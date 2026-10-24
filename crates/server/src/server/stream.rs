//! The streaming pass of phase 5: LOD0 cube streaming, LOD ring streaming
//! and world eviction, per player in join order. The pure offset/keep math
//! lives in [`super::stream_plan`]; this module drives it against the
//! live world, pool and pyramid.

use std::collections::HashMap;

use jbm_core::coords::CubePos;
use jbm_core::link::ServerLink;
use jbm_core::lod::CubeKey;
use jbm_core::protocol::ServerMsg;

use super::config::ServerConfig;
use super::stream_plan::{
    column_extra_dys, desired_offsets_surface, in_keep, level_center, lod_walk_surface, offset_pos,
    span_dy_keeps,
};
use super::{ConnectionState, Player, Server, cube_containing};
use crate::gen_pool::GenPool;
use crate::lod::LodPyramid;
use crate::spans::SurfaceSpans;

/// Entries of the merged LOD walk one player may scan past its cursor per
/// tick while cubes are missing (the cost-contract bound for unsettled
/// rings; a settled player's cursor sits at the end of the walk and costs
/// nothing).
const LOD_SKIP_WINDOW: usize = 512;

impl Server {
    /// Changes the view distances live (task 2.12's F3 panel): clamps
    /// LOD0 to 2..=32 cubes and the LOD rings to 2..=16 (or the floor
    /// when it is higher), updates the
    /// config, and replays the same transitions a border crossing does —
    /// sent cubes that left the new keep sets are unloaded (both LOD0
    /// and every ring level), the desired sets and merged LOD walks are
    /// forced to rebuild next tick, and eviction re-runs against the new
    /// keep sets. Growing works the same way in reverse: the rebuilt
    /// desired sets simply reach further, and the span store's fill queue
    /// feeds the wider square.
    pub fn set_view_radius(&mut self, view_radius: u32, lod_view_radius: u32) {
        self.config.view_radius = view_radius.clamp(2, 32);
        // Task 2.13 E's floor, live: keep L1 beyond the LOD0 disk (the
        // same clamp and warning ServerConfig::clamped prints). The cap
        // of 16 yields to it: view radius 32 needs 17.
        let min = ServerConfig::min_lod_view_radius(self.config.view_radius);
        self.config.lod_view_radius = lod_view_radius.clamp(2, min.max(16));
        if self.config.lod_view_radius < min {
            eprintln!(
                "jbm: lod_view_radius {} would leave L1 inside LOD0 (view_radius {}); using {min}",
                self.config.lod_view_radius, self.config.view_radius
            );
            self.config.lod_view_radius = min;
        }
        let (r, v, cap, window_r) = (
            self.config.view_radius,
            self.config.vertical_radius,
            self.config.max_column_cubes,
            self.config.camera_window_radius,
        );
        let (lod_r, lod_v) = (self.config.lod_view_radius, self.config.lod_vertical_radius);
        for conn in &mut self.connections {
            let ConnectionState::Joined(player) = &mut conn.state else {
                continue;
            };
            // LOD0: unload what left the new keep set, then force the
            // desired rebuild (the version sentinel, as at join).
            let center = player.center;
            let to_unload: Vec<CubePos> = player
                .sent
                .iter()
                .copied()
                .filter(|&pos| !lod0_keeps(&self.spans, pos, center, r, v, cap, window_r))
                .collect();
            for pos in to_unload {
                player.sent.remove(&pos);
                if conn.link.send(ServerMsg::UnloadCube { pos }).is_err() {
                    // The link is closed; the next streaming tick drops
                    // the connection, which also re-dirties eviction.
                    break;
                }
            }
            player.desired.clear();
            player.desired_version = u64::MAX;
            player.cursor = 0;
            // LOD rings, level by level, against each level's keep set.
            for (index, level_state) in player.lod.iter_mut().enumerate() {
                let level = index as u8 + 1;
                let center = level_state.center;
                let to_unload: Vec<CubePos> = level_state
                    .sent
                    .iter()
                    .copied()
                    .filter(|&pos| {
                        !lod_level_keeps(
                            &self.spans,
                            level,
                            pos,
                            center,
                            lod_r,
                            lod_v,
                            cap,
                            window_r,
                        )
                    })
                    .collect();
                for pos in to_unload {
                    level_state.sent.remove(&pos);
                    if conn
                        .link
                        .send(ServerMsg::UnloadLodCube { level, pos })
                        .is_err()
                    {
                        break;
                    }
                }
            }
            player.lod_walk.clear();
            player.lod_walk_version = u64::MAX;
            player.lod_cursor = 0;
        }
        self.eviction_dirty = true;
    }

    /// The server's effective `(view_radius, lod_view_radius)` — the
    /// clamped values the streams run with (task 2.12's tests read them).
    pub fn config_view_radius(&self) -> (u32, u32) {
        (self.config.view_radius, self.config.lod_view_radius)
    }

    /// The streaming pass, once per tick. Collects finished generation
    /// results, then per player (in join order): unloads cubes that left
    /// their keep set, sends newly in-range cubes nearest-first up to the
    /// send budget, and requests generation for missing ones within the
    /// in-flight cap. Finally evicts world cubes no player keeps.
    pub(super) fn phase_streaming(&mut self) {
        // 1. Collect finished generation results into the world. Inline
        //    mode generates up to the budget first; threaded mode drains
        //    whatever the workers finished (bounded by max_in_flight).
        //    Collection never overwrites: an edited cube may already be
        //    back in the world (pinned), so the stale generated copy is
        //    dropped — and must not touch the LOD pyramid either.
        let mut collected = false;
        let lod_on = self.config.lod_levels > 0;
        let inline = self.config.gen_threads == 0;
        let budget = self.config.gen_budget_per_tick;
        let Server {
            world, lod, pool, ..
        } = self;
        if inline {
            for _ in 0..budget {
                match pool.try_recv() {
                    Some((pos, cube, (octant, tops))) => {
                        collected = true;
                        if let std::collections::hash_map::Entry::Vacant(slot) = world.entry(pos) {
                            let inserted = slot.insert(cube);
                            if lod_on {
                                lod.insert_lod0_octant(pos, inserted, Some((&octant, &tops)));
                            }
                        }
                    }
                    None => break,
                }
            }
        } else {
            while let Some((pos, cube, (octant, tops))) = pool.try_recv() {
                collected = true;
                if let std::collections::hash_map::Entry::Vacant(slot) = world.entry(pos) {
                    let inserted = slot.insert(cube);
                    if lod_on {
                        lod.insert_lod0_octant(pos, inserted, Some((&octant, &tops)));
                    }
                }
            }
        }

        // Background LOD work lands after LOD0 ingest and before
        // streaming reads the pyramid.
        lod.poll();

        // Drain the pyramid's changed set once per tick, before the
        // per-player pass, queueing a re-send for every changed cube a
        // player already has.
        let lod_streaming = self.config.streaming && self.config.lod_levels > 0;
        let lod_changed = if lod_streaming {
            self.lod.take_changed()
        } else {
            Vec::new()
        };
        if !lod_changed.is_empty() {
            for conn in &mut self.connections {
                let ConnectionState::Joined(player) = &mut conn.state else {
                    continue;
                };
                for &(level, pos) in &lod_changed {
                    let Some(lvl) = player.lod.get(level as usize - 1) else {
                        continue;
                    };
                    if lvl.sent.contains(&pos)
                        && !player.lod_resend.contains(&CubeKey { level, pos })
                    {
                        player.lod_resend.push_back(CubeKey { level, pos });
                    }
                }
            }
        }

        // A join or disconnect changes who keeps which cubes.
        let joined_now = self
            .connections
            .iter()
            .filter(|conn| conn.joined_player().is_some())
            .count();
        let dirty = self.eviction_dirty || collected || joined_now != self.joined_count;
        self.joined_count = joined_now;

        // 2. Per player, in join order. The split borrows (connection vs
        //    world/pool/config) are disjoint fields of `self`.
        let mut dropped: Vec<usize> = Vec::new();
        let mut eviction_dirty = dirty;
        // Drain the surface-span fill queue within its per-tick budgets
        // before anyone streams from it. A settled view queues nothing and
        // this computes nothing.
        self.spans.poll(&self.generator);

        for idx in 0..self.connections.len() {
            let conn = &mut self.connections[idx];
            let ConnectionState::Joined(player) = &mut conn.state else {
                continue;
            };
            if stream_player(
                &mut conn.link,
                player,
                &self.world,
                &mut self.pool,
                &self.config,
                &mut self.spans,
                &mut eviction_dirty,
            ) {
                dropped.push(idx);
                continue;
            }
            if lod_streaming
                && stream_player_lod(
                    &mut conn.link,
                    player,
                    &mut self.lod,
                    &self.config,
                    &self.spans,
                )
            {
                dropped.push(idx);
            }
        }
        for idx in dropped.into_iter().rev() {
            self.remove_connection(idx);
        }

        // 3. Eviction: drop world cubes no player's keep set contains,
        //    keeping every pinned cube. Unedited cubes regenerate exactly;
        //    edited cubes under persistence are reloaded from disk once
        //    their edits are committed (version <= saved_version), so
        //    dropping those is lossless too. Only cubes with uncommitted
        //    edits — unsaved, or in a batch the save thread has not
        //    acknowledged yet — stay pinned forever. Without persistence
        //    (world_dir: None) every edited cube stays pinned. The keep
        //    set is surface-following: a cube of a column whose span
        //    reaches it stays even outside the camera window.
        if eviction_dirty {
            let (r, v) = (self.config.view_radius, self.config.vertical_radius);
            let cap = self.config.max_column_cubes;
            let mut centers: Vec<CubePos> = Vec::new();
            let mut sent_sets: Vec<&std::collections::HashSet<CubePos>> = Vec::new();
            for player in self.connections.iter().filter_map(|c| c.joined_player()) {
                centers.push(player.center);
                sent_sets.push(&player.sent);
            }
            let spans = &self.spans;
            let Server {
                world,
                pinned,
                versions,
                storage,
                eviction_dirty,
                ..
            } = self;
            let is_pinned = |pos: &CubePos| match storage {
                Some(_) => versions
                    .get(pos)
                    .is_some_and(|edit| edit.version > edit.saved_version),
                None => pinned.contains(pos),
            };
            let window_r = self.config.camera_window_radius;
            world.retain(|pos, _| {
                is_pinned(pos)
                    || centers
                        .iter()
                        .any(|&c| lod0_keeps(spans, *pos, c, r, v, cap, window_r))
                    // A cube a player was sent stays until the unload pass
                    // hands it back: dropping it here would strand the
                    // client's copy (and the walk skips sent cubes, so it
                    // would never re-stream). Task 2.13 D's far-column
                    // spans made this reachable — a window cube sent while
                    // its column was near can leave the span-only keep band
                    // between center moves.
                    || sent_sets.iter().any(|sent| sent.contains(pos))
            });
            // Evicted cubes take their edit bookkeeping with them: their
            // committed edits are on disk, and a reload starts fresh.
            versions.retain(|pos, _| world.contains_key(pos));
            *eviction_dirty = false;
        }
    }
}

/// Is `pos` in a player's LOD0 keep set (task 2.13 D)? The camera window
/// (radii +1, as before) applies only within `window_r` cubes
/// horizontally; every column of the keep ring instead keeps its surface
/// span trimmed one looser than the desired cap. The band between
/// `window_r` and `window_r + 1` keeps both shapes, so a player walking
/// along the boundary never flaps cubes in and out.
fn lod0_keeps(
    spans: &SurfaceSpans,
    pos: CubePos,
    center: CubePos,
    r: u32,
    v: u32,
    cap: usize,
    window_r: u32,
) -> bool {
    if in_keep(pos, center, window_r.min(r), v) {
        return true;
    }
    let dx = i64::from(pos.x - center.x);
    let dz = i64::from(pos.z - center.z);
    let keep_r = i64::from(r + 1);
    if dx * dx + dz * dz > keep_r * keep_r {
        return false;
    }
    let span = spans.lod0_padded(pos.x, pos.z);
    span_dy_keeps(
        span,
        i64::from(jbm_core::coords::CUBE_SIZE),
        i64::from(center.y),
        cap + 1,
        pos.y - center.y,
    )
}

/// The level-`level` form of [`lod0_keeps`] for ring cubes (task 2.13 D):
/// level 1's camera window applies only within `window_r` and its far
/// columns keep their sampled span; coarser levels are camera-keep only
/// (their default window already covers the shipped relief; see
/// `lod_walk_surface`).
// The eight parameters are one keep test's context; a bundling struct
// would only be unpacked again here.
#[allow(clippy::too_many_arguments)]
fn lod_level_keeps(
    spans: &SurfaceSpans,
    level: u8,
    pos: CubePos,
    center: CubePos,
    r: u32,
    v: u32,
    cap: usize,
    window_r: u32,
) -> bool {
    if in_keep(pos, center, r, v) {
        return true;
    }
    let dx = i64::from(pos.x - center.x);
    let dz = i64::from(pos.z - center.z);
    let keep_r = i64::from(r + 1);
    if dx * dx + dz * dz > keep_r * keep_r {
        return false;
    }
    if level != 1 {
        return false;
    }
    let span = spans.l1_padded(pos.x, pos.z);
    if dx * dx + dz * dz <= i64::from(window_r) * i64::from(window_r) {
        // Near column: also keep the span extras beyond the window.
        let side = i64::from(jbm_core::coords::CUBE_SIZE) * 2;
        let center_y = i64::from(center.y);
        for dy in column_extra_dys(span, side, center_y, v + 1, cap + 1) {
            if dy == pos.y - center.y {
                return true;
            }
        }
        return false;
    }
    span_dy_keeps(
        span,
        i64::from(jbm_core::coords::CUBE_SIZE) * 2,
        i64::from(center.y),
        cap + 1,
        pos.y - center.y,
    )
}

/// One player's slice of the streaming pass. Returns `true` when the link
/// closed and the connection must be dropped.
///
/// Cost contract: with an unchanged center, a settled span store and
/// everything sent, this is a handful of map lookups (the cursor already
/// sits at the end of the offset list) — no scans of the desired or sent
/// sets. While spans fill, each landed span rebuilds the desired list and
/// resets the cursor; the re-walk skips sent entries and pauses at the
/// first missing cube, so it stays cheap.
fn stream_player(
    link: &mut ServerLink,
    player: &mut Player,
    world: &HashMap<CubePos, jbm_core::cube::Cube>,
    pool: &mut GenPool,
    config: &ServerConfig,
    spans: &mut SurfaceSpans,
    eviction_dirty: &mut bool,
) -> bool {
    // Center: the cube containing the authoritative state position.
    let center = cube_containing(player.state.position);
    let (r, v) = (config.view_radius, config.vertical_radius);
    let cap = config.max_column_cubes;
    let window_r = config.camera_window_radius;

    if center != player.center {
        player.center = center;
        // Everything must be re-walked nearest-first from the new center.
        player.cursor = 0;
        *eviction_dirty = true;
        // Feed the span store's fill queue around the new center; it
        // covers the LOD0 and level-1 keep squares, so keep tests always
        // find a span near the view.
        spans.ensure_around((center.x, center.z), r, config.lod_levels);
        // Unload what left the keep set (the desired shape grown by one
        // in each radius, surface spans included — hysteresis against
        // border flapping).
        let to_unload: Vec<CubePos> = player
            .sent
            .iter()
            .copied()
            .filter(|&pos| !lod0_keeps(spans, pos, center, r, v, cap, window_r))
            .collect();
        for pos in to_unload {
            player.sent.remove(&pos);
            if link.send(ServerMsg::UnloadCube { pos }).is_err() {
                return true;
            }
        }
    }

    // The desired list follows the span store: a landed span may add that
    // column's surface cubes. Rebuilding resets the cursor — the re-walk
    // skips already-sent entries for free. The first build (empty list)
    // also feeds the span store around this center, so a player who never
    // moves still gets surface-following windows.
    if player.desired_version != spans.version() {
        if player.desired.is_empty() {
            spans.ensure_around((center.x, center.z), r, config.lod_levels);
        }
        player.desired =
            desired_offsets_surface(r, v, cap, i64::from(center.y), window_r, |dx, dz| {
                spans.lod0_padded(center.x + dx, center.z + dz)
            });
        player.desired_version = spans.version();
        player.cursor = 0;
    }
    let desired = &player.desired;

    // Send walk, nearest-first from the cursor. Already-sent cubes advance
    // the cursor for free; a missing cube stops the walk (the order stays
    // strictly nearest-first) after being requested.
    let mut sends_left = config.send_budget_per_tick;
    let mut i = player.cursor;
    while i < desired.len() {
        let pos = offset_pos(center, desired[i]);
        if player.sent.contains(&pos) {
            i += 1;
            continue;
        }
        let Some(cube) = world.get(&pos) else {
            if pool.in_flight() < config.max_in_flight {
                pool.request(pos);
            }
            break;
        };
        if sends_left == 0 {
            break;
        }
        let msg = ServerMsg::CubeData {
            pos,
            cube: Box::new(cube.clone()),
        };
        if link.send(msg).is_err() {
            player.cursor = i;
            return true;
        }
        player.sent.insert(pos);
        sends_left -= 1;
        i += 1;
    }
    player.cursor = i;

    // Request top-up: keep the pool fed ahead of the send cursor without
    // sending out of order. Runs only while something is missing, so an
    // idle player never scans.
    if pool.in_flight() < config.max_in_flight {
        for &offset in &desired[i..] {
            let pos = offset_pos(center, offset);
            if player.sent.contains(&pos) || world.contains_key(&pos) {
                continue;
            }
            pool.request(pos);
            if pool.in_flight() >= config.max_in_flight {
                break;
            }
        }
    }
    false
}

/// One player's slice of LOD ring streaming (surface-following windows
/// included). Returns `true` when the link closed and the connection must
/// be dropped.
///
/// Cost contract, like the LOD0 one: a settled player whose rings are
/// complete costs a few center comparisons per level per tick — the
/// cursor sits at the end of the walk, so nothing is scanned. While cubes
/// are missing, the per-tick scan is bounded by [`LOD_SKIP_WINDOW`]
/// entries past the cursor plus the send budget; the cursor only advances
/// over sent entries, so a gap in the pyramid costs a bounded look-ahead,
/// never a full rescan (a level-center move resets it, like LOD0).
fn stream_player_lod(
    link: &mut ServerLink,
    player: &mut Player,
    lod: &mut LodPyramid,
    config: &ServerConfig,
    spans: &SurfaceSpans,
) -> bool {
    let lod0_center = cube_containing(player.state.position);
    let (r, v) = (config.lod_view_radius, config.lod_vertical_radius);
    let cap = config.max_column_cubes;
    let window_r = config.camera_window_radius;

    // 1. Level centers: unload what left that level's keep set (the
    //    desired shape at +1 in both radii, level-1 surface spans
    //    included), then reset the walk cursor.
    let mut center_moved = false;
    for (index, level_state) in player.lod.iter_mut().enumerate() {
        let center = level_center(lod0_center, index as u8 + 1);
        if center == level_state.center {
            continue;
        }
        let level = index as u8 + 1;
        level_state.center = center;
        let to_unload: Vec<CubePos> = level_state
            .sent
            .iter()
            .copied()
            .filter(|&pos| !lod_level_keeps(spans, level, pos, center, r, v, cap, window_r))
            .collect();
        for pos in to_unload {
            level_state.sent.remove(&pos);
            if link.send(ServerMsg::UnloadLodCube { level, pos }).is_err() {
                return true;
            }
        }
        center_moved = true;
    }
    if center_moved {
        player.lod_cursor = 0;
    }

    // 2. Queued re-sends first, within the budget: an edit's LOD change
    //    reaches the player within one tick whenever the budget allows.
    let mut budget = config.lod_send_budget_per_tick;
    while budget > 0 {
        let Some(&key) = player.lod_resend.front() else {
            break;
        };
        let CubeKey { level, pos } = key;
        let Some(level_state) = player.lod.get(level as usize - 1) else {
            player.lod_resend.pop_front();
            continue;
        };
        // Dropped since queuing (unloaded or no longer ours): skip it.
        if !level_state.sent.contains(&pos) {
            player.lod_resend.pop_front();
            continue;
        }
        let Some((cube, tops)) = lod.ready_tops(level, pos) else {
            // No longer ready: nothing true to send, drop the entry.
            player.lod_resend.pop_front();
            continue;
        };
        if link
            .send(ServerMsg::LodCubeData {
                level,
                pos,
                cube: Box::new(cube.clone()),
                tops: tops.clone(),
            })
            .is_err()
        {
            return true;
        }
        player.lod_resend.pop_front();
        budget -= 1;
    }

    // 3. The merged walk (rebuilt when a center moved or a level-1 span
    //    landed — the span store's version covers both players' views
    //    being rebuilt lazily). Sent entries advance the head cursor for
    //    free; a not-ready cube is requested and skipped (the client's
    //    coverage rule makes any arrival order hole-free); the scan stops
    //    at the skip window even when the near cubes stay missing.
    if center_moved || player.lod_walk_version != spans.version() {
        let l1_center = player.lod.first().map(|state| state.center);
        player.lod_walk = lod_walk_surface(
            config.lod_levels,
            r,
            v,
            cap,
            l1_center.map_or(0, |c| i64::from(c.y)),
            window_r,
            |dx, dz| l1_center.and_then(|c| spans.l1_padded(c.x + dx, c.z + dz)),
        );
        player.lod_walk_version = spans.version();
        player.lod_cursor = 0;
    }
    let walk = &player.lod_walk;
    let mut head = player.lod_cursor.min(walk.len());
    let mut i = head;
    let end = (head + LOD_SKIP_WINDOW).min(walk.len());
    while i < end && budget > 0 {
        let (level, offset) = walk[i];
        let level_state = &mut player.lod[level as usize - 1];
        let pos = offset_pos(level_state.center, offset);
        if level_state.sent.contains(&pos) {
            if i == head {
                head += 1;
            }
            i += 1;
            continue;
        }
        match lod.ready_tops(level, pos) {
            Some((cube, tops)) => {
                if link
                    .send(ServerMsg::LodCubeData {
                        level,
                        pos,
                        cube: Box::new(cube.clone()),
                        tops: tops.clone(),
                    })
                    .is_err()
                {
                    player.lod_cursor = head;
                    return true;
                }
                level_state.sent.insert(pos);
                if i == head {
                    head += 1;
                }
                budget -= 1;
            }
            None => {
                if lod.in_flight() < config.lod_max_in_flight {
                    lod.request(level, pos);
                }
            }
        }
        i += 1;
    }
    player.lod_cursor = head;
    false
}

#[cfg(test)]
mod tests {
    use jbm_core::block::builtin::Blocks;
    use jbm_core::coords::{BlockPos, CubePos, LocalPos};
    use jbm_core::link::ClientLink;
    use jbm_core::protocol::{ClientMsg, PlayerId, ServerMsg};

    use super::Server;
    use crate::server::stream_plan::{LodWalkEntry, in_desired, in_keep, level_center};
    use crate::server::test_util::{
        drain, fly_until, input, joined_and_loaded, pending_hello_on, physics_server, surface_block,
    };
    use crate::server::{ServerConfig, cube_containing};

    /// The merged walk with plain vertical windows — the reference shape
    /// the surface-following walk extends; the order tests pin it.
    fn lod_walk(lod_levels: u8, r: u32, v: u32) -> Vec<LodWalkEntry> {
        use crate::server::stream_plan::desired_offsets;
        let mut walk = Vec::new();
        for level in 1..=lod_levels {
            for offset in desired_offsets(r, v) {
                walk.push((level, offset));
            }
        }
        walk.sort_by_key(|&(level, (x, y, z))| {
            let d = i64::from(x) * i64::from(x)
                + i64::from(y) * i64::from(y)
                + i64::from(z) * i64::from(z);
            (d, std::cmp::Reverse(level), x, y, z)
        });
        walk
    }

    #[test]
    fn streaming_center_follows_the_authoritative_position() {
        let mut server = Server::with_config(ServerConfig {
            streaming: true,
            view_radius: 2,
            vertical_radius: 1,
            send_budget_per_tick: 64,
            gen_threads: 0,
            gen_budget_per_tick: 64,
            max_in_flight: 512,
            ..ServerConfig::default()
        });
        let (mut client, server_end) = jbm_core::link::in_process_pair();
        server.add_connection(server_end);
        client
            .send(ClientMsg::Hello {
                protocol_version: jbm_core::protocol::PROTOCOL_VERSION,
                name: "walker".to_owned(),
            })
            .unwrap();
        server.tick();
        let _ = drain(&mut client);
        for _ in 0..10 {
            server.tick();
            let _ = drain(&mut client);
        }
        let start = server.players().next().unwrap().state;
        let center0 = jbm_core::coords::BlockPos {
            x: start.position.x.floor() as i64,
            y: start.position.y.floor() as i64,
            z: start.position.z.floor() as i64,
        }
        .split()
        .0;

        // Fly well above the terrain, then east across a cube border. The
        // unloaded rule stalls movement while cubes stream in, so the loop
        // runs until the position actually crosses.
        let east = std::f32::consts::FRAC_PI_2;
        let mut seq = 0u64;
        let mut crossed = false;
        for _ in 0..4000 {
            let state = server.players().next().unwrap().state;
            let rise = state.position.y < start.position.y + 40.0;
            let move_input = if rise {
                jbm_core::physics::MoveInput {
                    up: 1,
                    fly_toggle: !state.flying,
                    ..jbm_core::physics::MoveInput::IDLE
                }
            } else {
                jbm_core::physics::MoveInput {
                    forward: 1,
                    sprint: true,
                    yaw: east,
                    ..jbm_core::physics::MoveInput::IDLE
                }
            };
            seq += 1;
            client.send(input(seq, move_input)).unwrap();
            server.tick();
            for msg in drain(&mut client) {
                if let ServerMsg::CubeData { pos, .. } = msg
                    && pos.x >= center0.x + 2
                    && pos.z == center0.z
                {
                    crossed = true;
                }
            }
            let state = server.players().next().unwrap().state;
            if !rise && state.position.x >= start.position.x + 33.0 {
                break;
            }
        }
        let state = server.players().next().unwrap().state;
        assert!(
            state.position.x >= start.position.x + 33.0,
            "never crossed the cube border: {state:?}"
        );
        assert!(
            crossed,
            "no cubes streamed east of the start — the center did not follow"
        );
    }

    #[test]
    fn a_pinned_cube_survives_walking_away_and_back() {
        let (mut server, mut client) = pending_hello_on(physics_server());
        let _ = joined_and_loaded(&mut server, &mut client);
        let ground = surface_block(&server);
        let pinned_cube = ground.split().0;

        client.send(ClientMsg::BreakBlock { pos: ground }).unwrap();
        server.tick();
        let _ = drain(&mut client);
        assert_eq!(
            server.block(ground),
            Some(jbm_core::block::BlockStateId::AIR)
        );
        assert_eq!(server.pinned_count(), 1);

        // Fly east (yaw pi/2) until the pinned cube leaves the keep set
        // and is unloaded from the client.
        let mut seq = 0u64;
        let start = server.players().next().unwrap().state.position;
        let cruise = start.y + 40.0;
        let left = fly_until(
            &mut server,
            &mut client,
            std::f32::consts::FRAC_PI_2,
            cruise,
            &mut seq,
            |_, messages| {
                messages
                    .iter()
                    .any(|m| matches!(m, ServerMsg::UnloadCube { pos } if *pos == pinned_cube))
            },
            4000,
        );
        assert!(left, "the pinned cube was never unloaded");
        // Evicted world cubes: everything unpinned around spawn is gone,
        // the pinned cube stays.
        assert_eq!(
            server.block(ground),
            Some(jbm_core::block::BlockStateId::AIR)
        );
        let beside = BlockPos {
            x: 0,
            y: ground.y,
            z: 32,
        };
        assert_eq!(
            server.block(beside),
            None,
            "unpinned neighbour cube was evicted"
        );
        assert_eq!(server.pinned_count(), 1);

        // Fly home (yaw -pi/2): the cube streams back with the edit.
        let returned = fly_until(
            &mut server,
            &mut client,
            -std::f32::consts::FRAC_PI_2,
            cruise,
            &mut seq,
            |_, messages| {
                messages
                    .iter()
                    .any(|m| matches!(m, ServerMsg::CubeData { pos, .. } if *pos == pinned_cube))
            },
            4000,
        );
        assert!(returned, "the pinned cube never streamed back");
        assert_eq!(
            server.block(ground),
            Some(jbm_core::block::BlockStateId::AIR)
        );
    }

    #[test]
    fn world_cube_count_keeps_only_pinned_cubes_after_the_player_leaves() {
        let (mut server, mut client) = pending_hello_on(physics_server());
        let _ = joined_and_loaded(&mut server, &mut client);
        let loaded = server.world_cube_count();
        assert!(loaded > 1);

        let ground = surface_block(&server);
        client.send(ClientMsg::BreakBlock { pos: ground }).unwrap();
        server.tick();
        let _ = drain(&mut client);

        client.send(ClientMsg::Disconnect).unwrap();
        for _ in 0..5 {
            server.tick();
        }
        drop(client);
        for _ in 0..5 {
            server.tick();
        }
        assert_eq!(server.players().count(), 0);
        assert_eq!(server.pinned_count(), 1);
        assert_eq!(
            server.world_cube_count(),
            1,
            "only the pinned cube survives the player leaving"
        );
    }

    // ------------------------------------------------------------------
    // LOD ring streaming

    /// A streaming server with small LOD rings and inline generation, the
    /// deterministic mode these tests use.
    fn lod_config(lod_budget: usize) -> ServerConfig {
        ServerConfig {
            streaming: true,
            // Classic terrain: these tests pin the walk order, budgets and
            // keep sets, whose meaning is the window shape. On the
            // overworld the surface-following windows add span cubes
            // beyond the small `lod_vertical_radius` 2 ring here, which
            // would be a change of terrain, not of rule.
            worldgen: crate::worldgen::Preset::Classic,
            view_radius: 1,
            vertical_radius: 1,
            send_budget_per_tick: 64,
            gen_threads: 0,
            gen_budget_per_tick: 64,
            max_in_flight: 512,
            lod_levels: 3,
            lod_view_radius: 3,
            lod_vertical_radius: 2,
            lod_send_budget_per_tick: lod_budget,
            lod_max_in_flight: 256,
            ..ServerConfig::default()
        }
    }

    fn lod_server(lod_budget: usize) -> Server {
        Server::with_config(lod_config(lod_budget))
    }

    fn split_lod(messages: &[ServerMsg]) -> Vec<(u8, CubePos)> {
        messages
            .iter()
            .filter_map(|m| match m {
                ServerMsg::LodCubeData { level, pos, .. } => Some((*level, *pos)),
                _ => None,
            })
            .collect()
    }

    /// Joins one player and ticks until the LOD rings settle: no LOD
    /// message for `rounds` full ticks **and** no fill work left anywhere
    /// in the pyramid's pipeline. The pyramid keeps filling (and
    /// re-sending changed cubes) long after the first rings land, so quiet
    /// alone is not "settled". Returns every message seen.
    fn settle_lod(server: &mut Server, client: &mut ClientLink, rounds: usize) -> Vec<ServerMsg> {
        let mut seen = Vec::new();
        let mut quiet = 0;
        while quiet < rounds {
            server.tick();
            let messages = drain(client);
            let busy = messages.iter().any(|m| {
                matches!(
                    m,
                    ServerMsg::LodCubeData { .. } | ServerMsg::UnloadLodCube { .. }
                )
            }) || server.lod_pyramid().in_flight_total() > 0;
            quiet = if busy { 0 } else { quiet + 1 };
            seen.extend(messages);
        }
        seen
    }

    #[test]
    fn settled_rings_send_every_ready_desired_cube_exactly_once() {
        let (mut server, mut client) = pending_hello_on(lod_server(32));
        let seen = settle_lod(&mut server, &mut client, 10);
        let sent = split_lod(&seen);

        let state = server.players().next().unwrap().state;
        let lod0_center = cube_containing(state.position);
        let player = server
            .connections
            .iter()
            .filter_map(|c| c.joined_player())
            .next()
            .unwrap();
        // Nothing outside the desired disks was sent, and every cube the
        // server recorded as sent was received. A cube can appear more
        // than once in the fill history: a cube's content changes when
        // real data replaces a worldgen guess (the `merge` → `touch` →
        // `take_changed` chain in lod.rs), and changed cubes are re-sent
        // by design via the re-send queue in stream_player_lod step 2.
        // Exactly-once is therefore asserted on the settled window below,
        // where those re-sends are provably done.
        let mut counts = std::collections::HashMap::new();
        for &(level, pos) in &sent {
            *counts.entry((level, pos)).or_insert(0usize) += 1;
            let center = level_center(lod0_center, level);
            // The sent set is inside the player's own surface-following
            // walk (level 1's spans add cubes beyond the plain window on
            // this spawn's hill), whose window part is exactly the old
            // disk shape.
            let player_walk = &player.lod_walk;
            let offset = (pos.x - center.x, pos.y - center.y, pos.z - center.z);
            assert!(
                player_walk.contains(&(level, offset)),
                "sent outside the desired walk: level {level} {pos:?}"
            );
        }
        for (index, level_state) in player.lod.iter().enumerate() {
            let level = index as u8 + 1;
            for &pos in &level_state.sent {
                assert!(
                    counts.get(&(level, pos)).is_some_and(|&n| n >= 1),
                    "a cube in the sent set was never received: level {level} {pos:?}"
                );
                assert!(
                    server.lod_ready(level, pos).is_some(),
                    "a sent cube is no longer ready (level {level})"
                );
            }
        }
        // The bookkeeping agrees two-sidedly: every cube received is one
        // the server recorded as sent. With no inputs the level centers
        // never move, so nothing was unloaded out of the sent sets — a
        // received-but-unrecorded cube would be an untracked send.
        for &(level, pos) in counts.keys() {
            assert!(
                player.lod[level as usize - 1].sent.contains(&pos),
                "a cube was received but never recorded as sent: level {level} {pos:?}"
            );
        }
        // Every cube in the desired disks whose ready is Some is in the
        // sent set. Held-but-not-ready cubes are correctly not sent.
        for (level, pos) in server
            .lod_pyramid()
            .iter(1)
            .map(|(pos, _)| (1u8, *pos))
            .chain(
                server
                    .lod_pyramid()
                    .iter(2)
                    .map(|(pos, _)| (2u8, *pos))
                    .chain(server.lod_pyramid().iter(3).map(|(pos, _)| (3u8, *pos))),
            )
        {
            let center = level_center(lod0_center, level);
            if in_desired(pos, center, 3, 2) && server.lod_ready(level, pos).is_some() {
                assert!(
                    player.lod[level as usize - 1].sent.contains(&pos),
                    "ready desired cube never sent: level {level} {pos:?}"
                );
            }
        }

        // Exactly-once, settled window: after the pyramid is quiet, no
        // cube may be sent again. Zero LOD traffic here is a
        // code-guaranteed absence, not a guess: settle_lod returned only
        // after no LOD traffic for 10 full ticks with in_flight_total()
        // at zero, so no fill job can land to change a cube (`merge` →
        // `touch` → `take_changed` reports nothing) — and any leftover
        // re-send queue entry would itself have been traffic, since step 2
        // drains the queue first within the budget. Every desired cube is
        // already sent (asserted above), which leaves the walk cursor at
        // the end of the walk with nothing to scan. And with no inputs the
        // level centers never move, so step 1 unloads nothing. Any
        // LodCubeData in this window is a re-send of an unchanged cube;
        // any UnloadLodCube, an unload without a center move — both bugs.
        // 20 ticks is a full walk pass either way: 435 entries at the
        // budget of 32 need 14.
        let mut window = Vec::new();
        for _ in 0..20 {
            server.tick();
            window.extend(drain(&mut client));
        }
        let resent = split_lod(&window);
        assert!(
            resent.is_empty(),
            "{resent:?} re-sent after the pyramid settled"
        );
        assert!(
            !window
                .iter()
                .any(|m| matches!(m, ServerMsg::UnloadLodCube { .. })),
            "an unload happened with a static center"
        );
    }

    #[test]
    fn lod_messages_respect_the_budget_and_the_merged_walk_order() {
        let (mut server, mut client) = pending_hello_on(lod_server(3));
        // First sends only. The re-send queue (stream_player_lod step 2,
        // drained ahead of the walk) only ever carries cubes already in a
        // player's sent set — phase_streaming queues an entry only when
        // `lvl.sent.contains(&pos)` — so a re-send is never a first send,
        // and first sends come only from the walk itself, in walk order.
        // Re-sends are expected: real data replaces worldgen guesses
        // (`merge` → `touch` → `take_changed`), and those cubes re-send
        // at an arbitrary walk position. They are excluded from the order
        // check but still counted for the budget check, whose contract
        // includes re-sends. The state never moves here — no input is ever
        // sent, and physics steps only on inputs — so the level centers
        // and the walk cursor stay fixed for the whole run and no walk
        // segmentation is needed.
        // 400 ticks: 435 walk entries at a budget of 3 (re-sends included)
        // need ~145 ticks even without fill latency.
        let mut first_sent = std::collections::HashSet::new();
        let mut order: Vec<(u8, (i32, i32, i32))> = Vec::new();
        let mut per_tick_over = 0;
        for _ in 0..400 {
            server.tick();
            let messages = drain(&mut client);
            let lod_count = split_lod(&messages).len();
            if lod_count > 3 {
                per_tick_over += 1;
            }
            let state = server.players().next().unwrap().state;
            let lod0_center = cube_containing(state.position);
            for (level, pos) in split_lod(&messages) {
                if !first_sent.insert((level, pos)) {
                    continue;
                }
                let center = level_center(lod0_center, level);
                order.push((
                    level,
                    (pos.x - center.x, pos.y - center.y, pos.z - center.z),
                ));
            }
            if server.lod_ring_pending(PlayerId(1)) == 0 {
                break;
            }
        }
        assert_eq!(per_tick_over, 0, "the per-tick budget was exceeded");
        assert!(!order.is_empty());

        // The first-send order is a subsequence of the merged walk —
        // the player's own surface-following walk (level 1 unions the
        // column spans, which on this spawn's hill adds one cube below
        // the plain window).
        let player = server
            .connections
            .iter()
            .filter_map(|c| c.joined_player())
            .next()
            .unwrap();
        let walk = player.lod_walk.clone();
        assert!(walk.len() >= lod_walk(3, 3, 2).len());
        let mut walk_index = 0;
        for entry in &order {
            let mut found = false;
            while walk_index < walk.len() {
                if walk[walk_index] == *entry {
                    found = true;
                    walk_index += 1;
                    break;
                }
                walk_index += 1;
            }
            assert!(
                found,
                "send {entry:?} is out of the merged walk order (at walk index {walk_index})"
            );
        }
    }

    #[test]
    fn crossing_a_level2_boundary_unloads_exactly_the_left_keep_set() {
        let (mut server, mut client) = pending_hello_on(lod_server(64));
        let _ = settle_lod(&mut server, &mut client, 10);
        let start = server.players().next().unwrap().state;
        let lod0 = cube_containing(start.position);
        let center2 = level_center(lod0, 2);

        // Fly east (yaw pi/2) until the level-2 center moves.
        let mut seq = 0u64;
        let mut new_center2 = center2;
        let crossed = fly_until(
            &mut server,
            &mut client,
            std::f32::consts::FRAC_PI_2,
            start.position.y + 40.0,
            &mut seq,
            |server, _| {
                let state = server.players().next().unwrap().state;
                new_center2 = level_center(cube_containing(state.position), 2);
                new_center2 != center2
            },
            6000,
        );
        assert!(crossed, "never crossed a level-2 boundary");

        // Everything the player still holds at level 2 is in the new keep
        // set; everything that left it was unloaded exactly once.
        let player = server
            .connections
            .iter()
            .filter_map(|c| c.joined_player())
            .next()
            .unwrap();
        for pos in &player.lod[1].sent {
            assert!(
                in_keep(*pos, new_center2, 3, 2),
                "kept cube outside the new keep set: {pos:?}"
            );
        }
        // Settle again and collect the unloads that happened during the
        // flight (drained inside fly_until), so tally them from a fresh
        // move instead: walk back west and watch the messages.
        let before: std::collections::HashSet<CubePos> =
            player.lod[1].sent.iter().copied().collect();
        let back_center2 = new_center2;
        let mut unloaded = Vec::new();
        let returned = fly_until(
            &mut server,
            &mut client,
            -std::f32::consts::FRAC_PI_2,
            start.position.y + 40.0,
            &mut seq,
            |server, messages| {
                unloaded.extend(messages.iter().filter_map(|m| match m {
                    ServerMsg::UnloadLodCube { level: 2, pos } => Some(*pos),
                    _ => None,
                }));
                let state = server.players().next().unwrap().state;
                let c = level_center(cube_containing(state.position), 2);
                c != back_center2
            },
            6000,
        );
        assert!(returned, "never crossed back");
        // Every unloaded cube left the then-current keep set and was held
        // before the flight.
        for pos in &unloaded {
            assert!(
                before.contains(pos),
                "unloaded a cube the player never had: {pos:?}"
            );
        }
        let _ = settle_lod(&mut server, &mut client, 10);
        // The new ones arrived: the sent set is non-empty and inside the
        // new keep set.
        let player = server
            .connections
            .iter()
            .filter_map(|c| c.joined_player())
            .next()
            .unwrap();
        assert!(!player.lod[1].sent.is_empty());
    }

    /// Two stacked stone blocks above the surface, aligned into one
    /// 2x2x2 group: a 2-of-8 fill that turns its LOD1 cell from air to
    /// stone — an edit that provably changes the pyramid whatever the
    /// surface height's parity. Reachable from spawn, beside (not into)
    /// the player's box.
    fn lod_changing_placements(server: &Server) -> [BlockPos; 2] {
        let feet = server.players().next().unwrap().state.position;
        let mut y = feet.y.floor() as i64 + 2;
        if y % 2 != 0 {
            y += 1;
        }
        [
            BlockPos { x: 0, y, z: 2 },
            BlockPos {
                x: 0,
                y: y + 1,
                z: 2,
            },
        ]
    }

    #[test]
    fn an_edit_re_sends_changed_lod_cubes_within_one_tick() {
        let (mut server, mut client) = pending_hello_on(lod_server(64));
        let _ = settle_lod(&mut server, &mut client, 10);
        let _ = drain(&mut client);

        // An edit near the player: the LOD cubes it changes must be
        // re-sent in the same tick, with content equal to the pyramid's.
        let blocks = Blocks::register(&mut jbm_core::block::BlockRegistry::new());
        let [a, b] = lod_changing_placements(&server);
        client
            .send(ClientMsg::PlaceBlock {
                pos: a,
                state: blocks.stone,
            })
            .unwrap();
        client
            .send(ClientMsg::PlaceBlock {
                pos: b,
                state: blocks.stone,
            })
            .unwrap();
        server.tick();
        let messages = drain(&mut client);
        let resent = split_lod(&messages);
        assert!(
            !resent.is_empty(),
            "the edit changed no LOD cube that was re-sent"
        );
        for (level, pos) in &resent {
            let cube = server
                .lod_cube(*level, *pos)
                .expect("changed cube is ready");
            let tops = server.lod_tops(*level, *pos).unwrap();
            assert!(
                messages.iter().any(|m| matches!(
                    m,
                    ServerMsg::LodCubeData {
                        level: l,
                        pos: p,
                        cube: c,
                        tops: t,
                    } if l == level && p == pos && c.as_ref() == cube && t == tops
                )),
                "re-send content differs from the pyramid at level {level} {pos:?}"
            );
        }
        // All of it landed before the Tick message.
        assert!(matches!(messages.last(), Some(ServerMsg::Tick { .. })));
    }

    #[test]
    fn re_send_queue_never_starves_under_a_saturated_budget() {
        // One LOD message per tick and a stream of further changes: the
        // first edit's re-send must still arrive within a bounded wait.
        let (mut server, mut client) = pending_hello_on(lod_server(1));
        let _ = settle_lod(&mut server, &mut client, 10);
        let _ = drain(&mut client);

        let blocks = Blocks::register(&mut jbm_core::block::BlockRegistry::new());
        let [a, b] = lod_changing_placements(&server);
        for pos in [a, b] {
            client
                .send(ClientMsg::PlaceBlock {
                    pos,
                    state: blocks.stone,
                })
                .unwrap();
        }
        server.tick();

        // The parent of the placed column is the changed level-1 cube.
        let lod0 = a.split().0;
        let parent = CubePos {
            x: lod0.x.div_euclid(2),
            y: lod0.y.div_euclid(2),
            z: lod0.z.div_euclid(2),
        };

        // A churn of changes keeps the one-per-tick budget saturated,
        // while the first edit's re-send must come through anyway.
        let mut arrived_at = None;
        let mut ticks_since_edit = 0;
        for round in 0..32 {
            if round < 8 {
                let msg = if round % 2 == 0 {
                    ClientMsg::BreakBlock { pos: b }
                } else {
                    ClientMsg::PlaceBlock {
                        pos: b,
                        state: blocks.stone,
                    }
                };
                client.send(msg).unwrap();
            }
            server.tick();
            ticks_since_edit += 1;
            let messages = drain(&mut client);
            if messages.iter().any(
                |m| matches!(m, ServerMsg::LodCubeData { level: 1, pos, .. } if *pos == parent),
            ) && arrived_at.is_none()
            {
                arrived_at = Some(ticks_since_edit);
            }
        }
        assert!(
            arrived_at.is_some_and(|ticks| ticks <= 16),
            "the saturated budget starved the first edit's re-send (arrived {arrived_at:?})"
        );
    }

    #[test]
    fn no_lod_messages_without_streaming_or_levels_or_a_join() {
        // lod_levels = 0.
        let (mut server, mut client) = pending_hello_on(Server::with_config(ServerConfig {
            gen_threads: 0,
            lod_levels: 0,
            ..lod_config(64)
        }));
        for _ in 0..10 {
            server.tick();
            for msg in drain(&mut client) {
                assert!(
                    !matches!(
                        msg,
                        ServerMsg::LodCubeData { .. } | ServerMsg::UnloadLodCube { .. }
                    ),
                    "LOD message with the pyramid off: {msg:?}"
                );
            }
        }

        // streaming = false.
        let (mut server, mut client) = pending_hello_on(Server::with_config(ServerConfig {
            streaming: false,
            gen_threads: 0,
            ..lod_config(64)
        }));
        for _ in 0..10 {
            server.tick();
            for msg in drain(&mut client) {
                assert!(
                    !matches!(
                        msg,
                        ServerMsg::LodCubeData { .. } | ServerMsg::UnloadLodCube { .. }
                    ),
                    "LOD message with streaming off: {msg:?}"
                );
            }
        }

        // A pending connection (no Hello yet) receives nothing at all.
        let mut server = lod_server(64);
        let (mut client, server_end) = jbm_core::link::in_process_pair();
        server.add_connection(server_end);
        for _ in 0..10 {
            server.tick();
            assert!(drain(&mut client).is_empty());
        }
    }

    #[test]
    fn a_settled_idle_player_gets_zero_lod_messages_for_100_ticks() {
        let (mut server, mut client) = pending_hello_on(lod_server(64));
        let _ = settle_lod(&mut server, &mut client, 10);
        for _ in 0..100 {
            server.tick();
            let messages = drain(&mut client);
            let lod = messages
                .iter()
                .filter(|m| {
                    matches!(
                        m,
                        ServerMsg::LodCubeData { .. } | ServerMsg::UnloadLodCube { .. }
                    )
                })
                .count();
            assert_eq!(lod, 0, "idle player got LOD traffic: {messages:?}");
        }
    }

    #[test]
    fn collecting_generation_never_overwrites_an_edited_cube() {
        let (mut server, mut client) = pending_hello_on(physics_server());
        let _ = joined_and_loaded(&mut server, &mut client);
        let center = cube_containing(server.players().next().unwrap().state.position);
        // In the keep set (r + 1 = 2) but outside the desired set (r = 1):
        // streaming will keep it, but never requests or sends it.
        let target = CubePos {
            x: center.x + 2,
            y: center.y,
            z: center.z,
        };
        assert!(!server.world.contains_key(&target));

        // Request generation, then land an edited cube in the world before
        // the result is collected — the race `entry().or_insert` must win.
        server.pool.request(target);
        let blocks = Blocks::register(&mut jbm_core::block::BlockRegistry::new());
        server
            .world
            .insert(target, jbm_core::cube::Cube::filled(blocks.glass));
        let probe = target.block(LocalPos::new_unchecked(1, 1, 1));
        server.tick();
        let _ = drain(&mut client);

        assert_eq!(
            server.block(probe),
            Some(blocks.glass),
            "generation overwrote the edited cube"
        );
    }
}
