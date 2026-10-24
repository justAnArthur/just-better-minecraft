//! Read-only views of server state. [`Server::stats`] takes one snapshot
//! of every counter a debug panel or diagnostics view wants; the
//! finer-grained accessors below stay for the tests and benches that read
//! individual numbers (including per-cube queries a snapshot cannot
//! express).

use std::sync::Arc;

use jbm_core::coords::{BlockPos, CubePos};
use jbm_core::protocol::PlayerId;

use super::stream_plan::offset_pos;
use super::{Connection, Player, PlayerView, Server};
use crate::lod::LodPyramid;
use crate::storage::Storage;

/// One joined player's streaming counters, in the snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerStats {
    pub id: PlayerId,
    /// Cubes currently sent to (and not unloaded from) the player.
    pub sent_count: usize,
    /// Per-level LOD ring cubes sent (index 0 = level 1), plus the LOD0
    /// sent count at the end — the measurement hook for cubes desired per
    /// level.
    pub sent_counts_per_level: Vec<usize>,
    /// How many of the player's desired LOD0 cubes are not yet sent. Zero
    /// once the whole surface-following desired set has streamed.
    pub lod0_pending: usize,
    /// How many of the player's desired ring cubes are not yet sent,
    /// summed over levels. Zero once every ring the pyramid can fill has
    /// streamed.
    pub lod_ring_pending: usize,
}

/// LOD pyramid counters, in the snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LodStats {
    /// LOD jobs queued, running or finished-but-uncollected (the
    /// pyramid's own internal jobs excluded).
    pub in_flight: usize,
    /// Internal (pending-propagation) LOD jobs the pyramid has queued in
    /// total — the load-report number for the default view.
    pub internal_requests: u64,
    /// How many `sample_lod` sampling runs the generator has done — the
    /// test hook that proves storage reloads do not sample.
    pub samples: u64,
    /// LOD cubes held across all levels.
    pub cube_count: usize,
    /// Approximate heap bytes of the pyramid; diagnostic.
    pub heap_bytes: usize,
}

/// Surface-span store counters, in the snapshot: columns whose exact LOD0
/// bounds and sampled level-1 bounds have been computed, jobs still
/// queued, and the store's version (bumped per landed span).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanStats {
    pub computed_lod0: u64,
    pub computed_l1: u64,
    pub pending: usize,
    pub version: u64,
}

/// Persistence counters, in the snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaveStats {
    /// Cubes pinned by an accepted edit (never evicted; see
    /// [`Server::pinned_count`]).
    pub pinned_count: usize,
    /// Save batches committed by the save thread so far.
    pub saved_batches: u64,
    /// Save batches the save thread reported failed so far (their cubes
    /// stayed pinned for a retry).
    pub failed_batches: u64,
    /// Batches handed to the save thread and not yet answered (committed
    /// or failed). Tests wait for 0 before holding saves.
    pub unanswered_batches: u64,
    /// Cubes whose snapshot is with the save thread, unacknowledged.
    pub pending_save_count: usize,
}

/// A point-in-time snapshot of everything a debug panel or diagnostics
/// view wants from a [`Server`] — one call, no per-counter polling. The
/// values are the same ones the accessors below report.
#[derive(Debug, Clone, PartialEq)]
pub struct ServerStats {
    /// Ticks run so far.
    pub tick: u64,
    /// Joined players in join order.
    pub players: Vec<PlayerView>,
    /// Cubes the server currently keeps in memory.
    pub world_cube_count: usize,
    /// Generation pool: cubes queued, generating or finished-uncollected.
    pub gen_in_flight: usize,
    pub lod: LodStats,
    pub spans: SpanStats,
    pub saves: SaveStats,
    /// Per joined player, in join order.
    pub per_player: Vec<PlayerStats>,
}

/// How many of `player`'s desired LOD0 cubes are not yet sent.
fn lod0_pending_of(player: &Player) -> usize {
    player
        .desired
        .iter()
        .filter(|offset| !player.sent.contains(&offset_pos(player.center, **offset)))
        .count()
}

/// How many of `player`'s desired ring cubes are not yet sent, summed over
/// levels (a cube is only ever sent once ready, so this covers "not ready
/// and sent").
fn lod_ring_pending_of(player: &Player) -> usize {
    player
        .lod
        .iter()
        .enumerate()
        .map(|(index, level_state)| {
            let level = index as u8 + 1;
            player
                .lod_walk
                .iter()
                .filter(|(l, _)| *l == level)
                .filter(|&(_, offset)| {
                    !level_state
                        .sent
                        .contains(&offset_pos(level_state.center, *offset))
                })
                .count()
        })
        .sum()
}

/// Per-level LOD ring cubes sent (index 0 = level 1), plus the LOD0 sent
/// count at the end.
fn sent_counts_per_level_of(player: &Player) -> Vec<usize> {
    let mut counts: Vec<usize> = player.lod.iter().map(|l| l.sent.len()).collect();
    counts.push(player.sent.len());
    counts
}

impl Server {
    /// One snapshot of every counter: what a debug panel renders and what
    /// the tests poll. Cheap enough to call per frame or per assertion.
    pub fn stats(&self) -> ServerStats {
        ServerStats {
            tick: self.tick_number,
            players: self.players().collect(),
            world_cube_count: self.world.len(),
            gen_in_flight: self.pool.in_flight(),
            lod: LodStats {
                in_flight: self.lod.in_flight(),
                internal_requests: self.lod.internal_requests(),
                samples: self.generator.lod_sample_calls(),
                cube_count: self.lod.total_cubes(),
                heap_bytes: self.lod.heap_bytes(),
            },
            spans: SpanStats {
                computed_lod0: self.spans.computed_lod0,
                computed_l1: self.spans.computed_l1,
                pending: self.spans.pending(),
                version: self.spans.version(),
            },
            saves: SaveStats {
                pinned_count: self.pinned_count(),
                saved_batches: self.saved_batches,
                failed_batches: self.failed_batches,
                unanswered_batches: self.sent_batches - self.saved_batches - self.failed_batches,
                pending_save_count: self.pending_save.len(),
            },
            per_player: self
                .connections
                .iter()
                .filter_map(Connection::joined_player)
                .map(|player| PlayerStats {
                    id: player.id,
                    sent_count: player.sent.len(),
                    sent_counts_per_level: sent_counts_per_level_of(player),
                    lod0_pending: lod0_pending_of(player),
                    lod_ring_pending: lod_ring_pending_of(player),
                })
                .collect(),
        }
    }

    /// Cubes the server currently keeps in memory.
    pub fn world_cube_count(&self) -> usize {
        self.world.len()
    }

    /// Approximate heap bytes of the resident LOD0 world cubes: each
    /// `Cube`'s [`Cube::heap_bytes`] plus the `Cube` struct itself (the
    /// map entry overhead is excluded). Diagnostic; task 2.13's
    /// `far_metrics` reports it next to the pyramid's `lod_heap_bytes`.
    pub fn world_heap_bytes(&self) -> usize {
        self.world
            .values()
            .map(|cube| cube.heap_bytes() + std::mem::size_of::<jbm_core::cube::Cube>())
            .sum()
    }

    /// The level-`level` LOD cube at `pos`, when the pyramid holds one
    /// (levels are 1-based; `lod_levels` bounds the top).
    pub fn lod_cube(&self, level: u8, pos: CubePos) -> Option<&jbm_core::cube::Cube> {
        self.lod.get(level, pos).map(|lc| &lc.cube)
    }

    /// The cube at `(level, pos)` when it is ready to stream — every
    /// octant real or filled. Test and streaming access.
    pub fn lod_ready(&self, level: u8, pos: CubePos) -> Option<&jbm_core::cube::Cube> {
        self.lod.ready(level, pos)
    }

    /// The column tops of the level-`level` LOD cube at `pos`, when the
    /// pyramid holds one (task 2.14).
    pub fn lod_tops(&self, level: u8, pos: CubePos) -> Option<&jbm_core::lod::LodTops> {
        self.lod.get(level, pos).map(|lc| &lc.tops)
    }

    /// Forwards to [`LodPyramid::request`]: queue `(level, pos)` for
    /// load-or-sample. Test and streaming access.
    pub fn request_lod(&mut self, level: u8, pos: CubePos) {
        self.lod.request(level, pos);
    }

    /// LOD jobs queued, running or finished-but-uncollected. The pyramid's
    /// own (internal) jobs are excluded, as [`LodPyramid::in_flight`]
    /// documents.
    pub fn lod_in_flight(&self) -> usize {
        self.lod.in_flight()
    }

    /// Internal (pending-propagation) LOD jobs the pyramid has queued in
    /// total — the load-report number for the default view.
    pub fn lod_internal_requests(&self) -> u64 {
        self.lod.internal_requests()
    }

    /// How many `sample_lod` sampling runs the generator has done — the
    /// test hook that proves storage reloads do not sample.
    pub fn lod_samples(&self) -> u64 {
        self.generator.lod_sample_calls()
    }

    /// The known-octant mask of the level-`level` LOD cube at `pos`, when
    /// the pyramid holds one.
    pub fn lod_known(&self, level: u8, pos: CubePos) -> Option<u8> {
        self.lod.get(level, pos).map(|lc| lc.known)
    }

    /// LOD cubes held across all levels.
    pub fn lod_cube_count(&self) -> usize {
        self.lod.total_cubes()
    }

    /// Approximate heap bytes of the LOD pyramid (see
    /// [`LodPyramid::heap_bytes`]); diagnostic, used by the LOD bench's
    /// default-view report.
    pub fn lod_heap_bytes(&self) -> usize {
        self.lod.heap_bytes()
    }

    /// The whole pyramid, for tests and diagnostics.
    pub fn lod_pyramid(&self) -> &LodPyramid {
        &self.lod
    }

    /// Surface-span store counters: columns whose exact LOD0 bounds and
    /// sampled level-1 bounds have been computed, jobs still queued, and
    /// the store's version (bumped per landed span).
    pub fn span_stats(&self) -> (u64, u64, usize, u64) {
        (
            self.spans.computed_lod0,
            self.spans.computed_l1,
            self.spans.pending(),
            self.spans.version(),
        )
    }

    /// Cubes queued for generation, generating, or generated but not yet
    /// collected.
    pub fn in_flight(&self) -> usize {
        self.pool.in_flight()
    }

    /// LOD0 streaming completeness: how many of the player's desired LOD0
    /// cubes are not yet sent. Zero once the whole surface-following
    /// desired set has streamed.
    pub fn lod0_pending(&self, id: PlayerId) -> usize {
        self.connections
            .iter()
            .filter_map(Connection::joined_player)
            .find(|player| player.id == id)
            .map_or(0, lod0_pending_of)
    }

    /// LOD ring completeness: how many of the player's desired ring cubes
    /// are not yet sent, summed over levels (a cube is only ever sent once
    /// ready, so this covers "not ready and sent"). Zero once every ring
    /// the pyramid can fill has streamed.
    pub fn lod_ring_pending(&self, id: PlayerId) -> usize {
        self.connections
            .iter()
            .filter_map(Connection::joined_player)
            .find(|player| player.id == id)
            .map_or(0, lod_ring_pending_of)
    }

    /// Cubes currently sent to (and not unloaded from) the player.
    pub fn sent_count(&self, id: PlayerId) -> usize {
        self.connections
            .iter()
            .filter_map(Connection::joined_player)
            .find(|player| player.id == id)
            .map_or(0, |player| player.sent.len())
    }

    /// Per-level LOD ring cubes sent to the player (index 0 = level 1),
    /// plus the LOD0 sent count at the end — the measurement hook (cubes
    /// desired per level).
    pub fn sent_counts_per_level(&self, id: PlayerId) -> Vec<usize> {
        self.connections
            .iter()
            .filter_map(Connection::joined_player)
            .find(|player| player.id == id)
            .map_or(Vec::new(), sent_counts_per_level_of)
    }

    /// Cubes pinned by an accepted edit (never evicted). Without
    /// persistence that is the edited set; with a world directory it is
    /// every cube whose edit version is not yet committed
    /// (`version > saved_version`), which includes cubes in a save batch
    /// that has not been acknowledged yet.
    pub fn pinned_count(&self) -> usize {
        match &self.storage {
            Some(_) => self
                .versions
                .values()
                .filter(|v| v.version > v.saved_version)
                .count(),
            None => self.pinned.len(),
        }
    }

    /// Cubes with uncommitted edits (same as [`Server::pinned_count`]).
    pub fn dirty_count(&self) -> usize {
        self.pinned_count()
    }

    /// Cubes whose snapshot is with the save thread, unacknowledged.
    pub fn pending_save_count(&self) -> usize {
        self.pending_save.len()
    }

    /// Save batches committed by the save thread so far.
    pub fn saved_batches(&self) -> u64 {
        self.saved_batches
    }

    /// Save batches the save thread reported failed so far (their cubes
    /// stayed pinned for a retry).
    pub fn failed_batches(&self) -> u64 {
        self.failed_batches
    }

    /// Batches handed to the save thread and not yet answered (committed or
    /// failed). Tests wait for 0 before holding saves.
    pub fn unanswered_batches(&self) -> u64 {
        self.sent_batches - self.saved_batches - self.failed_batches
    }

    /// The open world storage, when persistence is on. Test and diagnostic
    /// access (the crash tests read the database through it).
    pub fn storage(&self) -> Option<&Arc<Storage>> {
        self.storage.as_ref()
    }

    /// The authoritative block at `pos`, or `None` when its cube is not
    /// in the server world.
    pub fn block(&self, pos: BlockPos) -> Option<jbm_core::block::BlockStateId> {
        let (cube_pos, local) = pos.split();
        self.world.get(&cube_pos).map(|cube| cube.get(local))
    }
}

#[cfg(test)]
mod tests {
    use super::ServerStats;
    use crate::server::test_util::{joined_and_loaded, pending_hello_on, physics_server};

    /// The snapshot and the individual accessors must always agree.
    #[test]
    fn stats_matches_the_individual_accessors() {
        let (mut server, mut client) = pending_hello_on(physics_server());
        joined_and_loaded(&mut server, &mut client);
        let ServerStats {
            tick,
            world_cube_count,
            gen_in_flight,
            lod,
            saves,
            per_player,
            ..
        } = server.stats();
        assert_eq!(tick, server.tick_count());
        assert!(world_cube_count > 1, "terrain streamed");
        assert_eq!(gen_in_flight, server.in_flight());
        assert_eq!(lod.cube_count, server.lod_cube_count());
        assert_eq!(lod.samples, server.lod_samples());
        assert_eq!(saves.pinned_count, server.pinned_count());
        assert_eq!(saves.unanswered_batches, server.unanswered_batches());
        let [player] = per_player.as_slice() else {
            panic!("one joined player expected");
        };
        assert_eq!(player.lod0_pending, server.lod0_pending(player.id));
        assert_eq!(player.lod_ring_pending, server.lod_ring_pending(player.id));
        assert_eq!(
            player.sent_counts_per_level,
            server.sent_counts_per_level(player.id)
        );
        assert_eq!(
            player.sent_count,
            *player.sent_counts_per_level.last().unwrap(),
        );
    }
}
