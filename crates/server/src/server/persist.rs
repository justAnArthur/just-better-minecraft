//! Persistence: the snapshot batches handed to the `jbm-save` thread, the
//! acknowledgements applied on the tick thread, and the flush that blocks
//! until everything dirty is committed.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use jbm_core::coords::CubePos;

use super::Server;
use crate::storage::{PlayerRecord, Storage};
use crate::tick::TICK_PERIOD;

/// How long [`Server::flush`] waits for the save thread before giving up
/// and logging loudly. Generous: a flush is a commit-to-disk, fsync
/// included.
const FLUSH_TIMEOUT: Duration = Duration::from_secs(30);

/// One snapshot batch handed to the save thread: cube clones with the edit
/// version they were taken at, LOD cube clones, and player records —
/// committed together, so LOD derived from an edit never lands on disk
/// without the LOD0 edit it came from.
#[derive(Default)]
pub(super) struct SaveBatch {
    cubes: Vec<(CubePos, jbm_core::cube::Cube, u64)>,
    lod: Vec<(u8, CubePos, jbm_core::cube::Cube, u8)>,
    players: Vec<(String, PlayerRecord)>,
}

impl SaveBatch {
    fn is_empty(&self) -> bool {
        self.cubes.is_empty() && self.lod.is_empty() && self.players.is_empty()
    }
}

/// The save thread's reply to a batch.
pub(super) enum SaveAck {
    /// The batch committed; these cube versions, LOD cubes and player
    /// records are now on disk.
    Committed(
        Vec<(CubePos, u64)>,
        Vec<(u8, CubePos)>,
        Vec<(String, PlayerRecord)>,
    ),
    /// The batch failed; these cubes stay dirty and retry next interval.
    Failed(Vec<CubePos>, Vec<(u8, CubePos)>),
}

/// Per-cube edit bookkeeping (persistence only): the edit version, bumped by
/// every accepted edit, and the last version known committed to disk.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct EditVersions {
    pub(super) version: u64,
    pub(super) saved_version: u64,
}

impl Server {
    /// Persistence slice of the tick, run right before phase 5: apply the
    /// save thread's acknowledgements, then every `save_interval` snapshot
    /// dirty cubes and LOD and hand them off. No-op without a world
    /// directory.
    pub(super) fn phase_save(&mut self) {
        if self.storage.is_none() {
            return;
        }
        self.drain_save_acks();
        self.time_since_save += TICK_PERIOD;
        if self.time_since_save < self.config.save_interval {
            return;
        }
        self.time_since_save = Duration::ZERO;
        let mut batch = SaveBatch::default();
        for (pos, edit) in &self.versions {
            // Pinned cubes whose current version is not already in flight.
            if edit.version > edit.saved_version
                && self.pending_save.get(pos) != Some(&edit.version)
                && let Some(cube) = self.world.get(pos)
            {
                batch.cubes.push((*pos, cube.clone(), edit.version));
            }
        }
        self.collect_lod_dirty(&mut batch);
        batch.players = self.player_snapshot();
        self.send_save_batch(batch);
    }

    /// Snapshots every LOD cube the pyramid marked dirty (ready cubes whose
    /// content changed) and not already in flight. A cube whose older save
    /// is still in flight is kept dirty, so the interval after that save's
    /// ack writes the newer content — dropping it here would lose it.
    fn collect_lod_dirty(&mut self, batch: &mut SaveBatch) {
        let dirty = self.lod.take_store_dirty();
        for (level, pos) in dirty {
            if self.lod_pending_save.contains(&(level, pos)) {
                self.lod.mark_store_dirty(&[(level, pos)]);
                continue;
            }
            if let Some(lc) = self.lod.get(level, pos)
                && lc.ready()
            {
                batch.lod.push((level, pos, lc.cube.clone(), lc.known));
            }
        }
    }

    /// Records of every joined player plus everyone who left since the last
    /// snapshot, keeping only those that differ from what is on disk.
    fn player_snapshot(&mut self) -> Vec<(String, PlayerRecord)> {
        let mut records: HashMap<String, PlayerRecord> = self.departed.drain(..).collect();
        for conn in &self.connections {
            if let Some(player) = conn.joined_player() {
                records.insert(player.name.clone(), player.record());
            }
        }
        let mut changed: Vec<(String, PlayerRecord)> = records
            .into_iter()
            .filter(|(name, record)| self.saved_players.get(name) != Some(record))
            .collect();
        changed.sort_by(|a, b| a.0.cmp(&b.0));
        changed
    }

    /// Applies every reply that has arrived from the save thread.
    pub(super) fn drain_save_acks(&mut self) {
        while let Ok(ack) = self.ack_rx.try_recv() {
            self.apply_save_ack(ack);
        }
    }

    fn apply_save_ack(&mut self, ack: SaveAck) {
        match ack {
            SaveAck::Committed(acks, lod, players) => {
                self.saved_batches += 1;
                self.saved_players.extend(players);
                for (pos, version) in acks {
                    let edit = self.versions.entry(pos).or_default();
                    edit.saved_version = edit.saved_version.max(version);
                    if self
                        .pending_save
                        .get(&pos)
                        .is_some_and(|&p| p <= edit.saved_version)
                    {
                        self.pending_save.remove(&pos);
                    }
                }
                for key in lod {
                    self.lod_pending_save.remove(&key);
                }
            }
            // A failed batch is logged and retried next interval: the cubes
            // stay dirty (never evicted), so edits are never dropped
            // silently. The batch's LOD cubes go back to the dirty set the
            // same way.
            // Players are re-sent next interval: `saved_players` only moves
            // on a commit, so their records still differ.
            SaveAck::Failed(positions, lod) => {
                self.failed_batches += 1;
                eprintln!(
                    "jbm: saving a batch of {} cubes and {} LOD cubes failed; retrying next interval",
                    positions.len(),
                    lod.len()
                );
                for pos in positions {
                    self.pending_save.remove(&pos);
                }
                // Only this batch's keys: other batches may still be in
                // flight behind it, and forgetting theirs would let a later
                // interval snapshot them again while their save still runs.
                for key in &lod {
                    self.lod_pending_save.remove(key);
                }
                self.lod.mark_store_dirty(&lod);
            }
        }
    }

    /// Hands a snapshot batch to the save thread and marks its versions
    /// in flight. A dead save thread is logged loudly — those edits are
    /// only protected by the retry on the next interval.
    fn send_save_batch(&mut self, batch: SaveBatch) {
        if batch.is_empty() {
            return;
        }
        for (pos, _, version) in &batch.cubes {
            self.pending_save.insert(*pos, *version);
        }
        self.lod_pending_save
            .extend(batch.lod.iter().map(|&(level, pos, _, _)| (level, pos)));
        match &self.save_tx {
            Some(tx) if tx.send(batch).is_ok() => self.sent_batches += 1,
            _ => eprintln!("jbm: the save thread is gone; unsaved edits may be lost"),
        }
    }

    /// Sends everything still dirty and blocks until the save thread has
    /// acknowledged it (or [`FLUSH_TIMEOUT`] passes, which is logged).
    /// Called by [`ServerHandle::stop`](crate::run::ServerHandle::stop)
    /// paths and by `Drop`, so every accepted edit is persisted by the time
    /// the server is gone.
    pub fn flush(&mut self) {
        if self.storage.is_none() {
            return;
        }
        self.drain_save_acks();
        let mut batch = SaveBatch::default();
        for (pos, edit) in &self.versions {
            if edit.version > edit.saved_version
                && let Some(cube) = self.world.get(pos)
            {
                batch.cubes.push((*pos, cube.clone(), edit.version));
            }
        }
        self.collect_lod_dirty(&mut batch);
        batch.players = self.player_snapshot();
        self.send_save_batch(batch);
        // The saver answers batches in order, so once every batch sent so
        // far is answered, this one (which may hold only player records,
        // with no pending cube to mark it) is too.
        let deadline = Instant::now() + FLUSH_TIMEOUT;
        while self.saved_batches + self.failed_batches < self.sent_batches {
            let timeout = deadline.saturating_duration_since(Instant::now());
            match self.ack_rx.recv_timeout(timeout) {
                Ok(ack) => self.apply_save_ack(ack),
                Err(RecvTimeoutError::Timeout) => {
                    eprintln!(
                        "jbm: flush timed out with {} cubes unsaved",
                        self.pending_save.len()
                    );
                    break;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
    }
}

/// The `jbm-save` thread's loop: take a snapshot batch, commit it as one
/// transaction, reply with acknowledgements. A failed commit is reported,
/// never swallowed — the tick thread keeps those cubes pinned and retries.
pub(super) fn save_loop(
    storage: Arc<Storage>,
    batches: Receiver<SaveBatch>,
    acks: Sender<SaveAck>,
) {
    while let Ok(batch) = batches.recv() {
        let refs: Vec<(CubePos, &jbm_core::cube::Cube)> = batch
            .cubes
            .iter()
            .map(|(pos, cube, _)| (*pos, cube))
            .collect();
        let lod_refs: Vec<(u8, CubePos, &jbm_core::cube::Cube, u8)> = batch
            .lod
            .iter()
            .map(|&(level, pos, ref cube, known)| (level, pos, cube, known))
            .collect();
        let lod_keys: Vec<(u8, CubePos)> = batch
            .lod
            .iter()
            .map(|&(level, pos, _, _)| (level, pos))
            .collect();
        let ack = match storage.save_lod_batch(&refs, &lod_refs, &batch.players) {
            Ok(()) => SaveAck::Committed(
                batch
                    .cubes
                    .iter()
                    .map(|(pos, _, version)| (*pos, *version))
                    .collect(),
                lod_keys,
                batch.players,
            ),
            Err(error) => {
                eprintln!("jbm: committing a save batch failed: {error}");
                SaveAck::Failed(refs.into_iter().map(|(pos, _)| pos).collect(), lod_keys)
            }
        };
        if acks.send(ack).is_err() {
            break; // the server is gone
        }
    }
}
