//! The world → mesh pool → coverage → terrain-upload pipeline, written
//! once. The app and the GPU tests all drive the same flow: server
//! messages enter through [`TerrainSync::apply`], dirty cubes are
//! re-meshed on the [`MeshPool`] and uploaded to the
//! [`TerrainPass`](crate::render::terrain::TerrainPass) within per-frame
//! budgets ([`TerrainSync::pump`]), and the coverage rule's verdicts are
//! applied to the renderer's meta slots. A synchronous
//! [`TerrainSync::settle_sync`] variant meshes inline (no pool round
//! trip) until the world is clean, for deterministic scenes.

use std::sync::Arc;

use glam::DVec3;
use jbm_core::protocol::ServerMsg;

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::lod::Coverage;
use crate::mesh::appearance::BlockAppearance;
use crate::mesh::pool::MeshPool;
use crate::render::terrain::TerrainPass;
use crate::world::{ClientWorld, WorldChange};

/// The client's terrain pipeline state: the cubes received from the
/// server, the mesh worker pool, the coverage rule and the in-flight
/// mesh-job count. The GPU-side pass is passed in by the caller (the app
/// has its window's pass, tests their offscreen one) — this struct owns
/// everything CPU-side.
pub struct TerrainSync {
    world: ClientWorld,
    pool: MeshPool,
    /// The coverage rule: which cube draws each volume. Fed by mesh
    /// uploads and unloads; its verdicts rewrite renderer metadata.
    coverage: Coverage,
    /// Mesh jobs submitted but not yet received back.
    in_flight: usize,
    /// The level >= 1 share of `in_flight`, for the overlay's LOD queue
    /// length (the pool itself does not break results down by level).
    in_flight_lod: usize,
}

impl TerrainSync {
    /// A fresh pipeline: empty world, a mesh pool over `appearance`
    /// (`threads` as in [`MeshPool::new`]), empty coverage.
    pub fn new(appearance: Arc<BlockAppearance>, threads: usize) -> TerrainSync {
        TerrainSync::with_world(ClientWorld::new(), appearance, threads)
    }

    /// The pipeline around an already-filled world (a streamed test
    /// scene). Same pool and coverage semantics as [`TerrainSync::new`].
    pub fn with_world(
        world: ClientWorld,
        appearance: Arc<BlockAppearance>,
        threads: usize,
    ) -> TerrainSync {
        TerrainSync {
            world,
            pool: MeshPool::new(appearance, threads),
            coverage: Coverage::default(),
            in_flight: 0,
            in_flight_lod: 0,
        }
    }

    /// The cubes received so far, plus re-mesh bookkeeping. Collision
    /// (`BlockShapes`), raycasts and optimistic edits read this.
    pub fn world(&self) -> &ClientWorld {
        &self.world
    }

    /// Mutable access for optimistic edits (`ClientWorld::set_block`).
    pub fn world_mut(&mut self) -> &mut ClientWorld {
        &mut self.world
    }

    /// Applies one server message to the world. On an unload the cube's
    /// GPU mesh and coverage readiness are dropped too; a load or edit
    /// just dirties cubes for the next [`TerrainSync::pump`]. Returns the
    /// cube-level change, if any (`None` when the message touched no
    /// cubes — callers handling their own protocol state ignore it).
    pub fn apply(
        &mut self,
        gpu: &Gpu,
        terrain: &mut TerrainPass,
        msg: &ServerMsg,
    ) -> Option<WorldChange> {
        match self.world.apply(msg) {
            WorldChange::None => None,
            WorldChange::Unloaded(key) => {
                terrain.remove(gpu, key);
                self.coverage.set_ready(key, false);
                Some(WorldChange::Unloaded(key))
            }
            change => Some(change),
        }
    }

    /// Submits dirty cubes (LOD0 and LOD, one budget — `submit_budget`
    /// per frame) to the pool and receives finished meshes, uploading at
    /// most `upload_budget` of them, so streaming never hitches a frame;
    /// leftover results stay queued for the next pump. Stale results (the
    /// cube changed again) are dropped. Every upload makes the cube
    /// coverage-ready, and the resulting draw-state changes are applied
    /// to the pass before the frame renders — meta is rewritten per
    /// change, never per frame. The camera orders the dirty queue
    /// (nearest first).
    pub fn pump(
        &mut self,
        gpu: &Gpu,
        terrain: &mut TerrainPass,
        camera: &Camera,
        submit_budget: usize,
        upload_budget: usize,
    ) {
        let inputs = self.world.take_dirty(submit_budget, camera.position);
        self.in_flight += inputs.len();
        for (key, version, input) in inputs {
            self.in_flight_lod += usize::from(key.level >= 1);
            self.pool.submit(key, version, input);
        }

        let mut uploads = 0;
        while uploads < upload_budget {
            let Some(result) = self.pool.try_recv() else {
                break;
            };
            self.in_flight -= 1;
            self.in_flight_lod -= usize::from(result.key.level >= 1);
            if self.world.version(result.key) != Some(result.version) {
                continue; // stale: a newer version exists and is queued
            }
            terrain.upload(gpu, result.key, &result.output);
            // An empty mesh counts as ready (uniform air, fully buried);
            // a re-mesh keeps the cube ready with its old mesh until this
            // one lands.
            self.coverage.set_ready(result.key, true);
            uploads += 1;
        }
        for (key, drawn, mask) in self.coverage.take_changed() {
            terrain.set_draw_state(gpu, key, drawn, mask);
        }
    }

    /// The synchronous variant: meshes everything dirty inline (no pool
    /// round trip) and uploads it, feeding the coverage rule, until the
    /// world is clean. For deterministic test scenes that arrive all at
    /// once. `camera` (a position is all the ordering needs) orders the
    /// dirty queue.
    pub fn settle_sync(
        &mut self,
        gpu: &Gpu,
        terrain: &mut TerrainPass,
        appearance: &BlockAppearance,
        camera: DVec3,
    ) {
        loop {
            let dirty = self.world.take_dirty(usize::MAX, camera);
            if dirty.is_empty() {
                break;
            }
            for (key, _version, input) in dirty {
                let output = crate::mesh::mesh(&input, appearance);
                terrain.upload(gpu, key, &output);
                // An empty mesh counts as ready.
                self.coverage.set_ready(key, true);
            }
            for (key, drawn, mask) in self.coverage.take_changed() {
                terrain.set_draw_state(gpu, key, drawn, mask);
            }
        }
    }

    /// No dirty cubes and no mesh jobs in flight.
    pub fn queue_clear(&self) -> bool {
        self.world.dirty_count() == 0 && self.in_flight == 0
    }

    /// Mesh jobs submitted but not yet received back (all levels).
    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    /// The level >= 1 share of the in-flight mesh jobs.
    pub fn in_flight_lod(&self) -> usize {
        self.in_flight_lod
    }
}
