//! Terrain statistics: the per-frame `TerrainStats` snapshot, the
//! per-level `LevelStats`, the GPU culling counters (`CullStats`), and the
//! readback ring that brings the GPU counters home one or two frames
//! late.

/// u32 words in the GPU stats buffer: 8 frame counters (7 plus the
/// region-overflow word, asserted zero by tests), then 7 per-level drawn
/// counts and 7 per-level face counts (task 2.12; levels 0..=6, the last
/// entry collecting anything beyond).
pub(super) const STATS_WORDS: usize = 22;
pub(super) const STATS_BYTES: u64 = STATS_WORDS as u64 * 4;

/// GPU culling counters, filled by cull.wgsl and read back one or two
/// frames late (see `TerrainPass::cull_stats`). On the cpu-cull and
/// fallback paths the same fields are filled CPU-side, immediately.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CullStats {
    /// Active slots this frame.
    pub candidates: u32,
    /// Active slots the frustum test rejected.
    pub frustum_culled: u32,
    /// Frustum-passing slots the Hi-Z test rejected.
    pub occluded: u32,
    /// Slots drawn in the early phase (were visible last frame).
    pub drawn_early: u32,
    /// Slots drawn in the late phase (newly visible).
    pub drawn_late: u32,
    /// Indirect commands emitted (early + late).
    pub commands: u32,
    /// Faces issued, counting only the non-skipped directions.
    pub faces: u32,
    /// Commands the shader dropped because their command-region slot was
    /// at or beyond the buffer capacity: a bound/capacity bug (the CPU
    /// frustum bound must be >= what the GPU emits). Always zero; the
    /// tests assert it.
    pub region_overflow: u32,
    /// Cubes drawn this frame per level (task 2.12): index L = level L,
    /// the last entry collecting anything beyond 6.
    pub level_drawn: [u32; 7],
    /// Faces issued this frame per level, the same indexing.
    pub level_faces: [u32; 7],
}

impl CullStats {
    fn from_words(words: &[u32]) -> CullStats {
        let mut level_drawn = [0u32; 7];
        let mut level_faces = [0u32; 7];
        level_drawn.copy_from_slice(&words[8..15]);
        level_faces.copy_from_slice(&words[15..22]);
        CullStats {
            candidates: words[0],
            frustum_culled: words[1],
            occluded: words[2],
            drawn_early: words[3],
            drawn_late: words[4],
            commands: words[5],
            faces: words[6],
            region_overflow: words[7],
            level_drawn,
            level_faces,
        }
    }

    /// The drawn count for the overlay: early + late.
    pub fn drawn(&self) -> u32 {
        self.drawn_early + self.drawn_late
    }
}

/// Per-LOD-level residency and coverage statistics, for the overlay and
/// the metrics example.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LevelStats {
    pub level: u8,
    /// Cubes with an uploaded mesh (empty or not).
    pub resident: u32,
    /// Cubes the coverage rule draws.
    pub drawn: u32,
    /// Masked octants summed over drawn cubes (each cube 0..=8).
    pub masked_octants: u32,
    /// Faces held by resident meshes.
    pub faces: u64,
    /// Faces emitted under the skirt rule.
    pub skirts: u32,
    /// Cubes the cull actually drew this frame at this level (task 2.12;
    /// on the gpu-cull path, one or two frames late like the rest of
    /// [`CullStats`]).
    pub frame_drawn: u32,
    /// Faces the cull issued this frame at this level.
    pub frame_faces: u64,
}

/// Per-frame terrain draw statistics, for the overlay and the metrics
/// example.
#[derive(Debug, Clone, Copy, Default)]
pub struct TerrainStats {
    /// Cubes with an uploaded mesh (empty or not).
    pub meshed: usize,
    /// Cubes passing the frustum test with at least one face. On gpu-cull
    /// this (and `faces`, `commands`) comes from the GPU counters and is
    /// one or two frames late.
    pub drawn: usize,
    /// Faces issued this frame, counting only the non-skipped directions.
    pub faces: u64,
    /// Indirect commands (MDI) or fallback draws issued this frame; one per
    /// visible cube and non-empty, non-skipped face direction.
    pub commands: u64,
    /// Render-pass draw API calls: one per non-empty page and phase with
    /// MDI, one per command on the fallback path.
    pub draw_calls: u64,
    /// Face pages allocated so far (pages are never freed).
    pub pages: usize,
    /// GPU bytes across all pages.
    pub page_bytes: u64,
    /// Faces currently allocated across all pages.
    pub used_faces: u64,
}

/// A ring of three MAP_READ buffers the GPU stats are copied into, one per
/// frame; completed maps are read without blocking, so the visible stats
/// are one or two frames late.
pub(super) struct StatsRing {
    pub(super) buffers: Vec<wgpu::Buffer>,
    slots: Vec<SlotState>,
    cursor: usize,
    last_copied: Option<usize>,
    pub(super) latest: CullStats,
}

enum SlotState {
    /// Idle: no copy since the last read.
    Free,
    /// A copy was recorded this frame. The map cannot be armed until that
    /// submit is on the queue, so the next frame (or the blocking read)
    /// arms it — a buffer with a pending map may not appear in a submit.
    Copied,
    /// map_async armed; waiting for the callback.
    Pending(std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>),
}

impl StatsRing {
    pub(super) fn new(device: &wgpu::Device) -> StatsRing {
        let buffers = (0..3)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jbm cull stats readback"),
                    size: STATS_BYTES,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        StatsRing {
            buffers,
            slots: vec![SlotState::Free, SlotState::Free, SlotState::Free],
            cursor: 0,
            last_copied: None,
            latest: CullStats::default(),
        }
    }

    /// One frame's start: complete what maps are done, then arm the maps
    /// for the slots copied last frame (their submit is on the queue by
    /// now, so arming is legal and the buffer never meets a submit while
    /// mapped).
    pub(super) fn begin_frame(&mut self, gpu: &crate::gpu::Gpu) {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        for i in 0..self.slots.len() {
            if let SlotState::Pending(receiver) = &self.slots[i] {
                match receiver.try_recv() {
                    Ok(Ok(())) => {
                        self.latest = Self::read(&self.buffers[i]);
                        self.buffers[i].unmap();
                        self.slots[i] = SlotState::Free;
                    }
                    Ok(Err(err)) => {
                        eprintln!("cull stats map failed: {err}");
                        self.buffers[i].unmap();
                        self.slots[i] = SlotState::Free;
                    }
                    Err(_) => {} // map not complete yet; retry next frame
                }
            }
            if matches!(self.slots[i], SlotState::Copied) {
                let (sender, receiver) = std::sync::mpsc::channel();
                self.buffers[i].map_async(wgpu::MapMode::Read, .., move |result| {
                    let _ = sender.send(result);
                });
                self.slots[i] = SlotState::Pending(receiver);
            }
        }
    }

    fn read(buffer: &wgpu::Buffer) -> CullStats {
        let data = buffer
            .slice(..)
            .get_mapped_range()
            .expect("stats buffer was just mapped");
        let mut words = [0u32; STATS_WORDS];
        for (i, word) in words.iter_mut().enumerate() {
            *word = u32::from_le_bytes([
                data[i * 4],
                data[i * 4 + 1],
                data[i * 4 + 2],
                data[i * 4 + 3],
            ]);
        }
        drop(data);
        CullStats::from_words(&words)
    }

    /// The next free slot for this frame's copy, if any (a still-mapped
    /// ring is skipped and the stats simply lag longer).
    pub(super) fn take_slot(&mut self) -> Option<usize> {
        for i in 0..self.slots.len() {
            let slot = (self.cursor + i) % self.slots.len();
            if matches!(self.slots[slot], SlotState::Free) {
                self.cursor = (slot + 1) % self.slots.len();
                self.last_copied = Some(slot);
                self.slots[slot] = SlotState::Copied;
                return Some(slot);
            }
        }
        None
    }

    /// Waits for the most recent copied slot and reads it (tests, benches
    /// and the metrics example). The caller must have submitted the frame.
    pub(super) fn read_blocking(&mut self, gpu: &crate::gpu::Gpu) -> CullStats {
        let Some(slot) = self.last_copied else {
            return self.latest;
        };
        // Arm the map now if the frame that copied the slot has not been
        // followed by another frame yet (arming after the submit is legal);
        // then wait it out.
        if matches!(self.slots[slot], SlotState::Copied) {
            let (sender, receiver) = std::sync::mpsc::channel();
            self.buffers[slot].map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.send(result);
            });
            self.slots[slot] = SlotState::Pending(receiver);
        }
        if let SlotState::Pending(receiver) =
            std::mem::replace(&mut self.slots[slot], SlotState::Free)
        {
            let poll = gpu.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            assert!(poll.is_ok(), "device poll failed: {poll:?}");
            receiver
                .recv()
                .expect("map callback dropped without sending")
                .expect("stats buffer mapping failed");
            self.latest = Self::read(&self.buffers[slot]);
            self.buffers[slot].unmap();
        }
        self.latest
    }
}
