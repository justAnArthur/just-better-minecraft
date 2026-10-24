//! GPU pass timing (task 2.16): a timestamp query at the start and the
//! end of each pass of a frame, read back without stalling, like the cull
//! stats — so the app's numbers are a few frames late, and the perf
//! harness waits for them (`read_blocking`).
//!
//! Only pass-boundary timestamps are used (`timestamp_writes` on the pass
//! descriptors, `Features::TIMESTAMP_QUERY`): Apple GPUs sample at stage
//! boundaries only, so a render pass spans its first vertex to its last
//! fragment. Passes can overlap on the GPU, so the per-pass times may add
//! up to more than [`GpuTimings::frame_ms`], the first start to the last
//! end. Without the feature the timer is never created and every pass
//! runs untimed.
//!
//! Each ring slot has its own query set, and a frame's queries are
//! resolved only once its submission has completed, in a later command
//! buffer: Metal does not order a counter resolve after the samples of
//! the same command buffer, and resolving there read stale or zero
//! end-of-pass values.

use std::cell::RefCell;
use std::sync::mpsc::Receiver;

use crate::gpu::Gpu;

/// The most passes one frame can time (two queries each); further passes
/// run untimed.
pub const MAX_TIMED_PASSES: u32 = 16;
const RING: usize = 4;

/// One frame's GPU times.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuTimings {
    /// Milliseconds per pass label in first-use order; a label used by
    /// several passes (the gpu-cull phases) holds their sum.
    pub passes: Vec<(&'static str, f64)>,
    /// The first pass's start to the last pass's end, in milliseconds.
    pub frame_ms: f64,
}

impl GpuTimings {
    /// The milliseconds recorded under `label`, 0 when absent.
    pub fn pass_ms(&self, label: &str) -> f64 {
        self.passes
            .iter()
            .find(|(l, _)| *l == label)
            .map_or(0.0, |(_, ms)| *ms)
    }
}

enum SlotState {
    Free,
    /// Written by the frame being recorded; armed after its submit.
    Written,
    /// Submitted; the receiver fires once the submission completed.
    Submitted(Receiver<()>),
    /// Complete on the GPU; resolved and copied by the next frame.
    Done,
    /// Resolved and copied this frame; the map is armed after the submit.
    Copied,
    Mapping(Receiver<Result<(), wgpu::BufferAsyncError>>),
}

struct Slot {
    query_set: wgpu::QuerySet,
    buffer: wgpu::Buffer,
    labels: Vec<&'static str>,
    /// The frame number the slot was written in: several maps can
    /// complete at once, and only the newest becomes `latest`.
    frame: u64,
    state: SlotState,
}

/// The per-frame pass timer. `begin_frame`, then one `render_writes` /
/// `compute_writes` per pass, `end_frame` before the submit and
/// `after_submit` after it.
pub struct GpuTimer {
    resolve: wgpu::Buffer,
    slots: Vec<Slot>,
    /// The slot this frame writes; `None` while the ring is full (the
    /// frame runs untimed).
    current: Option<usize>,
    /// This frame's pass labels, one per pass, in query order.
    labels: RefCell<Vec<&'static str>>,
    period_ns: f64,
    frame: u64,
    latest: Option<(u64, GpuTimings)>,
}

impl GpuTimer {
    /// A timer, or `None` when the device lacks `TIMESTAMP_QUERY`.
    pub fn new(gpu: &Gpu) -> Option<GpuTimer> {
        if !gpu
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return None;
        }
        let queries = 2 * MAX_TIMED_PASSES;
        let bytes = u64::from(queries) * 8;
        let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm timestamp resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = (0..RING)
            .map(|_| Slot {
                query_set: gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("jbm pass timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: queries,
                }),
                buffer: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jbm timestamp readback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                labels: Vec::new(),
                frame: 0,
                state: SlotState::Free,
            })
            .collect();
        Some(GpuTimer {
            resolve,
            slots,
            current: None,
            labels: RefCell::new(Vec::new()),
            period_ns: f64::from(gpu.queue.get_timestamp_period()),
            frame: 0,
            latest: None,
        })
    }

    /// Starts a frame: advances finished submissions and readbacks
    /// (non-blocking) and picks a free slot, if any.
    pub fn begin_frame(&mut self, gpu: &Gpu) {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        self.advance();
        self.labels.borrow_mut().clear();
        self.frame += 1;
        self.current = self
            .slots
            .iter()
            .position(|s| matches!(s.state, SlotState::Free));
    }

    /// The query set and first index of the next pass labelled `label`,
    /// or `None` when this frame is untimed or out of queries.
    fn next_pair(&self, label: &'static str) -> Option<(&wgpu::QuerySet, u32)> {
        let slot = self.current?;
        let mut labels = self.labels.borrow_mut();
        let index = labels.len() as u32;
        if index >= MAX_TIMED_PASSES {
            return None;
        }
        labels.push(label);
        Some((&self.slots[slot].query_set, 2 * index))
    }

    /// Timestamp writes for one render pass.
    pub fn render_writes(
        &self,
        label: &'static str,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.next_pair(label)
            .map(|(query_set, first)| wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(first),
                end_of_pass_write_index: Some(first + 1),
            })
    }

    /// Timestamp writes for one compute pass.
    pub fn compute_writes(
        &self,
        label: &'static str,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.next_pair(label)
            .map(|(query_set, first)| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(first),
                end_of_pass_write_index: Some(first + 1),
            })
    }

    /// Records this frame's labels and resolves the queries of earlier,
    /// completed frames into their readback buffers. Call once, after
    /// the last timed pass and before the submit.
    pub fn end_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.resolve_done(encoder);
        let Some(slot) = self.current.take() else {
            return;
        };
        let labels = std::mem::take(&mut *self.labels.borrow_mut());
        if labels.is_empty() {
            return;
        }
        self.slots[slot].labels = labels;
        self.slots[slot].frame = self.frame;
        self.slots[slot].state = SlotState::Written;
    }

    /// Arms the completion signal of this frame's slot and the maps of
    /// the slots resolved this frame. Call after the submit (a buffer may
    /// not be submitted while a map is pending).
    pub fn after_submit(&mut self, gpu: &Gpu) {
        for slot in &mut self.slots {
            match slot.state {
                SlotState::Written => {
                    let (sender, receiver) = std::sync::mpsc::channel();
                    gpu.queue.on_submitted_work_done(move || {
                        let _ = sender.send(());
                    });
                    slot.state = SlotState::Submitted(receiver);
                }
                SlotState::Copied => {
                    let (sender, receiver) = std::sync::mpsc::channel();
                    slot.buffer
                        .map_async(wgpu::MapMode::Read, .., move |result| {
                            let _ = sender.send(result);
                        });
                    slot.state = SlotState::Mapping(receiver);
                }
                _ => {}
            }
        }
    }

    /// The most recent frame's times read back so far (a few frames
    /// late), if any.
    pub fn latest(&self) -> Option<&GpuTimings> {
        self.latest.as_ref().map(|(_, timings)| timings)
    }

    /// Waits for every submitted frame and returns the newest times (the
    /// perf harness; the caller has submitted the frame). Resolves the
    /// completed frames in a command buffer of its own.
    pub fn read_blocking(&mut self, gpu: &Gpu) -> Option<GpuTimings> {
        let wait = |gpu: &Gpu| {
            let poll = gpu.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            assert!(poll.is_ok(), "device poll failed: {poll:?}");
        };
        wait(gpu);
        self.advance();
        if self
            .slots
            .iter()
            .any(|s| matches!(s.state, SlotState::Done))
        {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jbm timestamp resolve"),
                });
            self.resolve_done(&mut encoder);
            gpu.queue.submit([encoder.finish()]);
            self.after_submit(gpu);
            wait(gpu);
            self.advance();
        }
        self.latest().cloned()
    }

    /// Resolves and copies every completed slot.
    fn resolve_done(&mut self, encoder: &mut wgpu::CommandEncoder) {
        for slot in &mut self.slots {
            if !matches!(slot.state, SlotState::Done) {
                continue;
            }
            let queries = 2 * slot.labels.len() as u32;
            encoder.resolve_query_set(&slot.query_set, 0..queries, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(
                &self.resolve,
                0,
                &slot.buffer,
                0,
                u64::from(queries) * 8,
            );
            slot.state = SlotState::Copied;
        }
    }

    /// Moves completed submissions to `Done` and reads every completed
    /// map into `latest`, freeing its slot.
    fn advance(&mut self) {
        for i in 0..self.slots.len() {
            match &self.slots[i].state {
                SlotState::Submitted(receiver) => {
                    if receiver.try_recv().is_ok() {
                        self.slots[i].state = SlotState::Done;
                    }
                }
                SlotState::Mapping(receiver) => {
                    match receiver.try_recv() {
                        Ok(Ok(())) => {
                            let frame = self.slots[i].frame;
                            if self.latest.as_ref().is_none_or(|(f, _)| *f < frame) {
                                self.latest = Some((frame, self.read_slot(i)));
                            }
                        }
                        Ok(Err(err)) => eprintln!("timestamp map failed: {err}"),
                        Err(_) => continue, // not complete yet
                    }
                    self.slots[i].buffer.unmap();
                    self.slots[i].state = SlotState::Free;
                }
                _ => {}
            }
        }
    }

    fn read_slot(&self, slot: usize) -> GpuTimings {
        let slot = &self.slots[slot];
        let data = slot
            .buffer
            .slice(..)
            .get_mapped_range()
            .expect("timestamp buffer was just mapped");
        let ticks: Vec<u64> = data
            .as_chunks::<8>()
            .0
            .iter()
            .take(2 * slot.labels.len())
            .map(|c| u64::from_le_bytes(*c))
            .collect();
        drop(data);
        timings_from_ticks(&slot.labels, &ticks, self.period_ns)
    }
}

/// Folds a frame's begin/end tick pairs into per-label milliseconds and
/// the frame span, skipping pairs that were not sampled. Pure, so the
/// folding is unit-tested.
fn timings_from_ticks(labels: &[&'static str], ticks: &[u64], period_ns: f64) -> GpuTimings {
    let ms = |t: u64| t as f64 * period_ns * 1e-6;
    let mut timings = GpuTimings::default();
    let mut first = u64::MAX;
    let mut last = 0u64;
    for (label, pair) in labels.iter().zip(ticks.as_chunks::<2>().0) {
        let (begin, end) = (pair[0], pair[1]);
        // A pass the GPU did not sample (no work) reads zero.
        if begin == 0 || end < begin {
            continue;
        }
        first = first.min(begin);
        last = last.max(end);
        let pass = ms(end.saturating_sub(begin));
        match timings.passes.iter_mut().find(|(l, _)| l == label) {
            Some((_, total)) => *total += pass,
            None => timings.passes.push((label, pass)),
        }
    }
    timings.frame_ms = if last > first { ms(last - first) } else { 0.0 };
    timings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_fold_into_labels_and_a_frame_span() {
        let labels = ["terrain", "sky", "terrain"];
        // 1 tick = 1 µs.
        let ticks = [1_000, 3_000, 3_500, 4_000, 4_000, 7_000];
        let t = timings_from_ticks(&labels, &ticks, 1_000.0);
        assert_eq!(t.passes, vec![("terrain", 5.0), ("sky", 0.5)]);
        assert_eq!(t.frame_ms, 6.0);
        assert_eq!(t.pass_ms("sky"), 0.5);
        assert_eq!(t.pass_ms("missing"), 0.0);

        // An unsampled pair is skipped, not read as a huge span.
        let t = timings_from_ticks(&["cull", "scene"], &[0, 0, 2_000, 5_000], 1_000.0);
        assert_eq!(t.passes, vec![("scene", 3.0)]);
        assert_eq!(t.frame_ms, 3.0);
    }
}
