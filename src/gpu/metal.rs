use std::collections::VecDeque;
use std::mem;

use anyhow::{anyhow, Result};
use metal::{
    CommandBuffer, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize, SharedEvent,
};

use crate::prefix::Prefix;
use crate::table;

use super::ed_step_table::{mesh_step_table, step_table};
use super::{Match, MeshSearcher, SearchParams, Searcher};

const SHADER: &str = include_str!("../kernels/ed25519.metal");

/// Hit slots per Meshtastic launch (`MAX_HITS_PER_LAUNCH` in the shader).
const MESH_MAX_SLOTS: usize = 64;

/// The system default Metal device's name, if there is one.
pub(super) fn device_name() -> Option<String> {
    Device::system_default().map(|d| d.name().to_string())
}

/// Batch-inversion window for the Metal kernels: candidates per thread whose
/// affine y comes out of one fe_invert. Injected as a preprocessor #define
/// at shader-compile time (the shader keeps an #ifndef default).
///
/// Per candidate the symmetric affine walk costs 1.5 fe_mul for the point
/// addition plus 4 fe_mul of batch inversion, and the window adds the
/// inversion's ~265 squarings / W. The two per-thread slot arrays (ts/ds,
/// W + 1 field elements each) live in thread-private (stack) memory,
/// 2 * (W + 1) * 40 bytes = 20.5 KB at 256. Measured on an M1 Max, 512 is
/// within ~1% of 256 and 1024 exceeds the GPU stack budget ("Compute
/// function exceeds available stack space"), so 256 is the first choice;
/// the candidates are tried largest first so a GPU with a smaller stack
/// budget still gets the best window it fits. The window must be even (the
/// walk pairs candidates around the window middle).
const METAL_BATCH_INV_WINDOWS: [u32; 4] = [256, 128, 64, 32];

/// Command buffers kept in flight per searcher. The host prepares the next
/// batch of base scalars and reads back results while the GPU is busy, so
/// the GPU never idles between launches even when the feeder thread is
/// competing with the CPU workers for a core (combined mode used to lose
/// ~8% of GPU throughput to that). Consecutive launches are serialized on
/// the GPU with an MTLEvent (see `encode_launch`).
const PIPELINE_DEPTH: usize = 3;

/// Compile the shader with the largest batch-inversion window the device's
/// pipeline creation accepts (the window arrays live on the GPU stack).
fn build_pipeline(
    device: &Device,
    kernel: &str,
    extra_defines: &str,
) -> Result<(ComputePipelineState, u32)> {
    let options = CompileOptions::new();
    let mut last_error = None;
    for &window in &METAL_BATCH_INV_WINDOWS {
        // Inject the batch-inversion window as a preprocessor macro. metal 0.33
        // exposes set_preprocessor_macros only as a raw NSDictionary with no
        // safe wrapper, so prepend an equivalent #define to the source (the
        // shader guards it with #ifndef). This is the same preprocessor path
        // and keeps the knob adjustable without touching the shader.
        let source = format!("#define BATCH_INV_WINDOW {window}\n{extra_defines}{SHADER}");
        let library = device
            .new_library_with_source(&source, &options)
            .map_err(|error| anyhow!("failed to compile Metal shader: {error}"))?;
        let function = library
            .get_function(kernel, None)
            .map_err(|error| anyhow!("failed to load Metal kernel {kernel}: {error}"))?;
        match device.new_compute_pipeline_state_with_function(&function) {
            Ok(pipeline) => return Ok((pipeline, window)),
            Err(error) => last_error = Some(error),
        }
    }
    Err(anyhow!(
        "failed to create Metal pipeline for {kernel}: {}",
        last_error.unwrap_or_default()
    ))
}

fn check_completed(command_buffer: &CommandBuffer, what: &str) -> Result<()> {
    if command_buffer.status() != MTLCommandBufferStatus::Completed {
        return Err(anyhow!(
            "{what} ended with status {:?}",
            command_buffer.status()
        ));
    }
    Ok(())
}

fn new_shared_buffer(device: &Device, bytes: usize) -> metal::Buffer {
    device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
}

/// One in-flight launch's private buffers for the MeshCore searcher.
struct SearchSlot {
    scalars: metal::Buffer,
    found: metal::Buffer,
    found_index: metal::Buffer,
    found_iter: metal::Buffer,
    debug_prefix: metal::Buffer,
}

pub(super) struct MetalSearcher {
    pipeline: ComputePipelineState,
    command_queue: metal::CommandQueue,
    params_buffer: metal::Buffer,
    basepoint_table_buffer: metal::Buffer,
    /// Affine step multiples 1*S..window*S (see `step_table`).
    step_table_buffer: metal::Buffer,
    slots: Vec<SearchSlot>,
    /// Serializes consecutive launches on the GPU (see `encode_launch`).
    event: SharedEvent,
    launch_seq: u64,
    /// Launches committed but not yet collected, oldest first: (slot, command buffer).
    in_flight: VecDeque<(usize, CommandBuffer)>,
    /// Slot the next submit() writes into.
    next_slot: usize,
    batch_size: usize,
    iters: u32,
    /// Batch-inversion window the pipeline was compiled with (read by tests).
    #[cfg_attr(not(test), allow(dead_code))]
    window: u32,
    /// The full 32-byte compressed point (y encoding plus x sign bit) the
    /// kernel computed for the first candidate of gid 0 on the last launch,
    /// as eight little-endian u32 words.
    last_debug_point: [u32; 8],
}

impl MetalSearcher {
    pub(super) fn new(batch_size: usize, iters: u32, prefix: &Prefix) -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow!("no Metal device found"))?;
        let (pipeline, window) = build_pipeline(&device, "search_kernel", "")?;
        let command_queue = device.new_command_queue();

        // Each thread does one full scalar mult then visits `iters`
        // candidates base + 8*k through the affine step-table windows (see
        // the walk comment in the shader), amortizing one inversion per
        // window.
        let params = SearchParams::new(batch_size, iters, prefix);

        let params_buffer = device.new_buffer_with_data(
            &params as *const SearchParams as *const _,
            mem::size_of::<SearchParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let slots = (0..PIPELINE_DEPTH)
            .map(|_| SearchSlot {
                scalars: new_shared_buffer(&device, batch_size * 8 * mem::size_of::<u32>()),
                found: new_shared_buffer(&device, mem::size_of::<u32>()),
                found_index: new_shared_buffer(&device, mem::size_of::<u32>()),
                found_iter: new_shared_buffer(&device, mem::size_of::<u32>()),
                debug_prefix: new_shared_buffer(&device, 8 * mem::size_of::<u32>()),
            })
            .collect();
        let basepoint_table = table::basepoint_table();
        let basepoint_table_buffer = device.new_buffer_with_data(
            basepoint_table.as_ptr() as *const _,
            std::mem::size_of_val(basepoint_table) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let step_table = step_table(window);
        let step_table_buffer = device.new_buffer_with_data(
            step_table.as_ptr() as *const _,
            (step_table.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            pipeline,
            command_queue,
            params_buffer,
            basepoint_table_buffer,
            step_table_buffer,
            slots,
            event: device.new_shared_event(),
            launch_seq: 0,
            in_flight: VecDeque::with_capacity(PIPELINE_DEPTH),
            next_slot: 0,
            batch_size,
            iters,
            window,
            last_debug_point: [0u32; 8],
        })
    }

    fn encode_launch(&mut self, slot_index: usize) -> CommandBuffer {
        let slot = &self.slots[slot_index];
        let command_buffer = self.command_queue.new_command_buffer().to_owned();
        // Queued launches must not overlap on the GPU: the slots are
        // independent buffers, so without a dependency the driver overlaps
        // consecutive command buffers, which measurably slows this kernel
        // (~15%: two resident grids' per-thread stack windows thrash the
        // cache). Chain them through an event: launch n waits for n-1.
        let seq = self.launch_seq;
        self.launch_seq += 1;
        if seq > 0 {
            command_buffer.encode_wait_for_event(&self.event, seq);
        }
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&slot.scalars), 0);
        encoder.set_buffer(1, Some(&self.params_buffer), 0);
        encoder.set_buffer(2, Some(&slot.found), 0);
        encoder.set_buffer(3, Some(&slot.found_index), 0);
        encoder.set_buffer(4, Some(&slot.found_iter), 0);
        encoder.set_buffer(5, Some(&slot.debug_prefix), 0);
        encoder.set_buffer(6, Some(&self.basepoint_table_buffer), 0);
        encoder.set_buffer(7, Some(&self.step_table_buffer), 0);

        let width = self.pipeline.thread_execution_width().max(1) as u64;
        let threads_per_group = MTLSize {
            width: width.min(256),
            height: 1,
            depth: 1,
        };
        let threads = MTLSize {
            width: self.batch_size as u64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_threads(threads, threads_per_group);
        encoder.end_encoding();
        command_buffer.encode_signal_event(&self.event, seq + 1);
        command_buffer.commit();
        command_buffer
    }
}

impl Searcher for MetalSearcher {
    fn search_words(&mut self, words: &[u32]) -> Result<Option<Match>> {
        self.submit(words)?;
        self.collect()
    }

    fn last_debug_point(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for (chunk, word) in bytes.chunks_exact_mut(4).zip(self.last_debug_point) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    fn pipeline_depth(&self) -> usize {
        PIPELINE_DEPTH
    }

    fn submit(&mut self, words: &[u32]) -> Result<()> {
        if words.len() != self.batch_size * 8 {
            return Err(anyhow!(
                "expected {} scalar words, got {}",
                self.batch_size * 8,
                words.len()
            ));
        }
        if self.in_flight.len() >= PIPELINE_DEPTH {
            return Err(anyhow!("Metal pipeline is full; collect() before submit()"));
        }
        let slot_index = self.next_slot;
        self.next_slot = (self.next_slot + 1) % PIPELINE_DEPTH;
        let slot = &self.slots[slot_index];

        unsafe {
            std::ptr::copy_nonoverlapping(
                words.as_ptr(),
                slot.scalars.contents() as *mut u32,
                words.len(),
            );
            *(slot.found.contents() as *mut u32) = 0;
            *(slot.found_index.contents() as *mut u32) = u32::MAX;
            *(slot.found_iter.contents() as *mut u32) = 0;
            std::ptr::write_bytes(slot.debug_prefix.contents() as *mut u32, 0, 8);
        }

        let command_buffer = self.encode_launch(slot_index);
        self.in_flight.push_back((slot_index, command_buffer));
        Ok(())
    }

    fn collect(&mut self) -> Result<Option<Match>> {
        let (slot_index, command_buffer) = self
            .in_flight
            .pop_front()
            .ok_or_else(|| anyhow!("Metal collect() without a pending submit()"))?;
        command_buffer.wait_until_completed();
        if let Err(error) = check_completed(&command_buffer, "Metal command") {
            // A failed launch never signals the chain event; release any
            // queued launches so nothing is left waiting on the GPU.
            self.event.set_signaled_value(u64::MAX);
            return Err(error);
        }
        let slot = &self.slots[slot_index];

        self.last_debug_point = unsafe {
            let ptr = slot.debug_prefix.contents() as *const u32;
            let mut point = [0u32; 8];
            for (i, value) in point.iter_mut().enumerate() {
                *value = *ptr.add(i);
            }
            point
        };
        let found = unsafe { *(slot.found.contents() as *const u32) };
        if found == 0 {
            return Ok(None);
        }

        let index = unsafe { *(slot.found_index.contents() as *const u32) as usize };
        if index >= self.batch_size {
            return Err(anyhow!("Metal returned out-of-range index {index}"));
        }

        let iter = unsafe { *(slot.found_iter.contents() as *const u32) };
        if iter >= self.iters {
            return Err(anyhow!("Metal returned out-of-range iteration {iter}"));
        }

        Ok(Some(Match { index, iter }))
    }
}

// ---------------------------------------------------------------------------
// Meshtastic node-ID mode (mesh_kernel): the same launch shape with the
// pattern's parameter block and CRC/bitmap tables, and up to 64 hits per
// launch.
// ---------------------------------------------------------------------------

/// One in-flight launch's private buffers for the Meshtastic searcher.
struct MeshSlot {
    scalars: metal::Buffer,
    found_count: metal::Buffer,
    found_index: metal::Buffer,
    found_iter: metal::Buffer,
    debug_u: metal::Buffer,
}

pub(super) struct MetalMeshSearcher {
    pipeline: ComputePipelineState,
    command_queue: metal::CommandQueue,
    params_buffer: metal::Buffer,
    tables_buffer: metal::Buffer,
    basepoint_table_buffer: metal::Buffer,
    step_table_buffer: metal::Buffer,
    slots: Vec<MeshSlot>,
    event: SharedEvent,
    launch_seq: u64,
    in_flight: VecDeque<(usize, CommandBuffer)>,
    next_slot: usize,
    batch_size: usize,
    iters: u32,
    #[cfg_attr(not(test), allow(dead_code))]
    window: u32,
    /// X25519 public key of gid 0's first candidate on the last collected
    /// launch, as eight little-endian words.
    last_debug_u: [u32; 8],
}

impl MetalMeshSearcher {
    pub(super) fn new(
        batch_size: usize,
        iters: u32,
        pattern: &crate::meshtastic::NodeIdPattern,
    ) -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow!("no Metal device found"))?;
        // MTVK_MESH_CRC_SLICE4=0 selects the bytewise CRC table walk
        // (measurement knob; slicing-by-4 is the default, see the shader).
        let slice4 = std::env::var("MTVK_MESH_CRC_SLICE4").map_or(1, |v| if v.trim() == "0" { 0 } else { 1 });
        let (pipeline, window) = build_pipeline(
            &device,
            "mesh_kernel",
            &format!("#define MAX_HITS_PER_LAUNCH {MESH_MAX_SLOTS}\n#define MESH_CRC_SLICE4 {slice4}\n"),
        )?;
        let command_queue = device.new_command_queue();

        let params = super::MeshParams::new(batch_size, iters, pattern);
        let params_buffer = device.new_buffer_with_data(
            &params as *const super::MeshParams as *const _,
            mem::size_of::<super::MeshParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let tables = pattern.gpu_tables();
        let tables_buffer = device.new_buffer_with_data(
            tables.as_ptr() as *const _,
            (tables.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let slots = (0..PIPELINE_DEPTH)
            .map(|_| MeshSlot {
                scalars: new_shared_buffer(&device, batch_size * 8 * mem::size_of::<u32>()),
                found_count: new_shared_buffer(&device, mem::size_of::<u32>()),
                found_index: new_shared_buffer(&device, MESH_MAX_SLOTS * mem::size_of::<u32>()),
                found_iter: new_shared_buffer(&device, MESH_MAX_SLOTS * mem::size_of::<u32>()),
                debug_u: new_shared_buffer(&device, 8 * mem::size_of::<u32>()),
            })
            .collect();
        let basepoint_table = table::basepoint_table();
        let basepoint_table_buffer = device.new_buffer_with_data(
            basepoint_table.as_ptr() as *const _,
            std::mem::size_of_val(basepoint_table) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        // The d*x*y step table: see mesh_kernel's operand bounds.
        let step_table = mesh_step_table(window);
        let step_table_buffer = device.new_buffer_with_data(
            step_table.as_ptr() as *const _,
            (step_table.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            pipeline,
            command_queue,
            params_buffer,
            tables_buffer,
            basepoint_table_buffer,
            step_table_buffer,
            slots,
            event: device.new_shared_event(),
            launch_seq: 0,
            in_flight: VecDeque::with_capacity(PIPELINE_DEPTH),
            next_slot: 0,
            batch_size,
            iters,
            window,
            last_debug_u: [0u32; 8],
        })
    }

    /// The X25519 public key the kernel computed for gid 0's first candidate
    /// on the last collected launch.
    #[cfg(test)]
    pub(super) fn last_debug_u(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for (chunk, word) in bytes.chunks_exact_mut(4).zip(self.last_debug_u) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    fn encode_launch(&mut self, slot_index: usize) -> CommandBuffer {
        let slot = &self.slots[slot_index];
        let command_buffer = self.command_queue.new_command_buffer().to_owned();
        let seq = self.launch_seq;
        self.launch_seq += 1;
        if seq > 0 {
            command_buffer.encode_wait_for_event(&self.event, seq);
        }
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&slot.scalars), 0);
        encoder.set_buffer(1, Some(&self.params_buffer), 0);
        encoder.set_buffer(2, Some(&self.tables_buffer), 0);
        encoder.set_buffer(3, Some(&slot.found_count), 0);
        encoder.set_buffer(4, Some(&slot.found_index), 0);
        encoder.set_buffer(5, Some(&slot.found_iter), 0);
        encoder.set_buffer(6, Some(&slot.debug_u), 0);
        encoder.set_buffer(7, Some(&self.basepoint_table_buffer), 0);
        encoder.set_buffer(8, Some(&self.step_table_buffer), 0);

        let width = self.pipeline.thread_execution_width().max(1) as u64;
        let threads_per_group = MTLSize {
            width: width.min(256),
            height: 1,
            depth: 1,
        };
        let threads = MTLSize {
            width: self.batch_size as u64,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads, threads_per_group);
        encoder.end_encoding();
        command_buffer.encode_signal_event(&self.event, seq + 1);
        command_buffer.commit();
        command_buffer
    }
}

impl MeshSearcher for MetalMeshSearcher {
    fn search_words(&mut self, words: &[u32]) -> Result<Vec<Match>> {
        self.submit(words)?;
        self.collect()
    }

    fn pipeline_depth(&self) -> usize {
        PIPELINE_DEPTH
    }

    fn submit(&mut self, words: &[u32]) -> Result<()> {
        if words.len() != self.batch_size * 8 {
            return Err(anyhow!(
                "expected {} scalar words, got {}",
                self.batch_size * 8,
                words.len()
            ));
        }
        if self.in_flight.len() >= PIPELINE_DEPTH {
            return Err(anyhow!("Metal pipeline is full; collect() before submit()"));
        }
        let slot_index = self.next_slot;
        self.next_slot = (self.next_slot + 1) % PIPELINE_DEPTH;
        let slot = &self.slots[slot_index];

        unsafe {
            std::ptr::copy_nonoverlapping(
                words.as_ptr(),
                slot.scalars.contents() as *mut u32,
                words.len(),
            );
            *(slot.found_count.contents() as *mut u32) = 0;
            std::ptr::write_bytes(slot.found_index.contents() as *mut u32, 0, MESH_MAX_SLOTS);
            std::ptr::write_bytes(slot.found_iter.contents() as *mut u32, 0, MESH_MAX_SLOTS);
            std::ptr::write_bytes(slot.debug_u.contents() as *mut u32, 0, 8);
        }

        let command_buffer = self.encode_launch(slot_index);
        self.in_flight.push_back((slot_index, command_buffer));
        Ok(())
    }

    fn collect(&mut self) -> Result<Vec<Match>> {
        let (slot_index, command_buffer) = self
            .in_flight
            .pop_front()
            .ok_or_else(|| anyhow!("Metal mesh collect() without a pending submit()"))?;
        command_buffer.wait_until_completed();
        if let Err(error) = check_completed(&command_buffer, "Metal mesh command") {
            self.event.set_signaled_value(u64::MAX);
            return Err(error);
        }
        let slot = &self.slots[slot_index];

        self.last_debug_u = unsafe {
            let ptr = slot.debug_u.contents() as *const u32;
            let mut words = [0u32; 8];
            for (i, value) in words.iter_mut().enumerate() {
                *value = *ptr.add(i);
            }
            words
        };
        let count = unsafe { *(slot.found_count.contents() as *const u32) } as usize;
        let count = count.min(MESH_MAX_SLOTS);
        let mut hits = Vec::with_capacity(count);
        unsafe {
            let indexes = slot.found_index.contents() as *const u32;
            let iters = slot.found_iter.contents() as *const u32;
            for i in 0..count {
                let index = *indexes.add(i) as usize;
                let iter = *iters.add(i);
                if index >= self.batch_size || iter >= self.iters {
                    return Err(anyhow!(
                        "Metal mesh returned out-of-range match ({index}, {iter})"
                    ));
                }
                hits.push(Match { index, iter });
            }
        }
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::MetalSearcher;
    use crate::gpu::Searcher;
    use crate::keygen::{derive_candidate, scalar_words_for_gpu};
    use crate::prefix::Prefix;

    /// Known scalar/pubkey pair: the self-test seed. The public key is
    /// computed by curve25519-dalek's compress(), i.e. the authoritative
    /// canonical 32-byte encoding including the x sign bit.
    fn known_key() -> ([u32; 8], [u8; 32]) {
        let seed = [42u8; 32];
        let candidate = derive_candidate(seed);
        let (words, _) = scalar_words_for_gpu(seed);
        (words, candidate.public_key)
    }

    /// The kernel's full 32-byte compressed point (bytes 0..31 including the
    /// sign bit in byte 31) must equal dalek's compress() byte-for-byte.
    #[test]
    fn gpu_debug_point_matches_dalek_compress() {
        let (words, public_key) = known_key();
        let prefix = Prefix::parse("ABCD1234").unwrap();
        let mut searcher = MetalSearcher::new(1, 1, &prefix).expect("Metal searcher");
        let _ = searcher.search_words(&words).expect("Metal search");
        assert_eq!(
            searcher.last_debug_point(),
            public_key,
            "GPU compressed point != dalek compress()"
        );
    }

    /// A one-thread, one-iter search must find the known key when the prefix is
    /// exactly its leading bytes, and reject a near-miss prefix.
    #[test]
    fn gpu_matches_known_prefix_and_rejects_near_miss() {
        let (words, public_key) = known_key();
        let hit = hex::encode_upper(&public_key[..4]);
        let prefix = Prefix::parse(&hit).unwrap();
        let mut searcher = MetalSearcher::new(1, 1, &prefix).expect("Metal searcher");
        let found = searcher.search_words(&words).expect("Metal search");
        assert_eq!(found.map(|m| (m.index, m.iter)), Some((0, 0)));

        // Flip the last requested nibble: must no longer match.
        let mut miss: Vec<u8> = hit.into_bytes();
        let last = miss.len() - 1;
        miss[last] = if miss[last] == b'0' { b'1' } else { b'0' };
        let miss = String::from_utf8(miss).unwrap();
        let prefix = Prefix::parse(&miss).unwrap();
        let mut searcher = MetalSearcher::new(1, 1, &prefix).expect("Metal searcher");
        let found = searcher.search_words(&words).expect("Metal search");
        assert_eq!(found.map(|m| (m.index, m.iter)), None);
    }

    /// The affine step table must hold exactly i*S (S = 8*B) for every window
    /// the pipeline builder may pick: every entry's (x, y) must be on the
    /// curve and match dalek's compressed point (y bytes plus x parity),
    /// 2d*x*y must be consistent, and the entries that coincide with the
    /// basepoint table's digit-0 multiples (8B..32B) must reproduce its limbs
    /// exactly (same encoding, independent construction).
    #[test]
    fn step_table_matches_dalek_and_basepoint_table() {
        use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
        use curve25519_dalek::scalar::Scalar;
        use num_bigint::BigUint;
        use num_traits::One;

        const LIMB_BITS: [u32; 10] = [26, 25, 26, 25, 26, 25, 26, 25, 26, 25];
        let p = (BigUint::one() << 255u32) - BigUint::from(19u32);
        let field = |limbs: &[u32]| -> BigUint {
            let mut value = BigUint::ZERO;
            let mut pos = 0u32;
            for (limb, bits) in limbs.iter().zip(LIMB_BITS) {
                assert!((*limb as u64) < (1u64 << bits), "limb {limb} exceeds {bits} bits");
                value |= BigUint::from(*limb) << pos;
                pos += bits;
            }
            value
        };
        let modinv = |a: &BigUint| a.modpow(&(&p - BigUint::from(2u32)), &p);
        // d = -121665 / 121666 mod p.
        let d = (&p - BigUint::from(121665u32)) * modinv(&BigUint::from(121666u32)) % &p;
        let basepoint = crate::table::basepoint_table();

        for window in super::METAL_BATCH_INV_WINDOWS {
            let table = super::step_table(window);
            let half = window / 2;
            assert_eq!(table.len(), (half as usize + 1) * 30);
            for entry in 0..=half {
                let multiple = if entry == half { window } else { entry + 1 };
                let words = &table[entry as usize * 30..entry as usize * 30 + 30];
                let x = field(&words[..10]);
                let y = field(&words[10..20]);
                let t2d = field(&words[20..30]);
                assert!(x < p && y < p && t2d < p);
                // On the curve: -x^2 + y^2 = 1 + d x^2 y^2.
                let x2 = &x * &x % &p;
                let y2 = &y * &y % &p;
                let lhs = (&p - &x2 + &y2) % &p;
                let rhs = (BigUint::one() + &d * &x2 % &p * &y2) % &p;
                assert_eq!(lhs, rhs, "window {window} entry {entry} not on curve");
                assert_eq!(
                    t2d,
                    BigUint::from(2u32) * &d % &p * &x % &p * &y % &p,
                    "window {window} entry {entry} 2dxy"
                );
                // Same point as dalek's (8*multiple)*B.
                let expected = (Scalar::from(8u64 * multiple as u64) * ED25519_BASEPOINT_POINT)
                    .compress()
                    .to_bytes();
                let mut y_bytes = expected;
                y_bytes[31] &= 0x7F;
                let mut y_le = y.to_bytes_le();
                y_le.resize(32, 0);
                assert_eq!(y_le, y_bytes.to_vec(), "window {window} entry {entry} y");
                let x_parity = (x.to_bytes_le()[0] & 1) as u8;
                assert_eq!(x_parity, expected[31] >> 7, "window {window} entry {entry} x sign");
                // Basepoint table digit 0 holds j*B for j = 1..32 in Niels form.
                let j = 8 * multiple;
                if j <= 32 {
                    let base = (j as usize - 1) * 30;
                    let ypx = field(&basepoint[base..base + 10]);
                    let ymx = field(&basepoint[base + 10..base + 20]);
                    assert_eq!((&y + &x) % &p, ypx, "y+x vs basepoint table, j={j}");
                    assert_eq!((&p + &y - &x) % &p, ymx, "y-x vs basepoint table, j={j}");
                    assert_eq!(
                        &words[20..30],
                        &basepoint[base + 20..base + 30],
                        "2dxy limbs vs basepoint table, j={j}"
                    );
                }
            }
        }
    }

    /// The walk must find a key at every window-boundary position: the last
    /// candidate of a full window, the first of the next, the final candidate
    /// of a short tail window, and deep into a multi-window walk.
    #[test]
    fn gpu_search_finds_keys_across_inversion_windows() {
        use crate::keygen::{candidate_from_scalar, scalar_to_words};
        let (_, clamped) = scalar_words_for_gpu([42u8; 32]);
        let words = scalar_to_words(&clamped);
        let w = MetalSearcher::new(1, 1, &Prefix::parse("ABCD").unwrap())
            .expect("Metal searcher")
            .window;
        assert!(w >= 32);
        for (iters, target_iter) in [
            (w + 1, w - 1),
            (w + 1, w),
            (w + 37, w + 36),
            (2 * w + 5, 2 * w + 4),
            (3 * w, 3 * w - 1),
            // Exactly one full window, no base update.
            (w, w - 1),
            (w, 0),
            // The window middle (the base point itself) in a full window, in
            // a short window that still reaches it, and the first candidate
            // of a short window that does not.
            (w, w / 2),
            (w / 2 + 1, w / 2),
            (w / 2, w / 2 - 1),
            (w / 2, 0),
            // First candidate after a base update, and the second update.
            (2 * w + 1, 2 * w),
            (3 * w + 2, 3 * w + 1),
        ] {
            let stepped = crate::keygen::scalar_add_u64(clamped, 8 * target_iter as u64);
            let pubkey = candidate_from_scalar(stepped, [0u8; 32]).public_key;
            // A reserved first byte (00/FF) is not a MeshCore key: skip.
            let Ok(prefix) = Prefix::parse(&hex::encode_upper(&pubkey[..4])) else {
                continue;
            };
            let mut searcher = MetalSearcher::new(1, iters, &prefix).expect("Metal searcher");
            let found = searcher.search_words(&words).expect("Metal search");
            assert_eq!(
                found.map(|m| (m.index, m.iter)),
                Some((0, target_iter)),
                "iters={iters} target_iter={target_iter}"
            );
        }
    }

    /// Pipelined submit/collect must return each launch's own result in
    /// order, with all slots in flight at once.
    #[test]
    fn gpu_pipeline_returns_results_in_order() {
        use crate::keygen::{candidate_from_scalar, scalar_add_u64, scalar_to_words};
        let depth = super::PIPELINE_DEPTH;
        let (_, clamped) = scalar_words_for_gpu([42u8; 32]);
        // Launch k places the key at iteration k (batch of 4, thread 2).
        let iters = depth as u32 + 1;
        let pubkey = candidate_from_scalar(clamped, [0u8; 32]).public_key;
        let prefix = Prefix::parse(&hex::encode_upper(&pubkey[..4])).unwrap();
        let mut searcher = MetalSearcher::new(4, iters, &prefix).expect("Metal searcher");
        let filler = scalar_to_words(&scalar_add_u64(clamped, 8 * (iters as u64 + 7)));
        for k in 0..depth as u32 {
            let mut words = vec![0u32; 4 * 8];
            for t in 0..4 {
                words[t * 8..t * 8 + 8].copy_from_slice(&filler);
            }
            // Base = clamped - 8k, so candidate k of thread 2 is the key.
            let mut base = clamped;
            let delta = 8 * k as u64;
            let mut borrow = delta;
            for byte in base.iter_mut() {
                let sub = (*byte as u64).wrapping_sub(borrow & 0xFF);
                *byte = sub as u8;
                borrow = (borrow >> 8) + ((sub >> 63) & 1);
            }
            words[16..24].copy_from_slice(&scalar_to_words(&base));
            searcher.submit(&words).expect("submit");
        }
        for k in 0..depth as u32 {
            let found = searcher.collect().expect("collect");
            assert_eq!(found.map(|m| (m.index, m.iter)), Some((2, k)), "launch {k}");
        }
        assert!(searcher.collect().is_err(), "collect() on an empty pipeline must fail");
    }
}

#[cfg(test)]
mod mesh_tests {
    use super::MetalMeshSearcher;
    use crate::gpu::MeshSearcher;
    use crate::keygen::{random_base_scalar, scalar_add_u64, scalar_to_words};
    use crate::meshtastic::{crc32_bitwise, derive, oracle_vectors, NodeIdPattern};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn words_of(bases: &[[u8; 32]]) -> Vec<u32> {
        bases.iter().flat_map(|b| scalar_to_words(b)).collect()
    }

    /// The kernel's u of every oracle secret (gid 0's debug word, one
    /// launch per secret) must equal the oracle's X25519 public key byte for
    /// byte, and with all 64 of a batch's node ids as the (fully fixed)
    /// targets every thread must report exactly iteration 0: the device's
    /// encoding, CRC-32 and target test agree with the reference on 200
    /// vectors.
    #[test]
    fn mesh_kernel_matches_oracle_vectors() {
        let vectors: Vec<_> = oracle_vectors()
            .into_iter()
            // The all-ones secret has no walk headroom (bit 253 fix-up); the
            // kernel is only ever handed random_base_scalar bases.
            .filter(|(s, _, _)| !s[8..31].iter().all(|b| *b == 0xFF))
            .collect();
        assert!(vectors.len() >= 190);
        for chunk in vectors.chunks(64) {
            let ids: Vec<String> = chunk.iter().map(|(_, _, id)| format!("{id:08x}")).collect();
            let pattern = NodeIdPattern::parse(&ids.join(",")).unwrap();
            assert_eq!(pattern.mask, u32::MAX);
            let bases: Vec<[u8; 32]> = chunk.iter().map(|(s, _, _)| *s).collect();
            let mut searcher = MetalMeshSearcher::new(bases.len(), 1, &pattern).expect("Metal mesh searcher");
            let hits = searcher.search_words(&words_of(&bases)).expect("mesh launch");
            let mut got: Vec<(usize, u32)> = hits.iter().map(|h| (h.index, h.iter)).collect();
            got.sort_unstable();
            let wanted: Vec<(usize, u32)> = (0..bases.len()).map(|i| (i, 0)).collect();
            assert_eq!(got, wanted);
            assert_eq!(searcher.last_debug_u(), chunk[0].1, "debug u of {}", hex::encode(chunk[0].0));
        }
        // Debug u for every vector individually (gid 0 of a one-thread launch).
        let pattern = NodeIdPattern::parse("0").unwrap();
        let mut searcher = MetalMeshSearcher::new(1, 1, &pattern).expect("Metal mesh searcher");
        for (secret, pubkey, id) in &vectors {
            let hits = searcher.search_words(&scalar_to_words(secret)).expect("mesh launch");
            assert_eq!(searcher.last_debug_u(), *pubkey, "u of {}", hex::encode(secret));
            assert_eq!(hits.len(), usize::from(pattern.matches(*id)), "hit set of {}", hex::encode(secret));
        }
    }

    /// Keys planted at every window-boundary position of a multi-window
    /// walk (plus random positions) across 64 threads: the kernel must
    /// report exactly the planted (thread, iter) set, nothing else, with a
    /// fully fixed 8-digit target per plant.
    #[test]
    fn mesh_kernel_reports_exact_planted_set_across_windows() {
        let probe = NodeIdPattern::parse("0").unwrap();
        let w = MetalMeshSearcher::new(1, 1, &probe).expect("Metal mesh searcher").window;
        assert!(w >= 32);
        let iters = 2 * w + 9;
        let batch = 64usize;
        let mut rng = ChaCha20Rng::from_seed([21u8; 32]);
        let bases: Vec<[u8; 32]> = (0..batch).map(|_| random_base_scalar(&mut rng)).collect();
        let mut planted: Vec<(usize, u32)> = vec![
            (0, 0), (1, w / 2 - 1), (2, w / 2), (3, w / 2 + 1), (4, w - 1), (5, w), (6, w + 1),
            (7, 2 * w - 1), (8, 2 * w), (9, 2 * w + 8), (10, iters - 1), (10, 0),
        ];
        while planted.len() < 60 {
            let p = ((rand_core::RngCore::next_u32(&mut rng) as usize) % batch, rand_core::RngCore::next_u32(&mut rng) % iters);
            if !planted.contains(&p) {
                planted.push(p);
            }
        }
        let targets: Vec<String> = planted
            .iter()
            .map(|&(i, iter)| format!("{:08x}", derive(&scalar_add_u64(bases[i], 8 * iter as u64)).1))
            .collect();
        let pattern = NodeIdPattern::parse(&targets.join(",")).unwrap();
        let mut expected: Vec<(usize, u32)> = planted.clone();
        // A random candidate also hitting one of <= 60 exact ids: 2^-26 each.
        expected.sort_unstable();
        expected.dedup();
        let mut searcher = MetalMeshSearcher::new(batch, iters, &pattern).expect("Metal mesh searcher");
        for _ in 0..2 {
            let hits = searcher.search_words(&words_of(&bases)).expect("mesh launch");
            let mut got: Vec<(usize, u32)> = hits.iter().map(|h| (h.index, h.iter)).collect();
            got.sort_unstable();
            assert_eq!(got, expected);
        }
    }

    /// Short patterns on random bases: every reported hit re-derives (dalek
    /// + bitwise CRC) to a node id matching the pattern, the walk visits
    /// each candidate exactly once (no duplicate reports), and the count is
    /// statistically consistent with 2^-(4*digits) per candidate.
    #[test]
    fn mesh_kernel_hits_rederive_and_rate_is_honest() {
        let mut rng = ChaCha20Rng::from_seed([22u8; 32]);
        let batch = 256usize;
        let iters = 1024u32;
        let candidates = (batch as u32 * iters) as f64;
        for (text, digits) in [("a", 1u32), ("dc", 2), ("!1*3", 2), ("dead,beef,cafe", 4)] {
            let pattern = NodeIdPattern::parse(text).unwrap();
            let bases: Vec<[u8; 32]> = (0..batch).map(|_| random_base_scalar(&mut rng)).collect();
            let mut searcher = MetalMeshSearcher::new(batch, iters, &pattern).expect("Metal mesh searcher");
            let hits = searcher.search_words(&words_of(&bases)).expect("mesh launch");
            let mut seen: Vec<(usize, u32)> = hits.iter().map(|h| (h.index, h.iter)).collect();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), hits.len(), "duplicate reports for {text}");
            for h in &hits {
                let secret = scalar_add_u64(bases[h.index], 8 * h.iter as u64);
                let (pk, id) = derive(&secret);
                assert_eq!(crc32_bitwise(&pk), id);
                assert!(pattern.matches(id), "{text}: hit ({}, {}) -> {id:08x} does not match", h.index, h.iter);
            }
            // Expected hits; the slot cap (64) truncates the 1-digit case.
            let expected = candidates * pattern.targets.len() as f64 / 16f64.powi(digits as i32);
            let cap = 64.0f64;
            if expected > cap {
                assert_eq!(hits.len(), 64, "{text}: slot cap");
            } else {
                let sigma = expected.sqrt();
                assert!(
                    (hits.len() as f64 - expected).abs() < 5.0 * sigma + 2.0,
                    "{text}: {} hits, expected {expected:.1} +- {sigma:.1}",
                    hits.len()
                );
            }
        }
    }

    /// Pipelined launches return their own hits in submission order.
    #[test]
    fn mesh_pipeline_returns_hits_in_order() {
        let depth = super::PIPELINE_DEPTH;
        let mut rng = ChaCha20Rng::from_seed([23u8; 32]);
        let base = random_base_scalar(&mut rng);
        // Launch k walks base + 8*16k for 16 iterations; its target is the
        // id at its iteration 3.
        let ids: Vec<String> = (0..depth as u64)
            .map(|k| format!("{:08x}", derive(&scalar_add_u64(base, 8 * (16 * k + 3))).1))
            .collect();
        let pattern = NodeIdPattern::parse(&ids.join(",")).unwrap();
        let mut searcher = MetalMeshSearcher::new(1, 16, &pattern).expect("Metal mesh searcher");
        for k in 0..depth as u64 {
            searcher.submit(&scalar_to_words(&scalar_add_u64(base, 8 * 16 * k))).expect("submit");
        }
        for k in 0..depth {
            let hits = searcher.collect().expect("collect");
            assert_eq!(hits.len(), 1, "launch {k}: {hits:?}");
            assert_eq!((hits[0].index, hits[0].iter), (0, 3), "launch {k}");
        }
        assert!(searcher.collect().is_err());
    }
}
