//! GPU backends (Apple Metal, NVIDIA CUDA) and the host-side worker loops
//! that feed them.
//!
//! Both flows run the same kernel shape: each GPU thread computes one fixed-
//! base scalar multiplication for a CSPRNG-chosen clamped base scalar, then
//! walks `iters` candidates `base + 8k` through an affine step-table window
//! walk with one batched field inversion per window. The MeshCore kernel
//! (`search_kernel`) compares the canonical Ed25519 public key against a hex
//! prefix; the Meshtastic kernel (`mesh_kernel`) converts every point to its
//! X25519 public key, CRC-32s it and tests the node number against the
//! pattern. The host re-derives every reported `(base, iter)` with
//! curve25519-dalek before anything is reported.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use anyhow::{anyhow, Result};
use clap::ValueEnum;
use crossbeam_channel::Sender;
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;

use crate::keygen::{
    candidate_from_scalar, derive_candidate, fill_base_scalars, scalar_add_u64,
    scalar_words_for_gpu, FoundKey,
};
use crate::meshtastic::{MeshKey, NodeIdPattern, MAX_TARGETS};
use crate::prefix::Prefix;
use crate::Counters;

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
/// Host-C++ differential tests of src/kernels/ed25519.cu (no GPU needed).
#[cfg(test)]
mod cuda_host_tests;
/// Affine step tables of the window walk (shared by both backends).
pub(crate) mod ed_step_table;
#[cfg(target_os = "macos")]
mod metal;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Metal,
    Cuda,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Self::Metal => "Metal",
            Self::Cuda => "CUDA",
        }
    }

    pub fn source_name(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

/// Parameter block of the MeshCore kernel (`SearchParams` in the kernels).
#[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))] // GPU-only
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct SearchParams {
    pub count: u32,
    pub prefix_nibbles: u32,
    /// Low 4 prefix bytes (bytes 0..4), little-endian.
    pub prefix_bytes: u32,
    /// Candidates checked per thread; candidate k is `base_scalar + 8*k`.
    pub iters: u32,
    /// Prefix bytes 4..8 (always zero at this crate's 8-character cap; the
    /// kernels' fast path compares up to 16 hex characters).
    pub prefix_bytes_hi: u32,
    pub _pad: u32,
    /// All prefix bytes packed little-endian, byte i in word i/4.
    pub prefix_full: [u32; 8],
}

#[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))] // GPU-only
impl SearchParams {
    fn new(batch_size: usize, iters: u32, prefix: &Prefix) -> Self {
        let packed = prefix.packed_u64();
        SearchParams {
            count: batch_size as u32,
            prefix_nibbles: prefix.nibbles() as u32,
            prefix_bytes: packed as u32,
            iters,
            prefix_bytes_hi: (packed >> 32) as u32,
            _pad: 0,
            prefix_full: prefix.packed_words(),
        }
    }
}

/// Parameter block of the Meshtastic kernel (`MeshParams` in the kernels,
/// 68 words): the masked node-number targets.
#[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))] // GPU-only
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct MeshParams {
    pub count: u32,
    pub iters: u32,
    pub mask: u32,
    pub target_count: u32,
    pub targets: [u32; MAX_TARGETS],
}

#[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))] // GPU-only
impl MeshParams {
    pub(super) fn new(batch_size: usize, iters: u32, pattern: &NodeIdPattern) -> Self {
        MeshParams {
            count: batch_size as u32,
            iters,
            mask: pattern.mask,
            target_count: pattern.targets.len() as u32,
            targets: pattern.gpu_targets(),
        }
    }
}

/// A GPU match: base scalar `index` in the uploaded batch, matched at the
/// thread's iteration `iter` (i.e. scalar = base + 8*iter).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Match {
    pub index: usize,
    pub iter: u32,
}

/// The MeshCore (prefix) kernel: at most one match per launch.
pub(crate) trait Searcher {
    fn search_words(&mut self, words: &[u32]) -> Result<Option<Match>>;

    /// The canonical compressed public key of gid 0's first candidate on
    /// the last launch (the kernel's self-check word, compared with dalek).
    fn last_debug_point(&self) -> [u8; 32];

    /// Launches this backend can keep in flight at once (1 = synchronous).
    fn pipeline_depth(&self) -> usize {
        1
    }

    /// Queue one batch without waiting. Only called when `pipeline_depth()`
    /// is greater than one, and never with more than that many launches
    /// outstanding.
    fn submit(&mut self, _words: &[u32]) -> Result<()> {
        Err(anyhow!("this GPU backend does not pipeline launches"))
    }

    /// Wait for the oldest queued launch and return its match, if any.
    fn collect(&mut self) -> Result<Option<Match>> {
        Err(anyhow!("this GPU backend does not pipeline launches"))
    }
}

/// The Meshtastic (node-ID) kernel: every pattern match of a launch (up to
/// the kernel's 64 hit slots).
pub(crate) trait MeshSearcher {
    fn search_words(&mut self, words: &[u32]) -> Result<Vec<Match>>;

    fn pipeline_depth(&self) -> usize {
        1
    }

    fn submit(&mut self, _words: &[u32]) -> Result<()> {
        Err(anyhow!("this GPU backend does not pipeline launches"))
    }

    fn collect(&mut self) -> Result<Vec<Match>> {
        Err(anyhow!("this GPU backend does not pipeline launches"))
    }
}

// ---------------------------------------------------------------------------
// Backend selection and defaults.
// ---------------------------------------------------------------------------

pub fn default_backend() -> Result<Backend> {
    #[cfg(target_os = "macos")]
    {
        return Ok(Backend::Metal);
    }

    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    {
        return Ok(Backend::Cuda);
    }

    #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
    {
        Err(anyhow!(
            "no GPU backend is built in on this target; build with --features cuda for NVIDIA GPUs or use --cpu-only"
        ))
    }
}

pub fn ensure_supported(backend: Backend) -> Result<()> {
    match backend {
        Backend::Metal => {
            if cfg!(target_os = "macos") {
                Ok(())
            } else {
                Err(anyhow!("Metal backend is only available on macOS"))
            }
        }
        Backend::Cuda => {
            if cfg!(feature = "cuda") {
                Ok(())
            } else {
                Err(anyhow!(
                    "CUDA backend was not built; rebuild with cargo build --features cuda"
                ))
            }
        }
    }
}

/// `--gpu-iters` default for Metal: the system watchdog caps command-buffer
/// duration, so launches stay short.
pub const METAL_DEFAULT_GPU_ITERS: u32 = 1024;

/// `--gpu-iters` default for CUDA. One host thread feeds the GPU `--gpu-batch`
/// fresh base scalars per launch; at 16384 candidates per thread a launch
/// lasts tens of milliseconds, long enough to hide the host fill and short
/// enough for prompt stops.
pub const CUDA_DEFAULT_GPU_ITERS: u32 = 16384;

/// `--gpu-batch` default for Metal (and CUDA when the device query fails).
pub const DEFAULT_GPU_BATCH: usize = 32_768;

/// Full waves of 256-thread blocks the CUDA default batch covers (the walk
/// kernels run one block per SM; two waves measured best on an RTX 5090).
const CUDA_DEFAULT_WAVES: usize = 2;

/// The GPU walk length per launch: the user's value when given, else the
/// backend's default.
pub fn gpu_iters(backend: Backend, requested: Option<u32>) -> u32 {
    requested.unwrap_or(match backend {
        Backend::Cuda => CUDA_DEFAULT_GPU_ITERS,
        Backend::Metal => METAL_DEFAULT_GPU_ITERS,
    })
}

/// The GPU batch (threads per launch): the user's value when given, else
/// the backend's default. On CUDA that is a whole number of waves of the
/// kernel (SM count x block size x waves, queried from the device; the
/// plain default if the query fails).
pub fn gpu_batch(backend: Backend, requested: Option<usize>) -> usize {
    if let Some(batch) = requested {
        return batch;
    }
    #[cfg(feature = "cuda")]
    if backend == Backend::Cuda {
        if let Ok(sms) = cuda::multiprocessor_count() {
            return sms as usize * cuda::block_size() as usize * CUDA_DEFAULT_WAVES;
        }
    }
    let _ = (backend, CUDA_DEFAULT_WAVES);
    DEFAULT_GPU_BATCH
}

fn validate_iters(iters: u32) -> Result<()> {
    if iters == 0 {
        Err(anyhow!("--gpu-iters must be at least 1"))
    } else {
        Ok(())
    }
}

fn validate_batch_size(batch_size: usize) -> Result<()> {
    if batch_size == 0 || batch_size > u32::MAX as usize {
        Err(anyhow!("--gpu-batch must be between 1 and {}", u32::MAX))
    } else {
        Ok(())
    }
}

/// A human-readable description of the GPU device a backend would use
/// (for `--self-test` and the server's health report).
pub fn device_name(backend: Backend) -> Option<String> {
    match backend {
        Backend::Metal => {
            #[cfg(target_os = "macos")]
            {
                metal::device_name()
            }
            #[cfg(not(target_os = "macos"))]
            {
                None
            }
        }
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                cuda::device_name().ok()
            }
            #[cfg(not(feature = "cuda"))]
            {
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Searcher construction.
// ---------------------------------------------------------------------------

pub(crate) fn new_searcher(
    backend: Backend,
    batch_size: usize,
    iters: u32,
    prefix: &Prefix,
) -> Result<Box<dyn Searcher>> {
    match backend {
        Backend::Metal => {
            #[cfg(target_os = "macos")]
            {
                Ok(Box::new(metal::MetalSearcher::new(batch_size, iters, prefix)?))
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (batch_size, iters, prefix);
                Err(anyhow!("Metal backend is only available on macOS"))
            }
        }
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok(Box::new(cuda::CudaSearcher::new(batch_size, iters, prefix)?))
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (batch_size, iters, prefix);
                Err(anyhow!("CUDA backend was not built; rebuild with cargo build --features cuda"))
            }
        }
    }
}

pub(crate) fn new_mesh_searcher(
    backend: Backend,
    batch_size: usize,
    iters: u32,
    pattern: &NodeIdPattern,
) -> Result<Box<dyn MeshSearcher>> {
    match backend {
        Backend::Metal => {
            #[cfg(target_os = "macos")]
            {
                Ok(Box::new(metal::MetalMeshSearcher::new(batch_size, iters, pattern)?))
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (batch_size, iters, pattern);
                Err(anyhow!("Metal backend is only available on macOS"))
            }
        }
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok(Box::new(cuda::CudaMeshSearcher::new(batch_size, iters, pattern)?))
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (batch_size, iters, pattern);
                Err(anyhow!("CUDA backend was not built; rebuild with cargo build --features cuda"))
            }
        }
    }
}

/// Build the backend objects without running them, to surface device or
/// compile errors before any worker starts.
pub fn preflight_meshcore(backend: Backend, batch_size: usize, iters: u32, prefix: &Prefix) -> Result<()> {
    ensure_supported(backend)?;
    validate_batch_size(batch_size)?;
    validate_iters(iters)?;
    let _ = new_searcher(backend, batch_size, iters, prefix)?;
    Ok(())
}

pub fn preflight_meshtastic(backend: Backend, batch_size: usize, iters: u32, pattern: &NodeIdPattern) -> Result<()> {
    ensure_supported(backend)?;
    validate_batch_size(batch_size)?;
    validate_iters(iters)?;
    let _ = new_mesh_searcher(backend, batch_size, iters, pattern)?;
    Ok(())
}

/// One-key correctness check of both kernels on `backend`: a known scalar
/// must match its own public-key prefix at (0, 0) and its own node id, and
/// the kernels' self-check words must equal dalek's derivation.
pub fn self_test(backend: Backend) -> Result<()> {
    ensure_supported(backend)?;

    let seed = [42u8; 32];
    let candidate = derive_candidate(seed);
    let prefix = Prefix::parse(&hex::encode_upper(&candidate.public_key[..4]))?;
    let (words, clamped) = scalar_words_for_gpu(seed);
    let mut searcher = new_searcher(backend, 1, 1, &prefix)?;
    let result = searcher.search_words(&words)?;
    if searcher.last_debug_point() != candidate.public_key {
        return Err(anyhow!(
            "{} GPU public key {} differs from dalek's {}",
            backend.name(),
            hex::encode_upper(searcher.last_debug_point()),
            hex::encode_upper(candidate.public_key)
        ));
    }
    match result {
        Some(Match { index: 0, iter: 0 }) => {}
        Some(Match { index, iter }) => {
            return Err(anyhow!(
                "{} GPU reported an unexpected MeshCore match (index {index}, iter {iter})",
                backend.name()
            ))
        }
        None => {
            return Err(anyhow!(
                "{} GPU failed to match the known MeshCore key for prefix {}",
                backend.name(),
                prefix.text()
            ))
        }
    }

    let (_, node_id) = crate::meshtastic::derive(&clamped);
    let pattern = NodeIdPattern::parse(&format!("{node_id:08x}"))?;
    let mut mesh = new_mesh_searcher(backend, 1, 1, &pattern)?;
    let hits = mesh.search_words(&words)?;
    if hits != vec![Match { index: 0, iter: 0 }] {
        return Err(anyhow!(
            "{} GPU failed to match the known Meshtastic node id {}: got {hits:?}",
            backend.name(),
            pattern.text()
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Worker threads.
// ---------------------------------------------------------------------------

/// Spawn the GPU worker of the MeshCore search; it stops at the first
/// verified match (sent on `tx`) or when `stop` is raised.
pub fn spawn_meshcore_worker(
    backend: Backend,
    batch_size: usize,
    iters: u32,
    prefix: Arc<Prefix>,
    stop: Arc<AtomicBool>,
    counters: Arc<Counters>,
    tx: Sender<FoundKey>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name(format!("{}-meshcore", backend.source_name()))
        .spawn(move || {
            if let Err(error) = run_meshcore(backend, batch_size, iters, prefix, stop, counters, tx) {
                eprintln!("{} GPU worker disabled: {error:#}", backend.name());
            }
        })
        .expect("failed to spawn GPU worker")
}

fn run_meshcore(
    backend: Backend,
    batch_size: usize,
    iters: u32,
    prefix: Arc<Prefix>,
    stop: Arc<AtomicBool>,
    counters: Arc<Counters>,
    tx: Sender<FoundKey>,
) -> Result<()> {
    ensure_supported(backend)?;
    validate_batch_size(batch_size)?;
    validate_iters(iters)?;
    let mut searcher = new_searcher(backend, batch_size, iters, &prefix)?;
    let rng = crate::keygen::fresh_rng();
    search_loop(
        searcher.as_mut(),
        backend.source_name(),
        batch_size,
        iters,
        rng,
        &prefix,
        &stop,
        &counters,
        &tx,
    )
}

/// Spawn the GPU worker of the Meshtastic node-ID search. Every reported
/// `(base, iter)` is re-derived on the host with dalek and a bitwise CRC
/// (`meshtastic::verify_hit`) before it is sent on `tx`; the worker runs
/// until `stop`.
pub fn spawn_meshtastic_worker(
    backend: Backend,
    batch_size: usize,
    iters: u32,
    pattern: Arc<NodeIdPattern>,
    stop: Arc<AtomicBool>,
    counters: Arc<Counters>,
    tx: Sender<MeshKey>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name(format!("{}-meshtastic", backend.source_name()))
        .spawn(move || {
            if let Err(error) = run_meshtastic(backend, batch_size, iters, pattern, &stop, &counters, tx) {
                eprintln!("{} GPU worker disabled: {error:#}", backend.name());
            }
        })
        .expect("failed to spawn GPU worker")
}

fn run_meshtastic(
    backend: Backend,
    batch_size: usize,
    iters: u32,
    pattern: Arc<NodeIdPattern>,
    stop: &AtomicBool,
    counters: &Arc<Counters>,
    tx: Sender<MeshKey>,
) -> Result<()> {
    ensure_supported(backend)?;
    validate_batch_size(batch_size)?;
    validate_iters(iters)?;
    let mut searcher = new_mesh_searcher(backend, batch_size, iters, &pattern)?;
    let rng = crate::keygen::fresh_rng();
    let source = backend.source_name();
    // Report every verified hit, never stop by itself (the caller decides
    // how many hits it wants via `stop`).
    let verify = move |scalar: [u8; 32]| -> bool {
        if let Some(key) = crate::meshtastic::verify_hit(&pattern, scalar, source) {
            let _ = tx.try_send(key);
        }
        false
    };
    mesh_loop(searcher.as_mut(), batch_size, iters, rng, stop, counters, &verify)
}

/// Drive a Meshtastic backend until `stop` or until `verify` accepts a hit.
/// Base scalars come from `rng` (CSPRNG-seeded by the caller); every
/// launch's bases are retained until its result has been collected so a hit
/// can be re-derived as `base + 8*iter` no matter how deep the pipeline is.
fn mesh_loop(
    searcher: &mut dyn MeshSearcher,
    batch_size: usize,
    iters: u32,
    mut rng: ChaCha20Rng,
    stop: &AtomicBool,
    counters: &Counters,
    verify: &dyn Fn([u8; 32]) -> bool,
) -> Result<()> {
    let depth = searcher.pipeline_depth().max(1);
    let mut ring: Vec<Vec<u32>> = (0..depth).map(|_| vec![0u32; batch_size * 8]).collect();
    // One bulk draw per batch; the little-endian words are the scalar bytes.
    let mut scalar_bytes = vec![0u8; batch_size * 32];
    let fill = |rng: &mut ChaCha20Rng, scalar_bytes: &mut [u8], words: &mut [u32]| {
        fill_base_scalars(rng, scalar_bytes);
        for (word, chunk) in words.iter_mut().zip(scalar_bytes.chunks_exact(4)) {
            *word = u32::from_le_bytes(chunk.try_into().expect("4 bytes"));
        }
    };
    let check_hits = |words: &[u32], hits: Vec<Match>| -> bool {
        for hit in hits {
            let index8 = hit.index * 8;
            let mut scalar = [0u8; 32];
            for (i, word) in words[index8..index8 + 8].iter().enumerate() {
                scalar[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
            }
            let scalar = scalar_add_u64(scalar, 8 * hit.iter as u64);
            if verify(scalar) {
                return true;
            }
        }
        false
    };

    // Keep the pipeline full (depth-1 launches queued behind the one being
    // collected). `stop` (or an accepted hit) only ends submission: launches
    // already committed still run on the GPU, so they are collected, counted
    // and checked like any other, which also leaves the backend's buffers
    // quiescent before it is dropped.
    let mut head = 0usize;
    let mut tail = 0usize;
    let mut in_flight = 0usize;
    let mut done = false;
    loop {
        if !done && !stop.load(Ordering::Relaxed) {
            while in_flight < depth {
                fill(&mut rng, &mut scalar_bytes, &mut ring[tail]);
                if depth > 1 {
                    searcher.submit(&ring[tail])?;
                }
                tail = (tail + 1) % depth;
                in_flight += 1;
            }
        } else if in_flight == 0 {
            break;
        }
        let hits = if depth > 1 {
            searcher.collect()?
        } else {
            searcher.search_words(&ring[head])?
        };
        counters
            .gpu_attempts
            .fetch_add(batch_size as u64 * iters as u64, Ordering::Relaxed);
        if check_hits(&ring[head], hits) {
            done = true;
        }
        head = (head + 1) % depth;
        in_flight -= 1;
    }
    Ok(())
}

/// Drive a MeshCore backend until `stop` or a verified match (sent on `tx`).
#[allow(clippy::too_many_arguments)]
fn search_loop(
    searcher: &mut dyn Searcher,
    source: &'static str,
    batch_size: usize,
    iters: u32,
    mut rng: ChaCha20Rng,
    prefix: &Prefix,
    stop: &AtomicBool,
    counters: &Counters,
    tx: &Sender<FoundKey>,
) -> Result<()> {
    let depth = searcher.pipeline_depth().max(1);
    let mut bases: Vec<Vec<[u8; 32]>> = (0..depth).map(|_| vec![[0u8; 32]; batch_size]).collect();
    let mut words = vec![0u32; batch_size * 8];
    let mut scalar_bytes = vec![0u8; batch_size * 32];
    let fill = |rng: &mut ChaCha20Rng, scalar_bytes: &mut [u8], bases: &mut [[u8; 32]], words: &mut [u32]| {
        fill_base_scalars(rng, scalar_bytes);
        for ((base, scalar), words) in bases
            .iter_mut()
            .zip(scalar_bytes.chunks_exact(32))
            .zip(words.chunks_exact_mut(8))
        {
            base.copy_from_slice(scalar);
            for (word, bytes) in words.iter_mut().zip(scalar.chunks_exact(4)) {
                *word = u32::from_le_bytes(bytes.try_into().expect("4 bytes"));
            }
        }
    };
    // Reconstruct the exact matched scalar: base + 8*iter (the clamp is
    // preserved, see random_base_scalar), then verify with dalek.
    let on_match = |rng: &mut ChaCha20Rng, base: [u8; 32], found: Match| -> bool {
        let scalar = scalar_add_u64(base, 8 * found.iter as u64);
        let mut second_half = [0u8; 32];
        rng.fill_bytes(&mut second_half);
        let candidate = candidate_from_scalar(scalar, second_half);
        if prefix.matches(&candidate.public_key) {
            stop.store(true, Ordering::Relaxed);
            let _ = tx.try_send(candidate.into_found(source));
            return true;
        }
        false
    };

    let mut head = 0usize;
    let mut tail = 0usize;
    let mut in_flight = 0usize;
    let mut done = false;
    loop {
        if !done && !stop.load(Ordering::Relaxed) {
            while in_flight < depth {
                fill(&mut rng, &mut scalar_bytes, &mut bases[tail], &mut words);
                if depth > 1 {
                    searcher.submit(&words)?;
                }
                tail = (tail + 1) % depth;
                in_flight += 1;
            }
        } else if in_flight == 0 {
            break;
        }
        let result = if depth > 1 {
            searcher.collect()?
        } else {
            searcher.search_words(&words)?
        };
        counters
            .gpu_attempts
            .fetch_add(batch_size as u64 * iters as u64, Ordering::Relaxed);
        if let Some(found) = result {
            if !done {
                done = on_match(&mut rng, bases[head][found.index], found);
            }
        }
        head = (head + 1) % depth;
        in_flight -= 1;
    }
    Ok(())
}

#[cfg(test)]
mod pipeline_tests {
    //! The pipelined worker loops must pair every backend result with the
    //! base scalars of the launch that produced it, at any pipeline depth.
    //! Mock backends with a configurable depth make that deterministic.
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::{mesh_loop, search_loop, Match, MeshSearcher, Searcher};
    use crate::keygen::{candidate_from_scalar, scalar_add_u64};
    use crate::prefix::Prefix;

    fn scalar_at(words: &[u32], index: usize) -> [u8; 32] {
        let mut scalar = [0u8; 32];
        for (i, word) in words[index * 8..index * 8 + 8].iter().enumerate() {
            scalar[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
        }
        scalar
    }

    /// Reports candidate (index = launch % batch, iter = launch % iters) of
    /// every launch, and remembers which scalar that was.
    struct MockMesh {
        depth: usize,
        batch: usize,
        iters: u32,
        queued: std::collections::VecDeque<(usize, u32, [u8; 32])>,
        reported: Vec<[u8; 32]>,
        launches: usize,
    }

    impl MeshSearcher for MockMesh {
        fn search_words(&mut self, words: &[u32]) -> anyhow::Result<Vec<Match>> {
            self.submit(words)?;
            self.collect()
        }
        fn pipeline_depth(&self) -> usize {
            self.depth
        }
        fn submit(&mut self, words: &[u32]) -> anyhow::Result<()> {
            assert!(self.queued.len() < self.depth, "pipeline overfilled");
            assert_eq!(words.len(), self.batch * 8);
            let index = self.launches % self.batch;
            let iter = (self.launches % self.iters as usize) as u32;
            self.launches += 1;
            let scalar = scalar_add_u64(scalar_at(words, index), 8 * iter as u64);
            self.queued.push_back((index, iter, scalar));
            Ok(())
        }
        fn collect(&mut self) -> anyhow::Result<Vec<Match>> {
            let (index, iter, scalar) = self.queued.pop_front().expect("collect without submit");
            self.reported.push(scalar);
            Ok(vec![Match { index, iter }])
        }
    }

    #[test]
    fn mesh_loop_rederives_hits_from_the_right_launch_at_every_depth() {
        for depth in [1usize, 2, 3, 5] {
            let batch = 7usize;
            let iters = 5u32;
            let mut searcher = MockMesh {
                depth,
                batch,
                iters,
                queued: Default::default(),
                reported: Vec::new(),
                launches: 0,
            };
            let seen = Arc::new(Mutex::new(Vec::<[u8; 32]>::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let counters = crate::Counters::default();
            let verify = {
                let seen = Arc::clone(&seen);
                let stop = Arc::clone(&stop);
                move |scalar: [u8; 32]| -> bool {
                    let mut seen = seen.lock().unwrap();
                    seen.push(scalar);
                    if seen.len() == 11 {
                        stop.store(true, Ordering::Relaxed);
                    }
                    false
                }
            };
            mesh_loop(
                &mut searcher,
                batch,
                iters,
                ChaCha20Rng::from_seed([depth as u8; 32]),
                &stop,
                &counters,
                &verify,
            )
            .unwrap();
            let seen = seen.lock().unwrap();
            // The stop was raised while depth-1 further launches were in
            // flight; those still ran, so they are checked and counted too.
            assert_eq!(searcher.launches, 10 + depth, "depth {depth}");
            assert_eq!(seen.len(), searcher.launches, "depth {depth}: launches left unchecked");
            assert_eq!(searcher.reported, *seen, "depth {depth}: wrong base scalars");
            assert!(searcher.queued.is_empty(), "depth {depth}: pipeline not drained");
            assert_eq!(
                counters.gpu_attempts.load(Ordering::Relaxed),
                searcher.launches as u64 * batch as u64 * iters as u64,
                "depth {depth}: attempts must count every launch that ran"
            );
        }
    }

    struct MockSearcher {
        depth: usize,
        batch: usize,
        iters: u32,
        queued: std::collections::VecDeque<(usize, u32, [u8; 32])>,
        reported: Vec<[u8; 32]>,
        launches: usize,
    }

    impl Searcher for MockSearcher {
        fn search_words(&mut self, words: &[u32]) -> anyhow::Result<Option<Match>> {
            self.submit(words)?;
            self.collect()
        }
        fn last_debug_point(&self) -> [u8; 32] {
            [0u8; 32]
        }
        fn pipeline_depth(&self) -> usize {
            self.depth
        }
        fn submit(&mut self, words: &[u32]) -> anyhow::Result<()> {
            assert!(self.queued.len() < self.depth, "pipeline overfilled");
            let index = self.launches % self.batch;
            let iter = (self.launches % self.iters as usize) as u32;
            self.launches += 1;
            let scalar = scalar_add_u64(scalar_at(words, index), 8 * iter as u64);
            self.queued.push_back((index, iter, scalar));
            Ok(())
        }
        fn collect(&mut self) -> anyhow::Result<Option<Match>> {
            let (index, iter, scalar) = self.queued.pop_front().expect("collect without submit");
            self.reported.push(scalar);
            Ok(Some(Match { index, iter }))
        }
    }

    #[test]
    fn search_loop_reports_the_first_reported_candidate_matching_the_prefix() {
        // A one-nibble prefix matches 1/16 of random keys, so the mock's
        // reported candidates produce a hit after a handful of launches.
        let prefix = Prefix::parse("A").unwrap();
        for depth in [1usize, 2, 3, 4] {
            let batch = 5usize;
            let iters = 9u32;
            let mut searcher = MockSearcher {
                depth,
                batch,
                iters,
                queued: Default::default(),
                reported: Vec::new(),
                launches: 0,
            };
            let stop = AtomicBool::new(false);
            let counters = crate::Counters::default();
            let (tx, rx) = crossbeam_channel::bounded(1);
            search_loop(
                &mut searcher,
                "mock",
                batch,
                iters,
                ChaCha20Rng::from_seed([7 + depth as u8; 32]),
                &prefix,
                &stop,
                &counters,
                &tx,
            )
            .unwrap();
            let found = rx.try_recv().expect("a match must be reported");
            assert!(stop.load(Ordering::Relaxed));
            assert!(prefix.matches(&found.public_key), "depth {depth}");
            let first = searcher
                .reported
                .iter()
                .map(|scalar| candidate_from_scalar(*scalar, [0u8; 32]))
                .find(|candidate| prefix.matches(&candidate.public_key))
                .expect("some reported candidate matched");
            assert_eq!(found.public_key, first.public_key, "depth {depth}");
            assert_eq!(&found.private_key[..32], &first.private_key[..32], "depth {depth}");
            assert!(searcher.queued.is_empty(), "depth {depth}: pipeline not drained");
            assert!(rx.try_recv().is_err(), "depth {depth}: a second key was reported");
            assert_eq!(searcher.reported.len(), searcher.launches, "depth {depth}");
            assert_eq!(
                counters.gpu_attempts.load(Ordering::Relaxed),
                searcher.launches as u64 * batch as u64 * iters as u64,
                "depth {depth}: attempts must count every launch that ran"
            );
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod metal_loop_tests {
    //! End-to-end checks of the pipelined worker loops against the real Metal
    //! backend: a key planted in a launch that is collected while later
    //! launches are already in flight must be re-derived from that launch's
    //! own base scalars.
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::{mesh_loop, new_mesh_searcher, new_searcher, search_loop, Backend};
    use crate::keygen::{candidate_from_scalar, random_base_scalar, scalar_add_u64};
    use crate::meshtastic::NodeIdPattern;
    use crate::prefix::Prefix;

    /// The base scalar the worker loop will generate for candidate `index` of
    /// launch `launch` when seeded with `seed`.
    fn base_at(seed: [u8; 32], batch: usize, launch: usize, index: usize) -> [u8; 32] {
        let mut rng = ChaCha20Rng::from_seed(seed);
        let mut base = [0u8; 32];
        for l in 0..=launch {
            for i in 0..batch {
                let b = random_base_scalar(&mut rng);
                if l == launch && i == index {
                    base = b;
                }
            }
        }
        base
    }

    fn watchdog(stop: &Arc<AtomicBool>) {
        let stop = Arc::clone(stop);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(20));
            stop.store(true, Ordering::Relaxed);
        });
    }

    #[test]
    fn metal_search_loop_rederives_key_from_pipelined_launch() {
        let seed = [0x5Au8; 32];
        let batch = 16usize;
        let iters = 300u32; // crosses one 256-wide inversion window
        let (launch, index, iter) = (4usize, batch - 1, iters - 1);
        let base = base_at(seed, batch, launch, index);
        let scalar = scalar_add_u64(base, 8 * iter as u64);
        let pubkey = candidate_from_scalar(scalar, [0u8; 32]).public_key;
        let Ok(prefix) = Prefix::parse(&hex::encode_upper(&pubkey[..4])) else {
            // A reserved first byte (00/FF): the planted key is not a MeshCore key.
            return;
        };

        let mut searcher = new_searcher(Backend::Metal, batch, iters, &prefix).expect("metal searcher");
        let depth = searcher.pipeline_depth();
        assert!(depth > 1, "this test exercises the pipelined path");
        let stop = Arc::new(AtomicBool::new(false));
        watchdog(&stop);
        let counters = crate::Counters::default();
        let (tx, rx) = crossbeam_channel::bounded(1);
        search_loop(
            searcher.as_mut(),
            "metal",
            batch,
            iters,
            ChaCha20Rng::from_seed(seed),
            &prefix,
            &stop,
            &counters,
            &tx,
        )
        .unwrap();
        let found = rx.try_recv().expect("the planted key must be reported");
        assert!(stop.load(Ordering::Relaxed));
        assert_eq!(found.source, "metal");
        assert!(prefix.matches(&found.public_key));
        // A 4-byte prefix matches 2^-32 of random keys: the planted key is
        // the one found unless an astronomically unlikely chance match came
        // first in an earlier launch.
        assert_eq!(found.public_key, pubkey);
        assert_eq!(&found.private_key[..32], &scalar);
        assert!(rx.try_recv().is_err(), "exactly one key");
        assert_eq!(
            counters.gpu_attempts.load(Ordering::Relaxed),
            (launch + depth) as u64 * batch as u64 * iters as u64
        );
    }

    #[test]
    fn metal_mesh_loop_rederives_key_from_pipelined_launch() {
        let seed = [0xC3u8; 32];
        let batch = 16usize;
        let iters = 300u32;
        let (launch, index, iter) = (5usize, 7, 256u32); // first candidate of window 2
        let base = base_at(seed, batch, launch, index);
        let scalar = scalar_add_u64(base, 8 * iter as u64);
        let (_, node_id) = crate::meshtastic::derive(&scalar);
        let pattern = NodeIdPattern::parse(&format!("{node_id:08x}")).unwrap();

        let mut searcher = new_mesh_searcher(Backend::Metal, batch, iters, &pattern).expect("metal mesh searcher");
        let depth = searcher.pipeline_depth();
        assert!(depth > 1);
        let stop = Arc::new(AtomicBool::new(false));
        watchdog(&stop);
        let counters = crate::Counters::default();
        let seen = Mutex::new(Vec::<[u8; 32]>::new());
        let verify = |s: [u8; 32]| -> bool {
            seen.lock().unwrap().push(s);
            s == scalar
        };
        mesh_loop(
            searcher.as_mut(),
            batch,
            iters,
            ChaCha20Rng::from_seed(seed),
            &stop,
            &counters,
            &verify,
        )
        .unwrap();
        let seen = seen.into_inner().unwrap();
        assert_eq!(seen, vec![scalar], "exactly the planted scalar is re-derived");
        assert_eq!(
            counters.gpu_attempts.load(Ordering::Relaxed),
            (launch + depth) as u64 * batch as u64 * iters as u64
        );
    }
}
