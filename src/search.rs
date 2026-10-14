//! The search engine shared by the CLI and the API server: start CPU
//! workers and an optional GPU worker for one request, collect verified
//! hits until the requested count, a stop flag, an attempt limit or a
//! deadline, and report the attempt rate.
//!
//! Every hit that reaches a [`Hit`] has been re-derived with
//! curve25519-dalek on the host (twice for Meshtastic: once in the worker,
//! once here) and matched against the request, so nothing unverified can
//! ever be returned.
//!
//! [`Request`] and [`Hit`] are portable (the `stepper` shares them); the
//! threaded engine below them needs the `cli` feature.

#[cfg(feature = "cli")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "cli")]
use std::sync::Arc;
#[cfg(feature = "cli")]
use std::thread::JoinHandle;
#[cfg(feature = "cli")]
use std::time::{Duration, Instant};

#[cfg(feature = "cli")]
use anyhow::{anyhow, Context, Result};
#[cfg(feature = "cli")]
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError};
#[cfg(feature = "cli")]
use serde::Serialize;

#[cfg(feature = "cli")]
use crate::gpu::{self, Backend};
use crate::keygen::FoundKey;
#[cfg(feature = "cli")]
use crate::meshtastic;
use crate::meshtastic::{MeshKey, NodeIdPattern};
use crate::prefix::Prefix;
#[cfg(feature = "cli")]
use crate::Counters;

/// What to search for.
#[derive(Debug, Clone)]
pub enum Request {
    /// A Meshtastic node-ID pattern; stop after `hits` verified keys.
    Meshtastic { pattern: NodeIdPattern, hits: usize },
    /// A MeshCore public-key hex prefix (one key).
    Meshcore { prefix: Prefix },
}

impl Request {
    pub fn flow(&self) -> &'static str {
        match self {
            Request::Meshtastic { .. } => "meshtastic",
            Request::Meshcore { .. } => "meshcore",
        }
    }

    /// The pattern or prefix text.
    pub fn target(&self) -> &str {
        match self {
            Request::Meshtastic { pattern, .. } => pattern.text(),
            Request::Meshcore { prefix } => prefix.text(),
        }
    }

    /// Hits wanted.
    pub fn hits(&self) -> usize {
        match self {
            Request::Meshtastic { hits, .. } => *hits,
            Request::Meshcore { .. } => 1,
        }
    }

    /// Keys expected per hit.
    pub fn expected_attempts_per_hit(&self) -> f64 {
        match self {
            Request::Meshtastic { pattern, .. } => pattern.expected_attempts(),
            Request::Meshcore { prefix } => prefix.expected_attempts(),
        }
    }

    /// Keys expected for the whole request.
    pub fn expected_attempts(&self) -> f64 {
        self.expected_attempts_per_hit() * self.hits() as f64
    }

    /// Per-candidate success probability (for the progress line).
    pub fn match_probability(&self) -> f64 {
        1.0 / self.expected_attempts_per_hit()
    }
}

/// A verified key.
#[derive(Debug, Clone)]
pub enum Hit {
    Meshtastic(MeshKey),
    Meshcore(FoundKey),
}

impl Hit {
    pub fn source(&self) -> &'static str {
        match self {
            Hit::Meshtastic(k) => k.source,
            Hit::Meshcore(k) => k.source,
        }
    }

    /// The hit as the JSON document the CLI prints and the API returns.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Hit::Meshtastic(key) => serde_json::json!({
                "flow": "meshtastic",
                "node_id": key.node_id_str(),
                "node_num": key.node_id,
                "private_key_hex": key.private_key_hex(),
                "private_key_base64": key.private_key_base64(),
                "public_key_hex": key.public_key_hex(),
                "public_key_base64": key.public_key_base64(),
                "command": key.meshtastic_command(),
                "source": key.source,
            }),
            Hit::Meshcore(key) => serde_json::json!({
                "flow": "meshcore",
                "public_key": key.public_key_hex(),
                "private_key": key.private_key_hex(),
                "source": key.source,
            }),
        }
    }

    /// The hit as the CLI's text block.
    pub fn render_text(&self, n: usize) -> String {
        match self {
            Hit::Meshtastic(key) => format!(
                "hit {n}: node id {}\n  private key:     {}\n  private base64:  {}\n  public key:      {}\n  public base64:   {}\n  command:         {}\n  source:          {}\n",
                key.node_id_str(),
                key.private_key_hex(),
                key.private_key_base64(),
                key.public_key_hex(),
                key.public_key_base64(),
                key.meshtastic_command(),
                key.source
            ),
            Hit::Meshcore(key) => format!(
                "Public Key:  {}\nPrivate Key: {}\nSource:      {}\n",
                key.public_key_hex(),
                key.private_key_hex(),
                key.source
            ),
        }
    }
}

/// Re-derive `hit` from its private key with curve25519-dalek and check it
/// against `request`: `Some` only for a key that really matches. Both the
/// threaded engine and the `stepper` pass every hit through here last.
pub fn verify(request: &Request, hit: Hit) -> Option<Hit> {
    match (request, hit) {
        (Request::Meshtastic { pattern, .. }, Hit::Meshtastic(key)) => {
            crate::meshtastic::verify_hit(pattern, key.private_key, key.source).map(Hit::Meshtastic)
        }
        (Request::Meshcore { prefix }, Hit::Meshcore(key)) => {
            let mut scalar = [0u8; 32];
            scalar.copy_from_slice(&key.private_key[..32]);
            let recomputed = curve25519_dalek::edwards::EdwardsPoint::mul_base_clamped(scalar)
                .compress()
                .to_bytes();
            (recomputed == key.public_key && prefix.matches(&key.public_key)).then_some(Hit::Meshcore(key))
        }
        _ => None,
    }
}

/// Which workers a search runs.
#[cfg(feature = "cli")]
#[derive(Debug, Clone, Serialize)]
pub struct Workers {
    /// CPU worker threads (0 = none).
    pub cpu_threads: usize,
    /// The GPU backend with its launch shape, if any.
    pub gpu: Option<GpuWorker>,
}

#[cfg(feature = "cli")]
#[derive(Debug, Clone, Copy, Serialize)]
pub struct GpuWorker {
    pub backend: Backend,
    pub batch: usize,
    pub iters: u32,
}

/// How the caller asked for the workers (CLI flags / server options).
#[cfg(feature = "cli")]
#[derive(Debug, Clone, Default)]
pub struct WorkerOptions {
    pub cpu_only: bool,
    pub gpu_only: bool,
    pub gpu_backend: Option<Backend>,
    pub threads: Option<usize>,
    pub gpu_batch: Option<usize>,
    pub gpu_iters: Option<u32>,
}

#[cfg(feature = "cli")]
impl Workers {
    /// Resolve the options against this build and machine. A requested
    /// backend that is not built in is an error; with no request the
    /// platform default is used when available and the search is CPU-only
    /// otherwise (an error with `gpu_only`).
    pub fn resolve(options: &WorkerOptions) -> Result<Self> {
        if options.cpu_only && options.gpu_only {
            return Err(anyhow!("--cpu-only and --gpu-only are mutually exclusive"));
        }
        let cpu_threads = if options.gpu_only {
            0
        } else {
            let threads = options.threads.unwrap_or_else(|| num_cpus::get().max(1));
            if threads == 0 {
                return Err(anyhow!("--threads must be at least 1 when CPU search is enabled"));
            }
            threads
        };
        let gpu = if options.cpu_only {
            None
        } else {
            let backend = match options.gpu_backend {
                Some(backend) => {
                    gpu::ensure_supported(backend)?;
                    Some(backend)
                }
                None => match gpu::default_backend() {
                    Ok(backend) => Some(backend),
                    Err(error) if !options.gpu_only => {
                        eprintln!("GPU worker disabled: {error:#}");
                        None
                    }
                    Err(error) => return Err(error),
                },
            };
            backend.map(|backend| GpuWorker {
                backend,
                batch: gpu::gpu_batch(backend, options.gpu_batch),
                iters: gpu::gpu_iters(backend, options.gpu_iters),
            })
        };
        if cpu_threads == 0 && gpu.is_none() {
            return Err(anyhow!("no workers: GPU search is unavailable and CPU search is disabled"));
        }
        Ok(Workers { cpu_threads, gpu })
    }

    /// Build the GPU objects for `request` without running them, surfacing
    /// device or compile errors before a job starts.
    pub fn preflight(&self, request: &Request) -> Result<()> {
        if let Some(gpu) = &self.gpu {
            match request {
                Request::Meshtastic { pattern, .. } => {
                    gpu::preflight_meshtastic(gpu.backend, gpu.batch, gpu.iters, pattern)
                }
                Request::Meshcore { prefix } => gpu::preflight_meshcore(gpu.backend, gpu.batch, gpu.iters, prefix),
            }
            .with_context(|| format!("failed to initialize the {} GPU worker", gpu.backend.name()))?;
        }
        Ok(())
    }
}

/// Why a search ended.
#[cfg(feature = "cli")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The requested number of hits was found.
    Complete,
    /// The caller raised the stop flag (Ctrl-C, cancel).
    Stopped,
    /// `max_attempts` was reached.
    AttemptLimit,
    /// The deadline passed.
    Deadline,
    /// Every worker exited (all backends failed).
    NoWorkers,
}

/// The result of one search.
#[cfg(feature = "cli")]
#[derive(Debug)]
pub struct Outcome {
    pub hits: Vec<Hit>,
    pub attempts: u64,
    pub cpu_attempts: u64,
    pub gpu_attempts: u64,
    pub elapsed: Duration,
    pub reason: StopReason,
}

#[cfg(feature = "cli")]
impl Outcome {
    pub fn rate(&self) -> f64 {
        self.attempts as f64 / self.elapsed.as_secs_f64().max(0.001)
    }
}

/// A running search: the workers' shared stop flag and counters, and the
/// channel the verified hits arrive on.
#[cfg(feature = "cli")]
pub struct Search {
    request: Request,
    stop: Arc<AtomicBool>,
    counters: Arc<Counters>,
    started: Instant,
    rx: Receiver<Hit>,
    handles: Vec<JoinHandle<()>>,
}

#[cfg(feature = "cli")]
impl Search {
    /// Spawn the workers for `request`. `stop` may be shared with a signal
    /// handler or a job queue; raising it ends the search.
    pub fn start(request: Request, workers: &Workers, stop: Arc<AtomicBool>) -> Search {
        let counters = Arc::new(Counters::default());
        let (tx, rx) = bounded::<Hit>(1024);
        let mut handles = Vec::new();
        match &request {
            Request::Meshtastic { pattern, .. } => {
                let pattern = Arc::new(pattern.clone());
                let (mtx, mrx) = bounded::<MeshKey>(1024);
                if workers.cpu_threads > 0 {
                    handles.extend(meshtastic::spawn_cpu_workers(
                        workers.cpu_threads,
                        Arc::clone(&pattern),
                        Arc::clone(&stop),
                        Arc::clone(&counters),
                        mtx.clone(),
                    ));
                }
                if let Some(gpu) = &workers.gpu {
                    handles.push(gpu::spawn_meshtastic_worker(
                        gpu.backend,
                        gpu.batch,
                        gpu.iters,
                        Arc::clone(&pattern),
                        Arc::clone(&stop),
                        Arc::clone(&counters),
                        mtx.clone(),
                    ));
                }
                drop(mtx);
                // Forward (the channel closes when every worker has exited).
                handles.push(std::thread::spawn(move || {
                    for key in mrx {
                        if tx.send(Hit::Meshtastic(key)).is_err() {
                            break;
                        }
                    }
                }));
            }
            Request::Meshcore { prefix } => {
                let prefix = Arc::new(prefix.clone());
                let (ftx, frx) = bounded::<FoundKey>(16);
                if workers.cpu_threads > 0 {
                    handles.extend(crate::meshcore::spawn_workers(
                        workers.cpu_threads,
                        Arc::clone(&prefix),
                        Arc::clone(&stop),
                        Arc::clone(&counters),
                        ftx.clone(),
                    ));
                }
                if let Some(gpu) = &workers.gpu {
                    handles.push(gpu::spawn_meshcore_worker(
                        gpu.backend,
                        gpu.batch,
                        gpu.iters,
                        Arc::clone(&prefix),
                        Arc::clone(&stop),
                        Arc::clone(&counters),
                        ftx.clone(),
                    ));
                }
                drop(ftx);
                handles.push(std::thread::spawn(move || {
                    for key in frx {
                        if tx.send(Hit::Meshcore(key)).is_err() {
                            break;
                        }
                    }
                }));
            }
        }
        Search {
            request,
            stop,
            counters,
            started: Instant::now(),
            rx,
            handles,
        }
    }

    pub fn counters(&self) -> &Arc<Counters> {
        &self.counters
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    pub fn stop_flag(&self) -> &Arc<AtomicBool> {
        &self.stop
    }

    /// Re-verify a hit against the request (the workers already did; this
    /// is the last line of defence before anything leaves the engine).
    fn verify(&self, hit: Hit) -> Option<Hit> {
        verify(&self.request, hit)
    }

    /// Collect hits until the request is satisfied, `stop` is raised, the
    /// attempt limit or deadline is reached, or every worker has exited.
    /// `on_tick` runs about every `tick` (progress output); the workers are
    /// joined before this returns.
    pub fn run(
        mut self,
        max_attempts: Option<u64>,
        deadline: Option<Duration>,
        tick: Duration,
        mut on_tick: impl FnMut(&Search),
    ) -> Outcome {
        let wanted = self.request.hits();
        let mut hits: Vec<Hit> = Vec::new();
        let mut next_tick = Instant::now() + tick;
        let mut reason = loop {
            if hits.len() >= wanted {
                break StopReason::Complete;
            }
            if self.stop.load(Ordering::Relaxed) {
                break StopReason::Stopped;
            }
            if max_attempts.is_some_and(|max| self.counters.total() >= max) {
                break StopReason::AttemptLimit;
            }
            if deadline.is_some_and(|d| self.started.elapsed() >= d) {
                break StopReason::Deadline;
            }
            let now = Instant::now();
            if now >= next_tick {
                on_tick(&self);
                next_tick = now + tick;
            }
            let wait = next_tick.saturating_duration_since(now).min(Duration::from_millis(100));
            match self.rx.recv_timeout(wait) {
                Ok(hit) => {
                    match self.verify(hit) {
                        Some(hit) => hits.push(hit),
                        None => eprintln!("refusing a hit that does not re-derive"),
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    // The workers exit on the stop flag too, so the channel
                    // may close before this loop sees the flag.
                    break if self.stop.load(Ordering::Relaxed) {
                        StopReason::Stopped
                    } else {
                        StopReason::NoWorkers
                    };
                }
            }
        };
        self.stop.store(true, Ordering::Relaxed);
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
        // Hits that arrived while stopping are still good keys.
        while hits.len() < wanted {
            match self.rx.try_recv() {
                Ok(hit) => {
                    if let Some(hit) = self.verify(hit) {
                        hits.push(hit);
                    }
                }
                Err(_) => break,
            }
        }
        if hits.len() >= wanted {
            reason = StopReason::Complete;
        }
        Outcome {
            hits,
            attempts: self.counters.total(),
            cpu_attempts: self.counters.cpu_attempts.load(Ordering::Relaxed),
            gpu_attempts: self.counters.gpu_attempts.load(Ordering::Relaxed),
            elapsed: self.started.elapsed(),
            reason,
        }
    }
}

/// Measure the sustained key rate of `workers` by running a search that
/// cannot succeed (a full 8-digit node id, 2^32 keys expected, is stopped
/// after `duration`). Used by the server to size its limits and ETAs.
#[cfg(feature = "cli")]
pub fn measure_rate(workers: &Workers, duration: Duration) -> Result<f64> {
    let pattern = NodeIdPattern::parse("!0a1b2c3d")?;
    let request = Request::Meshtastic { pattern, hits: 1 };
    let search = Search::start(request, workers, Arc::new(AtomicBool::new(false)));
    let outcome = search.run(None, Some(duration), Duration::from_secs(3600), |_| {});
    if outcome.reason == StopReason::NoWorkers {
        return Err(anyhow!("no worker ran during the rate measurement"));
    }
    Ok(outcome.rate())
}

#[cfg(all(test, feature = "cli"))]
mod tests {
    use super::*;

    fn cpu_workers(threads: usize) -> Workers {
        Workers { cpu_threads: threads, gpu: None }
    }

    #[test]
    fn meshtastic_search_collects_verified_hits() {
        let pattern = NodeIdPattern::parse("a*b").unwrap();
        let request = Request::Meshtastic { pattern: pattern.clone(), hits: 3 };
        assert_eq!(request.flow(), "meshtastic");
        assert_eq!(request.expected_attempts(), 3.0 * 256.0);
        let search = Search::start(request, &cpu_workers(2), Arc::new(AtomicBool::new(false)));
        let outcome = search.run(None, Some(Duration::from_secs(120)), Duration::from_millis(50), |_| {});
        assert_eq!(outcome.reason, StopReason::Complete);
        assert_eq!(outcome.hits.len(), 3);
        assert!(outcome.attempts > 0 && outcome.cpu_attempts == outcome.attempts);
        for hit in &outcome.hits {
            let Hit::Meshtastic(key) = hit else { panic!("wrong flow") };
            let (pk, id) = meshtastic::derive(&key.private_key);
            assert_eq!((pk, id), (key.public_key, key.node_id));
            assert!(pattern.matches(id));
            let json = hit.to_json();
            assert_eq!(json["flow"], "meshtastic");
            assert_eq!(json["node_id"], key.node_id_str());
            assert!(hit.render_text(1).contains(&key.meshtastic_command()));
        }
    }

    #[test]
    fn meshcore_search_finds_one_verified_key() {
        let prefix = Prefix::parse("ab").unwrap();
        let request = Request::Meshcore { prefix: prefix.clone() };
        assert_eq!(request.hits(), 1);
        assert_eq!(request.expected_attempts(), 256.0);
        let search = Search::start(request, &cpu_workers(2), Arc::new(AtomicBool::new(false)));
        let outcome = search.run(None, Some(Duration::from_secs(120)), Duration::from_millis(50), |_| {});
        assert_eq!(outcome.reason, StopReason::Complete);
        assert_eq!(outcome.hits.len(), 1);
        let Hit::Meshcore(key) = &outcome.hits[0] else { panic!("wrong flow") };
        assert!(prefix.matches(&key.public_key));
        assert!(key.public_key_hex().starts_with("AB"));
        assert_eq!(key.private_key.len(), 64);
        let json = outcome.hits[0].to_json();
        assert_eq!(json["flow"], "meshcore");
        assert_eq!(json["public_key"], key.public_key_hex());
    }

    #[test]
    fn stop_flag_and_deadline_end_a_hopeless_search() {
        let pattern = NodeIdPattern::parse("!deadbeef").unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let search = Search::start(Request::Meshtastic { pattern: pattern.clone(), hits: 1 }, &cpu_workers(1), Arc::clone(&stop));
        let stopper = {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(300));
                stop.store(true, Ordering::Relaxed);
            })
        };
        let mut ticks = 0;
        let outcome = search.run(None, None, Duration::from_millis(50), |_| ticks += 1);
        stopper.join().unwrap();
        assert_eq!(outcome.reason, StopReason::Stopped);
        assert!(outcome.hits.is_empty());
        assert!(ticks >= 2, "ticks {ticks}");

        let search = Search::start(Request::Meshtastic { pattern: pattern.clone(), hits: 1 }, &cpu_workers(1), Arc::new(AtomicBool::new(false)));
        let outcome = search.run(None, Some(Duration::from_millis(300)), Duration::from_secs(10), |_| {});
        assert_eq!(outcome.reason, StopReason::Deadline);
        assert!(outcome.elapsed >= Duration::from_millis(300));

        let search = Search::start(Request::Meshtastic { pattern, hits: 1 }, &cpu_workers(1), Arc::new(AtomicBool::new(false)));
        let outcome = search.run(Some(1), Some(Duration::from_secs(60)), Duration::from_millis(10), |_| {});
        assert_eq!(outcome.reason, StopReason::AttemptLimit);
        assert!(outcome.attempts >= 1);
    }

    #[test]
    fn worker_options_are_validated() {
        let both = WorkerOptions { cpu_only: true, gpu_only: true, ..Default::default() };
        assert!(Workers::resolve(&both).is_err());
        let zero = WorkerOptions { cpu_only: true, threads: Some(0), ..Default::default() };
        assert!(Workers::resolve(&zero).is_err());
        let cpu = Workers::resolve(&WorkerOptions { cpu_only: true, threads: Some(3), ..Default::default() }).unwrap();
        assert_eq!(cpu.cpu_threads, 3);
        assert!(cpu.gpu.is_none());
    }

    #[test]
    fn rate_measurement_reports_a_positive_rate() {
        let rate = measure_rate(&cpu_workers(1), Duration::from_millis(400)).unwrap();
        assert!(rate > 1000.0, "rate {rate}");
    }
}
