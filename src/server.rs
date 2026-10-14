//! HTTP JSON API with a FIFO job queue in front of the single shared search
//! engine (one GPU, so jobs run one at a time).
//!
//! Endpoints (all JSON):
//!
//! * `GET /v1/meshtastic?pattern=<pattern>[&hits=N][&wait=true|<secs>]`
//! * `GET /v1/meshcore?prefix=<hex>[&wait=true|<secs>]`
//!   Validate and enqueue a job. With `wait`, block (up to the server's
//!   maximum wait) and return the finished job (200), else return the queued
//!   job's status with its queue position (202).
//! * `GET /v1/jobs/{id}`: status (`queued`, `running`, `done`, `failed`,
//!   `cancelled`), position, attempts, rate and ETA; a finished job's result
//!   is returned exactly once and then deleted.
//! * `DELETE /v1/jobs/{id}`: cancel (dequeue or stop) a job.
//! * `GET /v1/health`: backend, measured rate, queue length and limits.
//!
//! Errors are `{"error": "..."}` with a fitting status: 400 (bad request),
//! 401 (token), 404, 405, 413 (too long for this server's limits), 429
//! (queue full). Results live in memory only, for at most `result_ttl`
//! after completion, and private keys are never logged.

use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use rand_core::RngCore;
use serde_json::{json, Value};
use tiny_http::{Header, Method, Request as HttpRequest, Response, StatusCode};

use crate::gpu;
use crate::meshtastic::NodeIdPattern;
use crate::prefix::{Prefix, MAX_PREFIX_CHARS};
use crate::search::{Request, Search, StopReason, Workers};

/// Server configuration (see the `serve` subcommand for the defaults).
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// `host:port` to listen on (port 0 picks a free port).
    pub bind: String,
    /// Bearer token required on every endpoint except `/v1/health`.
    pub token: Option<String>,
    /// Jobs allowed in the queue (running job included); more get 429.
    pub max_queue: usize,
    /// Hits a single job may ask for.
    pub max_hits: usize,
    /// Reject jobs whose expected duration at the measured rate exceeds this.
    pub max_expected_seconds: f64,
    /// Explicit cap on a job's expected keys (hits x keys per hit), if any.
    pub max_expected_attempts: Option<f64>,
    /// Longest Meshtastic pattern (fixed hex digits) accepted.
    pub max_digits: u32,
    /// Longest MeshCore prefix accepted (at most [`MAX_PREFIX_CHARS`]).
    pub max_prefix_chars: usize,
    /// A running job is stopped (and fails) after this long.
    pub job_timeout: Duration,
    /// Finished jobs are deleted after this long if not retrieved.
    pub result_ttl: Duration,
    /// Upper bound of a `wait=` request.
    pub max_wait: Duration,
    /// The workers every job runs with.
    pub workers: Workers,
    /// Rate measurement at startup (zero skips it; limits then use
    /// `fallback_rate`).
    pub calibrate: Duration,
    /// Keys/s assumed until a rate has been measured.
    pub fallback_rate: f64,
    /// Handler threads (each `wait=true` request occupies one).
    pub handler_threads: usize,
    /// Print one line per request to stderr.
    pub log: bool,
}

impl ServerConfig {
    pub fn new(bind: &str, workers: Workers) -> Self {
        ServerConfig {
            bind: bind.to_string(),
            token: None,
            max_queue: 32,
            max_hits: 16,
            max_expected_seconds: 300.0,
            max_expected_attempts: None,
            max_digits: 8,
            max_prefix_chars: MAX_PREFIX_CHARS,
            job_timeout: Duration::from_secs(600),
            result_ttl: Duration::from_secs(600),
            max_wait: Duration::from_secs(60),
            workers,
            calibrate: Duration::from_secs(2),
            fallback_rate: 1e8,
            handler_threads: 8,
            log: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Queued => "queued",
            Status::Running => "running",
            Status::Done => "done",
            Status::Failed => "failed",
            Status::Cancelled => "cancelled",
        }
    }

    fn is_final(self) -> bool {
        matches!(self, Status::Done | Status::Failed | Status::Cancelled)
    }
}

struct Job {
    id: String,
    request: Request,
    status: Status,
    stop: Arc<AtomicBool>,
    /// The running search's counters and start time.
    counters: Option<Arc<crate::Counters>>,
    started: Option<Instant>,
    /// Attempts, elapsed and rate of the finished run.
    final_attempts: u64,
    final_elapsed: Duration,
    finished: Option<Instant>,
    /// Verified hits of a finished job (deleted once retrieved).
    result: Option<Vec<Value>>,
    error: Option<String>,
}

struct State {
    queue: VecDeque<String>,
    jobs: HashMap<String, Job>,
    running: Option<String>,
    /// Keys per second: the calibration, then an average over finished jobs.
    rate: Option<f64>,
    shutting_down: bool,
}

struct Shared {
    config: ServerConfig,
    state: Mutex<State>,
    /// Signalled on every queue or status change.
    changed: Condvar,
    device: Option<String>,
}

type ApiResult = std::result::Result<(u16, Value), (u16, String)>;

/// A running server; drop or `shutdown()` to stop it.
pub struct ServerHandle {
    addr: SocketAddr,
    server: Arc<tiny_http::Server>,
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl ServerHandle {
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Stop accepting requests, cancel the running job and join every thread.
    pub fn shutdown(mut self) {
        self.stop_threads();
    }

    fn stop_threads(&mut self) {
        {
            let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            state.shutting_down = true;
            for job in state.jobs.values() {
                job.stop.store(true, Ordering::Relaxed);
            }
        }
        self.shared.changed.notify_all();
        // `unblock` releases one thread blocked in `recv` per call.
        for _ in 0..self.shared.config.handler_threads.max(1) {
            self.server.unblock();
        }
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.stop_threads();
    }
}

/// Bind, measure the rate (unless disabled) and start the worker and
/// handler threads. Returns once the server is listening.
pub fn start(config: ServerConfig) -> Result<ServerHandle> {
    let server = tiny_http::Server::http(&config.bind)
        .map_err(|error| anyhow!("failed to bind {}: {error}", config.bind))?;
    let addr = match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => addr,
        #[allow(unreachable_patterns)]
        _ => return Err(anyhow!("unexpected listen address type")),
    };
    let server = Arc::new(server);

    let rate = if config.calibrate.is_zero() {
        None
    } else {
        let rate = crate::search::measure_rate(&config.workers, config.calibrate)
            .context("rate measurement failed")?;
        if config.log {
            eprintln!("measured {:.3e} keys/s over {:.1} s", rate, config.calibrate.as_secs_f64());
        }
        Some(rate)
    };
    let device = config.workers.gpu.and_then(|g| gpu::device_name(g.backend));
    let shared = Arc::new(Shared {
        config,
        state: Mutex::new(State {
            queue: VecDeque::new(),
            jobs: HashMap::new(),
            running: None,
            rate,
            shutting_down: false,
        }),
        changed: Condvar::new(),
        device,
    });

    let mut threads = Vec::new();
    {
        let shared = Arc::clone(&shared);
        threads.push(
            thread::Builder::new()
                .name("job-worker".into())
                .spawn(move || worker_loop(&shared))
                .context("failed to spawn the job worker")?,
        );
    }
    for i in 0..shared.config.handler_threads.max(1) {
        let shared = Arc::clone(&shared);
        let server = Arc::clone(&server);
        threads.push(
            thread::Builder::new()
                .name(format!("http-{i}"))
                .spawn(move || {
                    while let Ok(request) = server.recv() {
                        handle(&shared, request);
                    }
                })
                .context("failed to spawn an HTTP handler")?,
        );
    }
    Ok(ServerHandle {
        addr,
        server,
        shared,
        threads,
    })
}

// ---------------------------------------------------------------------------
// The job worker: one job at a time, FIFO.
// ---------------------------------------------------------------------------

fn worker_loop(shared: &Shared) {
    loop {
        let (id, request, stop) = {
            let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if state.shutting_down {
                    return;
                }
                purge_expired(&shared.config, &mut state);
                if let Some(id) = state.queue.pop_front() {
                    let picked = match state.jobs.get_mut(&id) {
                        Some(job) if job.status == Status::Queued => {
                            job.status = Status::Running;
                            job.started = Some(Instant::now());
                            Some((job.request.clone(), Arc::clone(&job.stop)))
                        }
                        _ => None,
                    };
                    if let Some((request, stop)) = picked {
                        state.running = Some(id.clone());
                        break (id, request, stop);
                    }
                    continue;
                }
                state = shared
                    .changed
                    .wait_timeout(state, Duration::from_millis(500))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        };
        shared.changed.notify_all();

        let search = Search::start(request.clone(), &shared.config.workers, stop);
        {
            let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(job) = state.jobs.get_mut(&id) {
                job.counters = Some(Arc::clone(search.counters()));
            }
        }
        let outcome = search.run(None, Some(shared.config.job_timeout), Duration::from_secs(3600), |_| {});

        let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        state.running = None;
        // Fold the measured rate in (ignoring runs too short to measure).
        if outcome.elapsed >= Duration::from_millis(200) && outcome.attempts > 0 {
            let r = outcome.rate();
            state.rate = Some(match state.rate {
                Some(old) => 0.7 * old + 0.3 * r,
                None => r,
            });
        }
        if let Some(job) = state.jobs.get_mut(&id) {
            job.final_attempts = outcome.attempts;
            job.final_elapsed = outcome.elapsed;
            job.finished = Some(Instant::now());
            job.counters = None;
            match outcome.reason {
                StopReason::Complete => {
                    job.status = Status::Done;
                    job.result = Some(outcome.hits.iter().map(|h| h.to_json()).collect());
                }
                StopReason::Stopped => {
                    job.status = Status::Cancelled;
                    job.error = Some("cancelled".into());
                }
                StopReason::Deadline => {
                    job.status = Status::Failed;
                    job.error = Some(format!(
                        "job time limit of {:.0} s exceeded after {} attempts",
                        shared.config.job_timeout.as_secs_f64(),
                        outcome.attempts
                    ));
                }
                StopReason::AttemptLimit => {
                    job.status = Status::Failed;
                    job.error = Some("attempt limit reached".into());
                }
                StopReason::NoWorkers => {
                    job.status = Status::Failed;
                    job.error = Some("no search worker could run (see the server log)".into());
                }
            }
            if shared.config.log {
                eprintln!(
                    "job {id} {} {} {} after {} attempts in {:.2} s",
                    request.flow(),
                    request.target(),
                    job.status.as_str(),
                    outcome.attempts,
                    outcome.elapsed.as_secs_f64()
                );
            }
        }
        drop(state);
        shared.changed.notify_all();
    }
}

/// Drop finished jobs older than the result TTL.
fn purge_expired(config: &ServerConfig, state: &mut State) {
    let ttl = config.result_ttl;
    state
        .jobs
        .retain(|_, job| !(job.status.is_final() && job.finished.is_some_and(|t| t.elapsed() > ttl)));
}

// ---------------------------------------------------------------------------
// HTTP handling.
// ---------------------------------------------------------------------------

fn handle(shared: &Arc<Shared>, mut request: HttpRequest) {
    // Drain any body so the connection can be reused.
    let mut sink = Vec::new();
    let _ = request.as_reader().take(1 << 16).read_to_end(&mut sink);

    let url = request.url().to_string();
    let (path, query) = match url.split_once('?') {
        Some((p, q)) => (p.to_string(), parse_query(q)),
        None => (url.clone(), HashMap::new()),
    };
    let method = request.method().clone();

    let result = if path == "/v1/health" && method == Method::Get {
        Ok((200, health(shared)))
    } else if let Err(e) = authorize(shared, &request) {
        Err(e)
    } else {
        match (&method, path.as_str()) {
            (&Method::Get, "/v1/meshtastic") => enqueue(shared, Flow::Meshtastic, &query),
            (&Method::Get, "/v1/meshcore") => enqueue(shared, Flow::Meshcore, &query),
            (&Method::Get, p) if p.starts_with("/v1/jobs/") => job_status(shared, &p["/v1/jobs/".len()..], true),
            (&Method::Delete, p) if p.starts_with("/v1/jobs/") => cancel(shared, &p["/v1/jobs/".len()..]),
            (_, "/v1/meshtastic" | "/v1/meshcore" | "/v1/health") => Err((405, "method not allowed".into())),
            (_, p) if p.starts_with("/v1/jobs/") => Err((405, "method not allowed".into())),
            _ => Err((404, "not found".into())),
        }
    };
    let (status, body) = match result {
        Ok((status, body)) => (status, body),
        Err((status, message)) => (status, json!({ "error": message })),
    };
    if shared.config.log {
        eprintln!("{method} {path} {status}");
    }
    let body = serde_json::to_vec(&body).unwrap_or_default();
    let response = Response::from_data(body)
        .with_status_code(StatusCode(status))
        .with_header(Header::from_bytes("Content-Type", "application/json").expect("header"))
        .with_header(Header::from_bytes("Cache-Control", "no-store").expect("header"));
    let _ = request.respond(response);
}

fn authorize(shared: &Shared, request: &HttpRequest) -> std::result::Result<(), (u16, String)> {
    let Some(token) = &shared.config.token else {
        return Ok(());
    };
    let header = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Authorization"))
        .map(|h| h.value.as_str().to_string())
        .unwrap_or_default();
    let presented = header.strip_prefix("Bearer ").unwrap_or("").trim();
    if !presented.is_empty() && constant_time_eq(presented.as_bytes(), token.as_bytes()) {
        Ok(())
    } else {
        Err((401, "missing or invalid bearer token".into()))
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn parse_query(query: &str) -> HashMap<String, String> {
    query
        .split('&')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (percent_decode(k), percent_decode(v))
        })
        .collect()
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = &text[i + 1..i + 3];
                match u8::from_str_radix(hex, 16) {
                    Ok(b) => {
                        out.push(b);
                        i += 3;
                        continue;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            b'+' => out.push(b' '),
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[derive(Clone, Copy)]
enum Flow {
    Meshtastic,
    Meshcore,
}

fn health(shared: &Shared) -> Value {
    let state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    let queued = state.queue.len();
    let config = &shared.config;
    json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "backend": config.workers.gpu.map(|g| g.backend.source_name()),
        "device": shared.device,
        "cpu_threads": config.workers.cpu_threads,
        "keys_per_second": state.rate,
        "queue_length": queued + usize::from(state.running.is_some()),
        "running": state.running.is_some(),
        "limits": {
            "max_queue": config.max_queue,
            "max_hits": config.max_hits,
            "max_expected_seconds": config.max_expected_seconds,
            "max_expected_attempts": config.max_expected_attempts,
            "max_digits": config.max_digits,
            "max_prefix_chars": config.max_prefix_chars,
            "job_timeout_seconds": config.job_timeout.as_secs_f64(),
            "result_ttl_seconds": config.result_ttl.as_secs_f64(),
            "max_wait_seconds": config.max_wait.as_secs_f64(),
        },
    })
}

fn enqueue(shared: &Arc<Shared>, flow: Flow, query: &HashMap<String, String>) -> ApiResult {
    let config = &shared.config;
    let hits = match query.get("hits") {
        None => 1,
        Some(h) => h.parse::<usize>().map_err(|_| (400, "hits must be a positive integer".to_string()))?,
    };
    if hits == 0 {
        return Err((400, "hits must be at least 1".into()));
    }
    if hits > config.max_hits {
        return Err((413, format!("hits must be at most {}", config.max_hits)));
    }
    let request = match flow {
        Flow::Meshtastic => {
            let text = query.get("pattern").ok_or((400, "missing pattern".to_string()))?;
            let pattern = NodeIdPattern::parse(text).map_err(|e| (400, e.to_string()))?;
            let digits = pattern.mask.count_ones() / 4;
            if digits > config.max_digits {
                return Err((413, format!(
                    "pattern fixes {digits} hex digits; this server accepts at most {}",
                    config.max_digits
                )));
            }
            Request::Meshtastic { pattern, hits }
        }
        Flow::Meshcore => {
            if hits != 1 {
                return Err((400, "the meshcore flow returns exactly one key (hits must be 1)".into()));
            }
            let text = query.get("prefix").ok_or((400, "missing prefix".to_string()))?;
            let prefix = Prefix::parse(text).map_err(|e| (400, e.to_string()))?;
            if prefix.nibbles() > config.max_prefix_chars {
                return Err((413, format!(
                    "prefix is {} characters; this server accepts at most {}",
                    prefix.nibbles(),
                    config.max_prefix_chars
                )));
            }
            Request::Meshcore { prefix }
        }
    };
    let wait = match query.get("wait") {
        None => None,
        Some(w) if w.is_empty() || w == "true" || w == "1" => Some(config.max_wait),
        Some(w) if w == "false" || w == "0" => None,
        Some(w) => {
            let secs: f64 = w.parse().map_err(|_| (400, "wait must be true, false or a number of seconds".to_string()))?;
            Some(Duration::from_secs_f64(secs.max(0.0)).min(config.max_wait))
        }
    };

    let id = {
        let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        purge_expired(config, &mut state);
        let rate = state.rate.unwrap_or(config.fallback_rate);
        let expected = request.expected_attempts();
        if let Some(cap) = config.max_expected_attempts {
            if expected > cap {
                return Err((413, format!(
                    "expected {expected:.3e} keys exceeds this server's limit of {cap:.3e}"
                )));
            }
        }
        let expected_seconds = expected / rate;
        if expected_seconds > config.max_expected_seconds {
            return Err((413, format!(
                "expected {expected:.3e} keys is about {expected_seconds:.0} s at {rate:.3e} keys/s; this server accepts jobs up to {:.0} s",
                config.max_expected_seconds
            )));
        }
        if state.queue.len() + usize::from(state.running.is_some()) >= config.max_queue {
            return Err((429, format!("queue is full ({} jobs); retry later", config.max_queue)));
        }
        let id = new_job_id();
        state.jobs.insert(
            id.clone(),
            Job {
                id: id.clone(),
                request,
                status: Status::Queued,
                stop: Arc::new(AtomicBool::new(false)),
                counters: None,
                started: None,
                final_attempts: 0,
                final_elapsed: Duration::ZERO,
                finished: None,
                result: None,
                error: None,
            },
        );
        state.queue.push_back(id.clone());
        id
    };
    shared.changed.notify_all();

    match wait {
        None => job_status(shared, &id, false).map(|(_, body)| (202, body)),
        Some(wait) => {
            let deadline = Instant::now() + wait;
            let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                let done = state.jobs.get(&id).is_none_or(|job| job.status.is_final());
                if done {
                    break;
                }
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                state = shared
                    .changed
                    .wait_timeout(state, deadline - now)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
            drop(state);
            let (status, body) = job_status(shared, &id, true)?;
            Ok((if status == 200 && body["status"] == "done" { 200 } else { 202 }, body))
        }
    }
}

fn new_job_id() -> String {
    let mut bytes = [0u8; 8];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    hex::encode(bytes)
}

/// The job's status document. A finished job's result is handed out once:
/// `consume` deletes the job afterwards (a `wait` request that timed out
/// leaves it in place by passing `consume` only when it is final anyway).
fn job_status(shared: &Shared, id: &str, consume: bool) -> ApiResult {
    let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    purge_expired(&shared.config, &mut state);
    let rate = state.rate.unwrap_or(shared.config.fallback_rate);
    // ETA of the jobs ahead in the queue (plus the running one).
    let position = state.queue.iter().position(|q| q == id);
    let ahead_seconds: f64 = {
        let mut secs = 0.0;
        if let Some(running) = state.running.as_ref().and_then(|r| state.jobs.get(r)) {
            secs += running.request.expected_attempts() / rate;
        }
        for q in state.queue.iter().take(position.unwrap_or(0)) {
            if let Some(job) = state.jobs.get(q) {
                secs += job.request.expected_attempts() / rate;
            }
        }
        secs
    };
    let Some(job) = state.jobs.get(id) else {
        return Err((404, "no such job (unknown, expired, or its result was already retrieved)".into()));
    };
    let (attempts, elapsed) = match (&job.counters, job.started) {
        (Some(counters), Some(started)) if job.status == Status::Running => (counters.total(), started.elapsed()),
        _ => (job.final_attempts, job.final_elapsed),
    };
    let expected = job.request.expected_attempts();
    let live_rate = if job.status == Status::Running && elapsed.as_secs_f64() > 0.5 {
        attempts as f64 / elapsed.as_secs_f64()
    } else {
        rate
    };
    let eta_seconds = match job.status {
        Status::Queued => Some(ahead_seconds + expected / live_rate),
        Status::Running => Some(expected / live_rate),
        _ => None,
    };
    let mut body = json!({
        "id": job.id,
        "flow": job.request.flow(),
        "target": job.request.target(),
        "hits_wanted": job.request.hits(),
        "status": job.status.as_str(),
        "position": position.map(|p| p + 1 + usize::from(state.running.is_some())),
        "attempts": attempts,
        "elapsed_seconds": elapsed.as_secs_f64(),
        "keys_per_second": if elapsed.as_secs_f64() > 0.0 { Some(attempts as f64 / elapsed.as_secs_f64()) } else { None },
        "expected_attempts": expected,
        "eta_seconds": eta_seconds,
    });
    if let Some(error) = &job.error {
        body["error"] = json!(error);
    }
    if job.status == Status::Done {
        body["result"] = json!({
            "hits": job.result.clone().unwrap_or_default(),
            "attempts": job.final_attempts,
            "elapsed_seconds": job.final_elapsed.as_secs_f64(),
            "keys_per_second": job.final_attempts as f64 / job.final_elapsed.as_secs_f64().max(0.001),
        });
    }
    if consume && job.status.is_final() {
        state.jobs.remove(id);
    }
    Ok((200, body))
}

fn cancel(shared: &Shared, id: &str) -> ApiResult {
    let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    let Some(job) = state.jobs.get_mut(id) else {
        return Err((404, "no such job".into()));
    };
    let previous = job.status;
    job.stop.store(true, Ordering::Relaxed);
    match previous {
        Status::Queued => {
            job.status = Status::Cancelled;
            job.error = Some("cancelled".into());
            job.finished = Some(Instant::now());
            state.queue.retain(|q| q != id);
        }
        Status::Running => {
            // The worker records the cancellation when the search unwinds.
        }
        Status::Done | Status::Failed | Status::Cancelled => {
            state.jobs.remove(id);
        }
    }
    drop(state);
    shared.changed.notify_all();
    Ok((200, json!({ "id": id, "status": "cancelled", "was": previous.as_str() })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_parsing_decodes_percent_and_plus() {
        let q = parse_query("pattern=%21dead%2Cbeef&hits=2&wait=true&x=a+b&flag");
        assert_eq!(q["pattern"], "!dead,beef");
        assert_eq!(q["hits"], "2");
        assert_eq!(q["wait"], "true");
        assert_eq!(q["x"], "a b");
        assert_eq!(q["flag"], "");
        assert_eq!(percent_decode("%zz%4"), "%zz%4");
        assert_eq!(percent_decode("100%"), "100%");
    }

    #[test]
    fn constant_time_eq_compares_whole_strings() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
    }
}
