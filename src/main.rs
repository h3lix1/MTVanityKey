use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use clap::{Args as ClapArgs, Parser, Subcommand};

use mtvanitykey::gpu::{self, Backend};
use mtvanitykey::meshtastic::NodeIdPattern;
use mtvanitykey::prefix::Prefix;
use mtvanitykey::progress::{format_duration, ProgressSnapshot};
use mtvanitykey::search::{Request, Search, StopReason, WorkerOptions, Workers};
use mtvanitykey::server::{self, ServerConfig};

#[derive(Parser, Debug)]
#[command(
    name = "mtvanitykey",
    version,
    about = "Vanity key generator for Meshtastic node IDs and MeshCore public keys (CPU, Metal, CUDA)",
    long_about = None
)]
struct Cli {
    /// Run a one-key correctness check of both GPU kernels on the selected
    /// backend (and the host derivation against the oracle vectors), then exit.
    #[arg(long)]
    self_test: bool,

    /// GPU backend for --self-test (defaults to the platform default).
    #[arg(long, value_enum, value_name = "BACKEND")]
    gpu_backend: Option<Backend>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Find X25519 identity keys whose Meshtastic node ID matches a pattern.
    Meshtastic(MeshtasticArgs),
    /// Find an Ed25519 MeshCore identity key whose public key starts with a hex prefix.
    Meshcore(MeshcoreArgs),
    /// Run the HTTP JSON API server with a FIFO job queue.
    Serve(ServeArgs),
}

#[derive(ClapArgs, Debug, Clone)]
struct SearchOptions {
    /// Search with CPU workers only.
    #[arg(long, conflicts_with = "gpu_only")]
    cpu_only: bool,

    /// Search with the GPU worker only.
    #[arg(long)]
    gpu_only: bool,

    /// GPU backend. Defaults to Metal on macOS and CUDA on CUDA builds.
    #[arg(long, value_enum, value_name = "BACKEND")]
    gpu_backend: Option<Backend>,

    /// Number of CPU worker threads. Defaults to the logical CPU count.
    #[arg(long, value_name = "N")]
    threads: Option<usize>,

    /// GPU threads per launch. Defaults to 32768 on Metal and two full waves
    /// of the device (SM count x 256 x 2) on CUDA.
    #[arg(long, value_name = "N")]
    gpu_batch: Option<usize>,

    /// Candidates each GPU thread walks per launch. Defaults to 1024 on Metal
    /// and 16384 on CUDA.
    #[arg(long, value_name = "N")]
    gpu_iters: Option<u32>,
}

impl SearchOptions {
    fn worker_options(&self) -> WorkerOptions {
        WorkerOptions {
            cpu_only: self.cpu_only,
            gpu_only: self.gpu_only,
            gpu_backend: self.gpu_backend,
            threads: self.threads,
            gpu_batch: self.gpu_batch,
            gpu_iters: self.gpu_iters,
        }
    }
}

#[derive(ClapArgs, Debug, Clone)]
struct RunOptions {
    /// Print the hits as one JSON document at the end instead of text.
    #[arg(long)]
    json: bool,

    /// Stop after approximately this many attempts.
    #[arg(long, value_name = "N")]
    max_attempts: Option<u64>,

    /// Update the progress line this often, in milliseconds.
    #[arg(long, default_value_t = 500, value_name = "MS")]
    progress_ms: u64,
}

#[derive(ClapArgs, Debug)]
struct MeshtasticArgs {
    /// Node-ID pattern: up to 64 comma-separated targets of up to 8 hex
    /// digits each (optional leading `!`; `*`, `?`, `.`, `x` are wildcards;
    /// shorter targets are prefixes). All targets must fix the same digit
    /// positions. Every fixed digit multiplies the expected search by 16.
    pattern: String,

    /// Stop after this many verified hits.
    #[arg(long, default_value_t = 1, value_name = "N")]
    hits: usize,

    #[command(flatten)]
    search: SearchOptions,

    #[command(flatten)]
    run: RunOptions,
}

#[derive(ClapArgs, Debug)]
struct MeshcoreArgs {
    /// Public-key hex prefix, 1 to 64 characters (case-insensitive; prefixes
    /// starting 00 or FF are reserved by MeshCore). Every character
    /// multiplies the expected search by 16.
    prefix: String,

    #[command(flatten)]
    search: SearchOptions,

    #[command(flatten)]
    run: RunOptions,
}

#[derive(ClapArgs, Debug)]
struct ServeArgs {
    /// Address to listen on.
    #[arg(long, default_value = "127.0.0.1:8080", value_name = "HOST:PORT")]
    bind: String,

    /// Bearer token clients must present (also read from MTVANITYKEY_TOKEN).
    #[arg(long, env = "MTVANITYKEY_TOKEN", value_name = "TOKEN", hide_env_values = true)]
    token: Option<String>,

    /// Jobs allowed in the queue (running job included).
    #[arg(long, default_value_t = 32, value_name = "N")]
    max_queue: usize,

    /// Hits a single Meshtastic job may ask for.
    #[arg(long, default_value_t = 16, value_name = "N")]
    max_hits: usize,

    /// Reject jobs expected to take longer than this many seconds at the
    /// measured rate.
    #[arg(long, default_value_t = 300.0, value_name = "SECS")]
    max_expected_seconds: f64,

    /// Reject jobs expecting more than this many keys (hits x keys per hit).
    #[arg(long, value_name = "N")]
    max_expected_attempts: Option<f64>,

    /// Longest Meshtastic pattern accepted, in fixed hex digits.
    #[arg(long, default_value_t = 8, value_name = "N")]
    max_digits: u32,

    /// Longest MeshCore prefix accepted, in hex characters (1 to 64; the
    /// default 64 leaves only the --max-expected-seconds limit).
    #[arg(long, default_value_t = mtvanitykey::prefix::MAX_PREFIX_CHARS, value_name = "N")]
    max_prefix_chars: usize,

    /// Stop (and fail) a running job after this many seconds.
    #[arg(long, default_value_t = 600, value_name = "SECS")]
    job_timeout: u64,

    /// Delete an unretrieved result after this many seconds.
    #[arg(long, default_value_t = 600, value_name = "SECS")]
    result_ttl: u64,

    /// Longest a `wait=` request may block, in seconds.
    #[arg(long, default_value_t = 60, value_name = "SECS")]
    max_wait: u64,

    /// Seconds spent measuring the key rate at startup (0 skips it).
    #[arg(long, default_value_t = 2.0, value_name = "SECS")]
    calibrate: f64,

    /// HTTP handler threads (each waiting request occupies one).
    #[arg(long, default_value_t = 8, value_name = "N")]
    handler_threads: usize,

    /// Do not log requests and job completions to stderr.
    #[arg(long)]
    quiet: bool,

    #[command(flatten)]
    search: SearchOptions,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.self_test {
        return self_test(cli.gpu_backend);
    }
    match cli.command {
        Some(Command::Meshtastic(args)) => {
            let pattern = NodeIdPattern::parse(&args.pattern)?;
            if args.hits == 0 {
                return Err(anyhow!("--hits must be at least 1"));
            }
            let request = Request::Meshtastic { pattern, hits: args.hits };
            run_search(request, &args.search, &args.run)
        }
        Some(Command::Meshcore(args)) => {
            let prefix = Prefix::parse(&args.prefix)?;
            run_search(Request::Meshcore { prefix }, &args.search, &args.run)
        }
        Some(Command::Serve(args)) => serve(args),
        None => Err(anyhow!("missing subcommand; run with --help for usage")),
    }
}

fn self_test(backend: Option<Backend>) -> Result<()> {
    mtvanitykey::meshtastic::self_test_host().context("host derivation self-test failed")?;
    println!("host derivation matches the oracle vectors.");
    let backend = match backend {
        Some(b) => {
            gpu::ensure_supported(b)?;
            b
        }
        None => gpu::default_backend().context("no GPU backend available for the self-test")?,
    };
    gpu::self_test(backend).with_context(|| format!("{} GPU self-test failed", backend.name()))?;
    println!(
        "{} GPU self-test passed{}.",
        backend.name(),
        gpu::device_name(backend).map(|d| format!(" ({d})")).unwrap_or_default()
    );
    Ok(())
}

fn run_search(request: Request, search: &SearchOptions, run: &RunOptions) -> Result<()> {
    let workers = Workers::resolve(&search.worker_options())?;
    if search.gpu_only {
        workers.preflight(&request)?;
    }
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        ctrlc::set_handler(move || stop.store(true, Ordering::Relaxed))
            .context("failed to install Ctrl-C handler")?;
    }
    if !run.json {
        match &request {
            Request::Meshtastic { pattern, hits } => println!(
                "pattern {}: {} target(s), {} fixed hex digit(s), {:.0} keys expected per hit; stopping after {} hit(s)",
                pattern.text(),
                pattern.targets.len(),
                pattern.mask.count_ones() / 4,
                pattern.expected_attempts(),
                hits
            ),
            Request::Meshcore { prefix } => println!(
                "prefix {}: {} hex character(s), {:.0} keys expected",
                prefix.text(),
                prefix.nibbles(),
                prefix.expected_attempts()
            ),
        }
        if let Some(gpu) = &workers.gpu {
            println!(
                "workers: {} CPU thread(s) + {} GPU (batch {}, {} iters/thread)",
                workers.cpu_threads,
                gpu.backend.name(),
                gpu.batch,
                gpu.iters
            );
        } else {
            println!("workers: {} CPU thread(s)", workers.cpu_threads);
        }
    }

    let json = run.json;
    let max_attempts = run.max_attempts;
    let label = request.target().to_string();
    let kind = match request {
        Request::Meshtastic { .. } => "pattern",
        Request::Meshcore { .. } => "prefix",
    };
    let probability = request.match_probability();
    let searching = Search::start(request, &workers, Arc::clone(&stop));
    let mut printed = 0usize;
    let mut progress_shown = false;
    let outcome = searching.run(
        max_attempts,
        None,
        Duration::from_millis(run.progress_ms.max(100)),
        |s| {
            let counters = s.counters();
            let snapshot = ProgressSnapshot::new(
                kind,
                &label,
                probability,
                counters.total(),
                counters.cpu_attempts.load(Ordering::Relaxed),
                counters.gpu_attempts.load(Ordering::Relaxed),
                s.elapsed(),
                max_attempts,
            );
            eprint!("\r{}", snapshot.render());
            let _ = io::stderr().flush();
            progress_shown = true;
        },
    );
    if progress_shown {
        eprint!("\r{}\r", " ".repeat(140));
        let _ = io::stderr().flush();
    }

    let rate = outcome.rate();
    if json {
        let value = serde_json::json!({
            "hits": outcome.hits.iter().map(|h| h.to_json()).collect::<Vec<_>>(),
            "attempts": outcome.attempts,
            "cpu_attempts": outcome.cpu_attempts,
            "gpu_attempts": outcome.gpu_attempts,
            "elapsed_seconds": outcome.elapsed.as_secs_f64(),
            "keys_per_second": rate,
            "stop_reason": outcome.reason,
        });
        println!("{}", serde_json::to_string_pretty(&value)?);
    } else {
        for hit in &outcome.hits {
            printed += 1;
            print!("{}", hit.render_text(printed));
        }
        match outcome.reason {
            StopReason::Complete => println!(
                "{} hit(s) after {} attempts in {} ({:.0} keys/sec).",
                outcome.hits.len(),
                outcome.attempts,
                format_duration(outcome.elapsed),
                rate
            ),
            StopReason::NoWorkers => println!("No worker completed successfully."),
            _ => println!(
                "Stopped after {} attempts in {} ({:.0} keys/sec), {} hit(s).",
                outcome.attempts,
                format_duration(outcome.elapsed),
                rate,
                outcome.hits.len()
            ),
        }
    }
    if outcome.reason == StopReason::NoWorkers {
        std::process::exit(2);
    }
    Ok(())
}

fn serve(args: ServeArgs) -> Result<()> {
    if !(1..=mtvanitykey::prefix::MAX_PREFIX_CHARS).contains(&args.max_prefix_chars) {
        return Err(anyhow!(
            "--max-prefix-chars must be 1 to {}",
            mtvanitykey::prefix::MAX_PREFIX_CHARS
        ));
    }
    let workers = Workers::resolve(&args.search.worker_options())?;
    // Surface GPU setup errors now rather than on the first job.
    workers.preflight(&Request::Meshtastic {
        pattern: NodeIdPattern::parse("0")?,
        hits: 1,
    })?;
    workers.preflight(&Request::Meshcore { prefix: Prefix::parse("1")? })?;
    let mut config = ServerConfig::new(&args.bind, workers);
    config.token = args.token.filter(|t| !t.is_empty());
    config.max_queue = args.max_queue.max(1);
    config.max_hits = args.max_hits.max(1);
    config.max_expected_seconds = args.max_expected_seconds;
    config.max_expected_attempts = args.max_expected_attempts;
    config.max_digits = args.max_digits.min(8);
    config.max_prefix_chars = args.max_prefix_chars;
    config.job_timeout = Duration::from_secs(args.job_timeout.max(1));
    config.result_ttl = Duration::from_secs(args.result_ttl.max(1));
    config.max_wait = Duration::from_secs(args.max_wait);
    config.calibrate = Duration::from_secs_f64(args.calibrate.max(0.0));
    config.handler_threads = args.handler_threads.max(1);
    config.log = !args.quiet;
    let has_token = config.token.is_some();
    let handle = server::start(config)?;
    eprintln!(
        "mtvanitykey {} listening on http://{}/v1/ ({})",
        env!("CARGO_PKG_VERSION"),
        handle.addr(),
        if has_token { "bearer token required" } else { "no authentication" }
    );
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        ctrlc::set_handler(move || stop.store(true, Ordering::Relaxed))
            .context("failed to install Ctrl-C handler")?;
    }
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(200));
    }
    eprintln!("shutting down");
    handle.shutdown();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meshtastic_subcommand_parses() {
        let cli = Cli::try_parse_from([
            "mtvanitykey", "meshtastic", "dc80", "--hits", "3", "--json", "--gpu-only", "--gpu-backend", "cuda",
            "--gpu-batch", "4096", "--gpu-iters", "512",
        ])
        .unwrap();
        match cli.command {
            Some(Command::Meshtastic(m)) => {
                assert_eq!(m.pattern, "dc80");
                assert_eq!(m.hits, 3);
                assert!(m.run.json && m.search.gpu_only);
                assert_eq!(m.search.gpu_backend, Some(Backend::Cuda));
                assert_eq!((m.search.gpu_batch, m.search.gpu_iters), (Some(4096), Some(512)));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn meshcore_subcommand_parses_and_rejects_conflicting_modes() {
        let cli = Cli::try_parse_from(["mtvanitykey", "meshcore", "F8A1", "--cpu-only", "--threads", "2"]).unwrap();
        match cli.command {
            Some(Command::Meshcore(m)) => {
                assert_eq!(m.prefix, "F8A1");
                assert!(m.search.cpu_only);
                assert_eq!(m.search.threads, Some(2));
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(Cli::try_parse_from(["mtvanitykey", "meshcore", "F8A1", "--cpu-only", "--gpu-only"]).is_err());
    }

    #[test]
    fn serve_subcommand_parses_limits() {
        let cli = Cli::try_parse_from([
            "mtvanitykey", "serve", "--bind", "0.0.0.0:9000", "--token", "secret", "--max-queue", "4",
            "--max-expected-seconds", "30", "--calibrate", "0",
        ])
        .unwrap();
        match cli.command {
            Some(Command::Serve(s)) => {
                assert_eq!(s.bind, "0.0.0.0:9000");
                assert_eq!(s.token.as_deref(), Some("secret"));
                assert_eq!(s.max_queue, 4);
                assert_eq!(s.max_expected_seconds, 30.0);
                assert_eq!(s.calibrate, 0.0);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn self_test_flag_parses_without_a_subcommand() {
        let cli = Cli::try_parse_from(["mtvanitykey", "--self-test"]).unwrap();
        assert!(cli.self_test && cli.command.is_none());
    }
}
