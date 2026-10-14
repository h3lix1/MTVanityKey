//! End-to-end tests of the HTTP API: a server on an ephemeral port with
//! CPU workers, driven over plain TCP (no HTTP client dependency). Every
//! returned key is re-derived independently with curve25519-dalek.
#![cfg(feature = "cli")]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use curve25519_dalek::edwards::EdwardsPoint;
use curve25519_dalek::montgomery::MontgomeryPoint;
use serde_json::Value;

use mtvanitykey::search::Workers;
use mtvanitykey::server::{start, ServerConfig, ServerHandle};

fn cpu_workers() -> Workers {
    Workers {
        cpu_threads: 2,
        gpu: None,
    }
}

fn config(bind: &str) -> ServerConfig {
    let mut config = ServerConfig::new(bind, cpu_workers());
    config.calibrate = Duration::from_millis(300);
    config.log = false;
    config.max_wait = Duration::from_secs(60);
    // Two CPU threads manage a few 10^7 keys/s: 6 fixed digits (16.7M keys)
    // pass, a full 8-digit id (4.3G keys, minutes) is refused.
    config.max_expected_seconds = 60.0;
    config
}

/// A job that cannot finish within a test: 16 hits of a full 8-digit id
/// (each ~2^32 keys), allowed by a server without the expected-time limit.
const ENDLESS: &str = "/v1/meshtastic?pattern=%210a1b2c3d&hits=16";

fn request(addr: SocketAddr, method: &str, path: &str, token: Option<&str>) -> (u16, Value) {
    let mut stream = TcpStream::connect(addr).expect("connect");
    stream.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
    let auth = token.map(|t| format!("Authorization: Bearer {t}\r\n")).unwrap_or_default();
    let req = format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\n{auth}Connection: close\r\n\r\n");
    stream.write_all(req.as_bytes()).expect("write");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read");
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").expect("http response");
    let status: u16 = head.split_whitespace().nth(1).expect("status").parse().expect("status code");
    let body: Value = serde_json::from_str(body).unwrap_or_else(|e| panic!("bad JSON body {body:?}: {e}"));
    (status, body)
}

fn get(addr: SocketAddr, path: &str) -> (u16, Value) {
    request(addr, "GET", path, None)
}

fn delete(addr: SocketAddr, path: &str) -> (u16, Value) {
    request(addr, "DELETE", path, None)
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

fn hex32(s: &str) -> [u8; 32] {
    hex::decode(s).unwrap().try_into().unwrap()
}

/// Re-derive a Meshtastic hit (dalek X25519 + bitwise CRC) and check the
/// node id against the pattern's fixed digits.
fn check_meshtastic_hit(hit: &Value, mask: u32, target: u32) {
    assert_eq!(hit["flow"], "meshtastic");
    let secret = hex32(hit["private_key_hex"].as_str().unwrap());
    assert_eq!(secret[0] & 7, 0);
    assert_eq!(secret[31] & 0xC0, 0x40);
    let pk = MontgomeryPoint::mul_base_clamped(secret).to_bytes();
    assert_eq!(hex::encode(pk), hit["public_key_hex"]);
    let id = crc32(&pk);
    assert_eq!(id as u64, hit["node_num"].as_u64().unwrap());
    assert_eq!(format!("!{id:08x}"), hit["node_id"]);
    assert_eq!(id & mask, target, "node id {id:08x} does not match");
    assert!(id >= 4 && id != u32::MAX);
    let b64 = hit["private_key_base64"].as_str().unwrap();
    assert_eq!(hit["command"], format!("meshtastic --set security.private_key \"base64:{b64}\""));
}

/// Re-derive a MeshCore hit (dalek Ed25519) and check the prefix.
fn check_meshcore_hit(hit: &Value, prefix: &str) {
    assert_eq!(hit["flow"], "meshcore");
    let private = hex::decode(hit["private_key"].as_str().unwrap()).unwrap();
    assert_eq!(private.len(), 64);
    let scalar: [u8; 32] = private[..32].try_into().unwrap();
    assert_eq!(scalar[0] & 7, 0);
    assert_eq!(scalar[31] & 0xC0, 0x40);
    let public = hex::encode_upper(EdwardsPoint::mul_base_clamped(scalar).compress().to_bytes());
    assert_eq!(public, hit["public_key"]);
    assert!(public.starts_with(&prefix.to_ascii_uppercase()), "{public} !~ {prefix}");
    assert!(!public.starts_with("00") && !public.starts_with("FF"));
}

fn start_server(config: ServerConfig) -> ServerHandle {
    start(config).expect("server starts")
}

#[test]
fn health_and_validation() {
    let server = start_server(config("127.0.0.1:0"));
    let addr = server.addr();
    let (status, health) = get(addr, "/v1/health");
    assert_eq!(status, 200);
    assert_eq!(health["status"], "ok");
    assert_eq!(health["cpu_threads"], 2);
    assert!(health["backend"].is_null());
    assert!(health["keys_per_second"].as_f64().unwrap() > 1000.0);
    assert_eq!(health["queue_length"], 0);
    assert_eq!(health["limits"]["max_queue"], 32);

    let (status, body) = get(addr, "/v1/meshtastic");
    assert_eq!(status, 400);
    assert_eq!(body["error"], "missing pattern");
    let (status, body) = get(addr, "/v1/meshtastic?pattern=12g4");
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("not a hex digit"));
    let (status, body) = get(addr, "/v1/meshtastic?pattern=ab,abc");
    assert_eq!((status, body["error"].as_str().unwrap()), (400, "every target in a set must fix the same digits"));
    let (status, _) = get(addr, "/v1/meshtastic?pattern=00000000");
    assert_eq!(status, 400);
    let (status, body) = get(addr, "/v1/meshtastic?pattern=ab&hits=0");
    assert_eq!((status, body["error"].as_str().unwrap()), (400, "hits must be at least 1"));
    let (status, body) = get(addr, "/v1/meshtastic?pattern=ab&hits=x");
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("hits"));
    let (status, _) = get(addr, "/v1/meshtastic?pattern=ab&hits=17");
    assert_eq!(status, 413);
    // A full 8-digit id takes 2^32 keys: far beyond 300 s on two CPU threads.
    let (status, body) = get(addr, "/v1/meshtastic?pattern=!deadbeef");
    assert_eq!(status, 413);
    assert!(body["error"].as_str().unwrap().contains("this server accepts jobs up to"));
    let (status, body) = get(addr, "/v1/meshcore?prefix=deadbeef0");
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("1 to 8 hex characters"));
    let (status, _) = get(addr, "/v1/meshcore?prefix=00ab");
    assert_eq!(status, 400);
    let (status, _) = get(addr, "/v1/meshcore?prefix=ab&hits=2");
    assert_eq!(status, 400);
    let (status, _) = get(addr, "/v1/meshcore");
    assert_eq!(status, 400);
    let (status, _) = get(addr, "/v1/nothing");
    assert_eq!(status, 404);
    let (status, _) = request(addr, "POST", "/v1/meshtastic?pattern=ab", None);
    assert_eq!(status, 405);
    let (status, body) = get(addr, "/v1/jobs/0123456789abcdef");
    assert_eq!(status, 404);
    assert!(body["error"].as_str().unwrap().contains("no such job"));
    let (status, _) = delete(addr, "/v1/jobs/0123456789abcdef");
    assert_eq!(status, 404);
    server.shutdown();
}

#[test]
fn wait_requests_return_verified_keys_for_both_flows() {
    let server = start_server(config("127.0.0.1:0"));
    let addr = server.addr();

    let (status, body) = get(addr, "/v1/meshtastic?pattern=%21dc8%2Cab7&hits=2&wait=true");
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["status"], "done");
    assert_eq!(body["flow"], "meshtastic");
    assert_eq!(body["target"], "!dc8,ab7");
    let hits = body["result"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2);
    for hit in hits {
        let id = u32::from_str_radix(&hit["node_id"].as_str().unwrap()[1..], 16).unwrap();
        let (mask, target) = if id >> 20 == 0xDC8 { (0xFFF0_0000, 0xDC80_0000) } else { (0xFFF0_0000, 0xAB70_0000) };
        check_meshtastic_hit(hit, mask, target);
    }
    assert!(body["result"]["attempts"].as_u64().unwrap() > 0);
    // The result was handed out once: the job is gone now.
    let id = body["id"].as_str().unwrap().to_string();
    let (status, _) = get(addr, &format!("/v1/jobs/{id}"));
    assert_eq!(status, 404);

    let (status, body) = get(addr, "/v1/meshcore?prefix=a1b&wait=30");
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["flow"], "meshcore");
    assert_eq!(body["target"], "A1B");
    let hits = body["result"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1);
    check_meshcore_hit(&hits[0], "A1B");

    // A wait that is too short returns the queued/running status instead.
    let (status, body) = get(addr, "/v1/meshtastic?pattern=abcde&wait=0.05");
    assert!(status == 202 || status == 200, "{status} {body}");
    if status == 202 {
        assert!(matches!(body["status"].as_str().unwrap(), "queued" | "running"));
        let id = body["id"].as_str().unwrap();
        let (status, _) = delete(addr, &format!("/v1/jobs/{id}"));
        assert_eq!(status, 200);
    }
    server.shutdown();
}

#[test]
fn queue_is_fifo_and_reports_positions_eta_cancel_and_fullness() {
    let mut config = config("127.0.0.1:0");
    config.max_queue = 4;
    config.max_expected_seconds = 1e12;
    let server = start_server(config);
    let addr = server.addr();

    // An endless job up front keeps the queue busy while the others are
    // examined (it is cancelled below).
    let (status, slow) = get(addr, ENDLESS);
    assert_eq!(status, 202, "{slow}");
    assert_eq!(slow["status"], "queued");
    assert_eq!(slow["position"], 1);
    assert!(slow["eta_seconds"].as_f64().unwrap() > 0.0);
    let slow_id = slow["id"].as_str().unwrap().to_string();

    // Three quick jobs submitted concurrently fill the queue (4 with the slow one).
    let handles: Vec<_> = ["ab", "cd", "ef"]
        .iter()
        .map(|p| {
            let path = format!("/v1/meshtastic?pattern={p}");
            std::thread::spawn(move || get(addr, &path))
        })
        .collect();
    let mut queued: Vec<Value> = handles.into_iter().map(|h| h.join().unwrap()).map(|(s, b)| {
        assert_eq!(s, 202, "{b}");
        b
    }).collect();
    queued.sort_by_key(|b| b["position"].as_u64().unwrap());
    let positions: Vec<u64> = queued.iter().map(|b| b["position"].as_u64().unwrap()).collect();
    // Whether or not the slow job has started yet, it is counted ahead.
    assert_eq!(positions, vec![2, 3, 4]);

    // The fifth job is refused.
    let (status, body) = get(addr, "/v1/meshtastic?pattern=12");
    assert_eq!(status, 429, "{body}");
    assert!(body["error"].as_str().unwrap().contains("queue is full"));
    let (_, health) = get(addr, "/v1/health");
    assert_eq!(health["queue_length"], 4);

    // Statuses while queued; the slow job should be running by now.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let (status, body) = get(addr, &format!("/v1/jobs/{slow_id}"));
        assert_eq!(status, 200);
        if body["status"] == "running" {
            assert!(body["position"].is_null());
            assert!(body["eta_seconds"].as_f64().is_some());
            break;
        }
        assert!(Instant::now() < deadline, "slow job never started: {body}");
        std::thread::sleep(Duration::from_millis(50));
    }
    let first_quick = queued[0]["id"].as_str().unwrap().to_string();
    let (status, body) = get(addr, &format!("/v1/jobs/{first_quick}"));
    assert_eq!(status, 200);
    assert_eq!(body["status"], "queued");
    assert_eq!(body["position"], 2);

    // Cancel the running slow job: the queue advances in FIFO order and the
    // quick jobs complete with verified keys.
    let (status, body) = delete(addr, &format!("/v1/jobs/{slow_id}"));
    assert_eq!(status, 200);
    assert_eq!(body["was"], "running");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let (status, body) = get(addr, &format!("/v1/jobs/{slow_id}"));
        if status == 200 && body["status"] == "cancelled" {
            assert_eq!(body["error"], "cancelled");
            break;
        }
        assert!(status == 200 || status == 404, "{status} {body}");
        if status == 404 {
            break; // already consumed above
        }
        assert!(Instant::now() < deadline, "cancel never took effect: {body}");
        std::thread::sleep(Duration::from_millis(50));
    }
    let (status, _) = get(addr, &format!("/v1/jobs/{slow_id}"));
    assert_eq!(status, 404, "a cancelled job is deleted once reported");

    let mut finish_order = Vec::new();
    for job in &queued {
        let id = job["id"].as_str().unwrap();
        let target = job["target"].as_str().unwrap();
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let (status, body) = get(addr, &format!("/v1/jobs/{id}"));
            assert_eq!(status, 200, "{body}");
            if body["status"] == "done" {
                let hits = body["result"]["hits"].as_array().unwrap();
                assert_eq!(hits.len(), 1);
                let t = u32::from_str_radix(target, 16).unwrap() << 24;
                check_meshtastic_hit(&hits[0], 0xFF00_0000, t);
                finish_order.push(id.to_string());
                break;
            }
            assert!(Instant::now() < deadline, "job {id} never finished: {body}");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    assert_eq!(finish_order.len(), 3);
    let (_, health) = get(addr, "/v1/health");
    assert_eq!(health["queue_length"], 0);

    // Cancelling a queued job removes it before it runs.
    let (status, body) = get(addr, ENDLESS);
    assert_eq!(status, 202);
    let a = body["id"].as_str().unwrap().to_string();
    let (status, body) = get(addr, ENDLESS);
    assert_eq!(status, 202);
    let b = body["id"].as_str().unwrap().to_string();
    let (status, body) = delete(addr, &format!("/v1/jobs/{b}"));
    assert_eq!(status, 200);
    assert_eq!(body["was"], "queued");
    let (status, body) = get(addr, &format!("/v1/jobs/{b}"));
    assert_eq!(status, 200);
    assert_eq!(body["status"], "cancelled");
    let (status, _) = get(addr, &format!("/v1/jobs/{b}"));
    assert_eq!(status, 404);
    let (status, _) = delete(addr, &format!("/v1/jobs/{a}"));
    assert_eq!(status, 200);
    server.shutdown();
}

#[test]
fn bearer_token_protects_everything_but_health() {
    let mut config = config("127.0.0.1:0");
    config.token = Some("s3cret".into());
    config.calibrate = Duration::ZERO;
    let server = start_server(config);
    let addr = server.addr();
    let (status, body) = get(addr, "/v1/health");
    assert_eq!(status, 200);
    assert!(body["keys_per_second"].is_null(), "no calibration: no rate");
    let (status, body) = get(addr, "/v1/meshtastic?pattern=ab");
    assert_eq!(status, 401);
    assert!(body["error"].as_str().unwrap().contains("bearer token"));
    let (status, _) = request(addr, "GET", "/v1/meshtastic?pattern=ab", Some("wrong"));
    assert_eq!(status, 401);
    let (status, _) = request(addr, "GET", "/v1/jobs/0123456789abcdef", None);
    assert_eq!(status, 401);
    let (status, body) = request(addr, "GET", "/v1/meshcore?prefix=7&wait=true", Some("s3cret"));
    assert_eq!(status, 200, "{body}");
    check_meshcore_hit(&body["result"]["hits"][0], "7");
    server.shutdown();
}

#[test]
fn job_time_limit_fails_the_job_and_results_expire() {
    let mut config = config("127.0.0.1:0");
    config.job_timeout = Duration::from_millis(400);
    config.result_ttl = Duration::from_millis(300);
    config.max_expected_seconds = 1e12;
    let server = start_server(config);
    let addr = server.addr();
    let (status, body) = get(addr, &format!("{ENDLESS}&wait=true"));
    assert_eq!(status, 202, "{body}");
    assert_eq!(body["status"], "failed");
    assert!(body["error"].as_str().unwrap().contains("time limit"));
    assert!(body["attempts"].as_u64().unwrap() > 0);

    let (status, body) = get(addr, "/v1/meshcore?prefix=7");
    assert_eq!(status, 202);
    let id = body["id"].as_str().unwrap().to_string();
    std::thread::sleep(Duration::from_millis(900));
    let (status, body) = get(addr, &format!("/v1/jobs/{id}"));
    assert_eq!(status, 404, "expired result must be gone: {body}");
    server.shutdown();
}

#[test]
fn shutdown_stops_a_running_job() {
    let mut config = config("127.0.0.1:0");
    config.max_expected_seconds = 1e12;
    let server = start_server(config);
    let addr = server.addr();
    let (status, _) = get(addr, ENDLESS);
    assert_eq!(status, 202);
    std::thread::sleep(Duration::from_millis(200));
    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    let started = Instant::now();
    let t = std::thread::spawn(move || {
        server.shutdown();
        flag.store(true, Ordering::Relaxed);
    });
    t.join().unwrap();
    assert!(done.load(Ordering::Relaxed));
    // The running (endless) job was stopped, not waited for.
    assert!(started.elapsed() < Duration::from_secs(5));
}
