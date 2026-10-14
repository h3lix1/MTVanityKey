//! A portable, single-threaded search the caller drives in slices: the CPU
//! workers' walks (`mesh_walk::UWalk` for Meshtastic, `ed_walk::Walk` for
//! MeshCore) without threads, channels, the OS CSPRNG or a clock, so it
//! runs anywhere `std` does, including `wasm32-unknown-unknown` (one
//! `Stepper` per Web Worker).
//!
//! The caller supplies the 32-byte seed (e.g. `crypto.getRandomValues`); it
//! keys a ChaCha20 stream exactly like the threaded workers' `fresh_rng`
//! (`keygen::rng_from_seed`, which refuses an all-zero or repeated seed),
//! and every base scalar is drawn from that stream with
//! `keygen::random_base_scalar`, so the keyspace is the full clamped range
//! (see `rng_audit_tests`). The caller also supplies the stopping rule: a
//! window count, a callback consulted at every window boundary (wasm passes
//! one that reads the JS clock, where `std::time::Instant` panics), or, off
//! wasm, a `Duration`.
//!
//! Every hit is re-derived with curve25519-dalek twice before it is
//! returned: once where the walk reports it (as in the workers) and once by
//! [`search::verify`], the engine's last check. The walks are checked
//! against dalek at every (re)start and every `RESYNC_WINDOWS` windows; on a
//! mismatch (never expected) the stepper drops to one dalek derivation per
//! candidate and reports it through [`Stepper::degraded`].

use anyhow::{anyhow, Result};
use rand_chacha::ChaCha20Rng;
use serde_json::{json, Value};

use crate::ed_walk::Walk;
use crate::keygen::{candidate_from_scalar, derive_candidate, random_base_scalar, random_seed, rng_from_seed};
use crate::mesh_walk::UWalk;
use crate::meshtastic::{self, Crc32, NodeIdPattern, BITMAP_WORDS};
use crate::prefix::Prefix;
use crate::search::{self, Hit, Request};
use crate::{meshcore, meshtastic::verify_hit};

/// The `source` of the hits this build reports.
pub const SOURCE: &str = if cfg!(target_arch = "wasm32") { "wasm" } else { "cpu" };

/// A call returns at the next window boundary once it holds this many hits
/// (a one-digit pattern matches every 16th key: without a cap a 100 ms
/// slice would return thousands of keys nobody asked for).
pub const MAX_HITS_PER_CALL: usize = 64;

/// Candidates per slice of the per-candidate fallback path.
const FALLBACK_CHUNK: u64 = 64;

/// Parse and validate a request the way the API server does: `flow` is
/// `meshtastic` or `meshcore`, `text` the pattern or prefix, `hits` at least
/// 1 (and exactly 1 for MeshCore). Errors carry the server's messages.
pub fn parse_request(flow: &str, text: &str, hits: usize) -> Result<Request> {
    if hits == 0 {
        return Err(anyhow!("hits must be at least 1"));
    }
    match flow {
        "meshtastic" => Ok(Request::Meshtastic {
            pattern: NodeIdPattern::parse(text)?,
            hits,
        }),
        "meshcore" => {
            if hits != 1 {
                return Err(anyhow!("the meshcore flow returns exactly one key (hits must be 1)"));
            }
            Ok(Request::Meshcore {
                prefix: Prefix::parse(text)?,
            })
        }
        other => Err(anyhow!("unknown flow \"{other}\" (expected meshtastic or meshcore)")),
    }
}

/// The verified hits and attempts of one `run_*` call.
#[derive(Debug, Default)]
pub struct Slice {
    /// Keys tried in this call.
    pub attempts: u64,
    /// Verified hits found in this call (at most a window past
    /// [`MAX_HITS_PER_CALL`]).
    pub hits: Vec<Hit>,
}

impl Slice {
    /// `{"attempts": N, "hits": [...]}` with the API server's hit objects.
    pub fn to_json(&self) -> Value {
        json!({
            "attempts": self.attempts,
            "hits": self.hits.iter().map(Hit::to_json).collect::<Vec<_>>(),
        })
    }
}

// One `Flow` per stepper, so the Meshtastic variant's inline bitmap is
// not worth a box.
#[allow(clippy::large_enum_variant)]
enum Flow {
    Meshtastic {
        pattern: NodeIdPattern,
        bitmap: [u32; BITMAP_WORDS],
        crc: Crc32,
        walk: Option<UWalk<{ meshtastic::LANES }>>,
    },
    Meshcore {
        prefix: Prefix,
        walk: Option<Walk<{ meshcore::LANES }>>,
    },
}

/// One single-threaded search; see the module documentation.
pub struct Stepper {
    request: Request,
    flow: Flow,
    rng: ChaCha20Rng,
    /// Scratch `(iter, lane)` list of a window's candidate hits.
    found: Vec<(u64, usize)>,
    windows: u64,
    attempts: u64,
    degraded: bool,
}

impl Stepper {
    /// A stepper for `request` keyed by `seed` (32 bytes from a CSPRNG; an
    /// all-zero seed or one already used in this process is an error).
    pub fn new(request: Request, seed: [u8; 32]) -> Result<Self> {
        let mut rng = rng_from_seed(seed)?;
        let flow = match &request {
            Request::Meshtastic { pattern, .. } => Flow::Meshtastic {
                pattern: pattern.clone(),
                bitmap: pattern.bitmap(),
                crc: Crc32::new(),
                walk: UWalk::new(&mut rng),
            },
            Request::Meshcore { prefix } => Flow::Meshcore {
                prefix: prefix.clone(),
                walk: Walk::new(&mut rng),
            },
        };
        let degraded = match &flow {
            Flow::Meshtastic { walk, .. } => walk.is_none(),
            Flow::Meshcore { walk, .. } => walk.is_none(),
        };
        Ok(Stepper {
            request,
            flow,
            rng,
            found: Vec::new(),
            windows: 0,
            attempts: 0,
            degraded,
        })
    }

    /// [`parse_request`] (one hit) then [`Stepper::new`].
    pub fn parse(flow: &str, text: &str, seed: [u8; 32]) -> Result<Self> {
        Self::new(parse_request(flow, text, 1)?, seed)
    }

    pub fn request(&self) -> &Request {
        &self.request
    }

    /// Keys tried since construction.
    pub fn total_attempts(&self) -> u64 {
        self.attempts
    }

    /// Whether the fast walk was abandoned for the per-candidate dalek
    /// derivation (a walk/dalek disagreement; slower, never wrong).
    pub fn degraded(&self) -> bool {
        self.degraded
    }

    /// Candidates per window of the fast walk (the granularity of every
    /// stopping rule).
    pub fn window_size(&self) -> u64 {
        match &self.flow {
            Flow::Meshtastic { walk: Some(_), .. } => UWalk::<{ meshtastic::LANES }>::CANDIDATES_PER_WINDOW,
            Flow::Meshcore { walk: Some(_), .. } => Walk::<{ meshcore::LANES }>::CANDIDATES_PER_WINDOW,
            _ => FALLBACK_CHUNK,
        }
    }

    /// Run whole windows while `more(attempts_so_far_in_this_call)` returns
    /// true (it is consulted after every window, so at least one window
    /// runs), or until [`MAX_HITS_PER_CALL`] hits are held.
    pub fn run_while(&mut self, mut more: impl FnMut(u64) -> bool) -> Slice {
        let mut slice = Slice::default();
        loop {
            slice.attempts += self.window(&mut slice.hits);
            if slice.hits.len() >= MAX_HITS_PER_CALL || !more(slice.attempts) {
                break;
            }
        }
        self.attempts += slice.attempts;
        slice
    }

    /// Run exactly `windows` windows (at least one), unless the hit cap ends
    /// the call earlier.
    pub fn run_windows(&mut self, windows: u64) -> Slice {
        let mut left = windows.max(1);
        self.run_while(|_| {
            left -= 1;
            left > 0
        })
    }

    /// Run for about `budget` (to the first window boundary past it). Not on
    /// wasm32-unknown-unknown, where `Instant` is unavailable: pass a clock
    /// to [`Stepper::run_while`] there.
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    pub fn run_for(&mut self, budget: std::time::Duration) -> Slice {
        let deadline = std::time::Instant::now() + budget;
        self.run_while(|_| std::time::Instant::now() < deadline)
    }

    /// One window of the fast walk (or one chunk of the fallback); verified
    /// hits are appended to `hits`. Returns the candidates tried.
    fn window(&mut self, hits: &mut Vec<Hit>) -> u64 {
        self.windows += 1;
        let windows = self.windows;
        let request = &self.request;
        let rng = &mut self.rng;
        let found = &mut self.found;
        let mut push = |hit: Hit| {
            if let Some(hit) = search::verify(request, hit) {
                hits.push(hit);
            }
        };
        let (tried, ok) = match &mut self.flow {
            Flow::Meshtastic { pattern, bitmap, crc, walk } => match walk {
                Some(w) => {
                    if w.iters_done() >= meshtastic::WALK_RESTART_ITERS && !w.restart(rng) {
                        *walk = None;
                        self.degraded = true;
                        return 0;
                    }
                    found.clear();
                    w.next_window(|lane, iter, u| {
                        if pattern.quick_matches(crc.of_bytes32(u), bitmap) {
                            found.push((iter, lane));
                        }
                    });
                    found.sort_unstable();
                    let mut ok = true;
                    for &(iter, lane) in found.iter() {
                        match verify_hit(pattern, w.scalar_at(lane, iter), SOURCE) {
                            Some(key) => push(Hit::Meshtastic(key)),
                            None => ok = false,
                        }
                    }
                    let desynced = windows % meshtastic::RESYNC_WINDOWS == 0 && !w.matches_dalek();
                    (UWalk::<{ meshtastic::LANES }>::CANDIDATES_PER_WINDOW, ok && !desynced)
                }
                None => {
                    for _ in 0..FALLBACK_CHUNK {
                        if let Some(key) = verify_hit(pattern, random_base_scalar(rng), SOURCE) {
                            push(Hit::Meshtastic(key));
                        }
                    }
                    (FALLBACK_CHUNK, true)
                }
            },
            Flow::Meshcore { prefix, walk } => match walk {
                Some(w) => {
                    if w.iters_done() >= meshcore::WALK_RESTART_ITERS && !w.restart(rng) {
                        *walk = None;
                        self.degraded = true;
                        return 0;
                    }
                    meshcore::scan_window(w, prefix, found);
                    let mut ok = true;
                    for &(iter, lane) in found.iter() {
                        let candidate = candidate_from_scalar(w.scalar_at(lane, iter), random_seed(rng));
                        if prefix.matches(&candidate.public_key) {
                            push(Hit::Meshcore(candidate.into_found(SOURCE)));
                        } else {
                            // `matches_y` is exact at this crate's prefix
                            // lengths, so a miss here means a broken walk.
                            ok = false;
                        }
                    }
                    let desynced = windows % meshcore::RESYNC_WINDOWS == 0 && !w.matches_dalek();
                    (Walk::<{ meshcore::LANES }>::CANDIDATES_PER_WINDOW, ok && !desynced)
                }
                None => {
                    for _ in 0..FALLBACK_CHUNK {
                        let candidate = derive_candidate(random_seed(rng));
                        if prefix.matches(&candidate.public_key) {
                            push(Hit::Meshcore(candidate.into_found(SOURCE)));
                        }
                    }
                    (FALLBACK_CHUNK, true)
                }
            },
        };
        if !ok {
            match &mut self.flow {
                Flow::Meshtastic { walk, .. } => *walk = None,
                Flow::Meshcore { walk, .. } => *walk = None,
            }
            self.degraded = true;
        }
        tried
    }

    /// Test hook: corrupt the fast walk so the fallback must take over.
    #[cfg(test)]
    fn desynchronise(&mut self) {
        match &mut self.flow {
            Flow::Meshtastic { walk: Some(w), .. } => (0..meshtastic::LANES).for_each(|l| w.desynchronise(l)),
            Flow::Meshcore { walk: Some(w), .. } => (0..meshcore::LANES).for_each(|l| w.desynchronise(l)),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use curve25519_dalek::edwards::EdwardsPoint;
    use curve25519_dalek::montgomery::MontgomeryPoint;

    /// A distinct seed per call site (the process-wide guard refuses
    /// repeats, and the tests share one process).
    fn seed(tag: &str) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        Sha256::digest(format!("stepper test seed: {tag}")).into()
    }

    /// Independent check of a Meshtastic hit object: dalek X25519 of the
    /// private key, the bitwise CRC-32, and the pattern.
    fn check_meshtastic(json: &Value, pattern: &NodeIdPattern) {
        let secret: [u8; 32] = hex::decode(json["private_key_hex"].as_str().unwrap()).unwrap().try_into().unwrap();
        assert!(meshtastic::is_clamped(&secret));
        let public = MontgomeryPoint::mul_base_clamped(secret).to_bytes();
        assert_eq!(json["public_key_hex"], hex::encode(public));
        assert_eq!(json["public_key_base64"], meshtastic::base64_encode(&public));
        assert_eq!(json["private_key_base64"], meshtastic::base64_encode(&secret));
        let id = meshtastic::crc32_bitwise(&public);
        assert_eq!(json["node_id"], format!("!{id:08x}"));
        assert!(pattern.matches(id), "!{id:08x} does not match {}", pattern.text());
        assert!(json["command"].as_str().unwrap().contains(json["private_key_base64"].as_str().unwrap()));
        assert_eq!(json["source"], SOURCE);
    }

    fn check_meshcore(json: &Value, prefix: &Prefix) {
        let private = hex::decode(json["private_key"].as_str().unwrap()).unwrap();
        assert_eq!(private.len(), 64);
        let scalar: [u8; 32] = private[..32].try_into().unwrap();
        assert_eq!(scalar[0] & 7, 0);
        assert_eq!(scalar[31] & 0xC0, 0x40);
        let public = EdwardsPoint::mul_base_clamped(scalar).compress().to_bytes();
        assert_eq!(json["public_key"], hex::encode_upper(public));
        assert!(prefix.matches(&public));
        assert!(hex::encode_upper(public).starts_with(prefix.text()));
        assert_eq!(json["source"], SOURCE);
    }

    #[test]
    fn requests_are_validated_like_the_server() {
        let r = parse_request("meshtastic", " dead, beef ", 3).unwrap();
        assert_eq!((r.flow(), r.target(), r.hits()), ("meshtastic", "dead,beef", 3));
        assert_eq!(r.expected_attempts(), 3.0 * 32768.0);
        let r = parse_request("meshcore", "beef", 1).unwrap();
        assert_eq!((r.target(), r.expected_attempts()), ("BEEF", 65536.0));
        let err = |f: &str, t: &str, h: usize| parse_request(f, t, h).unwrap_err().to_string();
        assert_eq!(err("meshcore", "beef", 2), "the meshcore flow returns exactly one key (hits must be 1)");
        assert_eq!(err("meshtastic", "dc80", 0), "hits must be at least 1");
        assert_eq!(err("meshtastic", "12g4", 1), "\"g\" is not a hex digit. Use 0-9, a-f, or * for any digit");
        assert_eq!(err("meshcore", "a1b2c3d4e", 1), "prefix must be 1 to 8 hex characters (got 9)");
        assert!(err("reticulum", "1", 1).starts_with("unknown flow"));
        assert!(Stepper::parse("meshtastic", "dc80", [0u8; 32]).is_err());
    }

    /// Short patterns are found on the fast walk, every hit verifies
    /// independently, and the JSON is the API server's hit shape.
    #[test]
    fn finds_short_meshtastic_patterns_and_hits_verify() {
        for (i, text) in ["dc8", "!12,34,ab", "*a*b"].into_iter().enumerate() {
            let request = parse_request("meshtastic", text, 1).unwrap();
            let Request::Meshtastic { pattern, .. } = &request else { unreachable!() };
            let pattern = pattern.clone();
            let mut stepper = Stepper::new(request, seed(&format!("mesh {i}"))).unwrap();
            let mut hits = Vec::new();
            while hits.len() < 3 {
                let slice = stepper.run_windows(64);
                assert!(slice.attempts > 0);
                hits.extend(slice.hits);
                assert!(stepper.total_attempts() < 1 << 24, "pattern {text} not found");
            }
            assert!(!stepper.degraded());
            for hit in &hits {
                let json = hit.to_json();
                assert_eq!(json["flow"], "meshtastic");
                check_meshtastic(&json, &pattern);
            }
            // Distinct keys.
            let mut keys: Vec<String> = hits.iter().map(|h| h.to_json()["private_key_hex"].to_string()).collect();
            keys.sort();
            keys.dedup();
            assert_eq!(keys.len(), hits.len());
        }
    }

    #[test]
    fn finds_short_meshcore_prefixes_and_hits_verify() {
        for (i, text) in ["A1B", "7", "c4d5"].into_iter().enumerate() {
            let mut stepper = Stepper::parse("meshcore", text, seed(&format!("core {i}"))).unwrap();
            let Request::Meshcore { prefix } = stepper.request().clone() else { unreachable!() };
            let hit = loop {
                let slice = stepper.run_windows(16);
                if let Some(hit) = slice.hits.into_iter().next() {
                    break hit;
                }
                assert!(stepper.total_attempts() < 1 << 24, "prefix {text} not found");
            };
            let json = hit.to_json();
            assert_eq!(json["flow"], "meshcore");
            check_meshcore(&json, &prefix);
            assert_ne!(&hex::decode(json["private_key"].as_str().unwrap()).unwrap()[32..], &[0u8; 32]);
        }
    }

    /// The stopping rules: whole windows, the callback sees the running
    /// attempt count, the hit cap ends a slice early, a time budget ends.
    #[test]
    fn stopping_rules_and_hit_cap() {
        let mut stepper = Stepper::parse("meshtastic", "!deadbeef", seed("rules")).unwrap();
        let w = stepper.window_size();
        assert_eq!(w, 512);
        assert_eq!(stepper.run_windows(0).attempts, w);
        assert_eq!(stepper.run_windows(3).attempts, 3 * w);
        let mut seen = Vec::new();
        let slice = stepper.run_while(|a| {
            seen.push(a);
            a < 5 * w
        });
        assert_eq!(seen, [w, 2 * w, 3 * w, 4 * w, 5 * w]);
        assert_eq!(slice.attempts, 5 * w);
        assert_eq!(stepper.total_attempts(), 9 * w);
        let slice = stepper.run_for(std::time::Duration::from_millis(50));
        assert!(slice.attempts >= w && slice.attempts % w == 0);
        assert_eq!(slice.to_json()["attempts"], slice.attempts);

        // One fixed digit: ~32 hits per window, so the cap ends the call.
        let mut stepper = Stepper::parse("meshtastic", "a", seed("cap")).unwrap();
        let slice = stepper.run_windows(1000);
        assert!(slice.hits.len() >= MAX_HITS_PER_CALL && slice.attempts < 100 * w, "{} hits", slice.hits.len());
        assert_eq!(slice.to_json()["hits"].as_array().unwrap().len(), slice.hits.len());
    }

    /// A corrupted walk is caught (a bogus hit or the periodic dalek check)
    /// and the per-candidate fallback still finds keys that verify.
    #[test]
    fn desynchronised_walk_falls_back_and_still_verifies() {
        let mut stepper = Stepper::parse("meshtastic", "ab", seed("desync mesh")).unwrap();
        stepper.desynchronise();
        let mut hits = Vec::new();
        while hits.is_empty() {
            hits.extend(stepper.run_windows(256).hits);
        }
        assert!(stepper.degraded());
        check_meshtastic(&hits[0].to_json(), &NodeIdPattern::parse("ab").unwrap());

        let mut stepper = Stepper::parse("meshcore", "A1", seed("desync core")).unwrap();
        stepper.desynchronise();
        let mut hits = Vec::new();
        while hits.is_empty() || !stepper.degraded() {
            hits.extend(stepper.run_windows(256).hits);
        }
        for hit in &hits {
            check_meshcore(&hit.to_json(), &Prefix::parse("A1").unwrap());
        }
    }

    /// The walk's candidates come from the seed's ChaCha20 stream exactly
    /// like a threaded worker's (`UWalk::new` on the same stream).
    #[test]
    fn seed_expansion_matches_the_workers() {
        use rand_core::SeedableRng;
        let s = seed("expansion");
        let mut stepper = Stepper::parse("meshtastic", "!0a1b2c3d", s).unwrap();
        let mut rng = ChaCha20Rng::from_seed(s);
        let bases = [random_base_scalar(&mut rng), random_base_scalar(&mut rng)];
        let Flow::Meshtastic { walk: Some(w), .. } = &stepper.flow else { panic!("no walk") };
        assert_eq!((w.scalar_at(0, 0), w.scalar_at(1, 5)), (bases[0], crate::keygen::scalar_add_u64(bases[1], 40)));
        stepper.run_windows(1);
        assert!(Stepper::parse("meshtastic", "1", s).is_err(), "a repeated seed is refused");
    }
}
