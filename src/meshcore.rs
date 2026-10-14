//! MeshCore identity-key vanity mode: an Ed25519 key whose 32-byte public
//! key, in hex, starts with a chosen prefix (at most [`crate::prefix::
//! MAX_PREFIX_CHARS`] characters). The private key is MeshCore's 64-byte
//! layout: the clamped 32-byte scalar that produces the public key followed
//! by 32 random bytes (the signing-nonce half of an expanded Ed25519 key).
//!
//! The CPU workers below walk `base + 8k` with [`ed_walk::Walk`], which
//! emits the public key's `y` coordinate for ~6 field multiplications per
//! candidate; the GPU kernels (`search_kernel`) do the same walk. Every
//! candidate is re-derived with curve25519-dalek (`candidate_from_scalar`)
//! and checked with [`Prefix::matches`] before it is reported.
//!
//! The thread workers need the `cli` feature; `stepper` runs the same walk
//! (`scan_window`) on the caller's thread.

#[cfg(feature = "cli")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "cli")]
use std::sync::Arc;
#[cfg(feature = "cli")]
use std::thread::{self, JoinHandle};

#[cfg(feature = "cli")]
use crossbeam_channel::Sender;
#[cfg(feature = "cli")]
use rand_chacha::ChaCha20Rng;

use crate::ed_walk::Walk;
#[cfg(feature = "cli")]
use crate::keygen::{candidate_from_scalar, derive_candidate, random_seed, FoundKey};
use crate::prefix::Prefix;
#[cfg(feature = "cli")]
use crate::Counters;

#[cfg(feature = "cli")]
const REPORT_EVERY: u64 = 4096;

/// Independent walks per thread (each lane is itself `ed_walk::WINDOW`
/// interleaved chains; two lanes keep the restart cadence and the
/// per-window batch of the original design).
pub(crate) const LANES: usize = 2;

/// Restart the walk from fresh random bases after this many candidates per
/// lane: keeps every found key close to an independently random scalar and
/// `8 * iter` far inside the clamp headroom `random_base_scalar` guarantees.
pub(crate) const WALK_RESTART_ITERS: u64 = 1 << 24;

/// Re-check the walk against dalek every this many windows (one exact
/// derivation per lane per ~512k candidates).
pub(crate) const RESYNC_WINDOWS: u64 = 1024;

#[cfg(feature = "cli")]
pub fn spawn_workers(
    threads: usize,
    prefix: Arc<Prefix>,
    stop: Arc<AtomicBool>,
    counters: Arc<Counters>,
    tx: Sender<FoundKey>,
) -> Vec<JoinHandle<()>> {
    (0..threads)
        .map(|worker_id| {
            let prefix = Arc::clone(&prefix);
            let stop = Arc::clone(&stop);
            let counters = Arc::clone(&counters);
            let tx = tx.clone();
            thread::Builder::new()
                .name(format!("cpu-meshcore-{worker_id}"))
                .spawn(move || run_worker(prefix, stop, counters, tx))
                .expect("failed to spawn CPU worker")
        })
        .collect()
}

/// Same search the GPU runs, on one CPU thread: CSPRNG-chosen clamped base
/// scalars (`random_base_scalar`), then `scalar += 8` per candidate with
/// the public key's y coordinate produced by `ed_walk::Walk` (a y-only
/// differential addition per candidate plus a share of one batched
/// inversion per window: ~6 field multiplications and 2 squarings per key
/// instead of SHA-512 plus a full fixed-base scalar multiplication and
/// compression).
///
/// A candidate is reported only after the exact key is re-derived with
/// dalek (`candidate_from_scalar`) and checked with `Prefix::matches`, so
/// the walk can never emit a key that does not match. The walk is checked
/// against dalek at every (re)start and every `RESYNC_WINDOWS` windows; on a
/// mismatch (never expected) the thread drops to the original per-candidate
/// derivation.
#[cfg(feature = "cli")]
fn run_worker(
    prefix: Arc<Prefix>,
    stop: Arc<AtomicBool>,
    counters: Arc<Counters>,
    tx: Sender<FoundKey>,
) {
    let mut rng = crate::keygen::fresh_rng();

    let walk = Walk::<LANES>::new(&mut rng);
    if walk.is_none() {
        eprintln!("cpu: incremental walk disagrees with dalek; using per-candidate derivation");
    }
    search(walk, rng, prefix, stop, counters, tx);
}

/// The worker loop proper: the incremental walk while `walk` is `Some`,
/// otherwise (or after a detected desynchronisation) the original
/// per-candidate derivation. Split from `run_worker` so tests can drive the
/// fallback path directly.
#[cfg(feature = "cli")]
fn search(
    mut walk: Option<Walk<LANES>>,
    mut rng: ChaCha20Rng,
    prefix: Arc<Prefix>,
    stop: Arc<AtomicBool>,
    counters: Arc<Counters>,
    tx: Sender<FoundKey>,
) {
    let mut pending = 0u64;
    let mut hits = Vec::new();
    let mut windows = 0u64;

    while !stop.load(Ordering::Relaxed) {
        if let Some(w) = walk.as_mut() {
            if w.iters_done() >= WALK_RESTART_ITERS && !w.restart(&mut rng) {
                eprintln!("cpu: incremental walk disagrees with dalek; using per-candidate derivation");
                walk = None;
                continue;
            }
            scan_window(w, &prefix, &mut hits);
            pending += Walk::<LANES>::CANDIDATES_PER_WINDOW;
            for &(iter, lane) in &hits {
                let scalar = w.scalar_at(lane, iter);
                let candidate = candidate_from_scalar(scalar, random_seed(&mut rng));
                if prefix.matches(&candidate.public_key) {
                    counters.cpu_attempts.fetch_add(pending, Ordering::Relaxed);
                    stop.store(true, Ordering::Relaxed);
                    let _ = tx.try_send(candidate.into_found("cpu"));
                    return;
                }
                // Unreachable for the prefix lengths this crate accepts
                // (`matches_y` is exact below 63 nibbles); kept as the
                // defensive re-check that makes the walk unable to report a
                // wrong key.
            }
            windows += 1;
            if windows % RESYNC_WINDOWS == 0 && !w.matches_dalek() {
                eprintln!("cpu: incremental walk desynchronised from dalek; using per-candidate derivation");
                walk = None;
            }
        } else {
            let candidate = derive_candidate(random_seed(&mut rng));
            pending += 1;
            if prefix.matches(&candidate.public_key) {
                counters.cpu_attempts.fetch_add(pending, Ordering::Relaxed);
                stop.store(true, Ordering::Relaxed);
                let _ = tx.try_send(candidate.into_found("cpu"));
                return;
            }
        }

        if pending >= REPORT_EVERY {
            counters.cpu_attempts.fetch_add(pending, Ordering::Relaxed);
            pending = 0;
        }
    }

    if pending > 0 {
        counters.cpu_attempts.fetch_add(pending, Ordering::Relaxed);
    }
}

/// Walk one window and collect the `(iter, lane)` of every candidate whose
/// y coordinate matches the prefix, lowest iteration first (so the reported
/// key does not depend on the batch-inversion traversal order).
/// `Prefix::matches_y` ignores the sign bit of x, which the walk does not
/// compute: exact for prefixes up to 62 nibbles, a superset for 63/64.
pub(crate) fn scan_window(walk: &mut Walk<LANES>, prefix: &Prefix, hits: &mut Vec<(u64, usize)>) {
    hits.clear();
    walk.next_window(|lane, iter, y| {
        if prefix.matches_y(y) {
            hits.push((iter, lane));
        }
    });
    hits.sort_unstable();
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;
    use crate::ed_walk::WINDOW;
    use crate::keygen::{candidate_from_scalar, random_base_scalar, scalar_add_u64};
    #[cfg(feature = "cli")]
    use crate::keygen::FoundKey;
    #[cfg(feature = "cli")]
    use crossbeam_channel::bounded;
    #[cfg(feature = "cli")]
    use curve25519_dalek::edwards::EdwardsPoint;
    #[cfg(feature = "cli")]
    use std::time::Duration;

    /// Public key of walk candidate `iter` from `base`, computed with dalek.
    fn dalek_key(base: [u8; 32], iter: u64) -> [u8; 32] {
        candidate_from_scalar(scalar_add_u64(base, 8 * iter), [0u8; 32]).public_key
    }

    /// Smallest iter >= from whose key does not start with a reserved byte.
    fn usable_iter(base: [u8; 32], from: u64) -> u64 {
        (from..)
            .find(|&iter| !matches!(dalek_key(base, iter)[0], 0x00 | 0xFF))
            .unwrap()
    }

    /// First `(iter, lane)` matching `prefix` within `limit` iterations.
    fn scan_until_hit(bases: [[u8; 32]; LANES], prefix: &Prefix, limit: u64) -> Option<(u64, usize)> {
        let mut walk = Walk::<LANES>::from_bases(bases).unwrap();
        let mut hits = Vec::new();
        while walk.iters_done() < limit {
            scan_window(&mut walk, prefix, &mut hits);
            if let Some(&(iter, lane)) = hits.first() {
                let candidate = candidate_from_scalar(walk.scalar_at(lane, iter), [0u8; 32]);
                if prefix.matches(&candidate.public_key) {
                    return Some((iter, lane));
                }
            }
        }
        None
    }

    #[test]
    fn scan_window_reports_the_exact_walk_index() {
        let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
        let bases = [random_base_scalar(&mut rng), random_base_scalar(&mut rng)];

        // 8-nibble prefix planted mid-window in lane 1 after crossing a
        // window boundary (earlier windows must report no hit).
        let iter = usable_iter(bases[1], WINDOW as u64 + 100);
        let key = dalek_key(bases[1], iter);
        let prefix = Prefix::parse(&hex::encode_upper(&key[..4])).unwrap();
        assert_eq!(scan_until_hit(bases, &prefix, 4 * WINDOW as u64), Some((iter, 1)));

        // Odd-length prefix exactly at a window boundary, lane 0.
        let iter = usable_iter(bases[0], WINDOW as u64);
        let key = dalek_key(bases[0], iter);
        let prefix = Prefix::parse(&hex::encode_upper(&key[..4])[..7]).unwrap();
        assert_eq!(scan_until_hit(bases, &prefix, 4 * WINDOW as u64), Some((iter, 0)));

        // An 8-character prefix (4 bytes) and a near miss
        // in its last nibble.
        let iter = usable_iter(bases[0], 77);
        let key = dalek_key(bases[0], iter);
        let full = hex::encode_upper(&key[..4]);
        let prefix = Prefix::parse(&full).unwrap();
        assert_eq!(scan_until_hit(bases, &prefix, 2 * WINDOW as u64), Some((iter, 0)));
        let mut wrong = key;
        wrong[3] ^= 0x01;
        let prefix = Prefix::parse(&hex::encode_upper(&wrong[..4])).unwrap();
        assert_ne!(scan_until_hit(bases, &prefix, 2 * WINDOW as u64), Some((iter, 0)));

        // Iteration 0 (the base scalar itself).
        let iter = usable_iter(bases[1], 0);
        let key = dalek_key(bases[1], iter);
        let prefix = Prefix::parse(&hex::encode_upper(&key[..4])).unwrap();
        assert_eq!(scan_until_hit(bases, &prefix, WINDOW as u64), Some((iter, 1)));

        // Chain/window boundaries: the last candidate of a window, the last
        // of the next, and the first of the one after, in both lanes.
        for (lane, from) in [(0, WINDOW as u64 - 1), (1, 2 * WINDOW as u64 - 1), (0, 3 * WINDOW as u64), (1, 3 * WINDOW as u64)] {
            let iter = usable_iter(bases[lane], from);
            let key = dalek_key(bases[lane], iter);
            let prefix = Prefix::parse(&hex::encode_upper(&key[..4])).unwrap();
            assert_eq!(scan_until_hit(bases, &prefix, 5 * WINDOW as u64), Some((iter, lane)), "lane {lane} from {from}");
        }
    }

    #[cfg(feature = "cli")]
    fn find(prefix_text: &str, threads: usize) -> FoundKey {
        let prefix = Arc::new(Prefix::parse(prefix_text).unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let counters = Arc::new(Counters::default());
        let (tx, rx) = bounded::<FoundKey>(1);
        let handles = spawn_workers(
            threads,
            Arc::clone(&prefix),
            Arc::clone(&stop),
            Arc::clone(&counters),
            tx,
        );
        let found = rx
            .recv_timeout(Duration::from_secs(60))
            .expect("CPU worker should find a short prefix quickly");
        stop.store(true, Ordering::Relaxed);
        for h in handles {
            h.join().unwrap();
        }
        assert!(counters.cpu_attempts.load(Ordering::Relaxed) > 0);
        assert!(prefix.matches(&found.public_key));
        found
    }

    /// Drive `search` directly with a given walk state and check the found
    /// key is a real, dalek-verified key for `prefix_text`.
    #[cfg(feature = "cli")]
    fn search_finds(walk: Option<Walk<LANES>>, seed: u8, prefix_text: &str) -> FoundKey {
        let prefix = Arc::new(Prefix::parse(prefix_text).unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let counters = Arc::new(Counters::default());
        let (tx, rx) = bounded::<FoundKey>(1);
        let rng = ChaCha20Rng::from_seed([seed; 32]);
        let handle = {
            let (prefix, stop, counters) = (Arc::clone(&prefix), Arc::clone(&stop), Arc::clone(&counters));
            thread::spawn(move || search(walk, rng, prefix, stop, counters, tx))
        };
        let found = rx
            .recv_timeout(Duration::from_secs(60))
            .expect("worker should find a short prefix quickly");
        handle.join().unwrap();
        assert!(stop.load(Ordering::Relaxed));
        assert!(counters.cpu_attempts.load(Ordering::Relaxed) > 0);
        assert_eq!(found.source, "cpu");
        assert!(prefix.matches(&found.public_key));
        let mut scalar = [0u8; 32];
        scalar.copy_from_slice(&found.private_key[..32]);
        assert_eq!(scalar[0] & 0b0000_0111, 0);
        assert_eq!(scalar[31] & 0b1100_0000, 0b0100_0000);
        assert_eq!(
            EdwardsPoint::mul_base_clamped(scalar).compress().to_bytes(),
            found.public_key
        );
        found
    }

    /// The fallback (per-candidate derivation) path, taken when the walk
    /// cannot start, still finds valid keys.
    #[cfg(feature = "cli")]
    #[test]
    fn search_fallback_path_finds_valid_key() {
        search_finds(None, 3, "A1");
    }

    /// A walk whose lane goes wrong is caught by the periodic dalek check
    /// and the worker continues on the fallback path and still finds a key.
    #[cfg(feature = "cli")]
    #[test]
    fn search_recovers_from_a_desynchronised_walk() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let mut walk = Walk::<LANES>::new(&mut rng).unwrap();
        walk.desynchronise(0);
        walk.desynchronise(1);
        // Before the resync check the wrong walk sees ~8 `matches_y` hits
        // (2 * 1024 * 256 candidates at 2^-16), each rejected by the dalek
        // re-derivation (chance one re-derives to a real match ~2^-13); the
        // fallback path must then find a real key.
        let found = search_finds(Some(walk), 6, "A1B2");
        assert!(hex::encode_upper(found.public_key).starts_with("A1B2"));
    }

    #[cfg(feature = "cli")]
    #[test]
    fn cpu_worker_finds_key_that_rederives_with_dalek() {
        for (text, threads) in [("A1B", 1usize), ("7", 2), ("C4D5", 2)] {
            let found = find(text, threads);
            assert_eq!(found.source, "cpu");
            let mut scalar = [0u8; 32];
            scalar.copy_from_slice(&found.private_key[..32]);
            // Clamped scalar layout.
            assert_eq!(scalar[0] & 0b0000_0111, 0);
            assert_eq!(scalar[31] & 0b1100_0000, 0b0100_0000);
            let recomputed = EdwardsPoint::mul_base_clamped(scalar).compress().to_bytes();
            assert_eq!(recomputed, found.public_key);
            assert!(hex::encode_upper(found.public_key).starts_with(text));
            assert_ne!(found.private_key[32..], [0u8; 32]);
        }
    }
}
