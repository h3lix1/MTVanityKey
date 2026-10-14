//! Statistical audit of the base-scalar generator the search workers use
//! (`keygen::fill_base_scalars`) and of the per-worker seeding
//! (`keygen::fresh_rng`).
//!
//! The tests draw tens of thousands of base scalars exactly as production
//! does and check: the documented fixed bits are fixed and every other bit
//! is fair; bytes are uniform (chi-square); consecutive scalars are
//! uncorrelated; no scalar repeats; the documented ranges hold. Bounds are
//! set at ~6 sigma so a healthy generator fails with probability < 1e-8 per
//! statistic while a stuck bit, a constant byte, an endianness slip that
//! duplicates a word or a reused seed fails deterministically.
//!
//! Each statistic is checked on a fixed-seed stream (deterministic) and on
//! a stream seeded through the production path (`fresh_rng`, OS CSPRNG),
//! and the caller-seeded path of the portable stepper (`rng_from_seed`) is
//! checked to expand a seed exactly like `fresh_rng` does.

use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use crate::keygen::{fill_base_scalars, rng_from_seed};
#[cfg(feature = "cli")]
use crate::keygen::fresh_rng;

const SAMPLES: usize = 1 << 16;

/// 6-sigma half-width of a Binomial(n, 1/2) count, as a fraction of n.
fn fair_bit_tolerance(n: usize) -> f64 {
    6.0 * 0.5 / (n as f64).sqrt()
}

/// Per-bit-position one-counts over `scalars` (little-endian bit index).
fn bit_counts(scalars: &[[u8; 32]]) -> [u64; 256] {
    let mut counts = [0u64; 256];
    for s in scalars {
        for (i, byte) in s.iter().enumerate() {
            for b in 0..8 {
                counts[8 * i + b] += ((byte >> b) & 1) as u64;
            }
        }
    }
    counts
}

/// Chi-square statistic of the byte histogram of byte positions `bytes`
/// of every scalar against uniform (255 degrees of freedom).
fn byte_chi_square(scalars: &[[u8; 32]], bytes: std::ops::Range<usize>) -> f64 {
    let mut hist = [0u64; 256];
    for s in scalars {
        for &b in &s[bytes.clone()] {
            hist[b as usize] += 1;
        }
    }
    let n = (scalars.len() * bytes.len()) as f64;
    let expected = n / 256.0;
    hist.iter()
        .map(|&h| {
            let d = h as f64 - expected;
            d * d / expected
        })
        .sum()
}

/// Chi-square with 255 df has mean 255 and standard deviation sqrt(510) ~ 22.6;
/// 400 is ~6.4 sigma above the mean (p ~ 1e-9).
const CHI2_255_BOUND: f64 = 400.0;

/// Fraction of differing bits between consecutive scalars at the positions
/// in `mask` (1 bits are compared): independent draws give 1/2.
fn consecutive_difference_fraction(scalars: &[[u8; 32]], mask: &[u8; 32]) -> f64 {
    let mut differing = 0u64;
    let mut compared = 0u64;
    for pair in scalars.windows(2) {
        for i in 0..32 {
            let x = (pair[0][i] ^ pair[1][i]) & mask[i];
            differing += x.count_ones() as u64;
            compared += mask[i].count_ones() as u64;
        }
    }
    differing as f64 / compared as f64
}

fn assert_no_duplicates(scalars: &[[u8; 32]], what: &str) {
    let mut sorted = scalars.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), scalars.len(), "{what}: duplicate base scalars");
}

/// Production Ed25519 draw: `fill_base_scalars` in one bulk call, as the GPU
/// feeder does per launch.
fn ed25519_scalars(rng: &mut ChaCha20Rng, n: usize) -> Vec<[u8; 32]> {
    let mut bytes = vec![0u8; n * 32];
    fill_base_scalars(rng, &mut bytes);
    bytes
        .chunks_exact(32)
        .map(|c| c.try_into().unwrap())
        .collect()
}

fn audit_ed25519(scalars: &[[u8; 32]], what: &str) {
    let n = scalars.len();
    let tol = fair_bit_tolerance(n);
    let counts = bit_counts(scalars);
    // Clamp: bits 0-2 clear, bit 254 set, bit 255 clear. Every other bit
    // (bit 253 included) is a fair coin.
    for bit in 0..256 {
        let frac = counts[bit] as f64 / n as f64;
        match bit {
            0..=2 | 255 => assert_eq!(counts[bit], 0, "{what}: bit {bit} must be clear"),
            254 => assert_eq!(counts[bit], n as u64, "{what}: bit 254 must be set"),
            _ => assert!(
                (frac - 0.5).abs() < tol,
                "{what}: bit {bit} frequency {frac:.4} outside 0.5 +- {tol:.4}"
            ),
        }
    }
    // Bytes 1..31 are fully random (byte 0 and 31 carry clamp bits).
    let chi2 = byte_chi_square(scalars, 1..31);
    assert!(chi2 < CHI2_255_BOUND, "{what}: byte histogram chi2 {chi2:.1}");
    let mut mask = [0xFFu8; 32];
    mask[0] = 0xF8;
    mask[31] = 0x3F;
    let diff = consecutive_difference_fraction(scalars, &mask);
    let tol_pairs = fair_bit_tolerance((n - 1) * 251);
    assert!(
        (diff - 0.5).abs() < tol_pairs,
        "{what}: consecutive scalars differ in {diff:.5} of free bits"
    );
    assert_no_duplicates(scalars, what);
    // Range: multiples of 8 in [2^254, 2^255), i.e. every clamped scalar.
    for s in scalars {
        assert_eq!(s[0] & 7, 0);
        assert_eq!(s[31] & 0xC0, 0x40, "{what}: top byte {:#x}", s[31]);
    }
}

#[test]
fn ed25519_base_scalars_are_uniform_fixed_seed() {
    let mut rng = ChaCha20Rng::from_seed([0xA5u8; 32]);
    let scalars = ed25519_scalars(&mut rng, SAMPLES);
    audit_ed25519(&scalars, "ed25519 fixed seed");
}

#[cfg(feature = "cli")]
#[test]
fn ed25519_base_scalars_are_uniform_production_seed() {
    let mut rng = fresh_rng();
    let scalars = ed25519_scalars(&mut rng, SAMPLES);
    audit_ed25519(&scalars, "ed25519 fresh_rng");
}

/// Two workers seeded through the production path get disjoint streams
/// (and `fresh_rng` refuses to hand out the same seed twice).
#[cfg(feature = "cli")]
#[test]
fn fresh_rngs_are_independent() {
    let mut a = fresh_rng();
    let mut b = fresh_rng();
    let sa = ed25519_scalars(&mut a, 4096);
    let sb = ed25519_scalars(&mut b, 4096);
    let mut all = sa.clone();
    all.extend_from_slice(&sb);
    assert_no_duplicates(&all, "two fresh_rng streams");
    let mut mask = [0xFFu8; 32];
    mask[0] = 0xF8;
    mask[31] = 0x3F;
    // Interleave the two streams and check neighbours (a-b pairs) differ in
    // half their bits: catches a correlated or offset-copied seed.
    let interleaved: Vec<[u8; 32]> = sa.iter().zip(&sb).flat_map(|(x, y)| [*x, *y]).collect();
    let diff = consecutive_difference_fraction(&interleaved, &mask);
    assert!((diff - 0.5).abs() < fair_bit_tolerance(8191 * 251), "cross-stream difference {diff:.5}");
}

/// The stepper's caller-seeded stream is plain ChaCha20 keyed with the
/// seed (the same expansion as `fresh_rng`, so the audit above applies to
/// it), and the guard refuses an all-zero seed and a repeated one.
#[test]
fn caller_seeds_expand_like_fresh_rng_and_are_guarded() {
    let seed = *b"rng audit: caller seed, 32 bytes";
    let mut a = rng_from_seed(seed).expect("fresh seed");
    let mut b = ChaCha20Rng::from_seed(seed);
    assert_eq!(ed25519_scalars(&mut a, 64), ed25519_scalars(&mut b, 64));
    let mut a = rng_from_seed(*b"rng audit: second seed, 32 bytes").expect("fresh seed");
    audit_ed25519(&ed25519_scalars(&mut a, SAMPLES), "ed25519 rng_from_seed");
    let repeat = rng_from_seed(seed).unwrap_err().to_string();
    assert!(repeat.contains("repeated"), "{repeat}");
    let zero = rng_from_seed([0u8; 32]).unwrap_err().to_string();
    assert!(zero.contains("all-zero"), "{zero}");
}

/// Which Ed25519 targets the search can reach. A candidate `a` finds target
/// scalar `t` iff `a == t (mod l)` (B has prime order l). The candidates are
/// `2^254 + 8k`, so `t` is reachable iff `k = (t - 2^254) / 8 (mod l)` lies
/// below the number of distinct `k` the generator can produce. Returns that
/// `k` as an integer.
fn ed25519_candidate_index(target: &[u8; 32]) -> num_bigint::BigUint {
    use curve25519_dalek::scalar::Scalar;
    let mut top = [0u8; 32];
    top[31] = 0x40;
    let t = Scalar::from_bytes_mod_order(*target);
    let k = (t - Scalar::from_bytes_mod_order(top)) * Scalar::from(8u8).invert();
    num_bigint::BigUint::from_bytes_le(&k.to_bytes())
}

/// Every standard clamped Ed25519 scalar (bits 0-2 clear, bit 254 set, bit
/// 255 clear, bit 253 FREE) must be reachable by some base scalar plus a
/// walk of `+8` steps: the base generator must not pin bit 253, or the half
/// of all real keys with bit 253 set could never be found (their residues
/// mod l are disjoint from those of the bit-253-clear half, because two such
/// scalars differ by less than 8l and the difference is a multiple of 8 only
/// when it is a multiple of 8l). `2^251` distinct indices `k` are needed.
#[test]
fn every_clamped_ed25519_scalar_is_reachable_from_some_base() {
    use num_bigint::BigUint;
    let mut rng = ChaCha20Rng::from_seed([0x77u8; 32]);
    let bases = ed25519_scalars(&mut rng, 4096);
    let reach = BigUint::from(1u8) << 251;
    // The generator's own output is reachable and indexes the full 2^251.
    let mut saw_253_set = 0;
    for b in &bases {
        let k = ed25519_candidate_index(b);
        assert!(k < reach, "base {} has index {k} >= 2^251", hex::encode(b));
        saw_253_set += (b[31] & 0x20 != 0) as usize;
    }
    assert!(
        saw_253_set > 1500 && saw_253_set < 2600,
        "bit 253 of base scalars is not free: set in {saw_253_set} of 4096"
    );
    // A standard key (SHA-512 of a seed, clamped) is reachable whatever
    // its bit 253.
    for i in 0..2048u32 {
        let mut seed = [0u8; 32];
        seed[..4].copy_from_slice(&i.to_le_bytes());
        let key = crate::keygen::derive_candidate(seed).private_key;
        let t: [u8; 32] = key[..32].try_into().unwrap();
        let k = ed25519_candidate_index(&t);
        assert!(k < reach, "clamped key {} (bit253={}) unreachable, index {k}", hex::encode(t), t[31] >> 5 & 1);
    }
}
