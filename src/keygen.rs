use curve25519_dalek::edwards::EdwardsPoint;
use rand_core::RngCore;
use sha2::{Digest, Sha512};

#[derive(Debug, Clone)]
pub struct FoundKey {
    pub public_key: [u8; 32],
    pub private_key: [u8; 64],
    pub source: &'static str,
}

impl FoundKey {
    pub fn public_key_hex(&self) -> String {
        hex::encode_upper(self.public_key)
    }

    pub fn private_key_hex(&self) -> String {
        hex::encode_upper(self.private_key)
    }
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub public_key: [u8; 32],
    pub private_key: [u8; 64],
}

impl Candidate {
    pub fn into_found(self, source: &'static str) -> FoundKey {
        FoundKey {
            public_key: self.public_key,
            private_key: self.private_key,
            source,
        }
    }
}

/// A fresh ChaCha20 stream for one search worker, keyed from the OS CSPRNG.
///
/// Every worker (CPU thread, GPU feeder thread) and every process restart
/// must start an independent stream, or two workers would re-walk the same
/// keys. The OS CSPRNG makes that so in practice; this helper adds the
/// guard a failed CSPRNG would otherwise defeat silently: the 256-bit key
/// must not be all-zero and must not repeat one already issued in this
/// process (a repeat has probability 2^-256 from a working CSPRNG, so one
/// is a broken RNG, not bad luck). Either case aborts the search rather than
/// burning GPU time on keys already searched.
pub fn fresh_rng() -> rand_chacha::ChaCha20Rng {
    use rand_core::SeedableRng;
    let mut seed = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut seed);
    assert!(seed != [0u8; 32], "OS CSPRNG returned an all-zero seed");
    let mut issued = ISSUED_SEEDS.lock().unwrap_or_else(|e| e.into_inner());
    assert!(
        issued.insert(seed),
        "OS CSPRNG repeated a worker seed within this process"
    );
    rand_chacha::ChaCha20Rng::from_seed(seed)
}

/// Seeds handed out by `fresh_rng` in this process (a few per worker; the
/// set never grows beyond a handful of entries in a search).
static ISSUED_SEEDS: std::sync::Mutex<std::collections::BTreeSet<[u8; 32]>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

pub fn random_seed(rng: &mut impl RngCore) -> [u8; 32] {
    let mut seed = [0u8; 32];
    rng.fill_bytes(&mut seed);
    seed
}

pub fn derive_candidate(seed: [u8; 32]) -> Candidate {
    let digest = Sha512::digest(seed);

    let mut clamped = [0u8; 32];
    clamped.copy_from_slice(&digest[..32]);
    clamp_scalar(&mut clamped);

    derive_candidate_from_clamped(clamped, &digest[32..64])
}

pub fn scalar_words_for_gpu(seed: [u8; 32]) -> ([u32; 8], [u8; 32]) {
    let digest = Sha512::digest(seed);
    let mut clamped = [0u8; 32];
    clamped.copy_from_slice(&digest[..32]);
    clamp_scalar(&mut clamped);

    (scalar_to_words(&clamped), clamped)
}

pub fn scalar_to_words(scalar: &[u8; 32]) -> [u32; 8] {
    let mut words = [0u32; 8];
    for (i, chunk) in scalar.chunks_exact(4).enumerate() {
        words[i] = u32::from_le_bytes(chunk.try_into().expect("chunk length is fixed"));
    }
    words
}

/// Random clamped base scalar for the incremental (`+8` per candidate)
/// search. It is a standard Ed25519 clamp (bits 0-2 clear, bit 254 set, bit
/// 255 clear) with bit 253 left random, so the bases index every clamped
/// scalar a real key can have: candidate `a` finds a key with scalar `t`
/// only if `a == t (mod l)`, and the bases with bit 253 clear reach no key
/// with bit 253 set (see `rng_audit_tests`). The walk's headroom is kept
/// by `clamp_base_scalar`: a base within 2^64 of the top of the clamped
/// range (probability 2^-190) has bit 253 cleared, so `scalar + 8*iter`
/// stays below 2^255 with bit 254 set for every `iter` below 2^61 and the
/// clamp is preserved for every candidate the walkers visit.
pub fn random_base_scalar(rng: &mut impl RngCore) -> [u8; 32] {
    let mut scalar = [0u8; 32];
    rng.fill_bytes(&mut scalar);
    clamp_base_scalar(&mut scalar);
    scalar
}

/// Fill `out` (a multiple of 32 bytes) with consecutive `random_base_scalar`
/// values: one bulk `fill_bytes` draw, then the base clamp on every 32-byte
/// scalar. Consumes the RNG stream exactly like the same number of
/// `random_base_scalar` calls would, at a fraction of the per-call cost;
/// the GPU feeder threads need millions of base scalars per second.
pub fn fill_base_scalars(rng: &mut impl RngCore, out: &mut [u8]) {
    debug_assert_eq!(out.len() % 32, 0);
    rng.fill_bytes(out);
    for scalar in out.chunks_exact_mut(32) {
        let scalar: &mut [u8; 32] = scalar.try_into().expect("32-byte chunk");
        clamp_base_scalar(scalar);
    }
}

/// Ed25519 clamp, then the walk headroom (see `random_base_scalar`): bit
/// 253 is cleared only when bits 64..=253 are all set, i.e. when fewer than
/// 2^64 values remain below 2^255 - 2^254 of walking room.
fn clamp_base_scalar(scalar: &mut [u8; 32]) {
    clamp_scalar(scalar);
    let near_top = scalar[8..31].iter().all(|b| *b == 0xFF) && scalar[31] & 0x3F == 0x3F;
    if near_top {
        scalar[31] &= !0x20;
    }
}

/// Minimum number of `+8` steps any base scalar can take before the clamp
/// bits would change (2^64 of room above the base, in units of 8).
#[cfg_attr(not(test), allow(dead_code))] // documents the invariant; asserted in tests
pub const BASE_SCALAR_HEADROOM_STEPS: u64 = 1 << 61;

/// Little-endian `scalar + delta`; the walkers use delta = 8*iter, which
/// cannot reach bit 254 given the headroom enforced by `random_base_scalar`
/// (`iter` < `BASE_SCALAR_HEADROOM_STEPS`).
pub fn scalar_add_u64(mut scalar: [u8; 32], delta: u64) -> [u8; 32] {
    let mut carry = delta as u128;
    for byte in scalar.iter_mut() {
        if carry == 0 {
            break;
        }
        let sum = *byte as u128 + (carry & 0xFF);
        *byte = sum as u8;
        carry = (carry >> 8) + (sum >> 8);
    }
    scalar
}

/// Build a candidate directly from an already-clamped scalar plus an
/// arbitrary 32-byte signing-nonce half (MeshCore only requires the first 32
/// bytes to be the clamped scalar producing the public key).
pub fn candidate_from_scalar(scalar: [u8; 32], second_half: [u8; 32]) -> Candidate {
    debug_assert_eq!(scalar[0] & 0b0000_0111, 0);
    debug_assert_eq!(scalar[31] & 0b1100_0000, 0b0100_0000);
    derive_candidate_from_clamped(scalar, &second_half)
}

fn derive_candidate_from_clamped(clamped: [u8; 32], second_half: &[u8]) -> Candidate {
    let point = EdwardsPoint::mul_base_clamped(clamped);
    let public_key = point.compress().to_bytes();

    let mut private_key = [0u8; 64];
    private_key[..32].copy_from_slice(&clamped);
    private_key[32..].copy_from_slice(second_half);

    Candidate {
        public_key,
        private_key,
    }
}

fn clamp_scalar(scalar: &mut [u8; 32]) {
    scalar[0] &= 248;
    scalar[31] &= 63;
    scalar[31] |= 64;
}

#[cfg(test)]
mod tests {
    use super::{
        candidate_from_scalar, derive_candidate, fill_base_scalars, random_base_scalar,
        scalar_add_u64,
    };
    use curve25519_dalek::edwards::EdwardsPoint;
    use rand_chacha::ChaCha20Rng;

    /// The bulk fill must produce exactly the scalars that the same number of
    /// `random_base_scalar` calls on an identically seeded RNG produce (the
    /// GPU workers switched to the bulk form; a hit's base scalar is looked up
    /// by index in the filled batch).
    #[test]
    fn bulk_fill_matches_sequential_base_scalars() {
        use rand_core::SeedableRng;
        for n in [1usize, 7, 64, 1000] {
            let mut a = ChaCha20Rng::seed_from_u64(n as u64);
            let mut b = ChaCha20Rng::seed_from_u64(n as u64);
            let mut bulk = vec![0u8; n * 32];
            fill_base_scalars(&mut a, &mut bulk);
            for (i, chunk) in bulk.chunks_exact(32).enumerate() {
                assert_eq!(chunk, &random_base_scalar(&mut b)[..], "scalar {i} of {n}");
            }
            // And the two RNGs are left in the same state.
            assert_eq!(random_base_scalar(&mut a), random_base_scalar(&mut b));
        }
    }
    use rand_core::SeedableRng;

    #[test]
    fn private_key_uses_meshcore_layout() {
        let candidate = derive_candidate([7u8; 32]);
        assert_eq!(candidate.public_key.len(), 32);
        assert_eq!(candidate.private_key.len(), 64);
        assert_eq!(candidate.private_key[0] & 0b0000_0111, 0);
        assert_eq!(candidate.private_key[31] & 0b1000_0000, 0);
        assert_ne!(candidate.private_key, [0u8; 64]);
    }

    /// Bit 253 is random in general but cleared when the base is within
    /// 2^64 of the top of the clamped range, so every base keeps at least
    /// `BASE_SCALAR_HEADROOM_STEPS` steps of `+8` before the clamp changes.
    #[test]
    fn base_scalar_keeps_bit_253_unless_near_the_top() {
        use super::{clamp_base_scalar, BASE_SCALAR_HEADROOM_STEPS};
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        let (mut set, mut clear) = (0, 0);
        for _ in 0..512 {
            let b = random_base_scalar(&mut rng);
            if b[31] & 0x20 != 0 {
                set += 1
            } else {
                clear += 1
            }
        }
        assert!(set > 150 && clear > 150, "bit 253 set {set} clear {clear}");

        // All ones: bits 64..=253 set, so bit 253 is cleared.
        let mut top = [0xFFu8; 32];
        clamp_base_scalar(&mut top);
        assert_eq!(top[31], 0x5F);
        // One below the fix-up threshold keeps bit 253.
        let mut below = [0xFFu8; 32];
        below[8] = 0xFE;
        clamp_base_scalar(&mut below);
        assert_eq!(below[31], 0x7F);
        // The largest unfixed base plus the guaranteed headroom still has
        // the clamp bits intact.
        let walked = scalar_add_u64(below, 8 * (BASE_SCALAR_HEADROOM_STEPS - 1));
        assert_eq!(walked[31] & 0b1100_0000, 0b0100_0000);
        assert_eq!(walked[0] & 7, 0);
    }

    #[test]
    fn scalar_add_u64_carries_across_bytes() {
        let mut scalar = [0u8; 32];
        scalar[0] = 0xF8;
        scalar[1] = 0xFF;
        scalar[2] = 0xFF;
        let sum = scalar_add_u64(scalar, 8);
        assert_eq!(sum[0], 0x00);
        assert_eq!(sum[1], 0x00);
        assert_eq!(sum[2], 0x00);
        assert_eq!(sum[3], 0x01);
    }

    #[test]
    fn incremented_base_scalar_stays_clamped_and_matches_pubkey() {
        let mut rng = ChaCha20Rng::from_seed([9u8; 32]);
        let base = random_base_scalar(&mut rng);
        assert_eq!(base[0] & 0b0000_0111, 0);
        assert_eq!(base[31] & 0b1100_0000, 0b0100_0000);

        for iter in [0u64, 1, 255, 1 << 20] {
            let scalar = scalar_add_u64(base, 8 * iter);
            // Clamp bits preserved (so mul_base_clamped's re-clamp is a no-op).
            assert_eq!(scalar[0] & 0b0000_0111, 0);
            assert_eq!(scalar[31] & 0b1100_0000, 0b0100_0000);

            let candidate = candidate_from_scalar(scalar, [3u8; 32]);
            let mut key_scalar = [0u8; 32];
            key_scalar.copy_from_slice(&candidate.private_key[..32]);
            assert_eq!(key_scalar, scalar);
            let recomputed = EdwardsPoint::mul_base_clamped(key_scalar)
                .compress()
                .to_bytes();
            assert_eq!(recomputed, candidate.public_key);
        }
    }
}
