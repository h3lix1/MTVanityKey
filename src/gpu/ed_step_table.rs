#![cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))] // tables feed the GPU kernels only
//! Affine step tables for the Ed25519 window walk of both GPU backends
//! (`walk_windows` / `window_forward` in the kernels) and for the host-C++
//! kernel emulation tests.

use crate::ed_walk::{mul_base, Fe};

/// Affine table of the walk's step multiples i*S, S = 8*B: entry i-1 holds
/// (x, y, 2d*x*y) of i*S for i = 1..=window/2, and entry window/2 holds
/// window*S, as ten non-negative ref10 radix-2^25.5 limbs each (the
/// basepoint table's encoding, see build.rs). The kernels derive a window's
/// candidates as M +- i*S around the window middle M and advance M by
/// window*S.
pub(crate) fn step_table(window: u32) -> Vec<u32> {
    assert!(window >= 2 && window % 2 == 0, "window must be even");
    let half = window / 2;
    let mut scalar = [0u8; 32];
    scalar[0] = 8;
    let step = mul_base(&scalar);
    let step_niels = step.to_niels();
    let mut words = Vec::with_capacity((half as usize + 1) * 30);
    let mut point = step;
    for i in 1..=window {
        if i <= half || i == window {
            let (x, y) = point.to_affine();
            let t2d = point.to_niels().t2d;
            for fe in [x, y, t2d] {
                words.extend_from_slice(&ref10_limbs(&fe.to_bytes()));
            }
        }
        if i < window {
            point = point.add_niels(&step_niels);
        }
    }
    words
}

/// The Meshtastic kernels' step table: identical to `step_table` except the
/// third component is `d*x*y` (half of the Niels `2d*x*y`), so the walk's
/// K = (X*Y)*(d*x*y) is half the vanity kernels' and the Montgomery-u
/// numerators and denominators `1 -+ K +- (B +- A)` sum only three fe_mul
/// outputs, inside the kernels' fe_mul input bound (see mesh_kernel).
pub(crate) fn mesh_step_table(window: u32) -> Vec<u32> {
    assert!(window >= 2 && window % 2 == 0, "window must be even");
    // (p + 1) / 2 = 2^254 - 9: the inverse of 2.
    const FE_HALF: Fe = Fe([0xFFFF_FFFF_FFFF_FFF7, u64::MAX, u64::MAX, 0x3FFF_FFFF_FFFF_FFFF]);
    let half = window / 2;
    let mut scalar = [0u8; 32];
    scalar[0] = 8;
    let step = mul_base(&scalar);
    let step_niels = step.to_niels();
    let mut words = Vec::with_capacity((half as usize + 1) * 30);
    let mut point = step;
    for i in 1..=window {
        if i <= half || i == window {
            let (x, y) = point.to_affine();
            let td = point.to_niels().t2d.mul(&FE_HALF);
            for fe in [x, y, td] {
                words.extend_from_slice(&ref10_limbs(&fe.to_bytes()));
            }
        }
        if i < window {
            point = point.add_niels(&step_niels);
        }
    }
    words
}

/// Number of u32 words a step table for `window` holds.
#[cfg(test)]
pub(crate) fn step_table_words(window: u32) -> usize {
    (window as usize / 2 + 1) * 30
}

/// The ten ref10 limbs (26/25-bit alternating, non-negative) of a canonical
/// little-endian field element encoding.
fn ref10_limbs(bytes: &[u8; 32]) -> [u32; 10] {
    const LIMB_BITS: [u32; 10] = [26, 25, 26, 25, 26, 25, 26, 25, 26, 25];
    let word = |i: usize| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().expect("8 bytes"));
    let value = [word(0), word(1), word(2), word(3)];
    let mut limbs = [0u32; 10];
    let mut pos = 0u32;
    for (limb, bits) in limbs.iter_mut().zip(LIMB_BITS) {
        let w = (pos / 64) as usize;
        let shift = pos % 64;
        let mut v = value[w] >> shift;
        if shift + bits > 64 && w + 1 < 4 {
            v |= value[w + 1] << (64 - shift);
        }
        *limb = (v & ((1u64 << bits) - 1)) as u32;
        pos += bits;
    }
    limbs
}

#[cfg(test)]
mod tests {
    use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
    use curve25519_dalek::scalar::Scalar;
    use num_bigint::BigUint;
    use num_traits::One;

    const LIMB_BITS: [u32; 10] = [26, 25, 26, 25, 26, 25, 26, 25, 26, 25];

    fn field(limbs: &[u32]) -> BigUint {
        let mut value = BigUint::ZERO;
        let mut pos = 0u32;
        for (limb, bits) in limbs.iter().zip(LIMB_BITS) {
            assert!((*limb as u64) < (1u64 << bits), "limb {limb} exceeds {bits} bits");
            value |= BigUint::from(*limb) << pos;
            pos += bits;
        }
        value
    }

    /// Every entry must be exactly i*S (S = 8*B) in dalek's view: y bytes and
    /// x parity equal dalek's compressed point, the point is on the curve and
    /// 2d*x*y is consistent; and the entries that coincide with the basepoint
    /// table's digit-0 multiples (8B..32B) reproduce its limbs exactly.
    #[test]
    fn step_table_matches_dalek_and_basepoint_table() {
        let p = (BigUint::one() << 255u32) - BigUint::from(19u32);
        let modinv = |a: &BigUint| a.modpow(&(&p - BigUint::from(2u32)), &p);
        let d = (&p - BigUint::from(121665u32)) * modinv(&BigUint::from(121666u32)) % &p;
        let basepoint = crate::table::basepoint_table();

        for window in [2u32, 8, 32, 256, 512] {
            let table = super::step_table(window);
            let half = window / 2;
            assert_eq!(table.len(), super::step_table_words(window));
            for entry in 0..=half {
                let multiple = if entry == half { window } else { entry + 1 };
                let words = &table[entry as usize * 30..entry as usize * 30 + 30];
                let x = field(&words[..10]);
                let y = field(&words[10..20]);
                let t2d = field(&words[20..30]);
                assert!(x < p && y < p && t2d < p);
                let x2 = &x * &x % &p;
                let y2 = &y * &y % &p;
                let lhs = (&p - &x2 + &y2) % &p;
                let rhs = (BigUint::one() + &d * &x2 % &p * &y2) % &p;
                assert_eq!(lhs, rhs, "entry {entry} (x{multiple}) off the curve");
                assert_eq!(
                    t2d,
                    BigUint::from(2u32) * &d % &p * &x % &p * &y % &p,
                    "entry {entry} 2dxy"
                );
                let point = ED25519_BASEPOINT_POINT * Scalar::from(8u32 * multiple);
                let compressed = point.compress().to_bytes();
                let mut y_bytes = compressed;
                y_bytes[31] &= 0x7F;
                assert_eq!(y.to_bytes_le(), trimmed(&y_bytes), "entry {entry} y");
                let x_parity = (compressed[31] >> 7) as u32;
                assert_eq!(
                    (&x % BigUint::from(2u32)).to_u32_digits().first().copied().unwrap_or(0),
                    x_parity,
                    "entry {entry} x parity"
                );
                // Basepoint table digit 0 stores j*B for j = 1..32 as
                // (y+x, y-x, 2dxy); the shared multiples must agree limb for
                // limb in value.
                let j = 8 * multiple;
                if j <= 32 {
                    let base = (j as usize - 1) * 30;
                    let bp_ypx = field(&basepoint[base..base + 10]);
                    let bp_ymx = field(&basepoint[base + 10..base + 20]);
                    let bp_t2d = field(&basepoint[base + 20..base + 30]);
                    assert_eq!((&y + &x) % &p, bp_ypx % &p, "entry {entry} y+x vs basepoint table");
                    assert_eq!((&y + &p - &x) % &p, bp_ymx % &p, "entry {entry} y-x vs basepoint table");
                    assert_eq!(t2d, bp_t2d % &p, "entry {entry} 2dxy vs basepoint table");
                }
            }
        }
    }

    /// The mesh table shares x and y with the vanity table and holds exactly
    /// half its 2dxy component (d*x*y), reduced.
    #[test]
    fn mesh_step_table_is_the_step_table_with_dxy() {
        let p = (BigUint::one() << 255u32) - BigUint::from(19u32);
        for window in [2u32, 32, 256] {
            let a = super::step_table(window);
            let b = super::mesh_step_table(window);
            assert_eq!(a.len(), b.len());
            for entry in 0..=(window / 2) as usize {
                let (wa, wb) = (&a[entry * 30..entry * 30 + 30], &b[entry * 30..entry * 30 + 30]);
                assert_eq!(&wa[..20], &wb[..20], "x, y of entry {entry}");
                let t2d = field(&wa[20..30]);
                let td = field(&wb[20..30]);
                assert!(td < p);
                assert_eq!((&td * BigUint::from(2u32)) % &p, t2d, "entry {entry} d*x*y");
            }
        }
    }

    fn trimmed(bytes: &[u8; 32]) -> Vec<u8> {
        let mut v = bytes.to_vec();
        while v.last() == Some(&0) {
            v.pop();
        }
        v
    }
}
