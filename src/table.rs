include!(concat!(env!("OUT_DIR"), "/basepoint_table.rs"));

pub fn basepoint_table() -> &'static [u32; BASEPOINT_TABLE_WORDS] {
    &BASEPOINT_TABLE
}

#[cfg(test)]
mod tests {
    use super::{basepoint_table, BASEPOINT_TABLE_WORDS};
    use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
    use curve25519_dalek::scalar::Scalar;
    use num_bigint::BigUint;
    use num_traits::One;

    const DIGITS: usize = 43;
    const MULTIPLES: usize = 32;
    const RADIX_BITS: usize = 6;
    /// ref10 radix-2^25.5 limb sizes (must match build.rs).
    const LIMB_BITS: [usize; 10] = [26, 25, 26, 25, 26, 25, 26, 25, 26, 25];

    fn p() -> BigUint {
        (BigUint::one() << 255usize) - BigUint::from(19u32)
    }

    fn inv(a: &BigUint, p: &BigUint) -> BigUint {
        a.modpow(&(p - BigUint::from(2u32)), p)
    }

    /// Reassemble a field element from its ten table limbs.
    fn field(words: &[u32]) -> BigUint {
        let mut value = BigUint::ZERO;
        let mut pos = 0usize;
        for (word, bits) in words.iter().zip(LIMB_BITS) {
            assert!(
                (*word as u64) < (1u64 << bits),
                "limb {word} does not fit in {bits} bits"
            );
            value |= BigUint::from(*word) << pos;
            pos += bits;
        }
        value
    }

    /// Decompress a dalek-encoded point (little-endian y, sign of x in the top
    /// bit) into affine (x, y) with plain bignum arithmetic, independently of
    /// both dalek's field code and build.rs.
    fn decompress(p: &BigUint, d: &BigUint, bytes: [u8; 32]) -> (BigUint, BigUint) {
        let x_is_odd = bytes[31] >> 7 == 1;
        let mut y_bytes = bytes;
        y_bytes[31] &= 0x7f;
        let y = BigUint::from_bytes_le(&y_bytes);
        assert!(&y < p, "compressed y is not reduced");

        // x^2 = (y^2 - 1) / (d y^2 + 1)
        let y2 = (&y * &y) % p;
        let u = (&y2 + p - BigUint::one()) % p;
        let v = (d * &y2 + BigUint::one()) % p;
        let x2 = (&u * inv(&v, p)) % p;
        // p == 5 (mod 8): a root is x2^((p+3)/8), possibly times sqrt(-1).
        let mut x = x2.modpow(&((p + BigUint::from(3u32)) >> 3usize), p);
        if (&x * &x) % p != x2 {
            let sqrt_m1 = BigUint::from(2u32).modpow(&((p - BigUint::one()) >> 2usize), p);
            x = (&x * sqrt_m1) % p;
        }
        assert_eq!((&x * &x) % p, x2, "not a square");
        if x.bit(0) != x_is_odd {
            x = p - &x;
        }
        (x, y)
    }

    #[test]
    fn basepoint_table_matches_dalek() {
        // Every entry of the generated table must be the affine Niels form
        // (y + x, y - x, 2 d x y) of (k + 1) * 2^(6 d) * B, with the point
        // itself computed by curve25519-dalek and decompressed here.
        let p = p();
        let d = ((&p - BigUint::from(121665u32)) * inv(&BigUint::from(121666u32), &p)) % &p;
        let two_d = (&d * 2u32) % &p;
        let table = basepoint_table();
        assert_eq!(BASEPOINT_TABLE_WORDS, DIGITS * MULTIPLES * 3 * LIMB_BITS.len());

        for digit in 0..DIGITS {
            for k in 0..MULTIPLES {
                let scalar = BigUint::from(k as u32 + 1) << (RADIX_BITS * digit);
                let mut wide = [0u8; 64];
                let bytes = scalar.to_bytes_le();
                wide[..bytes.len()].copy_from_slice(&bytes);
                let point = ED25519_BASEPOINT_POINT * Scalar::from_bytes_mod_order_wide(&wide);
                let (x, y) = decompress(&p, &d, point.compress().to_bytes());

                let off = (digit * MULTIPLES + k) * 3 * LIMB_BITS.len();
                let entry = &table[off..off + 3 * LIMB_BITS.len()];
                let y_plus_x = field(&entry[..10]);
                let y_minus_x = field(&entry[10..20]);
                let xy2d = field(&entry[20..30]);
                assert!(y_plus_x < p && y_minus_x < p && xy2d < p, "entry not reduced");
                let context = format!("digit {digit} multiple {}", k + 1);
                assert_eq!(y_plus_x, (&y + &x) % &p, "y+x of {context}");
                assert_eq!(y_minus_x, (&y + &p - &x) % &p, "y-x of {context}");
                assert_eq!(xy2d, (&two_d * &x % &p) * &y % &p, "2dxy of {context}");
            }
        }
    }
}
