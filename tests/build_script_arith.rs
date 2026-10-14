//! Differential tests for the arithmetic in build.rs.
//!
//! The build script is compiled straight into this test crate (`include!`),
//! so its private field / curve helpers can be exercised against plain
//! BigUint formulas (the ones the pristine script used, one modular inversion
//! per point) and against curve25519-dalek, on random and edge-case inputs.
//! The exhaustive table-vs-reference check lives next to the table
//! (src/table.rs); these tests cover the primitives on inputs the table walk
//! never reaches (0, 1, p-1, equal points, non-trivial Z, limb boundaries).
#![allow(dead_code)]

include!("../build.rs");

use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
use curve25519_dalek::scalar::Scalar;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn rng() -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(0x5eed_b01d_7ab1e)
}

fn random_below(rng: &mut ChaCha20Rng, p: &BigUint) -> BigUint {
    let mut bytes = [0u8; 40];
    rng.fill_bytes(&mut bytes);
    BigUint::from_bytes_le(&bytes) % p
}

/// Edge values of a prime field plus a few limb-boundary patterns.
fn edge_values(p: &BigUint) -> Vec<BigUint> {
    let mut v = vec![
        BigUint::zero(),
        BigUint::one(),
        BigUint::from(2u32),
        p - BigUint::one(),
        p - BigUint::from(2u32),
        (p - BigUint::one()) >> 1usize,
        ((p - BigUint::one()) >> 1usize) + BigUint::one(),
    ];
    for bits in [25usize, 26, 51, 52, 64, 77, 128, 191, 192, 230, 254, 255] {
        for cand in [
            (BigUint::one() << bits) - BigUint::one(),
            BigUint::one() << bits,
            (BigUint::one() << bits) + BigUint::one(),
        ] {
            if &cand < p {
                v.push(cand);
            }
        }
    }
    v
}

fn modpow_inv(a: &BigUint, p: &BigUint) -> BigUint {
    a.modpow(&(p - BigUint::from(2u32)), p)
}

fn fields() -> [Field; 1] {
    [Field { p: modulus() }]
}

#[test]
fn field_add_sub_mul_match_plain_biguint() {
    let mut rng = rng();
    for f in fields() {
        let p = &f.p;
        let mut values = edge_values(p);
        for _ in 0..200 {
            values.push(random_below(&mut rng, p));
        }
        for a in &values {
            for b in values.iter().chain(std::iter::once(a)) {
                let sum = (a + b) % p;
                let diff = (a + p - b) % p;
                let prod = (a * b) % p;
                assert_eq!(f.add(a, b), sum, "add {a} {b}");
                assert_eq!(f.add(b, a), sum, "add {b} {a}");
                assert_eq!(f.sub(a, b), diff, "sub {a} {b}");
                assert_eq!(f.mul(a, b), prod, "mul {a} {b}");
                assert!(f.add(a, b) < *p && f.sub(a, b) < *p && f.mul(a, b) < *p);
            }
            assert_eq!(f.sq(a), (a * a) % p, "sq {a}");
        }
    }
}

#[test]
fn inv_and_batch_inv_match_fermat() {
    let mut rng = rng();
    for f in fields() {
        let p = &f.p;
        let mut values: Vec<BigUint> = edge_values(p).into_iter().filter(|v| !v.is_zero()).collect();
        for _ in 0..100 {
            let v = random_below(&mut rng, p);
            if !v.is_zero() {
                values.push(v);
            }
        }
        for v in &values {
            assert_eq!(f.inv(v), modpow_inv(v, p), "inv {v}");
        }
        // Batches of every size from 1 up, including the table's row size (32),
        // and a batch with repeated elements.
        for len in [1usize, 2, 3, 31, 32, 33, values.len()] {
            let batch: Vec<BigUint> = values.iter().cycle().take(len).cloned().collect();
            let got = f.batch_inv(&batch);
            assert_eq!(got.len(), len);
            for (v, inv) in batch.iter().zip(&got) {
                assert_eq!(inv, &modpow_inv(v, p), "batch_inv len {len} value {v}");
                assert_eq!(f.mul(v, inv), BigUint::one());
            }
        }
        let repeated = vec![p - BigUint::one(); 32];
        for inv in f.batch_inv(&repeated) {
            assert_eq!(inv, p - BigUint::one());
        }
        assert!(f.batch_inv(&[]).is_empty());
    }
}

#[test]
#[should_panic(expected = "inverse of zero")]
fn batch_inv_rejects_zero_element() {
    let f = Field { p: modulus() };
    let mut batch = vec![BigUint::from(3u32); 32];
    batch[17] = BigUint::zero();
    let _ = f.batch_inv(&batch);
}

/// The pristine limb decomposition: BigUint mask / shift per limb.
fn write_field_reference(words: &mut Vec<u32>, value: &BigUint) {
    let mut rest = value.clone();
    for bits in LIMB_BITS {
        let mask = (BigUint::one() << bits) - BigUint::one();
        let limb: u32 = (&rest & &mask).try_into().unwrap();
        words.push(limb);
        rest >>= bits;
    }
    assert!(rest.is_zero());
}

#[test]
fn limb_extraction_matches_biguint_reference() {
    let mut rng = rng();
    let p = modulus();
    let mut ed_values = edge_values(&p);
    for _ in 0..2000 {
        ed_values.push(random_below(&mut rng, &p));
    }
    // Sparse patterns: single bits and runs of ones straddling every limb edge.
    for bit in 0..255usize {
        ed_values.push(BigUint::one() << bit);
        ed_values.push(((BigUint::one() << bit) - BigUint::one()) % &p);
    }
    let mut got = Vec::new();
    let mut want = Vec::new();
    for v in &ed_values {
        write_field(&mut got, v, &p);
        write_field_reference(&mut want, v);
        assert_eq!(got, want, "ed25519 limbs of {v}");
        let back: BigUint = got
            .iter()
            .zip(LIMB_BITS)
            .scan(0usize, |pos, (w, bits)| {
                let term = BigUint::from(*w) << *pos;
                *pos += bits;
                Some(term)
            })
            .sum();
        assert_eq!(back, *v);
        got.clear();
        want.clear();
    }
    // extract_bits over a 2-word boundary with every offset.
    let digits = [0xDEAD_BEEF_0123_4567u64, 0x89AB_CDEF_FEDC_BA98u64, 0, 0];
    let full = BigUint::from(digits[0]) | (BigUint::from(digits[1]) << 64usize);
    for pos in 0..64usize {
        for bits in [1usize, 25, 26, 32] {
            let want: u32 = ((&full >> pos) & ((BigUint::one() << bits) - BigUint::one()))
                .try_into()
                .unwrap();
            assert_eq!(extract_bits(&digits, pos, bits), want, "pos {pos} bits {bits}");
        }
    }
}

#[test]
#[should_panic(expected = "must be reduced")]
fn write_field_rejects_unreduced_value() {
    let p = modulus();
    let mut words = Vec::new();
    write_field(&mut words, &p, &p);
}

// ---------------------------------------------------------------------------
// Ed25519
// ---------------------------------------------------------------------------

/// Pristine affine addition (two Fermat inversions per add).
fn ed_affine_add_reference(p1: &(BigUint, BigUint), p2: &(BigUint, BigUint)) -> (BigUint, BigUint) {
    let p = modulus();
    let d = ((&p - BigUint::from(121665u32)) * modpow_inv(&BigUint::from(121666u32), &p)) % &p;
    let x1x2 = (&p1.0 * &p2.0) % &p;
    let y1y2 = (&p1.1 * &p2.1) % &p;
    let x1y2 = (&p1.0 * &p2.1) % &p;
    let y1x2 = (&p1.1 * &p2.0) % &p;
    let dxxyy = (&d * ((&x1x2 * &y1y2) % &p)) % &p;
    let x_num = (&x1y2 + &y1x2) % &p;
    let x_den = (BigUint::one() + &dxxyy) % &p;
    let y_num = (&y1y2 + &x1x2) % &p;
    let y_den = (BigUint::one() + &p - &dxxyy) % &p;
    (
        (&x_num * modpow_inv(&x_den, &p)) % &p,
        (&y_num * modpow_inv(&y_den, &p)) % &p,
    )
}

/// Decompress a dalek-encoded point with plain bignum arithmetic.
fn decompress(ed: &Edwards, bytes: [u8; 32]) -> (BigUint, BigUint) {
    let f = &ed.f;
    let p = &f.p;
    let x_is_odd = bytes[31] >> 7 == 1;
    let mut y_bytes = bytes;
    y_bytes[31] &= 0x7f;
    let y = BigUint::from_bytes_le(&y_bytes);
    let y2 = f.sq(&y);
    let u = f.sub(&y2, &BigUint::one());
    let v = f.add(&f.mul(&ed.d, &y2), &BigUint::one());
    let x2 = f.mul(&u, &modpow_inv(&v, p));
    let mut x = x2.modpow(&((p + BigUint::from(3u32)) >> 3usize), p);
    if f.sq(&x) != x2 {
        let sqrt_m1 = BigUint::from(2u32).modpow(&((p - BigUint::one()) >> 2usize), p);
        x = f.mul(&x, &sqrt_m1);
    }
    assert_eq!(f.sq(&x), x2, "not a square");
    if x.bit(0) != x_is_odd {
        x = p - &x;
    }
    (x, y)
}

fn to_affine(ed: &Edwards, p: &Extended) -> (BigUint, BigUint) {
    let zinv = ed.f.inv(&p.z);
    (ed.f.mul(&p.x, &zinv), ed.f.mul(&p.y, &zinv))
}

/// (x, y) in extended coordinates with a random projective scaling.
fn to_extended(ed: &Edwards, rng: &mut ChaCha20Rng, pt: &(BigUint, BigUint)) -> Extended {
    let f = &ed.f;
    let mut z = random_below(rng, &f.p);
    if z.is_zero() {
        z = BigUint::one();
    }
    Extended {
        x: f.mul(&pt.0, &z),
        y: f.mul(&pt.1, &z),
        z: z.clone(),
        t: f.mul(&f.mul(&pt.0, &pt.1), &z),
    }
}

fn random_ed_point(ed: &Edwards, rng: &mut ChaCha20Rng) -> (BigUint, BigUint) {
    let mut wide = [0u8; 64];
    rng.fill_bytes(&mut wide);
    let point = ED25519_BASEPOINT_POINT * Scalar::from_bytes_mod_order_wide(&wide);
    decompress(ed, point.compress().to_bytes())
}

#[test]
fn edwards_add_matches_affine_reference_and_dalek() {
    let mut rng = rng();
    let ed = Edwards::new();
    let f = &ed.f;
    let p = &f.p;
    let d_ref = ((p - BigUint::from(121665u32)) * modpow_inv(&BigUint::from(121666u32), p)) % p;
    assert_eq!(ed.d, d_ref);
    assert_eq!(ed.d2, (&d_ref * 2u32) % p);

    let b = basepoint(&ed);
    let (bx, by) = to_affine(&ed, &b);
    assert_eq!(decompress(&ed, ED25519_BASEPOINT_POINT.compress().to_bytes()), (bx, by));

    let identity = (BigUint::zero(), BigUint::one());
    let mut points = vec![identity.clone(), to_affine(&ed, &b)];
    for _ in 0..16 {
        points.push(random_ed_point(&ed, &mut rng));
    }
    // Negations: (-x, y) and the order-2 / order-4 points.
    let negs: Vec<_> = points.iter().map(|(x, y)| (f.sub(&BigUint::zero(), x), y.clone())).collect();
    points.extend(negs);
    points.push((BigUint::zero(), p - BigUint::one())); // order 2
    let sqrt_m1 = BigUint::from(2u32).modpow(&((p - BigUint::one()) >> 2usize), p);
    points.push((sqrt_m1.clone(), BigUint::zero())); // order 4
    points.push((f.sub(&BigUint::zero(), &sqrt_m1), BigUint::zero()));

    for pt in &points {
        assert!(ed.on_curve(&pt.0, &pt.1), "input not on curve");
    }
    let mut checked = 0usize;
    for (i, a) in points.iter().enumerate() {
        for bpt in points.iter().skip(i) {
            let ea = to_extended(&ed, &mut rng, a);
            let eb = to_extended(&ed, &mut rng, bpt);
            let sum = ed.add(&ea, &eb);
            let sum_rev = ed.add(&eb, &ea);
            assert!(!sum.z.is_zero(), "projective Z vanished for {a:?} + {bpt:?}");
            // T must stay consistent with X*Y/Z.
            assert_eq!(f.mul(&sum.x, &sum.y), f.mul(&sum.t, &sum.z), "T inconsistent");
            let got = to_affine(&ed, &sum);
            assert_eq!(got, to_affine(&ed, &sum_rev), "addition not commutative");
            assert_eq!(got, ed_affine_add_reference(a, bpt), "{a:?} + {bpt:?}");
            assert!(ed.on_curve(&got.0, &got.1));
            checked += 1;
        }
    }
    assert!(checked > 500);

    // Doubling through the unified add vs dalek, on random points.
    for _ in 0..40 {
        let mut wide = [0u8; 64];
        rng.fill_bytes(&mut wide);
        let s = Scalar::from_bytes_mod_order_wide(&wide);
        let pt = ED25519_BASEPOINT_POINT * s;
        let a = to_extended(&ed, &mut rng, &decompress(&ed, pt.compress().to_bytes()));
        let dbl = to_affine(&ed, &ed.add(&a, &a));
        assert_eq!(dbl, decompress(&ed, (pt + pt).compress().to_bytes()));
        let plus_b = to_affine(&ed, &ed.add(&a, &b));
        assert_eq!(plus_b, decompress(&ed, (pt + ED25519_BASEPOINT_POINT).compress().to_bytes()));
    }
}

#[test]
fn edwards_table_walk_matches_dalek_scalar_mults() {
    // Same walk as generate_table() (projective, one batch inversion per
    // row) checked at the affine level against dalek, including the row
    // boundary where `point` is advanced by six doublings.
    let mut rng = rng();
    let ed = Edwards::new();
    let f = &ed.f;
    let mut point = basepoint(&ed);
    for digit in 0..DIGITS {
        let mut multiples = vec![point.clone()];
        for k in 1..MULTIPLES {
            multiples.push(ed.add(&multiples[k - 1], &point));
        }
        let zinvs = f.batch_inv(&multiples.iter().map(|m| m.z.clone()).collect::<Vec<_>>());
        // Only spot-check rows (the exhaustive check is in src/table.rs).
        if digit % 7 == 0 || digit == DIGITS - 1 {
            for (k, (m, zinv)) in multiples.iter().zip(&zinvs).enumerate() {
                let got = (f.mul(&m.x, zinv), f.mul(&m.y, zinv));
                let scalar = BigUint::from(k as u32 + 1) << (RADIX_BITS * digit);
                let mut wide = [0u8; 64];
                let bytes = scalar.to_bytes_le();
                wide[..bytes.len()].copy_from_slice(&bytes);
                let want = ED25519_BASEPOINT_POINT * Scalar::from_bytes_mod_order_wide(&wide);
                assert_eq!(got, decompress(&ed, want.compress().to_bytes()), "digit {digit} k {k}");
            }
        }
        for _ in 0..RADIX_BITS {
            point = ed.add(&point, &point);
        }
        let _ = &mut rng;
    }
}

#[test]
fn generated_tables_are_deterministic_and_well_formed() {
    let a = generate_table();
    let b = generate_table();
    assert_eq!(a, b);
    assert_eq!(a.len(), WORDS);
    for (i, w) in a.iter().enumerate() {
        assert!((*w as u64) < (1u64 << LIMB_BITS[i % LIMBS]), "word {i} overflows its limb");
    }
}
