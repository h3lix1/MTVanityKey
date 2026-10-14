//! Incremental Ed25519 walk of the MeshCore CPU workers (and the field and
//! point arithmetic the Meshtastic u-walk in `mesh_walk` builds on).
//!
//! The search visits the clamped scalars `s, s+8, s+16, ...` exactly like
//! the GPU kernels. Deriving every candidate with a fresh fixed-base scalar
//! multiplication costs ~500 field multiplications plus an inversion per
//! key; instead the walk keeps, for `WINDOW` interleaved chains per lane,
//! the affine `y` coordinates of the chain's current point `P_i = s_i * B`
//! and of `P_i - D` (`D = 8 * WINDOW * B`), and steps every chain with the
//! y-only differential addition described at [`Walk`] (2 multiplications
//! and 2 squarings). The one inversion each step needs is batched over all
//! chains of the window (3 multiplications per chain), so a candidate
//! costs about 6 multiplications and 2 squarings, and the canonical `y`
//! encoding — the public key with the sign bit of `x` not yet known — falls
//! out of the step directly.
//!
//! `curve25519-dalek` v4 exposes neither field elements nor point
//! coordinates, so the arithmetic lives here: a 4x64-bit field (any value in
//! [0, 2^256) represents its residue, 2^256 = 38 is folded eagerly, so no
//! limb-bound tracking is needed; chosen over the radix-2^51 form because
//! the walk is bound by multiplier throughput and this needs 20 wide
//! multiplies per `mul` instead of 25), with hand-scheduled aarch64
//! assembly for `mul`/`sq`/`add`/`sub` that is bit-exact with the portable
//! code (differentially tested), dalek's complete formulas for doubling and
//! mixed addition (used to seed the chains), and a variable-time
//! double-and-add for the one scalar multiplication per lane per (re)start.
//!
//! Correctness does not rest on this code alone: every seed point of every
//! chain is compared with dalek's derivation at each (re)start, callers
//! re-check the walk against dalek periodically ([`Walk::matches_dalek`])
//! and drop to the per-candidate dalek derivation on a mismatch, and every
//! hit is re-derived with dalek from its scalar before it is reported. A
//! bug here could therefore only ever miss keys, never report a wrong one.

use std::cell::Cell;

use rand_core::RngCore;

use crate::keygen::{random_base_scalar, scalar_add_u64};

/// Field element mod 2^255 - 19 as four little-endian 64-bit words holding
/// any value in [0, 2^256) congruent to the element.
#[derive(Clone, Copy, Debug)]
pub struct Fe(pub [u64; 4]);

pub const FE_ZERO: Fe = Fe([0, 0, 0, 0]);
pub const FE_ONE: Fe = Fe([1, 0, 0, 0]);

/// Convert radix-2^51 limbs (each < 2^51, as dalek prints its constants)
/// to four 64-bit words.
const fn from_limbs51(l: [u64; 5]) -> Fe {
    Fe([
        l[0] | (l[1] << 51),
        (l[1] >> 13) | (l[2] << 38),
        (l[2] >> 26) | (l[3] << 25),
        (l[3] >> 39) | (l[4] << 12),
    ])
}

/// 2*d where d = -121665/121666 (Ed25519 curve constant).
const EDWARDS_D2: Fe = from_limbs51([
    1859910466990425,
    932731440258426,
    1072319116312658,
    1815898335770999,
    633789495995903,
]);

/// Ed25519 basepoint B in affine coordinates.
const BASE_X: Fe = from_limbs51([
    1738742601995546,
    1146398526822698,
    2070867633025821,
    562264141797630,
    587772402128613,
]);
const BASE_Y: Fe = from_limbs51([
    1801439850948184,
    1351079888211148,
    450359962737049,
    900719925474099,
    1801439850948198,
]);

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
fn m(a: u64, b: u64) -> u128 {
    (a as u128) * (b as u128)
}

/// Fold the 8-word product `t` (< 2^512) back to four words using
/// 2^256 = 38 (mod p). Output < 2^256.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
fn fold512(t: &[u64; 8]) -> Fe {
    let p = m(t[4], 38) + t[0] as u128;
    let r0 = p as u64;
    let p = m(t[5], 38) + t[1] as u128 + (p >> 64);
    let r1 = p as u64;
    let p = m(t[6], 38) + t[2] as u128 + (p >> 64);
    let r2 = p as u64;
    let p = m(t[7], 38) + t[3] as u128 + (p >> 64);
    let r3 = p as u64;
    // c <= 38 (39 * 2^256 - 39 < 2^262); fold c * 38 (<= 1444) again.
    let c = (p >> 64) as u64;
    let (r0, c) = r0.overflowing_add(c * 38);
    let (r1, c) = r1.overflowing_add(c as u64);
    let (r2, c) = r2.overflowing_add(c as u64);
    let (r3, c) = r3.overflowing_add(c as u64);
    // A carry here means the value wrapped past 2^256 to something < 1444,
    // so one more +38 cannot carry.
    Fe([r0 + (c as u64) * 38, r1, r2, r3])
}

/// The eight little-endian 32-bit limbs of a field element's words.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[inline(always)]
fn limbs32(a: &Fe) -> [u32; 8] {
    let w = &a.0;
    [
        w[0] as u32, (w[0] >> 32) as u32, w[1] as u32, (w[1] >> 32) as u32,
        w[2] as u32, (w[2] >> 32) as u32, w[3] as u32, (w[3] >> 32) as u32,
    ]
}

/// Sixteen 32-bit limbs of the 512-bit sum of the column sums `col` (each
/// below 2^37; the value is a product of two numbers below 2^256, so
/// nothing carries out of limb 15 and `col[16]` is zero).
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[inline(always)]
fn carry_columns(col: &[u64; 17]) -> [u32; 16] {
    let mut r = [0u32; 16];
    let mut carry = 0u64;
    for k in 0..16 {
        let v = col[k] + carry;
        r[k] = v as u32;
        carry = v >> 32;
    }
    r
}

/// `fold512` on sixteen 32-bit limbs: `low + 38 * high`, then the carry
/// out of 2^256 (at most 38) folded as `c * 38`, and a second wrap (value
/// then below 1444) as one more +38 that cannot carry. Output < 2^256.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[inline(always)]
fn fold512_32(t: &[u32; 16]) -> Fe {
    let mut r = [0u32; 8];
    let mut carry = 0u64;
    for k in 0..8 {
        let v = t[k] as u64 + 38 * t[k + 8] as u64 + carry;
        r[k] = v as u32;
        carry = v >> 32;
    }
    let mut c = carry * 38;
    for limb in r.iter_mut() {
        let v = *limb as u64 + c;
        *limb = v as u32;
        c = v >> 32;
    }
    let w = |i: usize| r[i] as u64 | (r[i + 1] as u64) << 32;
    Fe([w(0) + c * 38, w(2), w(4), w(6)])
}

impl Fe {
    /// Decode 32 little-endian bytes (any value in [0, 2^256) accepted).
    #[cfg(test)]
    pub fn from_bytes(bytes: &[u8; 32]) -> Fe {
        let w = |i: usize| u64::from_le_bytes(bytes[i..i + 8].try_into().expect("8 bytes"));
        Fe([w(0), w(8), w(16), w(24)])
    }

    /// Canonical little-endian encoding (value fully reduced into [0, p),
    /// so bit 255 is always clear). Identical to dalek's `to_bytes`.
    #[inline]
    pub fn to_bytes(&self) -> [u8; 32] {
        let w = self.0;
        // v = q * 2^255 + low, q in {0, 1}; v = low + 19 q (mod p), < 2^255 + 19.
        let q = w[3] >> 63;
        let (r0, c) = w[0].overflowing_add(19 * q);
        let (r1, c) = w[1].overflowing_add(c as u64);
        let (r2, c) = w[2].overflowing_add(c as u64);
        let r3 = (w[3] & 0x7FFF_FFFF_FFFF_FFFF) + c as u64;
        // r >= p  <=>  r + 19 >= 2^255: subtract p by adding 19 and masking.
        let (t0, c) = r0.overflowing_add(19);
        let (t1, c) = r1.overflowing_add(c as u64);
        let (t2, c) = r2.overflowing_add(c as u64);
        let t3 = r3 + c as u64;
        let ge = t3 >> 63; // 1 iff r >= p
        let mask = ge.wrapping_neg();
        let r0 = (r0 & !mask) | (t0 & mask);
        let r1 = (r1 & !mask) | (t1 & mask);
        let r2 = (r2 & !mask) | (t2 & mask);
        let r3 = (r3 & !mask) | ((t3 & 0x7FFF_FFFF_FFFF_FFFF) & mask);
        let mut out = [0u8; 32];
        out[0..8].copy_from_slice(&r0.to_le_bytes());
        out[8..16].copy_from_slice(&r1.to_le_bytes());
        out[16..24].copy_from_slice(&r2.to_le_bytes());
        out[24..32].copy_from_slice(&r3.to_le_bytes());
        out
    }

    /// `a + b` folded back below 2^256.
    #[inline(always)]
    pub fn add(&self, b: &Fe) -> Fe {
        #[cfg(target_arch = "aarch64")]
        {
            self.add_aarch64(b)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.add_portable(b)
        }
    }

    /// `a - b` (mod p), result < 2^256.
    #[inline(always)]
    pub fn sub(&self, b: &Fe) -> Fe {
        #[cfg(target_arch = "aarch64")]
        {
            self.sub_aarch64(b)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.sub_portable(b)
        }
    }

    /// `add_portable` as one `adds`/`adcs` chain per fold on aarch64: the
    /// 2^256 carry becomes +38 (`csel`), and a second wrap (value then
    /// < 38) one more +38 that cannot carry. Bit-exact with `add_portable`.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub fn add_aarch64(&self, b: &Fe) -> Fe {
        let a = &self.0;
        let b = &b.0;
        let (r0, r1, r2, r3): (u64, u64, u64, u64);
        // SAFETY: register-only arithmetic, no memory or stack use.
        unsafe {
            core::arch::asm!(
                "adds  {r0}, {a0}, {b0}",
                "adcs  {r1}, {a1}, {b1}",
                "adcs  {r2}, {a2}, {b2}",
                "adcs  {r3}, {a3}, {b3}",
                "mov   {k}, #38",
                "csel  {c}, {k}, xzr, cs",
                "adds  {r0}, {r0}, {c}",
                "adcs  {r1}, {r1}, xzr",
                "adcs  {r2}, {r2}, xzr",
                "adcs  {r3}, {r3}, xzr",
                "csel  {c}, {k}, xzr, cs",
                "add   {r0}, {r0}, {c}",
                r0 = out(reg) r0, r1 = out(reg) r1, r2 = out(reg) r2, r3 = out(reg) r3,
                k = out(reg) _, c = out(reg) _,
                a0 = in(reg) a[0], a1 = in(reg) a[1], a2 = in(reg) a[2], a3 = in(reg) a[3],
                b0 = in(reg) b[0], b1 = in(reg) b[1], b2 = in(reg) b[2], b3 = in(reg) b[3],
                options(pure, nomem, nostack),
            );
        }
        Fe([r0, r1, r2, r3])
    }

    /// `sub_portable` on aarch64 (a borrow out of 2^256 is repaid as -38,
    /// twice at most; `cc` is "borrow" for `subs`). Bit-exact with
    /// `sub_portable`.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub fn sub_aarch64(&self, b: &Fe) -> Fe {
        let a = &self.0;
        let b = &b.0;
        let (r0, r1, r2, r3): (u64, u64, u64, u64);
        // SAFETY: register-only arithmetic, no memory or stack use.
        unsafe {
            core::arch::asm!(
                "subs  {r0}, {a0}, {b0}",
                "sbcs  {r1}, {a1}, {b1}",
                "sbcs  {r2}, {a2}, {b2}",
                "sbcs  {r3}, {a3}, {b3}",
                "mov   {k}, #38",
                "csel  {c}, {k}, xzr, cc",
                "subs  {r0}, {r0}, {c}",
                "sbcs  {r1}, {r1}, xzr",
                "sbcs  {r2}, {r2}, xzr",
                "sbcs  {r3}, {r3}, xzr",
                "csel  {c}, {k}, xzr, cc",
                "sub   {r0}, {r0}, {c}",
                r0 = out(reg) r0, r1 = out(reg) r1, r2 = out(reg) r2, r3 = out(reg) r3,
                k = out(reg) _, c = out(reg) _,
                a0 = in(reg) a[0], a1 = in(reg) a[1], a2 = in(reg) a[2], a3 = in(reg) a[3],
                b0 = in(reg) b[0], b1 = in(reg) b[1], b2 = in(reg) b[2], b3 = in(reg) b[3],
                options(pure, nomem, nostack),
            );
        }
        Fe([r0, r1, r2, r3])
    }

    /// `a + b` folded back below 2^256 (portable).
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    #[inline(always)]
    pub fn add_portable(&self, b: &Fe) -> Fe {
        let a = &self.0;
        let b = &b.0;
        let (r0, c) = a[0].overflowing_add(b[0]);
        let (r1, c1) = a[1].overflowing_add(b[1]);
        let (r1, c2) = r1.overflowing_add(c as u64);
        let (r2, c1b) = a[2].overflowing_add(b[2]);
        let (r2, c2b) = r2.overflowing_add((c1 | c2) as u64);
        let (r3, c1c) = a[3].overflowing_add(b[3]);
        let (r3, c2c) = r3.overflowing_add((c1b | c2b) as u64);
        let c = (c1c | c2c) as u64;
        // Fold the 2^256 carry as +38; a second wrap leaves r < 38 so one
        // final +38 cannot carry.
        let (r0, c) = r0.overflowing_add(c * 38);
        let (r1, c) = r1.overflowing_add(c as u64);
        let (r2, c) = r2.overflowing_add(c as u64);
        let (r3, c) = r3.overflowing_add(c as u64);
        Fe([r0 + (c as u64) * 38, r1, r2, r3])
    }

    /// `a - b` (mod p), result < 2^256 (portable). A borrow out of 2^256
    /// is repaid as -38; if that borrows again the value is < 38 before the
    /// wrap, and one more -38 cannot borrow.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    #[inline(always)]
    pub fn sub_portable(&self, b: &Fe) -> Fe {
        let a = &self.0;
        let b = &b.0;
        let (r0, c) = a[0].overflowing_sub(b[0]);
        let (r1, c1) = a[1].overflowing_sub(b[1]);
        let (r1, c2) = r1.overflowing_sub(c as u64);
        let (r2, c1b) = a[2].overflowing_sub(b[2]);
        let (r2, c2b) = r2.overflowing_sub((c1 | c2) as u64);
        let (r3, c1c) = a[3].overflowing_sub(b[3]);
        let (r3, c2c) = r3.overflowing_sub((c1b | c2b) as u64);
        let c = (c1c | c2c) as u64;
        let (r0, c) = r0.overflowing_sub(c * 38);
        let (r1, c) = r1.overflowing_sub(c as u64);
        let (r2, c) = r2.overflowing_sub(c as u64);
        let (r3, c) = r3.overflowing_sub(c as u64);
        Fe([r0 - (c as u64) * 38, r1, r2, r3])
    }

    #[cfg(test)]
    pub fn neg(&self) -> Fe {
        FE_ZERO.sub(self)
    }

    /// `a * b` (mod p), result < 2^256: the hand-scheduled aarch64 schoolbook
    /// where available, the 32-bit-limb one on wasm32, otherwise the
    /// portable one.
    #[inline(always)]
    pub fn mul(&self, b: &Fe) -> Fe {
        #[cfg(target_arch = "aarch64")]
        {
            self.mul_aarch64(b)
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.mul_32(b)
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
        {
            self.mul_portable(b)
        }
    }

    /// Squaring (mod p), result < 2^256; see [`Fe::mul`].
    #[inline(always)]
    pub fn sq(&self) -> Fe {
        #[cfg(target_arch = "aarch64")]
        {
            self.sq_aarch64()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.sq_32()
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
        {
            self.sq_portable()
        }
    }

    /// `mul_portable` from 32x32->64-bit products, for targets without a
    /// 64x64->128-bit multiply: on wasm32 every `u128` product of the
    /// portable code is a call to compiler-rt's `__multi3`, while this is
    /// 64 inline `i64.mul`s. Product scanning over eight 32-bit limbs: the
    /// low and high halves of the products are summed per column (at most
    /// 8 * 2^32 each, so no carry until the end and no serial dependency
    /// between the multiplies), then one carry pass and the same
    /// `2^256 = 38` fold as `fold512`, so the result is bit-exact with
    /// `mul_portable` (differentially tested).
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    #[inline(always)]
    pub fn mul_32(&self, b: &Fe) -> Fe {
        let a = limbs32(self);
        let b = limbs32(b);
        let mut col = [0u64; 17];
        for i in 0..8 {
            for j in 0..8 {
                let p = a[i] as u64 * b[j] as u64;
                col[i + j] += p & 0xFFFF_FFFF;
                col[i + j + 1] += p >> 32;
            }
        }
        fold512_32(&carry_columns(&col))
    }

    /// `sq_portable` from 32x32->64-bit products (see `mul_32`): the 28
    /// off-diagonal products counted twice plus the 8 squares (36
    /// multiplies). Bit-exact with `sq_portable`.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    #[inline(always)]
    pub fn sq_32(&self) -> Fe {
        let a = limbs32(self);
        let mut col = [0u64; 17];
        for i in 0..8 {
            let p = a[i] as u64 * a[i] as u64;
            col[2 * i] += p & 0xFFFF_FFFF;
            col[2 * i + 1] += p >> 32;
            for j in i + 1..8 {
                let p = a[i] as u64 * a[j] as u64;
                col[i + j] += (p & 0xFFFF_FFFF) << 1;
                col[i + j + 1] += (p >> 32) << 1;
            }
        }
        fold512_32(&carry_columns(&col))
    }

    /// Schoolbook product (16 wide multiplies) folded by `fold512`.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    #[inline(always)]
    pub fn mul_portable(&self, b: &Fe) -> Fe {
        let a = &self.0;
        let b = &b.0;
        let mut t = [0u64; 8];

        let p = m(a[0], b[0]);
        t[0] = p as u64;
        let p = m(a[0], b[1]) + (p >> 64);
        t[1] = p as u64;
        let p = m(a[0], b[2]) + (p >> 64);
        t[2] = p as u64;
        let p = m(a[0], b[3]) + (p >> 64);
        t[3] = p as u64;
        t[4] = (p >> 64) as u64;

        let p = m(a[1], b[0]) + t[1] as u128;
        t[1] = p as u64;
        let p = m(a[1], b[1]) + t[2] as u128 + (p >> 64);
        t[2] = p as u64;
        let p = m(a[1], b[2]) + t[3] as u128 + (p >> 64);
        t[3] = p as u64;
        let p = m(a[1], b[3]) + t[4] as u128 + (p >> 64);
        t[4] = p as u64;
        t[5] = (p >> 64) as u64;

        let p = m(a[2], b[0]) + t[2] as u128;
        t[2] = p as u64;
        let p = m(a[2], b[1]) + t[3] as u128 + (p >> 64);
        t[3] = p as u64;
        let p = m(a[2], b[2]) + t[4] as u128 + (p >> 64);
        t[4] = p as u64;
        let p = m(a[2], b[3]) + t[5] as u128 + (p >> 64);
        t[5] = p as u64;
        t[6] = (p >> 64) as u64;

        let p = m(a[3], b[0]) + t[3] as u128;
        t[3] = p as u64;
        let p = m(a[3], b[1]) + t[4] as u128 + (p >> 64);
        t[4] = p as u64;
        let p = m(a[3], b[2]) + t[5] as u128 + (p >> 64);
        t[5] = p as u64;
        let p = m(a[3], b[3]) + t[6] as u128 + (p >> 64);
        t[6] = p as u64;
        t[7] = (p >> 64) as u64;

        fold512(&t)
    }

    /// Squaring (10 wide multiplies: the off-diagonal terms are doubled).
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    #[inline(always)]
    pub fn sq_portable(&self) -> Fe {
        let a = &self.0;
        // Off-diagonal terms (to be doubled): words 1..=6.
        let mut t = [0u64; 8];
        let p = m(a[0], a[1]);
        t[1] = p as u64;
        let p = m(a[0], a[2]) + (p >> 64);
        t[2] = p as u64;
        let p = m(a[0], a[3]) + (p >> 64);
        t[3] = p as u64;
        t[4] = (p >> 64) as u64;

        let p = m(a[1], a[2]) + t[3] as u128;
        t[3] = p as u64;
        let p = m(a[1], a[3]) + t[4] as u128 + (p >> 64);
        t[4] = p as u64;
        t[5] = (p >> 64) as u64;

        let p = m(a[2], a[3]) + t[5] as u128;
        t[5] = p as u64;
        t[6] = (p >> 64) as u64;

        // Double (sum of off-diagonal < 2^447, so t[7] takes the top bit).
        t[7] = t[6] >> 63;
        t[6] = (t[6] << 1) | (t[5] >> 63);
        t[5] = (t[5] << 1) | (t[4] >> 63);
        t[4] = (t[4] << 1) | (t[3] >> 63);
        t[3] = (t[3] << 1) | (t[2] >> 63);
        t[2] = (t[2] << 1) | (t[1] >> 63);
        t[1] <<= 1;

        // Add the diagonal.
        let p = m(a[0], a[0]);
        t[0] = p as u64;
        let p = t[1] as u128 + (p >> 64);
        t[1] = p as u64;
        let p = m(a[1], a[1]) + t[2] as u128 + (p >> 64);
        t[2] = p as u64;
        let p = t[3] as u128 + (p >> 64);
        t[3] = p as u64;
        let p = m(a[2], a[2]) + t[4] as u128 + (p >> 64);
        t[4] = p as u64;
        let p = t[5] as u128 + (p >> 64);
        t[5] = p as u64;
        let p = m(a[3], a[3]) + t[6] as u128 + (p >> 64);
        t[6] = p as u64;
        t[7] += (p >> 64) as u64;

        fold512(&t)
    }

    /// The same schoolbook product and `2^256 = 38` fold as `mul_portable`,
    /// hand-scheduled for aarch64: each row's four partial products are
    /// accumulated with two short `adds`/`adcs` chains instead of the
    /// `adds`/`cinc` pairs LLVM emits for `u128` arithmetic (87 instructions
    /// against 114; `extr` and other multi-µop instructions are avoided
    /// because they issue at only one per cycle on Apple cores). Bit-exact
    /// with `mul_portable` for every input (differentially tested).
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub fn mul_aarch64(&self, b: &Fe) -> Fe {
        let a = &self.0;
        let b = &b.0;
        let (t0, t1, t2, t3): (u64, u64, u64, u64);
        // SAFETY: pure register arithmetic on the eight input words; every
        // scratch register is declared as a clobbered output, no memory is
        // touched and the stack is not used.
        unsafe {
            core::arch::asm!(
                // row 0: t0..t4 = a0 * b
                "mul   {t0}, {a0}, {b0}",
                "umulh {t1}, {a0}, {b0}",
                "mul   {l1}, {a0}, {b1}",
                "umulh {h1}, {a0}, {b1}",
                "mul   {l2}, {a0}, {b2}",
                "umulh {h2}, {a0}, {b2}",
                "mul   {l3}, {a0}, {b3}",
                "umulh {h3}, {a0}, {b3}",
                "adds  {t1}, {t1}, {l1}",
                "adcs  {t2}, {h1}, {l2}",
                "adcs  {t3}, {h2}, {l3}",
                "adc   {t4}, {h3}, xzr",
                // row 1: t1..t5 += a1 * b (highs then lows; the running
                // value always fits its words, so no chain can overflow)
                "mul   {l0}, {a1}, {b0}",
                "umulh {h0}, {a1}, {b0}",
                "mul   {l1}, {a1}, {b1}",
                "umulh {h1}, {a1}, {b1}",
                "mul   {l2}, {a1}, {b2}",
                "umulh {h2}, {a1}, {b2}",
                "mul   {l3}, {a1}, {b3}",
                "umulh {h3}, {a1}, {b3}",
                "adds  {t1}, {t1}, {l0}",
                "adcs  {t2}, {t2}, {h0}",
                "adcs  {t3}, {t3}, {h1}",
                "adcs  {t4}, {t4}, {h2}",
                "adc   {t5}, {h3}, xzr",
                "adds  {t2}, {t2}, {l1}",
                "adcs  {t3}, {t3}, {l2}",
                "adcs  {t4}, {t4}, {l3}",
                "adc   {t5}, {t5}, xzr",
                // row 2
                "mul   {l0}, {a2}, {b0}",
                "umulh {h0}, {a2}, {b0}",
                "mul   {l1}, {a2}, {b1}",
                "umulh {h1}, {a2}, {b1}",
                "mul   {l2}, {a2}, {b2}",
                "umulh {h2}, {a2}, {b2}",
                "mul   {l3}, {a2}, {b3}",
                "umulh {h3}, {a2}, {b3}",
                "adds  {t2}, {t2}, {l0}",
                "adcs  {t3}, {t3}, {h0}",
                "adcs  {t4}, {t4}, {h1}",
                "adcs  {t5}, {t5}, {h2}",
                "adc   {t6}, {h3}, xzr",
                "adds  {t3}, {t3}, {l1}",
                "adcs  {t4}, {t4}, {l2}",
                "adcs  {t5}, {t5}, {l3}",
                "adc   {t6}, {t6}, xzr",
                // row 3
                "mul   {l0}, {a3}, {b0}",
                "umulh {h0}, {a3}, {b0}",
                "mul   {l1}, {a3}, {b1}",
                "umulh {h1}, {a3}, {b1}",
                "mul   {l2}, {a3}, {b2}",
                "umulh {h2}, {a3}, {b2}",
                "mul   {l3}, {a3}, {b3}",
                "umulh {h3}, {a3}, {b3}",
                "adds  {t3}, {t3}, {l0}",
                "adcs  {t4}, {t4}, {h0}",
                "adcs  {t5}, {t5}, {h1}",
                "adcs  {t6}, {t6}, {h2}",
                "adc   {t7}, {h3}, xzr",
                "adds  {t4}, {t4}, {l1}",
                "adcs  {t5}, {t5}, {l2}",
                "adcs  {t6}, {t6}, {l3}",
                "adc   {t7}, {t7}, xzr",
                // fold: r = t0..t3 + 38 * (t4..t7); the carry word c <= 38
                // is folded once more, and a final wrap leaves r < 1444 so
                // the last +38 cannot carry (as in `fold512`).
                "mov   {k}, #38",
                "mul   {l0}, {t4}, {k}",
                "umulh {h0}, {t4}, {k}",
                "mul   {l1}, {t5}, {k}",
                "umulh {h1}, {t5}, {k}",
                "mul   {l2}, {t6}, {k}",
                "umulh {h2}, {t6}, {k}",
                "mul   {l3}, {t7}, {k}",
                "umulh {h3}, {t7}, {k}",
                "adds  {t0}, {t0}, {l0}",
                "adcs  {t1}, {t1}, {l1}",
                "adcs  {t2}, {t2}, {l2}",
                "adcs  {t3}, {t3}, {l3}",
                "adc   {h3}, {h3}, xzr",
                "adds  {t1}, {t1}, {h0}",
                "adcs  {t2}, {t2}, {h1}",
                "adcs  {t3}, {t3}, {h2}",
                "adc   {h3}, {h3}, xzr",
                "mul   {h1}, {h3}, {k}",
                "adds  {t0}, {t0}, {h1}",
                "adcs  {t1}, {t1}, xzr",
                "adcs  {t2}, {t2}, xzr",
                "adcs  {t3}, {t3}, xzr",
                "csel  {h1}, {k}, xzr, cs",
                "add   {t0}, {t0}, {h1}",
                t0 = out(reg) t0, t1 = out(reg) t1, t2 = out(reg) t2, t3 = out(reg) t3,
                t4 = out(reg) _, t5 = out(reg) _, t6 = out(reg) _, t7 = out(reg) _,
                l0 = out(reg) _, l1 = out(reg) _, l2 = out(reg) _, l3 = out(reg) _,
                h0 = out(reg) _, h1 = out(reg) _, h2 = out(reg) _, h3 = out(reg) _,
                k = out(reg) _,
                a0 = in(reg) a[0], a1 = in(reg) a[1], a2 = in(reg) a[2], a3 = in(reg) a[3],
                b0 = in(reg) b[0], b1 = in(reg) b[1], b2 = in(reg) b[2], b3 = in(reg) b[3],
                options(pure, nomem, nostack),
            );
        }
        Fe([t0, t1, t2, t3])
    }

    /// `sq_portable` hand-scheduled for aarch64 (see `mul_aarch64`): the six
    /// off-diagonal products are accumulated, doubled by one `adds`/`adcs`
    /// chain, the diagonal added and the result folded exactly as in `mul`.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub fn sq_aarch64(&self) -> Fe {
        let a = &self.0;
        let (t0, t1, t2, t3): (u64, u64, u64, u64);
        // SAFETY: as for `mul_aarch64`.
        unsafe {
            core::arch::asm!(
                // off-diagonal: t1..t6 = a0a1 + (a0a2 + a1a2)<<64 + ... ;
                // the partial sums fit their words (sum < 2^447), so the
                // intermediate top words never overflow.
                "mul   {t1}, {a0}, {a1}",
                "umulh {h1}, {a0}, {a1}",
                "mul   {l2}, {a0}, {a2}",
                "umulh {h2}, {a0}, {a2}",
                "mul   {l3}, {a0}, {a3}",
                "umulh {h3}, {a0}, {a3}",
                "adds  {t2}, {h1}, {l2}",
                "adcs  {t3}, {h2}, {l3}",
                "adc   {t4}, {h3}, xzr",
                "mul   {l2}, {a1}, {a2}",
                "umulh {h2}, {a1}, {a2}",
                "mul   {l3}, {a1}, {a3}",
                "umulh {h3}, {a1}, {a3}",
                "adds  {t3}, {t3}, {l2}",
                "adcs  {t4}, {t4}, {h2}",
                "adc   {t5}, xzr, xzr",
                "adds  {t4}, {t4}, {l3}",
                "adcs  {t5}, {t5}, {h3}",
                "mul   {l3}, {a2}, {a3}",
                "umulh {h3}, {a2}, {a3}",
                "adds  {t5}, {t5}, {l3}",
                "adc   {t6}, {h3}, xzr",
                // double, top bit into t7
                "adds  {t1}, {t1}, {t1}",
                "adcs  {t2}, {t2}, {t2}",
                "adcs  {t3}, {t3}, {t3}",
                "adcs  {t4}, {t4}, {t4}",
                "adcs  {t5}, {t5}, {t5}",
                "adcs  {t6}, {t6}, {t6}",
                "adc   {t7}, xzr, xzr",
                // diagonal
                "mul   {t0}, {a0}, {a0}",
                "umulh {h0}, {a0}, {a0}",
                "mul   {l1}, {a1}, {a1}",
                "umulh {h1}, {a1}, {a1}",
                "mul   {l2}, {a2}, {a2}",
                "umulh {h2}, {a2}, {a2}",
                "mul   {l3}, {a3}, {a3}",
                "umulh {h3}, {a3}, {a3}",
                "adds  {t1}, {t1}, {h0}",
                "adcs  {t2}, {t2}, {l1}",
                "adcs  {t3}, {t3}, {h1}",
                "adcs  {t4}, {t4}, {l2}",
                "adcs  {t5}, {t5}, {h2}",
                "adcs  {t6}, {t6}, {l3}",
                "adc   {t7}, {t7}, {h3}",
                // fold (identical to mul_aarch64)
                "mov   {k}, #38",
                "mul   {l0}, {t4}, {k}",
                "umulh {h0}, {t4}, {k}",
                "mul   {l1}, {t5}, {k}",
                "umulh {h1}, {t5}, {k}",
                "mul   {l2}, {t6}, {k}",
                "umulh {h2}, {t6}, {k}",
                "mul   {l3}, {t7}, {k}",
                "umulh {h3}, {t7}, {k}",
                "adds  {t0}, {t0}, {l0}",
                "adcs  {t1}, {t1}, {l1}",
                "adcs  {t2}, {t2}, {l2}",
                "adcs  {t3}, {t3}, {l3}",
                "adc   {h3}, {h3}, xzr",
                "adds  {t1}, {t1}, {h0}",
                "adcs  {t2}, {t2}, {h1}",
                "adcs  {t3}, {t3}, {h2}",
                "adc   {h3}, {h3}, xzr",
                "mul   {h1}, {h3}, {k}",
                "adds  {t0}, {t0}, {h1}",
                "adcs  {t1}, {t1}, xzr",
                "adcs  {t2}, {t2}, xzr",
                "adcs  {t3}, {t3}, xzr",
                "csel  {h1}, {k}, xzr, cs",
                "add   {t0}, {t0}, {h1}",
                t0 = out(reg) t0, t1 = out(reg) t1, t2 = out(reg) t2, t3 = out(reg) t3,
                t4 = out(reg) _, t5 = out(reg) _, t6 = out(reg) _, t7 = out(reg) _,
                l0 = out(reg) _, l1 = out(reg) _, l2 = out(reg) _, l3 = out(reg) _,
                h0 = out(reg) _, h1 = out(reg) _, h2 = out(reg) _, h3 = out(reg) _,
                k = out(reg) _,
                a0 = in(reg) a[0], a1 = in(reg) a[1], a2 = in(reg) a[2], a3 = in(reg) a[3],
                options(pure, nomem, nostack),
            );
        }
        Fe([t0, t1, t2, t3])
    }

    /// `self^(2^k)`.
    pub fn pow2k(&self, k: u32) -> Fe {
        debug_assert!(k > 0);
        let mut r = self.sq();
        for _ in 1..k {
            r = r.sq();
        }
        r
    }

    /// Multiplicative inverse (`self^(p-2)`, dalek's addition chain);
    /// returns 0 for 0.
    pub fn invert(&self) -> Fe {
        let z = self;
        let t0 = z.sq(); // z^2
        let t1 = t0.pow2k(2); // z^8
        let t2 = z.mul(&t1); // z^9
        let t3 = t0.mul(&t2); // z^11
        let t4 = t3.sq(); // z^22
        let t5 = t2.mul(&t4); // z^(2^5 - 1)
        let t6 = t5.pow2k(5).mul(&t5); // 2^10 - 1
        let t7 = t6.pow2k(10).mul(&t6); // 2^20 - 1
        let t8 = t7.pow2k(20).mul(&t7); // 2^40 - 1
        let t9 = t8.pow2k(10).mul(&t6); // 2^50 - 1
        let t10 = t9.pow2k(50).mul(&t9); // 2^100 - 1
        let t11 = t10.pow2k(100).mul(&t10); // 2^200 - 1
        let t12 = t11.pow2k(50).mul(&t9); // 2^250 - 1
        t12.pow2k(5).mul(&t3) // 2^255 - 21 = p - 2
    }
}

/// Extended twisted Edwards coordinates (X:Y:Z:T), x = X/Z, y = Y/Z, xy = T/Z.
#[derive(Clone, Copy, Debug)]
pub struct Ext {
    pub x: Fe,
    pub y: Fe,
    pub z: Fe,
    pub t: Fe,
}

/// Affine "Niels" form of a point: (y+x, y-x, 2d*x*y).
#[derive(Clone, Copy, Debug)]
pub struct Niels {
    pub y_plus_x: Fe,
    pub y_minus_x: Fe,
    pub t2d: Fe,
}

impl Ext {
    pub const IDENTITY: Ext = Ext {
        x: FE_ZERO,
        y: FE_ONE,
        z: FE_ONE,
        t: FE_ZERO,
    };

    /// Mixed addition `self + q` (q affine Niels): dalek's complete
    /// `&EdwardsPoint + &AffineNielsPoint` formula, 7 multiplications, no
    /// exceptional inputs.
    #[inline(always)]
    pub fn add_niels(&self, q: &Niels) -> Ext {
        let y_plus_x = self.y.add(&self.x);
        let y_minus_x = self.y.sub(&self.x);
        let pp = y_plus_x.mul(&q.y_plus_x);
        let mm = y_minus_x.mul(&q.y_minus_x);
        let tt2d = self.t.mul(&q.t2d);
        let zz2 = self.z.add(&self.z);
        let e = pp.sub(&mm);
        let f = zz2.sub(&tt2d);
        let g = zz2.add(&tt2d);
        let h = pp.add(&mm);
        Ext {
            x: e.mul(&f),
            y: g.mul(&h),
            z: f.mul(&g),
            t: e.mul(&h),
        }
    }

    /// Point doubling (dbl-2008-hwcd, a = -1; dalek's completed-point
    /// doubling followed by the completed -> extended conversion).
    pub fn double(&self) -> Ext {
        let xx = self.x.sq();
        let yy = self.y.sq();
        let zz2 = self.z.sq().add(&self.z.sq());
        let x_plus_y_sq = self.x.add(&self.y).sq();
        let yy_plus_xx = yy.add(&xx);
        let yy_minus_xx = yy.sub(&xx);
        // E = (X+Y)^2 - (YY+XX), G = YY - XX, F = 2ZZ - G, H = YY + XX.
        let e = x_plus_y_sq.sub(&yy_plus_xx);
        let g = yy_minus_xx;
        let f = zz2.sub(&g);
        let h = yy_plus_xx;
        Ext {
            x: e.mul(&f),
            y: g.mul(&h),
            z: f.mul(&g),
            t: e.mul(&h),
        }
    }

    /// Affine (x, y) via one inversion.
    pub fn to_affine(&self) -> (Fe, Fe) {
        let zinv = self.z.invert();
        (self.x.mul(&zinv), self.y.mul(&zinv))
    }

    pub fn to_niels(&self) -> Niels {
        let (x, y) = self.to_affine();
        Niels {
            y_plus_x: y.add(&x),
            y_minus_x: y.sub(&x),
            t2d: x.mul(&y).mul(&EDWARDS_D2),
        }
    }

    /// Compressed Ed25519 encoding (y with the sign of x in bit 255).
    #[cfg(test)]
    pub fn compress(&self) -> [u8; 32] {
        let (x, y) = self.to_affine();
        let mut out = y.to_bytes();
        out[31] |= (x.to_bytes()[0] & 1) << 7;
        out
    }
}

pub fn basepoint() -> Ext {
    Ext {
        x: BASE_X,
        y: BASE_Y,
        z: FE_ONE,
        t: BASE_X.mul(&BASE_Y),
    }
}

/// Variable-time `scalar * B` (little-endian scalar bytes, all 256 bits
/// honoured; the callers pass clamped scalars). Only used once per walk
/// (re)start, so a plain double-and-add is fast enough.
pub fn mul_base(scalar: &[u8; 32]) -> Ext {
    let base = basepoint().to_niels();
    let mut acc = Ext::IDENTITY;
    for bit in (0..256).rev() {
        acc = acc.double();
        if (scalar[bit >> 3] >> (bit & 7)) & 1 == 1 {
            acc = acc.add_niels(&base);
        }
    }
    acc
}

/// Candidates per lane produced by one `next_window` call. Also the number
/// of independent y-only chains per lane: chain `k` visits the candidates
/// `k, k + WINDOW, k + 2*WINDOW, ...`, so one window is one step of every
/// chain and the per-step inversions of all `L * WINDOW` chains are batched
/// into a single field inversion. 256 keeps the working set (two windows of
/// y plus two scratch arrays, 32 KiB per lane) in L1 while amortising the
/// inversion to about half a squaring per candidate.
pub const WINDOW: usize = 256;

/// Interleaved running-product chains per lane in the batched inversion,
/// so the serial `acc *= z` dependency runs at multiplier throughput rather
/// than latency.
const INV_CHAINS: usize = 8;

/// Chains per lane re-derived with dalek by one `matches_dalek` call; the
/// cursor rotates so every chain is checked in turn.
const CHECK_CHAINS: usize = 2;

/// Little-endian `scalar + delta` for a signed `delta` (|delta| far below
/// 2^254, so a clamped scalar never borrows past bit 255).
pub(crate) fn scalar_offset(scalar: [u8; 32], delta: i64) -> [u8; 32] {
    if delta >= 0 {
        return scalar_add_u64(scalar, delta as u64);
    }
    let mut out = scalar;
    let mut borrow = delta.unsigned_abs() as u128;
    for byte in out.iter_mut() {
        if borrow == 0 {
            break;
        }
        let diff = *byte as i128 - (borrow & 0xFF) as i128;
        *byte = diff as u8;
        borrow = (borrow >> 8) + (diff < 0) as u128;
    }
    debug_assert_eq!(borrow, 0, "scalar_offset underflow");
    out
}

/// dalek's public key for any 256-bit scalar (reduced mod the group order,
/// which does not move the point) with the sign bit of x cleared: exactly
/// what the walk emits for that scalar.
fn dalek_y(scalar: &[u8; 32]) -> [u8; 32] {
    use curve25519_dalek::edwards::EdwardsPoint;
    use curve25519_dalek::scalar::Scalar;
    let mut y = EdwardsPoint::mul_base(&Scalar::from_bytes_mod_order(*scalar))
        .compress()
        .to_bytes();
    y[31] &= 0x7F;
    y
}

/// Incremental walk over the clamped scalars `base + 8*i`, i = 0, 1, 2, ...
/// of `L` independent lanes, emitting the canonical `y` coordinate of every
/// candidate.
///
/// Each lane is split into `WINDOW` chains; chain `k` holds the affine `y`
/// of its current point `P = (base + 8*(i*WINDOW + k)) * B` and of the
/// previous one `P - D`, where `D = 8*WINDOW*B` is the common step. A
/// window advances every chain once with the y-only differential addition
///
/// ```text
/// alpha = (y_P - y_D)^2,  beta = (y_P + y_D)^2
/// A = (1 + y_T) * alpha,  B = (1 - y_T) * beta        (y_T = y(P - D))
/// y(P + D) = (B - A) / (B + A)
/// ```
///
/// which follows from Montgomery's `x(P+Q) x(P-Q) (x_P - x_Q)^2 = (x_P x_Q
/// - 1)^2` on the birationally equivalent curve25519 (`u = (1+y)/(1-y)`),
/// and is valid whenever none of `P`, `D`, `P +- D` is the identity or has
/// small order and `P != +-D`; every walk point is an odd multiple-free
/// `s * B` with `s` in `(2^254 - 2^16, 2^255)` and `s = 0 mod 8`, which is
/// never `0` or `+-8*WINDOW` mod the group order, so no walk point ever
/// hits an exceptional case. The step costs 2 multiplications and 2
/// squarings plus one inversion, batched over all `L * WINDOW` chains (3
/// multiplications each plus the final `(B - A) * (B + A)^-1`): about 6
/// multiplications and 2 squarings per candidate, against 11
/// multiplications for a projective Edwards step plus normalisation.
///
/// Every candidate the walk emits is `candidate_from_scalar(scalar_at(lane,
/// i))` with the sign bit of x left clear; callers re-derive the exact key
/// with dalek for any hit.
pub struct Walk<const L: usize> {
    base: [[u8; 32]; L],
    /// Index of the first candidate of the next window (same for all lanes).
    next_iter: u64,
    /// Affine y of chain `k` of lane `l` for the last two windows:
    /// `ys[flip][l][k]` is candidate `next_iter - WINDOW + k`, the other
    /// array holds candidate `next_iter - 2*WINDOW + k` (both before the
    /// first candidate right after a restart).
    ys: [Box<[[Fe; WINDOW]; L]>; 2],
    flip: usize,
    /// Scratch: the denominators `B + A` and their running products.
    den: Box<[[Fe; WINDOW]; L]>,
    prods: Box<[[Fe; WINDOW]; L]>,
    /// Affine y of the chain step `D = 8*WINDOW*B`.
    step_y: Fe,
    /// First chain re-derived by the next `matches_dalek`.
    check_cursor: Cell<usize>,
}

impl<const L: usize> Walk<L> {
    /// Candidates produced by one `next_window` call (all lanes).
    pub const CANDIDATES_PER_WINDOW: u64 = (L * WINDOW) as u64;

    /// Start from fresh CSPRNG-chosen clamped base scalars
    /// (`random_base_scalar` semantics, so `base + 8*i` stays clamped for
    /// every `i` below 2^250). `None` if the computed starting points do not
    /// match dalek's derivation, which would indicate an arithmetic bug; the
    /// caller then falls back to per-candidate derivation.
    pub fn new(rng: &mut impl RngCore) -> Option<Self> {
        Self::from_bases(std::array::from_fn(|_| random_base_scalar(rng)))
    }

    /// Start from explicit bases (each must satisfy the `random_base_scalar`
    /// invariants); `None` on a dalek mismatch as for [`Walk::new`].
    pub fn from_bases(base: [[u8; 32]; L]) -> Option<Self> {
        let step_y = {
            let mut s = [0u8; 32];
            s[..8].copy_from_slice(&(8 * WINDOW as u64).to_le_bytes());
            mul_base(&s).to_affine().1
        };
        let mut walk = Walk {
            base,
            next_iter: 0,
            ys: [Box::new([[FE_ZERO; WINDOW]; L]), Box::new([[FE_ZERO; WINDOW]; L])],
            flip: 0,
            den: Box::new([[FE_ZERO; WINDOW]; L]),
            prods: Box::new([[FE_ZERO; WINDOW]; L]),
            step_y,
            check_cursor: Cell::new(0),
        };
        walk.restart_from(base).then_some(walk)
    }

    /// Restart from fresh CSPRNG-chosen bases. Returns whether the new
    /// starting points match dalek (false: stop using the walk).
    pub fn restart(&mut self, rng: &mut impl RngCore) -> bool {
        self.restart_from(std::array::from_fn(|_| random_base_scalar(rng)))
    }

    /// Restart from explicit bases; returns whether the starting points
    /// (every chain's two seed points, all `2 * L * WINDOW` of them) match
    /// dalek exactly.
    pub fn restart_from(&mut self, base: [[u8; 32]; L]) -> bool {
        self.base = base;
        self.next_iter = 0;
        self.flip = 0;
        self.check_cursor.set(0);

        // Seed: the 2*WINDOW points (base - 16*WINDOW + 8*j) * B, j = 0..,
        // i.e. the candidates of the two windows before the first one, by
        // one scalar multiplication and mixed Edwards additions per lane,
        // normalised by one batched inversion.
        let step8 = basepoint().double().double().double().to_niels();
        let n = 2 * WINDOW;
        let mut ys = vec![FE_ZERO; L * n];
        let mut zs = vec![FE_ZERO; L * n];
        for l in 0..L {
            let mut p = mul_base(&scalar_offset(base[l], -(16 * WINDOW as i64)));
            for j in 0..n {
                ys[l * n + j] = p.y;
                zs[l * n + j] = p.z;
                p = p.add_niels(&step8);
            }
        }
        let mut prods = vec![FE_ZERO; L * n];
        let mut acc = FE_ONE;
        for (prod, z) in prods.iter_mut().zip(zs.iter()) {
            acc = acc.mul(z);
            *prod = acc;
        }
        let mut inv = acc.invert();
        for i in (0..L * n).rev() {
            let zinv = if i == 0 { inv } else { inv.mul(&prods[i - 1]) };
            inv = inv.mul(&zs[i]);
            let y = ys[i].mul(&zinv);
            let (l, j) = (i / n, i % n);
            if j < WINDOW {
                self.ys[1][l][j] = y;
            } else {
                self.ys[0][l][j - WINDOW] = y;
            }
        }

        // Verify every seed point against dalek: one exact fixed-base
        // multiplication per lane, then dalek's own point additions.
        use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
        use curve25519_dalek::edwards::EdwardsPoint;
        use curve25519_dalek::scalar::Scalar;
        let d8 = ED25519_BASEPOINT_POINT.mul_by_cofactor();
        (0..L).all(|l| {
            let s = scalar_offset(base[l], -(16 * WINDOW as i64));
            let mut p = EdwardsPoint::mul_base(&Scalar::from_bytes_mod_order(s));
            (0..n).all(|j| {
                let mut expected = p.compress().to_bytes();
                expected[31] &= 0x7F;
                p += d8;
                let mine = if j < WINDOW { self.ys[1][l][j] } else { self.ys[0][l][j - WINDOW] };
                mine.to_bytes() == expected
            })
        })
    }

    /// The clamped private scalar of candidate `iter` of `lane`.
    pub fn scalar_at(&self, lane: usize, iter: u64) -> [u8; 32] {
        scalar_add_u64(self.base[lane], 8 * iter)
    }

    /// Scalar of `lane`'s first candidate of the next window; a slow
    /// fallback path resumes from here.
    pub fn next_scalar(&self, lane: usize) -> [u8; 32] {
        self.scalar_at(lane, self.next_iter)
    }

    /// Candidates visited per lane since the last (re)start.
    pub fn iters_done(&self) -> u64 {
        self.next_iter
    }

    /// Scalar of the point chain `k` of `lane` currently holds (its
    /// candidate of the last emitted window).
    fn chain_scalar(&self, lane: usize, k: usize) -> [u8; 32] {
        scalar_offset(
            self.base[lane],
            8 * (self.next_iter as i64 - WINDOW as i64 + k as i64),
        )
    }

    /// True iff the current points of the next `CHECK_CHAINS` chains of
    /// every lane (a rotating cursor covers all `WINDOW` chains in turn)
    /// agree with dalek's exact derivation of their scalars. A systematic
    /// arithmetic fault corrupts every chain and fails the first call; the
    /// seed state of every chain was verified in full at the (re)start.
    pub fn matches_dalek(&self) -> bool {
        let cursor = self.check_cursor.get();
        self.check_cursor.set((cursor + CHECK_CHAINS) % WINDOW);
        (0..L).all(|lane| {
            (0..CHECK_CHAINS).all(|i| {
                let k = (cursor + i) % WINDOW;
                self.ys[self.flip][lane][k].to_bytes() == dalek_y(&self.chain_scalar(lane, k))
            })
        })
    }

    /// Test hook: silently corrupt every chain of `lane` (as an arithmetic
    /// bug would) so callers' periodic `matches_dalek` checks and fallback
    /// paths can be exercised end to end.
    #[cfg(test)]
    pub fn desynchronise(&mut self, lane: usize) {
        for y in self.ys[self.flip][lane].iter_mut() {
            *y = y.add(&FE_ONE);
        }
    }

    /// Advance every lane by one window of `WINDOW` candidates and hand
    /// their canonical y encodings (bit 255 clear) to
    /// `visit(lane, iter, y_bytes)`; the visiting order within the window is
    /// unspecified. Returns the iteration index of the window's first
    /// candidate.
    #[inline]
    pub fn next_window(&mut self, mut visit: impl FnMut(usize, u64, &[u8; 32])) -> u64 {
        let first = self.next_iter;
        let [ys0, ys1] = &mut self.ys;
        let (cur, prev): (&[[Fe; WINDOW]; L], &mut [[Fe; WINDOW]; L]) = if self.flip == 0 {
            (ys0, ys1)
        } else {
            (ys1, ys0)
        };
        let den = &mut *self.den;
        let prods = &mut *self.prods;
        let yd = self.step_y;

        // Numerators (into `prev`, whose points are no longer needed) and
        // denominators of every chain's next y; no dependency between
        // chains, so the CPU overlaps them freely.
        for l in 0..L {
            for k in (0..WINDOW).step_by(2) {
                // Two chains interleaved stage by stage in program order:
                // the assembly field operations are opaque to the compiler's
                // scheduler, so independent work is placed adjacently by
                // hand (two chains measured best; four spill registers).
                let (yp0, yp1) = (cur[l][k], cur[l][k + 1]);
                let (yt0, yt1) = (prev[l][k], prev[l][k + 1]);
                let am0 = yp0.sub(&yd);
                let am1 = yp1.sub(&yd);
                let bp0 = yp0.add(&yd);
                let bp1 = yp1.add(&yd);
                let alpha0 = am0.sq();
                let alpha1 = am1.sq();
                let beta0 = bp0.sq();
                let beta1 = bp1.sq();
                let op0 = FE_ONE.add(&yt0);
                let op1 = FE_ONE.add(&yt1);
                let om0 = FE_ONE.sub(&yt0);
                let om1 = FE_ONE.sub(&yt1);
                let a0 = op0.mul(&alpha0);
                let a1 = op1.mul(&alpha1);
                let b0 = om0.mul(&beta0);
                let b1 = om1.mul(&beta1);
                den[l][k] = b0.add(&a0);
                den[l][k + 1] = b1.add(&a1);
                prev[l][k] = b0.sub(&a0);
                prev[l][k + 1] = b1.sub(&a1);
            }
        }

        // Montgomery batch inversion: INV_CHAINS interleaved running
        // products per lane (chain g covers k = g mod INV_CHAINS), the
        // chains' totals joined by one more short chain so a single field
        // inversion serves them all.
        for l in 0..L {
            prods[l][..INV_CHAINS].copy_from_slice(&den[l][..INV_CHAINS]);
            for k in INV_CHAINS..WINDOW {
                prods[l][k] = prods[l][k - INV_CHAINS].mul(&den[l][k]);
            }
        }
        let mut totals = [[FE_ONE; INV_CHAINS]; L];
        let mut joined = [[FE_ONE; INV_CHAINS]; L];
        let mut running = FE_ONE;
        for l in 0..L {
            for g in 0..INV_CHAINS {
                totals[l][g] = prods[l][WINDOW - INV_CHAINS + g];
                running = if l == 0 && g == 0 { totals[l][g] } else { running.mul(&totals[l][g]) };
                joined[l][g] = running;
            }
        }
        let mut inv_all = running.invert();
        let mut inv = [[FE_ONE; INV_CHAINS]; L];
        for l in (0..L).rev() {
            for g in (0..INV_CHAINS).rev() {
                if l == 0 && g == 0 {
                    inv[l][g] = inv_all;
                } else {
                    let (pl, pg) = if g == 0 { (l - 1, INV_CHAINS - 1) } else { (l, g - 1) };
                    inv[l][g] = inv_all.mul(&joined[pl][pg]);
                    inv_all = inv_all.mul(&totals[l][g]);
                }
            }
        }

        // Peel one inverse per chain off its running inverse, finish the
        // chain's new y and emit it.
        for l in 0..L {
            for k in (INV_CHAINS..WINDOW).rev() {
                let g = k % INV_CHAINS;
                let dinv = inv[l][g].mul(&prods[l][k - INV_CHAINS]);
                inv[l][g] = inv[l][g].mul(&den[l][k]);
                let y = prev[l][k].mul(&dinv);
                prev[l][k] = y;
                visit(l, first + k as u64, &y.to_bytes());
            }
            for k in (0..INV_CHAINS).rev() {
                let y = prev[l][k].mul(&inv[l][k]);
                prev[l][k] = y;
                visit(l, first + k as u64, &y.to_bytes());
            }
        }

        self.flip ^= 1;
        self.next_iter += WINDOW as u64;
        first
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keygen::candidate_from_scalar;
    use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
    use curve25519_dalek::edwards::EdwardsPoint;
    use curve25519_dalek::scalar::Scalar;
    use curve25519_dalek::traits::Identity;
    use num_bigint::BigUint;
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn p() -> BigUint {
        (BigUint::one() << 255u32) - BigUint::from(19u32)
    }

    fn to_big(bytes: &[u8; 32]) -> BigUint {
        BigUint::from_bytes_le(bytes)
    }

    fn from_big(v: &BigUint) -> [u8; 32] {
        let mut out = [0u8; 32];
        let b = v.to_bytes_le();
        out[..b.len()].copy_from_slice(&b);
        out
    }

    fn fe_big(v: &BigUint) -> Fe {
        Fe::from_bytes(&from_big(v))
    }

    /// Public key of walk candidate `iter` from `base`, computed with dalek,
    /// with the sign bit cleared (what the walk emits).
    fn dalek_y(base: [u8; 32], iter: u64) -> [u8; 32] {
        let mut y = candidate_from_scalar(scalar_add_u64(base, 8 * iter), [0u8; 32]).public_key;
        y[31] &= 0x7F;
        y
    }

    fn edge_values() -> Vec<BigUint> {
        let p = p();
        let two = BigUint::from(2u32);
        let mut v = vec![
            BigUint::zero(),
            BigUint::one(),
            two.clone(),
            BigUint::from(19u32),
            BigUint::from(37u32),
            BigUint::from(38u32),
            BigUint::from(121_665u32),
            BigUint::from(121_666u32),
            &p - BigUint::one(),
            &p - &two,
            &p - BigUint::from(19u32),
            p.clone(),
            &p + BigUint::one(),
            &p * &two,
            &p * &two - BigUint::one(),
            &p * &two + BigUint::one(),
            (BigUint::one() << 51u32) - BigUint::one(),
            BigUint::one() << 51u32,
            (BigUint::one() << 64u32) - BigUint::one(),
            BigUint::one() << 64u32,
            (BigUint::one() << 204u32) - BigUint::one(),
            BigUint::one() << 204u32,
            (BigUint::one() << 254u32) + BigUint::from(0x1234u32),
            (BigUint::one() << 255u32) - BigUint::one(),
            BigUint::one() << 255u32,
            (BigUint::one() << 255u32) + BigUint::from(18u32),
            (BigUint::one() << 255u32) + BigUint::from(19u32),
            (BigUint::one() << 255u32) + BigUint::from(20u32),
            (BigUint::one() << 256u32) - BigUint::one(),
            (BigUint::one() << 256u32) - BigUint::from(38u32),
            (BigUint::one() << 256u32) - BigUint::from(39u32),
            (BigUint::one() << 256u32) - BigUint::from(76u32),
            (BigUint::one() << 256u32) - BigUint::from(77u32),
            (BigUint::one() << 256u32) - BigUint::from(1444u32),
            (BigUint::one() << 256u32) - BigUint::from(1445u32),
        ];
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        for i in 0..400 {
            let mut b = [0u8; 32];
            rng.fill_bytes(&mut b);
            if i % 3 == 0 {
                b[31] &= 0x7F;
            }
            if i % 5 == 0 {
                // Near-maximal words to stress carries.
                for byte in b.iter_mut().take(24) {
                    *byte = 0xFF;
                }
            }
            if i % 7 == 0 {
                // Values just below a multiple of 2^64.
                for byte in b.iter_mut().skip(8).take(8) {
                    *byte = 0xFF;
                }
            }
            v.push(to_big(&b));
        }
        v
    }

    #[test]
    fn field_roundtrip_and_canonical_encoding() {
        for v in edge_values() {
            let fe = fe_big(&v);
            let bytes = fe.to_bytes();
            assert_eq!(to_big(&bytes), &v % p());
            assert_eq!(bytes[31] >> 7, 0);
            // Canonical bytes decode back to the same element.
            assert_eq!(Fe::from_bytes(&bytes).to_bytes(), bytes);
        }
        // Non-canonical input in [p, 2^255): must encode as v - p.
        let p = p();
        for k in 0u32..40 {
            let v = &p + BigUint::from(k);
            assert_eq!(to_big(&fe_big(&v).to_bytes()), BigUint::from(k));
        }
    }

    #[test]
    fn field_ops_match_bigint() {
        let p = p();
        let vals = edge_values();
        for (i, a) in vals.iter().enumerate() {
            let fa = fe_big(a);
            assert_eq!(to_big(&fa.sq().to_bytes()), (a * a) % &p, "sq {i}");
            assert_eq!(to_big(&fa.neg().to_bytes()), (&p - a % &p) % &p, "neg {i}");
            assert_eq!(to_big(&fa.pow2k(3).to_bytes()), a.modpow(&BigUint::from(8u32), &p), "pow2k {i}");
            let inv = fa.invert();
            if a % &p != BigUint::zero() {
                assert_eq!(
                    to_big(&inv.to_bytes()),
                    a.modpow(&(&p - BigUint::from(2u32)), &p),
                    "inv {i}"
                );
                assert_eq!(to_big(&fa.mul(&inv).to_bytes()), BigUint::one(), "inv {i}");
            } else {
                assert_eq!(to_big(&inv.to_bytes()), BigUint::zero());
            }
            for (j, b) in vals.iter().enumerate().step_by(5) {
                let fb = fe_big(b);
                assert_eq!(to_big(&fa.mul(&fb).to_bytes()), (a * b) % &p, "mul {i} {j}");
                assert_eq!(to_big(&fb.mul(&fa).to_bytes()), (a * b) % &p, "mul {j} {i}");
                assert_eq!(to_big(&fa.add(&fb).to_bytes()), (a + b) % &p, "add {i} {j}");
                assert_eq!(
                    to_big(&fa.sub(&fb).to_bytes()),
                    (a + &p - b % &p) % &p,
                    "sub {i} {j}"
                );
                // Every output is a valid input: chained additions (values
                // up to 2^256) and subtractions feed mul correctly.
                let chain = fa.add(&fb).add(&fa).add(&fb).sub(&fa);
                let expect = ((a + b) * BigUint::from(2u32) + &p - a % &p) % &p;
                assert_eq!(
                    to_big(&chain.mul(&chain).to_bytes()),
                    (&expect * &expect) % &p,
                    "chain {i} {j}"
                );
            }
        }
    }

    /// The aarch64 assembly `mul`/`sq` are bit-exact with the portable
    /// implementations (not just congruent: the same unreduced words) on the
    /// edge values, carry-stressing patterns and long random chains whose
    /// outputs are fed back in.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_field_ops_are_bit_exact_with_portable() {
        let vals: Vec<Fe> = edge_values().iter().map(fe_big).collect();
        for a in &vals {
            assert_eq!(a.sq_aarch64().0, a.sq_portable().0, "sq {a:?}");
            for b in &vals {
                assert_eq!(a.mul_aarch64(b).0, a.mul_portable(b).0, "mul {a:?} {b:?}");
                assert_eq!(a.add_aarch64(b).0, a.add_portable(b).0, "add {a:?} {b:?}");
                assert_eq!(a.sub_aarch64(b).0, a.sub_portable(b).0, "sub {a:?} {b:?}");
            }
        }
        let mut rng = ChaCha20Rng::from_seed([64u8; 32]);
        let word = |rng: &mut ChaCha20Rng| -> u64 {
            match rng.next_u32() % 8 {
                0 => u64::MAX,
                1 => 0,
                2 => u64::MAX - (rng.next_u64() % 4),
                3 => rng.next_u64() % 4,
                4 => 1u64 << 63,
                _ => rng.next_u64(),
            }
        };
        let mut x = vals[vals.len() - 1];
        let mut y = x;
        for i in 0..2_000_000 {
            let a = Fe([word(&mut rng), word(&mut rng), word(&mut rng), word(&mut rng)]);
            let b = Fe([word(&mut rng), word(&mut rng), word(&mut rng), word(&mut rng)]);
            assert_eq!(a.mul_aarch64(&b).0, a.mul_portable(&b).0, "mul {i}");
            assert_eq!(a.sq_aarch64().0, a.sq_portable().0, "sq {i}");
            assert_eq!(a.add_aarch64(&b).0, a.add_portable(&b).0, "add {i}");
            assert_eq!(a.sub_aarch64(&b).0, a.sub_portable(&b).0, "sub {i}");
            assert_eq!(b.sub_aarch64(&a).0, b.sub_portable(&a).0, "sub' {i}");
            // Chains: every output representation is fed back in.
            x = x.mul_aarch64(&a).sq_aarch64().add_aarch64(&b).sub_aarch64(&a);
            y = y.mul_portable(&a).sq_portable().add_portable(&b).sub_portable(&a);
            assert_eq!(x.0, y.0, "chain {i}");
        }
    }

    /// The 32-bit-limb `mul_32`/`sq_32` (wasm32's field ops) are bit-exact
    /// with the portable ones on the edge values, carry-stressing words and
    /// random chains whose outputs are fed back in. Runs on every target.
    #[test]
    fn limb32_field_ops_are_bit_exact_with_portable() {
        let vals: Vec<Fe> = edge_values().iter().map(fe_big).collect();
        for a in &vals {
            assert_eq!(a.sq_32().0, a.sq_portable().0, "sq {a:?}");
            for b in &vals {
                assert_eq!(a.mul_32(b).0, a.mul_portable(b).0, "mul {a:?} {b:?}");
            }
        }
        let mut rng = ChaCha20Rng::from_seed([32u8; 32]);
        let word = |rng: &mut ChaCha20Rng| -> u64 {
            match rng.next_u32() % 8 {
                0 => u64::MAX,
                1 => 0,
                2 => u64::MAX - (rng.next_u64() % 4),
                3 => rng.next_u64() % 4,
                4 => 1u64 << 63,
                5 => 0xFFFF_FFFF,
                _ => rng.next_u64(),
            }
        };
        let (mut x, mut y) = (vals[vals.len() - 1], vals[vals.len() - 1]);
        for i in 0..500_000 {
            let a = Fe([word(&mut rng), word(&mut rng), word(&mut rng), word(&mut rng)]);
            let b = Fe([word(&mut rng), word(&mut rng), word(&mut rng), word(&mut rng)]);
            assert_eq!(a.mul_32(&b).0, a.mul_portable(&b).0, "mul {i}");
            assert_eq!(a.sq_32().0, a.sq_portable().0, "sq {i}");
            x = x.mul_32(&a).sq_32().add_portable(&b);
            y = y.mul_portable(&a).sq_portable().add_portable(&b);
            assert_eq!(x.0, y.0, "chain {i}");
        }
    }

    #[test]
    fn basepoint_and_step_match_dalek() {
        let b = basepoint();
        assert_eq!(b.compress(), ED25519_BASEPOINT_POINT.compress().to_bytes());
        let two_b = b.double();
        assert_eq!(
            two_b.compress(),
            (ED25519_BASEPOINT_POINT + ED25519_BASEPOINT_POINT)
                .compress()
                .to_bytes()
        );
        let three_b = two_b.add_niels(&b.to_niels());
        assert_eq!(
            three_b.compress(),
            (ED25519_BASEPOINT_POINT + ED25519_BASEPOINT_POINT + ED25519_BASEPOINT_POINT)
                .compress()
                .to_bytes()
        );
        // The walk step 8B, both as repeated doubling and as mul_base(8).
        let eight_b = ED25519_BASEPOINT_POINT.mul_by_cofactor().compress().to_bytes();
        assert_eq!(b.double().double().double().compress(), eight_b);
        let mut eight = [0u8; 32];
        eight[0] = 8;
        assert_eq!(mul_base(&eight).compress(), eight_b);
        // The identity is a valid (complete-formula) addend and compresses as dalek's.
        assert_eq!(Ext::IDENTITY.add_niels(&b.to_niels()).compress(), b.compress());
        assert_eq!(
            Ext::IDENTITY.compress(),
            EdwardsPoint::identity().compress().to_bytes()
        );
    }

    #[test]
    fn mul_base_matches_dalek() {
        let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
        for _ in 0..64 {
            let s = random_base_scalar(&mut rng);
            let expected = EdwardsPoint::mul_base_clamped(s).compress().to_bytes();
            assert_eq!(mul_base(&s).compress(), expected);
        }
        // Extremes of the clamp range.
        let mut lo = [0u8; 32];
        lo[31] = 0x40;
        assert_eq!(
            mul_base(&lo).compress(),
            EdwardsPoint::mul_base_clamped(lo).compress().to_bytes()
        );
        let mut hi = [0xFFu8; 32];
        hi[0] = 0xF8;
        hi[31] = 0x7F;
        assert_eq!(
            mul_base(&hi).compress(),
            EdwardsPoint::mul_base_clamped(hi).compress().to_bytes()
        );
        // Unclamped scalars (every bit pattern honoured, including bit 255);
        // dalek reduces mod the group order, which does not move the point.
        let mut scalars = vec![[0u8; 32], [0xFFu8; 32]];
        let mut one = [0u8; 32];
        one[0] = 1;
        scalars.push(one);
        for _ in 0..16 {
            let mut s = [0u8; 32];
            rng.fill_bytes(&mut s);
            scalars.push(s);
        }
        for s in scalars {
            let mine = mul_base(&s);
            let expected = EdwardsPoint::mul_base(&Scalar::from_bytes_mod_order(s))
                .compress()
                .to_bytes();
            assert_eq!(mine.compress(), expected, "scalar {}", hex::encode(s));
            let expected2 = EdwardsPoint::mul_base(
                &(Scalar::from_bytes_mod_order(s) * Scalar::from(2u64)),
            )
            .compress()
            .to_bytes();
            assert_eq!(mine.double().compress(), expected2);
        }
    }

    #[test]
    fn walk_windows_match_dalek_candidates() {
        let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
        let mut walk = Walk::<2>::new(&mut rng).expect("walk start must match dalek");
        let bases = [walk.scalar_at(0, 0), walk.scalar_at(1, 0)];
        assert_ne!(bases[0], bases[1]);
        let mut seen = Vec::new();
        for window in 0..3u64 {
            let first = walk.next_window(|lane, iter, y| seen.push((lane, iter, *y)));
            assert_eq!(first, window * WINDOW as u64);
            assert_eq!(walk.iters_done(), (window + 1) * WINDOW as u64);
            assert!(walk.matches_dalek());
        }
        assert_eq!(seen.len() as u64, 3 * Walk::<2>::CANDIDATES_PER_WINDOW);
        let mut covered = [vec![false; 3 * WINDOW], vec![false; 3 * WINDOW]];
        for (lane, iter, y) in seen {
            assert_eq!(walk.scalar_at(lane, iter), scalar_add_u64(bases[lane], 8 * iter));
            assert_eq!(y, dalek_y(bases[lane], iter), "lane {lane} iter {iter}");
            assert!(!covered[lane][iter as usize]);
            covered[lane][iter as usize] = true;
        }
        assert!(covered.iter().all(|c| c.iter().all(|v| *v)));
        for lane in 0..2 {
            assert_eq!(walk.next_scalar(lane), scalar_add_u64(bases[lane], 8 * 3 * WINDOW as u64));
        }

        // Restart produces different, still-consistent walks.
        assert!(walk.restart(&mut rng));
        let bases2 = [walk.scalar_at(0, 0), walk.scalar_at(1, 0)];
        assert_ne!(bases2[0], bases[0]);
        assert_eq!(walk.iters_done(), 0);
        walk.next_window(|lane, iter, y| assert_eq!(*y, dalek_y(bases2[lane], iter)));

        // Extreme bases: lowest and highest clamped scalars with bit 253
        // clear, walked far enough to carry across several scalar bytes.
        let mut lo = [0u8; 32];
        lo[31] = 0x40;
        let mut hi = [0xFFu8; 32];
        hi[0] = 0xF8;
        hi[31] = 0x5F;
        assert!(walk.restart_from([lo, hi]));
        for _ in 0..4 {
            walk.next_window(|lane, iter, y| {
                let scalar = scalar_add_u64([lo, hi][lane], 8 * iter);
                assert_eq!(scalar[0] & 7, 0);
                assert_eq!(scalar[31] & 0xC0, 0x40);
                assert_eq!(*y, dalek_y([lo, hi][lane], iter));
            });
        }
        assert!(walk.matches_dalek());
    }

    #[test]
    fn walk_crosses_scalar_byte_carries() {
        // Start scalars whose low bytes are 0xF8/0xFF so that +8 steps carry
        // through several bytes inside the first window, in both lanes.
        let mut a = [0u8; 32];
        a[0] = 0xF8;
        a[1] = 0xFF;
        a[2] = 0xFF;
        a[3] = 0x12;
        a[31] = 0x40;
        let mut b = [0xFFu8; 32];
        b[0] = 0xF8;
        b[4] = 0x00;
        b[31] = 0x5F;
        let mut walk = Walk::<2>::from_bases([a, b]).unwrap();
        walk.next_window(|lane, iter, y| {
            assert_eq!(*y, dalek_y([a, b][lane], iter), "lane {lane} iter {iter}");
        });
        assert!(walk.matches_dalek());
    }

    #[test]
    fn walk_with_one_and_four_lanes() {
        let mut rng = ChaCha20Rng::from_seed([8u8; 32]);
        let mut one = Walk::<1>::new(&mut rng).unwrap();
        let base = one.scalar_at(0, 0);
        let mut count = 0u64;
        one.next_window(|lane, iter, y| {
            assert_eq!(lane, 0);
            assert_eq!(*y, dalek_y(base, iter));
            count += 1;
        });
        assert_eq!(count, Walk::<1>::CANDIDATES_PER_WINDOW);

        let mut four = Walk::<4>::new(&mut rng).unwrap();
        let bases: Vec<[u8; 32]> = (0..4).map(|l| four.scalar_at(l, 0)).collect();
        let mut count = 0u64;
        four.next_window(|lane, iter, y| {
            assert_eq!(*y, dalek_y(bases[lane], iter));
            count += 1;
        });
        assert_eq!(count, Walk::<4>::CANDIDATES_PER_WINDOW);
        assert!(four.matches_dalek());
    }

    #[test]
    fn matches_dalek_detects_a_desynchronised_point() {
        let mut rng = ChaCha20Rng::from_seed([9u8; 32]);
        let mut walk = Walk::<2>::new(&mut rng).unwrap();
        walk.next_window(|_, _, _| {});
        assert!(walk.matches_dalek());
        walk.desynchronise(1);
        assert!(!walk.matches_dalek());
        // A restart heals it.
        assert!(walk.restart(&mut rng));
        assert!(walk.matches_dalek());

        // A single corrupted chain is caught once the rotating cursor
        // reaches it, within WINDOW / CHECK_CHAINS calls.
        let k = WINDOW - 3;
        walk.ys[walk.flip][0][k] = walk.ys[walk.flip][0][k].add(&FE_ONE);
        let cursor = walk.check_cursor.get();
        let mut caught = None;
        for call in 0..WINDOW / CHECK_CHAINS {
            if !walk.matches_dalek() {
                caught = Some(call);
                break;
            }
        }
        assert_eq!(caught, Some((k - cursor) / CHECK_CHAINS));
        // The cursor keeps rotating: the same chain is caught again one
        // full rotation later, and nothing else is flagged in between.
        let calls_to_next = WINDOW / CHECK_CHAINS;
        for call in 0..calls_to_next {
            assert_eq!(!walk.matches_dalek(), call == calls_to_next - 1, "call {call}");
        }
    }

    /// The y-only differential addition agrees with dalek on arbitrary
    /// prime-order points P and steps D (not just walk-structured ones).
    #[test]
    fn y_only_differential_addition_matches_dalek() {
        let mut rng = ChaCha20Rng::from_seed([31u8; 32]);
        for _ in 0..200 {
            let sp = Scalar::from_bytes_mod_order(random_base_scalar(&mut rng));
            let sd = Scalar::from_bytes_mod_order(random_base_scalar(&mut rng));
            let (p, d) = (EdwardsPoint::mul_base(&sp), EdwardsPoint::mul_base(&sd));
            let y_of = |q: EdwardsPoint| {
                let mut b = q.compress().to_bytes();
                b[31] &= 0x7F;
                Fe::from_bytes(&b)
            };
            let (yp, yd, yt) = (y_of(p), y_of(d), y_of(p - d));
            let alpha = yp.sub(&yd).sq();
            let beta = yp.add(&yd).sq();
            let a = FE_ONE.add(&yt).mul(&alpha);
            let b = FE_ONE.sub(&yt).mul(&beta);
            let ys = b.sub(&a).mul(&b.add(&a).invert());
            assert_eq!(ys.to_bytes(), y_of(p + d).to_bytes());
        }
    }

    #[test]
    fn scalar_offset_matches_bigint() {
        let mut rng = ChaCha20Rng::from_seed([19u8; 32]);
        let mut scalars = vec![[0u8; 32], [0xFFu8; 32]];
        let mut lo = [0u8; 32];
        lo[31] = 0x40;
        scalars.push(lo);
        let mut a = [0u8; 32];
        a[0] = 0xF8;
        a[1] = 0xFF;
        a[2] = 0xFF;
        a[3] = 0x12;
        a[31] = 0x40;
        scalars.push(a);
        let mut b = [0u8; 32];
        b[2] = 1;
        b[31] = 0x40;
        scalars.push(b);
        for _ in 0..50 {
            scalars.push(random_base_scalar(&mut rng));
        }
        let deltas = [
            0i64,
            1,
            -1,
            8,
            -8,
            255,
            -255,
            256,
            -256,
            65535,
            -65535,
            65536,
            -65536,
            -(16 * WINDOW as i64),
            8 * (1 << 40),
            -8 * (1 << 40),
            i64::MAX,
            i64::MIN + 1,
        ];
        for s in &scalars {
            for &d in &deltas {
                let big = BigUint::from_bytes_le(s);
                let expect = if d >= 0 {
                    big + BigUint::from(d as u64)
                } else if big >= BigUint::from(d.unsigned_abs()) {
                    big - BigUint::from(d.unsigned_abs())
                } else {
                    continue; // would underflow: not a valid input
                };
                if expect >= (BigUint::one() << 256u32) {
                    continue; // would overflow: not a valid input
                }
                assert_eq!(to_big(&scalar_offset(*s, d)), expect, "{} {d}", hex::encode(s));
            }
        }
    }

    /// A walk from the lowest and highest valid bases seeds its chains from
    /// scalars below 2^254 (not clamped, but still exact) and every
    /// candidate of the first windows, including every chain, matches.
    #[test]
    fn walk_from_extreme_bases_matches_dalek_in_every_chain() {
        let mut lo = [0u8; 32];
        lo[31] = 0x40;
        let mut hi = [0xFFu8; 32];
        hi[0] = 0xF8;
        hi[31] = 0x5F;
        let mut walk = Walk::<2>::from_bases([lo, hi]).unwrap();
        for _ in 0..3 {
            walk.next_window(|lane, iter, y| {
                assert_eq!(*y, dalek_y([lo, hi][lane], iter), "lane {lane} iter {iter}");
            });
        }
        // Every chain, not just the rotating subset.
        for lane in 0..2 {
            for k in 0..WINDOW {
                assert_eq!(
                    walk.ys[walk.flip][lane][k].to_bytes(),
                    super::dalek_y(&walk.chain_scalar(lane, k)),
                    "lane {lane} chain {k}"
                );
            }
        }
    }

    /// Random expression fuzz: long chains of add/sub/mul/sq/neg/invert on
    /// unreduced inputs (any value in [0, 2^256), including structured
    /// carry-stressing words) are compared with BigUint after every step, so
    /// every intermediate representation the field can produce is fed back
    /// into every operation.
    #[test]
    fn field_random_expression_chains_match_bigint() {
        let p = p();
        let mut rng = ChaCha20Rng::from_seed([77u8; 32]);
        let random_value = |rng: &mut ChaCha20Rng| -> BigUint {
            let mut b = [0u8; 32];
            rng.fill_bytes(&mut b);
            match rng.next_u32() % 8 {
                0 => {
                    for w in 0..4 {
                        if rng.next_u32() & 1 == 1 {
                            b[w * 8..w * 8 + 8].copy_from_slice(&[0xFF; 8]);
                        }
                    }
                }
                1 => {
                    for w in 0..4 {
                        if rng.next_u32() & 1 == 1 {
                            b[w * 8..w * 8 + 8].copy_from_slice(&[0; 8]);
                        }
                    }
                }
                2 => {
                    // k * p + small, for k in 0..=2 (2p + 31 < 2^256).
                    let k = rng.next_u32() % 3;
                    let v = &p * BigUint::from(k) + BigUint::from(rng.next_u32() % 32);
                    b = from_big(&v);
                }
                3 => {
                    // 2^256 - small.
                    let v = (BigUint::one() << 256u32) - BigUint::from(1 + rng.next_u32() % 2000);
                    b = from_big(&v);
                }
                _ => {}
            }
            to_big(&b)
        };
        for chain in 0..1500 {
            let mut v = random_value(&mut rng);
            let mut fe = fe_big(&v);
            for step in 0..40 {
                let w = random_value(&mut rng);
                let fw = fe_big(&w);
                let op = rng.next_u32() % 7;
                (fe, v) = match op {
                    0 => (fe.add(&fw), (&v + &w) % &p),
                    1 => (fe.sub(&fw), (&v + &p * BigUint::from(2u32) - &w % &p) % &p),
                    2 => (fw.sub(&fe), (&w + &p * BigUint::from(2u32) - &v % &p) % &p),
                    3 => (fe.mul(&fw), (&v * &w) % &p),
                    4 => (fe.sq(), (&v * &v) % &p),
                    5 => (fe.neg(), (&p - &v % &p) % &p),
                    _ => {
                        let r = &v % &p;
                        if r.is_zero() {
                            (fe.invert(), BigUint::zero())
                        } else {
                            (fe.invert(), r.modpow(&(&p - BigUint::from(2u32)), &p))
                        }
                    }
                };
                assert_eq!(to_big(&fe.to_bytes()), &v % &p, "chain {chain} step {step} op {op}");
                // Keep the BigUint side in the same residue class but let
                // the Fe side keep whatever (unreduced) representation it
                // produced.
                v %= &p;
            }
        }
    }

    /// Every candidate of a long multi-window walk (both lanes) equals
    /// dalek's derivation of `base + 8*iter`, and the walk's own bookkeeping
    /// (`next_scalar`, `matches_dalek`) stays consistent throughout.
    #[test]
    fn walk_long_run_every_candidate_matches_dalek() {
        let mut rng = ChaCha20Rng::from_seed([123u8; 32]);
        let mut walk = Walk::<2>::new(&mut rng).unwrap();
        let bases = [walk.scalar_at(0, 0), walk.scalar_at(1, 0)];
        let windows = 40u64;
        let mut n = 0u64;
        for w in 0..windows {
            let first = walk.next_window(|lane, iter, y| {
                assert!(iter >= w * WINDOW as u64 && iter < (w + 1) * WINDOW as u64);
                assert_eq!(*y, dalek_y(bases[lane], iter), "lane {lane} iter {iter}");
                n += 1;
            });
            assert_eq!(first, w * WINDOW as u64);
        }
        assert_eq!(n, windows * Walk::<2>::CANDIDATES_PER_WINDOW);
        assert!(walk.matches_dalek());
        for lane in 0..2 {
            assert_eq!(walk.next_scalar(lane), scalar_add_u64(bases[lane], 8 * windows * WINDOW as u64));
        }

        // An odd lane count exercises the lane-join chain of the batched
        // inversion with a non-power-of-two L.
        let mut three = Walk::<3>::new(&mut rng).unwrap();
        let bases: Vec<[u8; 32]> = (0..3).map(|l| three.scalar_at(l, 0)).collect();
        let mut n = 0u64;
        for _ in 0..3 {
            three.next_window(|lane, iter, y| {
                assert_eq!(*y, dalek_y(bases[lane], iter), "lane {lane} iter {iter}");
                n += 1;
            });
        }
        assert_eq!(n, 3 * Walk::<3>::CANDIDATES_PER_WINDOW);
        assert!(three.matches_dalek());
    }

    /// A walk started from a scalar whose `+8` steps carry all the way into
    /// byte 31 (bit 253 clear by `random_base_scalar`, so the clamp bits are
    /// untouched) stays exact across the carry.
    #[test]
    fn walk_carries_into_the_top_byte() {
        let mut a = [0xFFu8; 32];
        a[0] = 0xF8;
        a[31] = 0x5F;
        // 0x5FFF..FFF8 + 8 = 0x6000..0000: bit 253 set, still clamped.
        let b = scalar_add_u64(a, 8);
        assert_eq!(b[31], 0x60);
        assert_eq!(b[0], 0x00);
        let mut walk = Walk::<2>::from_bases([a, b]).unwrap();
        for _ in 0..2 {
            walk.next_window(|lane, iter, y| {
                assert_eq!(*y, dalek_y([a, b][lane], iter), "lane {lane} iter {iter}");
            });
        }
        assert!(walk.matches_dalek());
    }

    #[test]
    #[ignore]
    fn timing_breakdown() {
        use std::time::Instant;
        let a = Fe([123456789, 987654321, 555555555, 444444444]);
        let n = 20_000_000u64;
        let mut x = a;
        let t = Instant::now();
        for _ in 0..n {
            x = x.mul(&a);
        }
        let dep = t.elapsed().as_nanos() as f64 / n as f64;
        std::hint::black_box(x);
        let (mut x0, mut x1, mut x2, mut x3) = (a, a.sq(), a.sq().sq(), a.sq().sq().sq());
        let t = Instant::now();
        for _ in 0..n / 4 {
            x0 = x0.mul(&a);
            x1 = x1.mul(&a);
            x2 = x2.mul(&a);
            x3 = x3.mul(&a);
        }
        let indep = t.elapsed().as_nanos() as f64 / n as f64;
        std::hint::black_box((x0, x1, x2, x3));
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        let mut acc = 0u64;
        let t = Instant::now();
        let mut w2r = Walk::<2>::new(&mut rng).unwrap();
        for _ in 0..9 {
            assert!(w2r.restart(&mut rng));
        }
        let restart_us = t.elapsed().as_micros() as f64 / 10.0;
        let t = Instant::now();
        for _ in 0..100 {
            assert!(w2r.matches_dalek());
        }
        let check_us = t.elapsed().as_micros() as f64 / 100.0;
        eprintln!("Walk::<2> (re)start {restart_us:.0} us, matches_dalek {check_us:.1} us");
        let mut w1 = Walk::<1>::new(&mut rng).unwrap();
        let t = Instant::now();
        for _ in 0..2000 {
            w1.next_window(|_, _, y| acc = acc.wrapping_add(y[0] as u64));
        }
        let per1 = t.elapsed().as_nanos() as f64 / (2000.0 * WINDOW as f64);
        let mut w2 = Walk::<2>::new(&mut rng).unwrap();
        let t = Instant::now();
        for _ in 0..2000 {
            w2.next_window(|_, _, y| acc = acc.wrapping_add(y[0] as u64));
        }
        let per2 = t.elapsed().as_nanos() as f64 / (2000.0 * 2.0 * WINDOW as f64);
        let mut w4 = Walk::<4>::new(&mut rng).unwrap();
        let t = Instant::now();
        for _ in 0..1000 {
            w4.next_window(|_, _, y| acc = acc.wrapping_add(y[0] as u64));
        }
        let per4 = t.elapsed().as_nanos() as f64 / (1000.0 * 4.0 * WINDOW as f64);
        std::hint::black_box(acc);
        eprintln!(
            "fe_mul dependent {dep:.2} ns, independent {indep:.2} ns; walk ns/candidate: 1 lane {per1:.1}, 2 lanes {per2:.1}, 4 lanes {per4:.1}"
        );
    }
}
