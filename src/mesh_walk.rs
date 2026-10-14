//! Incremental Curve25519 walk in Montgomery `u` for the Meshtastic node-ID
//! search: the twin of `ed_walk::Walk` that emits the X25519 public key
//! (the little-endian Montgomery u-coordinate of `s * B`) of every clamped
//! scalar `base + 8*i` instead of the Edwards `y`.
//!
//! Each lane holds `WINDOW` chains; chain `k` keeps the affine `u` of its
//! current point `P` and of `P - D` (`D = 8 * WINDOW * B`) and steps with
//! Montgomery's x-only differential addition
//!
//! ```text
//! u(P + D) = (u_P * u_D - 1)^2 / (u(P - D) * (u_P - u_D)^2)
//! ```
//!
//! (2 multiplications and 2 squarings), the inversions batched over all
//! `L * WINDOW` chains of a window (3 multiplications each plus the final
//! product): the same ~6M + 2S per candidate as the y-walk. The formula is
//! valid whenever `P - D` is not the 2-torsion point `u = 0`, none of the
//! points is the identity and `P != +-D`; every walk point is `s * B` with
//! `s` in `(2^254 - 2^16, 2^255)`, `s = 0 mod 8`, never `0` or
//! `+-8*WINDOW` mod the group order, so no exceptional case is reached.
//!
//! Seeds are the Edwards points `(base - 16*WINDOW + 8*j) * B` from
//! `ed_walk::mul_base` converted with `u = (Z + Y) / (Z - Y)` (one batched
//! inversion), and every seed is checked against curve25519-dalek's
//! `to_montgomery()`; callers re-check periodically (`matches_dalek`) and
//! re-derive every hit, so an arithmetic fault can only ever miss keys.

use std::cell::Cell;

use rand_core::RngCore;

use crate::ed_walk::{basepoint, mul_base, scalar_offset, Fe, FE_ONE, FE_ZERO, WINDOW};
use crate::keygen::{random_base_scalar, scalar_add_u64};

/// Interleaved running-product chains per lane in the batched inversion.
const INV_CHAINS: usize = 8;
/// Chains per lane re-derived with dalek by one `matches_dalek` call.
const CHECK_CHAINS: usize = 2;

/// dalek's X25519 public key (Montgomery u, little-endian) of any 256-bit
/// scalar reduced mod the group order: exactly what the walk emits.
fn dalek_u(scalar: &[u8; 32]) -> [u8; 32] {
    use curve25519_dalek::edwards::EdwardsPoint;
    use curve25519_dalek::scalar::Scalar;
    EdwardsPoint::mul_base(&Scalar::from_bytes_mod_order(*scalar))
        .to_montgomery()
        .to_bytes()
}

pub struct UWalk<const L: usize> {
    base: [[u8; 32]; L],
    next_iter: u64,
    /// `us[flip][l][k]` is the u of candidate `next_iter - WINDOW + k` of
    /// lane `l`; the other array holds candidate `next_iter - 2*WINDOW + k`.
    us: [Box<[[Fe; WINDOW]; L]>; 2],
    flip: usize,
    den: Box<[[Fe; WINDOW]; L]>,
    prods: Box<[[Fe; WINDOW]; L]>,
    /// Montgomery u of the chain step `D = 8*WINDOW*B`.
    step_u: Fe,
    check_cursor: Cell<usize>,
}

/// `u = (Z + Y) / (Z - Y)` of an extended point, given `1 / (Z - Y)`.
fn u_from_zy(z: &Fe, y: &Fe, zmy_inv: &Fe) -> Fe {
    z.add(y).mul(zmy_inv)
}

impl<const L: usize> UWalk<L> {
    pub const CANDIDATES_PER_WINDOW: u64 = (L * WINDOW) as u64;

    pub fn new(rng: &mut impl RngCore) -> Option<Self> {
        Self::from_bases(std::array::from_fn(|_| random_base_scalar(rng)))
    }

    pub fn from_bases(base: [[u8; 32]; L]) -> Option<Self> {
        let step_u = {
            let mut s = [0u8; 32];
            s[..8].copy_from_slice(&(8 * WINDOW as u64).to_le_bytes());
            let p = mul_base(&s);
            u_from_zy(&p.z, &p.y, &p.z.sub(&p.y).invert())
        };
        let mut walk = UWalk {
            base,
            next_iter: 0,
            us: [Box::new([[FE_ZERO; WINDOW]; L]), Box::new([[FE_ZERO; WINDOW]; L])],
            flip: 0,
            den: Box::new([[FE_ZERO; WINDOW]; L]),
            prods: Box::new([[FE_ZERO; WINDOW]; L]),
            step_u,
            check_cursor: Cell::new(0),
        };
        walk.restart_from(base).then_some(walk)
    }

    pub fn restart(&mut self, rng: &mut impl RngCore) -> bool {
        self.restart_from(std::array::from_fn(|_| random_base_scalar(rng)))
    }

    /// Restart from explicit bases; returns whether all `2 * L * WINDOW`
    /// seed points match dalek exactly.
    pub fn restart_from(&mut self, base: [[u8; 32]; L]) -> bool {
        self.base = base;
        self.next_iter = 0;
        self.flip = 0;
        self.check_cursor.set(0);

        let step8 = basepoint().double().double().double().to_niels();
        let n = 2 * WINDOW;
        let mut ys = vec![FE_ZERO; L * n];
        let mut zs = vec![FE_ZERO; L * n];
        let mut zmy = vec![FE_ZERO; L * n];
        for l in 0..L {
            let mut p = mul_base(&scalar_offset(base[l], -(16 * WINDOW as i64)));
            for j in 0..n {
                ys[l * n + j] = p.y;
                zs[l * n + j] = p.z;
                zmy[l * n + j] = p.z.sub(&p.y);
                p = p.add_niels(&step8);
            }
        }
        let mut prods = vec![FE_ZERO; L * n];
        let mut acc = FE_ONE;
        for (prod, d) in prods.iter_mut().zip(zmy.iter()) {
            acc = acc.mul(d);
            *prod = acc;
        }
        let mut inv = acc.invert();
        for i in (0..L * n).rev() {
            let dinv = if i == 0 { inv } else { inv.mul(&prods[i - 1]) };
            inv = inv.mul(&zmy[i]);
            let u = u_from_zy(&zs[i], &ys[i], &dinv);
            let (l, j) = (i / n, i % n);
            if j < WINDOW {
                self.us[1][l][j] = u;
            } else {
                self.us[0][l][j - WINDOW] = u;
            }
        }

        use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
        use curve25519_dalek::edwards::EdwardsPoint;
        use curve25519_dalek::scalar::Scalar;
        let d8 = ED25519_BASEPOINT_POINT.mul_by_cofactor();
        (0..L).all(|l| {
            let s = scalar_offset(base[l], -(16 * WINDOW as i64));
            let mut p = EdwardsPoint::mul_base(&Scalar::from_bytes_mod_order(s));
            (0..n).all(|j| {
                let expected = p.to_montgomery().to_bytes();
                p += d8;
                let mine = if j < WINDOW { self.us[1][l][j] } else { self.us[0][l][j - WINDOW] };
                mine.to_bytes() == expected
            })
        })
    }

    /// The clamped private scalar of candidate `iter` of `lane`.
    pub fn scalar_at(&self, lane: usize, iter: u64) -> [u8; 32] {
        scalar_add_u64(self.base[lane], 8 * iter)
    }

    pub fn iters_done(&self) -> u64 {
        self.next_iter
    }

    fn chain_scalar(&self, lane: usize, k: usize) -> [u8; 32] {
        scalar_offset(
            self.base[lane],
            8 * (self.next_iter as i64 - WINDOW as i64 + k as i64),
        )
    }

    /// True iff the current points of the next `CHECK_CHAINS` chains of
    /// every lane agree with dalek (rotating cursor over all chains).
    pub fn matches_dalek(&self) -> bool {
        let cursor = self.check_cursor.get();
        self.check_cursor.set((cursor + CHECK_CHAINS) % WINDOW);
        (0..L).all(|lane| {
            (0..CHECK_CHAINS).all(|i| {
                let k = (cursor + i) % WINDOW;
                self.us[self.flip][lane][k].to_bytes() == dalek_u(&self.chain_scalar(lane, k))
            })
        })
    }

    /// Test hook: corrupt every chain of `lane`.
    #[cfg(test)]
    pub fn desynchronise(&mut self, lane: usize) {
        for u in self.us[self.flip][lane].iter_mut() {
            *u = u.add(&FE_ONE);
        }
    }

    /// Advance every lane by one window and hand each candidate's X25519
    /// public key (canonical little-endian u) to `visit(lane, iter, u)`;
    /// the order within the window is unspecified. Returns the window's
    /// first iteration index.
    #[inline]
    pub fn next_window(&mut self, mut visit: impl FnMut(usize, u64, &[u8; 32])) -> u64 {
        let first = self.next_iter;
        let [us0, us1] = &mut self.us;
        let (cur, prev): (&[[Fe; WINDOW]; L], &mut [[Fe; WINDOW]; L]) = if self.flip == 0 {
            (us0, us1)
        } else {
            (us1, us0)
        };
        let den = &mut *self.den;
        let prods = &mut *self.prods;
        let ud = self.step_u;

        // Numerators (into `prev`, whose points are no longer needed) and
        // denominators of every chain's next u, two chains interleaved.
        for l in 0..L {
            for k in (0..WINDOW).step_by(2) {
                let (up0, up1) = (cur[l][k], cur[l][k + 1]);
                let (ut0, ut1) = (prev[l][k], prev[l][k + 1]);
                let m0 = up0.mul(&ud);
                let m1 = up1.mul(&ud);
                let d0 = up0.sub(&ud);
                let d1 = up1.sub(&ud);
                let m0 = m0.sub(&FE_ONE);
                let m1 = m1.sub(&FE_ONE);
                let num0 = m0.sq();
                let num1 = m1.sq();
                let dd0 = d0.sq();
                let dd1 = d1.sq();
                den[l][k] = ut0.mul(&dd0);
                den[l][k + 1] = ut1.mul(&dd1);
                prev[l][k] = num0;
                prev[l][k + 1] = num1;
            }
        }

        // Montgomery batch inversion, INV_CHAINS interleaved chains per lane
        // joined so one field inversion serves them all (as ed_walk::Walk).
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

        for l in 0..L {
            for k in (INV_CHAINS..WINDOW).rev() {
                let g = k % INV_CHAINS;
                let dinv = inv[l][g].mul(&prods[l][k - INV_CHAINS]);
                inv[l][g] = inv[l][g].mul(&den[l][k]);
                let u = prev[l][k].mul(&dinv);
                prev[l][k] = u;
                visit(l, first + k as u64, &u.to_bytes());
            }
            for k in (0..INV_CHAINS).rev() {
                let u = prev[l][k].mul(&inv[l][k]);
                prev[l][k] = u;
                visit(l, first + k as u64, &u.to_bytes());
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
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn dalek_u_at(base: [u8; 32], iter: u64) -> [u8; 32] {
        crate::meshtastic::public_key(&scalar_add_u64(base, 8 * iter))
    }

    /// Every candidate of several windows (both lanes, including the
    /// window boundaries and a restart) must equal dalek's X25519 public key
    /// of its scalar, byte for byte.
    #[test]
    fn walk_windows_match_dalek_x25519() {
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let bases = [random_base_scalar(&mut rng), random_base_scalar(&mut rng)];
        let mut walk = UWalk::<2>::from_bases(bases).expect("seeds match dalek");
        for window in 0..4u64 {
            let mut seen = 0;
            let first = walk.next_window(|lane, iter, u| {
                assert_eq!(u, &dalek_u_at(bases[lane], iter), "lane {lane} iter {iter}");
                seen += 1;
            });
            assert_eq!(first, window * WINDOW as u64);
            assert_eq!(seen, 2 * WINDOW);
            assert!(walk.matches_dalek());
        }
        assert_eq!(walk.iters_done(), 4 * WINDOW as u64);
        // Restart from new bases and check again.
        let bases2 = [random_base_scalar(&mut rng), random_base_scalar(&mut rng)];
        assert!(walk.restart_from(bases2));
        walk.next_window(|lane, iter, u| {
            assert_eq!(u, &dalek_u_at(bases2[lane], iter));
        });
        assert_eq!(walk.scalar_at(1, 5), scalar_add_u64(bases2[1], 40));
    }

    /// The oracle vectors' secrets as bases: the walk's first candidate of
    /// every lane is the vector's public key.
    #[test]
    fn walk_reproduces_oracle_public_keys() {
        for chunk in crate::meshtastic::oracle_vectors().chunks_exact(2).take(16) {
            // Skip bases without walk headroom (the all-ones vector).
            if chunk.iter().any(|(s, _, _)| s[8..31].iter().all(|b| *b == 0xFF)) {
                continue;
            }
            let bases = [chunk[0].0, chunk[1].0];
            let mut walk = UWalk::<2>::from_bases(bases).expect("seeds match dalek");
            walk.next_window(|lane, iter, u| {
                if iter == 0 {
                    assert_eq!(u, &chunk[lane].1);
                    assert_eq!(crate::meshtastic::crc32_bitwise(u), chunk[lane].2);
                }
            });
        }
    }

    #[test]
    fn walk_with_one_lane_and_byte_carries() {
        let mut rng = ChaCha20Rng::from_seed([12u8; 32]);
        let mut base = random_base_scalar(&mut rng);
        // Force carries across several bytes within the first windows.
        base[0] = 0xF8;
        base[1] = 0xFF;
        base[2] = 0xFF;
        let mut walk = UWalk::<1>::from_bases([base]).expect("seeds match dalek");
        for _ in 0..3 {
            walk.next_window(|_, iter, u| assert_eq!(u, &dalek_u_at(base, iter), "iter {iter}"));
        }
    }

    #[test]
    fn matches_dalek_detects_desynchronisation() {
        let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
        let mut walk = UWalk::<2>::new(&mut rng).unwrap();
        walk.next_window(|_, _, _| {});
        assert!(walk.matches_dalek());
        walk.desynchronise(1);
        assert!(!walk.matches_dalek());
    }
}
