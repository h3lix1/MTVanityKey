//! Differential tests for the Metal Ed25519 field primitives
//! (src/kernels/ed25519.metal: fe_mul, fe_sq, fe_invert).
//!
//! The shader is compiled together with a test kernel that also contains the
//! previous (ref10 rounded-carry) fe_mul / fe_sq verbatim, so every case is
//! evaluated three ways: the current GPU primitives, the old GPU primitives,
//! and num-bigint on the host. The canonical encodings (fe_tobytes32) must
//! agree, the new limbs must satisfy the bounds the shader documents, and a
//! host model of the new reduction (i128 with "fits a long" assertions)
//! must reproduce the GPU limbs bit for bit on inputs at the documented
//! input bounds, proving the 64-bit columns cannot overflow there.
#![cfg(target_os = "macos")]

use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

const SHADER: &str = include_str!("../src/kernels/ed25519.metal");

/// Previous implementation (ref10 rounded carry chain), copied verbatim from
/// the shader before the fe_carry change, plus the test kernel.
const TEST_SUFFIX: &str = r#"
inline Fe ref_fe_carry(long h0, long h1, long h2, long h3, long h4,
                       long h5, long h6, long h7, long h8, long h9) {
    long carry0, carry1, carry2, carry3, carry4, carry5, carry6, carry7, carry8, carry9;

    carry0 = (h0 + (long)(1 << 25)) >> 26; h1 += carry0; h0 -= carry0 << 26;
    carry4 = (h4 + (long)(1 << 25)) >> 26; h5 += carry4; h4 -= carry4 << 26;
    carry1 = (h1 + (long)(1 << 24)) >> 25; h2 += carry1; h1 -= carry1 << 25;
    carry5 = (h5 + (long)(1 << 24)) >> 25; h6 += carry5; h5 -= carry5 << 25;
    carry2 = (h2 + (long)(1 << 25)) >> 26; h3 += carry2; h2 -= carry2 << 26;
    carry6 = (h6 + (long)(1 << 25)) >> 26; h7 += carry6; h6 -= carry6 << 26;
    carry3 = (h3 + (long)(1 << 24)) >> 25; h4 += carry3; h3 -= carry3 << 25;
    carry7 = (h7 + (long)(1 << 24)) >> 25; h8 += carry7; h7 -= carry7 << 25;
    carry4 = (h4 + (long)(1 << 25)) >> 26; h5 += carry4; h4 -= carry4 << 26;
    carry8 = (h8 + (long)(1 << 25)) >> 26; h9 += carry8; h8 -= carry8 << 26;
    carry9 = (h9 + (long)(1 << 24)) >> 25; h0 += carry9 * 19; h9 -= carry9 << 25;
    carry0 = (h0 + (long)(1 << 25)) >> 26; h1 += carry0; h0 -= carry0 << 26;

    Fe h;
    h.v[0] = (int) h0;
    h.v[1] = (int) h1;
    h.v[2] = (int) h2;
    h.v[3] = (int) h3;
    h.v[4] = (int) h4;
    h.v[5] = (int) h5;
    h.v[6] = (int) h6;
    h.v[7] = (int) h7;
    h.v[8] = (int) h8;
    h.v[9] = (int) h9;
    return h;
}

inline Fe ref_fe_mul(Fe f, Fe g) {
    int f0 = f.v[0], f1 = f.v[1], f2 = f.v[2], f3 = f.v[3], f4 = f.v[4];
    int f5 = f.v[5], f6 = f.v[6], f7 = f.v[7], f8 = f.v[8], f9 = f.v[9];
    int g0 = g.v[0], g1 = g.v[1], g2 = g.v[2], g3 = g.v[3], g4 = g.v[4];
    int g5 = g.v[5], g6 = g.v[6], g7 = g.v[7], g8 = g.v[8], g9 = g.v[9];

    int g1_19 = 19 * g1;
    int g2_19 = 19 * g2;
    int g3_19 = 19 * g3;
    int g4_19 = 19 * g4;
    int g5_19 = 19 * g5;
    int g6_19 = 19 * g6;
    int g7_19 = 19 * g7;
    int g8_19 = 19 * g8;
    int g9_19 = 19 * g9;
    int f1_2 = 2 * f1;
    int f3_2 = 2 * f3;
    int f5_2 = 2 * f5;
    int f7_2 = 2 * f7;
    int f9_2 = 2 * f9;

    long f0g0    = f0   * (long) g0;
    long f0g1    = f0   * (long) g1;
    long f0g2    = f0   * (long) g2;
    long f0g3    = f0   * (long) g3;
    long f0g4    = f0   * (long) g4;
    long f0g5    = f0   * (long) g5;
    long f0g6    = f0   * (long) g6;
    long f0g7    = f0   * (long) g7;
    long f0g8    = f0   * (long) g8;
    long f0g9    = f0   * (long) g9;
    long f1g0    = f1   * (long) g0;
    long f1g1_2  = f1_2 * (long) g1;
    long f1g2    = f1   * (long) g2;
    long f1g3_2  = f1_2 * (long) g3;
    long f1g4    = f1   * (long) g4;
    long f1g5_2  = f1_2 * (long) g5;
    long f1g6    = f1   * (long) g6;
    long f1g7_2  = f1_2 * (long) g7;
    long f1g8    = f1   * (long) g8;
    long f1g9_38 = f1_2 * (long) g9_19;
    long f2g0    = f2   * (long) g0;
    long f2g1    = f2   * (long) g1;
    long f2g2    = f2   * (long) g2;
    long f2g3    = f2   * (long) g3;
    long f2g4    = f2   * (long) g4;
    long f2g5    = f2   * (long) g5;
    long f2g6    = f2   * (long) g6;
    long f2g7    = f2   * (long) g7;
    long f2g8_19 = f2   * (long) g8_19;
    long f2g9_19 = f2   * (long) g9_19;
    long f3g0    = f3   * (long) g0;
    long f3g1_2  = f3_2 * (long) g1;
    long f3g2    = f3   * (long) g2;
    long f3g3_2  = f3_2 * (long) g3;
    long f3g4    = f3   * (long) g4;
    long f3g5_2  = f3_2 * (long) g5;
    long f3g6    = f3   * (long) g6;
    long f3g7_38 = f3_2 * (long) g7_19;
    long f3g8_19 = f3   * (long) g8_19;
    long f3g9_38 = f3_2 * (long) g9_19;
    long f4g0    = f4   * (long) g0;
    long f4g1    = f4   * (long) g1;
    long f4g2    = f4   * (long) g2;
    long f4g3    = f4   * (long) g3;
    long f4g4    = f4   * (long) g4;
    long f4g5    = f4   * (long) g5;
    long f4g6_19 = f4   * (long) g6_19;
    long f4g7_19 = f4   * (long) g7_19;
    long f4g8_19 = f4   * (long) g8_19;
    long f4g9_19 = f4   * (long) g9_19;
    long f5g0    = f5   * (long) g0;
    long f5g1_2  = f5_2 * (long) g1;
    long f5g2    = f5   * (long) g2;
    long f5g3_2  = f5_2 * (long) g3;
    long f5g4    = f5   * (long) g4;
    long f5g5_38 = f5_2 * (long) g5_19;
    long f5g6_19 = f5   * (long) g6_19;
    long f5g7_38 = f5_2 * (long) g7_19;
    long f5g8_19 = f5   * (long) g8_19;
    long f5g9_38 = f5_2 * (long) g9_19;
    long f6g0    = f6   * (long) g0;
    long f6g1    = f6   * (long) g1;
    long f6g2    = f6   * (long) g2;
    long f6g3    = f6   * (long) g3;
    long f6g4_19 = f6   * (long) g4_19;
    long f6g5_19 = f6   * (long) g5_19;
    long f6g6_19 = f6   * (long) g6_19;
    long f6g7_19 = f6   * (long) g7_19;
    long f6g8_19 = f6   * (long) g8_19;
    long f6g9_19 = f6   * (long) g9_19;
    long f7g0    = f7   * (long) g0;
    long f7g1_2  = f7_2 * (long) g1;
    long f7g2    = f7   * (long) g2;
    long f7g3_38 = f7_2 * (long) g3_19;
    long f7g4_19 = f7   * (long) g4_19;
    long f7g5_38 = f7_2 * (long) g5_19;
    long f7g6_19 = f7   * (long) g6_19;
    long f7g7_38 = f7_2 * (long) g7_19;
    long f7g8_19 = f7   * (long) g8_19;
    long f7g9_38 = f7_2 * (long) g9_19;
    long f8g0    = f8   * (long) g0;
    long f8g1    = f8   * (long) g1;
    long f8g2_19 = f8   * (long) g2_19;
    long f8g3_19 = f8   * (long) g3_19;
    long f8g4_19 = f8   * (long) g4_19;
    long f8g5_19 = f8   * (long) g5_19;
    long f8g6_19 = f8   * (long) g6_19;
    long f8g7_19 = f8   * (long) g7_19;
    long f8g8_19 = f8   * (long) g8_19;
    long f8g9_19 = f8   * (long) g9_19;
    long f9g0    = f9   * (long) g0;
    long f9g1_38 = f9_2 * (long) g1_19;
    long f9g2_19 = f9   * (long) g2_19;
    long f9g3_38 = f9_2 * (long) g3_19;
    long f9g4_19 = f9   * (long) g4_19;
    long f9g5_38 = f9_2 * (long) g5_19;
    long f9g6_19 = f9   * (long) g6_19;
    long f9g7_38 = f9_2 * (long) g7_19;
    long f9g8_19 = f9   * (long) g8_19;
    long f9g9_38 = f9_2 * (long) g9_19;

    long h0 = f0g0 + f1g9_38 + f2g8_19 + f3g7_38 + f4g6_19 + f5g5_38 + f6g4_19 + f7g3_38 + f8g2_19 + f9g1_38;
    long h1 = f0g1 + f1g0    + f2g9_19 + f3g8_19 + f4g7_19 + f5g6_19 + f6g5_19 + f7g4_19 + f8g3_19 + f9g2_19;
    long h2 = f0g2 + f1g1_2  + f2g0    + f3g9_38 + f4g8_19 + f5g7_38 + f6g6_19 + f7g5_38 + f8g4_19 + f9g3_38;
    long h3 = f0g3 + f1g2    + f2g1    + f3g0    + f4g9_19 + f5g8_19 + f6g7_19 + f7g6_19 + f8g5_19 + f9g4_19;
    long h4 = f0g4 + f1g3_2  + f2g2    + f3g1_2  + f4g0    + f5g9_38 + f6g8_19 + f7g7_38 + f8g6_19 + f9g5_38;
    long h5 = f0g5 + f1g4    + f2g3    + f3g2    + f4g1    + f5g0    + f6g9_19 + f7g8_19 + f8g7_19 + f9g6_19;
    long h6 = f0g6 + f1g5_2  + f2g4    + f3g3_2  + f4g2    + f5g1_2  + f6g0    + f7g9_38 + f8g8_19 + f9g7_38;
    long h7 = f0g7 + f1g6    + f2g5    + f3g4    + f4g3    + f5g2    + f6g1    + f7g0    + f8g9_19 + f9g8_19;
    long h8 = f0g8 + f1g7_2  + f2g6    + f3g5_2  + f4g4    + f5g3_2  + f6g2    + f7g1_2  + f8g0    + f9g9_38;
    long h9 = f0g9 + f1g8    + f2g7    + f3g6    + f4g5    + f5g4    + f6g3    + f7g2    + f8g1    + f9g0;

    return ref_fe_carry(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9);
}

inline Fe ref_fe_sq(Fe f) {
    int f0 = f.v[0], f1 = f.v[1], f2 = f.v[2], f3 = f.v[3], f4 = f.v[4];
    int f5 = f.v[5], f6 = f.v[6], f7 = f.v[7], f8 = f.v[8], f9 = f.v[9];

    int f0_2 = 2 * f0;
    int f1_2 = 2 * f1;
    int f2_2 = 2 * f2;
    int f3_2 = 2 * f3;
    int f4_2 = 2 * f4;
    int f5_2 = 2 * f5;
    int f6_2 = 2 * f6;
    int f7_2 = 2 * f7;
    int f5_38 = 38 * f5;
    int f6_19 = 19 * f6;
    int f7_38 = 38 * f7;
    int f8_19 = 19 * f8;
    int f9_38 = 38 * f9;

    long f0f0    = f0   * (long) f0;
    long f0f1_2  = f0_2 * (long) f1;
    long f0f2_2  = f0_2 * (long) f2;
    long f0f3_2  = f0_2 * (long) f3;
    long f0f4_2  = f0_2 * (long) f4;
    long f0f5_2  = f0_2 * (long) f5;
    long f0f6_2  = f0_2 * (long) f6;
    long f0f7_2  = f0_2 * (long) f7;
    long f0f8_2  = f0_2 * (long) f8;
    long f0f9_2  = f0_2 * (long) f9;
    long f1f1_2  = f1_2 * (long) f1;
    long f1f2_2  = f1_2 * (long) f2;
    long f1f3_4  = f1_2 * (long) f3_2;
    long f1f4_2  = f1_2 * (long) f4;
    long f1f5_4  = f1_2 * (long) f5_2;
    long f1f6_2  = f1_2 * (long) f6;
    long f1f7_4  = f1_2 * (long) f7_2;
    long f1f8_2  = f1_2 * (long) f8;
    long f1f9_76 = f1_2 * (long) f9_38;
    long f2f2    = f2   * (long) f2;
    long f2f3_2  = f2_2 * (long) f3;
    long f2f4_2  = f2_2 * (long) f4;
    long f2f5_2  = f2_2 * (long) f5;
    long f2f6_2  = f2_2 * (long) f6;
    long f2f7_2  = f2_2 * (long) f7;
    long f2f8_38 = f2_2 * (long) f8_19;
    long f2f9_38 = f2   * (long) f9_38;
    long f3f3_2  = f3_2 * (long) f3;
    long f3f4_2  = f3_2 * (long) f4;
    long f3f5_4  = f3_2 * (long) f5_2;
    long f3f6_2  = f3_2 * (long) f6;
    long f3f7_76 = f3_2 * (long) f7_38;
    long f3f8_38 = f3_2 * (long) f8_19;
    long f3f9_76 = f3_2 * (long) f9_38;
    long f4f4    = f4   * (long) f4;
    long f4f5_2  = f4_2 * (long) f5;
    long f4f6_38 = f4_2 * (long) f6_19;
    long f4f7_38 = f4   * (long) f7_38;
    long f4f8_38 = f4_2 * (long) f8_19;
    long f4f9_38 = f4   * (long) f9_38;
    long f5f5_38 = f5   * (long) f5_38;
    long f5f6_38 = f5_2 * (long) f6_19;
    long f5f7_76 = f5_2 * (long) f7_38;
    long f5f8_38 = f5_2 * (long) f8_19;
    long f5f9_76 = f5_2 * (long) f9_38;
    long f6f6_19 = f6   * (long) f6_19;
    long f6f7_38 = f6   * (long) f7_38;
    long f6f8_38 = f6_2 * (long) f8_19;
    long f6f9_38 = f6   * (long) f9_38;
    long f7f7_38 = f7   * (long) f7_38;
    long f7f8_38 = f7_2 * (long) f8_19;
    long f7f9_76 = f7_2 * (long) f9_38;
    long f8f8_19 = f8   * (long) f8_19;
    long f8f9_38 = f8   * (long) f9_38;
    long f9f9_38 = f9   * (long) f9_38;

    long h0 = f0f0   + f1f9_76 + f2f8_38 + f3f7_76 + f4f6_38 + f5f5_38;
    long h1 = f0f1_2 + f2f9_38 + f3f8_38 + f4f7_38 + f5f6_38;
    long h2 = f0f2_2 + f1f1_2  + f3f9_76 + f4f8_38 + f5f7_76 + f6f6_19;
    long h3 = f0f3_2 + f1f2_2  + f4f9_38 + f5f8_38 + f6f7_38;
    long h4 = f0f4_2 + f1f3_4  + f2f2    + f5f9_76 + f6f8_38 + f7f7_38;
    long h5 = f0f5_2 + f1f4_2  + f2f3_2  + f6f9_38 + f7f8_38;
    long h6 = f0f6_2 + f1f5_4  + f2f4_2  + f3f3_2  + f7f9_76 + f8f8_19;
    long h7 = f0f7_2 + f1f6_2  + f2f5_2  + f3f4_2  + f8f9_38;
    long h8 = f0f8_2 + f1f7_4  + f2f6_2  + f3f5_4  + f4f4    + f9f9_38;
    long h9 = f0f9_2 + f1f8_2  + f2f7_2  + f3f6_2  + f4f5_2;

    return ref_fe_carry(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9);
}

inline Fe ref_fe_sq_times(Fe f, uint n) {
    for (uint i = 0; i < n; i++) {
        f = ref_fe_sq(f);
    }
    return f;
}

inline Fe ref_fe_invert(Fe z) {
    Fe t0 = ref_fe_sq(z);
    Fe t1 = ref_fe_sq_times(t0, 2);
    t1 = ref_fe_mul(z, t1);
    t0 = ref_fe_mul(t0, t1);
    Fe t2 = ref_fe_sq(t0);
    t1 = ref_fe_mul(t1, t2);
    t2 = ref_fe_sq_times(t1, 5);
    t1 = ref_fe_mul(t2, t1);
    t2 = ref_fe_sq_times(t1, 10);
    t2 = ref_fe_mul(t2, t1);
    Fe t3 = ref_fe_sq_times(t2, 20);
    t2 = ref_fe_mul(t3, t2);
    t2 = ref_fe_sq_times(t2, 10);
    t1 = ref_fe_mul(t2, t1);
    t2 = ref_fe_sq_times(t1, 50);
    t2 = ref_fe_mul(t2, t1);
    t3 = ref_fe_sq_times(t2, 100);
    t2 = ref_fe_mul(t3, t2);
    t2 = ref_fe_sq_times(t2, 50);
    t1 = ref_fe_mul(t2, t1);
    t1 = ref_fe_sq_times(t1, 5);
    return ref_fe_mul(t1, t0);
}

struct FeTestIn {
    int f[10];
    int g[10];
    uint op;
    uint pad[3];
};

struct FeTestOut {
    int new_limbs[10];
    int old_limbs[10];
    uint new_bytes[8];
    uint old_bytes[8];
};

inline void pack32(thread const uchar* enc, device uint* words) {
    for (uint i = 0; i < 8; i++) {
        words[i] = (uint)enc[4 * i] | ((uint)enc[4 * i + 1] << 8)
            | ((uint)enc[4 * i + 2] << 16) | ((uint)enc[4 * i + 3] << 24);
    }
}

kernel void fe_test_kernel(
    device const FeTestIn* in [[buffer(0)]],
    device FeTestOut* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    Fe f, g;
    for (uint i = 0; i < 10; i++) {
        f.v[i] = in[gid].f[i];
        g.v[i] = in[gid].g[i];
    }
    Fe n = fe_zero();
    Fe o = fe_zero();
    switch (in[gid].op) {
        case 0: n = fe_mul(f, g); o = ref_fe_mul(f, g); break;
        case 1: n = fe_sq(f); o = ref_fe_sq(f); break;
        case 2: n = fe_invert(f); o = ref_fe_invert(f); break;
        case 5: {
            // The mesh walk's operand pattern: numerators and denominators
            // 1 -+ K +- B +- A are sums of three fe_mul outputs (f, g and
            // f*g stand in for A, B and K).
            Fe k = fe_mul(f, g);
            Fe num = fe_add(fe_sub(fe_one(), k), fe_add(g, f));
            Fe den = fe_sub(fe_add(fe_one(), k), fe_sub(g, f));
            n = fe_mul(num, den);
            Fe ok = ref_fe_mul(f, g);
            Fe onum = fe_add(fe_sub(fe_one(), ok), fe_add(g, f));
            Fe oden = fe_sub(fe_add(fe_one(), ok), fe_sub(g, f));
            o = ref_fe_mul(onum, oden);
            break;
        }
        case 6: {
            // completed_to_extended's largest operands: (Z+Z+T) * (Z+Z-T)
            // with f, g standing in for fe_mul outputs.
            Fe z2 = fe_add(f, f);
            n = fe_mul(fe_add(z2, g), fe_sub(z2, g));
            o = ref_fe_mul(fe_add(z2, g), fe_sub(z2, g));
            break;
        }
        default: break;
    }
    for (uint i = 0; i < 10; i++) {
        out[gid].new_limbs[i] = n.v[i];
        out[gid].old_limbs[i] = o.v[i];
    }
    uchar enc[32];
    fe_tobytes32(enc, n);
    pack32(enc, out[gid].new_bytes);
    fe_tobytes32(enc, o);
    pack32(enc, out[gid].old_bytes);
}
"#;

const OP_MUL: u32 = 0;
const OP_SQ: u32 = 1;
const OP_INVERT: u32 = 2;
const OP_MESH: u32 = 5;
const OP_Z2T: u32 = 6;

/// Documented input bounds of fe_mul / fe_sq (see fe_carry in the shader):
/// |even limb| < 1.6 * 2^26, |odd limb| < 1.6 * 2^25.
const EVEN_MAX: i64 = 107_374_182;
const ODD_MAX: i64 = 53_687_091;
/// Limb positions of the ref10 radix-2^25.5 representation.
const POS: [u32; 10] = [0, 26, 51, 77, 102, 128, 153, 179, 204, 230];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct FeTestIn {
    f: [i32; 10],
    g: [i32; 10],
    op: u32,
    pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct FeTestOut {
    new_limbs: [i32; 10],
    old_limbs: [i32; 10],
    new_bytes: [u32; 8],
    old_bytes: [u32; 8],
}

fn p() -> BigInt {
    (BigInt::one() << 255usize) - BigInt::from(19)
}

fn limb_max(i: usize) -> i64 {
    if i % 2 == 0 {
        EVEN_MAX
    } else {
        ODD_MAX
    }
}

/// Integer value of a limb vector (signed, unreduced).
fn limbs_value(v: &[i32; 10]) -> BigInt {
    let mut acc = BigInt::zero();
    for i in 0..10 {
        acc += BigInt::from(v[i]) << POS[i] as usize;
    }
    acc
}

fn modp(x: &BigInt) -> BigUint {
    let p = p();
    let r = ((x % &p) + &p) % &p;
    r.to_biguint().unwrap()
}

fn canonical_bytes(x: &BigInt) -> [u8; 32] {
    let r = modp(x);
    let mut out = [0u8; 32];
    for (i, b) in r.to_bytes_le().into_iter().enumerate() {
        out[i] = b;
    }
    out
}

fn words_to_bytes(w: &[u32; 8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (chunk, word) in out.chunks_exact_mut(4).zip(w) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    out
}

/// Non-negative limbs of a value (top limb takes the remaining bits, so
/// values >= p and >= 2^255 are representable, non-canonically, up to the
/// odd-limb bound of the contract: about 1.6 * 2^255).
fn limbs_from_uint(x: &BigUint) -> [i32; 10] {
    assert!(x.bits() <= 256);
    let mut v = [0i32; 10];
    let mut rest = x.clone();
    for i in 0..10 {
        let bits = if i == 9 {
            26
        } else {
            (POS[i + 1] - POS[i]) as usize
        };
        let mask = (BigUint::one() << bits) - BigUint::one();
        let limb: BigUint = &rest & &mask;
        v[i] = u32::try_from(limb).unwrap() as i32;
        rest >>= bits;
    }
    assert!(rest.is_zero());
    assert!((v[9] as i64) <= ODD_MAX, "value too large for the limb contract");
    v
}

fn rng() -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(0xf1e1d_ca55_1e5)
}

fn random_limbs_in_bounds(rng: &mut ChaCha20Rng) -> [i32; 10] {
    let mut v = [0i32; 10];
    for i in 0..10 {
        let m = limb_max(i);
        v[i] = rng.gen_range(-m..=m) as i32;
    }
    v
}

/// Limbs of a fe_mul output: even in [-2^25, 2^25), odd in [-2^24, 2^24).
fn random_reduced_limbs(rng: &mut ChaCha20Rng) -> [i32; 10] {
    let mut v = [0i32; 10];
    for i in 0..10 {
        let bits = if i % 2 == 0 { 26 } else { 25 };
        v[i] = (rng.next_u32() & ((1u32 << bits) - 1)) as i32 - (1 << (bits - 1));
    }
    v
}

fn random_canonical(rng: &mut ChaCha20Rng) -> [i32; 10] {
    let mut bytes = [0u8; 40];
    rng.fill_bytes(&mut bytes);
    let x = BigUint::from_bytes_le(&bytes) % p().to_biguint().unwrap();
    limbs_from_uint(&x)
}

fn extreme_limbs(signs: u32) -> [i32; 10] {
    let mut v = [0i32; 10];
    for i in 0..10 {
        let m = limb_max(i) as i32;
        v[i] = if (signs >> i) & 1 == 1 { -m } else { m };
    }
    v
}

fn edge_values() -> Vec<[i32; 10]> {
    let p = p().to_biguint().unwrap();
    let two255 = BigUint::one() << 255usize;
    let mut vals = vec![
        BigUint::zero(),
        BigUint::one(),
        BigUint::from(2u32),
        &p - BigUint::one(),
        &p - BigUint::from(2u32),
        p.clone(),
        &p + BigUint::one(),
        &p + BigUint::from(18u32),
        &p + BigUint::from(19u32),
        &two255 - BigUint::one(),
        two255.clone(),
        &two255 + BigUint::from(18u32),
        &two255 + BigUint::from(19u32),
        &p + (BigUint::one() << 254usize),
        (BigUint::from(3u32) << 254usize) - BigUint::one(),
    ];
    // Single-limb patterns: one limb at its 26/25-bit maximum.
    for i in 0..10 {
        let bits = if i == 9 { 25 } else { (POS[i + 1] - POS[i]) as usize };
        vals.push(((BigUint::one() << bits) - BigUint::one()) << POS[i] as usize);
    }
    let mut out: Vec<[i32; 10]> = vals.iter().map(limbs_from_uint).collect();
    // Negated and ref10-style signed variants of the same values.
    let n = out.len();
    for k in 0..n {
        let mut neg = out[k];
        for limb in neg.iter_mut() {
            *limb = -*limb;
        }
        out.push(neg);
    }
    // Every limb at its documented bound, all sign patterns on a subset.
    out.push(extreme_limbs(0));
    out.push(extreme_limbs(0x3ff));
    out.push(extreme_limbs(0x155));
    out.push(extreme_limbs(0x2aa));
    out.push(extreme_limbs(0x0f0));
    out.push(extreme_limbs(0x30f));
    out
}

/// Host model of the new fe_mul: the ref10 product columns in i128 plus the
/// sequential rounded-carry chain, asserting at each point the GPU uses an
/// `int` or a `long` that the value fits one. Returns the limbs exactly as
/// the GPU produces them.
fn model_columns_mul(f: &[i32; 10], g: &[i32; 10]) -> [i128; 10] {
    let f: Vec<i128> = f.iter().map(|&x| x as i128).collect();
    let g: Vec<i128> = g.iter().map(|&x| x as i128).collect();
    let mut h = [0i128; 10];
    for i in 0..10 {
        for j in 0..10 {
            let mut w = 1i128;
            if i % 2 == 1 && j % 2 == 1 {
                w *= 2;
            }
            if i + j >= 10 {
                w *= 19;
            }
            // The GPU forms 19*g_j and 2*f_i as ints first.
            let gj = if i + j >= 10 { 19 * g[j] } else { g[j] };
            let fi = if i % 2 == 1 && j % 2 == 1 { 2 * f[i] } else { f[i] };
            assert!(fits_int(gj) && fits_int(fi), "precompute overflow");
            let prod = fi * gj;
            assert_eq!(prod, w * f[i] * g[j]);
            assert!(fits_long(prod), "product overflow");
            h[(i + j) % 10] += prod;
        }
    }
    h
}

fn model_columns_sq(f: &[i32; 10]) -> [i128; 10] {
    // fe_sq merges the symmetric terms of fe_mul(f, f): the column sums are
    // the same integers, but the int precomputes (2*f_i, 38*f_odd, 19*f6,
    // 19*f8) and the merged products (up to 76*f_i*f_j) are larger, so
    // check those exactly as the shader forms them.
    let f: Vec<i128> = f.iter().map(|&x| x as i128).collect();
    for i in 0..8 {
        assert!(fits_int(2 * f[i]), "2*f{i} overflow");
    }
    for i in [5, 7, 9] {
        assert!(fits_int(38 * f[i]), "38*f{i} overflow");
    }
    for i in [6, 8] {
        assert!(fits_int(19 * f[i]), "19*f{i} overflow");
    }
    let mut h = [0i128; 10];
    for i in 0..10 {
        for j in i..10 {
            let mut w = 1i128;
            if i != j {
                w *= 2;
            }
            if i % 2 == 1 && j % 2 == 1 {
                w *= 2;
            }
            if i + j >= 10 {
                w *= 19;
            }
            let prod = w * f[i] * f[j];
            assert!(fits_long(prod), "fe_sq product overflow");
            h[(i + j) % 10] += prod;
        }
    }
    assert_eq!(h, model_columns_mul(&f.iter().map(|&x| x as i32).collect::<Vec<_>>().try_into().unwrap(), &f.iter().map(|&x| x as i32).collect::<Vec<_>>().try_into().unwrap()));
    h
}

fn fits_long(x: i128) -> bool {
    x >= i64::MIN as i128 && x <= i64::MAX as i128
}

fn fits_int(x: i128) -> bool {
    x >= i32::MIN as i128 && x <= i32::MAX as i128
}

fn model_carry(mut h: [i128; 10]) -> [i32; 10] {
    for v in h.iter() {
        assert!(fits_long(*v), "column overflow: {v}");
    }
    // Rounded carries in the shader's sequential order; the 2^50
    // pass-through bias is modelled as what it is: the next column's bias.
    let mut r = [0i128; 10];
    for i in 0..9 {
        let bits = if i % 2 == 0 { 26 } else { 25 };
        let bias = 1i128 << (bits - 1);
        let hb = h[i] + bias;
        assert!(fits_long(hb + (1i128 << 50)), "biased column overflow");
        let c = hb >> bits;
        r[i] = (hb & ((1 << bits) - 1)) - bias;
        h[i + 1] += c;
        assert!(fits_long(h[i + 1] + (1i128 << 50)), "carry target overflow");
    }
    let hb = h[9] + (1i128 << 24);
    let c9 = hb >> 25;
    r[9] = (hb & ((1 << 25) - 1)) - (1i128 << 24);
    assert!(c9.abs() < (1i128 << 35));
    let h0b = r[0] + (1i128 << 25) + c9 * 19;
    assert!(fits_long(h0b));
    let c0 = h0b >> 26;
    assert!(fits_int(c0));
    r[0] = (h0b & ((1 << 26) - 1)) - (1i128 << 25);
    r[1] += c0;
    let mut out = [0i32; 10];
    for i in 0..10 {
        assert!(fits_int(r[i]));
        out[i] = r[i] as i32;
    }
    out
}

fn model_mul(f: &[i32; 10], g: &[i32; 10]) -> [i32; 10] {
    model_carry(model_columns_mul(f, g))
}

fn model_sq(f: &[i32; 10]) -> [i32; 10] {
    model_carry(model_columns_sq(f))
}

fn check_output_bounds(v: &[i32; 10], what: &str) {
    for i in 0..10 {
        let (lo, hi) = match i {
            1 => (-(1 << 24) - (1 << 17), (1 << 24) + (1 << 17)),
            i if i % 2 == 0 => (-(1 << 25), 1 << 25),
            _ => (-(1 << 24), 1 << 24),
        };
        assert!(
            v[i] >= lo && v[i] < hi,
            "{what}: limb {i} = {} outside [{lo}, {hi})",
            v[i]
        );
    }
}

fn run_gpu(cases: &[FeTestIn]) -> Vec<FeTestOut> {
    let device = Device::system_default().expect("no Metal device");
    let source = format!("#define BATCH_INV_WINDOW 32\n{SHADER}{TEST_SUFFIX}");
    let library = device
        .new_library_with_source(&source, &CompileOptions::new())
        .expect("compile test shader");
    let function = library.get_function("fe_test_kernel", None).expect("kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("pipeline");
    let queue = device.new_command_queue();
    let in_buf = device.new_buffer_with_data(
        cases.as_ptr() as *const _,
        std::mem::size_of_val(cases) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_len = cases.len() * std::mem::size_of::<FeTestOut>();
    let out_buf = device.new_buffer(out_len as u64, MTLResourceOptions::StorageModeShared);
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&in_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.dispatch_threads(
        MTLSize { width: cases.len() as u64, height: 1, depth: 1 },
        MTLSize { width: 64, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let mut out = vec![FeTestOut::default(); cases.len()];
    unsafe {
        std::ptr::copy_nonoverlapping(
            out_buf.contents() as *const FeTestOut,
            out.as_mut_ptr(),
            cases.len(),
        );
    }
    out
}

fn case(op: u32, f: [i32; 10], g: [i32; 10]) -> FeTestIn {
    FeTestIn { f, g, op, pad: [0; 3] }
}

/// Expected field value of a case, from num-bigint.
fn expected(c: &FeTestIn) -> BigInt {
    let p = p();
    let f = limbs_value(&c.f);
    let g = limbs_value(&c.g);
    match c.op {
        OP_MUL => f * g,
        OP_SQ => &f * &f,
        OP_INVERT => {
            let fm = modp(&f);
            if fm.is_zero() {
                BigInt::zero()
            } else {
                let e = (p.clone() - BigInt::from(2)).to_biguint().unwrap();
                BigInt::from_biguint(Sign::Plus, fm.modpow(&e, &p.to_biguint().unwrap()))
            }
        }
        OP_MESH => {
            let k = &f * &g;
            let num = (BigInt::one() - &k) + (&g + &f);
            let den = (BigInt::one() + &k) - (&g - &f);
            num * den
        }
        OP_Z2T => {
            let z2 = &f + &f;
            (&z2 + &g) * (&z2 - &g)
        }
        _ => unreachable!(),
    }
}

fn check_cases(cases: &[FeTestIn], outs: &[FeTestOut]) {
    for (c, o) in cases.iter().zip(outs) {
        let want = canonical_bytes(&expected(c));
        let got_new = words_to_bytes(&o.new_bytes);
        let got_old = words_to_bytes(&o.old_bytes);
        assert_eq!(got_new, want, "new != bigint for {c:?}");
        assert_eq!(got_old, want, "old != bigint for {c:?} (reference broken?)");
        check_output_bounds(&o.new_limbs, &format!("op {} output", c.op));
        // Both chains apply the same single wrap of column 9, so the limb
        // vectors denote the same integer, not just the same field element.
        // (The limbs themselves may differ on ties: ref10's interleaved
        // order can leave h5 at exactly +2^24 where the sequential chain
        // gives -2^24 and h6 + 1.)
        assert_eq!(
            limbs_value(&o.new_limbs),
            limbs_value(&o.old_limbs),
            "integer value differs from ref10 for {c:?}"
        );
        match c.op {
            OP_MUL => assert_eq!(o.new_limbs, model_mul(&c.f, &c.g), "limbs != model for {c:?}"),
            OP_SQ => assert_eq!(o.new_limbs, model_sq(&c.f), "limbs != model for {c:?}"),
            _ => {}
        }
    }
}

#[test]
fn metal_fe_mul_sq_match_old_and_bigint_on_edges_and_bounds() {
    let edges = edge_values();
    let mut rng = rng();
    let mut cases = Vec::new();
    for f in &edges {
        cases.push(case(OP_SQ, *f, [0; 10]));
        for g in &edges {
            cases.push(case(OP_MUL, *f, *g));
        }
        for _ in 0..8 {
            cases.push(case(OP_MUL, *f, random_limbs_in_bounds(&mut rng)));
            cases.push(case(OP_MUL, random_limbs_in_bounds(&mut rng), *f));
        }
    }
    // Every sign pattern of all-extreme limbs against itself and random.
    for signs in 0..1024u32 {
        let f = extreme_limbs(signs);
        cases.push(case(OP_MUL, f, extreme_limbs(signs.reverse_bits() >> 22)));
        cases.push(case(OP_SQ, f, [0; 10]));
        cases.push(case(OP_MUL, f, random_limbs_in_bounds(&mut rng)));
    }
    let outs = run_gpu(&cases);
    check_cases(&cases, &outs);
}

#[test]
fn metal_fe_mul_sq_match_old_and_bigint_on_random_inputs() {
    let mut rng = rng();
    let mut cases = Vec::new();
    for _ in 0..20_000 {
        cases.push(case(OP_MUL, random_limbs_in_bounds(&mut rng), random_limbs_in_bounds(&mut rng)));
        cases.push(case(OP_MUL, random_canonical(&mut rng), random_canonical(&mut rng)));
        cases.push(case(OP_SQ, random_limbs_in_bounds(&mut rng), [0; 10]));
        cases.push(case(OP_SQ, random_canonical(&mut rng), [0; 10]));
    }
    let outs = run_gpu(&cases);
    check_cases(&cases, &outs);
}

#[test]
fn metal_fe_invert_matches_old_and_bigint() {
    let mut rng = rng();
    let mut cases = Vec::new();
    for f in edge_values() {
        cases.push(case(OP_INVERT, f, [0; 10]));
    }
    for _ in 0..2000 {
        cases.push(case(OP_INVERT, random_canonical(&mut rng), [0; 10]));
        cases.push(case(OP_INVERT, random_limbs_in_bounds(&mut rng), [0; 10]));
    }
    let outs = run_gpu(&cases);
    check_cases(&cases, &outs);
    // 0 and multiples of p invert to 0 on both implementations (z^(p-2)).
    let zero = canonical_bytes(&BigInt::zero());
    for (c, o) in cases.iter().zip(&outs) {
        if modp(&limbs_value(&c.f)).is_zero() {
            assert_eq!(words_to_bytes(&o.new_bytes), zero);
            assert_eq!(words_to_bytes(&o.old_bytes), zero);
        }
    }
}

/// The mesh walk's three-output sums and completed_to_extended's operand
/// patterns, on reduced (fe_mul-output-shaped) limbs as the kernels feed
/// them: exercises the add/sub-grown operands against the documented input
/// contract.
#[test]
fn metal_fe_composed_kernel_patterns_match_old_and_bigint() {
    let mut rng = rng();
    let mut cases = Vec::new();
    let mut reduced_edges: Vec<[i32; 10]> = vec![[0; 10], {
        let mut one = [0; 10];
        one[0] = 1;
        one
    }];
    // Extreme patterns at the fe_mul output bounds, including h1's excess.
    let mut top = [0i32; 10];
    let mut bot = [0i32; 10];
    for i in 0..10 {
        top[i] = if i % 2 == 0 { (1 << 25) - 1 } else { (1 << 24) - 1 };
        bot[i] = if i % 2 == 0 { -(1 << 25) } else { -(1 << 24) };
    }
    reduced_edges.push(top);
    reduced_edges.push(bot);
    let mut top1 = top;
    top1[1] = (1 << 24) + (1 << 17) - 1;
    reduced_edges.push(top1);
    let mut bot1 = bot;
    bot1[1] = -(1 << 24) - (1 << 17);
    reduced_edges.push(bot1);
    // p - 1 and 2^255 - 1 as fe_mul would output them (centered limbs).
    let mut one = [0i32; 10];
    one[0] = 1;
    reduced_edges.push(model_mul(&limbs_from_uint(&(p().to_biguint().unwrap() - BigUint::one())), &one));
    reduced_edges.push(model_mul(&limbs_from_uint(&((BigUint::one() << 255usize) - BigUint::one())), &one));
    for f in &reduced_edges {
        for g in &reduced_edges {
            cases.push(case(OP_MESH, *f, *g));
            cases.push(case(OP_Z2T, *f, *g));
        }
    }
    for _ in 0..10_000 {
        let f = random_reduced_limbs(&mut rng);
        let g = random_reduced_limbs(&mut rng);
        cases.push(case(OP_MESH, f, g));
        cases.push(case(OP_Z2T, f, g));
        cases.push(case(OP_MESH, random_canonical(&mut rng), random_canonical(&mut rng)));
    }
    let outs = run_gpu(&cases);
    check_cases(&cases, &outs);
}

/// The host model is the proof that the documented input bounds keep every
/// 64-bit intermediate in range: every product, column and carry target is
/// asserted to fit a long for the all-extreme inputs (worst case for the
/// column magnitudes) and for many random inputs at the bounds.
#[test]
fn fe_carry_model_never_overflows_at_documented_bounds() {
    let mut rng = rng();
    for signs in 0..1024u32 {
        let f = extreme_limbs(signs);
        for other in [0u32, 0x3ff, signs, !signs & 0x3ff] {
            let g = extreme_limbs(other);
            let r = model_mul(&f, &g);
            assert_eq!(canonical_bytes(&limbs_value(&r)), canonical_bytes(&(limbs_value(&f) * limbs_value(&g))));
            check_output_bounds(&r, "model mul");
        }
        let r = model_sq(&f);
        assert_eq!(canonical_bytes(&limbs_value(&r)), canonical_bytes(&(limbs_value(&f) * limbs_value(&f))));
        check_output_bounds(&r, "model sq");
    }
    for _ in 0..20_000 {
        let f = random_limbs_in_bounds(&mut rng);
        let g = random_limbs_in_bounds(&mut rng);
        let r = model_mul(&f, &g);
        assert_eq!(canonical_bytes(&limbs_value(&r)), canonical_bytes(&(limbs_value(&f) * limbs_value(&g))));
        check_output_bounds(&r, "model mul");
    }
    // The worst column magnitude under the contract is below 2^63 by a
    // margin that also covers the largest carry added into a column.
    let e = EVEN_MAX as i128;
    let o = ODD_MAX as i128;
    let worst_h0 = e * e * (1 + 4 * 19) + o * o * (5 * 38);
    assert!(worst_h0 + (1i128 << 50) + (1i128 << 36) < (1i128 << 63), "contract bound too loose: {worst_h0}");
    assert!(fits_int(19 * e) && fits_int(38 * o), "precompute bound too loose");
}
