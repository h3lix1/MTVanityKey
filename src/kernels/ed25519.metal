// MTVanityKey Metal kernels: the Ed25519 window walk shared by the MeshCore
// prefix search (search_kernel) and the Meshtastic node-ID search
// (mesh_kernel). Field arithmetic is ref10's radix-2^25.5 representation;
// the basepoint table (build.rs) and the affine step tables
// (src/gpu/ed_step_table.rs) are uploaded by the host.
#include <metal_stdlib>
using namespace metal;

struct SearchParams {
    uint count;
    uint prefix_nibbles;
    uint prefix_bytes;      // low 4 bytes (bytes 0..4)
    uint iters;
    uint prefix_bytes_hi;   // high 4 bytes (bytes 4..8)
    uint pad;
    // All 32 prefix bytes, little-endian per word. Only the low 8 bytes
    // (prefix_bytes / prefix_bytes_hi) are read on this backend's fast path
    // (prefix_nibbles <= 16); longer prefixes are re-checked on the CPU.
    uint prefix_full[8];
};

// Montgomery batch-inversion window: one fe_invert amortized over W candidates.
// The host prepends a #define; keep a default so the shader compiles standalone.
#ifndef BATCH_INV_WINDOW
#define BATCH_INV_WINDOW 32
#endif

// Hit slots per Meshtastic launch; overflow hits are dropped (the host
// re-derives and re-verifies everything anyway, and the host sizes the
// launch so that short patterns do not saturate the slots).
#ifndef MAX_HITS_PER_LAUNCH
#define MAX_HITS_PER_LAUNCH 64
#endif

// Field element in ref10 radix-2^25.5: ten signed limbs, even limbs hold 26
// bits and odd limbs hold 25 bits (h = sum h[i]*2^ceil(25.5*i)). This mirrors
// src/kernels/ed25519.cu exactly so both backends consume the same basepoint
// table (10 limbs * 30 words per Niels point) produced by build.rs.
struct Fe {
    int v[10];
};

struct Niels {
    Fe y_plus_x;
    Fe y_minus_x;
    Fe xy2d;
};

struct EPoint {
    Fe X;
    Fe Y;
    Fe Z;
    Fe T;
};

struct CPoint {
    Fe X;
    Fe Y;
    Fe Z;
    Fe T;
};

inline Fe fe_zero() {
    Fe h;
    for (uint i = 0; i < 10; i++) {
        h.v[i] = 0;
    }
    return h;
}

inline Fe fe_one() {
    Fe h = fe_zero();
    h.v[0] = 1;
    return h;
}

inline Fe fe_add(Fe f, Fe g) {
    Fe h;
    for (uint i = 0; i < 10; i++) {
        h.v[i] = f.v[i] + g.v[i];
    }
    return h;
}

inline Fe fe_sub(Fe f, Fe g) {
    Fe h;
    for (uint i = 0; i < 10; i++) {
        h.v[i] = f.v[i] - g.v[i];
    }
    return h;
}

// Carry/reduce the ten 64-bit product columns of fe_mul / fe_sq into ref10
// limbs: even limbs in [-2^25, 2^25), odd limbs in [-2^24, 2^24), except
// h1 in [-2^24 - 2^17, 2^24 + 2^17) (the wrap carry of the last step).
// These are ref10's rounded carries ((h + 2^25) >> 26, h -= c << 26),
// computed in a cheaper way for a 32-bit ALU:
//
// * One sequential pass 0 -> 9 plus the wrap of column 9 into column 0
//   (11 steps) instead of ref10's interleaved 12 steps that carry columns
//   0 and 4 twice. Carrying is exact (round((a + b) / m) = round(a / m) +
//   round((a mod m + b) / m)), so the result denotes the same integer;
//   only on a tie can a limb differ in representation (ref10's second
//   carry4 can leave h5 at exactly +2^24 where this gives -2^24, h6 + 1),
//   within the bounds above either way.
// * The carry is an arithmetic shift of the biased column; the limb is the
//   masked low bits minus the bias (floor + mask, no 64-bit subtract).
// * The bias 2^25 / 2^24 that turns the floor into a rounded carry is only
//   added explicitly to column 0. Every other column receives its bias
//   through the carry chain: column k is biased by 2^50 as well (a
//   32-bit add to its high word, no carry-out possible since |h| < 2^61),
//   so its outgoing carry is c_k + 2^50 / 2^w_k = c_k + 2^24 (k even,
//   w = 26) or c_k + 2^25 (k odd, w = 25), which is exactly column k+1's
//   bias; the 2^50 vanishes in the mask. Column 9 passes nothing on; the
//   wrap step biases r0 with a 32-bit add. That replaces ten 64-bit
//   constant adds (measured ~8% of fe_mul) with one.
//
// Input contract for fe_mul / fe_sq: |even limb| < 1.6 * 2^26 and
// |odd limb| < 1.6 * 2^25, so the int precomputes 19*g and 38*f fit and
// the worst column (h0 = f0g0 + 19*(f2g8 + f4g6 + f6g4 + f8g2) +
// 38*(f1g9 + f3g7 + f5g5 + f7g3 + f9g1), same weights for fe_sq) stays
// below 1275 * 2^50 < 2^60.4, every carry below 2^35, and nothing
// overflows a long. The kernels stay inside: fe_mul / fe_sq outputs are
// bounded as above, the basepoint table holds ref10-reduced limbs, and
// the largest operand ever formed is a sum of three such values
// (completed_to_extended: Z+Z+T*xy2d, < 1.5 * 2^26 + 3 * 2^17 even).
// Measured on an M1 Max: ~8% faster per fe_mul and ~12% per fe_sq than the
// ref10 chain (dependent-chain microbenchmark), +12% keys/s for both
// search_kernel and mesh_kernel (GPU-timestamped launches).
inline Fe fe_carry(long h0, long h1, long h2, long h3, long h4,
                   long h5, long h6, long h7, long h8, long h9) {
    const int m26 = (1 << 26) - 1;
    const int m25 = (1 << 25) - 1;
    const int b26 = 1 << 25;
    const int b25 = 1 << 24;
    const long pass = (long)1 << 50;
    h0 += pass + b26;
    h1 += pass; h2 += pass; h3 += pass; h4 += pass;
    h5 += pass; h6 += pass; h7 += pass; h8 += pass;
    long c;
    c = h0 >> 26; h1 += c; int r0 = ((int)(h0 & m26)) - b26;
    c = h1 >> 25; h2 += c; int r1 = ((int)(h1 & m25)) - b25;
    c = h2 >> 26; h3 += c; int r2 = ((int)(h2 & m26)) - b26;
    c = h3 >> 25; h4 += c; int r3 = ((int)(h3 & m25)) - b25;
    c = h4 >> 26; h5 += c; int r4 = ((int)(h4 & m26)) - b26;
    c = h5 >> 25; h6 += c; int r5 = ((int)(h5 & m25)) - b25;
    c = h6 >> 26; h7 += c; int r6 = ((int)(h6 & m26)) - b26;
    c = h7 >> 25; h8 += c; int r7 = ((int)(h7 & m25)) - b25;
    c = h8 >> 26; h9 += c; int r8 = ((int)(h8 & m26)) - b26;
    // |c| < 2^35 here, so h0b < 2^40: its carry fits an int, as does
    // r1 + c0.
    c = h9 >> 25; long h0b = (long)(r0 + b26) + c * 19; int r9 = ((int)(h9 & m25)) - b25;
    int c0 = (int)(h0b >> 26); r1 += c0; r0 = ((int)(h0b & m26)) - b26;

    Fe h;
    h.v[0] = r0;
    h.v[1] = r1;
    h.v[2] = r2;
    h.v[3] = r3;
    h.v[4] = r4;
    h.v[5] = r5;
    h.v[6] = r6;
    h.v[7] = r7;
    h.v[8] = r8;
    h.v[9] = r9;
    return h;
}

// ref10 fe_mul: h = f * g mod 2^255-19 (products as in ref10, reduction in
// fe_carry; see its contract for the input bounds).
inline Fe fe_mul(Fe f, Fe g) {
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

    return fe_carry(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9);
}

// ref10 fe_sq: h = f * f mod 2^255-19 (reduction in fe_carry).
inline Fe fe_sq(Fe f) {
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

    return fe_carry(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9);
}

inline Fe fe_sq_times(Fe f, uint n) {
    for (uint i = 0; i < n; i++) {
        f = fe_sq(f);
    }
    return f;
}

// z^(p-2) via the standard curve25519 ref10 addition chain
// (254 squarings + 11 multiplications).
inline Fe fe_invert(Fe z) {
    Fe t0 = fe_sq(z);
    Fe t1 = fe_sq_times(t0, 2);
    t1 = fe_mul(z, t1);
    t0 = fe_mul(t0, t1);
    Fe t2 = fe_sq(t0);
    t1 = fe_mul(t1, t2);
    t2 = fe_sq_times(t1, 5);
    t1 = fe_mul(t2, t1);
    t2 = fe_sq_times(t1, 10);
    t2 = fe_mul(t2, t1);
    Fe t3 = fe_sq_times(t2, 20);
    t2 = fe_mul(t3, t2);
    t2 = fe_sq_times(t2, 10);
    t1 = fe_mul(t2, t1);
    t2 = fe_sq_times(t1, 50);
    t2 = fe_mul(t2, t1);
    t3 = fe_sq_times(t2, 100);
    t2 = fe_mul(t3, t2);
    t2 = fe_sq_times(t2, 50);
    t1 = fe_mul(t2, t1);
    t1 = fe_sq_times(t1, 5);
    return fe_mul(t1, t0);
}

// ref10 fe_tobytes, truncated to the low 8 bytes of the canonical 32-byte
// little-endian encoding. Fully reduces h mod 2^255-19 first.
inline ulong fe_tobytes_prefix8(Fe input) {
    int h0 = input.v[0], h1 = input.v[1], h2 = input.v[2], h3 = input.v[3], h4 = input.v[4];
    int h5 = input.v[5], h6 = input.v[6], h7 = input.v[7], h8 = input.v[8], h9 = input.v[9];

    int q = (19 * h9 + (1 << 24)) >> 25;
    q = (h0 + q) >> 26;
    q = (h1 + q) >> 25;
    q = (h2 + q) >> 26;
    q = (h3 + q) >> 25;
    q = (h4 + q) >> 26;
    q = (h5 + q) >> 25;
    q = (h6 + q) >> 26;
    q = (h7 + q) >> 25;
    q = (h8 + q) >> 26;
    q = (h9 + q) >> 25;

    h0 += 19 * q;
    int carry0 = h0 >> 26; h1 += carry0; h0 -= carry0 << 26;
    int carry1 = h1 >> 25; h2 += carry1; h1 -= carry1 << 25;
    int carry2 = h2 >> 26; h3 += carry2; h2 -= carry2 << 26;
    int carry3 = h3 >> 25; h4 += carry3; h3 -= carry3 << 25;
    int carry4 = h4 >> 26; h5 += carry4; h4 -= carry4 << 26;
    int carry5 = h5 >> 25; h6 += carry5; h5 -= carry5 << 25;
    int carry6 = h6 >> 26; h7 += carry6; h6 -= carry6 << 26;
    int carry7 = h7 >> 25; h8 += carry7; h7 -= carry7 << 25;
    int carry8 = h8 >> 26; h9 += carry8; h8 -= carry8 << 26;
    int carry9 = h9 >> 25;                h9 -= carry9 << 25;

    // Bytes 0..7: all of h0 (26 bits), all of h1 (25 bits) and low 13 bits of h2.
    return static_cast<ulong>(static_cast<uint>(h0))
        | (static_cast<ulong>(static_cast<uint>(h1)) << 26)
        | (static_cast<ulong>(static_cast<uint>(h2)) << 51);
}

// Full ref10 fe_tobytes: canonical 32-byte little-endian encoding. Only used
// for the once-per-launch gid-0 debug write, so it stays out of the hot loop.
inline void fe_tobytes32(thread uchar* out, Fe input) {
    int h0 = input.v[0], h1 = input.v[1], h2 = input.v[2], h3 = input.v[3], h4 = input.v[4];
    int h5 = input.v[5], h6 = input.v[6], h7 = input.v[7], h8 = input.v[8], h9 = input.v[9];

    int q = (19 * h9 + (1 << 24)) >> 25;
    q = (h0 + q) >> 26;
    q = (h1 + q) >> 25;
    q = (h2 + q) >> 26;
    q = (h3 + q) >> 25;
    q = (h4 + q) >> 26;
    q = (h5 + q) >> 25;
    q = (h6 + q) >> 26;
    q = (h7 + q) >> 25;
    q = (h8 + q) >> 26;
    q = (h9 + q) >> 25;

    h0 += 19 * q;
    int carry0 = h0 >> 26; h1 += carry0; h0 -= carry0 << 26;
    int carry1 = h1 >> 25; h2 += carry1; h1 -= carry1 << 25;
    int carry2 = h2 >> 26; h3 += carry2; h2 -= carry2 << 26;
    int carry3 = h3 >> 25; h4 += carry3; h3 -= carry3 << 25;
    int carry4 = h4 >> 26; h5 += carry4; h4 -= carry4 << 26;
    int carry5 = h5 >> 25; h6 += carry5; h5 -= carry5 << 25;
    int carry6 = h6 >> 26; h7 += carry6; h6 -= carry6 << 26;
    int carry7 = h7 >> 25; h8 += carry7; h7 -= carry7 << 25;
    int carry8 = h8 >> 26; h9 += carry8; h8 -= carry8 << 26;
    int carry9 = h9 >> 25;                h9 -= carry9 << 25;

    out[0]  = static_cast<uchar>(h0 >> 0);
    out[1]  = static_cast<uchar>(h0 >> 8);
    out[2]  = static_cast<uchar>(h0 >> 16);
    out[3]  = static_cast<uchar>((h0 >> 24) | (h1 << 2));
    out[4]  = static_cast<uchar>(h1 >> 6);
    out[5]  = static_cast<uchar>(h1 >> 14);
    out[6]  = static_cast<uchar>((h1 >> 22) | (h2 << 3));
    out[7]  = static_cast<uchar>(h2 >> 5);
    out[8]  = static_cast<uchar>(h2 >> 13);
    out[9]  = static_cast<uchar>((h2 >> 21) | (h3 << 5));
    out[10] = static_cast<uchar>(h3 >> 3);
    out[11] = static_cast<uchar>(h3 >> 11);
    out[12] = static_cast<uchar>((h3 >> 19) | (h4 << 6));
    out[13] = static_cast<uchar>(h4 >> 2);
    out[14] = static_cast<uchar>(h4 >> 10);
    out[15] = static_cast<uchar>(h4 >> 18);
    out[16] = static_cast<uchar>(h5 >> 0);
    out[17] = static_cast<uchar>(h5 >> 8);
    out[18] = static_cast<uchar>(h5 >> 16);
    out[19] = static_cast<uchar>((h5 >> 24) | (h6 << 1));
    out[20] = static_cast<uchar>(h6 >> 7);
    out[21] = static_cast<uchar>(h6 >> 15);
    out[22] = static_cast<uchar>((h6 >> 23) | (h7 << 3));
    out[23] = static_cast<uchar>(h7 >> 5);
    out[24] = static_cast<uchar>(h7 >> 13);
    out[25] = static_cast<uchar>((h7 >> 21) | (h8 << 4));
    out[26] = static_cast<uchar>(h8 >> 4);
    out[27] = static_cast<uchar>(h8 >> 12);
    out[28] = static_cast<uchar>((h8 >> 20) | (h9 << 6));
    out[29] = static_cast<uchar>(h9 >> 2);
    out[30] = static_cast<uchar>(h9 >> 10);
    out[31] = static_cast<uchar>(h9 >> 18);
}

inline Fe fe_load_table(device const uint* table, uint offset) {
    Fe h;
    for (uint i = 0; i < 10; i++) {
        h.v[i] = static_cast<int>(table[offset + i]);
    }
    return h;
}

inline Niels load_niels(device const uint* table, uint digit_index, int digit) {
    uint abs_digit = digit < 0 ? static_cast<uint>(-digit) : static_cast<uint>(digit);
    uint point_index = digit_index * 32 + (abs_digit - 1);
    uint base = point_index * 3 * 10;

    Niels n;
    Fe y_plus_x = fe_load_table(table, base);
    Fe y_minus_x = fe_load_table(table, base + 10);
    Fe xy2d = fe_load_table(table, base + 20);

    if (digit < 0) {
        n.y_plus_x = y_minus_x;
        n.y_minus_x = y_plus_x;
        n.xy2d = fe_sub(fe_zero(), xy2d);
    } else {
        n.y_plus_x = y_plus_x;
        n.y_minus_x = y_minus_x;
        n.xy2d = xy2d;
    }

    return n;
}

inline EPoint extended_identity() {
    EPoint p;
    p.X = fe_zero();
    p.Y = fe_one();
    p.Z = fe_one();
    p.T = fe_zero();
    return p;
}

inline EPoint completed_to_extended(CPoint c) {
    EPoint p;
    p.X = fe_mul(c.X, c.T);
    p.Y = fe_mul(c.Y, c.Z);
    p.Z = fe_mul(c.Z, c.T);
    p.T = fe_mul(c.X, c.Y);
    return p;
}

inline EPoint add_niels(EPoint p, Niels n) {
    Fe y_plus_x = fe_add(p.Y, p.X);
    Fe y_minus_x = fe_sub(p.Y, p.X);
    Fe pp = fe_mul(y_plus_x, n.y_plus_x);
    Fe mm = fe_mul(y_minus_x, n.y_minus_x);
    Fe txy2d = fe_mul(p.T, n.xy2d);
    Fe z2 = fe_add(p.Z, p.Z);

    CPoint c;
    c.X = fe_sub(pp, mm);
    c.Y = fe_add(pp, mm);
    c.Z = fe_add(z2, txy2d);
    c.T = fe_sub(z2, txy2d);
    return completed_to_extended(c);
}

inline EPoint edwards_mul_base(device const uint* scalar, device const uint* table) {
    int digits[43];
    int carry = 0;
    for (uint i = 0; i < 43; i++) {
        uint bit_offset = i * 6;
        uint word_index = bit_offset >> 5;
        uint bit_index = bit_offset & 31;
        uint bit_buf = scalar[word_index] >> bit_index;
        if (bit_index > 26 && word_index < 7) {
            bit_buf |= scalar[word_index + 1] << (32 - bit_index);
        }

        int coef = carry + static_cast<int>(bit_buf & 63);
        carry = (coef + 32) >> 6;
        int digit = coef - (carry << 6);
        digits[i] = digit;
    }
    digits[42] += carry << 6;

    EPoint p = extended_identity();
    for (uint i = 0; i < 43; i++) {
        int digit = digits[i];
        if (digit != 0) {
            p = add_niels(p, load_niels(table, i, digit));
        }
    }

    return p;
}

inline bool matches_prefix(ulong y_prefix8, constant SearchParams& params) {
    uint first = static_cast<uint>(y_prefix8) & 0xFF;
    if (first == 0 || first == 0xFF) {
        return false;
    }

    ulong prefix_bytes = static_cast<ulong>(params.prefix_bytes)
        | (static_cast<ulong>(params.prefix_bytes_hi) << 32);

    // Only the low 8 bytes (16 hex characters) are compared here; the host
    // re-checks every candidate against the whole prefix.
    uint full_bytes = min(params.prefix_nibbles >> 1, 8u);
    for (uint i = 0; i < full_bytes; i++) {
        uint y_byte = static_cast<uint>(y_prefix8 >> (8 * i)) & 0xFF;
        uint prefix_byte = static_cast<uint>(prefix_bytes >> (8 * i)) & 0xFF;
        if (y_byte != prefix_byte) {
            return false;
        }
    }

    if ((params.prefix_nibbles & 1) != 0 && full_bytes < 8) {
        uint i = full_bytes;
        uint y_byte = static_cast<uint>(y_prefix8 >> (8 * i)) & 0xFF;
        uint prefix_byte = static_cast<uint>(prefix_bytes >> (8 * i)) & 0xFF;
        if ((y_byte & 0xF0) != (prefix_byte & 0xF0)) {
            return false;
        }
    }

    return true;
}

// Compress an affine point given Y, X and the shared Zinv (RFC 8032 / dalek
// compress()): canonical 32-byte y encoding with bit 255 set to parity of x.
inline void compress_point(thread uchar* out, Fe y, Fe x, Fe zinv) {
    fe_tobytes32(out, fe_mul(y, zinv));
    uchar x_bytes[32];
    fe_tobytes32(x_bytes, fe_mul(x, zinv));
    out[31] = static_cast<uchar>(out[31] | ((x_bytes[0] & 1) << 7));
}

// ---------------------------------------------------------------------------
// Symmetric affine window walk (shared by search_kernel and, in its
// Montgomery-u form, mesh_kernel).
//
// Each thread visits the candidates P + k*S (S = 8*B, k = 0..iters-1) in
// windows of W = BATCH_INV_WINDOW. Window w keeps its MIDDLE point
// M = P + (w*W + H)*S, H = W/2, in projective form (X:Y:Z) and derives
// candidate j of the window, j = 0..W-1, as M + (j-H)*S from a host-built
// affine table of i*S (x, y, 2d*x*y; entry i-1 for i = 1..H, entry H for
// W*S; buffer index 7 / 9). With A = X*x2, B = Y*y2, K = (X*Y)*(2d*x2*y2)
// the complete twisted-Edwards addition law (a = -1, d non-square: no
// exceptional inputs) gives both signs from the same three products:
//
//   y(M + i*S) = 2Z*(B + A) / (2Z^2 - K)
//   y(M - i*S) = 2Z*(B - A) / (2Z^2 + K)
//
// i.e. 1.5 fe_mul per candidate plus the 4 fe_mul of a Montgomery batch
// inversion (one fe_invert per window), no squarings: 5.5M against the
// x-only ladder's 5M+2S step plus 4M.
//
// The batch inversion runs as TWO independent product chains (the plus
// candidates on chain A, the minus candidates on chain B, inverted together
// as 1/(prodA*prodB)) so that each candidate costs half a dependent fe_mul
// on the thread's critical path instead of a whole one; the GPU hides the
// rest with other threads. The common factor 2Z is carried through the
// inversion (inv = 2Z / prod), so the first window runs straight off the
// extended point of edwards_mul_base + H*S without an inversion of its own;
// after the first window the base is affine (Z = 1). The base advances by
// W*S through the same products (x3 needs 1/(2Z^2 + K) too, so the update's
// denominator joins chain A as the product (2Z^2 - K)(2Z^2 + K)).
//
// Slots: 0 = j = H (the base), 2i-1 = j = H+i (chain A) and 2i = j = H-i
// (chain B) for i = 1..H-1, W-1 = j = 0 (i = H, chain B), W = base update
// (chain A). A short final window (iters not a multiple of W) computes all
// W candidates and reports only j < w.
// ---------------------------------------------------------------------------

#define WALK_HALF (BATCH_INV_WINDOW / 2)

struct StepEntry {
    Fe x;
    Fe y;
    Fe xy2d;
};

inline StepEntry load_step(constant uint* step_table, uint entry) {
    uint base = entry * 30;
    StepEntry s;
    for (uint i = 0; i < 10; i++) {
        s.x.v[i] = static_cast<int>(step_table[base + i]);
        s.y.v[i] = static_cast<int>(step_table[base + 10 + i]);
        s.xy2d.v[i] = static_cast<int>(step_table[base + 20 + i]);
    }
    return s;
}

// Table entry as a Niels point for add_niels (the one-off M = P + H*S).
inline Niels load_niels_step(constant uint* step_table, uint entry) {
    StepEntry s = load_step(step_table, entry);
    Niels n;
    n.y_plus_x = fe_add(s.y, s.x);
    n.y_minus_x = fe_sub(s.y, s.x);
    n.xy2d = s.xy2d;
    return n;
}

struct WindowUpdate {
    Fe num_x;   // Y*x2 + X*y2
    Fe num_y;   // Y*y2 + X*x2
    Fe dm;      // 2Z^2 - K
    Fe dp;      // 2Z^2 + K
};

struct WindowChains {
    Fe inv_a;
    Fe inv_b;
};

// Forward half of a window: ts[e] = numerator * (chain product before slot
// e), ds[e] = denominator, in slot order; when `more`, slot W holds the
// base update. Returns the two chains' inverses (with the 2Z factor).
inline WindowChains window_forward(
    thread Fe* ts,
    thread Fe* ds,
    Fe BX, Fe BY, Fe BZ,
    bool more,
    constant uint* step_table,
    thread WindowUpdate& upd
) {
    Fe z2 = fe_sq(BZ);
    Fe z2x2 = fe_add(z2, z2);
    Fe xy = fe_mul(BX, BY);

    // Slot 0: the base, y = Y/Z = 2Z*Y / (2Z^2).
    ts[0] = BY;
    ds[0] = z2x2;
    Fe prod_a = z2x2;
    Fe prod_b = fe_one();
    for (uint i = 1; i <= WALK_HALF; i++) {
        StepEntry s = load_step(step_table, i - 1);
        Fe a = fe_mul(BX, s.x);
        Fe b = fe_mul(BY, s.y);
        Fe kk = fe_mul(xy, s.xy2d);
        Fe num_p = fe_add(b, a);
        Fe num_m = fe_sub(b, a);
        Fe den_p = fe_sub(z2x2, kk);
        Fe den_m = fe_add(z2x2, kk);
        uint e_m = (i < WALK_HALF) ? 2 * i : BATCH_INV_WINDOW - 1;
        ds[e_m] = den_m;
        ts[e_m] = (i == 1) ? num_m : fe_mul(num_m, prod_b);
        prod_b = (i == 1) ? den_m : fe_mul(prod_b, den_m);
        if (i < WALK_HALF) {
            uint e_p = 2 * i - 1;
            ds[e_p] = den_p;
            ts[e_p] = fe_mul(num_p, prod_a);
            prod_a = fe_mul(prod_a, den_p);
        }
    }
    if (more) {
        StepEntry s = load_step(step_table, WALK_HALF);
        Fe a = fe_mul(BX, s.x);
        Fe b = fe_mul(BY, s.y);
        Fe kk = fe_mul(xy, s.xy2d);
        Fe pp = fe_mul(fe_add(BY, BX), fe_add(s.y, s.x));
        upd.num_x = fe_sub(fe_sub(pp, a), b);
        upd.num_y = fe_add(b, a);
        upd.dm = fe_sub(z2x2, kk);
        upd.dp = fe_add(z2x2, kk);
        Fe uden = fe_mul(upd.dm, upd.dp);
        ts[BATCH_INV_WINDOW] = prod_a;
        ds[BATCH_INV_WINDOW] = uden;
        prod_a = fe_mul(prod_a, uden);
    }
    Fe inv = fe_mul(fe_add(BZ, BZ), fe_invert(fe_mul(prod_a, prod_b)));
    WindowChains chains;
    chains.inv_a = fe_mul(inv, prod_b);
    chains.inv_b = fe_mul(inv, prod_a);
    return chains;
}

// Advances the base to M + W*S (affine) from slot W and chain A's value
// inv_a = 2Z/prod_a; returns chain A's value for its next slot.
inline Fe window_advance(
    thread Fe* ts,
    thread Fe* ds,
    thread Fe& BX, thread Fe& BY, thread Fe& BZ,
    Fe inv_a,
    thread const WindowUpdate& upd
) {
    Fe iu = fe_mul(ts[BATCH_INV_WINDOW], inv_a);  // 2Z / ((2Z^2 - K)(2Z^2 + K))
    BX = fe_mul(upd.num_x, fe_mul(upd.dm, iu));
    BY = fe_mul(upd.num_y, fe_mul(upd.dp, iu));
    BZ = fe_one();
    return fe_mul(inv_a, ds[BATCH_INV_WINDOW]);
}

// Backward half of a window: visits every candidate's affine y (chain B
// slots and chain A slots interleaved, pair by pair). WALK_CHECK(cand, idx) is
// the kernel's per-candidate test; k is the window's first candidate.
#define WALK_BACKWARD(ts, ds, chains, w)                                      \
    {                                                                        \
        Fe inv_a = chains.inv_a;                                             \
        Fe inv_b = chains.inv_b;                                             \
        for (uint i = WALK_HALF; i >= 1; i--) {                              \
            uint e_m = (i < WALK_HALF) ? 2 * i : BATCH_INV_WINDOW - 1;       \
            Fe y_m = fe_mul(ts[e_m], inv_b);                                 \
            inv_b = fe_mul(inv_b, ds[e_m]);                                  \
            if (WALK_HALF - i < (w)) {                                       \
                WALK_CHECK(y_m, WALK_HALF - i);                              \
            }                                                                \
            if (i < WALK_HALF) {                                             \
                uint e_p = 2 * i - 1;                                        \
                Fe y_p = fe_mul(ts[e_p], inv_a);                             \
                inv_a = fe_mul(inv_a, ds[e_p]);                              \
                if (WALK_HALF + i < (w)) {                                   \
                    WALK_CHECK(y_p, WALK_HALF + i);                          \
                }                                                            \
            }                                                                \
        }                                                                    \
        Fe y_0 = fe_mul(ts[0], inv_a);                                       \
        if (WALK_HALF < (w)) {                                               \
            WALK_CHECK(y_0, WALK_HALF);                                      \
        }                                                                    \
    }

// Each thread computes ONE full scalar mult P = base*B, then visits the
// candidates P + i*8*B (scalar += 8 preserves Ed25519 clamping) window by
// window through the affine step table (see window_forward); the affine y
// needed for prefix matching comes out of one Montgomery batch inversion per
// window of BATCH_INV_WINDOW candidates.
kernel void search_kernel(
    device const uint* scalars [[buffer(0)]],
    constant SearchParams& params [[buffer(1)]],
    device atomic_uint* found [[buffer(2)]],
    device uint* found_index [[buffer(3)]],
    device uint* found_iter [[buffer(4)]],
    device uint* debug_prefix [[buffer(5)]],
    device const uint* basepoint_table [[buffer(6)]],
    constant uint* step_table [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) {
        return;
    }

    device const uint* scalar = scalars + gid * 8;
    EPoint p = edwards_mul_base(scalar, basepoint_table);

    // Full compressed point of gid 0's first candidate, written once per launch
    // as eight little-endian words; the host compares it byte-for-byte against
    // dalek's compress() in tests.
    if (gid == 0) {
        uchar enc[32];
        compress_point(enc, p.Y, p.X, fe_invert(p.Z));
        for (uint i = 0; i < 8; i++) {
            debug_prefix[i] = static_cast<uint>(enc[4 * i])
                | (static_cast<uint>(enc[4 * i + 1]) << 8)
                | (static_cast<uint>(enc[4 * i + 2]) << 16)
                | (static_cast<uint>(enc[4 * i + 3]) << 24);
        }
    }

    // Early out if another thread already found a match this launch.
    if (atomic_load_explicit(found, memory_order_relaxed) != 0) {
        return;
    }

    // Window middle M = P + H*S in projective form (affine after the first
    // window update).
    EPoint m = add_niels(p, load_niels_step(step_table, WALK_HALF - 1));
    Fe BX = m.X;
    Fe BY = m.Y;
    Fe BZ = m.Z;

    const uint iters = params.iters;
    Fe ts[BATCH_INV_WINDOW + 1];
    Fe ds[BATCH_INV_WINDOW + 1];

    for (uint k = 0; k < iters; k += BATCH_INV_WINDOW) {
        uint w = iters - k;
        if (w > BATCH_INV_WINDOW) {
            w = BATCH_INV_WINDOW;
        }
        const bool more = (k + w) < iters;

        WindowUpdate upd;
        WindowChains chains = window_forward(ts, ds, BX, BY, BZ, more, step_table, upd);
        if (more) {
            chains.inv_a = window_advance(ts, ds, BX, BY, BZ, chains.inv_a, upd);
        }
#define WALK_CHECK(cand, idx)                                                \
        {                                                                    \
            ulong y_prefix8 = fe_tobytes_prefix8(cand);                      \
            if (matches_prefix(y_prefix8, params)) {                         \
                uint expected = 0;                                           \
                if (atomic_compare_exchange_weak_explicit(                   \
                        found, &expected, 1,                                 \
                        memory_order_relaxed, memory_order_relaxed)) {       \
                    *found_index = gid;                                      \
                    *found_iter = k + (idx);                                 \
                }                                                            \
            }                                                                \
        }
        WALK_BACKWARD(ts, ds, chains, w)
#undef WALK_CHECK
    }
}

// ---------------------------------------------------------------------------
// Meshtastic node-ID mode: the same window walk, but each candidate's
// Montgomery u (its X25519 public key) is encoded, CRC-32'd and tested
// against the masked node-number targets. See src/meshtastic.rs for the
// firmware derivation.
//
// With the window middle M affine (Z = 1, see mesh_kernel) and the mesh step
// table carrying d*x*y (so K = (X*Y)*(d*x_i*y_i) is HALF the vanity walk's
// K), the complete addition law reads
//
//   y(M + iS) = (B + A) / (1 - K),   y(M - iS) = (B - A) / (1 + K)
//
// and the birational map to Curve25519, u = (1 + y) / (1 - y) = (d + n) /
// (d - n) for y = n / d, costs nothing extra: the same three products per
// pair, with the batch inversion running over d - n instead of d. Every
// numerator and denominator is 1 -+ K +- B +- A, a sum of three fe_mul
// outputs (|even limb| <= 3 * 2^25 + 1 < 1.6 * 2^26, |odd| <= 3 * 2^24 +
// 2^17 < 1.6 * 2^25), inside fe_carry's input contract. (The vanity walk's
// 2d table would put five outputs in one operand and overflow the int
// 19*g precomputes: measured as ~17% wrong candidates.)
// ---------------------------------------------------------------------------

struct MeshParams {
    uint count;
    uint iters;
    uint mask;
    uint target_count;
    uint targets[64];
};

// Table buffer: four CRC-32 slicing tables (T0..T3, 256 words each) then
// the 4096-bit target bitmap; staged into threadgroup memory per launch.
#define MESH_CRC_WORDS 1024
#define MESH_BITMAP_WORDS 128
#define MESH_TABLE_WORDS (MESH_CRC_WORDS + MESH_BITMAP_WORDS)

// 1: slicing-by-4 (eight rounds of four independent lookups); 0: bytewise
// (32 dependent lookups through T0).
#ifndef MESH_CRC_SLICE4
#define MESH_CRC_SLICE4 1
#endif

// Full ref10 fe_tobytes as eight little-endian 32-bit words (the canonical
// encoding of h mod 2^255-19, bit 255 clear).
inline void fe_tobytes_words8(thread uint* out, Fe input) {
    int h0 = input.v[0], h1 = input.v[1], h2 = input.v[2], h3 = input.v[3], h4 = input.v[4];
    int h5 = input.v[5], h6 = input.v[6], h7 = input.v[7], h8 = input.v[8], h9 = input.v[9];

    int q = (19 * h9 + (1 << 24)) >> 25;
    q = (h0 + q) >> 26;
    q = (h1 + q) >> 25;
    q = (h2 + q) >> 26;
    q = (h3 + q) >> 25;
    q = (h4 + q) >> 26;
    q = (h5 + q) >> 25;
    q = (h6 + q) >> 26;
    q = (h7 + q) >> 25;
    q = (h8 + q) >> 26;
    q = (h9 + q) >> 25;

    h0 += 19 * q;
    int carry0 = h0 >> 26; h1 += carry0; h0 -= carry0 << 26;
    int carry1 = h1 >> 25; h2 += carry1; h1 -= carry1 << 25;
    int carry2 = h2 >> 26; h3 += carry2; h2 -= carry2 << 26;
    int carry3 = h3 >> 25; h4 += carry3; h3 -= carry3 << 25;
    int carry4 = h4 >> 26; h5 += carry4; h4 -= carry4 << 26;
    int carry5 = h5 >> 25; h6 += carry5; h5 -= carry5 << 25;
    int carry6 = h6 >> 26; h7 += carry6; h6 -= carry6 << 26;
    int carry7 = h7 >> 25; h8 += carry7; h7 -= carry7 << 25;
    int carry8 = h8 >> 26; h9 += carry8; h8 -= carry8 << 26;
    int carry9 = h9 >> 25;                h9 -= carry9 << 25;

    // Limb k starts at bit 0, 26, 51, 77, 102, 128, 153, 179, 204, 230.
    uint u0 = (uint)h0, u1 = (uint)h1, u2 = (uint)h2, u3 = (uint)h3, u4 = (uint)h4;
    uint u5 = (uint)h5, u6 = (uint)h6, u7 = (uint)h7, u8 = (uint)h8, u9 = (uint)h9;
    out[0] = u0 | (u1 << 26);
    out[1] = (u1 >> 6) | (u2 << 19);
    out[2] = (u2 >> 13) | (u3 << 13);
    out[3] = (u3 >> 19) | (u4 << 6);
    out[4] = u5 | (u6 << 25);
    out[5] = (u6 >> 7) | (u7 << 19);
    out[6] = (u7 >> 13) | (u8 << 12);
    out[7] = (u8 >> 20) | (u9 << 6);
}

// Standard reflected CRC-32 (poly 0xEDB88320, init/xorout 0xFFFFFFFF) of
// 32 bytes given as eight little-endian words: crc32Buffer of the firmware.
inline uint crc32_words8(thread const uint* w, threadgroup const uint* tab) {
    uint crc = 0xFFFFFFFFu;
    for (uint i = 0; i < 8; i++) {
        crc ^= w[i];
#if MESH_CRC_SLICE4
        crc = tab[768 + (crc & 0xFFu)]
            ^ tab[512 + ((crc >> 8) & 0xFFu)]
            ^ tab[256 + ((crc >> 16) & 0xFFu)]
            ^ tab[crc >> 24];
#else
        for (uint b = 0; b < 4; b++) {
            crc = tab[crc & 0xFFu] ^ (crc >> 8);
        }
#endif
    }
    return ~crc;
}

// Table entry of the mesh step table (x, y, d*x*y) as a Niels point for the
// one-off M = P + H*S: add_niels wants 2d*x*y.
inline Niels load_niels_step_dxy(constant uint* step_table, uint entry) {
    StepEntry s = load_step(step_table, entry);
    Niels n;
    n.y_plus_x = fe_add(s.y, s.x);
    n.y_minus_x = fe_sub(s.y, s.x);
    n.xy2d = fe_add(s.xy2d, s.xy2d);
    return n;
}

// Exact membership (after the bitmap): the firmware's reserved numbers are
// never a hit, whatever the mask.
inline bool mesh_is_target(uint masked, uint id, constant MeshParams& params) {
    if (id < 4u || id == 0xFFFFFFFFu) {
        return false;
    }
    for (uint i = 0; i < params.target_count; i++) {
        if (params.targets[i] == masked) {
            return true;
        }
    }
    return false;
}

// Forward half of a window for the u walk (base affine, Z = 1): slot e
// holds u's numerator times the chain product before it and u's
// denominator d - n. Slot 0 is the base itself, u = (1 + y) / (1 - y).
inline WindowChains window_forward_u(
    thread Fe* ts,
    thread Fe* ds,
    Fe BX, Fe BY,
    bool more,
    constant uint* step_table,
    thread WindowUpdate& upd
) {
    const Fe one = fe_one();
    Fe xy = fe_mul(BX, BY);

    ts[0] = fe_add(one, BY);
    ds[0] = fe_sub(one, BY);
    Fe prod_a = ds[0];
    Fe prod_b = fe_one();
    for (uint i = 1; i <= WALK_HALF; i++) {
        StepEntry s = load_step(step_table, i - 1);   // (x, y, d*x*y) of i*S
        Fe a = fe_mul(BX, s.x);
        Fe b = fe_mul(BY, s.y);
        Fe kk = fe_mul(xy, s.xy2d);      // K = X*Y*d*x*y
        Fe np = fe_add(b, a);            // y(M+iS) = np / dp
        Fe nm = fe_sub(b, a);            // y(M-iS) = nm / dm
        Fe dp = fe_sub(one, kk);
        Fe dm = fe_add(one, kk);
        Fe up_num = fe_add(dp, np);      // u(M+iS) = (dp + np) / (dp - np)
        Fe up_den = fe_sub(dp, np);
        Fe um_num = fe_add(dm, nm);
        Fe um_den = fe_sub(dm, nm);
        uint e_m = (i < WALK_HALF) ? 2 * i : BATCH_INV_WINDOW - 1;
        ds[e_m] = um_den;
        ts[e_m] = (i == 1) ? um_num : fe_mul(um_num, prod_b);
        prod_b = (i == 1) ? um_den : fe_mul(prod_b, um_den);
        if (i < WALK_HALF) {
            uint e_p = 2 * i - 1;
            ds[e_p] = up_den;
            ts[e_p] = fe_mul(up_num, prod_a);
            prod_a = fe_mul(prod_a, up_den);
        }
    }
    if (more) {
        StepEntry s = load_step(step_table, WALK_HALF);
        Fe a = fe_mul(BX, s.x);
        Fe b = fe_mul(BY, s.y);
        Fe kk = fe_mul(xy, s.xy2d);
        // x(M+WS) = (Y x + X y) / (1 + K), y(M+WS) = (B + A) / (1 - K).
        upd.num_x = fe_add(fe_mul(BY, s.x), fe_mul(BX, s.y));
        upd.num_y = fe_add(b, a);
        upd.dm = fe_sub(one, kk);
        upd.dp = fe_add(one, kk);
        Fe uden = fe_mul(upd.dm, upd.dp);
        ts[BATCH_INV_WINDOW] = prod_a;
        ds[BATCH_INV_WINDOW] = uden;
        prod_a = fe_mul(prod_a, uden);
    }
    Fe inv = fe_invert(fe_mul(prod_a, prod_b));
    WindowChains chains;
    chains.inv_a = fe_mul(inv, prod_b);
    chains.inv_b = fe_mul(inv, prod_a);
    return chains;
}

// Advances the affine base to M + W*S from slot W and chain A's inverse;
// returns chain A's value for its next slot.
inline Fe window_advance_u(
    thread Fe* ts,
    thread Fe* ds,
    thread Fe& BX, thread Fe& BY,
    Fe inv_a,
    thread const WindowUpdate& upd
) {
    Fe iu = fe_mul(ts[BATCH_INV_WINDOW], inv_a);  // 1 / (dm * dp)
    BX = fe_mul(upd.num_x, fe_mul(upd.dm, iu));   // num_x / dp
    BY = fe_mul(upd.num_y, fe_mul(upd.dp, iu));   // num_y / dm
    return fe_mul(inv_a, ds[BATCH_INV_WINDOW]);
}

// One full scalar mult per thread, then `iters` candidates base + 8k through
// the u-walk windows; each candidate's X25519 public key is CRC-32'd and
// tested against the targets. Hits (gid, iter) go to the slot arrays; the
// host re-derives and verifies every one.
kernel void mesh_kernel(
    device const uint* scalars [[buffer(0)]],
    constant MeshParams& params [[buffer(1)]],
    device const uint* mesh_tables [[buffer(2)]],
    device uint* found_count [[buffer(3)]],
    device uint* found_index [[buffer(4)]],
    device uint* found_iter [[buffer(5)]],
    device uint* debug_u [[buffer(6)]],
    device const uint* basepoint_table [[buffer(7)]],
    constant uint* step_table [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Stage the CRC tables and the target bitmap (4.5 KB) into threadgroup
    // memory before any thread can return: every thread takes part in the
    // barrier.
    threadgroup uint tab[MESH_TABLE_WORDS];
    for (uint i = lid; i < MESH_TABLE_WORDS; i += tg_size) {
        tab[i] = mesh_tables[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid >= params.count) {
        return;
    }

    device const uint* scalar = scalars + gid * 8;
    EPoint p = edwards_mul_base(scalar, basepoint_table);

    // X25519 public key (u = (Z + Y) / (Z - Y)) of gid 0's first candidate,
    // compared byte-for-byte against dalek on the host in tests.
    if (gid == 0) {
        Fe u = fe_mul(fe_add(p.Z, p.Y), fe_invert(fe_sub(p.Z, p.Y)));
        uint uw[8];
        fe_tobytes_words8(uw, u);
        for (uint i = 0; i < 8; i++) {
            debug_u[i] = uw[i];
        }
    }

    // Window middle M = P + H*S, made affine once: the u chains below assume
    // Z = 1 (one inversion per thread per launch).
    EPoint m = add_niels(p, load_niels_step_dxy(step_table, WALK_HALF - 1));
    Fe zinv = fe_invert(m.Z);
    Fe BX = fe_mul(m.X, zinv);
    Fe BY = fe_mul(m.Y, zinv);

    const uint iters = params.iters;
    const uint mask = params.mask;
    Fe ts[BATCH_INV_WINDOW + 1];
    Fe ds[BATCH_INV_WINDOW + 1];

    for (uint k = 0; k < iters; k += BATCH_INV_WINDOW) {
        uint w = iters - k;
        if (w > BATCH_INV_WINDOW) {
            w = BATCH_INV_WINDOW;
        }
        const bool more = (k + w) < iters;

        WindowUpdate upd;
        WindowChains chains = window_forward_u(ts, ds, BX, BY, more, step_table, upd);
        if (more) {
            chains.inv_a = window_advance_u(ts, ds, BX, BY, chains.inv_a, upd);
        }
#define WALK_CHECK(cand, idx)                                                \
        {                                                                    \
            uint uw[8];                                                      \
            fe_tobytes_words8(uw, cand);                                     \
            uint id = crc32_words8(uw, tab);                                 \
            uint masked = id & mask;                                         \
            uint bit = (masked * 0x9E3779B1u) >> 20;                         \
            if ((tab[MESH_CRC_WORDS + (bit >> 5)] >> (bit & 31u)) & 1u) {   \
                if (mesh_is_target(masked, id, params)) {                    \
                    uint slot = atomic_fetch_add_explicit(                   \
                        (device atomic_uint*)found_count, 1u,                \
                        memory_order_relaxed);                               \
                    if (slot < MAX_HITS_PER_LAUNCH) {                        \
                        found_index[slot] = gid;                             \
                        found_iter[slot] = k + (idx);                        \
                    }                                                        \
                }                                                            \
            }                                                                \
        }
        WALK_BACKWARD(ts, ds, chains, w)
#undef WALK_CHECK
    }
}
