// MeshVanityKey CUDA kernels: the Ed25519 window walk shared by the MeshCore
// prefix search (search_kernel) and the Meshtastic node-ID search
// (mesh_kernel). Mirrors src/kernels/ed25519.metal; compiled by build.rs
// with nvcc, or as host C++ through tests/cuda_host/shim.h for the
// differential tests.
#include <stdint.h>

struct SearchParams {
    uint32_t count;
    uint32_t prefix_nibbles;
    uint32_t prefix_bytes;     // low 4 bytes (bytes 0..4)
    uint32_t iters;
    uint32_t prefix_bytes_hi;  // high 4 bytes (bytes 4..8)
    uint32_t pad;
    // All 32 prefix bytes, little-endian per word (byte i in word i/4 at bit
    // 8*(i%4)); unused by this kernel (the fast path reads prefix_bytes /
    // prefix_bytes_hi). Must stay in lockstep with SearchParams in src/gpu.rs.
    uint32_t prefix_full[8];
};

// Montgomery batch-inversion window: one fe_invert amortized over W candidates.
#ifndef BATCH_INV_WINDOW
#define BATCH_INV_WINDOW 256
#endif

// Launch-bounds knobs, overridable at compile time (see build.rs) so the
// occupancy/register trade-off can be swept empirically.
#ifndef LB_THREADS
#define LB_THREADS 256
#endif
#ifndef LB_MIN_BLOCKS
#define LB_MIN_BLOCKS 1
#endif
#if LB_MIN_BLOCKS > 1
#define MTVK_LAUNCH_BOUNDS __launch_bounds__(LB_THREADS, LB_MIN_BLOCKS)
#else
#define MTVK_LAUNCH_BOUNDS __launch_bounds__(LB_THREADS)
#endif

// Field element in ref10 radix-2^25.5: ten signed limbs, even limbs hold 26
// bits and odd limbs hold 25 bits (h = sum h[i]*2^ceil(25.5*i)).
struct Fe {
    int32_t v[10];
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

__device__ __forceinline__ Fe fe_zero() {
    Fe h;
    for (uint32_t i = 0; i < 10; i++) {
        h.v[i] = 0;
    }
    return h;
}

__device__ __forceinline__ Fe fe_one() {
    Fe h = fe_zero();
    h.v[0] = 1;
    return h;
}

__device__ __forceinline__ Fe fe_add(Fe f, Fe g) {
    Fe h;
    for (uint32_t i = 0; i < 10; i++) {
        h.v[i] = f.v[i] + g.v[i];
    }
    return h;
}

__device__ __forceinline__ Fe fe_sub(Fe f, Fe g) {
    Fe h;
    for (uint32_t i = 0; i < 10; i++) {
        h.v[i] = f.v[i] - g.v[i];
    }
    return h;
}

// ---------------------------------------------------------------------------
// 32x32->64 multiply-accumulate primitives.
//
// Every product in fe_mul / fe_sq is a signed 32-bit limb times a signed
// 32-bit limb accumulated into a 64-bit column. On NVIDIA that is exactly one
// IMAD.WIDE; written as plain C++ `(int64_t)a * b` the optimizer can
// re-associate the sign extensions of limbs that were just truncated from a
// 64-bit column (sext(trunc(x)) -> shl/ashr) and then lose the mul.wide
// pattern, emitting an emulated 64-bit multiply (three IMADs plus adds) per
// product instead. Inline PTX pins the mul.wide / mad.wide form; ptxas still
// schedules, folds immediates and allocates registers across these as usual.
// The host-C++ build (tests/cuda_host) uses the plain-C definitions, which
// compute the same wrapping 64-bit values.
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__)
__device__ __forceinline__ int64_t mulw(int32_t a, int32_t b) {
    int64_t r;
    asm("mul.wide.s32 %0, %1, %2;" : "=l"(r) : "r"(a), "r"(b));
    return r;
}
__device__ __forceinline__ int64_t madw(int32_t a, int32_t b, int64_t c) {
    int64_t r;
    asm("mad.wide.s32 %0, %1, %2, %3;" : "=l"(r) : "r"(a), "r"(b), "l"(c));
    return r;
}
#else
__device__ __forceinline__ int64_t mulw(int32_t a, int32_t b) {
    return static_cast<int64_t>(a) * static_cast<int64_t>(b);
}
__device__ __forceinline__ int64_t madw(int32_t a, int32_t b, int64_t c) {
    return static_cast<int64_t>(static_cast<uint64_t>(static_cast<int64_t>(a) * static_cast<int64_t>(b))
        + static_cast<uint64_t>(c));
}
#endif

// ref10 carry step, h = h_low + carry * 2^bits with h_low in
// [-2^(bits-1), 2^(bits-1)), i.e. carry = (h + 2^(bits-1)) >> bits and
// h_low = h - (carry << bits). The low limb is formed from the low 32 bits
// of t = h + 2^(bits-1) alone: h - (carry << bits) = (t mod 2^bits) -
// 2^(bits-1), and t mod 2^bits is t's low `bits` bits in two's complement.
// Exact for every int64 h (no bound needed); it just avoids the 64-bit
// shift/subtract ref10 spends on the low half.
__device__ __forceinline__ int64_t carry_step26(int64_t h, int32_t& low) {
    int64_t t = h + static_cast<int64_t>(1 << 25);
    low = static_cast<int32_t>(static_cast<uint32_t>(t) & 0x3FFFFFFu) - (1 << 25);
    return t >> 26;
}
__device__ __forceinline__ int64_t carry_step25(int64_t h, int32_t& low) {
    int64_t t = h + static_cast<int64_t>(1 << 24);
    low = static_cast<int32_t>(static_cast<uint32_t>(t) & 0x1FFFFFFu) - (1 << 24);
    return t >> 25;
}

// The ref10 carry chain over ten 64-bit columns (same order as ref10 fe_mul /
// fe_sq: 0,4,1,5,2,6,3,7,4,8,9,0), producing the ten reduced int32 limbs.
// The two second-round carries (4 again, then 0 again after the 19*carry9
// fold) act on values that are provably small for ANY int64 columns:
//   h4' = low4 + carry3, |low4| <= 2^25, |carry3| < 2^39  -> carry4' < 2^14,
//   h0' = low0 + 19*carry9, |carry9| < 2^39                -> carry0' < 2^18,
// so they fit int32 and the adds into low5 / low1 (|.| <= 2^24) cannot
// overflow int32. Bit-for-bit the same limbs as ref10's all-int64 chain.
__device__ __forceinline__ Fe fe_carry_chain(
    int64_t h0, int64_t h1, int64_t h2, int64_t h3, int64_t h4,
    int64_t h5, int64_t h6, int64_t h7, int64_t h8, int64_t h9
) {
    int32_t l0, l1, l2, l3, l4, l5, l6, l7, l8, l9;
    int64_t carry;

    carry = carry_step26(h0, l0); h1 += carry;
    carry = carry_step26(h4, l4); h5 += carry;
    carry = carry_step25(h1, l1); h2 += carry;
    carry = carry_step25(h5, l5); h6 += carry;
    carry = carry_step26(h2, l2); h3 += carry;
    carry = carry_step26(h6, l6); h7 += carry;
    int64_t carry3 = carry_step25(h3, l3);
    carry = carry_step25(h7, l7); h8 += carry;
    // Second carry4: h4' = l4 + carry3.
    {
        int64_t t = static_cast<int64_t>(l4 + (1 << 25)) + carry3;
        l4 = static_cast<int32_t>(static_cast<uint32_t>(t) & 0x3FFFFFFu) - (1 << 25);
        l5 += static_cast<int32_t>(t >> 26);
    }
    carry = carry_step26(h8, l8); h9 += carry;
    int64_t carry9 = carry_step25(h9, l9);
    // Second carry0: h0' = l0 + 19 * carry9.
    {
        int64_t t = static_cast<int64_t>(l0 + (1 << 25)) + carry9 * 19;
        l0 = static_cast<int32_t>(static_cast<uint32_t>(t) & 0x3FFFFFFu) - (1 << 25);
        l1 += static_cast<int32_t>(t >> 26);
    }

    Fe h;
    h.v[0] = l0; h.v[1] = l1; h.v[2] = l2; h.v[3] = l3; h.v[4] = l4;
    h.v[5] = l5; h.v[6] = l6; h.v[7] = l7; h.v[8] = l8; h.v[9] = l9;
    return h;
}

// ref10 fe_mul: h = f * g mod 2^255-19.
// |f|, |g| limbs bounded by 1.65*2^26 (even) / 1.65*2^25 (odd);
// output limbs bounded by ~1.01*2^25 (even) / ~1.01*2^24 (odd).
// Same products and column sums as ref10 (each column is one mad.wide
// chain), then the ref10 carry chain.
__device__ __forceinline__ Fe fe_mul(Fe f, Fe g) {
    int32_t f0 = f.v[0], f1 = f.v[1], f2 = f.v[2], f3 = f.v[3], f4 = f.v[4];
    int32_t f5 = f.v[5], f6 = f.v[6], f7 = f.v[7], f8 = f.v[8], f9 = f.v[9];
    int32_t g0 = g.v[0], g1 = g.v[1], g2 = g.v[2], g3 = g.v[3], g4 = g.v[4];
    int32_t g5 = g.v[5], g6 = g.v[6], g7 = g.v[7], g8 = g.v[8], g9 = g.v[9];

    int32_t g1_19 = 19 * g1; // 1.959375*2^29
    int32_t g2_19 = 19 * g2;
    int32_t g3_19 = 19 * g3;
    int32_t g4_19 = 19 * g4;
    int32_t g5_19 = 19 * g5;
    int32_t g6_19 = 19 * g6;
    int32_t g7_19 = 19 * g7;
    int32_t g8_19 = 19 * g8;
    int32_t g9_19 = 19 * g9;
    int32_t f1_2 = 2 * f1;
    int32_t f3_2 = 2 * f3;
    int32_t f5_2 = 2 * f5;
    int32_t f7_2 = 2 * f7;
    int32_t f9_2 = 2 * f9;

    int64_t h0 = mulw(f0, g0);
    h0 = madw(f1_2, g9_19, h0);
    h0 = madw(f2,   g8_19, h0);
    h0 = madw(f3_2, g7_19, h0);
    h0 = madw(f4,   g6_19, h0);
    h0 = madw(f5_2, g5_19, h0);
    h0 = madw(f6,   g4_19, h0);
    h0 = madw(f7_2, g3_19, h0);
    h0 = madw(f8,   g2_19, h0);
    h0 = madw(f9_2, g1_19, h0);

    int64_t h1 = mulw(f0, g1);
    h1 = madw(f1, g0,    h1);
    h1 = madw(f2, g9_19, h1);
    h1 = madw(f3, g8_19, h1);
    h1 = madw(f4, g7_19, h1);
    h1 = madw(f5, g6_19, h1);
    h1 = madw(f6, g5_19, h1);
    h1 = madw(f7, g4_19, h1);
    h1 = madw(f8, g3_19, h1);
    h1 = madw(f9, g2_19, h1);

    int64_t h2 = mulw(f0, g2);
    h2 = madw(f1_2, g1,    h2);
    h2 = madw(f2,   g0,    h2);
    h2 = madw(f3_2, g9_19, h2);
    h2 = madw(f4,   g8_19, h2);
    h2 = madw(f5_2, g7_19, h2);
    h2 = madw(f6,   g6_19, h2);
    h2 = madw(f7_2, g5_19, h2);
    h2 = madw(f8,   g4_19, h2);
    h2 = madw(f9_2, g3_19, h2);

    int64_t h3 = mulw(f0, g3);
    h3 = madw(f1, g2,    h3);
    h3 = madw(f2, g1,    h3);
    h3 = madw(f3, g0,    h3);
    h3 = madw(f4, g9_19, h3);
    h3 = madw(f5, g8_19, h3);
    h3 = madw(f6, g7_19, h3);
    h3 = madw(f7, g6_19, h3);
    h3 = madw(f8, g5_19, h3);
    h3 = madw(f9, g4_19, h3);

    int64_t h4 = mulw(f0, g4);
    h4 = madw(f1_2, g3,    h4);
    h4 = madw(f2,   g2,    h4);
    h4 = madw(f3_2, g1,    h4);
    h4 = madw(f4,   g0,    h4);
    h4 = madw(f5_2, g9_19, h4);
    h4 = madw(f6,   g8_19, h4);
    h4 = madw(f7_2, g7_19, h4);
    h4 = madw(f8,   g6_19, h4);
    h4 = madw(f9_2, g5_19, h4);

    int64_t h5 = mulw(f0, g5);
    h5 = madw(f1, g4,    h5);
    h5 = madw(f2, g3,    h5);
    h5 = madw(f3, g2,    h5);
    h5 = madw(f4, g1,    h5);
    h5 = madw(f5, g0,    h5);
    h5 = madw(f6, g9_19, h5);
    h5 = madw(f7, g8_19, h5);
    h5 = madw(f8, g7_19, h5);
    h5 = madw(f9, g6_19, h5);

    int64_t h6 = mulw(f0, g6);
    h6 = madw(f1_2, g5,    h6);
    h6 = madw(f2,   g4,    h6);
    h6 = madw(f3_2, g3,    h6);
    h6 = madw(f4,   g2,    h6);
    h6 = madw(f5_2, g1,    h6);
    h6 = madw(f6,   g0,    h6);
    h6 = madw(f7_2, g9_19, h6);
    h6 = madw(f8,   g8_19, h6);
    h6 = madw(f9_2, g7_19, h6);

    int64_t h7 = mulw(f0, g7);
    h7 = madw(f1, g6,    h7);
    h7 = madw(f2, g5,    h7);
    h7 = madw(f3, g4,    h7);
    h7 = madw(f4, g3,    h7);
    h7 = madw(f5, g2,    h7);
    h7 = madw(f6, g1,    h7);
    h7 = madw(f7, g0,    h7);
    h7 = madw(f8, g9_19, h7);
    h7 = madw(f9, g8_19, h7);

    int64_t h8 = mulw(f0, g8);
    h8 = madw(f1_2, g7,    h8);
    h8 = madw(f2,   g6,    h8);
    h8 = madw(f3_2, g5,    h8);
    h8 = madw(f4,   g4,    h8);
    h8 = madw(f5_2, g3,    h8);
    h8 = madw(f6,   g2,    h8);
    h8 = madw(f7_2, g1,    h8);
    h8 = madw(f8,   g0,    h8);
    h8 = madw(f9_2, g9_19, h8);

    int64_t h9 = mulw(f0, g9);
    h9 = madw(f1, g8, h9);
    h9 = madw(f2, g7, h9);
    h9 = madw(f3, g6, h9);
    h9 = madw(f4, g5, h9);
    h9 = madw(f5, g4, h9);
    h9 = madw(f6, g3, h9);
    h9 = madw(f7, g2, h9);
    h9 = madw(f8, g1, h9);
    h9 = madw(f9, g0, h9);

    return fe_carry_chain(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9);
}

// ref10 fe_sq: h = f * f mod 2^255-19. Same bounds as fe_mul.
__device__ __forceinline__ Fe fe_sq(Fe f) {
    int32_t f0 = f.v[0], f1 = f.v[1], f2 = f.v[2], f3 = f.v[3], f4 = f.v[4];
    int32_t f5 = f.v[5], f6 = f.v[6], f7 = f.v[7], f8 = f.v[8], f9 = f.v[9];

    int32_t f0_2 = 2 * f0;
    int32_t f1_2 = 2 * f1;
    int32_t f2_2 = 2 * f2;
    int32_t f3_2 = 2 * f3;
    int32_t f4_2 = 2 * f4;
    int32_t f5_2 = 2 * f5;
    int32_t f6_2 = 2 * f6;
    int32_t f7_2 = 2 * f7;
    int32_t f5_38 = 38 * f5;
    int32_t f6_19 = 19 * f6;
    int32_t f7_38 = 38 * f7;
    int32_t f8_19 = 19 * f8;
    int32_t f9_38 = 38 * f9;

    int64_t h0 = mulw(f0, f0);
    h0 = madw(f1_2, f9_38, h0);
    h0 = madw(f2_2, f8_19, h0);
    h0 = madw(f3_2, f7_38, h0);
    h0 = madw(f4_2, f6_19, h0);
    h0 = madw(f5,   f5_38, h0);

    int64_t h1 = mulw(f0_2, f1);
    h1 = madw(f2,   f9_38, h1);
    h1 = madw(f3_2, f8_19, h1);
    h1 = madw(f4,   f7_38, h1);
    h1 = madw(f5_2, f6_19, h1);

    int64_t h2 = mulw(f0_2, f2);
    h2 = madw(f1_2, f1,    h2);
    h2 = madw(f3_2, f9_38, h2);
    h2 = madw(f4_2, f8_19, h2);
    h2 = madw(f5_2, f7_38, h2);
    h2 = madw(f6,   f6_19, h2);

    int64_t h3 = mulw(f0_2, f3);
    h3 = madw(f1_2, f2,    h3);
    h3 = madw(f4,   f9_38, h3);
    h3 = madw(f5_2, f8_19, h3);
    h3 = madw(f6,   f7_38, h3);

    int64_t h4 = mulw(f0_2, f4);
    h4 = madw(f1_2, f3_2,  h4);
    h4 = madw(f2,   f2,    h4);
    h4 = madw(f5_2, f9_38, h4);
    h4 = madw(f6_2, f8_19, h4);
    h4 = madw(f7,   f7_38, h4);

    int64_t h5 = mulw(f0_2, f5);
    h5 = madw(f1_2, f4,    h5);
    h5 = madw(f2_2, f3,    h5);
    h5 = madw(f6,   f9_38, h5);
    h5 = madw(f7_2, f8_19, h5);

    int64_t h6 = mulw(f0_2, f6);
    h6 = madw(f1_2, f5_2,  h6);
    h6 = madw(f2_2, f4,    h6);
    h6 = madw(f3_2, f3,    h6);
    h6 = madw(f7_2, f9_38, h6);
    h6 = madw(f8,   f8_19, h6);

    int64_t h7 = mulw(f0_2, f7);
    h7 = madw(f1_2, f6, h7);
    h7 = madw(f2_2, f5, h7);
    h7 = madw(f3_2, f4, h7);
    h7 = madw(f8,   f9_38, h7);

    int64_t h8 = mulw(f0_2, f8);
    h8 = madw(f1_2, f7_2,  h8);
    h8 = madw(f2_2, f6,    h8);
    h8 = madw(f3_2, f5_2,  h8);
    h8 = madw(f4,   f4,    h8);
    h8 = madw(f9,   f9_38, h8);

    int64_t h9 = mulw(f0_2, f9);
    h9 = madw(f1_2, f8, h9);
    h9 = madw(f2_2, f7, h9);
    h9 = madw(f3_2, f6, h9);
    h9 = madw(f4_2, f5, h9);

    return fe_carry_chain(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9);
}

// Kept a loop in every compiler: the 254-squaring chain of fe_invert runs
// once per window, and unrolling it (~60K instructions) would evict the
// per-candidate walk loop from the instruction cache for nothing.
__device__ __forceinline__ Fe fe_sq_times(Fe f, uint32_t n) {
#pragma unroll 1
    for (uint32_t i = 0; i < n; i++) {
        f = fe_sq(f);
    }
    return f;
}

// z^(p-2) via the standard curve25519 ref10 addition chain
// (254 squarings + 11 multiplications).
__device__ __forceinline__ Fe fe_invert(Fe z) {
    Fe t0 = fe_sq(z);                       // z^2
    Fe t1 = fe_sq_times(t0, 2);             // z^8
    t1 = fe_mul(z, t1);                     // z^9
    t0 = fe_mul(t0, t1);                    // z^11
    Fe t2 = fe_sq(t0);                      // z^22
    t1 = fe_mul(t1, t2);                    // z^31 = z^(2^5 - 1)
    t2 = fe_sq_times(t1, 5);                // z^(2^10 - 2^5)
    t1 = fe_mul(t2, t1);                    // z^(2^10 - 1)
    t2 = fe_sq_times(t1, 10);               // z^(2^20 - 2^10)
    t2 = fe_mul(t2, t1);                    // z^(2^20 - 1)
    Fe t3 = fe_sq_times(t2, 20);            // z^(2^40 - 2^20)
    t2 = fe_mul(t3, t2);                    // z^(2^40 - 1)
    t2 = fe_sq_times(t2, 10);               // z^(2^50 - 2^10)
    t1 = fe_mul(t2, t1);                    // z^(2^50 - 1)
    t2 = fe_sq_times(t1, 50);               // z^(2^100 - 2^50)
    t2 = fe_mul(t2, t1);                    // z^(2^100 - 1)
    t3 = fe_sq_times(t2, 100);              // z^(2^200 - 2^100)
    t2 = fe_mul(t3, t2);                    // z^(2^200 - 1)
    t2 = fe_sq_times(t2, 50);               // z^(2^250 - 2^50)
    t1 = fe_mul(t2, t1);                    // z^(2^250 - 1)
    t1 = fe_sq_times(t1, 5);                // z^(2^255 - 2^5)
    return fe_mul(t1, t0);                  // z^(2^255 - 21) = z^(p - 2)
}

// ref10 fe_tobytes, truncated to the low 8 bytes of the canonical 32-byte
// little-endian encoding. Fully reduces h mod 2^255-19 first, so the result
// is the canonical prefix. (Byte 31 would carry the x sign bit in a full
// compressed point; it cannot affect bytes 0..7.)
__device__ __forceinline__ uint64_t fe_tobytes_prefix8(Fe input) {
    int32_t h0 = input.v[0], h1 = input.v[1], h2 = input.v[2], h3 = input.v[3], h4 = input.v[4];
    int32_t h5 = input.v[5], h6 = input.v[6], h7 = input.v[7], h8 = input.v[8], h9 = input.v[9];

    int32_t q = (19 * h9 + (1 << 24)) >> 25;
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

    // Goal: output h - (2^255 - 19) q, which is between 0 and 2^255 - 20.
    h0 += 19 * q;
    // Goal: output h % 2^255; carry chain, top carry discarded.
    int32_t carry0 = h0 >> 26; h1 += carry0; h0 -= carry0 << 26;
    int32_t carry1 = h1 >> 25; h2 += carry1; h1 -= carry1 << 25;
    int32_t carry2 = h2 >> 26; h3 += carry2; h2 -= carry2 << 26;
    int32_t carry3 = h3 >> 25; h4 += carry3; h3 -= carry3 << 25;
    int32_t carry4 = h4 >> 26; h5 += carry4; h4 -= carry4 << 26;
    int32_t carry5 = h5 >> 25; h6 += carry5; h5 -= carry5 << 25;
    int32_t carry6 = h6 >> 26; h7 += carry6; h6 -= carry6 << 26;
    int32_t carry7 = h7 >> 25; h8 += carry7; h7 -= carry7 << 25;
    int32_t carry8 = h8 >> 26; h9 += carry8; h8 -= carry8 << 26;
    int32_t carry9 = h9 >> 25;                h9 -= carry9 << 25;

    // Bytes 0..7 of the canonical encoding. The canonical integer is
    // h0 + h1*2^26 + h2*2^51 + h3*2^77 + ...; the low 64 bits therefore take
    // all of h0 (26 bits), all of h1 (25 bits) and the low 13 bits of h2.
    // h3 starts at bit 77, so it cannot affect bytes 0..7.
    return static_cast<uint64_t>(static_cast<uint32_t>(h0))
        | (static_cast<uint64_t>(static_cast<uint32_t>(h1)) << 26)
        | (static_cast<uint64_t>(static_cast<uint32_t>(h2)) << 51);
}

// Full ref10 fe_tobytes: the canonical 32-byte little-endian encoding of h
// mod 2^255-19. Same reduction as fe_tobytes_prefix8, then the standard
// ref10 byte packing of the ten reduced limbs. Only used for the
// once-per-launch gid-0 self-check write, so it stays out of the hot loop.
__device__ __forceinline__ void fe_tobytes32(uint8_t out[32], Fe input) {
    int32_t h0 = input.v[0], h1 = input.v[1], h2 = input.v[2], h3 = input.v[3], h4 = input.v[4];
    int32_t h5 = input.v[5], h6 = input.v[6], h7 = input.v[7], h8 = input.v[8], h9 = input.v[9];

    int32_t q = (19 * h9 + (1 << 24)) >> 25;
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

    // Output h - (2^255 - 19) q, i.e. the canonical representative.
    h0 += 19 * q;
    int32_t carry0 = h0 >> 26; h1 += carry0; h0 -= carry0 << 26;
    int32_t carry1 = h1 >> 25; h2 += carry1; h1 -= carry1 << 25;
    int32_t carry2 = h2 >> 26; h3 += carry2; h2 -= carry2 << 26;
    int32_t carry3 = h3 >> 25; h4 += carry3; h3 -= carry3 << 25;
    int32_t carry4 = h4 >> 26; h5 += carry4; h4 -= carry4 << 26;
    int32_t carry5 = h5 >> 25; h6 += carry5; h5 -= carry5 << 25;
    int32_t carry6 = h6 >> 26; h7 += carry6; h6 -= carry6 << 26;
    int32_t carry7 = h7 >> 25; h8 += carry7; h7 -= carry7 << 25;
    int32_t carry8 = h8 >> 26; h9 += carry8; h8 -= carry8 << 26;
    int32_t carry9 = h9 >> 25;                h9 -= carry9 << 25;

    out[0]  = static_cast<uint8_t>(h0 >> 0);
    out[1]  = static_cast<uint8_t>(h0 >> 8);
    out[2]  = static_cast<uint8_t>(h0 >> 16);
    out[3]  = static_cast<uint8_t>((h0 >> 24) | (h1 << 2));
    out[4]  = static_cast<uint8_t>(h1 >> 6);
    out[5]  = static_cast<uint8_t>(h1 >> 14);
    out[6]  = static_cast<uint8_t>((h1 >> 22) | (h2 << 3));
    out[7]  = static_cast<uint8_t>(h2 >> 5);
    out[8]  = static_cast<uint8_t>(h2 >> 13);
    out[9]  = static_cast<uint8_t>((h2 >> 21) | (h3 << 5));
    out[10] = static_cast<uint8_t>(h3 >> 3);
    out[11] = static_cast<uint8_t>(h3 >> 11);
    out[12] = static_cast<uint8_t>((h3 >> 19) | (h4 << 6));
    out[13] = static_cast<uint8_t>(h4 >> 2);
    out[14] = static_cast<uint8_t>(h4 >> 10);
    out[15] = static_cast<uint8_t>(h4 >> 18);
    out[16] = static_cast<uint8_t>(h5 >> 0);
    out[17] = static_cast<uint8_t>(h5 >> 8);
    out[18] = static_cast<uint8_t>(h5 >> 16);
    out[19] = static_cast<uint8_t>((h5 >> 24) | (h6 << 1));
    out[20] = static_cast<uint8_t>(h6 >> 7);
    out[21] = static_cast<uint8_t>(h6 >> 15);
    out[22] = static_cast<uint8_t>((h6 >> 23) | (h7 << 3));
    out[23] = static_cast<uint8_t>(h7 >> 5);
    out[24] = static_cast<uint8_t>(h7 >> 13);
    out[25] = static_cast<uint8_t>((h7 >> 21) | (h8 << 4));
    out[26] = static_cast<uint8_t>(h8 >> 4);
    out[27] = static_cast<uint8_t>(h8 >> 12);
    out[28] = static_cast<uint8_t>((h8 >> 20) | (h9 << 6));
    out[29] = static_cast<uint8_t>(h9 >> 2);
    out[30] = static_cast<uint8_t>(h9 >> 10);
    out[31] = static_cast<uint8_t>(h9 >> 18);
}

// The table is read-only for the kernel's lifetime: __ldg routes the loads
// through the non-coherent (read-only) cache path.
__device__ __forceinline__ Fe fe_load_table(const uint32_t* table, uint32_t offset) {
    Fe h;
    for (uint32_t i = 0; i < 10; i++) {
        h.v[i] = static_cast<int32_t>(__ldg(table + offset + i));
    }
    return h;
}

__device__ __forceinline__ Niels load_niels(
    const uint32_t* table,
    uint32_t digit_index,
    int digit
) {
    uint32_t abs_digit = digit < 0 ? static_cast<uint32_t>(-digit) : static_cast<uint32_t>(digit);
    uint32_t point_index = digit_index * 32 + (abs_digit - 1);
    uint32_t base = point_index * 3 * 10;

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

__device__ __forceinline__ EPoint extended_identity() {
    EPoint p;
    p.X = fe_zero();
    p.Y = fe_one();
    p.Z = fe_one();
    p.T = fe_zero();
    return p;
}

__device__ __forceinline__ EPoint completed_to_extended(CPoint c) {
    EPoint p;
    p.X = fe_mul(c.X, c.T);
    p.Y = fe_mul(c.Y, c.Z);
    p.Z = fe_mul(c.Z, c.T);
    p.T = fe_mul(c.X, c.Y);
    return p;
}

__device__ __forceinline__ EPoint add_niels(EPoint p, Niels n) {
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

__device__ __forceinline__ EPoint edwards_mul_base(
    const uint32_t* scalar,
    const uint32_t* table
) {
    int digits[43];
    int carry = 0;
    for (uint32_t i = 0; i < 43; i++) {
        uint32_t bit_offset = i * 6;
        uint32_t word_index = bit_offset >> 5;
        uint32_t bit_index = bit_offset & 31;
        uint32_t bit_buf = scalar[word_index] >> bit_index;
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
    for (uint32_t i = 0; i < 43; i++) {
        int digit = digits[i];
        if (digit != 0) {
            p = add_niels(p, load_niels(table, i, digit));
        }
    }

    return p;
}

// Prefix compare on the canonical low 8 bytes of y (prefixes up to 16
// nibbles; the host re-checks longer ones against the whole prefix). The byte/nibble loop of
// Prefix::matches is folded into one 64-bit mask computed once per thread:
// full prefix bytes are 0xFF, an odd trailing nibble masks the HIGH half
// (0xF0) of the next byte (the prefix's first hex digit is the high nibble
// of byte 0). The mask saturates for longer prefixes anyway.
__device__ __forceinline__ uint64_t prefix_mask8(uint32_t prefix_nibbles) {
    uint32_t full_bytes = prefix_nibbles >> 1;
    uint64_t mask = full_bytes >= 8 ? ~0ULL : ((1ULL << (8 * full_bytes)) - 1);
    if ((prefix_nibbles & 1) != 0 && full_bytes < 8) {
        mask |= 0xF0ULL << (8 * full_bytes);
    }
    return mask;
}

__device__ __forceinline__ bool matches_prefix_masked(
    uint64_t y_prefix8,
    uint64_t prefix_bytes,
    uint64_t mask
) {
    uint32_t first = static_cast<uint32_t>(y_prefix8) & 0xFF;
    if (first == 0 || first == 0xFF) {
        return false;
    }
    return ((y_prefix8 ^ prefix_bytes) & mask) == 0;
}

// Compress an affine point given Y, X and the shared Zinv: the canonical
// 32-byte y encoding with bit 255 set to the parity of canonical x
// (RFC 8032 / dalek compress()).
__device__ __forceinline__ void compress_point(uint8_t out[32], Fe y, Fe x, Fe zinv) {
    fe_tobytes32(out, fe_mul(y, zinv));
    uint8_t x_bytes[32];
    fe_tobytes32(x_bytes, fe_mul(x, zinv));
    out[31] = static_cast<uint8_t>(out[31] | ((x_bytes[0] & 1) << 7));
}

// ---------------------------------------------------------------------------
// Symmetric affine window walk (search_kernel; mesh_kernel runs the same
// walk in Montgomery u, see walk_windows_u). Same algorithm as the Metal
// kernel (src/kernels/ed25519.metal, "Symmetric affine window walk"),
// restated here with the storage layout the CUDA kernel uses.
//
// Each thread visits the candidates P + k*S (S = 8*B, k = 0..iters-1) in
// windows of W = BATCH_INV_WINDOW. Window w keeps its MIDDLE point
// M = P + (w*W + H)*S, H = W/2, in projective form (X:Y:Z) and derives
// candidate j of the window, j = 0..W-1, as M + (j-H)*S from the host-built
// affine step table of i*S (x, y, 2d*x*y; entry i-1 for i = 1..H, entry H
// for W*S). With A = X*x_i, B = Y*y_i, K = (X*Y)*(2d*x_i*y_i) the complete
// twisted-Edwards addition law (a = -1, d non-square: no exceptional
// inputs) gives both signs from the same three products:
//
//   y(M + i*S) = 2Z*(B + A) / (2Z^2 - K)
//   y(M - i*S) = 2Z*(B - A) / (2Z^2 + K)
//
// i.e. 1.5 fe_mul per candidate plus the 4 fe_mul of a Montgomery batch
// inversion (one fe_invert per window) and no squarings, against the
// x-only ladder's 5M+2S+2 const-M forward step plus 3M backward.
//
// The batch inversion runs as two independent product chains: chain A holds
// slot 0 (the base, y = Y/Z = 2Z*Y / 2Z^2), the plus candidates i = 1..H-1
// and, unless this is the last window, the base update; chain B holds the
// minus candidates i = 1..H. Both are inverted together as 1/(prodA*prodB)
// with the common factor 2Z carried through (inv = 2Z / prod), so the first
// window runs straight off the extended point of edwards_mul_base + H*S
// without an inversion of its own; after the first window the base is
// affine (Z = 1). The base advances by W*S through the same products (x3
// needs 1/(2Z^2 + K) too, so the update's denominator joins chain A as the
// product (2Z^2 - K)(2Z^2 + K)).
//
// Per pair i only three field elements are kept in local memory: the two
// chain terms tsA[i-1] = (B+A)*prodA_before and tsB[i-1] = (B-A)*prodB_before
// and K itself; the backward pass recomputes the denominators 2Z^2 -+ K from
// the register copy of 2Z^2 (3H+1 Fe per thread instead of the ladder's 3W).
// A short final window (iters not a multiple of W) computes all W candidates
// and reports only j < w.
//
// Bounds: fe_mul / fe_sq outputs are <= 1.01*2^25 (even limbs) / 1.01*2^24
// (odd) and table limbs are canonical (< 2^26 / 2^25). Every fe_mul input
// below is either such a value, a sum/difference of two of them
// (B +- A, 2Z^2 -+ K: <= 3.03*2^25) or 2Z (<= 2.02*2^25), all inside the
// ref10 input bound of 1.65*2^26. The update's x numerator is formed
// directly as Y*x + X*y (two extra fe_mul once per window) rather than from
// (Y+X)(y+x), whose y+x factor could reach 2^27.
// ---------------------------------------------------------------------------

#define WALK_HALF (BATCH_INV_WINDOW / 2)

struct StepEntry {
    Fe x;
    Fe y;
    Fe xy2d;
};

// Entry `entry` of the affine step table (30 words: x, y, 2d*x*y as ten
// non-negative ref10 limbs each). The table is read-only for the kernel's
// lifetime and every lane of a warp reads the same entry at the same time,
// so __ldg broadcasts it from the read-only cache path.
__device__ __forceinline__ StepEntry load_step(const uint32_t* step_table, uint32_t entry) {
    uint32_t base = entry * 30;
    StepEntry s;
    s.x = fe_load_table(step_table, base);
    s.y = fe_load_table(step_table, base + 10);
    s.xy2d = fe_load_table(step_table, base + 20);
    return s;
}

// Table entry as a Niels point for add_niels (the one-off M = P + H*S).
__device__ __forceinline__ Niels load_niels_step(const uint32_t* step_table, uint32_t entry) {
    StepEntry s = load_step(step_table, entry);
    Niels n;
    n.y_plus_x = fe_add(s.y, s.x);
    n.y_minus_x = fe_sub(s.y, s.x);
    n.xy2d = s.xy2d;
    return n;
}

// Local-memory slot storage, kept behind these helpers so the layout can
// change without touching the walk.
//
// SLOT_VEC=0 (default): a slot is a plain Fe (40 bytes, ten 32-bit local
// accesses). SLOT_VEC=1: 48 bytes, 16-byte aligned, written and read as two
// 128-bit and one 64-bit accesses (STL/LDL.128 + .64). Measured on an RTX
// 5090 (32768 x 16384, two runs each) the two are identical within noise
// (10.3-10.4 G/s vanity), so the smaller stack frame stays the default: the
// kernel is not bound by local-memory instruction count.
#ifndef SLOT_VEC
#define SLOT_VEC 0
#endif

#if SLOT_VEC
#if defined(__CUDACC__)
typedef int4 slot_v4;
typedef int2 slot_v2;
#define SLOT_ALIGN __align__(16)
#else
struct slot_v4 { int32_t x, y, z, w; };
struct slot_v2 { int32_t x, y; };
#define SLOT_ALIGN alignas(16)
#endif
struct SLOT_ALIGN FeSlot {
    slot_v4 a;
    slot_v4 b;
    slot_v2 c;
};
__device__ __forceinline__ void slot_store(FeSlot* arr, uint32_t idx, Fe v) {
    FeSlot s;
    s.a.x = v.v[0]; s.a.y = v.v[1]; s.a.z = v.v[2]; s.a.w = v.v[3];
    s.b.x = v.v[4]; s.b.y = v.v[5]; s.b.z = v.v[6]; s.b.w = v.v[7];
    s.c.x = v.v[8]; s.c.y = v.v[9];
    arr[idx] = s;
}
__device__ __forceinline__ Fe slot_load(const FeSlot* arr, uint32_t idx) {
    FeSlot s = arr[idx];
    Fe v;
    v.v[0] = s.a.x; v.v[1] = s.a.y; v.v[2] = s.a.z; v.v[3] = s.a.w;
    v.v[4] = s.b.x; v.v[5] = s.b.y; v.v[6] = s.b.z; v.v[7] = s.b.w;
    v.v[8] = s.c.x; v.v[9] = s.c.y;
    return v;
}
#else
typedef Fe FeSlot;
__device__ __forceinline__ void slot_store(FeSlot* arr, uint32_t idx, Fe v) {
    arr[idx] = v;
}
__device__ __forceinline__ Fe slot_load(const FeSlot* arr, uint32_t idx) {
    return arr[idx];
}
#endif

// Runs every window of the walk. `check(y, idx)` is called once per
// candidate idx = k + j (k the window's first candidate, j < w its slot in
// the window) with the affine y (fe_mul output magnitude, unreduced). Hits
// are recorded by `check`.
template <class Check>
__device__ __forceinline__ void walk_windows(
    EPoint m,
    uint32_t iters,
    const uint32_t* step_table,
    Check&& check
) {
    Fe BX = m.X;
    Fe BY = m.Y;
    Fe BZ = m.Z;

    FeSlot tsA[WALK_HALF];      // slot i-1: plus candidate i (i = 1..H-1); slot H-1: base update
    FeSlot tsB[WALK_HALF];      // slot i-1: minus candidate i (i = 1..H)
    FeSlot kk[WALK_HALF + 1];   // slot i-1: K of pair i (i = 1..H); slot H: K of the update

    for (uint32_t k = 0; k < iters; k += BATCH_INV_WINDOW) {
        uint32_t w = iters - k;
        if (w > BATCH_INV_WINDOW) {
            w = BATCH_INV_WINDOW;
        }
        const bool more = (k + w) < iters;

        // ---- forward: chain products -------------------------------------
        Fe z2 = fe_sq(BZ);
        Fe z2x2 = fe_add(z2, z2);
        Fe xy = fe_mul(BX, BY);
        Fe prod_a = z2x2;        // slot 0 (the base): ts = BY, den = 2Z^2
        Fe prod_b = fe_one();

#pragma unroll 1
        for (uint32_t i = 1; i <= WALK_HALF; i++) {
            StepEntry s = load_step(step_table, i - 1);
            Fe a = fe_mul(BX, s.x);
            Fe b = fe_mul(BY, s.y);
            Fe k_i = fe_mul(xy, s.xy2d);
            Fe num_p = fe_add(b, a);
            Fe num_m = fe_sub(b, a);
            Fe den_p = fe_sub(z2x2, k_i);
            Fe den_m = fe_add(z2x2, k_i);
            slot_store(kk, i - 1, k_i);
            slot_store(tsB, i - 1, fe_mul(num_m, prod_b));
            prod_b = fe_mul(prod_b, den_m);
            if (i < WALK_HALF) {
                slot_store(tsA, i - 1, fe_mul(num_p, prod_a));
                prod_a = fe_mul(prod_a, den_p);
            }
        }

        Fe upd_num_x = fe_zero();
        Fe upd_num_y = fe_zero();
        if (more) {
            StepEntry s = load_step(step_table, WALK_HALF);
            Fe a = fe_mul(BX, s.x);
            Fe b = fe_mul(BY, s.y);
            Fe k_u = fe_mul(xy, s.xy2d);
            // x numerator Y*x + X*y formed directly (two multiplications once
            // per window keep every fe_mul input inside the ref10 bound).
            upd_num_x = fe_add(fe_mul(BY, s.x), fe_mul(BX, s.y));
            upd_num_y = fe_add(b, a);
            Fe uden = fe_mul(fe_sub(z2x2, k_u), fe_add(z2x2, k_u));
            slot_store(kk, WALK_HALF, k_u);
            slot_store(tsA, WALK_HALF - 1, prod_a);
            prod_a = fe_mul(prod_a, uden);
        }

        Fe inv = fe_mul(fe_add(BZ, BZ), fe_invert(fe_mul(prod_a, prod_b)));
        Fe inv_a = fe_mul(inv, prod_b);   // 2Z / prod_a
        Fe inv_b = fe_mul(inv, prod_a);   // 2Z / prod_b

        // ---- base update (consumes chain A's last slot) --------------------
        const Fe base_y = BY;
        if (more) {
            Fe k_u = slot_load(kk, WALK_HALF);
            Fe dm = fe_sub(z2x2, k_u);
            Fe dp = fe_add(z2x2, k_u);
            Fe iu = fe_mul(slot_load(tsA, WALK_HALF - 1), inv_a);  // 2Z / (dm*dp)
            BX = fe_mul(upd_num_x, fe_mul(dm, iu));
            BY = fe_mul(upd_num_y, fe_mul(dp, iu));
            BZ = fe_one();
            inv_a = fe_mul(inv_a, fe_mul(dm, dp));
        }

        // ---- backward: every candidate's affine y --------------------------
#pragma unroll 1
        for (uint32_t i = WALK_HALF; i >= 1; i--) {
            Fe k_i = slot_load(kk, i - 1);
            Fe y_m = fe_mul(slot_load(tsB, i - 1), inv_b);
            inv_b = fe_mul(inv_b, fe_add(z2x2, k_i));
            if (WALK_HALF - i < w) {
                check(y_m, k + (WALK_HALF - i));
            }
            if (i < WALK_HALF) {
                Fe y_p = fe_mul(slot_load(tsA, i - 1), inv_a);
                inv_a = fe_mul(inv_a, fe_sub(z2x2, k_i));
                if (WALK_HALF + i < w) {
                    check(y_p, k + (WALK_HALF + i));
                }
            }
        }
        Fe y_0 = fe_mul(base_y, inv_a);
        if (WALK_HALF < w) {
            check(y_0, k + WALK_HALF);
        }
    }
}

// Each thread computes ONE full scalar mult P = base*B, then visits the
// candidates P + i*8*B (scalar += 8 preserves Ed25519 clamping) window by
// window through the affine step table (see walk_windows); the affine y
// needed for prefix matching comes out of one Montgomery batch inversion per
// window of BATCH_INV_WINDOW candidates. The first match of a launch wins
// (atomicCAS on `found`); the host re-derives it with dalek.
extern "C" __global__ void MTVK_LAUNCH_BOUNDS search_kernel(
    const uint32_t* scalars,
    const SearchParams* params,
    uint32_t* found,
    uint32_t* found_index,
    uint32_t* found_iter,
    uint32_t* debug_prefix,
    const uint32_t* basepoint_table,
    const uint32_t* step_table
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params->count) {
        return;
    }

    const uint32_t* scalar = scalars + gid * 8;
    EPoint p = edwards_mul_base(scalar, basepoint_table);

    // Full compressed point (y encoding + x sign bit) of gid 0's first
    // candidate, written once per launch as eight little-endian words; the
    // host compares it byte-for-byte against dalek's compress() in tests.
    // The dedicated fe_invert keeps this entirely out of the search loop.
    if (gid == 0) {
        uint8_t enc[32];
        compress_point(enc, p.Y, p.X, fe_invert(p.Z));
        for (uint32_t i = 0; i < 8; i++) {
            debug_prefix[i] = static_cast<uint32_t>(enc[4 * i])
                | (static_cast<uint32_t>(enc[4 * i + 1]) << 8)
                | (static_cast<uint32_t>(enc[4 * i + 2]) << 16)
                | (static_cast<uint32_t>(enc[4 * i + 3]) << 24);
        }
    }

    const uint32_t iters = params->iters;
    // Launch-constant prefix data, read once: the loop below stores to
    // found_index/found_iter, so re-reading params through a pointer after
    // the (rarely taken) hit branch would cost a reload per candidate.
    const uint64_t prefix_bytes = static_cast<uint64_t>(params->prefix_bytes)
        | (static_cast<uint64_t>(params->prefix_bytes_hi) << 32);
    const uint64_t prefix_mask = prefix_mask8(params->prefix_nibbles);

    // Window middle M = P + H*S in projective form (affine after the first
    // window update).
    EPoint m = add_niels(p, load_niels_step(step_table, WALK_HALF - 1));

    auto check = [&](Fe y, uint32_t idx) {
        uint64_t y_prefix8 = fe_tobytes_prefix8(y);
        if (matches_prefix_masked(y_prefix8, prefix_bytes, prefix_mask)) {
            if (atomicCAS(found, 0U, 1U) == 0U) {
                *found_index = gid;
                *found_iter = idx;
            }
        }
    };
    walk_windows(m, iters, step_table, check);
}

#ifndef MAX_HITS_PER_LAUNCH
#define MAX_HITS_PER_LAUNCH 64
#endif

// ---------------------------------------------------------------------------
// Meshtastic node-ID mode (mirrors mesh_kernel in src/kernels/ed25519.metal):
// the window walk in Montgomery u. With the window middle M affine (Z = 1)
// and the mesh step table carrying d*x*y (so K = (X*Y)*(d*x_i*y_i) is HALF
// the vanity walk's K) the candidates are y(M + iS) = (B + A) / (1 - K) and
// y(M - iS) = (B - A) / (1 + K); the birational map to Curve25519,
// u = (1 + y) / (1 - y) = (d + n) / (d - n) for y = n / d, costs nothing
// extra: the batch inversion runs over d - n instead of d. Each candidate's
// u (its X25519 public key) is encoded as eight words, CRC-32'd
// (slicing-by-4 tables in shared memory) and the masked node number is
// tested against a 4096-bit bitmap, then exactly against the targets. See
// src/meshtastic.rs.
//
// Operand bounds: every numerator and denominator 1 -+ K +- B +- A sums
// three fe_mul outputs, so |even limb| <= 3.03 * 2^25 + 1 < 1.65 * 2^26
// (the ref10 input bound) and |odd| <= 3.03 * 2^24 + 1. (With the vanity
// walk's 2d table the sums reach five outputs and the int32 19*g
// precomputes overflow: measured as ~17% wrong candidates.)
// ---------------------------------------------------------------------------

struct MeshParams {
    uint32_t count;
    uint32_t iters;
    uint32_t mask;
    uint32_t target_count;
    uint32_t targets[64];
};

#define MESH_CRC_WORDS 1024
#define MESH_BITMAP_WORDS 128
#define MESH_TABLE_WORDS (MESH_CRC_WORDS + MESH_BITMAP_WORDS)

// 1: slicing-by-4 (eight rounds of four independent lookups); 0: bytewise
// (32 dependent lookups through T0). Overridable via MTVK_CUDA_DEFINES.
#ifndef MESH_CRC_SLICE4
#define MESH_CRC_SLICE4 1
#endif

// Full ref10 fe_tobytes as eight little-endian 32-bit words (canonical
// encoding of h mod 2^255-19, bit 255 clear). Limb k starts at bit
// 0, 26, 51, 77, 102, 128, 153, 179, 204, 230.
__device__ __forceinline__ void fe_tobytes_words8(uint32_t out[8], Fe input) {
    int32_t h0 = input.v[0], h1 = input.v[1], h2 = input.v[2], h3 = input.v[3], h4 = input.v[4];
    int32_t h5 = input.v[5], h6 = input.v[6], h7 = input.v[7], h8 = input.v[8], h9 = input.v[9];

    int32_t q = (19 * h9 + (1 << 24)) >> 25;
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
    int32_t carry0 = h0 >> 26; h1 += carry0; h0 -= carry0 << 26;
    int32_t carry1 = h1 >> 25; h2 += carry1; h1 -= carry1 << 25;
    int32_t carry2 = h2 >> 26; h3 += carry2; h2 -= carry2 << 26;
    int32_t carry3 = h3 >> 25; h4 += carry3; h3 -= carry3 << 25;
    int32_t carry4 = h4 >> 26; h5 += carry4; h4 -= carry4 << 26;
    int32_t carry5 = h5 >> 25; h6 += carry5; h5 -= carry5 << 25;
    int32_t carry6 = h6 >> 26; h7 += carry6; h6 -= carry6 << 26;
    int32_t carry7 = h7 >> 25; h8 += carry7; h7 -= carry7 << 25;
    int32_t carry8 = h8 >> 26; h9 += carry8; h8 -= carry8 << 26;
    int32_t carry9 = h9 >> 25;                h9 -= carry9 << 25;

    uint32_t u0 = (uint32_t)h0, u1 = (uint32_t)h1, u2 = (uint32_t)h2, u3 = (uint32_t)h3, u4 = (uint32_t)h4;
    uint32_t u5 = (uint32_t)h5, u6 = (uint32_t)h6, u7 = (uint32_t)h7, u8 = (uint32_t)h8, u9 = (uint32_t)h9;
    out[0] = u0 | (u1 << 26);
    out[1] = (u1 >> 6) | (u2 << 19);
    out[2] = (u2 >> 13) | (u3 << 13);
    out[3] = (u3 >> 19) | (u4 << 6);
    out[4] = u5 | (u6 << 25);
    out[5] = (u6 >> 7) | (u7 << 19);
    out[6] = (u7 >> 13) | (u8 << 12);
    out[7] = (u8 >> 20) | (u9 << 6);
}

// Standard reflected CRC-32 (poly 0xEDB88320, init/xorout 0xFFFFFFFF) of 32
// bytes given as eight little-endian words: the firmware's crc32Buffer.
__device__ __forceinline__ uint32_t crc32_words8(const uint32_t w[8], const uint32_t* tab) {
    uint32_t crc = 0xFFFFFFFFu;
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        crc ^= w[i];
#if MESH_CRC_SLICE4
        crc = tab[768 + (crc & 0xFFu)]
            ^ tab[512 + ((crc >> 8) & 0xFFu)]
            ^ tab[256 + ((crc >> 16) & 0xFFu)]
            ^ tab[crc >> 24];
#else
#pragma unroll
        for (uint32_t b = 0; b < 4; b++) {
            crc = tab[crc & 0xFFu] ^ (crc >> 8);
        }
#endif
    }
    return ~crc;
}

// Table entry of the mesh step table (x, y, d*x*y) as a Niels point for the
// one-off M = P + H*S: add_niels wants 2d*x*y.
__device__ __forceinline__ Niels load_niels_step_dxy(const uint32_t* step_table, uint32_t entry) {
    StepEntry s = load_step(step_table, entry);
    Niels n;
    n.y_plus_x = fe_add(s.y, s.x);
    n.y_minus_x = fe_sub(s.y, s.x);
    n.xy2d = fe_add(s.xy2d, s.xy2d);
    return n;
}

// Exact membership after the bitmap; the firmware's reserved node numbers
// (< 4, 0xFFFFFFFF) are never a hit, whatever the mask.
__device__ __forceinline__ bool mesh_is_target(uint32_t masked, uint32_t id, const MeshParams* params) {
    if (id < 4u || id == 0xFFFFFFFFu) {
        return false;
    }
    const uint32_t n = params->target_count;
    for (uint32_t i = 0; i < n; i++) {
        if (params->targets[i] == masked) {
            return true;
        }
    }
    return false;
}

// walk_windows in Montgomery u: `check(u, idx)` receives the affine u
// (fe_mul output magnitude) of candidate idx. The window middle is made
// affine once at the start (one fe_invert per thread per launch) so every
// window's chains run with Z = 1.
template <class Check>
__device__ __forceinline__ void walk_windows_u(
    EPoint m,
    uint32_t iters,
    const uint32_t* step_table,
    Check&& check
) {
    Fe zinv = fe_invert(m.Z);
    Fe BX = fe_mul(m.X, zinv);
    Fe BY = fe_mul(m.Y, zinv);
    const Fe one = fe_one();

    FeSlot tsA[WALK_HALF];      // slot i-1: plus candidate i (i = 1..H-1); slot H-1: base update
    FeSlot tsB[WALK_HALF];      // slot i-1: minus candidate i (i = 1..H)
    FeSlot dsA[WALK_HALF];      // denominators of chain A (slot H-1: dm*dp of the update)
    FeSlot dsB[WALK_HALF];      // denominators of chain B
    FeSlot ku[1];               // K of the update

    for (uint32_t k = 0; k < iters; k += BATCH_INV_WINDOW) {
        uint32_t w = iters - k;
        if (w > BATCH_INV_WINDOW) {
            w = BATCH_INV_WINDOW;
        }
        const bool more = (k + w) < iters;

        // ---- forward: chain products -------------------------------------
        Fe xy = fe_mul(BX, BY);
        Fe prod_a = fe_sub(one, BY);   // slot 0 (the base): u = (1 + y) / (1 - y)
        Fe prod_b = fe_one();

#pragma unroll 1
        for (uint32_t i = 1; i <= WALK_HALF; i++) {
            StepEntry s = load_step(step_table, i - 1);   // (x, y, d*x*y) of i*S
            Fe a = fe_mul(BX, s.x);
            Fe b = fe_mul(BY, s.y);
            Fe k_i = fe_mul(xy, s.xy2d);     // K = X*Y*d*x*y
            Fe np = fe_add(b, a);            // y(M+iS) = np / dp
            Fe nm = fe_sub(b, a);            // y(M-iS) = nm / dm
            Fe dp = fe_sub(one, k_i);
            Fe dm = fe_add(one, k_i);
            Fe up_num = fe_add(dp, np);      // u = (d + n) / (d - n)
            Fe up_den = fe_sub(dp, np);
            Fe um_num = fe_add(dm, nm);
            Fe um_den = fe_sub(dm, nm);
            slot_store(dsB, i - 1, um_den);
            slot_store(tsB, i - 1, fe_mul(um_num, prod_b));
            prod_b = fe_mul(prod_b, um_den);
            if (i < WALK_HALF) {
                slot_store(dsA, i - 1, up_den);
                slot_store(tsA, i - 1, fe_mul(up_num, prod_a));
                prod_a = fe_mul(prod_a, up_den);
            }
        }

        Fe upd_num_x = fe_zero();
        Fe upd_num_y = fe_zero();
        if (more) {
            StepEntry s = load_step(step_table, WALK_HALF);
            Fe a = fe_mul(BX, s.x);
            Fe b = fe_mul(BY, s.y);
            Fe k_u = fe_mul(xy, s.xy2d);
            // x(M+WS) = (Y x + X y) / (1 + K), y(M+WS) = (B + A) / (1 - K).
            upd_num_x = fe_add(fe_mul(BY, s.x), fe_mul(BX, s.y));
            upd_num_y = fe_add(b, a);
            Fe uden = fe_mul(fe_sub(one, k_u), fe_add(one, k_u));
            slot_store(ku, 0, k_u);
            slot_store(tsA, WALK_HALF - 1, prod_a);
            slot_store(dsA, WALK_HALF - 1, uden);
            prod_a = fe_mul(prod_a, uden);
        }

        Fe inv = fe_invert(fe_mul(prod_a, prod_b));
        Fe inv_a = fe_mul(inv, prod_b);   // 1 / prod_a
        Fe inv_b = fe_mul(inv, prod_a);   // 1 / prod_b

        // ---- base update (consumes chain A's last slot) --------------------
        const Fe base_num = fe_add(one, BY);
        if (more) {
            Fe k_u = slot_load(ku, 0);
            Fe dm = fe_sub(one, k_u);
            Fe dp = fe_add(one, k_u);
            Fe iu = fe_mul(slot_load(tsA, WALK_HALF - 1), inv_a);  // 1 / (dm*dp)
            BX = fe_mul(upd_num_x, fe_mul(dm, iu));
            BY = fe_mul(upd_num_y, fe_mul(dp, iu));
            inv_a = fe_mul(inv_a, slot_load(dsA, WALK_HALF - 1));
        }

        // ---- backward: every candidate's affine u --------------------------
#pragma unroll 1
        for (uint32_t i = WALK_HALF; i >= 1; i--) {
            Fe u_m = fe_mul(slot_load(tsB, i - 1), inv_b);
            inv_b = fe_mul(inv_b, slot_load(dsB, i - 1));
            if (WALK_HALF - i < w) {
                check(u_m, k + (WALK_HALF - i));
            }
            if (i < WALK_HALF) {
                Fe u_p = fe_mul(slot_load(tsA, i - 1), inv_a);
                inv_a = fe_mul(inv_a, slot_load(dsA, i - 1));
                if (WALK_HALF + i < w) {
                    check(u_p, k + (WALK_HALF + i));
                }
            }
        }
        Fe u_0 = fe_mul(base_num, inv_a);
        if (WALK_HALF < w) {
            check(u_0, k + WALK_HALF);
        }
    }
}

extern "C" __global__ void MTVK_LAUNCH_BOUNDS mesh_kernel(
    const uint32_t* scalars,
    const MeshParams* params,
    const uint32_t* mesh_tables,
    uint32_t* found_count,
    uint32_t* found_index,
    uint32_t* found_iter,
    uint32_t* debug_u,
    const uint32_t* basepoint_table,
    const uint32_t* step_table
) {
    // Stage the CRC tables and the target bitmap (4.5 KB) into shared
    // memory before any thread can return: every thread joins the barrier.
    __shared__ uint32_t tab[MESH_TABLE_WORDS];
    for (uint32_t i = threadIdx.x; i < MESH_TABLE_WORDS; i += blockDim.x) {
        tab[i] = mesh_tables[i];
    }
    __syncthreads();

    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params->count) {
        return;
    }

    const uint32_t* scalar = scalars + gid * 8;
    EPoint p = edwards_mul_base(scalar, basepoint_table);

    // X25519 public key (u = (Z + Y) / (Z - Y)) of gid 0's first candidate,
    // compared byte-for-byte against dalek on the host in tests.
    if (gid == 0) {
        Fe u = fe_mul(fe_add(p.Z, p.Y), fe_invert(fe_sub(p.Z, p.Y)));
        uint32_t uw[8];
        fe_tobytes_words8(uw, u);
        for (uint32_t i = 0; i < 8; i++) {
            debug_u[i] = uw[i];
        }
    }

    const uint32_t iters = params->iters;
    const uint32_t mask = params->mask;

    // Window middle M = P + H*S (walk_windows_u makes it affine).
    EPoint m = add_niels(p, load_niels_step_dxy(step_table, WALK_HALF - 1));

    auto check = [&](Fe u, uint32_t idx) {
        uint32_t uw[8];
        fe_tobytes_words8(uw, u);
#ifdef MESH_BENCH_NO_CRC
        // Measurement only (MTVK_CUDA_DEFINES=-DMESH_BENCH_NO_CRC): the
        // CRC's cost is the difference to the default build. Not a node id.
        uint32_t id = uw[0] ^ uw[1] ^ uw[2] ^ uw[3] ^ uw[4] ^ uw[5] ^ uw[6] ^ uw[7];
#else
        uint32_t id = crc32_words8(uw, tab);
#endif
        uint32_t masked = id & mask;
        uint32_t bit = (masked * 0x9E3779B1u) >> 20;
        if ((tab[MESH_CRC_WORDS + (bit >> 5)] >> (bit & 31u)) & 1u) {
            if (mesh_is_target(masked, id, params)) {
                uint32_t slot = atomicAdd(found_count, 1u);
                if (slot < MAX_HITS_PER_LAUNCH) {
                    found_index[slot] = gid;
                    found_iter[slot] = idx;
                }
            }
        }
    };
    walk_windows_u(m, iters, step_table, check);
}
