// Host-side stand-ins for the CUDA keywords and intrinsics that
// src/kernels/ed25519.cu uses, so the kernel source compiles as plain C++
// for the differential tests in src/gpu/cuda_host_tests.rs (no NVIDIA GPU
// needed).
#pragma once

#include <stdint.h>

#define __device__
#define __global__
#define __host__
#define __constant__
#define __forceinline__ inline
#define __noinline__
#define __launch_bounds__(...)

struct HostDim3 {
    unsigned int x, y, z;
};

// The harness sets these per emulated thread before calling a kernel.
static HostDim3 blockIdx = {0, 0, 0};
static HostDim3 blockDim = {1, 1, 1};
static HostDim3 threadIdx = {0, 0, 0};

static inline uint32_t __umulhi(uint32_t a, uint32_t b) {
    return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
}

template <class T>
static inline T __ldg(const T* p) {
    return *p;
}

static inline uint32_t atomicAdd(uint32_t* p, uint32_t v) {
    uint32_t old = *p;
    *p += v;
    return old;
}

// Block-shared memory and the block barrier, for the Meshtastic kernel's
// table staging: the harness emulates one thread per block (blockDim.x = 1),
// so a `static` array that every emulated thread refills is equivalent.
#define __shared__ static
static inline void __syncthreads() {}
