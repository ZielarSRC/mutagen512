#ifndef RIPEMD160_AVX512_H
#define RIPEMD160_AVX512_H

#include <immintrin.h>

#include <cstdint>

namespace ripemd160avx512 {

// Initializing Ripemd160 state
void Initialize(__m512i *state);

// Transform function for processing 16 blocks at once with AVX-512
void Transform(__m512i *state, uint8_t *blocks[16]);

// Hashing functions for different batch sizes
// 16 x 32-byte inputs, optimized for Xeon Platinum 8488C
void ripemd160avx512_16x32(unsigned char *inputs[16], unsigned char *outputs[16]);

// Process 32 messages in parallel (even better for Xeon 8488C)
void ripemd160avx512_32x32(unsigned char *inputs[32], unsigned char *outputs[32]);

// Original AVX2 function for backward compatibility
void ripemd160avx512_8x32(unsigned char *i0, unsigned char *i1, unsigned char *i2,
                          unsigned char *i3, unsigned char *i4, unsigned char *i5,
                          unsigned char *i6, unsigned char *i7, unsigned char *d0,
                          unsigned char *d1, unsigned char *d2, unsigned char *d3,
                          unsigned char *d4, unsigned char *d5, unsigned char *d6,
                          unsigned char *d7);

// Function for streaming multiple batches using all available cores
void ripemd160avx512_stream(unsigned char **inputs, unsigned char **outputs, size_t num_messages);

// Prefetch helpers for optimizing cache usage
void PrefetchInputs(unsigned char **inputs, int count, int hint = _MM_HINT_T0);

// Helper functions for threading
int GetOptimalThreadCount();
void SetThreadAffinity(int thread_id);

}  // namespace ripemd160avx512

#endif  // RIPEMD160_AVX512_H
