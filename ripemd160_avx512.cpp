#include <immintrin.h>
#include <omp.h>
#include <cstdint>
#include <cstring>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cstdio>

#include "ripemd160_avx512.h"

namespace ripemd160avx512 {

// Aligned state initialization values for AVX-512
#ifdef WIN64
static const __declspec(align(64)) uint32_t _init[] = {
#else
static const uint32_t _init[] __attribute__((aligned(64))) = {
#endif
    // 16 copies of A
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,

    // 16 copies of B
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,

    // 16 copies of C
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,

    // 16 copies of D
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,

    // 16 copies of E
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul,
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul,
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul};

// AVX-512 optimized operations
#define _mm512_not_si512(x) _mm512_xor_si512((x), _mm512_set1_epi32(-1))
#define ROL(x, n) _mm512_or_si512(_mm512_slli_epi32(x, n), _mm512_srli_epi32(x, 32 - n))

// RIPEMD-160 functions - optimized for AVX-512
#define f1(x, y, z) _mm512_xor_si512(x, _mm512_xor_si512(y, z))
#define f2(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0xE8)  // (x & y) | (~x & z)
#define f3(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0x96)  // (x | ~y) ^ z
#define f4(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0xE2)  // (x & z) | (y & ~z)
#define f5(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0x78)  // x ^ (y | ~z)

// Adding helpers with AVX-512
#define add3(x0, x1, x2) _mm512_add_epi32(_mm512_add_epi32(x0, x1), x2)
#define add4(x0, x1, x2, x3) _mm512_add_epi32(_mm512_add_epi32(x0, x1), _mm512_add_epi32(x2, x3))

// Round function optimized for AVX-512
#define Round(a, b, c, d, e, f, x, k, r)   \
  u = add4(a, f, x, _mm512_set1_epi32(k)); \
  a = _mm512_add_epi32(ROL(u, r), e);      \
  c = ROL(c, 10);

// Macroses for each round
#define R11(a, b, c, d, e, x, r) Round(a, b, c, d, e, f1(b, c, d), x, 0, r)
#define R21(a, b, c, d, e, x, r) Round(a, b, c, d, e, f2(b, c, d), x, 0x5A827999ul, r)
#define R31(a, b, c, d, e, x, r) Round(a, b, c, d, e, f3(b, c, d), x, 0x6ED9EBA1ul, r)
#define R41(a, b, c, d, e, x, r) Round(a, b, c, d, e, f4(b, c, d), x, 0x8F1BBCDCul, r)
#define R51(a, b, c, d, e, x, r) Round(a, b, c, d, e, f5(b, c, d), x, 0xA953FD4Eul, r)
#define R12(a, b, c, d, e, x, r) Round(a, b, c, d, e, f5(b, c, d), x, 0x50A28BE6ul, r)
#define R22(a, b, c, d, e, x, r) Round(a, b, c, d, e, f4(b, c, d), x, 0x5C4DD124ul, r)
#define R32(a, b, c, d, e, x, r) Round(a, b, c, d, e, f3(b, c, d), x, 0x6D703EF3ul, r)
#define R42(a, b, c, d, e, x, r) Round(a, b, c, d, e, f2(b, c, d), x, 0x7A6D76E9ul, r)
#define R52(a, b, c, d, e, x, r) Round(a, b, c, d, e, f1(b, c, d), x, 0, r)

// Macros to load words from the message blocks for 16 messages at once
#define LOADW_16(i)                                                                       \
  _mm512_set_epi32(                                                                       \
      *((uint32_t *)blk[0] + i), *((uint32_t *)blk[1] + i), *((uint32_t *)blk[2] + i),    \
      *((uint32_t *)blk[3] + i), *((uint32_t *)blk[4] + i), *((uint32_t *)blk[5] + i),    \
      *((uint32_t *)blk[6] + i), *((uint32_t *)blk[7] + i), *((uint32_t *)blk[8] + i),    \
      *((uint32_t *)blk[9] + i), *((uint32_t *)blk[10] + i), *((uint32_t *)blk[11] + i),  \
      *((uint32_t *)blk[12] + i), *((uint32_t *)blk[13] + i), *((uint32_t *)blk[14] + i), \
      *((uint32_t *)blk[15] + i))

// Initialize state for AVX-512
void Initialize(__m512i *s) { memcpy(s, _init, sizeof(uint32_t) * 16 * 5); }

// Transform function for processing 16 blocks at once with AVX-512
void Transform(__m512i *s, uint8_t *blk[16]) {
  // Prefetch the data into L1 cache
  for (int i = 0; i < 16; i++) {
    _mm_prefetch((const char *)blk[i], _MM_HINT_T0);
    _mm_prefetch((const char *)blk[i] + 64, _MM_HINT_T0);
  }

  // Load state variables
  __m512i a1 = _mm512_load_si512(s + 0);
  __m512i b1 = _mm512_load_si512(s + 1);
  __m512i c1 = _mm512_load_si512(s + 2);
  __m512i d1 = _mm512_load_si512(s + 3);
  __m512i e1 = _mm512_load_si512(s + 4);

  // Initialize second set of variables
  __m512i a2 = a1;
  __m512i b2 = b1;
  __m512i c2 = c1;
  __m512i d2 = d1;
  __m512i e2 = e1;

  __m512i u;
  __m512i w[16];

// Load all message blocks with prefetching to improve cache performance
#pragma unroll 16
  for (int i = 0; i < 16; ++i) {
    w[i] = LOADW_16(i);
  }

  // Main rounds 0-15 of Ripemd160 - implemented with AVX-512
  R11(a1, b1, c1, d1, e1, w[0], 11);
  R12(a2, b2, c2, d2, e2, w[5], 8);
  R11(e1, a1, b1, c1, d1, w[1], 14);
  R12(e2, a2, b2, c2, d2, w[14], 9);
  R11(d1, e1, a1, b1, c1, w[2], 15);
  R12(d2, e2, a2, b2, c2, w[7], 9);
  R11(c1, d1, e1, a1, b1, w[3], 12);
  R12(c2, d2, e2, a2, b2, w[0], 11);
  R11(b1, c1, d1, e1, a1, w[4], 5);
  R12(b2, c2, d2, e2, a2, w[9], 13);
  R11(a1, b1, c1, d1, e1, w[5], 8);
  R12(a2, b2, c2, d2, e2, w[2], 15);
  R11(e1, a1, b1, c1, d1, w[6], 7);
  R12(e2, a2, b2, c2, d2, w[11], 15);
  R11(d1, e1, a1, b1, c1, w[7], 9);
  R12(d2, e2, a2, b2, c2, w[4], 5);
  R11(c1, d1, e1, a1, b1, w[8], 11);
  R12(c2, d2, e2, a2, b2, w[13], 7);
  R11(b1, c1, d1, e1, a1, w[9], 13);
  R12(b2, c2, d2, e2, a2, w[6], 7);
  R11(a1, b1, c1, d1, e1, w[10], 14);
  R12(a2, b2, c2, d2, e2, w[15], 8);
  R11(e1, a1, b1, c1, d1, w[11], 15);
  R12(e2, a2, b2, c2, d2, w[8], 11);
  R11(d1, e1, a1, b1, c1, w[12], 6);
  R12(d2, e2, a2, b2, c2, w[1], 14);
  R11(c1, d1, e1, a1, b1, w[13], 7);
  R12(c2, d2, e2, a2, b2, w[10], 14);
  R11(b1, c1, d1, e1, a1, w[14], 9);
  R12(b2, c2, d2, e2, a2, w[3], 12);
  R11(a1, b1, c1, d1, e1, w[15], 8);
  R12(a2, b2, c2, d2, e2, w[12], 6);

  R21(e1, a1, b1, c1, d1, w[7], 7);
  R22(e2, a2, b2, c2, d2, w[6], 9);
  R21(d1, e1, a1, b1, c1, w[4], 6);
  R22(d2, e2, a2, b2, c2, w[11], 13);
  R21(c1, d1, e1, a1, b1, w[13], 8);
  R22(c2, d2, e2, a2, b2, w[3], 15);
  R21(b1, c1, d1, e1, a1, w[1], 13);
  R22(b2, c2, d2, e2, a2, w[7], 7);
  R21(a1, b1, c1, d1, e1, w[10], 11);
  R22(a2, b2, c2, d2, e2, w[0], 12);
  R21(e1, a1, b1, c1, d1, w[6], 9);
  R22(e2, a2, b2, c2, d2, w[13], 8);
  R21(d1, e1, a1, b1, c1, w[15], 7);
  R22(d2, e2, a2, b2, c2, w[5], 9);
  R21(c1, d1, e1, a1, b1, w[3], 15);
  R22(c2, d2, e2, a2, b2, w[10], 11);
  R21(b1, c1, d1, e1, a1, w[12], 7);
  R22(b2, c2, d2, e2, a2, w[14], 7);
  R21(a1, b1, c1, d1, e1, w[0], 12);
  R22(a2, b2, c2, d2, e2, w[15], 7);
  R21(e1, a1, b1, c1, d1, w[9], 15);
  R22(e2, a2, b2, c2, d2, w[8], 12);
  R21(d1, e1, a1, b1, c1, w[5], 9);
  R22(d2, e2, a2, b2, c2, w[12], 7);
  R21(c1, d1, e1, a1, b1, w[2], 11);
  R22(c2, d2, e2, a2, b2, w[4], 6);
  R21(b1, c1, d1, e1, a1, w[14], 7);
  R22(b2, c2, d2, e2, a2, w[9], 15);
  R21(a1, b1, c1, d1, e1, w[11], 13);
  R22(a2, b2, c2, d2, e2, w[1], 13);
  R21(e1, a1, b1, c1, d1, w[8], 12);
  R22(e2, a2, b2, c2, d2, w[2], 11);

  R31(d1, e1, a1, b1, c1, w[3], 11);
  R32(d2, e2, a2, b2, c2, w[15], 9);
  R31(c1, d1, e1, a1, b1, w[10], 13);
  R32(c2, d2, e2, a2, b2, w[5], 7);
  R31(b1, c1, d1, e1, a1, w[14], 6);
  R32(b2, c2, d2, e2, a2, w[1], 15);
  R31(a1, b1, c1, d1, e1, w[4], 7);
  R32(a2, b2, c2, d2, e2, w[3], 11);
  R31(e1, a1, b1, c1, d1, w[9], 14);
  R32(e2, a2, b2, c2, d2, w[7], 8);
  R31(d1, e1, a1, b1, c1, w[15], 9);
  R32(d2, e2, a2, b2, c2, w[14], 6);
  R31(c1, d1, e1, a1, b1, w[8], 13);
  R32(c2, d2, e2, a2, b2, w[6], 6);
  R31(b1, c1, d1, e1, a1, w[1], 15);
  R32(b2, c2, d2, e2, a2, w[9], 14);
  R31(a1, b1, c1, d1, e1, w[2], 14);
  R32(a2, b2, c2, d2, e2, w[11], 12);
  R31(e1, a1, b1, c1, d1, w[7], 8);
  R32(e2, a2, b2, c2, d2, w[8], 13);
  R31(d1, e1, a1, b1, c1, w[0], 13);
  R32(d2, e2, a2, b2, c2, w[12], 5);
  R31(c1, d1, e1, a1, b1, w[6], 6);
  R32(c2, d2, e2, a2, b2, w[2], 14);
  R31(b1, c1, d1, e1, a1, w[13], 5);
  R32(b2, c2, d2, e2, a2, w[10], 13);
  R31(a1, b1, c1, d1, e1, w[11], 12);
  R32(a2, b2, c2, d2, e2, w[0], 13);
  R31(e1, a1, b1, c1, d1, w[5], 7);
  R32(e2, a2, b2, c2, d2, w[4], 7);
  R31(d1, e1, a1, b1, c1, w[12], 5);
  R32(d2, e2, a2, b2, c2, w[13], 5);

  R41(c1, d1, e1, a1, b1, w[1], 11);
  R42(c2, d2, e2, a2, b2, w[8], 15);
  R41(b1, c1, d1, e1, a1, w[9], 12);
  R42(b2, c2, d2, e2, a2, w[6], 5);
  R41(a1, b1, c1, d1, e1, w[11], 14);
  R42(a2, b2, c2, d2, e2, w[4], 8);
  R41(e1, a1, b1, c1, d1, w[10], 15);
  R42(e2, a2, b2, c2, d2, w[1], 11);
  R41(d1, e1, a1, b1, c1, w[0], 14);
  R42(d2, e2, a2, b2, c2, w[3], 14);
  R41(c1, d1, e1, a1, b1, w[8], 15);
  R42(c2, d2, e2, a2, b2, w[11], 14);
  R41(b1, c1, d1, e1, a1, w[12], 9);
  R42(b2, c2, d2, e2, a2, w[15], 6);
  R41(a1, b1, c1, d1, e1, w[4], 8);
  R42(a2, b2, c2, d2, e2, w[0], 14);
  R41(e1, a1, b1, c1, d1, w[13], 9);
  R42(e2, a2, b2, c2, d2, w[5], 6);
  R41(d1, e1, a1, b1, c1, w[3], 14);
  R42(d2, e2, a2, b2, c2, w[12], 9);
  R41(c1, d1, e1, a1, b1, w[7], 5);
  R42(c2, d2, e2, a2, b2, w[2], 12);
  R41(b1, c1, d1, e1, a1, w[15], 6);
  R42(b2, c2, d2, e2, a2, w[13], 9);
  R41(a1, b1, c1, d1, e1, w[14], 8);
  R42(a2, b2, c2, d2, e2, w[9], 12);
  R41(e1, a1, b1, c1, d1, w[5], 6);
  R42(e2, a2, b2, c2, d2, w[7], 5);
  R41(d1, e1, a1, b1, c1, w[6], 5);
  R42(d2, e2, a2, b2, c2, w[10], 15);
  R41(c1, d1, e1, a1, b1, w[2], 12);
  R42(c2, d2, e2, a2, b2, w[14], 8);

  R51(b1, c1, d1, e1, a1, w[4], 9);
  R52(b2, c2, d2, e2, a2, w[12], 8);
  R51(a1, b1, c1, d1, e1, w[0], 15);
  R52(a2, b2, c2, d2, e2, w[15], 5);
  R51(e1, a1, b1, c1, d1, w[5], 5);
  R52(e2, a2, b2, c2, d2, w[10], 12);
  R51(d1, e1, a1, b1, c1, w[9], 11);
  R52(d2, e2, a2, b2, c2, w[4], 9);
  R51(c1, d1, e1, a1, b1, w[7], 6);
  R52(c2, d2, e2, a2, b2, w[1], 12);
  R51(b1, c1, d1, e1, a1, w[12], 8);
  R52(b2, c2, d2, e2, a2, w[5], 5);
  R51(a1, b1, c1, d1, e1, w[2], 13);
  R52(a2, b2, c2, d2, e2, w[8], 14);
  R51(e1, a1, b1, c1, d1, w[10], 12);
  R52(e2, a2, b2, c2, d2, w[7], 6);
  R51(d1, e1, a1, b1, c1, w[14], 5);
  R52(d2, e2, a2, b2, c2, w[6], 8);
  R51(c1, d1, e1, a1, b1, w[1], 12);
  R52(c2, d2, e2, a2, b2, w[2], 13);
  R51(b1, c1, d1, e1, a1, w[3], 13);
  R52(b2, c2, d2, e2, a2, w[13], 6);
  R51(a1, b1, c1, d1, e1, w[8], 14);
  R52(a2, b2, c2, d2, e2, w[14], 5);
  R51(e1, a1, b1, c1, d1, w[11], 11);
  R52(e2, a2, b2, c2, d2, w[0], 15);
  R51(d1, e1, a1, b1, c1, w[6], 8);
  R52(d2, e2, a2, b2, c2, w[3], 13);
  R51(c1, d1, e1, a1, b1, w[15], 5);
  R52(c2, d2, e2, a2, b2, w[9], 11);
  R51(b1, c1, d1, e1, a1, w[13], 6);
  R52(b2, c2, d2, e2, a2, w[11], 11);

  // Combine results and update state using AVX-512
  __m512i t = s[0];
  s[0] = add3(s[1], c1, d2);
  s[1] = add3(s[2], d1, e2);
  s[2] = add3(s[3], e1, a2);
  s[3] = add3(s[4], a1, b2);
  s[4] = add3(t, b1, c2);
}

// AVX-512 optimized extraction macros
#ifdef WIN64
#define DEPACK512(d, i)                               \
  ((uint32_t *)d)[0] = _mm512_extract_epi32(s[0], i); \
  ((uint32_t *)d)[1] = _mm512_extract_epi32(s[1], i); \
  ((uint32_t *)d)[2] = _mm512_extract_epi32(s[2], i); \
  ((uint32_t *)d)[3] = _mm512_extract_epi32(s[3], i); \
  ((uint32_t *)d)[4] = _mm512_extract_epi32(s[4], i);
#else
#define DEPACK512(d, i)                        \
  ((uint32_t *)d)[0] = ((uint32_t *)&s[0])[i]; \
  ((uint32_t *)d)[1] = ((uint32_t *)&s[1])[i]; \
  ((uint32_t *)d)[2] = ((uint32_t *)&s[2])[i]; \
  ((uint32_t *)d)[3] = ((uint32_t *)&s[3])[i]; \
  ((uint32_t *)d)[4] = ((uint32_t *)&s[4])[i];
#endif

static const uint64_t sizedesc_32 = 32 << 3;
static const unsigned char pad[64] = {0x80};

// Process 16 messages in parallel (optimized for AVX-512)
void ripemd160avx512_16x32(unsigned char *inputs[16], unsigned char *outputs[16]) {
  __m512i s[5];
  uint8_t *padded_inputs[16];

  // Allocate aligned memory for padded inputs
  for (int i = 0; i < 16; i++) {
    padded_inputs[i] = (uint8_t *)_mm_malloc(64, 64);  // Align to 64 bytes for AVX-512

    // Copy input data
    memcpy(padded_inputs[i], inputs[i], 32);

    // Add padding and length
    memcpy(padded_inputs[i] + 32, pad, 24);
    memcpy(padded_inputs[i] + 56, &sizedesc_32, 8);

    // Prefetch for faster processing
    _mm_prefetch((const char *)padded_inputs[i], _MM_HINT_T0);
  }

  // Initialize state
  Initialize(s);

  // Process all blocks
  Transform(s, padded_inputs);

// Extract results to output buffers
#pragma unroll 16
  for (int i = 0; i < 16; i++) {
    DEPACK512(outputs[i], 15 - i);
    _mm_free(padded_inputs[i]);
  }
}

// Processes 32 messages in parallel using two AVX-512 contexts
void ripemd160avx512_32x32(unsigned char *inputs[32], unsigned char *outputs[32]) {
// Split into two batches for maximum throughput
#pragma omp parallel sections
  {
#pragma omp section
    {
      // First batch of 16 messages
      unsigned char *batch1_in[16];
      unsigned char *batch1_out[16];

      for (int i = 0; i < 16; i++) {
        batch1_in[i] = inputs[i];
        batch1_out[i] = outputs[i];
      }

      ripemd160avx512_16x32(batch1_in, batch1_out);
    }

#pragma omp section
    {
      // Second batch of 16 messages
      unsigned char *batch2_in[16];
      unsigned char *batch2_out[16];

      for (int i = 0; i < 16; i++) {
        batch2_in[i] = inputs[i + 16];
        batch2_out[i] = outputs[i + 16];
      }

      ripemd160avx512_16x32(batch2_in, batch2_out);
    }
  }
}

// Compatibility function with AVX2 version
void ripemd160avx512_8x32(unsigned char *i0, unsigned char *i1, unsigned char *i2,
                           unsigned char *i3, unsigned char *i4, unsigned char *i5,
                           unsigned char *i6, unsigned char *i7, unsigned char *d0,
                           unsigned char *d1, unsigned char *d2, unsigned char *d3,
                           unsigned char *d4, unsigned char *d5, unsigned char *d6,
                           unsigned char *d7) {
  // Pack into arrays for the 16x32 function
  unsigned char *inputs[16] = {i0, i1, i2, i3, i4, i5, i6, i7,
                               // Duplicate inputs to fill all 16 slots
                               i0, i1, i2, i3, i4, i5, i6, i7};

  unsigned char *outputs[16] = {d0, d1, d2, d3, d4, d5, d6, d7,
                                // Dummy outputs for duplicated inputs
                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                nullptr};

  // Use the 16x32 function but only process 8 real messages
  __m512i s[5];
  uint8_t *padded_inputs[16];

  // Allocate aligned memory for padded inputs
  for (int i = 0; i < 8; i++) {
    padded_inputs[i] = (uint8_t *)_mm_malloc(64, 64);
    padded_inputs[i + 8] = (uint8_t *)_mm_malloc(64, 64);

    // Copy input data
    memcpy(padded_inputs[i], inputs[i], 32);
    memcpy(padded_inputs[i + 8], inputs[i], 32);  // Duplicated

    // Add padding and length
    memcpy(padded_inputs[i] + 32, pad, 24);
    memcpy(padded_inputs[i] + 56, &sizedesc_32, 8);
    memcpy(padded_inputs[i + 8] + 32, pad, 24);
    memcpy(padded_inputs[i + 8] + 56, &sizedesc_32, 8);
  }

  // Initialize state
  Initialize(s);

  // Process all blocks
  Transform(s, padded_inputs);

// Extract results to output buffers
#pragma unroll 8
  for (int i = 0; i < 8; i++) {
    DEPACK512(outputs[i], 15 - i);
    _mm_free(padded_inputs[i]);
    _mm_free(padded_inputs[i + 8]);
  }
}

// Stream processing of multiple batches
void ripemd160avx512_stream(unsigned char **inputs, unsigned char **outputs, size_t num_messages) {
  // Get optimal thread count
  int thread_count = GetOptimalThreadCount();

// Set thread affinity for better performance
#pragma omp parallel num_threads(thread_count)
  {
    int thread_id = omp_get_thread_num();
    SetThreadAffinity(thread_id);

#pragma omp for schedule(dynamic, 32)
    for (size_t i = 0; i < num_messages; i += 16) {
      size_t batch_size = std::min((size_t)16, num_messages - i);

      if (batch_size == 16) {
        // Full batch
        unsigned char *batch_in[16];
        unsigned char *batch_out[16];

        for (size_t j = 0; j < 16; j++) {
          batch_in[j] = inputs[i + j];
          batch_out[j] = outputs[i + j];
        }

        ripemd160avx512_16x32(batch_in, batch_out);
      } else {
        // Partial batch - use padding
        unsigned char *batch_in[16];
        unsigned char *batch_out[16];

        for (size_t j = 0; j < batch_size; j++) {
          batch_in[j] = inputs[i + j];
          batch_out[j] = outputs[i + j];
        }

        // Pad with duplicates
        for (size_t j = batch_size; j < 16; j++) {
          batch_in[j] = batch_in[0];
          batch_out[j] = nullptr;  // No output needed
        }

        ripemd160avx512_16x32(batch_in, batch_out);
      }
    }
  }
}

// Prefetch inputs to improve cache usage
void PrefetchInputs(unsigned char **inputs, int count, int hint) {
  for (int i = 0; i < count; i++) {
    _mm_prefetch((const char *)inputs[i], (_mm_hint)hint);
  }
}

// Get optimal thread count for the Xeon Platinum 8488C
int GetOptimalThreadCount() {
  int max_threads = omp_get_max_threads();
  int num_physical_cores = std::thread::hardware_concurrency() / 2;  // Assuming 2 threads per core

  // For RIPEMD160, optimal is usually around 75% of physical cores
  return std::min(max_threads, std::max(1, (int)(num_physical_cores * 0.75)));
}

// Set thread affinity for better performance
void SetThreadAffinity(int thread_id) {
#ifdef _WIN32
  // Windows implementation
  DWORD_PTR affinityMask = 1ULL << (thread_id % 112);  // 8488C has 112 logical cores
  HANDLE currentThread = GetCurrentThread();
  if (!SetThreadAffinityMask(currentThread, affinityMask)) {
    // Failed to set affinity
    DWORD error = GetLastError();
    fprintf(stderr, "Failed to set thread affinity: error code %lu\n", error);
  }
#else
  // Linux/Unix implementation
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  // For Xeon 8488C, spread threads across all available cores
  // but try to keep them on the same socket if possible
  int core_id = thread_id % 112;  // 8488C has 112 logical cores

  // Map thread to specific core
  CPU_SET(core_id, &cpuset);

  pthread_t current_thread = pthread_self();
  int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

  if (result != 0) {
    // Failed to set affinity
    fprintf(stderr, "Failed to set thread affinity: error code %d\n", result);
  }
#endif
}

// Performance monitoring function for AVX-512 operations
inline void MonitorAVX512Performance(int operation_count) {
  static int total_operations = 0;
  static auto start_time = std::chrono::high_resolution_clock::now();

  total_operations += operation_count;

  // Log performance metrics every 1 million operations
  if (total_operations >= 1000000) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    if (duration > 0) {
      double ops_per_ms = total_operations / (double)duration;
// Only log if not in a time-critical section
#pragma omp critical(logging)
      { fprintf(stderr, "AVX-512 Performance: %.2f MOps/s\n", ops_per_ms / 1000.0); }
    }

    // Reset counters
    total_operations = 0;
    start_time = std::chrono::high_resolution_clock::now();
  }
}

// Helper function to determine optimal batch size based on current system load
int GetOptimalBatchSize() {
  // Check current CPU usage
  double cpu_usage = 0.0;

#ifdef _WIN32
  FILETIME idle_time, kernel_time, user_time;
  static FILETIME last_idle_time = {0}, last_kernel_time = {0}, last_user_time = {0};

  if (GetSystemTimes(&idle_time, &kernel_time, &user_time)) {
    ULARGE_INTEGER ul_idle, ul_kernel, ul_user;
    ul_idle.LowPart = idle_time.dwLowDateTime;
    ul_idle.HighPart = idle_time.dwHighDateTime;
    ul_kernel.LowPart = kernel_time.dwLowDateTime;
    ul_kernel.HighPart = kernel_time.dwHighDateTime;
    ul_user.LowPart = user_time.dwLowDateTime;
    ul_user.HighPart = user_time.dwHighDateTime;

    if (last_idle_time.dwLowDateTime != 0) {
      ULARGE_INTEGER ul_last_idle, ul_last_kernel, ul_last_user;
      ul_last_idle.LowPart = last_idle_time.dwLowDateTime;
      ul_last_idle.HighPart = last_idle_time.dwHighDateTime;
      ul_last_kernel.LowPart = last_kernel_time.dwLowDateTime;
      ul_last_kernel.HighPart = last_kernel_time.dwHighDateTime;
      ul_last_user.LowPart = last_user_time.dwLowDateTime;
      ul_last_user.HighPart = last_user_time.dwHighDateTime;

      ULONGLONG idle_diff = ul_idle.QuadPart - ul_last_idle.QuadPart;
      ULONGLONG kernel_diff = ul_kernel.QuadPart - ul_last_kernel.QuadPart;
      ULONGLONG user_diff = ul_user.QuadPart - ul_last_user.QuadPart;
      ULONGLONG total_diff = kernel_diff + user_diff;

      if (total_diff > 0) {
        cpu_usage = 1.0 - ((double)idle_diff / total_diff);
      }
    }

    last_idle_time = idle_time;
    last_kernel_time = kernel_time;
    last_user_time = user_time;
  }
#else
  // Linux implementation
  FILE* file = std::fopen("/proc/stat", "r");
  if (file) {
    static unsigned long long last_user = 0, last_nice = 0, last_system = 0, last_idle = 0;
    static unsigned long long last_iowait = 0, last_irq = 0, last_softirq = 0, last_steal = 0;

    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
    char cpu[10];

    if (std::fscanf(file, "%s %llu %llu %llu %llu %llu %llu %llu %llu", cpu, &user, &nice, &system,
               &idle, &iowait, &irq, &softirq, &steal) >= 9) {
      if (last_user > 0) {
        unsigned long long total_time = (user - last_user) + (nice - last_nice) +
                                        (system - last_system) + (idle - last_idle) +
                                        (iowait - last_iowait) + (irq - last_irq) +
                                        (softirq - last_softirq) + (steal - last_steal);

        unsigned long long idle_time = (idle - last_idle) + (iowait - last_iowait);

        if (total_time > 0) {
          cpu_usage = 1.0 - ((double)idle_time / total_time);
        }
      }

      last_user = user;
      last_nice = nice;
      last_system = system;
      last_idle = idle;
      last_iowait = iowait;
      last_irq = irq;
      last_softirq = softirq;
      last_steal = steal;
    }

    std::fclose(file);
  }
#endif

  // Adjust batch size based on CPU usage
  if (cpu_usage < 0.3) {
    return 64;  // Light load, use larger batches
  } else if (cpu_usage < 0.7) {
    return 32;  // Medium load
  } else {
    return 16;  // Heavy load, use smaller batches
  }
}

// Optimize RIPEMD160 computation for Xeon 8488C with frequency scaling
void OptimizeForFrequencyScaling() {
  // Signal to the CPU that we're about to do intensive AVX-512 work
  // This can help prevent frequency throttling on some Intel CPUs

  // Warm up the CPU with a short burst of AVX-512 instructions
  __m512i dummy = _mm512_set1_epi32(0);
  for (int i = 0; i < 1000; i++) {
    dummy = _mm512_add_epi32(dummy, _mm512_set1_epi32(1));
    _mm_pause();  // Short pause to prevent overheating
  }

  // Force result to be used to prevent optimization
  if (_mm512_reduce_add_epi32(dummy) == 0) {
    // This branch is unlikely to be taken, but prevents the compiler
    // from optimizing away our warm-up code
    fprintf(stderr, "AVX-512 warm-up complete\n");
  }
}

}  // namespace ripemd160avx512