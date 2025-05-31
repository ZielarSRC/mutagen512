#include <immintrin.h>

#include <cstdint>
#include <cstring>

#include "ripemd160_avx512.h"

namespace ripemd160avx512 {

#ifdef WIN64
static const __declspec(align(64)) uint32_t _init[] = {
#else
static const uint32_t _init[] __attribute__((aligned(64))) = {
#endif
    // 16 copies of A for AVX-512
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,
    0x67452301ul,

    // 16 copies of B
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,
    0xEFCDAB89ul,

    // 16 copies of C
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,
    0x98BADCFEul,

    // 16 copies of D
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,
    0x10325476ul,

    // 16 copies of E
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul,
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul,
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul,
    0xC3D2E1F0ul};

// AVX-512 operations optimized for Xeon Platinum 8488C
#define _mm512_not_si512(x) _mm512_xor_si512((x), _mm512_set1_epi32(-1))
#define ROL512(x, n) \
  _mm512_or_si512(_mm512_slli_epi32(x, n), _mm512_srli_epi32(x, 32 - n))

// RIPEMD-160 functions for AVX-512
#define f1_512(x, y, z) _mm512_xor_si512(x, _mm512_xor_si512(y, z))
#define f2_512(x, y, z) \
  _mm512_or_si512(_mm512_and_si512(x, y), _mm512_andnot_si512(x, z))
#define f3_512(x, y, z) \
  _mm512_xor_si512(_mm512_or_si512(x, _mm512_not_si512(y)), z)
#define f4_512(x, y, z) \
  _mm512_or_si512(_mm512_and_si512(x, z), _mm512_andnot_si512(z, y))
#define f5_512(x, y, z) \
  _mm512_xor_si512(x, _mm512_or_si512(y, _mm512_not_si512(z)))

// Adding helpers for AVX-512
#define add3_512(x0, x1, x2) _mm512_add_epi32(_mm512_add_epi32(x0, x1), x2)
#define add4_512(x0, x1, x2, x3) \
  _mm512_add_epi32(_mm512_add_epi32(x0, x1), _mm512_add_epi32(x2, x3))

// Round function optimized for AVX-512
#define Round512(a, b, c, d, e, f, x, k, r)    \
  u = add4_512(a, f, x, _mm512_set1_epi32(k)); \
  a = _mm512_add_epi32(ROL512(u, r), e);       \
  c = ROL512(c, 10);

// Macros for each round using AVX-512
#define R11_512(a, b, c, d, e, x, r) \
  Round512(a, b, c, d, e, f1_512(b, c, d), x, 0, r)
#define R21_512(a, b, c, d, e, x, r) \
  Round512(a, b, c, d, e, f2_512(b, c, d), x, 0x5A827999ul, r)
#define R31_512(a, b, c, d, e, x, r) \
  Round512(a, b, c, d, e, f3_512(b, c, d), x, 0x6ED9EBA1ul, r)
#define R41_512(a, b, c, d, e, x, r) \
  Round512(a, b, c, d, e, f4_512(b, c, d), x, 0x8F1BBCDCul, r)
#define R51_512(a, b, c, d, e, x, r) \
  Round512(a, b, c, d, e, f5_512(b, c, d), x, 0xA953FD4Eul, r)
#define R12_512(a, b, c, d, e, x, r) \
  Round512(a, b, c, d, e, f5_512(b, c, d), x, 0x50A28BE6ul, r)
#define R22_512(a, b, c, d, e, x, r) \
  Round512(a, b, c, d, e, f4_512(b, c, d), x, 0x5C4DD124ul, r)
#define R32_512(a, b, c, d, e, x, r) \
  Round512(a, b, c, d, e, f3_512(b, c, d), x, 0x6D703EF3ul, r)
#define R42_512(a, b, c, d, e, x, r) \
  Round512(a, b, c, d, e, f2_512(b, c, d), x, 0x7A6D76E9ul, r)
#define R52_512(a, b, c, d, e, x, r) \
  Round512(a, b, c, d, e, f1_512(b, c, d), x, 0, r)

// Macro to load words from 16 message blocks for AVX-512
#define LOADW512(i)                                                        \
  _mm512_set_epi32(*((uint32_t *)blk[0] + i), *((uint32_t *)blk[1] + i),   \
                   *((uint32_t *)blk[2] + i), *((uint32_t *)blk[3] + i),   \
                   *((uint32_t *)blk[4] + i), *((uint32_t *)blk[5] + i),   \
                   *((uint32_t *)blk[6] + i), *((uint32_t *)blk[7] + i),   \
                   *((uint32_t *)blk[8] + i), *((uint32_t *)blk[9] + i),   \
                   *((uint32_t *)blk[10] + i), *((uint32_t *)blk[11] + i), \
                   *((uint32_t *)blk[12] + i), *((uint32_t *)blk[13] + i), \
                   *((uint32_t *)blk[14] + i), *((uint32_t *)blk[15] + i))

// Initialize state for 16 parallel computations
void Initialize(__m512i *s) { memcpy(s, _init, sizeof(_init)); }

// Transform function processes one block for each of 16 messages
void Transform(__m512i *s, uint8_t *blk[16]) {
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

  // Load message words using optimized AVX-512 operations
  for (int i = 0; i < 16; ++i) {
    w[i] = LOADW512(i);
  }

  // Main rounds 0-15 of RIPEMD-160 optimized for Xeon Platinum 8488C
  R11_512(a1, b1, c1, d1, e1, w[0], 11);
  R12_512(a2, b2, c2, d2, e2, w[5], 8);
  R11_512(e1, a1, b1, c1, d1, w[1], 14);
  R12_512(e2, a2, b2, c2, d2, w[14], 9);
  R11_512(d1, e1, a1, b1, c1, w[2], 15);
  R12_512(d2, e2, a2, b2, c2, w[7], 9);
  R11_512(c1, d1, e1, a1, b1, w[3], 12);
  R12_512(c2, d2, e2, a2, b2, w[0], 11);
  R11_512(b1, c1, d1, e1, a1, w[4], 5);
  R12_512(b2, c2, d2, e2, a2, w[9], 13);
  R11_512(a1, b1, c1, d1, e1, w[5], 8);
  R12_512(a2, b2, c2, d2, e2, w[2], 15);
  R11_512(e1, a1, b1, c1, d1, w[6], 7);
  R12_512(e2, a2, b2, c2, d2, w[11], 15);
  R11_512(d1, e1, a1, b1, c1, w[7], 9);
  R12_512(d2, e2, a2, b2, c2, w[4], 5);
  R11_512(c1, d1, e1, a1, b1, w[8], 11);
  R12_512(c2, d2, e2, a2, b2, w[13], 7);
  R11_512(b1, c1, d1, e1, a1, w[9], 13);
  R12_512(b2, c2, d2, e2, a2, w[6], 7);
  R11_512(a1, b1, c1, d1, e1, w[10], 14);
  R12_512(a2, b2, c2, d2, e2, w[15], 8);
  R11_512(e1, a1, b1, c1, d1, w[11], 15);
  R12_512(e2, a2, b2, c2, d2, w[8], 11);
  R11_512(d1, e1, a1, b1, c1, w[12], 6);
  R12_512(d2, e2, a2, b2, c2, w[1], 14);
  R11_512(c1, d1, e1, a1, b1, w[13], 7);
  R12_512(c2, d2, e2, a2, b2, w[10], 14);
  R11_512(b1, c1, d1, e1, a1, w[14], 9);
  R12_512(b2, c2, d2, e2, a2, w[3], 12);
  R11_512(a1, b1, c1, d1, e1, w[15], 8);
  R12_512(a2, b2, c2, d2, e2, w[12], 6);

  R21_512(e1, a1, b1, c1, d1, w[7], 7);
  R22_512(e2, a2, b2, c2, d2, w[6], 9);
  R21_512(d1, e1, a1, b1, c1, w[4], 6);
  R22_512(d2, e2, a2, b2, c2, w[11], 13);
  R21_512(c1, d1, e1, a1, b1, w[13], 8);
  R22_512(c2, d2, e2, a2, b2, w[3], 15);
  R21_512(b1, c1, d1, e1, a1, w[1], 13);
  R22_512(b2, c2, d2, e2, a2, w[7], 7);
  R21_512(a1, b1, c1, d1, e1, w[10], 11);
  R22_512(a2, b2, c2, d2, e2, w[0], 12);
  R21_512(e1, a1, b1, c1, d1, w[6], 9);
  R22_512(e2, a2, b2, c2, d2, w[13], 8);
  R21_512(d1, e1, a1, b1, c1, w[15], 7);
  R22_512(d2, e2, a2, b2, c2, w[5], 9);
  R21_512(c1, d1, e1, a1, b1, w[3], 15);
  R22_512(c2, d2, e2, a2, b2, w[10], 11);
  R21_512(b1, c1, d1, e1, a1, w[12], 7);
  R22_512(b2, c2, d2, e2, a2, w[14], 7);
  R21_512(a1, b1, c1, d1, e1, w[0], 12);
  R22_512(a2, b2, c2, d2, e2, w[15], 7);
  R21_512(e1, a1, b1, c1, d1, w[9], 15);
  R22_512(e2, a2, b2, c2, d2, w[8], 12);
  R21_512(d1, e1, a1, b1, c1, w[5], 9);
  R22_512(d2, e2, a2, b2, c2, w[12], 7);
  R21_512(c1, d1, e1, a1, b1, w[2], 11);
  R22_512(c2, d2, e2, a2, b2, w[4], 6);
  R21_512(b1, c1, d1, e1, a1, w[14], 7);
  R22_512(b2, c2, d2, e2, a2, w[9], 15);
  R21_512(a1, b1, c1, d1, e1, w[11], 13);
  R22_512(a2, b2, c2, d2, e2, w[1], 13);
  R21_512(e1, a1, b1, c1, d1, w[8], 12);
  R22_512(e2, a2, b2, c2, d2, w[2], 11);

  R31_512(d1, e1, a1, b1, c1, w[3], 11);
  R32_512(d2, e2, a2, b2, c2, w[15], 9);
  R31_512(c1, d1, e1, a1, b1, w[10], 13);
  R32_512(c2, d2, e2, a2, b2, w[5], 7);
  R31_512(b1, c1, d1, e1, a1, w[14], 6);
  R32_512(b2, c2, d2, e2, a2, w[1], 15);
  R31_512(a1, b1, c1, d1, e1, w[4], 7);
  R32_512(a2, b2, c2, d2, e2, w[3], 11);
  R31_512(e1, a1, b1, c1, d1, w[9], 14);
  R32_512(e2, a2, b2, c2, d2, w[7], 8);
  R31_512(d1, e1, a1, b1, c1, w[15], 9);
  R32_512(d2, e2, a2, b2, c2, w[14], 6);
  R31_512(c1, d1, e1, a1, b1, w[8], 13);
  R32_512(c2, d2, e2, a2, b2, w[6], 6);
  R31_512(b1, c1, d1, e1, a1, w[1], 15);
  R32_512(b2, c2, d2, e2, a2, w[9], 14);
  R31_512(a1, b1, c1, d1, e1, w[2], 14);
  R32_512(a2, b2, c2, d2, e2, w[11], 12);
  R31_512(e1, a1, b1, c1, d1, w[7], 8);
  R32_512(e2, a2, b2, c2, d2, w[8], 13);
  R31_512(d1, e1, a1, b1, c1, w[0], 13);
  R32_512(d2, e2, a2, b2, c2, w[12], 5);
  R31_512(c1, d1, e1, a1, b1, w[6], 6);
  R32_512(c2, d2, e2, a2, b2, w[2], 14);
  R31_512(b1, c1, d1, e1, a1, w[13], 5);
  R32_512(b2, c2, d2, e2, a2, w[10], 13);
  R31_512(a1, b1, c1, d1, e1, w[11], 12);
  R32_512(a2, b2, c2, d2, e2, w[0], 13);
  R31_512(e1, a1, b1, c1, d1, w[5], 7);
  R32_512(e2, a2, b2, c2, d2, w[4], 7);
  R31_512(d1, e1, a1, b1, c1, w[12], 5);
  R32_512(d2, e2, a2, b2, c2, w[13], 5);

  R41_512(c1, d1, e1, a1, b1, w[1], 11);
  R42_512(c2, d2, e2, a2, b2, w[8], 15);
  R41_512(b1, c1, d1, e1, a1, w[9], 12);
  R42_512(b2, c2, d2, e2, a2, w[6], 5);
  R41_512(a1, b1, c1, d1, e1, w[11], 14);
  R42_512(a2, b2, c2, d2, e2, w[4], 8);
  R41_512(e1, a1, b1, c1, d1, w[10], 15);
  R42_512(e2, a2, b2, c2, d2, w[1], 11);
  R41_512(d1, e1, a1, b1, c1, w[0], 14);
  R42_512(d2, e2, a2, b2, c2, w[3], 14);
  R41_512(c1, d1, e1, a1, b1, w[8], 15);
  R42_512(c2, d2, e2, a2, b2, w[11], 14);
  R41_512(b1, c1, d1, e1, a1, w[12], 9);
  R42_512(b2, c2, d2, e2, a2, w[15], 6);
  R41_512(a1, b1, c1, d1, e1, w[4], 8);
  R42_512(a2, b2, c2, d2, e2, w[0], 14);
  R41_512(e1, a1, b1, c1, d1, w[13], 9);
  R42_512(e2, a2, b2, c2, d2, w[5], 6);
  R41_512(d1, e1, a1, b1, c1, w[3], 14);
  R42_512(d2, e2, a2, b2, c2, w[12], 9);
  R41_512(c1, d1, e1, a1, b1, w[7], 5);
  R42_512(c2, d2, e2, a2, b2, w[2], 12);
  R41_512(b1, c1, d1, e1, a1, w[15], 6);
  R42_512(b2, c2, d2, e2, a2, w[13], 9);
  R41_512(a1, b1, c1, d1, e1, w[14], 8);
  R42_512(a2, b2, c2, d2, e2, w[9], 12);
  R41_512(e1, a1, b1, c1, d1, w[5], 6);
  R42_512(e2, a2, b2, c2, d2, w[7], 5);
  R41_512(d1, e1, a1, b1, c1, w[6], 5);
  R42_512(d2, e2, a2, b2, c2, w[10], 15);
  R41_512(c1, d1, e1, a1, b1, w[2], 12);
  R42_512(c2, d2, e2, a2, b2, w[14], 8);

  R51_512(b1, c1, d1, e1, a1, w[4], 9);
  R52_512(b2, c2, d2, e2, a2, w[12], 8);
  R51_512(a1, b1, c1, d1, e1, w[0], 15);
  R52_512(a2, b2, c2, d2, e2, w[15], 5);
  R51_512(e1, a1, b1, c1, d1, w[5], 5);
  R52_512(e2, a2, b2, c2, d2, w[10], 12);
  R51_512(d1, e1, a1, b1, c1, w[9], 11);
  R52_512(d2, e2, a2, b2, c2, w[4], 9);
  R51_512(c1, d1, e1, a1, b1, w[7], 6);
  R52_512(c2, d2, e2, a2, b2, w[1], 12);
  R51_512(b1, c1, d1, e1, a1, w[12], 8);
  R52_512(b2, c2, d2, e2, a2, w[5], 5);
  R51_512(a1, b1, c1, d1, e1, w[2], 13);
  R52_512(a2, b2, c2, d2, e2, w[8], 14);
  R51_512(e1, a1, b1, c1, d1, w[10], 12);
  R52_512(e2, a2, b2, c2, d2, w[7], 6);
  R51_512(d1, e1, a1, b1, c1, w[14], 5);
  R52_512(d2, e2, a2, b2, c2, w[6], 8);
  R51_512(c1, d1, e1, a1, b1, w[1], 12);
  R52_512(c2, d2, e2, a2, b2, w[2], 13);
  R51_512(b1, c1, d1, e1, a1, w[3], 13);
  R52_512(b2, c2, d2, e2, a2, w[13], 6);
  R51_512(a1, b1, c1, d1, e1, w[8], 14);
  R52_512(a2, b2, c2, d2, e2, w[14], 5);
  R51_512(e1, a1, b1, c1, d1, w[11], 11);
  R52_512(e2, a2, b2, c2, d2, w[0], 15);
  R51_512(d1, e1, a1, b1, c1, w[6], 8);
  R52_512(d2, e2, a2, b2, c2, w[3], 13);
  R51_512(c1, d1, e1, a1, b1, w[15], 5);
  R52_512(c2, d2, e2, a2, b2, w[9], 11);
  R51_512(b1, c1, d1, e1, a1, w[13], 6);
  R52_512(b2, c2, d2, e2, a2, w[11], 11);

  // Combine results and update state
  __m512i t = s[0];
  s[0] = add3_512(s[1], c1, d2);
  s[1] = add3_512(s[2], d1, e2);
  s[2] = add3_512(s[3], e1, a2);
  s[3] = add3_512(s[4], a1, b2);
  s[4] = add3_512(t, b1, c2);
}

// Optimized DEPACK for AVX-512
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

// Main function to compute RIPEMD-160 hash for 16 messages of 32 bytes each
void ripemd160avx512_32(unsigned char *i0, unsigned char *i1, unsigned char *i2,
                        unsigned char *i3, unsigned char *i4, unsigned char *i5,
                        unsigned char *i6, unsigned char *i7, unsigned char *i8,
                        unsigned char *i9, unsigned char *i10,
                        unsigned char *i11, unsigned char *i12,
                        unsigned char *i13, unsigned char *i14,
                        unsigned char *i15, unsigned char *d0,
                        unsigned char *d1, unsigned char *d2, unsigned char *d3,
                        unsigned char *d4, unsigned char *d5, unsigned char *d6,
                        unsigned char *d7, unsigned char *d8, unsigned char *d9,
                        unsigned char *d10, unsigned char *d11,
                        unsigned char *d12, unsigned char *d13,
                        unsigned char *d14, unsigned char *d15) {
  __m512i s[5];
  uint8_t *bs[] = {i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                   i8, i9, i10, i11, i12, i13, i14, i15};

  // Initialize state for 16 parallel computations
  ripemd160avx512::Initialize(s);

  // Add padding and length for all 16 messages
  for (int i = 0; i < 16; ++i) {
    memcpy(bs[i] + 32, pad, 24);
    memcpy(bs[i] + 56, &sizedesc_32, 8);
  }

  // Process message blocks using optimized AVX-512 transform
  ripemd160avx512::Transform(s, bs);

#ifndef WIN64
  uint32_t *s0 = (uint32_t *)&s[0];
  uint32_t *s1 = (uint32_t *)&s[1];
  uint32_t *s2 = (uint32_t *)&s[2];
  uint32_t *s3 = (uint32_t *)&s[3];
  uint32_t *s4 = (uint32_t *)&s[4];
#endif

  // Unpack the hash values to the output buffers - optimized for Xeon Platinum
  // 8488C
  unsigned char *hashArray[16] = {d0, d1, d2,  d3,  d4,  d5,  d6,  d7,
                                  d8, d9, d10, d11, d12, d13, d14, d15};

  // Optimized unpacking using AVX-512 lanes
  DEPACK512(hashArray[0], 15);
  DEPACK512(hashArray[1], 14);
  DEPACK512(hashArray[2], 13);
  DEPACK512(hashArray[3], 12);
  DEPACK512(hashArray[4], 11);
  DEPACK512(hashArray[5], 10);
  DEPACK512(hashArray[6], 9);
  DEPACK512(hashArray[7], 8);
  DEPACK512(hashArray[8], 7);
  DEPACK512(hashArray[9], 6);
  DEPACK512(hashArray[10], 5);
  DEPACK512(hashArray[11], 4);
  DEPACK512(hashArray[12], 3);
  DEPACK512(hashArray[13], 2);
  DEPACK512(hashArray[14], 1);
  DEPACK512(hashArray[15], 0);
}

// Batch processing function for easier integration with mutagen
void ripemd160avx512_batch(unsigned char **inputs, unsigned char **outputs) {
  ripemd160avx512_32(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                     inputs[5], inputs[6], inputs[7], inputs[8], inputs[9],
                     inputs[10], inputs[11], inputs[12], inputs[13], inputs[14],
                     inputs[15], outputs[0], outputs[1], outputs[2], outputs[3],
                     outputs[4], outputs[5], outputs[6], outputs[7], outputs[8],
                     outputs[9], outputs[10], outputs[11], outputs[12],
                     outputs[13], outputs[14], outputs[15]);
}

}  // namespace ripemd160avx512
