#include <immintrin.h>
#include <string.h>

#include "ripemd160_avx512.h"

namespace ripemd160avx512 {

#define ROL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

// Constants
static const uint32_t K0 = 0x00000000;
static const uint32_t K1 = 0x5A827999;
static const uint32_t K2 = 0x6ED9EBA1;
static const uint32_t K3 = 0x8F1BBCDC;
static const uint32_t K4 = 0xA953FD4E;

static const uint32_t KK0 = 0x50A28BE6;
static const uint32_t KK1 = 0x5C4DD124;
static const uint32_t KK2 = 0x6D703EF3;
static const uint32_t KK3 = 0x7A6D76E9;
static const uint32_t KK4 = 0x00000000;

// Functions
#define F(x, y, z) ((x) ^ (y) ^ (z))
#define G(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define H(x, y, z) (((x) | ~(y)) ^ (z))
#define I(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define J(x, y, z) ((x) ^ ((y) | ~(z)))

// These define the indices into the input chunk for each round
static const int r[80] = {0, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
                          7, 4,  13, 1,  10, 6,  15, 3,  12, 0, 9,  5,  2,  14, 11, 8,
                          3, 10, 14, 4,  9,  15, 8,  1,  2,  7, 0,  6,  13, 11, 5,  12,
                          1, 9,  11, 10, 0,  8,  12, 4,  13, 3, 7,  15, 14, 5,  6,  2,
                          4, 0,  5,  9,  7,  12, 2,  10, 14, 1, 3,  8,  11, 6,  15, 13};

static const int rr[80] = {5,  14, 7,  0, 9, 2,  11, 4,  13, 6,  15, 8,  1,  10, 3,  12,
                           6,  11, 3,  7, 0, 13, 5,  10, 14, 15, 8,  12, 4,  9,  1,  2,
                           15, 5,  1,  3, 7, 14, 6,  9,  11, 8,  12, 2,  10, 0,  4,  13,
                           8,  6,  4,  1, 3, 11, 15, 0,  5,  12, 2,  13, 9,  7,  10, 14,
                           12, 15, 10, 4, 1, 5,  8,  7,  6,  2,  13, 14, 0,  3,  9,  11};

static const int s[80] = {11, 14, 15, 12, 5,  8,  7,  9,  11, 13, 14, 15, 6,  7,  9,  8,
                          7,  6,  8,  13, 11, 9,  7,  15, 7,  12, 15, 9,  11, 7,  13, 12,
                          11, 13, 6,  7,  14, 9,  13, 15, 14, 8,  13, 6,  5,  12, 7,  5,
                          11, 12, 14, 15, 14, 15, 9,  8,  9,  14, 5,  6,  8,  6,  5,  12,
                          9,  15, 5,  11, 6,  8,  13, 12, 5,  12, 13, 14, 11, 8,  5,  6};

static const int ss[80] = {8,  9,  9,  11, 13, 15, 15, 5,  7,  7,  8,  11, 14, 14, 12, 6,
                           9,  13, 15, 7,  12, 8,  9,  11, 7,  7,  12, 7,  6,  15, 13, 11,
                           9,  7,  15, 11, 8,  6,  6,  14, 12, 13, 5,  14, 13, 13, 7,  5,
                           15, 5,  8,  11, 14, 14, 6,  14, 6,  9,  12, 9,  12, 5,  15, 8,
                           8,  5,  12, 9,  12, 5,  14, 6,  8,  13, 6,  5,  15, 13, 11, 11};

// Optimized version for AVX-512
#pragma unroll(16)
static inline void compress_block_avx512(__m512i* state, const __m512i* chunk) {
  __m512i a = state[0];
  __m512i b = state[1];
  __m512i c = state[2];
  __m512i d = state[3];
  __m512i e = state[4];
  __m512i aa = state[0];
  __m512i bb = state[1];
  __m512i cc = state[2];
  __m512i dd = state[3];
  __m512i ee = state[4];
  __m512i t;

  // Unpack 16 message blocks from chunk
  __m512i m[16];
  for (int i = 0; i < 16; i++) {
    m[i] = chunk[i];
  }

  // Round 1
  for (int j = 0; j < 16; j++) {
    t = _mm512_add_epi32(a, _mm512_add_epi32(m[r[j]], _mm512_set1_epi32(K0)));
    t = _mm512_add_epi32(ROL32(t, s[j]), _mm512_add_epi32(F(b, c, d), e));
    a = e;
    e = d;
    d = ROL32(c, 10);
    c = b;
    b = t;

    t = _mm512_add_epi32(aa, _mm512_add_epi32(m[rr[j]], _mm512_set1_epi32(KK0)));
    t = _mm512_add_epi32(ROL32(t, ss[j]), _mm512_add_epi32(J(bb, cc, dd), ee));
    aa = ee;
    ee = dd;
    dd = ROL32(cc, 10);
    cc = bb;
    bb = t;
  }

  // Round 2
  for (int j = 16; j < 32; j++) {
    t = _mm512_add_epi32(a, _mm512_add_epi32(m[r[j]], _mm512_set1_epi32(K1)));
    t = _mm512_add_epi32(ROL32(t, s[j]), _mm512_add_epi32(G(b, c, d), e));
    a = e;
    e = d;
    d = ROL32(c, 10);
    c = b;
    b = t;

    t = _mm512_add_epi32(aa, _mm512_add_epi32(m[rr[j]], _mm512_set1_epi32(KK1)));
    t = _mm512_add_epi32(ROL32(t, ss[j]), _mm512_add_epi32(I(bb, cc, dd), ee));
    aa = ee;
    ee = dd;
    dd = ROL32(cc, 10);
    cc = bb;
    bb = t;
  }

  // Round 3
  for (int j = 32; j < 48; j++) {
    t = _mm512_add_epi32(a, _mm512_add_epi32(m[r[j]], _mm512_set1_epi32(K2)));
    t = _mm512_add_epi32(ROL32(t, s[j]), _mm512_add_epi32(H(b, c, d), e));
    a = e;
    e = d;
    d = ROL32(c, 10);
    c = b;
    b = t;

    t = _mm512_add_epi32(aa, _mm512_add_epi32(m[rr[j]], _mm512_set1_epi32(KK2)));
    t = _mm512_add_epi32(ROL32(t, ss[j]), _mm512_add_epi32(H(bb, cc, dd), ee));
    aa = ee;
    ee = dd;
    dd = ROL32(cc, 10);
    cc = bb;
    bb = t;
  }

  // Round 4
  for (int j = 48; j < 64; j++) {
    t = _mm512_add_epi32(a, _mm512_add_epi32(m[r[j]], _mm512_set1_epi32(K3)));
    t = _mm512_add_epi32(ROL32(t, s[j]), _mm512_add_epi32(I(b, c, d), e));
    a = e;
    e = d;
    d = ROL32(c, 10);
    c = b;
    b = t;

    t = _mm512_add_epi32(aa, _mm512_add_epi32(m[rr[j]], _mm512_set1_epi32(KK3)));
    t = _mm512_add_epi32(ROL32(t, ss[j]), _mm512_add_epi32(G(bb, cc, dd), ee));
    aa = ee;
    ee = dd;
    dd = ROL32(cc, 10);
    cc = bb;
    bb = t;
  }

  // Round 5
  for (int j = 64; j < 80; j++) {
    t = _mm512_add_epi32(a, _mm512_add_epi32(m[r[j]], _mm512_set1_epi32(K4)));
    t = _mm512_add_epi32(ROL32(t, s[j]), _mm512_add_epi32(J(b, c, d), e));
    a = e;
    e = d;
    d = ROL32(c, 10);
    c = b;
    b = t;

    t = _mm512_add_epi32(aa, _mm512_add_epi32(m[rr[j]], _mm512_set1_epi32(KK4)));
    t = _mm512_add_epi32(ROL32(t, ss[j]), _mm512_add_epi32(F(bb, cc, dd), ee));
    aa = ee;
    ee = dd;
    dd = ROL32(cc, 10);
    cc = bb;
    bb = t;
  }

  // Combine results
  t = _mm512_add_epi32(state[1], _mm512_add_epi32(c, dd));
  state[1] = _mm512_add_epi32(state[2], _mm512_add_epi32(d, ee));
  state[2] = _mm512_add_epi32(state[3], _mm512_add_epi32(e, aa));
  state[3] = _mm512_add_epi32(state[4], _mm512_add_epi32(a, bb));
  state[4] = _mm512_add_epi32(state[0], _mm512_add_epi32(b, cc));
  state[0] = t;
}

// Poprawione makro DEPACK_AVX512
#define DEPACK_AVX512(d, i)         \
  {                                 \
    alignas(64) uint32_t temp[16];  \
    _mm512_store_epi32(temp, s[0]); \
    ((uint32_t*)d)[0] = temp[i];    \
  }

// Main function to compute RIPEMD-160 for 16 input blocks simultaneously
void ripemd160avx512_16(unsigned char* in0, unsigned char* in1, unsigned char* in2,
                        unsigned char* in3, unsigned char* in4, unsigned char* in5,
                        unsigned char* in6, unsigned char* in7, unsigned char* in8,
                        unsigned char* in9, unsigned char* in10, unsigned char* in11,
                        unsigned char* in12, unsigned char* in13, unsigned char* in14,
                        unsigned char* in15, unsigned char* out0, unsigned char* out1,
                        unsigned char* out2, unsigned char* out3, unsigned char* out4,
                        unsigned char* out5, unsigned char* out6, unsigned char* out7,
                        unsigned char* out8, unsigned char* out9, unsigned char* out10,
                        unsigned char* out11, unsigned char* out12, unsigned char* out13,
                        unsigned char* out14, unsigned char* out15) {
  // Initial hash values
  __m512i state[5];
  state[0] = _mm512_set1_epi32(0x67452301);
  state[1] = _mm512_set1_epi32(0xEFCDAB89);
  state[2] = _mm512_set1_epi32(0x98BADCFE);
  state[3] = _mm512_set1_epi32(0x10325476);
  state[4] = _mm512_set1_epi32(0xC3D2E1F0);

  // Prepare message blocks (16 inputs × 64 bytes each)
  __m512i chunk[16];
  unsigned char* inputs[16] = {in0, in1, in2,  in3,  in4,  in5,  in6,  in7,
                               in8, in9, in10, in11, in12, in13, in14, in15};

  // Process each 64-byte chunk
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      uint32_t values[16];
      for (int k = 0; k < 16; k++) {
        values[k] = ((uint32_t)inputs[k][i * 4 + 0]) | ((uint32_t)inputs[k][i * 4 + 1] << 8) |
                    ((uint32_t)inputs[k][i * 4 + 2] << 16) | ((uint32_t)inputs[k][i * 4 + 3] << 24);
      }
      chunk[i] = _mm512_loadu_si512((__m512i*)values);
    }
    compress_block_avx512(state, chunk);
  }

  // Extract results
  __m512i s[1];
  unsigned char* outputs[16] = {out0, out1, out2,  out3,  out4,  out5,  out6,  out7,
                                out8, out9, out10, out11, out12, out13, out14, out15};

  for (int i = 0; i < 5; i++) {
    s[0] = state[i];
    for (int j = 0; j < 16; j++) {
      DEPACK_AVX512(outputs[j] + i * 4, j);
    }
  }
}

// Version for 64 inputs simultaneously using 4 × 16 blocks
void ripemd160avx512_64(unsigned char* in0, unsigned char* in1, unsigned char* in2,
                        unsigned char* in3, unsigned char* in4, unsigned char* in5,
                        unsigned char* in6, unsigned char* in7, unsigned char* in8,
                        unsigned char* in9, unsigned char* in10, unsigned char* in11,
                        unsigned char* in12, unsigned char* in13, unsigned char* in14,
                        unsigned char* in15, unsigned char* out0, unsigned char* out1,
                        unsigned char* out2, unsigned char* out3, unsigned char* out4,
                        unsigned char* out5, unsigned char* out6, unsigned char* out7,
                        unsigned char* out8, unsigned char* out9, unsigned char* out10,
                        unsigned char* out11, unsigned char* out12, unsigned char* out13,
                        unsigned char* out14, unsigned char* out15) {
#pragma omp parallel sections
  {
#pragma omp section
    ripemd160avx512_16(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13,
                       in14, in15, out0, out1, out2, out3, out4, out5, out6, out7, out8, out9,
                       out10, out11, out12, out13, out14, out15);
  }
}

}  // namespace ripemd160avx512
