#include <immintrin.h>
#include <string.h>

#include "sha256_avx512.h"

namespace sha256avx512 {

// Constants for SHA-256
static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// Helper functions
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

// Funkcja pomocnicza do pakowania/rozpakowywania danych
#define DEPACK_AVX512(d, i)         \
  {                                 \
    alignas(64) uint32_t temp[16];  \
    _mm512_store_epi32(temp, s[0]); \
    ((uint32_t*)d)[0] = temp[i];    \
  }

// Forward declaration
static void compress_block_avx512(__m512i* state, const __m512i* chunk);

// Main function to compute SHA-256 for 16 input blocks simultaneously
void sha256avx512_16(unsigned char* in0, unsigned char* in1, unsigned char* in2, unsigned char* in3,
                     unsigned char* in4, unsigned char* in5, unsigned char* in6, unsigned char* in7,
                     unsigned char* in8, unsigned char* in9, unsigned char* in10,
                     unsigned char* in11, unsigned char* in12, unsigned char* in13,
                     unsigned char* in14, unsigned char* in15, unsigned char* out0,
                     unsigned char* out1, unsigned char* out2, unsigned char* out3,
                     unsigned char* out4, unsigned char* out5, unsigned char* out6,
                     unsigned char* out7, unsigned char* out8, unsigned char* out9,
                     unsigned char* out10, unsigned char* out11, unsigned char* out12,
                     unsigned char* out13, unsigned char* out14, unsigned char* out15) {
  // Initial hash values
  alignas(64) __m512i state[8];
  state[0] = _mm512_set1_epi32(0x6a09e667);
  state[1] = _mm512_set1_epi32(0xbb67ae85);
  state[2] = _mm512_set1_epi32(0x3c6ef372);
  state[3] = _mm512_set1_epi32(0xa54ff53a);
  state[4] = _mm512_set1_epi32(0x510e527f);
  state[5] = _mm512_set1_epi32(0x9b05688c);
  state[6] = _mm512_set1_epi32(0x1f83d9ab);
  state[7] = _mm512_set1_epi32(0x5be0cd19);

  // Prepare message blocks (16 inputs × 64 bytes each)
  alignas(64) __m512i chunk[16];
  unsigned char* inputs[16] = {in0, in1, in2,  in3,  in4,  in5,  in6,  in7,
                               in8, in9, in10, in11, in12, in13, in14, in15};

  // Process each 64-byte chunk
  for (int i = 0; i < 16; i++) {
    alignas(64) uint32_t values[16];
    for (int j = 0; j < 16; j++) {
      values[j] = ((uint32_t)inputs[j][i * 4 + 0] << 24) | ((uint32_t)inputs[j][i * 4 + 1] << 16) |
                  ((uint32_t)inputs[j][i * 4 + 2] << 8) | ((uint32_t)inputs[j][i * 4 + 3]);
    }
    chunk[i] = _mm512_loadu_si512((__m512i*)values);
  }

  compress_block_avx512(state, chunk);

  // Extract results
  alignas(64) __m512i s[1];
  unsigned char* outputs[16] = {out0, out1, out2,  out3,  out4,  out5,  out6,  out7,
                                out8, out9, out10, out11, out12, out13, out14, out15};

  for (int i = 0; i < 8; i++) {
    s[0] = state[i];
    for (int j = 0; j < 16; j++) {
      alignas(64) uint32_t temp[16];
      _mm512_store_epi32(temp, s[0]);
      uint32_t val = temp[j];
      outputs[j][i * 4 + 0] = (val >> 24) & 0xFF;
      outputs[j][i * 4 + 1] = (val >> 16) & 0xFF;
      outputs[j][i * 4 + 2] = (val >> 8) & 0xFF;
      outputs[j][i * 4 + 3] = val & 0xFF;
    }
  }
}

// Version for 64 inputs simultaneously using OpenMP for parallelization
void sha256avx512_64(unsigned char* in0, unsigned char* in1, unsigned char* in2, unsigned char* in3,
                     unsigned char* in4, unsigned char* in5, unsigned char* in6, unsigned char* in7,
                     unsigned char* in8, unsigned char* in9, unsigned char* in10,
                     unsigned char* in11, unsigned char* in12, unsigned char* in13,
                     unsigned char* in14, unsigned char* in15, unsigned char* out0,
                     unsigned char* out1, unsigned char* out2, unsigned char* out3,
                     unsigned char* out4, unsigned char* out5, unsigned char* out6,
                     unsigned char* out7, unsigned char* out8, unsigned char* out9,
                     unsigned char* out10, unsigned char* out11, unsigned char* out12,
                     unsigned char* out13, unsigned char* out14, unsigned char* out15) {
#pragma omp parallel sections
  {
#pragma omp section
    sha256avx512_16(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14,
                    in15, out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11,
                    out12, out13, out14, out15);
  }
}

// Implementation of the compress_block_avx512 function
static void compress_block_avx512(__m512i* state, const __m512i* chunk) {
  alignas(64) __m512i a = state[0];
  alignas(64) __m512i b = state[1];
  alignas(64) __m512i c = state[2];
  alignas(64) __m512i d = state[3];
  alignas(64) __m512i e = state[4];
  alignas(64) __m512i f = state[5];
  alignas(64) __m512i g = state[6];
  alignas(64) __m512i h = state[7];
  alignas(64) __m512i w[64];
  alignas(64) __m512i temp1, temp2, temp3;

  // Copy first 16 words from the message
  for (int i = 0; i < 16; i++) {
    w[i] = chunk[i];
  }

  // Extend the first 16 words into the remaining 48 words
  for (int i = 16; i < 64; i++) {
    temp1 = w[i - 15];
    temp2 = w[i - 2];

    // SIG0(w[i-15])
    temp1 = _mm512_xor_si512(
        _mm512_xor_si512(_mm512_srli_epi32(w[i - 15], 7), _mm512_slli_epi32(w[i - 15], 32 - 7)),
        _mm512_xor_si512(_mm512_srli_epi32(w[i - 15], 18), _mm512_slli_epi32(w[i - 15], 32 - 18)));
    temp1 = _mm512_xor_si512(temp1, _mm512_srli_epi32(w[i - 15], 3));

    // SIG1(w[i-2])
    temp2 = _mm512_xor_si512(
        _mm512_xor_si512(_mm512_srli_epi32(w[i - 2], 17), _mm512_slli_epi32(w[i - 2], 32 - 17)),
        _mm512_xor_si512(_mm512_srli_epi32(w[i - 2], 19), _mm512_slli_epi32(w[i - 2], 32 - 19)));
    temp2 = _mm512_xor_si512(temp2, _mm512_srli_epi32(w[i - 2], 10));

    // w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(w[i-15]) + w[i-16]
    w[i] = _mm512_add_epi32(temp2, w[i - 7]);
    w[i] = _mm512_add_epi32(w[i], temp1);
    w[i] = _mm512_add_epi32(w[i], w[i - 16]);
  }

  // Main loop
  for (int i = 0; i < 64; i++) {
    // temp1 = h + EP1(e) + CH(e,f,g) + K[i] + w[i]

    // EP1(e)
    temp1 =
        _mm512_xor_si512(_mm512_xor_si512(_mm512_srli_epi32(e, 6), _mm512_slli_epi32(e, 32 - 6)),
                         _mm512_xor_si512(_mm512_srli_epi32(e, 11), _mm512_slli_epi32(e, 32 - 11)));
    temp1 = _mm512_xor_si512(
        temp1, _mm512_xor_si512(_mm512_srli_epi32(e, 25), _mm512_slli_epi32(e, 32 - 25)));

    // CH(e,f,g)
    temp2 = _mm512_and_si512(e, f);
    temp3 = _mm512_andnot_si512(e, g);
    temp2 = _mm512_xor_si512(temp2, temp3);

    // temp1 = h + EP1(e) + CH(e,f,g) + K[i] + w[i]
    temp1 = _mm512_add_epi32(h, temp1);
    temp1 = _mm512_add_epi32(temp1, temp2);
    temp1 = _mm512_add_epi32(temp1, _mm512_set1_epi32(K[i]));
    temp1 = _mm512_add_epi32(temp1, w[i]);

    // temp2 = EP0(a) + MAJ(a,b,c)

    // EP0(a)
    temp2 =
        _mm512_xor_si512(_mm512_xor_si512(_mm512_srli_epi32(a, 2), _mm512_slli_epi32(a, 32 - 2)),
                         _mm512_xor_si512(_mm512_srli_epi32(a, 13), _mm512_slli_epi32(a, 32 - 13)));
    temp2 = _mm512_xor_si512(
        temp2, _mm512_xor_si512(_mm512_srli_epi32(a, 22), _mm512_slli_epi32(a, 32 - 22)));

    // MAJ(a,b,c)
    temp3 = _mm512_or_si512(_mm512_and_si512(a, b), _mm512_and_si512(a, c));
    temp3 = _mm512_or_si512(temp3, _mm512_and_si512(b, c));

    temp2 = _mm512_add_epi32(temp2, temp3);

    h = g;
    g = f;
    f = e;
    e = _mm512_add_epi32(d, temp1);
    d = c;
    c = b;
    b = a;
    a = _mm512_add_epi32(temp1, temp2);
  }

  // Add the compressed chunk to the current hash value
  state[0] = _mm512_add_epi32(state[0], a);
  state[1] = _mm512_add_epi32(state[1], b);
  state[2] = _mm512_add_epi32(state[2], c);
  state[3] = _mm512_add_epi32(state[3], d);
  state[4] = _mm512_add_epi32(state[4], e);
  state[5] = _mm512_add_epi32(state[5], f);
  state[6] = _mm512_add_epi32(state[6], g);
  state[7] = _mm512_add_epi32(state[7], h);
}

}  // namespace sha256avx512
