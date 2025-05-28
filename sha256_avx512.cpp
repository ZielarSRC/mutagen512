#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "sha256_avx512.h"

namespace _sha256avx512 {

#define ALIGN64 __attribute__((aligned(64)))

// Initialize SHA-256 state with initial hash values for 16 parallel
// computations
void Initialize(__m512i* s) {
  const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

  for (int i = 0; i < 8; ++i) {
    s[i] = _mm512_set1_epi32(init[i]);
  }
}

// SHA-256 macros optimized for AVX-512
#define Maj(x, y, z)                      \
  _mm512_or_si512(_mm512_and_si512(x, y), \
                  _mm512_and_si512(z, _mm512_or_si512(x, y)))
#define Ch(x, y, z) \
  _mm512_xor_si512(_mm512_and_si512(x, y), _mm512_andnot_si512(x, z))
#define ROR(x, n) \
  _mm512_or_si512(_mm512_srli_epi32(x, n), _mm512_slli_epi32(x, 32 - n))
#define SHR(x, n) _mm512_srli_epi32(x, n)

#define S0(x) \
  (_mm512_xor_si512(ROR(x, 2), _mm512_xor_si512(ROR(x, 13), ROR(x, 22))))
#define S1(x) \
  (_mm512_xor_si512(ROR(x, 6), _mm512_xor_si512(ROR(x, 11), ROR(x, 25))))
#define s0(x) \
  (_mm512_xor_si512(ROR(x, 7), _mm512_xor_si512(ROR(x, 18), SHR(x, 3))))
#define s1(x) \
  (_mm512_xor_si512(ROR(x, 17), _mm512_xor_si512(ROR(x, 19), SHR(x, 10))))

#define Round(a, b, c, d, e, f, g, h, Kt, Wt)                             \
  T1 = _mm512_add_epi32(                                                  \
      _mm512_add_epi32(                                                   \
          _mm512_add_epi32(_mm512_add_epi32(h, S1(e)), Ch(e, f, g)), Kt), \
      Wt);                                                                \
  T2 = _mm512_add_epi32(S0(a), Maj(a, b, c));                             \
  h = g;                                                                  \
  g = f;                                                                  \
  f = e;                                                                  \
  e = _mm512_add_epi32(d, T1);                                            \
  d = c;                                                                  \
  c = b;                                                                  \
  b = a;                                                                  \
  a = _mm512_add_epi32(T1, T2);

void Transform(__m512i* state, const uint8_t* data[16]) {
  __m512i a, b, c, d, e, f, g, h;
  __m512i W[64];
  __m512i T1, T2;

  // Load state into local variables
  a = state[0];
  b = state[1];
  c = state[2];
  d = state[3];
  e = state[4];
  f = state[5];
  g = state[6];
  h = state[7];

  // Prepare message schedule W[0..15] for 16 parallel computations
  for (int t = 0; t < 16; ++t) {
    uint32_t wt[16];
    for (int i = 0; i < 16; ++i) {
      const uint8_t* ptr = data[i] + t * 4;
      wt[i] = ((uint32_t)ptr[0] << 24) | ((uint32_t)ptr[1] << 16) |
              ((uint32_t)ptr[2] << 8) | ((uint32_t)ptr[3]);
    }
    W[t] = _mm512_setr_epi32(wt[0], wt[1], wt[2], wt[3], wt[4], wt[5], wt[6],
                             wt[7], wt[8], wt[9], wt[10], wt[11], wt[12],
                             wt[13], wt[14], wt[15]);
  }

  // Message schedule extension with optimized AVX-512 operations
  for (int t = 16; t < 64; ++t) {
    W[t] = _mm512_add_epi32(_mm512_add_epi32(s1(W[t - 2]), W[t - 7]),
                            _mm512_add_epi32(s0(W[t - 15]), W[t - 16]));
  }

  // SHA-256 round constants
  static const uint32_t K[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
      0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

  // Main compression loop with 64 rounds
  for (int t = 0; t < 64; ++t) {
    __m512i Kt = _mm512_set1_epi32(K[t]);
    Round(a, b, c, d, e, f, g, h, Kt, W[t]);
  }

  // Add compressed chunk to current hash value
  state[0] = _mm512_add_epi32(state[0], a);
  state[1] = _mm512_add_epi32(state[1], b);
  state[2] = _mm512_add_epi32(state[2], c);
  state[3] = _mm512_add_epi32(state[3], d);
  state[4] = _mm512_add_epi32(state[4], e);
  state[5] = _mm512_add_epi32(state[5], f);
  state[6] = _mm512_add_epi32(state[6], g);
  state[7] = _mm512_add_epi32(state[7], h);
}

}  // namespace _sha256avx512

void sha256avx512_16B(
    const uint8_t* data0, const uint8_t* data1, const uint8_t* data2,
    const uint8_t* data3, const uint8_t* data4, const uint8_t* data5,
    const uint8_t* data6, const uint8_t* data7, const uint8_t* data8,
    const uint8_t* data9, const uint8_t* data10, const uint8_t* data11,
    const uint8_t* data12, const uint8_t* data13, const uint8_t* data14,
    const uint8_t* data15, unsigned char* hash0, unsigned char* hash1,
    unsigned char* hash2, unsigned char* hash3, unsigned char* hash4,
    unsigned char* hash5, unsigned char* hash6, unsigned char* hash7,
    unsigned char* hash8, unsigned char* hash9, unsigned char* hash10,
    unsigned char* hash11, unsigned char* hash12, unsigned char* hash13,
    unsigned char* hash14, unsigned char* hash15) {
  __m512i state[8];

  // Initialize with SHA-256 initial hash values
  _sha256avx512::Initialize(state);

  const uint8_t* data[16] = {data0,  data1,  data2,  data3, data4,  data5,
                             data6,  data7,  data8,  data9, data10, data11,
                             data12, data13, data14, data15};

  // Process 16 data blocks in parallel
  _sha256avx512::Transform(state, data);

  // Extract and store results with optimal memory layout
  ALIGN64 uint32_t digest[8][16];  // digest[state_index][element_index]

  for (int i = 0; i < 8; ++i) {
    _mm512_store_si512((__m512i*)digest[i], state[i]);
  }

  unsigned char* hashArray[16] = {hash0,  hash1,  hash2,  hash3, hash4,  hash5,
                                  hash6,  hash7,  hash8,  hash9, hash10, hash11,
                                  hash12, hash13, hash14, hash15};

  // Extract hash values with byte order conversion for little-endian output
  for (int i = 0; i < 16; ++i) {
    unsigned char* hash = hashArray[i];
    for (int j = 0; j < 8; ++j) {
      uint32_t word = digest[j][i];
      word = __builtin_bswap32(word);
      memcpy(hash + j * 4, &word, 4);
    }
  }
}
