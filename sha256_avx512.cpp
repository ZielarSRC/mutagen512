#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "sha256_avx512.h"

namespace _sha256avx512 {

// Use 64-byte alignment for optimal AVX-512 performance
alignas(64) static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// Initialize SHA-256 state with initial hash values (optimized for AVX-512)
inline void Initialize(__m512i* s) {
  s[0] = _mm512_set1_epi32(0x6a09e667);
  s[1] = _mm512_set1_epi32(0xbb67ae85);
  s[2] = _mm512_set1_epi32(0x3c6ef372);
  s[3] = _mm512_set1_epi32(0xa54ff53a);
  s[4] = _mm512_set1_epi32(0x510e527f);
  s[5] = _mm512_set1_epi32(0x9b05688c);
  s[6] = _mm512_set1_epi32(0x1f83d9ab);
  s[7] = _mm512_set1_epi32(0x5be0cd19);
}

// SHA-256 macros with AVX-512 intrinsics for improved performance
#define ROR(x, n) _mm512_or_si512(_mm512_srli_epi32(x, n), _mm512_slli_epi32(x, 32 - (n)))
#define SHR(x, n) _mm512_srli_epi32(x, n)

// Optimized sigma functions using AVX-512 ternary logic
// The _mm512_ternarylogic_epi32 instruction performs arbitrary 3-input bitwise logic
// This is much faster than using multiple XOR/AND/OR operations
#define S0(x) _mm512_ternarylogic_epi32(ROR(x, 2), ROR(x, 13), ROR(x, 22), 0x96)  // XOR of 3 values
#define S1(x) _mm512_ternarylogic_epi32(ROR(x, 6), ROR(x, 11), ROR(x, 25), 0x96)  // XOR of 3 values
#define s0(x) _mm512_ternarylogic_epi32(ROR(x, 7), ROR(x, 18), SHR(x, 3), 0x96)   // XOR of 3 values
#define s1(x) \
  _mm512_ternarylogic_epi32(ROR(x, 17), ROR(x, 19), SHR(x, 10), 0x96)  // XOR of 3 values

// Optimized Ch and Maj functions using AVX-512 ternary logic
// Ch(x,y,z) = (x & y) ^ (~x & z) - equivalent to ternary pattern 0xCA
// Maj(x,y,z) = (x & y) ^ (x & z) ^ (y & z) - equivalent to ternary pattern 0xE8
#define Ch(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0xCA)
#define Maj(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0xE8)

// SHA-256 round function optimized for AVX-512
#define Round(a, b, c, d, e, f, g, h, Kt, Wt)                                                 \
  {                                                                                           \
    __m512i T1 = _mm512_add_epi32(                                                            \
        h, _mm512_add_epi32(S1(e), _mm512_add_epi32(Ch(e, f, g), _mm512_add_epi32(Kt, Wt)))); \
    __m512i T2 = _mm512_add_epi32(S0(a), Maj(a, b, c));                                       \
    h = g;                                                                                    \
    g = f;                                                                                    \
    f = e;                                                                                    \
    e = _mm512_add_epi32(d, T1);                                                              \
    d = c;                                                                                    \
    c = b;                                                                                    \
    b = a;                                                                                    \
    a = _mm512_add_epi32(T1, T2);                                                             \
  }

// Main transformation function that processes 16 message blocks in parallel
inline void Transform(__m512i* state, const uint8_t* data[16]) {
  // Initialize working variables with current state
  __m512i a = state[0], b = state[1], c = state[2], d = state[3];
  __m512i e = state[4], f = state[5], g = state[6], h = state[7];

  // Message schedule array (16 registers, each holding 16 message words)
  alignas(64) __m512i W[16];

// Prepare message schedule W[0..15] - load 16 message blocks
// Each __m512i register holds data from 16 different message blocks
#pragma unroll(4)
  for (int t = 0; t < 16; t += 4) {
#pragma unroll(4)
    for (int i = 0; i < 4; ++i) {
      // Prepare 16 words (one from each message) for this position
      alignas(64) uint32_t wt[16];

#pragma unroll(16)
      for (int j = 0; j < 16; ++j) {
        const uint8_t* ptr = data[j] + (t + i) * 4;
        // Convert from big-endian
        wt[j] =
            ((uint32_t)ptr[0] << 24) | ((uint32_t)ptr[1] << 16) | ((uint32_t)ptr[2] << 8) | ptr[3];
      }

      // Load all 16 words into one AVX-512 register
      W[t + i] = _mm512_loadu_si512((__m512i*)wt);
    }
  }

// Main SHA-256 compression function - 64 rounds
// First 16 rounds use the preloaded message words
#pragma unroll(16)
  for (int t = 0; t < 16; ++t) {
    Round(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[t]), W[t]);
  }

// Remaining 48 rounds with message schedule expansion
#pragma unroll(16)
  for (int t = 16; t < 64; t += 16) {
#pragma unroll(16)
    for (int i = 0; i < 16; ++i) {
      __m512i newW =
          _mm512_add_epi32(_mm512_add_epi32(s1(W[(t + i - 2) & 0xf]), W[(t + i - 7) & 0xf]),
                           _mm512_add_epi32(s0(W[(t + i - 15) & 0xf]), W[(t + i - 16) & 0xf]));
      W[(t + i) & 0xf] = newW;
      Round(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[t + i]), newW);
    }
  }

  // Update state with the processed values
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

// Main SHA-256 hash function for 16 message blocks in parallel
void sha256avx512_16B(const uint8_t* data0, const uint8_t* data1, const uint8_t* data2,
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
  // Ensure proper alignment for AVX-512
  alignas(64) __m512i state[8];

  // Arrays to hold input pointers and output pointers
  const uint8_t* data[16] = {data0, data1, data2,  data3,  data4,  data5,  data6,  data7,
                             data8, data9, data10, data11, data12, data13, data14, data15};

  unsigned char* hashArray[16] = {hash0, hash1, hash2,  hash3,  hash4,  hash5,  hash6,  hash7,
                                  hash8, hash9, hash10, hash11, hash12, hash13, hash14, hash15};

  // Initialize SHA-256 state
  _sha256avx512::Initialize(state);

  // Process all 16 message blocks in parallel
  _sha256avx512::Transform(state, data);

  // Extract and store hash values with proper big-endian conversion
  alignas(64) uint32_t digest[8][16];  // digest[state_index][element_index]

// Store state to aligned memory
#pragma unroll(8)
  for (int i = 0; i < 8; ++i) {
    _mm512_store_si512((__m512i*)digest[i], state[i]);
  }

// Copy to output buffers with endian conversion
#pragma omp parallel for simd
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 8; ++j) {
      uint32_t word = digest[j][i];
// Convert to big-endian
#ifdef _MSC_VER
      word = _byteswap_ulong(word);
#else
      word = __builtin_bswap32(word);
#endif
      // Copy to output
      memcpy(hashArray[i] + j * 4, &word, 4);
    }
  }
}
