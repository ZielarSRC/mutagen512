#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <x86intrin.h>

#include "sha256_avx512.h"

namespace _sha256avx512 {

#define ALIGN64 __attribute__((aligned(64)))

static const ALIGN64 uint32_t K[64] = {
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

// AVX-512 SHA-256 rotation and shift operations
#define ROR512(x, n) \
  _mm512_or_si512(_mm512_srli_epi32(x, n), _mm512_slli_epi32(x, 32 - (n)))
#define SHR512(x, n) _mm512_srli_epi32(x, n)

// SHA-256 functions optimized for AVX-512
#define S0_512(x)                 \
  (_mm512_xor_si512(ROR512(x, 2), \
                    _mm512_xor_si512(ROR512(x, 13), ROR512(x, 22))))
#define S1_512(x)                 \
  (_mm512_xor_si512(ROR512(x, 6), \
                    _mm512_xor_si512(ROR512(x, 11), ROR512(x, 25))))
#define s0_512(x)                 \
  (_mm512_xor_si512(ROR512(x, 7), \
                    _mm512_xor_si512(ROR512(x, 18), SHR512(x, 3))))
#define s1_512(x)                  \
  (_mm512_xor_si512(ROR512(x, 17), \
                    _mm512_xor_si512(ROR512(x, 19), SHR512(x, 10))))

// Optimized logical functions using ternary logic (Sapphire Rapids feature)
#define Ch512(x, y, z) \
  _mm512_ternarylogic_epi32(x, y, z, 0xCA)  // (x & y) ^ (~x & z)
#define Maj512(x, y, z) \
  _mm512_ternarylogic_epi32(x, y, z, 0xE8)  // (x & y) ^ (x & z) ^ (y & z)

// SHA-256 round function
#define Round512(a, b, c, d, e, f, g, h, Kt, Wt)                             \
  {                                                                          \
    __m512i T1 = _mm512_add_epi32(                                           \
        h, _mm512_add_epi32(                                                 \
               S1_512(e),                                                    \
               _mm512_add_epi32(Ch512(e, f, g), _mm512_add_epi32(Kt, Wt)))); \
    __m512i T2 = _mm512_add_epi32(S0_512(a), Maj512(a, b, c));               \
    h = g;                                                                   \
    g = f;                                                                   \
    f = e;                                                                   \
    e = _mm512_add_epi32(d, T1);                                             \
    d = c;                                                                   \
    c = b;                                                                   \
    b = a;                                                                   \
    a = _mm512_add_epi32(T1, T2);                                            \
  }

// Main transformation function for 16-way parallel SHA-256
inline void Transform(__m512i* state, const uint8_t* data[16]) {
  __m512i a = state[0], b = state[1], c = state[2], d = state[3];
  __m512i e = state[4], f = state[5], g = state[6], h = state[7];

  ALIGN64 __m512i W[16];

  // Prepare message schedule W[0..15] - load and convert endianness
  for (int t = 0; t < 16; ++t) {
    ALIGN64 uint32_t wt[16];
    for (int i = 0; i < 16; ++i) {
      const uint8_t* ptr = data[i] + t * 4;
      wt[i] = ((uint32_t)ptr[0] << 24) | ((uint32_t)ptr[1] << 16) |
              ((uint32_t)ptr[2] << 8) | ptr[3];
    }
    W[t] = _mm512_load_si512((__m512i*)wt);
  }

  // **KOMPLETNE** 64 rundy SHA-256 - każda runda oddzielnie

  // Rundy 0-15 (używamy W[0] do W[15])
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[0]), W[0]);
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[1]), W[1]);
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[2]), W[2]);
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[3]), W[3]);
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[4]), W[4]);
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[5]), W[5]);
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[6]), W[6]);
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[7]), W[7]);
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[8]), W[8]);
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[9]), W[9]);
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[10]), W[10]);
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[11]), W[11]);
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[12]), W[12]);
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[13]), W[13]);
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[14]), W[14]);
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[15]), W[15]);

  // Rundy 16-31 - obliczamy nowe W wartości
  __m512i W16 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W[14]), W[9]),
                                 _mm512_add_epi32(s0_512(W[1]), W[0]));
  W[0] = W16;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[16]), W16);

  __m512i W17 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W[15]), W[10]),
                                 _mm512_add_epi32(s0_512(W[2]), W[1]));
  W[1] = W17;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[17]), W17);

  __m512i W18 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W16), W[11]),
                                 _mm512_add_epi32(s0_512(W[3]), W[2]));
  W[2] = W18;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[18]), W18);

  __m512i W19 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W17), W[12]),
                                 _mm512_add_epi32(s0_512(W[4]), W[3]));
  W[3] = W19;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[19]), W19);

  __m512i W20 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W18), W[13]),
                                 _mm512_add_epi32(s0_512(W[5]), W[4]));
  W[4] = W20;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[20]), W20);

  __m512i W21 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W19), W[14]),
                                 _mm512_add_epi32(s0_512(W[6]), W[5]));
  W[5] = W21;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[21]), W21);

  __m512i W22 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W20), W[15]),
                                 _mm512_add_epi32(s0_512(W[7]), W[6]));
  W[6] = W22;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[22]), W22);

  __m512i W23 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W21), W16),
                                 _mm512_add_epi32(s0_512(W[8]), W[7]));
  W[7] = W23;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[23]), W23);

  __m512i W24 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W22), W17),
                                 _mm512_add_epi32(s0_512(W[9]), W[8]));
  W[8] = W24;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[24]), W24);

  __m512i W25 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W23), W18),
                                 _mm512_add_epi32(s0_512(W[10]), W[9]));
  W[9] = W25;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[25]), W25);

  __m512i W26 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W24), W19),
                                 _mm512_add_epi32(s0_512(W[11]), W[10]));
  W[10] = W26;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[26]), W26);

  __m512i W27 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W25), W20),
                                 _mm512_add_epi32(s0_512(W[12]), W[11]));
  W[11] = W27;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[27]), W27);

  __m512i W28 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W26), W21),
                                 _mm512_add_epi32(s0_512(W[13]), W[12]));
  W[12] = W28;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[28]), W28);

  __m512i W29 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W27), W22),
                                 _mm512_add_epi32(s0_512(W[14]), W[13]));
  W[13] = W29;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[29]), W29);

  __m512i W30 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W28), W23),
                                 _mm512_add_epi32(s0_512(W[15]), W[14]));
  W[14] = W30;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[30]), W30);

  __m512i W31 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W29), W24),
                                 _mm512_add_epi32(s0_512(W16), W[15]));
  W[15] = W31;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[31]), W31);

  // Rundy 32-47
  __m512i W32 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W30), W25),
                                 _mm512_add_epi32(s0_512(W17), W16));
  W[0] = W32;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[32]), W32);

  __m512i W33 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W31), W26),
                                 _mm512_add_epi32(s0_512(W18), W17));
  W[1] = W33;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[33]), W33);

  __m512i W34 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W32), W27),
                                 _mm512_add_epi32(s0_512(W19), W18));
  W[2] = W34;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[34]), W34);

  __m512i W35 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W33), W28),
                                 _mm512_add_epi32(s0_512(W20), W19));
  W[3] = W35;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[35]), W35);

  __m512i W36 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W34), W29),
                                 _mm512_add_epi32(s0_512(W21), W20));
  W[4] = W36;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[36]), W36);

  __m512i W37 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W35), W30),
                                 _mm512_add_epi32(s0_512(W22), W21));
  W[5] = W37;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[37]), W37);

  __m512i W38 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W36), W31),
                                 _mm512_add_epi32(s0_512(W23), W22));
  W[6] = W38;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[38]), W38);

  __m512i W39 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W37), W32),
                                 _mm512_add_epi32(s0_512(W24), W23));
  W[7] = W39;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[39]), W39);

  __m512i W40 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W38), W33),
                                 _mm512_add_epi32(s0_512(W25), W24));
  W[8] = W40;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[40]), W40);

  __m512i W41 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W39), W34),
                                 _mm512_add_epi32(s0_512(W26), W25));
  W[9] = W41;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[41]), W41);

  __m512i W42 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W40), W35),
                                 _mm512_add_epi32(s0_512(W27), W26));
  W[10] = W42;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[42]), W42);

  __m512i W43 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W41), W36),
                                 _mm512_add_epi32(s0_512(W28), W27));
  W[11] = W43;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[43]), W43);

  __m512i W44 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W42), W37),
                                 _mm512_add_epi32(s0_512(W29), W28));
  W[12] = W44;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[44]), W44);

  __m512i W45 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W43), W38),
                                 _mm512_add_epi32(s0_512(W30), W29));
  W[13] = W45;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[45]), W45);

  __m512i W46 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W44), W39),
                                 _mm512_add_epi32(s0_512(W31), W30));
  W[14] = W46;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[46]), W46);

  __m512i W47 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W45), W40),
                                 _mm512_add_epi32(s0_512(W32), W31));
  W[15] = W47;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[47]), W47);

  // Rundy 48-63 (ostatnie)
  __m512i W48 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W46), W41),
                                 _mm512_add_epi32(s0_512(W33), W32));
  W[0] = W48;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[48]), W48);

  __m512i W49 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W47), W42),
                                 _mm512_add_epi32(s0_512(W34), W33));
  W[1] = W49;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[49]), W49);

  __m512i W50 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W48), W43),
                                 _mm512_add_epi32(s0_512(W35), W34));
  W[2] = W50;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[50]), W50);

  __m512i W51 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W49), W44),
                                 _mm512_add_epi32(s0_512(W36), W35));
  W[3] = W51;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[51]), W51);

  __m512i W52 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W50), W45),
                                 _mm512_add_epi32(s0_512(W37), W36));
  W[4] = W52;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[52]), W52);

  __m512i W53 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W51), W46),
                                 _mm512_add_epi32(s0_512(W38), W37));
  W[5] = W53;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[53]), W53);

  __m512i W54 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W52), W47),
                                 _mm512_add_epi32(s0_512(W39), W38));
  W[6] = W54;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[54]), W54);

  __m512i W55 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W53), W48),
                                 _mm512_add_epi32(s0_512(W40), W39));
  W[7] = W55;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[55]), W55);

  __m512i W56 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W54), W49),
                                 _mm512_add_epi32(s0_512(W41), W40));
  W[8] = W56;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[56]), W56);

  __m512i W57 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W55), W50),
                                 _mm512_add_epi32(s0_512(W42), W41));
  W[9] = W57;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[57]), W57);

  __m512i W58 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W56), W51),
                                 _mm512_add_epi32(s0_512(W43), W42));
  W[10] = W58;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[58]), W58);

  __m512i W59 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W57), W52),
                                 _mm512_add_epi32(s0_512(W44), W43));
  W[11] = W59;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[59]), W59);

  __m512i W60 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W58), W53),
                                 _mm512_add_epi32(s0_512(W45), W44));
  W[12] = W60;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[60]), W60);

  __m512i W61 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W59), W54),
                                 _mm512_add_epi32(s0_512(W46), W45));
  W[13] = W61;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[61]), W61);

  __m512i W62 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W60), W55),
                                 _mm512_add_epi32(s0_512(W47), W46));
  W[14] = W62;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[62]), W62);

  __m512i W63 = _mm512_add_epi32(_mm512_add_epi32(s1_512(W61), W56),
                                 _mm512_add_epi32(s0_512(W48), W47));
  W[15] = W63;
  Round512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[63]), W63);

  // Add computed values to state
  state[0] = _mm512_add_epi32(state[0], a);
  state[1] = _mm512_add_epi32(state[1], b);
  state[2] = _mm512_add_epi32(state[2], c);
  state[3] = _mm512_add_epi32(state[3], d);
  state[4] = _mm512_add_epi32(state[4], e);
  state[5] = _mm512_add_epi32(state[5], f);
  state[6] = _mm512_add_epi32(state[6], g);
  state[7] = _mm512_add_epi32(state[7], h);
}

inline void ExtractHashes(__m512i* state, unsigned char* hashArray[16]) {
  ALIGN64 uint32_t digest[8][16];

  for (int i = 0; i < 8; ++i) {
    _mm512_store_si512((__m512i*)digest[i], state[i]);
  }

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 8; ++j) {
      uint32_t word = __builtin_bswap32(digest[j][i]);
      *((uint32_t*)(hashArray[i] + j * 4)) = word;
    }
  }
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
  ALIGN64 __m512i state[8];
  const uint8_t* data[16] = {data0,  data1,  data2,  data3, data4,  data5,
                             data6,  data7,  data8,  data9, data10, data11,
                             data12, data13, data14, data15};
  unsigned char* hashArray[16] = {hash0,  hash1,  hash2,  hash3, hash4,  hash5,
                                  hash6,  hash7,  hash8,  hash9, hash10, hash11,
                                  hash12, hash13, hash14, hash15};

  _sha256avx512::Initialize(state);
  _sha256avx512::Transform(state, data);
  _sha256avx512::ExtractHashes(state, hashArray);
}
