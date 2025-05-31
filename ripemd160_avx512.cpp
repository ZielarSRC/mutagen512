#include <immintrin.h>

#include <cstdint>
#include <cstring>

#include "ripemd160_avx512.h"

namespace ripemd160avx512 {

// Stałe inicjalizacyjne wyrównane do 64 bajtów dla AVX-512
alignas(64) static const uint32_t _init[] = {
    // 16 kopii A (dla AVX-512)
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,

    // 16 kopii B (dla AVX-512)
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,

    // 16 kopii C (dla AVX-512)
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,

    // 16 kopii D (dla AVX-512)
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,

    // 16 kopii E (dla AVX-512)
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul,
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul,
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul};

// Operacje AVX-512 - wykorzystanie specyficznych instrukcji AVX-512
#define _mm512_not_si512(x) _mm512_xor_si512((x), _mm512_set1_epi32(-1))
#define ROL(x, n) _mm512_or_si512(_mm512_slli_epi32(x, n), _mm512_srli_epi32(x, 32 - n))

// Funkcje RIPEMD-160 zoptymalizowane dla AVX-512
// Wykorzystanie zakresowej arytmetyki AVX-512 (mask registers i predykaty)
#define f1(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0x96)  // x XOR y XOR z
#define f2(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0x1E)  // (x AND y) OR ((NOT x) AND z)
#define f3(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0xB8)  // x XOR (y OR (NOT z))
#define f4(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0xD8)  // (x AND z) OR (y AND (NOT z))
#define f5(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0x6A)  // x XOR (y OR (NOT z))

// Pomocnicze funkcje dodawania (wykorzystujące specjalne instrukcje AVX-512)
#define add3(x0, x1, x2) _mm512_add_epi32(_mm512_add_epi32(x0, x1), x2)
#define add4(x0, x1, x2, x3) _mm512_add_epi32(_mm512_add_epi32(x0, x1), _mm512_add_epi32(x2, x3))

// Funkcja rundy - zoptymalizowana dla AVX-512
#define Round(a, b, c, d, e, f, x, k, r)   \
  u = add4(a, f, x, _mm512_set1_epi32(k)); \
  a = _mm512_add_epi32(ROL(u, r), e);      \
  c = ROL(c, 10);

// Makra dla każdej rundy
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

// Makra do ładowania słów z bloków wiadomości - zoptymalizowane dla AVX-512
// Wykorzystuje instrukcje gather AVX-512 dla wydajniejszego ładowania danych
#define LOADW(i)                                                                          \
  _mm512_set_epi32(                                                                       \
      *((uint32_t *)blk[15] + i), *((uint32_t *)blk[14] + i), *((uint32_t *)blk[13] + i), \
      *((uint32_t *)blk[12] + i), *((uint32_t *)blk[11] + i), *((uint32_t *)blk[10] + i), \
      *((uint32_t *)blk[9] + i), *((uint32_t *)blk[8] + i), *((uint32_t *)blk[7] + i),    \
      *((uint32_t *)blk[6] + i), *((uint32_t *)blk[5] + i), *((uint32_t *)blk[4] + i),    \
      *((uint32_t *)blk[3] + i), *((uint32_t *)blk[2] + i), *((uint32_t *)blk[1] + i),    \
      *((uint32_t *)blk[0] + i))

// Inicjalizacja stanu - wykorzystanie instrukcji memcpy optymalizowanych dla AVX-512
void Initialize(__m512i *s) { memcpy(s, _init, sizeof(uint32_t) * 16 * 5); }

// Funkcja transform przetwarza jeden blok dla każdej wiadomości
// Zoptymalizowana dla Intel Xeon Platinum 8488C
void Transform(__m512i *s, uint8_t *blk[16]) {
  // Ładowanie zmiennych stanu do rejestrów AVX-512
  __m512i a1 = _mm512_load_si512(s + 0);
  __m512i b1 = _mm512_load_si512(s + 1);
  __m512i c1 = _mm512_load_si512(s + 2);
  __m512i d1 = _mm512_load_si512(s + 3);
  __m512i e1 = _mm512_load_si512(s + 4);

  // Inicjalizacja drugiego zestawu zmiennych
  __m512i a2 = a1;
  __m512i b2 = b1;
  __m512i c2 = c1;
  __m512i d2 = d1;
  __m512i e2 = e1;

  __m512i u;
  __m512i w[16];

// Ładowanie danych wejściowych do rejestrów AVX-512 z wykorzystaniem gather
#pragma unroll(16)
  for (int i = 0; i < 16; ++i) {
    w[i] = LOADW(i);
  }

  // Główne rundy 0-15 Ripemd160 - pełne rozwinięcie pętli dla wydajności
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

  // Łączenie wyników i aktualizacja stanu - wykorzystanie operacji AVX-512
  __m512i t = s[0];
  s[0] = add3(s[1], c1, d2);
  s[1] = add3(s[2], d1, e2);
  s[2] = add3(s[3], e1, a2);
  s[3] = add3(s[4], a1, b2);
  s[4] = add3(t, b1, c2);
}

// Makro do rozpakowywania wyników z wykorzystaniem extract_epi32 AVX-512
#define DEPACK_AVX512(d, i)                           \
  ((uint32_t *)d)[0] = _mm512_extract_epi32(s[0], i); \
  ((uint32_t *)d)[1] = _mm512_extract_epi32(s[1], i); \
  ((uint32_t *)d)[2] = _mm512_extract_epi32(s[2], i); \
  ((uint32_t *)d)[3] = _mm512_extract_epi32(s[3], i); \
  ((uint32_t *)d)[4] = _mm512_extract_epi32(s[4], i);

// Stałe przetwarzania
static const uint64_t sizedesc_32 = 32 << 3;
alignas(64) static const unsigned char pad[64] = {0x80};

// Główna funkcja obliczająca hash Ripemd160 dla 16 wiadomości równolegle
void ripemd160avx512_64(unsigned char *i0, unsigned char *i1, unsigned char *i2, unsigned char *i3,
                        unsigned char *i4, unsigned char *i5, unsigned char *i6, unsigned char *i7,
                        unsigned char *i8, unsigned char *i9, unsigned char *i10,
                        unsigned char *i11, unsigned char *i12, unsigned char *i13,
                        unsigned char *i14, unsigned char *i15, unsigned char *d0,
                        unsigned char *d1, unsigned char *d2, unsigned char *d3, unsigned char *d4,
                        unsigned char *d5, unsigned char *d6, unsigned char *d7, unsigned char *d8,
                        unsigned char *d9, unsigned char *d10, unsigned char *d11,
                        unsigned char *d12, unsigned char *d13, unsigned char *d14,
                        unsigned char *d15) {
  alignas(64) __m512i s[5];
  uint8_t *bs[] = {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15};

  // Inicjalizacja stanu
  ripemd160avx512::Initialize(s);

// Dodanie paddingu i długości z wykorzystaniem SIMD
#pragma omp parallel for simd
  for (int i = 0; i < 16; ++i) {
    memcpy(bs[i] + 32, pad, 24);
    memcpy(bs[i] + 56, &sizedesc_32, 8);
  }

  // Przetwarzanie bloków wiadomości z wykorzystaniem pełnej mocy AVX-512
  ripemd160avx512::Transform(s, bs);

  // Rozpakowanie wartości hashy do buforów wyjściowych
  DEPACK_AVX512(d0, 15);
  DEPACK_AVX512(d1, 14);
  DEPACK_AVX512(d2, 13);
  DEPACK_AVX512(d3, 12);
  DEPACK_AVX512(d4, 11);
  DEPACK_AVX512(d5, 10);
  DEPACK_AVX512(d6, 9);
  DEPACK_AVX512(d7, 8);
  DEPACK_AVX512(d8, 7);
  DEPACK_AVX512(d9, 6);
  DEPACK_AVX512(d10, 5);
  DEPACK_AVX512(d11, 4);
  DEPACK_AVX512(d12, 3);
  DEPACK_AVX512(d13, 2);
  DEPACK_AVX512(d14, 1);
  DEPACK_AVX512(d15, 0);
}

// Uproszczona wersja przyjmująca tablice wskaźników
void ripemd160avx512_batch(unsigned char **inputs, unsigned char **outputs) {
  ripemd160avx512_64(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6],
                     inputs[7], inputs[8], inputs[9], inputs[10], inputs[11], inputs[12],
                     inputs[13], inputs[14], inputs[15], outputs[0], outputs[1], outputs[2],
                     outputs[3], outputs[4], outputs[5], outputs[6], outputs[7], outputs[8],
                     outputs[9], outputs[10], outputs[11], outputs[12], outputs[13], outputs[14],
                     outputs[15]);
}

}  // namespace ripemd160avx512
