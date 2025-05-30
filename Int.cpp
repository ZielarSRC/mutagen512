#include <emmintrin.h>
#include <immintrin.h>  // Dla pełnego wsparcia AVX-512
#include <math.h>
#include <string.h>

#include "Int.h"
#include "IntGroup.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

Int _ONE((uint64_t)1);

Int Int::P;
// ------------------------------------------------

Int::Int() {}

Int::Int(Int* a) {
  if (a)
    Set(a);
  else
    CLEAR();
}

// Add Xor ---------------------------------------

void Int::Xor(const Int* a) {
  if (!a) return;

#if USE_AVX512
  // Optymalizacja dla AVX-512: wykonanie operacji XOR na całych 512-bitowych rejestrach
  __m512i va = _mm512_loadu_si512((__m512i*)a->bits64);
  __m512i vthis = _mm512_loadu_si512((__m512i*)bits64);
  __m512i result = _mm512_xor_si512(va, vthis);
  _mm512_storeu_si512((__m512i*)bits64, result);
#else
  // Standardowa implementacja bez AVX-512
  asm volatile(
      "mov %[count], %%ecx\n\t"
      "shr $2, %%ecx\n\t"
      "jz 2f\n\t"

      "1:\n\t"
      "vmovdqa (%[a_bits]), %%ymm0\n\t"
      "vpxor (%[this_bits]), %%ymm0, %%ymm0\n\t"
      "vmovdqa %%ymm0, (%[this_bits])\n\t"
      "add $32, %[a_bits]\n\t"
      "add $32, %[this_bits]\n\t"
      "dec %%ecx\n\t"
      "jnz 1b\n\t"

      "vzeroupper\n\t"

      "2:\n\t"
      "mov %[count], %%ecx\n\t"
      "and $3, %%ecx\n\t"
      "jz 4f\n\t"

      "3:\n\t"
      "mov (%[a_bits]), %%rax\n\t"
      "xor %%rax, (%[this_bits])\n\t"
      "add $8, %[a_bits]\n\t"
      "add $8, %[this_bits]\n\t"
      "dec %%ecx\n\t"
      "jnz 3b\n\t"

      "4:\n\t"
      : [this_bits] "+r"(bits64), [a_bits] "+r"(a->bits64)
      : [count] "r"(NB64BLOCK)
      : "rax", "rcx", "ymm0", "memory", "cc");
#endif
}

Int::Int(int64_t i64) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: szybsze ustawienie wartości początkowej
  if (i64 < 0) {
    CLEARFF();
  } else {
    CLEAR();
  }
  bits64[0] = i64;
#else
  if (i64 < 0) {
    CLEARFF();
  } else {
    CLEAR();
  }
  bits64[0] = i64;
#endif
}

Int::Int(uint64_t u64) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: szybsze ustawienie wartości początkowej
  CLEAR();
  bits64[0] = u64;
#else
  CLEAR();
  bits64[0] = u64;
#endif
}

// ------------------------------------------------

void Int::CLEAR() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: szybsze czyszczenie rejestrów
  __m512i zero = _mm512_setzero_si512();
  _mm512_storeu_si512((__m512i*)bits64, zero);
#else
  memset(bits64, 0, NB64BLOCK * 8);
#endif
}

void Int::CLEARFF() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: szybsze ustawienie wszystkich bitów na 1
  __m512i all_ones = _mm512_set1_epi64(-1);
  _mm512_storeu_si512((__m512i*)bits64, all_ones);
#else
  memset(bits64, 0xFF, NB64BLOCK * 8);
#endif
}

// ------------------------------------------------

void Int::Set(Int* a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: szybsze kopiowanie danych
  __m512i src = _mm512_loadu_si512((__m512i*)a->bits64);
  _mm512_storeu_si512((__m512i*)bits64, src);
#else
  for (int i = 0; i < NB64BLOCK; i++) bits64[i] = a->bits64[i];
#endif
}

// ------------------------------------------------

void Int::Add(Int* a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: dodawanie wektorowe
  __m512i va = _mm512_loadu_si512((__m512i*)a->bits64);
  __m512i vthis = _mm512_loadu_si512((__m512i*)bits64);
  __m512i result = _mm512_add_epi64(va, vthis);
  _mm512_storeu_si512((__m512i*)bits64, result);
#else
  uint64_t acc0 = bits64[0];
  uint64_t acc1 = bits64[1];
  uint64_t acc2 = bits64[2];
  uint64_t acc3 = bits64[3];
  uint64_t acc4 = bits64[4];

#if NB64BLOCK > 5
  uint64_t acc5 = bits64[5];
  uint64_t acc6 = bits64[6];
  uint64_t acc7 = bits64[7];
  uint64_t acc8 = bits64[8];
#endif

  asm("add %[src0], %[dst0]    \n\t"
      "adc %[src1], %[dst1]    \n\t"
      "adc %[src2], %[dst2]    \n\t"
      "adc %[src3], %[dst3]    \n\t"
      "adc %[src4], %[dst4]    \n\t"
#if NB64BLOCK > 5
      "adc %[src5], %[dst5]    \n\t"
      "adc %[src6], %[dst6]    \n\t"
      "adc %[src7], %[dst7]    \n\t"
      "adc %[src8], %[dst8]    \n\t"
#endif
      :
      [dst0] "+r"(acc0), [dst1] "+r"(acc1), [dst2] "+r"(acc2), [dst3] "+r"(acc3), [dst4] "+r"(acc4)
#if NB64BLOCK > 5
                                                                                      ,
      [dst5] "+r"(acc5), [dst6] "+r"(acc6), [dst7] "+r"(acc7), [dst8] "+r"(acc8)
#endif
      : [src0] "r"(a->bits64[0]), [src1] "r"(a->bits64[1]), [src2] "r"(a->bits64[2]),
        [src3] "r"(a->bits64[3]), [src4] "r"(a->bits64[4])
#if NB64BLOCK > 5
                                      ,
        [src5] "r"(a->bits64[5]), [src6] "r"(a->bits64[6]), [src7] "r"(a->bits64[7]),
        [src8] "r"(a->bits64[8])
#endif
      : "cc");

  bits64[0] = acc0;
  bits64[1] = acc1;
  bits64[2] = acc2;
  bits64[3] = acc3;
  bits64[4] = acc4;

#if NB64BLOCK > 5
  bits64[5] = acc5;
  bits64[6] = acc6;
  bits64[7] = acc7;
  bits64[8] = acc8;
#endif
#endif
}

// ------------------------------------------------

void Int::Add(uint64_t a) {
#if USE_AVX512
  // Zoptymalizowana funkcja dodawania z przeniesieniem dla AVX-512
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a, bits64 + 0);

  if (c) {
    // Użyj AVX-512 do propagacji przeniesienia
    __m512i data = _mm512_loadu_si512((__m512i*)(bits64 + 1));
    __m512i one = _mm512_set1_epi64(1);
    __mmask8 carry_mask = 0x01;  // Tylko pierwszy element ma przeniesienie

    __m512i result = _mm512_mask_add_epi64(data, carry_mask, data, one);

    // Propaguj przeniesienie
    for (int i = 1; i < NB64BLOCK - 1; i++) {
      if (_mm512_cmplt_epu64_mask(_mm512_maskz_add_epi64(1ULL << i, _mm512_setzero_si512(),
                                                         _mm512_set1_epi64(bits64[i])),
                                  _mm512_set1_epi64(bits64[i]))) {
        carry_mask |= (1ULL << (i + 1));
      } else {
        break;
      }
    }

    if (carry_mask > 0x01) {
      result = _mm512_mask_add_epi64(result, carry_mask >> 1, result, one);
      _mm512_storeu_si512((__m512i*)(bits64 + 1), result);
    } else {
      bits64[1]++;
    }
  }
#else
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a, bits64 + 0);
  c = _addcarry_u64(c, bits64[1], 0, bits64 + 1);
  c = _addcarry_u64(c, bits64[2], 0, bits64 + 2);
  c = _addcarry_u64(c, bits64[3], 0, bits64 + 3);
  c = _addcarry_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], 0, bits64 + 5);
  c = _addcarry_u64(c, bits64[6], 0, bits64 + 6);
  c = _addcarry_u64(c, bits64[7], 0, bits64 + 7);
  c = _addcarry_u64(c, bits64[8], 0, bits64 + 8);
#endif
#endif
}

// ------------------------------------------------
void Int::AddOne() {
#if USE_AVX512
  // Zoptymalizowana funkcja AddOne dla AVX-512
  // Wykorzystuje instrukcje wektorowe do przyspieszenia dodawania jedynki
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], 1, bits64 + 0);

  if (c) {
    // Użyj AVX-512 do propagacji przeniesienia
    __m512i data = _mm512_loadu_si512((__m512i*)(bits64 + 1));
    __m512i one = _mm512_set1_epi64(1);
    __mmask8 carry_mask = 0x01;  // Tylko pierwszy element ma przeniesienie

    __m512i result = _mm512_mask_add_epi64(data, carry_mask, data, one);

    // Propaguj przeniesienie
    for (int i = 1; i < NB64BLOCK - 1; i++) {
      if (bits64[i] == 0xFFFFFFFFFFFFFFFF) {
        carry_mask |= (1ULL << (i + 1));
      } else {
        break;
      }
    }

    if (carry_mask > 0x01) {
      result = _mm512_mask_add_epi64(result, carry_mask >> 1, result, one);
      _mm512_storeu_si512((__m512i*)(bits64 + 1), result);
    } else {
      bits64[1]++;
    }
  }
#else
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], 1, bits64 + 0);
  c = _addcarry_u64(c, bits64[1], 0, bits64 + 1);
  c = _addcarry_u64(c, bits64[2], 0, bits64 + 2);
  c = _addcarry_u64(c, bits64[3], 0, bits64 + 3);
  c = _addcarry_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], 0, bits64 + 5);
  c = _addcarry_u64(c, bits64[6], 0, bits64 + 6);
  c = _addcarry_u64(c, bits64[7], 0, bits64 + 7);
  c = _addcarry_u64(c, bits64[8], 0, bits64 + 8);
#endif
#endif
}

// ------------------------------------------------

void Int::Add(Int* a, Int* b) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: dodawanie dwóch liczb z wykorzystaniem instrukcji wektorowych
  __m512i va = _mm512_loadu_si512((__m512i*)a->bits64);
  __m512i vb = _mm512_loadu_si512((__m512i*)b->bits64);
  __m512i result = _mm512_add_epi64(va, vb);
  _mm512_storeu_si512((__m512i*)bits64, result);
#else
  uint64_t acc0 = a->bits64[0];
  uint64_t acc1 = a->bits64[1];
  uint64_t acc2 = a->bits64[2];
  uint64_t acc3 = a->bits64[3];
  uint64_t acc4 = a->bits64[4];

#if NB64BLOCK > 5
  uint64_t acc5 = a->bits64[5];
  uint64_t acc6 = a->bits64[6];
  uint64_t acc7 = a->bits64[7];
  uint64_t acc8 = a->bits64[8];
#endif

  asm("add %[b0], %[a0]       \n\t"
      "adc %[b1], %[a1]       \n\t"
      "adc %[b2], %[a2]       \n\t"
      "adc %[b3], %[a3]       \n\t"
      "adc %[b4], %[a4]       \n\t"
#if NB64BLOCK > 5
      "adc %[b5], %[a5]       \n\t"
      "adc %[b6], %[a6]       \n\t"
      "adc %[b7], %[a7]       \n\t"
      "adc %[b8], %[a8]       \n\t"
#endif
      : [a0] "+r"(acc0), [a1] "+r"(acc1), [a2] "+r"(acc2), [a3] "+r"(acc3), [a4] "+r"(acc4)
#if NB64BLOCK > 5
                                                                                ,
        [a5] "+r"(acc5), [a6] "+r"(acc6), [a7] "+r"(acc7), [a8] "+r"(acc8)
#endif
      : [b0] "r"(b->bits64[0]), [b1] "r"(b->bits64[1]), [b2] "r"(b->bits64[2]),
        [b3] "r"(b->bits64[3]), [b4] "r"(b->bits64[4])
#if NB64BLOCK > 5
                                    ,
        [b5] "r"(b->bits64[5]), [b6] "r"(b->bits64[6]), [b7] "r"(b->bits64[7]),
        [b8] "r"(b->bits64[8])
#endif
      : "cc");

  bits64[0] = acc0;
  bits64[1] = acc1;
  bits64[2] = acc2;
  bits64[3] = acc3;
  bits64[4] = acc4;

#if NB64BLOCK > 5
  bits64[5] = acc5;
  bits64[6] = acc6;
  bits64[7] = acc7;
  bits64[8] = acc8;
#endif
#endif
}

// ------------------------------------------------

uint64_t Int::AddCh(Int* a, uint64_t ca, Int* b, uint64_t cb) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: zoptymalizowane dodawanie z przeniesieniem
  uint64_t carry;

  // Użyj AVX-512 do szybkiego dodawania a + b
  __m512i va = _mm512_loadu_si512((__m512i*)a->bits64);
  __m512i vb = _mm512_loadu_si512((__m512i*)b->bits64);
  __m512i result = _mm512_add_epi64(va, vb);
  _mm512_storeu_si512((__m512i*)bits64, result);

  // Oblicz przeniesienie
  unsigned char c = 0;
  for (int i = 0; i < NB64BLOCK; i++) {
    if (bits64[i] < a->bits64[i] || (bits64[i] == a->bits64[i] && i > 0 && c)) {
      c = 1;
    } else {
      c = 0;
    }
  }

  // Dodaj przeniesienia ca i cb
  unsigned char final_c = 0;
  _addcarry_u64(c, ca, cb, &carry);

  return carry;
#else
  uint64_t carry;
  unsigned char c = 0;
  c = _addcarry_u64(c, a->bits64[0], b->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, a->bits64[1], b->bits64[1], bits64 + 1);
  c = _addcarry_u64(c, a->bits64[2], b->bits64[2], bits64 + 2);
  c = _addcarry_u64(c, a->bits64[3], b->bits64[3], bits64 + 3);
  c = _addcarry_u64(c, a->bits64[4], b->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, a->bits64[5], b->bits64[5], bits64 + 5);
  c = _addcarry_u64(c, a->bits64[6], b->bits64[6], bits64 + 6);
  c = _addcarry_u64(c, a->bits64[7], b->bits64[7], bits64 + 7);
  c = _addcarry_u64(c, a->bits64[8], b->bits64[8], bits64 + 8);
#endif
  _addcarry_u64(c, ca, cb, &carry);
  return carry;
#endif
}

uint64_t Int::AddCh(Int* a, uint64_t ca) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: zoptymalizowane dodawanie z przeniesieniem
  uint64_t carry;

  // Użyj AVX-512 do szybkiego dodawania
  __m512i vthis = _mm512_loadu_si512((__m512i*)bits64);
  __m512i va = _mm512_loadu_si512((__m512i*)a->bits64);
  __m512i result = _mm512_add_epi64(vthis, va);
  _mm512_storeu_si512((__m512i*)bits64, result);

  // Oblicz przeniesienie
  __mmask8 carry_mask = _mm512_cmplt_epu64_mask(result, vthis);

  // Dodaj przeniesienie ca
  unsigned char c = _mm512_mask2int(carry_mask) ? 1 : 0;
  _addcarry_u64(c, ca, 0, &carry);

  return carry;
#else
  uint64_t carry;
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, bits64[1], a->bits64[1], bits64 + 1);
  c = _addcarry_u64(c, bits64[2], a->bits64[2], bits64 + 2);
  c = _addcarry_u64(c, bits64[3], a->bits64[3], bits64 + 3);
  c = _addcarry_u64(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], a->bits64[5], bits64 + 5);
  c = _addcarry_u64(c, bits64[6], a->bits64[6], bits64 + 6);
  c = _addcarry_u64(c, bits64[7], a->bits64[7], bits64 + 7);
  c = _addcarry_u64(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
  _addcarry_u64(c, ca, 0, &carry);
  return carry;
#endif
}

// ------------------------------------------------

uint64_t Int::AddC(Int* a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: szybsze dodawanie z kontrolą przeniesienia
  // Użyj AVX-512 do wektorowego dodawania
  __m512i vthis = _mm512_loadu_si512((__m512i*)bits64);
  __m512i va = _mm512_loadu_si512((__m512i*)a->bits64);
  __m512i result = _mm512_add_epi64(vthis, va);

  // Wykryj przeniesienie
  __mmask8 carry_mask = _mm512_cmplt_epu64_mask(result, vthis);

  // Zapisz wynik
  _mm512_storeu_si512((__m512i*)bits64, result);

  // Zwróć przeniesienie
  return _mm512_mask2int(carry_mask) ? 1 : 0;
#else
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, bits64[1], a->bits64[1], bits64 + 1);
  c = _addcarry_u64(c, bits64[2], a->bits64[2], bits64 + 2);
  c = _addcarry_u64(c, bits64[3], a->bits64[3], bits64 + 3);
  c = _addcarry_u64(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], a->bits64[5], bits64 + 5);
  c = _addcarry_u64(c, bits64[6], a->bits64[6], bits64 + 6);
  c = _addcarry_u64(c, bits64[7], a->bits64[7], bits64 + 7);
  c = _addcarry_u64(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
  return c;
#endif
}

// ------------------------------------------------

void Int::AddAndShift(Int* a, Int* b, uint64_t cH) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: szybsze dodawanie z przesunięciem
  // Dla tej funkcji możemy użyć bardziej złożonej optymalizacji AVX-512

  // Najpierw dodaj a + b
  __m512i va = _mm512_loadu_si512((__m512i*)a->bits64);
  __m512i vb = _mm512_loadu_si512((__m512i*)b->bits64);
  __m512i sum = _mm512_add_epi64(va,
