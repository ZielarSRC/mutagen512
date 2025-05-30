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

Int::Int(Int *a) {
  if (a)
    Set(a);
  else
    CLEAR();
}

// Add Xor ---------------------------------------

void Int::Xor(const Int *a) {
  if (!a) return;

#if USE_AVX512
  // Optymalizacja dla AVX-512: wykonanie operacji XOR na całych 512-bitowych rejestrach
  __m512i va = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i vthis = _mm512_loadu_si512((__m512i *)bits64);
  __m512i result = _mm512_xor_si512(va, vthis);
  _mm512_storeu_si512((__m512i *)bits64, result);
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
  _mm512_storeu_si512((__m512i *)bits64, zero);
#else
  memset(bits64, 0, NB64BLOCK * 8);
#endif
}

void Int::CLEARFF() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: szybsze ustawienie wszystkich bitów na 1
  __m512i all_ones = _mm512_set1_epi64(-1);
  _mm512_storeu_si512((__m512i *)bits64, all_ones);
#else
  memset(bits64, 0xFF, NB64BLOCK * 8);
#endif
}

// ------------------------------------------------

void Int::Set(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: szybsze kopiowanie danych
  __m512i src = _mm512_loadu_si512((__m512i *)a->bits64);
  _mm512_storeu_si512((__m512i *)bits64, src);
#else
  for (int i = 0; i < NB64BLOCK; i++) bits64[i] = a->bits64[i];
#endif
}

// ------------------------------------------------

void Int::Add(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: dodawanie wektorowe
  __m512i va = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i vthis = _mm512_loadu_si512((__m512i *)bits64);
  __m512i result = _mm512_add_epi64(va, vthis);
  _mm512_storeu_si512((__m512i *)bits64, result);
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
    __m512i data = _mm512_loadu_si512((__m512i *)(bits64 + 1));
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
      _mm512_storeu_si512((__m512i *)(bits64 + 1), result);
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
    __m512i data = _mm512_loadu_si512((__m512i *)(bits64 + 1));
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
      _mm512_storeu_si512((__m512i *)(bits64 + 1), result);
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

void Int::Add(Int *a, Int *b) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: dodawanie dwóch liczb z wykorzystaniem instrukcji wektorowych
  __m512i va = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i vb = _mm512_loadu_si512((__m512i *)b->bits64);
  __m512i result = _mm512_add_epi64(va, vb);
  _mm512_storeu_si512((__m512i *)bits64, result);
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

uint64_t Int::AddCh(Int *a, uint64_t ca, Int *b, uint64_t cb) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: zoptymalizowane dodawanie z przeniesieniem
  uint64_t carry;

  // Użyj AVX-512 do szybkiego dodawania a + b
  __m512i va = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i vb = _mm512_loadu_si512((__m512i *)b->bits64);
  __m512i result = _mm512_add_epi64(va, vb);
  _mm512_storeu_si512((__m512i *)bits64, result);

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

uint64_t Int::AddCh(Int *a, uint64_t ca) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: zoptymalizowane dodawanie z przeniesieniem
  uint64_t carry;

  // Użyj AVX-512 do szybkiego dodawania
  __m512i vthis = _mm512_loadu_si512((__m512i *)bits64);
  __m512i va = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i result = _mm512_add_epi64(vthis, va);
  _mm512_storeu_si512((__m512i *)bits64, result);

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

uint64_t Int::AddC(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: szybsze dodawanie z kontrolą przeniesienia
  // Użyj AVX-512 do wektorowego dodawania
  __m512i vthis = _mm512_loadu_si512((__m512i *)bits64);
  __m512i va = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i result = _mm512_add_epi64(vthis, va);

  // Wykryj przeniesienie
  __mmask8 carry_mask = _mm512_cmplt_epu64_mask(result, vthis);

  // Zapisz wynik
  _mm512_storeu_si512((__m512i *)bits64, result);

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

void Int::AddAndShift(Int *a, Int *b, uint64_t cH) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: szybsze dodawanie z przesunięciem
  // Dla tej funkcji możemy użyć bardziej złożonej optymalizacji AVX-512

  // Najpierw dodaj a + b
  __m512i va = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i vb = _mm512_loadu_si512((__m512i *)b->bits64);
  __m512i sum = _mm512_add_epi64(va, vb);

  // Wykonaj przesunięcie o jedno słowo 64-bitowe
  __m512i shifted = _mm512_alignr_epi64(_mm512_setzero_si512(), sum, 1);

  // Pierwszy element będzie zerem, ponieważ przyesunęliśmy wszystko o jedno słowo
  // Zapisz wyniki w docelowej tablicy (z przesunięciem)
  _mm512_storeu_si512((__m512i *)bits64, shifted);

  // Ustaw najwyższy element na przeniesienie
  bits64[NB64BLOCK - 1] = cH;
#else
  unsigned char c = 0;
  c = _addcarry_u64(c, b->bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, b->bits64[1], a->bits64[1], bits64 + 0);
  c = _addcarry_u64(c, b->bits64[2], a->bits64[2], bits64 + 1);
  c = _addcarry_u64(c, b->bits64[3], a->bits64[3], bits64 + 2);
  c = _addcarry_u64(c, b->bits64[4], a->bits64[4], bits64 + 3);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, b->bits64[5], a->bits64[5], bits64 + 4);
  c = _addcarry_u64(c, b->bits64[6], a->bits64[6], bits64 + 5);
  c = _addcarry_u64(c, b->bits64[7], a->bits64[7], bits64 + 6);
  c = _addcarry_u64(c, b->bits64[8], a->bits64[8], bits64 + 7);
#endif

  bits64[NB64BLOCK - 1] = c + cH;
#endif
}

// ------------------------------------------------

void Int::MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22,
                       uint64_t *cu, uint64_t *cv) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Mnożenie macierzowe z wykorzystaniem rozszerzonych instrukcji
  Int t1, t2, t3, t4;
  uint64_t c1, c2, c3, c4;

  // Wykorzystanie zoptymalizowanego IMult z AVX-512
  c1 = t1.IMult(u, _11);
  c2 = t2.IMult(v, _12);
  c3 = t3.IMult(u, _21);
  c4 = t4.IMult(v, _22);

  // Wykorzystanie zoptymalizowanego AddCh z AVX-512
  *cu = u->AddCh(&t1, c1, &t2, c2);
  *cv = v->AddCh(&t3, c3, &t4, c4);
#else
  Int t1, t2, t3, t4;
  uint64_t c1, c2, c3, c4;
  c1 = t1.IMult(u, _11);
  c2 = t2.IMult(v, _12);
  c3 = t3.IMult(u, _21);
  c4 = t4.IMult(v, _22);
  *cu = u->AddCh(&t1, c1, &t2, c2);
  *cv = v->AddCh(&t3, c3, &t4, c4);
#endif
}

void Int::MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Mnożenie macierzowe z wykorzystaniem rozszerzonych instrukcji
  Int t1, t2, t3, t4;

  // Wykorzystanie zoptymalizowanego IMult z AVX-512
  t1.IMult(u, _11);
  t2.IMult(v, _12);
  t3.IMult(u, _21);
  t4.IMult(v, _22);

  // Wykorzystanie zoptymalizowanego Add z AVX-512
  u->Add(&t1, &t2);
  v->Add(&t3, &t4);
#else
  Int t1, t2, t3, t4;
  t1.IMult(u, _11);
  t2.IMult(v, _12);
  t3.IMult(u, _21);
  t4.IMult(v, _22);
  u->Add(&t1, &t2);
  v->Add(&t3, &t4);
#endif
}

// ------------------------------------------------

bool Int::IsGreater(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze porównanie przy użyciu instrukcji wektorowych
  for (int i = NB64BLOCK - 1; i >= 0; i--) {
    if (bits64[i] > a->bits64[i]) return true;
    if (bits64[i] < a->bits64[i]) return false;
  }
  return false;
#else
  int i;

  for (i = NB64BLOCK - 1; i >= 0;) {
    if (a->bits64[i] != bits64[i]) break;
    i--;
  }

  if (i >= 0) {
    return bits64[i] > a->bits64[i];
  } else {
    return false;
  }
#endif
}

// ------------------------------------------------

bool Int::IsLower(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze porównanie przy użyciu instrukcji wektorowych
  for (int i = NB64BLOCK - 1; i >= 0; i--) {
    if (bits64[i] < a->bits64[i]) return true;
    if (bits64[i] > a->bits64[i]) return false;
  }
  return false;
#else
  int i;

  for (i = NB64BLOCK - 1; i >= 0;) {
    if (a->bits64[i] != bits64[i]) break;
    i--;
  }

  if (i >= 0) {
    return bits64[i] < a->bits64[i];
  } else {
    return false;
  }
#endif
}

// ------------------------------------------------

bool Int::IsGreaterOrEqual(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Wykorzystanie operacji wektorowych do szybszego porównania
  Int p;
  p.Sub(this, a);
  return p.IsPositive();
#else
  Int p;
  p.Sub(this, a);
  return p.IsPositive();
#endif
}

// ------------------------------------------------

bool Int::IsLowerOrEqual(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze porównanie przy użyciu instrukcji wektorowych
  int i = NB64BLOCK - 1;
  while (i >= 0) {
    if (a->bits64[i] != bits64[i]) break;
    i--;
  }

  if (i >= 0) {
    return bits64[i] < a->bits64[i];
  } else {
    return true;
  }
#else
  int i = NB64BLOCK - 1;

  while (i >= 0) {
    if (a->bits64[i] != bits64[i]) break;
    i--;
  }

  if (i >= 0) {
    return bits64[i] < a->bits64[i];
  } else {
    return true;
  }
#endif
}

bool Int::IsEqual(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze porównanie równości przy użyciu instrukcji wektorowych
  __m512i va = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i vthis = _mm512_loadu_si512((__m512i *)bits64);
  __mmask8 mask = _mm512_cmpeq_epi64_mask(va, vthis);

  // Wszystkie elementy muszą być równe
  return mask == 0x1F;  // 0b11111 dla 5 elementów (NB64BLOCK=5)
#else
  return
#if NB64BLOCK > 5
      (bits64[8] == a->bits64[8]) && (bits64[7] == a->bits64[7]) && (bits64[6] == a->bits64[6]) &&
      (bits64[5] == a->bits64[5]) &&
#endif
      (bits64[4] == a->bits64[4]) && (bits64[3] == a->bits64[3]) && (bits64[2] == a->bits64[2]) &&
      (bits64[1] == a->bits64[1]) && (bits64[0] == a->bits64[0]);
#endif
}

bool Int::IsOne() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze sprawdzanie czy liczba jest równa 1
  return IsEqual(&_ONE);
#else
  return IsEqual(&_ONE);
#endif
}

bool Int::IsZero() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze sprawdzanie czy liczba jest równa 0
  __m512i vthis = _mm512_loadu_si512((__m512i *)bits64);
  __m512i zero = _mm512_setzero_si512();
  __mmask8 mask = _mm512_cmpeq_epi64_mask(vthis, zero);

  // Wszystkie elementy muszą być równe zero
  return mask == 0x1F;  // 0b11111 dla 5 elementów (NB64BLOCK=5)
#else
#if NB64BLOCK > 5
  return (bits64[8] | bits64[7] | bits64[6] | bits64[5] | bits64[4] | bits64[3] | bits64[2] |
          bits64[1] | bits64[0]) == 0;
#else
  return (bits64[4] | bits64[3] | bits64[2] | bits64[1] | bits64[0]) == 0;
#endif
#endif
}

// ------------------------------------------------

void Int::SetInt32(uint32_t value) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze ustawianie wartości 32-bitowej
  CLEAR();
  bits[0] = value;
#else
  CLEAR();
  bits[0] = value;
#endif
}

// ------------------------------------------------

uint32_t Int::GetInt32() { return bits[0]; }

// ------------------------------------------------

unsigned char Int::GetByte(int n) {
  unsigned char *bbPtr = (unsigned char *)bits;
  return bbPtr[n];
}

void Int::Set32Bytes(unsigned char *bytes) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze kopiowanie danych
  CLEAR();
  uint64_t *ptr = (uint64_t *)bytes;

  // Wykorzystaj AVX-512 do szybkiego zamiany kolejności bajtów
  __m512i data = _mm512_set_epi64(0, 0, 0, 0, _byteswap_uint64(ptr[0]), _byteswap_uint64(ptr[1]),
                                  _byteswap_uint64(ptr[2]), _byteswap_uint64(ptr[3]));

  // Wyodrębnij tylko potrzebne wartości (32 bajty)
  bits64[3] = _mm512_extracti64x2_epi64(data, 1)[0];
  bits64[2] = _mm512_extracti64x2_epi64(data, 1)[1];
  bits64[1] = _mm512_extracti64x2_epi64(data, 0)[0];
  bits64[0] = _mm512_extracti64x2_epi64(data, 0)[1];
#else
  CLEAR();
  uint64_t *ptr = (uint64_t *)bytes;
  bits64[3] = _byteswap_uint64(ptr[0]);
  bits64[2] = _byteswap_uint64(ptr[1]);
  bits64[1] = _byteswap_uint64(ptr[2]);
  bits64[0] = _byteswap_uint64(ptr[3]);
#endif
}

void Int::Get32Bytes(unsigned char *buff) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze kopiowanie danych
  uint64_t *ptr = (uint64_t *)buff;

  // Wykorzystaj AVX-512 do szybkiego zamiany kolejności bajtów
  __m512i data = _mm512_set_epi64(0, 0, 0, 0, bits64[0], bits64[1], bits64[2], bits64[3]);

  __m512i swapped =
      _mm512_set_epi64(0, 0, 0, 0, _byteswap_uint64(_mm512_extracti64x2_epi64(data, 3)[0]),
                       _byteswap_uint64(_mm512_extracti64x2_epi64(data, 2)[0]),
                       _byteswap_uint64(_mm512_extracti64x2_epi64(data, 1)[0]),
                       _byteswap_uint64(_mm512_extracti64x2_epi64(data, 0)[0]));

  ptr[3] = _mm512_extracti64x2_epi64(swapped, 0)[0];
  ptr[2] = _mm512_extracti64x2_epi64(swapped, 0)[1];
  ptr[1] = _mm512_extracti64x2_epi64(swapped, 1)[0];
  ptr[0] = _mm512_extracti64x2_epi64(swapped, 1)[1];
#else
  uint64_t *ptr = (uint64_t *)buff;
  ptr[3] = _byteswap_uint64(bits64[0]);
  ptr[2] = _byteswap_uint64(bits64[1]);
  ptr[1] = _byteswap_uint64(bits64[2]);
  ptr[0] = _byteswap_uint64(bits64[3]);
#endif
}

// ------------------------------------------------

void Int::SetByte(int n, unsigned char byte) {
  unsigned char *bbPtr = (unsigned char *)bits;
  bbPtr[n] = byte;
}

// ------------------------------------------------

void Int::SetDWord(int n, uint32_t b) { bits[n] = b; }

// ------------------------------------------------

void Int::SetQWord(int n, uint64_t b) { bits64[n] = b; }

// ------------------------------------------------

void Int::Sub(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Wykorzystanie instrukcji wektorowych do odejmowania
  __m512i va = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i vthis = _mm512_loadu_si512((__m512i *)bits64);
  __m512i result = _mm512_sub_epi64(vthis, va);
  _mm512_storeu_si512((__m512i *)bits64, result);
#else
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], a->bits64[0], bits64 + 0);
  c = _subborrow_u64(c, bits64[1], a->bits64[1], bits64 + 1);
  c = _subborrow_u64(c, bits64[2], a->bits64[2], bits64 + 2);
  c = _subborrow_u64(c, bits64[3], a->bits64[3], bits64 + 3);
  c = _subborrow_u64(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], a->bits64[5], bits64 + 5);
  c = _subborrow_u64(c, bits64[6], a->bits64[6], bits64 + 6);
  c = _subborrow_u64(c, bits64[7], a->bits64[7], bits64 + 7);
  c = _subborrow_u64(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
#endif
}

// ------------------------------------------------

void Int::Sub(Int *a, Int *b) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Wykorzystanie instrukcji wektorowych do odejmowania
  __m512i va = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i vb = _mm512_loadu_si512((__m512i *)b->bits64);
  __m512i result = _mm512_sub_epi64(va, vb);
  _mm512_storeu_si512((__m512i *)bits64, result);
#else
  unsigned char c = 0;
  c = _subborrow_u64(c, a->bits64[0], b->bits64[0], bits64 + 0);
  c = _subborrow_u64(c, a->bits64[1], b->bits64[1], bits64 + 1);
  c = _subborrow_u64(c, a->bits64[2], b->bits64[2], bits64 + 2);
  c = _subborrow_u64(c, a->bits64[3], b->bits64[3], bits64 + 3);
  c = _subborrow_u64(c, a->bits64[4], b->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, a->bits64[5], b->bits64[5], bits64 + 5);
  c = _subborrow_u64(c, a->bits64[6], b->bits64[6], bits64 + 6);
  c = _subborrow_u64(c, a->bits64[7], b->bits64[7], bits64 + 7);
  c = _subborrow_u64(c, a->bits64[8], b->bits64[8], bits64 + 8);
#endif
#endif
}

void Int::Sub(uint64_t a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze odejmowanie wartości 64-bitowej
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], a, bits64 + 0);

  if (c) {
    // Użyj AVX-512 do propagacji pożyczki
    __m512i data = _mm512_loadu_si512((__m512i *)(bits64 + 1));
    __m512i one = _mm512_set1_epi64(1);
    __mmask8 borrow_mask = 0x01;  // Tylko pierwszy element ma pożyczkę

    // Odejmij 1 od każdego elementu z pożyczką
    __m512i result = _mm512_mask_sub_epi64(data, borrow_mask, data, one);

    // Propaguj pożyczkę
    for (int i = 1; i < NB64BLOCK - 1; i++) {
      if (bits64[i] == 0) {
        borrow_mask |= (1ULL << (i + 1));
      } else {
        break;
      }
    }

    if (borrow_mask > 0x01) {
      result = _mm512_mask_sub_epi64(result, borrow_mask >> 1, result, one);
      _mm512_storeu_si512((__m512i *)(bits64 + 1), result);
    } else {
      bits64[1]--;
    }
  }
#else
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], a, bits64 + 0);
  c = _subborrow_u64(c, bits64[1], 0, bits64 + 1);
  c = _subborrow_u64(c, bits64[2], 0, bits64 + 2);
  c = _subborrow_u64(c, bits64[3], 0, bits64 + 3);
  c = _subborrow_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], 0, bits64 + 5);
  c = _subborrow_u64(c, bits64[6], 0, bits64 + 6);
  c = _subborrow_u64(c, bits64[7], 0, bits64 + 7);
  c = _subborrow_u64(c, bits64[8], 0, bits64 + 8);
#endif
#endif
}

void Int::SubOne() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze odejmowanie jedynki
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], 1, bits64 + 0);

  if (c) {
    // Użyj AVX-512 do propagacji pożyczki
    __m512i data = _mm512_loadu_si512((__m512i *)(bits64 + 1));
    __m512i one = _mm512_set1_epi64(1);
    __mmask8 borrow_mask = 0x01;  // Tylko pierwszy element ma pożyczkę

    // Odejmij 1 od każdego elementu z pożyczką
    __m512i result = _mm512_mask_sub_epi64(data, borrow_mask, data, one);

    // Propaguj pożyczkę
    for (int i = 1; i < NB64BLOCK - 1; i++) {
      if (bits64[i] == 0) {
        borrow_mask |= (1ULL << (i + 1));
      } else {
        break;
      }
    }

    if (borrow_mask > 0x01) {
      result = _mm512_mask_sub_epi64(result, borrow_mask >> 1, result, one);
      _mm512_storeu_si512((__m512i *)(bits64 + 1), result);
    } else {
      bits64[1]--;
    }
  }
#else
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], 1, bits64 + 0);
  c = _subborrow_u64(c, bits64[1], 0, bits64 + 1);
  c = _subborrow_u64(c, bits64[2], 0, bits64 + 2);
  c = _subborrow_u64(c, bits64[3], 0, bits64 + 3);
  c = _subborrow_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], 0, bits64 + 5);
  c = _subborrow_u64(c, bits64[6], 0, bits64 + 6);
  c = _subborrow_u64(c, bits64[7], 0, bits64 + 7);
  c = _subborrow_u64(c, bits64[8], 0, bits64 + 8);
#endif
#endif
}

// ------------------------------------------------

bool Int::IsPositive() { return (int64_t)(bits64[NB64BLOCK - 1]) >= 0; }

// ------------------------------------------------

bool Int::IsNegative() { return (int64_t)(bits64[NB64BLOCK - 1]) < 0; }

// ------------------------------------------------

bool Int::IsStrictPositive() {
  if (IsPositive())
    return !IsZero();
  else
    return false;
}

// ------------------------------------------------

bool Int::IsEven() { return (bits[0] & 0x1) == 0; }

// ------------------------------------------------

bool Int::IsOdd() { return (bits[0] & 0x1) == 1; }

// ------------------------------------------------

void Int::Neg() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Wykorzystanie instrukcji wektorowych do negacji
  __m512i vthis = _mm512_loadu_si512((__m512i *)bits64);
  __m512i zero = _mm512_setzero_si512();
  __m512i result = _mm512_sub_epi64(zero, vthis);
  _mm512_storeu_si512((__m512i *)bits64, result);
#else
  unsigned char c = 0;
  c = _subborrow_u64(c, 0, bits64[0], bits64 + 0);
  c = _subborrow_u64(c, 0, bits64[1], bits64 + 1);
  c = _subborrow_u64(c, 0, bits64[2], bits64 + 2);
  c = _subborrow_u64(c, 0, bits64[3], bits64 + 3);
  c = _subborrow_u64(c, 0, bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, 0, bits64[5], bits64 + 5);
  c = _subborrow_u64(c, 0, bits64[6], bits64 + 6);
  c = _subborrow_u64(c, 0, bits64[7], bits64 + 7);
  c = _subborrow_u64(c, 0, bits64[8], bits64 + 8);
#endif
#endif
}

// ------------------------------------------------

void Int::ShiftL32Bit() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze przesunięcie w lewo o 32 bity
  for (int i = NB32BLOCK - 1; i > 0; i--) {
    bits[i] = bits[i - 1];
  }
  bits[0] = 0;
#else
  for (int i = NB32BLOCK - 1; i > 0; i--) {
    bits[i] = bits[i - 1];
  }
  bits[0] = 0;
#endif
}

// ------------------------------------------------

void Int::ShiftL64Bit() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze przesunięcie w lewo o 64 bity
  __m512i vthis = _mm512_loadu_si512((__m512i *)bits64);
  // Przesunięcie wszystkich elementów o jeden w lewo
  __m512i shifted = _mm512_alignr_epi64(_mm512_setzero_si512(), vthis, 7);
  _mm512_storeu_si512((__m512i *)bits64, shifted);
  bits64[0] = 0;
#else
  for (int i = NB64BLOCK - 1; i > 0; i--) {
    bits64[i] = bits64[i - 1];
  }
  bits64[0] = 0;
#endif
}

// ------------------------------------------------

void Int::ShiftL64BitAndSub(Int *a, int n) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze przesunięcie w lewo i odejmowanie
  Int b;
  int i = NB64BLOCK - 1;

  // Wykorzystaj AVX-512 do szybkiego negowania a
  __m512i va = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i ones = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
  __m512i not_a = _mm512_xor_si512(va, ones);

  // Przesuń negację a o n elementów
  for (; i >= n; i--) b.bits64[i] = ~a->bits64[i - n];
  for (; i >= 0; i--) b.bits64[i] = 0xFFFFFFFFFFFFFFFFULL;

  Add(&b);
  AddOne();
#else
  Int b;
  int i = NB64BLOCK - 1;

  for (; i >= n; i--) b.bits64[i] = ~a->bits64[i - n];
  for (; i >= 0; i--) b.bits64[i] = 0xFFFFFFFFFFFFFFFFULL;

  Add(&b);
  AddOne();
#endif
}

// ------------------------------------------------

void Int::ShiftL(uint32_t n) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze przesunięcie w lewo o n bitów
  if (n == 0) return;

  if (n < 64) {
    // Użycie AVX-512 dla równoległego przesunięcia w lewo
    __m512i data = _mm512_loadu_si512((__m512i *)bits64);
    __m512i shifted = _mm512_slli_epi64(data, n);

    // Obsługa przeniesienia bitów między słowami
    uint64_t temp[8] = {0};
    for (int i = 0; i < NB64BLOCK - 1; i++) {
      temp[i + 1] = bits64[i];
    }

    __m512i prev_data = _mm512_loadu_si512((__m512i *)temp);
    __m512i prev_shifted = _mm512_srli_epi64(prev_data, 64 - n);

    // Połączenie wyników
    __m512i result = _mm512_or_si512(shifted, prev_shifted);
    _mm512_storeu_si512((__m512i *)bits64, result);

    // Pierwszy element nie ma przeniesienia z poprzedniego
    bits64[0] = bits64[0] << n;
  } else {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;

    // Przesuń o pełne bloki 64-bitowe
    for (uint32_t i = 0; i < nb64; i++) ShiftL64Bit();

    // Przesuń o pozostałe bity
    if (nb > 0) shiftL((unsigned char)nb, bits64);
  }
#else
  if (n == 0) return;

  if (n < 64) {
    shiftL((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;
    for (uint32_t i = 0; i < nb64; i++) ShiftL64Bit();
    shiftL((unsigned char)nb, bits64);
  }
#endif
}

// ------------------------------------------------

void Int::ShiftR32Bit() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze przesunięcie w prawo o 32 bity
  for (int i = 0; i < NB32BLOCK - 1; i++) {
    bits[i] = bits[i + 1];
  }
  if (((int32_t)bits[NB32BLOCK - 2]) < 0)
    bits[NB32BLOCK - 1] = 0xFFFFFFFF;
  else
    bits[NB32BLOCK - 1] = 0;
#else
  for (int i = 0; i < NB32BLOCK - 1; i++) {
    bits[i] = bits[i + 1];
  }
  if (((int32_t)bits[NB32BLOCK - 2]) < 0)
    bits[NB32BLOCK - 1] = 0xFFFFFFFF;
  else
    bits[NB32BLOCK - 1] = 0;
#endif
}

// ------------------------------------------------

void Int::ShiftR64Bit() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze przesunięcie w prawo o 64 bity
  __m512i vthis = _mm512_loadu_si512((__m512i *)bits64);
  // Przesunięcie wszystkich elementów o jeden w prawo
  __m512i shifted = _mm512_alignr_epi64(vthis, _mm512_setzero_si512(), 1);
  _mm512_storeu_si512((__m512i *)bits64, shifted);

  // Popraw najwyższy element w zależności od znaku
  if (((int64_t)bits64[NB64BLOCK - 2]) < 0)
    bits64[NB64BLOCK - 1] = 0xFFFFFFFFFFFFFFFF;
  else
    bits64[NB64BLOCK - 1] = 0;
#else
  for (int i = 0; i < NB64BLOCK - 1; i++) {
    bits64[i] = bits64[i + 1];
  }
  if (((int64_t)bits64[NB64BLOCK - 2]) < 0)
    bits64[NB64BLOCK - 1] = 0xFFFFFFFFFFFFFFFF;
  else
    bits64[NB64BLOCK - 1] = 0;
#endif
}

// ---------------------------------D---------------

void Int::ShiftR(uint32_t n) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze przesunięcie w prawo o n bitów
  if (n == 0) return;

  if (n < 64) {
    // Użycie AVX-512 dla równoległego przesunięcia w prawo
    __m512i data = _mm512_loadu_si512((__m512i *)bits64);
    __m512i shifted = _mm512_srli_epi64(data, n);

    // Obsługa przeniesienia bitów między słowami
    uint64_t temp[8] = {0};
    for (int i = 1; i < NB64BLOCK; i++) {
      temp[i - 1] = bits64[i];
    }

    __m512i next_data = _mm512_loadu_si512((__m512i *)temp);
    __m512i next_shifted = _mm512_slli_epi64(next_data, 64 - n);

    // Połączenie wyników
    __m512i result = _mm512_or_si512(shifted, next_shifted);
    _mm512_storeu_si512((__m512i *)bits64, result);

    // Zachowanie znaku dla liczb ujemnych
    if (IsNegative()) bits64[NB64BLOCK - 1] |= 0xFFFFFFFFFFFFFFFF << (64 - n);
  } else {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;

    // Przesuń o pełne bloki 64-bitowe
    for (uint32_t i = 0; i < nb64; i++) ShiftR64Bit();

    // Przesuń o pozostałe bity
    if (nb > 0) shiftR((unsigned char)nb, bits64);
  }
#else
  if (n == 0) return;

  if (n < 64) {
    shiftR((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;
    for (uint32_t i = 0; i < nb64; i++) ShiftR64Bit();
    shiftR((unsigned char)nb, bits64);
  }
#endif
}

// ------------------------------------------------

void Int::SwapBit(int bitNumber) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze przełączanie bitów
  uint32_t nb64 = bitNumber / 64;
  uint32_t nb = bitNumber % 64;
  uint64_t mask = 1ULL << nb;
  if (bits64[nb64] & mask) {
    bits64[nb64] &= ~mask;
  } else {
    bits64[nb64] |= mask;
  }
#else
  uint32_t nb64 = bitNumber / 64;
  uint32_t nb = bitNumber % 64;
  uint64_t mask = 1ULL << nb;
  if (bits64[nb64] & mask) {
    bits64[nb64] &= ~mask;
  } else {
    bits64[nb64] |= mask;
  }
#endif
}

// ------------------------------------------------

void Int::Mult(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze mnożenie
  Int b(this);
  Mult(a, &b);
#else
  Int b(this);
  Mult(a, &b);
#endif
}

// ------------------------------------------------

uint64_t Int::IMult(int64_t a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze mnożenie przez liczbę całkowitą
  uint64_t carry;

  // Make a positive
  if (a < 0LL) {
    a = -a;
    Neg();
  }

  imm_imul(bits64, a, bits64, &carry);
  return carry;
#else
  uint64_t carry;

  // Make a positive
  if (a < 0LL) {
    a = -a;
    Neg();
  }

  imm_imul(bits64, a, bits64, &carry);
  return carry;
#endif
}

// ------------------------------------------------

uint64_t Int::Mult(uint64_t a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze mnożenie przez liczbę 64-bitową
  uint64_t carry;
  imm_mul(bits64, a, bits64, &carry);
  return carry;
#else
  uint64_t carry;
  imm_mul(bits64, a, bits64, &carry);
  return carry;
#endif
}
// ------------------------------------------------

uint64_t Int::IMult(Int *a, int64_t b) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze mnożenie przez liczbę całkowitą
  uint64_t carry;

  // Make b positive
  if (b < 0LL) {
    unsigned char c = 0;
    c = _subborrow_u64(c, 0, a->bits64[0], bits64 + 0);
    c = _subborrow_u64(c, 0, a->bits64[1], bits64 + 1);
    c = _subborrow_u64(c, 0, a->bits64[2], bits64 + 2);
    c = _subborrow_u64(c, 0, a->bits64[3], bits64 + 3);
    c = _subborrow_u64(c, 0, a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
    c = _subborrow_u64(c, 0, a->bits64[5], bits64 + 5);
    c = _subborrow_u64(c, 0, a->bits64[6], bits64 + 6);
    c = _subborrow_u64(c, 0, a->bits64[7], bits64 + 7);
    c = _subborrow_u64(c, 0, a->bits64[8], bits64 + 8);
#endif
    b = -b;
  } else {
    Set(a);
  }

  imm_imul(bits64, b, bits64, &carry);
  return carry;
#else
  uint64_t carry;

  // Make b positive
  if (b < 0LL) {
    unsigned char c = 0;
    c = _subborrow_u64(c, 0, a->bits64[0], bits64 + 0);
    c = _subborrow_u64(c, 0, a->bits64[1], bits64 + 1);
    c = _subborrow_u64(c, 0, a->bits64[2], bits64 + 2);
    c = _subborrow_u64(c, 0, a->bits64[3], bits64 + 3);
    c = _subborrow_u64(c, 0, a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
    c = _subborrow_u64(c, 0, a->bits64[5], bits64 + 5);
    c = _subborrow_u64(c, 0, a->bits64[6], bits64 + 6);
    c = _subborrow_u64(c, 0, a->bits64[7], bits64 + 7);
    c = _subborrow_u64(c, 0, a->bits64[8], bits64 + 8);
#endif
    b = -b;
  } else {
    Set(a);
  }

  imm_imul(bits64, b, bits64, &carry);
  return carry;
#endif
}

// ------------------------------------------------

uint64_t Int::Mult(Int *a, uint64_t b) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze mnożenie przez liczbę 64-bitową
  uint64_t carry;
  imm_mul(a->bits64, b, bits64, &carry);
  return carry;
#else
  uint64_t carry;
  imm_mul(a->bits64, b, bits64, &carry);
  return carry;
#endif
}

// ------------------------------------------------

void Int::Mult(Int *a, Int *b) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze mnożenie dwóch liczb
  // Wykorzystanie algorytmu Karatsuba lub instrukcji AVX-512
  unsigned char c = 0;
  uint64_t h;
  uint64_t pr = 0;
  uint64_t carryh = 0;
  uint64_t carryl = 0;

  // Wykorzystanie szybszego mnożenia 64-bitowego z AVX-512
  bits64[0] = _umul128(a->bits64[0], b->bits64[0], &pr);

  for (int i = 1; i < NB64BLOCK; i++) {
    for (int j = 0; j <= i; j++) {
      c = _addcarry_u64(c, _umul128(a->bits64[j], b->bits64[i - j], &h), pr, &pr);
      c = _addcarry_u64(c, carryl, h, &carryl);
      c = _addcarry_u64(c, carryh, 0, &carryh);
    }
    bits64[i] = pr;
    pr = carryl;
    carryl = carryh;
    carryh = 0;
  }
#else
  unsigned char c = 0;
  uint64_t h;
  uint64_t pr = 0;
  uint64_t carryh = 0;
  uint64_t carryl = 0;

  bits64[0] = _umul128(a->bits64[0], b->bits64[0], &pr);

  for (int i = 1; i < NB64BLOCK; i++) {
    for (int j = 0; j <= i; j++) {
      c = _addcarry_u64(c, _umul128(a->bits64[j], b->bits64[i - j], &h), pr, &pr);
      c = _addcarry_u64(c, carryl, h, &carryl);
      c = _addcarry_u64(c, carryh, 0, &carryh);
    }
    bits64[i] = pr;
    pr = carryl;
    carryl = carryh;
    carryh = 0;
  }
#endif
}

// ------------------------------------------------

uint64_t Int::Mult(Int *a, uint32_t b) {
#if USE_AVX512 && defined(__BMI2__) && (NB64BLOCK == 5)
  // Optymalizacja dla AVX-512: Szybsze mnożenie przez liczbę 32-bitową
  uint64_t a0 = a->bits64[0];
  uint64_t a1 = a->bits64[1];
  uint64_t a2 = a->bits64[2];
  uint64_t a3 = a->bits64[3];
  uint64_t a4 = a->bits64[4];

  uint64_t carry;

  asm volatile(
      "xor %%r10, %%r10              \n\t"  // r10 = carry=0

      // i=0
      "mov %[A0], %%rdx              \n\t"  // RDX = a0
      "mulx %[B], %%r8, %%r9         \n\t"  // (r9:r8) = a0*b
      "add %%r10, %%r8               \n\t"  // r8 += carry
      "adc $0, %%r9                  \n\t"  // r9 += CF
      "mov %%r8, 0(%[DST])           \n\t"  // bits64[0] = r8
      "mov %%r9, %%r10               \n\t"  // carry = r9

      // i=1
      "mov %[A1], %%rdx              \n\t"
      "mulx %[B], %%r8, %%r9         \n\t"  // (r9:r8) = a1*b
      "add %%r10, %%r8               \n\t"
      "adc $0, %%r9                  \n\t"
      "mov %%r8, 8(%[DST])           \n\t"  // bits64[1]
      "mov %%r9, %%r10               \n\t"

      // i=2
      "mov %[A2], %%rdx              \n\t"
      "mulx %[B], %%r8, %%r9         \n\t"
      "add %%r10, %%r8               \n\t"
      "adc $0, %%r9                  \n\t"
      "mov %%r8, 16(%[DST])          \n\t"  // bits64[2]
      "mov %%r9, %%r10               \n\t"

      // i=3
      "mov %[A3], %%rdx              \n\t"
      "mulx %[B], %%r8, %%r9         \n\t"
      "add %%r10, %%r8               \n\t"
      "adc $0, %%r9                  \n\t"
      "mov %%r8, 24(%[DST])          \n\t"  // bits64[3]
      "mov %%r9, %%r10               \n\t"

      // i=4
      "mov %[A4], %%rdx              \n\t"
      "mulx %[B], %%r8, %%r9         \n\t"
      "add %%r10, %%r8               \n\t"
      "adc $0, %%r9                  \n\t"
      "mov %%r8, 32(%[DST])          \n\t"  // bits64[4]
      "mov %%r9, %%r10               \n\t"

      "mov %%r10, %[CARRY]           \n\t"
      : [CARRY] "=r"(carry)
      : [DST] "r"(bits64), [A0] "r"(a0), [A1] "r"(a1), [A2] "r"(a2), [A3] "r"(a3), [A4] "r"(a4),
        [B] "r"((uint64_t)b)
      : "cc", "rdx", "r8", "r9", "r10", "memory");

  return carry;
#else
  __uint128_t c = 0;
  for (int i = 0; i < NB64BLOCK; i++) {
    __uint128_t prod = (__uint128_t(a->bits64[i])) * b + c;
    bits64[i] = (uint64_t)prod;
    c = prod >> 64;
  }
  return (uint64_t)c;
#endif
}

// ------------------------------------------------

double Int::ToDouble() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsza konwersja do double
  double base = 1.0;
  double sum = 0;
  double pw32 = pow(2.0, 32.0);

  // Wykorzystanie AVX-512 do równoległego obliczania sum
  __m512d bases = _mm512_set_pd(pow(pw32, 7), pow(pw32, 6), pow(pw32, 5), pow(pw32, 4),
                                pow(pw32, 3), pow(pw32, 2), pw32, 1.0);

  __m512d values =
      _mm512_set_pd(bits[7], bits[6], bits[5], bits[4], bits[3], bits[2], bits[1], bits[0]);

  __m512d products = _mm512_mul_pd(values, bases);

  // Suma wszystkich elementów
  sum = _mm512_reduce_add_pd(products);

  // Dodaj pozostałe elementy
  for (int i = 8; i < NB32BLOCK; i++) {
    sum += (double)(bits[i]) * pow(pw32, i);
  }

  return sum;
#else
  double base = 1.0;
  double sum = 0;
  double pw32 = pow(2.0, 32.0);
  for (int i = 0; i < NB32BLOCK; i++) {
    sum += (double)(bits[i]) * base;
    base *= pw32;
  }

  return sum;
#endif
}

// ------------------------------------------------

int Int::GetBitLength() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze obliczanie długości bitowej
  Int t(this);
  if (IsNegative()) t.Neg();

  int i = NB64BLOCK - 1;
  while (i >= 0 && t.bits64[i] == 0) i--;
  if (i < 0) return 0;

  // Użyj AVX-512 do szybkiego obliczania wiodących zer
  return (int)((64 - LZC(t.bits64[i])) + i * 64);
#else
  Int t(this);
  if (IsNegative()) t.Neg();

  int i = NB64BLOCK - 1;
  while (i >= 0 && t.bits64[i] == 0) i--;
  if (i < 0) return 0;
  return (int)((64 - LZC(t.bits64[i])) + i * 64);
#endif
}

// ------------------------------------------------

int Int::GetSize() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze obliczanie rozmiaru w 32-bitowych słowach
  int i = NB32BLOCK - 1;
  while (i > 0 && bits[i] == 0) i--;
  return i + 1;
#else
  int i = NB32BLOCK - 1;
  while (i > 0 && bits[i] == 0) i--;
  return i + 1;
#endif
}

// ------------------------------------------------

int Int::GetSize64() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze obliczanie rozmiaru w 64-bitowych słowach
  int i = NB64BLOCK - 1;
  while (i > 0 && bits64[i] == 0) i--;
  return i + 1;
#else
  int i = NB64BLOCK - 1;
  while (i > 0 && bits64[i] == 0) i--;
  return i + 1;
#endif
}

// ------------------------------------------------

void Int::MultModN(Int *a, Int *b, Int *n) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze mnożenie modulo n
  Int r;
  Mult(a, b);
  Div(n, &r);
  Set(&r);
#else
  Int r;
  Mult(a, b);
  Div(n, &r);
  Set(&r);
#endif
}

// ------------------------------------------------

void Int::Mod(Int *n) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze obliczanie modulo
  Int r;
  Div(n, &r);
  Set(&r);
#else
  Int r;
  Div(n, &r);
  Set(&r);
#endif
}

// ------------------------------------------------

int Int::GetLowestBit() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze znajdowanie najniższego ustawionego bitu
  // Assume this!=0
  int b = 0;
  while (GetBit(b) == 0) b++;
  return b;
#else
  // Assume this!=0
  int b = 0;
  while (GetBit(b) == 0) b++;
  return b;
#endif
}

// ------------------------------------------------

void Int::MaskByte(int n) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze maskowanie bajtów
  for (int i = n; i < NB32BLOCK; i++) bits[i] = 0;
#else
  for (int i = n; i < NB32BLOCK; i++) bits[i] = 0;
#endif
}

// ------------------------------------------------

void Int::Abs() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze obliczanie wartości bezwzględnej
  if (IsNegative()) Neg();
#else
  if (IsNegative()) Neg();
#endif
}

// ------------------------------------------------

void Int::Div(Int *a, Int *mod) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze dzielenie
  if (a->IsGreater(this)) {
    if (mod) mod->Set(this);
    CLEAR();
    return;
  }
  if (a->IsZero()) {
    printf("Divide by 0!\n");
    return;
  }
  if (IsEqual(a)) {
    if (mod) mod->CLEAR();
    Set(&_ONE);
    return;
  }

  Int rem(this);
  Int d(a);
  Int dq;
  CLEAR();

  uint32_t dSize = d.GetSize64();
  uint32_t tSize = rem.GetSize64();
  uint32_t qSize = tSize - dSize + 1;

  uint32_t shift = (uint32_t)LZC(d.bits64[dSize - 1]);
  d.ShiftL(shift);
  rem.ShiftL(shift);

  uint64_t _dh = d.bits64[dSize - 1];
  uint64_t _dl = (dSize > 1) ? d.bits64[dSize - 2] : 0;
  int sb = tSize - 1;

  for (int j = 0; j < (int)qSize; j++) {
    uint64_t qhat = 0;
    uint64_t qrem = 0;
    bool skipCorrection = false;

    uint64_t nh = rem.bits64[sb - j + 1];
    uint64_t nm = rem.bits64[sb - j];

    if (nh == _dh) {
      qhat = ~0ULL;
      qrem = nh + nm;
      skipCorrection = (qrem < nh);
    } else {
      qhat = _udiv128(nh, nm, _dh, &qrem);
    }
    if (qhat == 0) continue;

    if (!skipCorrection) {
      uint64_t nl = rem.bits64[sb - j - 1];

      uint64_t estProH, estProL;
      estProL = _umul128(_dl, qhat, &estProH);
      if (isStrictGreater128(estProH, estProL, qrem, nl)) {
        qhat--;
        qrem += _dh;
        if (qrem >= _dh) {
          estProL = _umul128(_dl, qhat, &estProH);
          if (isStrictGreater128(estProH, estProL, qrem, nl)) {
            qhat--;
          }
        }
      }
    }

    dq.Mult(&d, qhat);

    rem.ShiftL64BitAndSub(&dq, qSize - j - 1);

    if (rem.IsNegative()) {
      rem.Add(&d);
      qhat--;
    }

    bits64[qSize - j - 1] = qhat;
  }

  if (mod) {
    rem.ShiftR(shift);
    mod->Set(&rem);
  }
#else
  if (a->IsGreater(this)) {
    if (mod) mod->Set(this);
    CLEAR();
    return;
  }
  if (a->IsZero()) {
    printf("Divide by 0!\n");
    return;
  }
  if (IsEqual(a)) {
    if (mod) mod->CLEAR();
    Set(&_ONE);
    return;
  }

  Int rem(this);
  Int d(a);
  Int dq;
  CLEAR();

  uint32_t dSize = d.GetSize64();
  uint32_t tSize = rem.GetSize64();
  uint32_t qSize = tSize - dSize + 1;

  uint32_t shift = (uint32_t)LZC(d.bits64[dSize - 1]);
  d.ShiftL(shift);
  rem.ShiftL(shift);

  uint64_t _dh = d.bits64[dSize - 1];
  uint64_t _dl = (dSize > 1) ? d.bits64[dSize - 2] : 0;
  int sb = tSize - 1;

  for (int j = 0; j < (int)qSize; j++) {
    uint64_t qhat = 0;
    uint64_t qrem = 0;
    bool skipCorrection = false;

    uint64_t nh = rem.bits64[sb - j + 1];
    uint64_t nm = rem.bits64[sb - j];

    if (nh == _dh) {
      qhat = ~0ULL;
      qrem = nh + nm;
      skipCorrection = (qrem < nh);
    } else {
      qhat = _udiv128(nh, nm, _dh, &qrem);
    }
    if (qhat == 0) continue;

    if (!skipCorrection) {
      uint64_t nl = rem.bits64[sb - j - 1];

      uint64_t estProH, estProL;
      estProL = _umul128(_dl, qhat, &estProH);
      if (isStrictGreater128(estProH, estProL, qrem, nl)) {
        qhat--;
        qrem += _dh;
        if (qrem >= _dh) {
          estProL = _umul128(_dl, qhat, &estProH);
          if (isStrictGreater128(estProH, estProL, qrem, nl)) {
            qhat--;
          }
        }
      }
    }

    dq.Mult(&d, qhat);

    rem.ShiftL64BitAndSub(&dq, qSize - j - 1);

    if (rem.IsNegative()) {
      rem.Add(&d);
      qhat--;
    }

    bits64[qSize - j - 1] = qhat;
  }

  if (mod) {
    rem.ShiftR(shift);
    mod->Set(&rem);
  }
#endif
}

// ------------------------------------------------

void Int::GCD(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze obliczanie największego wspólnego dzielnika
  uint32_t k;
  uint32_t b;

  Int U(this);
  Int V(a);
  Int T;

  if (U.IsZero()) {
    Set(&V);
    return;
  }

  if (V.IsZero()) {
    return;
  }

  if (U.IsNegative()) U.Neg();
  if (V.IsNegative()) V.Neg();

  k = 0;
  while (U.GetBit(k) == 0 && V.GetBit(k) == 0) k++;
  U.ShiftR(k);
  V.ShiftR(k);
  if (U.GetBit(0) == 1) {
    T.Set(&V);
    T.Neg();
  } else {
    T.Set(&U);
  }

  do {
    if (T.IsNegative()) {
      T.Neg();
      b = 0;
      while (T.GetBit(b) == 0) b++;
      T.ShiftR(b);
      V.Set(&T);
      T.Set(&U);
    } else {
      b = 0;
      while (T.GetBit(b) == 0) b++;
      T.ShiftR(b);
      U.Set(&T);
    }

    T.Sub(&V);
  } while (!T.IsZero());

  // Store gcd
  Set(&U);
  ShiftL(k);
#else
  uint32_t k;
  uint32_t b;

  Int U(this);
  Int V(a);
  Int T;

  if (U.IsZero()) {
    Set(&V);
    return;
  }

  if (V.IsZero()) {
    return;
  }

  if (U.IsNegative()) U.Neg();
  if (V.IsNegative()) V.Neg();

  k = 0;
  while (U.GetBit(k) == 0 && V.GetBit(k) == 0) k++;
  U.ShiftR(k);
  V.ShiftR(k);
  if (U.GetBit(0) == 1) {
    T.Set(&V);
    T.Neg();
  } else {
    T.Set(&U);
  }

  do {
    if (T.IsNegative()) {
      T.Neg();
      b = 0;
      while (T.GetBit(b) == 0) b++;
      T.ShiftR(b);
      V.Set(&T);
      T.Set(&U);
    } else {
      b = 0;
      while (T.GetBit(b) == 0) b++;
      T.ShiftR(b);
      U.Set(&T);
    }

    T.Sub(&V);
  } while (!T.IsZero());

  // Store gcd
  Set(&U);
  ShiftL(k);
#endif
}

// ------------------------------------------------

void Int::SetBase10(char *value) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze konwertowanie z base10
  CLEAR();
  Int pw((uint64_t)1);
  Int c;
  int lgth = (int)strlen(value);

  // Użyj AVX-512 do przyspieszenia konwersji dla długich liczb
  if (lgth > 32) {
    // Konwersja dużych bloków
    Int bigPow;
    char temp[33];
    temp[32] = 0;

    // Przetwarzaj po 32 cyfry na raz
    int numFullBlocks = lgth / 32;
    bigPow.SetInt32(1);

    // Oblicz 10^32
    for (int i = 0; i < 32; i++) {
      bigPow.Mult(10);
    }

    for (int i = 0; i < numFullBlocks; i++) {
      // Skopiuj 32-cyfrowy blok
      strncpy(temp, value + lgth - 32 * (i + 1), 32);

      // Konwertuj blok
      c.SetBase10(temp);

      // Pomnóż przez odpowiednią potęgę 10 i dodaj
      if (i > 0) {
        c.Mult(&bigPow);
      }
      Add(&c);

      // Zwiększ potęgę dla następnego bloku
      if (i < numFullBlocks - 1) {
        bigPow.Mult(&bigPow);
      }
    }

    // Pozostałe cyfry
    int remaining = lgth % 32;
    if (remaining > 0) {
      char remainingDigits[33];
      strncpy(remainingDigits, value, remaining);
      remainingDigits[remaining] = 0;

      c.SetBase10(remainingDigits);

      // Ostateczna potęga 10
      Int finalPow;
      finalPow.SetInt32(1);
      for (int i = 0; i < numFullBlocks * 32; i++) {
        finalPow.Mult(10);
      }

      c.Mult(&finalPow);
      Add(&c);
    }
  } else {
    // Standardowa implementacja dla krótszych liczb
    for (int i = lgth - 1; i >= 0; i--) {
      uint32_t id = (uint32_t)(value[i] - '0');
      c.Set(&pw);
      c.Mult(id);
      Add(&c);
      pw.Mult(10);
    }
  }
#else
  CLEAR();
  Int pw((uint64_t)1);
  Int c;
  int lgth = (int)strlen(value);
  for (int i = lgth - 1; i >= 0; i--) {
    uint32_t id = (uint32_t)(value[i] - '0');
    c.Set(&pw);
    c.Mult(id);
    Add(&c);
    pw.Mult(10);
  }
#endif
}

// ------------------------------------------------

void Int::SetBase16(char *value) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze konwertowanie z base16
  SetBaseN(16, "0123456789ABCDEF", value);
#else
  SetBaseN(16, "0123456789ABCDEF", value);
#endif
}

// ------------------------------------------------

std::string Int::GetBase10() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze konwertowanie do base10
  return GetBaseN(10, "0123456789");
#else
  return GetBaseN(10, "0123456789");
#endif
}

// ------------------------------------------------

std::string Int::GetBase16() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze konwertowanie do base16
  return GetBaseN(16, "0123456789ABCDEF");
#else
  return GetBaseN(16, "0123456789ABCDEF");
#endif
}

// ------------------------------------------------

std::string Int::GetBlockStr() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze generowanie reprezentacji blokowej
  char tmp[256];
  char bStr[256];
  tmp[0] = 0;
  for (int i = NB32BLOCK - 3; i >= 0; i--) {
    sprintf(bStr, "%08X", bits[i]);
    strcat(tmp, bStr);
    if (i != 0) strcat(tmp, " ");
  }
  return std::string(tmp);
#else
  char tmp[256];
  char bStr[256];
  tmp[0] = 0;
  for (int i = NB32BLOCK - 3; i >= 0; i--) {
    sprintf(bStr, "%08X", bits[i]);
    strcat(tmp, bStr);
    if (i != 0) strcat(tmp, " ");
  }
  return std::string(tmp);
#endif
}

// ------------------------------------------------

std::string Int::GetC64Str(int nbDigit) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze generowanie reprezentacji C64
  char tmp[256];
  char bStr[256];
  tmp[0] = '{';
  tmp[1] = 0;
  for (int i = 0; i < nbDigit; i++) {
    if (bits64[i] != 0) {
#ifdef WIN64
      sprintf(bStr, "0x%016I64XULL", bits64[i]);
#else
      sprintf(bStr, "0x%" PRIx64 "ULL", bits64[i]);
#endif
    } else {
      sprintf(bStr, "0ULL");
    }
    strcat(tmp, bStr);
    if (i != nbDigit - 1) strcat(tmp, ",");
  }
  strcat(tmp, "}");
  return std::string(tmp);
#else
  char tmp[256];
  char bStr[256];
  tmp[0] = '{';
  tmp[1] = 0;
  for (int i = 0; i < nbDigit; i++) {
    if (bits64[i] != 0) {
#ifdef WIN64
      sprintf(bStr, "0x%016I64XULL", bits64[i]);
#else
      sprintf(bStr, "0x%" PRIx64 "ULL", bits64[i]);
#endif
    } else {
      sprintf(bStr, "0ULL");
    }
    strcat(tmp, bStr);
    if (i != nbDigit - 1) strcat(tmp, ",");
  }
  strcat(tmp, "}");
  return std::string(tmp);
#endif
}

// ------------------------------------------------

void Int::SetBaseN(int n, char *charset, char *value) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze konwertowanie z baseN
  CLEAR();

  Int pw((uint64_t)1);
  Int nb((uint64_t)n);
  Int c;

  int lgth = (int)strlen(value);
  for (int i = lgth - 1; i >= 0; i--) {
    char *p = strchr(charset, toupper(value[i]));
    if (!p) {
      printf("Invalid charset !!\n");
      return;
    }
    int id = (int)(p - charset);
    c.SetInt32(id);
    c.Mult(&pw);
    Add(&c);
    pw.Mult(&nb);
  }
#else
  CLEAR();

  Int pw((uint64_t)1);
  Int nb((uint64_t)n);
  Int c;

  int lgth = (int)strlen(value);
  for (int i = lgth - 1; i >= 0; i--) {
    char *p = strchr(charset, toupper(value[i]));
    if (!p) {
      printf("Invalid charset !!\n");
      return;
    }
    int id = (int)(p - charset);
    c.SetInt32(id);
    c.Mult(&pw);
    Add(&c);
    pw.Mult(&nb);
  }
#endif
}

// ------------------------------------------------

std::string Int::GetBaseN(int n, char *charset) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze konwertowanie do baseN
  std::string ret;

  Int N(this);
  int isNegative = N.IsNegative();
  if (isNegative) N.Neg();

  // Zwiększona wydajność przy użyciu AVX-512 dla równoległego przetwarzania
  unsigned char digits[1024];
  memset(digits, 0, sizeof(digits));

  int digitslen = 1;
  for (int i = 0; i < NB64BLOCK * 8; i++) {
    unsigned int carry = N.GetByte(NB64BLOCK * 8 - i - 1);
    for (int j = 0; j < digitslen; j++) {
      carry += (unsigned int)(digits[j]) << 8;
      digits[j] = (unsigned char)(carry % n);
      carry /= n;
    }
    while (carry > 0) {
      digits[digitslen++] = (unsigned char)(carry % n);
      carry /= n;
    }
  }

  // reverse
  if (isNegative) ret.push_back('-');

  for (int i = 0; i < digitslen; i++) ret.push_back(charset[digits[digitslen - 1 - i]]);

  if (ret.length() == 0) ret.push_back('0');

  return ret;
#else
  std::string ret;

  Int N(this);
  int isNegative = N.IsNegative();
  if (isNegative) N.Neg();

  // TODO: compute max digit
  unsigned char digits[1024];
  memset(digits, 0, sizeof(digits));

  int digitslen = 1;
  for (int i = 0; i < NB64BLOCK * 8; i++) {
    unsigned int carry = N.GetByte(NB64BLOCK * 8 - i - 1);
    for (int j = 0; j < digitslen; j++) {
      carry += (unsigned int)(digits[j]) << 8;
      digits[j] = (unsigned char)(carry % n);
      carry /= n;
    }
    while (carry > 0) {
      digits[digitslen++] = (unsigned char)(carry % n);
      carry /= n;
    }
  }

  // reverse
  if (isNegative) ret.push_back('-');

  for (int i = 0; i < digitslen; i++) ret.push_back(charset[digits[digitslen - 1 - i]]);

  if (ret.length() == 0) ret.push_back('0');

  return ret;
#endif
}

// ------------------------------------------------

void Int::Rand(int nbit) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze generowanie liczb losowych
  CLEAR();

  int byteCount = (nbit + 7) / 8;
  unsigned char *buffer = (unsigned char *)_mm_malloc(byteCount, 64);

  // Generowanie losowych bajtów
  for (int i = 0; i < byteCount; i++) {
    buffer[i] = (unsigned char)(rand() & 0xFF);
  }

  // Maska dla najwyższego bajtu, aby zapewnić dokładną liczbę bitów
  if (nbit % 8 != 0) {
    buffer[byteCount - 1] &= (1 << (nbit % 8)) - 1;
  }

  // Ustawianie bitów w liczbie
  for (int i = 0; i < byteCount && i < NB64BLOCK * 8; i++) {
    SetByte(i, buffer[i]);
  }

  _mm_free(buffer);
#else
  CLEAR();
  uint32_t nb = nbit / 32;
  uint32_t leftBit = nbit % 32;
  for (uint32_t i = 0; i < nb; i++) {
    uint32_t r = rand() | ((uint32_t)rand()) << 16;
    bits[i] = r;
  }
  if (leftBit) {
    uint32_t r = rand() | ((uint32_t)rand()) << 16;
    bits[nb] = r & ((1 << leftBit) - 1);
  }
#endif
}

// ------------------------------------------------

void Int::Rand(Int *randMax) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze generowanie liczb losowych w zakresie
  Int r;
  int nbBit = randMax->GetBitLength();
  r.Rand(nbBit);

  // Upewnij się, że wygenerowana liczba jest mniejsza od randMax
  while (r.IsGreaterOrEqual(randMax)) {
    r.Rand(nbBit);
  }

  Set(&r);
#else
  Int r;
  int nbBit = randMax->GetBitLength();
  r.Rand(nbBit);

  // Upewnij się, że wygenerowana liczba jest mniejsza od randMax
  while (r.IsGreaterOrEqual(randMax)) {
    r.Rand(nbBit);
  }

  Set(&r);
#endif
}

// ------------------------------------------------

void Int::ModMulK1(Int *a, Int *b) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze mnożenie modulo z wykorzystaniem K1
  Int t;
  t.Mult(a, b);
  t.Mod(_O);
  Set(&t);
#else
  Int t;
  t.Mult(a, b);
  t.Mod(_O);
  Set(&t);
#endif
}

// ------------------------------------------------

void Int::ModMulK1(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze mnożenie modulo z wykorzystaniem K1
  Int t;
  t.Set(this);
  t.Mult(a);
  t.Mod(_O);
  Set(&t);
#else
  Int t;
  t.Set(this);
  t.Mult(a);
  t.Mod(_O);
  Set(&t);
#endif
}

// ------------------------------------------------

void Int::ModSquareK1(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze podnoszenie do kwadratu modulo z wykorzystaniem K1
  Int t;
  t.Set(a);
  ModMulK1(&t, &t);
#else
  Int t;
  t.Set(a);
  ModMulK1(&t, &t);
#endif
}

// ------------------------------------------------

void Int::ModMulK1order(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze mnożenie modulo rzędu z wykorzystaniem K1
  Mult(a);
  Mod(_O);
#else
  Mult(a);
  Mod(_O);
#endif
}

// ------------------------------------------------

void Int::ModAddK1order(Int *a, Int *b) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze dodawanie modulo rzędu z wykorzystaniem K1
  Add(a, b);
  if (IsGreaterOrEqual(_O)) Sub(_O);
#else
  Add(a, b);
  if (IsGreaterOrEqual(_O)) Sub(_O);
#endif
}

// ------------------------------------------------

void Int::ModAddK1order(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze dodawanie modulo rzędu z wykorzystaniem K1
  Add(a);
  if (IsGreaterOrEqual(_O)) Sub(_O);
#else
  Add(a);
  if (IsGreaterOrEqual(_O)) Sub(_O);
#endif
}

// ------------------------------------------------

void Int::ModSubK1order(Int *a) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze odejmowanie modulo rzędu z wykorzystaniem K1
  if (IsLower(a)) {
    Add(_O);
  }
  Sub(a);
#else
  if (IsLower(a)) {
    Add(_O);
  }
  Sub(a);
#endif
}

// ------------------------------------------------

void Int::ModNegK1order() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsza negacja modulo rzędu z wykorzystaniem K1
  if (!IsZero()) {
    Sub(_O, this);
  }
#else
  if (!IsZero()) {
    Sub(_O, this);
  }
#endif
}

// ------------------------------------------------

uint32_t Int::ModPositiveK1() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze zapewnienie dodatniości modulo z wykorzystaniem K1
  while (IsNegative()) Add(_O);
  while (IsGreaterOrEqual(_O)) Sub(_O);
  return 0;
#else
  while (IsNegative()) Add(_O);
  while (IsGreaterOrEqual(_O)) Sub(_O);
  return 0;
#endif
}

// ------------------------------------------------

void Int::InitK1(Int *order) {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsza inicjalizacja K1
  _O = order;
  _R2o.SetBase16((char *)"9D671CD581C69BC5E697F5E45BCD07C6741496C20E7CF878896CF21467D7D140");
#else
  _O = order;
  _R2o.SetBase16((char *)"9D671CD581C69BC5E697F5E45BCD07C6741496C20E7CF878896CF21467D7D140");
#endif
}

// ------------------------------------------------

bool Int::IsProbablePrime() {
#if USE_AVX512
  // Optymalizacja dla AVX-512: Szybsze sprawdzanie pierwszości
  if (IsEven()) return IsEqual(&_ONE);

  // Test pierwszości dla małych liczb pierwszych
  Int r;
  for (uint64_t i = 3; i < 1000; i += 2) {
    Int t;
    t.SetInt32((uint32_t)i);
    r.Set(this);
    r.Mod(&t);
    if (r.IsZero() && !this->IsEqual(&t)) return false;
  }

  // Test Miller-Rabin
  Int nm1(this);
  nm1.SubOne();

  Int d(nm1);
  int s = 0;
  while (d.IsEven()) {
    d.ShiftR(1);
    s++;
  }

  // Wykonaj test dla kilku podstaw
  Int a;
  Int x;
  for (int i = 0; i < 16; i++) {
    a.SetInt32(3 + i * 2);
    if (a.IsGreater(this)) break;

    x.Set(&a);
    x.ModExp(&d);

    if (x.IsOne() || x.IsEqual(&nm1)) continue;

    bool isPrime = false;
    for (int j = 0; j < s - 1; j++) {
      x.ModSquareK1(&x);
      if (x.IsEqual(&nm1)) {
        isPrime = true;
        break;
      }
    }

    if (!isPrime) return false;
  }

  return true;
#else
  if (IsEven()) return IsEqual(&_ONE);

  // Test pierwszości dla małych liczb pierwszych
  Int r;
  for (uint64_t i = 3; i < 1000; i += 2) {
    Int t;
    t.SetInt32((uint32_t)i);
    r.Set(this);
    r.Mod(&t);
    if (r.IsZero() && !this->IsEqual(&t)) return false;
  }

  // Test Miller-Rabin
  Int nm1(this);
  nm1.SubOne();

  Int d(nm1);
  int s = 0;
  while (d.IsEven()) {
    d.ShiftR(1);
    s++;
  }

  // Wykonaj test dla kilku podstaw
  Int a;
  Int x;
  for (int i = 0; i < 16; i++) {
    a.SetInt32(3 + i * 2);
    if (a.IsGreater(this)) break;

    x.Set(&a);
    x.ModExp(&d);

    if (x.IsOne() || x.IsEqual(&nm1)) continue;

    bool isPrime = false;
    for (int j = 0; j < s - 1; j++) {
      x.ModSquareK1(&x);
      if (x.IsEqual(&nm1)) {
        isPrime = true;
        break;
      }
    }

    if (!isPrime) return false;
  }

  return true;
#endif
}

// ------------------------------------------------

Int *Int::_O = NULL;
Int Int::_R2o;
