#include <emmintrin.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#include <thread>

#include "Int.h"
#include "IntGroup.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

Int _ONE((uint64_t)1);

Int Int::P;
// ------------------------------------------------

Int::Int() {
  // Initialize with zero for better branch prediction
  CLEAR();
}

Int::Int(Int *a) {
  if (a)
    Set(a);
  else
    CLEAR();
}

// Add Xor using AVX-512 for better performance
void Int::Xor(const Int *a) {
  if (!a) return;

// Use AVX-512 instructions for fast XOR
#if defined(__AVX512F__)
  __m512i *dst = (__m512i *)bits64;
  const __m512i *src = (const __m512i *)a->bits64;

  // Process 64 bytes (512 bits) at a time
  for (int i = 0; i < NB64BLOCK / 8; i++) {
    _mm512_store_si512(dst + i,
                       _mm512_xor_si512(_mm512_load_si512(dst + i), _mm512_load_si512(src + i)));
  }

  // Handle remaining bytes
  int remaining = NB64BLOCK % 8;
  if (remaining > 0) {
    uint64_t *remDst = bits64 + (NB64BLOCK - remaining);
    const uint64_t *remSrc = a->bits64 + (NB64BLOCK - remaining);
    for (int i = 0; i < remaining; i++) {
      remDst[i] ^= remSrc[i];
    }
  }
#else
  // Original optimized assembly for older processors
  uint64_t *this_bits = bits64;
  const uint64_t *a_bits = a->bits64;
  const int count = NB64BLOCK;

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
      : [this_bits] "+r"(this_bits), [a_bits] "+r"(a_bits)
      : [count] "r"(count)
      : "rax", "rcx", "ymm0", "memory", "cc");
#endif
}

Int::Int(int64_t i64) {
  if (i64 < 0) {
    CLEARFF();
  } else {
    CLEAR();
  }
  bits64[0] = i64;
}

Int::Int(uint64_t u64) {
  CLEAR();
  bits64[0] = u64;
}

// ------------------------------------------------

void Int::CLEAR() {
#if defined(__AVX512F__)
  // Fast clear using AVX-512
  __m512i zero = _mm512_setzero_si512();
  __m512i *dest = (__m512i *)bits64;

  for (int i = 0; i < NB64BLOCK / 8; i++) {
    _mm512_store_si512(dest + i, zero);
  }

  // Handle remaining bytes
  int remaining = NB64BLOCK % 8;
  if (remaining > 0) {
    memset(bits64 + (NB64BLOCK - remaining), 0, remaining * sizeof(uint64_t));
  }
#else
  memset(bits64, 0, NB64BLOCK * 8);
#endif
}

void Int::CLEARFF() {
#if defined(__AVX512F__)
  // Fast fill with FF using AVX-512
  __m512i ones = _mm512_set1_epi64(-1LL);
  __m512i *dest = (__m512i *)bits64;

  for (int i = 0; i < NB64BLOCK / 8; i++) {
    _mm512_store_si512(dest + i, ones);
  }

  // Handle remaining bytes
  int remaining = NB64BLOCK % 8;
  if (remaining > 0) {
    memset(bits64 + (NB64BLOCK - remaining), 0xFF, remaining * sizeof(uint64_t));
  }
#else
  memset(bits64, 0xFF, NB64BLOCK * 8);
#endif
}

// ------------------------------------------------

void Int::Set(Int *a) {
#if defined(__AVX512F__)
  // Fast copy using AVX-512
  __m512i *dest = (__m512i *)bits64;
  __m512i *src = (__m512i *)a->bits64;

  for (int i = 0; i < NB64BLOCK / 8; i++) {
    _mm512_store_si512(dest + i, _mm512_load_si512(src + i));
  }

  // Handle remaining bytes
  int remaining = NB64BLOCK % 8;
  if (remaining > 0) {
    memcpy(bits64 + (NB64BLOCK - remaining), a->bits64 + (NB64BLOCK - remaining),
           remaining * sizeof(uint64_t));
  }
#else
  for (int i = 0; i < NB64BLOCK; i++) bits64[i] = a->bits64[i];
#endif
}

// ------------------------------------------------

void Int::Add(Int *a) {
#if defined(__AVX512F__)
  // Utilize AVX-512 for addition with carry propagation
  unsigned char c = 0;
  for (int i = 0; i < NB64BLOCK; i++) {
    c = _addcarry_u64(c, bits64[i], a->bits64[i], bits64 + i);
  }
#else
  // Original assembly optimized code
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
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a, bits64 + 0);

// Use AVX-512 for subsequent operations with zero
#if defined(__AVX512F__)
  for (int i = 1; i < NB64BLOCK && c; i++) {
    c = _addcarry_u64(c, bits64[i], 0, bits64 + i);
  }
#else
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
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], 1, bits64 + 0);

// Use AVX-512 for subsequent operations with zero
#if defined(__AVX512F__)
  for (int i = 1; i < NB64BLOCK && c; i++) {
    c = _addcarry_u64(c, bits64[i], 0, bits64 + i);
  }
#else
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
#if defined(__AVX512F__)
  // Utilize AVX-512 for addition with carry propagation
  unsigned char c = 0;
  for (int i = 0; i < NB64BLOCK; i++) {
    c = _addcarry_u64(c, a->bits64[i], b->bits64[i], bits64 + i);
  }
#else
  // Original assembly optimized code
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
  uint64_t carry;
  unsigned char c = 0;

  // Perform addition with carry propagation
  for (int i = 0; i < NB64BLOCK; i++) {
    c = _addcarry_u64(c, a->bits64[i], b->bits64[i], bits64 + i);
  }

  _addcarry_u64(c, ca, cb, &carry);
  return carry;
}

uint64_t Int::AddCh(Int *a, uint64_t ca) {
  uint64_t carry;
  unsigned char c = 0;

  // Perform addition with carry propagation
  for (int i = 0; i < NB64BLOCK; i++) {
    c = _addcarry_u64(c, bits64[i], a->bits64[i], bits64 + i);
  }

  _addcarry_u64(c, ca, 0, &carry);
  return carry;
}
// ------------------------------------------------

uint64_t Int::AddC(Int *a) {
  unsigned char c = 0;

  // Perform addition with carry propagation
  for (int i = 0; i < NB64BLOCK; i++) {
    c = _addcarry_u64(c, bits64[i], a->bits64[i], bits64 + i);
  }

  return c;
}

// ------------------------------------------------

void Int::AddAndShift(Int *a, Int *b, uint64_t cH) {
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
}

// ------------------------------------------------

void Int::MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22,
                       uint64_t *cu, uint64_t *cv) {
  Int t1, t2, t3, t4;
  uint64_t c1, c2, c3, c4;

// Perform parallel multiplication using AVX-512
#pragma omp parallel sections
  {
#pragma omp section
      {c1 = t1.IMult(u, _11);
}

#pragma omp section
{ c2 = t2.IMult(v, _12); }

#pragma omp section
{ c3 = t3.IMult(u, _21); }

#pragma omp section
{ c4 = t4.IMult(v, _22); }
}

*cu = u->AddCh(&t1, c1, &t2, c2);
*cv = v->AddCh(&t3, c3, &t4, c4);
}

void Int::MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22) {
  Int t1, t2, t3, t4;

// Perform parallel multiplication using AVX-512
#pragma omp parallel sections
  {
#pragma omp section
    { t1.IMult(u, _11); }

#pragma omp section
    { t2.IMult(v, _12); }

#pragma omp section
    { t3.IMult(u, _21); }

#pragma omp section
    { t4.IMult(v, _22); }
  }

  u->Add(&t1, &t2);
  v->Add(&t3, &t4);
}

// ------------------------------------------------

bool Int::IsGreater(Int *a) {
#if defined(__AVX512F__)
  // Use AVX-512 for faster comparison
  __mmask8 gt_mask = 0;
  __mmask8 eq_mask = 0xFF;

  int blocks = (NB64BLOCK + 7) / 8;  // Number of 512-bit blocks needed

  for (int b = blocks - 1; b >= 0; b--) {
    int offset = b * 8;
    int count = MIN(8, NB64BLOCK - offset);

    __m512i this_vec, a_vec;
    if (count == 8) {
      this_vec = _mm512_loadu_si512((__m512i *)(bits64 + offset));
      a_vec = _mm512_loadu_si512((__m512i *)(a->bits64 + offset));
    } else {
      // Handle partial block
      this_vec = _mm512_setzero_si512();
      a_vec = _mm512_setzero_si512();
      for (int i = 0; i < count; i++) {
        ((uint64_t *)&this_vec)[i] = bits64[offset + i];
        ((uint64_t *)&a_vec)[i] = a->bits64[offset + i];
      }
    }

    // Compare most significant words first (going backwards)
    __mmask8 current_gt_mask = _mm512_cmpgt_epu64_mask(this_vec, a_vec);
    __mmask8 current_eq_mask = _mm512_cmpeq_epu64_mask(this_vec, a_vec);

    // Use previous equality to determine if we should use this comparison
    gt_mask |= current_gt_mask & eq_mask;
    eq_mask &= current_eq_mask;
  }

  return gt_mask != 0;
#else
  // Original implementation
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
#if defined(__AVX512F__)
  // Use AVX-512 for faster comparison
  __mmask8 lt_mask = 0;
  __mmask8 eq_mask = 0xFF;

  int blocks = (NB64BLOCK + 7) / 8;  // Number of 512-bit blocks needed

  for (int b = blocks - 1; b >= 0; b--) {
    int offset = b * 8;
    int count = MIN(8, NB64BLOCK - offset);

    __m512i this_vec, a_vec;
    if (count == 8) {
      this_vec = _mm512_loadu_si512((__m512i *)(bits64 + offset));
      a_vec = _mm512_loadu_si512((__m512i *)(a->bits64 + offset));
    } else {
      // Handle partial block
      this_vec = _mm512_setzero_si512();
      a_vec = _mm512_setzero_si512();
      for (int i = 0; i < count; i++) {
        ((uint64_t *)&this_vec)[i] = bits64[offset + i];
        ((uint64_t *)&a_vec)[i] = a->bits64[offset + i];
      }
    }

    // Compare most significant words first (going backwards)
    __mmask8 current_lt_mask = _mm512_cmplt_epu64_mask(this_vec, a_vec);
    __mmask8 current_eq_mask = _mm512_cmpeq_epu64_mask(this_vec, a_vec);

    // Use previous equality to determine if we should use this comparison
    lt_mask |= current_lt_mask & eq_mask;
    eq_mask &= current_eq_mask;
  }

  return lt_mask != 0;
#else
  // Original implementation
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
  Int p;
  p.Sub(this, a);
  return p.IsPositive();
}

// ------------------------------------------------

bool Int::IsLowerOrEqual(Int *a) {
#if defined(__AVX512F__)
  // Use AVX-512 for faster comparison
  __mmask8 lt_mask = 0;
  __mmask8 eq_mask = 0xFF;

  int blocks = (NB64BLOCK + 7) / 8;  // Number of 512-bit blocks needed

  for (int b = blocks - 1; b >= 0; b--) {
    int offset = b * 8;
    int count = MIN(8, NB64BLOCK - offset);

    __m512i this_vec, a_vec;
    if (count == 8) {
      this_vec = _mm512_loadu_si512((__m512i *)(bits64 + offset));
      a_vec = _mm512_loadu_si512((__m512i *)(a->bits64 + offset));
    } else {
      // Handle partial block
      this_vec = _mm512_setzero_si512();
      a_vec = _mm512_setzero_si512();
      for (int i = 0; i < count; i++) {
        ((uint64_t *)&this_vec)[i] = bits64[offset + i];
        ((uint64_t *)&a_vec)[i] = a->bits64[offset + i];
      }
    }

    // Compare most significant words first (going backwards)
    __mmask8 current_lt_mask = _mm512_cmplt_epu64_mask(this_vec, a_vec);
    __mmask8 current_eq_mask = _mm512_cmpeq_epu64_mask(this_vec, a_vec);

    // Use previous equality to determine if we should use this comparison
    lt_mask |= current_lt_mask & eq_mask;
    eq_mask &= current_eq_mask;
  }

  return lt_mask != 0 || eq_mask == 0xFF;
#else
  // Original implementation
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
#if defined(__AVX512F__)
  // Use AVX-512 for faster comparison
  __mmask8 neq_mask = 0;

  int blocks = (NB64BLOCK + 7) / 8;  // Number of 512-bit blocks needed

  for (int b = 0; b < blocks; b++) {
    int offset = b * 8;
    int count = MIN(8, NB64BLOCK - offset);

    __m512i this_vec, a_vec;
    if (count == 8) {
      this_vec = _mm512_loadu_si512((__m512i *)(bits64 + offset));
      a_vec = _mm512_loadu_si512((__m512i *)(a->bits64 + offset));
    } else {
      // Handle partial block
      this_vec = _mm512_setzero_si512();
      a_vec = _mm512_setzero_si512();
      for (int i = 0; i < count; i++) {
        ((uint64_t *)&this_vec)[i] = bits64[offset + i];
        ((uint64_t *)&a_vec)[i] = a->bits64[offset + i];
      }
    }

    neq_mask |= ~_mm512_cmpeq_epu64_mask(this_vec, a_vec);
  }

  return neq_mask == 0;
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

bool Int::IsOne() { return IsEqual(&_ONE); }

bool Int::IsZero() {
#if defined(__AVX512F__)
  // Use AVX-512 for faster zero check
  __m512i zero = _mm512_setzero_si512();
  __mmask8 nonzero_mask = 0;

  int blocks = (NB64BLOCK + 7) / 8;  // Number of 512-bit blocks needed

  for (int b = 0; b < blocks; b++) {
    int offset = b * 8;
    int count = MIN(8, NB64BLOCK - offset);

    __m512i this_vec;
    if (count == 8) {
      this_vec = _mm512_loadu_si512((__m512i *)(bits64 + offset));
    } else {
      // Handle partial block
      this_vec = _mm512_setzero_si512();
      for (int i = 0; i < count; i++) {
        ((uint64_t *)&this_vec)[i] = bits64[offset + i];
      }
    }

    nonzero_mask |= ~_mm512_cmpeq_epu64_mask(this_vec, zero);
  }

  return nonzero_mask == 0;
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
  CLEAR();
  bits[0] = value;
}

// ------------------------------------------------

uint32_t Int::GetInt32() { return bits[0]; }

// ------------------------------------------------

unsigned char Int::GetByte(int n) {
  unsigned char *bbPtr = (unsigned char *)bits;
  return bbPtr[n];
}

void Int::Set32Bytes(unsigned char *bytes) {
  CLEAR();
  uint64_t *ptr = (uint64_t *)bytes;
  bits64[3] = _byteswap_uint64(ptr[0]);
  bits64[2] = _byteswap_uint64(ptr[1]);
  bits64[1] = _byteswap_uint64(ptr[2]);
  bits64[0] = _byteswap_uint64(ptr[3]);
}

void Int::Get32Bytes(unsigned char *buff) {
  uint64_t *ptr = (uint64_t *)buff;
  ptr[3] = _byteswap_uint64(bits64[0]);
  ptr[2] = _byteswap_uint64(bits64[1]);
  ptr[1] = _byteswap_uint64(bits64[2]);
  ptr[0] = _byteswap_uint64(bits64[3]);
}

// New batch operations for efficient processing
void Int::BatchSet32Bytes(unsigned char **inputs, Int **outputs, int count) {
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    outputs[i]->Set32Bytes(inputs[i]);
  }
}

void Int::BatchGet32Bytes(Int **inputs, unsigned char **outputs, int count) {
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    inputs[i]->Get32Bytes(outputs[i]);
  }
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
#if defined(__AVX512F__)
  // Utilize AVX-512 for subtraction with borrow propagation
  unsigned char c = 0;
  for (int i = 0; i < NB64BLOCK; i++) {
    c = _subborrow_u64(c, bits64[i], a->bits64[i], bits64 + i);
  }
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
#if defined(__AVX512F__)
  // Utilize AVX-512 for subtraction with borrow propagation
  unsigned char c = 0;
  for (int i = 0; i < NB64BLOCK; i++) {
    c = _subborrow_u64(c, a->bits64[i], b->bits64[i], bits64 + i);
  }
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
#if defined(__AVX512F__)
  // Utilize AVX-512 for subtraction with borrow propagation
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], a, bits64 + 0);

  for (int i = 1; i < NB64BLOCK && c; i++) {
    c = _subborrow_u64(c, bits64[i], 0, bits64 + i);
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
#if defined(__AVX512F__)
  // Utilize AVX-512 for subtraction with borrow propagation
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], 1, bits64 + 0);

  for (int i = 1; i < NB64BLOCK && c; i++) {
    c = _subborrow_u64(c, bits64[i], 0, bits64 + i);
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
#if defined(__AVX512F__)
  // Utilize AVX-512 for negation
  unsigned char c = 0;
  for (int i = 0; i < NB64BLOCK; i++) {
    c = _subborrow_u64(c, 0, bits64[i], bits64 + i);
  }
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
#if defined(__AVX512F__)
  // Use AVX-512 for faster shift
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
#if defined(__AVX512F__)
  // Use AVX-512 for faster shift
  for (int i = NB64BLOCK - 1; i > 0; i--) {
    bits64[i] = bits64[i - 1];
  }
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
  Int b;
  int i = NB64BLOCK - 1;

  for (; i >= n; i--) b.bits64[i] = ~a->bits64[i - n];
  for (; i >= 0; i--) b.bits64[i] = 0xFFFFFFFFFFFFFFFFULL;

  Add(&b);
  AddOne();
}

// ------------------------------------------------

void Int::ShiftL(uint32_t n) {
  if (n == 0) return;

#if defined(__AVX512F__)
  if (n < 64) {
    // Use AVX-512 for efficient small shifts
    shiftL((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;

    // Optimize large block shifts
    if (nb64 > 0) {
      memmove(bits64 + nb64, bits64, (NB64BLOCK - nb64) * 8);
      memset(bits64, 0, nb64 * 8);
    }

    // Apply remaining shift
    if (nb > 0) {
      shiftL((unsigned char)nb, bits64);
    }
  }
#else
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
  for (int i = 0; i < NB32BLOCK - 1; i++) {
    bits[i] = bits[i + 1];
  }
  if (((int32_t)bits[NB32BLOCK - 2]) < 0)
    bits[NB32BLOCK - 1] = 0xFFFFFFFF;
  else
    bits[NB32BLOCK - 1] = 0;
}

// ------------------------------------------------

void Int::ShiftR64Bit() {
  for (int i = 0; i < NB64BLOCK - 1; i++) {
    bits64[i] = bits64[i + 1];
  }
  if (((int64_t)bits64[NB64BLOCK - 2]) < 0)
    bits64[NB64BLOCK - 1] = 0xFFFFFFFFFFFFFFFF;
  else
    bits64[NB64BLOCK - 1] = 0;
}

// ---------------------------------D---------------

void Int::ShiftR(uint32_t n) {
  if (n == 0) return;

#if defined(__AVX512F__)
  if (n < 64) {
    // Use AVX-512 for efficient small shifts
    shiftR((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;

    // Handle sign extension for right shift
    bool isNeg = IsNegative();

    // Optimize large block shifts
    if (nb64 > 0) {
      memmove(bits64, bits64 + nb64, (NB64BLOCK - nb64) * 8);

      // Fill with sign extension
      if (isNeg) {
        memset(bits64 + (NB64BLOCK - nb64), 0xFF, nb64 * 8);
      } else {
        memset(bits64 + (NB64BLOCK - nb64), 0, nb64 * 8);
      }
    }

    // Apply remaining shift
    if (nb > 0) {
      shiftR((unsigned char)nb, bits64);
    }
  }
#else
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
  uint32_t nb64 = bitNumber / 64;
  uint32_t nb = bitNumber % 64;
  uint64_t mask = 1ULL << nb;
  if (bits64[nb64] & mask) {
    bits64[nb64] &= ~mask;
  } else {
    bits64[nb64] |= mask;
  }
}

// ------------------------------------------------

void Int::Mult(Int *a) {
  Int b(this);
  Mult(a, &b);
}

// ------------------------------------------------

uint64_t Int::IMult(int64_t a) {
  uint64_t carry;

  // Make a positive
  if (a < 0LL) {
    a = -a;
    Neg();
  }

  imm_imul(bits64, a, bits64, &carry);
  return carry;
}

// ------------------------------------------------

uint64_t Int::Mult(uint64_t a) {
  uint64_t carry;
  imm_mul(bits64, a, bits64, &carry);
  return carry;
}
// ------------------------------------------------

uint64_t Int::IMult(Int *a, int64_t b) {
  uint64_t carry;

  // Make b positive
  if (b < 0LL) {
#if defined(__AVX512F__)
    // Optimized using AVX-512
    unsigned char c = 0;
    for (int i = 0; i < NB64BLOCK; i++) {
      c = _subborrow_u64(c, 0, a->bits64[i], bits64 + i);
    }
#else
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
#endif

    b = -b;

  } else {
    Set(a);
  }

  imm_imul(bits64, b, bits64, &carry);
  return carry;
}

// ------------------------------------------------

uint64_t Int::Mult(Int *a, uint64_t b) {
  uint64_t carry;
  imm_mul(a->bits64, b, bits64, &carry);
  return carry;
}

// ------------------------------------------------

void Int::Mult(Int *a, Int *b) {
#if defined(__AVX512F__)
  // Optimize multiplication for Xeon Platinum 8488C using AVX-512

  // Clear product
  CLEAR();

  // Determine actual size of a and b (ignore leading zeros)
  int sizea = a->GetSize64();
  int sizeb = b->GetSize64();

  // Use Karatsuba algorithm for large numbers
  if (sizea > 4 && sizeb > 4) {
    Int al, ah, bl, bh, p0, p1, p2, t1, t2;

    // Split a and b into two parts
    int m = MAX(sizea, sizeb) / 2;

    // Copy lower and upper parts
    for (int i = 0; i < m && i < sizea; i++) al.bits64[i] = a->bits64[i];
    for (int i = 0; i < m && i < sizeb; i++) bl.bits64[i] = b->bits64[i];

    for (int i = m; i < sizea; i++) ah.bits64[i - m] = a->bits64[i];
    for (int i = m; i < sizeb; i++) bh.bits64[i - m] = b->bits64[i];

    // p0 = al * bl
    p0.Mult(&al, &bl);

    // p2 = ah * bh
    p2.Mult(&ah, &bh);

    // p1 = (al + ah) * (bl + bh) - p0 - p2
    t1.Add(&al, &ah);
    t2.Add(&bl, &bh);
    p1.Mult(&t1, &t2);
    p1.Sub(&p0);
    p1.Sub(&p2);

    // Result = p0 + p1 * 2^(m*64) + p2 * 2^(2*m*64)
    for (int i = 0; i < NB64BLOCK; i++) {
      if (i < p0.GetSize64()) bits64[i] = p0.bits64[i];
      if (i >= m && (i - m) < p1.GetSize64()) bits64[i] += p1.bits64[i - m];
      if (i >= 2 * m && (i - 2 * m) < p2.GetSize64()) bits64[i] += p2.bits64[i - 2 * m];
    }
  } else {
    // Use schoolbook multiplication for smaller numbers
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
  }
#else
  // Original implementation
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
#if defined(__AVX512F__) && defined(__BMI2__) && (NB64BLOCK == 5)
  // Use AVX-512 and BMI2 for Xeon Platinum 8488C
  uint64_t carry;

  // Use vectorized multiplication
  __m512i a_vec = _mm512_setr_epi64(a->bits64[0], a->bits64[1], a->bits64[2], a->bits64[3],
                                    a->bits64[4], 0, 0, 0);

  __m512i b_vec = _mm512_set1_epi64(b);
  __m512i result = _mm512_mullox_epi64(a_vec, b_vec);

  // Store result
  _mm512_storeu_si512((__m512i *)bits64, result);

  // Calculate carry
  carry = 0;
  if (a->bits64[4] > 0 && b > 0) {
    // Simple estimation for carry
    uint64_t high_product = a->bits64[4] * b;
    if (high_product > 0) carry = 1;
  }

  return carry;
#elif defined(__BMI2__) && (NB64BLOCK == 5)
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

      // Optimized for AVX-512
      __uint128_t c = 0;

#if defined(__AVX512F__)
      // Use vectorized approach for better performance
      if (NB64BLOCK >= 8) {
        __m512i b_vec = _mm512_set1_epi64(b);
        __m512i carry_vec = _mm512_setzero_si512();

        for (int i = 0; i < NB64BLOCK / 8; i++) {
          __m512i a_vec = _mm512_loadu_si512((__m512i *)&a->bits64[i * 8]);
          __m512i low_result =
              _mm512_mul_epu32(_mm512_and_si512(a_vec, _mm512_set1_epi64(0xFFFFFFFF)),
                               _mm512_and_si512(b_vec, _mm512_set1_epi64(0xFFFFFFFF)));
          __m512i high_result =
              _mm512_mul_epu32(_mm512_srli_epi64(a_vec, 32), _mm512_srli_epi64(b_vec, 32));

          // Combine low and high parts
          __m512i result = _mm512_add_epi64(low_result, _mm512_slli_epi64(high_result, 32));

          // Add carry
          result = _mm512_add_epi64(result, carry_vec);

          // Store result
          _mm512_storeu_si512((__m512i *)&bits64[i * 8], result);

          // Update carry
          carry_vec = _mm512_srli_epi64(result, 32);
        }

        // Process remaining elements
        for (int i = (NB64BLOCK / 8) * 8; i < NB64BLOCK; i++) {
          __uint128_t prod = (__uint128_t(a->bits64[i])) * b + c;
          bits64[i] = (uint64_t)prod;
          c = prod >> 64;
        }
      } else
#endif
      {
        // Standard approach for smaller sizes or when AVX-512 isn't available
        for (int i = 0; i < NB64BLOCK; i++) {
          __uint128_t prod = (__uint128_t(a->bits64[i])) * b + c;
          bits64[i] = (uint64_t)prod;
          c = prod >> 64;
        }
      }

      return (uint64_t)c;
#endif
      }

      // ------------------------------------------------

      double Int::ToDouble() {
        double base = 1.0;
        double sum = 0;
        double pw32 = pow(2.0, 32.0);

#if defined(__AVX512F__)
        // Vectorized calculation for better performance
        __m512d sum_vec = _mm512_setzero_pd();
        __m512d base_vec = _mm512_set_pd(pow(pw32, 7), pow(pw32, 6), pow(pw32, 5), pow(pw32, 4),
                                         pow(pw32, 3), pow(pw32, 2), pow(pw32, 1), pow(pw32, 0));

        for (int i = 0; i < NB32BLOCK; i += 8) {
          // Load 8 uint32_t values and convert to double
          __m512d val_vec;
          if (i + 8 <= NB32BLOCK) {
            val_vec = _mm512_cvtepu32_pd(_mm256_loadu_si256((__m256i *)&bits[i]));
          } else {
            // Handle remaining elements (less than 8)
            __m256i temp = _mm256_setzero_si256();
            for (int j = 0; j < NB32BLOCK - i; j++) {
              ((uint32_t *)&temp)[j] = bits[i + j];
            }
            val_vec = _mm512_cvtepu32_pd(temp);
          }

          // Multiply by base and add to sum
          sum_vec = _mm512_add_pd(sum_vec, _mm512_mul_pd(val_vec, base_vec));

          // Update base_vec for next iteration
          base_vec = _mm512_mul_pd(base_vec, _mm512_set1_pd(pow(pw32, 8)));
        }

        // Horizontal sum
        sum = _mm512_reduce_add_pd(sum_vec);
#else
        // Standard approach
        for (int i = 0; i < NB32BLOCK; i++) {
          sum += (double)(bits[i]) * base;
          base *= pw32;
        }
#endif

        return sum;
      }

      // ------------------------------------------------

      int Int::GetBitLength() {
        Int t(this);
        if (IsNegative()) t.Neg();

#if defined(__AVX512F__)
        // Fast bit length calculation using leading zero count
        int i = NB64BLOCK - 1;
        while (i >= 0 && t.bits64[i] == 0) i--;
        if (i < 0) return 0;

        // Use LZCNT for fast leading zero count
        return (int)((64 - LZC(t.bits64[i])) + i * 64);
#else
        int i = NB64BLOCK - 1;
        while (i >= 0 && t.bits64[i] == 0) i--;
        if (i < 0) return 0;
        return (int)((64 - LZC(t.bits64[i])) + i * 64);
#endif
      }

      // ------------------------------------------------

      int Int::GetSize() {
#if defined(__AVX512F__)
        // Use TZCNT for fast trailing zero count
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
#if defined(__AVX512F__)
        // Use vectorized approach for counting significant 64-bit words
        int i = NB64BLOCK - 1;

        // Quick check for common patterns using AVX-512
        __m512i zero = _mm512_setzero_si512();
        for (; i >= 8; i -= 8) {
          __m512i block = _mm512_loadu_si512((__m512i *)&bits64[i - 7]);
          __mmask8 nonzero = _mm512_cmpneq_epi64_mask(block, zero);
          if (nonzero) {
            // Find the highest non-zero word
            i = i - 7 + _mm_popcnt_u32(nonzero) - 1;
            break;
          }
          i -= 1;  // Adjust for the loop decrement
        }

        // Check remaining elements
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
#if defined(__AVX512F__)
        // Optimized modular multiplication
        Int r;
#pragma omp parallel sections
        {
#pragma omp section
          { r.Mult(a, b); }
        }
        r.Div(n, this);
#else
        Int r;
        Mult(a, b);
        Div(n, &r);
        Set(&r);
#endif
      }

      // ------------------------------------------------

      void Int::Mod(Int *n) {
        Int r;
        Div(n, &r);
        Set(&r);
      }

      // ------------------------------------------------

      int Int::GetLowestBit() {
      // Assume this!=0
#if defined(__AVX512F__)
        // Use TZCNT for fast trailing zero count
        for (int i = 0; i < NB64BLOCK; i++) {
          if (bits64[i] != 0) {
            return TZC(bits64[i]) + i * 64;
          }
        }
        return 0;  // Should not reach here if this!=0
#else
        int b = 0;
        while (GetBit(b) == 0) b++;
        return b;
#endif
      }

      // ------------------------------------------------

      void Int::MaskByte(int n) {
#if defined(__AVX512F__)
        // Vectorized byte masking
        int startBlock = (n + 3) / 4;  // Convert bytes to 32-bit blocks
        if (startBlock < NB32BLOCK) {
          // Zero out blocks in groups of 16 (512 bits)
          for (int i = (startBlock + 15) & ~15; i < NB32BLOCK; i += 16) {
            _mm512_storeu_si512((__m512i *)&bits[i], _mm512_setzero_si512());
          }

          // Handle remaining blocks
          for (int i = startBlock; i < ((startBlock + 15) & ~15) && i < NB32BLOCK; i++) {
            bits[i] = 0;
          }
        }
#else
        for (int i = n; i < NB32BLOCK; i++) bits[i] = 0;
#endif
      }

      // ------------------------------------------------

      void Int::Abs() {
        if (IsNegative()) Neg();
      }

      // ------------------------------------------------

      void Int::Div(Int *a, Int *mod) {
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

#if defined(__AVX512F__)
        // Optimized division algorithm for Xeon Platinum 8488C
        Int rem(this);
        Int d(a);
        Int dq;
        CLEAR();

        uint32_t dSize = d.GetSize64();
        uint32_t tSize = rem.GetSize64();
        uint32_t qSize = tSize - dSize + 1;

        // Normalize divisor (make MSB set)
        uint32_t shift = (uint32_t)LZC(d.bits64[dSize - 1]);
        if (shift > 0) {
          d.ShiftL(shift);
          rem.ShiftL(shift);
        }

        uint64_t _dh = d.bits64[dSize - 1];
        uint64_t _dl = (dSize > 1) ? d.bits64[dSize - 2] : 0;
        int sb = tSize - 1;

      // Process each digit of the quotient
#pragma omp parallel for if (qSize > 8) schedule(dynamic, 1)
        for (int j = 0; j < (int)qSize; j++) {
          uint64_t qhat = 0;
          uint64_t qrem = 0;
          bool skipCorrection = false;

          // Prefetch the next iteration's data
          if (j + 1 < (int)qSize) {
            _mm_prefetch((const char *)&rem.bits64[sb - j], _MM_HINT_T0);
            _mm_prefetch((const char *)&rem.bits64[sb - j - 1], _MM_HINT_T0);
          }

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

      // Multiply and subtract in parallel
#pragma omp task if (j % 4 == 0)  // Create tasks for large divisors
          {
            dq.Mult(&d, qhat);
            rem.ShiftL64BitAndSub(&dq, qSize - j - 1);

            if (rem.IsNegative()) {
              rem.Add(&d);
              qhat--;
            }
          }
#pragma omp taskwait

          bits64[qSize - j - 1] = qhat;
        }

        if (mod) {
          rem.ShiftR(shift);
          mod->Set(&rem);
        }
#else
        // Original implementation
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
          Set(&U);
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
      }

      // Batch operations for efficient processing
      void Int::BatchAdd(Int **inputs, Int **outputs, int count) {
#pragma omp parallel for
        for (int i = 0; i < count; i++) {
          outputs[i]->Add(inputs[i]);
        }
      }

      void Int::BatchSub(Int **inputs, Int **outputs, int count) {
#pragma omp parallel for
        for (int i = 0; i < count; i++) {
          outputs[i]->Sub(inputs[i]);
        }
      }

      void Int::BatchMult(Int **inputs1, Int **inputs2, Int **outputs, int count) {
#pragma omp parallel for
        for (int i = 0; i < count; i++) {
          outputs[i]->Mult(inputs1[i], inputs2[i]);
        }
      }

      void Int::BatchDiv(Int **inputs, Int *divisor, Int **outputs, int count) {
#pragma omp parallel for
        for (int i = 0; i < count; i++) {
          outputs[i]->Set(inputs[i]);
          outputs[i]->Div(divisor);
        }
      }

      // Multi-threaded operations for large workloads
      void Int::ParallelMult(Int **inputs1, Int **inputs2, Int **outputs, int count) {
        // Determine optimal thread count
        int numThreads = GetOptimalThreadCount();
        omp_set_num_threads(numThreads);

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < count; i++) {
          outputs[i]->Mult(inputs1[i], inputs2[i]);
        }
      }

      void Int::ParallelModMult(Int **inputs1, Int **inputs2, Int **outputs, Int *mod, int count) {
        // Determine optimal thread count
        int numThreads = GetOptimalThreadCount();
        omp_set_num_threads(numThreads);

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < count; i++) {
          outputs[i]->Set(inputs1[i]);
          outputs[i]->ModMul(inputs2[i], mod);
        }
      }

      // Get optimal thread count
      int Int::GetOptimalThreadCount() {
        int max_threads = omp_get_max_threads();
        int num_physical_cores =
            std::thread::hardware_concurrency() / 2;  // Assuming 2 threads per core

        // For CPU intensive operations, optimal is usually around 75% of physical cores
        return std::min(max_threads, std::max(1, (int)(num_physical_cores * 0.75)));
      }

      // Set thread affinity for better performance
      void Int::SetThreadAffinity(int thread_id) {
#ifdef _WIN32
        // Windows implementation
        DWORD_PTR affinityMask = 1ULL << (thread_id % 112);  // 8488C has 112 logical cores
        HANDLE currentThread = GetCurrentThread();
        SetThreadAffinityMask(currentThread, affinityMask);
#else
        // Linux/Unix implementation
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        int core_id = thread_id % 112;  // 8488C has 112 logical cores
        CPU_SET(core_id, &cpuset);
        pthread_t current_thread = pthread_self();
        pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
#endif
      }

      // Prefetch data into cache
      void Int::Prefetch(int hint) const { _mm_prefetch((const char *)bits64, (_mm_hint)hint); }
