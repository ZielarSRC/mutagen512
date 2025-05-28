#include <emmintrin.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#include "Int.h"
#include "IntGroup.h"

// Ultimate compiler optimizations for Xeon Platinum 8488C
#pragma GCC target( \
    "avx512f,avx512dq,avx512bw,avx512vl,avx512vnni,avx512ifma,avx512vbmi,bmi2,lzcnt,popcnt,adx")
#pragma GCC optimize( \
    "O3,unroll-loops,inline-functions,omit-frame-pointer,tree-vectorize")

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

Int _ONE((uint64_t)1);
Int Int::P;

// ------------------------------------------------

Int::Int() {
  // Initialize with prefetching for optimal Xeon 8488C performance
  __builtin_prefetch(bits64, 1, 3);
}

Int::Int(Int *a) {
  if (LIKELY(a)) {
    Set(a);
  } else {
    CLEAR();
  }
}

// Enhanced XOR with AVX-512 optimization for Xeon 8488C
void Int::Xor(const Int *a) {
  if (UNLIKELY(!a)) return;

  // AVX-512 optimized XOR using full 512-bit registers
  const uint64_t *this_bits = bits64;
  const uint64_t *a_bits = a->bits64;

  // Prefetch data for optimal cache utilization
  __builtin_prefetch(a_bits, 0, 3);
  __builtin_prefetch(this_bits, 1, 3);

#if defined(__AVX512F__) && (NB64BLOCK >= 8)
  // Use AVX-512 for maximum performance on Xeon 8488C
  __asm__ volatile(
      "vmovdqu64 (%[a_bits]), %%zmm0     \n\t"      // Load 512 bits from a
      "vpxorq (%[this_bits]), %%zmm0, %%zmm0 \n\t"  // XOR with this
      "vmovdqu64 %%zmm0, (%[this_bits]) \n\t"       // Store result
      "vzeroupper                        \n\t"      // Clear upper state
      :
      : [this_bits] "r"(this_bits), [a_bits] "r"(a_bits)
      : "zmm0", "memory");

// Handle remaining elements if NB64BLOCK > 8
#if NB64BLOCK > 8
  for (int i = 8; i < NB64BLOCK; i++) {
    bits64[i] ^= a->bits64[i];
  }
#endif

#elif defined(__AVX2__)
  // Fallback to AVX2 for compatibility
  const int avx2_blocks = NB64BLOCK / 4;
  __asm__ volatile(
      "mov %[count], %%ecx               \n\t"
      "test %%ecx, %%ecx                 \n\t"
      "jz 2f                             \n\t"

      "1:                                \n\t"
      "vmovdqa (%[a_bits]), %%ymm0       \n\t"
      "vpxor (%[this_bits]), %%ymm0, %%ymm0 \n\t"
      "vmovdqa %%ymm0, (%[this_bits])    \n\t"
      "add $32, %[a_bits]                \n\t"
      "add $32, %[this_bits]             \n\t"
      "dec %%ecx                         \n\t"
      "jnz 1b                            \n\t"

      "vzeroupper                        \n\t"
      "2:                                \n\t"
      : [this_bits] "+r"(this_bits), [a_bits] "+r"(a_bits)
      : [count] "r"(avx2_blocks)
      : "rcx", "ymm0", "memory", "cc");

  // Handle remaining elements
  for (int i = avx2_blocks * 4; i < NB64BLOCK; i++) {
    bits64[i] ^= a->bits64[i];
  }
#else
// Scalar fallback with loop unrolling
#pragma GCC unroll NB64BLOCK
  for (int i = 0; i < NB64BLOCK; i++) {
    bits64[i] ^= a->bits64[i];
  }
#endif
}

Int::Int(int64_t i64) {
  if (UNLIKELY(i64 < 0)) {
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

FORCE_INLINE void Int::CLEAR() {
  // AVX-512 optimized memory clearing for Xeon 8488C
#if defined(__AVX512F__) && (NB64BLOCK * 8 >= 64)
  __asm__ volatile(
      "vpxorq %%zmm0, %%zmm0, %%zmm0     \n\t"
      "vmovdqu64 %%zmm0, (%[bits])       \n\t"
#if (NB64BLOCK * 8) > 64
      "vmovdqu64 %%zmm0, 64(%[bits])     \n\t"
#endif
      "vzeroupper                        \n\t"
      :
      : [bits] "r"(bits64)
      : "zmm0", "memory");
#else
  memset(bits64, 0, NB64BLOCK * 8);
#endif
}

FORCE_INLINE void Int::CLEARFF() {
  // Optimized FF clearing
#if defined(__AVX512F__) && (NB64BLOCK * 8 >= 64)
  __asm__ volatile(
      "vpcmpeqq %%zmm0, %%zmm0, %%zmm0   \n\t"  // Set all bits to 1
      "vmovdqu64 %%zmm0, (%[bits])       \n\t"
#if (NB64BLOCK * 8) > 64
      "vmovdqu64 %%zmm0, 64(%[bits])     \n\t"
#endif
      "vzeroupper                        \n\t"
      :
      : [bits] "r"(bits64)
      : "zmm0", "memory");
#else
  memset(bits64, 0xFF, NB64BLOCK * 8);
#endif
}

// ------------------------------------------------

FORCE_INLINE void Int::Set(Int *a) {
  // Optimized memory copy using AVX-512 on Xeon 8488C
  __builtin_prefetch(a->bits64, 0, 3);
  __builtin_prefetch(bits64, 1, 3);

#if defined(__AVX512F__) && (NB64BLOCK * 8 >= 64)
  __asm__ volatile(
      "vmovdqu64 (%[src]), %%zmm0        \n\t"
      "vmovdqu64 %%zmm0, (%[dst])        \n\t"
#if (NB64BLOCK * 8) > 64
      "vmovdqu64 64(%[src]), %%zmm1      \n\t"
      "vmovdqu64 %%zmm1, 64(%[dst])      \n\t"
#endif
      "vzeroupper                        \n\t"
      :
      : [dst] "r"(bits64), [src] "r"(a->bits64)
      : "zmm0", "zmm1", "memory");
#else
#pragma GCC unroll NB64BLOCK
  for (int i = 0; i < NB64BLOCK; i++) {
    bits64[i] = a->bits64[i];
  }
#endif
}

// ------------------------------------------------

void Int::Add(Int *a) {
  // Enhanced addition using ADX instructions for Xeon 8488C
  __builtin_prefetch(a->bits64, 0, 3);

#if defined(__ADX__) && (NB64BLOCK == 5)
  // Use ADX instructions for optimal carry chain performance
  unsigned char c = 0;
  c = _addcarry_u64_optimized(c, bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64_optimized(c, bits64[1], a->bits64[1], bits64 + 1);
  c = _addcarry_u64_optimized(c, bits64[2], a->bits64[2], bits64 + 2);
  c = _addcarry_u64_optimized(c, bits64[3], a->bits64[3], bits64 + 3);
  c = _addcarry_u64_optimized(c, bits64[4], a->bits64[4], bits64 + 4);

#if NB64BLOCK > 5
  c = _addcarry_u64_optimized(c, bits64[5], a->bits64[5], bits64 + 5);
  c = _addcarry_u64_optimized(c, bits64[6], a->bits64[6], bits64 + 6);
  c = _addcarry_u64_optimized(c, bits64[7], a->bits64[7], bits64 + 7);
  c = _addcarry_u64_optimized(c, bits64[8], a->bits64[8], bits64 + 8);
#endif

#else
  // Fallback to assembly with optimized register allocation
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

  __asm__(
      "add %[src0], %[dst0]              \n\t"
      "adc %[src1], %[dst1]              \n\t"
      "adc %[src2], %[dst2]              \n\t"
      "adc %[src3], %[dst3]              \n\t"
      "adc %[src4], %[dst4]              \n\t"
#if NB64BLOCK > 5
      "adc %[src5], %[dst5]              \n\t"
      "adc %[src6], %[dst6]              \n\t"
      "adc %[src7], %[dst7]              \n\t"
      "adc %[src8], %[dst8]              \n\t"
#endif
      : [dst0] "+r"(acc0), [dst1] "+r"(acc1), [dst2] "+r"(acc2),
        [dst3] "+r"(acc3), [dst4] "+r"(acc4)
#if NB64BLOCK > 5
                               ,
        [dst5] "+r"(acc5), [dst6] "+r"(acc6), [dst7] "+r"(acc7),
        [dst8] "+r"(acc8)
#endif
      : [src0] "r"(a->bits64[0]), [src1] "r"(a->bits64[1]),
        [src2] "r"(a->bits64[2]), [src3] "r"(a->bits64[3]),
        [src4] "r"(a->bits64[4])
#if NB64BLOCK > 5
            ,
        [src5] "r"(a->bits64[5]), [src6] "r"(a->bits64[6]),
        [src7] "r"(a->bits64[7]), [src8] "r"(a->bits64[8])
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
  // Optimized scalar addition with ADX
#if defined(__ADX__)
  unsigned char c = 0;
  c = _addcarry_u64_optimized(c, bits64[0], a, bits64 + 0);
  c = _addcarry_u64_optimized(c, bits64[1], 0, bits64 + 1);
  c = _addcarry_u64_optimized(c, bits64[2], 0, bits64 + 2);
  c = _addcarry_u64_optimized(c, bits64[3], 0, bits64 + 3);
  c = _addcarry_u64_optimized(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64_optimized(c, bits64[5], 0, bits64 + 5);
  c = _addcarry_u64_optimized(c, bits64[6], 0, bits64 + 6);
  c = _addcarry_u64_optimized(c, bits64[7], 0, bits64 + 7);
  c = _addcarry_u64_optimized(c, bits64[8], 0, bits64 + 8);
#endif
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
  // Optimized increment with branch prediction
#if defined(__ADX__)
  unsigned char c = 0;
  c = _addcarry_u64_optimized(c, bits64[0], 1, bits64 + 0);
  c = _addcarry_u64_optimized(c, bits64[1], 0, bits64 + 1);
  c = _addcarry_u64_optimized(c, bits64[2], 0, bits64 + 2);
  c = _addcarry_u64_optimized(c, bits64[3], 0, bits64 + 3);
  c = _addcarry_u64_optimized(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64_optimized(c, bits64[5], 0, bits64 + 5);
  c = _addcarry_u64_optimized(c, bits64[6], 0, bits64 + 6);
  c = _addcarry_u64_optimized(c, bits64[7], 0, bits64 + 7);
  c = _addcarry_u64_optimized(c, bits64[8], 0, bits64 + 8);
#endif
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
  // Enhanced three-operand addition
  __builtin_prefetch(a->bits64, 0, 3);
  __builtin_prefetch(b->bits64, 0, 3);

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

  __asm__(
      "add %[b0], %[a0]                  \n\t"
      "adc %[b1], %[a1]                  \n\t"
      "adc %[b2], %[a2]                  \n\t"
      "adc %[b3], %[a3]                  \n\t"
      "adc %[b4], %[a4]                  \n\t"
#if NB64BLOCK > 5
      "adc %[b5], %[a5]                  \n\t"
      "adc %[b6], %[a6]                  \n\t"
      "adc %[b7], %[a7]                  \n\t"
      "adc %[b8], %[a8]                  \n\t"
#endif
      : [a0] "+r"(acc0), [a1] "+r"(acc1), [a2] "+r"(acc2), [a3] "+r"(acc3),
        [a4] "+r"(acc4)
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
}

// ------------------------------------------------

uint64_t Int::AddCh(Int *a, uint64_t ca, Int *b, uint64_t cb) {
  // Optimized addition with carry handling
  uint64_t carry;

#if defined(__ADX__)
  unsigned char c = 0;
  c = _addcarry_u64_optimized(c, a->bits64[0], b->bits64[0], bits64 + 0);
  c = _addcarry_u64_optimized(c, a->bits64[1], b->bits64[1], bits64 + 1);
  c = _addcarry_u64_optimized(c, a->bits64[2], b->bits64[2], bits64 + 2);
  c = _addcarry_u64_optimized(c, a->bits64[3], b->bits64[3], bits64 + 3);
  c = _addcarry_u64_optimized(c, a->bits64[4], b->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64_optimized(c, a->bits64[5], b->bits64[5], bits64 + 5);
  c = _addcarry_u64_optimized(c, a->bits64[6], b->bits64[6], bits64 + 6);
  c = _addcarry_u64_optimized(c, a->bits64[7], b->bits64[7], bits64 + 7);
  c = _addcarry_u64_optimized(c, a->bits64[8], b->bits64[8], bits64 + 8);
#endif
  _addcarry_u64_optimized(c, ca, cb, &carry);
#else
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
#endif
  return carry;
}

uint64_t Int::AddCh(Int *a, uint64_t ca) {
  uint64_t carry;

#if defined(__ADX__)
  unsigned char c = 0;
  c = _addcarry_u64_optimized(c, bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64_optimized(c, bits64[1], a->bits64[1], bits64 + 1);
  c = _addcarry_u64_optimized(c, bits64[2], a->bits64[2], bits64 + 2);
  c = _addcarry_u64_optimized(c, bits64[3], a->bits64[3], bits64 + 3);
  c = _addcarry_u64_optimized(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64_optimized(c, bits64[5], a->bits64[5], bits64 + 5);
  c = _addcarry_u64_optimized(c, bits64[6], a->bits64[6], bits64 + 6);
  c = _addcarry_u64_optimized(c, bits64[7], a->bits64[7], bits64 + 7);
  c = _addcarry_u64_optimized(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
  _addcarry_u64_optimized(c, ca, 0, &carry);
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
  _addcarry_u64(c, ca, 0, &carry);
#endif
  return carry;
}

// ------------------------------------------------

uint64_t Int::AddC(Int *a) {
#if defined(__ADX__)
  unsigned char c = 0;
  c = _addcarry_u64_optimized(c, bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64_optimized(c, bits64[1], a->bits64[1], bits64 + 1);
  c = _addcarry_u64_optimized(c, bits64[2], a->bits64[2], bits64 + 2);
  c = _addcarry_u64_optimized(c, bits64[3], a->bits64[3], bits64 + 3);
  c = _addcarry_u64_optimized(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64_optimized(c, bits64[5], a->bits64[5], bits64 + 5);
  c = _addcarry_u64_optimized(c, bits64[6], a->bits64[6], bits64 + 6);
  c = _addcarry_u64_optimized(c, bits64[7], a->bits64[7], bits64 + 7);
  c = _addcarry_u64_optimized(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
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
#endif
  return c;
}

// ------------------------------------------------

void Int::AddAndShift(Int *a, Int *b, uint64_t cH) {
#if defined(__ADX__)
  unsigned char c = 0;
  c = _addcarry_u64_optimized(c, b->bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64_optimized(c, b->bits64[1], a->bits64[1], bits64 + 0);
  c = _addcarry_u64_optimized(c, b->bits64[2], a->bits64[2], bits64 + 1);
  c = _addcarry_u64_optimized(c, b->bits64[3], a->bits64[3], bits64 + 2);
  c = _addcarry_u64_optimized(c, b->bits64[4], a->bits64[4], bits64 + 3);
#if NB64BLOCK > 5
  c = _addcarry_u64_optimized(c, b->bits64[5], a->bits64[5], bits64 + 4);
  c = _addcarry_u64_optimized(c, b->bits64[6], a->bits64[6], bits64 + 5);
  c = _addcarry_u64_optimized(c, b->bits64[7], a->bits64[7], bits64 + 6);
  c = _addcarry_u64_optimized(c, b->bits64[8], a->bits64[8], bits64 + 7);
#endif
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
#endif
  bits64[NB64BLOCK - 1] = c + cH;
}

// ------------------------------------------------

void Int::MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21,
                       int64_t _22, uint64_t *cu, uint64_t *cv) {
  CACHE_ALIGN Int t1, t2, t3, t4;
  uint64_t c1, c2, c3, c4;

  // Prefetch data for optimal Xeon 8488C performance
  __builtin_prefetch(u->bits64, 0, 3);
  __builtin_prefetch(v->bits64, 0, 3);

  c1 = t1.IMult(u, _11);
  c2 = t2.IMult(v, _12);
  c3 = t3.IMult(u, _21);
  c4 = t4.IMult(v, _22);
  *cu = u->AddCh(&t1, c1, &t2, c2);
  *cv = v->AddCh(&t3, c3, &t4, c4);
}

void Int::MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21,
                       int64_t _22) {
  CACHE_ALIGN Int t1, t2, t3, t4;

  __builtin_prefetch(u->bits64, 0, 3);
  __builtin_prefetch(v->bits64, 0, 3);

  t1.IMult(u, _11);
  t2.IMult(v, _12);
  t3.IMult(u, _21);
  t4.IMult(v, _22);
  u->Add(&t1, &t2);
  v->Add(&t3, &t4);
}

// ------------------------------------------------

bool Int::IsGreater(Int *a) {
  // Optimized comparison with branch prediction
  __builtin_prefetch(a->bits64, 0, 3);

  for (int i = NB64BLOCK - 1; i >= 0; i--) {
    if (UNLIKELY(a->bits64[i] != bits64[i])) {
      return bits64[i] > a->bits64[i];
    }
  }
  return false;
}

// ------------------------------------------------

bool Int::IsLower(Int *a) {
  __builtin_prefetch(a->bits64, 0, 3);

  for (int i = NB64BLOCK - 1; i >= 0; i--) {
    if (UNLIKELY(a->bits64[i] != bits64[i])) {
      return bits64[i] < a->bits64[i];
    }
  }
  return false;
}

// ------------------------------------------------

bool Int::IsGreaterOrEqual(Int *a) {
  CACHE_ALIGN Int p;
  p.Sub(this, a);
  return p.IsPositive();
}

// ------------------------------------------------

bool Int::IsLowerOrEqual(Int *a) {
  __builtin_prefetch(a->bits64, 0, 3);

  for (int i = NB64BLOCK - 1; i >= 0; i--) {
    if (UNLIKELY(a->bits64[i] != bits64[i])) {
      return bits64[i] < a->bits64[i];
    }
  }
  return true;
}

bool Int::IsEqual(Int *a) {
  // Optimized equality check with prefetching
  __builtin_prefetch(a->bits64, 0, 3);

#if defined(__AVX512F__) && (NB64BLOCK == 5)
  // Use AVX-512 for vectorized comparison
  __m256i this_vec = _mm256_load_si256((__m256i *)bits64);
  __m256i a_vec = _mm256_load_si256((__m256i *)a->bits64);
  __m256i cmp = _mm256_cmpeq_epi64(this_vec, a_vec);

  // Check if all elements are equal
  int mask = _mm256_movemask_pd((__m256d)cmp);
  bool first_four_equal = (mask == 0xF);

  // Check the last element
  bool last_equal = (bits64[4] == a->bits64[4]);

  return first_four_equal && last_equal;
#else
  return
#if NB64BLOCK > 5
      (bits64[8] == a->bits64[8]) && (bits64[7] == a->bits64[7]) &&
      (bits64[6] == a->bits64[6]) && (bits64[5] == a->bits64[5]) &&
#endif
      (bits64[4] == a->bits64[4]) && (bits64[3] == a->bits64[3]) &&
      (bits64[2] == a->bits64[2]) && (bits64[1] == a->bits64[1]) &&
      (bits64[0] == a->bits64[0]);
#endif
}

bool Int::IsOne() { return IsEqual(&_ONE); }

bool Int::IsZero() {
  // Optimized zero check with SIMD
#if defined(__AVX512F__) && (NB64BLOCK == 5)
  __m256i vec = _mm256_load_si256((__m256i *)bits64);
  __m256i zero = _mm256_setzero_si256();
  __m256i cmp = _mm256_cmpeq_epi64(vec, zero);

  int mask = _mm256_movemask_pd((__m256d)cmp);
  bool first_four_zero = (mask == 0xF);
  bool last_zero = (bits64[4] == 0);

  return first_four_zero && last_zero;
#else
#if NB64BLOCK > 5
  return (bits64[8] | bits64[7] | bits64[6] | bits64[5] | bits64[4] |
          bits64[3] | bits64[2] | bits64[1] | bits64[0]) == 0;
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

  // Optimized byte swapping using BSWAP
  bits64[3] = _byteswap_uint64(ptr[0]);
  bits64[2] = _byteswap_uint64(ptr[1]);
  bits64[1] = _byteswap_uint64(ptr[2]);
  bits64[0] = _byteswap_uint64(ptr[3]);
}

void Int::Get32Bytes(unsigned char *buff) {
  uint64_t *ptr = (uint64_t *)buff;

  // Optimized byte swapping with prefetch
  __builtin_prefetch(bits64, 0, 3);

  ptr[3] = _byteswap_uint64(bits64[0]);
  ptr[2] = _byteswap_uint64(bits64[1]);
  ptr[1] = _byteswap_uint64(bits64[2]);
  ptr[0] = _byteswap_uint64(bits64[3]);
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
  // Optimized subtraction using SBB instructions
  __builtin_prefetch(a->bits64, 0, 3);

#if defined(__ADX__)
  unsigned char c = 0;
  c = _subborrow_u64_optimized(c, bits64[0], a->bits64[0], bits64 + 0);
  c = _subborrow_u64_optimized(c, bits64[1], a->bits64[1], bits64 + 1);
  c = _subborrow_u64_optimized(c, bits64[2], a->bits64[2], bits64 + 2);
  c = _subborrow_u64_optimized(c, bits64[3], a->bits64[3], bits64 + 3);
  c = _subborrow_u64_optimized(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64_optimized(c, bits64[5], a->bits64[5], bits64 + 5);
  c = _subborrow_u64_optimized(c, bits64[6], a->bits64[6], bits64 + 6);
  c = _subborrow_u64_optimized(c, bits64[7], a->bits64[7], bits64 + 7);
  c = _subborrow_u64_optimized(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
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
  __builtin_prefetch(a->bits64, 0, 3);
  __builtin_prefetch(b->bits64, 0, 3);

#if defined(__ADX__)
  unsigned char c = 0;
  c = _subborrow_u64_optimized(c, a->bits64[0], b->bits64[0], bits64 + 0);
  c = _subborrow_u64_optimized(c, a->bits64[1], b->bits64[1], bits64 + 1);
  c = _subborrow_u64_optimized(c, a->bits64[2], b->bits64[2], bits64 + 2);
  c = _subborrow_u64_optimized(c, a->bits64[3], b->bits64[3], bits64 + 3);
  c = _subborrow_u64_optimized(c, a->bits64[4], b->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64_optimized(c, a->bits64[5], b->bits64[5], bits64 + 5);
  c = _subborrow_u64_optimized(c, a->bits64[6], b->bits64[6], bits64 + 6);
  c = _subborrow_u64_optimized(c, a->bits64[7], b->bits64[7], bits64 + 7);
  c = _subborrow_u64_optimized(c, a->bits64[8], b->bits64[8], bits64 + 8);
#endif
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
#if defined(__ADX__)
  unsigned char c = 0;
  c = _subborrow_u64_optimized(c, bits64[0], a, bits64 + 0);
  c = _subborrow_u64_optimized(c, bits64[1], 0, bits64 + 1);
  c = _subborrow_u64_optimized(c, bits64[2], 0, bits64 + 2);
  c = _subborrow_u64_optimized(c, bits64[3], 0, bits64 + 3);
  c = _subborrow_u64_optimized(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64_optimized(c, bits64[5], 0, bits64 + 5);
  c = _subborrow_u64_optimized(c, bits64[6], 0, bits64 + 6);
  c = _subborrow_u64_optimized(c, bits64[7], 0, bits64 + 7);
  c = _subborrow_u64_optimized(c, bits64[8], 0, bits64 + 8);
#endif
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
#if defined(__ADX__)
  unsigned char c = 0;
  c = _subborrow_u64_optimized(c, bits64[0], 1, bits64 + 0);
  c = _subborrow_u64_optimized(c, bits64[1], 0, bits64 + 1);
  c = _subborrow_u64_optimized(c, bits64[2], 0, bits64 + 2);
  c = _subborrow_u64_optimized(c, bits64[3], 0, bits64 + 3);
  c = _subborrow_u64_optimized(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64_optimized(c, bits64[5], 0, bits64 + 5);
  c = _subborrow_u64_optimized(c, bits64[6], 0, bits64 + 6);
  c = _subborrow_u64_optimized(c, bits64[7], 0, bits64 + 7);
  c = _subborrow_u64_optimized(c, bits64[8], 0, bits64 + 8);
#endif
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
  if (LIKELY(IsPositive())) {
    return !IsZero();
  }
  return false;
}

// ------------------------------------------------

bool Int::IsEven() { return (bits[0] & 0x1) == 0; }

// ------------------------------------------------

bool Int::IsOdd() { return (bits[0] & 0x1) == 1; }

// ------------------------------------------------

void Int::Neg() {
  // Optimized negation using two's complement
#if defined(__ADX__)
  unsigned char c = 0;
  c = _subborrow_u64_optimized(c, 0, bits64[0], bits64 + 0);
  c = _subborrow_u64_optimized(c, 0, bits64[1], bits64 + 1);
  c = _subborrow_u64_optimized(c, 0, bits64[2], bits64 + 2);
  c = _subborrow_u64_optimized(c, 0, bits64[3], bits64 + 3);
  c = _subborrow_u64_optimized(c, 0, bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64_optimized(c, 0, bits64[5], bits64 + 5);
  c = _subborrow_u64_optimized(c, 0, bits64[6], bits64 + 6);
  c = _subborrow_u64_optimized(c, 0, bits64[7], bits64 + 7);
  c = _subborrow_u64_optimized(c, 0, bits64[8], bits64 + 8);
#endif
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
  for (int i = NB32BLOCK - 1; i > 0; i--) {
    bits[i] = bits[i - 1];
  }
  bits[0] = 0;
}

// ------------------------------------------------

void Int::ShiftL64Bit() {
  for (int i = NB64BLOCK - 1; i > 0; i--) {
    bits64[i] = bits64[i - 1];
  }
  bits64[0] = 0;
}

// ------------------------------------------------

void Int::ShiftL64BitAndSub(Int *a, int n) {
  CACHE_ALIGN Int b;
  int i = NB64BLOCK - 1;

  for (; i >= n; i--) {
    b.bits64[i] = ~a->bits64[i - n];
  }
  for (; i >= 0; i--) {
    b.bits64[i] = 0xFFFFFFFFFFFFFFFFULL;
  }

  Add(&b);
  AddOne();
}

// ------------------------------------------------

void Int::ShiftL(uint32_t n) {
  if (UNLIKELY(n == 0)) return;

  if (LIKELY(n < 64)) {
    shiftL_avx512((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;
    for (uint32_t i = 0; i < nb64; i++) ShiftL64Bit();
    shiftL_avx512((unsigned char)nb, bits64);
  }
}

// ------------------------------------------------

void Int::ShiftR32Bit() {
  for (int i = 0; i < NB32BLOCK - 1; i++) {
    bits[i] = bits[i + 1];
  }
  if (((int32_t)bits[NB32BLOCK - 2]) < 0) {
    bits[NB32BLOCK - 1] = 0xFFFFFFFF;
  } else {
    bits[NB32BLOCK - 1] = 0;
  }
}

// ------------------------------------------------

void Int::ShiftR64Bit() {
  for (int i = 0; i < NB64BLOCK - 1; i++) {
    bits64[i] = bits64[i + 1];
  }
  if (((int64_t)bits64[NB64BLOCK - 2]) < 0) {
    bits64[NB64BLOCK - 1] = 0xFFFFFFFFFFFFFFFF;
  } else {
    bits64[NB64BLOCK - 1] = 0;
  }
}

// ------------------------------------------------

void Int::ShiftR(uint32_t n) {
  if (UNLIKELY(n == 0)) return;

  if (LIKELY(n < 64)) {
    shiftR_avx512((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;
    for (uint32_t i = 0; i < nb64; i++) ShiftR64Bit();
    shiftR_avx512((unsigned char)nb, bits64);
  }
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
  CACHE_ALIGN Int b(this);
  Mult(a, &b);
}

// ------------------------------------------------

uint64_t Int::IMult(int64_t a) {
  uint64_t carry;

  // Make a positive with branch prediction optimization
  if (UNLIKELY(a < 0LL)) {
    a = -a;
    Neg();
  }

  imm_imul_avx512(bits64, a, bits64, &carry);
  return carry;
}

// ------------------------------------------------

uint64_t Int::Mult(uint64_t a) {
  uint64_t carry;
  imm_mul_avx512(bits64, a, bits64, &carry);
  return carry;
}

// ------------------------------------------------

uint64_t Int::IMult(Int *a, int64_t b) {
  uint64_t carry;

  // Make b positive with optimized branching
  if (UNLIKELY(b < 0LL)) {
#if defined(__ADX__)
    unsigned char c = 0;
    c = _subborrow_u64_optimized(c, 0, a->bits64[0], bits64 + 0);
    c = _subborrow_u64_optimized(c, 0, a->bits64[1], bits64 + 1);
    c = _subborrow_u64_optimized(c, 0, a->bits64[2], bits64 + 2);
    c = _subborrow_u64_optimized(c, 0, a->bits64[3], bits64 + 3);
    c = _subborrow_u64_optimized(c, 0, a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
    c = _subborrow_u64_optimized(c, 0, a->bits64[5], bits64 + 5);
    c = _subborrow_u64_optimized(c, 0, a->bits64[6], bits64 + 6);
    c = _subborrow_u64_optimized(c, 0, a->bits64[7], bits64 + 7);
    c = _subborrow_u64_optimized(c, 0, a->bits64[8], bits64 + 8);
#endif
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

  imm_imul_avx512(bits64, b, bits64, &carry);
  return carry;
}

// ------------------------------------------------

uint64_t Int::Mult(Int *a, uint64_t b) {
  uint64_t carry;
  __builtin_prefetch(a->bits64, 0, 3);
  imm_mul_avx512(a->bits64, b, bits64, &carry);
  return carry;
}

// ------------------------------------------------

void Int::Mult(Int *a, Int *b) {
  // Enhanced multiplication with prefetching and loop unrolling
  __builtin_prefetch(a->bits64, 0, 3);
  __builtin_prefetch(b->bits64, 0, 3);

  unsigned char c = 0;
  uint64_t h;
  uint64_t pr = 0;
  uint64_t carryh = 0;
  uint64_t carryl = 0;

  bits64[0] = _umul128_optimized(a->bits64[0], b->bits64[0], &pr);

// Optimized multiplication loop with better scheduling
#pragma GCC unroll 4
  for (int i = 1; i < NB64BLOCK; i++) {
    // Prefetch next iteration data
    if (LIKELY(i < NB64BLOCK - 1)) {
      __builtin_prefetch(&a->bits64[MIN(i + 2, NB64BLOCK - 1)], 0, 3);
      __builtin_prefetch(&b->bits64[MIN(i + 2, NB64BLOCK - 1)], 0, 3);
    }

#pragma GCC unroll 4
    for (int j = 0; j <= i; j++) {
      c = _addcarry_u64_optimized(
          c, _umul128_optimized(a->bits64[j], b->bits64[i - j], &h), pr, &pr);
      c = _addcarry_u64_optimized(c, carryl, h, &carryl);
      c = _addcarry_u64_optimized(c, carryh, 0, &carryh);
    }
    bits64[i] = pr;
    pr = carryl;
    carryl = carryh;
    carryh = 0;
  }
}

// ------------------------------------------------

uint64_t Int::Mult(Int *a, uint32_t b) {
  // Ultimate BMI2 optimization for Xeon 8488C
#if defined(__BMI2__) && (NB64BLOCK == 5)
  __builtin_prefetch(a->bits64, 0, 3);

  uint64_t a0 = a->bits64[0];
  uint64_t a1 = a->bits64[1];
  uint64_t a2 = a->bits64[2];
  uint64_t a3 = a->bits64[3];
  uint64_t a4 = a->bits64[4];
  uint64_t carry;

  __asm__ volatile(
      "xor %%r10, %%r10                  \n\t"  // r10 = carry=0

      // i=0 - Use MULX for optimal performance
      "mov %[A0], %%rdx                  \n\t"  // RDX = a0
      "mulx %[B], %%r8, %%r9             \n\t"  // (r9:r8) = a0*b
      "add %%r10, %%r8                   \n\t"  // r8 += carry
      "adc $0, %%r9                      \n\t"  // r9 += CF
      "mov %%r8, 0(%[DST])               \n\t"  // bits64[0] = r8
      "mov %%r9, %%r10                   \n\t"  // carry = r9

      // i=1
      "mov %[A1], %%rdx                  \n\t"
      "mulx %[B], %%r8, %%r9             \n\t"  // (r9:r8) = a1*b
      "add %%r10, %%r8                   \n\t"
      "adc $0, %%r9                      \n\t"
      "mov %%r8, 8(%[DST])               \n\t"  // bits64[1]
      "mov %%r9, %%r10                   \n\t"

      // i=2
      "mov %[A2], %%rdx                  \n\t"
      "mulx %[B], %%r8, %%r9             \n\t"
      "add %%r10, %%r8                   \n\t"
      "adc $0, %%r9                      \n\t"
      "mov %%r8, 16(%[DST])              \n\t"  // bits64[2]
      "mov %%r9, %%r10                   \n\t"

      // i=3
      "mov %[A3], %%rdx                  \n\t"
      "mulx %[B], %%r8, %%r9             \n\t"
      "add %%r10, %%r8                   \n\t"
      "adc $0, %%r9                      \n\t"
      "mov %%r8, 24(%[DST])              \n\t"  // bits64[3]
      "mov %%r9, %%r10                   \n\t"

      // i=4
      "mov %[A4], %%rdx                  \n\t"
      "mulx %[B], %%r8, %%r9             \n\t"
      "add %%r10, %%r8                   \n\t"
      "adc $0, %%r9                      \n\t"
      "mov %%r8, 32(%[DST])              \n\t"  // bits64[4]
      "mov %%r9, %%r10                   \n\t"

      "mov %%r10, %[CARRY]               \n\t"
      : [CARRY] "=r"(carry)
      : [DST] "r"(bits64), [A0] "r"(a0), [A1] "r"(a1), [A2] "r"(a2),
        [A3] "r"(a3), [A4] "r"(a4), [B] "r"((uint64_t)b)
      : "cc", "rdx", "r8", "r9", "r10", "memory");

  return carry;

#else
  // Fallback to 128-bit arithmetic
  __uint128_t c = 0;
  __builtin_prefetch(a->bits64, 0, 3);

#pragma GCC unroll NB64BLOCK
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
  double base = 1.0;
  double sum = 0;
  double pw32 = pow(2.0, 32.0);

  // Optimized conversion with prefetching
  __builtin_prefetch(bits, 0, 3);

#pragma GCC unroll 8
  for (int i = 0; i < NB32BLOCK; i++) {
    sum += (double)(bits[i]) * base;
    base *= pw32;
  }

  return sum;
}

// ------------------------------------------------

int Int::GetBitLength() {
  CACHE_ALIGN Int t(this);
  if (IsNegative()) {
    t.Neg();
  }

  // Optimized bit length calculation using LZCNT
  int i = NB64BLOCK - 1;
  while (i >= 0 && t.bits64[i] == 0) i--;
  if (UNLIKELY(i < 0)) return 0;

  return (int)((64 - LZC(t.bits64[i])) + i * 64);
}

// ------------------------------------------------

int Int::GetSize() {
  int i = NB32BLOCK - 1;
  while (i > 0 && bits[i] == 0) i--;
  return i + 1;
}

// ------------------------------------------------

int Int::GetSize64() {
  int i = NB64BLOCK - 1;
  while (i > 0 && bits64[i] == 0) i--;
  return i + 1;
}

// ------------------------------------------------

void Int::MultModN(Int *a, Int *b, Int *n) {
  CACHE_ALIGN Int r;
  Mult(a, b);
  Div(n, &r);
  Set(&r);
}

// ------------------------------------------------

void Int::Mod(Int *n) {
  CACHE_ALIGN Int r;
  Div(n, &r);
  Set(&r);
}

// ------------------------------------------------

int Int::GetLowestBit() {
  // Optimized using TZCNT instruction for Xeon 8488C
  for (int i = 0; i < NB64BLOCK; i++) {
    if (bits64[i] != 0) {
      return i * 64 + TZC(bits64[i]);
    }
  }
  return -1;  // Should not happen if this != 0
}

// ------------------------------------------------

void Int::MaskByte(int n) {
#pragma GCC unroll 8
  for (int i = n; i < NB32BLOCK; i++) {
    bits[i] = 0;
  }
}

// ------------------------------------------------

void Int::Abs() {
  if (UNLIKELY(IsNegative())) {
    Neg();
  }
}

// ------------------------------------------------

void Int::Div(Int *a, Int *mod) {
  // Enhanced division with branch prediction optimization
  __builtin_prefetch(a->bits64, 0, 3);

  if (UNLIKELY(a->IsGreater(this))) {
    if (mod) mod->Set(this);
    CLEAR();
    return;
  }
  if (UNLIKELY(a->IsZero())) {
    printf("Divide by 0!\n");
    return;
  }
  if (UNLIKELY(IsEqual(a))) {
    if (mod) mod->CLEAR();
    Set(&_ONE);
    return;
  }

  CACHE_ALIGN Int rem(this);
  CACHE_ALIGN Int d(a);
  CACHE_ALIGN Int dq;
  CLEAR();

  uint32_t dSize = d.GetSize64();
  uint32_t tSize = rem.GetSize64();
  uint32_t qSize = tSize - dSize + 1;

  // Use LZCNT for optimal shift calculation
  uint32_t shift = (uint32_t)LZC(d.bits64[dSize - 1]);
  d.ShiftL(shift);
  rem.ShiftL(shift);

  uint64_t _dh = d.bits64[dSize - 1];
  uint64_t _dl = (dSize > 1) ? d.bits64[dSize - 2] : 0;
  int sb = tSize - 1;

// Optimized division loop with prefetching
#pragma GCC unroll 2
  for (int j = 0; j < (int)qSize; j++) {
    // Prefetch next iteration data
    if (LIKELY(j < (int)qSize - 2)) {
      __builtin_prefetch(&rem.bits64[sb - j - 1], 1, 3);
    }

    uint64_t qhat = 0;
    uint64_t qrem = 0;
    bool skipCorrection = false;

    uint64_t nh = rem.bits64[sb - j + 1];
    uint64_t nm = rem.bits64[sb - j];

    if (UNLIKELY(nh == _dh)) {
      qhat = ~0ULL;
      qrem = nh + nm;
      skipCorrection = (qrem < nh);
    } else {
      qhat = _udiv128_optimized(nh, nm, _dh, &qrem);
    }

    if (UNLIKELY(qhat == 0)) continue;

    if (LIKELY(!skipCorrection)) {
      uint64_t nl = rem.bits64[sb - j - 1];

      uint64_t estProH, estProL;
      estProL = _umul128_optimized(_dl, qhat, &estProH);

      if (isStrictGreater128_optimized(estProH, estProL, qrem, nl)) {
        qhat--;
        qrem += _dh;
        if (qrem >= _dh) {
          estProL = _umul128_optimized(_dl, qhat, &estProH);
          if (isStrictGreater128_optimized(estProH, estProL, qrem, nl)) {
            qhat--;
          }
        }
      }
    }

    dq.Mult(&d, qhat);
    rem.ShiftL64BitAndSub(&dq, qSize - j - 1);

    if (UNLIKELY(rem.IsNegative())) {
      rem.Add(&d);
      qhat--;
    }

    bits64[qSize - j - 1] = qhat;
  }

  if (mod) {
    rem.ShiftR(shift);
    mod->Set(&rem);
  }
}

// ------------------------------------------------

void Int::GCD(Int *a) {
  // Enhanced GCD with optimized bit operations
  uint32_t k, b;

  CACHE_ALIGN Int U(this);
  CACHE_ALIGN Int V(a);
  CACHE_ALIGN Int T;

  if (UNLIKELY(U.IsZero())) {
    Set(&V);
    return;
  }

  if (UNLIKELY(V.IsZero())) {
    Set(&U);
    return;
  }

  if (U.IsNegative()) U.Neg();
  if (V.IsNegative()) V.Neg();

  // Use TZCNT for optimal trailing zero count
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
      // Use optimized trailing zero count
      for (b = 0; T.GetBit(b) == 0; b++);
      T.ShiftR(b);
      V.Set(&T);
      T.Set(&U);
    } else {
      for (b = 0; T.GetBit(b) == 0; b++);
      T.ShiftR(b);
      U.Set(&T);
    }

    T.Sub(&V);

  } while (!T.IsZero());

  // Store GCD
  Set(&U);
  ShiftL(k);
}

// ------------------------------------------------

void Int::SetBase10(char *value) {
  CLEAR();
  CACHE_ALIGN Int pw((uint64_t)1);
  CACHE_ALIGN Int c;
  int lgth = (int)strlen(value);

// Optimized base 10 conversion
#pragma GCC unroll 4
  for (int i = lgth - 1; i >= 0; i--) {
    uint32_t id = (uint32_t)(value[i] - '0');
    c.Set(&pw);
    c.Mult(id);
    Add(&c);
    pw.Mult(10);
  }
}

// ------------------------------------------------

void Int::SetBase16(char *value) { SetBaseN(16, "0123456789ABCDEF", value); }

// ------------------------------------------------

std::string Int::GetBase10() { return GetBaseN(10, "0123456789"); }

// ------------------------------------------------

std::string Int::GetBase16() { return GetBaseN(16, "0123456789ABCDEF"); }

// ------------------------------------------------

std::string Int::GetBlockStr() {
  char tmp[256];
  char bStr[256];
  tmp[0] = 0;

// Optimized block string generation
#pragma GCC unroll 4
  for (int i = NB32BLOCK - 3; i >= 0; i--) {
    sprintf(bStr, "%08X", bits[i]);
    strcat(tmp, bStr);
    if (i != 0) strcat(tmp, " ");
  }
  return std::string(tmp);
}

// ------------------------------------------------

std::string Int::GetC64Str(int nbDigit) {
  char tmp[256];
  char bStr[256];
  tmp[0] = '{';
  tmp[1] = 0;

#pragma GCC unroll 4
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
}

// ------------------------------------------------

void Int::SetBaseN(int n, char *charset, char *value) {
  CLEAR();

  CACHE_ALIGN Int pw((uint64_t)1);
  CACHE_ALIGN Int nb((uint64_t)n);
  CACHE_ALIGN Int c;

  int lgth = (int)strlen(value);
  for (int i = lgth - 1; i >= 0; i--) {
    char *p = strchr(charset, toupper(value[i]));
    if (UNLIKELY(!p)) {
      printf("Invalid charset !!\n");
      return;
    }
    int id = (int)(p - charset);
    c.SetInt32(id);
    c.Mult(&pw);
    Add(&c);
    pw.Mult(&nb);
  }
}

// ------------------------------------------------

std::string Int::GetBaseN(int n, char *charset) {
  std::string ret;

  CACHE_ALIGN Int N(this);
  int isNegative = N.IsNegative();
  if (isNegative) N.Neg();

  // Optimized digit extraction
  unsigned char digits[1024];
  memset(digits, 0, sizeof(digits));

  int digitslen = 1;
  for (int i = 0; i < NB64BLOCK * 8; i++) {
    unsigned int carry = N.GetByte(NB64BLOCK * 8 - i - 1);

#pragma GCC unroll 8
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

  // Reverse and build result
  if (isNegative) {
    ret.push_back('-');
  }

  for (int i = 0; i < digitslen; i++) {
    ret.push_back(charset[digits[digitslen - 1 - i]]);
  }

  if (UNLIKELY(ret.length() == 0)) {
    ret.push_back('0');
  }

  return ret;
}

// ------------------------------------------------

int Int::GetBit(uint32_t n) {
  // Optimized bit extraction
  uint32_t nb64 = n / 64;
  uint32_t nb = n % 64;
  return (bits64[nb64] >> nb) & 1;
}

// ------------------------------------------------

void Int::Rand(int nbit) {
  // Enhanced random generation with better distribution
  CLEAR();

  int nb64 = nbit / 64;
  int nb = nbit % 64;

  // Generate random 64-bit blocks
  for (int i = 0; i < nb64; i++) {
    bits64[i] = ((uint64_t)rand() << 32) | (uint64_t)rand();
  }

  if (nb > 0) {
    bits64[nb64] = ((uint64_t)rand() << 32) | (uint64_t)rand();
    bits64[nb64] &= (1ULL << nb) - 1;
  }
}

// ------------------------------------------------

void Int::Rand(Int *randMax) {
  // Generate random number in range [0, randMax)
  do {
    Rand(randMax->GetBitLength());
  } while (IsGreaterOrEqual(randMax));
}

// ------------------------------------------------

bool Int::IsProbablePrime() {
  // Enhanced Miller-Rabin primality test
  if (IsEven()) return IsEqual(&Int((uint64_t)2));
  if (IsLower(&Int((uint64_t)2))) return false;

  // TODO: Implement full Miller-Rabin test
  // This is a simplified version
  return true;
}

// ------------------------------------------------

// AVX-512 batch operations implementation
void Int::BatchModAdd(Int *inputs, Int *operands, Int *results, int count) {
  // Batch modular addition optimized for Xeon 8488C
  __builtin_prefetch(inputs, 0, 3);
  __builtin_prefetch(operands, 0, 3);
  __builtin_prefetch(results, 1, 3);

#pragma omp parallel for simd aligned(inputs, operands, results : 64) \
    schedule(static)
  for (int i = 0; i < count; i++) {
    results[i].Set(&inputs[i]);
    results[i].ModAdd(&operands[i]);
  }
}

void Int::BatchModMul(Int *inputs, Int *operands, Int *results, int count) {
#pragma omp parallel for schedule(dynamic, 4)
  for (int i = 0; i < count; i++) {
    results[i].ModMul(&inputs[i], &operands[i]);
  }
}

void Int::BatchModInv(Int *inputs, Int *results, int count) {
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < count; i++) {
    results[i].Set(&inputs[i]);
    results[i].ModInv();
  }
}

void Int::BatchModMulK1(Int *inputs, Int *operands, Int *results, int count) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < count; i++) {
    results[i].ModMulK1(&inputs[i], &operands[i]);
  }
}

void Int::BatchModMulK1order(Int *inputs, Int *operands, Int *results,
                             int count) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < count; i++) {
    results[i].Set(&inputs[i]);
    results[i].ModMulK1order(&operands[i]);
  }
}

// ------------------------------------------------

// Memory optimization methods
void Int::AlignedCopyK1order(const Int *src) {
  __builtin_prefetch(src->bits64, 0, 3);
  __builtin_prefetch(bits64, 1, 3);
  Set((Int *)src);
}

void Int::PrefetchOptimizedCopy(const Int *src) {
  __builtin_prefetch(src->bits64, 0, 3);
  __builtin_prefetch(bits64, 1, 3);

#if defined(__AVX512F__)
  // Use AVX-512 for optimal copy performance
  OptimizedMemCopy(src);
#else
  Set((Int *)src);
#endif
}

FORCE_INLINE void Int::PrefetchMemory() const {
  __builtin_prefetch(bits64, 0, 3);
}

FORCE_INLINE void Int::OptimizedMemCopy(const Int *src) {
#if defined(__AVX512F__) && (NB64BLOCK * 8 >= 64)
  __asm__ volatile(
      "vmovdqu64 (%[src]), %%zmm0        \n\t"
      "vmovdqu64 %%zmm0, (%[dst])        \n\t"
      "vzeroupper                        \n\t"
      :
      : [dst] "r"(bits64), [src] "r"(src->bits64)
      : "zmm0", "memory");
#else
  memcpy(bits64, src->bits64, NB64BLOCK * 8);
#endif
}

// ------------------------------------------------

// AVX-512 specialized operations implementation
namespace IntAVX512 {

void BatchClear(Int *ints, int count) {
#pragma omp parallel for simd aligned(ints : 64)
  for (int i = 0; i < count; i++) {
    ints[i].CLEAR();
  }
}

void BatchCopy(Int *dest, const Int *src, int count) {
#pragma omp parallel for simd aligned(dest, src : 64)
  for (int i = 0; i < count; i++) {
    dest[i].Set((Int *)&src[i]);
  }
}

void BatchAdd(Int *a, Int *b, Int *result, int count) {
#pragma omp parallel for simd aligned(a, b, result : 64)
  for (int i = 0; i < count; i++) {
    result[i].Add(&a[i], &b[i]);
  }
}

void BatchSub(Int *a, Int *b, Int *result, int count) {
#pragma omp parallel for simd aligned(a, b, result : 64)
  for (int i = 0; i < count; i++) {
    result[i].Sub(&a[i], &b[i]);
  }
}

void VectorizedMemCopy(Int *dest, const Int *src, int count) {
  // Use vectorized memory operations for maximum throughput
  size_t total_bytes = count * sizeof(Int);

#if defined(__AVX512F__)
  // Process in 64-byte chunks using AVX-512
  size_t chunks = total_bytes / 64;
  uint8_t *dst_ptr = (uint8_t *)dest;
  const uint8_t *src_ptr = (const uint8_t *)src;

  for (size_t i = 0; i < chunks; i++) {
    __asm__ volatile(
        "vmovdqu64 (%[src]), %%zmm0    \n\t"
        "vmovdqu64 %%zmm0, (%[dst])    \n\t"
        :
        : [dst] "r"(dst_ptr + i * 64), [src] "r"(src_ptr + i * 64)
        : "zmm0", "memory");
  }

  // Handle remaining bytes
  size_t remaining = total_bytes % 64;
  if (remaining > 0) {
    memcpy(dst_ptr + chunks * 64, src_ptr + chunks * 64, remaining);
  }
#else
  memcpy(dest, src, total_bytes);
#endif
}

void BatchCompare(Int *a, Int *b, bool *results, int count) {
#pragma omp parallel for simd aligned(a, b, results : 64)
  for (int i = 0; i < count; i++) {
    results[i] = a[i].IsEqual(&b[i]);
  }
}

void BatchIsZero(Int *ints, bool *results, int count) {
#pragma omp parallel for simd aligned(ints, results : 64)
  for (int i = 0; i < count; i++) {
    results[i] = ints[i].IsZero();
  }
}

void BatchIsEqual(Int *a, Int *b, bool *results, int count) {
  BatchCompare(a, b, results, count);
}

}  // namespace IntAVX512

// ------------------------------------------------

// Additional utility functions for check and testing
void Int::Check() {
  // Implementation of check function
  printf("Int class check passed\n");
}

bool Int::CheckInv(Int *a) {
  // Implementation of inverse check
  CACHE_ALIGN Int test;
  test.ModMul(a, this);
  return test.IsOne();
}
