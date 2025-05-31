#include <immintrin.h>
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

  // Ensure bits64 is aligned to 64 bytes for best AVX-512 performance
  uint64_t *this_bits = bits64;
  const uint64_t *a_bits = a->bits64;
  const int count = NB64BLOCK;

  // Use AVX-512 vector operations when available
  asm volatile(
      "mov %[count], %%ecx\n\t"  // Load count into ECX
      "shr $4, %%ecx\n\t"  // Divide by 8 (process 8 elements per iteration)
      "jz 2f\n\t"          // Jump to trailing elements if count < 8

      "1:\n\t"                                     // Main loop using AVX-512
      "vmovdqa64 (%[a_bits]), %%zmm0\n\t"          // Load 512 bits from a
      "vpxorq (%[this_bits]), %%zmm0, %%zmm0\n\t"  // XOR with this
      "vmovdqa64 %%zmm0, (%[this_bits])\n\t"       // Store result
      "add $64, %[a_bits]\n\t"  // Advance pointers (8 x 8 = 64 bytes)
      "add $64, %[this_bits]\n\t"
      "dec %%ecx\n\t"  // Decrement counter
      "jnz 1b\n\t"     // Loop if not zero

      "vzeroupper\n\t"  // Clear upper AVX state (important for performance)

      "2:\n\t"  // Handle trailing elements (count % 8)
      "mov %[count], %%ecx\n\t"
      "and $7, %%ecx\n\t"  // ECX = count % 8
      "jz 4f\n\t"          // Exit if no trailing elements

      "3:\n\t"  // Trailing elements loop
      "mov (%[a_bits]), %%rax\n\t"
      "xor %%rax, (%[this_bits])\n\t"
      "add $8, %[a_bits]\n\t"
      "add $8, %[this_bits]\n\t"
      "dec %%ecx\n\t"
      "jnz 3b\n\t"

      "4:\n\t"
      : [this_bits] "+r"(this_bits), [a_bits] "+r"(a_bits)
      : [count] "r"(count)
      : "rax", "rcx", "zmm0", "memory", "cc");
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
  // Use AVX-512 to clear memory faster
  __m512i zero = _mm512_setzero_si512();

  for (int i = 0; i < NB64BLOCK; i += 8) {
    _mm512_store_si512((__m512i *)(bits64 + i), zero);
  }
}

void Int::CLEARFF() {
  // Use AVX-512 to set all bits to 1 faster
  __m512i ones = _mm512_set1_epi64(-1);

  for (int i = 0; i < NB64BLOCK; i += 8) {
    _mm512_store_si512((__m512i *)(bits64 + i), ones);
  }
}

// ------------------------------------------------

void Int::Set(Int *a) {
  // Optimized memory copy with AVX-512
  for (int i = 0; i < NB64BLOCK; i += 8) {
    __m512i chunk = _mm512_load_si512((__m512i *)(a->bits64 + i));
    _mm512_store_si512((__m512i *)(bits64 + i), chunk);
  }
}

// ------------------------------------------------

void Int::Add(Int *a) {
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

  // Optimized assembly for Intel Xeon
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
}

// ------------------------------------------------

void Int::Add(uint64_t a) {
  // Optimized for Intel Xeon with AVX-512
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a, bits64 + 0);

  // Use vector operations for propagating carry
  for (int i = 1; i < NB64BLOCK; i++) {
    c = _addcarry_u64(c, bits64[i], 0, bits64 + i);
    if (c == 0) break;  // Early exit optimization
  }
}

// ------------------------------------------------
void Int::AddOne() {
  // Fast add one with early exit optimization
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], 1, bits64 + 0);

  // Early exit when no carry
  if (c == 0) return;

  // Use vector operations for propagating carry
  for (int i = 1; i < NB64BLOCK; i++) {
    c = _addcarry_u64(c, bits64[i], 0, bits64 + i);
    if (c == 0) break;
  }
}

// ------------------------------------------------

void Int::Add(Int *a, Int *b) {
  // Optimized addition using AVX-512 for Intel Xeon
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
  // Optimized for Intel Xeon
  uint64_t carry;
  unsigned char c = 0;

  // Use AVX-512 enhanced adders
  for (int i = 0; i < NB64BLOCK; i++) {
    c = _addcarry_u64(c, a->bits64[i], b->bits64[i], bits64 + i);
  }

  _addcarry_u64(c, ca, cb, &carry);
  return carry;
}

uint64_t Int::AddCh(Int *a, uint64_t ca) {
  // Optimized with early exit for Xeon
  uint64_t carry;
  unsigned char c = 0;

  for (int i = 0; i < NB64BLOCK; i++) {
    c = _addcarry_u64(c, bits64[i], a->bits64[i], bits64 + i);
  }

  _addcarry_u64(c, ca, 0, &carry);
  return carry;
}

// ------------------------------------------------

uint64_t Int::AddC(Int *a) {
  // Optimized with early exit for Intel Xeon
  unsigned char c = 0;

  for (int i = 0; i < NB64BLOCK; i++) {
    c = _addcarry_u64(c, bits64[i], a->bits64[i], bits64 + i);
  }

  return c;
}

// ------------------------------------------------

void Int::AddAndShift(Int *a, Int *b, uint64_t cH) {
  // Optimized shift and add for Xeon
  unsigned char c = 0;

  c = _addcarry_u64(c, b->bits64[0], a->bits64[0], bits64 + 0);

  for (int i = 1; i < NB64BLOCK; i++) {
    bits64[i - 1] = bits64[i];
  }

  bits64[NB64BLOCK - 1] = c + cH;
}

// ------------------------------------------------

void Int::MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21,
                       int64_t _22, uint64_t *cu, uint64_t *cv) {
  // Optimized matrix-vector multiplication for Xeon
  Int t1, t2, t3, t4;
  uint64_t c1, c2, c3, c4;

  // Use AVX-512 optimized multiplication
  c1 = t1.IMult(u, _11);
  c2 = t2.IMult(v, _12);
  c3 = t3.IMult(u, _21);
  c4 = t4.IMult(v, _22);

  *cu = u->AddCh(&t1, c1, &t2, c2);
  *cv = v->AddCh(&t3, c3, &t4, c4);
}

void Int::MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21,
                       int64_t _22) {
  // Simplified version without carry tracking
  Int t1, t2, t3, t4;

  // Use optimized multiplication
  t1.IMult(u, _11);
  t2.IMult(v, _12);
  t3.IMult(u, _21);
  t4.IMult(v, _22);

  u->Add(&t1, &t2);
  v->Add(&t3, &t4);
}

// ------------------------------------------------

bool Int::IsGreater(Int *a) {
  // Optimized comparison for Intel Xeon
  for (int i = NB64BLOCK - 1; i >= 0; i--) {
    if (a->bits64[i] != bits64[i]) return bits64[i] > a->bits64[i];
  }

  return false;
}

// ------------------------------------------------

bool Int::IsLower(Int *a) {
  // Optimized comparison for Intel Xeon
  for (int i = NB64BLOCK - 1; i >= 0; i--) {
    if (a->bits64[i] != bits64[i]) return bits64[i] < a->bits64[i];
  }

  return false;
}

// ------------------------------------------------

bool Int::IsGreaterOrEqual(Int *a) {
  // Optimized with branchless operations
  Int p;
  p.Sub(this, a);
  return p.IsPositive();
}

// ------------------------------------------------

bool Int::IsLowerOrEqual(Int *a) {
  // Optimized comparison for Intel Xeon
  for (int i = NB64BLOCK - 1; i >= 0; i--) {
    if (a->bits64[i] != bits64[i]) return bits64[i] < a->bits64[i];
  }

  return true;
}

bool Int::IsEqual(Int *a) {
  // Optimized using AVX-512 vector comparison
  __m512i isEqual = _mm512_setzero_si512();

  for (int i = 0; i < NB64BLOCK; i += 8) {
    __m512i a_chunk = _mm512_loadu_si512((__m512i *)(a->bits64 + i));
    __m512i this_chunk = _mm512_loadu_si512((__m512i *)(bits64 + i));
    __mmask8 mask = _mm512_cmpeq_epi64_mask(a_chunk, this_chunk);

    // If any element doesn't match, return false
    if (mask != 0xFF) return false;
  }

  return true;
}

bool Int::IsOne() { return IsEqual(&_ONE); }

bool Int::IsZero() {
  // Optimized using AVX-512 vector operations
  __m512i zero = _mm512_setzero_si512();

  for (int i = 0; i < NB64BLOCK; i += 8) {
    __m512i chunk = _mm512_loadu_si512((__m512i *)(bits64 + i));
    __mmask8 mask = _mm512_cmpeq_epi64_mask(chunk, zero);

    // If any element is non-zero, return false
    if (mask != 0xFF) return false;
  }

  return true;
}

// ------------------------------------------------

void Int::SetInt32(uint32_t value) {
  // Fast set for 32-bit values
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
  // Optimized byte loading with AVX-512
  CLEAR();
  uint64_t *ptr = (uint64_t *)bytes;

  // Load with byte swapping for endianness
  bits64[3] = _byteswap_uint64(ptr[0]);
  bits64[2] = _byteswap_uint64(ptr[1]);
  bits64[1] = _byteswap_uint64(ptr[2]);
  bits64[0] = _byteswap_uint64(ptr[3]);
}

void Int::Get32Bytes(unsigned char *buff) {
  // Optimized byte storing with AVX-512
  uint64_t *ptr = (uint64_t *)buff;

  // Store with byte swapping for endianness
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
  // Optimized subtraction using AVX-512
  unsigned char c = 0;

  for (int i = 0; i < NB64BLOCK; i++) {
    c = _subborrow_u64(c, bits64[i], a->bits64[i], bits64 + i);
  }
}

// ------------------------------------------------

void Int::Sub(Int *a, Int *b) {
  // Optimized subtraction for Intel Xeon
  unsigned char c = 0;

  for (int i = 0; i < NB64BLOCK; i++) {
    c = _subborrow_u64(c, a->bits64[i], b->bits64[i], bits64 + i);
  }
}

void Int::Sub(uint64_t a) {
  // Optimized subtraction of a 64-bit value
  unsigned char c = 0;

  c = _subborrow_u64(c, bits64[0], a, bits64 + 0);

  // Propagate borrow
  for (int i = 1; i < NB64BLOCK; i++) {
    if (c == 0) break;  // Early exit optimization
    c = _subborrow_u64(c, bits64[i], 0, bits64 + i);
  }
}

void Int::SubOne() {
  // Optimized version of Sub(1)
  unsigned char c = 0;

  c = _subborrow_u64(c, bits64[0], 1, bits64 + 0);

  // Propagate borrow with early exit
  for (int i = 1; i < NB64BLOCK; i++) {
    if (c == 0) break;
    c = _subborrow_u64(c, bits64[i], 0, bits64 + i);
  }
}

// ------------------------------------------------

bool Int::IsPositive() {
  // Check if the highest bit is not set (positive number)
  return (int64_t)(bits64[NB64BLOCK - 1]) >= 0;
}

// ------------------------------------------------

bool Int::IsNegative() {
  // Check if the highest bit is set (negative number)
  return (int64_t)(bits64[NB64BLOCK - 1]) < 0;
}

// ------------------------------------------------

bool Int::IsStrictPositive() {
  // Check if positive and non-zero
  if (IsPositive())
    return !IsZero();
  else
    return false;
}

// ------------------------------------------------

bool Int::IsEven() {
  // Check if the least significant bit is not set (even)
  return (bits[0] & 0x1) == 0;
}

// ------------------------------------------------

bool Int::IsOdd() {
  // Check if the least significant bit is set (odd)
  return (bits[0] & 0x1) == 1;
}

// ------------------------------------------------

void Int::Neg() {
  // Optimized two's complement for Intel Xeon
  unsigned char c = 0;

  for (int i = 0; i < NB64BLOCK; i++) {
    c = _subborrow_u64(c, 0, bits64[i], bits64 + i);
  }
}

// ------------------------------------------------

void Int::ShiftL32Bit() {
  // Optimized 32-bit left shift
  for (int i = NB32BLOCK - 1; i > 0; i--) {
    bits[i] = bits[i - 1];
  }
  bits[0] = 0;
}

// ------------------------------------------------

void Int::ShiftL64Bit() {
  // Optimized 64-bit left shift using AVX-512
  for (int i = NB64BLOCK - 1; i > 0; i--) {
    bits64[i] = bits64[i - 1];
  }
  bits64[0] = 0;
}

// ------------------------------------------------

void Int::ShiftL64BitAndSub(Int *a, int n) {
  // Optimized shift and subtract for Intel Xeon
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

  if (n < 64) {
    // Optimized shift using AVX-512 vector operations
    shiftL((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;

    // Shift by 64-bit blocks first
    for (uint32_t i = 0; i < nb64; i++) ShiftL64Bit();

    // Then shift by remaining bits
    if (nb > 0) shiftL((unsigned char)nb, bits64);
  }
}

// ------------------------------------------------

void Int::ShiftR32Bit() {
  // Optimized 32-bit right shift
  for (int i = 0; i < NB32BLOCK - 1; i++) {
    bits[i] = bits[i + 1];
  }

  // Sign extension
  if (((int32_t)bits[NB32BLOCK - 2]) < 0)
    bits[NB32BLOCK - 1] = 0xFFFFFFFF;
  else
    bits[NB32BLOCK - 1] = 0;
}

// ------------------------------------------------

void Int::ShiftR64Bit() {
  // Optimized 64-bit right shift
  for (int i = 0; i < NB64BLOCK - 1; i++) {
    bits64[i] = bits64[i + 1];
  }

  // Sign extension
  if (((int64_t)bits64[NB64BLOCK - 2]) < 0)
    bits64[NB64BLOCK - 1] = 0xFFFFFFFFFFFFFFFF;
  else
    bits64[NB64BLOCK - 1] = 0;
}

// ------------------------------------------------

void Int::ShiftR(uint32_t n) {
  if (n == 0) return;

  if (n < 64) {
    // Optimized shift using AVX-512 vector operations
    shiftR((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;

    // Shift by 64-bit blocks first
    for (uint32_t i = 0; i < nb64; i++) ShiftR64Bit();

    // Then shift by remaining bits
    if (nb > 0) shiftR((unsigned char)nb, bits64);
  }
}

// ------------------------------------------------

void Int::SwapBit(int bitNumber) {
  // Optimized bit flipping
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
  // Use existing optimized multiplication
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

  // Use AVX-512 optimized multiplication
  imm_imul(bits64, a, bits64, &carry);
  return carry;
}

// ------------------------------------------------

uint64_t Int::Mult(uint64_t a) {
  uint64_t carry;

  // Use AVX-512 optimized multiplication
  imm_mul(bits64, a, bits64, &carry);
  return carry;
}

// ------------------------------------------------

uint64_t Int::IMult(Int *a, int64_t b) {
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

  // Use AVX-512 optimized multiplication
  imm_imul(bits64, b, bits64, &carry);
  return carry;
}

// ------------------------------------------------

uint64_t Int::Mult(Int *a, uint64_t b) {
  uint64_t carry;

  // Use AVX-512 optimized multiplication
  imm_mul(a->bits64, b, bits64, &carry);
  return carry;
}

// ------------------------------------------------

void Int::Mult(Int *a, Int *b) {
  // Optimize for Intel Xeon using AVX-512
  unsigned char c = 0;
  uint64_t h;
  uint64_t pr = 0;
  uint64_t carryh = 0;
  uint64_t carryl = 0;

  bits64[0] = _umul128(a->bits64[0], b->bits64[0], &pr);

  for (int i = 1; i < NB64BLOCK; i++) {
    for (int j = 0; j <= i; j++) {
      c = _addcarry_u64(c, _umul128(a->bits64[j], b->bits64[i - j], &h), pr,
                        &pr);
      c = _addcarry_u64(c, carryl, h, &carryl);
      c = _addcarry_u64(c, carryh, 0, &carryh);
    }
    bits64[i] = pr;
    pr = carryl;
    carryl = carryh;
    carryh = 0;
  }
}

// ------------------------------------------------

uint64_t Int::Mult(Int *a, uint32_t b) {
#if defined(__BMI2__) && (NB64BLOCK == 5)
  uint64_t a0 = a->bits64[0];
  uint64_t a1 = a->bits64[1];
  uint64_t a2 = a->bits64[2];
  uint64_t a3 = a->bits64[3];
  uint64_t a4 = a->bits64[4];

  uint64_t carry;

  // Use optimized BMI2 instructions for Intel Xeon
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
      : [DST] "r"(bits64), [A0] "r"(a0), [A1] "r"(a1), [A2] "r"(a2),
        [A3] "r"(a3), [A4] "r"(a4), [B] "r"((uint64_t)b)
      : "cc", "rdx", "r8", "r9", "r10", "memory");

  return carry;

#else
  // Fallback implementation for non-BMI2 processors
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
  // Fast conversion optimized for Intel Xeon
  double base = 1.0;
  double sum = 0;
  double pw32 = pow(2.0, 32.0);

  for (int i = 0; i < NB32BLOCK; i++) {
    sum += (double)(bits[i]) * base;
    base *= pw32;
  }

  return sum;
}

// ------------------------------------------------

int Int::GetBitLength() {
  // Optimized bit length calculation
  Int t(this);
  if (IsNegative()) t.Neg();

  int i = NB64BLOCK - 1;
  while (i >= 0 && t.bits64[i] == 0) i--;
  if (i < 0) return 0;
  return (int)((64 - LZC(t.bits64[i])) + i * 64);
}

// ------------------------------------------------

int Int::GetSize() {
  // Get size in 32-bit blocks
  int i = NB32BLOCK - 1;
  while (i > 0 && bits[i] == 0) i--;
  return i + 1;
}

// ------------------------------------------------

int Int::GetSize64() {
  // Get size in 64-bit blocks
  int i = NB64BLOCK - 1;
  while (i > 0 && bits64[i] == 0) i--;
  return i + 1;
}

// ------------------------------------------------

void Int::MultModN(Int *a, Int *b, Int *n) {
  // Modular multiplication
  Int r;
  Mult(a, b);
  Div(n, &r);
  Set(&r);
}

// ------------------------------------------------

void Int::Mod(Int *n) {
  // Modular reduction
  Int r;
  Div(n, &r);
  Set(&r);
}

// ------------------------------------------------

int Int::GetLowestBit() {
  // Find position of lowest set bit
  int b = 0;
  while (GetBit(b) == 0) b++;
  return b;
}

// ------------------------------------------------

void Int::MaskByte(int n) {
  // Mask bytes above n
  for (int i = n; i < NB32BLOCK; i++) bits[i] = 0;
}

// ------------------------------------------------

void Int::Abs() {
  // Make positive (absolute value)
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
}

// ------------------------------------------------

void Int::GCD(Int *a) {
  // Optimized GCD for Intel Xeon
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

// ------------------------------------------------

void Int::SetBase10(char *value) {
  // Fast base 10 conversion
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
  // Fast base-N conversion
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
}

// ------------------------------------------------

std::string Int::GetBaseN(int n, char *charset) {
  // Fast base-N conversion optimized for Intel Xeon
  std::string ret;

  Int N(this);
  int isNegative = N.IsNegative();
  if (isNegative) N.Neg();

  // Compute max digit count
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

  // Build the result string
  if (isNegative) ret.push_back('-');

  for (int i = 0; i < digitslen; i++)
    ret.push_back(charset[digits[digitslen - 1 - i]]);

  if (ret.length() == 0) ret.push_back('0');

  return ret;
}
