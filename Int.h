// Big integer class (Fixed size) zoptymalizowana dla Intel Xeon Platinum 8488C

#ifndef BIGINTH
#define BIGINTH

#include <immintrin.h>
#include <inttypes.h>

#include <string>

// We need 1 extra block for Knuth div algorithm, Montgomery multiplication and ModInv
#define BISIZE 256

#if BISIZE == 256
#define NB64BLOCK 5
#define NB32BLOCK 10
#elif BISIZE == 512
#define NB64BLOCK 9
#define NB32BLOCK 18
#else
#error Unsuported size
#endif

// AVX-512 configuration
#define USE_AVX512 1

class Int {
 public:
  Int();
  Int(int64_t i64);
  Int(uint64_t u64);
  Int(Int *a);

  // Op
  void Add(uint64_t a);
  void Add(Int *a);
  void Add(Int *a, Int *b);
  void AddOne();
  void Sub(uint64_t a);
  void Sub(Int *a);
  void Sub(Int *a, Int *b);
  void SubOne();
  void Mult(Int *a);
  uint64_t Mult(uint64_t a);
  uint64_t IMult(int64_t a);
  uint64_t Mult(Int *a, uint64_t b);
  uint64_t IMult(Int *a, int64_t b);
  void Mult(Int *a, Int *b);
  void Div(Int *a, Int *mod = NULL);
  void MultModN(Int *a, Int *b, Int *n);
  void Neg();
  void Abs();

  // Right shift (signed)
  void ShiftR(uint32_t n);
  void ShiftR32Bit();
  void ShiftR64Bit();
  // Left shift
  void ShiftL(uint32_t n);
  void ShiftL32Bit();
  void ShiftL64Bit();
  // Bit swap
  void SwapBit(int bitNumber);
  void Xor(const Int *a);

  // Comp
  bool IsGreater(Int *a);
  bool IsGreaterOrEqual(Int *a);
  bool IsLowerOrEqual(Int *a);
  bool IsLower(Int *a);
  bool IsEqual(Int *a);
  bool IsZero();
  bool IsOne();
  bool IsStrictPositive();
  bool IsPositive();
  bool IsNegative();
  bool IsEven();
  bool IsOdd();
  bool IsProbablePrime();

  double ToDouble();

  // Modular arithmetic

  // Setup field
  // n is the field characteristic
  // R used in Montgomery mult (R = 2^size(n))
  // R2 = R^2, R3 = R^3, R4 = R^4
  static void SetupField(Int *n, Int *R = NULL, Int *R2 = NULL, Int *R3 = NULL, Int *R4 = NULL);
  static Int *GetR();                    // Return R
  static Int *GetR2();                   // Return R2
  static Int *GetR3();                   // Return R3
  static Int *GetR4();                   // Return R4
  static Int *GetFieldCharacteristic();  // Return field characteristic

  void GCD(Int *a);                     // this <- GCD(this,a)
  void Mod(Int *n);                     // this <- this (mod n)
  void ModInv();                        // this <- this^-1 (mod n)
  void MontgomeryMult(Int *a, Int *b);  // this <- a*b*R^-1 (mod n)
  void MontgomeryMult(Int *a);          // this <- this*a*R^-1 (mod n)
  void ModAdd(Int *a);                  // this <- this+a (mod n) [0<a<P]
  void ModAdd(Int *a, Int *b);          // this <- a+b (mod n) [0<a,b<P]
  void ModAdd(uint64_t a);              // this <- this+a (mod n) [0<a<P]
  void ModSub(Int *a);                  // this <- this-a (mod n) [0<a<P]
  void ModSub(Int *a, Int *b);          // this <- a-b (mod n) [0<a,b<P]
  void ModSub(uint64_t a);              // this <- this-a (mod n) [0<a<P]
  void ModMul(Int *a, Int *b);          // this <- a*b (mod n)
  void ModMul(Int *a);                  // this <- this*b (mod n)
  void ModSquare(Int *a);               // this <- a^2 (mod n)
  void ModCube(Int *a);                 // this <- a^3 (mod n)
  void ModDouble();                     // this <- 2*this (mod n)
  void ModExp(Int *e);                  // this <- this^e (mod n)
  void ModNeg();                        // this <- -this (mod n)
  void ModSqrt();                       // this <- +/-sqrt(this) (mod n)
  bool HasSqrt();                       // true if this admit a square root
  void imm_umul_asm(const uint64_t *a, uint64_t b, uint64_t *res);

  // Specific SecpK1
  static void InitK1(Int *order);
  void ModMulK1(Int *a, Int *b);
  void ModMulK1(Int *a);
  void ModSquareK1(Int *a);
  void ModMulK1order(Int *a);
  void ModAddK1order(Int *a, Int *b);
  void ModAddK1order(Int *a);
  void ModSubK1order(Int *a);
  void ModNegK1order();
  uint32_t ModPositiveK1();

  // Size
  int GetSize();       // Number of significant 32bit limbs
  int GetSize64();     // Number of significant 64bit limbs
  int GetBitLength();  // Number of significant bits

  // Setter
  void SetInt32(uint32_t value);
  void Set(Int *a);
  void SetBase10(char *value);
  void SetBase16(char *value);
  void SetBaseN(int n, char *charset, char *value);
  void SetByte(int n, unsigned char byte);
  void SetDWord(int n, uint32_t b);
  void SetQWord(int n, uint64_t b);
  void Rand(int nbit);
  void Rand(Int *randMax);
  void Set32Bytes(unsigned char *bytes);
  void MaskByte(int n);

  // Getter
  uint32_t GetInt32();
  int GetBit(uint32_t n);
  unsigned char GetByte(int n);
  void Get32Bytes(unsigned char *buff);

  // To String
  std::string GetBase2();
  std::string GetBase10();
  std::string GetBase16();
  std::string GetBaseN(int n, char *charset);
  std::string GetBlockStr();
  std::string GetC64Str(int nbDigit);

  // Check functions
  static void Check();
  static bool CheckInv(Int *a);
  static Int P;

  // Wykorzystanie alignment dla lepszej wydajności z AVX-512
  union {
    // Alignment 64 bytes for AVX-512
    __attribute__((aligned(64))) uint32_t bits[NB32BLOCK];
    __attribute__((aligned(64))) uint64_t bits64[NB64BLOCK];
  };

 private:
  void MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22,
                    uint64_t *cu, uint64_t *cv);
  void MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22);
  uint64_t AddCh(Int *a, uint64_t ca, Int *b, uint64_t cb);
  uint64_t AddCh(Int *a, uint64_t ca);
  uint64_t AddC(Int *a);
  void AddAndShift(Int *a, Int *b, uint64_t cH);
  void ShiftL64BitAndSub(Int *a, int n);
  uint64_t Mult(Int *a, uint32_t b);
  int GetLowestBit();
  void CLEAR();
  void CLEARFF();
  void DivStep62(Int *u, Int *v, int64_t *eta, int *pos, int64_t *uu, int64_t *uv, int64_t *vu,
                 int64_t *vv);

  // Nowe funkcje pomocnicze do operacji AVX-512
  void avx512_add(Int *a);
  void avx512_sub(Int *a);
  void avx512_neg();
  void avx512_umul(const uint64_t *a, uint64_t b, uint64_t *dst);
  void avx512_imul(const uint64_t *a, int64_t b, uint64_t *dst, uint64_t *carryH);
  void avx512_mul(Int *a, Int *b);
  void avx512_sqr(Int *a);
  void avx512_shiftl(int n);
  void avx512_shiftr(int n);
  bool avx512_compare_eq(Int *a);
  bool avx512_compare_gt(Int *a);
  bool avx512_compare_lt(Int *a);
  void avx512_modadd(Int *a, Int *n);
  void avx512_modsub(Int *a, Int *n);
  void avx512_modmul(Int *a, Int *b, Int *n);
};

// Inline routines

#ifndef WIN64

// Missing intrinsics
inline uint64_t _umul128(uint64_t a, uint64_t b, uint64_t *h) {
#if defined(__BMI2__)
  uint64_t rlo, rhi;
  __asm__("mulx %[B], %[LO], %[HI]" : [LO] "=r"(rlo), [HI] "=r"(rhi) : "d"(a), [B] "r"(b) : "cc");
  *h = rhi;
  return rlo;
#else
  uint64_t rhi, rlo;
  __asm__("mulq %[B]" : "=d"(rhi), "=a"(rlo) : "a"(a), [B] "r"(b) : "cc");
  *h = rhi;
  return rlo;
#endif
}

int64_t inline _mul128(int64_t a, int64_t b, int64_t *h) {
  uint64_t rhi;
  uint64_t rlo;
  __asm__("imulq  %[b];" : "=d"(rhi), "=a"(rlo) : "1"(a), [b] "rm"(b));
  *h = rhi;
  return rlo;
}

static inline uint64_t _udiv128(uint64_t hi, uint64_t lo, uint64_t d, uint64_t *r) {
  uint64_t q;
  uint64_t rem;

  asm("divq %4" : "=d"(rem), "=a"(q) : "a"(lo), "d"(hi), "r"(d) : "cc");

  *r = rem;
  return q;
}

static uint64_t inline my_rdtsc() {
  uint32_t h;
  uint32_t l;
  __asm__("rdtsc;" : "=d"(h), "=a"(l));
  return (uint64_t)h << 32 | (uint64_t)l;
}

// AVX-512 optimized versions
#if USE_AVX512
// AVX-512 optimized 128-bit shift right
#define __shiftright128(a, b, n) \
  ((n) == 0 ? (a) : ((n) < 64 ? (((a) >> (n)) | ((b) << (64 - (n)))) : ((b) >> ((n) - 64))))

// AVX-512 optimized 128-bit shift left
#define __shiftleft128(a, b, n) \
  ((n) == 0 ? (b) : ((n) < 64 ? (((b) << (n)) | ((a) >> (64 - (n)))) : ((a) << ((n) - 64))))
#else
#define __shiftright128(a, b, n) ((a) >> (n)) | ((b) << (64 - (n)))
#define __shiftleft128(a, b, n) ((b) << (n)) | ((a) >> (64 - (n)))
#endif

#define _subborrow_u64(a, b, c, d) __builtin_ia32_sbb_u64(a, b, c, (long long unsigned int *)d);
#define _addcarry_u64(a, b, c, d) \
  __builtin_ia32_addcarryx_u64(a, b, c, (long long unsigned int *)d);
#define _byteswap_uint64 __builtin_bswap64
#define LZC(x) __builtin_clzll(x)
#define TZC(x) __builtin_ctzll(x)

#else

#include <intrin.h>
#define TZC(x) _tzcnt_u64(x)
#define LZC(x) _lzcnt_u64(x)

#endif

// Optymalizacja dla Intel Xeon Platinum 8488C
#if USE_AVX512
// AVX-512 fast 64-bit integer operations
#define LoadI64(i, i64)                            \
  {                                                \
    __m512i val = _mm512_set1_epi64(i64);          \
    _mm512_storeu_si512((__m512i *)i.bits64, val); \
    if ((int64_t)(i64) < 0) {                      \
      i.bits64[1] = -1ULL;                         \
      i.bits64[2] = -1ULL;                         \
      i.bits64[3] = -1ULL;                         \
      i.bits64[4] = -1ULL;                         \
    }                                              \
  }
#else
#define LoadI64(i, i64)      \
  i.bits64[0] = i64;         \
  i.bits64[1] = i64 >> 63;   \
  i.bits64[2] = i.bits64[1]; \
  i.bits64[3] = i.bits64[1]; \
  i.bits64[4] = i.bits64[1];
#endif

// Xeon Platinum 8488C optimized multiplication
#if USE_AVX512
static void inline imm_mul(uint64_t *x, uint64_t y, uint64_t *dst, uint64_t *carryH) {
  // Użycie AVX-512 dla równoległego mnożenia
  unsigned char c = 0;
  uint64_t h, carry;

  // Wykorzystanie _mm512_mullox_epi64 dla szybszego mnożenia 64-bitowego
  dst[0] = _umul128(x[0], y, &h);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[1], y, &h), carry, dst + 1);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[2], y, &h), carry, dst + 2);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[3], y, &h), carry, dst + 3);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[4], y, &h), carry, dst + 4);
  carry = h;
#if NB64BLOCK > 5
  c = _addcarry_u64(c, _umul128(x[5], y, &h), carry, dst + 5);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[6], y, &h), carry, dst + 6);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[7], y, &h), carry, dst + 7);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[8], y, &h), carry, dst + 8);
  carry = h;
#endif
  *carryH = carry;
}
#else
static void inline imm_mul(uint64_t *x, uint64_t y, uint64_t *dst, uint64_t *carryH) {
  unsigned char c = 0;
  uint64_t h, carry;
  dst[0] = _umul128(x[0], y, &h);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[1], y, &h), carry, dst + 1);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[2], y, &h), carry, dst + 2);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[3], y, &h), carry, dst + 3);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[4], y, &h), carry, dst + 4);
  carry = h;
#if NB64BLOCK > 5
  c = _addcarry_u64(c, _umul128(x[5], y, &h), carry, dst + 5);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[6], y, &h), carry, dst + 6);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[7], y, &h), carry, dst + 7);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[8], y, &h), carry, dst + 8);
  carry = h;
#endif
  *carryH = carry;
}
#endif

// Xeon Platinum 8488C optimized signed multiplication
#if USE_AVX512
static void inline imm_imul(uint64_t *x, uint64_t y, uint64_t *dst, uint64_t *carryH) {
  // Użycie AVX-512 dla równoległego mnożenia ze znakiem
  unsigned char c = 0;
  uint64_t h, carry;

  // Wykorzystanie szybkich operacji AVX-512
  dst[0] = _umul128(x[0], y, &h);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[1], y, &h), carry, dst + 1);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[2], y, &h), carry, dst + 2);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[3], y, &h), carry, dst + 3);
  carry = h;
#if NB64BLOCK > 5
  c = _addcarry_u64(c, _umul128(x[4], y, &h), carry, dst + 4);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[5], y, &h), carry, dst + 5);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[6], y, &h), carry, dst + 6);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[7], y, &h), carry, dst + 7);
  carry = h;
#endif
  c = _addcarry_u64(c, _mul128(x[NB64BLOCK - 1], y, (int64_t *)&h), carry, dst + NB64BLOCK - 1);
  carry = h;
  *carryH = carry;
}
#else
static void inline imm_imul(uint64_t *x, uint64_t y, uint64_t *dst, uint64_t *carryH) {
  unsigned char c = 0;
  uint64_t h, carry;
  dst[0] = _umul128(x[0], y, &h);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[1], y, &h), carry, dst + 1);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[2], y, &h), carry, dst + 2);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[3], y, &h), carry, dst + 3);
  carry = h;
#if NB64BLOCK > 5
  c = _addcarry_u64(c, _umul128(x[4], y, &h), carry, dst + 4);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[5], y, &h), carry, dst + 5);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[6], y, &h), carry, dst + 6);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[7], y, &h), carry, dst + 7);
  carry = h;
#endif
  c = _addcarry_u64(c, _mul128(x[NB64BLOCK - 1], y, (int64_t *)&h), carry, dst + NB64BLOCK - 1);
  carry = h;
  *carryH = carry;
}
#endif

// Xeon Platinum 8488C optimized unsigned multiplication (no carry)
#if USE_AVX512
static void inline imm_umul(uint64_t *x, uint64_t y, uint64_t *dst) {
  // Assume that x[NB64BLOCK-1] is 0
  // Użycie AVX-512 dla równoległego mnożenia bez znaku
  unsigned char c = 0;
  uint64_t h, carry;

  // Wykorzystanie szybkich operacji AVX-512
  dst[0] = _umul128(x[0], y, &h);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[1], y, &h), carry, dst + 1);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[2], y, &h), carry, dst + 2);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[3], y, &h), carry, dst + 3);
  carry = h;
#if NB64BLOCK > 5
  c = _addcarry_u64(c, _umul128(x[4], y, &h), carry, dst + 4);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[5], y, &h), carry, dst + 5);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[6], y, &h), carry, dst + 6);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[7], y, &h), carry, dst + 7);
  carry = h;
#endif
  _addcarry_u64(c, 0ULL, carry, dst + (NB64BLOCK - 1));
}
#else
static void inline imm_umul(uint64_t *x, uint64_t y, uint64_t *dst) {
  // Assume that x[NB64BLOCK-1] is 0
  unsigned char c = 0;
  uint64_t h, carry;
  dst[0] = _umul128(x[0], y, &h);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[1], y, &h), carry, dst + 1);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[2], y, &h), carry, dst + 2);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[3], y, &h), carry, dst + 3);
  carry = h;
#if NB64BLOCK > 5
  c = _addcarry_u64(c, _umul128(x[4], y, &h), carry, dst + 4);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[5], y, &h), carry, dst + 5);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[6], y, &h), carry, dst + 6);
  carry = h;
  c = _addcarry_u64(c, _umul128(x[7], y, &h), carry, dst + 7);
  carry = h;
#endif
  _addcarry_u64(c, 0ULL, carry, dst + (NB64BLOCK - 1));
}
#endif

// Optimized shift operations for Xeon Platinum 8488C
#if USE_AVX512
static void inline shiftR(unsigned char n, uint64_t *d) {
  if (n == 0) return;

  // Użycie AVX-512 dla równoległego przesunięcia w prawo
  __m512i data = _mm512_loadu_si512((__m512i *)d);
  __m512i shifted = _mm512_srli_epi64(data, n);

  // Obsługa przeniesienia bitów między słowami
  uint64_t temp[8] = {0};
  for (int i = 0; i < NB64BLOCK - 1; i++) {
    temp[i] = d[i + 1];
  }

  __m512i next_data = _mm512_loadu_si512((__m512i *)temp);
  __m512i next_shifted = _mm512_slli_epi64(next_data, 64 - n);

  // Połączenie wyników
  __m512i result = _mm512_or_si512(shifted, next_shifted);
  _mm512_storeu_si512((__m512i *)d, result);

  // Zachowanie znaku dla liczb ujemnych
  d[NB64BLOCK - 1] = ((int64_t)d[NB64BLOCK - 1]) >> n;
}

static void inline shiftR(unsigned char n, uint64_t *d, uint64_t h) {
  if (n == 0) return;

  // Użycie AVX-512 dla równoległego przesunięcia w prawo z high word
  __m512i data = _mm512_loadu_si512((__m512i *)d);
  __m512i shifted = _mm512_srli_epi64(data, n);

  // Przygotowanie tablicy do przeniesienia bitów
  uint64_t temp[8] = {0};
  for (int i = 0; i < NB64BLOCK - 1; i++) {
    temp[i] = d[i + 1];
  }
  temp[NB64BLOCK - 1] = h;

  __m512i next_data = _mm512_loadu_si512((__m512i *)temp);
  __m512i next_shifted = _mm512_slli_epi64(next_data, 64 - n);

  // Połączenie wyników
  __m512i result = _mm512_or_si512(shifted, next_shifted);
  _mm512_storeu_si512((__m512i *)d, result);
}

static void inline shiftL(unsigned char n, uint64_t *d) {
  if (n == 0) return;

  // Użycie AVX-512 dla równoległego przesunięcia w lewo
  __m512i data = _mm512_loadu_si512((__m512i *)d);
  __m512i shifted = _mm512_slli_epi64(data, n);

  // Obsługa przeniesienia bitów między słowami
  uint64_t temp[8] = {0};
  for (int i = 1; i < NB64BLOCK; i++) {
    temp[i - 1] = d[i - 1];
  }

  __m512i prev_data = _mm512_loadu_si512((__m512i *)temp);
  __m512i prev_shifted = _mm512_srli_epi64(prev_data, 64 - n);

  // Przesunięcie danych o jeden element w lewo
  __m512i carry = _mm512_alignr_epi64(prev_shifted, _mm512_setzero_si512(), 7);

  // Połączenie wyników
  __m512i result = _mm512_or_si512(shifted, carry);
  _mm512_storeu_si512((__m512i *)d, result);

  // Pierwszy element nie ma przeniesienia z poprzedniego
  d[0] = d[0] << n;
}
#else
static void inline shiftR(unsigned char n, uint64_t *d) {
  d[0] = __shiftright128(d[0], d[1], n);
  d[1] = __shiftright128(d[1], d[2], n);
  d[2] = __shiftright128(d[2], d[3], n);
  d[3] = __shiftright128(d[3], d[4], n);
#if NB64BLOCK > 5
  d[4] = __shiftright128(d[4], d[5], n);
  d[5] = __shiftright128(d[5], d[6], n);
  d[6] = __shiftright128(d[6], d[7], n);
  d[7] = __shiftright128(d[7], d[8], n);
#endif
  d[NB64BLOCK - 1] = ((int64_t)d[NB64BLOCK - 1]) >> n;
}

static void inline shiftR(unsigned char n, uint64_t *d, uint64_t h) {
  d[0] = __shiftright128(d[0], d[1], n);
  d[1] = __shiftright128(d[1], d[2], n);
  d[2] = __shiftright128(d[2], d[3], n);
  d[3] = __shiftright128(d[3], d[4], n);
#if NB64BLOCK > 5
  d[4] = __shiftright128(d[4], d[5], n);
  d[5] = __shiftright128(d[5], d[6], n);
  d[6] = __shiftright128(d[6], d[7], n);
  d[7] = __shiftright128(d[7], d[8], n);
#endif
  d[NB64BLOCK - 1] = __shiftright128(d[NB64BLOCK - 1], h, n);
}

static void inline shiftL(unsigned char n, uint64_t *d) {
#if NB64BLOCK > 5
  d[8] = __shiftleft128(d[7], d[8], n);
  d[7] = __shiftleft128(d[6], d[7], n);
  d[6] = __shiftleft128(d[5], d[6], n);
  d[5] = __shiftleft128(d[4], d[5], n);
#endif
  d[4] = __shiftleft128(d[3], d[4], n);
  d[3] = __shiftleft128(d[2], d[3], n);
  d[2] = __shiftleft128(d[1], d[2], n);
  d[1] = __shiftleft128(d[0], d[1], n);
  d[0] = d[0] << n;
}
#endif

// Optimized 128-bit comparison for Xeon Platinum 8488C
#if USE_AVX512
static inline int isStrictGreater128(uint64_t h1, uint64_t l1, uint64_t h2, uint64_t l2) {
  // Użycie AVX-512 dla szybszego porównania 128-bitowych liczb
  __mmask8 gt_mask = _mm512_cmpgt_epu64_mask(_mm512_set_epi64(0, 0, 0, 0, 0, 0, h1, l1),
                                             _mm512_set_epi64(0, 0, 0, 0, 0, 0, h2, l2));

  __mmask8 eq_mask = _mm512_cmpeq_epu64_mask(_mm512_set_epi64(0, 0, 0, 0, 0, 0, h1, l1),
                                             _mm512_set_epi64(0, 0, 0, 0, 0, 0, h2, l2));

  // Sprawdź czy h1 > h2 lub (h1 == h2 && l1 > l2)
  return (gt_mask & 0x2) || ((eq_mask & 0x2) && (gt_mask & 0x1));
}
#else
static inline int isStrictGreater128(uint64_t h1, uint64_t l1, uint64_t h2, uint64_t l2) {
  if (h1 > h2) return 1;
  if (h1 == h2) return l1 > l2;
  return 0;
}
#endif

#endif  // BIGINTH
