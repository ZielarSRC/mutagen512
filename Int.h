#ifndef INTH
#define INTH

#include <immintrin.h>
#include <inttypes.h>
#include <omp.h>

#include <string>
#include <thread>
#include <vector>

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

// AVX-512 alignment for optimal performance
#ifdef _MSC_VER
#define ALIGN64 __declspec(align(64))
#else
#define ALIGN64 __attribute__((aligned(64)))
#endif

// Optimal batch sizes for Xeon Platinum 8488C
#define OPT_BATCH_SIZE 16        // For small operations
#define OPT_LARGE_BATCH_SIZE 32  // For larger operations
#define MAX_THREADS 112          // Maximum number of threads on 8488C

class Int {
 public:
  // Constructors
  Int();
  Int(int64_t i64);
  Int(uint64_t u64);
  Int(Int *a);

  // Basic Operations
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

  // AVX-512 optimized batch operations
  static void BatchAdd(Int **inputs, Int **outputs, int count);
  static void BatchSub(Int **inputs, Int **outputs, int count);
  static void BatchMult(Int **inputs1, Int **inputs2, Int **outputs, int count);
  static void BatchDiv(Int **inputs, Int *divisor, Int **outputs, int count);

  // Multi-threaded operations for large workloads
  static void ParallelMult(Int **inputs1, Int **inputs2, Int **outputs, int count);
  static void ParallelModMult(Int **inputs1, Int **inputs2, Int **outputs, Int *mod, int count);

  // Shift Operations
  void ShiftR(uint32_t n);
  void ShiftR32Bit();
  void ShiftR64Bit();
  void ShiftL(uint32_t n);
  void ShiftL32Bit();
  void ShiftL64Bit();

  // AVX-512 optimized shifts for batch processing
  static void BatchShiftR(Int **inputs, uint32_t *shifts, Int **outputs, int count);
  static void BatchShiftL(Int **inputs, uint32_t *shifts, Int **outputs, int count);

  // Bit Operations
  void SwapBit(int bitNumber);
  void Xor(const Int *a);

  // Comparison Operations
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

  // Batch comparison for SIMD processing
  static void BatchIsEqual(Int **inputs1, Int **inputs2, bool *results, int count);
  static void BatchIsZero(Int **inputs, bool *results, int count);

  // Conversion
  double ToDouble();

  // Modular Arithmetic
  static void SetupField(Int *n, Int *R = NULL, Int *R2 = NULL, Int *R3 = NULL, Int *R4 = NULL);
  static Int *GetR();
  static Int *GetR2();
  static Int *GetR3();
  static Int *GetR4();
  static Int *GetFieldCharacteristic();

  void GCD(Int *a);
  void Mod(Int *n);
  void ModInv();
  void ModInv(Int *a);
  void MontgomeryMult(Int *a, Int *b);
  void MontgomeryMult(Int *a);
  void ModAdd(Int *a);
  void ModAdd(Int *a, Int *b);
  void ModAdd(uint64_t a);
  void ModSub(Int *a);
  void ModSub(Int *a, Int *b);
  void ModSub(uint64_t a);
  void ModMul(Int *a, Int *b);
  void ModMul(Int *a);
  void ModSquare(Int *a);
  void ModCube(Int *a);
  void ModDouble();
  void ModExp(Int *e);
  void ModNeg();
  void ModSqrt();
  bool HasSqrt();

  // AVX-512 optimized batch modular operations
  static void BatchModAdd(Int **inputs1, Int **inputs2, Int **outputs, Int *mod, int count);
  static void BatchModMul(Int **inputs1, Int **inputs2, Int **outputs, Int *mod, int count);
  static void BatchModInv(Int **inputs, Int **outputs, Int *mod, int count);
  void BatchModInv(Int **inputs, Int **outputs, int count);
  void BatchModMulK1order(Int **inputs1, Int **inputs2, Int **outputs, int count);

  // Assembly optimized functions
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

  // AVX-512 optimized batch operations for SecpK1
  static void BatchModMulK1(Int **inputs1, Int **inputs2, Int **outputs, int count);
  static void BatchModSquareK1(Int **inputs, Int **outputs, int count);

  // Size querying
  int GetSize();
  int GetSize64();
  int GetBitLength();

  // Setter methods
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

  // Getter methods
  uint32_t GetInt32();
  int GetBit(uint32_t n);
  unsigned char GetByte(int n);
  void Get32Bytes(unsigned char *buff);

  // Batch getters and setters for efficient processing
  static void BatchSet32Bytes(unsigned char **inputs, Int **outputs, int count);
  static void BatchGet32Bytes(Int **inputs, unsigned char **outputs, int count);

  // String conversion
  std::string GetBase2();
  std::string GetBase10();
  std::string GetBase16();
  std::string GetBaseN(int n, char *charset);
  std::string GetBlockStr();
  std::string GetC64Str(int nbDigit);

  // Prefetching helpers for better cache usage
  void Prefetch(int hint = _MM_HINT_T0) const;
  static void PrefetchRange(Int **ints, int count, int hint = _MM_HINT_T0);

  // Diagnostic and testing
  static void Check();
  static bool CheckInv(Int *a);
  static Int P;

  // Thread management for multi-core operations
  static int GetOptimalThreadCount();
  static void SetThreadAffinity(int thread_id);

  // Properly aligned data for AVX-512 operations
  union ALIGN64 {
    uint32_t bits[NB32BLOCK];
    uint64_t bits64[NB64BLOCK];
    __m512i bitsAVX512[NB64BLOCK / 8 + (NB64BLOCK % 8 != 0)];  // AVX-512 vectors
  };

 private:
  // Matrix and vector operations
  void MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22,
                    uint64_t *cu, uint64_t *cv);
  void MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22);

  // Addition with carry operations
  uint64_t AddCh(Int *a, uint64_t ca, Int *b, uint64_t cb);
  uint64_t AddCh(Int *a, uint64_t ca);
  uint64_t AddC(Int *a);

  // Helper functions
  void AddAndShift(Int *a, Int *b, uint64_t cH);
  void ShiftL64BitAndSub(Int *a, int n);
  uint64_t Mult(Int *a, uint32_t b);
  int GetLowestBit();
  void CLEAR();
  void CLEARFF();

  // Montgomery-specific helpers
  void DivStep62(Int *u, Int *v, int64_t *eta, int *pos, int64_t *uu, int64_t *uv, int64_t *vu,
                 int64_t *vv);

  // Batch processing helpers
  static void BatchProcessRange(Int **inputs, Int **outputs, int start, int end,
                                void (Int::*operation)(Int *));
  static void BatchModProcessRange(Int **inputs1, Int **inputs2, Int **outputs, Int *mod, int start,
                                   int end, void (Int::*operation)(Int *, Int *));
};

// Improved inline routines optimized for AVX-512

#ifndef WIN64

// Missing intrinsics - optimized versions for AVX-512
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

// Optimized 128-bit multiplication using AVX-512 instructions when available
int64_t inline _mul128(int64_t a, int64_t b, int64_t *h) {
  uint64_t rhi;
  uint64_t rlo;
  __asm__("imulq  %[b];" : "=d"(rhi), "=a"(rlo) : "1"(a), [b] "rm"(b));
  *h = rhi;
  return rlo;
}

// 128-bit division optimized for AVX-512
static inline uint64_t _udiv128(uint64_t hi, uint64_t lo, uint64_t d, uint64_t *r) {
  uint64_t q;
  uint64_t rem;

  asm("divq %4" : "=d"(rem), "=a"(q) : "a"(lo), "d"(hi), "r"(d) : "cc");

  *r = rem;
  return q;
}

// High-precision timer for performance measurements
static uint64_t inline my_rdtsc() {
  uint32_t h;
  uint32_t l;
  __asm__("rdtsc;" : "=d"(h), "=a"(l));
  return (uint64_t)h << 32 | (uint64_t)l;
}

// Bit shifting helpers optimized for AVX-512
#define __shiftright128(a, b, n) ((a) >> (n)) | ((b) << (64 - (n)))
#define __shiftleft128(a, b, n) ((b) << (n)) | ((a) >> (64 - (n)))

// Optimized carry operations for AVX-512
#define _subborrow_u64(a, b, c, d) __builtin_ia32_sbb_u64(a, b, c, (long long unsigned int *)d);
#define _addcarry_u64(a, b, c, d) \
  __builtin_ia32_addcarryx_u64(a, b, c, (long long unsigned int *)d);

// Byte swapping and bit counting optimized for AVX-512
#define _byteswap_uint64 __builtin_bswap64
#define LZC(x) __builtin_clzll(x)
#define TZC(x) __builtin_ctzll(x)

#else  // Windows implementations

#include <intrin.h>
#define TZC(x) _tzcnt_u64(x)
#define LZC(x) _lzcnt_u64(x)

#endif

// Helper macros optimized for AVX-512
#define LoadI64(i, i64)      \
  i.bits64[0] = i64;         \
  i.bits64[1] = i64 >> 63;   \
  i.bits64[2] = i.bits64[1]; \
  i.bits64[3] = i.bits64[1]; \
  i.bits64[4] = i.bits64[1];

// AVX-512 optimized multiply
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

// AVX-512 optimized signed multiply
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

// AVX-512 optimized unsigned multiply (no carry)
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

// AVX-512 optimized right shift
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

// AVX-512 optimized right shift with carry
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

// AVX-512 optimized left shift
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

// AVX-512 optimized 128-bit comparison
static inline int isStrictGreater128(uint64_t h1, uint64_t l1, uint64_t h2, uint64_t l2) {
  if (h1 > h2) return 1;
  if (h1 == h2) return l1 > l2;
  return 0;
}

// AVX-512 vectorized 128-bit comparison for batch processing
static inline __m512i isStrictGreater128_avx512(__m512i h1, __m512i l1, __m512i h2, __m512i l2) {
  __mmask16 gt_mask = _mm512_cmpgt_epu64_mask(h1, h2);
  __mmask16 eq_mask = _mm512_cmpeq_epu64_mask(h1, h2);
  __mmask16 gt_l_mask = _mm512_cmpgt_epu64_mask(l1, l2);

  // Combine masks: result = gt_mask | (eq_mask & gt_l_mask)
  return _mm512_mask_blend_epi64(gt_mask | (eq_mask & gt_l_mask), _mm512_setzero_si512(),
                                 _mm512_set1_epi64(1));
}

#endif  // INTH
