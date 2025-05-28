#ifndef BIGINTH
#define BIGINTH

#include <immintrin.h>
#include <inttypes.h>
#include <x86intrin.h>

#include <string>

// Ultimate compiler optimizations for Xeon Platinum 8488C
#pragma GCC target( \
    "avx512f,avx512dq,avx512bw,avx512vl,avx512vnni,avx512ifma,avx512vbmi,bmi2,lzcnt,popcnt,adx")

// We need 1 extra block for Knuth div algorithm, Montgomery multiplication and
// ModInv
#define BISIZE 256

#if BISIZE == 256
#define NB64BLOCK 5
#define NB32BLOCK 10
#elif BISIZE == 512
#define NB64BLOCK 9
#define NB32BLOCK 18
#else
#error Unsupported size
#endif

// Cache alignment and performance macros for Xeon 8488C
#define CACHE_ALIGN alignas(64)
#define FORCE_INLINE __attribute__((always_inline)) inline
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

class Int {
 public:
  Int();
  Int(int64_t i64);
  Int(uint64_t u64);
  Int(Int *a);

  // Basic Operations - optimized for Xeon 8488C
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

  // Bit Operations - AVX-512 optimized
  void ShiftR(uint32_t n);
  void ShiftR32Bit();
  void ShiftR64Bit();
  void ShiftL(uint32_t n);
  void ShiftL32Bit();
  void ShiftL64Bit();
  void SwapBit(int bitNumber);
  void Xor(const Int *a);

  // Comparison Operations - branch prediction optimized
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

  // Modular Arithmetic - Montgomery optimized for Xeon 8488C
  static void SetupField(Int *n, Int *R = NULL, Int *R2 = NULL, Int *R3 = NULL,
                         Int *R4 = NULL);
  static Int *GetR();
  static Int *GetR2();
  static Int *GetR3();
  static Int *GetR4();
  static Int *GetFieldCharacteristic();

  void GCD(Int *a);
  void Mod(Int *n);
  void ModInv();
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

  // SECP256K1 Specific - ultimate optimization
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

  // AVX-512 Batch Operations for Xeon 8488C
  static void BatchModAdd(Int *inputs, Int *operands, Int *results, int count);
  static void BatchModMul(Int *inputs, Int *operands, Int *results, int count);
  static void BatchModInv(Int *inputs, Int *results, int count);
  static void BatchModMulK1(Int *inputs, Int *operands, Int *results,
                            int count);
  static void BatchModMulK1order(Int *inputs, Int *operands, Int *results,
                                 int count);

  // Memory-optimized operations
  void AlignedCopyK1order(const Int *src);
  void PrefetchOptimizedCopy(const Int *src);

  // Size and Properties
  int GetSize();
  int GetSize64();
  int GetBitLength();

  // Setters - optimized for cache performance
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

  // Getters
  uint32_t GetInt32();
  int GetBit(uint32_t n);
  unsigned char GetByte(int n);
  void Get32Bytes(unsigned char *buff);

  // String Conversion
  std::string GetBase2();
  std::string GetBase10();
  std::string GetBase16();
  std::string GetBaseN(int n, char *charset);
  std::string GetBlockStr();
  std::string GetC64Str(int nbDigit);

  // Verification and Testing
  static void Check();
  static bool CheckInv(Int *a);
  static Int P;

  // Cache-aligned memory layout for optimal Xeon 8488C performance
  union {
    CACHE_ALIGN uint32_t bits[NB32BLOCK];
    CACHE_ALIGN uint64_t bits64[NB64BLOCK];
  };

 private:
  void MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21,
                    int64_t _22, uint64_t *cu, uint64_t *cv);
  void MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21,
                    int64_t _22);
  uint64_t AddCh(Int *a, uint64_t ca, Int *b, uint64_t cb);
  uint64_t AddCh(Int *a, uint64_t ca);
  uint64_t AddC(Int *a);
  void AddAndShift(Int *a, Int *b, uint64_t cH);
  void ShiftL64BitAndSub(Int *a, int n);
  uint64_t Mult(Int *a, uint32_t b);
  int GetLowestBit();
  void CLEAR();
  void CLEARFF();
  void DivStep62(Int *u, Int *v, int64_t *eta, int *pos, int64_t *uu,
                 int64_t *uv, int64_t *vu, int64_t *vv);

  // Xeon 8488C specific optimizations
  void MontgomerySquare(Int *a);  // Dedicated squaring function
  FORCE_INLINE void PrefetchMemory() const;
  FORCE_INLINE void OptimizedMemCopy(const Int *src);
};

// Optimized inline routines for Xeon 8488C

// Enhanced multiplication using BMI2/ADX for maximum performance
FORCE_INLINE uint64_t _umul128_optimized(uint64_t a, uint64_t b, uint64_t *h) {
#if defined(__BMI2__)
  // Use MULX instruction for better performance on Xeon 8488C
  uint64_t rlo, rhi;
  __asm__("mulx %[B], %[LO], %[HI]"
          : [LO] "=r"(rlo), [HI] "=r"(rhi)
          : "d"(a), [B] "r"(b)
          : "cc");
  *h = rhi;
  return rlo;
#else
  uint64_t rhi, rlo;
  __asm__("mulq %[B]" : "=d"(rhi), "=a"(rlo) : "a"(a), [B] "r"(b) : "cc");
  *h = rhi;
  return rlo;
#endif
}

FORCE_INLINE int64_t _mul128_optimized(int64_t a, int64_t b, int64_t *h) {
  uint64_t rhi, rlo;
  __asm__("imulq %[b];" : "=d"(rhi), "=a"(rlo) : "1"(a), [b] "rm"(b));
  *h = rhi;
  return rlo;
}

FORCE_INLINE uint64_t _udiv128_optimized(uint64_t hi, uint64_t lo, uint64_t d,
                                         uint64_t *r) {
  uint64_t q, rem;
  __asm__("divq %4" : "=d"(rem), "=a"(q) : "a"(lo), "d"(hi), "r"(d) : "cc");
  *r = rem;
  return q;
}

FORCE_INLINE uint64_t optimized_rdtsc() {
  uint32_t h, l;
  __asm__("rdtsc;" : "=d"(h), "=a"(l));
  return ((uint64_t)h << 32) | (uint64_t)l;
}

// Enhanced bit manipulation using LZCNT/TZCNT
#define __shiftright128(a, b, n) ((a) >> (n)) | ((b) << (64 - (n)))
#define __shiftleft128(a, b, n) ((b) << (n)) | ((a) >> (64 - (n)))

// Optimized carry operations using ADX when available
#if defined(__ADX__)
#define _subborrow_u64_optimized(a, b, c, d) \
  __builtin_ia32_sbb_u64(a, b, c, (long long unsigned int *)d)
#define _addcarry_u64_optimized(a, b, c, d) \
  __builtin_ia32_addcarryx_u64(a, b, c, (long long unsigned int *)d)
#else
#define _subborrow_u64_optimized(a, b, c, d) \
  __builtin_ia32_sbb_u64(a, b, c, (long long unsigned int *)d)
#define _addcarry_u64_optimized(a, b, c, d) \
  __builtin_ia32_addcarryx_u64(a, b, c, (long long unsigned int *)d)
#endif

#define _byteswap_uint64 __builtin_bswap64

// Use LZCNT/TZCNT instructions for optimal bit counting on Xeon 8488C
#define LZC(x) __lzcnt64(x)
#define TZC(x) __tzcnt_u64(x)

// Redefine _umul128 to use optimized version
#undef _umul128
#define _umul128(a, b, h) _umul128_optimized((a), (b), (h))

#define LoadI64(i, i64)      \
  i.bits64[0] = i64;         \
  i.bits64[1] = i64 >> 63;   \
  i.bits64[2] = i.bits64[1]; \
  i.bits64[3] = i.bits64[1]; \
  i.bits64[4] = i.bits64[1];

// Ultimate optimized multiplication using ADX/BMI2 for Xeon 8488C
FORCE_INLINE void imm_mul_avx512(uint64_t *x, uint64_t y, uint64_t *dst,
                                 uint64_t *carryH) {
  unsigned char c = 0;
  uint64_t h, carry;

  // Prefetch data for optimal cache utilization
  __builtin_prefetch(x, 0, 3);
  __builtin_prefetch(dst, 1, 3);

  dst[0] = _umul128_optimized(x[0], y, &h);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[1], y, &h), carry,
                              dst + 1);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[2], y, &h), carry,
                              dst + 2);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[3], y, &h), carry,
                              dst + 3);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[4], y, &h), carry,
                              dst + 4);
  carry = h;

#if NB64BLOCK > 5
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[5], y, &h), carry,
                              dst + 5);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[6], y, &h), carry,
                              dst + 6);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[7], y, &h), carry,
                              dst + 7);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[8], y, &h), carry,
                              dst + 8);
  carry = h;
#endif
  *carryH = carry;
}

FORCE_INLINE void imm_imul_avx512(uint64_t *x, uint64_t y, uint64_t *dst,
                                  uint64_t *carryH) {
  unsigned char c = 0;
  uint64_t h, carry;

  __builtin_prefetch(x, 0, 3);
  __builtin_prefetch(dst, 1, 3);

  dst[0] = _umul128_optimized(x[0], y, &h);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[1], y, &h), carry,
                              dst + 1);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[2], y, &h), carry,
                              dst + 2);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[3], y, &h), carry,
                              dst + 3);
  carry = h;

#if NB64BLOCK > 5
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[4], y, &h), carry,
                              dst + 4);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[5], y, &h), carry,
                              dst + 5);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[6], y, &h), carry,
                              dst + 6);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[7], y, &h), carry,
                              dst + 7);
  carry = h;
#endif

  c = _addcarry_u64_optimized(
      c, _mul128_optimized(x[NB64BLOCK - 1], y, (int64_t *)&h), carry,
      dst + NB64BLOCK - 1);
  carry = h;
  *carryH = carry;
}

FORCE_INLINE void imm_umul_avx512(uint64_t *x, uint64_t y, uint64_t *dst) {
  // Optimized unsigned multiplication assuming x[NB64BLOCK-1] is 0
  unsigned char c = 0;
  uint64_t h, carry;

  // Strategic prefetching for Xeon 8488C cache hierarchy
  __builtin_prefetch(x, 0, 3);
  __builtin_prefetch(dst, 1, 3);

  dst[0] = _umul128_optimized(x[0], y, &h);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[1], y, &h), carry,
                              dst + 1);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[2], y, &h), carry,
                              dst + 2);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[3], y, &h), carry,
                              dst + 3);
  carry = h;

#if NB64BLOCK > 5
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[4], y, &h), carry,
                              dst + 4);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[5], y, &h), carry,
                              dst + 5);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[6], y, &h), carry,
                              dst + 6);
  carry = h;
  c = _addcarry_u64_optimized(c, _umul128_optimized(x[7], y, &h), carry,
                              dst + 7);
  carry = h;
#endif

  _addcarry_u64_optimized(c, 0ULL, carry, dst + (NB64BLOCK - 1));
}

// Use the optimized versions by default
#define imm_mul imm_mul_avx512
#define imm_imul imm_imul_avx512
#define imm_umul imm_umul_avx512

// Enhanced bit shifting with AVX-512 optimization hints
FORCE_INLINE void shiftR_avx512(unsigned char n, uint64_t *d) {
  // Prefetch data for optimal cache performance
  __builtin_prefetch(d, 1, 3);

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

FORCE_INLINE void shiftR_avx512(unsigned char n, uint64_t *d, uint64_t h) {
  __builtin_prefetch(d, 1, 3);

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

FORCE_INLINE void shiftL_avx512(unsigned char n, uint64_t *d) {
  __builtin_prefetch(d, 1, 3);

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

// Use optimized versions by default
#define shiftR shiftR_avx512
#define shiftL shiftL_avx512

// Enhanced 128-bit comparison with branch prediction hints
FORCE_INLINE int isStrictGreater128_optimized(uint64_t h1, uint64_t l1,
                                              uint64_t h2, uint64_t l2) {
  if (LIKELY(h1 != h2)) return h1 > h2;
  return l1 > l2;
}

#define isStrictGreater128 isStrictGreater128_optimized

// AVX-512 memory operations namespace for specialized functions
namespace IntAVX512 {
// Vectorized operations using full AVX-512 capability
void BatchClear(Int *ints, int count);
void BatchCopy(Int *dest, const Int *src, int count);
void BatchAdd(Int *a, Int *b, Int *result, int count);
void BatchSub(Int *a, Int *b, Int *result, int count);
void VectorizedMemCopy(Int *dest, const Int *src, int count);

// Parallel comparison operations
void BatchCompare(Int *a, Int *b, bool *results, int count);
void BatchIsZero(Int *ints, bool *results, int count);
void BatchIsEqual(Int *a, Int *b, bool *results, int count);
}  // namespace IntAVX512

// Performance monitoring utilities for Xeon 8488C
class IntPerformanceMonitor {
 private:
  uint64_t start_cycles;
  uint64_t operation_count;

 public:
  void StartTiming() {
    __builtin_ia32_lfence();
    start_cycles = optimized_rdtsc();
  }

  void EndTiming() {
    __builtin_ia32_lfence();
    uint64_t end_cycles = optimized_rdtsc();
    operation_count = end_cycles - start_cycles;
  }

  double GetOperationsPerSecond(int ops, double cpu_freq_ghz = 3.9) const {
    double seconds = (double)operation_count / (cpu_freq_ghz * 1e9);
    return (double)ops / seconds;
  }

  uint64_t GetCycles() const { return operation_count; }
};

// Memory pool for frequent Int allocations optimized for Xeon 8488C
class IntMemoryPool {
 private:
  static constexpr int POOL_SIZE = 1024;
  CACHE_ALIGN Int memory_pool[POOL_SIZE];
  bool used[POOL_SIZE];
  int next_free;

 public:
  IntMemoryPool() : next_free(0) { std::fill(used, used + POOL_SIZE, false); }

  Int *Allocate() {
    for (int i = next_free; i < POOL_SIZE; i++) {
      if (!used[i]) {
        used[i] = true;
        next_free = i + 1;
        return &memory_pool[i];
      }
    }

    // Search from beginning
    for (int i = 0; i < next_free; i++) {
      if (!used[i]) {
        used[i] = true;
        next_free = i + 1;
        return &memory_pool[i];
      }
    }

    return nullptr;  // Pool exhausted
  }

  void Deallocate(Int *ptr) {
    if (ptr >= memory_pool && ptr < memory_pool + POOL_SIZE) {
      int index = ptr - memory_pool;
      used[index] = false;
      if (index < next_free) next_free = index;
    }
  }
};

#endif  // BIGINTH
