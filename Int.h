// Big integer class (Fixed size) zoptymalizowana dla Intel Xeon Platinum 8488C

#ifndef BIGINTH
#define BIGINTH

#include <immintrin.h>  // Dla pełnego wsparcia AVX-512
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

// Tablica małych liczb pierwszych dla testów pierwszości
const int primeCount = 11;
const uint32_t primes[primeCount] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31};

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

  // Dane
  union {
    uint32_t bits[NB32BLOCK];
    uint64_t bits64[NB64BLOCK];
  };

 private:
  // Funkcje pomocnicze zoptymalizowane dla AVX-512
  void imm_umul(uint64_t *x, uint64_t y, uint64_t *dst);
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

#define __shiftright128(a, b, n) ((a) >> (n)) | ((b) << (64 - (n)))
#define __shiftleft128(a, b, n) ((b) << (n)) | ((a) >> (64 - (n)))

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

// Optymalizacje dla AVX-512
#if defined(__AVX512F__)
// Optymalizacja shiftR dla AVX-512
static void inline shiftR(unsigned char n, uint64_t *d) {
  if (n == 0) return;

  __m512i data = _mm512_loadu_si512((__m512i *)d);
  __m512i shifted = _mm512_srli_epi64(data, n);

  // Obsługa przeniesienia bitów między słowami
  __m512i data_shifted_left = _mm512_slli_epi64(data, 64 - n);
  __m512i carry = _mm512_alignr_epi64(_mm512_setzero_si512(), data_shifted_left, NB64BLOCK - 1);

  // Połączenie wyników
  __m512i result = _mm512_or_si512(shifted, carry);
  _mm512_storeu_si512((__m512i *)d, result);

  // Zachowanie znaku dla liczb ujemnych
  if ((int64_t)d[NB64BLOCK - 1] < 0) {
    d[NB64BLOCK - 1] |= (0xFFFFFFFFFFFFFFFFULL << (64 - n));
  }
}

// Optymalizacja shiftR z wykorzystaniem high word dla AVX-512
static void inline shiftR(unsigned char n, uint64_t *d, uint64_t h) {
  if (n == 0) return;

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

// Optymalizacja shiftL dla AVX-512
static void inline shiftL(unsigned char n, uint64_t *d) {
  if (n == 0) return;

  __m512i data = _mm512_loadu_si512((__m512i *)d);
  __m512i shifted = _mm512_slli_epi64(data, n);

  // Obsługa przeniesienia bitów między słowami
  __m512i data_shifted_right = _mm512_srli_epi64(data, 64 - n);
  __m512i carry = _mm512_alignr_epi64(data_shifted_right, _mm512_setzero_si512(), 1);

  // Połączenie wyników
  __m512i result = _mm512_or_si512(shifted, carry);
  _mm512_storeu_si512((__m512i *)d, result);
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

static inline int isStrictGreater128(uint64_t h1, uint64_t l1, uint64_t h2, uint64_t l2) {
  if (h1 > h2) return 1;
  if (h1 == h2) return l1 > l2;
  return 0;
}

#endif  // BIGINTH
