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
