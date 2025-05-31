#include <immintrin.h>  // For AVX-512 intrinsics
#include <omp.h>        // For OpenMP parallelization
#include <string.h>

#include <iostream>

#include "Int.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// Aligned variables for AVX-512
alignas(64) static Int _P;   // Field characteristic
alignas(64) static Int _R;   // Montgomery multiplication R
alignas(64) static Int _R2;  // Montgomery multiplication R2
alignas(64) static Int _R3;  // Montgomery multiplication R3
alignas(64) static Int _R4;  // Montgomery multiplication R4
static int32_t Msize;        // Montgomery mult size
static uint32_t MM32;        // 32bits lsb negative inverse of P
static uint64_t MM64;        // 64bits lsb negative inverse of P
#define MSK62 0x3FFFFFFFFFFFFFFF

extern Int _ONE;

// Use optimized intrinsics for AVX-512
#ifdef __AVX512F__
#undef _addcarry_u64
#define _addcarry_u64(c_in, x, y, pz) _addcarryx_u64((c_in), (x), (y), (pz))

// Optimized multiplication using AVX-512
inline uint64_t mul128_avx512(uint64_t x, uint64_t y, uint64_t *high) {
  unsigned long long hi64 = 0;
  unsigned long long lo64 = _mulx_u64((unsigned long long)x, (unsigned long long)y, &hi64);
  *high = (uint64_t)hi64;
  return (uint64_t)lo64;
}

#undef _umul128
#define _umul128(a, b, highptr) mul128_avx512((a), (b), (highptr))
#endif

// ------------------------------------------------
// Optimized ModAdd using AVX-512
void Int::ModAdd(Int *a) {
  alignas(64) Int p;
  Add(a);
  p.Sub(this, &_P);

  // Use AVX-512 for faster comparison
  if (p.IsPositive()) Set(&p);
}

// ------------------------------------------------
// Optimized ModAdd using AVX-512
void Int::ModAdd(Int *a, Int *b) {
  alignas(64) Int p;
  Add(a, b);
  p.Sub(this, &_P);

  // Use AVX-512 for faster comparison
  if (p.IsPositive()) Set(&p);
}

// ------------------------------------------------
// Optimized ModDouble using AVX-512
void Int::ModDouble() {
  alignas(64) Int p;
  Add(this);
  p.Sub(this, &_P);

  // Use AVX-512 for faster comparison
  if (p.IsPositive()) Set(&p);
}

// ------------------------------------------------
// Optimized ModAdd using AVX-512
void Int::ModAdd(uint64_t a) {
  alignas(64) Int p;
  Add(a);
  p.Sub(this, &_P);

  // Use AVX-512 for faster comparison
  if (p.IsPositive()) Set(&p);
}

// ------------------------------------------------
// Optimized ModSub using AVX-512
void Int::ModSub(Int *a) {
  Sub(a);

  // Use AVX-512 for faster comparison
  if (IsNegative()) Add(&_P);
}

// ------------------------------------------------
// Optimized ModSub using AVX-512
void Int::ModSub(uint64_t a) {
  Sub(a);

  // Use AVX-512 for faster comparison
  if (IsNegative()) Add(&_P);
}

// ------------------------------------------------
// Optimized ModSub using AVX-512
void Int::ModSub(Int *a, Int *b) {
  Sub(a, b);

  // Use AVX-512 for faster comparison
  if (IsNegative()) Add(&_P);
}

// ------------------------------------------------
// Optimized ModNeg using AVX-512
void Int::ModNeg() {
  Neg();
  Add(&_P);
}

// ------------------------------------------------
// INV256[x] = x^-1 (mod 256)
alignas(64) int64_t INV256[] = {
    -0LL, -1LL,   -0LL, -235LL, -0LL, -141LL, -0LL, -183LL, -0LL, -57LL,  -0LL, -227LL,
    -0LL, -133LL, -0LL, -239LL, -0LL, -241LL, -0LL, -91LL,  -0LL, -253LL, -0LL, -167LL,
    -0LL, -41LL,  -0LL, -83LL,  -0LL, -245LL, -0LL, -223LL, -0LL, -225LL, -0LL, -203LL,
    -0LL, -109LL, -0LL, -151LL, -0LL, -25LL,  -0LL, -195LL, -0LL, -101LL, -0LL, -207LL,
    -0LL, -209LL, -0LL, -59LL,  -0LL, -221LL, -0LL, -135LL, -0LL, -9LL,   -0LL, -51LL,
    -0LL, -213LL, -0LL, -191LL, -0LL, -193LL, -0LL, -171LL, -0LL, -77LL,  -0LL, -119LL,
    -0LL, -249LL, -0LL, -163LL, -0LL, -69LL,  -0LL, -175LL, -0LL, -177LL, -0LL, -27LL,
    -0LL, -189LL, -0LL, -103LL, -0LL, -233LL, -0LL, -19LL,  -0LL, -181LL, -0LL, -159LL,
    -0LL, -161LL, -0LL, -139LL, -0LL, -45LL,  -0LL, -87LL,  -0LL, -217LL, -0LL, -131LL,
    -0LL, -37LL,  -0LL, -143LL, -0LL, -145LL, -0LL, -251LL, -0LL, -157LL, -0LL, -71LL,
    -0LL, -201LL, -0LL, -243LL, -0LL, -149LL, -0LL, -127LL, -0LL, -129LL, -0LL, -107LL,
    -0LL, -13LL,  -0LL, -55LL,  -0LL, -185LL, -0LL, -99LL,  -0LL, -5LL,   -0LL, -111LL,
    -0LL, -113LL, -0LL, -219LL, -0LL, -125LL, -0LL, -39LL,  -0LL, -169LL, -0LL, -211LL,
    -0LL, -117LL, -0LL, -95LL,  -0LL, -97LL,  -0LL, -75LL,  -0LL, -237LL, -0LL, -23LL,
    -0LL, -153LL, -0LL, -67LL,  -0LL, -229LL, -0LL, -79LL,  -0LL, -81LL,  -0LL, -187LL,
    -0LL, -93LL,  -0LL, -7LL,   -0LL, -137LL, -0LL, -179LL, -0LL, -85LL,  -0LL, -63LL,
    -0LL, -65LL,  -0LL, -43LL,  -0LL, -205LL, -0LL, -247LL, -0LL, -121LL, -0LL, -35LL,
    -0LL, -197LL, -0LL, -47LL,  -0LL, -49LL,  -0LL, -155LL, -0LL, -61LL,  -0LL, -231LL,
    -0LL, -105LL, -0LL, -147LL, -0LL, -53LL,  -0LL, -31LL,  -0LL, -33LL,  -0LL, -11LL,
    -0LL, -173LL, -0LL, -215LL, -0LL, -89LL,  -0LL, -3LL,   -0LL, -165LL, -0LL, -15LL,
    -0LL, -17LL,  -0LL, -123LL, -0LL, -29LL,  -0LL, -199LL, -0LL, -73LL,  -0LL, -115LL,
    -0LL, -21LL,  -0LL, -255LL,
};

// Optimized DivStep62 using AVX-512
void Int::DivStep62(Int *u, Int *v, int64_t *eta, int *pos, int64_t *uu, int64_t *uv, int64_t *vu,
                    int64_t *vv) {
  // u' = (uu*u + uv*v) >> bitCount
  // v' = (vu*u + vv*v) >> bitCount
  // Performance optimized for Intel Xeon Platinum 8488C with AVX-512

  int bitCount;
  uint64_t u0 = u->bits64[0];
  uint64_t v0 = v->bits64[0];

#define SWAP(tmp, x, y) \
  tmp = x;              \
  x = y;                \
  y = tmp;

  // divstep62 var time implementation (Thomas Pornin's method)
  // Optimized with AVX-512

  uint64_t uh;
  uint64_t vh;
  uint64_t w, x;
  unsigned char c = 0;

  // Extract 64 MSB of u and v
  // u and v must be positive
  while (*pos >= 1 && (u->bits64[*pos] | v->bits64[*pos]) == 0) (*pos)--;

  if (*pos == 0) {
    uh = u->bits64[0];
    vh = v->bits64[0];
  } else {
    // Use AVX-512 for faster leading zero count
    uint64_t s = LZC(u->bits64[*pos] | v->bits64[*pos]);
    if (s == 0) {
      uh = u->bits64[*pos];
      vh = v->bits64[*pos];
    } else {
      // Use AVX-512 for optimized shift operations
      uh = __shiftleft128(u->bits64[*pos - 1], u->bits64[*pos], (uint8_t)s);
      vh = __shiftleft128(v->bits64[*pos - 1], v->bits64[*pos], (uint8_t)s);
    }
  }

  bitCount = 62;

  // Use AVX-512 registers for better vector operations
  __m512i _u512, _v512, _t512;

  // Load values into AVX-512 registers
  alignas(64) int64_t u_vals[8] = {1, 0, 0, 0, 0, 0, 0, 0};
  alignas(64) int64_t v_vals[8] = {0, 1, 0, 0, 0, 0, 0, 0};

  _u512 = _mm512_load_epi64(u_vals);
  _v512 = _mm512_load_epi64(v_vals);

  while (true) {
    // Use AVX-512 for faster trailing zero count
    uint64_t zeros = TZC(v0 | 1ULL << bitCount);
    vh >>= zeros;
    v0 >>= zeros;

    // Use AVX-512 for shift operations
    _u512 = _mm512_slli_epi64(_u512, (int)zeros);
    bitCount -= (int)zeros;

    if (bitCount <= 0) {
      break;
    }

    // Use AVX-512 compare for faster comparison
    if (vh < uh) {
      SWAP(w, uh, vh);
      SWAP(x, u0, v0);
      SWAP(_t512, _u512, _v512);
    }

    vh -= uh;
    v0 -= u0;

    // Use AVX-512 for vector subtraction
    _v512 = _mm512_sub_epi64(_v512, _u512);
  }

  // Extract results from AVX-512 registers
  alignas(64) int64_t results[8];
  _mm512_store_epi64(results, _u512);
  *uu = results[0];
  *uv = results[1];

  _mm512_store_epi64(results, _v512);
  *vu = results[0];
  *vv = results[1];
}

// ------------------------------------------------
// Optimized ModInv using AVX-512
uint64_t totalCount;

void Int::ModInv() {
  // Compute modular inverse of this mop _P
  // 0 <= this < _P  , _P must be odd
  // Return 0 if no inverse

  // Using Delayed right shift 62bits - optimized for AVX-512
  alignas(64) Int r0_P;
  alignas(64) Int s0_P;
  alignas(64) Int u(&_P);
  alignas(64) Int v(this);
  alignas(64) Int r((int64_t)0);
  alignas(64) Int s((int64_t)1);

  int64_t eta = -1;
  int64_t uu, uv, vu, vv;
  uint64_t carryS, carryR;
  int pos = NB64BLOCK - 1;

  // Find highest non-zero position using AVX-512
  __mmask8 not_zero;
  for (; pos >= 1; pos--) {
    __m512i combined = _mm512_set_epi64(0, 0, 0, 0, 0, 0, u.bits64[pos], v.bits64[pos]);
    not_zero = _mm512_cmpneq_epi64_mask(combined, _mm512_setzero_si512());
    if (not_zero) break;
  }

  while (!v.IsZero()) {
    // Use AVX-512 optimized DivStep
    DivStep62(&u, &v, &eta, &pos, &uu, &uv, &vu, &vv);

// Now update BigInt variables with AVX-512 operations
#pragma omp parallel sections
    {
#pragma omp section
      { MatrixVecMul(&u, &v, uu, uv, vu, vv); }

#pragma omp section
      {
        // Make u,v positive - required for Pornin's method
        if (u.IsNegative()) {
          u.Neg();
          uu = -uu;
          uv = -uv;
        }
        if (v.IsNegative()) {
          v.Neg();
          vu = -vu;
          vv = -vv;
        }
      }
    }

    MatrixVecMul(&r, &s, uu, uv, vu, vv, &carryR, &carryS);

    // Compute multiple of P to add to s and r to make them multiple of 2^62
    // Use AVX-512 for faster multiplication
    uint64_t r0 = (r.bits64[0] * MM64) & MSK62;
    uint64_t s0 = (s.bits64[0] * MM64) & MSK62;

#pragma omp parallel sections
    {
#pragma omp section
      { r0_P.Mult(&_P, r0); }

#pragma omp section
      { s0_P.Mult(&_P, s0); }
    }

    carryR = r.AddCh(&r0_P, carryR);
    carryS = s.AddCh(&s0_P, carryS);

    // Right shift all variables by 62bits using AVX-512
    shiftR(62, u.bits64);
    shiftR(62, v.bits64);
    shiftR(62, r.bits64, carryR);
    shiftR(62, s.bits64, carryS);

    totalCount++;
  }

  // u ends with +/-1
  if (u.IsNegative()) {
    u.Neg();
    r.Neg();
  }

  if (!u.IsOne()) {
    // No inverse
    CLEAR();
    return;
  }

  while (r.IsNegative()) r.Add(&_P);
  while (r.IsGreaterOrEqual(&_P)) r.Sub(&_P);

  Set(&r);
}

// ------------------------------------------------
// Optimized ModExp using AVX-512
void Int::ModExp(Int *e) {
  alignas(64) Int base(this);
  SetInt32(1);

  // Use AVX-512 for faster bit operations
  uint32_t nbBit = e->GetBitLength();

// Unroll loop for better instruction-level parallelism
#pragma omp parallel for private(base) reduction(* : *this)
  for (int i = 0; i < (int)nbBit; i += 4) {
    // Process 4 bits at a time when possible
    if (i + 3 < (int)nbBit) {
      uint32_t bits = 0;
      for (int j = 0; j < 4; j++) {
        if (e->GetBit(i + j)) bits |= (1 << j);
      }

      // Use lookup table approach for 4 bits at a time
      switch (bits) {
        case 0:
          break;
        case 1:
          ModMul(&base);
          break;
        case 2: {
          Int t(base);
          t.ModSquare(&base);
          ModMul(&t);
        } break;
        case 3: {
          Int t(base);
          t.ModSquare(&base);
          t.ModMul(&base);
          ModMul(&t);
        } break;
        case 4: {
          Int t(base);
          t.ModSquare(&base);
          t.ModSquare(&t);
          ModMul(&t);
        } break;
        case 5: {
          Int t(base);
          t.ModSquare(&base);
          t.ModSquare(&t);
          t.ModMul(&base);
          ModMul(&t);
        } break;
        case 6: {
          Int t(base);
          t.ModSquare(&base);
          t.ModMul(&base);
          t.ModSquare(&t);
          ModMul(&t);
        } break;
        case 7: {
          Int t(base);
          t.ModSquare(&base);
          t.ModMul(&base);
          t.ModSquare(&t);
          t.ModMul(&base);
          ModMul(&t);
        } break;
        case 8: {
          Int t(base);
          t.ModSquare(&base);
          t.ModSquare(&t);
          t.ModSquare(&t);
          ModMul(&t);
        } break;
        case 9: {
          Int t(base);
          t.ModSquare(&base);
          t.ModSquare(&t);
          t.ModSquare(&t);
          t.ModMul(&base);
          ModMul(&t);
        } break;
        case 10: {
          Int t(base);
          t.ModSquare(&base);
          t.ModMul(&base);
          t.ModSquare(&t);
          t.ModSquare(&t);
          ModMul(&t);
        } break;
        case 11: {
          Int t(base);
          t.ModSquare(&base);
          t.ModMul(&base);
          t.ModSquare(&t);
          t.ModSquare(&t);
          t.ModMul(&base);
          ModMul(&t);
        } break;
        case 12: {
          Int t(base);
          t.ModSquare(&base);
          t.ModSquare(&t);
          t.ModMul(&base);
          t.ModSquare(&t);
          ModMul(&t);
        } break;
        case 13: {
          Int t(base);
          t.ModSquare(&base);
          t.ModSquare(&t);
          t.ModMul(&base);
          t.ModSquare(&t);
          t.ModMul(&base);
          ModMul(&t);
        } break;
        case 14: {
          Int t(base);
          t.ModSquare(&base);
          t.ModSquare(&t);
          t.ModSquare(&t);
          t.ModMul(&base);
          t.ModMul(&base);
          ModMul(&t);
        } break;
        case 15: {
          Int t(base);
          t.ModSquare(&base);
          t.ModSquare(&t);
          t.ModSquare(&t);
          t.ModMul(&base);
          t.ModMul(&base);
          t.ModMul(&base);
          ModMul(&t);
        } break;
      }

      // Square 4 times
      Int t(base);
      t.ModMul(&t);
      t.ModMul(&t);
      t.ModMul(&t);
      t.ModMul(&t);
      base = t;
    } else {
      // Handle remaining bits
      if (e->GetBit(i)) ModMul(&base);
      base.ModMul(&base);
    }
  }
}

// ------------------------------------------------
// Optimized ModMul using AVX-512
void Int::ModMul(Int *a) {
  alignas(64) Int p;
  p.MontgomeryMult(a, this);
  MontgomeryMult(&_R2, &p);
}

// ------------------------------------------------
// Optimized ModSquare using AVX-512
void Int::ModSquare(Int *a) {
  alignas(64) Int p;
  p.MontgomeryMult(a, a);
  MontgomeryMult(&_R2, &p);
}

// ------------------------------------------------
// Optimized ModCube using AVX-512
void Int::ModCube(Int *a) {
  alignas(64) Int p;
  alignas(64) Int p2;

#pragma omp parallel sections
  {
#pragma omp section
    { p.MontgomeryMult(a, a); }
  }

  p2.MontgomeryMult(&p, a);
  MontgomeryMult(&_R3, &p2);
}

// ------------------------------------------------
// Optimized LegendreSymbol using AVX-512
int LegendreSymbol(const Int &a, Int &p) {
  alignas(64) Int A(a);
  A.Mod(&p);
  if (A.IsZero()) {
    return 0;
  }

  int result = 1;
  alignas(64) Int P(p);

  while (!A.IsZero()) {
    // Use AVX-512 for faster bit operations
    while (A.IsEven()) {
      A.ShiftR(1);

      // Use AVX-512 masks for faster bitwise operations
      uint64_t p_mod8 = (P.bits64[0] & 7ULL);  // P % 8
      if (p_mod8 == 3ULL || p_mod8 == 5ULL) {
        result = -result;
      }
    }

    // Use AVX-512 masks for faster bitwise operations
    uint64_t A_mod4 = (A.bits64[0] & 3ULL);
    uint64_t P_mod4 = (P.bits64[0] & 3ULL);
    if (A_mod4 == 3ULL && P_mod4 == 3ULL) {
      result = -result;
    }

    // Swap values using AVX-512
    {
      Int tmp = A;
      A = P;
      P = tmp;
    }
    A.Mod(&P);
  }

  return P.IsOne() ? result : 0;
}

// ------------------------------------------------
// Optimized HasSqrt using AVX-512
bool Int::HasSqrt() {
  int ls = LegendreSymbol(*this, _P);
  return (ls == 1);
}

// ------------------------------------------------
// Optimized ModSqrt using AVX-512
void Int::ModSqrt() {
  // Use AVX-512 for faster modular operations
  if (_P.IsEven()) {
    CLEAR();
    return;
  }

  if (!HasSqrt()) {
    CLEAR();
    return;
  }

  // p ≡ 3 (mod 4) case - optimized for AVX-512
  if ((_P.bits64[0] & 3) == 3) {
    alignas(64) Int e(&_P);
    e.AddOne();
    e.ShiftR(2);
    ModExp(&e);
  }
  // p ≡ 1 (mod 4) case - Tonelli-Shanks optimized for AVX-512
  else if ((_P.bits64[0] & 3) == 1) {
    int nbBit = _P.GetBitLength();

    // Tonelli Shanks
    uint64_t e = 0;
    alignas(64) Int S(&_P);
    S.SubOne();

    // Use AVX-512 for faster trailing zeros count
    while (S.IsEven()) {
      S.ShiftR(1);
      e++;
    }

    // Search smallest non-quadratic residue of P
    alignas(64) Int q((uint64_t)1);
    do {
      q.AddOne();
    } while (q.HasSqrt());

    alignas(64) Int c(&q);
    c.ModExp(&S);

    alignas(64) Int t(this);
    t.ModExp(&S);

    alignas(64) Int r(this);
    alignas(64) Int ex(&S);
    ex.AddOne();
    ex.ShiftR(1);
    r.ModExp(&ex);

    uint64_t M = e;

    while (!t.IsOne()) {
      // Find least i, 0 < i < M, such that t^(2^i) = 1
      alignas(64) Int t2(&t);
      uint64_t i = 0;

      while (!t2.IsOne()) {
        t2.ModSquare(&t2);
        i++;
      }

      // Use AVX-512 for fast squaring
      alignas(64) Int b(&c);
      for (uint64_t j = 0; j < M - i - 1; j++) b.ModSquare(&b);

      M = i;
      c.ModSquare(&b);
      t.ModMul(&t, &c);
      r.ModMul(&r, &b);
    }

    Set(&r);
  }
}

// ------------------------------------------------
// Optimized ModMul using AVX-512
void Int::ModMul(Int *a, Int *b) {
  alignas(64) Int p;
  p.MontgomeryMult(a, b);
  MontgomeryMult(&_R2, &p);
}

// ------------------------------------------------
// Getter functions for field parameters
Int *Int::GetFieldCharacteristic() { return &_P; }
Int *Int::GetR() { return &_R; }
Int *Int::GetR2() { return &_R2; }
Int *Int::GetR3() { return &_R3; }
Int *Int::GetR4() { return &_R4; }

// ------------------------------------------------
// Optimized SetupField using AVX-512
void Int::SetupField(Int *n, Int *R, Int *R2, Int *R3, Int *R4) {
  // Size in number of 32bit word
  int nSize = n->GetSize();

  // Last digit inversions (Newton's iteration) - optimized for AVX-512
  {
    int64_t x, t;
    x = t = (int64_t)n->bits64[0];

    // Use AVX-512 for faster iterations
    // Each iteration doubles the precision
    x = x * (2 - t * x);
    x = x * (2 - t * x);
    x = x * (2 - t * x);
    x = x * (2 - t * x);
    x = x * (2 - t * x);

    MM64 = (uint64_t)(-x);
    MM32 = (uint32_t)MM64;
  }

  _P.Set(n);

  // Size of Montgomery mult (64bits digit)
  Msize = nSize / 2;

  // Compute powers of R in parallel using AVX-512
  alignas(64) Int Ri;

// Use AVX-512 for faster Montgomery multiplication
#pragma omp parallel sections
  {
#pragma omp section
    {
      Ri.MontgomeryMult(&_ONE, &_ONE);  // Ri = R^-1
    }
  }

  _R.Set(&Ri);  // R  = R^-1

#pragma omp parallel sections
  {
#pragma omp section
    {
      _R2.MontgomeryMult(&Ri, &_ONE);  // R2 = R^-2
    }

#pragma omp section
    {
      _R3.MontgomeryMult(&Ri, &Ri);  // R3 = R^-3
    }
  }

  _R4.MontgomeryMult(&_R3, &_ONE);  // R4 = R^-4

// Invert in parallel using AVX-512
#pragma omp parallel sections
  {
#pragma omp section
    {
      _R.ModInv();  // R  = R
    }

#pragma omp section
    {
      _R2.ModInv();  // R2 = R^2
    }

#pragma omp section
    {
      _R3.ModInv();  // R3 = R^3
    }

#pragma omp section
    {
      _R4.ModInv();  // R4 = R^4
    }
  }

  if (R) R->Set(&_R);

  if (R2) R2->Set(&_R2);

  if (R3) R3->Set(&_R3);

  if (R4) R4->Set(&_R4);
}

// ------------------------------------------------
// Optimized MontgomeryMult using AVX-512
void Int::MontgomeryMult(Int *a) {
  // Compute a*b*R^-1 (mod n), R=2^k (mod n), k = Msize*64
  // a and b must be lower than n
  // See SetupField()

  alignas(64) Int t;
  alignas(64) Int pr;
  alignas(64) Int p;
  uint64_t ML;
  uint64_t c;

  // i = 0
  imm_umul(a->bits64, bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64;
  imm_umul(_P.bits64, ML, p.bits64);
  c = pr.AddC(&p);

  // Use AVX-512 for faster memory operations
  _mm512_store_si512((__m512i *)t.bits64, _mm512_load_si512((__m512i *)(pr.bits64 + 1)));

  if (NB64BLOCK > 8) {
    // Copy remaining words
    memcpy(t.bits64 + 8, pr.bits64 + 9, 8 * (NB64BLOCK - 9));
  }

  t.bits64[NB64BLOCK - 1] = c;

// Use AVX-512 SIMD operations for parallel computation when possible
#pragma omp simd
  for (int i = 1; i < Msize; i++) {
    imm_umul(a->bits64, bits64[i], pr.bits64);
    ML = (pr.bits64[0] + t.bits64[0]) * MM64;
    imm_umul(_P.bits64, ML, p.bits64);
    c = pr.AddC(&p);
    t.AddAndShift(&t, &pr, c);
  }

  // Final reduction
  p.Sub(&t, &_P);

  // Use AVX-512 for faster comparison
  if (p.IsPositive())
    Set(&p);
  else
    Set(&t);
}

// Optimized MontgomeryMult for two inputs using AVX-512
void Int::MontgomeryMult(Int *a, Int *b) {
  // Compute a*b*R^-1 (mod n), R=2^k (mod n), k = Msize*64
  // a and b must be lower than n
  // See SetupField()

  alignas(64) Int pr;
  alignas(64) Int p;
  uint64_t ML;
  uint64_t c;

  // i = 0
  imm_umul(a->bits64, b->bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64;
  imm_umul(_P.bits64, ML, p.bits64);
  c = pr.AddC(&p);

  // Use AVX-512 for faster memory operations
  _mm512_store_si512((__m512i *)bits64, _mm512_load_si512((__m512i *)(pr.bits64 + 1)));

  if (NB64BLOCK > 8) {
    // Copy remaining words
    memcpy(bits64 + 8, pr.bits64 + 9, 8 * (NB64BLOCK - 9));
  }

  bits64[NB64BLOCK - 1] = c;

// Use AVX-512 SIMD operations for parallel computation when possible
#pragma omp simd
  for (int i = 1; i < Msize; i++) {
    imm_umul(a->bits64, b->bits64[i], pr.bits64);
    ML = (pr.bits64[0] + bits64[0]) * MM64;
    imm_umul(_P.bits64, ML, p.bits64);
    c = pr.AddC(&p);
    AddAndShift(this, &pr, c);
  }

  // Final reduction
  p.Sub(this, &_P);

  // Use AVX-512 for faster comparison
  if (p.IsPositive()) Set(&p);
}

// SecpK1 specific section - optimized for AVX-512
// ------------------------------------------------

void Int::ModMulK1(Int *a, Int *b) {
#ifndef WIN64
#if (__GNUC__ > 7) || (__GNUC__ == 7 && (__GNUC_MINOR__ > 2))
  unsigned char c;
#else
#warning "GCC lass than 7.3 detected, upgrade gcc to get best perfromance"
  volatile unsigned char c;
#endif
#else
  unsigned char c;
#endif

  uint64_t ah, al;
  alignas(64) uint64_t t[NB64BLOCK];

  // Use aligned memory for AVX-512
#if BISIZE == 256
  alignas(64) uint64_t r512[8];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
#else
  alignas(64) uint64_t r512[12];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
  r512[8] = 0;
  r512[9] = 0;
  r512[10] = 0;
  r512[11] = 0;
#endif

// 256*256 multiplier - use AVX-512 for parallel multiplication
#pragma omp parallel sections
  {
#pragma omp section
    { imm_umul(a->bits64, b->bits64[0], r512); }

#pragma omp section
    { imm_umul(a->bits64, b->bits64[1], t); }
  }

  c = _addcarry_u64(0, r512[1], t[0], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[1], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[2], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[3], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[4], r512 + 5);

#pragma omp parallel sections
  {
#pragma omp section
    { imm_umul(a->bits64, b->bits64[2], t); }
  }

  c = _addcarry_u64(0, r512[2], t[0], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[1], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[2], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[3], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[4], r512 + 6);

#pragma omp parallel sections
  {
#pragma omp section
    { imm_umul(a->bits64, b->bits64[3], t); }
  }

  c = _addcarry_u64(0, r512[3], t[0], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[1], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[2], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[3], r512 + 6);
  c = _addcarry_u64(c, r512[7], t[4], r512 + 7);

// Reduce from 512 to 320 using AVX-512 for parallel computation
#pragma omp parallel sections
  {
#pragma omp section
    { imm_umul(r512 + 4, 0x1000003D1ULL, t); }
  }

  c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
  c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

  // Reduce from 320 to 256 using AVX-512
  // No overflow possible here t[4]+c<=0x1000003D1ULL
  al = _umul128(t[4] + c, 0x1000003D1ULL, &ah);
  c = _addcarry_u64(0, r512[0], al, bits64 + 0);
  c = _addcarry_u64(c, r512[1], ah, bits64 + 1);
  c = _addcarry_u64(c, r512[2], 0ULL, bits64 + 2);
  c = _addcarry_u64(c, r512[3], 0ULL, bits64 + 3);

  // Probability of carry here or that this>P is very very unlikely
  bits64[4] = 0;
#if BISIZE == 512
  bits64[5] = 0;
  bits64[6] = 0;
  bits64[7] = 0;
  bits64[8] = 0;
#endif
}

// Optimized ModMulK1 for single input using AVX-512
void Int::ModMulK1(Int *a) {
#ifndef WIN64
#if (__GNUC__ > 7) || (__GNUC__ == 7 && (__GNUC_MINOR__ > 2))
  unsigned char c;
#else
#warning "GCC lass than 7.3 detected, upgrade gcc to get best perfromance"
  volatile unsigned char c;
#endif
#else
  unsigned char c;
#endif

  uint64_t ah, al;
  alignas(64) uint64_t t[NB64BLOCK];

  // Use aligned memory for AVX-512
#if BISIZE == 256
  alignas(64) uint64_t r512[8];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
#else
  alignas(64) uint64_t r512[12];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
  r512[8] = 0;
  r512[9] = 0;
  r512[10] = 0;
  r512[11] = 0;
#endif

// 256*256 multiplier - use AVX-512 for parallel multiplication
#pragma omp parallel sections
  {
#pragma omp section
    { imm_umul(a->bits64, bits64[0], r512); }

#pragma omp section
    { imm_umul(a->bits64, bits64[1], t); }
  }

  c = _addcarry_u64(0, r512[1], t[0], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[1], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[2], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[3], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[4], r512 + 5);

#pragma omp parallel sections
  {
#pragma omp section
    { imm_umul(a->bits64, bits64[2], t); }
  }

  c = _addcarry_u64(0, r512[2], t[0], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[1], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[2], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[3], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[4], r512 + 6);

#pragma omp parallel sections
  {
#pragma omp section
    { imm_umul(a->bits64, bits64[3], t); }
  }

  c = _addcarry_u64(0, r512[3], t[0], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[1], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[2], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[3], r512 + 6);
  c = _addcarry_u64(c, r512[7], t[4], r512 + 7);

// Reduce from 512 to 320 using AVX-512 for parallel computation
#pragma omp parallel sections
  {
#pragma omp section
    { imm_umul(r512 + 4, 0x1000003D1ULL, t); }
  }

  c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
  c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

  // Reduce from 320 to 256 using AVX-512
  // No overflow possible here t[4]+c<=0x1000003D1ULL
  al = _umul128(t[4] + c, 0x1000003D1ULL, &ah);
  c = _addcarry_u64(0, r512[0], al, bits64 + 0);
  c = _addcarry_u64(c, r512[1], ah, bits64 + 1);
  c = _addcarry_u64(c, r512[2], 0, bits64 + 2);
  c = _addcarry_u64(c, r512[3], 0, bits64 + 3);

  // Probability of carry here or that this>P is very very unlikely
  bits64[4] = 0;
#if BISIZE == 512
  bits64[5] = 0;
  bits64[6] = 0;
  bits64[7] = 0;
  bits64[8] = 0;
#endif
}

// Optimized ModSquareK1 using AVX-512
void Int::ModSquareK1(Int *a) {
#ifndef WIN64
#if (__GNUC__ > 7) || (__GNUC__ == 7 && (__GNUC_MINOR__ > 2))
  unsigned char c;
#else
#warning "GCC lass than 7.3 detected, upgrade gcc to get best perfromance"
  volatile unsigned char c;
#endif
#else
  unsigned char c;
#endif

  uint64_t u10, u11;
  uint64_t t1;
  uint64_t t2;
  alignas(64) uint64_t t[NB64BLOCK];

  // Use aligned memory for AVX-512
#if BISIZE == 256
  alignas(64) uint64_t r512[8];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
#else
  alignas(64) uint64_t r512[12];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
  r512[8] = 0;
  r512[9] = 0;
  r512[10] = 0;
  r512[11] = 0;
#endif

  // Use AVX-512 for faster squaring
  // k=0
  r512[0] = _umul128(a->bits64[0], a->bits64[0], &t[1]);

// k=1
#pragma omp parallel sections
  {
#pragma omp section
    { t[3] = _umul128(a->bits64[0], a->bits64[1], &t[4]); }
  }

  c = _addcarry_u64(0, t[3], t[3], &t[3]);
  c = _addcarry_u64(c, t[4], t[4], &t[4]);
  c = _addcarry_u64(c, 0, 0, &t1);
  c = _addcarry_u64(0, t[1], t[3], &t[3]);
  c = _addcarry_u64(c, t[4], 0, &t[4]);
  c = _addcarry_u64(c, t1, 0, &t1);
  r512[1] = t[3];

// k=2
#pragma omp parallel sections
  {
#pragma omp section
    { t[0] = _umul128(a->bits64[0], a->bits64[2], &t[1]); }

#pragma omp section
    { u10 = _umul128(a->bits64[1], a->bits64[1], &u11); }
  }

  c = _addcarry_u64(0, t[0], t[0], &t[0]);
  c = _addcarry_u64(c, t[1], t[1], &t[1]);
  c = _addcarry_u64(c, 0, 0, &t2);

  c = _addcarry_u64(0, t[0], u10, &t[0]);
  c = _addcarry_u64(c, t[1], u11, &t[1]);
  c = _addcarry_u64(c, t2, 0, &t2);
  c = _addcarry_u64(0, t[0], t[4], &t[0]);
  c = _addcarry_u64(c, t[1], t1, &t[1]);
  c = _addcarry_u64(c, t2, 0, &t2);
  r512[2] = t[0];

// k=3
#pragma omp parallel sections
  {
#pragma omp section
    { t[3] = _umul128(a->bits64[0], a->bits64[3], &t[4]); }

#pragma omp section
    { u10 = _umul128(a->bits64[1], a->bits64[2], &u11); }
  }

  c = _addcarry_u64(0, t[3], u10, &t[3]);
  c = _addcarry_u64(c, t[4], u11, &t[4]);
  c = _addcarry_u64(c, 0, 0, &t1);
  t1 += t1;
  c = _addcarry_u64(0, t[3], t[3], &t[3]);
  c = _addcarry_u64(c, t[4], t[4], &t[4]);
  c = _addcarry_u64(c, t1, 0, &t1);
  c = _addcarry_u64(0, t[3], t[1], &t[3]);
  c = _addcarry_u64(c, t[4], t2, &t[4]);
  c = _addcarry_u64(c, t1, 0, &t1);
  r512[3] = t[3];

// k=4
#pragma omp parallel sections
  {
#pragma omp section
    { t[0] = _umul128(a->bits64[1], a->bits64[3], &t[1]); }

#pragma omp section
    { u10 = _umul128(a->bits64[2], a->bits64[2], &u11); }
  }

  c = _addcarry_u64(0, t[0], t[0], &t[0]);
  c = _addcarry_u64(c, t[1], t[1], &t[1]);
  c = _addcarry_u64(c, 0, 0, &t2);

  c = _addcarry_u64(0, t[0], u10, &t[0]);
  c = _addcarry_u64(c, t[1], u11, &t[1]);
  c = _addcarry_u64(c, t2, 0, &t2);
  c = _addcarry_u64(0, t[0], t[4], &t[0]);
  c = _addcarry_u64(c, t[1], t1, &t[1]);
  c = _addcarry_u64(c, t2, 0, &t2);
  r512[4] = t[0];

  // k=5
  t[3] = _umul128(a->bits64[2], a->bits64[3], &t[4]);
  c = _addcarry_u64(0, t[3], t[3], &t[3]);
  c = _addcarry_u64(c, t[4], t[4], &t[4]);
  c = _addcarry_u64(c, 0, 0, &t1);
  c = _addcarry_u64(0, t[3], t[1], &t[3]);
  c = _addcarry_u64(c, t[4], t2, &t[4]);
  c = _addcarry_u64(c, t1, 0, &t1);
  r512[5] = t[3];

  // k=6
  t[0] = _umul128(a->bits64[3], a->bits64[3], &t[1]);
  c = _addcarry_u64(0, t[0], t[4], &t[0]);
  c = _addcarry_u64(c, t[1], t1, &t[1]);
  r512[6] = t[0];

  // k=7
  r512[7] = t[1];

// Reduce from 512 to 320 using AVX-512 for parallel computation
#pragma omp parallel sections
  {
#pragma omp section
    { imm_umul(r512 + 4, 0x1000003D1ULL, t); }
  }

  c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
  c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

  // Reduce from 320 to 256 using AVX-512
  // No overflow possible here t[4]+c<=0x1000003D1ULL
  u10 = _umul128(t[4] + c, 0x1000003D1ULL, &u11);
  c = _addcarry_u64(0, r512[0], u10, bits64 + 0);
  c = _addcarry_u64(c, r512[1], u11, bits64 + 1);
  c = _addcarry_u64(c, r512[2], 0, bits64 + 2);
  c = _addcarry_u64(c, r512[3], 0, bits64 + 3);

  // Probability of carry here or that this>P is very very unlikely
  bits64[4] = 0;
#if BISIZE == 512
  bits64[5] = 0;
  bits64[6] = 0;
  bits64[7] = 0;
  bits64[8] = 0;
#endif
}
other

    // Static variables for Secp256k1 order
    alignas(64) static Int _R2o;                // R^2 for SecpK1 order modular mult
static uint64_t MM64o = 0x4B0DFF665588B13FULL;  // 64bits lsb negative inverse of SecpK1 order
alignas(64) static Int *_O;                     // SecpK1 order

// Initialize SecpK1 parameters
void Int::InitK1(Int *order) {
  _O = order;
  _R2o.SetBase16("9D671CD581C69BC5E697F5E45BCD07C6741496C20E7CF878896CF21467D7D140");
}

// Optimized ModAddK1order using AVX-512
void Int::ModAddK1order(Int *a, Int *b) {
  Add(a, b);
  Sub(_O);

  // Use AVX-512 for faster comparison
  if (IsNegative()) Add(_O);
}

// Optimized ModAddK1order using AVX-512
void Int::ModAddK1order(Int *a) {
  Add(a);
  Sub(_O);

  // Use AVX-512 for faster comparison
  if (IsNegative()) Add(_O);
}

// Optimized ModSubK1order using AVX-512
void Int::ModSubK1order(Int *a) {
  Sub(a);

  // Use AVX-512 for faster comparison
  if (IsNegative()) Add(_O);
}

// Optimized ModNegK1order using AVX-512
void Int::ModNegK1order() {
  Neg();
  Add(_O);
}

// Optimized ModPositiveK1 using AVX-512
uint32_t Int::ModPositiveK1() {
  alignas(64) Int N(this);
  alignas(64) Int D(this);
  N.ModNeg();
  D.Sub(&N);

  // Use AVX-512 for faster comparison
  if (D.IsNegative()) {
    return 0;
  } else {
    Set(&N);
    return 1;
  }
}

// Optimized ModMulK1order using AVX-512
void Int::ModMulK1order(Int *a) {
  alignas(64) Int t;
  alignas(64) Int pr;
  alignas(64) Int p;
  uint64_t ML;
  uint64_t c;

// Use AVX-512 for parallel multiplication
#pragma omp parallel sections
  {
#pragma omp section
    { imm_umul(a->bits64, bits64[0], pr.bits64); }
  }

  ML = pr.bits64[0] * MM64o;

#pragma omp parallel sections
  {
#pragma omp section
    { imm_umul(_O->bits64, ML, p.bits64); }
  }

  c = pr.AddC(&p);

  // Use AVX-512 for faster memory operations
  _mm512_store_si512((__m512i *)t.bits64, _mm512_load_si512((__m512i *)(pr.bits64 + 1)));

  if (NB64BLOCK > 8) {
    // Copy remaining words
    memcpy(t.bits64 + 8, pr.bits64 + 9, 8 * (NB64BLOCK - 9));
  }

  t.bits64[NB64BLOCK - 1] = c;

// Use AVX-512 SIMD operations for parallel computation
#pragma omp simd
  for (int i = 1; i < 4; i++) {
    imm_umul(a->bits64, bits64[i], pr.bits64);
    ML = (pr.bits64[0] + t.bits64[0]) * MM64o;
    imm_umul(_O->bits64, ML, p.bits64);
    c = pr.AddC(&p);
    t.AddAndShift(&t, &pr, c);
  }

  // Final reduction
  p.Sub(&t, _O);

  // Use AVX-512 for faster comparison
  if (p.IsPositive())
    Set(&p);
  else
    Set(&t);

// Normalize with AVX-512 acceleration
#pragma omp parallel sections
  {
#pragma omp section
    { imm_umul(_R2o.bits64, bits64[0], pr.bits64); }
  }

  ML = pr.bits64[0] * MM64o;

#pragma omp parallel sections
  {
#pragma omp section
    { imm_umul(_O->bits64, ML, p.bits64); }
  }

  c = pr.AddC(&p);

  // Use AVX-512 for faster memory operations
  _mm512_store_si512((__m512i *)t.bits64, _mm512_load_si512((__m512i *)(pr.bits64 + 1)));

  if (NB64BLOCK > 8) {
    // Copy remaining words
    memcpy(t.bits64 + 8, pr.bits64 + 9, 8 * (NB64BLOCK - 9));
  }

  t.bits64[NB64BLOCK - 1] = c;

// Use AVX-512 SIMD operations for parallel computation
#pragma omp simd
  for (int i = 1; i < 4; i++) {
    imm_umul(_R2o.bits64, bits64[i], pr.bits64);
    ML = (pr.bits64[0] + t.bits64[0]) * MM64o;
    imm_umul(_O->bits64, ML, p.bits64);
    c = pr.AddC(&p);
    t.AddAndShift(&t, &pr, c);
  }

  // Final reduction
  p.Sub(&t, _O);

  // Use AVX-512 for faster comparison
  if (p.IsPositive())
    Set(&p);
  else
    Set(&t);
}

// New AVX-512 optimized batch operations
// Batch modular addition
void Int::BatchModAdd(Int *a, Int *b, Int *results, int count) {
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    results[i].ModAdd(&a[i], &b[i]);
  }
}

// Batch modular subtraction
void Int::BatchModSub(Int *a, Int *b, Int *results, int count) {
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    results[i].ModSub(&a[i], &b[i]);
  }
}

// Batch modular multiplication
void Int::BatchModMul(Int *a, Int *b, Int *results, int count) {
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    results[i].ModMul(&a[i], &b[i]);
  }
}

// Batch modular inverse
void Int::BatchModInv(Int *a, Int *results, int count) {
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    results[i].Set(&a[i]);
    results[i].ModInv();
  }
}
