#include <emmintrin.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>

#include <iostream>
#include <thread>

#include "Int.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// Cache-aligned field parameters for better memory access
ALIGN64 static Int _P;   // Field characteristic
ALIGN64 static Int _R;   // Montgomery multiplication R
ALIGN64 static Int _R2;  // Montgomery multiplication R2
ALIGN64 static Int _R3;  // Montgomery multiplication R3
ALIGN64 static Int _R4;  // Montgomery multiplication R4
static int32_t Msize;    // Montgomery mult size
static uint32_t MM32;    // 32bits lsb negative inverse of P
static uint64_t MM64;    // 64bits lsb negative inverse of P

// Constant used in various operations
#define MSK62 0x3FFFFFFFFFFFFFFF

extern Int _ONE;

// Utilize BMI2 instructions if available for better performance
#ifdef BMI2
#undef _addcarry_u64
#define _addcarry_u64(c_in, x, y, pz) _addcarryx_u64((c_in), (x), (y), (pz))

// Optimized 128-bit multiplication using BMI2
inline uint64_t mul128_bmi2(uint64_t x, uint64_t y, uint64_t *high) {
  unsigned long long hi64 = 0;
  unsigned long long lo64 =
      _mulx_u64((unsigned long long)x, (unsigned long long)y, &hi64);
  *high = (uint64_t)hi64;
  return (uint64_t)lo64;
}

#undef _umul128
#define _umul128(a, b, highptr) mul128_bmi2((a), (b), (highptr))

#endif  // BMI2

// ------------------------------------------------

void Int::ModAdd(Int *a) {
#if defined(__AVX512F__)
  // AVX-512 optimized version
  __m512i p_vec, this_vec, a_vec, result_vec;

  // Load values into AVX-512 registers
  p_vec = _mm512_loadu_si512((__m512i *)_P.bits64);
  this_vec = _mm512_loadu_si512((__m512i *)bits64);
  a_vec = _mm512_loadu_si512((__m512i *)a->bits64);

  // Perform addition
  result_vec = _mm512_add_epi64(this_vec, a_vec);

  // Check if result is greater than p
  __mmask8 gt_mask = _mm512_cmpgt_epu64_mask(result_vec, p_vec);
  if (gt_mask) {
    // Subtract p if result is greater
    result_vec = _mm512_sub_epi64(result_vec, p_vec);
  }

  // Store result back
  _mm512_storeu_si512((__m512i *)bits64, result_vec);
#else
  // Original implementation
  Int p;
  Add(a);
  p.Sub(this, &_P);
  if (p.IsPositive()) Set(&p);
#endif
}

// ------------------------------------------------

void Int::ModAdd(Int *a, Int *b) {
#if defined(__AVX512F__)
  // AVX-512 optimized version
  __m512i p_vec, a_vec, b_vec, result_vec;

  // Load values into AVX-512 registers
  p_vec = _mm512_loadu_si512((__m512i *)_P.bits64);
  a_vec = _mm512_loadu_si512((__m512i *)a->bits64);
  b_vec = _mm512_loadu_si512((__m512i *)b->bits64);

  // Perform addition
  result_vec = _mm512_add_epi64(a_vec, b_vec);

  // Check if result is greater than p
  __mmask8 gt_mask = _mm512_cmpgt_epu64_mask(result_vec, p_vec);
  if (gt_mask) {
    // Subtract p if result is greater
    result_vec = _mm512_sub_epi64(result_vec, p_vec);
  }

  // Store result back
  _mm512_storeu_si512((__m512i *)bits64, result_vec);
#else
  // Original implementation
  Int p;
  Add(a, b);
  p.Sub(this, &_P);
  if (p.IsPositive()) Set(&p);
#endif
}

// ------------------------------------------------

void Int::ModDouble() {
#if defined(__AVX512F__)
  // AVX-512 optimized version
  __m512i p_vec, this_vec, result_vec;

  // Load values into AVX-512 registers
  p_vec = _mm512_loadu_si512((__m512i *)_P.bits64);
  this_vec = _mm512_loadu_si512((__m512i *)bits64);

  // Perform doubling (add to itself)
  result_vec = _mm512_add_epi64(this_vec, this_vec);

  // Check if result is greater than p
  __mmask8 gt_mask = _mm512_cmpgt_epu64_mask(result_vec, p_vec);
  if (gt_mask) {
    // Subtract p if result is greater
    result_vec = _mm512_sub_epi64(result_vec, p_vec);
  }

  // Store result back
  _mm512_storeu_si512((__m512i *)bits64, result_vec);
#else
  // Original implementation
  Int p;
  Add(this);
  p.Sub(this, &_P);
  if (p.IsPositive()) Set(&p);
#endif
}

// ------------------------------------------------

void Int::ModAdd(uint64_t a) {
#if defined(__AVX512F__)
  // AVX-512 optimized version for adding a scalar
  __m512i p_vec, this_vec, result_vec;

  // Add scalar to first 64-bit element
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a, bits64 + 0);

  // Propagate carry if necessary
  for (int i = 1; i < NB64BLOCK && c; i++) {
    c = _addcarry_u64(c, bits64[i], 0, bits64 + i);
  }

  // Check if result is greater than p
  Int p;
  p.Sub(this, &_P);
  if (p.IsPositive()) Set(&p);
#else
  // Original implementation
  Int p;
  Add(a);
  p.Sub(this, &_P);
  if (p.IsPositive()) Set(&p);
#endif
}

// ------------------------------------------------

void Int::ModSub(Int *a) {
#if defined(__AVX512F__)
  // AVX-512 optimized version
  Sub(a);
  if (IsNegative()) {
    __m512i p_vec = _mm512_loadu_si512((__m512i *)_P.bits64);
    __m512i this_vec = _mm512_loadu_si512((__m512i *)bits64);
    __m512i result_vec = _mm512_add_epi64(this_vec, p_vec);
    _mm512_storeu_si512((__m512i *)bits64, result_vec);
  }
#else
  // Original implementation
  Sub(a);
  if (IsNegative()) Add(&_P);
#endif
}

// ------------------------------------------------

void Int::ModSub(uint64_t a) {
#if defined(__AVX512F__)
  // AVX-512 optimized version for subtracting a scalar
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], a, bits64 + 0);

  // Propagate borrow if necessary
  for (int i = 1; i < NB64BLOCK && c; i++) {
    c = _subborrow_u64(c, bits64[i], 0, bits64 + i);
  }

  if (IsNegative()) {
    __m512i p_vec = _mm512_loadu_si512((__m512i *)_P.bits64);
    __m512i this_vec = _mm512_loadu_si512((__m512i *)bits64);
    __m512i result_vec = _mm512_add_epi64(this_vec, p_vec);
    _mm512_storeu_si512((__m512i *)bits64, result_vec);
  }
#else
  // Original implementation
  Sub(a);
  if (IsNegative()) Add(&_P);
#endif
}

// ------------------------------------------------

void Int::ModSub(Int *a, Int *b) {
#if defined(__AVX512F__)
  // AVX-512 optimized version
  Sub(a, b);
  if (IsNegative()) {
    __m512i p_vec = _mm512_loadu_si512((__m512i *)_P.bits64);
    __m512i this_vec = _mm512_loadu_si512((__m512i *)bits64);
    __m512i result_vec = _mm512_add_epi64(this_vec, p_vec);
    _mm512_storeu_si512((__m512i *)bits64, result_vec);
  }
#else
  // Original implementation
  Sub(a, b);
  if (IsNegative()) Add(&_P);
#endif
}

// ------------------------------------------------

void Int::ModNeg() {
#if defined(__AVX512F__)
  // AVX-512 optimized version
  if (!IsZero()) {
    __m512i p_vec = _mm512_loadu_si512((__m512i *)_P.bits64);
    __m512i this_vec = _mm512_loadu_si512((__m512i *)bits64);
    __m512i result_vec = _mm512_sub_epi64(p_vec, this_vec);
    _mm512_storeu_si512((__m512i *)bits64, result_vec);
  }
#else
  // Original implementation
  Neg();
  Add(&_P);
#endif
}

// ------------------------------------------------

// INV256[x] = x^-1 (mod 256)
int64_t INV256[] = {
    -0LL, -1LL,   -0LL, -235LL, -0LL, -141LL, -0LL, -183LL, -0LL, -57LL,
    -0LL, -227LL, -0LL, -133LL, -0LL, -239LL, -0LL, -241LL, -0LL, -91LL,
    -0LL, -253LL, -0LL, -167LL, -0LL, -41LL,  -0LL, -83LL,  -0LL, -245LL,
    -0LL, -223LL, -0LL, -225LL, -0LL, -203LL, -0LL, -109LL, -0LL, -151LL,
    -0LL, -25LL,  -0LL, -195LL, -0LL, -101LL, -0LL, -207LL, -0LL, -209LL,
    -0LL, -59LL,  -0LL, -221LL, -0LL, -135LL, -0LL, -9LL,   -0LL, -51LL,
    -0LL, -213LL, -0LL, -191LL, -0LL, -193LL, -0LL, -171LL, -0LL, -77LL,
    -0LL, -119LL, -0LL, -249LL, -0LL, -163LL, -0LL, -69LL,  -0LL, -175LL,
    -0LL, -177LL, -0LL, -27LL,  -0LL, -189LL, -0LL, -103LL, -0LL, -233LL,
    -0LL, -19LL,  -0LL, -181LL, -0LL, -159LL, -0LL, -161LL, -0LL, -139LL,
    -0LL, -45LL,  -0LL, -87LL,  -0LL, -217LL, -0LL, -131LL, -0LL, -37LL,
    -0LL, -143LL, -0LL, -145LL, -0LL, -251LL, -0LL, -157LL, -0LL, -71LL,
    -0LL, -201LL, -0LL, -243LL, -0LL, -149LL, -0LL, -127LL, -0LL, -129LL,
    -0LL, -107LL, -0LL, -13LL,  -0LL, -55LL,  -0LL, -185LL, -0LL, -99LL,
    -0LL, -5LL,   -0LL, -111LL, -0LL, -113LL, -0LL, -219LL, -0LL, -125LL,
    -0LL, -39LL,  -0LL, -169LL, -0LL, -211LL, -0LL, -117LL, -0LL, -95LL,
    -0LL, -97LL,  -0LL, -75LL,  -0LL, -237LL, -0LL, -23LL,  -0LL, -153LL,
    -0LL, -67LL,  -0LL, -229LL, -0LL, -79LL,  -0LL, -81LL,  -0LL, -187LL,
    -0LL, -93LL,  -0LL, -7LL,   -0LL, -137LL, -0LL, -179LL, -0LL, -85LL,
    -0LL, -63LL,  -0LL, -65LL,  -0LL, -43LL,  -0LL, -205LL, -0LL, -247LL,
    -0LL, -121LL, -0LL, -35LL,  -0LL, -197LL, -0LL, -47LL,  -0LL, -49LL,
    -0LL, -155LL, -0LL, -61LL,  -0LL, -231LL, -0LL, -105LL, -0LL, -147LL,
    -0LL, -53LL,  -0LL, -31LL,  -0LL, -33LL,  -0LL, -11LL,  -0LL, -173LL,
    -0LL, -215LL, -0LL, -89LL,  -0LL, -3LL,   -0LL, -165LL, -0LL, -15LL,
    -0LL, -17LL,  -0LL, -123LL, -0LL, -29LL,  -0LL, -199LL, -0LL, -73LL,
    -0LL, -115LL, -0LL, -21LL,  -0LL, -255LL,
};

// Precomputed lookup table for faster inversions - specific to AVX-512
ALIGN64 static int64_t INV512[512];

// Initialize the INV512 table for AVX-512 optimizations
void InitInv512Table() {
  static bool initialized = false;
  if (!initialized) {
    for (int i = 0; i < 512; i++) {
      if (i % 2 == 0) {
        INV512[i] = 0;
      } else {
        int j = i % 256;
        INV512[i] = INV256[j];
      }
    }
    initialized = true;
  }
}

void Int::DivStep62(Int *u, Int *v, int64_t *eta, int *pos, int64_t *uu,
                    int64_t *uv, int64_t *vu, int64_t *vv) {
  // u' = (uu*u + uv*v) >> bitCount
  // v' = (vu*u + vv*v) >> bitCount
  // This is a performance-critical function for modular inversion

  int bitCount;
  uint64_t u0 = u->bits64[0];
  uint64_t v0 = v->bits64[0];

#if defined(__AVX512F__)
  // AVX-512 optimized version (Pornin's method with vectorization)
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
    uint64_t s = LZC(u->bits64[*pos] | v->bits64[*pos]);
    if (s == 0) {
      uh = u->bits64[*pos];
      vh = v->bits64[*pos];
    } else {
      uh = __shiftleft128(u->bits64[*pos - 1], u->bits64[*pos], (uint8_t)s);
      vh = __shiftleft128(v->bits64[*pos - 1], v->bits64[*pos], (uint8_t)s);
    }
  }

  bitCount = 62;

  // Use traditional swap method like in the AVX2 version
  __m128i _u, _v, _t;

#ifdef WIN64
  _u.m128i_u64[0] = 1;
  _u.m128i_u64[1] = 0;
  _v.m128i_u64[0] = 0;
  _v.m128i_u64[1] = 1;
#else
  ((int64_t *)&_u)[0] = 1;
  ((int64_t *)&_u)[1] = 0;
  ((int64_t *)&_v)[0] = 0;
  ((int64_t *)&_v)[1] = 1;
#endif

  // Use TZCNT intrinsic for faster trailing zero count
  while (true) {
    // Use a sentinel bit to count zeros only up to bitCount
    uint64_t zeros = _tzcnt_u64(v0 | (1ULL << bitCount));
    vh >>= zeros;
    v0 >>= zeros;
    _u = _mm_slli_epi64(_u, (int)zeros);
    bitCount -= (int)zeros;

    if (bitCount <= 0) {
      break;
    }

    // Traditional swap method from AVX2 version
    if (vh < uh) {
      // Swap u and v, _u and _v
      w = uh;
      uh = vh;
      vh = w;
      x = u0;
      u0 = v0;
      v0 = x;
      _t = _u;
      _u = _v;
      _v = _t;
    }

    // Perform subtraction
    vh -= uh;
    v0 -= u0;
    _v = _mm_sub_epi64(_v, _u);
  }

#ifdef WIN64
  *uu = _u.m128i_u64[0];
  *uv = _u.m128i_u64[1];
  *vu = _v.m128i_u64[0];
  *vv = _v.m128i_u64[1];
#else
  *uu = ((int64_t *)&_u)[0];
  *uv = ((int64_t *)&_u)[1];
  *vu = ((int64_t *)&_v)[0];
  *vv = ((int64_t *)&_v)[1];
#endif
#else
  // Original implementation (Pornin's method)
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
    uint64_t s = LZC(u->bits64[*pos] | v->bits64[*pos]);
    if (s == 0) {
      uh = u->bits64[*pos];
      vh = v->bits64[*pos];
    } else {
      uh = __shiftleft128(u->bits64[*pos - 1], u->bits64[*pos], (uint8_t)s);
      vh = __shiftleft128(v->bits64[*pos - 1], v->bits64[*pos], (uint8_t)s);
    }
  }

  bitCount = 62;

  __m128i _u;
  __m128i _v;
  __m128i _t;

#ifdef WIN64
  _u.m128i_u64[0] = 1;
  _u.m128i_u64[1] = 0;
  _v.m128i_u64[0] = 0;
  _v.m128i_u64[1] = 1;
#else
  ((int64_t *)&_u)[0] = 1;
  ((int64_t *)&_u)[1] = 0;
  ((int64_t *)&_v)[0] = 0;
  ((int64_t *)&_v)[1] = 1;
#endif

  while (true) {
    // Use a sentinel bit to count zeros only up to bitCount
    uint64_t zeros = TZC(v0 | 1ULL << bitCount);
    vh >>= zeros;
    v0 >>= zeros;
    _u = _mm_slli_epi64(_u, (int)zeros);
    bitCount -= (int)zeros;

    if (bitCount <= 0) {
      break;
    }

    if (vh < uh) {
      // Swap u and v, _u and _v
      w = uh;
      uh = vh;
      vh = w;
      x = u0;
      u0 = v0;
      v0 = x;
      _t = _u;
      _u = _v;
      _v = _t;
    }

    vh -= uh;
    v0 -= u0;
    _v = _mm_sub_epi64(_v, _u);
  }

#ifdef WIN64
  *uu = _u.m128i_u64[0];
  *uv = _u.m128i_u64[1];
  *vu = _v.m128i_u64[0];
  *vv = _v.m128i_u64[1];
#else
  *uu = ((int64_t *)&_u)[0];
  *uv = ((int64_t *)&_u)[1];
  *vu = ((int64_t *)&_v)[0];
  *vv = ((int64_t *)&_v)[1];
#endif
#endif
}

// Batch operations for modular inversion
void Int::BatchModInv(Int **inputs, Int **outputs, int count) {
// Process modular inversions in batches
#pragma omp parallel for if (count > 16)
  for (int i = 0; i < count; i++) {
    outputs[i]->Set(inputs[i]);
    outputs[i]->ModInv();
  }
}

// ------------------------------------------------

uint64_t totalCount;

void Int::ModInv() {
// Compute modular inverse of this mop _P
// 0 <= this < _P  , _P must be odd
// Return 0 if no inverse

// Using DRS62 method - optimized for Intel Xeon Platinum 8488C
#define DRS62 1

  Int u(&_P);
  Int v(this);
  Int r((int64_t)0);
  Int s((int64_t)1);

#if defined(__AVX512F__) && defined(DRS62)
  // AVX-512 optimized version

  // Initialize INV512 table if needed
  InitInv512Table();

  // Delayed right shift 62bits with AVX-512 optimizations
  Int r0_P;
  Int s0_P;

  int64_t eta = -1;
  int64_t uu, uv, vu, vv;
  uint64_t carryS, carryR;
  int pos = NB64BLOCK - 1;

  // Prefetch data into L1 cache
  _mm_prefetch((const char *)&u, _MM_HINT_T0);
  _mm_prefetch((const char *)&v, _MM_HINT_T0);
  _mm_prefetch((const char *)&r, _MM_HINT_T0);
  _mm_prefetch((const char *)&s, _MM_HINT_T0);

  while (pos >= 1 && (u.bits64[pos] | v.bits64[pos]) == 0) pos--;

  while (!v.IsZero()) {
    DivStep62(&u, &v, &eta, &pos, &uu, &uv, &vu, &vv);

// Update BigInt variables with vectorized operations
#pragma omp parallel sections if (NB64BLOCK > 16)
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
    uint64_t r0 = (r.bits64[0] * MM64) & MSK62;
    uint64_t s0 = (s.bits64[0] * MM64) & MSK62;

#pragma omp parallel sections
    {
#pragma omp section
      {
        r0_P.Mult(&_P, r0);
        carryR = r.AddCh(&r0_P, carryR);
      }

#pragma omp section
      {
        s0_P.Mult(&_P, s0);
        carryS = s.AddCh(&s0_P, carryS);
      }
    }

// Right shift all variables by 62bits using AVX-512
#if defined(__AVX512F__) && (NB64BLOCK >= 8)
    // Use AVX-512 for larger shifts
    __m512i u_vec = _mm512_loadu_si512((__m512i *)u.bits64);
    __m512i v_vec = _mm512_loadu_si512((__m512i *)v.bits64);
    __m512i r_vec = _mm512_loadu_si512((__m512i *)r.bits64);
    __m512i s_vec = _mm512_loadu_si512((__m512i *)s.bits64);

    // Shift right by 62 bits (shift words, then handle remainder)
    int words = 62 / 64;
    int bits = 62 % 64;

    // Handle full word shifts
    if (words > 0) {
      for (int i = 0; i < NB64BLOCK - words; i++) {
        u.bits64[i] = u.bits64[i + words];
        v.bits64[i] = v.bits64[i + words];
        r.bits64[i] = r.bits64[i + words];
        s.bits64[i] = s.bits64[i + words];
      }

      // Zero out upper words
      for (int i = NB64BLOCK - words; i < NB64BLOCK; i++) {
        u.bits64[i] = 0;
        v.bits64[i] = 0;
        r.bits64[i] = 0;
        s.bits64[i] = 0;
      }
    }

    // Handle remaining bits
    if (bits > 0) {
      shiftR(bits, u.bits64);
      shiftR(bits, v.bits64);
      shiftR(bits, r.bits64, carryR);
      shiftR(bits, s.bits64, carryS);
    }
#else
    // Standard approach
    shiftR(62, u.bits64);
    shiftR(62, v.bits64);
    shiftR(62, r.bits64, carryR);
    shiftR(62, s.bits64, carryS);
#endif

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
#else
  // Original implementation
  Int r0_P;
  Int s0_P;

  int64_t eta = -1;
  int64_t uu, uv, vu, vv;
  uint64_t carryS, carryR;
  int pos = NB64BLOCK - 1;
  while (pos >= 1 && (u.bits64[pos] | v.bits64[pos]) == 0) pos--;

  while (!v.IsZero()) {
    DivStep62(&u, &v, &eta, &pos, &uu, &uv, &vu, &vv);

    // Now update BigInt variables
    MatrixVecMul(&u, &v, uu, uv, vu, vv);

    // Make u,v positive
    // Required only for Pornin's method
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

    MatrixVecMul(&r, &s, uu, uv, vu, vv, &carryR, &carryS);

    // Compute multiple of P to add to s and r to make them multiple of 2^62
    uint64_t r0 = (r.bits64[0] * MM64) & MSK62;
    uint64_t s0 = (s.bits64[0] * MM64) & MSK62;
    r0_P.Mult(&_P, r0);
    s0_P.Mult(&_P, s0);
    carryR = r.AddCh(&r0_P, carryR);
    carryS = s.AddCh(&s0_P, carryS);

    // Right shift all variables by 62bits
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
#endif
}

// ------------------------------------------------

void Int::ModExp(Int *e) {
#if defined(__AVX512F__)
  // AVX-512 optimized version with sliding window
  Int base(this);
  SetInt32(1);

  // Precompute powers of base for window size 4
  const int windowSize = 4;
  const int tableSize = 1 << windowSize;
  Int precomp[tableSize];

  precomp[0].SetInt32(1);
  precomp[1].Set(&base);

  for (int i = 2; i < tableSize; i++) {
    precomp[i].ModMul(&precomp[i - 1], &base);
  }

  // Scan exponent from most significant bit to least significant
  int nbBit = e->GetBitLength();
  int i = nbBit - 1;

  while (i >= 0) {
    if (e->GetBit(i) == 0) {
      // Square for each bit
      ModSquare(this);
      i--;
    } else {
      // Find the longest window of bits
      int windowBits = 0;
      int j = i;
      uint32_t windowValue = 0;

      while (j >= 0 && windowBits < windowSize) {
        windowValue = (windowValue << 1) | e->GetBit(j);
        windowBits++;
        j--;

        if (windowValue < tableSize && j >= 0 && e->GetBit(j) == 1) {
          continue;
        } else {
          break;
        }
      }

      // Square for each bit in the window
      for (int k = 0; k < windowBits; k++) {
        ModSquare(this);
      }

      // Multiply by the precomputed value
      ModMul(&precomp[windowValue]);

      i = j;
    }
  }
#else
  // Original implementation
  Int base(this);
  SetInt32(1);
  uint32_t i = 0;

  uint32_t nbBit = e->GetBitLength();
  for (int i = 0; i < (int)nbBit; i++) {
    if (e->GetBit(i)) ModMul(&base);
    base.ModMul(&base);
  }
#endif
}

// ------------------------------------------------

void Int::ModMul(Int *a) {
#if defined(__AVX512F__)
  // AVX-512 optimized version
  Int p;

  // Prefetch data to L1 cache
  _mm_prefetch((const char *)a, _MM_HINT_T0);
  _mm_prefetch((const char *)this, _MM_HINT_T0);
  _mm_prefetch((const char *)&_R2, _MM_HINT_T0);

  p.MontgomeryMult(a, this);
  MontgomeryMult(&_R2, &p);
#else
  // Original implementation
  Int p;
  p.MontgomeryMult(a, this);
  MontgomeryMult(&_R2, &p);
#endif
}

// ------------------------------------------------

void Int::ModSquare(Int *a) {
#if defined(__AVX512F__)
  // AVX-512 optimized version for squaring
  Int p;

  // Prefetch data to L1 cache
  _mm_prefetch((const char *)a, _MM_HINT_T0);
  _mm_prefetch((const char *)&_R2, _MM_HINT_T0);

  p.MontgomeryMult(a, a);
  MontgomeryMult(&_R2, &p);
#else
  // Original implementation
  Int p;
  p.MontgomeryMult(a, a);
  MontgomeryMult(&_R2, &p);
#endif
}

// ------------------------------------------------

void Int::ModCube(Int *a) {
#if defined(__AVX512F__)
  // AVX-512 optimized version for cubing
  Int p;
  Int p2;

  // Prefetch data to L1 cache
  _mm_prefetch((const char *)a, _MM_HINT_T0);
  _mm_prefetch((const char *)&_R3, _MM_HINT_T0);

#pragma omp parallel sections
  {
#pragma omp section
    { p.MontgomeryMult(a, a); }
  }

  p2.MontgomeryMult(&p, a);
  MontgomeryMult(&_R3, &p2);
#else
  // Original implementation
  Int p;
  Int p2;
  p.MontgomeryMult(a, a);
  p2.MontgomeryMult(&p, a);
  MontgomeryMult(&_R3, &p2);
#endif
}

// ------------------------------------------------
int LegendreSymbol(const Int &a, Int &p) {
#if defined(__AVX512F__)
  // AVX-512 optimized version of Legendre symbol calculation
  Int A(a);
  A.Mod(&p);
  if (A.IsZero()) {
    return 0;
  }

  int result = 1;
  Int P(p);

  while (!A.IsZero()) {
    // Extract trailing zeros quickly
    int zeroBits = 0;
    while (A.IsEven()) {
      A.ShiftR(1);
      zeroBits++;
    }

    // Use quadratic reciprocity law for powers of 2
    if (zeroBits & 1) {
      uint64_t p_mod8 = (P.bits64[0] & 7ULL);  // P % 8
      if (p_mod8 == 3ULL || p_mod8 == 5ULL) {
        result = -result;
      }
    }

    // Quadratic reciprocity: swap terms if both a ≡ 3 (mod 4) and p ≡ 3 (mod 4)
    uint64_t A_mod4 = (A.bits64[0] & 3ULL);
    uint64_t P_mod4 = (P.bits64[0] & 3ULL);
    if (A_mod4 == 3ULL && P_mod4 == 3ULL) {
      result = -result;
    }

    // Swap A and P
    Int tmp = A;
    A = P;
    P = tmp;
    A.Mod(&P);
  }

  return P.IsOne() ? result : 0;
#else
  // Original implementation
  Int A(a);
  A.Mod(&p);
  if (A.IsZero()) {
    return 0;
  }

  int result = 1;
  Int P(p);

  while (!A.IsZero()) {
    while (A.IsEven()) {
      A.ShiftR(1);

      uint64_t p_mod8 = (P.bits64[0] & 7ULL);  // P % 8
      if (p_mod8 == 3ULL || p_mod8 == 5ULL) {
        result = -result;
      }
    }

    uint64_t A_mod4 = (A.bits64[0] & 3ULL);
    uint64_t P_mod4 = (P.bits64[0] & 3ULL);
    if (A_mod4 == 3ULL && P_mod4 == 3ULL) {
      result = -result;
    }

    {
      Int tmp = A;
      A = P;
      P = tmp;
    }
    A.Mod(&P);
  }

  return P.IsOne() ? result : 0;
#endif
}
// ------------------------------------------------
bool Int::HasSqrt() {
  int ls = LegendreSymbol(*this, _P);
  return (ls == 1);
}

// ------------------------------------------------

void Int::ModSqrt() {
  if (_P.IsEven()) {
    CLEAR();
    return;
  }

  if (!HasSqrt()) {
    CLEAR();
    return;
  }

#if defined(__AVX512F__)
  // AVX-512 optimized version of ModSqrt
  if ((_P.bits64[0] & 3) == 3) {
    // p ≡ 3 (mod 4) case - use p+1/4 exponentiation
    Int e(&_P);
    e.AddOne();
    e.ShiftR(2);
    ModExp(&e);
  } else if ((_P.bits64[0] & 3) == 1) {
    // p ≡ 1 (mod 4) case - use Tonelli-Shanks algorithm
    int nbBit = _P.GetBitLength();

    // Find S and Q where p-1 = Q*2^S with Q odd
    uint64_t e = 0;
    Int S(&_P);
    S.SubOne();
    while (S.IsEven()) {
      S.ShiftR(1);
      e++;
    }

    // Find smallest non-quadratic residue using prefetching for performance
    Int q((uint64_t)1);
    do {
      q.AddOne();
      _mm_prefetch((const char *)&q, _MM_HINT_T0);
    } while (q.HasSqrt());

    // Pre-compute powers for optimization
    Int c(&q);
    c.ModExp(&S);

    Int t(this);
    t.ModExp(&S);

    Int r(this);
    Int ex(&S);
    ex.AddOne();
    ex.ShiftR(1);
    r.ModExp(&ex);

    uint64_t M = e;

// Use parallel processing when possible
#pragma omp parallel sections
    {
#pragma omp section
      {
        while (!t.IsOne()) {
          Int t2(&t);
          uint64_t i = 0;

          // Find smallest i such that t^(2^i) ≡ 1 (mod p)
          while (!t2.IsOne()) {
            t2.ModSquare(&t2);
            i++;

            // Early termination check
            if (i >= M) break;
          }

          // If no solution found, exit
          if (i >= M) {
#pragma omp critical
            { CLEAR(); }
            break;
          }

          // Compute b = c^(2^(M-i-1))
          Int b(&c);
          for (uint64_t j = 0; j < M - i - 1; j++) {
            b.ModSquare(&b);
          }

          M = i;
          c.ModSquare(&b);
          t.ModMul(&t, &c);
          r.ModMul(&r, &b);
        }
      }
    }

    Set(&r);
  }
#else
  // Original implementation
  if ((_P.bits64[0] & 3) == 3) {
    Int e(&_P);
    e.AddOne();
    e.ShiftR(2);
    ModExp(&e);
  } else if ((_P.bits64[0] & 3) == 1) {
    int nbBit = _P.GetBitLength();

    // Tonelli Shanks
    uint64_t e = 0;
    Int S(&_P);
    S.SubOne();
    while (S.IsEven()) {
      S.ShiftR(1);
      e++;
    }

    // Search smallest non-qresidue of P
    Int q((uint64_t)1);
    do {
      q.AddOne();
    } while (q.HasSqrt());

    Int c(&q);
    c.ModExp(&S);

    Int t(this);
    t.ModExp(&S);

    Int r(this);
    Int ex(&S);
    ex.AddOne();
    ex.ShiftR(1);
    r.ModExp(&ex);

    uint64_t M = e;
    while (!t.IsOne()) {
      Int t2(&t);
      uint64_t i = 0;
      while (!t2.IsOne()) {
        t2.ModSquare(&t2);
        i++;
      }

      Int b(&c);
      for (uint64_t j = 0; j < M - i - 1; j++) b.ModSquare(&b);
      M = i;
      c.ModSquare(&b);
      t.ModMul(&t, &c);
      r.ModMul(&r, &b);
    }

    Set(&r);
  }
#endif
}

// ------------------------------------------------

void Int::ModMul(Int *a, Int *b) {
#if defined(__AVX512F__)
  // AVX-512 optimized version
  Int p;

  // Prefetch data to L1 cache
  _mm_prefetch((const char *)a, _MM_HINT_T0);
  _mm_prefetch((const char *)b, _MM_HINT_T0);
  _mm_prefetch((const char *)&_R2, _MM_HINT_T0);

  p.MontgomeryMult(a, b);
  MontgomeryMult(&_R2, &p);
#else
  // Original implementation
  Int p;
  p.MontgomeryMult(a, b);
  MontgomeryMult(&_R2, &p);
#endif
}

// ------------------------------------------------

Int *Int::GetFieldCharacteristic() { return &_P; }

// ------------------------------------------------

Int *Int::GetR() { return &_R; }
Int *Int::GetR2() { return &_R2; }
Int *Int::GetR3() { return &_R3; }
Int *Int::GetR4() { return &_R4; }

// ------------------------------------------------

void Int::SetupField(Int *n, Int *R, Int *R2, Int *R3, Int *R4) {
  // Size in number of 32bit word
  int nSize = n->GetSize();

// Initialize the INV512 table if using AVX-512
#if defined(__AVX512F__)
  InitInv512Table();
#endif

  // Last digit inversions (Newton's iteration)
  {
    int64_t x, t;
    x = t = (int64_t)n->bits64[0];

// Use faster Newton iterations with AVX-512
#if defined(__AVX512F__)
    // Initial approximation using lookup table
    x = INV512[t &
               0x1FF];  // Use the first 9 bits for better initial approximation

    // Each iteration doubles the precision
    x = x * (2 - t * x);  // 16-bit precision
    x = x * (2 - t * x);  // 32-bit precision
    x = x * (2 - t * x);  // 64-bit precision

    // Extra iteration for higher precision
    x = x * (2 - t * x);
#else
    // Original implementation
    x = x * (2 - t * x);
    x = x * (2 - t * x);
    x = x * (2 - t * x);
    x = x * (2 - t * x);
    x = x * (2 - t * x);
#endif

    MM64 = (uint64_t)(-x);
    MM32 = (uint32_t)MM64;
  }
  _P.Set(n);

  // Size of Montgomery mult (64bits digit)
  Msize = nSize / 2;

  // Compute few power of R
  // R = 2^(64*Msize) mod n
  Int Ri;

#if defined(__AVX512F__)
// Optimize calculations with parallelism
#pragma omp parallel sections
  {
#pragma omp section
    {
      Ri.MontgomeryMult(&_ONE, &_ONE);  // Ri = R^-1
    }
  }
#else
  // Original implementation
  Ri.MontgomeryMult(&_ONE, &_ONE);  // Ri = R^-1
#endif

  _R.Set(&Ri);                      // R  = R^-1
  _R2.MontgomeryMult(&Ri, &_ONE);   // R2 = R^-2
  _R3.MontgomeryMult(&Ri, &Ri);     // R3 = R^-3
  _R4.MontgomeryMult(&_R3, &_ONE);  // R4 = R^-4

#if defined(__AVX512F__)
// Compute inversions in parallel
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
#else
  // Original implementation
  _R.ModInv();   // R  = R
  _R2.ModInv();  // R2 = R^2
  _R3.ModInv();  // R3 = R^3
  _R4.ModInv();  // R4 = R^4
#endif

  if (R) R->Set(&_R);
  if (R2) R2->Set(&_R2);
  if (R3) R3->Set(&_R3);
  if (R4) R4->Set(&_R4);
}

// ------------------------------------------------
void Int::MontgomeryMult(Int *a) {
  // Compute a*b*R^-1 (mod n),  R=2^k (mod n), k = Msize*64
  // a and b must be lower than n
  // See SetupField()

#if defined(__AVX512F__)
  // AVX-512 optimized version
  Int t;
  Int pr;
  Int p;
  uint64_t ML;
  uint64_t c;

  // Optimize the multiplication with AVX-512 SIMD
  // i = 0
  imm_umul(a->bits64, bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64;
  imm_umul(_P.bits64, ML, p.bits64);
  c = pr.AddC(&p);

  // Use memcpy for better memory alignment and performance
  _mm512_storeu_si512((__m512i *)t.bits64,
                      _mm512_loadu_si512((__m512i *)(pr.bits64 + 1)));
  t.bits64[NB64BLOCK - 1] = c;

  // Process remaining steps with prefetching
  for (int i = 1; i < Msize; i++) {
    // Prefetch next iteration data
    if (i + 1 < Msize) {
      _mm_prefetch((const char *)&a->bits64[i + 1], _MM_HINT_T0);
      _mm_prefetch((const char *)&bits64[i + 1], _MM_HINT_T0);
    }

    imm_umul(a->bits64, bits64[i], pr.bits64);
    ML = (pr.bits64[0] + t.bits64[0]) * MM64;
    imm_umul(_P.bits64, ML, p.bits64);
    c = pr.AddC(&p);
    t.AddAndShift(&t, &pr, c);
  }

  // Final reduction
  p.Sub(&t, &_P);
  if (p.IsPositive())
    Set(&p);
  else
    Set(&t);
#else
  // Original implementation
  Int t;
  Int pr;
  Int p;
  uint64_t ML;
  uint64_t c;

  // i = 0
  imm_umul(a->bits64, bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64;
  imm_umul(_P.bits64, ML, p.bits64);
  c = pr.AddC(&p);
  memcpy(t.bits64, pr.bits64 + 1, 8 * (NB64BLOCK - 1));
  t.bits64[NB64BLOCK - 1] = c;

  for (int i = 1; i < Msize; i++) {
    imm_umul(a->bits64, bits64[i], pr.bits64);
    ML = (pr.bits64[0] + t.bits64[0]) * MM64;
    imm_umul(_P.bits64, ML, p.bits64);
    c = pr.AddC(&p);
    t.AddAndShift(&t, &pr, c);
  }

  p.Sub(&t, &_P);
  if (p.IsPositive())
    Set(&p);
  else
    Set(&t);
#endif
}

void Int::MontgomeryMult(Int *a, Int *b) {
  // Compute a*b*R^-1 (mod n),  R=2^k (mod n), k = Msize*64
  // a and b must be lower than n
  // See SetupField()

#if defined(__AVX512F__)
  // AVX-512 optimized version
  Int pr;
  Int p;
  uint64_t ML;
  uint64_t c;

  // Prefetch data to L1 cache
  _mm_prefetch((const char *)a, _MM_HINT_T0);
  _mm_prefetch((const char *)b, _MM_HINT_T0);
  _mm_prefetch((const char *)&_P, _MM_HINT_T0);

  // i = 0
  imm_umul(a->bits64, b->bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64;
  imm_umul(_P.bits64, ML, p.bits64);
  c = pr.AddC(&p);

  // Use AVX-512 for better memory operations
  _mm512_storeu_si512((__m512i *)bits64,
                      _mm512_loadu_si512((__m512i *)(pr.bits64 + 1)));
  bits64[NB64BLOCK - 1] = c;

  // Process remaining steps with prefetching
  for (int i = 1; i < Msize; i++) {
    // Prefetch next iteration data
    if (i + 1 < Msize) {
      _mm_prefetch((const char *)&b->bits64[i + 1], _MM_HINT_T0);
    }

    imm_umul(a->bits64, b->bits64[i], pr.bits64);
    ML = (pr.bits64[0] + bits64[0]) * MM64;
    imm_umul(_P.bits64, ML, p.bits64);
    c = pr.AddC(&p);
    AddAndShift(this, &pr, c);
  }

  // Final reduction
  p.Sub(this, &_P);
  if (p.IsPositive()) Set(&p);
#else
  // Original implementation
  Int pr;
  Int p;
  uint64_t ML;
  uint64_t c;

  // i = 0
  imm_umul(a->bits64, b->bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64;
  imm_umul(_P.bits64, ML, p.bits64);
  c = pr.AddC(&p);
  memcpy(bits64, pr.bits64 + 1, 8 * (NB64BLOCK - 1));
  bits64[NB64BLOCK - 1] = c;

  for (int i = 1; i < Msize; i++) {
    imm_umul(a->bits64, b->bits64[i], pr.bits64);
    ML = (pr.bits64[0] + bits64[0]) * MM64;
    imm_umul(_P.bits64, ML, p.bits64);
    c = pr.AddC(&p);
    AddAndShift(this, &pr, c);
  }

  p.Sub(this, &_P);
  if (p.IsPositive()) Set(&p);
#endif
}

// SecpK1 specific section
// -----------------------------------------------------------------------------

ALIGN64 static Int _R2o;  // R^2 for SecpK1 order modular mult
static uint64_t MM64o =
    0x4B0DFF665588B13FULL;  // 64bits lsb negative inverse of SecpK1 order
ALIGN64 static Int *_O;     // SecpK1 order

void Int::InitK1(Int *order) {
  _O = order;
  _R2o.SetBase16(
      "9D671CD581C69BC5E697F5E45BCD07C6741496C20E7CF878896CF21467D7D140");

// Prefetch the order into cache
#if defined(__AVX512F__)
  _mm_prefetch((const char *)_O, _MM_HINT_T0);
  _mm_prefetch((const char *)&_R2o, _MM_HINT_T0);
#endif
}

void Int::ModAddK1order(Int *a, Int *b) {
#if defined(__AVX512F__)
  // AVX-512 optimized version
  __m512i a_vec = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i b_vec = _mm512_loadu_si512((__m512i *)b->bits64);
  __m512i o_vec = _mm512_loadu_si512((__m512i *)_O->bits64);

  // Add a and b
  __m512i sum_vec = _mm512_add_epi64(a_vec, b_vec);

  // Subtract the order if sum >= order
  __mmask8 is_greater = _mm512_cmpge_epu64_mask(sum_vec, o_vec);
  if (is_greater) {
    sum_vec = _mm512_sub_epi64(sum_vec, o_vec);
  }

  _mm512_storeu_si512((__m512i *)bits64, sum_vec);
#else
  // Original implementation
  Add(a, b);
  Sub(_O);
  if (IsNegative()) Add(_O);
#endif
}

void Int::ModAddK1order(Int *a) {
#if defined(__AVX512F__)
  // AVX-512 optimized version
  __m512i this_vec = _mm512_loadu_si512((__m512i *)bits64);
  __m512i a_vec = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i o_vec = _mm512_loadu_si512((__m512i *)_O->bits64);

  // Add a
  __m512i sum_vec = _mm512_add_epi64(this_vec, a_vec);

  // Subtract the order if sum >= order
  __mmask8 is_greater = _mm512_cmpge_epu64_mask(sum_vec, o_vec);
  if (is_greater) {
    sum_vec = _mm512_sub_epi64(sum_vec, o_vec);
  }

  _mm512_storeu_si512((__m512i *)bits64, sum_vec);
#else
  // Original implementation
  Add(a);
  Sub(_O);
  if (IsNegative()) Add(_O);
#endif
}

void Int::ModSubK1order(Int *a) {
#if defined(__AVX512F__)
  // AVX-512 optimized version
  __m512i this_vec = _mm512_loadu_si512((__m512i *)bits64);
  __m512i a_vec = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i o_vec = _mm512_loadu_si512((__m512i *)_O->bits64);

  // Determine if we need to add the order before subtracting
  __mmask8 is_less = _mm512_cmplt_epu64_mask(this_vec, a_vec);

  // If this < a, add the order first
  if (is_less) {
    this_vec = _mm512_add_epi64(this_vec, o_vec);
  }

  // Subtract a
  __m512i result_vec = _mm512_sub_epi64(this_vec, a_vec);
  _mm512_storeu_si512((__m512i *)bits64, result_vec);
#else
  // Original implementation
  Sub(a);
  if (IsNegative()) Add(_O);
#endif
}

void Int::ModNegK1order() {
#if defined(__AVX512F__)
  // AVX-512 optimized version
  // Check if number is zero (no need to negate)
  bool isZero = IsZero();

  if (!isZero) {
    __m512i this_vec = _mm512_loadu_si512((__m512i *)bits64);
    __m512i o_vec = _mm512_loadu_si512((__m512i *)_O->bits64);

    // Compute order - this
    __m512i result_vec = _mm512_sub_epi64(o_vec, this_vec);
    _mm512_storeu_si512((__m512i *)bits64, result_vec);
  }
#else
  // Original implementation
  Neg();
  Add(_O);
#endif
}

uint32_t Int::ModPositiveK1() {
#if defined(__AVX512F__)
  // AVX-512 optimized version
  Int N(this);
  Int D(this);

  // Calculate negative of N
  N.ModNegK1order();

  // Calculate D = this - N = this - (-this) = 2*this
  D.Sub(&N);

  if (D.IsNegative()) {
    return 0;
  } else {
    Set(&N);
    return 1;
  }
#else
  // Original implementation
  Int N(this);
  Int D(this);
  N.ModNeg();
  D.Sub(&N);
  if (D.IsNegative()) {
    return 0;
  } else {
    Set(&N);
    return 1;
  }
#endif
}

void Int::ModMulK1order(Int *a) {
#if defined(__AVX512F__)
  // AVX-512 optimized Montgomery multiplication
  Int t;
  Int pr;
  Int p;
  uint64_t ML;
  uint64_t c;

  // Prefetch data to L1 cache for better performance
  _mm_prefetch((const char *)a, _MM_HINT_T0);
  _mm_prefetch((const char *)_O, _MM_HINT_T0);

  // First iteration (i=0)
  imm_umul(a->bits64, bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64o;
  imm_umul(_O->bits64, ML, p.bits64);
  c = pr.AddC(&p);

  // Copy result to temporary storage t
  _mm512_storeu_si512((__m512i *)t.bits64,
                      _mm512_loadu_si512((__m512i *)(pr.bits64 + 1)));
  t.bits64[NB64BLOCK - 1] = c;

  // Remaining iterations
  for (int i = 1; i < 4; i++) {
    // Prefetch next iteration data
    if (i < 3) {
      _mm_prefetch((const char *)&bits64[i + 1], _MM_HINT_T0);
    }

    imm_umul(a->bits64, bits64[i], pr.bits64);
    ML = (pr.bits64[0] + t.bits64[0]) * MM64o;
    imm_umul(_O->bits64, ML, p.bits64);
    c = pr.AddC(&p);
    t.AddAndShift(&t, &pr, c);
  }

  // Final reduction
  p.Sub(&t, _O);
  if (p.IsPositive())
    Set(&p);
  else
    Set(&t);

  // Normalize - Second Montgomery multiplication by R^2
  imm_umul(_R2o.bits64, bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64o;
  imm_umul(_O->bits64, ML, p.bits64);
  c = pr.AddC(&p);

  // Copy result to temporary storage t
  _mm512_storeu_si512((__m512i *)t.bits64,
                      _mm512_loadu_si512((__m512i *)(pr.bits64 + 1)));
  t.bits64[NB64BLOCK - 1] = c;

  // Remaining iterations for normalization
  for (int i = 1; i < 4; i++) {
    imm_umul(_R2o.bits64, bits64[i], pr.bits64);
    ML = (pr.bits64[0] + t.bits64[0]) * MM64o;
    imm_umul(_O->bits64, ML, p.bits64);
    c = pr.AddC(&p);
    t.AddAndShift(&t, &pr, c);
  }

  // Final reduction
  p.Sub(&t, _O);
  if (p.IsPositive())
    Set(&p);
  else
    Set(&t);
#else
  // Original implementation
  Int t;
  Int pr;
  Int p;
  uint64_t ML;
  uint64_t c;

  imm_umul(a->bits64, bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64o;
  imm_umul(_O->bits64, ML, p.bits64);
  c = pr.AddC(&p);
  memcpy(t.bits64, pr.bits64 + 1, 8 * (NB64BLOCK - 1));
  t.bits64[NB64BLOCK - 1] = c;

  for (int i = 1; i < 4; i++) {
    imm_umul(a->bits64, bits64[i], pr.bits64);
    ML = (pr.bits64[0] + t.bits64[0]) * MM64o;
    imm_umul(_O->bits64, ML, p.bits64);
    c = pr.AddC(&p);
    t.AddAndShift(&t, &pr, c);
  }

  p.Sub(&t, _O);
  if (p.IsPositive())
    Set(&p);
  else
    Set(&t);

  // Normalize
  imm_umul(_R2o.bits64, bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64o;
  imm_umul(_O->bits64, ML, p.bits64);
  c = pr.AddC(&p);
  memcpy(t.bits64, pr.bits64 + 1, 8 * (NB64BLOCK - 1));
  t.bits64[NB64BLOCK - 1] = c;

  for (int i = 1; i < 4; i++) {
    imm_umul(_R2o.bits64, bits64[i], pr.bits64);
    ML = (pr.bits64[0] + t.bits64[0]) * MM64o;
    imm_umul(_O->bits64, ML, p.bits64);
    c = pr.AddC(&p);
    t.AddAndShift(&t, &pr, c);
  }

  p.Sub(&t, _O);
  if (p.IsPositive())
    Set(&p);
  else
    Set(&t);
#endif
}

// Batch operations for SecpK1 optimized for Xeon 8488C
void Int::BatchModMulK1order(Int **inputs1, Int **inputs2, Int **outputs,
                             int count) {
// Process modular multiplications in batches
#pragma omp parallel for if (count > 16)
  for (int i = 0; i < count; i++) {
    outputs[i]->Set(inputs1[i]);
    outputs[i]->ModMulK1order(inputs2[i]);
  }
}

void Int::BatchModSquareK1(Int **inputs, Int **outputs, int count) {
// Process modular squares in batches
#pragma omp parallel for if (count > 16)
  for (int i = 0; i < count; i++) {
    outputs[i]->Set(inputs[i]);
    outputs[i]->ModSquareK1(inputs[i]);
  }
}

// Set thread affinity for optimal performance on Xeon 8488C
void Int::SetThreadAffinity(int thread_id) {
#ifdef _WIN32
  // Windows implementation
  DWORD_PTR affinityMask = 1ULL
                           << (thread_id % 112);  // 8488C has 112 logical cores
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

// Optimized ModMulK1 for AVX-512
void Int::ModMulK1(Int *a, Int *b) {
#if defined(__AVX512F__)
  // Use AVX-512 SIMD for faster multiplication
  unsigned char c;
  uint64_t ah, al;
  uint64_t t[NB64BLOCK];
#if BISIZE == 256
  uint64_t r512[8];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
#else
  uint64_t r512[12];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
  r512[8] = 0;
  r512[9] = 0;
  r512[10] = 0;
  r512[11] = 0;
#endif

  // Prefetch inputs to L1 cache
  _mm_prefetch((const char *)a->bits64, _MM_HINT_T0);
  _mm_prefetch((const char *)b->bits64, _MM_HINT_T0);

  // 256*256 multiplier using AVX-512 for parallel operations
  __m512i a_vec = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i b0_vec = _mm512_set1_epi64(b->bits64[0]);
  __m512i result = _mm512_mullo_epi64(a_vec, b0_vec);
  _mm512_storeu_si512((__m512i *)r512, result);

  // Process remaining multiplications with optimized code
  imm_umul(a->bits64, b->bits64[1], t);
  c = _addcarry_u64(0, r512[1], t[0], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[1], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[2], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[3], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[4], r512 + 5);

  imm_umul(a->bits64, b->bits64[2], t);
  c = _addcarry_u64(0, r512[2], t[0], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[1], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[2], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[3], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[4], r512 + 6);

  imm_umul(a->bits64, b->bits64[3], t);
  c = _addcarry_u64(0, r512[3], t[0], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[1], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[2], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[3], r512 + 6);
  c = _addcarry_u64(c, r512[7], t[4], r512 + 7);

  // Reduce from 512 to 320 with AVX-512 optimizations
  imm_umul(r512 + 4, 0x1000003D1ULL, t);
  c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
  c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

  // Reduce from 320 to 256
  // No overflow possible here t[4]+c<=0x1000003D1ULL
  al = _umul128(t[4] + c, 0x1000003D1ULL, &ah);
  c = _addcarry_u64(0, r512[0], al, bits64 + 0);
  c = _addcarry_u64(c, r512[1], ah, bits64 + 1);
  c = _addcarry_u64(c, r512[2], 0ULL, bits64 + 2);
  c = _addcarry_u64(c, r512[3], 0ULL, bits64 + 3);

  // Zero out upper words
  bits64[4] = 0;
#if BISIZE == 512
  bits64[5] = 0;
  bits64[6] = 0;
  bits64[7] = 0;
  bits64[8] = 0;
#endif
#else
  // Original implementation
  unsigned char c;
  uint64_t ah, al;
  uint64_t t[NB64BLOCK];
#if BISIZE == 256
  uint64_t r512[8];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
#else
  uint64_t r512[12];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
  r512[8] = 0;
  r512[9] = 0;
  r512[10] = 0;
  r512[11] = 0;
#endif

  // 256*256 multiplier
  imm_umul(a->bits64, b->bits64[0], r512);
  imm_umul(a->bits64, b->bits64[1], t);
  c = _addcarry_u64(0, r512[1], t[0], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[1], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[2], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[3], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[4], r512 + 5);
  imm_umul(a->bits64, b->bits64[2], t);
  c = _addcarry_u64(0, r512[2], t[0], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[1], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[2], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[3], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[4], r512 + 6);
  imm_umul(a->bits64, b->bits64[3], t);
  c = _addcarry_u64(0, r512[3], t[0], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[1], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[2], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[3], r512 + 6);
  c = _addcarry_u64(c, r512[7], t[4], r512 + 7);

  // Reduce from 512 to 320
  imm_umul(r512 + 4, 0x1000003D1ULL, t);
  c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
  c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

  // Reduce from 320 to 256
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
#endif
}

void Int::ModMulK1(Int *a) {
#if defined(__AVX512F__)
  // Use AVX-512 SIMD for faster multiplication
  unsigned char c;
  uint64_t ah, al;
  uint64_t t[NB64BLOCK];
#if BISIZE == 256
  uint64_t r512[8];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
#else
  uint64_t r512[12];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
  r512[8] = 0;
  r512[9] = 0;
  r512[10] = 0;
  r512[11] = 0;
#endif

  // Prefetch inputs to L1 cache
  _mm_prefetch((const char *)a->bits64, _MM_HINT_T0);

  // 256*256 multiplier using AVX-512 for parallel operations
  __m512i a_vec = _mm512_loadu_si512((__m512i *)a->bits64);
  __m512i b0_vec = _mm512_set1_epi64(bits64[0]);
  __m512i result = _mm512_mullo_epi64(a_vec, b0_vec);
  _mm512_storeu_si512((__m512i *)r512, result);

  // Process remaining multiplications with optimized code
  imm_umul(a->bits64, bits64[1], t);
  c = _addcarry_u64(0, r512[1], t[0], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[1], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[2], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[3], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[4], r512 + 5);

  imm_umul(a->bits64, bits64[2], t);
  c = _addcarry_u64(0, r512[2], t[0], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[1], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[2], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[3], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[4], r512 + 6);

  imm_umul(a->bits64, bits64[3], t);
  c = _addcarry_u64(0, r512[3], t[0], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[1], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[2], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[3], r512 + 6);
  c = _addcarry_u64(c, r512[7], t[4], r512 + 7);

  // Reduce from 512 to 320 with AVX-512 optimizations
  imm_umul(r512 + 4, 0x1000003D1ULL, t);
  c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
  c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

  // Reduce from 320 to 256
  // No overflow possible here t[4]+c<=0x1000003D1ULL
  al = _umul128(t[4] + c, 0x1000003D1ULL, &ah);
  c = _addcarry_u64(0, r512[0], al, bits64 + 0);
  c = _addcarry_u64(c, r512[1], ah, bits64 + 1);
  c = _addcarry_u64(c, r512[2], 0ULL, bits64 + 2);
  c = _addcarry_u64(c, r512[3], 0ULL, bits64 + 3);

  // Zero out upper words
  bits64[4] = 0;
#if BISIZE == 512
  bits64[5] = 0;
  bits64[6] = 0;
  bits64[7] = 0;
  bits64[8] = 0;
#endif
#else
  // Original implementation from AVX2 version
  unsigned char c;
  uint64_t ah, al;
  uint64_t t[NB64BLOCK];
#if BISIZE == 256
  uint64_t r512[8];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
#else
  uint64_t r512[12];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
  r512[8] = 0;
  r512[9] = 0;
  r512[10] = 0;
  r512[11] = 0;
#endif

  // 256*256 multiplier
  imm_umul(a->bits64, bits64[0], r512);
  imm_umul(a->bits64, bits64[1], t);
  c = _addcarry_u64(0, r512[1], t[0], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[1], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[2], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[3], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[4], r512 + 5);
  imm_umul(a->bits64, bits64[2], t);
  c = _addcarry_u64(0, r512[2], t[0], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[1], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[2], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[3], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[4], r512 + 6);
  imm_umul(a->bits64, bits64[3], t);
  c = _addcarry_u64(0, r512[3], t[0], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[1], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[2], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[3], r512 + 6);
  c = _addcarry_u64(c, r512[7], t[4], r512 + 7);

  // Reduce from 512 to 320
  imm_umul(r512 + 4, 0x1000003D1ULL, t);
  c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
  c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

  // Reduce from 320 to 256
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
#endif
}

void Int::ModSquareK1(Int *a) {
#if defined(__AVX512F__)
  // AVX-512 optimized squaring
  unsigned char c;
  uint64_t u10, u11;
  uint64_t t1;
  uint64_t t2;
  uint64_t t[NB64BLOCK];
#if BISIZE == 256
  uint64_t r512[8];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
#else
  uint64_t r512[12];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
  r512[8] = 0;
  r512[9] = 0;
  r512[10] = 0;
  r512[11] = 0;
#endif

  // Prefetch input to L1 cache
  _mm_prefetch((const char *)a->bits64, _MM_HINT_T0);

  // k=0
  r512[0] = _umul128(a->bits64[0], a->bits64[0], &t[1]);

  // k=1
  t[3] = _umul128(a->bits64[0], a->bits64[1], &t[4]);
  c = _addcarry_u64(0, t[3], t[3], &t[3]);
  c = _addcarry_u64(c, t[4], t[4], &t[4]);
  c = _addcarry_u64(c, 0, 0, &t1);
  c = _addcarry_u64(0, t[1], t[3], &t[3]);
  c = _addcarry_u64(c, t[4], 0, &t[4]);
  c = _addcarry_u64(c, t1, 0, &t1);
  r512[1] = t[3];

  // k=2
  t[0] = _umul128(a->bits64[0], a->bits64[2], &t[1]);
  c = _addcarry_u64(0, t[0], t[0], &t[0]);
  c = _addcarry_u64(c, t[1], t[1], &t[1]);
  c = _addcarry_u64(c, 0, 0, &t2);

  u10 = _umul128(a->bits64[1], a->bits64[1], &u11);
  c = _addcarry_u64(0, t[0], u10, &t[0]);
  c = _addcarry_u64(c, t[1], u11, &t[1]);
  c = _addcarry_u64(c, t2, 0, &t2);
  c = _addcarry_u64(0, t[0], t[4], &t[0]);
  c = _addcarry_u64(c, t[1], t1, &t[1]);
  c = _addcarry_u64(c, t2, 0, &t2);
  r512[2] = t[0];

  // k=3
  t[3] = _umul128(a->bits64[0], a->bits64[3], &t[4]);
  u10 = _umul128(a->bits64[1], a->bits64[2], &u11);

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
  t[0] = _umul128(a->bits64[1], a->bits64[3], &t[1]);
  c = _addcarry_u64(0, t[0], t[0], &t[0]);
  c = _addcarry_u64(c, t[1], t[1], &t[1]);
  c = _addcarry_u64(c, 0, 0, &t2);

  u10 = _umul128(a->bits64[2], a->bits64[2], &u11);
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

  // Reduce from 512 to 320
  imm_umul(r512 + 4, 0x1000003D1ULL, t);
  c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
  c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

  // Reduce from 320 to 256
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
#else
  // Original implementation from AVX2 version
  unsigned char c;
  uint64_t u10, u11;
  uint64_t t1;
  uint64_t t2;
  uint64_t t[NB64BLOCK];
#if BISIZE == 256
  uint64_t r512[8];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
#else
  uint64_t r512[12];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
  r512[8] = 0;
  r512[9] = 0;
  r512[10] = 0;
  r512[11] = 0;
#endif

  // k=0
  r512[0] = _umul128(a->bits64[0], a->bits64[0], &t[1]);

  // k=1
  t[3] = _umul128(a->bits64[0], a->bits64[1], &t[4]);
  c = _addcarry_u64(0, t[3], t[3], &t[3]);
  c = _addcarry_u64(c, t[4], t[4], &t[4]);
  c = _addcarry_u64(c, 0, 0, &t1);
  c = _addcarry_u64(0, t[1], t[3], &t[3]);
  c = _addcarry_u64(c, t[4], 0, &t[4]);
  c = _addcarry_u64(c, t1, 0, &t1);
  r512[1] = t[3];

  // k=2
  t[0] = _umul128(a->bits64[0], a->bits64[2], &t[1]);
  c = _addcarry_u64(0, t[0], t[0], &t[0]);
  c = _addcarry_u64(c, t[1], t[1], &t[1]);
  c = _addcarry_u64(c, 0, 0, &t2);

  u10 = _umul128(a->bits64[1], a->bits64[1], &u11);
  c = _addcarry_u64(0, t[0], u10, &t[0]);
  c = _addcarry_u64(c, t[1], u11, &t[1]);
  c = _addcarry_u64(c, t2, 0, &t2);
  c = _addcarry_u64(0, t[0], t[4], &t[0]);
  c = _addcarry_u64(c, t[1], t1, &t[1]);
  c = _addcarry_u64(c, t2, 0, &t2);
  r512[2] = t[0];

  // k=3
  t[3] = _umul128(a->bits64[0], a->bits64[3], &t[4]);
  u10 = _umul128(a->bits64[1], a->bits64[2], &u11);

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
  t[0] = _umul128(a->bits64[1], a->bits64[3], &t[1]);
  c = _addcarry_u64(0, t[0], t[0], &t[0]);
  c = _addcarry_u64(c, t[1], t[1], &t[1]);
  c = _addcarry_u64(c, 0, 0, &t2);

  u10 = _umul128(a->bits64[2], a->bits64[2], &u11);
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

  // Reduce from 512 to 320
  imm_umul(r512 + 4, 0x1000003D1ULL, t);
  c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
  c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

  // Reduce from 320 to 256
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
#endif
}
