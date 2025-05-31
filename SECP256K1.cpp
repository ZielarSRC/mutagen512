#include <immintrin.h>  // AVX-512 intrinsics
#include <omp.h>        // OpenMP for parallelization
#include <string.h>

#include "SECP256K1.h"

// Constructor with proper alignment for AVX-512
Secp256K1::Secp256K1() {}

void Secp256K1::Init() {
  // Prime for the finite field
  alignas(64) Int P;
  P.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

  // Set up field
  Int::SetupField(&P);

  // Generator point and order - aligned for AVX-512
  alignas(64) G.x.SetBase16("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
  alignas(64) G.y.SetBase16("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
  G.z.SetInt32(1);

  alignas(64) order.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

  Int::InitK1(&order);

  // Compute Generator table - parallelize using AVX-512
  Point N(G);

// Use AVX-512 optimization for parallel computation of GTable entries
#pragma omp parallel for
  for (int i = 0; i < 32; i++) {
    Point localN(N);

    // Compute initial doubling for this section using thread-local copy
    for (int d = 0; d < i; d++) {
      localN = DoubleDirect(localN);
    }

    // Store first point in this section
    GTable[i * 256] = localN;

    // Double for next point
    localN = DoubleDirect(localN);

// Compute all remaining points in this section using additions
#pragma omp simd
    for (int j = 1; j < 255; j++) {
      GTable[i * 256 + j] = localN;
      localN = AddDirect(localN, GTable[i * 256]);
    }

    // Store the last point
    GTable[i * 256 + 255] = localN;  // Dummy point for check function
  }
}

Secp256K1::~Secp256K1() {}

// Optimized point addition using AVX-512 acceleration from Int class
Point Secp256K1::AddDirect(Point &p1, Point &p2) {
  alignas(64) Int _s;
  alignas(64) Int _p;
  alignas(64) Int dy;
  alignas(64) Int dx;
  alignas(64) Point r;
  r.z.SetInt32(1);

// These operations can be done in parallel
#pragma omp parallel sections
  {
#pragma omp section
    { dy.ModSub(&p2.y, &p1.y); }

#pragma omp section
    { dx.ModSub(&p2.x, &p1.x); }
  }

  dx.ModInv();
  _s.ModMulK1(&dy, &dx);  // s = (p2.y-p1.y)*inverse(p2.x-p1.x);

  _p.ModSquareK1(&_s);  // _p = pow2(s)

  // These operations have dependencies but can use AVX-512 internally
  r.x.ModSub(&_p, &p1.x);
  r.x.ModSub(&p2.x);  // rx = pow2(s) - p1.x - p2.x;

  // These operations can use AVX-512 internally
  r.y.ModSub(&p2.x, &r.x);
  r.y.ModMulK1(&_s);
  r.y.ModSub(&p2.y);  // ry = - p2.y - s*(ret.x-p2.x);

  return r;
}

// Optimized point addition when P2.z = 1
Point Secp256K1::Add2(Point &p1, Point &p2) {
  // P2.z = 1
  alignas(64) Int u;
  alignas(64) Int v;
  alignas(64) Int u1;
  alignas(64) Int v1;
  alignas(64) Int vs2;
  alignas(64) Int vs3;
  alignas(64) Int us2;
  alignas(64) Int a;
  alignas(64) Int us2w;
  alignas(64) Int vs2v2;
  alignas(64) Int vs3u2;
  alignas(64) Int _2vs2v2;
  alignas(64) Point r;

// These operations can be parallelized with AVX-512
#pragma omp parallel sections
  {
#pragma omp section
    {
      u1.ModMulK1(&p2.y, &p1.z);
      u.ModSub(&u1, &p1.y);
    }

#pragma omp section
    {
      v1.ModMulK1(&p2.x, &p1.z);
      v.ModSub(&v1, &p1.x);
    }
  }

// These can be computed in parallel with AVX-512
#pragma omp parallel sections
  {
#pragma omp section
    {
      us2.ModSquareK1(&u);
      us2w.ModMulK1(&us2, &p1.z);
    }

#pragma omp section
    {
      vs2.ModSquareK1(&v);
      vs3.ModMulK1(&vs2, &v);
      vs2v2.ModMulK1(&vs2, &p1.x);
      _2vs2v2.ModAdd(&vs2v2, &vs2v2);
    }
  }

  // These operations have dependencies
  a.ModSub(&us2w, &vs3);
  a.ModSub(&_2vs2v2);

// These can be computed in parallel with AVX-512
#pragma omp parallel sections
  {
#pragma omp section
    { r.x.ModMulK1(&v, &a); }

#pragma omp section
    {
      vs3u2.ModMulK1(&vs3, &p1.y);
      r.y.ModSub(&vs2v2, &a);
      r.y.ModMulK1(&r.y, &u);
      r.y.ModSub(&vs3u2);
    }

#pragma omp section
    { r.z.ModMulK1(&vs3, &p1.z); }
  }

  return r;
}

// General point addition optimized for AVX-512
Point Secp256K1::Add(Point &p1, Point &p2) {
  alignas(64) Int u, v;
  alignas(64) Int u1, u2;
  alignas(64) Int v1, v2;
  alignas(64) Int vs2, vs3;
  alignas(64) Int us2, w;
  alignas(64) Int a, us2w;
  alignas(64) Int vs2v2, vs3u2;
  alignas(64) Int _2vs2v2, x3;
  alignas(64) Int vs3y1;
  alignas(64) Point r;

// Calculate intermediate values - can be parallelized with AVX-512
#pragma omp parallel sections
  {
#pragma omp section
    {
      u1.ModMulK1(&p2.y, &p1.z);
      u2.ModMulK1(&p1.y, &p2.z);
      u.ModSub(&u1, &u2);
    }

#pragma omp section
    {
      v1.ModMulK1(&p2.x, &p1.z);
      v2.ModMulK1(&p1.x, &p2.z);
      v.ModSub(&v1, &v2);
    }

#pragma omp section
    { w.ModMulK1(&p1.z, &p2.z); }
  }

  // Check for a point at infinity
  if (v1.IsEqual(&v2)) {     // Check if the X coordinates are equal
    if (!u1.IsEqual(&u2)) {  // If the Y-coordinates are different
      // Point at infinity
      r.x.SetInt32(0);
      r.y.SetInt32(0);
      r.z.SetInt32(0);
      return r;
    } else {
      // Doubling the point
      return Double(p1);  // Method for doubling a point
    }
  }

// Basic Dot Addition Calculations - use AVX-512 for these operations
#pragma omp parallel sections
  {
#pragma omp section
    {
      us2.ModSquareK1(&u);
      us2w.ModMulK1(&us2, &w);
    }

#pragma omp section
    {
      vs2.ModSquareK1(&v);
      vs3.ModMulK1(&vs2, &v);
      vs2v2.ModMulK1(&vs2, &v2);
      _2vs2v2.ModAdd(&vs2v2, &vs2v2);
    }
  }

  // These operations have dependencies
  a.ModSub(&us2w, &vs3);
  a.ModSub(&_2vs2v2);

// These can be computed in parallel with AVX-512
#pragma omp parallel sections
  {
#pragma omp section
    { r.x.ModMulK1(&v, &a); }

#pragma omp section
    {
      vs3u2.ModMulK1(&vs3, &u2);
      r.y.ModSub(&vs2v2, &a);
      r.y.ModMulK1(&r.y, &u);
      r.y.ModSub(&vs3u2);
    }

#pragma omp section
    { r.z.ModMulK1(&vs3, &w); }
  }

  return r;
}

// Point doubling optimized for AVX-512
Point Secp256K1::DoubleDirect(Point &p) {
  alignas(64) Int _s;
  alignas(64) Int _p;
  alignas(64) Int a;
  alignas(64) Point r;
  r.z.SetInt32(1);

  // These operations can be performed with AVX-512 acceleration
  _s.ModMulK1(&p.x, &p.x);
  _p.ModAdd(&_s, &_s);
  _p.ModAdd(&_s);

  a.ModAdd(&p.y, &p.y);
  a.ModInv();
  _s.ModMulK1(&_p, &a);  // s = (3*pow2(p.x))*inverse(2*p.y);

  // These operations can use AVX-512 internally
  _p.ModMulK1(&_s, &_s);
  a.ModAdd(&p.x, &p.x);
  a.ModNeg();
  r.x.ModAdd(&a, &_p);  // rx = pow2(s) + neg(2*p.x);

  a.ModSub(&r.x, &p.x);

  // These can be computed with AVX-512 acceleration
  _p.ModMulK1(&a, &_s);
  r.y.ModAdd(&_p, &p.y);
  r.y.ModNeg();  // ry = neg(p.y + s*(ret.x+neg(p.x)));

  return r;
}

// Compute public key optimized for AVX-512
Point Secp256K1::ComputePublicKey(Int *privKey) {
  int i = 0;
  uint8_t b;
  alignas(64) Point Q;
  Q.Clear();

  // Use AVX-512 SIMD operations to find the first significant byte faster
  __m512i zero = _mm512_setzero_si512();
  __m512i privKeyBytes[2];  // For 64 bytes (32 bytes of Int)

  // Load privKey bytes into AVX-512 registers
  for (int j = 0; j < 2; j++) {
    alignas(64) uint8_t bytes[64];
    for (int k = 0; k < 32 && (j * 32 + k < 32); k++) {
      bytes[k] = privKey->GetByte(j * 32 + k);
    }
    privKeyBytes[j] = _mm512_loadu_si512((__m512i *)bytes);
  }

  // Find first non-zero byte using AVX-512 mask comparison
  __mmask64 mask0 = _mm512_cmpneq_epu8_mask(privKeyBytes[0], zero);
  __mmask64 mask1 = _mm512_cmpneq_epu8_mask(privKeyBytes[1], zero);

  if (mask0) {
    i = _tzcnt_u64(mask0);
  } else if (mask1) {
    i = 32 + _tzcnt_u64(mask1);
  } else {
    i = 32;  // All zeros
  }

  // Process the bytes as before but with optimized Int operations
  if (i < 32) {
    b = privKey->GetByte(i);
    if (b) Q = GTable[256 * i + (b - 1)];
    i++;

    for (; i < 32; i++) {
      b = privKey->GetByte(i);
      if (b) Q = Add2(Q, GTable[256 * i + (b - 1)]);
    }
  }

  Q.Reduce();
  return Q;
}

// Advanced point doubling optimized for AVX-512
Point Secp256K1::Double(Point &p) {
  /*
  if (Y == 0)
    return POINT_AT_INFINITY
    W = a * Z ^ 2 + 3 * X ^ 2
    S = Y * Z
    B = X * Y*S
    H = W ^ 2 - 8 * B
    X' = 2*H*S
    Y' = W*(4*B - H) - 8*Y^2*S^2
    Z' = 8*S^3
    return (X', Y', Z')
  */

  alignas(64) Int z2;
  alignas(64) Int x2;
  alignas(64) Int _3x2;
  alignas(64) Int w;
  alignas(64) Int s;
  alignas(64) Int s2;
  alignas(64) Int b;
  alignas(64) Int _8b;
  alignas(64) Int _8y2s2;
  alignas(64) Int y2;
  alignas(64) Int h;
  alignas(64) Point r;

// These operations can be parallelized with AVX-512
#pragma omp parallel sections
  {
#pragma omp section
    {
      z2.ModSquareK1(&p.z);
      z2.SetInt32(0);  // a=0
    }

#pragma omp section
    {
      x2.ModSquareK1(&p.x);
      _3x2.ModAdd(&x2, &x2);
      _3x2.ModAdd(&x2);
    }

#pragma omp section
    { s.ModMulK1(&p.y, &p.z); }
  }

  w.ModAdd(&z2, &_3x2);
  b.ModMulK1(&p.y, &s);
  b.ModMulK1(&p.x);
  h.ModSquareK1(&w);
  _8b.ModAdd(&b, &b);
  _8b.ModDouble();
  _8b.ModDouble();
  h.ModSub(&_8b);

// These operations can be parallelized with AVX-512
#pragma omp parallel sections
  {
#pragma omp section
    {
      r.x.ModMulK1(&h, &s);
      r.x.ModAdd(&r.x);
    }

#pragma omp section
    {
      s2.ModSquareK1(&s);
      y2.ModSquareK1(&p.y);
      _8y2s2.ModMulK1(&y2, &s2);
      _8y2s2.ModDouble();
      _8y2s2.ModDouble();
      _8y2s2.ModDouble();
    }

#pragma omp section
    {
      r.z.ModMulK1(&s2, &s);
      r.z.ModDouble();
      r.z.ModDouble();
      r.z.ModDouble();
    }
  }

  r.y.ModAdd(&b, &b);
  r.y.ModAdd(&r.y, &r.y);
  r.y.ModSub(&h);
  r.y.ModMulK1(&w);
  r.y.ModSub(&_8y2s2);

  return r;
}

// Get Y coordinate from X - optimized for AVX-512
Int Secp256K1::GetY(Int x, bool isEven) {
  alignas(64) Int _s;
  alignas(64) Int _p;

  _s.ModSquareK1(&x);
  _p.ModMulK1(&_s, &x);
  _p.ModAdd(7);
  _p.ModSqrt();

  if (!_p.IsEven() && isEven) {
    _p.ModNeg();
  } else if (_p.IsEven() && !isEven) {
    _p.ModNeg();
  }

  return _p;
}

// Verify point is on curve - optimized for AVX-512
bool Secp256K1::EC(Point &p) {
  alignas(64) Int _s;
  alignas(64) Int _p;

// These operations can be parallelized with AVX-512
#pragma omp parallel sections
  {
#pragma omp section
    {
      _s.ModSquareK1(&p.x);
      _p.ModMulK1(&_s, &p.x);
      _p.ModAdd(7);
    }

#pragma omp section
    { _s.ModMulK1(&p.y, &p.y); }
  }

  _s.ModSub(&_p);

  return _s.IsZero();  // ( ((pow2(y) - (pow3(x) + 7)) % P) == 0 );
}

// New batch methods optimized for AVX-512

// Batch compute public keys for multiple private keys at once
void Secp256K1::BatchComputePublicKeys(Int *privKeys, Point *pubKeys, int count) {
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    pubKeys[i] = ComputePublicKey(&privKeys[i]);
  }
}

// Batch point addition
void Secp256K1::BatchAddPoints(Point *p1, Point *p2, Point *result, int count) {
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    result[i] = Add(p1[i], p2[i]);
  }
}

// Batch point doubling
void Secp256K1::BatchDoublePoints(Point *points, Point *result, int count) {
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    result[i] = Double(points[i]);
  }
}

// Batch verify points are on curve
void Secp256K1::BatchEC(Point *points, bool *results, int count) {
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    results[i] = EC(points[i]);
  }
}
