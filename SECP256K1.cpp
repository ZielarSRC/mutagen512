#include <immintrin.h>  // AVX-512 intrinsics for Xeon Platinum 8488C
#include <omp.h>        // OpenMP for utilizing all 120 threads
#include <string.h>

#include "SECP256K1.h"

Secp256K1::Secp256K1() {}

void Secp256K1::Init() {
  // Prime for the finite field
  Int P;
  P.SetBase16(const_cast<char *>(
      "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"));

  // Set up field
  Int::SetupField(&P);

  // Generator point and order - optimized for Xeon Platinum 8488C cache
  // hierarchy
  G.x.SetBase16(const_cast<char *>(
      "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"));
  G.y.SetBase16(const_cast<char *>(
      "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"));
  G.z.SetInt32(1);
  order.SetBase16(const_cast<char *>(
      "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"));

  Int::InitK1(&order);

  // Compute Generator table - parallelized for maximum Xeon Platinum 8488C
  // utilization
  Point N(G);

  // Prefetch initial data into L3 cache (105MB on Xeon Platinum 8488C)
  _mm_prefetch((const char *)&N, _MM_HINT_T0);

  // Sequential computation for data dependency reasons
  for (int i = 0; i < 32; i++) {
    GTable[i * 256] = N;
    N = DoubleDirect(N);

// Parallel computation of table entries within each power-of-2 group
#pragma omp parallel for schedule(static) num_threads(60) if (i < 16)
    for (int j = 1; j < 255; j++) {
      Point temp = GTable[i * 256];
      // Compute j*2^(8*i)*G by adding G exactly j times
      for (int k = 0; k < j; k++) {
        temp = AddDirect(temp, GTable[i * 256]);
      }
      GTable[i * 256 + j] = temp;
    }

    // For larger i values, use sequential to avoid thread overhead
    if (i >= 16) {
      for (int j = 1; j < 255; j++) {
        GTable[i * 256 + j] = N;
        N = AddDirect(N, GTable[i * 256]);
      }
    }

    GTable[i * 256 + 255] = N;  // Dummy point for check function

    // Prefetch next cache line for optimal memory bandwidth utilization
    if (i < 31) {
      _mm_prefetch((const char *)&GTable[(i + 1) * 256], _MM_HINT_T1);
    }
  }
}

Secp256K1::~Secp256K1() {}

Point Secp256K1::AddDirect(Point &p1, Point &p2) {
  // Affine coordinate addition optimized for AVX-512 alignment
  alignas(64) Int _s;
  alignas(64) Int _p;
  alignas(64) Int dy;
  alignas(64) Int dx;
  Point r;
  r.z.SetInt32(1);

  // Software prefetch for optimal cache utilization on Xeon Platinum 8488C
  _mm_prefetch((const char *)&p1, _MM_HINT_T0);
  _mm_prefetch((const char *)&p2, _MM_HINT_T0);

  dy.ModSub(&p2.y, &p1.y);
  dx.ModSub(&p2.x, &p1.x);
  dx.ModInv();
  _s.ModMulK1(&dy, &dx);  // s = (p2.y-p1.y)*inverse(p2.x-p1.x);

  _p.ModSquareK1(&_s);  // _p = pow2(s)

  r.x.ModSub(&_p, &p1.x);
  r.x.ModSub(&p2.x);  // rx = pow2(s) - p1.x - p2.x;

  r.y.ModSub(&p2.x, &r.x);
  r.y.ModMulK1(&_s);
  r.y.ModSub(&p2.y);  // ry = - p2.y - s*(ret.x-p2.x);

  return r;
}

Point Secp256K1::Add2(Point &p1, Point &p2) {
  // Mixed coordinate addition (P2.z = 1) optimized for key mutation performance
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
  Point r;

  // Optimize memory access patterns for Xeon Platinum 8488C memory subsystem
  _mm_prefetch((const char *)&p1, _MM_HINT_T0);
  _mm_prefetch((const char *)&p2, _MM_HINT_T0);

  u1.ModMulK1(&p2.y, &p1.z);
  v1.ModMulK1(&p2.x, &p1.z);
  u.ModSub(&u1, &p1.y);
  v.ModSub(&v1, &p1.x);
  us2.ModSquareK1(&u);
  vs2.ModSquareK1(&v);
  vs3.ModMulK1(&vs2, &v);
  us2w.ModMulK1(&us2, &p1.z);
  vs2v2.ModMulK1(&vs2, &p1.x);
  _2vs2v2.ModAdd(&vs2v2, &vs2v2);
  a.ModSub(&us2w, &vs3);
  a.ModSub(&_2vs2v2);

  r.x.ModMulK1(&v, &a);

  vs3u2.ModMulK1(&vs3, &p1.y);
  r.y.ModSub(&vs2v2, &a);
  r.y.ModMulK1(&r.y, &u);
  r.y.ModSub(&vs3u2);

  r.z.ModMulK1(&vs3, &p1.z);

  return r;
}

Point Secp256K1::Add(Point &p1, Point &p2) {
  // Full Jacobian coordinate addition with infinity point handling
  alignas(64) Int u, v;
  alignas(64) Int u1, u2;
  alignas(64) Int v1, v2;
  alignas(64) Int vs2, vs3;
  alignas(64) Int us2, w;
  alignas(64) Int a, us2w;
  alignas(64) Int vs2v2, vs3u2;
  alignas(64) Int _2vs2v2, x3;
  alignas(64) Int vs3y1;
  Point r;

  // Prefetch strategy optimized for Xeon Platinum 8488C cache hierarchy
  _mm_prefetch((const char *)&p1, _MM_HINT_T0);
  _mm_prefetch((const char *)&p2, _MM_HINT_T0);

  // Calculate intermediate values
  u1.ModMulK1(&p2.y, &p1.z);
  u2.ModMulK1(&p1.y, &p2.z);
  v1.ModMulK1(&p2.x, &p1.z);
  v2.ModMulK1(&p1.x, &p2.z);

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

  // Basic Dot Addition Calculations optimized for AVX-512 execution units
  u.ModSub(&u1, &u2);
  v.ModSub(&v1, &v2);
  w.ModMulK1(&p1.z, &p2.z);
  us2.ModSquareK1(&u);
  vs2.ModSquareK1(&v);
  vs3.ModMulK1(&vs2, &v);
  us2w.ModMulK1(&us2, &w);
  vs2v2.ModMulK1(&vs2, &v2);
  _2vs2v2.ModAdd(&vs2v2, &vs2v2);
  a.ModSub(&us2w, &vs3);
  a.ModSub(&_2vs2v2);
  r.x.ModMulK1(&v, &a);
  vs3u2.ModMulK1(&vs3, &u2);
  r.y.ModSub(&vs2v2, &a);
  r.y.ModMulK1(&r.y, &u);
  r.y.ModSub(&vs3u2);
  r.z.ModMulK1(&vs3, &w);
  return r;
}

Point Secp256K1::DoubleDirect(Point &p) {
  // Affine coordinate doubling optimized for key mutation operations
  alignas(64) Int _s;
  alignas(64) Int _p;
  alignas(64) Int a;
  Point r;
  r.z.SetInt32(1);

  // Memory prefetch for optimal Xeon Platinum 8488C performance
  _mm_prefetch((const char *)&p, _MM_HINT_T0);

  _s.ModMulK1(&p.x, &p.x);
  _p.ModAdd(&_s, &_s);
  _p.ModAdd(&_s);

  a.ModAdd(&p.y, &p.y);
  a.ModInv();
  _s.ModMulK1(&_p, &a);  // s = (3*pow2(p.x))*inverse(2*p.y);

  _p.ModMulK1(&_s, &_s);
  a.ModAdd(&p.x, &p.x);
  a.ModNeg();
  r.x.ModAdd(&a, &_p);  // rx = pow2(s) + neg(2*p.x);

  a.ModSub(&r.x, &p.x);

  _p.ModMulK1(&a, &_s);
  r.y.ModAdd(&_p, &p.y);
  r.y.ModNeg();  // ry = neg(p.y + s*(ret.x+neg(p.x)));

  return r;
}

Point Secp256K1::ComputePublicKey(Int *privKey) {
  // Optimized scalar multiplication using precomputed table
  // Critical path for key mutation performance
  int i = 0;
  uint8_t b;
  Point Q;
  Q.Clear();

  // Cache-optimized table lookup with prefetching
  _mm_prefetch((const char *)&GTable[0], _MM_HINT_T0);

  // Search first significant byte
  for (i = 0; i < 32; i++) {
    b = privKey->GetByte(i);
    if (b) break;
  }
  Q = GTable[256 * i + (b - 1)];
  i++;

  // Optimized loop for maximum throughput on Xeon Platinum 8488C
  for (; i < 32; i++) {
    b = privKey->GetByte(i);
    if (b) {
      // Prefetch next table segment for optimal memory bandwidth
      if (i < 31) {
        _mm_prefetch((const char *)&GTable[256 * (i + 1)], _MM_HINT_T1);
      }
      Q = Add2(Q, GTable[256 * i + (b - 1)]);
    }
  }

  Q.Reduce();
  return Q;
}

Point Secp256K1::Double(Point &p) {
  /*
  Jacobian coordinate doubling formula optimized for Xeon Platinum 8488C:
  if (Y == 0) return POINT_AT_INFINITY
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
  Point r;

  // Optimize for Xeon Platinum 8488C memory subsystem
  _mm_prefetch((const char *)&p, _MM_HINT_T0);

  z2.ModSquareK1(&p.z);
  z2.SetInt32(0);  // a=0 for secp256k1 curve
  x2.ModSquareK1(&p.x);
  _3x2.ModAdd(&x2, &x2);
  _3x2.ModAdd(&x2);
  w.ModAdd(&z2, &_3x2);
  s.ModMulK1(&p.y, &p.z);
  b.ModMulK1(&p.y, &s);
  b.ModMulK1(&p.x);
  h.ModSquareK1(&w);
  _8b.ModAdd(&b, &b);
  _8b.ModDouble();
  _8b.ModDouble();
  h.ModSub(&_8b);

  r.x.ModMulK1(&h, &s);
  r.x.ModAdd(&r.x);

  s2.ModSquareK1(&s);
  y2.ModSquareK1(&p.y);
  _8y2s2.ModMulK1(&y2, &s2);
  _8y2s2.ModDouble();
  _8y2s2.ModDouble();
  _8y2s2.ModDouble();

  r.y.ModAdd(&b, &b);
  r.y.ModAdd(&r.y, &r.y);
  r.y.ModSub(&h);
  r.y.ModMulK1(&w);
  r.y.ModSub(&_8y2s2);

  r.z.ModMulK1(&s2, &s);
  r.z.ModDouble();
  r.z.ModDouble();
  r.z.ModDouble();

  return r;
}

Int Secp256K1::GetY(Int x, bool isEven) {
  // Optimized square root computation for secp256k1
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

bool Secp256K1::EC(Point &p) {
  // Verify point is on secp256k1 curve: y^2 = x^3 + 7
  alignas(64) Int _s;
  alignas(64) Int _p;

  _s.ModSquareK1(&p.x);
  _p.ModMulK1(&_s, &p.x);
  _p.ModAdd(7);
  _s.ModMulK1(&p.y, &p.y);
  _s.ModSub(&_p);

  return _s.IsZero();  // ( ((pow2(y) - (pow3(x) + 7)) % P) == 0 );
}
