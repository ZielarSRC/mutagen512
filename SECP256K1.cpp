#include <string.h>
#include <immintrin.h>

#include "SECP256K1.h"

Secp256K1::Secp256K1() {}

void Secp256K1::Init() {
  // Prime for the finite field
  Int P;
  P.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

  // Set up field
  Int::SetupField(&P);

  // Generator point and order
  G.x.SetBase16("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
  G.y.SetBase16("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
  G.z.SetInt32(1);
  order.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

  Int::InitK1(&order);

  // Compute Generator table with AVX-512 optimizations
  Point N(G);
  for (int i = 0; i < 32; i++) {
    // Prefetch next table entries for better cache performance
    if (i < 31) {
      _mm_prefetch((char*)&GTable[(i + 1) * 256], _MM_HINT_T1);
    }
    
    GTable[i * 256] = N;
    N = DoubleDirect(N);
    
    for (int j = 1; j < 255; j++) {
      // Prefetch ahead for cache optimization
      if (j % 8 == 0 && j + 8 < 255) {
        _mm_prefetch((char*)&GTable[i * 256 + j + 8], _MM_HINT_T0);
      }
      
      GTable[i * 256 + j] = N;
      N = AddDirect(N, GTable[i * 256]);
      
      // Apply fast reduction periodically for better performance
      N.x.ModReduceK1AVX512();
      N.y.ModReduceK1AVX512();
    }
    GTable[i * 256 + 255] = N;  // Dummy point for check function
  }
  
  // Memory fence to ensure table initialization is complete
  _mm_mfence();
}

Secp256K1::~Secp256K1() {}

Point Secp256K1::AddDirect(Point &p1, Point &p2) {
  Int _s;
  Int _p;
  Int dy;
  Int dx;
  Point r;
  r.z.SetInt32(1);

  // Prefetch input data for better cache performance
  _mm_prefetch((char*)p1.x.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p1.y.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p2.x.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p2.y.bits64, _MM_HINT_T0);

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

  // Apply fast reduction for secp256k1
  r.x.ModReduceK1AVX512();
  r.y.ModReduceK1AVX512();

  return r;
}

Point Secp256K1::Add2(Point &p1, Point &p2) {
  // P2.z = 1

  Int u;
  Int v;
  Int u1;
  Int v1;
  Int vs2;
  Int vs3;
  Int us2;
  Int a;
  Int us2w;
  Int vs2v2;
  Int vs3u2;
  Int _2vs2v2;
  Point r;

  // Prefetch input data
  _mm_prefetch((char*)p1.x.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p1.y.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p1.z.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p2.x.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p2.y.bits64, _MM_HINT_T0);

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

  // Apply fast reduction for secp256k1
  r.x.ModReduceK1AVX512();
  r.y.ModReduceK1AVX512();
  r.z.ModReduceK1AVX512();

  return r;
}

Point Secp256K1::Add(Point &p1, Point &p2) {
  Int u, v;
  Int u1, u2;
  Int v1, v2;
  Int vs2, vs3;
  Int us2, w;
  Int a, us2w;
  Int vs2v2, vs3u2;
  Int _2vs2v2, x3;
  Int vs3y1;
  Point r;

  // Prefetch input data for optimal cache performance
  _mm_prefetch((char*)p1.x.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p1.y.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p1.z.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p2.x.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p2.y.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p2.z.bits64, _MM_HINT_T0);

  // Calculate intermediate values
  u1.ModMulK1(&p2.y, &p1.z);
  u2.ModMulK1(&p1.y, &p2.z);
  v1.ModMulK1(&p2.x, &p1.z);
  v2.ModMulK1(&p1.x, &p2.z);

  // Check for a point at infinity
  if (v1.IsEqual(&v2)) {  // Check if the X coordinates are equal
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

  // Basic Dot Addition Calculations
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

  // Apply fast reduction for secp256k1
  r.x.ModReduceK1AVX512();
  r.y.ModReduceK1AVX512();
  r.z.ModReduceK1AVX512();

  return r;
}

Point Secp256K1::DoubleDirect(Point &p) {
  Int _s;
  Int _p;
  Int a;
  Point r;
  r.z.SetInt32(1);

  // Prefetch input data
  _mm_prefetch((char*)p.x.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p.y.bits64, _MM_HINT_T0);

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

  // Apply fast reduction for secp256k1
  r.x.ModReduceK1AVX512();
  r.y.ModReduceK1AVX512();

  return r;
}

Point Secp256K1::ComputePublicKey(Int *privKey) {
  int i = 0;
  uint8_t b;
  Point Q;
  Q.Clear();

  // Prefetch private key data
  _mm_prefetch((char*)privKey->bits64, _MM_HINT_T0);

  // Search first significant byte
  for (i = 0; i < 32; i++) {
    b = privKey->GetByte(i);
    if (b) break;
  }
  
  // Prefetch first table entry
  _mm_prefetch((char*)&GTable[256 * i + (b - 1)], _MM_HINT_T0);
  Q = GTable[256 * i + (b - 1)];
  i++;

  for (; i < 32; i++) {
    b = privKey->GetByte(i);
    if (b) {
      // Prefetch next table entry
      _mm_prefetch((char*)&GTable[256 * i + (b - 1)], _MM_HINT_T0);
      Q = Add2(Q, GTable[256 * i + (b - 1)]);
      
      // Apply periodic reduction for better performance
      Q.x.ModReduceK1AVX512();
      Q.y.ModReduceK1AVX512();
      Q.z.ModReduceK1AVX512();
    }
  }

  Q.Reduce();
  return Q;
}

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

  Int z2;
  Int x2;
  Int _3x2;
  Int w;
  Int s;
  Int s2;
  Int b;
  Int _8b;
  Int _8y2s2;
  Int y2;
  Int h;
  Point r;

  // Prefetch input point data
  _mm_prefetch((char*)p.x.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p.y.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p.z.bits64, _MM_HINT_T0);

  z2.ModSquareK1(&p.z);
  z2.SetInt32(0);  // a=0
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

  // Apply fast reduction for secp256k1
  r.x.ModReduceK1AVX512();
  r.y.ModReduceK1AVX512();
  r.z.ModReduceK1AVX512();

  return r;
}

Int Secp256K1::GetY(Int x, bool isEven) {
  Int _s;
  Int _p;

  // Prefetch input data
  _mm_prefetch((char*)x.bits64, _MM_HINT_T0);

  _s.ModSquareK1(&x);
  _p.ModMulK1(&_s, &x);
  _p.ModAdd(7);
  _p.ModSqrt();

  if (!_p.IsEven() && isEven) {
    _p.ModNeg();
  } else if (_p.IsEven() && !isEven) {
    _p.ModNeg();
  }

  // Apply fast reduction
  _p.ModReduceK1AVX512();

  return _p;
}

bool Secp256K1::EC(Point &p) {
  Int _s;
  Int _p;

  // Prefetch point data
  _mm_prefetch((char*)p.x.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p.y.bits64, _MM_HINT_T0);

  _s.ModSquareK1(&p.x);
  _p.ModMulK1(&_s, &p.x);
  _p.ModAdd(7);
  _s.ModMulK1(&p.y, &p.y);
  _s.ModSub(&_p);

  return _s.IsZero();  // ( ((pow2(y) - (pow3(x) + 7)) % P) == 0 );
}
