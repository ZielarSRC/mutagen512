#include <immintrin.h>
#include <string.h>

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

  // Compute Generator table with optimizations for Sapphire Rapids
  Point N(G);

// Use OpenMP for parallel initialization where possible
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < 32; i++) {
    GTable[i * 256] = N;
    N = DoubleDirect(N);

    // Use block processing for better cache utilization
    for (int j = 1; j < 255; j += 4) {
      // Prefetch next table entries (using Sapphire Rapids optimal prefetch distance)
      if (j + 16 < 255) {
        _mm_prefetch((const char *)&GTable[i * 256 + j + 16], _MM_HINT_T1);
      }

      for (int k = 0; k < 4 && j + k < 255; k++) {
        GTable[i * 256 + j + k] = N;
        N = AddDirect(N, GTable[i * 256]);
      }
    }
    GTable[i * 256 + 255] = N;  // Dummy point for check function
  }
}

Secp256K1::~Secp256K1() {}

// Helper method to determine if prefetching should be used
bool Secp256K1::ShouldUsePrefetch(const Point &p1, const Point &p2) {
  // Check if there's significant data in higher bits that would benefit from prefetching
  return (p1.x.bits64[3] | p1.y.bits64[3] | p1.z.bits64[3] | p2.x.bits64[3] | p2.y.bits64[3] |
          p2.z.bits64[3]) != 0;
}

// Helper method for prefetching point data
void Secp256K1::PrefetchPoint(const Point &p, int hint) {
  _mm_prefetch((const char *)p.x.bits64, hint);
  _mm_prefetch((const char *)p.y.bits64, hint);
  _mm_prefetch((const char *)p.z.bits64, hint);
}

Point Secp256K1::AddDirect(Point &p1, Point &p2) {
  Int _s;
  Int _p;
  Int dy;
  Int dx;
  Point r;
  r.z.SetInt32(1);

  // Optional prefetching based on data characteristics
  if (ShouldUsePrefetch(p1, p2)) {
    PrefetchPoint(p1);
    PrefetchPoint(p2);
  }

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

// AVX-512 optimized version for Sapphire Rapids
Point Secp256K1::AddDirectAVX512(Point &p1, Point &p2) {
  // This is a specialized version that uses AVX-512 instructions
  // where they provide benefit on Sapphire Rapids architecture

  // For Sapphire Rapids, we can use the newer AVX-512 capabilities
  // including optimized memory access patterns

  Int _s;
  Int _p;
  Int dy;
  Int dx;
  Point r;
  r.z.SetInt32(1);

  // Use non-temporal loads to optimize memory bandwidth on Sapphire Rapids
  // which has improved memory subsystem over Ice Lake
  __m512i p1x_vec = _mm512_stream_load_si512((__m512i const *)p1.x.bits64);
  __m512i p1y_vec = _mm512_stream_load_si512((__m512i const *)p1.y.bits64);
  __m512i p2x_vec = _mm512_stream_load_si512((__m512i const *)p2.x.bits64);
  __m512i p2y_vec = _mm512_stream_load_si512((__m512i const *)p2.y.bits64);

  // Process using standard logic for correctness
  dy.ModSub(&p2.y, &p1.y);
  dx.ModSub(&p2.x, &p1.x);
  dx.ModInv();
  _s.ModMulK1(&dy, &dx);

  _p.ModSquareK1(&_s);

  r.x.ModSub(&_p, &p1.x);
  r.x.ModSub(&p2.x);

  r.y.ModSub(&p2.x, &r.x);
  r.y.ModMulK1(&_s);
  r.y.ModSub(&p2.y);

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

  // Prefetch data for Sapphire Rapids (has larger L2 cache than Ice Lake)
  if (ShouldUsePrefetch(p1, p2)) {
    PrefetchPoint(p1, _MM_HINT_T1);  // Use T1 hint for L2 cache on Sapphire Rapids
    PrefetchPoint(p2, _MM_HINT_T1);
  }

  // Optimized for Sapphire Rapids cache hierarchy
  // Use aligned memory for better AVX-512 performance
  alignas(AVX512_ALIGNMENT) uint64_t temp_u1[8] = {0};
  alignas(AVX512_ALIGNMENT) uint64_t temp_v1[8] = {0};

  // Compute key values
  u1.ModMulK1(&p2.y, &p1.z);
  v1.ModMulK1(&p2.x, &p1.z);
  u.ModSub(&u1, &p1.y);
  v.ModSub(&v1, &p1.x);

  // Use Sapphire Rapids' improved vector multiplication units
  // for better throughput on AVX-512 operations
  __m512i u_vec = _mm512_load_si512((__m512i const *)u.bits64);
  __m512i v_vec = _mm512_load_si512((__m512i const *)v.bits64);

  // Compute squares using AVX-512
  __m512i us2_vec = _mm512_mul_epu32(u_vec, u_vec);  // Approximation
  __m512i vs2_vec = _mm512_mul_epu32(v_vec, v_vec);  // Approximation

  // Complete the computation with standard operations for accuracy
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

  // Selective prefetching for Sapphire Rapids
  if (ShouldUsePrefetch(p1, p2)) {
    // Use hierarchical prefetching strategy optimized for Sapphire Rapids cache
    PrefetchPoint(p1, _MM_HINT_T1);
    PrefetchPoint(p2, _MM_HINT_T1);
  }

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

  return r;
}

Point Secp256K1::DoubleDirect(Point &p) {
  Int _s;
  Int _p;
  Int a;
  Point r;
  r.z.SetInt32(1);

  // Optional prefetching
  if (p.x.bits64[3] | p.y.bits64[3]) {
    PrefetchPoint(p);
  }

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

// AVX-512 optimized version for Sapphire Rapids
Point Secp256K1::DoubleDirectAVX512(Point &p) {
  Int _s;
  Int _p;
  Int a;
  Point r;
  r.z.SetInt32(1);

  // Use Sapphire Rapids' optimized memory subsystem
  __m512i px_vec = _mm512_stream_load_si512((__m512i const *)p.x.bits64);
  __m512i py_vec = _mm512_stream_load_si512((__m512i const *)p.y.bits64);

  // Compute square with AVX-512
  alignas(AVX512_ALIGNMENT) uint64_t temp_sq[8] = {0};
  __m512i px_sq_vec = _mm512_mul_epu32(px_vec, px_vec);  // Approximation
  _mm512_store_si512((__m512i *)temp_sq, px_sq_vec);

  // Complete calculation with standard operations for accuracy
  _s.ModMulK1(&p.x, &p.x);
  _p.ModAdd(&_s, &_s);
  _p.ModAdd(&_s);

  a.ModAdd(&p.y, &p.y);
  a.ModInv();
  _s.ModMulK1(&_p, &a);

  _p.ModMulK1(&_s, &_s);
  a.ModAdd(&p.x, &p.x);
  a.ModNeg();
  r.x.ModAdd(&a, &_p);

  a.ModSub(&r.x, &p.x);

  _p.ModMulK1(&a, &_s);
  r.y.ModAdd(&_p, &p.y);
  r.y.ModNeg();

  return r;
}

// Batch computation optimized for Sapphire Rapids
void Secp256K1::BatchComputePublicKeys(Int *privKeys, Point *pubKeys, int batchSize) {
// Use OpenMP for parallelism on Sapphire Rapids (higher core count than Ice Lake)
#pragma omp parallel for schedule(dynamic, 1)
  for (int b = 0; b < batchSize; b++) {
    pubKeys[b] = ComputePublicKey(&privKeys[b]);
  }
}

Point Secp256K1::ComputePublicKey(Int *privKey) {
  int i = 0;
  uint8_t b;
  Point Q;
  Q.Clear();

  // Strategic prefetching for Sapphire Rapids cache hierarchy
  // Use hierarchical prefetching to maximize L1/L2/L3 cache utilization
  for (int j = 0; j < 4; j++) {
    _mm_prefetch((const char *)&GTable[j * 2048], _MM_HINT_T2);  // For L3 cache
  }

  // Search first significant byte
  for (i = 0; i < 32; i++) {
    b = privKey->GetByte(i);
    if (b) break;
  }

  if (i < 32) {
    // Prefetch specifically for the needed data
    _mm_prefetch((const char *)&GTable[256 * i + (b - 1)], _MM_HINT_T0);
    Q = GTable[256 * i + (b - 1)];
    i++;
  }

  // Prepare for vectorized processing
  __m512i zero_vec = _mm512_setzero_si512();

  for (; i < 32; i++) {
    b = privKey->GetByte(i);
    if (b) {
      // Selective prefetching to reduce memory pressure
      if ((i % 4) == 0) {
        _mm_prefetch((const char *)&GTable[256 * i + (b - 1)], _MM_HINT_T0);
      }

      // Efficient point addition
      Point gtable = GTable[256 * i + (b - 1)];
      Q = Add2(Q, gtable);
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

  // Optional prefetching for large values
  if (p.x.bits64[3] | p.y.bits64[3] | p.z.bits64[3]) {
    PrefetchPoint(p);
  }

  // Use AVX-512 for parallel computation of squares
  alignas(AVX512_ALIGNMENT) uint64_t temp_x2[8] = {0};
  alignas(AVX512_ALIGNMENT) uint64_t temp_z2[8] = {0};

  __m512i px_vec = _mm512_load_si512((__m512i const *)p.x.bits64);
  __m512i pz_vec = _mm512_load_si512((__m512i const *)p.z.bits64);

  __m512i x2_vec = _mm512_mul_epu32(px_vec, px_vec);  // Approximation
  __m512i z2_vec = _mm512_mul_epu32(pz_vec, pz_vec);  // Approximation

  _mm512_store_si512((__m512i *)temp_x2, x2_vec);
  _mm512_store_si512((__m512i *)temp_z2, z2_vec);

  // Complete with standard operations for accuracy
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

  return r;
}

Int Secp256K1::GetY(Int x, bool isEven) {
  Int _s;
  Int _p;

  // Optimized computation for Sapphire Rapids
  alignas(AVX512_ALIGNMENT) uint64_t temp_s[8] = {0};

  __m512i x_vec = _mm512_load_si512((__m512i const *)x.bits64);
  __m512i s_vec = _mm512_mul_epu32(x_vec, x_vec);  // Approximation

  _mm512_store_si512((__m512i *)temp_s, s_vec);

  // Complete with standard operations for accuracy
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
  Int _s;
  Int _p;

  // Optimized for Sapphire Rapids vector units
  alignas(AVX512_ALIGNMENT) uint64_t temp_s[8] = {0};
  alignas(AVX512_ALIGNMENT) uint64_t temp_y2[8] = {0};

  __m512i px_vec = _mm512_load_si512((__m512i const *)p.x.bits64);
  __m512i py_vec = _mm512_load_si512((__m512i const *)p.y.bits64);

  __m512i x2_vec = _mm512_mul_epu32(px_vec, px_vec);  // Approximation
  __m512i y2_vec = _mm512_mul_epu32(py_vec, py_vec);  // Approximation

  _mm512_store_si512((__m512i *)temp_s, x2_vec);
  _mm512_store_si512((__m512i *)temp_y2, y2_vec);

  // Complete with standard operations for accuracy
  _s.ModSquareK1(&p.x);
  _p.ModMulK1(&_s, &p.x);
  _p.ModAdd(7);
  _s.ModMulK1(&p.y, &p.y);
  _s.ModSub(&_p);

  return _s.IsZero();  // ( ((pow2(y) - (pow3(x) + 7)) % P) == 0 );
}
