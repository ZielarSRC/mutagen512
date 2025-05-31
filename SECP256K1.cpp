#include <immintrin.h>  // Dla instrukcji AVX-512
#include <omp.h>        // Dla OpenMP
#include <string.h>

#include "SECP256K1.h"

Secp256K1::Secp256K1() {}

void Secp256K1::Init() {
  // Prime for the finite field
  alignas(64) Int P;
  P.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

  // Set up field
  Int::SetupField(&P);

  // Generator point and order
  // Deklaracja zmiennych tymczasowych z wyrównaniem 64 bajtów dla AVX-512
  alignas(64) Int x_aligned, y_aligned, order_aligned;

  // Inicjalizacja zmiennych tymczasowych
  x_aligned.SetBase16("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
  y_aligned.SetBase16("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
  order_aligned.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

  // Prefetching dla lepszej wydajności pamięci
  _mm_prefetch((const char *)&x_aligned, _MM_HINT_T0);
  _mm_prefetch((const char *)&y_aligned, _MM_HINT_T0);

  // Przypisanie do właściwych zmiennych
  G.x = x_aligned;
  G.y = y_aligned;
  G.z.SetInt32(1);
  order = order_aligned;

  Int::InitK1(&order);

  // Compute Generator table - wykorzystanie OpenMP dla równoległego przetwarzania
  Point N(G);
  GTable[0] = N;

#pragma omp parallel
  {
#pragma omp single
    {
      for (int i = 0; i < 32; i++) {
#pragma omp task firstprivate(i)
        {
          Point localN;
          if (i == 0) {
            localN = N;
          } else {
            localN = GTable[i * 256];
            localN = DoubleDirect(localN);
          }

          GTable[i * 256] = localN;

          // Prefetch dla następnych iteracji
          if (i < 31) {
            _mm_prefetch((const char *)&GTable[(i + 1) * 256], _MM_HINT_T0);
          }

          for (int j = 1; j < 255; j++) {
            Point add = AddDirect(localN, GTable[i * 256]);
            GTable[i * 256 + j] = add;
            localN = add;
          }
          GTable[i * 256 + 255] = localN;  // Dummy point for check function
        }
      }
    }
  }
}

Secp256K1::~Secp256K1() {}

Point Secp256K1::AddDirect(Point &p1, Point &p2) {
  // Wyrównane zmienne dla lepszego dostępu do pamięci
  alignas(64) Int _s;
  alignas(64) Int _p;
  alignas(64) Int dy;
  alignas(64) Int dx;
  Point r;
  r.z.SetInt32(1);

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
  // P2.z = 1
  // Wyrównane zmienne dla AVX-512
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

  // Prefetching dla kluczowych danych
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
  // Wyrównane zmienne dla AVX-512
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

  // Prefetching
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
  // Wyrównane zmienne dla AVX-512
  alignas(64) Int _s;
  alignas(64) Int _p;
  alignas(64) Int a;
  Point r;
  r.z.SetInt32(1);

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
  int i = 0;
  uint8_t b;
  Point Q;
  Q.Clear();

  // Optymalizacja: prefetching dla tablicy GTable
  _mm_prefetch((const char *)&GTable[0], _MM_HINT_T0);

  // Search first significant byte
  for (i = 0; i < 32; i++) {
    b = privKey->GetByte(i);
    if (b) break;
  }

  if (i < 32) {
    Q = GTable[256 * i + (b - 1)];
    i++;

    // Przetwarzanie pozostałych bajtów z prefetchingiem
    for (; i < 32; i++) {
      b = privKey->GetByte(i);
      if (b) {
        // Prefetch następnego punktu z GTable
        if (i < 31) {
          _mm_prefetch((const char *)&GTable[256 * (i + 1)], _MM_HINT_T0);
        }
        Q = Add2(Q, GTable[256 * i + (b - 1)]);
      }
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

  // Wyrównane zmienne dla AVX-512
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

  // Prefetching
  _mm_prefetch((const char *)&p, _MM_HINT_T0);

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
  // Wyrównane zmienne dla AVX-512
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
  // Wyrównane zmienne dla AVX-512
  alignas(64) Int _s;
  alignas(64) Int _p;

  _s.ModSquareK1(&p.x);
  _p.ModMulK1(&_s, &p.x);
  _p.ModAdd(7);
  _s.ModMulK1(&p.y, &p.y);
  _s.ModSub(&_p);

  return _s.IsZero();  // ( ((pow2(y) - (pow3(x) + 7)) % P) == 0 );
}
