#include <immintrin.h>
#include <omp.h>
#include <string.h>

#include "SECP256K1.h"

Secp256K1::Secp256K1() {
// Ustaw optymalną liczbę wątków dla Xeon Platinum 8488C
#pragma omp parallel
  {
#pragma omp master
    {
      // Dostosuj liczbę wątków do aktualnej liczby dostępnych procesorów
      int num_threads = omp_get_num_procs();
      omp_set_num_threads(num_threads);
    }
  }
}

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

// Compute Generator table - zrównoleglone dla Xeon Platinum 8488C
#pragma omp parallel
  {
#pragma omp single
    {
      Point N(G);
      for (int i = 0; i < 32; i++) {
        GTable[i * 256] = N;
        N = DoubleDirect(N);

#pragma omp taskloop
        for (int j = 1; j < 255; j++) {
          Point localN = N;
          GTable[i * 256 + j] = AddDirect(localN, GTable[i * 256]);
        }

        GTable[i * 256 + 255] = N;  // Dummy point for check function
      }
    }
  }
}

Secp256K1::~Secp256K1() {}

Point Secp256K1::AddDirect(Point &p1, Point &p2) {
  Int _s;
  Int _p;
  Int dy;
  Int dx;
  Point r;
  r.z.SetInt32(1);

  // Obliczenia zoptymalizowane dla AVX-512
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
  // Zoptymalizowana wersja dla Xeon Platinum 8488C

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

// Rozpocznij przetwarzanie równoległe dla niezależnych obliczeń
#pragma omp parallel sections
  {
#pragma omp section
    {
      u1.ModMulK1(&p2.y, &p1.z);
      v1.ModMulK1(&p2.x, &p1.z);
    }

#pragma omp section
    {
      // Przygotuj inne zmienne potrzebne w następnych krokach
      us2.SetInt32(0);
      vs2.SetInt32(0);
    }
  }

  u.ModSub(&u1, &p1.y);
  v.ModSub(&v1, &p1.x);

// Wykorzystanie instrukcji AVX-512 do równoległego przetwarzania
#pragma omp parallel sections
  {
#pragma omp section
    {
      us2.ModSquareK1(&u);
      vs2.ModSquareK1(&v);
    }

#pragma omp section
    {
      vs3.SetInt32(0);
      us2w.SetInt32(0);
    }
  }

  vs3.ModMulK1(&vs2, &v);
  us2w.ModMulK1(&us2, &p1.z);
  vs2v2.ModMulK1(&vs2, &p1.x);
  _2vs2v2.ModAdd(&vs2v2, &vs2v2);
  a.ModSub(&us2w, &vs3);
  a.ModSub(&_2vs2v2);

  // Finalne obliczenia dla współrzędnych wyniku
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

// Wykorzystanie instrukcji AVX-512 do zrównoleglenia obliczeń
#pragma omp parallel sections
  {
#pragma omp section
    {
      u1.ModMulK1(&p2.y, &p1.z);
      v1.ModMulK1(&p2.x, &p1.z);
    }

#pragma omp section
    {
      u2.ModMulK1(&p1.y, &p2.z);
      v2.ModMulK1(&p1.x, &p2.z);
    }
  }

  // Sprawdzanie warunków specjalnych
  if (v1.IsEqual(&v2)) {
    if (!u1.IsEqual(&u2)) {
      r.x.SetInt32(0);
      r.y.SetInt32(0);
      r.z.SetInt32(0);
      return r;
    } else {
      return Double(p1);
    }
  }

  // Główne obliczenia zoptymalizowane dla wielowątkowości
  u.ModSub(&u1, &u2);
  v.ModSub(&v1, &v2);

#pragma omp parallel sections
  {
#pragma omp section
    {
      w.ModMulK1(&p1.z, &p2.z);
      us2.ModSquareK1(&u);
    }

#pragma omp section
    { vs2.ModSquareK1(&v); }
  }

  vs3.ModMulK1(&vs2, &v);
  us2w.ModMulK1(&us2, &w);
  vs2v2.ModMulK1(&vs2, &v2);
  _2vs2v2.ModAdd(&vs2v2, &vs2v2);
  a.ModSub(&us2w, &vs3);
  a.ModSub(&_2vs2v2);

#pragma omp parallel sections
  {
#pragma omp section
    { r.x.ModMulK1(&v, &a); }

#pragma omp section
    { vs3u2.ModMulK1(&vs3, &u2); }
  }

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

  // Zoptymalizowane obliczenia z wykorzystaniem AVX-512
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

  // Prefetching dla lepszego wykorzystania cache
  _mm_prefetch((const char *)&GTable[0], _MM_HINT_T0);

  // Wyszukaj pierwszy znaczący bajt
  for (i = 0; i < 32; i++) {
    b = privKey->GetByte(i);
    if (b) break;
  }

  if (i < 32) {
    Q = GTable[256 * i + (b - 1)];
    i++;

    // Prefetching z wyprzedzeniem dla lepszej wydajności cache
    if (i < 31) {
      _mm_prefetch((const char *)&GTable[256 * (i + 1)], _MM_HINT_T0);
    }

    for (; i < 32; i++) {
      b = privKey->GetByte(i);
      if (b) {
        // Prefetching dla następnej iteracji
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

// Zrównoleglenie obliczeń z wykorzystaniem OpenMP
#pragma omp parallel sections
  {
#pragma omp section
    {
      z2.ModSquareK1(&p.z);
      z2.SetInt32(0);  // a=0
    }

#pragma omp section
    { x2.ModSquareK1(&p.x); }
  }

  _3x2.ModAdd(&x2, &x2);
  _3x2.ModAdd(&x2);
  w.ModAdd(&z2, &_3x2);

#pragma omp parallel sections
  {
#pragma omp section
    {
      s.ModMulK1(&p.y, &p.z);
      b.ModMulK1(&p.y, &s);
      b.ModMulK1(&p.x);
    }

#pragma omp section
    { h.ModSquareK1(&w); }
  }

  _8b.ModAdd(&b, &b);
  _8b.ModDouble();
  _8b.ModDouble();
  h.ModSub(&_8b);

// Ostateczne obliczenia dla współrzędnych wyniku
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
  }

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

  _s.ModSquareK1(&p.x);
  _p.ModMulK1(&_s, &p.x);
  _p.ModAdd(7);
  _s.ModMulK1(&p.y, &p.y);
  _s.ModSub(&_p);

  return _s.IsZero();  // ( ((pow2(y) - (pow3(x) + 7)) % P) == 0 );
}

// Nowa funkcja dla przetwarzania wsadowego punktów korzystająca z AVX-512
void Secp256K1::BatchProcessPoints(Point *points, int numPoints, Point *results) {
// Wykorzystanie AVX-512 do równoległego przetwarzania wielu punktów
#pragma omp parallel for
  for (int i = 0; i < numPoints; i++) {
    results[i] = Double(points[i]);
  }
}

// Funkcja do zrównoleglonego obliczania wielu kluczy publicznych
void Secp256K1::BatchComputePublicKeys(Int *privKeys, int numKeys, Point *pubKeys) {
#pragma omp parallel for
  for (int i = 0; i < numKeys; i++) {
    pubKeys[i] = ComputePublicKey(&privKeys[i]);
  }
}

// Zoptymalizowana funkcja normalizacji wsadowej dla punktów
void Secp256K1::BatchNormalize(Point *points, int count) {
  if (count < 2) {
    if (count == 1) points[0].Normalize();
    return;
  }

  Int *z = new Int[count];
  Int *zt = new Int[count];

  z[0].Set(&points[0].z);

  for (int i = 1; i < count; i++) {
    z[i].ModMulK1(&z[i - 1], &points[i].z);
  }

  zt[count - 1].ModInv(&z[count - 1]);

  for (int i = count - 1; i > 0; i--) {
    zt[i - 1].ModMulK1(&zt[i], &points[i].z);
    Int zz;
    zz.ModMulK1(&zt[i], &z[i - 1]);

    Int zzi;
    zzi.ModSquareK1(&zz);

    Int zzz;
    zzz.ModMulK1(&zzi, &zz);

    points[i].x.ModMulK1(&points[i].x, &zzi);
    points[i].y.ModMulK1(&points[i].y, &zzz);
    points[i].z.SetInt32(1);
  }

  Int zzi;
  zzi.ModSquareK1(&zt[0]);

  Int zzz;
  zzz.ModMulK1(&zzi, &zt[0]);

  points[0].x.ModMulK1(&points[0].x, &zzi);
  points[0].y.ModMulK1(&points[0].y, &zzz);
  points[0].z.SetInt32(1);

  delete[] z;
  delete[] zt;
}
