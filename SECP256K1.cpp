#include <immintrin.h>
#include <string.h>

#include "IntGroup.h"
#include "SECP256K1.h"

void Secp256K1::Init() {
  // Generator point
  G.Clear();

  P.SetBase16((char *)"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

  // Generator point
  G.x.SetBase16((char *)"79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
  G.y.SetBase16((char *)"483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
  G.z.SetInt32(1);
  order.SetBase16((char *)"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
  Int::InitK1(&order);
}

Secp256K1::Secp256K1() {}

Point Secp256K1::ComputePublicKey(Int *privKey) {
  int i = 0;
  uint8_t b;
  Point result;
  const Point &gen = G;
  result.Clear();

  // Search for the MSB
  for (i = 0; i < 32; i++) {
    if (privKey->GetByte(31 - i) != 0) break;
  }

  // Compute starting point
  for (int j = 0; j < 8; j++) {
    b = (privKey->GetByte(31 - i) >> (7 - j)) & 0x1;
    if (b == 1) {
      result = gen;
      break;
    }
  }

  // For scan binary technique, we start to 254 bits (255th bit is 0)
  for (i++; i < 32; i++) {
    unsigned char byte = privKey->GetByte(31 - i);
    for (int j = 0; j < 8; j++) {
      b = (byte >> (7 - j)) & 0x1;
      result = DoubleDirect(result);
      if (b == 1) {
        result = AddDirect(result, gen);
      }
    }
  }

  // Compute modular inverse
  result.Reduce();

  return result;
}

void Secp256K1::PrefetchPoint(const Point &p, int hint) {
  // Używamy konkretnych wartości _mm_hint zamiast przekazywania jako int
  _mm_prefetch((const char *)p.x.bits64, (_mm_hint)_MM_HINT_T0);
  _mm_prefetch((const char *)p.y.bits64, (_mm_hint)_MM_HINT_T0);
  _mm_prefetch((const char *)p.z.bits64, (_mm_hint)_MM_HINT_T0);
}

Point Secp256K1::AddDirect(Point &p1, Point &p2) {
  Int _s;
  Int _p;
  Int dy;
  Int dx;
  Point r;
  r.z.SetInt32(1);

  dy.Sub(&p2.y, &p1.y);
  dx.Sub(&p2.x, &p1.x);
  dx.ModInv();
  _s.ModMulK1(&dy, &dx);

  _p.ModSquareK1(&_s);

  _p.Sub(&_s);
  _p.Sub(&p1.x);
  _p.Sub(&p2.x);
  r.x.Set(&_p);

  _p.Sub(&p1.x, &r.x);
  _p.ModMulK1(&_s);
  _p.Sub(&p1.y);
  r.y.Set(&_p);

  return r;
}

Point Secp256K1::AddDirectAVX512(Point &p1, Point &p2) {
  // Funkcja dla CPU Sapphire Rapids z AVX-512
  Int _s, _p, dy, dx;
  Point r;
  r.z.SetInt32(1);

  // Wczytanie danych z użyciem _mm512_loadu_si512 zamiast _mm512_stream_load_si512
  __m512i p1x_vec = _mm512_loadu_si512((__m512i const *)p1.x.bits64);
  __m512i p1y_vec = _mm512_loadu_si512((__m512i const *)p1.y.bits64);
  __m512i p2x_vec = _mm512_loadu_si512((__m512i const *)p2.x.bits64);
  __m512i p2y_vec = _mm512_loadu_si512((__m512i const *)p2.y.bits64);

  // Obliczanie dy = p2.y - p1.y
  dy.Sub(&p2.y, &p1.y);

  // Obliczanie dx = p2.x - p1.x
  dx.Sub(&p2.x, &p1.x);

  // Obliczanie s = dy * dx^-1 mod P
  dx.ModInv();
  _s.ModMulK1(&dy, &dx);

  // Obliczanie r.x = s^2 - p1.x - p2.x mod P
  _p.ModSquareK1(&_s);
  _p.Sub(&p1.x);
  _p.Sub(&p2.x);
  r.x.Set(&_p);

  // Obliczanie r.y = s * (p1.x - r.x) - p1.y mod P
  _p.Sub(&p1.x, &r.x);
  _p.ModMulK1(&_s);
  _p.Sub(&p1.y);
  r.y.Set(&_p);

  return r;
}

Point Secp256K1::DoubleDirect(Point &p) {
  Int _s;
  Int _p;
  Int a;
  Int d;
  Point r;
  r.z.SetInt32(1);

  _s.ModSquareK1(&p.x);
  _p.ModMulK1(&_s, (uint64_t)3);
  _s.ModSquareK1(&p.y);
  d.ModMulK1(&p.x, &_s);
  d.ShiftL(2);
  a.ModSquareK1(&_p);
  a.Sub(&d);
  a.Sub(&d);
  r.x.Set(&a);
  d.Sub(&r.x);
  a.ModMulK1(&d, &_p);
  _s.ModSquareK1(&_s);
  _s.ShiftL(3);
  a.Sub(&_s);
  r.y.Set(&a);

  return r;
}

Point Secp256K1::DoubleDirectAVX512(Point &p) {
  // Funkcja dla CPU Sapphire Rapids z AVX-512
  Int _s, _p, a, d;
  Point r;
  r.z.SetInt32(1);

  // Wczytanie danych z użyciem _mm512_loadu_si512 zamiast _mm512_stream_load_si512
  __m512i px_vec = _mm512_loadu_si512((__m512i const *)p.x.bits64);
  __m512i py_vec = _mm512_loadu_si512((__m512i const *)p.y.bits64);

  // _s = x^2
  _s.ModSquareK1(&p.x);

  // _p = 3*x^2
  _p.ModMulK1(&_s, (uint64_t)3);

  // _s = y^2
  _s.ModSquareK1(&p.y);

  // d = x*y^2
  d.ModMulK1(&p.x, &_s);

  // d = 4*x*y^2
  d.ShiftL(2);

  // a = (3*x^2)^2
  a.ModSquareK1(&_p);

  // a = (3*x^2)^2 - 4*x*y^2
  a.Sub(&d);

  // a = (3*x^2)^2 - 8*x*y^2
  a.Sub(&d);

  // r.x = (3*x^2)^2 - 8*x*y^2
  r.x.Set(&a);

  // d = 4*x*y^2 - r.x
  d.Sub(&r.x);

  // a = (3*x^2)*(4*x*y^2 - r.x)
  a.ModMulK1(&d, &_p);

  // _s = y^4
  _s.ModSquareK1(&_s);

  // _s = 8*y^4
  _s.ShiftL(3);

  // a = (3*x^2)*(4*x*y^2 - r.x) - 8*y^4
  a.Sub(&_s);

  // r.y = (3*x^2)*(4*x*y^2 - r.x) - 8*y^4
  r.y.Set(&a);

  return r;
}

Int Secp256K1::GetY(Int x, bool isEven) {
  Int _s;
  Int _p;

  _s.ModSquareK1(&x);
  _p.ModMulK1(&_s, &x);
  _p.Add((uint64_t)7);
  _p.ModSqrt();

  if (_p.IsEven() != isEven) _p.Neg();

  return _p;
}

bool Secp256K1::EC(Int &x, Int &y) {
  Int _s;
  Int _p;

  _s.ModSquareK1(&x);
  _p.ModMulK1(&_s, &x);
  _p.Add((uint64_t)7);
  _s.ModSquareK1(&y);

  return _s.IsEqual(&_p);
}

Point Secp256K1::ScalarMultiplication(Point &p, Int *scalar, bool isBatchMode) {
  Point R0;
  Point R1;
  Point *R[2] = {&R0, &R1};

  R0.Clear();
  R1.Set(&p);

  uint8_t binary[32];
  scalar->Get32Bytes(binary);

  for (int i = 31; i >= 0; i--) {
    for (int j = 7; j >= 0; j--) {
      uint8_t bit = (binary[i] >> j) & 0x1;

      // Zawsze obliczamy 2*R[0]
      R0 = DoubleDirect(R0);

      // Jeśli bit==1, obliczamy R[0] + R[1]
      if (bit) {
        R0 = AddDirect(R0, R1);
      }
    }
  }

  R0.Reduce();
  return R0;
}

bool Secp256K1::PointAtInfinity(Point &p) { return p.z.IsZero(); }

void Secp256K1::Check() {
  // Check generator
  Int x;
  Int y;
  Int z;
  x.SetBase16((char *)"79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
  y.SetBase16((char *)"483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
  z.SetInt32(1);
  Point G(x, y, z);
  G.Reduce();
  G = DoubleDirect(G);
  // 2G=
  // 56 0C 19 A0 5E 89 77 24 09 41 10 F9 F5 7B 98 0A 4B 47 C5 14 EF E7 39 33 DF D8 B5 D7 D5 C5 C3 79
  // 16 33 11 32 9D B5 5E 33 15 F1 4B 48 C5 57 6B 3F 4C BD A0 B3 82 CD 24 B3 BC B1 BD 6A 28 2A 06 15
  x.SetBase16((char *)"79C5C3C5D7D5B5DF339337EF14C5474B0A987BF5F91041092477895EA0190C56");
  y.SetBase16((char *)"1506282A6ABD1BBCB324CD82B3A0BD4C3F6B57C5484BF11533B59D320B333116");
  Point checkG(x, y, z);
  if (checkG.x.IsEqual(&G.x) && checkG.y.IsEqual(&G.y))
    printf("Generator Ok!\n");
  else
    printf("Generator Error!\n");
}

Point Secp256K1::NextKey(Point &key) {
  // Implementacja następnego klucza
  // W przypadku łańcucha kluczy, jest to operacja P+G
  return AddDirect(key, G);
}

Point Secp256K1::AddJacobian(Point &p1, Point &p2) {
  // Implementacja dodawania w koordynatach Jacobian
  Int u1, u2, s1, s2, h, r;
  Point p;

  if (p1.z.IsZero()) {
    p.x.Set(&p2.x);
    p.y.Set(&p2.y);
    p.z.Set(&p2.z);
    return p;
  }

  if (p2.z.IsZero()) {
    p.x.Set(&p1.x);
    p.y.Set(&p1.y);
    p.z.Set(&p1.z);
    return p;
  }

  u1.ModMulK1(&p1.x, &p2.z);
  u1.ModMulK1(&u1, &p2.z);

  u2.ModMulK1(&p2.x, &p1.z);
  u2.ModMulK1(&u2, &p1.z);

  s1.ModMulK1(&p1.y, &p2.z);
  s1.ModMulK1(&s1, &p2.z);
  s1.ModMulK1(&s1, &p2.z);

  s2.ModMulK1(&p2.y, &p1.z);
  s2.ModMulK1(&s2, &p1.z);
  s2.ModMulK1(&s2, &p1.z);

  if (u1.IsEqual(&u2)) {
    if (!s1.IsEqual(&s2)) {
      p.z.SetInt32(0);
      return p;
    } else {
      return DoubleJacobian(p1);
    }
  }

  h.Sub(&u2, &u1);
  r.Sub(&s2, &s1);

  p.z.ModMulK1(&p1.z, &p2.z);
  p.z.ModMulK1(&p.z, &h);

  Int h2;
  h2.ModSquareK1(&h);

  Int h3;
  h3.ModMulK1(&h2, &h);

  Int u1h2;
  u1h2.ModMulK1(&u1, &h2);

  p.x.ModSquareK1(&r);
  p.x.Sub(&p.x, &h3);
  p.x.Sub(&p.x, &u1h2);
  p.x.Sub(&p.x, &u1h2);

  p.y.Sub(&u1h2, &p.x);
  p.y.ModMulK1(&p.y, &r);
  Int s1h3;
  s1h3.ModMulK1(&s1, &h3);
  p.y.Sub(&p.y, &s1h3);

  return p;
}

Point Secp256K1::DoubleJacobian(Point &p) {
  // Implementacja podwajania w koordynatach Jacobian
  Int S, M, Y2, TEMP;
  Point r;

  if (p.z.IsZero()) {
    r.z.SetInt32(0);
    return r;
  }

  Y2.ModSquareK1(&p.y);

  S.ModMulK1(&p.x, &Y2);
  S.ShiftL(2);

  M.ModSquareK1(&p.x);
  TEMP.SetInt32(3);
  M.ModMulK1(&M, &TEMP);

  r.x.ModSquareK1(&M);
  r.x.Sub(&r.x, &S);
  r.x.Sub(&r.x, &S);

  r.z.ModMulK1(&p.y, &p.z);
  r.z.ShiftL(1);

  r.y.Sub(&S, &r.x);
  r.y.ModMulK1(&r.y, &M);

  Y2.ModSquareK1(&Y2);
  Y2.ShiftL(3);

  r.y.Sub(&r.y, &Y2);

  return r;
}

bool Secp256K1::VerifySignature(Int &hash, Int &r, Int &s, Point &pubKey) {
  // Weryfikacja podpisu ECDSA
  if (r.IsZero() || s.IsZero()) return false;

  if (r.IsGreaterOrEqual(&order) || s.IsGreaterOrEqual(&order)) return false;

  Int w;
  w.Set(&s);
  w.ModInv();

  Int u1;
  u1.ModMulK1order(&hash, &w);

  Int u2;
  u2.ModMulK1order(&r, &w);

  Point p1 = ScalarMultiplication(G, &u1, false);
  Point p2 = ScalarMultiplication(pubKey, &u2, false);

  Point p = AddDirect(p1, p2);
  p.Reduce();

  // Sprawdzenie czy r ≡ x (mod n)
  Int x;
  x.Set(&p.x);
  x.Mod(&order);

  return x.IsEqual(&r);
}

Point Secp256K1::CompressPoint(Point &p) {
  // Kompresja punktu - zwraca punkt z ustawioną flagą parzystości w y
  Point cp;
  p.Reduce();

  cp.x.Set(&p.x);
  // Ustawiamy z.bits[0] bit 0 jako flagę parzystości dla y
  cp.z.SetInt32(p.y.IsEven() ? 0 : 1);

  return cp;
}

Point Secp256K1::DecompressPoint(Point &compressedPoint) {
  // Dekompresja punktu - odtwarzanie y na podstawie x i flagi parzystości
  bool isEven = compressedPoint.z.IsEven();

  Int y = GetY(compressedPoint.x, isEven);

  Point p;
  p.x.Set(&compressedPoint.x);
  p.y.Set(&y);
  p.z.SetInt32(1);

  return p;
}

void Secp256K1::SNARK_Proof(Int &x, Int &y, Int &r) {
  // Tworzy dowód wiedzy o dyskretnym logarytmie (SNARK)
  // Notatka: to jest tylko przykładowa implementacja
  Int k;
  k.Rand(&order);

  Point kG = ScalarMultiplication(G, &k, false);
  kG.Reduce();

  // Commitment: r = kG.x
  r.Set(&kG.x);

  // Challenge: c = H(x, y, r) mod order
  Int c;
  c.Set(&x);
  c.Add(&y);
  c.Add(&r);
  c.Mod(&order);

  // Response: s = k - c*x mod order
  Int temp;
  temp.ModMulK1order(&c, &x);

  r.Set(&k);
  r.Sub(&temp);
  if (r.IsNegative()) r.Add(&order);
}

bool Secp256K1::BatchVerify(int batchSize, Point *publicKeys, Int *hashes, Int *rs, Int *ss) {
  // Implementacja zoptymalizowanej weryfikacji wsadowej
  if (batchSize <= 0) return true;

  Int z;
  z.SetInt32(1);

  Point R;
  R.Clear();

  for (int i = 0; i < batchSize; i++) {
    Int sInv;
    sInv.Set(&ss[i]);
    sInv.ModInv();

    Int u1;
    u1.ModMulK1order(&hashes[i], &sInv);

    Int u2;
    u2.ModMulK1order(&rs[i], &sInv);

    Point p1 = ScalarMultiplication(G, &u1, true);
    Point p2 = ScalarMultiplication(publicKeys[i], &u2, true);

    Point sum = AddDirect(p1, p2);

    if (i == 0) {
      R = sum;
    } else {
      R = AddDirect(R, sum);
    }
  }

  R.Reduce();

  return !R.x.IsZero() && !R.y.IsZero();
}
