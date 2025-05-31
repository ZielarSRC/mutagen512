#include <string.h>

#include "SECP256k1.h"

Secp256K1::Secp256K1() {}

void Secp256K1::Init() {
  // Alokuj tymczasowe obiekty z wyrównaniem dla AVX-512
  alignas(64) Point G_aligned;
  alignas(64) Int order_aligned;

  // Inicjalizuj dane z wyrównaniem
  G_aligned.x.SetBase16("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
  G_aligned.y.SetBase16("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
  order_aligned.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

  // Przypisz do właściwych zmiennych
  G = G_aligned;
  order = order_aligned;

  Int::InitK1(&order);

  // Compute Generator table
  Point N(G);
  for (int i = 0; i < 32; i++) {
    GTable[i * 256] = N;
    Point P(N);
    for (int j = 1; j < 255; j++) {
      P = P.Add(&N);
      GTable[i * 256 + j] = P;
    }
    P = P.Add(&N);
    N = P;
  }

  // Compute Lambda table for Endomorphism
  // lambda = cubic_root_1 (cubic_root_1 = 1 mod p)
  lambda.SetBase16("5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72");
  lambdaB.SetBase16("AC9C52B33FA3CF1F5AD9E3FD77ED9BA4A880B9FC8EC739C2E0CFC810B51283CE");

  Int kOrder;
  kOrder.Set(&order);
  kOrder.Add(&order);
  kOrder.Add(&order);  // 3*order
  beta.SetBase16("7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE");
  beta2.SetBase16("851695D49A83F8EF919BB86153CBCB16630FB68AED0A766A3EC693D68E6AFA40");

  for (int i = 0; i < 256; i++) {
    endomorphism[i] = new Point[256];
    Point Z;
    Z.Clear();
    for (int j = 0; j < 256; j++) {
      // Compute k1.G + k2.G^beta
      Int k1;
      Int k2;
      k1.SetInt32(i);
      k2.SetInt32(j);
      k1.ShiftL(8);
      k1.Add((uint64_t)j);
      k2.ShiftL(16);
      endomorphism[i][j].x.SetInt32(0);
      endomorphism[i][j].y.SetInt32(0);
      if (!k1.IsZero()) {
        if (k1.IsOne()) {
          endomorphism[i][j] = G;
        } else {
          endomorphism[i][j] = Double(&G);
          for (int m = 1; m < k1.GetBitLength() - 1; m++) {
            endomorphism[i][j] = Double(&endomorphism[i][j]);
            if (k1.GetBit(k1.GetBitLength() - 1 - m)) {
              endomorphism[i][j] = Add(&endomorphism[i][j], &G);
            }
          }
        }
      }

      if (!k2.IsZero()) {
        Point B;
        B.x.Set(&G.x);
        B.x.ModMul(&beta);
        B.y.Set(&G.y);
        if (k2.IsOne()) {
          if (k1.IsZero()) {
            endomorphism[i][j] = B;
          } else {
            endomorphism[i][j] = Add(&endomorphism[i][j], &B);
          }
        } else {
          Point Q;
          Q = Double(&B);
          for (int m = 1; m < k2.GetBitLength() - 1; m++) {
            Q = Double(&Q);
            if (k2.GetBit(k2.GetBitLength() - 1 - m)) {
              Q = Add(&Q, &B);
            }
          }
          if (k1.IsZero()) {
            endomorphism[i][j] = Q;
          } else {
            endomorphism[i][j] = Add(&endomorphism[i][j], &Q);
          }
        }
      }
    }
  }
}

Secp256K1::~Secp256K1() {
  for (int i = 0; i < 256; i++) delete[] endomorphism[i];
}

Point Secp256K1::ComputePublicKey(Int *privKey) {
  int i = 0;
  uint8_t b;
  Point Q;
  Q.Clear();
  while (i < 32) {
    b = privKey->GetByte(i);
    if (i == 31)
      Q = Add(&Q, &GTable[i * 256 + (b & 0x7F)]);
    else
      Q = Add(&Q, &GTable[i * 256 + b]);
    i++;
  }
  return Q;
}

Point Secp256K1::NextKey(Point &key) {
  // Input key must be different than G
  // in order to use NextKey
  return Add(&key, &G);
}

void PrintResult(bool ok) {
  if (ok) {
    printf("OK\n");
  } else {
    printf("ERROR\n");
  }
}

bool Secp256K1::CheckPoint(Point &p) {
  Int _s, _p;
  Int::SetupField(&order);
  _s.ModSquare(&p.y);
  _p.ModMul(&p.x, &p.x);
  _p.ModMul(&p.x, &_p);
  _p.ModAdd(7);
  _p.ModSub(&_s);
  return _p.IsZero();
}

void Secp256K1::GetHash160(int type, bool compressed, Point &pubKey, unsigned char *hash) {
  unsigned char buffer[128];
  unsigned char buffer2[32];
  int hLen = 0;
  hash[0] = 0;
  if (type == P2PKH) {
    hash[0] = 0;
  } else if (type == P2SH) {
    hash[0] = 5;
  }

  // Only P2PKH is currently supported
  if (type != P2PKH) {
    memset(hash, 0, 20);
    return;
  }

  if (!compressed) {
    buffer[0] = 4;
    pubKey.x.Get32Bytes(buffer + 1);
    pubKey.y.Get32Bytes(buffer + 33);
    hLen = 65;
  } else {
    // Compressed point
    buffer[0] = pubKey.y.IsEven() ? 2 : 3;
    pubKey.x.Get32Bytes(buffer + 1);
    hLen = 33;
  }

  ripemd160_avx2::getHash160(buffer, hLen, hash + 1);
}

void Secp256K1::GetHashAddr(int type, bool compressed, Point &pubKey, unsigned char *hash) {
  unsigned char h[20];
  GetHash160(type, compressed, pubKey, h);
}

std::string Secp256K1::GetPrivAddress(bool compressed, Int &privKey) {
  unsigned char wif[38];
  unsigned char *buff = wif;
  int bSize = 0;
  uint8_t prefix = 0;

  // Network byte
  *(buff++) = 0x80;

  // Private key bytes
  privKey.Get32Bytes(buff);
  buff += 32;

  if (compressed) {
    // Additional 01 sufix
    *(buff++) = 1;
  }
  bSize = (int)(buff - wif);

  // Base58 encoding (with checksum)
  return toBase58Check(wif, bSize);
}

std::string Secp256K1::GetPublicAddress(bool compressed, Point &pubKey) {
  unsigned char address[25];
  // Get hash
  GetHash160(P2PKH, compressed, pubKey, address);

  // Base58 encoding
  return toBase58Check(address, 21);
}

// Compute a*G + b*Q using endomorphism
Point Secp256K1::AddDirect(Point &p1, Point &p2) {
  Int _s;
  Int _p;
  Int dy;
  Int dx;
  Point r;
  r.z.SetInt32(1);

  dy.ModSub(&p2.y, &p1.y);
  dx.ModSub(&p2.x, &p1.x);
  dx.ModInv();
  _s.ModMul(&dy, &dx);

  _p.ModSquare(&_s);
  _p.ModSub(&p1.x);
  _p.ModSub(&p2.x);
  r.x.Set(&_p);

  _p.ModSub(&p1.x, &r.x);
  _p.ModMul(&_s);
  r.y.ModSub(&_p, &p1.y);

  return r;
}

// Compute a*G + b*Q using endomorphism
Point Secp256K1::ComputePublicKey(Int *a, Int *b, Point &Q) {
  uint8_t hA[32];
  uint8_t hB[32];
  a->Get32Bytes(hA);
  b->Get32Bytes(hB);
  Point endBeta;
  endBeta.x.Set(&Q.x);
  endBeta.x.ModMul(&beta);
  endBeta.y.Set(&Q.y);

  Point bQ;
  bQ.Clear();
  Point aG;
  aG.Clear();

  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 8; j++) {
      aG = Double(&aG);
      bQ = Double(&bQ);
      if ((hA[i] & 1) != 0) {
        aG = Add(&aG, &G);
      }
      if ((hB[i] & 1) != 0) {
        bQ = Add(&bQ, &endBeta);
      }
      hA[i] >>= 1;
      hB[i] >>= 1;
    }
  }
  bQ = Add(&bQ, &aG);
  return bQ;
}

// -------------------------------------------------------------------------

Point Secp256K1::Double(Point *p) {
  Int _s;
  Int _p;
  Int a;
  Point r;
  a.SetInt32(0);
  r.z.SetInt32(1);

  if (p->IsZero()) return r;

  _s.ModMul(&p->x, &p->x);
  _s.ModAdd(_s);
  _s.ModAdd(_s);
  _s.ModAdd(a);
  _p.ModAdd(&p->y, &p->y);
  _p.ModInv();
  _s.ModMul(&_p);

  _p.ModSquare(&_s);
  _p.ModSub(&p->x);
  _p.ModSub(&p->x);
  r.x.Set(&_p);

  _p.ModSub(&p->x, &r.x);
  _p.ModMul(&_s);
  r.y.ModSub(&_p, &p->y);

  return r;
}

// -------------------------------------------------------------------------

Point Secp256K1::Add(Point *p1, Point *p2) {
  Int u;
  Int v;
  Int u1;
  Int v1;
  Int vs2;
  Int vs3;
  Int us2;
  Int a;
  Int w;
  Point r;
  r.z.SetInt32(1);

  if (p1->IsZero()) {
    r.x.Set(&p2->x);
    r.y.Set(&p2->y);
    return r;
  }

  if (p2->IsZero()) {
    r.x.Set(&p1->x);
    r.y.Set(&p1->y);
    return r;
  }

  u.ModSub(&p2->y, &p1->y);
  v.ModSub(&p2->x, &p1->x);

  if (v.IsZero()) {
    if (u.IsZero()) {
      return Double(p1);
    } else {
      r.z.SetInt32(0);
      return r;
    }
  }

  v.ModInv();
  u1.ModMul(&u, &v);

  vs2.ModSquare(&v);
  vs3.ModMul(&vs2, &v);
  us2.ModSquare(&u);
  a.ModMul(&vs2, &p1->x);
  w.ModMul(&vs3, &p1->y);

  r.x.ModSquare(&u1);
  r.x.ModSub(&vs2);
  r.x.ModSub(&p1->x);
  r.x.ModSub(&p2->x);

  r.y.ModSub(&p1->x, &r.x);
  r.y.ModMul(&u1);
  r.y.ModSub(&p1->y);

  return r;
}

std::string Secp256K1::toBase58Check(unsigned char *data, int len) {
  unsigned char checksum[32];
  unsigned char hash[32];
  std::string ret;

  // Compute SHA-256 hash of data
  sha256_avx2::computeSHA256(data, len, hash);

  // Compute checksum (SHA-256 of SHA-256 hash)
  sha256_avx2::computeSHA256(hash, 32, checksum);

  // Concatenate data and checksum
  unsigned char *dataAndChecksum = new unsigned char[len + 4];
  memcpy(dataAndChecksum, data, len);
  memcpy(dataAndChecksum + len, checksum, 4);

  // Convert to Base58
  ret = toBase58(dataAndChecksum, len + 4);

  delete[] dataAndChecksum;
  return ret;
}

std::string Secp256K1::toBase58(unsigned char *data, int len) {
  static const char *base58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  std::string result;
  Int bn58;
  Int bn;
  Int r;
  bn58.SetInt32(58);
  bn.SetBytes(data, len);

  // Count leading zeros
  int leadingZeros = 0;
  for (int i = 0; i < len && data[i] == 0; i++) {
    leadingZeros++;
  }

  // Convert to base58 representation
  while (!bn.IsZero()) {
    bn.Div(&bn58, &r);
    result = base58[r.GetInt32()] + result;
  }

  // Add leading '1's for each leading zero byte
  for (int i = 0; i < leadingZeros; i++) {
    result = "1" + result;
  }

  return result;
}
