#ifndef SECP256K1H
#define SECP256K1H

#include "Point.h"
#include <string>
#include <vector>
#include <immintrin.h>

// Address type
#define P2PKH  0
#define P2SH   1
#define BECH32 2

class Secp256K1 {

public:

  Secp256K1();
  ~Secp256K1();
  void Init();
  Point ComputePublicKey(Int *privKey);
  Point NextKey(Point &key);
  void Check();
  bool  EC(Point &p);
  Int GetY(Int x, bool isEven);

  // Enhanced batch processing with AVX-512
  void GetHash160_AVX512_16(int type, bool compressed,
    Point &k0, Point &k1, Point &k2, Point &k3, Point &k4, Point &k5, Point &k6, Point &k7,
    Point &k8, Point &k9, Point &k10, Point &k11, Point &k12, Point &k13, Point &k14, Point &k15,
    uint8_t *h0, uint8_t *h1, uint8_t *h2, uint8_t *h3, uint8_t *h4, uint8_t *h5, uint8_t *h6, uint8_t *h7,
    uint8_t *h8, uint8_t *h9, uint8_t *h10, uint8_t *h11, uint8_t *h12, uint8_t *h13, uint8_t *h14, uint8_t *h15);

  void GetHash160(int type, bool compressed,
    Point &k0, Point &k1, Point &k2, Point &k3,
    uint8_t *h0, uint8_t *h1, uint8_t *h2, uint8_t *h3);

  void GetHash160(int type, bool compressed, Point &pubKey, unsigned char *hash);

  // Batch key computation with AVX-512 acceleration
  void ComputePublicKeys_AVX512_16(Int *privKeys, Point *pubKeys);
  void ComputePublicKeys_AVX512_8(Int *privKeys, Point *pubKeys);

  std::string GetAddress(int type, bool compressed, Point &pubKey);
  std::string GetAddress(int type, bool compressed, unsigned char *hash160);
  std::vector<std::string> GetAddress(int type, bool compressed, unsigned char *h1, unsigned char *h2, unsigned char *h3, unsigned char *h4);
  std::string GetPrivAddress(bool compressed, Int &privKey);
  std::string GetPublicKeyHex(bool compressed, Point &p);
  Point ParsePublicKeyHex(std::string str, bool &isCompressed);

  bool CheckPudAddress(std::string address);

  static Int DecodePrivateKey(char *key, bool *compressed);

  // Optimized point operations for batch processing
  Point Add(Point &p1, Point &p2);
  Point Add2(Point &p1, Point &p2);
  Point AddDirect(Point &p1, Point &p2);
  Point Double(Point &p);
  Point DoubleDirect(Point &p);

  // Batch point operations with AVX-512
  void AddBatch_AVX512_8(Point *p1, Point *p2, Point *result);
  void DoubleBatch_AVX512_8(Point *p, Point *result);

  Point G;                 // Generator
  Int   order;             // Curve order

private:

  uint8_t GetByte(std::string &str, int idx);

  // Enhanced generator table with better cache alignment
  Point GTable[256*32] __attribute__((aligned(64)));

  // AVX-512 optimized internal functions
  void PrecomputeTable_AVX512();
  void BatchScalarMult_AVX512_16(Int *scalars, Point *results);

};

#endif // SECP256K1H
