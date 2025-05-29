#ifndef SECP256K1H
#define SECP256K1H

#include <immintrin.h>

#include <string>
#include <vector>

#include "Point.h"

// Address type
#define P2PKH 0
#define P2SH 1
#define BECH32 2

// Alignment for AVX-512 operations
#define AVX512_ALIGNMENT 64

class Secp256K1 {
 public:
  Secp256K1();
  ~Secp256K1();
  void Init();
  Point ComputePublicKey(Int *privKey);
  Point NextKey(Point &key);
  void Check();
  bool EC(Point &p);
  Int GetY(Int x, bool isEven);

  // Batch operations optimized for AVX-512
  void BatchComputePublicKeys(Int *privKeys, Point *pubKeys, int batchSize);

  // AVX-512 optimized versions of key operations
  Point AddDirectAVX512(Point &p1, Point &p2);
  Point DoubleDirectAVX512(Point &p);

  void GetHash160(int type, bool compressed, Point &k0, Point &k1, Point &k2, Point &k3,
                  uint8_t *h0, uint8_t *h1, uint8_t *h2, uint8_t *h3);

  void GetHash160(int type, bool compressed, Point &pubKey, unsigned char *hash);

  std::string GetAddress(int type, bool compressed, Point &pubKey);
  std::string GetAddress(int type, bool compressed, unsigned char *hash160);
  std::vector<std::string> GetAddress(int type, bool compressed, unsigned char *h1,
                                      unsigned char *h2, unsigned char *h3, unsigned char *h4);
  std::string GetPrivAddress(bool compressed, Int &privKey);
  std::string GetPublicKeyHex(bool compressed, Point &p);
  Point ParsePublicKeyHex(std::string str, bool &isCompressed);

  bool CheckPudAddress(std::string address);

  static Int DecodePrivateKey(char *key, bool *compressed);

  // Standard operations
  Point Add(Point &p1, Point &p2);
  Point Add2(Point &p1, Point &p2);
  Point AddDirect(Point &p1, Point &p2);
  Point Double(Point &p);
  Point DoubleDirect(Point &p);

  Point G;    // Generator
  Int order;  // Curve order

 private:
  uint8_t GetByte(std::string &str, int idx);

  // Helper methods for AVX-512 optimizations
  void PrefetchPoint(const Point &p, int hint = _MM_HINT_T0);
  bool ShouldUsePrefetch(const Point &p1, const Point &p2);

  alignas(AVX512_ALIGNMENT) Point GTable[256 * 32];  // Generator table aligned for AVX-512
};

#endif  // SECP256K1H
