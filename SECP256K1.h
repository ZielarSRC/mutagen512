#ifndef SECP256K1H
#define SECP256K1H

#include <immintrin.h>  // For AVX-512 intrinsics

#include <string>
#include <vector>

#include "Point.h"

// Address type
#define P2PKH 0
#define P2SH 1
#define BECH32 2

// Secp256K1 class optimized for AVX-512 on Intel Xeon Platinum 8488C
class alignas(64) Secp256K1 {
 public:
  Secp256K1();
  ~Secp256K1();

  // Initialization with AVX-512 optimizations
  void Init();

  // Core ECC operations optimized for AVX-512
  Point ComputePublicKey(Int *privKey);
  Point NextKey(Point &key);
  void Check();
  bool EC(Point &p);
  Int GetY(Int x, bool isEven);

  // Hash operations with AVX-512 acceleration
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

  // Point arithmetic optimized for AVX-512
  Point Add(Point &p1, Point &p2);
  Point Add2(Point &p1, Point &p2);
  Point AddDirect(Point &p1, Point &p2);
  Point Double(Point &p);
  Point DoubleDirect(Point &p);

  // New batch methods optimized for AVX-512
  void BatchComputePublicKeys(Int *privKeys, Point *pubKeys, int count);
  void BatchAddPoints(Point *p1, Point *p2, Point *result, int count);
  void BatchDoublePoints(Point *points, Point *result, int count);
  void BatchEC(Point *points, bool *results, int count);

  // Aligned for AVX-512
  alignas(64) Point G;    // Generator
  alignas(64) Int order;  // Curve order

 private:
  uint8_t GetByte(std::string &str, int idx);

  // Generator table aligned for AVX-512 operations
  alignas(64) Point GTable[256 * 32];  // Generator table
};

#endif  // SECP256K1H
