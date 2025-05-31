#ifndef SECP256K1_AVX512_H
#define SECP256K1_AVX512_H

#include <immintrin.h>

#include <cstdint>
#include <string>
#include <vector>

#include "Point.h"

// Address types
#define P2PKH 0
#define P2SH 1
#define BECH32 2

class Secp256K1_AVX512 {
 public:
  Secp256K1_AVX512();
  ~Secp256K1_AVX512();
  void Init();

  // Single point operations
  Point ComputePublicKey(Int *privKey);
  Point NextKey(Point &key);
  void Check();
  bool EC(Point &p);
  Int GetY(Int x, bool isEven);

  // 16-way parallel operations for maximum Bitcoin puzzle performance
  void ComputePublicKeys16(Int *privKeys[16], Point *pubKeys[16]);
  void GetHash160_16(int type, bool compressed, Point *keys[16],
                     uint8_t *hashes[16]);

  // Hash160 operations
  void GetHash160(int type, bool compressed, Point &k0, Point &k1, Point &k2,
                  Point &k3, uint8_t *h0, uint8_t *h1, uint8_t *h2,
                  uint8_t *h3);
  void GetHash160(int type, bool compressed, Point &pubKey,
                  unsigned char *hash);

  // Address generation
  std::string GetAddress(int type, bool compressed, Point &pubKey);
  std::string GetAddress(int type, bool compressed, unsigned char *hash160);
  std::vector<std::string> GetAddress(int type, bool compressed,
                                      unsigned char *h1, unsigned char *h2,
                                      unsigned char *h3, unsigned char *h4);
  std::string GetPrivAddress(bool compressed, Int &privKey);
  std::string GetPublicKeyHex(bool compressed, Point &p);
  Point ParsePublicKeyHex(std::string str, bool &isCompressed);

  bool CheckPudAddress(std::string address);
  static Int DecodePrivateKey(char *key, bool *compressed);

  // Core ECC operations
  Point Add(Point &p1, Point &p2);
  Point Add2(Point &p1, Point &p2);
  Point AddDirect(Point &p1, Point &p2);
  Point Double(Point &p);
  Point DoubleDirect(Point &p);

  // 16-way parallel ECC operations for Sapphire Rapids
  void Add16_AVX512(Point *p1[16], Point *p2[16], Point *results[16]);
  void Double16_AVX512(Point *points[16], Point *results[16]);
  void ComputePublicKeys16_Optimized(Int *privKeys[16], Point *pubKeys[16]);

  Point G;    // Generator
  Int order;  // Curve order

 private:
  uint8_t GetByte(std::string &str, int idx);

  // AVX-512 optimized lookup tables for Sapphire Rapids
  Point GTable[256 * 32];  // Generator table

  // 16-way parallel modular arithmetic helpers
  void ModAdd16_AVX512(__m512i *a, __m512i *b, __m512i *result);
  void ModSub16_AVX512(__m512i *a, __m512i *b, __m512i *result);
  void ModMul16_AVX512(__m512i *a, __m512i *b, __m512i *result);
  void ModSquare16_AVX512(__m512i *a, __m512i *result);
  void ModInv16_AVX512(__m512i *a, __m512i *result);
};

#endif  // SECP256K1_AVX512_H
