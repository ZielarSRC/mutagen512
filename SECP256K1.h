#ifndef SECP256K1H
#define SECP256K1H

#include <immintrin.h>

#include <array>
#include <string>
#include <vector>

#include "Point.h"

// Address type
#define P2PKH 0
#define P2SH 1
#define BECH32 2

// Sapphire Rapids optimizations
#define SIMD_BATCH_SIZE 8  // Process 8 keys simultaneously with AVX-512

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

  // Batch operations for mutation optimization
  void ComputePublicKeysBatch(
      const std::array<Int *, SIMD_BATCH_SIZE> &privKeys,
      std::array<Point *, SIMD_BATCH_SIZE> &pubKeys);

  void GetHash160(int type, bool compressed, Point &k0, Point &k1, Point &k2,
                  Point &k3, uint8_t *h0, uint8_t *h1, uint8_t *h2,
                  uint8_t *h3);

  void GetHash160(int type, bool compressed, Point &pubKey,
                  unsigned char *hash);

  void GetHash160Batch(int type, bool compressed,
                       const std::array<Point *, SIMD_BATCH_SIZE> &keys,
                       std::array<uint8_t *, SIMD_BATCH_SIZE> &hashes);

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

  Point Add(Point &p1, Point &p2);
  Point Add2(Point &p1, Point &p2);
  Point AddDirect(Point &p1, Point &p2);
  Point Double(Point &p);
  Point DoubleDirect(Point &p);

  // High-performance key mutation for puzzle challenge
  void MutateKeyRange(const Int &baseKey, uint64_t startOffset,
                      uint64_t endOffset, const std::string &targetHash160);

  Point G;    // Generator
  Int order;  // Curve order

 private:
  uint8_t GetByte(std::string &str, int idx);

  // Original table size optimized for cache efficiency
  alignas(64) Point GTable[256 * 32];  // 8192 points, cache-aligned

  // Mutation helpers
  alignas(64) Point
      MutationTable[256];  // Pre-computed 2^i points for bit flips

  // AVX-512 constants for vectorized modular arithmetic
  alignas(64) __m512i ModulusVec[8];
  alignas(64) __m512i OrderVec[8];
};

#endif  // SECP256K1H
