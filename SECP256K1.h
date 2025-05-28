#ifndef SECP256K1H
#define SECP256K1H

#include <immintrin.h>  // Dla instrukcji AVX-512
#include <omp.h>        // Dla obsługi wielowątkowości

#include <string>
#include <vector>

#include "Point.h"

// Address type
#define P2PKH 0
#define P2SH 1
#define BECH32 2

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

  // Standardowe funkcje hash i adresowe
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

  // Podstawowe operacje na krzywej eliptycznej
  Point Add(Point &p1, Point &p2);
  Point Add2(Point &p1, Point &p2);
  Point AddDirect(Point &p1, Point &p2);
  Point Double(Point &p);
  Point DoubleDirect(Point &p);

  // Nowe metody zoptymalizowane dla Intel Xeon Platinum 8488C

  // Funkcja do wsadowego przetwarzania punktów z wykorzystaniem AVX-512
  void BatchProcessPoints(Point *points, int numPoints, Point *results);

  // Funkcja do równoległego obliczania wielu kluczy publicznych
  void BatchComputePublicKeys(Int *privKeys, int numKeys, Point *pubKeys);

  // Zoptymalizowana funkcja normalizacji wsadowej dla punktów
  void BatchNormalize(Point *points, int count);

  // Zoptymalizowana wersja GetHash160 dla jednoczesnego przetwarzania 16 punktów
  void GetHash160_AVX512(int type, bool compressed, Point **pubKeys, uint8_t **hashes, int count);

  // Funkcja do wsadowego podwajania wielu punktów
  void BatchDouble(Point *points, int count, Point *results);

  // Funkcja do wsadowego dodawania punktów
  void BatchAdd(Point *points1, Point *points2, int count, Point *results);

  Point G;    // Generator
  Int order;  // Curve order

 private:
  uint8_t GetByte(std::string &str, int idx);

  // Ustawienie optymalnej liczby wątków dla procesora
  void ConfigureThreads();

  // Prefetching danych dla lepszego wykorzystania pamięci podręcznej
  void PrefetchPoint(const Point *p, int hint = _MM_HINT_T0);

  // Wykonanie instrukcji AVX-512 VNNI dla przyspieszenia obliczeń
  void VnniAccelerate(Int *data, int count);

  Point GTable[256 * 32];  // Generator table
};

// Globalne funkcje pomocnicze do przetwarzania równoległego
void sha256_avx512_batch(uint8_t **inputs, int inputLen, uint8_t **outputs, int count);
void ripemd160_avx512_batch(uint8_t **inputs, int inputLen, uint8_t **outputs, int count);

#endif  // SECP256K1H
