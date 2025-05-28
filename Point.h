#ifndef POINTH
#define POINTH

#include <immintrin.h>  // Dla instrukcji AVX-512
#include <omp.h>        // Dla obsługi wielowątkowości

#include "Int.h"

class Point {
 public:
  // Standardowe konstruktory
  Point();
  Point(Int *cx, Int *cy, Int *cz);
  Point(Int *cx, Int *cz);
  Point(const Point &p);
  ~Point();

  // Podstawowe operacje
  bool IsZero() const;
  bool Equals(const Point &p) const;
  void Set(const Point &p);
  void Set(Int *cx, Int *cy, Int *cz);
  void Clear();
  void Reduce();
  void Normalize();

  // Operatory dla wygodniejszej pracy
  bool operator==(const Point &p) const;
  Point &operator=(const Point &p);

  // Metody zoptymalizowane dla AVX-512
  void ReduceFast();  // Szybsza redukcja z wykorzystaniem AVX-512

  // Nowe metody do wsadowego przetwarzania
  static void BatchReduce(Point *points, int count);
  static void BatchNormalize(Point *points, int count);

  // Prefetching dla lepszej wydajności pamięci podręcznej
  void Prefetch(int hint = _MM_HINT_T0) const;

  // Funkcje pomocnicze
  bool IsOnCurve() const;   // Sprawdza czy punkt leży na krzywej
  bool IsInfinity() const;  // Sprawdza czy to punkt w nieskończoności

  // Współrzędne punktu
  Int x;
  Int y;
  Int z;

 private:
  // Pomocnicze metody do optymalizacji
  void ComputeZ1Inv(Int &zInv) const;
};

// Funkcje wsadowe (statyczne)
void PointBatchOperation(Point *points, int count, void (*operation)(Point &));

#endif  // POINTH
