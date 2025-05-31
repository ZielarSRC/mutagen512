#ifndef POINTH
#define POINTH

#include <immintrin.h>  // For AVX-512 intrinsics

#include "Int.h"

// Point class optimized for AVX-512 operations on Intel Xeon Platinum 8488C
class alignas(64) Point {
 public:
  // Constructors optimized for AVX-512
  Point();
  Point(Int *cx, Int *cy, Int *cz);
  Point(Int *cx, Int *cz);
  Point(const Point &p);
  ~Point();

  // Fast AVX-512 optimized operations
  bool isZero();
  bool equals(Point &p);
  void Set(Point &p);
  void Set(Int *cx, Int *cy, Int *cz);
  void Clear();
  void Reduce();

  // AVX-512 optimized coordinate storage
  alignas(64) Int x;
  alignas(64) Int y;
  alignas(64) Int z;

  // Batch processing methods for AVX-512
  static void BatchReduce(Point *points, int count);
  static void BatchEquals(Point *points1, Point *points2, bool *results, int count);
};

#endif  // POINTH
