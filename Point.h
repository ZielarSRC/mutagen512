#ifndef POINTH
#define POINTH

#include <immintrin.h>
#include <omp.h>

#include "Int.h"

// Cache line alignment for Xeon 8488C optimal performance
#define CACHE_ALIGN alignas(64)

class Point {
 public:
  Point();
  Point(Int *cx, Int *cy, Int *cz);
  Point(Int *cx, Int *cz);
  Point(const Point &p);
  ~Point();

  bool isZero();
  bool equals(Point &p);
  void Set(Point &p);
  void Set(Int *cx, Int *cy, Int *cz);
  void Clear();
  void Reduce();

  // AVX-512 optimized batch operations for Xeon 8488C
  static void BatchReduce(Point *points, int count);
  static void BatchEquals(Point *points1, Point *points2, bool *results,
                          int count);
  static void BatchIsZero(Point *points, bool *results, int count);

  // VNNI-optimized vector operations
  void VectorizedSet(const Point &p);
  void PrefetchOptimizedCopy(const Point &p);

  // Memory-aligned operations for Xeon 8488C L3 cache optimization
  void CACHE_ALIGN AlignedSet(const Point &p);

  CACHE_ALIGN Int x;
  CACHE_ALIGN Int y;
  CACHE_ALIGN Int z;

 private:
  // Xeon 8488C specific optimizations
  static void OptimizedModInvBatch(Int *targets, int count);
  void PrefetchMemory() const;
};

// AVX-512 vectorized point operations
namespace PointAVX512 {
void BatchClear(Point *points, int count);
void ParallelReduce(Point *points, int count, int num_threads);
bool VectorizedEquals(const Point &p1, const Point &p2);
}  // namespace PointAVX512

#endif  // POINTH
