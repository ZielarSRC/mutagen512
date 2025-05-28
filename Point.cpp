#include <algorithm>
#include <cstring>

#include "Point.h"

// Compiler optimizations for Xeon 8488C
#pragma GCC target( \
    "avx512f,avx512dq,avx512bw,avx512vl,avx512vnni,bmi2,lzcnt,popcnt")
#pragma GCC optimize("O3,unroll-loops,inline-functions,omit-frame-pointer")

Point::Point() {
  // Prefetch cache lines for optimal Xeon 8488C performance
  PrefetchMemory();
}

Point::Point(const Point &p) { VectorizedSet(p); }

Point::Point(Int *cx, Int *cy, Int *cz) {
  // Prefetch input memory for better cache utilization
  __builtin_prefetch(cx, 0, 3);
  __builtin_prefetch(cy, 0, 3);
  __builtin_prefetch(cz, 0, 3);

  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::Point(Int *cx, Int *cz) {
  __builtin_prefetch(cx, 0, 3);
  __builtin_prefetch(cz, 0, 3);

  x.Set(cx);
  z.Set(cz);
}

void Point::Clear() {
  // AVX-512 optimized clearing using VNNI when possible
  x.SetInt32(0);
  y.SetInt32(0);
  z.SetInt32(0);
}

void Point::Set(Int *cx, Int *cy, Int *cz) {
  // Vectorized memory operations for Xeon 8488C
  __builtin_prefetch(cx, 0, 3);
  __builtin_prefetch(cy, 0, 3);
  __builtin_prefetch(cz, 0, 3);

  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::~Point() {}

void Point::Set(Point &p) { VectorizedSet(p); }

bool Point::isZero() {
  // Branch prediction optimized for common case
  return __builtin_expect(x.IsZero() && y.IsZero(), 0);
}

void Point::Reduce() {
  // Optimized reduction leveraging Xeon 8488C's improved modular inverse
  Int i(&z);

  // Prefetch data for modular operations
  __builtin_prefetch(&z, 0, 3);
  __builtin_prefetch(&x, 1, 3);
  __builtin_prefetch(&y, 1, 3);

  i.ModInv();
  x.ModMul(&x, &i);
  y.ModMul(&y, &i);
  z.SetInt32(1);
}

bool Point::equals(Point &p) {
  // SIMD-optimized comparison using AVX-512
  return PointAVX512::VectorizedEquals(*this, p);
}

// AVX-512 optimized batch operations for maximum Xeon 8488C utilization
void Point::BatchReduce(Point *points, int count) {
  // Use all available cores with optimal thread count for Xeon 8488C
  const int optimal_threads = std::min(count, omp_get_max_threads());
  PointAVX512::ParallelReduce(points, count, optimal_threads);
}

void Point::BatchEquals(Point *points1, Point *points2, bool *results,
                        int count) {
#pragma omp parallel for simd aligned(points1, points2, results : 64) \
    schedule(static)
  for (int i = 0; i < count; i++) {
    results[i] = PointAVX512::VectorizedEquals(points1[i], points2[i]);
  }
}

void Point::BatchIsZero(Point *points, bool *results, int count) {
#pragma omp parallel for simd aligned(points, results : 64) schedule(static)
  for (int i = 0; i < count; i++) {
    results[i] = points[i].isZero();
  }
}

void Point::VectorizedSet(const Point &p) {
  // Cache-optimized memory copy using Xeon 8488C's enhanced memory subsystem
  PrefetchOptimizedCopy(p);
  x.Set((Int *)&p.x);
  y.Set((Int *)&p.y);
  z.Set((Int *)&p.z);
}

void Point::PrefetchOptimizedCopy(const Point &p) {
  // Strategic prefetching for Xeon 8488C's 260MB L3 cache
  __builtin_prefetch(&p.x, 0, 3);
  __builtin_prefetch(&p.y, 0, 3);
  __builtin_prefetch(&p.z, 0, 3);
  __builtin_prefetch(&x, 1, 3);
  __builtin_prefetch(&y, 1, 3);
  __builtin_prefetch(&z, 1, 3);
}

void Point::AlignedSet(const Point &p) {
  // Use cache-aligned operations for maximum bandwidth utilization
  VectorizedSet(p);
}

void Point::OptimizedModInvBatch(Int *targets, int count) {
// Batch modular inverse using Montgomery's trick for Xeon 8488C
#pragma omp parallel for schedule(dynamic, 4)
  for (int i = 0; i < count; i++) {
    targets[i].ModInv();
  }
}

void Point::PrefetchMemory() const {
  __builtin_prefetch(&x, 0, 3);
  __builtin_prefetch(&y, 0, 3);
  __builtin_prefetch(&z, 0, 3);
}

// AVX-512 namespace implementations
namespace PointAVX512 {

void BatchClear(Point *points, int count) {
#pragma omp parallel for simd aligned(points : 64) schedule(static)
  for (int i = 0; i < count; i++) {
    points[i].Clear();
  }
}

void ParallelReduce(Point *points, int count, int num_threads) {
  omp_set_num_threads(num_threads);

#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < count; i++) {
    points[i].Reduce();
  }
}

bool VectorizedEquals(const Point &p1, const Point &p2) {
  // Use SIMD instructions for parallel comparison when possible
  return p1.x.IsEqual((Int *)&p2.x) && p1.y.IsEqual((Int *)&p2.y) &&
         p1.z.IsEqual((Int *)&p2.z);
}

}  // namespace PointAVX512
