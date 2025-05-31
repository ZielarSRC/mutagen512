#include <omp.h>

#include "Point.h"

Point::Point() {
  // Default constructor - no initialization needed
}

Point::Point(const Point &p) {
  // Use AVX-512 operations from Int class
  x.Set((Int *)&p.x);
  y.Set((Int *)&p.y);
  z.Set((Int *)&p.z);
}

Point::Point(Int *cx, Int *cy, Int *cz) {
  // Use AVX-512 operations from Int class
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::Point(Int *cx, Int *cz) {
  // Use AVX-512 operations from Int class
  x.Set(cx);
  z.Set(cz);
}

void Point::Clear() {
  // Fast AVX-512 zeroing using Int class methods
  x.SetInt32(0);
  y.SetInt32(0);
  z.SetInt32(0);
}

void Point::Set(Int *cx, Int *cy, Int *cz) {
  // Use AVX-512 operations from Int class
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::~Point() {
  // Destructor - no cleanup needed
}

void Point::Set(Point &p) {
  // Use AVX-512 operations from Int class
  x.Set(&p.x);
  y.Set(&p.y);
  z.Set(&p.z);
}

bool Point::isZero() {
  // Optimized for AVX-512 by using Int's IsZero which uses _mm512_reduce_or_epi64
  return x.IsZero() && y.IsZero();
}

void Point::Reduce() {
  // Use ModInv and ModMul from Int class which are optimized for AVX-512
  Int i(&z);
  i.ModInv();

// These operations can execute in parallel
#pragma omp parallel sections
  {
#pragma omp section
    { x.ModMul(&x, &i); }

#pragma omp section
    { y.ModMul(&y, &i); }
  }

  z.SetInt32(1);
}

bool Point::equals(Point &p) {
  // Optimized for AVX-512 by using Int's IsEqual which uses _mm512_cmpeq_epi64_mask
  return x.IsEqual(&p.x) && y.IsEqual(&p.y) && z.IsEqual(&p.z);
}

// New batch methods for AVX-512

void Point::BatchReduce(Point *points, int count) {
// Process multiple points in parallel using AVX-512
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    points[i].Reduce();
  }
}

void Point::BatchEquals(Point *points1, Point *points2, bool *results, int count) {
// Compare multiple points in parallel using AVX-512
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    results[i] = points1[i].equals(points2[i]);
  }
}
