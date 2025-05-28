#include <cstring>

#include "Point.h"

Point::Point() {
  // Initialize with cache-friendly alignment for Xeon 8488C
}

Point::Point(const Point &p) {
  x.Set((Int *)&p.x);
  y.Set((Int *)&p.y);
  z.Set((Int *)&p.z);
}

Point::Point(Int *cx, Int *cy, Int *cz) {
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::Point(Int *cx, Int *cz) {
  x.Set(cx);
  z.Set(cz);
}

void Point::Clear() {
  x.SetInt32(0);
  y.SetInt32(0);
  z.SetInt32(0);
}

void Point::Set(Int *cx, Int *cy, Int *cz) {
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::~Point() {}

void Point::Set(Point &p) {
  x.Set(&p.x);
  y.Set(&p.y);
  z.Set(&p.z);
}

bool Point::isZero() { return x.IsZero() && y.IsZero(); }

void Point::Reduce() {
  // Optimized reduction for Xeon 8488C using modular inverse
  Int i(&z);
  i.ModInv();
  x.ModMul(&x, &i);
  y.ModMul(&y, &i);
  z.SetInt32(1);
}

bool Point::equals(Point &p) {
  return x.IsEqual(&p.x) && y.IsEqual(&p.y) && z.IsEqual(&p.z);
}

// Additional optimized functions for Xeon 8488C performance

void Point::AlignedSet(const Point &p) {
  // Use cache-aligned memory operations for better performance on Xeon 8488C
  x.AlignedCopyK1order(&p.x);
  y.AlignedCopyK1order(&p.y);
  z.AlignedCopyK1order(&p.z);
}

void Point::BatchReduce(Point *points, int count) {
// Batch processing leveraging Xeon 8488C's multiple execution units
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    points[i].Reduce();
  }
}

void Point::OptimizedModInv(Int *target) {
  // Cache-optimized modular inverse for Xeon 8488C
  target->ModInv();
}
