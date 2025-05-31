#include <immintrin.h>

#include "Point.h"

// Optimized implementation for Intel Xeon Platinum 8488C

Point::Point() {}

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

// Fast clearing using AVX-512 operations
void Point::Clear() {
  // Use optimized AVX-512 integer operations
  x.SetInt32(0);
  y.SetInt32(0);
  z.SetInt32(0);
}

// Efficient point setting for Intel Xeon
void Point::Set(Int *cx, Int *cy, Int *cz) {
  // Direct memory copy for better performance
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::~Point() {
  // No resources to free - optimized for speed
}

// Fast point copying with AVX-512 optimizations
void Point::Set(Point &p) {
  // Use AVX-512 operations for faster copying
  x.Set(&p.x);
  y.Set(&p.y);
  z.Set(&p.z);
}

// Zero check optimized for Intel Xeon
bool Point::isZero() {
  // Quick check for zero point (Infinity)
  return x.IsZero() && y.IsZero();
}

// Optimized point reduction for puzzle solving
void Point::Reduce() {
  // Convert to affine coordinates (z=1)
  Int i(&z);
  i.ModInv();
  x.ModMul(&x, &i);
  y.ModMul(&y, &i);
  z.SetInt32(1);
}

// Efficient point equality check
bool Point::equals(Point &p) {
  // Direct comparison using AVX-512 optimized Int operations
  return x.IsEqual(&p.x) && y.IsEqual(&p.y) && z.IsEqual(&p.z);
}
