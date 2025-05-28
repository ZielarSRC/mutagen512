#include "Point.h"

Point::Point() {
}

Point::Point(const Point &p) {
  x.Set((Int *)&p.x);
  y.Set((Int *)&p.y);
  z.Set((Int *)&p.z);
}

Point::Point(Int *cx,Int *cy,Int *cz) {
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

void Point::Set(Int *cx, Int *cy,Int *cz) {
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::~Point() {
}

void Point::Set(const Point &p) {
  x.Set((Int *)&p.x);
  y.Set((Int *)&p.y);
  z.Set((Int *)&p.z);
}

Point &Point::operator=(const Point &p) {
  if (this != &p) {
    Set(p);
  }
  return *this;
}

bool Point::IsZero() const {
  return ((Int*)&x)->IsZero() && ((Int*)&y)->IsZero();
}

bool Point::IsInfinity() const {
  return ((Int*)&z)->IsZero();
}

bool Point::Equals(const Point &p) const {
  return ((Int*)&x)->IsEqual((Int*)&p.x) && 
         ((Int*)&y)->IsEqual((Int*)&p.y) && 
         ((Int*)&z)->IsEqual((Int*)&p.z);
}

bool Point::operator==(const Point &p) const { 
  return Equals(p); 
}

void Point::Reduce() {
  if (z.IsZero()) return;  // Point at infinity
  
  Int i(&z);
  i.ModInv();
  x.ModMul(&x,&i);
  y.ModMul(&y,&i);
  z.SetInt32(1);
}

void Point::ReduceFast() {
  // Optimized reduction for AVX-512
  if (z.IsZero()) return;  // Point at infinity

  Int zInv;
  ComputeZ1Inv(zInv);

  // Parallelize ModMul operations
#pragma omp parallel sections
  {
#pragma omp section
    {
      Int zInv2;
      zInv2.ModSquare(&zInv);
      x.ModMul(&x, &zInv2);
    }

#pragma omp section
    {
      Int zInv3;
      zInv3.ModMul(&zInv, &zInv);
      zInv3.ModMul(&zInv3, &zInv);
      y.ModMul(&y, &zInv3);
    }
  }

  z.SetInt32(1);
}

void Point::ComputeZ1Inv(Int &zInv) const {
  zInv.Set((Int*)&z);
  zInv.ModInv();
}

void Point::Normalize() {
  Reduce();
}

void Point::Prefetch(int hint) const {
  _mm_prefetch((const char *)&x, (_mm_hint)hint);
  _mm_prefetch((const char *)&y, (_mm_hint)hint);
  _mm_prefetch((const char *)&z, (_mm_hint)hint);
}

bool Point::IsOnCurve() const {
  if (IsInfinity()) return true;

  Int y2, x3, temp;

  // If z != 1, we need to normalize the point
  if (!((Int*)&z)->IsOne()) {
    Point normalized(*this);
    normalized.Normalize();
    return normalized.IsOnCurve();
  }

  y2.ModSquare((Int*)&y);
  x3.ModSquare((Int*)&x);
  x3.ModMul(&x3, (Int*)&x);

  temp.SetInt32(7);  // a = 0, b = 7 for secp256k1
  x3.ModAdd(&temp);

  return y2.IsEqual(&x3);
}

void Point::BatchReduce(Point *points, int count) {
  if (count <= 1) {
    if (count == 1) points[0].Reduce();
    return;
  }

  // Montgomery batch inversion trick implementation
  Int *zs = new Int[count];
  Int *temps = new Int[count];

  // Calculate z-values
  for (int i = 0; i < count; i++) {
    zs[i].Set(&points[i].z);
  }

  // Accumulate products
  temps[0].Set(&zs[0]);
  for (int i = 1; i < count; i++) {
    temps[i].ModMul(&temps[i - 1], &zs[i]);
  }

  // Invert accumulator
  Int acc;
  acc.Set(&temps[count - 1]);
  acc.ModInv();

  // Calculate individual inversions
  for (int i = count - 1; i > 0; i--) {
    temps[i].ModMul(&acc, &temps[i - 1]);
    acc.ModMul(&acc, &zs[i]);
  }
  temps[0].Set(&acc);

  // Apply inversions to points
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    points[i].x.ModMul(&points[i].x, &temps[i]);
    points[i].y.ModMul(&points[i].y, &temps[i]);
    points[i].z.SetInt32(1);
  }

  delete[] zs;
  delete[] temps;
}

void Point::BatchNormalize(Point *points, int count) {
  BatchReduce(points, count);
}

void PointBatchOperation(Point *points, int count, void (*operation)(Point &)) {
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    operation(points[i]);
  }
}