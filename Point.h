#ifndef POINTH
#define POINTH

#include "Int.h"

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

  // Memory-aligned operations for Xeon 8488C optimization
  void AlignedSet(const Point &p);
  void BatchReduce(Point *points, int count);

  Int x;
  Int y;
  Int z;

 private:
  // Cache-aligned helper functions for Xeon 8488C
  void OptimizedModInv(Int *target);
};

#endif  // POINTH
