#ifndef INTGROUPH
#define INTGROUPH

#include <immintrin.h>  // For AVX-512 intrinsics

#include <vector>

#include "Int.h"

// Optimized for AVX-512 on Intel Xeon Platinum 8488C
class alignas(64) IntGroup {
 public:
  // Constructor with proper alignment for AVX-512
  IntGroup(int size);
  ~IntGroup();

  // AVX-512 optimized batch operations
  void Set(Int *pts);
  void ModInv();

  // Additional AVX-512 optimized batch methods
  void BatchModMul(Int *a, Int *b, Int *result, int count);
  void BatchModAdd(Int *a, Int *b, Int *result, int count);
  void BatchModSub(Int *a, Int *b, Int *result, int count);

 private:
  Int *ints;
  alignas(64) Int *subp;  // Aligned for AVX-512
  int size;

  // Helper methods for AVX-512 optimization
  void AlignedModInv();
};

#endif  // INTGROUPH
