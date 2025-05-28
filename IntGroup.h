#ifndef INTGROUPH
#define INTGROUPH

#include <immintrin.h>
#include <omp.h>

#include <vector>

#include "Int.h"

// Cache line alignment for optimal Xeon 8488C performance
#define CACHE_ALIGN alignas(64)
#define FORCE_INLINE __attribute__((always_inline)) inline

class IntGroup {
 public:
  IntGroup(int size);
  ~IntGroup();

  void Set(Int *pts);
  void ModInv();

  // AVX-512 optimized versions for Xeon 8488C
  void ParallelModInv();
  void VectorizedModInv();

  // Performance monitoring
  uint64_t GetOperationCount() const { return operation_count; }
  void ResetOperationCount() { operation_count = 0; }

 private:
  CACHE_ALIGN Int *ints;
  CACHE_ALIGN Int *subp;
  int size;
  uint64_t operation_count;

  // Optimized helper functions
  FORCE_INLINE void PrefetchMemory(int index);
};

#endif  // INTGROUPH
