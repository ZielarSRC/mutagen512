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

  // AVX-512 optimized batch operations for Xeon 8488C
  void ParallelModInv();
  void VectorizedModInv();
  void OptimizedModInv();

  // Advanced batch operations leveraging Xeon 8488C capabilities
  static void BatchModInv(IntGroup *groups, int group_count);
  static void ParallelBatchModInv(IntGroup *groups, int group_count,
                                  int num_threads);

  // Memory management optimized for Xeon 8488C cache hierarchy
  void PrefetchData();
  void AlignedMemoryOperations();

  // Performance monitoring
  uint64_t GetOperationCount() const { return operation_count; }
  void ResetOperationCount() { operation_count = 0; }

 private:
  CACHE_ALIGN Int *ints;
  CACHE_ALIGN Int *subp;
  CACHE_ALIGN Int *temp_buffer;  // Additional buffer for vectorized operations
  int size;
  uint64_t operation_count;

  // Xeon 8488C specific optimizations
  void OptimizedForwardPass();
  void OptimizedBackwardPass();
  void VectorizedForwardPass();
  void VectorizedBackwardPass();

  // Memory prefetching strategies
  FORCE_INLINE void PrefetchForward(int index);
  FORCE_INLINE void PrefetchBackward(int index);

  // AVX-512 optimized helper functions
  void AVX512ModMulBatch(Int *a, Int *b, Int *result, int count);
  void ParallelChunkProcess(int chunk_start, int chunk_end);
};

// Namespace for AVX-512 specialized operations
namespace IntGroupAVX512 {
void BatchSetup(IntGroup *groups, int count);
void VectorizedInversion(Int *data, int size);
void ParallelModularInversion(Int *data, int size, int num_threads);
}  // namespace IntGroupAVX512

#endif  // INTGROUPH
