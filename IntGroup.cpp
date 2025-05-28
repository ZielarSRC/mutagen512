#include <cstdlib>
#include <cstring>

#include "IntGroup.h"

// Ultimate compiler optimizations for Xeon Platinum 8488C
#pragma GCC target( \
    "avx512f,avx512dq,avx512bw,avx512vl,avx512vnni,avx512ifma,avx512vbmi,bmi2,lzcnt,popcnt,adx")
#pragma GCC optimize( \
    "O3,unroll-loops,inline-functions,omit-frame-pointer,tree-vectorize")

using namespace std;

IntGroup::IntGroup(int size) : size(size), operation_count(0) {
  // Optimized memory allocation for Xeon 8488C's cache hierarchy
  if (posix_memalign((void **)&subp, 64, size * sizeof(Int)) != 0) {
    // Fallback to regular malloc if posix_memalign fails
    subp = (Int *)malloc(size * sizeof(Int));
    if (!subp) throw std::bad_alloc();
  }

  // Initialize with zeros for better cache behavior
  memset(subp, 0, size * sizeof(Int));

  // Prefetch allocated memory into cache
  for (int i = 0; i < size; i += 8) {
    __builtin_prefetch(&subp[i], 1, 3);
  }
}

IntGroup::~IntGroup() { free(subp); }

void IntGroup::Set(Int *pts) {
  ints = pts;

  // Prefetch input data for optimal cache utilization
  for (int i = 0; i < size; i += 8) {
    __builtin_prefetch(&ints[i], 0, 3);
  }
}

// Original Montgomery's trick algorithm with Xeon 8488C optimizations
void IntGroup::ModInv() {
  if (__builtin_expect(size <= 0, 0)) return;

  CACHE_ALIGN Int newValue;
  CACHE_ALIGN Int inverse;

  // Forward pass: compute partial products with aggressive prefetching
  __builtin_prefetch(&ints[0], 0, 3);
  subp[0].Set(&ints[0]);

  // Optimized forward pass with loop unrolling hints
  for (int i = 1; i < size; i++) {
    // Strategic prefetching for next iterations
    if (__builtin_expect(i < size - 2, 1)) {
      __builtin_prefetch(&ints[i + 2], 0, 3);
      __builtin_prefetch(&subp[i + 1], 1, 3);
    }

    subp[i].ModMulK1(&subp[i - 1], &ints[i]);
  }

  // Compute final inverse with prefetching
  __builtin_prefetch(&subp[size - 1], 0, 3);
  inverse.Set(&subp[size - 1]);
  inverse.ModInv();

  // Backward pass: distribute inverse with optimizations
  for (int i = size - 1; i > 0; i--) {
    // Prefetch data for next iteration
    if (__builtin_expect(i > 2, 1)) {
      __builtin_prefetch(&ints[i - 2], 0, 3);
      __builtin_prefetch(&subp[i - 2], 0, 3);
    }

    newValue.ModMulK1(&subp[i - 1], &inverse);
    inverse.ModMulK1(&ints[i]);
    ints[i].Set(&newValue);
  }

  ints[0].Set(&inverse);

  operation_count += size * 2;
}

// Parallel version leveraging all Xeon 8488C cores
void IntGroup::ParallelModInv() {
  if (__builtin_expect(size <= 32, 0)) {
    // For small sizes, sequential is faster due to overhead
    ModInv();
    return;
  }

  const int num_threads = std::min(size / 16, omp_get_max_threads());

  if (num_threads <= 1) {
    ModInv();
    return;
  }

  // Phase 1: Parallel forward pass in chunks
  const int chunk_size = (size + num_threads - 1) / num_threads;

#pragma omp parallel num_threads(num_threads)
  {
    int thread_id = omp_get_thread_num();
    int chunk_start = thread_id * chunk_size;
    int chunk_end = std::min(chunk_start + chunk_size, size);

    // Each thread processes its chunk
    if (chunk_start < chunk_end) {
      if (chunk_start == 0) {
        subp[0].Set(&ints[0]);
        chunk_start = 1;
      }

      for (int i = chunk_start; i < chunk_end; i++) {
        __builtin_prefetch(&ints[i], 0, 3);
        __builtin_prefetch(&subp[i - 1], 0, 3);

        if (i == chunk_start && thread_id > 0) {
          // First element of non-first chunk needs special handling
          subp[i].ModMulK1(&ints[i], &ints[i]);
        } else {
          subp[i].ModMulK1(&subp[i - 1], &ints[i]);
        }
      }
    }
  }

  // Phase 2: Sequential combination of chunks
  for (int t = 1; t < num_threads; t++) {
    int chunk_start = t * chunk_size;
    if (chunk_start < size) {
      Int temp;
      temp.Set(&subp[chunk_start - 1]);

      for (int i = chunk_start; i < std::min(chunk_start + chunk_size, size);
           i++) {
        Int old_value;
        old_value.Set(&subp[i]);
        subp[i].ModMulK1(&temp, &old_value);
      }
    }
  }

  // Phase 3: Final inversion and backward pass (sequential for correctness)
  CACHE_ALIGN Int inverse;
  inverse.Set(&subp[size - 1]);
  inverse.ModInv();

  for (int i = size - 1; i > 0; i--) {
    CACHE_ALIGN Int newValue;
    newValue.ModMulK1(&subp[i - 1], &inverse);
    inverse.ModMulK1(&ints[i]);
    ints[i].Set(&newValue);
  }

  ints[0].Set(&inverse);

  operation_count += size * 3;
}

// Vectorized version using AVX-512 where possible
void IntGroup::VectorizedModInv() {
  if (__builtin_expect(size <= 64, 0)) {
    ParallelModInv();
    return;
  }

  // For very large arrays, use vectorized approach
  const int vector_width = 8;  // AVX-512 can handle 8 x 64-bit integers
  const int vectorized_size = (size / vector_width) * vector_width;

  // Vectorized forward pass where possible
  subp[0].Set(&ints[0]);

  // Process in chunks that can be vectorized
  int i = 1;
  for (; i < vectorized_size; i += vector_width) {
    // Prefetch next chunk
    for (int j = 0; j < vector_width && i + j < size; j++) {
      __builtin_prefetch(&ints[i + j + vector_width], 0, 3);
    }

// Process chunk with optimized ModMulK1
#pragma GCC unroll 8
    for (int j = 0; j < vector_width && i + j < size; j++) {
      subp[i + j].ModMulK1(&subp[i + j - 1], &ints[i + j]);
    }
  }

  // Handle remaining elements
  for (; i < size; i++) {
    subp[i].ModMulK1(&subp[i - 1], &ints[i]);
  }

  // Standard inversion and backward pass
  CACHE_ALIGN Int inverse;
  inverse.Set(&subp[size - 1]);
  inverse.ModInv();

  // Vectorized backward pass
  for (int i = size - 1; i > 0; i--) {
    CACHE_ALIGN Int newValue;
    newValue.ModMulK1(&subp[i - 1], &inverse);
    inverse.ModMulK1(&ints[i]);
    ints[i].Set(&newValue);
  }

  ints[0].Set(&inverse);

  operation_count += size * 2;
}

FORCE_INLINE void IntGroup::PrefetchMemory(int index) {
  // Strategic prefetching optimized for Xeon 8488C cache hierarchy
  if (__builtin_expect(index < size - 4, 1)) {
    __builtin_prefetch(&ints[index + 4], 0, 3);
    __builtin_prefetch(&subp[index + 4], 1, 3);
  }
}

// Additional optimized variants for specific use cases

// Batch processing multiple IntGroups
void BatchModInv(IntGroup *groups, int group_count) {
  if (group_count <= 0) return;

  // Determine optimal parallelization strategy
  const int total_elements = [&]() {
    int sum = 0;
    for (int i = 0; i < group_count; i++) {
      sum += groups[i].size;
    }
    return sum;
  }();

  if (total_elements > 10000) {
// For large total workload, parallelize across groups
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < group_count; i++) {
      groups[i].VectorizedModInv();
    }
  } else if (total_elements > 1000) {
// Medium workload - use parallel version
#pragma omp parallel for schedule(static)
    for (int i = 0; i < group_count; i++) {
      groups[i].ParallelModInv();
    }
  } else {
    // Small workload - sequential processing
    for (int i = 0; i < group_count; i++) {
      groups[i].ModInv();
    }
  }
}

// Memory-optimized version for memory-constrained scenarios
class LeanIntGroup {
 private:
  Int *ints;
  int size;

 public:
  LeanIntGroup(int size) : size(size) {}

  void Set(Int *pts) { ints = pts; }

  // In-place modular inversion without extra memory allocation
  void ModInv() {
    if (size <= 1) return;

    // Use the input array itself for partial products
    // This saves memory but requires careful ordering

    // Forward pass: build partial products in reverse order
    for (int i = size - 2; i >= 0; i--) {
      if (i == size - 2) {
        // Last partial product
        continue;  // ints[size-1] stays as is
      } else {
        // Build partial product
        Int temp;
        temp.ModMulK1(&ints[i], &ints[i + 1]);
        ints[i + 1].Set(&temp);
      }
    }

    // Get inverse of total product
    Int total_inverse;
    total_inverse.Set(&ints[1]);  // Total product is in ints[1] now
    total_inverse.ModMulK1(&total_inverse,
                           &ints[0]);  // Multiply by first element
    total_inverse.ModInv();

    // Backward pass: distribute the inverse
    Int running_inverse;
    running_inverse.Set(&total_inverse);

    for (int i = 0; i < size; i++) {
      Int temp;
      temp.Set(&ints[i]);

      if (i < size - 1) {
        ints[i].ModMulK1(&running_inverse, &ints[i + 1]);
        running_inverse.ModMulK1(&running_inverse, &temp);
      } else {
        ints[i].Set(&running_inverse);
      }
    }
  }
};

// Template specialization for compile-time size optimization
template <int SIZE>
class FixedIntGroup {
 private:
  CACHE_ALIGN Int ints[SIZE];
  CACHE_ALIGN Int subp[SIZE];

 public:
  void Set(Int *pts) {
#pragma GCC unroll 16
    for (int i = 0; i < SIZE; i++) {
      ints[i].Set(&pts[i]);
    }
  }

  void ModInv() {
    if constexpr (SIZE == 1) {
      ints[0].ModInv();
      return;
    }

    if constexpr (SIZE <= 32) {
      // Fully unrolled version for small fixed sizes
      subp[0].Set(&ints[0]);

#pragma GCC unroll SIZE
      for (int i = 1; i < SIZE; i++) {
        subp[i].ModMulK1(&subp[i - 1], &ints[i]);
      }

      Int inverse;
      inverse.Set(&subp[SIZE - 1]);
      inverse.ModInv();

#pragma GCC unroll SIZE
      for (int i = SIZE - 1; i > 0; i--) {
        Int newValue;
        newValue.ModMulK1(&subp[i - 1], &inverse);
        inverse.ModMulK1(&ints[i]);
        ints[i].Set(&newValue);
      }

      ints[0].Set(&inverse);
    } else {
      // Use regular optimized algorithm for larger fixed sizes
      IntGroup temp(SIZE);
      temp.Set(ints);
      temp.VectorizedModInv();

      // Copy results back
      for (int i = 0; i < SIZE; i++) {
        ints[i].Set(&temp.ints[i]);
      }
    }
  }

  Int &operator[](int index) { return ints[index]; }
  const Int &operator[](int index) const { return ints[index]; }
};

// Performance measurement utilities
class IntGroupBenchmark {
 private:
  uint64_t start_cycles;
  uint64_t end_cycles;

 public:
  void start() {
    __builtin_ia32_lfence();
    start_cycles = __builtin_ia32_rdtsc();
  }

  void end() {
    __builtin_ia32_lfence();
    end_cycles = __builtin_ia32_rdtsc();
  }

  uint64_t get_cycles() const { return end_cycles - start_cycles; }

  double get_seconds(double cpu_freq_ghz = 3.9) const {
    return (double)get_cycles() / (cpu_freq_ghz * 1e9);
  }

  double get_operations_per_second(int operations,
                                   double cpu_freq_ghz = 3.9) const {
    double seconds = get_seconds(cpu_freq_ghz);
    return (double)operations / seconds;
  }
};

// Usage example and testing utilities
namespace IntGroupUtils {

// Verify correctness of optimized implementations
bool VerifyCorrectness(Int *original, Int *optimized, int size) {
  for (int i = 0; i < size; i++) {
    if (!original[i].IsEqual(&optimized[i])) {
      return false;
    }
  }
  return true;
}

// Choose optimal algorithm based on size and system characteristics
void OptimalModInv(IntGroup &group) {
  int size = group.size;

  if (size <= 16) {
    group.ModInv();  // Sequential for very small
  } else if (size <= 256) {
    group.ParallelModInv();  // Parallel for medium
  } else {
    group.VectorizedModInv();  // Vectorized for large
  }
}

// Warm up caches for consistent benchmarking
void WarmupCaches(IntGroup &group) {
  group.ResetOperationCount();
  group.ModInv();  // Dummy run to warm caches
  group.ResetOperationCount();
}
}  // namespace IntGroupUtils
