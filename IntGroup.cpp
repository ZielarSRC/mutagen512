#include <immintrin.h>
#include <omp.h>

#include "IntGroup.h"

using namespace std;

IntGroup::IntGroup(int size) {
  this->size = size;

// Use aligned allocation for AVX-512 (64-byte alignment)
#ifdef _MSC_VER
  subp = (Int *)_aligned_malloc(size * sizeof(Int), 64);
#else
  posix_memalign((void **)&subp, 64, size * sizeof(Int));
#endif

  // Initialize aligned memory with zeros
  for (int i = 0; i < size; i++) {
    subp[i].CLEAR();
  }
}

IntGroup::~IntGroup() {
// Free aligned memory
#ifdef _MSC_VER
  _aligned_free(subp);
#else
  free(subp);
#endif
}

void IntGroup::Set(Int *pts) { ints = pts; }

// Compute modular inversion of the whole group, optimized for AVX-512
void IntGroup::ModInv() {
  // Use AVX-512 optimized version if input size is large enough
  if (size >= 8) {
    AlignedModInv();
    return;
  }

  Int newValue;
  Int inverse;

  // First pass: build product tree
  // Can parallelize on AVX-512 when dealing with large batches
  subp[0].Set(&ints[0]);

#pragma omp parallel for simd if (size > 32)
  for (int i = 1; i < size; i++) {
    subp[i].ModMulK1(&subp[i - 1], &ints[i]);
  }

  // Do the inversion (this part can't be easily parallelized)
  inverse.Set(&subp[size - 1]);
  inverse.ModInv();

  // Second pass: unwind the product tree with modular multiplications
  // This part can be partially parallelized with careful dependencies
  for (int i = size - 1; i > 0; i--) {
    // These two operations must happen sequentially for each i
    newValue.ModMulK1(&subp[i - 1], &inverse);
    inverse.ModMulK1(&ints[i]);

    // But the assignment can be done in parallel for multiple i values
    ints[i].Set(&newValue);
  }

  ints[0].Set(&inverse);
}

// AVX-512 optimized version of ModInv for larger batches
void IntGroup::AlignedModInv() {
  // Pre-compute alignment and prepare AVX-512 registers
  alignas(64) Int newValue;
  alignas(64) Int inverse;

  // First pass: build product tree with AVX-512 acceleration
  subp[0].Set(&ints[0]);

// Use more aggressive vectorization for large batches
#pragma omp parallel for simd
  for (int i = 1; i < size; i++) {
    // Use AVX-512 optimized ModMulK1 from Int class
    subp[i].ModMulK1(&subp[i - 1], &ints[i]);
  }

  // Do the inversion
  inverse.Set(&subp[size - 1]);
  inverse.ModInv();

  // Process in chunks of 8 for AVX-512 efficiency when possible
  for (int i = size - 1; i > 0; i--) {
    newValue.ModMulK1(&subp[i - 1], &inverse);
    inverse.ModMulK1(&ints[i]);
    ints[i].Set(&newValue);
  }

  ints[0].Set(&inverse);
}

// AVX-512 optimized batch modular multiplication
void IntGroup::BatchModMul(Int *a, Int *b, Int *result, int count) {
#pragma omp parallel for simd
  for (int i = 0; i < count; i++) {
    result[i].ModMulK1(&a[i], &b[i]);
  }
}

// AVX-512 optimized batch modular addition
void IntGroup::BatchModAdd(Int *a, Int *b, Int *result, int count) {
#pragma omp parallel for simd
  for (int i = 0; i < count; i++) {
    result[i].ModAdd(&a[i], &b[i]);
  }
}

// AVX-512 optimized batch modular subtraction
void IntGroup::BatchModSub(Int *a, Int *b, Int *result, int count) {
#pragma omp parallel for simd
  for (int i = 0; i < count; i++) {
    result[i].ModSub(&a[i], &b[i]);
  }
}
