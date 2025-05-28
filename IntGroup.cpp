#include <immintrin.h>
#include "IntGroup.h"

using namespace std;

IntGroup::IntGroup(int size) {
  this->size = size;
  subp = (Int *)malloc(size * sizeof(Int));
  
  // Initialize Int objects properly
  for (int i = 0; i < size; i++) {
    new (&subp[i]) Int();
  }
}

IntGroup::~IntGroup() {
  // Properly destroy Int objects before freeing memory
  for (int i = 0; i < size; i++) {
    subp[i].~Int();
  }
  free(subp);
}

void IntGroup::Set(Int *pts) {
  ints = pts;
}

// Compute modular inversion of the whole group - AVX-512 optimized
void IntGroup::ModInv() {

  Int newValue;
  Int inverse;

  // First phase: compute cumulative products with AVX-512 optimizations
  subp[0].Set(&ints[0]);
  
  for (int i = 1; i < size; i++) {
    // Prefetch next elements for better cache performance
    if (i + 1 < size) {
      _mm_prefetch((char*)ints[i + 1].bits64, _MM_HINT_T0);
    }
    if (i + 2 < size) {
      _mm_prefetch((char*)ints[i + 2].bits64, _MM_HINT_T1);
    }
    
    // Use optimized K1 multiplication
    subp[i].ModMulK1(&subp[i - 1], &ints[i]);
    
    // Apply fast reduction for better performance
    subp[i].ModReduceK1AVX512();
  }

  // Do the inversion on the final cumulative product
  inverse.Set(&subp[size - 1]);
  inverse.ModInv();

  // Second phase: back-substitution with AVX-512 optimizations
  for (int i = size - 1; i > 0; i--) {
    // Prefetch next iteration data
    if (i > 1) {
      _mm_prefetch((char*)subp[i - 2].bits64, _MM_HINT_T0);
      _mm_prefetch((char*)ints[i - 1].bits64, _MM_HINT_T0);
    }
    
    // Compute newValue = subp[i-1] * inverse
    newValue.ModMulK1(&subp[i - 1], &inverse);
    
    // Update inverse = inverse * ints[i]
    inverse.ModMulK1(&ints[i]);
    
    // Store result with fast reduction
    ints[i].Set(&newValue);
    ints[i].ModReduceK1AVX512();
  }

  // Set final result
  ints[0].Set(&inverse);
  ints[0].ModReduceK1AVX512();
  
  // Memory fence to ensure all operations complete
  _mm_mfence();
}
