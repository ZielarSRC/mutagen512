#include <immintrin.h>

#include "IntGroup.h"

using namespace std;

IntGroup::IntGroup(int size) {
  this->size = size;
  // Align to 64-byte boundary for optimal AVX-512 performance
  subp = (Int *)aligned_alloc(64, size * sizeof(Int));
}

IntGroup::~IntGroup() { free(subp); }

void IntGroup::Set(Int *pts) { ints = pts; }

// Compute modular inversion of the whole group with batch operations
void IntGroup::ModInv() {
  Int newValue;
  Int inverse;

  // Compute the product chain
  subp[0].Set(&ints[0]);
#pragma omp simd
  for (int i = 1; i < size; i++) {
    subp[i].ModMulK1(&subp[i - 1], &ints[i]);
  }

  // Do the inversion of the product
  inverse.Set(&subp[size - 1]);
  inverse.ModInv();

  // Distribute the result to all elements
  for (int i = size - 1; i > 0; i--) {
    newValue.ModMulK1(&subp[i - 1], &inverse);
    inverse.ModMulK1(&ints[i]);
    ints[i].Set(&newValue);
  }

  ints[0].Set(&inverse);
}
