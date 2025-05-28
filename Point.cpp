#include "Point.h"
#include <immintrin.h>

Point::Point() {
}

Point::Point(const Point &p) {
  // Prefetch source data for optimal cache performance
  _mm_prefetch((char*)p.x.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p.y.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p.z.bits64, _MM_HINT_T0);
  
  x.Set((Int *)&p.x);
  y.Set((Int *)&p.y);
  z.Set((Int *)&p.z);
}

Point::Point(Int *cx,Int *cy,Int *cz) {
  // Prefetch input data
  _mm_prefetch((char*)cx->bits64, _MM_HINT_T0);
  _mm_prefetch((char*)cy->bits64, _MM_HINT_T0);
  _mm_prefetch((char*)cz->bits64, _MM_HINT_T0);
  
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::Point(Int *cx, Int *cz) {
  // Prefetch input data
  _mm_prefetch((char*)cx->bits64, _MM_HINT_T0);
  _mm_prefetch((char*)cz->bits64, _MM_HINT_T0);
  
  x.Set(cx);
  z.Set(cz);
}

void Point::Clear() {
  // Use vectorized clear operations where possible
  x.SetInt32(0);
  y.SetInt32(0);
  z.SetInt32(0);
  
  // Ensure all high bits are cleared using AVX-512
  #if BISIZE >= 256
  // Clear any remaining high bits efficiently
  _mm512_storeu_si512((__m512i*)x.bits64, _mm512_setzero_si512());
  _mm512_storeu_si512((__m512i*)y.bits64, _mm512_setzero_si512());
  _mm512_storeu_si512((__m512i*)z.bits64, _mm512_setzero_si512());
  #endif
}

void Point::Set(Int *cx, Int *cy,Int *cz) {
  // Prefetch all input data for optimal cache performance
  _mm_prefetch((char*)cx->bits64, _MM_HINT_T0);
  _mm_prefetch((char*)cy->bits64, _MM_HINT_T0);
  _mm_prefetch((char*)cz->bits64, _MM_HINT_T0);
  
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::~Point() {
}

void Point::Set(Point &p) {
  // Prefetch source point data
  _mm_prefetch((char*)p.x.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p.y.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p.z.bits64, _MM_HINT_T0);
  
  x.Set(&p.x);
  y.Set(&p.y);
  z.Set(&p.z);
}

bool Point::isZero() {
  // Prefetch data for zero check
  _mm_prefetch((char*)x.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)y.bits64, _MM_HINT_T0);
  
  return x.IsZero() && y.IsZero();
}

void Point::Reduce() {
  // Prefetch point data for reduction
  _mm_prefetch((char*)x.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)y.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)z.bits64, _MM_HINT_T0);

  Int i(&z);
  i.ModInv();
  
  // Use optimized K1 multiplication for secp256k1
  x.ModMulK1(&x,&i);
  y.ModMulK1(&y,&i);
  
  // Apply fast reduction for secp256k1
  x.ModReduceK1AVX512();
  y.ModReduceK1AVX512();
  
  z.SetInt32(1);
  
  // Memory fence to ensure reduction is complete
  _mm_mfence();
}

bool Point::equals(Point &p) {
  // Prefetch both points' data for comparison
  _mm_prefetch((char*)x.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)y.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)z.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p.x.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p.y.bits64, _MM_HINT_T0);
  _mm_prefetch((char*)p.z.bits64, _MM_HINT_T0);
  
  return x.IsEqual(&p.x) && y.IsEqual(&p.y) && z.IsEqual(&p.z);
}
