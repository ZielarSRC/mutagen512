#include "Point.h"

Point::Point() {
}

Point::Point(const Point &p) {
  // Selektywny prefetch tylko dla dużych bloków danych
  // dla procesora Sapphire Rapids
  #if NB64BLOCK > 5
  _mm_prefetch((const char*)p.x.bits64, _MM_HINT_T0);
  _mm_prefetch((const char*)p.y.bits64, _MM_HINT_T0);
  _mm_prefetch((const char*)p.z.bits64, _MM_HINT_T0);
  #endif
  
  x.Set((Int *)&p.x);
  y.Set((Int *)&p.y);
  z.Set((Int *)&p.z);
}

Point::Point(Int *cx,Int *cy,Int *cz) {
  // Selektywny prefetch tylko dla dużych bloków danych
  #if NB64BLOCK > 5
  _mm_prefetch((const char*)cx->bits64, _MM_HINT_T0);
  _mm_prefetch((const char*)cy->bits64, _MM_HINT_T0);
  _mm_prefetch((const char*)cz->bits64, _MM_HINT_T0);
  #endif
  
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::Point(Int *cx, Int *cz) {
  // Selektywny prefetch tylko dla dużych bloków danych
  #if NB64BLOCK > 5
  _mm_prefetch((const char*)cx->bits64, _MM_HINT_T0);
  _mm_prefetch((const char*)cz->bits64, _MM_HINT_T0);
  #endif
  
  x.Set(cx);
  z.Set(cz);
}

void Point::Clear() {
  // Używamy istniejących metod z Int, które są już zoptymalizowane
  // dla procesora Sapphire Rapids w pliku Int.cpp
  x.CLEAR();
  y.CLEAR();
  z.CLEAR();
}

void Point::Set(Int *cx, Int *cy,Int *cz) {
  // Selektywny prefetch tylko dla dużych bloków danych
  #if NB64BLOCK > 5
  _mm_prefetch((const char*)cx->bits64, _MM_HINT_T0);
  _mm_prefetch((const char*)cy->bits64, _MM_HINT_T0);
  _mm_prefetch((const char*)cz->bits64, _MM_HINT_T0);
  #endif
  
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::~Point() {
}

void Point::Set(Point &p) {
  // Selektywny prefetch tylko dla dużych bloków danych
  #if NB64BLOCK > 5
  _mm_prefetch((const char*)p.x.bits64, _MM_HINT_T0);
  _mm_prefetch((const char*)p.y.bits64, _MM_HINT_T0);
  _mm_prefetch((const char*)p.z.bits64, _MM_HINT_T0);
  #endif
  
  x.Set(&p.x);
  y.Set(&p.y);
  z.Set(&p.z);
}

bool Point::isZero() {
  return x.IsZero() && y.IsZero();
}

void Point::Reduce() {
  // Selektywny prefetch dla dużych bloków danych
  #if NB64BLOCK > 5
  _mm_prefetch((const char*)z.bits64, _MM_HINT_T0);
  #endif

  Int i(&z);
  i.ModInv();
  
  // Używamy zoptymalizowanych operacji ModMulK1 dla secp256k1
  // które wykorzystują AVX-512 na procesorze Sapphire Rapids
  x.ModMulK1(&x, &i);
  y.ModMulK1(&y, &i);
  
  // Automatyczna redukcja już zawarta w ModMulK1
  z.SetInt32(1);
}

bool Point::equals(Point &p) {
  // Prefetching nie jest potrzebny dla prostej operacji porównania
  return x.IsEqual(&p.x) && y.IsEqual(&p.y) && z.IsEqual(&p.z);
}
