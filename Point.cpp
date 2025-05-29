#include <immintrin.h>

#include "Point.h"

Point::Point() {}

Point::Point(const Point &p) {
// Selektywny prefetch tylko dla dużych bloków danych
#if NB64BLOCK > 5
  _mm_prefetch((const char *)p.x.bits64, _MM_HINT_T0);
  _mm_prefetch((const char *)p.y.bits64, _MM_HINT_T0);
  _mm_prefetch((const char *)p.z.bits64, _MM_HINT_T0);
#endif

  x.Set((Int *)&p.x);
  y.Set((Int *)&p.y);
  z.Set((Int *)&p.z);
}

Point::Point(Int *cx, Int *cy, Int *cz) {
// Selektywny prefetch tylko dla dużych bloków danych
#if NB64BLOCK > 5
  _mm_prefetch((const char *)cx->bits64, _MM_HINT_T0);
  _mm_prefetch((const char *)cy->bits64, _MM_HINT_T0);
  _mm_prefetch((const char *)cz->bits64, _MM_HINT_T0);
#endif

  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::Point(Int *cx, Int *cz) {
// Selektywny prefetch tylko dla dużych bloków danych
#if NB64BLOCK > 5
  _mm_prefetch((const char *)cx->bits64, _MM_HINT_T0);
  _mm_prefetch((const char *)cz->bits64, _MM_HINT_T0);
#endif

  x.Set(cx);
  z.Set(cz);
}

// Konstruktory przyjmujące referencje
Point::Point(Int &cx, Int &cy, Int &cz) {
// Selektywny prefetch tylko dla dużych bloków danych
#if NB64BLOCK > 5
  _mm_prefetch((const char *)cx.bits64, _MM_HINT_T0);
  _mm_prefetch((const char *)cy.bits64, _MM_HINT_T0);
  _mm_prefetch((const char *)cz.bits64, _MM_HINT_T0);
#endif

  x.Set(&cx);
  y.Set(&cy);
  z.Set(&cz);
}

Point::Point(Int &cx, Int &cy) {
// Selektywny prefetch tylko dla dużych bloków danych
#if NB64BLOCK > 5
  _mm_prefetch((const char *)cx.bits64, _MM_HINT_T0);
  _mm_prefetch((const char *)cy.bits64, _MM_HINT_T0);
#endif

  x.Set(&cx);
  y.Set(&cy);
  z.SetInt32(1);  // Domyślnie ustawiamy z=1
}

void Point::Clear() {
  // Używamy publicznych metod zamiast prywatnej CLEAR()
  x.SetInt32(0);
  y.SetInt32(0);
  z.SetInt32(0);
}

void Point::Set(Int *cx, Int *cy, Int *cz) {
// Selektywny prefetch tylko dla dużych bloków danych
#if NB64BLOCK > 5
  _mm_prefetch((const char *)cx->bits64, _MM_HINT_T0);
  _mm_prefetch((const char *)cy->bits64, _MM_HINT_T0);
  _mm_prefetch((const char *)cz->bits64, _MM_HINT_T0);
#endif

  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::~Point() {}

void Point::Set(Point &p) {
// Selektywny prefetch tylko dla dużych bloków danych
#if NB64BLOCK > 5
  _mm_prefetch((const char *)p.x.bits64, _MM_HINT_T0);
  _mm_prefetch((const char *)p.y.bits64, _MM_HINT_T0);
  _mm_prefetch((const char *)p.z.bits64, _MM_HINT_T0);
#endif

  x.Set(&p.x);
  y.Set(&p.y);
  z.Set(&p.z);
}

bool Point::isZero() { return x.IsZero() && y.IsZero(); }

void Point::Reduce() {
// Selektywny prefetch dla dużych bloków danych
#if NB64BLOCK > 5
  _mm_prefetch((const char *)z.bits64, _MM_HINT_T0);
#endif

  // Sprawdzenie czy z jest różne od zera
  if (z.IsZero()) {
    // Punkt w nieskończoności
    Clear();
    return;
  }

  Int i(&z);
  i.ModInv();

  // Używamy konwersji z uint64_t na Int zamiast przekazywania jako wskaźnik
  Int tmp;

  // x = x * z^-1 mod P
  tmp.ModMulK1(&x, &i);
  x.Set(&tmp);

  // y = y * z^-1 mod P
  tmp.ModMulK1(&y, &i);
  y.Set(&tmp);

  // z = 1
  z.SetInt32(1);
}

bool Point::equals(Point &p) {
  // Prefetching nie jest potrzebny dla prostej operacji porównania
  return x.IsEqual(&p.x) && y.IsEqual(&p.y) && z.IsEqual(&p.z);
}
