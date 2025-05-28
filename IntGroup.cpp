#include <emmintrin.h>
#include <immintrin.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <thread>

#include "IntGroup.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// Globalnie dostępne zmienne statyczne
static Int _P;         // Field characteristic
static Int _R;         // Montgomery multiplication R
static Int _R2;        // Montgomery multiplication R2
static Int _R3;        // Montgomery multiplication R3
static Int _R4;        // Montgomery multiplication R4
static int32_t Msize;  // Montgomery mult size
static uint32_t MM32;  // 32bits lsb negative inverse of P
static uint64_t MM64;  // 64bits lsb negative inverse of P
#define MSK62 0x3FFFFFFFFFFFFFFF

extern Int _ONE;

#ifdef BMI2
#undef _addcarry_u64
#define _addcarry_u64(c_in, x, y, pz) _addcarryx_u64((c_in), (x), (y), (pz))

inline uint64_t mul128_bmi2(uint64_t x, uint64_t y, uint64_t* high) {
  unsigned long long hi64 = 0;
  unsigned long long lo64 = _mulx_u64((unsigned long long)x, (unsigned long long)y, &hi64);
  *high = (uint64_t)hi64;
  return (uint64_t)lo64;
}

#undef _umul128
#define _umul128(a, b, highptr) mul128_bmi2((a), (b), (highptr))
#endif  // BMI2

// Konstruktor
IntGroup::IntGroup(int size) {
  // Wykryj i dostosuj optymalną liczbę wątków dla procesora
  int maxThreads = omp_get_max_threads();
  omp_set_num_threads(std::min(maxThreads, MAX_THREADS));

  this->size = size;

  // Alokuj pamięć z wyrównaniem do 64 bajtów dla lepszej wydajności AVX-512
  ints = (Int*)_mm_malloc(size * sizeof(Int), 64);
  subp = (Int*)_mm_malloc(size * sizeof(Int), 64);

// Inicjalizacja z użyciem OpenMP
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    ints[i].SetInt32(0);
    subp[i].SetInt32(0);
  }
}

// Destruktor
IntGroup::~IntGroup() {
  // Zwolnij pamięć zaalokowaną z wyrównaniem
  _mm_free(ints);
  _mm_free(subp);
}

// Ustaw wartości w grupie
void IntGroup::Set(Int* pts) {
// Wykorzystaj AVX-512 do szybkiego kopiowania dużych bloków danych
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    ints[i].Set(&pts[i]);

    // Prefetching następnego elementu
    if (i + 1 < size) {
      _mm_prefetch((const char*)&pts[i + 1], _MM_HINT_T0);
    }
  }
}

// Oryginalna metoda ModInv
void IntGroup::ModInv() {
  // Przeprowadź algorytm Montgomery Batch Inversion

  // Inicjalizacja
  subp[0].Set(&ints[0]);

  for (int i = 1; i < size; i++) {
    subp[i].ModMulK1(&subp[i - 1], &ints[i]);
  }

  // Oblicz odwrotność ostatniej wartości
  Int inv;
  inv.Set(&subp[size - 1]);
  inv.ModInv();

  // Przetwarzanie od końca
  for (int i = size - 1; i > 0; i--) {
    ints[i].ModMulK1(&subp[i - 1], &inv);
    inv.ModMulK1(&ints[i], &inv);
  }

  ints[0].Set(&inv);
}

// Zoptymalizowana metoda ModInv dla AVX-512
void IntGroup::ModInvBatch() {
  if (size <= 1) {
    if (size == 1) ints[0].ModInv();
    return;
  }

  // Wykorzystaj prefetching dla lepszej wydajności pamięci podręcznej
  _mm_prefetch((const char*)&ints[0], _MM_HINT_T0);

  // Inicjalizacja
  subp[0].Set(&ints[0]);

// Faza 1: Obliczanie produktów narastających
#pragma omp parallel for
  for (int i = 1; i < size; i += 8) {
    // Prefetching dla lepszej wydajności
    if (i + 8 < size) {
      _mm_prefetch((const char*)&ints[i + 8], _MM_HINT_T0);
    }

    // Przetwarzanie 8 elementów jednocześnie
    for (int j = i; j < std::min(i + 8, size); j++) {
      subp[j].ModMulK1(&subp[j - 1], &ints[j]);
    }
  }

  // Oblicz odwrotność ostatniej wartości
  Int inv;
  inv.Set(&subp[size - 1]);
  inv.ModInv();

// Faza 2: Obliczanie odwrotności
#pragma omp parallel for
  for (int i = size - 8; i >= 0; i -= 8) {
    // Przetwarzanie 8 elementów jednocześnie
    for (int j = std::min(i + 7, size - 1); j >= i; j--) {
      if (j > 0) {
        ints[j].ModMulK1(&subp[j - 1], &inv);
        inv.ModMulK1(&ints[j], &inv);
      } else {
        ints[0].Set(&inv);
      }
    }
  }
}

// Wersja z wielowątkowym przetwarzaniem
void IntGroup::ModInvParallel() {
  if (size <= 1) {
    if (size == 1) ints[0].ModInv();
    return;
  }

  // Określenie optymalnej liczby wątków i rozmiaru partycji
  int numThreads, chunkSize;
  OptimizeThreads(numThreads, chunkSize);

// Faza 1: Przetwarzanie lokalne w obrębie każdej partycji
#pragma omp parallel num_threads(numThreads)
  {
#pragma omp for schedule(dynamic, chunkSize)
    for (int i = 0; i < size; i++) {
      if (i == 0) {
        subp[0].Set(&ints[0]);
      } else {
        subp[i].ModMulK1(&subp[i - 1], &ints[i]);
      }
    }
  }

  // Oblicz odwrotność ostatniej wartości
  Int inv;
  inv.Set(&subp[size - 1]);
  inv.ModInv();

// Faza 2: Wsteczne przetwarzanie
#pragma omp parallel num_threads(numThreads)
  {
#pragma omp for schedule(dynamic, chunkSize)
    for (int i = size - 1; i >= 0; i--) {
      if (i > 0) {
        ints[i].ModMulK1(&subp[i - 1], &inv);

// Synchronizacja jest tutaj konieczna
#pragma omp critical
        { inv.ModMulK1(&ints[i], &inv); }
      } else {
        ints[0].Set(&inv);
      }
    }
  }
}

// Metoda z optymalnym wykorzystaniem pamięci podręcznej
void IntGroup::ModInvBatchOptimized() {
  if (size <= 1) {
    if (size == 1) ints[0].ModInv();
    return;
  }

  BatchInversionPrecompute();

  // Określenie optymalnej liczby wątków i rozmiaru partycji
  int numThreads, chunkSize;
  OptimizeThreads(numThreads, chunkSize);

// Przetwarzanie równoległe partycji
#pragma omp parallel num_threads(numThreads)
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < numThreads; i++) {
      int start = i * (size / numThreads);
      int end = (i == numThreads - 1) ? size : (i + 1) * (size / numThreads);
      BatchInversionProcessRange(start, end);
    }
  }
}

// Metoda pomocnicza do prekomputacji dla zoptymalizowanej wersji
void IntGroup::BatchInversionPrecompute() {
  // Wykorzystaj prefetching i przetwarzanie wsadowe z optymalnym wykorzystaniem pamięci podręcznej
  constexpr int CACHE_LINE_SIZE = 64;  // Rozmiar linii pamięci podręcznej w bajtach
  constexpr int ELEMENTS_PER_CACHE_LINE = CACHE_LINE_SIZE / sizeof(Int*);

  // Inicjalizacja pierwszego elementu
  subp[0].Set(&ints[0]);

  // Faza 1: Produkty narastające
  for (int i = 1; i < size; i++) {
    // Prefetching przyszłych elementów, które będą potrzebne
    if (i + ELEMENTS_PER_CACHE_LINE < size) {
      _mm_prefetch((const char*)&ints[i + ELEMENTS_PER_CACHE_LINE], _MM_HINT_T0);
      _mm_prefetch((const char*)&subp[i + ELEMENTS_PER_CACHE_LINE - 1], _MM_HINT_T0);
    }

    // Korzystaj z zoptymalizowanych funkcji mnożenia z wykorzystaniem BMI2 i AVX-512
    subp[i].ModMulK1(&subp[i - 1], &ints[i]);
  }
}

// Metoda pomocnicza do przetwarzania zakresu indeksów
void IntGroup::BatchInversionProcessRange(int start, int end) {
  if (start >= end) return;

  // Specjalne przetwarzanie dla pierwszego i ostatniego elementu
  if (start == 0 && end == size) {
    // Pełne przetwarzanie całej grupy
    Int inv;
    inv.Set(&subp[size - 1]);
    inv.ModInv();

    for (int i = size - 1; i > 0; i--) {
      ints[i].ModMulK1(&subp[i - 1], &inv);
      inv.ModMulK1(&ints[i], &inv);
    }

    ints[0].Set(&inv);
  } else {
    // Przetwarzanie podgrupy
    Int groupInv;
    if (end == size) {
      // Jeśli jest to ostatnia podgrupa, możemy wykorzystać już obliczoną wartość
      groupInv.Set(&subp[end - 1]);
      groupInv.ModInv();
    } else {
      // W przeciwnym razie musimy obliczyć odwrotność dla tej podgrupy
      groupInv.Set(&subp[end - 1]);
      Int temp;
      if (start > 0) {
        temp.Set(&subp[start - 1]);
        temp.ModInv();
        groupInv.ModMulK1(&temp);
      }
      groupInv.ModInv();
    }

    // Przetwarzanie elementów w tej podgrupie
    for (int i = end - 1; i >= start; i--) {
      if (i > start) {
        ints[i].ModMulK1(&subp[i - 1], &groupInv);
        Int prevSubp;
        if (start > 0) {
          prevSubp.Set(&subp[start - 1]);
        } else {
          prevSubp.SetInt32(1);
        }
        Int div;
        div.Set(&subp[i - 1]);
        // Create a temporary variable to hold the result
        Int temp;
        prevSubp.ModInv(&temp);  // Or if ModInv now takes no parameters: temp = prevSubp; temp.ModInv();
        div.ModMulK1(&temp);
        ints[i].ModMulK1(&div);
      } else if (i == start) {
        if (start == 0) {
          ints[0].Set(&groupInv);
        } else {
          Int prevInv;
          prevInv.Set(&subp[start - 1]);
          prevInv.ModInv();
          ints[start].ModMulK1(&prevInv, &groupInv);
        }
      }

      if (i > start) {
        groupInv.ModMulK1(&ints[i]);
      }
    }
  }
}

// Metoda pomocnicza do optymalizacji liczby wątków i rozmiaru partycji
void IntGroup::OptimizeThreads(int& numThreads, int& chunkSize) {
  // Podstawowa heurystyka oparta na rozmiarze zadania i liczbie dostępnych rdzeni
  int maxThreads = omp_get_max_threads();

  if (size <= 16) {
    numThreads = 1;
    chunkSize = size;
  } else if (size <= 64) {
    numThreads = std::min(4, maxThreads);
    chunkSize = (size + numThreads - 1) / numThreads;
  } else if (size <= 256) {
    numThreads = std::min(16, maxThreads);
    chunkSize = (size + numThreads - 1) / numThreads;
  } else if (size <= 1024) {
    numThreads = std::min(32, maxThreads);
    chunkSize = (size + numThreads - 1) / numThreads;
  } else {
    numThreads = std::min(60, maxThreads);  // Xeon 8488C ma 60 rdzeni
    chunkSize = (size + numThreads - 1) / numThreads;
  }

  // Upewnij się, że rozmiar partycji jest przynajmniej 1
  chunkSize = std::max(1, chunkSize);
}

// Dostęp do konkretnego elementu
Int* IntGroup::GetElement(int idx) {
  if (idx < 0 || idx >= size) {
    return nullptr;
  }
  return &ints[idx];
}

// Ustawienie wartości elementu
void IntGroup::SetElement(int idx, Int* val) {
  if (idx < 0 || idx >= size) {
    return;
  }
  ints[idx].Set(val);
}

// Prefetching dla lepszej wydajności pamięci podręcznej
void IntGroup::PrefetchAll(int hint) {
  for (int i = 0; i < size; i++) {
    _mm_prefetch((const char*)&ints[i], (_mm_hint)hint);
  }
}

// Prefetching dla zakresu elementów
void IntGroup::PrefetchRange(int start, int end, int hint) {
  for (int i = start; i < end && i < size; i++) {
    _mm_prefetch((const char*)&ints[i], (_mm_hint)hint);
  }
}

// Wykonanie operacji na wszystkich elementach grupy
void IntGroup::BatchOperation(void (*operation)(Int*)) {
  for (int i = 0; i < size; i++) {
    operation(&ints[i]);
  }
}

// Równoległe wykonanie operacji na wszystkich elementach grupy
void IntGroup::ParallelBatchOperation(void (*operation)(Int*)) {
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    operation(&ints[i]);
  }
}
