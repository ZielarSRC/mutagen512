#ifndef INTGROUPH
#define INTGROUPH

#include <immintrin.h>  // Dla instrukcji AVX-512
#include <omp.h>        // Dla obsługi wielowątkowości

#include <vector>

#include "Int.h"

// Optymalne wartości dla Xeon Platinum 8488C
#define BATCH_SIZE_SMALL 16  // Optymalny rozmiar dla małych wsadów
#define BATCH_SIZE_LARGE 64  // Optymalny rozmiar dla dużych wsadów
#define MAX_THREADS 112      // Maksymalna liczba wątków dla 8488C

class IntGroup {
 public:
  // Konstruktory
  IntGroup(int size);
  ~IntGroup();

  // Podstawowe operacje
  void Set(Int* pts);
  void ModInv();

  // Nowe zoptymalizowane metody
  void ModInvBatch();           // Zoptymalizowana dla AVX-512
  void ModInvParallel();        // Wersja z wielowątkowym przetwarzaniem
  void ModInvBatchOptimized();  // Wersja z optymalnym wykorzystaniem pamięci podręcznej

  // Funkcje pomocnicze
  Int* GetElement(int idx);            // Dostęp do konkretnego elementu
  void SetElement(int idx, Int* val);  // Ustawienie wartości elementu

  // Prefetching dla lepszej wydajności pamięci podręcznej
  void PrefetchAll(int hint = _MM_HINT_T0);
  void PrefetchRange(int start, int end, int hint = _MM_HINT_T0);

  // Operacje wsadowe
  void BatchOperation(void (*operation)(Int*));
  void ParallelBatchOperation(void (*operation)(Int*));

  // Uzyskiwanie danych
  int GetSize() const { return size; }
  Int* GetInts() { return ints; }

 private:
  Int* ints;  // Tablica elementów Int
  Int* subp;  // Tablica do obliczeń pomocniczych
  int size;   // Rozmiar grupy

  // Metody pomocnicze dla zoptymalizowanych operacji
  void BatchInversionPrecompute();
  void BatchInversionProcessRange(int start, int end);
  void OptimizeThreads(int& numThreads, int& chunkSize);
};

#endif  // INTGROUPH
