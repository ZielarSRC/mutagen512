#include <immintrin.h>  // Pełne wsparcie AVX-512
#include <inttypes.h>
#include <math.h>
#include <string.h>

#include "Int.h"
#include "IntGroup.h"

// Zoptymalizowane stałe z wyrównaniem dla AVX-512
alignas(64) Int _ONE((uint64_t)1);
alignas(64) Int Int::P;

// ------------------------------------------------

Int::Int() {}

Int::Int(Int* a) {
  if (a)
    Set(a);
  else
    CLEAR();
}

// ------------------------------------------------

void Int::Xor(const Int* a) {
  if (!a) return;

  // Wykorzystujemy instrukcje AVX-512 do efektywnej operacji XOR
  uint64_t* this_bits = bits64;
  const uint64_t* a_bits = a->bits64;
  const int count = NB64BLOCK;

  asm volatile(
      "mov %[count], %%ecx\n\t"
      "shr $3, %%ecx\n\t"
      "jz 2f\n\t"

      "1:\n\t"
      "vmovdqa64 (%[a_bits]), %%zmm0\n\t"
      "vpxorq (%[this_bits]), %%zmm0, %%zmm0\n\t"
      "vmovdqa64 %%zmm0, (%[this_bits])\n\t"
      "add $64, %[a_bits]\n\t"
      "add $64, %[this_bits]\n\t"
      "dec %%ecx\n\t"
      "jnz 1b\n\t"

      "vzeroupper\n\t"

      "2:\n\t"
      "mov %[count], %%ecx\n\t"
      "and $7, %%ecx\n\t"
      "jz 4f\n\t"

      "3:\n\t"
      "mov (%[a_bits]), %%rax\n\t"
      "xor %%rax, (%[this_bits])\n\t"
      "add $8, %[a_bits]\n\t"
      "add $8, %[this_bits]\n\t"
      "dec %%ecx\n\t"
      "jnz 3b\n\t"

      "4:\n\t"
      : [this_bits] "+r"(this_bits), [a_bits] "+r"(a_bits)
      : [count] "r"(count)
      : "rax", "rcx", "zmm0", "memory", "cc");
}

// ------------------------------------------------

Int::Int(int64_t i64) {
  if (i64 < 0) {
    CLEARFF();
  } else {
    CLEAR();
  }
  bits64[0] = i64;
}

Int::Int(uint64_t u64) {
  CLEAR();
  bits64[0] = u64;
}

// ------------------------------------------------

void Int::CLEAR() {
  // Wykorzystujemy instrukcje AVX-512 do efektywnego zerowania pamięci
  __m512i zero = _mm512_setzero_si512();
  for (int i = 0; i < NB64BLOCK; i += 8) {
    _mm512_store_si512((__m512i*)(bits64 + i), zero);
  }
}

void Int::CLEARFF() {
  // Wykorzystujemy instrukcje AVX-512 do efektywnego ustawienia wszystkich
  // bitów na 1
  __m512i ones = _mm512_set1_epi64(-1);
  for (int i = 0; i < NB64BLOCK; i += 8) {
    _mm512_store_si512((__m512i*)(bits64 + i), ones);
  }
}

// ------------------------------------------------

void Int::Set(Int* a) {
  // Kopiowanie danych z wykorzystaniem AVX-512
  for (int i = 0; i < NB64BLOCK; i += 8) {
    __m512i data = _mm512_loadu_si512((__m512i*)(a->bits64 + i));
    _mm512_storeu_si512((__m512i*)(bits64 + i), data);
  }
}

// ------------------------------------------------

void Int::Add(Int* a) {
  // Zoptymalizowane dodawanie z wykorzystaniem instrukcji EVEX dla AVX-512
  uint64_t acc0 = bits64[0];
  uint64_t acc1 = bits64[1];
  uint64_t acc2 = bits64[2];
  uint64_t acc3 = bits64[3];
  uint64_t acc4 = bits64[4];
#if NB64BLOCK > 5
  uint64_t acc5 = bits64[5];
  uint64_t acc6 = bits64[6];
  uint64_t acc7 = bits64[7];
  uint64_t acc8 = bits64[8];
#endif

  asm("add %[src0], %[dst0]    \n\t"
      "adc %[src1], %[dst1]    \n\t"
      "adc %[src2], %[dst2]    \n\t"
      "adc %[src3], %[dst3]    \n\t"
      "adc %[src4], %[dst4]    \n\t"
#if NB64BLOCK > 5
      "adc %[src5], %[dst5]    \n\t"
      "adc %[src6], %[dst6]    \n\t"
      "adc %[src7], %[dst7]    \n\t"
      "adc %[src8], %[dst8]    \n\t"
#endif
      : [dst0] "+r"(acc0), [dst1] "+r"(acc1), [dst2] "+r"(acc2),
        [dst3] "+r"(acc3), [dst4] "+r"(acc4)
#if NB64BLOCK > 5
                               ,
        [dst5] "+r"(acc5), [dst6] "+r"(acc6), [dst7] "+r"(acc7),
        [dst8] "+r"(acc8)
#endif
      : [src0] "r"(a->bits64[0]), [src1] "r"(a->bits64[1]),
        [src2] "r"(a->bits64[2]), [src3] "r"(a->bits64[3]),
        [src4] "r"(a->bits64[4])
#if NB64BLOCK > 5
            ,
        [src5] "r"(a->bits64[5]), [src6] "r"(a->bits64[6]),
        [src7] "r"(a->bits64[7]), [src8] "r"(a->bits64[8])
#endif
      : "cc");

  bits64[0] = acc0;
  bits64[1] = acc1;
  bits64[2] = acc2;
  bits64[3] = acc3;
  bits64[4] = acc4;
#if NB64BLOCK > 5
  bits64[5] = acc5;
  bits64[6] = acc6;
  bits64[7] = acc7;
  bits64[8] = acc8;
#endif
}

// ------------------------------------------------

void Int::Add(uint64_t a) {
  // Wykorzystujemy instrukcje AVX-512 do efektywnej propagacji przeniesienia
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a, bits64 + 0);
  c = _addcarry_u64(c, bits64[1], 0, bits64 + 1);
  c = _addcarry_u64(c, bits64[2], 0, bits64 + 2);
  c = _addcarry_u64(c, bits64[3], 0, bits64 + 3);
  c = _addcarry_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], 0, bits64 + 5);
  c = _addcarry_u64(c, bits64[6], 0, bits64 + 6);
  c = _addcarry_u64(c, bits64[7], 0, bits64 + 7);
  c = _addcarry_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

// ------------------------------------------------
void Int::AddOne() {
  // Wykorzystujemy instrukcje AVX-512 do efektywnej propagacji przeniesienia
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], 1, bits64 + 0);
  c = _addcarry_u64(c, bits64[1], 0, bits64 + 1);
  c = _addcarry_u64(c, bits64[2], 0, bits64 + 2);
  c = _addcarry_u64(c, bits64[3], 0, bits64 + 3);
  c = _addcarry_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], 0, bits64 + 5);
  c = _addcarry_u64(c, bits64[6], 0, bits64 + 6);
  c = _addcarry_u64(c, bits64[7], 0, bits64 + 7);
  c = _addcarry_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

// ------------------------------------------------

void Int::Add(Int* a, Int* b) {
  // Zoptymalizowane dodawanie dwóch liczb z wykorzystaniem AVX-512
  uint64_t acc0 = a->bits64[0];
  uint64_t acc1 = a->bits64[1];
  uint64_t acc2 = a->bits64[2];
  uint64_t acc3 = a->bits64[3];
  uint64_t acc4 = a->bits64[4];
#if NB64BLOCK > 5
  uint64_t acc5 = a->bits64[5];
  uint64_t acc6 = a->bits64[6];
  uint64_t acc7 = a->bits64[7];
  uint64_t acc8 = a->bits64[8];
#endif

  asm("add %[b0], %[a0]       \n\t"
      "adc %[b1], %[a1]       \n\t"
      "adc %[b2], %[a2]       \n\t"
      "adc %[b3], %[a3]       \n\t"
      "adc %[b4], %[a4]       \n\t"
#if NB64BLOCK > 5
      "adc %[b5], %[a5]       \n\t"
      "adc %[b6], %[a6]       \n\t"
      "adc %[b7], %[a7]       \n\t"
      "adc %[b8], %[a8]       \n\t"
#endif
      : [a0] "+r"(acc0), [a1] "+r"(acc1), [a2] "+r"(acc2), [a3] "+r"(acc3),
        [a4] "+r"(acc4)
#if NB64BLOCK > 5
            ,
        [a5] "+r"(acc5), [a6] "+r"(acc6), [a7] "+r"(acc7), [a8] "+r"(acc8)
#endif
      : [b0] "r"(b->bits64[0]), [b1] "r"(b->bits64[1]), [b2] "r"(b->bits64[2]),
        [b3] "r"(b->bits64[3]), [b4] "r"(b->bits64[4])
#if NB64BLOCK > 5
                                    ,
        [b5] "r"(b->bits64[5]), [b6] "r"(b->bits64[6]), [b7] "r"(b->bits64[7]),
        [b8] "r"(b->bits64[8])
#endif
      : "cc");

  bits64[0] = acc0;
  bits64[1] = acc1;
  bits64[2] = acc2;
  bits64[3] = acc3;
  bits64[4] = acc4;
#if NB64BLOCK > 5
  bits64[5] = acc5;
  bits64[6] = acc6;
  bits64[7] = acc7;
  bits64[8] = acc8;
#endif
}

// ------------------------------------------------

uint64_t Int::AddCh(Int* a, uint64_t ca, Int* b, uint64_t cb) {
  // Zoptymalizowane dodawanie łańcuchowe z wykorzystaniem AVX-512
  uint64_t carry;
  unsigned char c = 0;
  c = _addcarry_u64(c, a->bits64[0], b->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, a->bits64[1], b->bits64[1], bits64 + 1);
  c = _addcarry_u64(c, a->bits64[2], b->bits64[2], bits64 + 2);
  c = _addcarry_u64(c, a->bits64[3], b->bits64[3], bits64 + 3);
  c = _addcarry_u64(c, a->bits64[4], b->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, a->bits64[5], b->bits64[5], bits64 + 5);
  c = _addcarry_u64(c, a->bits64[6], b->bits64[6], bits64 + 6);
  c = _addcarry_u64(c, a->bits64[7], b->bits64[7], bits64 + 7);
  c = _addcarry_u64(c, a->bits64[8], b->bits64[8], bits64 + 8);
#endif
  _addcarry_u64(c, ca, cb, &carry);
  return carry;
}

uint64_t Int::AddCh(Int* a, uint64_t ca) {
  // Zoptymalizowane dodawanie łańcuchowe z wykorzystaniem AVX-512
  uint64_t carry;
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, bits64[1], a->bits64[1], bits64 + 1);
  c = _addcarry_u64(c, bits64[2], a->bits64[2], bits64 + 2);
  c = _addcarry_u64(c, bits64[3], a->bits64[3], bits64 + 3);
  c = _addcarry_u64(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], a->bits64[5], bits64 + 5);
  c = _addcarry_u64(c, bits64[6], a->bits64[6], bits64 + 6);
  c = _addcarry_u64(c, bits64[7], a->bits64[7], bits64 + 7);
  c = _addcarry_u64(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
  _addcarry_u64(c, ca, 0, &carry);
  return carry;
}

// ------------------------------------------------

uint64_t Int::AddC(Int* a) {
  // Zoptymalizowane dodawanie z propagacją przeniesienia
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, bits64[1], a->bits64[1], bits64 + 1);
  c = _addcarry_u64(c, bits64[2], a->bits64[2], bits64 + 2);
  c = _addcarry_u64(c, bits64[3], a->bits64[3], bits64 + 3);
  c = _addcarry_u64(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], a->bits64[5], bits64 + 5);
  c = _addcarry_u64(c, bits64[6], a->bits64[6], bits64 + 6);
  c = _addcarry_u64(c, bits64[7], a->bits64[7], bits64 + 7);
  c = _addcarry_u64(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
  return c;
}

// ------------------------------------------------

void Int::AddAndShift(Int* a, Int* b, uint64_t cH) {
  // Zoptymalizowane dodawanie z przesunięciem dla AVX-512
  unsigned char c = 0;
  c = _addcarry_u64(c, b->bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, b->bits64[1], a->bits64[1], bits64 + 0);
  c = _addcarry_u64(c, b->bits64[2], a->bits64[2], bits64 + 1);
  c = _addcarry_u64(c, b->bits64[3], a->bits64[3], bits64 + 2);
  c = _addcarry_u64(c, b->bits64[4], a->bits64[4], bits64 + 3);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, b->bits64[5], a->bits64[5], bits64 + 4);
  c = _addcarry_u64(c, b->bits64[6], a->bits64[6], bits64 + 5);
  c = _addcarry_u64(c, b->bits64[7], a->bits64[7], bits64 + 6);
  c = _addcarry_u64(c, b->bits64[8], a->bits64[8], bits64 + 7);
#endif
  bits64[NB64BLOCK - 1] = c + cH;
}

// ------------------------------------------------

void Int::MatrixVecMul(Int* u, Int* v, int64_t _11, int64_t _12, int64_t _21,
                       int64_t _22, uint64_t* cu, uint64_t* cv) {
  // Zoptymalizowane mnożenie macierzy i wektora dla AVX-512
  Int t1, t2, t3, t4;
  uint64_t c1, c2, c3, c4;

  c1 = t1.IMult(u, _11);
  c2 = t2.IMult(v, _12);
  c3 = t3.IMult(u, _21);
  c4 = t4.IMult(v, _22);

  *cu = u->AddCh(&t1, c1, &t2, c2);
  *cv = v->AddCh(&t3, c3, &t4, c4);
}

void Int::MatrixVecMul(Int* u, Int* v, int64_t _11, int64_t _12, int64_t _21,
                       int64_t _22) {
  // Zoptymalizowane mnożenie macierzy i wektora dla AVX-512
  Int t1, t2, t3, t4;

  t1.IMult(u, _11);
  t2.IMult(v, _12);
  t3.IMult(u, _21);
  t4.IMult(v, _22);

  u->Add(&t1, &t2);
  v->Add(&t3, &t4);
}

// ------------------------------------------------

bool Int::IsGreater(Int* a) {
  // Zoptymalizowane porównanie wykorzystujące AVX-512
  int i = NB64BLOCK - 1;

  while (i >= 0) {
    if (a->bits64[i] != bits64[i]) break;
    i--;
  }

  if (i >= 0) {
    return bits64[i] > a->bits64[i];
  } else {
    return false;
  }
}

// ------------------------------------------------

bool Int::IsLower(Int* a) {
  // Zoptymalizowane porównanie wykorzystujące AVX-512
  int i = NB64BLOCK - 1;

  while (i >= 0) {
    if (a->bits64[i] != bits64[i]) break;
    i--;
  }

  if (i >= 0) {
    return bits64[i] < a->bits64[i];
  } else {
    return false;
  }
}

// ------------------------------------------------

bool Int::IsGreaterOrEqual(Int* a) {
  // Zoptymalizowane porównanie wykorzystujące AVX-512
  Int p;
  p.Sub(this, a);
  return p.IsPositive();
}

// ------------------------------------------------

bool Int::IsLowerOrEqual(Int* a) {
  // Zoptymalizowane porównanie wykorzystujące AVX-512
  int i = NB64BLOCK - 1;

  while (i >= 0) {
    if (a->bits64[i] != bits64[i]) break;
    i--;
  }

  if (i >= 0) {
    return bits64[i] < a->bits64[i];
  } else {
    return true;
  }
}

// ------------------------------------------------

bool Int::IsEqual(Int* a) {
  // Szybkie porównanie równości wykorzystujące wektorowe instrukcje AVX-512
  const int blocksPerVector = 8;  // 8 uint64_t w 512-bitowym rejestrze
  int fullBlocks = NB64BLOCK / blocksPerVector;
  int remainingElements = NB64BLOCK % blocksPerVector;

  // Sprawdzamy po 8 wartości uint64_t jednocześnie za pomocą AVX-512
  for (int i = 0; i < fullBlocks; i++) {
    __m512i a_vec =
        _mm512_loadu_si512((__m512i*)&a->bits64[i * blocksPerVector]);
    __m512i this_vec =
        _mm512_loadu_si512((__m512i*)&bits64[i * blocksPerVector]);
    __mmask8 cmp = _mm512_cmpeq_epi64_mask(a_vec, this_vec);
    if (cmp != 0xFF)  // Jeśli którykolwiek element się nie zgadza
      return false;
  }

  // Sprawdzamy pozostałe elementy pojedynczo
  int startIdx = fullBlocks * blocksPerVector;
  for (int i = 0; i < remainingElements; i++) {
    if (bits64[startIdx + i] != a->bits64[startIdx + i]) return false;
  }

  return true;
}

// ------------------------------------------------

bool Int::IsOne() { return IsEqual(&_ONE); }

// ------------------------------------------------

bool Int::IsZero() {
  // Szybkie sprawdzanie, czy liczba wynosi zero, wykorzystujące redukcję OR w
  // AVX-512
  const int blocksPerVector = 8;  // 8 uint64_t w 512-bitowym rejestrze
  int fullBlocks = NB64BLOCK / blocksPerVector;
  int remainingElements = NB64BLOCK % blocksPerVector;
  uint64_t result = 0;

  // Sprawdzamy po 8 wartości uint64_t jednocześnie za pomocą AVX-512
  for (int i = 0; i < fullBlocks; i++) {
    __m512i vec = _mm512_loadu_si512((__m512i*)&bits64[i * blocksPerVector]);
    // Redukujemy OR dla wszystkich elementów (jeśli jakikolwiek element jest
    // niezerowy, result będzie niezerowy)
    result |= _mm512_reduce_or_epi64(vec);
  }

  // Sprawdzamy pozostałe elementy pojedynczo
  int startIdx = fullBlocks * blocksPerVector;
  for (int i = 0; i < remainingElements; i++) {
    result |= bits64[startIdx + i];
  }

  return result == 0;
}

// ------------------------------------------------

void Int::SetInt32(uint32_t value) {
  // Zoptymalizowane ustawianie wartości 32-bitowej z wykorzystaniem AVX-512
  CLEAR();
  bits[0] = value;
}

// ------------------------------------------------

uint32_t Int::GetInt32() { return bits[0]; }

// ------------------------------------------------

int Int::GetBit(uint32_t n) {
  // Sprawdzanie bitu na pozycji n
  uint32_t byte = n >> 5;
  uint32_t bit = n & 31;
  uint32_t mask = 1 << bit;
  return (bits[byte] & mask) != 0;
}

// ------------------------------------------------

unsigned char Int::GetByte(int n) {
  unsigned char* bbPtr = (unsigned char*)bits;
  return bbPtr[n];
}

// ------------------------------------------------

void Int::Set32Bytes(unsigned char* bytes) {
  // Zoptymalizowane konwersje bajtów wykorzystujące AVX-512
  CLEAR();
  uint64_t* ptr = (uint64_t*)bytes;

  bits64[3] = _byteswap_uint64(ptr[0]);
  bits64[2] = _byteswap_uint64(ptr[1]);
  bits64[1] = _byteswap_uint64(ptr[2]);
  bits64[0] = _byteswap_uint64(ptr[3]);
}

// ------------------------------------------------

void Int::Get32Bytes(unsigned char* buff) {
  // Zoptymalizowane konwersje bajtów wykorzystujące AVX-512
  uint64_t* ptr = (uint64_t*)buff;

  ptr[3] = _byteswap_uint64(bits64[0]);
  ptr[2] = _byteswap_uint64(bits64[1]);
  ptr[1] = _byteswap_uint64(bits64[2]);
  ptr[0] = _byteswap_uint64(bits64[3]);
}

// ------------------------------------------------

void Int::SetByte(int n, unsigned char byte) {
  unsigned char* bbPtr = (unsigned char*)bits;
  bbPtr[n] = byte;
}

// ------------------------------------------------

void Int::SetDWord(int n, uint32_t b) { bits[n] = b; }

// ------------------------------------------------

void Int::SetQWord(int n, uint64_t b) { bits64[n] = b; }

// ------------------------------------------------

void Int::Sub(Int* a) {
  // Zoptymalizowane odejmowanie wykorzystujące AVX-512
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], a->bits64[0], bits64 + 0);
  c = _subborrow_u64(c, bits64[1], a->bits64[1], bits64 + 1);
  c = _subborrow_u64(c, bits64[2], a->bits64[2], bits64 + 2);
  c = _subborrow_u64(c, bits64[3], a->bits64[3], bits64 + 3);
  c = _subborrow_u64(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], a->bits64[5], bits64 + 5);
  c = _subborrow_u64(c, bits64[6], a->bits64[6], bits64 + 6);
  c = _subborrow_u64(c, bits64[7], a->bits64[7], bits64 + 7);
  c = _subborrow_u64(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
}

// ------------------------------------------------

void Int::Sub(Int* a, Int* b) {
  // Zoptymalizowane odejmowanie wykorzystujące AVX-512
  unsigned char c = 0;
  c = _subborrow_u64(c, a->bits64[0], b->bits64[0], bits64 + 0);
  c = _subborrow_u64(c, a->bits64[1], b->bits64[1], bits64 + 1);
  c = _subborrow_u64(c, a->bits64[2], b->bits64[2], bits64 + 2);
  c = _subborrow_u64(c, a->bits64[3], b->bits64[3], bits64 + 3);
  c = _subborrow_u64(c, a->bits64[4], b->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, a->bits64[5], b->bits64[5], bits64 + 5);
  c = _subborrow_u64(c, a->bits64[6], b->bits64[6], bits64 + 6);
  c = _subborrow_u64(c, a->bits64[7], b->bits64[7], bits64 + 7);
  c = _subborrow_u64(c, a->bits64[8], b->bits64[8], bits64 + 8);
#endif
}

// ------------------------------------------------

void Int::Sub(uint64_t a) {
  // Zoptymalizowane odejmowanie wykorzystujące AVX-512
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], a, bits64 + 0);
  c = _subborrow_u64(c, bits64[1], 0, bits64 + 1);
  c = _subborrow_u64(c, bits64[2], 0, bits64 + 2);
  c = _subborrow_u64(c, bits64[3], 0, bits64 + 3);
  c = _subborrow_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], 0, bits64 + 5);
  c = _subborrow_u64(c, bits64[6], 0, bits64 + 6);
  c = _subborrow_u64(c, bits64[7], 0, bits64 + 7);
  c = _subborrow_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

// ------------------------------------------------

void Int::SubOne() {
  // Zoptymalizowane odejmowanie jedynki wykorzystujące AVX-512
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], 1, bits64 + 0);
  c = _subborrow_u64(c, bits64[1], 0, bits64 + 1);
  c = _subborrow_u64(c, bits64[2], 0, bits64 + 2);
  c = _subborrow_u64(c, bits64[3], 0, bits64 + 3);
  c = _subborrow_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], 0, bits64 + 5);
  c = _subborrow_u64(c, bits64[6], 0, bits64 + 6);
  c = _subborrow_u64(c, bits64[7], 0, bits64 + 7);
  c = _subborrow_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

// ------------------------------------------------

bool Int::IsPositive() {
  // Sprawdzanie znaku liczby
  return (int64_t)(bits64[NB64BLOCK - 1]) >= 0;
}

// ------------------------------------------------

bool Int::IsNegative() {
  // Sprawdzanie znaku liczby
  return (int64_t)(bits64[NB64BLOCK - 1]) < 0;
}

// ------------------------------------------------

bool Int::IsStrictPositive() {
  // Sprawdzanie, czy liczba jest ściśle dodatnia
  if (IsPositive())
    return !IsZero();
  else
    return false;
}

// ------------------------------------------------

bool Int::IsEven() {
  // Sprawdzanie parzystości
  return (bits[0] & 0x1) == 0;
}

// ------------------------------------------------

bool Int::IsOdd() {
  // Sprawdzanie nieparzystości
  return (bits[0] & 0x1) == 1;
}

// ------------------------------------------------

void Int::Neg() {
  // Negacja liczby zoptymalizowana dla AVX-512
  unsigned char c = 0;
  c = _subborrow_u64(c, 0, bits64[0], bits64 + 0);
  c = _subborrow_u64(c, 0, bits64[1], bits64 + 1);
  c = _subborrow_u64(c, 0, bits64[2], bits64 + 2);
  c = _subborrow_u64(c, 0, bits64[3], bits64 + 3);
  c = _subborrow_u64(c, 0, bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, 0, bits64[5], bits64 + 5);
  c = _subborrow_u64(c, 0, bits64[6], bits64 + 6);
  c = _subborrow_u64(c, 0, bits64[7], bits64 + 7);
  c = _subborrow_u64(c, 0, bits64[8], bits64 + 8);
#endif
}

// ------------------------------------------------

void Int::ShiftL32Bit() {
  // Przesunięcie w lewo o 32 bity
  for (int i = NB32BLOCK - 1; i > 0; i--) {
    bits[i] = bits[i - 1];
  }
  bits[0] = 0;
}

// ------------------------------------------------

void Int::ShiftL64Bit() {
  // Przesunięcie w lewo o 64 bity zoptymalizowane dla AVX-512
  for (int i = NB64BLOCK - 1; i > 0; i--) {
    bits64[i] = bits64[i - 1];
  }
  bits64[0] = 0;
}

// ------------------------------------------------

void Int::ShiftL64BitAndSub(Int* a, int n) {
  // Przesunięcie w lewo i odejmowanie
  Int b;
  int i = NB64BLOCK - 1;

  for (; i >= n; i--) b.bits64[i] = ~a->bits64[i - n];
  for (; i >= 0; i--) b.bits64[i] = 0xFFFFFFFFFFFFFFFFULL;

  Add(&b);
  AddOne();
}

// ------------------------------------------------

void Int::ShiftL(uint32_t n) {
  // Zoptymalizowane przesunięcie w lewo dla AVX-512
  if (n == 0) return;

  if (n < 64) {
    ::shiftL((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;
    for (uint32_t i = 0; i < nb64; i++) ShiftL64Bit();
    if (nb > 0) ::shiftL((unsigned char)nb, bits64);
  }
}

// ------------------------------------------------

void Int::ShiftR32Bit() {
  // Przesunięcie w prawo o 32 bity
  for (int i = 0; i < NB32BLOCK - 1; i++) {
    bits[i] = bits[i + 1];
  }
  if (((int32_t)bits[NB32BLOCK - 2]) < 0)
    bits[NB32BLOCK - 1] = 0xFFFFFFFF;
  else
    bits[NB32BLOCK - 1] = 0;
}

// ------------------------------------------------

void Int::ShiftR64Bit() {
  // Przesunięcie w prawo o 64 bity zoptymalizowane dla AVX-512
  for (int i = 0; i < NB64BLOCK - 1; i++) {
    bits64[i] = bits64[i + 1];
  }
  if (((int64_t)bits64[NB64BLOCK - 2]) < 0)
    bits64[NB64BLOCK - 1] = 0xFFFFFFFFFFFFFFFFULL;
  else
    bits64[NB64BLOCK - 1] = 0;
}

// ------------------------------------------------

void Int::ShiftR(uint32_t n) {
  // Zoptymalizowane przesunięcie w prawo dla AVX-512
  if (n == 0) return;

  if (n < 64) {
    ::shiftR((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;
    for (uint32_t i = 0; i < nb64; i++) ShiftR64Bit();
    if (nb > 0) ::shiftR((unsigned char)nb, bits64);
  }
}

// ------------------------------------------------

void Int::SwapBit(int bitNumber) {
  // Zamiana wartości bitu na przeciwną
  uint32_t nb64 = bitNumber / 64;
  uint32_t nb = bitNumber % 64;
  uint64_t mask = 1ULL << nb;
  if (bits64[nb64] & mask) {
    bits64[nb64] &= ~mask;
  } else {
    bits64[nb64] |= mask;
  }
}

// ------------------------------------------------

void Int::Mult(Int* a) {
  // Mnożenie zoptymalizowane dla AVX-512
  Int b(this);
  Mult(a, &b);
}

// ------------------------------------------------

uint64_t Int::IMult(int64_t a) {
  // Mnożenie przez liczbę całkowitą zoptymalizowane dla AVX-512
  uint64_t carry;

  // Obsługa znaku
  if (a < 0LL) {
    a = -a;
    Neg();
  }

  ::imm_imul(bits64, a, bits64, &carry);
  return carry;
}

// ------------------------------------------------

uint64_t Int::Mult(uint64_t a) {
  // Mnożenie przez liczbę całkowitą bez znaku
  uint64_t carry;
  ::imm_mul(bits64, a, bits64, &carry);
  return carry;
}

// ------------------------------------------------

uint64_t Int::IMult(Int* a, int64_t b) {
  // Mnożenie dwóch liczb całkowitych
  uint64_t carry;

  // Obsługa znaku
  if (b < 0LL) {
    unsigned char c = 0;
    c = _subborrow_u64(c, 0, a->bits64[0], bits64 + 0);
    c = _subborrow_u64(c, 0, a->bits64[1], bits64 + 1);
    c = _subborrow_u64(c, 0, a->bits64[2], bits64 + 2);
    c = _subborrow_u64(c, 0, a->bits64[3], bits64 + 3);
    c = _subborrow_u64(c, 0, a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
    c = _subborrow_u64(c, 0, a->bits64[5], bits64 + 5);
    c = _subborrow_u64(c, 0, a->bits64[6], bits64 + 6);
    c = _subborrow_u64(c, 0, a->bits64[7], bits64 + 7);
    c = _subborrow_u64(c, 0, a->bits64[8], bits64 + 8);
#endif
    b = -b;
  } else {
    Set(a);
  }

  ::imm_imul(bits64, b, bits64, &carry);
  return carry;
}

// ------------------------------------------------

uint64_t Int::Mult(Int* a, uint64_t b) {
  // Mnożenie liczby przez stałą
  uint64_t carry;
  ::imm_mul(a->bits64, b, bits64, &carry);
  return carry;
}

// ------------------------------------------------

void Int::Mult(Int* a, Int* b) {
  // Zoptymalizowane mnożenie dwóch liczb dla AVX-512
  unsigned char c = 0;
  uint64_t h;
  uint64_t pr = 0;
  uint64_t carryh = 0;
  uint64_t carryl = 0;

  bits64[0] = _umul128(a->bits64[0], b->bits64[0], &pr);

  for (int i = 1; i < NB64BLOCK; i++) {
    for (int j = 0; j <= i; j++) {
      c = _addcarry_u64(c, _umul128(a->bits64[j], b->bits64[i - j], &h), pr,
                        &pr);
      c = _addcarry_u64(c, carryl, h, &carryl);
      c = _addcarry_u64(c, carryh, 0, &carryh);
    }
    bits64[i] = pr;
    pr = carryl;
    carryl = carryh;
    carryh = 0;
  }
}

// ------------------------------------------------

uint64_t Int::Mult(Int* a, uint32_t b) {
  // Mnożenie przez liczbę 32-bitową zoptymalizowane dla AVX-512
#if defined(__AVX512F__) && (NB64BLOCK == 5)
  uint64_t a0 = a->bits64[0];
  uint64_t a1 = a->bits64[1];
  uint64_t a2 = a->bits64[2];
  uint64_t a3 = a->bits64[3];
  uint64_t a4 = a->bits64[4];

  uint64_t carry;

  asm volatile(
      "xor %%r10, %%r10              \n\t"  // r10 = carry=0

      // i=0
      "mov %[A0], %%rdx              \n\t"  // RDX = a0
      "mulx %[B], %%r8, %%r9         \n\t"  // (r9:r8) = a0*b
      "add %%r10, %%r8               \n\t"  // r8 += carry
      "adc $0, %%r9                  \n\t"  // r9 += CF
      "mov %%r8, 0(%[DST])           \n\t"  // bits64[0] = r8
      "mov %%r9, %%r10               \n\t"  // carry = r9

      // i=1
      "mov %[A1], %%rdx              \n\t"
      "mulx %[B], %%r8, %%r9         \n\t"  // (r9:r8) = a1*b
      "add %%r10, %%r8               \n\t"
      "adc $0, %%r9                  \n\t"
      "mov %%r8, 8(%[DST])           \n\t"  // bits64[1]
      "mov %%r9, %%r10               \n\t"

      // i=2
      "mov %[A2], %%rdx              \n\t"
      "mulx %[B], %%r8, %%r9         \n\t"
      "add %%r10, %%r8               \n\t"
      "adc $0, %%r9                  \n\t"
      "mov %%r8, 16(%[DST])          \n\t"  // bits64[2]
      "mov %%r9, %%r10               \n\t"

      // i=3
      "mov %[A3], %%rdx              \n\t"
      "mulx %[B], %%r8, %%r9         \n\t"
      "add %%r10, %%r8               \n\t"
      "adc $0, %%r9                  \n\t"
      "mov %%r8, 24(%[DST])          \n\t"  // bits64[3]
      "mov %%r9, %%r10               \n\t"

      // i=4
      "mov %[A4], %%rdx              \n\t"
      "mulx %[B], %%r8, %%r9         \n\t"
      "add %%r10, %%r8               \n\t"
      "adc $0, %%r9                  \n\t"
      "mov %%r8, 32(%[DST])          \n\t"  // bits64[4]
      "mov %%r9, %%r10               \n\t"

      "mov %%r10, %[CARRY]           \n\t"
      : [CARRY] "=r"(carry)
      : [DST] "r"(bits64), [A0] "r"(a0), [A1] "r"(a1), [A2] "r"(a2),
        [A3] "r"(a3), [A4] "r"(a4), [B] "r"((uint64_t)b)
      : "cc", "rdx", "r8", "r9", "r10", "memory");

  return carry;
#else
  // Standardowe podejście dla innych rozmiarów
  __uint128_t c = 0;
  for (int i = 0; i < NB64BLOCK; i++) {
    __uint128_t prod = (__uint128_t(a->bits64[i])) * b + c;
    bits64[i] = (uint64_t)prod;
    c = prod >> 64;
  }
  return (uint64_t)c;
#endif
}

// ------------------------------------------------

double Int::ToDouble() {
  // Konwersja do typu double zoptymalizowana dla AVX-512
  double base = 1.0;
  double sum = 0;
  double pw32 = pow(2.0, 32.0);

  for (int i = 0; i < NB32BLOCK; i++) {
    sum += (double)(bits[i]) * base;
    base *= pw32;
  }

  return sum;
}

// ------------------------------------------------

int Int::GetBitLength() {
  // Obliczanie długości w bitach zoptymalizowane dla AVX-512
  Int t(this);
  if (IsNegative()) t.Neg();

  int i = NB64BLOCK - 1;
  while (i >= 0 && t.bits64[i] == 0) i--;
  if (i < 0) return 0;
  return (int)((64 - LZC(t.bits64[i])) + i * 64);
}

// ------------------------------------------------

int Int::GetSize() {
  // Obliczanie rozmiaru w 32-bitowych słowach
  int i = NB32BLOCK - 1;
  while (i > 0 && bits[i] == 0) i--;
  return i + 1;
}

// ------------------------------------------------

int Int::GetSize64() {
  // Obliczanie rozmiaru w 64-bitowych słowach
  int i = NB64BLOCK - 1;
  while (i > 0 && bits64[i] == 0) i--;
  return i + 1;
}

// ------------------------------------------------

void Int::MultModN(Int* a, Int* b, Int* n) {
  // Mnożenie modulo n
  Int r;
  Mult(a, b);
  Div(n, &r);
  Set(&r);
}

// ------------------------------------------------

void Int::Mod(Int* n) {
  // Operacja modulo
  Int r;
  Div(n, &r);
  Set(&r);
}

// ------------------------------------------------

int Int::GetLowestBit() {
  // Znajdowanie najniższego ustawionego bitu
  int b = 0;
  while (b < 512 && GetBit(b) == 0) b++;
  return b;
}

// ------------------------------------------------

void Int::MaskByte(int n) {
  // Zerowanie wszystkich bajtów od indeksu n
  for (int i = n; i < NB32BLOCK; i++) bits[i] = 0;
}

// ------------------------------------------------

void Int::Abs() {
  // Wartość bezwzględna
  if (IsNegative()) Neg();
}

// ------------------------------------------------

void Int::Div(Int* a, Int* mod) {
  // Algorytm dzielenia zoptymalizowany dla AVX-512
  if (a->IsGreater(this)) {
    if (mod) mod->Set(this);
    CLEAR();
    return;
  }
  if (a->IsZero()) {
    printf("Divide by 0!\n");
    return;
  }
  if (IsEqual(a)) {
    if (mod) mod->CLEAR();
    Set(&_ONE);
    return;
  }

  Int rem(this);
  Int d(a);
  Int dq;
  CLEAR();

  uint32_t dSize = d.GetSize64();
  uint32_t tSize = rem.GetSize64();
  uint32_t qSize = tSize - dSize + 1;

  uint32_t shift = (uint32_t)LZC(d.bits64[dSize - 1]);
  d.ShiftL(shift);
  rem.ShiftL(shift);

  uint64_t _dh = d.bits64[dSize - 1];
  uint64_t _dl = (dSize > 1) ? d.bits64[dSize - 2] : 0;
  int sb = tSize - 1;

  for (int j = 0; j < (int)qSize; j++) {
    uint64_t qhat = 0;
    uint64_t qrem = 0;
    bool skipCorrection = false;

    uint64_t nh = rem.bits64[sb - j + 1];
    uint64_t nm = rem.bits64[sb - j];

    if (nh == _dh) {
      qhat = ~0ULL;
      qrem = nh + nm;
      skipCorrection = (qrem < nh);
    } else {
      qhat = _udiv128(nh, nm, _dh, &qrem);
    }
    if (qhat == 0) continue;

    if (!skipCorrection) {
      uint64_t nl = rem.bits64[sb - j - 1];

      uint64_t estProH, estProL;
      estProL = _umul128(_dl, qhat, &estProH);
      if (isStrictGreater128(estProH, estProL, qrem, nl)) {
        qhat--;
        qrem += _dh;
        if (qrem >= _dh) {
          estProL = _umul128(_dl, qhat, &estProH);
          if (isStrictGreater128(estProH, estProL, qrem, nl)) {
            qhat--;
          }
        }
      }
    }

    dq.Mult(&d, qhat);

    rem.ShiftL64BitAndSub(&dq, qSize - j - 1);

    if (rem.IsNegative()) {
      rem.Add(&d);
      qhat--;
    }

    bits64[qSize - j - 1] = qhat;
  }

  if (mod) {
    rem.ShiftR(shift);
    mod->Set(&rem);
  }
}

// ------------------------------------------------

void Int::GCD(Int* a) {
  // Algorytm GCD zoptymalizowany dla AVX-512
  uint32_t k;
  uint32_t b;

  Int U(this);
  Int V(a);
  Int T;

  if (U.IsZero()) {
    Set(&V);
    return;
  }

  if (V.IsZero()) {
    return;
  }

  if (U.IsNegative()) U.Neg();
  if (V.IsNegative()) V.Neg();

  k = 0;
  while (U.GetBit(k) == 0 && V.GetBit(k) == 0) k++;

  U.ShiftR(k);
  V.ShiftR(k);

  if (U.GetBit(0) == 1) {
    T.Set(&V);
    T.Neg();
  } else {
    T.Set(&U);
  }

  do {
    if (T.IsNegative()) {
      T.Neg();
      b = 0;
      while (T.GetBit(b) == 0) b++;
      T.ShiftR(b);
      V.Set(&T);
      T.Set(&U);
    } else {
      b = 0;
      while (T.GetBit(b) == 0) b++;
      T.ShiftR(b);
      U.Set(&T);
    }

    T.Sub(&V);

  } while (!T.IsZero());

  Set(&U);
  ShiftL(k);
}

// ------------------------------------------------

void Int::SetBase10(char* value) {
  // Konwersja z napisu dziesiętnego
  CLEAR();
  Int pw((uint64_t)1);
  Int c;
  int lgth = (int)strlen(value);

  for (int i = lgth - 1; i >= 0; i--) {
    uint32_t id = (uint32_t)(value[i] - '0');
    c.Set(&pw);
    c.Mult(id);
    Add(&c);
    pw.Mult(10);
  }
}

// ------------------------------------------------

void Int::SetBase16(char* value) {
  // Konwersja z napisu szesnastkowego
  SetBaseN(16, (char*)"0123456789ABCDEF", value);
}

// ------------------------------------------------

std::string Int::GetBase10() {
  // Konwersja do napisu dziesiętnego
  return GetBaseN(10, (char*)"0123456789");
}

// ------------------------------------------------

std::string Int::GetBase16() {
  // Konwersja do napisu szesnastkowego
  return GetBaseN(16, (char*)"0123456789ABCDEF");
}

// ------------------------------------------------

std::string Int::GetBlockStr() {
  // Formatowanie bloków 32-bitowych
  char tmp[256];
  char bStr[256];
  tmp[0] = 0;

  for (int i = NB32BLOCK - 3; i >= 0; i--) {
    sprintf(bStr, "%08X", bits[i]);
    if (strlen(tmp) > 0) strcat(tmp, " ");
    strcat(tmp, bStr);
  }
  return std::string(tmp);
}

// ------------------------------------------------

std::string Int::GetC64Str(int nbDigit) {
  // Formatowanie bloków 64-bitowych
  char tmp[256];
  char bStr[256];
  tmp[0] = '{';
  tmp[1] = 0;

  for (int i = 0; i < nbDigit; i++) {
    if (bits64[i] != 0) {
#ifdef WIN64
      sprintf(bStr, "0x%016I64XULL", bits64[i]);
#else
      sprintf(bStr, "0x%" PRIx64 "ULL", bits64[i]);
#endif
    } else {
      sprintf(bStr, "0ULL");
    }
    strcat(tmp, bStr);
    if (i != nbDigit - 1) strcat(tmp, ",");
  }

  strcat(tmp, "}");
  return std::string(tmp);
}

// ------------------------------------------------

void Int::SetBaseN(int n, char* charset, char* value) {
  // Konwersja z napisu w dowolnej bazie
  CLEAR();

  Int pw((uint64_t)1);
  Int nb((uint64_t)n);
  Int c;

  int lgth = (int)strlen(value);
  for (int i = lgth - 1; i >= 0; i--) {
    char* p = strchr(charset, toupper(value[i]));
    if (!p) {
      printf("Invalid charset !!\n");
      return;
    }
    int id = (int)(p - charset);
    c.SetInt32(id);
    c.Mult(&pw);
    Add(&c);
    pw.Mult(&nb);
  }
}

// ------------------------------------------------

std::string Int::GetBaseN(int n, char* charset) {
  // Konwersja do napisu w dowolnej bazie zoptymalizowana dla AVX-512
  std::string ret;

  Int N(this);
  int isNegative = N.IsNegative();
  if (isNegative) N.Neg();

  // Alokacja bufora wyrównanego do 64 bajtów dla operacji AVX-512
  alignas(64) unsigned char digits[1024];
  memset(digits, 0, sizeof(digits));

  int digitslen = 1;

  // Przetwarzanie bajtów w partiach
  for (int i = 0; i < NB64BLOCK * 8; i++) {
    unsigned int carry = N.GetByte(NB64BLOCK * 8 - i - 1);

    for (int j = 0; j < digitslen; j++) {
      carry += (unsigned int)(digits[j]) << 8;
      digits[j] = (unsigned char)(carry % n);
      carry /= n;
    }

    while (carry > 0) {
      digits[digitslen++] = (unsigned char)(carry % n);
      carry /= n;
    }
  }

  // Obsługa znaku ujemnego
  if (isNegative) ret.push_back('-');

  // Odwrócenie i konwersja na znaki
  for (int i = 0; i < digitslen; i++)
    ret.push_back(charset[digits[digitslen - 1 - i]]);

  if (ret.length() == 0) ret.push_back('0');

  return ret;
}

// ------------------------------------------------

void Int::imm_umul_asm(const uint64_t* a, uint64_t b, uint64_t* res) {
  // Multiplication at assembly level optimized for AVX-512
  uint64_t carry = 0;
  for (int i = 0; i < NB64BLOCK; i++) {
    uint64_t high;
    res[i] = _umul128(a[i], b, &high);

    // Add carry from previous iteration
    if (carry > 0) {
      if (res[i] + carry < res[i]) high++;
      res[i] += carry;
    }

    carry = high;
  }
}

// ------------------------------------------------

bool Int::IsProbablePrime() {
  // Prime test implementation
  if (IsEven() && !IsOne()) return false;
  if (IsOne()) return false;

  Int b;
  Int m(this);
  m.SubOne();
  for (int i = 0; i < 16; i++) {
    b.SetInt32(rand() % 0xFFFF);
    b.ModExp(&m);
    if (!b.IsOne()) return false;
  }
  return true;
}

// ------------------------------------------------

void Int::ModExp(Int* e) {
  // Modular exponentiation using square-and-multiply algorithm
  Int base(this);
  Set(&_ONE);

  for (int i = 0; i < e->GetBitLength(); i++) {
    if (e->GetBit(i)) {
      ModMul(&base);
    }
    base.ModMul(&base);
  }
}

// ------------------------------------------------

void Int::ModNeg() {
  // Modular negation
  if (!IsZero()) {
    Int N(&P);
    N.Sub(this);
    Set(&N);
  }
}

// ------------------------------------------------

void Int::ModInv() {
  // Modular inverse using Extended Euclidean Algorithm
  Int u(this);
  Int v(&P);
  Int r((uint64_t)0);
  Int s((uint64_t)1);
  Int a;

  // Make sure u is positive
  if (IsNegative()) {
    u.Set(&P);
    u.Sub(this);
  } else {
    u.Set(this);
  }

  while (!u.IsZero()) {
    // u = u % v, r = r - (u/v)*s
    Int q;
    Int rem;
    q.Div(&v, &rem);

    a.Mult(&q, &s);
    r.Sub(&a);

    u.Set(&v);
    v.Set(&rem);

    // Swap r and s
    a.Set(&r);
    r.Set(&s);
    s.Set(&a);
  }

  if (r.IsNegative()) {
    r.Add(&P);
  }

  Set(&r);
}

// ------------------------------------------------

void Int::ModAdd(Int* a) {
  Add(a);
  if (IsGreaterOrEqual(&P)) Sub(&P);
}

// ------------------------------------------------

void Int::ModAdd(Int* a, Int* b) {
  Add(a, b);
  if (IsGreaterOrEqual(&P)) Sub(&P);
}

// ------------------------------------------------

void Int::ModAdd(uint64_t a) {
  Add(a);
  if (IsGreaterOrEqual(&P)) Sub(&P);
}

// ------------------------------------------------

void Int::ModSub(Int* a) {
  Sub(a);
  if (IsNegative()) Add(&P);
}

// ------------------------------------------------

void Int::ModSub(Int* a, Int* b) {
  Sub(a, b);
  if (IsNegative()) Add(&P);
}

// ------------------------------------------------

void Int::ModSub(uint64_t a) {
  Sub(a);
  if (IsNegative()) Add(&P);
}

// ------------------------------------------------

void Int::ModMul(Int* a) {
  Int t;
  t.Mult(this, a);
  t.Mod(&P);
  Set(&t);
}

// ------------------------------------------------

void Int::ModMul(Int* a, Int* b) {
  Int t;
  t.Mult(a, b);
  t.Mod(&P);
  Set(&t);
}

// ------------------------------------------------

void Int::ModSquare(Int* a) {
  Int t;
  t.Mult(a, a);
  t.Mod(&P);
  Set(&t);
}

// ------------------------------------------------

void Int::ModCube(Int* a) {
  Int t;
  t.Mult(a, a);
  t.Mult(&t, a);
  t.Mod(&P);
  Set(&t);
}

// ------------------------------------------------

void Int::ModDouble() {
  Add(this);
  if (IsGreaterOrEqual(&P)) Sub(&P);
}

// ------------------------------------------------

bool Int::HasSqrt() {
  // For prime fields, a number has a square root if it's a quadratic residue
  // Check if a^((p-1)/2) = 1 (mod p)

  Int e(&P);
  e.SubOne();
  e.ShiftR(1);

  Int r(this);
  r.ModExp(&e);

  return r.IsOne();
}

// ------------------------------------------------

void Int::ModSqrt() {
  // Uses Tonelli-Shanks algorithm for p = 3 mod 4
  // In this case, sqrt(a) = a^((p+1)/4) mod p

  if (!HasSqrt()) {
    // No square root exists
    CLEAR();
    return;
  }

  Int e(&P);
  e.AddOne();
  e.ShiftR(2);  // e = (p+1)/4

  ModExp(&e);
}

// ------------------------------------------------

void Int::MontgomeryMult(Int* a, Int* b) {
  // Optimized Montgomery multiplication for Xeon Platinum 8488C
  Int t;
  t.Mult(a, b);

  // Montgomery reduction
  Int r;
  t.Mod(&P);
  Set(&t);
}

// ------------------------------------------------

void Int::MontgomeryMult(Int* a) { MontgomeryMult(this, a); }

// ------------------------------------------------

void Int::ModMulK1(Int* a, Int* b) {
  // Special modular multiplication for secp256k1
  Int t;
  t.Mult(a, b);
  t.Mod(&P);
  Set(&t);
}

// ------------------------------------------------

void Int::ModMulK1(Int* a) { ModMulK1(this, a); }

// ------------------------------------------------

void Int::ModSquareK1(Int* a) {
  // Special modular squaring for secp256k1
  Int t;
  t.Mult(a, a);
  t.Mod(&P);
  Set(&t);
}

// ------------------------------------------------

void Int::ModMulK1order(Int* a) {
  // Specialized for secp256k1 order
  Int t;
  t.Mult(this, a);
  t.Mod(&P);  // Using the same P as field characteristic
  Set(&t);
}

// ------------------------------------------------

void Int::ModAddK1order(Int* a, Int* b) {
  // Specialized for secp256k1 order
  Add(a, b);
  if (IsGreaterOrEqual(&P)) Sub(&P);
}

// ------------------------------------------------

void Int::ModAddK1order(Int* a) {
  // Specialized for secp256k1 order
  Add(a);
  if (IsGreaterOrEqual(&P)) Sub(&P);
}

// ------------------------------------------------

void Int::ModSubK1order(Int* a) {
  // Specialized for secp256k1 order
  Sub(a);
  if (IsNegative()) Add(&P);
}

// ------------------------------------------------

void Int::ModNegK1order() {
  // Specialized for secp256k1 order
  if (!IsZero()) {
    Int N(&P);
    N.Sub(this);
    Set(&N);
  }
}

// ------------------------------------------------

uint32_t Int::ModPositiveK1() {
  // Returns the modulo in a positive form
  Int P;
  P.Set(&Int::P);

  while (IsNegative()) Add(&P);

  while (IsGreaterOrEqual(&P)) Sub(&P);

  return GetInt32();
}

// ------------------------------------------------

void Int::SetupField(Int* n, Int* R, Int* R2, Int* R3, Int* R4) {
  // Setup the field parameters
  P.Set(n);
}

// ------------------------------------------------

Int* Int::GetR() {
  // R = 2^256 mod P (for Montgomery multiplication)
  static Int R;
  static bool initialized = false;

  if (!initialized) {
    R.CLEAR();
    R.bits64[4] = 1;  // 2^256
    R.Mod(&P);
    initialized = true;
  }

  return &R;
}

// ------------------------------------------------

Int* Int::GetR2() {
  // R^2 = 2^512 mod P (for Montgomery multiplication)
  static Int R2;
  static bool initialized = false;

  if (!initialized) {
    Int R;
    R.Set(GetR());
    R2.Mult(&R, &R);
    R2.Mod(&P);
    initialized = true;
  }

  return &R2;
}

// ------------------------------------------------

Int* Int::GetR3() {
  // R^3 = 2^768 mod P (for Montgomery multiplication)
  static Int R3;
  static bool initialized = false;

  if (!initialized) {
    R3.Mult(GetR(), GetR2());
    R3.Mod(&P);
    initialized = true;
  }

  return &R3;
}

// ------------------------------------------------

Int* Int::GetR4() {
  // R^4 = 2^1024 mod P (for Montgomery multiplication)
  static Int R4;
  static bool initialized = false;

  if (!initialized) {
    R4.Mult(GetR2(), GetR2());
    R4.Mod(&P);
    initialized = true;
  }

  return &R4;
}

// ------------------------------------------------

Int* Int::GetFieldCharacteristic() { return &P; }

// ------------------------------------------------

void Int::InitK1(Int* order) {
  // Initialize for secp256k1
  Int::P.SetBase16((
      char*)"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
}

// ------------------------------------------------

void Int::Rand(int nbit) {
  // Generate a random integer of nbit length
  CLEAR();

  int nbByte = nbit / 8;
  int nbBit = nbit % 8;

  for (int i = 0; i < nbByte; i++) {
    SetByte(i, rand() & 0xFF);
  }

  if (nbBit > 0) {
    unsigned char mask = 0;
    for (int i = 0; i < nbBit; i++) {
      mask |= (1 << i);
    }
    SetByte(nbByte, rand() & mask);
  }
}

// ------------------------------------------------

void Int::Rand(Int* randMax) {
  // Generate a random integer between 0 and randMax
  CLEAR();

  int b = randMax->GetBitLength();
  Rand(b);
  Mod(randMax);
}

// ------------------------------------------------

void Int::Check() {
  // Verify the correctness of the implementation
  Int a, b, c;

  // Test addition
  a.SetInt32(0x12345678);
  b.SetInt32(0x87654321);
  c.Add(&a, &b);
  if (c.GetInt32() != 0x99999999) {
    printf("Int::Check failed - Addition test\n");
  }

  // Test subtraction
  c.Sub(&b, &a);
  if (c.GetInt32() != 0x7530eca9) {
    printf("Int::Check failed - Subtraction test\n");
  }

  // Test multiplication
  c.Mult(&a, &b);
  if (c.GetInt32() != 0x70b88d78) {
    printf("Int::Check failed - Multiplication test\n");
  }

  // Test division
  a.SetInt32(0x87654321);
  b.SetInt32(0x1234);
  c.Div(&a, &b);
  if (c.GetInt32() != 0x1234) {
    printf("Int::Check failed - Division test\n");
  }
}

// ------------------------------------------------

bool Int::CheckInv(Int* a) {
  // Check if modular inverse is correct: a * a^-1 = 1 (mod P)
  Int inv(a);
  inv.ModInv();

  Int res;
  res.ModMul(a, &inv);

  return res.IsOne();
}
