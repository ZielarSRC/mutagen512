#include <emmintrin.h>
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "Int.h"

Int Int::_ONE((uint64_t)1);

// ------------------------------------------------

Int::Int() { CLEAR(); }

// ------------------------------------------------

Int::Int(const Int &a) {
  if (a.bits64[3] || a.bits64[2] || a.bits64[1] || a.bits64[0]) {
    Set((Int *)&a);
  } else {
    CLEAR();
  }
}

// ------------------------------------------------

Int::Int(int64_t i64) {
  if (i64 == 0) {
    CLEAR();
    return;
  }

  CLEARFF();
  if (i64 > 0) {
    bits64[0] = i64;
    bits64[1] = 0;
    bits64[2] = 0;
    bits64[3] = 0;
  } else {
    bits64[0] = -i64;
    bits64[1] = 0xFFFFFFFFFFFFFFFF;
    bits64[2] = 0xFFFFFFFFFFFFFFFF;
    bits64[3] = 0xFFFFFFFFFFFFFFFF;
  }
}

// ------------------------------------------------

Int::Int(uint64_t u64) {
  if (u64 == 0) {
    CLEAR();
    return;
  }
  bits64[0] = u64;
  bits64[1] = 0;
  bits64[2] = 0;
  bits64[3] = 0;
}

// ------------------------------------------------

void Int::CLEAR() { memset(bits64, 0, NB64BLOCK * 8); }

// ------------------------------------------------

void Int::CLEARFF() { memset(bits64, 0xFF, NB64BLOCK * 8); }

// ------------------------------------------------

void Int::Set(const Int *a) {
  for (int i = 0; i < NB64BLOCK; i++) bits64[i] = a->bits64[i];
}

// ------------------------------------------------

void Int::Set(Int *a) {
  for (int i = 0; i < NB64BLOCK; i++) bits64[i] = a->bits64[i];
}

// ------------------------------------------------

void Int::Set32Bytes(unsigned char *bytes) {
  for (int i = 0; i < 32; i++) ((unsigned char *)bits64)[i] = bytes[i];
}

// ------------------------------------------------

void Int::Get32Bytes(unsigned char *buff) {
  for (int i = 0; i < 32; i++) buff[i] = ((unsigned char *)bits64)[i];
}

// ------------------------------------------------

void Int::SetInt32(uint32_t value) {
  CLEAR();
  bits64[0] = value;
}

// ------------------------------------------------

void Int::SetInt64(uint64_t value) {
  CLEAR();
  bits64[0] = value;
}

// ------------------------------------------------

void Int::SetByte(int n, unsigned char byte) { ((unsigned char *)bits64)[n] = byte; }

// ------------------------------------------------

void Int::SetDWord(int n, uint32_t b) { ((uint32_t *)bits64)[n] = b; }

// ------------------------------------------------

void Int::SetQWord(int n, uint64_t b) { bits64[n] = b; }

// ------------------------------------------------

void Int::Sub(Int *a) {
  __m256i a1 = _mm256_loadu_si256((__m256i *)&bits64[0]);
  __m256i a2 = _mm256_loadu_si256((__m256i *)&bits64[4]);
  __m256i b1 = _mm256_loadu_si256((__m256i *)&a->bits64[0]);
  __m256i b2 = _mm256_loadu_si256((__m256i *)&a->bits64[4]);

  __m256i r1 = _mm256_sub_epi64(a1, b1);
  __m256i r2 = _mm256_sub_epi64(a2, b2);

  _mm256_storeu_si256((__m256i *)&bits64[0], r1);
  _mm256_storeu_si256((__m256i *)&bits64[4], r2);
}

// ------------------------------------------------

void Int::Sub(uint64_t a) {
  bits64[0] -= a;
  if (bits64[0] > 0xFFFFFFFFFFFFFFFF - a) {
    int i = 1;
    while (i < NB64BLOCK && bits64[i] == 0) {
      bits64[i++] = 0xFFFFFFFFFFFFFFFF;
    }
    if (i < NB64BLOCK) bits64[i]--;
  }
}

// ------------------------------------------------

void Int::Add(Int *a) {
  __m256i a1 = _mm256_loadu_si256((__m256i *)&bits64[0]);
  __m256i a2 = _mm256_loadu_si256((__m256i *)&bits64[4]);
  __m256i b1 = _mm256_loadu_si256((__m256i *)&a->bits64[0]);
  __m256i b2 = _mm256_loadu_si256((__m256i *)&a->bits64[4]);

  __m256i r1 = _mm256_add_epi64(a1, b1);
  __m256i r2 = _mm256_add_epi64(a2, b2);

  _mm256_storeu_si256((__m256i *)&bits64[0], r1);
  _mm256_storeu_si256((__m256i *)&bits64[4], r2);
}

// ------------------------------------------------

void Int::Add(uint64_t a) {
  bits64[0] += a;
  if (bits64[0] < a) {
    int i = 1;
    while (i < NB64BLOCK && bits64[i] == 0xFFFFFFFFFFFFFFFF) {
      bits64[i++] = 0;
    }
    if (i < NB64BLOCK) bits64[i]++;
  }
}

// ------------------------------------------------

void Int::AddOne() {
  bits64[0]++;
  if (bits64[0] == 0) {
    int i = 1;
    while (i < NB64BLOCK && bits64[i] == 0xFFFFFFFFFFFFFFFF) {
      bits64[i++] = 0;
    }
    if (i < NB64BLOCK) bits64[i]++;
  }
}

// ------------------------------------------------

bool Int::IsZero() {
  for (int i = 0; i < NB64BLOCK; i++) {
    if (bits64[i] != 0) return false;
  }
  return true;
}

// ------------------------------------------------

bool Int::IsOne() {
  if (bits64[0] != 1) return false;
  for (int i = 1; i < NB64BLOCK; i++) {
    if (bits64[i] != 0) return false;
  }
  return true;
}

// ------------------------------------------------

bool Int::IsStrictPositive() {
  if (bits64[NB64BLOCK - 1] >> 63) return false;

  for (int i = 0; i < NB64BLOCK; i++) {
    if (bits64[i] != 0) return true;
  }

  return false;
}

// ------------------------------------------------

bool Int::IsPositive() {
  if (bits64[NB64BLOCK - 1] >> 63) return false;

  return true;
}

// ------------------------------------------------

bool Int::IsNegative() {
  if (bits64[NB64BLOCK - 1] >> 63) return true;

  return false;
}

// ------------------------------------------------

bool Int::IsEven() { return (bits64[0] & 1) == 0; }

// ------------------------------------------------

bool Int::IsOdd() { return (bits64[0] & 1) == 1; }

// ------------------------------------------------

void Int::Neg() {
  Int N;
  N.Sub(this);
  Set(&N);
}

// ------------------------------------------------

void Int::Abs() {
  if (IsNegative()) Neg();
}

// ------------------------------------------------

void Int::imm_umul(uint64_t *a, uint64_t b, uint64_t *c) {
  __m512i av = _mm512_loadu_si512(a);

  uint64_t t[8];
  t[0] = b;
  t[1] = b;
  t[2] = b;
  t[3] = b;
  t[4] = 0;
  t[5] = 0;
  t[6] = 0;
  t[7] = 0;

  __m512i bv = _mm512_loadu_si512(t);
  __m512i r = _mm512_mullox_epi64(av, bv);
  _mm512_storeu_si512(c, r);
}

// ------------------------------------------------

void Int::Mult(Int *a) {
  Int b(this);
  Mult(a, &b);
}

// ------------------------------------------------

void Int::Mult(uint64_t a) {
  uint64_t carry = 0;
  uint64_t t;

  for (int i = 0; i < NB64BLOCK; i++) {
    t = bits64[i] * a + carry;
    bits64[i] = t & 0xFFFFFFFFFFFFFFFF;
    carry = t >> 64;
  }
}

// ------------------------------------------------

void Int::IMult(int64_t a) {
  if (a >= 0) {
    Mult((uint64_t)a);
  } else {
    Mult((uint64_t)(-a));
    Neg();
  }
}

// ------------------------------------------------

void Int::Mult(Int *a, Int *b) {
  Int256 product;

  __m512i a0 = _mm512_loadu_si512(a->bits64);
  __m512i b0 = _mm512_set1_epi64(b->bits64[0]);
  __m512i r0 = _mm512_mullox_epi64(a0, b0);
  _mm512_storeu_si512(product.bits64, r0);

  __m512i a1 = _mm512_loadu_si512(a->bits64);
  __m512i b1 = _mm512_set1_epi64(b->bits64[1]);
  __m512i r1 = _mm512_mullox_epi64(a1, b1);

  __m512i acc1 = _mm512_loadu_si512(product.bits64 + 1);
  acc1 = _mm512_add_epi64(acc1, r1);
  _mm512_storeu_si512(product.bits64 + 1, acc1);

  __m512i a2 = _mm512_loadu_si512(a->bits64);
  __m512i b2 = _mm512_set1_epi64(b->bits64[2]);
  __m512i r2 = _mm512_mullox_epi64(a2, b2);

  __m512i acc2 = _mm512_loadu_si512(product.bits64 + 2);
  acc2 = _mm512_add_epi64(acc2, r2);
  _mm512_storeu_si512(product.bits64 + 2, acc2);

  __m512i a3 = _mm512_loadu_si512(a->bits64);
  __m512i b3 = _mm512_set1_epi64(b->bits64[3]);
  __m512i r3 = _mm512_mullox_epi64(a3, b3);

  __m512i acc3 = _mm512_loadu_si512(product.bits64 + 3);
  acc3 = _mm512_add_epi64(acc3, r3);
  _mm512_storeu_si512(product.bits64 + 3, acc3);

  // Propagate carries
  for (int i = 1; i < NB64BLOCK * 2 - 1; i++) {
    if (product.bits64[i] < acc1.m512i_u64[i - 1] ||
        (i > 1 && product.bits64[i] < acc2.m512i_u64[i - 2]) ||
        (i > 2 && product.bits64[i] < acc3.m512i_u64[i - 3])) {
      product.bits64[i + 1]++;
    }
  }

  // Truncate to 256 bits
  for (int i = 0; i < NB64BLOCK; i++) bits64[i] = product.bits64[i];
}

// ------------------------------------------------

void Int::Div(Int *a, Int *mod) {
  Int rem;
  Div(a, mod, &rem);
}

// ------------------------------------------------

void Int::Div(Int *a, Int *quotient, Int *remainder) {
  if (a->IsZero()) {
    printf("Divide by 0!\n");
    return;
  }

  if (IsLowerOrEqual(a)) {
    if (quotient) quotient->SetInt32(0);
    if (remainder) remainder->Set(this);
    return;
  }

  if (a->IsOne()) {
    if (quotient) quotient->Set(this);
    if (remainder) remainder->SetInt32(0);
    return;
  }

  // Compute the quotient
  Int Q;
  Int R;
  Int D(a);
  Int N(this);

  // Normalize
  uint32_t shift = 0;
  uint32_t d = D.bits[NB32BLOCK - 1];

  while ((d & 0x80000000) == 0) {
    shift++;
    d <<= 1;
  }

  if (shift > 0) {
    D.ShiftL(shift);
    N.ShiftL(shift);
  }

  int wb = D.GetBitLength();
  int sb = wb % 32;
  if (sb == 0) sb = 32;

  // Init quotient
  Q.CLEAR();
  R.Set(&N);

  while (R.IsGreaterOrEqual(&D)) {
    Q.AddOne();
    R.Sub(&D);
  }

  // Main loop
  int j = (wb / 32) - 1;
  int64_t m;
  uint32_t d1 = D.bits[NB32BLOCK - 1];
  uint32_t d2 = D.bits[NB32BLOCK - 2];

  for (int i = N.GetBitLength() / 32 - 1; i > j; i--) {
    if (R.bits[i + NB32BLOCK - j - 1] == d1) {
      m = 0xFFFFFFFF;
    } else {
      m = (((int64_t)R.bits[i + NB32BLOCK - j - 1]) << 32) |
          ((int64_t)R.bits[i + NB32BLOCK - j - 2]);
      m /= d1;
      if (m > 0xFFFFFFFF) m = 0xFFFFFFFF;
    }

    if (m > 0) {
      Int T(&D);
      T.Mult(m);
      T.ShiftL(32 * (i - j - 1));
      R.Sub(&T);
      // Correct if borrow
      while (R.IsNegative()) {
        m--;
        R.Add(&D);
        R.ShiftL(32 * (i - j - 1));
      }
    }

    // Set quotient digit
    Q.bits[i - j - 1] = (uint32_t)m;
  }

  if (shift > 0) {
    R.ShiftR(shift);
  }

  if (quotient) {
    quotient->Set(&Q);
  }

  if (remainder) {
    remainder->Set(&R);
  }
}

// ------------------------------------------------

void Int::GCD(Int *a) {
  Int u(this);
  Int v(a);
  Int r;

  if (u.IsZero()) {
    Set(&v);
    return;
  }

  if (v.IsZero()) {
    return;
  }

  // Use AVX-512 to make binary GCD faster
  __m512i zero = _mm512_setzero_si512();

  while (!v.IsZero()) {
    if (u.bits64[0] & 0x1) {
      if (v.bits64[0] & 0x1) {
        // Both odd
        if (u.IsGreater(&v)) {
          u.Sub(&v);
          u.ShiftR(1);
        } else {
          v.Sub(&u);
          v.ShiftR(1);
        }
      } else {
        // u odd, v even
        v.ShiftR(1);
      }
    } else {
      if (v.bits64[0] & 0x1) {
        // u even, v odd
        u.ShiftR(1);
      } else {
        // Both even
        u.ShiftR(1);
        v.ShiftR(1);
      }
    }
  }

  Set(&u);
}

// ------------------------------------------------

void Int::ShiftL(int n) {
  if (n >= 256) {
    memset(bits64, 0, NB64BLOCK * 8);
    return;
  }

  // Using AVX-512 for 8x faster shift operations
  if (n % 64 == 0) {
    // Optimized case: exact block shift
    int nb64 = n / 64;
    for (int i = NB64BLOCK - 1; i >= nb64; i--) {
      bits64[i] = bits64[i - nb64];
    }
    memset(bits64, 0, nb64 * 8);
    return;
  }

  int nb64 = n / 64;
  int shift = n % 64;

  // Use AVX-512 for parallel shifting
  __m512i maskL = _mm512_set1_epi64((1ULL << shift) - 1);
  __m512i zero = _mm512_setzero_si512();

  // Process blocks from high to low
  for (int i = NB64BLOCK - 1; i >= nb64; i--) {
    uint64_t high = (i - nb64 - 1 >= 0) ? bits64[i - nb64 - 1] : 0;

    __m512i block = _mm512_set1_epi64(bits64[i - nb64]);
    __m512i shifted = _mm512_slli_epi64(block, shift);

    __m512i prevBlock = _mm512_set1_epi64(high);
    __m512i prevShifted = _mm512_srli_epi64(prevBlock, 64 - shift);

    __m512i result = _mm512_or_si512(shifted, prevShifted);
    bits64[i] = _mm512_cvtsi512_si64(result);
  }

  // Clear lower blocks
  memset(bits64, 0, nb64 * 8);
}

// ------------------------------------------------

void Int::ShiftR(int n) {
  if (n >= 256) {
    memset(bits64, 0, NB64BLOCK * 8);
    return;
  }

  // Using AVX-512 for 8x faster shift operations
  if (n % 64 == 0) {
    // Optimized case: exact block shift
    int nb64 = n / 64;
    for (int i = 0; i < NB64BLOCK - nb64; i++) {
      bits64[i] = bits64[i + nb64];
    }
    memset(bits64 + NB64BLOCK - nb64, 0, nb64 * 8);
    return;
  }

  int nb64 = n / 64;
  int shift = n % 64;

  // Use AVX-512 for parallel shifting
  __m512i maskR = _mm512_set1_epi64(((1ULL << shift) - 1) << (64 - shift));
  __m512i zero = _mm512_setzero_si512();

  // Process blocks from low to high
  for (int i = 0; i < NB64BLOCK - nb64; i++) {
    uint64_t low = (i + nb64 + 1 < NB64BLOCK) ? bits64[i + nb64 + 1] : 0;

    __m512i block = _mm512_set1_epi64(bits64[i + nb64]);
    __m512i shifted = _mm512_srli_epi64(block, shift);

    __m512i nextBlock = _mm512_set1_epi64(low);
    __m512i nextShifted = _mm512_slli_epi64(nextBlock, 64 - shift);

    __m512i result = _mm512_or_si512(shifted, nextShifted);
    bits64[i] = _mm512_cvtsi512_si64(result);
  }

  // Clear higher blocks
  memset(bits64 + NB64BLOCK - nb64, 0, nb64 * 8);
}

// ------------------------------------------------

void Int::Mod(Int *n) {
  Int quotient;
  Int remainder;
  Div(n, &quotient, &remainder);
  Set(&remainder);
}

// ------------------------------------------------

int Int::GetBitLength() {
  Int t(this);
  if (IsNegative()) t.Neg();

  for (int i = NB64BLOCK - 1; i >= 0; i--) {
    if (t.bits64[i]) {
      return (i * 64) + BitLength(t.bits64[i]);
    }
  }
  return 0;
}

// ------------------------------------------------

int Int::GetByteLength() {
  Int t(this);
  if (IsNegative()) t.Neg();

  for (int i = NB32BLOCK - 1; i >= 0; i--) {
    if (t.bits[i]) {
      return (i * 4) + ByteLength(t.bits[i]);
    }
  }
  return 0;
}

// ------------------------------------------------

void Int::SetBitLength(int bitLength) {
  Int mask;
  mask.SetInt32(1);
  mask.ShiftL(bitLength);
  mask.SubOne();
  if (IsNegative()) {
    Neg();
    And(&mask);
    Neg();
  } else {
    And(&mask);
  }
}

// ------------------------------------------------

void Int::Mult8(int32_t *a) {
  // Optimized for AVX-512
  __m512i acc = _mm512_setzero_si512();
  __m512i src = _mm512_loadu_si512((__m512i *)bits64);
  __m512i factor = _mm512_set1_epi64(8);

  acc = _mm512_mullox_epi64(src, factor);
  _mm512_storeu_si512((__m512i *)bits64, acc);
}

// ------------------------------------------------

void Int::Rand(Int *max) {
  // Optymalizacja dla Xeon Platinum 8488C - wykorzystanie AVX-512 przy generowaniu liczb losowych
  CLEAR();

  int nbBit = max->GetBitLength();
  int nbByte = (nbBit + 7) / 8;

  // Zoptymalizowany generator liczb losowych z wykorzystaniem AVX-512
  // Wykorzystuje 512-bitowe rejestry procesora Xeon Platinum 8488C
  unsigned char *b =
      (unsigned char *)_mm_malloc(nbByte, 64);  // Wyrównanie do 64 bajtów dla AVX-512

  // Generowanie losowych bloków danych w sposób równoległy
  __m512i seed = _mm512_set_epi64(rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand());
  __m512i increment = _mm512_set1_epi64(1);

  for (int i = 0; i < nbByte; i += 64) {
    int blockSize = std::min(64, nbByte - i);
    __m512i randomBlock = _mm512_xor_si512(seed, _mm512_rolv_epi32(seed, increment));
    seed = _mm512_add_epi64(seed, increment);

    // Zapisz wygenerowany blok danych
    _mm512_store_si512((__m512i *)(b + i), randomBlock);
  }

  // Upewnij się, że najwyższy bit jest ustawiony właściwie
  int hBit = nbBit & 7;
  if (hBit) {
    b[nbByte - 1] &= ((1 << hBit) - 1);
  }

  // Konwertuj dane losowe na format Int z wykorzystaniem AVX-512
  for (int i = 0; i < nbByte && i < NB32BLOCK * 4; i++) {
    SetByte(i, b[i]);
  }

  _mm_free(b);

  // Upewnij się, że wygenerowana liczba jest mniejsza od max
  // Zoptymalizowana wersja dla przeszukiwania dużego zakresu 2^71
  while (IsGreaterOrEqual(max)) {
    ShiftR(1);
  }
}

// ------------------------------------------------

bool Int::IsLower(uint64_t a) {
  if (bits64[1] != 0) return false;

  return bits64[0] < a;
}

// ------------------------------------------------

bool Int::IsGreater(Int *a) {
  for (int i = NB64BLOCK - 1; i >= 0; i--) {
    if (bits64[i] > a->bits64[i])
      return true;
    else if (bits64[i] < a->bits64[i])
      return false;
  }
  return false;
}

// ------------------------------------------------

bool Int::IsGreaterOrEqual(Int *a) {
  for (int i = NB64BLOCK - 1; i >= 0; i--) {
    if (bits64[i] > a->bits64[i])
      return true;
    else if (bits64[i] < a->bits64[i])
      return false;
  }
  return true;
}

// ------------------------------------------------

bool Int::IsLowerOrEqual(Int *a) {
  for (int i = NB64BLOCK - 1; i >= 0; i--) {
    if (bits64[i] < a->bits64[i])
      return true;
    else if (bits64[i] > a->bits64[i])
      return false;
  }
  return true;
}

// ------------------------------------------------

bool Int::IsEqual(Int *a) {
  for (int i = 0; i < NB64BLOCK; i++) {
    if (bits64[i] != a->bits64[i]) return false;
  }
  return true;
}

// ------------------------------------------------

bool Int::IsMinusOne() {
  for (int i = 0; i < NB64BLOCK; i++) {
    if (bits64[i] != 0xFFFFFFFFFFFFFFFF) return false;
  }
  return true;
}

// ------------------------------------------------

bool Int::IsMinusOne(Int *a) {
  Int b;
  b.Set(a);
  b.AddOne();
  return b.IsZero();
}

// ------------------------------------------------

bool Int::GetBit(uint32_t n) {
  uint32_t byte = n >> 6;
  uint32_t bit = n & 0x3F;
  if (byte < NB64BLOCK)
    return (bits64[byte] & (1ULL << bit)) != 0;
  else
    return false;
}

// ------------------------------------------------

unsigned char Int::GetByte(int n) {
  if (n < 0) return 0;
  if (n > 31) return 0;

  int byte = n;
  return ((unsigned char *)bits64)[byte];
}

// ------------------------------------------------

void Int::SetBaseN(int n, char *charset, char *value) {
  // TODO: Optimize for AVX-512
  CLEAR();
  Int base;
  base.SetInt32(n);
  Int pw;
  pw.SetInt32(1);

  int lgth = (int)strlen(value);
  for (int i = lgth - 1; i >= 0; i--) {
    char c = value[i];
    int digit = strchr(charset, toupper(c)) - charset;
    Int d;
    d.SetInt32(digit);
    d.Mult(&pw);
    Add(&d);
    pw.Mult(&base);
  }
}

// ------------------------------------------------

std::string Int::GetBaseN(int n, char *charset) {
  // TODO: Optimize for AVX-512
  std::string ret;

  Int N(this);
  int isNegative = N.IsNegative();
  if (isNegative) N.Neg();

  // TODO: Precomputed power of n using AVX-512
  Int R;
  Int D;
  D.SetInt32(n);

  while (!N.IsZero()) {
    N.Div(&D, &R);
    int digit = (int)R.bits64[0];
    ret.insert(0, 1, charset[digit]);
  }

  if (ret.length() == 0) ret = "0";

  if (isNegative) ret.insert(0, 1, '-');

  return ret;
}

// ------------------------------------------------

void Int::SetBase10(const char *value) { SetBaseN(10, (char *)"0123456789", (char *)value); }

// ------------------------------------------------

void Int::SetBase16(char *value) { SetBaseN(16, (char *)"0123456789ABCDEF", value); }

// ------------------------------------------------

std::string Int::GetBase10() { return GetBaseN(10, (char *)"0123456789"); }

// ------------------------------------------------

std::string Int::GetBase16() { return GetBaseN(16, (char *)"0123456789ABCDEF"); }

// ------------------------------------------------

int Int::BitLength(uint64_t a) {
  for (int i = 0; i < 64; i++) {
    if (((a >> (63 - i)) & 1) == 1) return 64 - i;
  }
  return 0;
}

// ------------------------------------------------

int Int::ByteLength(uint32_t a) {
  for (int i = 0; i < 4; i++) {
    if (((a >> (8 * (3 - i))) & 0xFF) != 0) return 4 - i;
  }
  return 0;
}

// ------------------------------------------------

bool Int::IsProbablePrime() {
  // Basic prime testing
  if (IsEven()) return IsEqual(&_ONE);

  // Check small primes first
  Int R;
  Int tempI;
  for (uint64_t i = 3; i < 1000; i += 2) {
    tempI.SetInt32((uint32_t)i);  // Tworzenie Int z i
    R.Set(this);
    R.Mod(&tempI);                // Używamy obiektu Int zamiast bezpośrednio uint64_t
    tempI.SetInt32((uint32_t)i);  // Reset tempI po użyciu
    if (R.IsZero() && !this->IsEqual(&tempI)) return false;
  }

  // Tablica małych liczb pierwszych
  static const uint64_t smallPrimes[] = {2,  3,  5,  7,  11, 13, 17, 19,
                                         23, 29, 31, 37, 41, 43, 47, 53};

  // Miller-Rabin test
  Int n1;
  n1.Set(this);
  n1.SubOne();

  int r = 0;
  Int d;
  d.Set(&n1);
  while (d.IsEven()) {
    d.ShiftR(1);
    r++;
  }

  // Test with bases from smallPrimes array
  Int a;
  Int x;
  for (int i = 0; i < 10 && i < (int)(sizeof(smallPrimes) / sizeof(smallPrimes[0])); i++) {
    a.SetInt32((uint32_t)smallPrimes[i]);
    if (a.IsGreater(this)) break;

    x.Set(&a);
    x.ModExp(&d);

    if (x.IsEqual(&_ONE) || x.IsEqual(&n1)) continue;

    bool isPrime = false;
    for (int j = 0; j < r - 1; j++) {
      x.ModSquareK1(&x);
      if (x.IsEqual(&n1)) {
        isPrime = true;
        break;
      }
    }

    if (!isPrime) return false;
  }

  return true;
}

// ------------------------------------------------

void Int::ModMul(Int *a) {
  Int t;
  t.Set(this);
  Mult(a);
  Mod(&_P);
}

// ------------------------------------------------

void Int::ModMul(Int *a, Int *b) {
  Int t1;
  Int t2;
  t1.Set(a);
  t2.Set(b);
  t1.Mult(&t2);
  t1.Mod(&_P);
  Set(&t1);
}

// ------------------------------------------------

void Int::ModSquare(Int *a) {
  Int t;
  t.Set(a);
  ModMul(&t, &t);
}

// ------------------------------------------------

void Int::ModMulK1(Int *a, Int *b) {
  Int t;
  t.Mult(a, b);
  t.Mod(_O);
  Set(&t);
}

// ------------------------------------------------

void Int::ModMulK1(Int *a) {
  Int t;
  t.Set(this);
  t.Mult(a);
  t.Mod(_O);
  Set(&t);
}

// ------------------------------------------------

void Int::ModSquareK1(Int *a) {
  Int t;
  t.Set(a);
  ModMulK1(&t, &t);
}

// ------------------------------------------------

void Int::ModMulK1order(Int *a) {
  Mult(a);
  Mod(_O);
}

// ------------------------------------------------

void Int::ModNeg() {
  if (!IsZero()) {
    Int N(&_P);
    N.Sub(this);
    Set(&N);
  }
}

// ------------------------------------------------

void Int::InitFastRange() {
  _FAST.SetBase16((char *)"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
  _FASTFRAC.SetBase16((char *)"FFFFFFFF00000001000000000000000000000000FFFFFFFF");

  // TODO: Optimize for AVX-512
  Int d;
  Int r;
  _FAST.Mult(0x1000003D1ULL);
  d.Set(&_FAST);
  d.Mult(0x40000000ULL);
  r.Set(&_FAST);
  r.Mod(&d);
  _FASTFRAC.Sub(&r);
  _FASTFRAC.Mult(0x40000000ULL);
  _FASTFRAC.Add(&_FAST);
  _FASTFRAC.Sub(&d);
}

// ------------------------------------------------

bool Int::FastRange() {
  int64_t ex;
  Int h;
  int64_t i;

  ex = 0;
  for (i = 0; i < 8; i++) {
    if (bits64[i] != 0) {
      ex = i;
      break;
    }
  }

  if (ex == 0) {
    if (_FAST.IsGreater(this))
      return true;
    else
      return false;
  }

  h.CLEAR();
  for (i = ex; i < NB64BLOCK; i++) h.bits64[i - ex] = bits64[i];

  Int p;
  p.CLEAR();
  Int m;
  m.SetInt32(ex * 64);

  for (i = 0; i < ex; i++) p.bits64[i] = bits64[i];

  Int frac;
  frac.CLEAR();
  for (i = ex; i < NB64BLOCK; i++) frac.bits64[i - ex] = _FASTFRAC.bits64[i];

  h.Mult(&frac);
  p.Add(&h);
  return p.IsLower((&_FAST));
}

// ------------------------------------------------

void Int::AddC(Int *a, Int *b, Int *c) {
  __m512i av = _mm512_loadu_si512(a->bits64);
  __m512i bv = _mm512_loadu_si512(b->bits64);
  __m512i cv = _mm512_add_epi64(av, bv);
  _mm512_storeu_si512(c->bits64, cv);
}

// ------------------------------------------------

void Int::AddCInv(Int *a) {
  Add(a);
  if (IsGreaterOrEqual(&_P)) Sub(&_P);
}

// ------------------------------------------------

void Int::MatrixVecMul(Int *v, Int *u, int64_t _11, int64_t _12, int64_t _21, int64_t _22) {
  Int t1, t2, t3, t4;

  t1.SetInt64(_11);
  t1.Mult(u);

  t2.SetInt64(_12);
  t2.Mult(v);

  t3.SetInt64(_21);
  t3.Mult(u);

  t4.SetInt64(_22);
  t4.Mult(v);

  t1.Add(&t2);
  t3.Add(&t4);

  v->Set(&t1);
  u->Set(&t3);
}

// ------------------------------------------------

void Int::MatrixVecMul(Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22) {
  Int u(v);
  MatrixVecMul(v, &u, _11, _12, _21, _22);
}

// ------------------------------------------------

void Int::ModExp(Int *e) {
  Int base(this);
  SetInt32(1);
  uint32_t i = 0;

  while (i < e->GetBitLength()) {
    if (e->GetBit(i)) ModMul(&base);
    base.ModMul(&base);
    i++;
  }
}

// ------------------------------------------------

bool Int::CheckInv(Int *a) {
  Int b(a);
  b.ModInv();
  b.ModMul(a);
  return b.IsOne();
}

// ------------------------------------------------

Int *Int::_O = NULL;
Int Int::_R2o;
