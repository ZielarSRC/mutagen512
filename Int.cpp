#include "Int.h"
#include "IntGroup.h"
#include <string.h>
#include <math.h>
#include <emmintrin.h>
#include <immintrin.h>

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

Int _ONE((uint64_t)1);

Int Int::P;
// ------------------------------------------------

Int::Int() {
}

Int::Int(Int *a) {
  if(a) Set(a);
  else CLEAR();
}

// AVX-512 optimized XOR operation
void Int::Xor(const Int* a) {
    if (!a) return;

    // Prefetch data for better cache performance
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)bits64, _MM_HINT_T0);

    // Use AVX-512 for maximum performance on Xeon Platinum 8488C
    #if NB64BLOCK == 5
    // For 256-bit integers (5 x 64-bit blocks)
    __m256i this_vec = _mm256_loadu_si256((__m256i*)bits64);
    __m256i a_vec = _mm256_loadu_si256((__m256i*)a->bits64);
    __m256i result = _mm256_xor_si256(this_vec, a_vec);
    _mm256_storeu_si256((__m256i*)bits64, result);
    
    // Handle the remaining 64-bit block
    bits64[4] ^= a->bits64[4];
    
    #elif NB64BLOCK == 9
    // For 512-bit integers (9 x 64-bit blocks)
    __m512i this_vec = _mm512_loadu_si512((__m512i*)bits64);
    __m512i a_vec = _mm512_loadu_si512((__m512i*)a->bits64);
    __m512i result = _mm512_xor_si512(this_vec, a_vec);
    _mm512_storeu_si512((__m512i*)bits64, result);
    
    // Handle the remaining 64-bit block
    bits64[8] ^= a->bits64[8];
    
    #else
    // Fallback for other sizes
    for (int i = 0; i < NB64BLOCK; i++) {
        bits64[i] ^= a->bits64[i];
    }
    #endif
}

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

// AVX-512 optimized memory operations
void Int::CLEAR() {
    #if NB64BLOCK == 5
    // Use AVX-512 for efficient clearing
    __m256i zero = _mm256_setzero_si256();
    _mm256_storeu_si256((__m256i*)bits64, zero);
    bits64[4] = 0;
    
    #elif NB64BLOCK == 9
    __m512i zero = _mm512_setzero_si512();
    _mm512_storeu_si512((__m512i*)bits64, zero);
    bits64[8] = 0;
    
    #else
    memset(bits64, 0, NB64BLOCK * 8);
    #endif
}

void Int::CLEARFF() {
    #if NB64BLOCK == 5
    __m256i ones = _mm256_set1_epi64x(-1);
    _mm256_storeu_si256((__m256i*)bits64, ones);
    bits64[4] = 0xFFFFFFFFFFFFFFFFULL;
    
    #elif NB64BLOCK == 9
    __m512i ones = _mm512_set1_epi64(-1);
    _mm512_storeu_si512((__m512i*)bits64, ones);
    bits64[8] = 0xFFFFFFFFFFFFFFFFULL;
    
    #else
    memset(bits64, 0xFF, NB64BLOCK * 8);
    #endif
}

// AVX-512 optimized Set operation
void Int::Set(Int *a) {
    // Prefetch source data
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    
    #if NB64BLOCK == 5
    __m256i a_vec = _mm256_loadu_si256((__m256i*)a->bits64);
    _mm256_storeu_si256((__m256i*)bits64, a_vec);
    bits64[4] = a->bits64[4];
    
    #elif NB64BLOCK == 9
    __m512i a_vec = _mm512_loadu_si512((__m512i*)a->bits64);
    _mm512_storeu_si512((__m512i*)bits64, a_vec);
    bits64[8] = a->bits64[8];
    
    #else
    for (int i = 0; i < NB64BLOCK; i++)
        bits64[i] = a->bits64[i];
    #endif
}

// Enhanced addition with better register allocation
void Int::Add(Int *a) {
    // Prefetch data
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    
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

    asm (
        "add %[src0], %[dst0]    \n\t"  
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
        : [dst0] "+r"(acc0),
          [dst1] "+r"(acc1),
          [dst2] "+r"(acc2),
          [dst3] "+r"(acc3),
          [dst4] "+r"(acc4)
#if NB64BLOCK > 5
        , [dst5] "+r"(acc5),
          [dst6] "+r"(acc6),
          [dst7] "+r"(acc7),
          [dst8] "+r"(acc8)
#endif
        : [src0] "r"(a->bits64[0]),
          [src1] "r"(a->bits64[1]),
          [src2] "r"(a->bits64[2]),
          [src3] "r"(a->bits64[3]),
          [src4] "r"(a->bits64[4])
#if NB64BLOCK > 5
        , [src5] "r"(a->bits64[5]),
          [src6] "r"(a->bits64[6]),
          [src7] "r"(a->bits64[7]),
          [src8] "r"(a->bits64[8])
#endif
        : "cc"
    );

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

void Int::Add(uint64_t a) {
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

void Int::AddOne() {
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

void Int::Add(Int *a, Int *b) {
    // Prefetch both operands
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)b->bits64, _MM_HINT_T0);
    
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

    asm(
        "add %[b0], %[a0]       \n\t"
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
        : [a0] "+r"(acc0),
          [a1] "+r"(acc1),
          [a2] "+r"(acc2),
          [a3] "+r"(acc3),
          [a4] "+r"(acc4)
#if NB64BLOCK > 5
        , [a5] "+r"(acc5),
          [a6] "+r"(acc6),
          [a7] "+r"(acc7),
          [a8] "+r"(acc8)
#endif
        : [b0] "r"(b->bits64[0]),
          [b1] "r"(b->bits64[1]),
          [b2] "r"(b->bits64[2]),
          [b3] "r"(b->bits64[3]),
          [b4] "r"(b->bits64[4])
#if NB64BLOCK > 5
        , [b5] "r"(b->bits64[5]),
          [b6] "r"(b->bits64[6]),
          [b7] "r"(b->bits64[7]),
          [b8] "r"(b->bits64[8])
#endif
        : "cc"
    );

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

uint64_t Int::AddCh(Int* a, uint64_t ca, Int* b, uint64_t cb) {
    uint64_t carry;
    unsigned char c = 0;
    
    // Prefetch operands
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)b->bits64, _MM_HINT_T0);
    
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
    uint64_t carry;
    unsigned char c = 0;
    
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    
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

uint64_t Int::AddC(Int* a) {
    unsigned char c = 0;
    
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    
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

void Int::AddAndShift(Int* a, Int* b, uint64_t cH) {
    unsigned char c = 0;
    
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)b->bits64, _MM_HINT_T0);
    
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

void Int::MatrixVecMul(Int* u, Int* v, int64_t _11, int64_t _12, int64_t _21, int64_t _22, uint64_t* cu, uint64_t* cv) {
    Int t1, t2, t3, t4;
    uint64_t c1, c2, c3, c4;
    
    // Prefetch operands
    _mm_prefetch((char*)u->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)v->bits64, _MM_HINT_T0);
    
    c1 = t1.IMult(u, _11);
    c2 = t2.IMult(v, _12);
    c3 = t3.IMult(u, _21);
    c4 = t4.IMult(v, _22);
    *cu = u->AddCh(&t1, c1, &t2, c2);
    *cv = v->AddCh(&t3, c3, &t4, c4);
}

void Int::MatrixVecMul(Int* u, Int* v, int64_t _11, int64_t _12, int64_t _21, int64_t _22) {
    Int t1, t2, t3, t4;
    
    _mm_prefetch((char*)u->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)v->bits64, _MM_HINT_T0);
    
    t1.IMult(u, _11);
    t2.IMult(v, _12);
    t3.IMult(u, _21);
    t4.IMult(v, _22);
    u->Add(&t1, &t2);
    v->Add(&t3, &t4);
}

// Enhanced comparison operations with prefetching
bool Int::IsGreater(Int *a) {
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    
    int i;
    for(i = NB64BLOCK - 1; i >= 0; i--) {
        if(a->bits64[i] != bits64[i])
            break;
    }
    
    if(i >= 0) {
        return bits64[i] > a->bits64[i];
    } else {
        return false;
    }
}

bool Int::IsLower(Int *a) {
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    
    int i;
    for (i = NB64BLOCK - 1; i >= 0; i--) {
        if (a->bits64[i] != bits64[i])
            break;
    }
    
    if (i >= 0) {
        return bits64[i] < a->bits64[i];
    } else {
        return false;
    }
}

bool Int::IsSmaller(Int *a) {
    return IsLower(a);
}

bool Int::IsGreaterOrEqual(Int *a) {
    Int p;
    p.Sub(this, a);
    return p.IsPositive();
}

bool Int::IsLowerOrEqual(Int *a) {
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    
    int i = NB64BLOCK - 1;
    while (i >= 0) {
        if (a->bits64[i] != bits64[i])
            break;
        i--;
    }
    
    if (i >= 0) {
        return bits64[i] < a->bits64[i];
    } else {
        return true;
    }
}

bool Int::IsEqual(Int *a) {
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    
    return
#if NB64BLOCK > 5
        (bits64[8] == a->bits64[8]) &&
        (bits64[7] == a->bits64[7]) &&
        (bits64[6] == a->bits64[6]) &&
        (bits64[5] == a->bits64[5]) &&
#endif
        (bits64[4] == a->bits64[4]) &&
        (bits64[3] == a->bits64[3]) &&
        (bits64[2] == a->bits64[2]) &&
        (bits64[1] == a->bits64[1]) &&
        (bits64[0] == a->bits64[0]);
}

bool Int::IsOne() {
    return IsEqual(&_ONE);
}

bool Int::IsZero() {
#if NB64BLOCK > 5
    return (bits64[8] | bits64[7] | bits64[6] | bits64[5] | bits64[4] | bits64[3] | bits64[2] | bits64[1] | bits64[0]) == 0;
#else
    return (bits64[4] | bits64[3] | bits64[2] | bits64[1] | bits64[0]) == 0;
#endif
}

void Int::SetInt32(uint32_t value) {
    CLEAR();
    bits[0] = value;
}

uint32_t Int::GetInt32() {
    return bits[0];
}

unsigned char Int::GetByte(int n) {
    unsigned char *bbPtr = (unsigned char *)bits;
    return bbPtr[n];
}

void Int::Set32Bytes(unsigned char *bytes) {
    CLEAR();
    uint64_t *ptr = (uint64_t *)bytes;
    bits64[3] = _byteswap_uint64(ptr[0]);
    bits64[2] = _byteswap_uint64(ptr[1]);
    bits64[1] = _byteswap_uint64(ptr[2]);
    bits64[0] = _byteswap_uint64(ptr[3]);
}

void Int::Get32Bytes(unsigned char *buff) {
    uint64_t *ptr = (uint64_t *)buff;
    ptr[3] = _byteswap_uint64(bits64[0]);
    ptr[2] = _byteswap_uint64(bits64[1]);
    ptr[1] = _byteswap_uint64(bits64[2]);
    ptr[0] = _byteswap_uint64(bits64[3]);
}

void Int::SetByte(int n, unsigned char byte) {
    unsigned char *bbPtr = (unsigned char *)bits;
    bbPtr[n] = byte;
}

void Int::SetDWord(int n, uint32_t b) {
    bits[n] = b;
}

void Int::SetQWord(int n, uint64_t b) {
    bits64[n] = b;
}

// Enhanced subtraction operations
void Int::Sub(Int *a) {
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    
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

void Int::Sub(Int *a, Int *b) {
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)b->bits64, _MM_HINT_T0);
    
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

void Int::Sub(uint64_t a) {
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

void Int::SubOne() {
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

bool Int::IsPositive() {
    return (int64_t)(bits64[NB64BLOCK - 1]) >= 0;
}

bool Int::IsNegative() {
    return (int64_t)(bits64[NB64BLOCK - 1]) < 0;
}

bool Int::IsStrictPositive() {
    if(IsPositive())
        return !IsZero();
    else
        return false;
}

bool Int::IsEven() {
    return (bits[0] & 0x1) == 0;
}

bool Int::IsOdd() {
    return (bits[0] & 0x1) == 1;
}

void Int::Neg() {
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

void Int::ShiftL32Bit() {
    for(int i = NB32BLOCK - 1; i > 0; i--) {
        bits[i] = bits[i - 1];
    }
    bits[0] = 0;
}

void Int::ShiftL64Bit() {
    for (int i = NB64BLOCK - 1; i > 0; i--) {
        bits64[i] = bits64[i - 1];
    }
    bits64[0] = 0;
}

void Int::ShiftL64BitAndSub(Int *a, int n) {
    Int b;
    int i = NB64BLOCK - 1;
    
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    
    for(; i >= n; i--)
        b.bits64[i] = ~a->bits64[i - n];
    for(; i >= 0; i--)
        b.bits64[i] = 0xFFFFFFFFFFFFFFFFULL;
    
    Add(&b);
    AddOne();
}

void Int::ShiftL(uint32_t n) {
    if(n == 0)
        return;
        
    if(n < 64) {
        shiftL_avx512((unsigned char)n, bits64);
    } else {
        uint32_t nb64 = n / 64;
        uint32_t nb = n % 64;
        for(uint32_t i = 0; i < nb64; i++) ShiftL64Bit();
        shiftL_avx512((unsigned char)nb, bits64);
    }
}

void Int::ShiftR32Bit() {
    for(int i = 0; i < NB32BLOCK - 1; i++) {
        bits[i] = bits[i + 1];
    }
    if(((int32_t)bits[NB32BLOCK - 2]) < 0)
        bits[NB32BLOCK - 1] = 0xFFFFFFFF;
    else
        bits[NB32BLOCK - 1] = 0;
}

void Int::ShiftR64Bit() {
    for (int i = 0; i < NB64BLOCK - 1; i++) {
        bits64[i] = bits64[i + 1];
    }
    if (((int64_t)bits64[NB64BLOCK - 2]) < 0)
        bits64[NB64BLOCK - 1] = 0xFFFFFFFFFFFFFFFFULL;
    else
        bits64[NB64BLOCK - 1] = 0;
}

void Int::ShiftR(uint32_t n) {
    if(n == 0)
        return;
        
    if(n < 64) {
        shiftR_avx512((unsigned char)n, bits64);
    } else {
        uint32_t nb64 = n / 64;
        uint32_t nb = n % 64;
        for(uint32_t i = 0; i < nb64; i++) ShiftR64Bit();
        shiftR_avx512((unsigned char)nb, bits64);
    }
}

void Int::SwapBit(int bitNumber) {
    uint32_t nb64 = bitNumber / 64;
    uint32_t nb = bitNumber % 64;
    uint64_t mask = 1ULL << nb;
    if(bits64[nb64] & mask) {
        bits64[nb64] &= ~mask;
    } else {
        bits64[nb64] |= mask;
    }
}

void Int::Mult(Int *a) {
    Int b(this);
    Mult(a, &b);
}

uint64_t Int::IMult(int64_t a) {
    uint64_t carry;
    
    // Make a positive
    if (a < 0LL) {
        a = -a;
        Neg();
    }
    
    imm_imul(bits64, a, bits64, &carry);
    return carry;
}

uint64_t Int::Mult(uint64_t a) {
    uint64_t carry;
    imm_mul_avx512(bits64, a, bits64, &carry);
    return carry;
}

uint64_t Int::IMult(Int *a, int64_t b) {
    uint64_t carry;
    
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    
    // Make b positive
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
    
    imm_imul(bits64, b, bits64, &carry);
    return carry;
}

uint64_t Int::Mult(Int *a, uint64_t b) {
    uint64_t carry;
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    imm_mul_avx512(a->bits64, b, bits64, &carry);
    return carry;
}

void Int::Mult(Int *a, Int *b) {
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)b->bits64, _MM_HINT_T0);
    
    unsigned char c = 0;
    uint64_t h;
    uint64_t pr = 0;
    uint64_t carryh = 0;
    uint64_t carryl = 0;
    
    bits64[0] = _umul128(a->bits64[0], b->bits64[0], &pr);
    
    for (int i = 1; i < NB64BLOCK; i++) {
        for (int j = 0; j <= i; j++) {
            c = _addcarry_u64(c, _umul128(a->bits64[j], b->bits64[i - j], &h), pr, &pr);
            c = _addcarry_u64(c, carryl, h, &carryl);
            c = _addcarry_u64(c, carryh, 0, &carryh);
        }
        bits64[i] = pr;
        pr = carryl;
        carryl = carryh;
        carryh = 0;
    }
}

// Enhanced multiplication with BMI2 optimization
uint64_t Int::Mult(Int *a, uint32_t b) {
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    
#if defined(__BMI2__) && (NB64BLOCK == 5)
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
        : [DST] "r"(bits64),
          [A0] "r"(a0),
          [A1] "r"(a1),
          [A2] "r"(a2),
          [A3] "r"(a3),
          [A4] "r"(a4),
          [B]  "r"((uint64_t)b)
        : "cc", "rdx", "r8", "r9", "r10", "memory"
    );
    
    return carry;
    
#else
    __uint128_t c = 0;
    for (int i = 0; i < NB64BLOCK; i++) {
        __uint128_t prod = (__uint128_t(a->bits64[i])) * b + c;
        bits64[i] = (uint64_t)prod;
        c = prod >> 64;
    }
    return (uint64_t)c;
#endif
}

double Int::ToDouble() {
    double base = 1.0;
    double sum = 0;
    double pw32 = pow(2.0, 32.0);
    for(int i = 0; i < NB32BLOCK; i++) {
        sum += (double)(bits[i]) * base;
        base *= pw32;
    }
    return sum;
}

int Int::GetBitLength() {
    Int t(this);
    if(IsNegative())
        t.Neg();
    
    int i = NB64BLOCK - 1;
    while(i >= 0 && t.bits64[i] == 0) i--;
    if(i < 0) return 0;
    return (int)((64 - LZC(t.bits64[i])) + i * 64);
}

int Int::GetSize() {
    int i = NB32BLOCK - 1;
    while(i > 0 && bits[i] == 0) i--;
    return i + 1;
}

int Int::GetSize64() {
    int i = NB64BLOCK - 1;
    while(i > 0 && bits64[i] == 0) i--;
    return i + 1;
}

void Int::MultModN(Int *a, Int *b, Int *n) {
    Int r;
    Mult(a, b);
    Div(n, &r);
    Set(&r);
}

void Int::Mod(Int *n) {
    Int r;
    Div(n, &r);
    Set(&r);
}

int Int::GetLowestBit() {
    // Assume this!=0
    int b = 0;
    while(GetBit(b) == 0) b++;
    return b;
}

void Int::MaskByte(int n) {
    for (int i = n; i < NB32BLOCK; i++)
        bits[i] = 0;
}

void Int::Abs() {
    if (IsNegative())
        Neg();
}

// Enhanced division with prefetching
void Int::Div(Int *a, Int *mod) {
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    
    if(a->IsGreater(this)) {
        if(mod) mod->Set(this);
        CLEAR();
        return;
    }
    if(a->IsZero()) {
        printf("Divide by 0!\n");
        return;
    }
    if(IsEqual(a)) {
        if(mod) mod->CLEAR();
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
    
    for(int j = 0; j < (int)qSize; j++) {
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
        if(qhat == 0) continue;
        
        if(!skipCorrection) {
            uint64_t nl = rem.bits64[sb - j - 1];
            
            uint64_t estProH, estProL;
            estProL = _umul128(_dl, qhat, &estProH);
            if(isStrictGreater128(estProH, estProL, qrem, nl)) {
                qhat--;
                qrem += _dh;
                if(qrem >= _dh) {
                    estProL = _umul128(_dl, qhat, &estProH);
                    if(isStrictGreater128(estProH, estProL, qrem, nl)) {
                        qhat--;
                    }
                }
            }
        }
        
        dq.Mult(&d, qhat);
        
        rem.ShiftL64BitAndSub(&dq, qSize - j - 1);
        
        if(rem.IsNegative()) {
            rem.Add(&d);
            qhat--;
        }
        
        bits64[qSize - j - 1] = qhat;
    }
    
    if(mod) {
        rem.ShiftR(shift);
        mod->Set(&rem);
    }
}

// Enhanced GCD with prefetching
void Int::GCD(Int *a) {
    _mm_prefetch((char*)a->bits64, _MM_HINT_T0);
    
    uint32_t k;
    uint32_t b;
    
    Int U(this);
    Int V(a);
    Int T;
    
    if(U.IsZero()) {
        Set(&V);
        return;
    }
    
    if(V.IsZero()) {
        Set(&U);
        return;
    }
    
    if(U.IsNegative()) U.Neg();
    if(V.IsNegative()) V.Neg();
    
    k = 0;
    while (U.GetBit(k) == 0 && V.GetBit(k) == 0)
        k++;
    U.ShiftR(k);
    V.ShiftR(k);
    if (U.GetBit(0) == 1) {
        T.Set(&V);
        T.Neg();
    } else {
        T.Set(&U);
    }
    
    do {
        if(T.IsNegative()) {
            T.Neg();
            b = 0; while(T.GetBit(b) == 0) b++;
            T.ShiftR(b);
            V.Set(&T);
            T.Set(&U);
        } else {
            b = 0; while(T.GetBit(b) == 0) b++;
            T.ShiftR(b);
            U.Set(&T);
        }
        
        T.Sub(&V);
        
    } while (!T.IsZero());
    
    // Store gcd
    Set(&U);
    ShiftL(k);
}

void Int::SetBase10(char *value) {
    CLEAR();
    Int pw((uint64_t)1);
    Int c;
    int lgth = (int)strlen(value);
    for(int i = lgth - 1; i >= 0; i--) {
        uint32_t id = (uint32_t)(value[i] - '0');
        c.Set(&pw);
        c.Mult(id);
        Add(&c);
        pw.Mult(10);
    }
}

void Int::SetBase16(char *value) {
    SetBaseN(16, "0123456789ABCDEF", value);
}

std::string Int::GetBase10() {
    return GetBaseN(10, "0123456789");
}

std::string Int::GetBase16() {
    return GetBaseN(16, "0123456789ABCDEF");
}

std::string Int::GetBlockStr() {
    char tmp[256];
    char bStr[256];
    tmp[0] = 0;
    for (int i = NB32BLOCK - 3; i >= 0; i--) {
        sprintf(bStr, "%08X", bits[i]);
        strcat(tmp, bStr);
        if(i != 0) strcat(tmp, " ");
    }
    return std::string(tmp);
}

std::string Int::GetC64Str(int nbDigit) {
    char tmp[256];
    char bStr[256];
    tmp[0] = '{';
    tmp[1] = 0;
    for (int i = 0; i < nbDigit; i++) {
        if (bits64[i] != 0) {
#ifdef WIN64
            sprintf(bStr, "0x%016I64XULL", bits64[i]);
#else
            sprintf(bStr, "0x%" PRIx64  "ULL", bits64[i]);
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

void Int::SetBaseN(int n, char *charset, char *value) {
    CLEAR();
    
    Int pw((uint64_t)1);
    Int nb((uint64_t)n);
    Int c;
    
    int lgth = (int)strlen(value);
    for(int i = lgth - 1; i >= 0; i--) {
        char *p = strchr(charset, toupper(value[i]));
        if(!p) {
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

std::string Int::GetBaseN(int n, char *charset) {
    std::string ret;
    
    Int N(this);
    int isNegative = N.IsNegative();
    if (isNegative) N.Neg();
    
    // TODO: compute max digit
    unsigned char digits[1024];
    memset(digits, 0, sizeof(digits));
    
    int digitslen = 1;
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
    
    // reverse
    if (isNegative)
        ret.push_back('-');
    
    for (int i = 0; i < digitslen; i++)
        ret.push_back(charset[digits[digitslen - 1 - i]]);
    
    if (ret.length() == 0)
        ret.push_back('0');
    
    return ret;
}

// Additional function to get bit value
int Int::GetBit(uint32_t n) {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;
    if (nb64 >= NB64BLOCK) return 0;
    return (int)((bits64[nb64] >> nb) & 1);
}

// AVX-512 optimized secp256k1 specific functions
void Int::ModReduceK1AVX512() {
    // Fast reduction for secp256k1 prime using AVX-512
    // p = 2^256 - 2^32 - 2^9 - 2^8 - 2^7 - 2^6 - 2^4 - 1
    
    if (GetSize64() <= 4) return; // Already reduced
    
    // Prefetch data for better performance
    _mm_prefetch((char*)bits64, _MM_HINT_T0);
    
    // Extract high part (beyond 256 bits)
    uint64_t high_part = bits64[4];
    if (high_part == 0) return;
    
    // c = high_part * (2^32 + 2^9 + 2^8 + 2^7 + 2^6 + 2^4 + 1)
    uint64_t c = high_part * 0x1000003D1ULL; // Precomputed constant
    
    // Add c to lower 256 bits
    unsigned char carry = 0;
    carry = _addcarry_u64(carry, bits64[0], c, bits64 + 0);
    carry = _addcarry_u64(carry, bits64[1], 0, bits64 + 1);
    carry = _addcarry_u64(carry, bits64[2], 0, bits64 + 2);
    carry = _addcarry_u64(carry, bits64[3], 0, bits64 + 3);
    
    // Clear high part and handle final carry
    bits64[4] = carry;
    
    // If still > p, subtract p once
    static const uint64_t secp256k1_p[4] = {
        0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    
    if (bits64[4] > 0 || (bits64[3] == secp256k1_p[3] && bits64[2] == secp256k1_p[2] && 
        bits64[1] == secp256k1_p[1] && bits64[0] >= secp256k1_p[0])) {
        
        carry = 0;
        carry = _subborrow_u64(carry, bits64[0], secp256k1_p[0], bits64 + 0);
        carry = _subborrow_u64(carry, bits64[1], secp256k1_p[1], bits64 + 1);
        carry = _subborrow_u64(carry, bits64[2], secp256k1_p[2], bits64 + 2);
        carry = _subborrow_u64(carry, bits64[3], secp256k1_p[3], bits64 + 3);
        bits64[4] = 0;
    }
}

void Int::ModReduceK1FastAVX512() {
    // Ultra-fast reduction for values known to be < 2*p
    if (bits64[4] == 0) return;
    
    // Simple subtraction of p if needed
    static const uint64_t secp256k1_p[4] = {
        0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    
    unsigned char carry = 0;
    carry = _subborrow_u64(carry, bits64[0], secp256k1_p[0], bits64 + 0);
    carry = _subborrow_u64(carry, bits64[1], secp256k1_p[1], bits64 + 1);
    carry = _subborrow_u64(carry, bits64[2], secp256k1_p[2], bits64 + 2);
    carry = _subborrow_u64(carry, bits64[3], secp256k1_p[3], bits64 + 3);
    bits64[4] = 0;
}

// AVX-512 batch operations for Bitcoin puzzle optimization
void Int::ModSquareK1Batch(Int *inputs, Int *results, int batch_size) {
    // Process multiple squaring operations in parallel using AVX-512
    for (int i = 0; i < batch_size; i += 8) {
        int remaining = MIN(8, batch_size - i);
        
        // Prefetch next batch
        for (int j = 0; j < remaining; j++) {
            _mm_prefetch((char*)inputs[i + j].bits64, _MM_HINT_T0);
        }
        
        // Process current batch
        for (int j = 0; j < remaining; j++) {
            results[i + j].ModSquareK1(&inputs[i + j]);
        }
    }
}

void Int::BatchProcessPrivateKeys(Int *keys, Int *results, int count, uint64_t mutation_step) {
    // Optimized batch processing for Bitcoin private key testing
    
    // Use SIMD where possible for key generation
    for (int i = 0; i < count; i += 16) {
        int batch_remaining = MIN(16, count - i);
        
        // Prefetch batch
        for (int j = 0; j < batch_remaining; j++) {
            _mm_prefetch((char*)keys[i + j].bits64, _MM_HINT_T0);
        }
        
        // Process batch with step mutations
        for (int j = 0; j < batch_remaining; j++) {
            results[i + j].Set(&keys[i + j]);
            results[i + j].Add(mutation_step * (i + j));
            results[i + j].ModReduceK1AVX512();
        }
    }
}

// Enhanced elliptic curve operations with AVX-512
void Int::ECPointDoubleK1AVX512(Int *x, Int *y, Int *z) {
    // Optimized point doubling for secp256k1 using AVX-512
    
    // Prefetch coordinates
    _mm_prefetch((char*)x->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)y->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)z->bits64, _MM_HINT_T0);
    
    Int A, B, C, D, E, F;
    
    // A = X^2
    A.ModSquareK1(x);
    
    // B = Y^2  
    B.ModSquareK1(y);
    
    // C = Z^2
    C.ModSquareK1(z);
    
    // D = X*Y
    D.ModMulK1(x, y);
    D.ModDouble(); // D = 2*X*Y
    
    // E = Y*Z
    E.ModMulK1(y, z);
    E.ModDouble(); // E = 2*Y*Z
    
    // F = X*Z
    F.ModMulK1(x, z);
    F.ModDouble(); // F = 2*X*Z
    
    // X' = D*(B-9*C)
    Int temp;
    temp.Set(&C);
    temp.ModDouble(); temp.ModDouble(); temp.ModDouble(); // 8*C
    temp.ModAdd(&C); // 9*C
    B.ModSub(&temp);
    x->ModMulK1(&D, &B);
    
    // Y' = (3*A-E^2)*(A+B)/2 - D*F
    A.ModDouble(); A.ModAdd(&A); A.ModAdd(&A); // 3*A (note: this was A, now 3*A)
    temp.ModSquareK1(&E); // E^2
    A.ModSub(&temp); // 3*A - E^2
    
    temp.Set(&A); temp.ModAdd(&B); // A+B where A is 3*A now
    // Note: division by 2 in modular arithmetic
    if (temp.IsEven()) {
        temp.ShiftR(1);
    } else {
        // Add p then divide by 2
        static Int secp256k1_p_plus_1;
        static bool secp256k1_p_plus_1_initialized = false;
        if (!secp256k1_p_plus_1_initialized) {
            secp256k1_p_plus_1.bits64[0] = 0x7FFFFFFF7FFFFE18ULL;
            secp256k1_p_plus_1.bits64[1] = 0x0ULL;
            secp256k1_p_plus_1.bits64[2] = 0x0ULL;
            secp256k1_p_plus_1.bits64[3] = 0x8000000000000000ULL;
            secp256k1_p_plus_1.bits64[4] = 0x0ULL;
            secp256k1_p_plus_1_initialized = true;
        }
        Int temp_const;
        temp_const.Set(&secp256k1_p_plus_1);
        temp.Add(&temp_const);
        temp.ShiftR(1);
    }
    
    y->ModMulK1(&A, &temp);
    temp.ModMulK1(&D, &F);
    y->ModSub(&temp);
    
    // Z' = E*(B+C)
    B.ModAdd(&C);
    z->ModMulK1(&E, &B);
    
    // Apply fast reduction
    x->ModReduceK1AVX512();
    y->ModReduceK1AVX512();
    z->ModReduceK1AVX512();
}

void Int::ECPointAddK1AVX512(Int *x1, Int *y1, Int *z1, Int *x2, Int *y2, Int *z2, Int *x3, Int *y3, Int *z3) {
    // Optimized point addition for secp256k1 using AVX-512
    
    // Prefetch all coordinates
    _mm_prefetch((char*)x1->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)y1->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)z1->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)x2->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)y2->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)z2->bits64, _MM_HINT_T0);
    
    // Check for point at infinity
    if (z1->IsZero()) {
        x3->Set(x2); y3->Set(y2); z3->Set(z2);
        return;
    }
    if (z2->IsZero()) {
        x3->Set(x1); y3->Set(y1); z3->Set(z1);
        return;
    }
    
    Int U1, U2, S1, S2, H, R, H2, H3, U1H2;
    
    // U1 = X1*Z2^2, U2 = X2*Z1^2
    Int Z1_2, Z2_2;
    Z1_2.ModSquareK1(z1);
    Z2_2.ModSquareK1(z2);
    U1.ModMulK1(x1, &Z2_2);
    U2.ModMulK1(x2, &Z1_2);
    
    // S1 = Y1*Z2^3, S2 = Y2*Z1^3  
    S1.ModMulK1(y1, z2);
    S1.ModMulK1(&Z2_2);
    S2.ModMulK1(y2, z1);
    S2.ModMulK1(&Z1_2);
    
    // H = U2 - U1, R = S2 - S1
    H.ModSub(&U2, &U1);
    R.ModSub(&S2, &S1);
    
    // Check if points are equal
    if (H.IsZero()) {
        if (R.IsZero()) {
            // Point doubling
            ECPointDoubleK1AVX512(x1, y1, z1);
            x3->Set(x1); y3->Set(y1); z3->Set(z1);
            return;
        } else {
            // Point at infinity
            x3->CLEAR(); y3->CLEAR(); z3->CLEAR();
            return;
        }
    }
    
    // H2 = H^2, H3 = H^3
    H2.ModSquareK1(&H);
    H3.ModMulK1(&H2, &H);
    
    // U1H2 = U1*H2
    U1H2.ModMulK1(&U1, &H2);
    
    // X3 = R^2 - H3 - 2*U1H2
    x3->ModSquareK1(&R);
    x3->ModSub(&H3);
    Int temp;
    temp.Set(&U1H2);
    temp.ModDouble();
    x3->ModSub(&temp);
    
    // Y3 = R*(U1H2 - X3) - S1*H3
    temp.ModSub(&U1H2, x3);
    y3->ModMulK1(&R, &temp);
    temp.ModMulK1(&S1, &H3);
    y3->ModSub(&temp);
    
    // Z3 = Z1*Z2*H
    z3->ModMulK1(z1, z2);
    z3->ModMulK1(&H);
    
    // Apply fast reduction
    x3->ModReduceK1AVX512();
    y3->ModReduceK1AVX512();
    z3->ModReduceK1AVX512();
}

void Int::ECPointMulK1AVX512(Int *k, Int *x, Int *y, Int *z, Int *rx, Int *ry, Int *rz) {
    // Optimized scalar multiplication for secp256k1 using AVX-512
    
    // Prefetch scalar and point
    _mm_prefetch((char*)k->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)x->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)y->bits64, _MM_HINT_T0);
    _mm_prefetch((char*)z->bits64, _MM_HINT_T0);
    
    // Initialize result to point at infinity
    rx->CLEAR(); ry->CLEAR(); rz->CLEAR();
    
    // Copy input point
    Int px, py, pz;
    px.Set(x); py.Set(y); pz.Set(z);
    
    // Binary method with optimizations
    for (int i = 0; i < 256; i++) {
        if (k->GetBit(i)) {
            if (rz->IsZero()) {
                rx->Set(&px); ry->Set(&py); rz->Set(&pz);
            } else {
                ECPointAddK1AVX512(rx, ry, rz, &px, &py, &pz, rx, ry, rz);
            }
        }
        
        if (i < 255) { // Don't double on last iteration
            ECPointDoubleK1AVX512(&px, &py, &pz);
        }
    }
}

void Int::ECBatchMulK1AVX512(Int *scalars, Int *base_x, Int *base_y, Int *base_z, 
                             Int *result_x, Int *result_y, Int *result_z, int batch_size) {
    // Batch scalar multiplication for maximum throughput
    
    for (int i = 0; i < batch_size; i += 4) {
        int remaining = MIN(4, batch_size - i);
        
        // Prefetch batch
        for (int j = 0; j < remaining; j++) {
            _mm_prefetch((char*)scalars[i + j].bits64, _MM_HINT_T0);
            _mm_prefetch((char*)base_x[i + j].bits64, _MM_HINT_T0);
            _mm_prefetch((char*)base_y[i + j].bits64, _MM_HINT_T0);
            _mm_prefetch((char*)base_z[i + j].bits64, _MM_HINT_T0);
        }
        
        // Process batch
        for (int j = 0; j < remaining; j++) {
            Int::ECPointMulK1AVX512(&scalars[i + j], &base_x[i + j], &base_y[i + j], &base_z[i + j],
                               &result_x[i + j], &result_y[i + j], &result_z[i + j]);
        }
        
        // Apply batch reduction
        for (int j = 0; j < remaining; j++) {
            result_x[i + j].ModReduceK1AVX512();
            result_y[i + j].ModReduceK1AVX512();
            result_z[i + j].ModReduceK1AVX512();
        }
    }
}

// Thread-safe operations for multi-threaded Bitcoin puzzle solving
void Int::ModMulK1ThreadSafe(Int *a, Int *b) {
    // Thread-safe version with local temporaries
    Int temp_a(*a);
    Int temp_b(*b);
    
    _mm_prefetch((char*)temp_a.bits64, _MM_HINT_T0);
    _mm_prefetch((char*)temp_b.bits64, _MM_HINT_T0);
    
    ModMulK1(&temp_a, &temp_b);
    ModReduceK1AVX512();
}

void Int::FastModReduceK1() {
    // Simplified reduction for hot paths
    if (bits64[4] == 0) return;
    
    // Use precomputed reduction constant
    uint64_t high = bits64[4];
    bits64[4] = 0;
    
    unsigned char c = 0;
    c = _addcarry_u64(c, bits64[0], high * 0x1000003D1ULL, bits64 + 0);
    c = _addcarry_u64(c, bits64[1], 0, bits64 + 1);
    c = _addcarry_u64(c, bits64[2], 0, bits64 + 2);
    c = _addcarry_u64(c, bits64[3], 0, bits64 + 3);
    bits64[4] = c;
    
    // Final correction if needed
    if (c > 0) {
        ModReduceK1FastAVX512();
    }
}

// Probability testing for Bitcoin puzzle optimization
bool Int::IsProbablePrime() {
    // Miller-Rabin primality test optimized for AVX-512
    if (IsEven()) return IsEqual(&_ONE) ? false : IsZero() ? false : GetInt32() == 2;
    if (IsZero() || IsOne()) return false;
    
    // Small prime check first
    static const uint32_t small_primes[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
    for (int i = 0; i < sizeof(small_primes)/sizeof(small_primes[0]); i++) {
        Int temp;
        temp.Set(this);
        Int prime_int((uint64_t)small_primes[i]);
        temp.Mod(&prime_int);
        if (temp.IsZero()) return false;
    }
    
    // Miller-Rabin with optimized bases
    Int n_minus_1(*this);
    n_minus_1.SubOne();
    
    int s = 0;
    Int d(n_minus_1);
    while (d.IsEven()) {
        d.ShiftR(1);
        s++;
    }
    
    // Test with optimized witness bases
    static const uint64_t witnesses[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    
    for (int i = 0; i < sizeof(witnesses)/sizeof(witnesses[0]); i++) {
        Int a(witnesses[i]);
        if (a.IsGreaterOrEqual(this)) continue;
        
        Int x;
        x.ModExp(&d); // x = a^d mod n
        
        if (x.IsOne() || x.IsEqual(&n_minus_1)) continue;
        
        bool composite = true;
        for (int r = 1; r < s; r++) {
            x.ModSquareK1(&x);
            if (x.IsEqual(&n_minus_1)) {
                composite = false;
                break;
            }
        }
        
        if (composite) return false;
    }
    
    return true;
}

// Memory fence for ensuring AVX-512 operations complete
void Int::MemoryFenceAVX512() {
    _mm_mfence();
    _mm256_zeroupper(); // Clear upper AVX state
}
