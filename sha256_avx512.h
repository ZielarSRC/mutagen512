#ifndef SHA256_AVX512_H
#define SHA256_AVX512_H

#include <immintrin.h>

#include <cstdint>

namespace sha256avx512 {

// Initialize SHA-256 state for 16 parallel computations
void Initialize(__m512i* state);

// Transform 16 parallel SHA-256 blocks using AVX-512
void Transform(__m512i* state, const uint8_t* data[16]);

// Main hashing function for 16 messages of 64 bytes each
void sha256avx512_16B(
    const uint8_t* data0, const uint8_t* data1, const uint8_t* data2,
    const uint8_t* data3, const uint8_t* data4, const uint8_t* data5,
    const uint8_t* data6, const uint8_t* data7, const uint8_t* data8,
    const uint8_t* data9, const uint8_t* data10, const uint8_t* data11,
    const uint8_t* data12, const uint8_t* data13, const uint8_t* data14,
    const uint8_t* data15, unsigned char* hash0, unsigned char* hash1,
    unsigned char* hash2, unsigned char* hash3, unsigned char* hash4,
    unsigned char* hash5, unsigned char* hash6, unsigned char* hash7,
    unsigned char* hash8, unsigned char* hash9, unsigned char* hash10,
    unsigned char* hash11, unsigned char* hash12, unsigned char* hash13,
    unsigned char* hash14, unsigned char* hash15);

}  // namespace sha256avx512

#endif  // SHA256_AVX512_H
