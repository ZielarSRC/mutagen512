#ifndef SHA256_AVX512_H
#define SHA256_AVX512_H

#include <cstdint>

// AVX-512 optimized SHA-256 for 16 parallel computations
void sha256avx512_16B(
    const uint8_t* data0, const uint8_t* data1, const uint8_t* data2, const uint8_t* data3,
    const uint8_t* data4, const uint8_t* data5, const uint8_t* data6, const uint8_t* data7,
    const uint8_t* data8, const uint8_t* data9, const uint8_t* data10, const uint8_t* data11,
    const uint8_t* data12, const uint8_t* data13, const uint8_t* data14, const uint8_t* data15,
    unsigned char* hash0, unsigned char* hash1, unsigned char* hash2, unsigned char* hash3,
    unsigned char* hash4, unsigned char* hash5, unsigned char* hash6, unsigned char* hash7,
    unsigned char* hash8, unsigned char* hash9, unsigned char* hash10, unsigned char* hash11,
    unsigned char* hash12, unsigned char* hash13, unsigned char* hash14, unsigned char* hash15
);

// Backward compatibility wrapper for 8 parallel computations
void sha256avx512_8B(
    const uint8_t* data0, const uint8_t* data1, const uint8_t* data2, const uint8_t* data3,
    const uint8_t* data4, const uint8_t* data5, const uint8_t* data6, const uint8_t* data7,
    unsigned char* hash0, unsigned char* hash1, unsigned char* hash2, unsigned char* hash3,
    unsigned char* hash4, unsigned char* hash5, unsigned char* hash6, unsigned char* hash7
);

// Batch processing for variable number of inputs
void sha256avx512_batch(
    const uint8_t** data_array, 
    unsigned char** hash_array, 
    int count
);

#endif // SHA256_AVX512_H
