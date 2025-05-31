#ifndef SHA256_AVX512_H
#define SHA256_AVX512_H

#include <cstdint>
#include <immintrin.h>

/**
 * Highly optimized SHA-256 implementation using AVX-512 instructions
 * Specifically tuned for Intel Xeon Platinum 8488C
 * 
 * This implementation processes 16 input blocks in parallel to maximize throughput
 * on AVX-512 capable processors.
 */

/**
 * Process 16 input blocks (each 64 bytes) in parallel to produce 16 SHA-256 hashes
 * 
 * @param data0-data15  Pointers to 16 input blocks (each 64 bytes)
 * @param hash0-hash15  Pointers to store the resulting 16 hashes (each 32 bytes)
 */
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

/**
 * Process a single input block to produce a SHA-256 hash (fallback for when full batch is not available)
 * 
 * @param data  Pointer to input block (64 bytes)
 * @param hash  Pointer to store the resulting hash (32 bytes)
 */
void sha256avx512_single(const uint8_t* data, unsigned char* hash);

/**
 * Prepare input blocks for SHA-256 computation (add padding and length)
 * 
 * @param data    Pointer to input data
 * @param length  Length of input data in bytes
 * @param block   Pointer to output block (64 bytes)
 */
void sha256_prepare_block(const uint8_t* data, size_t length, uint8_t* block);

#endif // SHA256_AVX512_H
