#ifndef SHA256_AVX512_H
#define SHA256_AVX512_H

#include <immintrin.h>

#include <cstdint>

// Process 8 hashes in parallel (same interface as AVX2 version for compatibility)
void sha256avx512_8B(const uint8_t* data0, const uint8_t* data1, const uint8_t* data2,
                     const uint8_t* data3, const uint8_t* data4, const uint8_t* data5,
                     const uint8_t* data6, const uint8_t* data7, unsigned char* hash0,
                     unsigned char* hash1, unsigned char* hash2, unsigned char* hash3,
                     unsigned char* hash4, unsigned char* hash5, unsigned char* hash6,
                     unsigned char* hash7);

// Process 16 hashes in parallel (new AVX-512 optimized function)
void sha256avx512_16B(const uint8_t* data[16], unsigned char* hash[16]);

// Process 32 hashes in parallel for even higher throughput
void sha256avx512_32B(const uint8_t* data[32], unsigned char* hash[32]);

// Batch processing with optimal threading for Xeon 8488C
void sha256avx512_batch(const uint8_t** data, unsigned char** hash, size_t num_inputs);

// Stream processing for continuous data
void sha256avx512_stream(const uint8_t** data_blocks, size_t num_blocks, size_t block_size,
                         unsigned char** hashes);

// Get optimal thread count for the current system
int sha256avx512_get_optimal_threads();

// Prefetch data into cache levels
void sha256avx512_prefetch(const void* ptr, int hint);

#endif  // SHA256_AVX512_H
