#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "sha256_avx512.h"

namespace _sha256avx512 {

#ifdef _MSC_VER
#define ALIGN64 __declspec(align(64))
#else
#define ALIGN64 __attribute__((aligned(64)))
#endif

// Constants for SHA-256 - optimized for AVX-512
static const ALIGN64 uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Initialize SHA-256 state with initial hash values using AVX-512
inline void Initialize(__m512i* s) {
    s[0] = _mm512_set1_epi32(0x6a09e667);
    s[1] = _mm512_set1_epi32(0xbb67ae85);
    s[2] = _mm512_set1_epi32(0x3c6ef372);
    s[3] = _mm512_set1_epi32(0xa54ff53a);
    s[4] = _mm512_set1_epi32(0x510e527f);
    s[5] = _mm512_set1_epi32(0x9b05688c);
    s[6] = _mm512_set1_epi32(0x1f83d9ab);
    s[7] = _mm512_set1_epi32(0x5be0cd19);
}

// SHA-256 macros with AVX-512 intrinsics - optimized for Xeon Platinum 8488C
#define ROR_AVX512(x, n) \
    _mm512_or_si512(_mm512_srli_epi32(x, n), _mm512_slli_epi32(x, 32 - (n)))
#define SHR_AVX512(x, n) _mm512_srli_epi32(x, n)

#define S0_AVX512(x) \
    (_mm512_xor_si512(ROR_AVX512(x, 2), _mm512_xor_si512(ROR_AVX512(x, 13), ROR_AVX512(x, 22))))
#define S1_AVX512(x) \
    (_mm512_xor_si512(ROR_AVX512(x, 6), _mm512_xor_si512(ROR_AVX512(x, 11), ROR_AVX512(x, 25))))
#define s0_AVX512(x) \
    (_mm512_xor_si512(ROR_AVX512(x, 7), _mm512_xor_si512(ROR_AVX512(x, 18), SHR_AVX512(x, 3))))
#define s1_AVX512(x) \
    (_mm512_xor_si512(ROR_AVX512(x, 17), _mm512_xor_si512(ROR_AVX512(x, 19), SHR_AVX512(x, 10))))

#define Ch_AVX512(x, y, z) \
    _mm512_xor_si512(_mm512_and_si512(x, y), _mm512_andnot_si512(x, z))
#define Maj_AVX512(x, y, z) \
    _mm512_or_si512(_mm512_and_si512(x, y), _mm512_and_si512(z, _mm512_or_si512(x, y)))

#define Round_AVX512(a, b, c, d, e, f, g, h, Kt, Wt) \
    { \
        __m512i T1 = _mm512_add_epi32( \
            h, \
            _mm512_add_epi32( \
                S1_AVX512(e), _mm512_add_epi32(Ch_AVX512(e, f, g), _mm512_add_epi32(Kt, Wt)))); \
        __m512i T2 = _mm512_add_epi32(S0_AVX512(a), Maj_AVX512(a, b, c)); \
        h = g; \
        g = f; \
        f = e; \
        e = _mm512_add_epi32(d, T1); \
        d = c; \
        c = b; \
        b = a; \
        a = _mm512_add_epi32(T1, T2); \
    }

inline void Transform(__m512i* state, const uint8_t* data[16]) {
    __m512i a = state[0], b = state[1], c = state[2], d = state[3];
    __m512i e = state[4], f = state[5], g = state[6], h = state[7];
    __m512i W[16];

    // Prepare message schedule W[0..15] with AVX-512 optimizations
    for (int t = 0; t < 16; t += 4) {
        for (int i = 0; i < 4; ++i) {
            uint32_t wt[16];
            for (int j = 0; j < 16; ++j) {
                const uint8_t* ptr = data[j] + (t + i) * 4;
                wt[j] = ((uint32_t)ptr[0] << 24) | ((uint32_t)ptr[1] << 16) |
                        ((uint32_t)ptr[2] << 8) | ptr[3];
            }
            W[t + i] = _mm512_setr_epi32(
                wt[0], wt[1], wt[2], wt[3], wt[4], wt[5], wt[6], wt[7],
                wt[8], wt[9], wt[10], wt[11], wt[12], wt[13], wt[14], wt[15]
            );
        }
    }

    // Main loop of SHA-256 - optimized for AVX-512
    // First 16 rounds
    for (int t = 0; t < 16; ++t) {
        Round_AVX512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[t]), W[t]);
    }

    // Remaining 48 rounds with message schedule expansion
    for (int t = 16; t < 64; ++t) {
        __m512i newW = _mm512_add_epi32(
            _mm512_add_epi32(s1_AVX512(W[(t - 2) & 0xf]), W[(t - 7) & 0xf]),
            _mm512_add_epi32(s0_AVX512(W[(t - 15) & 0xf]), W[(t - 16) & 0xf]));
        W[t & 0xf] = newW;
        Round_AVX512(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[t]), newW);
    }

    // Update state with vectorized addition
    state[0] = _mm512_add_epi32(state[0], a);
    state[1] = _mm512_add_epi32(state[1], b);
    state[2] = _mm512_add_epi32(state[2], c);
    state[3] = _mm512_add_epi32(state[3], d);
    state[4] = _mm512_add_epi32(state[4], e);
    state[5] = _mm512_add_epi32(state[5], f);
    state[6] = _mm512_add_epi32(state[6], g);
    state[7] = _mm512_add_epi32(state[7], h);
}

}  // namespace _sha256avx512

// Main AVX-512 function for 16 parallel SHA-256 computations
void sha256avx512_16B(
    const uint8_t* data0, const uint8_t* data1, const uint8_t* data2, const uint8_t* data3,
    const uint8_t* data4, const uint8_t* data5, const uint8_t* data6, const uint8_t* data7,
    const uint8_t* data8, const uint8_t* data9, const uint8_t* data10, const uint8_t* data11,
    const uint8_t* data12, const uint8_t* data13, const uint8_t* data14, const uint8_t* data15,
    unsigned char* hash0, unsigned char* hash1, unsigned char* hash2, unsigned char* hash3,
    unsigned char* hash4, unsigned char* hash5, unsigned char* hash6, unsigned char* hash7,
    unsigned char* hash8, unsigned char* hash9, unsigned char* hash10, unsigned char* hash11,
    unsigned char* hash12, unsigned char* hash13, unsigned char* hash14, unsigned char* hash15) {
    
    ALIGN64 __m512i state[8];
    const uint8_t* data[16] = {
        data0, data1, data2, data3, data4, data5, data6, data7,
        data8, data9, data10, data11, data12, data13, data14, data15
    };
    unsigned char* hashArray[16] = {
        hash0, hash1, hash2, hash3, hash4, hash5, hash6, hash7,
        hash8, hash9, hash10, hash11, hash12, hash13, hash14, hash15
    };

    // Prefetch input data for optimal cache performance
    for (int i = 0; i < 16; i++) {
        _mm_prefetch((char*)data[i], _MM_HINT_T0);
    }

    // Initialize the state with the initial hash values
    _sha256avx512::Initialize(state);

    // Process the data blocks
    _sha256avx512::Transform(state, data);

    // Store the resulting state
    ALIGN64 uint32_t digest[8][16];  // digest[state_index][element_index]

    // Extract and store hash values using optimized stores
    for (int i = 0; i < 8; ++i) {
        _mm512_store_si512((__m512i*)digest[i], state[i]);
    }

    // Convert and store final hash values with byte swapping
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 8; ++j) {
            uint32_t word = digest[j][i];
            #ifdef _MSC_VER
            word = _byteswap_ulong(word);
            #else
            word = __builtin_bswap32(word);
            #endif
            memcpy(hashArray[i] + j * 4, &word, 4);
        }
    }
    
    // Memory fence to ensure all operations complete
    _mm_mfence();
}

// Backward compatibility wrapper for 8 parallel computations
void sha256avx512_8B(
    const uint8_t* data0, const uint8_t* data1, const uint8_t* data2, const uint8_t* data3,
    const uint8_t* data4, const uint8_t* data5, const uint8_t* data6, const uint8_t* data7,
    unsigned char* hash0, unsigned char* hash1, unsigned char* hash2, unsigned char* hash3,
    unsigned char* hash4, unsigned char* hash5, unsigned char* hash6, unsigned char* hash7) {
    
    // Create dummy data for unused lanes
    static const uint8_t dummy_data[64] = {0};
    static unsigned char dummy_hash[32];
    
    sha256avx512_16B(
        data0, data1, data2, data3, data4, data5, data6, data7,
        dummy_data, dummy_data, dummy_data, dummy_data, dummy_data, dummy_data, dummy_data, dummy_data,
        hash0, hash1, hash2, hash3, hash4, hash5, hash6, hash7,
        dummy_hash, dummy_hash, dummy_hash, dummy_hash, dummy_hash, dummy_hash, dummy_hash, dummy_hash
    );
}

// Batch processing for variable number of inputs
void sha256avx512_batch(const uint8_t** data_array, unsigned char** hash_array, int count) {
    
    const int batch_size = 16;
    
    for (int i = 0; i < count; i += batch_size) {
        int remaining = (count - i < batch_size) ? count - i : batch_size;
        
        // Prepare batch arrays
        const uint8_t* batch_data[16];
        unsigned char* batch_hash[16];
        static const uint8_t dummy_data[64] = {0};
        static unsigned char dummy_hash[32];
        
        // Fill actual data
        for (int j = 0; j < remaining; j++) {
            batch_data[j] = data_array[i + j];
            batch_hash[j] = hash_array[i + j];
        }
        
        // Fill dummy data for unused lanes
        for (int j = remaining; j < 16; j++) {
            batch_data[j] = dummy_data;
            batch_hash[j] = dummy_hash;
        }
        
        // Process batch
        sha256avx512_16B(
            batch_data[0], batch_data[1], batch_data[2], batch_data[3],
            batch_data[4], batch_data[5], batch_data[6], batch_data[7],
            batch_data[8], batch_data[9], batch_data[10], batch_data[11],
            batch_data[12], batch_data[13], batch_data[14], batch_data[15],
            batch_hash[0], batch_hash[1], batch_hash[2], batch_hash[3],
            batch_hash[4], batch_hash[5], batch_hash[6], batch_hash[7],
            batch_hash[8], batch_hash[9], batch_hash[10], batch_hash[11],
            batch_hash[12], batch_hash[13], batch_hash[14], batch_hash[15]
        );
    }
}
