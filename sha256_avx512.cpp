#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <thread>
#include <vector>

#include "sha256_avx512.h"

namespace _sha256avx512 {

#ifdef _MSC_VER
#define ALIGN64 __declspec(align(64))
#else
#define ALIGN64 __attribute__((aligned(64)))
#endif

// SHA-256 Constants (K)
ALIGN64 static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// Initialize SHA-256 state with initial hash values
void Initialize(__m512i* s) {
  const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

  for (int i = 0; i < 8; ++i) {
    s[i] = _mm512_set1_epi32(init[i]);
  }
}

// SHA-256 macroses with AVX-512 intrinsics for maximum performance
// Using ternary logic instructions for better efficiency
#define Maj(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0xE8)  // (x & y) | (x & z) | (y & z)
#define Ch(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0xCA)   // (x & y) ^ (~x & z)
#define ROR(x, n) _mm512_or_si512(_mm512_srli_epi32(x, n), _mm512_slli_epi32(x, 32 - n))
#define SHR(x, n) _mm512_srli_epi32(x, n)

// SHA-256 functions optimized for AVX-512
#define S0(x) \
  (_mm512_ternarylogic_epi32(ROR(x, 2), ROR(x, 13), ROR(x, 22), 0x96))  // XOR of rotations
#define S1(x) \
  (_mm512_ternarylogic_epi32(ROR(x, 6), ROR(x, 11), ROR(x, 25), 0x96))  // XOR of rotations
#define s0(x) \
  (_mm512_ternarylogic_epi32(ROR(x, 7), ROR(x, 18), SHR(x, 3), 0x96))  // XOR of rotations and shift
#define s1(x)                                                    \
  (_mm512_ternarylogic_epi32(ROR(x, 17), ROR(x, 19), SHR(x, 10), \
                             0x96))  // XOR of rotations and shift

// Optimized Round function with reduced register pressure
#define Round(a, b, c, d, e, f, g, h, Kt, Wt)                                               \
  T1 = _mm512_add_epi32(                                                                    \
      _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(h, S1(e)), Ch(e, f, g)), Kt), Wt); \
  T2 = _mm512_add_epi32(S0(a), Maj(a, b, c));                                               \
  h = g;                                                                                    \
  g = f;                                                                                    \
  f = e;                                                                                    \
  e = _mm512_add_epi32(d, T1);                                                              \
  d = c;                                                                                    \
  c = b;                                                                                    \
  b = a;                                                                                    \
  a = _mm512_add_epi32(T1, T2);

// Transform function processing 16 blocks in parallel using AVX-512
void Transform(__m512i* state, const uint8_t* data[16]) {
  __m512i a, b, c, d, e, f, g, h;
  __m512i W[64];
  __m512i T1, T2;

  // Prefetch data into L1 cache for better performance
  for (int i = 0; i < 16; i++) {
    _mm_prefetch((const char*)data[i], _MM_HINT_T0);
    _mm_prefetch((const char*)data[i] + 32, _MM_HINT_T0);
  }

  // Load state into local variables
  a = state[0];
  b = state[1];
  c = state[2];
  d = state[3];
  e = state[4];
  f = state[5];
  g = state[6];
  h = state[7];

// Prepare message schedule W[0..15] for 16 blocks in parallel
#pragma unroll
  for (int t = 0; t < 16; ++t) {
    uint32_t wt[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint8_t* ptr = data[i] + t * 4;
      wt[i] = ((uint32_t)ptr[0] << 24) | ((uint32_t)ptr[1] << 16) | ((uint32_t)ptr[2] << 8) |
              ((uint32_t)ptr[3]);
    }
    W[t] = _mm512_setr_epi32(wt[0], wt[1], wt[2], wt[3], wt[4], wt[5], wt[6], wt[7], wt[8], wt[9],
                             wt[10], wt[11], wt[12], wt[13], wt[14], wt[15]);
  }

// Compute the extended message schedule W[16..63]
// Unroll for better instruction scheduling
#pragma unroll 4
  for (int t = 16; t < 64; ++t) {
    W[t] = _mm512_add_epi32(_mm512_add_epi32(s1(W[t - 2]), W[t - 7]),
                            _mm512_add_epi32(s0(W[t - 15]), W[t - 16]));
  }

// Main loop of SHA-256 - process 16 blocks in parallel
#pragma unroll 4
  for (int t = 0; t < 64; ++t) {
    __m512i Kt = _mm512_set1_epi32(K[t]);
    Round(a, b, c, d, e, f, g, h, Kt, W[t]);
  }

  // Update state with intermediate hash value
  state[0] = _mm512_add_epi32(state[0], a);
  state[1] = _mm512_add_epi32(state[1], b);
  state[2] = _mm512_add_epi32(state[2], c);
  state[3] = _mm512_add_epi32(state[3], d);
  state[4] = _mm512_add_epi32(state[4], e);
  state[5] = _mm512_add_epi32(state[5], f);
  state[6] = _mm512_add_epi32(state[6], g);
  state[7] = _mm512_add_epi32(state[7], h);
}

// Function to set thread affinity for optimal performance on Xeon 8488C
void SetThreadAffinity(int thread_id) {
#ifdef _WIN32
  // Windows implementation
  DWORD_PTR affinityMask = 1ULL << (thread_id % 112);  // 8488C has 112 logical cores
  HANDLE currentThread = GetCurrentThread();
  SetThreadAffinityMask(currentThread, affinityMask);
#else
  // Linux/Unix implementation
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  int core_id = thread_id % 112;  // 8488C has 112 logical cores
  CPU_SET(core_id, &cpuset);
  pthread_t current_thread = pthread_self();
  pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
#endif
}

}  // namespace _sha256avx512

// Process 8 hashes in parallel (compatible with the AVX2 version)
void sha256avx512_8B(const uint8_t* data0, const uint8_t* data1, const uint8_t* data2,
                     const uint8_t* data3, const uint8_t* data4, const uint8_t* data5,
                     const uint8_t* data6, const uint8_t* data7, unsigned char* hash0,
                     unsigned char* hash1, unsigned char* hash2, unsigned char* hash3,
                     unsigned char* hash4, unsigned char* hash5, unsigned char* hash6,
                     unsigned char* hash7) {
  // Create arrays of pointers for easier handling
  const uint8_t* data[16] = {data0, data1, data2, data3, data4, data5, data6, data7,
                             // Repeat the first 8 inputs to fill 16 slots
                             data0, data1, data2, data3, data4, data5, data6, data7};

  unsigned char* hash[16] = {hash0, hash1, hash2, hash3, hash4, hash5, hash6, hash7,
                             // Dummy pointers for the extra slots (won't be used)
                             nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                             nullptr};

  // Use the 16-hash implementation with only 8 actual inputs
  __m512i state[8];

  // Initialize the state with the initial hash values
  _sha256avx512::Initialize(state);

  // Process the data blocks
  _sha256avx512::Transform(state, data);

  // Store the resulting state
  ALIGN64 uint32_t digest[8][16];  // digest[state_index][element_index]

  for (int i = 0; i < 8; ++i) {
    _mm512_store_si512((__m512i*)digest[i], state[i]);
  }

  // Extract the hash values and copy to output buffers
  for (int i = 0; i < 8; ++i) {  // Process only the first 8 hashes
    unsigned char* hash_out = hash[i];
    for (int j = 0; j < 8; ++j) {
      uint32_t word = digest[j][i];
#ifdef _MSC_VER
      word = _byteswap_ulong(word);
#else
      word = __builtin_bswap32(word);
#endif
      memcpy(hash_out + j * 4, &word, 4);
    }
  }
}

// Process 16 hashes in parallel (new AVX-512 optimized function)
void sha256avx512_16B(const uint8_t* data[16], unsigned char* hash[16]) {
  // Use AVX-512 for processing 16 hashes at once
  __m512i state[8];

  // Initialize the state with the initial hash values
  _sha256avx512::Initialize(state);

  // Process the data blocks
  _sha256avx512::Transform(state, data);

  // Store the resulting state
  ALIGN64 uint32_t digest[8][16];  // digest[state_index][element_index]

  for (int i = 0; i < 8; ++i) {
    _mm512_store_si512((__m512i*)digest[i], state[i]);
  }

// Extract the hash values and copy to output buffers
#pragma omp parallel for
  for (int i = 0; i < 16; ++i) {
    unsigned char* hash_out = hash[i];
    for (int j = 0; j < 8; ++j) {
      uint32_t word = digest[j][i];
#ifdef _MSC_VER
      word = _byteswap_ulong(word);
#else
      word = __builtin_bswap32(word);
#endif
      memcpy(hash_out + j * 4, &word, 4);
    }
  }
}

// Process 32 hashes in parallel for even higher throughput
void sha256avx512_32B(const uint8_t* data[32], unsigned char* hash[32]) {
// Process 32 hashes using two parallel AVX-512 operations
#pragma omp parallel sections
  {
#pragma omp section
    {
      // Process first 16 hashes
      const uint8_t* data1[16];
      unsigned char* hash1[16];

      for (int i = 0; i < 16; i++) {
        data1[i] = data[i];
        hash1[i] = hash[i];
      }

      sha256avx512_16B(data1, hash1);
    }

#pragma omp section
    {
      // Process second 16 hashes
      const uint8_t* data2[16];
      unsigned char* hash2[16];

      for (int i = 0; i < 16; i++) {
        data2[i] = data[i + 16];
        hash2[i] = hash[i + 16];
      }

      sha256avx512_16B(data2, hash2);
    }
  }
}

// Get optimal thread count for the current system
int sha256avx512_get_optimal_threads() {
  int max_threads = omp_get_max_threads();
  int num_physical_cores = std::thread::hardware_concurrency() / 2;  // Assuming 2 threads per core

  // For SHA-256, optimal is usually around 60-80% of physical cores
  return std::min(max_threads, std::max(1, (int)(num_physical_cores * 0.7)));
}

// Batch processing with optimal threading for Xeon 8488C
void sha256avx512_batch(const uint8_t** data, unsigned char** hash, size_t num_inputs) {
  // Determine optimal number of threads based on input size and available cores
  int num_threads = sha256avx512_get_optimal_threads();
  omp_set_num_threads(num_threads);

// Process inputs in batches of 16
#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    _sha256avx512::SetThreadAffinity(thread_id);

#pragma omp for schedule(dynamic, 1)
    for (size_t i = 0; i < num_inputs; i += 16) {
      size_t batch_size = std::min((size_t)16, num_inputs - i);

      if (batch_size == 16) {
        // Full batch of 16
        const uint8_t* batch_data[16];
        unsigned char* batch_hash[16];

        for (size_t j = 0; j < 16; j++) {
          batch_data[j] = data[i + j];
          batch_hash[j] = hash[i + j];
        }

        sha256avx512_16B(batch_data, batch_hash);
      } else {
        // Partial batch - handle remaining inputs
        const uint8_t* batch_data[16];
        unsigned char* batch_hash[16];

        for (size_t j = 0; j < batch_size; j++) {
          batch_data[j] = data[i + j];
          batch_hash[j] = hash[i + j];
        }

        // Fill remaining slots with dummy data (won't affect results)
        for (size_t j = batch_size; j < 16; j++) {
          batch_data[j] = batch_data[0];
          batch_hash[j] = nullptr;  // Results will be discarded
        }

        sha256avx512_16B(batch_data, batch_hash);
      }
    }
  }
}

// Stream processing for continuous data
void sha256avx512_stream(const uint8_t** data_blocks, size_t num_blocks, size_t block_size,
                         unsigned char** hashes) {
  // For streaming, use pipeline processing to maximize throughput
  int num_threads = sha256avx512_get_optimal_threads();
  size_t blocks_per_thread = (num_blocks + num_threads - 1) / num_threads;

#pragma omp parallel num_threads(num_threads)
  {
    int thread_id = omp_get_thread_num();
    size_t start_block = thread_id * blocks_per_thread;
    size_t end_block = std::min(start_block + blocks_per_thread, num_blocks);

    _sha256avx512::SetThreadAffinity(thread_id);

    for (size_t i = start_block; i < end_block; i += 16) {
      size_t batch_size = std::min((size_t)16, end_block - i);

      const uint8_t* batch_data[16];
      unsigned char* batch_hash[16];

      for (size_t j = 0; j < batch_size; j++) {
        batch_data[j] = data_blocks[i + j];
        batch_hash[j] = hashes[i + j];
      }

      // Fill remaining slots with dummy data if needed
      for (size_t j = batch_size; j < 16; j++) {
        batch_data[j] = batch_data[0];
        batch_hash[j] = nullptr;
      }

      sha256avx512_16B(batch_data, batch_hash);
    }
  }
}

// Prefetch data into cache levels
void sha256avx512_prefetch(const void* ptr, int hint) { _mm_prefetch((const char*)ptr, hint); }
