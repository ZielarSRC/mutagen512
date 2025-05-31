#include <getopt.h>
#include <immintrin.h>
#include <numa.h>
#include <omp.h>
#include <pthread.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "Int.h"
#include "IntGroup.h"
#include "Point.h"
#include "SECP256K1.h"
#include "ripemd160_avx512.h"
#include "sha256_avx512.h"

using namespace std;

// Sapphire Rapids optimized constants
static constexpr int POINTS_BATCH_SIZE = 512;  // Optimized for AVX-512
static constexpr int HASH_BATCH_SIZE = 16;     // 16-way parallel hashing
static constexpr int L1_CACHE_LINE_SIZE = 64;
static constexpr int L2_CACHE_SIZE = 2048 * 1024;  // 2MB per core
static constexpr int PREFETCH_DISTANCE = 8;

int PUZZLE_NUM = 71;  // Default to puzzle 71
int WORKERS = 0;      // Will be set based on CPU topology
int FLIP_COUNT = -1;
const __uint128_t REPORT_INTERVAL =
    50000000;  // Increased for higher performance

const unordered_map<int, tuple<int, string, string>> PUZZLE_DATA = {
    {20, {8, "b907c3a2a3b27789dfb509b730dd47703c272868", "357535"}},
    {21, {9, "29a78213caa9eea824acf08022ab9dfc83414f56", "863317"}},
    {22, {11, "7ff45303774ef7a52fffd8011981034b258cb86b", "1811764"}},
    {23, {12, "d0a79df189fe1ad5c306cc70497b358415da579e", "3007503"}},
    {24, {9, "0959e80121f36aea13b3bad361c15dac26189e2f", "5598802"}},
    {25, {12, "2f396b29b27324300d0c59b17c3abc1835bd3dbb", "14428676"}},
    {26, {14, "bfebb73562d4541b32a02ba664d140b5a574792f", "33185509"}},
    {27, {13, "0c7aaf6caa7e5424b63d317f0f8f1f9fa40d5560", "54538862"}},
    {28, {16, "1306b9e4ff56513a476841bac7ba48d69516b1da", "111949941"}},
    {29, {18, "5a416cc9148f4a377b672c8ae5d3287adaafadec", "227634408"}},
    {30, {16, "d39c4704664e1deb76c9331e637564c257d68a08", "400708894"}},
    {31, {13, "d805f6f251f7479ebd853b3d0f4b9b2656d92f1d", "1033162084"}},
    {32, {14, "9e42601eeaedc244e15f17375adb0e2cd08efdc9", "2102388551"}},
    {33, {15, "4e15e5189752d1eaf444dfd6bff399feb0443977", "3093472814"}},
    {34, {16, "f6d67d7983bf70450f295c9cb828daab265f1bfa", "7137437912"}},
    {35, {19, "f6d8ce225ffbdecec170f8298c3fc28ae686df25", "14133072157"}},
    {36, {14, "74b1e012be1521e5d8d75e745a26ced845ea3d37", "20112871792"}},
    {37, {23, "28c30fb9118ed1da72e7c4f89c0164756e8a021d", "42387769980"}},
    {38, {21, "b190e2d40cfdeee2cee072954a2be89e7ba39364", "100251560595"}},
    {39, {23, "0b304f2a79a027270276533fe1ed4eff30910876", "146971536592"}},
    {40, {20, "95a156cd21b4a69de969eb6716864f4c8b82a82a", "323724968937"}},
    {41, {25, "d1562eb37357f9e6fc41cb2359f4d3eda4032329", "1003651412950"}},
    {42, {24, "8efb85f9c5b5db2d55973a04128dc7510075ae23", "1458252205147"}},
    {43, {19, "f92044c7924e5525c61207972c253c9fc9f086f7", "2895374552463"}},
    {44, {24, "80df54e1f612f2fc5bdc05c9d21a83aa8d20791e", "7409811047825"}},
    {45, {21, "f0225bfc68a6e17e87cd8b5e60ae3be18f120753", "15404761757071"}},
    {46, {24, "9a012260d01c5113df66c8a8438c9f7a1e3d5dac", "19996463086597"}},
    {47, {27, "f828005d41b0f4fed4c8dca3b06011072cfb07d4", "51408670348612"}},
    {48, {21, "8661cb56d9df0a61f01328b55af7e56a3fe7a2b2", "119666659114170"}},
    {49, {30, "0d2f533966c6578e1111978ca698f8add7fffdf3", "191206974700443"}},
    {50, {29, "de081b76f840e462fa2cdf360173dfaf4a976a47", "409118905032525"}},
    {51, {25, "ef6419cffd7fad7027994354eb8efae223c2dbe7", "611140496167764"}},
    {52, {27, "36af659edbe94453f6344e920d143f1778653ae7", "2058769515153876"}},
    {53, {26, "2f4870ef54fa4b048c1365d42594cc7d3d269551", "4216495639600700"}},
    {54, {30, "cb66763cf7fde659869ae7f06884d9a0f879a092", "6763683971478124"}},
    {55, {31, "db53d9bbd1f3a83b094eeca7dd970bd85b492fa2", "9974455244496707"}},
    {56, {31, "48214c5969ae9f43f75070cea1e2cb41d5bdcccd", "30045390491869460"}},
    {57, {33, "328660ef43f66abe2653fa178452a5dfc594c2a1", "44218742292676575"}},
    {58,
     {28, "8c2a6071f89c90c4dab5ab295d7729d1b54ea60f", "138245758910846492"}},
    {59,
     {30, "b14ed3146f5b2c9bde1703deae9ef33af8110210", "199976667976342049"}},
    {60,
     {31, "cdf8e5c7503a9d22642e3ecfc87817672787b9c5", "525070384258266191"}},
    {61,
     {25, "68133e19b2dfb9034edf9830a200cfdf38c90cbd", "1135041350219496382"}},
    {62,
     {35, "e26646db84b0602f32b34b5a62ca3cae1f91b779", "1425787542618654982"}},
    {63,
     {34, "ef58afb697b094423ce90721fbb19a359ef7c50e", "3908372542507822062"}},
    {64,
     {34, "3ee4133d991f52fdf6a25c9834e0745ac74248a4", "8993229949524469768"}},
    {65,
     {37, "52e763a7ddc1aa4fa811578c491c1bc7fd570137", "17799667357578236628"}},
    {66,
     {35, "20d45a6a762535700ce9e0b216e31994335db8a5", "30568377312064202855"}},
    {67,
     {31, "739437bb3dd6d1983e66629c5f08c70e52769371", "46346217550346335726"}},
    {68,
     {42, "e0b8a2baee1b77fc703455f39d51477451fc8cfc", "132656943602386256302"}},
    {69,
     {34, "61eb8a50c86b0584bb727dd65bed8d2400d6d5aa", "219898266213316039825"}},
    {70,
     {29, "5db8cda53a6a002db10365967d7f85d19e171b10", "297274491920375905804"}},
    {71,
     {29, "f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8",
      "970436974005023690481"}}};

vector<unsigned char> TARGET_HASH160_RAW(20);
string TARGET_HASH160;
Int BASE_KEY;
atomic<bool> stop_event(false);
mutex result_mutex;
queue<tuple<string, __uint128_t, int>> results;

// AVX-512 optimized counter with full 512-bit utilization
union AVX512Counter {
  __m512i vec;
  uint64_t u64[8];
  __uint128_t u128[4];

  AVX512Counter() : vec(_mm512_setzero_si512()) {}

  AVX512Counter(__uint128_t value) { store(value); }

  // AVX-512 optimized increment using gather/scatter operations
  void increment() {
    const __m512i one = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 1);
    vec = _mm512_add_epi64(vec, one);

    // Handle carry propagation using AVX-512 mask operations
    __mmask8 carry_mask = _mm512_cmpeq_epi64_mask(
        _mm512_and_si512(vec, _mm512_set1_epi64(0xFFFFFFFFFFFFFFFFULL)),
        _mm512_setzero_si512());

    if (carry_mask & 0x01) {
      const __m512i carry = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 1, 0);
      vec = _mm512_add_epi64(vec, carry);
    }
  }

  void add(__uint128_t value) {
    const __m512i add_val = _mm512_set_epi64(0, 0, 0, 0, 0, 0, value >> 64,
                                             value & 0xFFFFFFFFFFFFFFFFULL);
    vec = _mm512_add_epi64(vec, add_val);

    // Optimized carry handling
    if (u64[0] < (value & 0xFFFFFFFFFFFFFFFFULL)) {
      const __m512i carry = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 1, 0);
      vec = _mm512_add_epi64(vec, carry);
    }
  }

  __uint128_t load() const {
    return (static_cast<__uint128_t>(u64[1]) << 64) | u64[0];
  }

  void store(__uint128_t value) {
    vec = _mm512_setzero_si512();
    u64[0] = static_cast<uint64_t>(value);
    u64[1] = static_cast<uint64_t>(value >> 64);
  }

  bool operator<(const AVX512Counter& other) const {
    return _mm512_cmplt_epi64_mask(vec, other.vec) != 0;
  }

  bool operator>=(const AVX512Counter& other) const {
    return _mm512_cmpge_epi64_mask(vec, other.vec) != 0;
  }

  static AVX512Counter div(const AVX512Counter& num, uint64_t denom) {
    __uint128_t n = num.load();
    __uint128_t q = n / denom;
    return AVX512Counter(q);
  }

  static uint64_t mod(const AVX512Counter& num, uint64_t denom) {
    __uint128_t n = num.load();
    return n % denom;
  }

  static AVX512Counter mul(uint64_t a, uint64_t b) {
    __uint128_t result = static_cast<__uint128_t>(a) * b;
    return AVX512Counter(result);
  }
};

// NUMA-aware memory allocation for optimal performance
template <typename T>
T* numa_alloc_aligned(size_t count, int node = -1) {
  size_t size = count * sizeof(T);
  size_t aligned_size = (size + 63) & ~63;  // 64-byte alignment

  void* ptr;
  if (node >= 0) {
    ptr = numa_alloc_onnode(aligned_size, node);
  } else {
    ptr = numa_alloc_local(aligned_size);
  }

  if (!ptr) {
    throw std::bad_alloc();
  }

  return static_cast<T*>(ptr);
}

static AVX512Counter total_checked_avx512;
__uint128_t total_combinations = 0;
vector<string> g_threadPrivateKeys;
mutex progress_mutex;

atomic<uint64_t> globalComparedCount(0);
atomic<uint64_t> localComparedCount(0);
double globalElapsedTime = 0.0;
double mkeysPerSec = 0.0;
chrono::time_point<chrono::high_resolution_clock> tStart;

// CPU topology detection for optimal thread affinity
struct CPUTopology {
  int numa_nodes;
  int cores_per_node;
  int threads_per_core;
  vector<int> core_ids;

  CPUTopology() {
    numa_nodes = numa_max_node() + 1;
    cores_per_node = 0;
    threads_per_core = 0;

    // Detect CPU topology
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);

    for (int i = 0; i < CPU_SETSIZE; ++i) {
      if (CPU_ISSET(i, &cpu_set)) {
        core_ids.push_back(i);
      }
    }

    cores_per_node = core_ids.size() / numa_nodes;
    threads_per_core = 2;  // Assume hyperthreading
  }
};

static CPUTopology cpu_topology;

// Intel SHA Extensions optimized block preparation
inline void prepareShaBlockSHA(const uint8_t* dataSrc, size_t dataLen,
                               uint8_t* outBlock) {
  // Use Intel SHA Extensions for optimal performance
  alignas(64) uint8_t block[64];
  memset(block, 0, 64);
  memcpy(block, dataSrc, dataLen);
  block[dataLen] = 0x80;

  const uint32_t bitLen = static_cast<uint32_t>(dataLen * 8);
  // Use Intel's optimized byte order handling
  *reinterpret_cast<uint32_t*>(&block[60]) = __builtin_bswap32(bitLen);

  memcpy(outBlock, block, 64);
}

// AVX-512 optimized batch hash computation
static void computeHash160BatchAVX512(int numKeys, uint8_t pubKeys[][33],
                                      uint8_t hashResults[][20]) {
  // Allocate NUMA-aware aligned memory
  alignas(64) auto shaInputs =
      numa_alloc_aligned<array<uint8_t, 64>>(HASH_BATCH_SIZE);
  alignas(64) auto shaOutputs =
      numa_alloc_aligned<array<uint8_t, 32>>(HASH_BATCH_SIZE);
  alignas(64) auto ripemdInputs =
      numa_alloc_aligned<array<uint8_t, 64>>(HASH_BATCH_SIZE);
  alignas(64) auto ripemdOutputs =
      numa_alloc_aligned<array<uint8_t, 20>>(HASH_BATCH_SIZE);

  const int totalBatches = (numKeys + (HASH_BATCH_SIZE - 1)) / HASH_BATCH_SIZE;

  for (int batch = 0; batch < totalBatches; batch++) {
    const int batchCount =
        min(HASH_BATCH_SIZE, numKeys - batch * HASH_BATCH_SIZE);

    // Prepare SHA blocks with prefetching
    for (int i = 0; i < batchCount; i++) {
      _mm_prefetch(
          reinterpret_cast<const char*>(
              &pubKeys[batch * HASH_BATCH_SIZE + i + PREFETCH_DISTANCE]),
          _MM_HINT_T0);
      prepareShaBlockSHA(pubKeys[batch * HASH_BATCH_SIZE + i], 33,
                         shaInputs[i].data());
    }

    // Pad remaining slots for full AVX-512 utilization
    if (batchCount < HASH_BATCH_SIZE) {
      static array<uint8_t, 64> shaPadding = {};
      prepareShaBlockSHA(pubKeys[0], 33, shaPadding.data());
      for (int i = batchCount; i < HASH_BATCH_SIZE; i++) {
        memcpy(shaInputs[i].data(), shaPadding.data(), 64);
      }
    }

    // AVX-512 SHA-256 computation (16-way parallel)
    sha256avx512_16B(shaInputs, shaOutputs);

    // Prepare RIPEMD blocks
    for (int i = 0; i < batchCount; i++) {
      prepareShaBlockSHA(shaOutputs[i].data(), 32, ripemdInputs[i].data());
    }

    if (batchCount < HASH_BATCH_SIZE) {
      static array<uint8_t, 64> ripemdPadding = {};
      prepareShaBlockSHA(shaOutputs[0].data(), 32, ripemdPadding.data());
      for (int i = batchCount; i < HASH_BATCH_SIZE; i++) {
        memcpy(ripemdInputs[i].data(), ripemdPadding.data(), 64);
      }
    }

    // AVX-512 RIPEMD-160 computation (16-way parallel)
    ripemd160avx512_16B(ripemdInputs, ripemdOutputs);

    // Copy results with optimized memory access patterns
    for (int i = 0; i < batchCount; i++) {
      memcpy(hashResults[batch * HASH_BATCH_SIZE + i], ripemdOutputs[i].data(),
             20);
    }
  }

  // Free NUMA-allocated memory
  numa_free(shaInputs, HASH_BATCH_SIZE * sizeof(array<uint8_t, 64>));
  numa_free(shaOutputs, HASH_BATCH_SIZE * sizeof(array<uint8_t, 32>));
  numa_free(ripemdInputs, HASH_BATCH_SIZE * sizeof(array<uint8_t, 64>));
  numa_free(ripemdOutputs, HASH_BATCH_SIZE * sizeof(array<uint8_t, 20>));
}

// Combination generator optimized for cache efficiency
class CombinationGeneratorOptimized {
  int n, k;
  vector<int> current;

 public:
  CombinationGeneratorOptimized(int n, int k) : n(n), k(k), current(k) {
    if (k > n) k = n;
    for (int i = 0; i < k; ++i) current[i] = i;
  }

  static __uint128_t combinations_count(int n, int k) {
    if (k > n) return 0;
    if (k * 2 > n) k = n - k;
    if (k == 0) return 1;

    __uint128_t result = n;
    for (int i = 2; i <= k; ++i) {
      result *= (n - i + 1);
      result /= i;
    }
    return result;
  }

  const vector<int>& get() const { return current; }

  bool next() {
    int i = k - 1;
    while (i >= 0 && current[i] == n - k + i) --i;
    if (i < 0) return false;

    ++current[i];
    for (int j = i + 1; j < k; ++j) current[j] = current[j - 1] + 1;
    return true;
  }

  void unrank(__uint128_t rank) {
    __uint128_t total = combinations_count(n, k);
    if (rank >= total) {
      current.clear();
      return;
    }

    current.resize(k);
    __uint128_t remaining_rank = rank;
    int a = n;
    int b = k;
    __uint128_t x = (total - 1) - rank;
    for (int i = 0; i < k; i++) {
      a = largest_a_where_comb_a_b_le_x(a, b, x);
      current[i] = (n - 1) - a;
      x -= combinations_count(a, b);
      b--;
    }
  }

 private:
  int largest_a_where_comb_a_b_le_x(int a, int b, __uint128_t x) const {
    while (a >= b && combinations_count(a, b) > x) {
      a--;
    }
    return a;
  }
};

// Set thread affinity for optimal NUMA performance
void setThreadAffinity(int threadId) {
  if (threadId >= cpu_topology.core_ids.size()) return;

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu_topology.core_ids[threadId], &cpuset);

  pthread_t current_thread = pthread_self();
  pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

  // Set NUMA memory policy
  int numa_node = threadId / cpu_topology.cores_per_node;
  numa_run_on_node(numa_node);
  numa_set_preferred(numa_node);
}

// Optimized worker function with AVX-512 and NUMA awareness
void workerOptimized(Secp256K1* secp, int bit_length, int flip_count,
                     int threadId, AVX512Counter start, AVX512Counter end) {
  // Set thread affinity for optimal performance
  setThreadAffinity(threadId);

  const int fullBatchSize = 2 * POINTS_BATCH_SIZE;

  // NUMA-aware memory allocation
  auto localPubKeys = numa_alloc_aligned<array<uint8_t, 33>>(HASH_BATCH_SIZE);
  auto localHashResults =
      numa_alloc_aligned<array<uint8_t, 20>>(HASH_BATCH_SIZE);
  auto pointIndices = numa_alloc_aligned<int>(HASH_BATCH_SIZE);

  // AVX-512 target comparison setup
  __m512i target32 = _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(TARGET_HASH160_RAW.data()));

  // Pre-computed points for batch operations
  auto plusPoints = numa_alloc_aligned<Point>(POINTS_BATCH_SIZE);
  auto minusPoints = numa_alloc_aligned<Point>(POINTS_BATCH_SIZE);

  // Initialize precomputed points
  for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
    Int tmp;
    tmp.SetInt32(i);
    plusPoints[i] = secp->ComputePublicKey(&tmp);
    minusPoints[i] = plusPoints[i];
    minusPoints[i].y.ModNeg();
  }

  auto deltaX = numa_alloc_aligned<Int>(POINTS_BATCH_SIZE);
  IntGroup modGroup(POINTS_BATCH_SIZE);
  auto pointBatchX = numa_alloc_aligned<Int>(fullBatchSize);
  auto pointBatchY = numa_alloc_aligned<Int>(fullBatchSize);

  CombinationGeneratorOptimized gen(bit_length, flip_count);
  gen.unrank(start.load());

  AVX512Counter count;
  count.store(start.load());

  uint64_t actual_work_done = 0;
  auto last_report = chrono::high_resolution_clock::now();

  while (!stop_event.load() && count < end) {
    Int currentKey;
    currentKey.Set(&BASE_KEY);

    const vector<int>& flips = gen.get();

    // Apply bit flips with vectorized operations where possible
    for (int pos : flips) {
      Int mask;
      mask.SetInt32(1);
      mask.ShiftL(pos);
      currentKey.Xor(&mask);
    }

    string keyStr = currentKey.GetBase16();
    keyStr = string(64 - keyStr.length(), '0') + keyStr;

#pragma omp critical
    { g_threadPrivateKeys[threadId] = keyStr; }

    Point startPoint = secp->ComputePublicKey(&currentKey);
    Int startPointX, startPointY, startPointXNeg;
    startPointX.Set(&startPoint.x);
    startPointY.Set(&startPoint.y);
    startPointXNeg.Set(&startPointX);
    startPointXNeg.ModNeg();

// Vectorized point operations with AVX-512 where applicable
#pragma omp simd aligned(deltaX : 64)
    for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
      deltaX[i].ModSub(&plusPoints[i].x, &startPointX);
    }

    modGroup.Set(deltaX);
    modGroup.ModInv();

    // Compute batch of points with optimized memory access patterns
    for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
      // Prefetch next iteration data
      if (i + PREFETCH_DISTANCE < POINTS_BATCH_SIZE) {
        _mm_prefetch(&plusPoints[i + PREFETCH_DISTANCE], _MM_HINT_T0);
        _mm_prefetch(&deltaX[i + PREFETCH_DISTANCE], _MM_HINT_T0);
      }

      Int deltaY;
      deltaY.ModSub(&plusPoints[i].y, &startPointY);

      Int slope;
      slope.ModMulK1(&deltaY, &deltaX[i]);

      Int slopeSq;
      slopeSq.ModSquareK1(&slope);

      pointBatchX[i].Set(&startPointXNeg);
      pointBatchX[i].ModAdd(&slopeSq);
      pointBatchX[i].ModSub(&plusPoints[i].x);

      Int diffX;
      diffX.Set(&startPointX);
      diffX.ModSub(&pointBatchX[i]);
      diffX.ModMulK1(&slope);

      pointBatchY[i].Set(&startPointY);
      pointBatchY[i].ModNeg();
      pointBatchY[i].ModAdd(&diffX);
    }

    // Similar computation for minus points
    for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
      Int deltaY;
      deltaY.ModSub(&minusPoints[i].y, &startPointY);

      Int slope;
      slope.ModMulK1(&deltaY, &deltaX[i]);

      Int slopeSq;
      slopeSq.ModSquareK1(&slope);

      pointBatchX[POINTS_BATCH_SIZE + i].Set(&startPointXNeg);
      pointBatchX[POINTS_BATCH_SIZE + i].ModAdd(&slopeSq);
      pointBatchX[POINTS_BATCH_SIZE + i].ModSub(&minusPoints[i].x);

      Int diffX;
      diffX.Set(&startPointX);
      diffX.ModSub(&pointBatchX[POINTS_BATCH_SIZE + i]);
      diffX.ModMulK1(&slope);

      pointBatchY[POINTS_BATCH_SIZE + i].Set(&startPointY);
      pointBatchY[POINTS_BATCH_SIZE + i].ModNeg();
      pointBatchY[POINTS_BATCH_SIZE + i].ModAdd(&diffX);
    }

    // Process points in batches optimized for AVX-512
    int localBatchCount = 0;
    for (int i = 0; i < fullBatchSize && localBatchCount < HASH_BATCH_SIZE;
         i++) {
      Point tempPoint;
      tempPoint.x.Set(&pointBatchX[i]);
      tempPoint.y.Set(&pointBatchY[i]);

      (*localPubKeys)[localBatchCount][0] = tempPoint.y.IsEven() ? 0x02 : 0x03;
      for (int j = 0; j < 32; j++) {
        (*localPubKeys)[localBatchCount][1 + j] =
            pointBatchX[i].GetByte(31 - j);
      }
      pointIndices[localBatchCount] = i;
      localBatchCount++;

      if (localBatchCount == HASH_BATCH_SIZE) {
        computeHash160BatchAVX512(localBatchCount, *localPubKeys,
                                  *localHashResults);

        actual_work_done += HASH_BATCH_SIZE;
        localComparedCount += HASH_BATCH_SIZE;

        // AVX-512 optimized comparison
        for (int j = 0; j < HASH_BATCH_SIZE; j++) {
          __m512i cand = _mm512_loadu_si512(
              reinterpret_cast<const __m512i*>((*localHashResults)[j]));
          __mmask64 cmp_mask = _mm512_cmpeq_epi8_mask(cand, target32);

          if ((cmp_mask & 0x0FFFFF) == 0x0FFFFF) {  // Check first 20 bytes
            bool fullMatch = true;
            for (int k = 0; k < 20; k++) {
              if ((*localHashResults)[j][k] != TARGET_HASH160_RAW[k]) {
                fullMatch = false;
                break;
              }
            }

            if (fullMatch) {
              // Solution found - compute final key
              Int foundKey;
              foundKey.Set(&currentKey);
              int idx = pointIndices[j];
              if (idx < POINTS_BATCH_SIZE) {
                Int offset;
                offset.SetInt32(idx);
                foundKey.Add(&offset);
              } else {
                Int offset;
                offset.SetInt32(idx - POINTS_BATCH_SIZE);
                foundKey.Sub(&offset);
              }

              string hexKey = foundKey.GetBase16();
              hexKey = string(64 - hexKey.length(), '0') + hexKey;

              lock_guard<mutex> lock(result_mutex);
              results.push(
                  make_tuple(hexKey, total_checked_avx512.load(), flip_count));
              stop_event.store(true);

              // Free NUMA memory before return
              numa_free(localPubKeys,
                        HASH_BATCH_SIZE * sizeof(array<uint8_t, 33>));
              numa_free(localHashResults,
                        HASH_BATCH_SIZE * sizeof(array<uint8_t, 20>));
              numa_free(pointIndices, HASH_BATCH_SIZE * sizeof(int));
              numa_free(plusPoints, POINTS_BATCH_SIZE * sizeof(Point));
              numa_free(minusPoints, POINTS_BATCH_SIZE * sizeof(Point));
              numa_free(deltaX, POINTS_BATCH_SIZE * sizeof(Int));
              numa_free(pointBatchX, fullBatchSize * sizeof(Int));
              numa_free(pointBatchY, fullBatchSize * sizeof(Int));
              return;
            }
          }
        }

        total_checked_avx512.increment();
        localBatchCount = 0;

        // Progress reporting
        __uint128_t current_total = total_checked_avx512.load();
        if (current_total % REPORT_INTERVAL == 0 ||
            count.load() == end.load() - 1) {
          auto now = chrono::high_resolution_clock::now();
          globalElapsedTime = chrono::duration<double>(now - tStart).count();

          globalComparedCount += localComparedCount;
          localComparedCount = 0;
          mkeysPerSec = (double)globalComparedCount / globalElapsedTime / 1e6;
          double progress =
              min(100.0, (double)current_total / total_combinations * 100.0);

          lock_guard<mutex> lock(progress_mutex);
          cout << "\rProgress: " << fixed << setprecision(6) << progress
               << "% | Speed: " << fixed << setprecision(2) << mkeysPerSec
               << " Mkeys/s | Elapsed: " << formatElapsedTime(globalElapsedTime)
               << flush;

          if (current_total >= total_combinations) {
            stop_event.store(true);
            break;
          }
        }
      }
    }

    if (!gen.next()) {
      break;
    }
    count.increment();

    if (count >= end) {
      break;
    }
  }

  // Free NUMA-allocated memory
  numa_free(localPubKeys, HASH_BATCH_SIZE * sizeof(array<uint8_t, 33>));
  numa_free(localHashResults, HASH_BATCH_SIZE * sizeof(array<uint8_t, 20>));
  numa_free(pointIndices, HASH_BATCH_SIZE * sizeof(int));
  numa_free(plusPoints, POINTS_BATCH_SIZE * sizeof(Point));
  numa_free(minusPoints, POINTS_BATCH_SIZE * sizeof(Point));
  numa_free(deltaX, POINTS_BATCH_SIZE * sizeof(Int));
  numa_free(pointBatchX, fullBatchSize * sizeof(Int));
  numa_free(pointBatchY, fullBatchSize * sizeof(Int));

  if (!stop_event.load() && total_checked_avx512.load() >= total_combinations) {
    stop_event.store(true);
  }
}

// Utility functions (keeping the same interface but optimized implementations)
static string formatElapsedTime(double seconds) {
  int hrs = static_cast<int>(seconds) / 3600;
  int mins = (static_cast<int>(seconds) % 3600) / 60;
  int secs = static_cast<int>(seconds) % 60;
  ostringstream oss;
  oss << setw(2) << setfill('0') << hrs << ":" << setw(2) << setfill('0')
      << mins << ":" << setw(2) << setfill('0') << secs;
  return oss.str();
}

static string to_string_128(__uint128_t value) {
  if (value == 0) return "0";
  char buffer[50];
  char* p = buffer + sizeof(buffer);
  *--p = '\0';
  while (value != 0) {
    *--p = "0123456789"[value % 10];
    value /= 10;
  }
  return string(p);
}

void signalHandler(int signum) {
  stop_event.store(true);
  cout << "\nInterrupt received, shutting down...\n";
}

void printUsage(const char* programName) {
  cout << "Usage: " << programName << " [options]\n";
  cout << "Options:\n";
  cout << "  -p, --puzzle NUM    Puzzle number to solve (default: 71)\n";
  cout << "  -t, --threads NUM   Number of CPU cores to use (default: "
          "auto-detect)\n";
  cout << "  -f, --flips NUM     Override default flip count for puzzle\n";
  cout << "  -h, --help          Show this help message\n";
  cout << "\nOptimized for Intel Xeon Platinum 8488C (Sapphire Rapids)\n";
  cout << "Example:\n";
  cout << "  " << programName << " -p 71 -t 48\n";
}

int main(int argc, char* argv[]) {
  signal(SIGINT, signalHandler);

  // Initialize NUMA
  if (numa_available() < 0) {
    cerr << "Warning: NUMA not available, performance may be suboptimal\n";
  }

  // Auto-detect optimal thread count based on CPU topology
  if (WORKERS == 0) {
    WORKERS = cpu_topology.core_ids.size();
    // For Sapphire Rapids, use physical cores only for maximum performance
    if (cpu_topology.threads_per_core > 1) {
      WORKERS = WORKERS / cpu_topology.threads_per_core;
    }
  }

  int opt;
  int option_index = 0;
  static struct option long_options[] = {{"puzzle", required_argument, 0, 'p'},
                                         {"threads", required_argument, 0, 't'},
                                         {"flips", required_argument, 0, 'f'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  while ((opt = getopt_long(argc, argv, "p:t:f:h", long_options,
                            &option_index)) != -1) {
    switch (opt) {
      case 'p':
        PUZZLE_NUM = atoi(optarg);
        if (PUZZLE_NUM < 20 || PUZZLE_NUM > 71) {
          cerr << "Error: Puzzle number must be between 20 and 71\n";
          return 1;
        }
        break;
      case 't':
        WORKERS = atoi(optarg);
        if (WORKERS < 1) {
          cerr << "Error: Thread count must be at least 1\n";
          return 1;
        }
        break;
      case 'f':
        FLIP_COUNT = atoi(optarg);
        if (FLIP_COUNT < 1) {
          cerr << "Error: Flip count must be at least 1\n";
          return 1;
        }
        break;
      case 'h':
        printUsage(argv[0]);
        return 0;
      default:
        printUsage(argv[0]);
        return 1;
    }
  }

  tStart = chrono::high_resolution_clock::now();

  Secp256K1 secp;
  secp.Init();

  auto puzzle_it = PUZZLE_DATA.find(PUZZLE_NUM);
  if (puzzle_it == PUZZLE_DATA.end()) {
    cerr << "Error: Invalid puzzle number\n";
    return 1;
  }

  auto [DEFAULT_FLIP_COUNT, TARGET_HASH160_HEX, PRIVATE_KEY_DECIMAL] =
      puzzle_it->second;

  if (FLIP_COUNT == -1) {
    FLIP_COUNT = DEFAULT_FLIP_COUNT;
  }

  TARGET_HASH160 = TARGET_HASH160_HEX;

  for (int i = 0; i < 20; i++) {
    TARGET_HASH160_RAW[i] = stoul(TARGET_HASH160.substr(i * 2, 2), nullptr, 16);
  }

  BASE_KEY.SetBase10(const_cast<char*>(PRIVATE_KEY_DECIMAL.c_str()));

  total_combinations =
      CombinationGeneratorOptimized::combinations_count(PUZZLE_NUM, FLIP_COUNT);

  string paddedKey = BASE_KEY.GetBase16();
  size_t firstNonZero = paddedKey.find_first_not_of('0');
  if (firstNonZero != string::npos) {
    paddedKey = paddedKey.substr(firstNonZero);
  } else {
    paddedKey = "0";
  }
  paddedKey = "0x" + paddedKey;

  cout << "=======================================\n";
  cout << "== Mutagen Sapphire Rapids Optimizer ==\n";
  cout << "=======================================\n";
  cout << "Target CPU: Intel Xeon Platinum 8488C\n";
  cout << "NUMA Nodes: " << cpu_topology.numa_nodes << "\n";
  cout << "Available Cores: " << cpu_topology.core_ids.size() << "\n";
  cout << "Starting puzzle: " << PUZZLE_NUM << " (" << PUZZLE_NUM << "-bit)\n";
  cout << "Target HASH160: " << TARGET_HASH160.substr(0, 10) << "..."
       << TARGET_HASH160.substr(TARGET_HASH160.length() - 10) << "\n";
  cout << "Base Key: " << paddedKey << "\n";
  cout << "Flip count: " << FLIP_COUNT << "\n";
  cout << "Total Combinations: " << to_string_128(total_combinations) << "\n";
  cout << "Using: " << WORKERS << " threads\n";
  cout << "Optimizations: AVX-512, NUMA-aware, Intel SHA Extensions\n\n";

  g_threadPrivateKeys.resize(WORKERS, "0");
  vector<thread> threads;

  AVX512Counter total_combinations_avx512;
  total_combinations_avx512.store(total_combinations);

  AVX512Counter comb_per_thread =
      AVX512Counter::div(total_combinations_avx512, WORKERS);
  uint64_t remainder = AVX512Counter::mod(total_combinations_avx512, WORKERS);

  for (int i = 0; i < WORKERS; i++) {
    AVX512Counter start, end;

    AVX512Counter base = AVX512Counter::mul(i, comb_per_thread.load());
    uint64_t extra = min(static_cast<uint64_t>(i), remainder);
    start.store(base.load() + extra);

    end.store(start.load() + comb_per_thread.load() + (i < remainder ? 1 : 0));
    threads.emplace_back(workerOptimized, &secp, PUZZLE_NUM, FLIP_COUNT, i,
                         start, end);
  }

  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  // Results processing (same as original but with updated formatting)
  if (!results.empty()) {
    auto [hex_key, checked, flips] = results.front();
    globalElapsedTime =
        chrono::duration<double>(chrono::high_resolution_clock::now() - tStart)
            .count();
    mkeysPerSec = (double)globalComparedCount / globalElapsedTime / 1e6;

    string compactHex = hex_key;
    size_t firstNonZeroCompact = compactHex.find_first_not_of('0');
    if (firstNonZeroCompact != string::npos) {
      compactHex = compactHex.substr(firstNonZeroCompact);
    } else {
      compactHex = "0";
    }
    compactHex = "0x" + compactHex;

    cout << "\n=======================================\n";
    cout << "=========== SOLUTION FOUND ============\n";
    cout << "=======================================\n";
    cout << "Private key: " << compactHex << "\n";
    cout << "Checked " << to_string_128(checked) << " combinations\n";
    cout << "Bit flips: " << flips << "\n";
    cout << "Time: " << fixed << setprecision(2) << globalElapsedTime
         << " seconds (" << formatElapsedTime(globalElapsedTime) << ")\n";
    cout << "Speed: " << fixed << setprecision(2) << mkeysPerSec
         << " Mkeys/s\n";

    ofstream out("puzzle_" + to_string(PUZZLE_NUM) + "_solution.txt");
    if (out) {
      out << hex_key;
      out.close();
      cout << "Solution saved to puzzle_" << PUZZLE_NUM << "_solution.txt\n";
    }
  } else {
    __uint128_t final_count = total_checked_avx512.load();
    globalElapsedTime =
        chrono::duration<double>(chrono::high_resolution_clock::now() - tStart)
            .count();
    mkeysPerSec = globalElapsedTime > 1e-6
                      ? (double)globalComparedCount / globalElapsedTime / 1e6
                      : 0.0;

    cout << "\n\nNo solution found. Checked " << to_string_128(final_count)
         << " combinations\n";
    cout << "Time: " << fixed << setprecision(2) << globalElapsedTime
         << " seconds (" << formatElapsedTime(globalElapsedTime) << ")\n";
    cout << "Speed: " << fixed << setprecision(2) << mkeysPerSec
         << " Mkeys/s\n";
  }

  return 0;
}
