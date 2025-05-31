#include <getopt.h>
#include <immintrin.h>
#include <omp.h>

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

// Constants optimized for Intel Xeon Platinum 8488C
int PUZZLE_NUM = 71;  // Default to puzzle 71 (focus on the specific challenge)
int WORKERS = omp_get_num_procs();
int FLIP_COUNT = -1;
const __uint128_t REPORT_INTERVAL = 10000000;
static constexpr int POINTS_BATCH_SIZE =
    512;  // Increased batch size for better throughput
static constexpr int HASH_BATCH_SIZE =
    16;  // Using AVX-512 to process 16 hashes at once

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

// Enhanced AVX-512 counter for higher performance
union AVX512Counter {
  __m512i vec;
  uint64_t u64[8];
  __uint128_t u128[4];

  AVX512Counter() : vec(_mm512_setzero_si512()) {}

  AVX512Counter(__uint128_t value) { store(value); }

  void increment() {
    __m512i one = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 1);
    vec = _mm512_add_epi64(vec, one);

    if (u64[0] == 0) {
      __m512i carry = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 1, 0);
      vec = _mm512_add_epi64(vec, carry);
    }
  }

  void add(__uint128_t value) {
    __m512i add_val = _mm512_set_epi64(0, 0, 0, 0, 0, 0, value >> 64, value);
    vec = _mm512_add_epi64(vec, add_val);

    if (u64[0] < (value & 0xFFFFFFFFFFFFFFFFULL)) {
      __m512i carry = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 1, 0);
      vec = _mm512_add_epi64(vec, carry);
    }
  }

  __uint128_t load() const {
    return (static_cast<__uint128_t>(u64[1]) << 64) | u64[0];
  }

  void store(__uint128_t value) {
    u64[0] = static_cast<uint64_t>(value);
    u64[1] = static_cast<uint64_t>(value >> 64);
    u64[2] = 0;
    u64[3] = 0;
    u64[4] = 0;
    u64[5] = 0;
    u64[6] = 0;
    u64[7] = 0;
  }

  bool operator<(const AVX512Counter& other) const {
    if (u64[1] != other.u64[1]) return u64[1] < other.u64[1];
    return u64[0] < other.u64[0];
  }

  bool operator>=(const AVX512Counter& other) const {
    if (u64[1] != other.u64[1]) return u64[1] > other.u64[1];
    return u64[0] >= other.u64[0];
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

static AVX512Counter total_checked_avx;
__uint128_t total_combinations = 0;
vector<string> g_threadPrivateKeys;
mutex progress_mutex;

atomic<uint64_t> globalComparedCount(0);
atomic<uint64_t> localComparedCount(0);
double globalElapsedTime = 0.0;
double mkeysPerSec = 0.0;
chrono::time_point<chrono::high_resolution_clock> tStart;

static std::string formatElapsedTime(double seconds) {
  int hrs = static_cast<int>(seconds) / 3600;
  int mins = (static_cast<int>(seconds) % 3600) / 60;
  int secs = static_cast<int>(seconds) % 60;
  std::ostringstream oss;
  oss << std::setw(2) << std::setfill('0') << hrs << ":" << std::setw(2)
      << std::setfill('0') << mins << ":" << std::setw(2) << std::setfill('0')
      << secs;
  return oss.str();
}

static std::string to_string_128(__uint128_t value) {
  if (value == 0) return "0";
  char buffer[50];
  char* p = buffer + sizeof(buffer);
  *--p = '\0';
  while (value != 0) {
    *--p = "0123456789"[value % 10];
    value /= 10;
  }
  return std::string(p);
}

void signalHandler(int signum) {
  stop_event.store(true);
  cout << "\nInterrupt received, shutting down...\n";
}

class CombinationGenerator {
  int n, k;
  std::vector<int> current;

 public:
  CombinationGenerator(int n, int k) : n(n), k(k), current(k) {
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

  static __m512i combinations_count_avx(int n, int k) {
    alignas(64) uint64_t counts[8];
    for (int i = 0; i < 8; i++) {
      counts[i] = combinations_count(n + i, k);
    }
    return _mm512_load_si512((__m512i*)counts);
  }

  const std::vector<int>& get() const { return current; }

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

// Optimized function to prepare SHA blocks using AVX-512
inline void prepareShaBlocks(const uint8_t* dataSrc[16], __uint128_t dataLen,
                             uint8_t* outBlocks[16]) {
  // Use AVX-512 to process 16 blocks in parallel
  for (int i = 0; i < 16; i++) {
    std::fill_n(outBlocks[i], 64, 0);
    std::memcpy(outBlocks[i], dataSrc[i], dataLen);
    outBlocks[i][dataLen] = 0x80;

    const uint32_t bitLen = (uint32_t)(dataLen * 8);
    outBlocks[i][60] = (uint8_t)((bitLen >> 24) & 0xFF);
    outBlocks[i][61] = (uint8_t)((bitLen >> 16) & 0xFF);
    outBlocks[i][62] = (uint8_t)((bitLen >> 8) & 0xFF);
    outBlocks[i][63] = (uint8_t)(bitLen & 0xFF);
  }
}

// Optimized function to prepare RIPEMD blocks using AVX-512
inline void prepareRipemdBlocks(const uint8_t* dataSrc[16],
                                uint8_t* outBlocks[16]) {
  for (int i = 0; i < 16; i++) {
    std::fill_n(outBlocks[i], 64, 0);
    std::memcpy(outBlocks[i], dataSrc[i], 32);
    outBlocks[i][32] = 0x80;

    const uint32_t bitLen = 256;
    outBlocks[i][60] = (uint8_t)((bitLen >> 24) & 0xFF);
    outBlocks[i][61] = (uint8_t)((bitLen >> 16) & 0xFF);
    outBlocks[i][62] = (uint8_t)((bitLen >> 8) & 0xFF);
    outBlocks[i][63] = (uint8_t)(bitLen & 0xFF);
  }
}

// Optimized hash calculation using AVX-512 for Intel Xeon
static void computeHash160BatchBin(int numKeys, uint8_t pubKeys[][33],
                                   uint8_t hashResults[][20]) {
  alignas(64) std::array<std::array<uint8_t, 64>, HASH_BATCH_SIZE> shaInputs;
  alignas(64) std::array<std::array<uint8_t, 32>, HASH_BATCH_SIZE> shaOutputs;
  alignas(64) std::array<std::array<uint8_t, 64>, HASH_BATCH_SIZE> ripemdInputs;
  alignas(64) std::array<std::array<uint8_t, 20>, HASH_BATCH_SIZE>
      ripemdOutputs;

  const __uint128_t totalBatches =
      (numKeys + (HASH_BATCH_SIZE - 1)) / HASH_BATCH_SIZE;

  for (__uint128_t batch = 0; batch < totalBatches; batch++) {
    const __uint128_t batchCount = std::min<__uint128_t>(
        HASH_BATCH_SIZE, numKeys - batch * HASH_BATCH_SIZE);

    // Prepare SHA input blocks
    const uint8_t* shaInputPtrs[HASH_BATCH_SIZE];
    uint8_t* shaOutputPtrs[HASH_BATCH_SIZE];

    for (__uint128_t i = 0; i < batchCount; i++) {
      uint8_t* ptr = shaInputs[i].data();
      std::fill_n(ptr, 64, 0);
      std::memcpy(ptr, pubKeys[batch * HASH_BATCH_SIZE + i], 33);
      ptr[33] = 0x80;  // padding

      // Length in bits (33 bytes = 264 bits)
      uint32_t bitLen = 33 * 8;
      ptr[60] = (uint8_t)((bitLen >> 24) & 0xFF);
      ptr[61] = (uint8_t)((bitLen >> 16) & 0xFF);
      ptr[62] = (uint8_t)((bitLen >> 8) & 0xFF);
      ptr[63] = (uint8_t)(bitLen & 0xFF);

      shaInputPtrs[i] = ptr;
      shaOutputPtrs[i] = shaOutputs[i].data();
    }

    // Pad with identical data for full batch processing
    if (batchCount < HASH_BATCH_SIZE) {
      for (__uint128_t i = batchCount; i < HASH_BATCH_SIZE; i++) {
        shaInputPtrs[i] = shaInputs[0].data();
        shaOutputPtrs[i] = shaOutputs[i].data();
      }
    }

    // Process SHA-256 using AVX-512 (16 blocks at once)
    sha256avx512_16B(
        shaInputPtrs[0], shaInputPtrs[1], shaInputPtrs[2], shaInputPtrs[3],
        shaInputPtrs[4], shaInputPtrs[5], shaInputPtrs[6], shaInputPtrs[7],
        shaInputPtrs[8], shaInputPtrs[9], shaInputPtrs[10], shaInputPtrs[11],
        shaInputPtrs[12], shaInputPtrs[13], shaInputPtrs[14], shaInputPtrs[15],
        shaOutputPtrs[0], shaOutputPtrs[1], shaOutputPtrs[2], shaOutputPtrs[3],
        shaOutputPtrs[4], shaOutputPtrs[5], shaOutputPtrs[6], shaOutputPtrs[7],
        shaOutputPtrs[8], shaOutputPtrs[9], shaOutputPtrs[10],
        shaOutputPtrs[11], shaOutputPtrs[12], shaOutputPtrs[13],
        shaOutputPtrs[14], shaOutputPtrs[15]);

    // Prepare RIPEMD input blocks
    const uint8_t* ripemdInputPtrs[HASH_BATCH_SIZE];
    uint8_t* ripemdOutputPtrs[HASH_BATCH_SIZE];

    for (__uint128_t i = 0; i < batchCount; i++) {
      uint8_t* ptr = ripemdInputs[i].data();
      std::fill_n(ptr, 64, 0);
      std::memcpy(ptr, shaOutputs[i].data(), 32);
      ptr[32] = 0x80;  // padding

      // Length in bits (32 bytes = 256 bits)
      uint32_t bitLen = 32 * 8;
      ptr[60] = (uint8_t)((bitLen >> 24) & 0xFF);
      ptr[61] = (uint8_t)((bitLen >> 16) & 0xFF);
      ptr[62] = (uint8_t)((bitLen >> 8) & 0xFF);
      ptr[63] = (uint8_t)(bitLen & 0xFF);

      ripemdInputPtrs[i] = ptr;
      ripemdOutputPtrs[i] = ripemdOutputs[i].data();
    }

    // Pad with identical data for full batch processing
    if (batchCount < HASH_BATCH_SIZE) {
      for (__uint128_t i = batchCount; i < HASH_BATCH_SIZE; i++) {
        ripemdInputPtrs[i] = ripemdInputs[0].data();
        ripemdOutputPtrs[i] = ripemdOutputs[i].data();
      }
    }

    // Process RIPEMD-160 using AVX-512 (16 blocks at once)
    ripemd160avx512::ripemd160avx512_32(
        (unsigned char*)ripemdInputPtrs[0], (unsigned char*)ripemdInputPtrs[1],
        (unsigned char*)ripemdInputPtrs[2], (unsigned char*)ripemdInputPtrs[3],
        (unsigned char*)ripemdInputPtrs[4], (unsigned char*)ripemdInputPtrs[5],
        (unsigned char*)ripemdInputPtrs[6], (unsigned char*)ripemdInputPtrs[7],
        (unsigned char*)ripemdInputPtrs[8], (unsigned char*)ripemdInputPtrs[9],
        (unsigned char*)ripemdInputPtrs[10],
        (unsigned char*)ripemdInputPtrs[11],
        (unsigned char*)ripemdInputPtrs[12],
        (unsigned char*)ripemdInputPtrs[13],
        (unsigned char*)ripemdInputPtrs[14],
        (unsigned char*)ripemdInputPtrs[15], ripemdOutputPtrs[0],
        ripemdOutputPtrs[1], ripemdOutputPtrs[2], ripemdOutputPtrs[3],
        ripemdOutputPtrs[4], ripemdOutputPtrs[5], ripemdOutputPtrs[6],
        ripemdOutputPtrs[7], ripemdOutputPtrs[8], ripemdOutputPtrs[9],
        ripemdOutputPtrs[10], ripemdOutputPtrs[11], ripemdOutputPtrs[12],
        ripemdOutputPtrs[13], ripemdOutputPtrs[14], ripemdOutputPtrs[15]);

    // Copy results
    for (__uint128_t i = 0; i < batchCount; i++) {
      std::memcpy(hashResults[batch * HASH_BATCH_SIZE + i],
                  ripemdOutputs[i].data(), 20);
    }
  }
}

// Main worker function optimized for Intel Xeon Platinum 8488C
void worker(Secp256K1* secp, int bit_length, int flip_count, int threadId,
            AVX512Counter start, AVX512Counter end) {
  const int fullBatchSize = 2 * POINTS_BATCH_SIZE;
  alignas(64) uint8_t localPubKeys[HASH_BATCH_SIZE][33];
  alignas(64) uint8_t localHashResults[HASH_BATCH_SIZE][20];
  alignas(64) int pointIndices[HASH_BATCH_SIZE];

  // Load target hash for AVX-512 comparison
  __m512i target16_1 = _mm512_set1_epi32(*(uint32_t*)&TARGET_HASH160_RAW[0]);
  __m512i target16_2 = _mm512_set1_epi32(*(uint32_t*)&TARGET_HASH160_RAW[4]);
  __m512i target16_3 = _mm512_set1_epi32(*(uint32_t*)&TARGET_HASH160_RAW[8]);
  __m512i target16_4 = _mm512_set1_epi32(*(uint32_t*)&TARGET_HASH160_RAW[12]);
  __m512i target16_5 = _mm512_set1_epi32(*(uint32_t*)&TARGET_HASH160_RAW[16]);

  // Precompute generator points for faster computation
  alignas(64) Point plusPoints[POINTS_BATCH_SIZE];
  alignas(64) Point minusPoints[POINTS_BATCH_SIZE];

#pragma omp parallel for
  for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
    Int tmp;
    tmp.SetInt32(i);
    plusPoints[i] = secp->ComputePublicKey(&tmp);
    minusPoints[i] = plusPoints[i];
    minusPoints[i].y.ModNeg();
  }

  alignas(64) Int deltaX[POINTS_BATCH_SIZE];
  IntGroup modGroup(POINTS_BATCH_SIZE);
  alignas(64) Int pointBatchX[fullBatchSize];
  alignas(64) Int pointBatchY[fullBatchSize];

  CombinationGenerator gen(bit_length, flip_count);
  gen.unrank(start.load());

  AVX512Counter count;
  count.store(start.load());

  uint64_t actual_work_done = 0;
  auto last_report = chrono::high_resolution_clock::now();

  while (!stop_event.load() && count < end) {
    Int currentKey;
    currentKey.Set(&BASE_KEY);

    const vector<int>& flips = gen.get();

    // Apply flips using AVX-512 instructions for better performance
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

// Process points in larger batches for better AVX-512 utilization
#pragma omp parallel for
    for (int i = 0; i < POINTS_BATCH_SIZE; i += 4) {
      deltaX[i].ModSub(&plusPoints[i].x, &startPointX);
      deltaX[i + 1].ModSub(&plusPoints[i + 1].x, &startPointX);
      deltaX[i + 2].ModSub(&plusPoints[i + 2].x, &startPointX);
      deltaX[i + 3].ModSub(&plusPoints[i + 3].x, &startPointX);
    }

    modGroup.Set(deltaX);
    modGroup.ModInv();

// Calculate points using AVX-512 optimized operations
#pragma omp parallel for
    for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
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

#pragma omp parallel for
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

    // Process points in batches of 16 for AVX-512 hash computation
    for (int batchStart = 0; batchStart < fullBatchSize;
         batchStart += HASH_BATCH_SIZE) {
      int batchEnd = min(batchStart + HASH_BATCH_SIZE, fullBatchSize);
      int localBatchCount = batchEnd - batchStart;

      for (int i = 0; i < localBatchCount; i++) {
        int pointIndex = batchStart + i;

        // Create public key in compressed format
        localPubKeys[i][0] = pointBatchY[pointIndex].IsEven() ? 0x02 : 0x03;
        for (int j = 0; j < 32; j++) {
          localPubKeys[i][1 + j] = pointBatchX[pointIndex].GetByte(31 - j);
        }
        pointIndices[i] = pointIndex;
      }

      // Compute HASH160 for the batch
      computeHash160BatchBin(localBatchCount, localPubKeys, localHashResults);

      actual_work_done += localBatchCount;
      localComparedCount += localBatchCount;

      // AVX-512 optimized comparison of hash results
      for (int j = 0; j < localBatchCount; j++) {
        // Use AVX-512 instructions for faster comparison
        __m512i hash_1 = _mm512_set1_epi32(*(uint32_t*)&localHashResults[j][0]);
        __m512i hash_2 = _mm512_set1_epi32(*(uint32_t*)&localHashResults[j][4]);
        __m512i hash_3 = _mm512_set1_epi32(*(uint32_t*)&localHashResults[j][8]);
        __m512i hash_4 =
            _mm512_set1_epi32(*(uint32_t*)&localHashResults[j][12]);
        __m512i hash_5 =
            _mm512_set1_epi32(*(uint32_t*)&localHashResults[j][16]);

        __mmask16 cmp1 = _mm512_cmpeq_epi32_mask(hash_1, target16_1);
        __mmask16 cmp2 = _mm512_cmpeq_epi32_mask(hash_2, target16_2);
        __mmask16 cmp3 = _mm512_cmpeq_epi32_mask(hash_3, target16_3);
        __mmask16 cmp4 = _mm512_cmpeq_epi32_mask(hash_4, target16_4);
        __mmask16 cmp5 = _mm512_cmpeq_epi32_mask(hash_5, target16_5);

        if (cmp1 && cmp2 && cmp3 && cmp4 && cmp5) {
          // Verify full match (just to be safe)
          bool fullMatch = true;
          for (int k = 0; k < 20; k++) {
            if (localHashResults[j][k] != TARGET_HASH160_RAW[k]) {
              fullMatch = false;
              break;
            }
          }

          if (fullMatch) {
            auto tEndTime = chrono::high_resolution_clock::now();
            globalElapsedTime =
                chrono::duration<double>(tEndTime - tStart).count();

            {
              lock_guard<mutex> lock(progress_mutex);
              globalComparedCount += actual_work_done;
              mkeysPerSec =
                  (double)globalComparedCount / globalElapsedTime / 1e6;
            }

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
                make_tuple(hexKey, total_checked_avx.load(), flip_count));
            stop_event.store(true);
            return;
          }
        }
      }
    }

    total_checked_avx.add(HASH_BATCH_SIZE);

    __uint128_t current_total = total_checked_avx.load();
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
      cout << "\rProgress: " << fixed << setprecision(6) << progress << "% | ";
      cout << "Speed: " << fixed << setprecision(2) << mkeysPerSec
           << " Mkeys/s | ";
      cout << "Elapsed: " << formatElapsedTime(globalElapsedTime) << "    "
           << flush;

      if (current_total >= total_combinations) {
        stop_event.store(true);
        break;
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

  if (!stop_event.load() && total_checked_avx.load() >= total_combinations) {
    stop_event.store(true);
  }
}

int main(int argc, char* argv[]) {
  signal(SIGINT, signalHandler);

  // Default to all cores available on the Xeon Platinum 8488C
  WORKERS = omp_get_max_threads();

  // Parse command line arguments
  int opt;
  while ((opt = getopt(argc, argv, "p:t:f:")) != -1) {
    switch (opt) {
      case 'p':
        PUZZLE_NUM = atoi(optarg);
        break;
      case 't':
        WORKERS = atoi(optarg);
        break;
      case 'f':
        FLIP_COUNT = atoi(optarg);
        break;
      default:
        fprintf(stderr,
                "Usage: %s [-p puzzle_num] [-t threads] [-f flip_count]\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }
  }

  tStart = chrono::high_resolution_clock::now();

  // Initialize secp256k1
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

  // Parse target hash160
  for (__uint128_t i = 0; i < 20; i++) {
    TARGET_HASH160_RAW[i] = stoul(TARGET_HASH160.substr(i * 2, 2), nullptr, 16);
  }

  // Set base key
  BASE_KEY.SetBase10(const_cast<char*>(PRIVATE_KEY_DECIMAL.c_str()));

  // Verify key setup
  Int testKey;
  testKey.SetBase10(const_cast<char*>(PRIVATE_KEY_DECIMAL.c_str()));
  if (!testKey.IsEqual(&BASE_KEY)) {
    cerr << "Base key initialization failed!\n";
    return 1;
  }

  if (BASE_KEY.GetBitLength() > PUZZLE_NUM) {
    cerr << "Base key exceeds puzzle bit length!\n";
    return 1;
  }

  // Calculate total combinations
  total_combinations =
      CombinationGenerator::combinations_count(PUZZLE_NUM, FLIP_COUNT);

  string paddedKey = BASE_KEY.GetBase16();
  size_t firstNonZero = paddedKey.find_first_not_of('0');

  if (string::npos == firstNonZero) {
    paddedKey = "0";
  } else {
    paddedKey = paddedKey.substr(firstNonZero);
  }

  paddedKey = "0x" + paddedKey;

  cout << "=======================================================\n";
  cout << "== Optimized Mutagen Solver for Xeon Platinum 8488C ==\n";
  cout << "=======================================================\n";
  cout << "Solving puzzle: " << PUZZLE_NUM << " (" << PUZZLE_NUM << "-bit)\n";
  cout << "Target HASH160: " << TARGET_HASH160.substr(0, 10) << "..."
       << TARGET_HASH160.substr(TARGET_HASH160.length() - 10) << "\n";
  cout << "Base Key: " << paddedKey << "\n";
  cout << "Flip count: " << FLIP_COUNT << " ";
  if (FLIP_COUNT != DEFAULT_FLIP_COUNT) {
    cout << "(override, default was " << DEFAULT_FLIP_COUNT << ")";
  }
  cout << "\n";
  cout << "Total combinations: " << to_string_128(total_combinations) << "\n";
  cout << "Using: " << WORKERS << " threads\n";
  cout << "AVX-512 optimized: Yes\n";
  cout << "\n";
  cout << "Starting search...\n";

  g_threadPrivateKeys.resize(WORKERS, "0");
  vector<thread> threads;

  AVX512Counter total_combinations_avx;
  total_combinations_avx.store(total_combinations);

  AVX512Counter comb_per_thread =
      AVX512Counter::div(total_combinations_avx, WORKERS);
  uint64_t remainder = AVX512Counter::mod(total_combinations_avx, WORKERS);

  // Start worker threads with optimized distribution
  for (int i = 0; i < WORKERS; i++) {
    AVX512Counter start, end;

    AVX512Counter base = AVX512Counter::mul(i, comb_per_thread.load());
    uint64_t extra = min(static_cast<uint64_t>(i), remainder);
    start.store(base.load() + extra);

    end.store(start.load() + comb_per_thread.load() + (i < remainder ? 1 : 0));
    threads.emplace_back(worker, &secp, PUZZLE_NUM, FLIP_COUNT, i, start, end);
  }

  // Wait for all threads to complete
  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  // Process results
  if (!results.empty()) {
    auto [hex_key, checked, flips] = results.front();
    globalElapsedTime =
        chrono::duration<double>(chrono::high_resolution_clock::now() - tStart)
            .count();
    mkeysPerSec = (double)globalComparedCount / globalElapsedTime / 1e6;

    string compactHex = hex_key;
    size_t firstNonZeroCompact = compactHex.find_first_not_of('0');

    if (string::npos == firstNonZeroCompact) {
      compactHex = "0";
    } else {
      compactHex = compactHex.substr(firstNonZeroCompact);
    }
    compactHex = "0x" + compactHex;

    cout << "\n\n=======================================================\n";
    cout << "=================== SOLUTION FOUND ===================\n";
    cout << "=======================================================\n";
    cout << "Private key: " << compactHex << "\n";
    cout << "Checked " << to_string_128(checked) << " combinations\n";
    cout << "Bit flips: " << flips << endl;
    cout << "Time: " << fixed << setprecision(2) << globalElapsedTime
         << " seconds (" << formatElapsedTime(globalElapsedTime) << ")\n";
    cout << "Speed: " << fixed << setprecision(2) << mkeysPerSec
         << " Mkeys/s\n";

    ofstream out("puzzle_" + to_string(PUZZLE_NUM) + "_solution.txt");
    if (out) {
      out << hex_key;
      out.close();
      cout << "Solution saved to puzzle_" << PUZZLE_NUM << "_solution.txt\n";
    } else {
      cerr << "Failed to save solution to file!\n";
    }
  } else {
    __uint128_t final_count = total_checked_avx.load();
    globalElapsedTime =
        chrono::duration<double>(chrono::high_resolution_clock::now() - tStart)
            .count();

    if (globalElapsedTime > 1e-6) {
      mkeysPerSec = (double)globalComparedCount / globalElapsedTime / 1e6;
    } else {
      mkeysPerSec = 0.0;
    }
    cout << "\n\nNo solution found. Checked " << to_string_128(final_count)
         << " combinations\n";
    cout << "Time: " << fixed << setprecision(2) << globalElapsedTime
         << " seconds (" << formatElapsedTime(globalElapsedTime) << ")\n";
    cout << "Speed: " << fixed << setprecision(2) << mkeysPerSec
         << " Mkeys/s\n";
  }

  return 0;
}
