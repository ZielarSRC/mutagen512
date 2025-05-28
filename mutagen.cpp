#include <iostream>
#include <array>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <queue>
#include <mutex>
#include <cstring>
#include <unordered_map>
#include <cmath>
#include <immintrin.h>
#include <omp.h>
#include <csignal>
#include <random>
#include <algorithm>
#include <getopt.h>

#ifdef _WIN32
    #include <windows.h>
#endif

#include "sha256_avx512.h"
#include "ripemd160_avx512.h"
#include "SECP256K1.h"
#include "Point.h"
#include "Int.h"
#include "IntGroup.h"

using namespace std;

void initConsole() {
#ifdef _WIN32
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    GetConsoleMode(hConsole, &mode);
    SetConsoleMode(hConsole, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
}

void clearTerminal() {
#ifdef _WIN32
    HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD coord = {0, 0};
    DWORD count;
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(hStdOut, &csbi);
    FillConsoleOutputCharacter(hStdOut, ' ', csbi.dwSize.X * csbi.dwSize.Y, coord, &count);
    SetConsoleCursorPosition(hStdOut, coord);
#else
    std::cout << "\033[2J\033[H";
#endif
    std::cout.flush();
}

void moveCursorTo(int x, int y) {
#ifdef _WIN32
    HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD coord = {(SHORT)x, (SHORT)y};
    SetConsoleCursorPosition(hStdOut, coord);
#else
    std::cout << "\033[" << y << ";" << x << "H";
#endif
    std::cout.flush();
}

int PUZZLE_NUM = 71;
int WORKERS = omp_get_num_procs();
int FLIP_COUNT = -1; 
const __uint128_t REPORT_INTERVAL = 20000000; // Increased for AVX-512 performance
static constexpr int POINTS_BATCH_SIZE = 512; // Doubled for AVX-512
static constexpr int HASH_BATCH_SIZE = 16; // Doubled for AVX-512

const unordered_map<int, tuple<int, string, string>> PUZZLE_DATA = {
    {20, {8, "b907c3a2a3b27789dfb509b730dd47703c272868",  "357535"}}, 
    {21, {9, "29a78213caa9eea824acf08022ab9dfc83414f56",  "863317"}},
    {22, {11, "7ff45303774ef7a52fffd8011981034b258cb86b", "1811764"}}, 
    {23, {12, "d0a79df189fe1ad5c306cc70497b358415da579e", "3007503"}},
    {24, {9, "0959e80121f36aea13b3bad361c15dac26189e2f",  "5598802"}},
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
    {37, {23, "28c30fb9118ed1da72e7c4f89c0164756e8a021d",   "42387769980"}},
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
    {58, {28, "8c2a6071f89c90c4dab5ab295d7729d1b54ea60f", "138245758910846492"}},
    {59, {30, "b14ed3146f5b2c9bde1703deae9ef33af8110210", "199976667976342049"}},
    {60, {31, "cdf8e5c7503a9d22642e3ecfc87817672787b9c5", "525070384258266191"}},
    {61, {25, "68133e19b2dfb9034edf9830a200cfdf38c90cbd", "1135041350219496382"}},
    {62, {35, "e26646db84b0602f32b34b5a62ca3cae1f91b779", "1425787542618654982"}},
    {63, {34, "ef58afb697b094423ce90721fbb19a359ef7c50e", "3908372542507822062"}},
    {64, {34, "3ee4133d991f52fdf6a25c9834e0745ac74248a4", "8993229949524469768"}},
    {65, {37, "52e763a7ddc1aa4fa811578c491c1bc7fd570137", "17799667357578236628"}},
    {66, {35, "20d45a6a762535700ce9e0b216e31994335db8a5", "30568377312064202855"}},
    {67, {31, "739437bb3dd6d1983e66629c5f08c70e52769371", "46346217550346335726"}},
    {68, {42, "e0b8a2baee1b77fc703455f39d51477451fc8cfc", "132656943602386256302"}},
    {69, {34, "61eb8a50c86b0584bb727dd65bed8d2400d6d5aa", "219898266213316039825"}},
    {70, {29, "5db8cda53a6a002db10365967d7f85d19e171b10", "297274491920375905804"}},
    {71, {29, "f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8", "970436974005023690481"}}
};

vector<unsigned char> TARGET_HASH160_RAW(20);
string TARGET_HASH160;
Int BASE_KEY;
atomic<bool> stop_event(false);
mutex result_mutex;
queue<tuple<string, __uint128_t, int>> results;

union AVXCounter {
    alignas(64) __m512i vec512;
    alignas(64) __m256i vec256[2];
    alignas(64) uint64_t u64[8];
    alignas(64) __uint128_t u128[4];

    AVXCounter() : vec512(_mm512_setzero_si512()) {}

    AVXCounter(__uint128_t value) {
        store(value);
    }

    void increment() {
        // AVX-512 optimized increment
        __m512i one = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 1);
        vec512 = _mm512_add_epi64(vec512, one);
        
        // Handle carry propagation for 128-bit arithmetic
        if (u64[0] == 0) {
            __m512i carry = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 1, 0);
            vec512 = _mm512_add_epi64(vec512, carry);
        }
    }

    void add(__uint128_t value) {
        __m512i add_val = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 
                                          static_cast<uint64_t>(value >> 64), 
                                          static_cast<uint64_t>(value));
        vec512 = _mm512_add_epi64(vec512, add_val);

        // Handle carry for 128-bit arithmetic
        if (u64[0] < (value & 0xFFFFFFFFFFFFFFFFULL)) {
            __m512i carry = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 1, 0);
            vec512 = _mm512_add_epi64(vec512, carry);
        }
    }

    __uint128_t load() const {
        return (static_cast<__uint128_t>(u64[1]) << 64) | u64[0];
    }

    void store(__uint128_t value) {
        vec512 = _mm512_setzero_si512();
        u64[0] = static_cast<uint64_t>(value);
        u64[1] = static_cast<uint64_t>(value >> 64);
    }

    bool operator<(const AVXCounter& other) const {
        if (u64[1] != other.u64[1])
            return u64[1] < other.u64[1];
        return u64[0] < other.u64[0];
    }

    bool operator>=(const AVXCounter& other) const {
        if (u64[1] != other.u64[1])
            return u64[1] > other.u64[1];
        return u64[0] >= other.u64[0];
    }

    static AVXCounter div(const AVXCounter& num, uint64_t denom) {
        __uint128_t n = num.load();
        __uint128_t q = n / denom;
        return AVXCounter(q);
    }

    static uint64_t mod(const AVXCounter& num, uint64_t denom) {
        __uint128_t n = num.load();
        return n % denom;
    }

    static AVXCounter mul(uint64_t a, uint64_t b) {
        __uint128_t result = static_cast<__uint128_t>(a) * b;
        return AVXCounter(result);
    }
};

static AVXCounter total_checked_avx;
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
    oss << std::setw(2) << std::setfill('0') << hrs << ":"
        << std::setw(2) << std::setfill('0') << mins << ":"
        << std::setw(2) << std::setfill('0') << secs;
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
        for(int i = 2; i <= k; ++i) {
            result *= (n - i + 1);
            result /= i;  
        }
        return result;
    }

    static __m512i combinations_count_avx512(int n, int k) {
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
        for (int j = i + 1; j < k; ++j)
            current[j] = current[j-1] + 1;
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

inline void prepareShaBlock(const uint8_t* dataSrc, __uint128_t dataLen, uint8_t* outBlock) {
    // AVX-512 optimized block preparation
    __m512i zero = _mm512_setzero_si512();
    _mm512_storeu_si512((__m512i*)outBlock, zero);
    
    std::memcpy(outBlock, dataSrc, dataLen);
    outBlock[dataLen] = 0x80;
    const uint32_t bitLen = (uint32_t)(dataLen * 8);
    outBlock[60] = (uint8_t)((bitLen >> 24) & 0xFF);
    outBlock[61] = (uint8_t)((bitLen >> 16) & 0xFF);
    outBlock[62] = (uint8_t)((bitLen >>  8) & 0xFF);
    outBlock[63] = (uint8_t)( bitLen        & 0xFF);
}

inline void prepareRipemdBlock(const uint8_t* dataSrc, uint8_t* outBlock) {
    // AVX-512 optimized block preparation
    __m512i zero = _mm512_setzero_si512();
    _mm512_storeu_si512((__m512i*)outBlock, zero);
    
    std::memcpy(outBlock, dataSrc, 32);
    outBlock[32] = 0x80;
    const uint32_t bitLen = 256;
    outBlock[60] = (uint8_t)((bitLen >> 24) & 0xFF);
    outBlock[61] = (uint8_t)((bitLen >> 16) & 0xFF);
    outBlock[62] = (uint8_t)((bitLen >>  8) & 0xFF);
    outBlock[63] = (uint8_t)( bitLen        & 0xFF);
}

static void computeHash160BatchBinSingle(int numKeys,
                                       uint8_t pubKeys[][33],
                                       uint8_t hashResults[][20])
{
    alignas(64) std::array<std::array<uint8_t, 64>, HASH_BATCH_SIZE> shaInputs;
    alignas(64) std::array<std::array<uint8_t, 32>, HASH_BATCH_SIZE> shaOutputs;
    alignas(64) std::array<std::array<uint8_t, 64>, HASH_BATCH_SIZE> ripemdInputs;
    alignas(64) std::array<std::array<uint8_t, 20>, HASH_BATCH_SIZE> ripemdOutputs;
    
    const __uint128_t totalBatches = (numKeys + (HASH_BATCH_SIZE - 1)) / HASH_BATCH_SIZE;
    
    for (__uint128_t batch = 0; batch < totalBatches; batch++) {
        const __uint128_t batchCount = std::min<__uint128_t>(HASH_BATCH_SIZE, numKeys - batch * HASH_BATCH_SIZE);

        // Prefetch input data for better cache performance
        for (__uint128_t i = 0; i < batchCount; i++) {
            _mm_prefetch((char*)pubKeys[batch * HASH_BATCH_SIZE + i], _MM_HINT_T0);
        }

        for (__uint128_t i = 0; i < batchCount; i++) {
            prepareShaBlock(pubKeys[batch * HASH_BATCH_SIZE + i], 33, shaInputs[i].data());
        }

        if (batchCount < HASH_BATCH_SIZE) {
            static std::array<uint8_t, 64> shaPadding = {};
            prepareShaBlock(pubKeys[0], 33, shaPadding.data());
            for (__uint128_t i = batchCount; i < HASH_BATCH_SIZE; i++) {
                std::memcpy(shaInputs[i].data(), shaPadding.data(), 64);
            }
        }

        const uint8_t* inPtr[HASH_BATCH_SIZE];
        uint8_t* outPtr[HASH_BATCH_SIZE];
        for (int i = 0; i < HASH_BATCH_SIZE; i++) {
            inPtr[i]  = shaInputs[i].data();
            outPtr[i] = shaOutputs[i].data();
        }

        // Use AVX-512 optimized SHA-256
        sha256avx512_16B(inPtr[0], inPtr[1], inPtr[2], inPtr[3],
                        inPtr[4], inPtr[5], inPtr[6], inPtr[7],
                        inPtr[8], inPtr[9], inPtr[10], inPtr[11],
                        inPtr[12], inPtr[13], inPtr[14], inPtr[15],
                        outPtr[0], outPtr[1], outPtr[2], outPtr[3],
                        outPtr[4], outPtr[5], outPtr[6], outPtr[7],
                        outPtr[8], outPtr[9], outPtr[10], outPtr[11],
                        outPtr[12], outPtr[13], outPtr[14], outPtr[15]);

        for (__uint128_t i = 0; i < batchCount; i++) {
            prepareRipemdBlock(shaOutputs[i].data(), ripemdInputs[i].data());
        }

        if (batchCount < HASH_BATCH_SIZE) {
            static std::array<uint8_t, 64> ripemdPadding = {};
            prepareRipemdBlock(shaOutputs[0].data(), ripemdPadding.data());
            for (__uint128_t i = batchCount; i < HASH_BATCH_SIZE; i++) {
                std::memcpy(ripemdInputs[i].data(), ripemdPadding.data(), 64);
            }
        }

        for (int i = 0; i < HASH_BATCH_SIZE; i++) {
            inPtr[i]  = ripemdInputs[i].data();
            outPtr[i] = ripemdOutputs[i].data();
        }

        // Use AVX-512 optimized RIPEMD160
        ripemd160avx512::ripemd160avx512_32(
            (unsigned char*)inPtr[0], (unsigned char*)inPtr[1],
            (unsigned char*)inPtr[2], (unsigned char*)inPtr[3],
            (unsigned char*)inPtr[4], (unsigned char*)inPtr[5],
            (unsigned char*)inPtr[6], (unsigned char*)inPtr[7],
            (unsigned char*)inPtr[8], (unsigned char*)inPtr[9],
            (unsigned char*)inPtr[10], (unsigned char*)inPtr[11],
            (unsigned char*)inPtr[12], (unsigned char*)inPtr[13],
            (unsigned char*)inPtr[14], (unsigned char*)inPtr[15],
            outPtr[0], outPtr[1], outPtr[2], outPtr[3],
            outPtr[4], outPtr[5], outPtr[6], outPtr[7],
            outPtr[8], outPtr[9], outPtr[10], outPtr[11],
            outPtr[12], outPtr[13], outPtr[14], outPtr[15]
        );

        for (__uint128_t i = 0; i < batchCount; i++) {
            std::memcpy(hashResults[batch * HASH_BATCH_SIZE + i], ripemdOutputs[i].data(), 20);
        }
    }
}

void worker(Secp256K1* secp, int bit_length, int flip_count, int threadId, AVXCounter start, AVXCounter end) {
    const int fullBatchSize = 2 * POINTS_BATCH_SIZE;
    alignas(64) uint8_t localPubKeys[HASH_BATCH_SIZE][33];
    alignas(64) uint8_t localHashResults[HASH_BATCH_SIZE][20];
    alignas(64) int pointIndices[HASH_BATCH_SIZE];

    // AVX-512 optimized target comparison
    __m512i target32 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(TARGET_HASH160_RAW.data()));

    alignas(64) Point plusPoints[POINTS_BATCH_SIZE];
    alignas(64) Point minusPoints[POINTS_BATCH_SIZE];

    // MOVED: Initialize points FIRST, then prefetch
    for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
        Int tmp;
        tmp.SetInt32(i);
        plusPoints[i] = secp->ComputePublicKey(&tmp);
        minusPoints[i] = plusPoints[i];
        minusPoints[i].y.ModNeg();
        
        // Apply AVX-512 reduction for better performance
        plusPoints[i].x.ModReduceK1AVX512();
        plusPoints[i].y.ModReduceK1AVX512();
        minusPoints[i].x.ModReduceK1AVX512();
        minusPoints[i].y.ModReduceK1AVX512();
    }

    // NOW prefetch after initialization
    _mm_prefetch((char*)plusPoints, _MM_HINT_T0);
    _mm_prefetch((char*)minusPoints, _MM_HINT_T0);

    alignas(64) Int deltaX[POINTS_BATCH_SIZE];
    IntGroup modGroup(POINTS_BATCH_SIZE);
    alignas(64) Int pointBatchX[fullBatchSize];
    alignas(64) Int pointBatchY[fullBatchSize];

    CombinationGenerator gen(bit_length, flip_count);
    gen.unrank(start.load());

    AVXCounter count;
    count.store(start.load());

    uint64_t actual_work_done = 0;
    auto last_report = chrono::high_resolution_clock::now();

    while (!stop_event.load() && count < end)
    {
        Int currentKey;
        currentKey.Set(&BASE_KEY);

        const vector<int> &flips = gen.get();

        // Apply flips with AVX-512 optimizations
        for (int pos : flips) {
            Int mask;
            mask.SetInt32(1);
            mask.ShiftL(pos);
            currentKey.Xor(&mask);
        }

        string keyStr = currentKey.GetBase16();
        keyStr = string(64 - keyStr.length(), '0') + keyStr;

        #pragma omp critical
        {
            g_threadPrivateKeys[threadId] = keyStr;
        }

        Point startPoint = secp->ComputePublicKey(&currentKey);
        
        // Apply fast reduction for secp256k1 with AVX-512
        startPoint.x.ModReduceK1AVX512();
        startPoint.y.ModReduceK1AVX512();
        
        Int startPointX, startPointY, startPointXNeg;
        startPointX.Set(&startPoint.x);
        startPointY.Set(&startPoint.y);
        startPointXNeg.Set(&startPointX);
        startPointXNeg.ModNeg();

        // AVX-512 optimized point arithmetic with unrolling
        #pragma unroll(4)
        for (int i = 0; i < POINTS_BATCH_SIZE; i += 4) {
            // Prefetch next iteration data
            if (i + 4 < POINTS_BATCH_SIZE) {
                _mm_prefetch((char*)&plusPoints[i + 4], _MM_HINT_T0);
            }
            
            deltaX[i].ModSub(&plusPoints[i].x, &startPointX);
            deltaX[i+1].ModSub(&plusPoints[i+1].x, &startPointX);
            deltaX[i+2].ModSub(&plusPoints[i+2].x, &startPointX);
            deltaX[i+3].ModSub(&plusPoints[i+3].x, &startPointX);
            
            // Apply fast reduction for better performance
            deltaX[i].ModReduceK1AVX512();
            deltaX[i+1].ModReduceK1AVX512();
            deltaX[i+2].ModReduceK1AVX512();
            deltaX[i+3].ModReduceK1AVX512();
        }
        
        modGroup.Set(deltaX);
        modGroup.ModInv();

        // AVX-512 optimized point computations
        for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
            // Prefetch next iteration data
            if (i + 1 < POINTS_BATCH_SIZE) {
                _mm_prefetch((char*)&plusPoints[i + 1], _MM_HINT_T0);
                _mm_prefetch((char*)&deltaX[i + 1], _MM_HINT_T0);
            }
            
            Int deltaY;
            deltaY.ModSub(&plusPoints[i].y, &startPointY);

            Int slope;
            slope.ModMulK1(&deltaY, &deltaX[i]);
            slope.ModReduceK1AVX512();

            Int slopeSq;
            slopeSq.ModSquareK1(&slope);
            slopeSq.ModReduceK1AVX512();

            pointBatchX[i].Set(&startPointXNeg);
            pointBatchX[i].ModAdd(&slopeSq);
            pointBatchX[i].ModSub(&plusPoints[i].x);
            pointBatchX[i].ModReduceK1AVX512();

            Int diffX;
            diffX.Set(&startPointX);
            diffX.ModSub(&pointBatchX[i]);
            diffX.ModMulK1(&slope);
            diffX.ModReduceK1AVX512();

            pointBatchY[i].Set(&startPointY);
            pointBatchY[i].ModNeg();
            pointBatchY[i].ModAdd(&diffX);
            pointBatchY[i].ModReduceK1AVX512();
        }

        for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
            // Prefetch next iteration data
            if (i + 1 < POINTS_BATCH_SIZE) {
                _mm_prefetch((char*)&minusPoints[i + 1], _MM_HINT_T0);
            }
            
            Int deltaY;
            deltaY.ModSub(&minusPoints[i].y, &startPointY);

            Int slope;
            slope.ModMulK1(&deltaY, &deltaX[i]);
            slope.ModReduceK1AVX512();

            Int slopeSq;
            slopeSq.ModSquareK1(&slope);
            slopeSq.ModReduceK1AVX512();

            pointBatchX[POINTS_BATCH_SIZE + i].Set(&startPointXNeg);
            pointBatchX[POINTS_BATCH_SIZE + i].ModAdd(&slopeSq);
            pointBatchX[POINTS_BATCH_SIZE + i].ModSub(&minusPoints[i].x);
            pointBatchX[POINTS_BATCH_SIZE + i].ModReduceK1AVX512();

            Int diffX;
            diffX.Set(&startPointX);
            diffX.ModSub(&pointBatchX[POINTS_BATCH_SIZE + i]);
            diffX.ModMulK1(&slope);
            diffX.ModReduceK1AVX512();

            pointBatchY[POINTS_BATCH_SIZE + i].Set(&startPointY);
            pointBatchY[POINTS_BATCH_SIZE + i].ModNeg();
            pointBatchY[POINTS_BATCH_SIZE + i].ModAdd(&diffX);
            pointBatchY[POINTS_BATCH_SIZE + i].ModReduceK1AVX512();
        }

        int localBatchCount = 0;
        for (int i = 0; i < fullBatchSize && localBatchCount < HASH_BATCH_SIZE; i++) {
            Point tempPoint;
            tempPoint.x.Set(&pointBatchX[i]);
            tempPoint.y.Set(&pointBatchY[i]);

            localPubKeys[localBatchCount][0] = tempPoint.y.IsEven() ? 0x02 : 0x03;
            for (int j = 0; j < 32; j++) {
                localPubKeys[localBatchCount][1 + j] = pointBatchX[i].GetByte(31 - j);
            }
            pointIndices[localBatchCount] = i;
            localBatchCount++;

            if (localBatchCount == HASH_BATCH_SIZE) {
                computeHash160BatchBinSingle(localBatchCount, localPubKeys, localHashResults);

                actual_work_done += HASH_BATCH_SIZE;
                localComparedCount += HASH_BATCH_SIZE;

                // AVX-512 optimized hash comparison
                for (int j = 0; j < HASH_BATCH_SIZE; j += 4) {
                    // Process 4 hashes at once with AVX-512
                    for (int k = 0; k < 4 && (j + k) < HASH_BATCH_SIZE; k++) {
                        int idx = j + k;
                        
                        // AVX-512 comparison for first 16 bytes
                        __m128i cand = _mm_loadu_si128(reinterpret_cast<const __m128i*>(localHashResults[idx]));
                        __m128i target = _mm_loadu_si128(reinterpret_cast<const __m128i*>(TARGET_HASH160_RAW.data()));
                        __m128i cmp = _mm_cmpeq_epi8(cand, target);
                        int mask = _mm_movemask_epi8(cmp);

                        if (mask == 0xFFFF) {  // First 16 bytes match
                            // Check remaining 4 bytes
                            bool fullMatch = true;
                            for (int l = 16; l < 20; l++) {
                                if (localHashResults[idx][l] != TARGET_HASH160_RAW[l]) {
                                    fullMatch = false;
                                    break;
                                }
                            }

                            if (fullMatch) {
                                auto tEndTime = chrono::high_resolution_clock::now();
                                globalElapsedTime = chrono::duration<double>(tEndTime - tStart).count();

                                {
                                    lock_guard<mutex> lock(progress_mutex);
                                    globalComparedCount += actual_work_done;
                                    mkeysPerSec = (double)globalComparedCount / globalElapsedTime / 1e6;
                                }

                                Int foundKey;
                                foundKey.Set(&currentKey);
                                int pointIdx = pointIndices[idx];
                                if (pointIdx < POINTS_BATCH_SIZE) {
                                    Int offset;
                                    offset.SetInt32(pointIdx);
                                    foundKey.Add(&offset);
                                } else {
                                    Int offset;
                                    offset.SetInt32(pointIdx - POINTS_BATCH_SIZE);
                                    foundKey.Sub(&offset);
                                }

                                string hexKey = foundKey.GetBase16();
                                hexKey = string(64 - hexKey.length(), '0') + hexKey;

                                lock_guard<mutex> lock(result_mutex);
                                results.push(make_tuple(hexKey, total_checked_avx.load(), flip_count));
                                stop_event.store(true);
                                return;
                            }
                        }
                    }
                }

                total_checked_avx.increment();
                localBatchCount = 0;

                __uint128_t current_total = total_checked_avx.load();
                if (current_total % REPORT_INTERVAL == 0 || count.load() == end.load() - 1) {
                    auto now = chrono::high_resolution_clock::now();
                    globalElapsedTime = chrono::duration<double>(now - tStart).count();

                    globalComparedCount += localComparedCount;
                    localComparedCount = 0;
                    mkeysPerSec = (double)globalComparedCount / globalElapsedTime / 1e6;
                    double progress = min(100.0, (double)current_total / total_combinations * 100.0);

                    lock_guard<mutex> lock(progress_mutex);
                    moveCursorTo(0, 10);
                    cout << "Progress: " << fixed << setprecision(6) << progress << "%\n";
                    cout << "Processed: " << to_string_128(current_total) << "\n";
                    cout << "Speed: " << fixed << setprecision(2) << mkeysPerSec << " Mkeys/s\n";
                    cout << "Elapsed Time: " << formatElapsedTime(globalElapsedTime) << "\n";
                    cout.flush();

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

    if (!stop_event.load() && total_checked_avx.load() >= total_combinations) {
        stop_event.store(true);
    }
}

void printUsage(const char* programName) {
    cout << "Usage: " << programName << " [options]\n";
    cout << "Options:\n";
    cout << "  -p, --puzzle NUM    Puzzle number to solve (default: 71)\n"; 
    cout << "  -t, --threads NUM   Number of CPU cores to use (default: all)\n";
    cout << "  -f, --flips NUM     Override default flip count for puzzle\n";
    cout << "  -h, --help          Show this help message\n";
    cout << "\nExample:\n";
    cout << "  " << programName << " -p 71 -t 12\n"; 
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signalHandler);

    // Check for AVX-512 support
    if (!__builtin_cpu_supports("avx512f")) {
        cerr << "Error: This program requires AVX-512 support\n";
        return 1;
    }

    int opt;
    int option_index = 0;
    static struct option long_options[] = {
        {"puzzle", required_argument, 0, 'p'},
        {"threads", required_argument, 0, 't'},
        {"flips", required_argument, 0, 'f'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "p:t:f:h", long_options, &option_index)) != -1) {
        if (opt == -1) break;
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

    auto [DEFAULT_FLIP_COUNT, TARGET_HASH160_HEX, PRIVATE_KEY_DECIMAL] = puzzle_it->second;

    if (FLIP_COUNT == -1) {
        FLIP_COUNT = DEFAULT_FLIP_COUNT;
    }

    TARGET_HASH160 = TARGET_HASH160_HEX;

    for (__uint128_t i = 0; i < 20; i++) {
        TARGET_HASH160_RAW[i] = stoul(TARGET_HASH160.substr(i * 2, 2), nullptr, 16);
    }

    BASE_KEY.SetBase10(const_cast<char*>(PRIVATE_KEY_DECIMAL.c_str()));

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

    total_combinations = CombinationGenerator::combinations_count(PUZZLE_NUM, FLIP_COUNT);

    string paddedKey = BASE_KEY.GetBase16();
    size_t firstNonZero = paddedKey.find_first_not_of('0');

    if (string::npos == firstNonZero) {
         paddedKey = "0";
    } else {
        paddedKey = paddedKey.substr(firstNonZero);
    }

    paddedKey = "0x" + paddedKey;

    clearTerminal();
    cout << "=======================================\n";
    cout << "== Mutagen AVX-512 Solver by Denevron ==\n";
    cout << "=======================================\n";
    cout << "Starting puzzle: " << PUZZLE_NUM << " (" << PUZZLE_NUM << "-bit)\n";
    cout << "Target HASH160: " << TARGET_HASH160.substr(0, 10) << "..." << TARGET_HASH160.substr(TARGET_HASH160.length()-10) << "\n";
    cout << "Base Key: " << paddedKey << "\n";
    cout << "Flip count: " << FLIP_COUNT << " ";
    if (FLIP_COUNT != DEFAULT_FLIP_COUNT) {
        cout << "(override, default was " << DEFAULT_FLIP_COUNT << ")";
    }
    cout << "\n";

    if (PUZZLE_NUM == 71 && FLIP_COUNT == 29) { 
        cout << "*** WARNING: Flip count is an ESTIMATE for Puzzle 71 and might be incorrect! ***\n";
    }
    cout << "Total Flips: " << to_string_128(total_combinations) << "\n";
    cout << "Using: " << WORKERS << " threads with AVX-512 optimization\n";
    cout << "Batch sizes: Points=" << POINTS_BATCH_SIZE << ", Hash=" << HASH_BATCH_SIZE << "\n";
    cout << "\n";

    g_threadPrivateKeys.resize(WORKERS, "0");
    vector<thread> threads;

    AVXCounter total_combinations_avx;
    total_combinations_avx.store(total_combinations);

    AVXCounter comb_per_thread = AVXCounter::div(total_combinations_avx, WORKERS);
    uint64_t remainder = AVXCounter::mod(total_combinations_avx, WORKERS);

    for (int i = 0; i < WORKERS; i++) {
        AVXCounter start, end;

        AVXCounter base = AVXCounter::mul(i, comb_per_thread.load());
        uint64_t extra = min(static_cast<uint64_t>(i), remainder);
        start.store(base.load() + extra);

        end.store(start.load() + comb_per_thread.load() + (i < remainder ? 1 : 0));
        threads.emplace_back(worker, &secp, PUZZLE_NUM, FLIP_COUNT, i, start, end);
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    if (!results.empty()) {
        auto [hex_key, checked, flips] = results.front();
        globalElapsedTime = chrono::duration<double>(chrono::high_resolution_clock::now() - tStart).count();
        mkeysPerSec = (double)globalComparedCount / globalElapsedTime / 1e6;

        string compactHex = hex_key;
        size_t firstNonZeroCompact = compactHex.find_first_not_of('0');

        if (string::npos == firstNonZeroCompact) {
             compactHex = "0";
        } else {
            compactHex = compactHex.substr(firstNonZeroCompact);
        }
        compactHex = "0x" + compactHex;

        cout << "=======================================\n";
        cout << "========== SOLUTION FOUND =============\n";
        cout << "=======================================\n";
        cout << "Private key: " << compactHex << "\n";
        cout << "Checked " << to_string_128(checked) << " combinations\n";
        cout << "Bit flips: " << flips << endl;
        cout << "Time: " << fixed << setprecision(2) << globalElapsedTime << " seconds ("
             << formatElapsedTime(globalElapsedTime) << ")\n";
        cout << "Speed: " << fixed << setprecision(2) << mkeysPerSec << " Mkeys/s (AVX-512)\n";

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
        globalElapsedTime = chrono::duration<double>(chrono::high_resolution_clock::now() - tStart).count();

        if (globalElapsedTime > 1e-6) {
             mkeysPerSec = (double)globalComparedCount / globalElapsedTime / 1e6;
        } else {
             mkeysPerSec = 0.0;
        }
        cout << "\n\nNo solution found. Checked " << to_string_128(final_count) << " combinations\n";
        cout << "Time: " << fixed << setprecision(2) << globalElapsedTime << " seconds ("
             << formatElapsedTime(globalElapsedTime) << ")\n";
        cout << "Speed: " << fixed << setprecision(2) << mkeysPerSec << " Mkeys/s (AVX-512)\n";
    }

    return 0;
}
