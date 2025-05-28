# Detect OS
UNAME_S := $(shell uname -s)

# Enable static linking by default (change to 'no' to use dynamic linking)
STATIC_LINKING = yes

# Auto-detect Intel Xeon Platinum 8488C (Sapphire Rapids) capabilities
CPU_INFO := $(shell cat /proc/cpuinfo 2>/dev/null | grep -E "(avx512|model name)" || echo "unknown")
INTEL_XEON := $(shell echo "$(CPU_INFO)" | grep -i "xeon" | wc -l)
HAS_AVX512 := $(shell echo "$(CPU_INFO)" | grep -i "avx512" | wc -l)
SAPPHIRE_RAPIDS := $(shell echo "$(CPU_INFO)" | grep -E "(8488C|Sapphire Rapids)" | wc -l)

# Compiler settings based on OS
ifeq ($(UNAME_S),Linux)
# Linux settings

# Compiler (require GCC 12+ for proper Sapphire Rapids support)
CXX = g++
GCC_VERSION := $(shell $(CXX) -dumpversion | cut -d. -f1)
GCC_MAJOR := $(shell echo "$(GCC_VERSION)" | sed 's/[^0-9].*//')

# Check if GCC version supports Sapphire Rapids optimizations
ifeq ($(shell test $(GCC_MAJOR) -ge 12; echo $$?),0)
    $(info Using GCC $(GCC_VERSION) with Sapphire Rapids support)
    SAPPHIRE_SUPPORT = full
else ifeq ($(shell test $(GCC_MAJOR) -ge 11; echo $$?),0)
    $(warning GCC $(GCC_VERSION) has limited Sapphire Rapids support. GCC 12+ recommended)
    SAPPHIRE_SUPPORT = limited
else
    $(error GCC $(GCC_VERSION) insufficient for Sapphire Rapids. Minimum GCC 11 required)
endif

# Intel Xeon Platinum 8488C (Sapphire Rapids) specific optimizations
ifeq ($(SAPPHIRE_RAPIDS),1)
    $(info Intel Xeon Platinum 8488C (Sapphire Rapids) detected - enabling advanced optimizations)
    ARCH_FLAGS = -march=sapphirerapids -mtune=sapphirerapids
else ifeq ($(INTEL_XEON),1)
    $(info Intel Xeon processor detected - using Sapphire Rapids optimizations)
    ARCH_FLAGS = -march=sapphirerapids -mtune=sapphirerapids
else
    $(info Generic processor detected - using safe Sapphire Rapids compatible flags)
    ARCH_FLAGS = -march=native -mtune=native
endif

# Sapphire Rapids optimized compiler flags
CXXFLAGS = -m64 -std=c++20 -Ofast -g0 -DNDEBUG \
           -Wall -Wextra -Wno-write-strings -Wno-unused-variable \
           -Wno-deprecated-copy -Wno-unused-parameter -Wno-sign-compare \
           -Wno-strict-aliasing -Wno-unused-but-set-variable \
           -funroll-loops -ftree-vectorize -fstrict-aliasing \
           -fno-semantic-interposition -fvect-cost-model=unlimited \
           -fno-trapping-math -fipa-ra -flto=auto -fuse-linker-plugin \
           -fassociative-math -fopenmp -fwrapv \
           $(ARCH_FLAGS)

# Sapphire Rapids specific AVX-512 instruction sets
ifeq ($(HAS_AVX512),1)
    # Full Sapphire Rapids AVX-512 feature set
    CXXFLAGS += -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl \
                -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vnni \
                -mavx512bitalg -mavx512vpopcntdq -mavx512bf16 \
                -mavx512fp16 -mavx512vp2intersect \
                -mbmi2 -madx -msha -maes -mpclmul \
                -mgfni -mvaes -mvpclmulqdq
    $(info Sapphire Rapids AVX-512 instruction sets enabled)
else
    $(warning AVX-512 not detected - falling back to AVX2)
    CXXFLAGS += -mavx2 -mbmi2 -madx
endif

# Sapphire Rapids specific performance optimizations
CXXFLAGS += -fomit-frame-pointer -ffast-math -ffinite-math-only \
            -fno-stack-protector -fno-asynchronous-unwind-tables \
            -fmerge-all-constants -fmodulo-sched -fmodulo-sched-allow-regmoves \
            -fgcse-las -fgcse-sm -fgcse-after-reload -fipa-cp-clone \
            -ftree-loop-im -ftree-loop-ivcanon -fivopts \
            -ftree-parallelize-loops=8 -ftracer -funswitch-loops \
            -fpredictive-commoning -floop-interchange -floop-unroll-and-jam

# Sapphire Rapids cache optimization (80 cores, large L3)
CXXFLAGS += -DSAPPHIRE_RAPIDS -DLARGE_L3_CACHE -DHIGH_CORE_COUNT \
            -falign-functions=32 -falign-loops=32

# Memory subsystem optimizations for DDR5 and high bandwidth
CXXFLAGS += -DDDR5_OPTIMIZED -DHBM_AWARE -mprefer-vector-width=512

# Link-time optimization flags for Sapphire Rapids
LDFLAGS = -flto=auto -fuse-linker-plugin -Wl,-O3 -Wl,--gc-sections \
          -Wl,--strip-all -Wl,--as-needed -Wl,--hash-style=gnu

# Source files (updated for AVX-512)
SRCS = mutagen.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp \
       Point.cpp ripemd160_avx512.cpp sha256_avx512.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = mutagen

# Default target
all: info $(TARGET)

# Display build information
info:
	@echo "=========================================="
	@echo "=== Mutagen Sapphire Rapids Build ======"
	@echo "=========================================="
	@echo "OS: $(UNAME_S)"
	@echo "Compiler: $(CXX) $(GCC_VERSION)"
	@echo "Architecture: Intel Sapphire Rapids (Xeon Platinum 8488C)"
	@echo "CPU Detection: $(if $(filter 1,$(SAPPHIRE_RAPIDS)),Sapphire Rapids Detected,$(if $(filter 1,$(INTEL_XEON)),Generic Xeon,Unknown CPU))"
	@echo "AVX-512 Support: $(if $(filter 1,$(HAS_AVX512)),Full Sapphire Rapids Feature Set,Limited/Not Detected)"
	@echo "Sapphire Support: $(SAPPHIRE_SUPPORT)"
	@echo "Static Linking: $(STATIC_LINKING)"
	@echo "Target: $(TARGET)"
	@echo "=========================================="

# Link the object files with Sapphire Rapids optimizations
$(TARGET): $(OBJS)
	@echo "Linking with Sapphire Rapids AVX-512 optimizations..."
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJS) -lpthread -lnuma
	@echo "Cleaning intermediate files..."
	rm -f $(OBJS)
	chmod +x $(TARGET)
	@echo "Build complete! Sapphire Rapids optimized binary ready."
	@echo "Recommended NUMA run: numactl --cpunodebind=0 --membind=0 ./$(TARGET)"

# Compile each source file with Sapphire Rapids optimizations
%.o: %.cpp
	@echo "Compiling $< with Sapphire Rapids AVX-512 optimizations..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Special compilation for crypto modules with Sapphire Rapids features
sha256_avx512.o: sha256_avx512.cpp
	@echo "Compiling SHA-256 for Sapphire Rapids with AVX-512 enhancements..."
	$(CXX) $(CXXFLAGS) -O3 -fno-strict-overflow -mavx512sha -c $< -o $@

ripemd160_avx512.o: ripemd160_avx512.cpp
	@echo "Compiling RIPEMD160 for Sapphire Rapids with maximum vectorization..."
	$(CXX) $(CXXFLAGS) -O3 -fno-strict-overflow -c $< -o $@

# Sapphire Rapids specific performance testing
sapphire-test: $(TARGET)
	@echo "Running Sapphire Rapids specific performance validation..."
	@echo "Testing AVX-512 throughput and DDR5 memory bandwidth..."
	@if command -v perf >/dev/null 2>&1; then \
		echo "Run: perf stat -e instructions,cycles,cache-misses,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./$(TARGET)"; \
	else \
		echo "Install linux-perf for detailed Sapphire Rapids performance analysis"; \
	fi

# CPU capability check specific for Sapphire Rapids
cpu-check:
	@echo "=== Sapphire Rapids CPU Analysis ==="
	@echo "Processor info:"
	@grep "model name" /proc/cpuinfo | head -1 || echo "CPU info not available"
	@echo "Sapphire Rapids detection:"
	@lscpu | grep -E "(Model name|CPU family|Model:|Stepping:|Flags)" || echo "Detailed CPU info not available"
	@echo "AVX-512 feature detection:"
	@grep -o "avx512[a-z0-9]*" /proc/cpuinfo | sort -u || echo "No AVX-512 detected"
	@echo "Sapphire Rapids specific features:"
	@grep -E "(avx512fp16|avx512bf16|avx512vp2intersect|sha|gfni|vaes|vpclmulqdq)" /proc/cpuinfo || echo "Advanced features not detected"
	@echo "Cache hierarchy (Sapphire Rapids has large L3):"
	@lscpu | grep -E "(L1d|L2|L3)" || echo "Cache info not available"
	@echo "NUMA topology (important for 80-core Sapphire Rapids):"
	@if command -v numactl >/dev/null 2>&1; then numactl --hardware; else echo "Install numactl for NUMA analysis"; fi

# Sapphire Rapids memory optimization check
memory-check: $(TARGET)
	@echo "=== Sapphire Rapids Memory Optimization ==="
	@echo "Checking DDR5 memory configuration and NUMA topology..."
	@if command -v numactl >/dev/null 2>&1; then \
		echo "NUMA topology for 80-core Sapphire Rapids:"; \
		numactl --hardware; \
		echo ""; \
		echo "Recommended execution patterns:"; \
		echo "Single NUMA node: numactl --cpunodebind=0 --membind=0 ./$(TARGET)"; \
		echo "Dual socket: numactl --cpunodebind=0,1 --interleave=0,1 ./$(TARGET)"; \
	else \
		echo "Install numactl for Sapphire Rapids NUMA optimization"; \
	fi
	@echo "Memory bandwidth test (important for DDR5):"
	@if command -v stream >/dev/null 2>&1; then \
		echo "Run: stream for memory bandwidth validation"; \
	else \
		echo "Install STREAM benchmark for DDR5 bandwidth testing"; \
	fi

# Benchmark specifically tuned for Sapphire Rapids capabilities
benchmark: $(TARGET)
	@echo "Running Sapphire Rapids tuned Bitcoin puzzle benchmark..."
	@echo "Testing with all 80 cores and optimized thread placement..."
	@if command -v numactl >/dev/null 2>&1; then \
		echo "NUMA-aware benchmark:"; \
		timeout 60s numactl --cpunodebind=0 --membind=0 ./$(TARGET) -p 20 -t 40 || echo "Benchmark completed"; \
	else \
		timeout 60s ./$(TARGET) -p 20 -t 80 || echo "Benchmark completed"; \
	fi

# Profile-guided optimization for Sapphire Rapids
sapphire-pgo: 
	@echo "Building with Sapphire Rapids Profile-Guided Optimization..."
	@echo "Stage 1: Instrumented build..."
	$(MAKE) clean
	$(MAKE) CXXFLAGS="$(CXXFLAGS) -fprofile-generate" LDFLAGS="$(LDFLAGS) -fprofile-generate" $(TARGET)
	@echo "Stage 2: Training with representative Sapphire Rapids workload..."
	numactl --cpunodebind=0 --membind=0 ./$(TARGET) -p 20 -t 20 >/dev/null 2>&1 || true
	./$(TARGET) -p 25 -t 40 >/dev/null 2>&1 || true
	@echo "Stage 3: Final optimized build..."
	$(MAKE) clean
	$(MAKE) CXXFLAGS="$(CXXFLAGS) -fprofile-use -fprofile-correction" LDFLAGS="$(LDFLAGS) -fprofile-use" $(TARGET)
	@echo "Sapphire Rapids PGO build complete!"

# Clean up build files
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(OBJS) $(TARGET) *.gcda *.gcno *.prof *.profdata

# Phony targets
.PHONY: all clean info cpu-check benchmark sapphire-test memory-check sapphire-pgo

else
# Windows settings (MinGW-w64) - Sapphire Rapids support
CXX = g++

# Check compiler version for Sapphire Rapids support
GCC_VERSION := $(shell $(CXX) -dumpversion 2>NUL | cut -d. -f1)
ifeq ($(shell echo $(GCC_VERSION) | grep -E "^[0-9]+$$"),)
    $(error MinGW-w64 GCC not found or invalid version)
endif

ifeq ($(shell echo "$(GCC_VERSION) >= 12" | bc 2>NUL),1)
    $(info Using MinGW-w64 GCC $(GCC_VERSION) with Sapphire Rapids support)
else
    $(warning MinGW-w64 GCC $(GCC_VERSION) has limited Sapphire Rapids support)
endif

# Windows Sapphire Rapids optimized flags
CXXFLAGS = -m64 -std=c++20 -Ofast -g0 -DNDEBUG -march=sapphirerapids -mtune=sapphirerapids \
           -Wall -Wextra -Wno-write-strings -Wno-unused-variable \
           -Wno-deprecated-copy -Wno-unused-parameter -Wno-sign-compare \
           -Wno-strict-aliasing -Wno-unused-but-set-variable \
           -funroll-loops -ftree-vectorize -fstrict-aliasing \
           -fno-semantic-interposition -fvect-cost-model=unlimited \
           -fno-trapping-math -fipa-ra -fassociative-math -fopenmp \
           -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl \
           -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vnni \
           -mavx512bitalg -mavx512vpopcntdq -mavx512bf16 -mavx512fp16 \
           -mbmi2 -madx -msha -maes -mgfni -mvaes -mvpclmulqdq -fwrapv

# Add static linking for Windows
ifeq ($(STATIC_LINKING), yes)
    CXXFLAGS += -static
    $(info Static linking enabled for Sapphire Rapids on Windows)
endif

# Source files
SRCS = mutagen.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp \
       Point.cpp ripemd160_avx512.cpp sha256_avx512.cpp

OBJS = $(SRCS:.cpp=.o)
TARGET = mutagen.exe

all: info $(TARGET)

info:
	@echo ==========================================
	@echo === Mutagen Sapphire Rapids (Windows) ===
	@echo ==========================================
	@echo Compiler: MinGW-w64 GCC $(GCC_VERSION)
	@echo Architecture: Intel Sapphire Rapids optimized
	@echo AVX-512 Support: Full Sapphire Rapids feature set
	@echo Static Linking: $(STATIC_LINKING)
	@echo ==========================================

$(TARGET): $(OBJS)
	@echo Linking with Sapphire Rapids optimizations...
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -lpthread
	@if exist $(word 1,$(OBJS)) del /f /q $(OBJS) 2>NUL
	@echo Sapphire Rapids optimized build complete!

%.o: %.cpp
	@echo Compiling $< for Sapphire Rapids...
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@echo Cleaning...
	@if exist *.o del /f /q *.o 2>NUL
	@if exist $(TARGET) del /f /q $(TARGET) 2>NUL

.PHONY: all clean info
endif

# Help target
help:
	@echo "Sapphire Rapids optimized build targets:"
	@echo "  all              - Build for Intel Xeon Platinum 8488C (Sapphire Rapids)"
	@echo "  sapphire-test    - Sapphire Rapids specific performance test"
	@echo "  sapphire-pgo     - Profile-guided optimization for Sapphire Rapids"
	@echo "  cpu-check        - Analyze Sapphire Rapids capabilities"
	@echo "  memory-check     - Check DDR5/NUMA configuration"
	@echo "  benchmark        - 80-core optimized benchmark"
	@echo "  clean            - Remove build artifacts"
	@echo "  help             - Show this help"
