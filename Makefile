# Makefile optimized for Intel Xeon Platinum 8488C with AVX-512
# Detect OS
UNAME_S := $(shell uname -s)

# Enable static linking by default (change to 'no' to use dynamic linking)
STATIC_LINKING = yes

# Compiler settings based on OS
ifeq ($(UNAME_S),Linux)
# Linux settings

# Compiler - Intel oneAPI compiler for best AVX-512 optimization
# If not available, fall back to GCC
INTEL_CXX := $(shell which icpx 2>/dev/null)
ifdef INTEL_CXX
    CXX = icpx
    $(info Using Intel oneAPI compiler for optimal Xeon 8488C performance)
    # Intel compiler flags optimized for Xeon Platinum
    CXXFLAGS = -std=c++17 -O3 -xHOST -qopt-zmm-usage=high -march=sapphirerapids \
               -fiopenmp -qopenmp-simd -qopt-report=5 -qopt-report-phase=vec \
               -qopt-prefetch=5 -qoverride-limits -Wall -Wextra -Wno-deprecated-declarations \
               -Wno-unknown-pragmas -ipo -inline-forceinline -finline-functions \
               -falign-functions=32 -falign-loops=32 -ip -DNDEBUG
else
    CXX = g++
    $(info Using GCC with AVX-512 optimizations for Xeon 8488C)
    # GCC flags optimized for Xeon Platinum with AVX-512
    CXXFLAGS = -m64 -std=c++17 -Ofast -march=sapphirerapids -mtune=sapphirerapids \
               -Wall -Wextra -Wno-write-strings -Wno-unused-variable -Wno-deprecated-copy \
               -Wno-unused-parameter -Wno-sign-compare -Wno-strict-aliasing \
               -funroll-loops -ftree-vectorize -fstrict-aliasing -fno-semantic-interposition \
               -fvect-cost-model=unlimited -fno-trapping-math -fipa-ra -flto -ffast-math \
               -fassociative-math -fopenmp -mavx512f -mavx512cd -mavx512bw -mavx512dq \
               -mavx512vl -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vnni \
               -mavx512bitalg -mavx512vpopcntdq -mprefer-vector-width=512 \
               -mbmi2 -madx -fwrapv -falign-functions=64 -falign-loops=64
endif

# Source files - updated to use AVX-512 versions
SRCS = mutagen.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp \
       Point.cpp ripemd160_avx512.cpp sha256_avx512.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = mutagen

# Link the object files to create the executable and then delete .o files
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
	rm -f $(OBJS) && chmod +x $(TARGET)

# Compile each source file into an object file with optimization reports
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	@echo "Cleaning..."
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean 

else
# Windows settings (MinGW-w64 or MSVC)

# Check if Intel compiler is available
INTEL_CXX := $(shell where icpx 2>nul)
ifdef INTEL_CXX
    CXX = icpx
    $(info Using Intel oneAPI compiler for optimal Xeon 8488C performance)
    # Intel compiler flags for Windows
    CXXFLAGS = -std=c++17 -O3 -xHOST -qopt-zmm-usage=high -march=sapphirerapids \
               -Qopenmp -Qopenmp-simd -Qopt-report:5 -Qopt-report-phase:vec \
               -Qopt-prefetch:5 -Qoverride-limits -Wall -Wextra -ipo -Qinline-forceinline \
               -Qfnalign:32 -Qunroll-aggressive -Qipo -DNDEBUG
else
    # Check if MSVC cl.exe is available
    MSVC_CXX := $(shell where cl.exe 2>nul)
    ifdef MSVC_CXX
        CXX = cl.exe
        $(info Using MSVC with AVX-512 optimizations)
        # MSVC flags for AVX-512
        CXXFLAGS = /std:c++17 /O2 /Ob3 /Oi /Ot /Oy /GL /arch:AVX512 /fp:fast \
                   /openmp /openmp:llvm /favor:INTEL64 /Qpar /Qvec-report:2 \
                   /D "NDEBUG" /EHsc /MD
        # Different linking command for MSVC
        LINK_CMD = $(CXX) $(CXXFLAGS) /Fe:$(TARGET) $(OBJS)
        OBJ_EXT = .obj
    else
        CXX = g++
        $(info Using MinGW-w64 with AVX-512 optimizations)
        # MinGW flags for AVX-512
        CXXFLAGS = -m64 -std=c++17 -Ofast -march=sapphirerapids -mtune=sapphirerapids \
                   -Wall -Wextra -Wno-write-strings -Wno-unused-variable -Wno-deprecated-copy \
                   -Wno-unused-parameter -Wno-sign-compare -Wno-strict-aliasing \
                   -funroll-loops -ftree-vectorize -fstrict-aliasing -fno-semantic-interposition \
                   -fvect-cost-model=unlimited -fno-trapping-math -fipa-ra -ffast-math \
                   -fassociative-math -fopenmp -mavx512f -mavx512cd -mavx512bw -mavx512dq \
                   -mavx512vl -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vnni \
                   -mavx512bitalg -mavx512vpopcntdq -mprefer-vector-width=512 \
                   -mbmi2 -madx -fwrapv -falign-functions=64 -falign-loops=64
        # Default linking command for MinGW
        LINK_CMD = $(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
        OBJ_EXT = .o
    endif
endif

# Add -static flag if STATIC_LINKING is enabled and not using MSVC
ifeq ($(STATIC_LINKING), yes)
    ifndef MSVC_CXX
        CXXFLAGS += -static
    endif
else
    $(info Dynamic linking will be used. Ensure required DLLs are distributed)
endif

# Source files - updated to use AVX-512 versions
SRCS = mutagen.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp \
       Point.cpp ripemd160_avx512.cpp sha256_avx512.cpp

# Object files - handle MSVC differently
ifndef MSVC_CXX
    OBJS = $(SRCS:.cpp=.o)
else
    OBJS = $(SRCS:.cpp=.obj)
endif

# Target executable
TARGET = mutagen_avx512.exe

# Default target
all: $(TARGET)

# Link the object files to create the executable
$(TARGET): $(OBJS)
	$(LINK_CMD)
	del /q $(OBJS)

# Compile each source file into an object file
ifndef MSVC_CXX
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
else
%.obj: %.cpp
	$(CXX) $(CXXFLAGS) /c $< /Fo$@
endif

# Clean up build files
clean:
	@echo Cleaning...
	del /q $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean
endif
