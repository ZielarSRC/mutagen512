# Detect OS
UNAME_S := $(shell uname -s)

# Enable static linking by default (change to 'no' to use dynamic linking)
STATIC_LINKING = yes

# Compiler settings based on OS
ifeq ($(UNAME_S),Linux)
# Linux settings for Xeon Platinum 8488C

# Compiler
CXX = g++

# Compiler flags optimized for Xeon Platinum 8488C with AVX-512
CXXFLAGS = -m64 -std=c++17 -Ofast -Wall -Wextra \
           -Wno-write-strings -Wno-unused-variable -Wno-deprecated-copy \
           -Wno-unused-parameter -Wno-sign-compare -Wno-strict-aliasing \
           -Wno-unused-but-set-variable \
           -funroll-loops -ftree-vectorize -fstrict-aliasing -fno-semantic-interposition \
           -fvect-cost-model=unlimited -fno-trapping-math -fipa-ra -flto \
           -fassociative-math -fopenmp -fwrapv \
           -mavx2 -mbmi2 -madx \
           -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512ifma -mavx512vbmi \
           -mtune=icelake-server -mprefer-vector-width=512 \
           -fprefetch-loop-arrays -falign-functions=64 -falign-loops=64

# Link flags
LDFLAGS = -ltbb -lnuma

# Source files
SRCS = mutagen.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp \
       Point.cpp ripemd160_avx512.cpp sha256_avx512.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = mutagen

# Default target
all: $(TARGET)

# Link the object files to create the executable and then delete .o files
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)
	rm -f $(OBJS) && chmod +x $(TARGET)

# Compile each source file into an object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	@echo "Cleaning..."
	rm -f $(OBJS) $(TARGET)

# Generate optimized build for current CPU
native:
	$(CXX) $(CXXFLAGS) -march=native -o $(TARGET) $(SRCS) $(LDFLAGS)

# Phony targets
.PHONY: all clean native

else
# Windows settings (MinGW-w64) for Xeon Platinum 8488C

# Compiler
CXX = g++

# Check if compiler is found
CHECK_COMPILER := $(shell which $(CXX))

# Add MSYS path if the compiler is not found
ifeq ($(CHECK_COMPILER),)
  $(info Compiler not found. Adding MSYS path to the environment...)
  SHELL := powershell
  PATH := C:\msys64\mingw64\bin;$(PATH)
endif

# Compiler flags optimized for Xeon Platinum 8488C with AVX-512 (without LTO for Windows compatibility)
CXXFLAGS = -m64 -std=c++17 -Ofast -Wall -Wextra \
           -Wno-write-strings -Wno-unused-variable -Wno-deprecated-copy \
           -Wno-unused-parameter -Wno-sign-compare -Wno-strict-aliasing \
           -Wno-unused-but-set-variable -funroll-loops -ftree-vectorize \
           -fstrict-aliasing -fno-semantic-interposition -fvect-cost-model=unlimited \
           -fno-trapping-math -fipa-ra -fassociative-math -fopenmp -fwrapv \
           -mavx2 -mbmi2 -madx \
           -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl \
           -mtune=icelake-server -mprefer-vector-width=512 \
           -fprefetch-loop-arrays -falign-functions=64 -falign-loops=64

# Add -static flag if STATIC_LINKING is enabled
ifeq ($(STATIC_LINKING), yes)
    CXXFLAGS += -static
else
    $(info Dynamic linking will be used. Ensure required DLLs are distributed)
endif

# Link flags - note that NUMA isn't typically available on Windows
LDFLAGS = -ltbb

# Source files
SRCS = mutagen.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp \
       Point.cpp ripemd160_avx512.cpp sha256_avx512.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = mutagen.exe

# Default target
all: $(TARGET)

# Link the object files to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)
	del /q $(OBJS)

# Compile each source file into an object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	@echo Cleaning...
	del /q $(OBJS) $(TARGET)

# Generate optimized build for current CPU
native:
	$(CXX) $(CXXFLAGS) -march=native -o $(TARGET) $(SRCS) $(LDFLAGS)

# Phony targets
.PHONY: all clean native
endif
