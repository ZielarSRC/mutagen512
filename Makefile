# Detect OS
UNAME_S := $(shell uname -s)

# Enable static linking by default (change to 'no' to use dynamic linking)
STATIC_LINKING = yes

# Compiler settings based on OS
ifeq ($(UNAME_S),Linux)
# Linux settings

# Compiler
CXX = g++

# Intel Xeon Platinum 8488C (Sapphire Rapids) optimized flags
CXXFLAGS = -O3 -m64 -std=c++17 -Ofast -march=sapphirerapids -mtune=sapphirerapids \
           -Wall -Wextra -Wno-write-strings -Wno-unused-variable \
           -Wno-deprecated-copy -Wno-unused-parameter -Wno-sign-compare \
           -Wno-strict-aliasing -Wno-unused-but-set-variable -Wno-unknown-pragmas \
           -funroll-loops -ftree-vectorize -fstrict-aliasing \
           -fno-semantic-interposition -fvect-cost-model=unlimited \
           -fno-trapping-math -fipa-ra -fassociative-math -fopenmp \
           -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl \
           -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vnni \
           -mavx512bitalg -mavx512vpopcntdq -mavx512bf16 -mavx512fp16 \
           -mbmi2 -madx -msha -maes -mgfni -mvaes -mvpclmulqdq \
           -fwrapv -mprefer-vector-width=512

# Source files
SRCS = mutagen.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp \
       Point.cpp ripemd160_avx512.cpp sha256_avx512.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = mutagen

# Link the object files to create the executable and then delete .o files
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -lpthread
	rm -f $(OBJS) && chmod +x $(TARGET)

# Compile each source file into an object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	@echo "Cleaning..."
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean 

else
# Windows settings (MinGW-w64)

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

# Intel Xeon Platinum 8488C optimized flags for Windows
CXXFLAGS = -O3 -m64 -std=c++17 -Ofast -march=sapphirerapids -mtune=sapphirerapids \
           -Wall -Wextra -Wno-write-strings -Wno-unused-variable \
           -Wno-deprecated-copy -Wno-unused-parameter -Wno-sign-compare \
           -Wno-strict-aliasing -Wno-unused-but-set-variable -Wno-unknown-pragmas \
           -funroll-loops -ftree-vectorize -fstrict-aliasing \
           -fno-semantic-interposition -fvect-cost-model=unlimited \
           -fno-trapping-math -fipa-ra -fassociative-math -fopenmp \
           -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl \
           -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vnni \
           -mavx512bitalg -mavx512vpopcntdq -mavx512bf16 -mavx512fp16 \
           -mbmi2 -madx -msha -maes -mgfni -mvaes -mvpclmulqdq \
           -fwrapv -mprefer-vector-width=512

# Add -static flag if STATIC_LINKING is enabled
ifeq ($(STATIC_LINKING), yes)
    CXXFLAGS += -static
else
    $(info Dynamic linking will be used. Ensure required DLLs are distributed)
endif

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
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -lpthread
	del /q $(OBJS)

# Compile each source file into an object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	@echo Cleaning...
	del /q $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean
endif
