#!/bin/bash

# Ścieżka do kompilatora Intel
INTEL_CXX="/opt/intel/oneapi/compiler/2025.1/bin/icpx"
INTEL_CC="/opt/intel/oneapi/compiler/2025.1/bin/icx"

# Inicjalizacja środowiska OneAPI (jeśli potrzebne)
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
fi

# Upewnij się, że kompilator istnieje
if [ ! -x "$INTEL_CXX" ]; then
    echo "Kompilator $INTEL_CXX nie istnieje lub nie ma uprawnień do wykonania"
    exit 1
fi

# Usuń poprzednie pliki obiektowe
rm -f *.o mutagen

# Flagi optymalizacji dla Xeon Platinum 8488C
CXXFLAGS="-std=c++17 -O3 -xHOST -qopt-zmm-usage=high -ipo -qopenmp -fp-model fast=2"
AVX512_FLAGS="-mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vnni -mavx512bitalg -mavx512vpopcntdq"

# Optymalizacje dla pamięci i prefetching
MEM_FLAGS="-qopt-prefetch=5 -qopt-mem-layout-trans=3"

# Dodatkowe optymalizacje
EXTRA_FLAGS="-fargument-noalias -ansi-alias -qoverride-limits -qopt-dynamic-align"

# Wszystkie flagi razem
ALL_FLAGS="$CXXFLAGS $AVX512_FLAGS $MEM_FLAGS $EXTRA_FLAGS"

# Lista plików źródłowych
SOURCES=("mutagen.cpp" "SECP256K1.cpp" "Int.cpp" "IntGroup.cpp" "IntMod.cpp" "Point.cpp" "ripemd160_avx512.cpp" "sha256_avx512.cpp")

# Kompilacja plików źródłowych
echo "Kompilacja z użyciem $INTEL_CXX (OneAPI 2025.1)"
echo "Optymalizacja dla Xeon Platinum 8488C z AVX-512"

for src in "${SOURCES[@]}"; do
    obj="${src%.cpp}.o"
    echo "Kompilacja: $src -> $obj"
    $INTEL_CXX $ALL_FLAGS -c "$src" -o "$obj"
    
    # Sprawdź czy kompilacja się powiodła
    if [ $? -ne 0 ]; then
        echo "Błąd podczas kompilacji $src"
        exit 1
    fi
done

# Linkowanie
echo "Linkowanie..."
$INTEL_CXX $ALL_FLAGS -o mutagen *.o

if [ $? -eq 0 ]; then
    echo "Kompilacja zakończona sukcesem: utworzono plik mutagen"
    # Opcjonalnie: pokazuje informacje o pliku wykonywalnym
    ls -lh mutagen
else
    echo "Błąd podczas linkowania"
    exit 1
fi
