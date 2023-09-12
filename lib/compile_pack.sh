#!/bin/bash -x

SMALL_ROOT=/home/nicholai/prog/SMaLLFramework
PLATFORM=$1
PLATFORM_FLAGS=""

if [ ${PLATFORM} == "zen2" ]; then
    PLATFORM_FLAGS="-DZEN2 -DUARCH_ZEN2 -mavx2 -mfma -I${SMALL_ROOT}/include/small/platforms/zen2"
elif [ ${PLATFORM} == "arm" ]; then
    PLATFORM_FLAGS="-DARM -DUARCH_ARM-I${SMALL_ROOT}/include/small/platforms/arm"
elif [ ${PLATFORM} == "ref" ]; then
    PLATFORM_FLAGS="-DREF -DUARCH_REF -I${SMALL_ROOT}/include/small/platforms/reference"
else
    echo "Invalid platform"
    exit -1
fi

g++ -O3 -fopenmp -fPIC -march=native -I${SMALL_ROOT}/include -I${SMALL_ROOT}/include/small ${PLATFORM_FLAGS} --shared ${SMALL_ROOT}/lib/pack.cpp -o ${SMALL_ROOT}/lib/libpack.so;

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SMALL_ROOT}/lib
