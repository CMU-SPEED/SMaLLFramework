#!/bin/bash -x

SMALL_ROOT=/home/nicholai/prog/SMaLLFramework
PLATFORM=$1
MODEL_FILE=$2

# echo "SMALL_ROOT: ${SMALL_ROOT}"
# echo "PLATFORM: ${PLATFORM}"
# echo "MODEL_FILE: ${MODEL_FILE}"

# update lib
rm -rf ${SMALL_ROOT}/lib/libsmall.so

# zen2
if [ ${PLATFORM} == "zen2" ]; then
    g++ -O3 -fPIC -DZEN2 -DUARCH_ZEN2 -mavx2 -mfma -march=native -I${SMALL_ROOT}/lib/include -I${SMALL_ROOT}/include -I${SMALL_ROOT}/include/small -I${SMALL_ROOT}/include/small/platforms/zen2 --shared ${SMALL_ROOT}/lib/layers.cpp -o ${SMALL_ROOT}/lib/libsmall.so
fi

# ref 
if [ ${PLATFORM} == "ref" ]; then
    g++ -O3 -fPIC -DREF -DUARCH_REF -I${SMALL_ROOT}/lib/include -I${SMALL_ROOT}/include -I${SMALL_ROOT}/include/small -I${SMALL_ROOT}/include/small/platforms/reference --shared ${SMALL_ROOT}/lib/layers.cpp -o ${SMALL_ROOT}/lib/libsmall.so
fi

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SMALL_ROOT}/lib

echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

python run_onnx_model.py -o /home/nicholai/prog/nico-onnx-mlir ${MODEL_FILE}
