#!/bin/bash

path_to_build=$1
path_to_tld=$2
uarch=$3

cp -r ./small $path_to_build/

# Set up the include dir
mkdir -p $path_to_build/small/quantized/include
mkdir $path_to_build/small/quantized/include/small
mkdir $path_to_build/small/quantized/include/small/platforms

cp $path_to_tld/include/small/*.* $path_to_build/small/quantized/include/small
cp $path_to_tld/include/small/platforms/$uarch/*.* $path_to_build/small/quantized/include/small/platforms
