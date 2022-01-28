#!/bin/bash
export OMP_NUM_THREADS=$3
echo "Parallelizing over ${3} threads fusing ${2}"
# echo "Single block of output channels, square panels"
echo -n "56x56x64x256 "
$1 64  64  56 56 256 
echo -n "28x28x128x512 "
$1 128 128 28 28 512 
echo -n "14x14x256x1024 "
$1 256 256  14 14 1024
echo -n "7x7x512x2048 "
$1 512 512  7 7 2048
