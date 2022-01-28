#!/bin/bash
export OMP_NUM_THREADS=$3
echo "Parallelizing over ${3} threads fusing ${2}"
# echo "Single block of output channels, square panels"
echo -n "56x56x256x64 "
$1 64  256  56 56 64
echo -n "28x28x512x128 "
$1 128 512 28 28 128
echo -n "14x14x1024x256 "
$1 256 1024 14 14 256 
echo -n "7x7x2048x512 "
$1 512 2048 7 7  512