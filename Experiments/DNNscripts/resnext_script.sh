#!/bin/bash
export OMP_NUM_THREADS=$3
echo "Parallelizing over ${3} threads fusing ${2}"
# echo "Single block of output channels, square panels"
echo -n "56x56x128 "
$1   128 128 56 56
echo -n "28x28x256 "
$1  256 256 28 28
echo -n "14x14x512 "
$1 512 512 14 14
echo -n "7x7x1024 "
$1  1024 1024 7 7
