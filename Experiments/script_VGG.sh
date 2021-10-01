#!/bin/bash

echo "Parallelizing over output channels, square panels"
export OMP_NUM_THREADS=$2;

echo $OMP_NUM_THREADS;

#for i in 222 114 54 30 12 6;

echo -n 224 224 64 ' '
$1 64 64 224 224
echo -n 112 112 128 ' '
$1 128 128 112 112
echo -n 56 56 256 ' '
$1 256 256 56 56
echo -n 28 28 512 ' '
$1 512 512 28 28
echo -n 14 14 512 ' '
$1 512 512 14 14

echo -n 7 7 512 ' '
$1 512 1024 7 7