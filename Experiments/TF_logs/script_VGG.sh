#!/bin/bash

echo "Parallelizing over output channels, square panels"
export OMP_NUM_THREADS=$2;

echo $OMP_NUM_THREADS;

#for i in 222 114 54 30 12 6;

echo -n 224x224x64 ' '
$1 64 64 224 224
echo -n 112x112x128 ' '
$1 128 128 112 112
echo -n 56x56x256 ' '
$1 256 256 56 56
echo -n 28x28x512 ' '
$1 512 512 28 28
echo -n 14x14x512 ' '
$1 512 512 14 14

echo -n 7x7x512 ' '
$1 512 1024 7 7