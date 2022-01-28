#!/bin/bash

echo "Parallelizing over output channels, square panels"
export OMP_NUM_THREADS=$2;

echo $OMP_NUM_THREADS;

#for i in 222 114 54 30 12 6;


echo -n 224x222x64 ' '
$1  64 64 222 222
echo -n 112x114x128 ' '
$1  128 128 114 114
echo -n 56x54x256 ' '
$1  256 256 54 54
echo -n 30x30x512 ' '
$1  512 512 28 30
echo -n 12x12x512 ' '
$1  512 512 14 12
echo -n 6x6x512 ' '
$1  512 512 6 6
