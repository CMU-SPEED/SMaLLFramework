#!/bin/bash

echo "Parallelizing over output channels, square panels"
export OMP_NUM_THREADS=$2;

echo $OMP_NUM_THREADS;

#for i in 222 114 54 30 12 6;

# echo -n 224x224x96 ' '
# $1 96 96 224 224
# echo -n 112x112x192 ' '
# $1 192 192 112 112
echo -n 56x56x384 ' '
$1 384 384 56 56
echo -n 28x28x768 ' '
$1 768 768 28 28
echo -n 14x14x768 ' '
$1 768 768 14 14

echo -n 7x7x1536 ' '
$1 1536 1536 7 7