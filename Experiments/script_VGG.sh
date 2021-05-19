#!/bin/bash

echo "Parallelizing over output channels, square panels"
export OMP_NUM_THREADS=1;
echo $OMP_NUM_THREADS;

#for i in 222 114 54 30 12 6;
echo -n 222, 222, 64, 
$1 64 64 222 222
echo -n 114, 114, 128, 
$1 128 128 114 114
echo -n 54, 54, 256, 
$1 256 256 54 54
echo -n 30, 30, 512, 
$1 512 512 30 30
echo -n 12, 12, 512, 
$1 512 512 12 12
