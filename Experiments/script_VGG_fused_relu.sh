#!/bin/bash
# echo "Parallelizing over output channels, square panels"

export config_kernel_size=$3
export config_stride=$4
export padding=$5

export OMP_NUM_THREADS=$2
echo $OMP_NUM_THREADS;
echo $config_kernel_size
echo $config_stride


#for i in 222 114 54 30 12 6;


echo -n 224x224x64 ', '
$1  64 224 224 $config_kernel_size  $4 $padding  64   1 1 v 64    
echo -n 112x112x128 ', '
$1  128  112 112 $config_kernel_size  $4 $padding  128  1 1 v 128   
echo -n 56x56x256 ', '
$1  256  56 56 $config_kernel_size  $4 $padding  256  1 1 v 256   
echo -n 28x28x512 ', '
$1  512  28 28 $config_kernel_size  $4 $padding  512  1 1 v 512   
echo -n 14x14x512 ', '
$1  512  14 14 $config_kernel_size  $4 $padding  512  1 1 v 512   
echo -n 7x7x512 ', '
$1  512  7 7 $config_kernel_size  $4 $padding  512  1 1 v 512   
