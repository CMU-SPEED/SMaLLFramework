#!/bin/bash
# echo "Parallelizing over output channels, square panels"

export config_kernel_size=$3
export config_stride=$4


export OMP_NUM_THREADS=$2
echo $OMP_NUM_THREADS;
echo $config_kernel_size
echo $config_stride


#for i in 222 114 54 30 12 6;


echo -n 224x222x64 ', '
$1  64 222 222 $config_kernel_size  $4 'v'  64    
echo -n 112x114x128 ', '
$1  128  114 114 $config_kernel_size  $4 'v'  128   
echo -n 56x54x256 ', '
$1  256  54 54 $config_kernel_size  $4 'v'  256   
echo -n 30x30x512 ', '
$1  512  28 30 $config_kernel_size  $4 'v'  512   
echo -n 12x12x512 ', '
$1  512  14 12 $config_kernel_size  $4 'v'  512   
echo -n 6x6x512 ', '
$1  512  6 6 $config_kernel_size  $4 'v'  512   



