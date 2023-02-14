#!/bin/bash
# echo "Parallelizing over output channels, square panels"

export Wob=6
export Cob=16

export config_kernel_size=$3
export config_stride=$4


# export OMP_NUM_THREADS=1
export NUM_BLOCKS=$2

echo "OMP_NUM_THREADS: " $OMP_NUM_THREADS
echo "NUM_BLOCKS:      " $NUM_BLOCKS
echo "kernel_size:     " $config_kernel_size
echo "stride:          " $config_stride

export padding=$5

let "min_ip_height = $config_kernel_size"
let "min_ip_width = ($Wob - 1)*$config_stride+$config_kernel_size"

echo "testing layer type: ", $1
let "op_blocks = $Cob*NUM_BLOCKS"

echo "** Single Output element ", $config_kernel_size ,"x", $config_kernel_size, "x", $Cob
echo "Command: " $1  $Cob $config_kernel_size $config_kernel_size $config_kernel_size  $config_stride $padding $Cob
$1  $Cob $config_kernel_size $config_kernel_size $config_kernel_size  $config_stride $padding  $Cob 

echo "** Single Output tile ", $min_ip_height ,"x", $min_ip_width, "x", $Cob
echo "Command: " $1  $Cob $min_ip_height $min_ip_width  $config_kernel_size  $config_stride $padding $Cob
$1  $Cob $min_ip_height $min_ip_width  $config_kernel_size  $config_stride $padding  $Cob

echo "** Many output channel blocks in 1 group", $min_ip_height ,"x", $min_ip_width, "x", $op_blocks
echo "Command: " $1  $Cob $min_ip_height $min_ip_width  $config_kernel_size  $config_stride $padding $op_blocks
$1  $Cob $min_ip_height $min_ip_width  $config_kernel_size  $config_stride $padding  $op_blocks  

echo "** Many input channel blocks", $min_ip_height ,"x", $min_ip_width, "x", $op_blocks
echo "Command: " $1  $op_blocks $min_ip_height $min_ip_width  $config_kernel_size  $config_stride $padding  $Cob
$1  $op_blocks $min_ip_height $min_ip_width  $config_kernel_size  $config_stride $padding  $Cob  

echo "** Large Size", 30 ,"x", 30, "x", $op_blocks
echo "Command: " $1  $op_blocks 30 30 $config_kernel_size  $config_stride $padding $op_blocks
$1  $op_blocks 30 30 $config_kernel_size  $config_stride $padding  $op_blocks


# echo -n 224x222x64 ', '
# $1  64 222 222 $config_kernel_size  $4 'v'  64   
# echo -n 114x114x128 ', '
# $1  128  116 116 $config_kernel_size  $4 'v'  128   
# echo -n 56x56x256 ', '
# $1  256  56 56 $config_kernel_size  $4 'v'  256   
# echo -n 32x32x512 ', '
# $1  512  28 32 $config_kernel_size  $4 'v'  512   
# echo -n 12x12x512 ', '
# $1  512  14 14 $config_kernel_size  $4 'v'  512   
# echo -n 6x6x512 ', '
# $1  512  8 8 $config_kernel_size  $4 'v'  512   



