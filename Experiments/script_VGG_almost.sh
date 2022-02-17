#!/bin/bash
# echo "Parallelizing over output channels, square panels"










export config_kernel_size=5
export config_stride=2
export config_channel_stride=1

echo $OMP_NUM_THREADS;
echo $config_kernel_size
echo $config_stride
echo $config_channel_stride


#for i in 222 114 54 30 12 6;


echo -n 224x222x64 ', '
$1  64 64 222 222 $config_kernel_size $config_stride $config_channel_stride
echo -n 112x114x128 ', '
$1  128 128 114 114 $config_kernel_size $config_stride $config_channel_stride
echo -n 56x54x256 ', '
$1  256 256 54 54 $config_kernel_size $config_stride $config_channel_stride
echo -n 30x30x512 ', '
$1  512 512 28 30 $config_kernel_size $config_stride $config_channel_stride
echo -n 12x12x512 ', '
$1  512 512 14 12 $config_kernel_size $config_stride $config_channel_stride
echo -n 6x6x512 ', '
$1  512 512 6 6 $config_kernel_size $config_stride $config_channel_stride



