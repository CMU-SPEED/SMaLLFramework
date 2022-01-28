
#!/bin/bash

echo "Parallelizing over output channels, square panels"
export OMP_NUM_THREADS=$2;

echo $OMP_NUM_THREADS;

#for i in 222 114 54 30 12 6;

G=1
F_conv=1
S_conv=1

F_dw=3
S_dw=2
block=dw

echo "stride 1 sizes"

echo -n 56 56 64 128 ' '
$1  64 128 56 56
echo -n 28 28 128 256 ' '
$1  128 256 28 28
echo -n 14 14 256 512 ' '
$1  256 512 14 14
echo -n 14 14 512 512 ' '
$1  512 512 14 14
# echo -n 6 6 512 ' '
# $1  512 512 6 6