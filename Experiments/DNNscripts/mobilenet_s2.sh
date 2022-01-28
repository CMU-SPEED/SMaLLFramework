
#!/bin/bash

echo "Parallelizing over output channels square panels"
export OMP_NUM_THREADS=$2;

echo $OMP_NUM_THREADS;

#for i in 224 114 54 30 12 6;

G=1
F_conv=1
S_conv=1

F_dw=3
S_dw=2
block=dw



echo "stride2 sizes"
echo -n 224 224 32 32 ' '
$1  32 32 224 224 $G $F_conv $S_conv $F_dw $S_dw $block
echo -n 112 112 32 64 ' '
$1  32 64 112 112 $G $F_conv $S_conv $F_dw $S_dw $block
echo -n 56 56 128 128 ' '
$1  128 128 56 56 $G $F_conv $S_conv $F_dw $S_dw $block
echo -n 28 28 256 128 ' '
$1  256 256 28 28 $G $F_conv $S_conv $F_dw $S_dw $block
echo -n 14 14 512 512 ' '
$1  512 512 14 14 $G $F_conv $S_conv $F_dw $S_dw $block
echo -n 7 7 512 ' '
$1  512 1024 7 7 $G $F_conv $S_conv $F_dw $S_dw $block