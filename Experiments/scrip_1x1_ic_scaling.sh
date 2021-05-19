#!/bin/bash

echo "4x6 cachelines loaded in the fused microkernel, square panels"
for k in 16 32 64 128 256 512;
do
    #48 64 96 128 144 192 256 384 512 768 1024 ;
    for j in 1 2 3 4;
    do
	for l in 12 30 54 114 222 504
	    echo -n $j $k $l 
	    ./torch_conv  $((${1}*${j})) ${k}   $l $l
	done 
   
done
