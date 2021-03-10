#!/bin/bash

echo "Parallelizing over output channels, square panels"
export OMP_NUM_THREADS=4;
echo $OMP_NUM_THREADS;
for i in 12 36 96 126 252 504;
do

    for j in 64;
	     do
		 for k in 32 48 64 96 128 144 192 256 384 512 768 1024 ;
		 do
		     echo -n  ${i},  ${k},
		     ./torch_pool  ${j} ${k} ${i} ${i}
		 done 
    done
done
