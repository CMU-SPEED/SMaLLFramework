#!/bin/bash

echo "Parallelizing over output channels, square panels"
export OMP_NUM_THREADS=$2;
echo $OMP_NUM_THREADS;
for i in 12 30 54 114 222 504;
do

    for j in 64;
	     do
		 for k in 64 ;
		 do
		     echo -n  ${i},  ${k}, " "
		     $1  ${j} ${k} ${i} ${i}
		 done 
    done
done
