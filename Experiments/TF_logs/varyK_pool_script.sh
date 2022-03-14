#!/bin/bash

echo "Parallelizing over output channels, square panels"
export OMP_NUM_THREADS=$2;
echo $OMP_NUM_THREADS;
for i in 12 30 504;
do

    for j in 64;
	     do
<<<<<<< HEAD
		 for k in 64 ;
=======
		 for k in 32 64 128  256  512  1024 ;
>>>>>>> 8d469ed6fc7afebbb6de0bc9d9cdc32f1ee75f4c
		 do
		     echo -n  ${i},  ${k}, " "
		     $1  ${j} ${k} ${i} ${i}
		 done 
    done
done
