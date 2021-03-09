#!/bin/bash

echo "Parallelizing over output channels, square panels"
export OMP_NUM_THREADS=4;
echo $OMP_NUM_THREADS;

#for i in 222 114 54 30 12 6;
	 
./torch_pool 64 64 222 222
./torch_pool 128 128 114 114
./torch_pool 256 256 54 54
./torch_pool 512 512 30 30
./torch_pool 512 512 12 12

