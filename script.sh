#!/bin/bash

for i in `seq 12 6 36`;
do
    for j in `seq 3  2 33`;
    do
	for k in `seq 32 16 64`;
	do
	    echo ${i}, ${j} , ${k}
	    ./torch_pool  ${k} 16  ${j} ${i}
	done 
    done
done
