#!/bin/bash

echo "Single block of output channels, square panels"
for i in 12 36 96 126 252 504;
do
   
	for k in  32 48 64 96 128 144 192 256 384 512 768 1024 ;
	do
	    echo -n  ${i},  ${k},
	    ./torch_pool  ${k} 16  ${i} ${i}
	done 
   
done
