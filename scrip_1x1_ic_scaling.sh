#!/bin/bash

echo "4x6 cachelines loaded in the fused microkernel, square panels"
   
	for k in 16 32 48 64 96 128 144 192 256 384 512 768 1024 ;
	do
	    
	    ./torch_1x1 16 16  ${k}   54 54
	done 
   

