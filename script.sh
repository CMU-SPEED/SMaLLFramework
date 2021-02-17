#!/bin/bash

for i in `seq 6 6 222`;
do
    ./torch_1x1  32 16 16 1 ${i}
done
