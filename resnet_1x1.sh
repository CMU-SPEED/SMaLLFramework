#!/bin/bash

echo "Single block of output channels, square panels"
echo -n "conv2 54x54 "
./torch_pool 64 256 54 54
echo -n "conv3 30x30 "
./torch_pool 128 512 30 30
echo -n "conv4 12x12 "
./torch_pool 256 1024 12 12
echo -n "conv5 6x6"
./torch_pool 512 2048 6 6
