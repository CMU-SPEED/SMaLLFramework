#!/bin/bash

./Experiments/varyK_pool_script.sh ./torch_avg_pool 6 | tee vary_K_l_avgpool_l3_6_mar18.txt
./Experiments/varyK_pool_script.sh ./torch_avg_pool 2 | tee vary_K_l_avgpool_l3_2_mar18.txt
./Experiments/varyK_pool_script.sh ./torch_avg_pool 4 | tee vary_K_l_avgpool_l3_4_mar18.txt
