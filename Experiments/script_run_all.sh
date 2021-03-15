#!/bin/bash
echo "Parallel Expts"
echo "Power of 2 sizes"
./Experiments/script.sh ./torch_pool > ./Results/unfused_sizes_March14.txt
./Experiments/script.sh ./torch_pool_fused > ./Results/fused_sizes_March14.txt
./Experiments/script.sh ./torch_pool_not_buffered > ./Results/not_buffered_sizes_March14.txt
./Experiments/script.sh ./torch_pool_complete > ./Results/complete_sizes_March14.txt
echo "VGG sizes"
./Experiments/script_VGG.sh ./torch_pool > ./Results/unfused_VGG_sizes_March14.txt
./Experiments/script_VGG.sh ./torch_pool_fused > ./Results/fused_VGG_sizes_March14.txt
./Experiments/script_VGG.sh ./torch_pool_not_buffered > ./Results/not_buffered_VGG_sizes_March14.txt
./Experiments/script_VGG.sh ./torch_pool_complete > ./Results/complete_VGG_sizes_March14.txt

echo "VaryK"
./Experiments/varyK_pool_script.sh ./torch_pool > ./Results/unfused_varyK_March14.txt
./Experiments/varyK_pool_script.sh ./torch_pool_fused > ./Results/fused_varyK_March14.txt
echo "VaryC"
./Experiments/varyC_pool_script.sh ./torch_pool > ./Results/unfused_varyC_March14.txt
./Experiments/varyC_pool_script.sh ./torch_pool_fused > ./Results/fused_varyC_March14.txt

echo "Sequential Expts"

echo "Power of 2 sizes"
./Experiments/script.sh ./seq_torch_pool > ./Results/seq_sizes_March14.txt
./Experiments/script.sh ./seq_torch_pool_fused > ./Results/seq_fused_sizes_March14.txt
./Experiments/script.sh ./seq_torch_pool_not_buffered > ./Results/seq_not_buffered_sizes_March14.txt
./Experiments/script.sh ./seq_torch_pool_complete > ./Results/seq_complete_sizes_March14.txt
echo "VGG sizes"
./Experiments/script_VGG.sh ./seq_torch_pool > ./Results/seq_VGG_sizes_March14.txt
./Experiments/script_VGG.sh ./seq_torch_pool_fused > ./Results/seq_fused_VGG_sizes_March14.txt
./Experiments/script_VGG.sh ./seq_torch_pool_not_buffered > ./Results/seq_not_buffered_VGG_sizes_March14.txt
./Experiments/script_VGG.sh ./seq_torch_pool_complete > ./Results/seq_complete_VGG_sizes_March14.txt

echo "VaryK"
./Experiments/varyK_pool_script.sh ./seq_torch_pool > ./Results/seq_varyK_March14.txt
./Experiments/varyK_pool_script.sh ./seq_torch_pool_fused > ./Results/seq_fused_varyK_March14.txt
echo "VaryC"
./Experiments/varyC_pool_script.sh ./seq_torch_pool > ./Results/seq_varyC_March14.txt
./Experiments/varyC_pool_script.sh ./seq_torch_pool_fused > ./Results/seq_fused_varyC_March14.txt
