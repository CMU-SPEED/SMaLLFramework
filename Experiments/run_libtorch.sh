#!/bin/bash
export CONV_PATH=$1
echo $CONV_PATH
for type in conv 1x1 group dw ;
do
    echo $type
    $CONV_PATH/Experiments/script_VGG_almost.sh $CONV_PATH/build/torch_$type.x $type 6
done
# $CONV_PATH/Experiments/script_VGG_almost.sh $CONV_PATH/torch_1x1 bneck 6
# $CONV_PATH/Experiments/script_VGG_almost.sh ./torch_group group 6
# $CONV_PATH/Experiments/script_VGG_almost.sh ./torch_dw dw 6
