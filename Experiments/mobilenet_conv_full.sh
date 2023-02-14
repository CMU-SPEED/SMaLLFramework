echo "Mobilenet Layer By Layer Multiples of kernel size"

export config_kernel_size=$3
export config_stride=$4
export padding=$5


output_dims=(48  24 24  12  12  6   6   3   3)
input_dims=( 48  48 24  24  12  12  6   6   3)
channel_dims=(32 64 128 128 256 256 512 512 1024 1024)

for i in ${!output_dims[@]}; do
    let "Ci  = ${channel_dims[$i]}"
    let "Co  = ${channel_dims[$i + 1]}"
    let "kernel_size = $3"
    stride=1
    if [ $kernel_size == '1' ]
    then
        let "stride = 1"
    else
        let "stride = ${input_dims[$i]}/${output_dims[$i]}"
    fi
    echo  -n $1  $Ci ${input_dims[$i]} ${input_dims[$i]} $kernel_size  $stride $padding  $Co " "
    $1  $Ci ${input_dims[$i]} ${input_dims[$i]} $kernel_size  $stride $padding  $Co
done