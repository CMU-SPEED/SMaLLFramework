#!/bin/bash
shopt -s nullglob
small_root=$1
cd $small_root/build
files=($( ls ./test/*.exe ))
echo $files
for ((i=0; i<${#files[@]}; i++)); do
    #do something to each element of array
    ${files[$i]}
    if [ $? -eq 0 ]; then
        echo "Passed"
    else
        exit -1;
    fi
done