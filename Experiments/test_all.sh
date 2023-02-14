#!/bin/bash

# Scott usage from build dir: ../Experiments/test_all.sh abstract_interface sweep_sizes

current_date=`date +"%Y-%m-%d"`
results_dir="./results"
path_to_results="./results/"${current_date}

[! -d $results_dir ] && echo " Creating results directory." && mkdir $results_dir 

[ ! -d $path_to_results ] && echo " Creating directory " $path_to_results && mkdir $path_to_results

num_th=${OMP_NUM_THREADS}
# num_th=1
test_type=$1
expt=$2
echo "Writing results to today's directory: " $path_to_results
valid_log_file=${path_to_results}/${test_type}_${expt}_valid_output_${num_th}.log
padding_log_file=${path_to_results}/${test_type}_${expt}_padding_output_${num_th}.log
echo "Tesing different layers and padding config on " $num_th " threads"

echo "Checking " $test_type " on " $expt "experiment"
echo "Run " $expt " Experiment? [Y/n]"
read check_run

if [ $check_run == "Y" ]; then
echo "outputs in files: " ${valid_log_file} " and " ${padding_log_file}
echo " Testing RELU"
../Experiments/$expt.sh ./${test_type}_RELU.exe 6 1 1 v > $valid_log_file
echo " "
echo " "

echo " Testing POOL"
../Experiments/$expt.sh ./${test_type}_POOL.exe 6 3 2 v >> $valid_log_file
echo " "
echo " "

echo " Testing DW"
../Experiments/$expt.sh ./${test_type}_DW_CONV.exe 6 3 1 v >> $valid_log_file
echo " "
echo " "

echo " Testing CONV"
../Experiments/$expt.sh ./${test_type}_CONV.exe 6 3 1 v >> $valid_log_file
echo " "
echo " "

echo " Testing 1x1 CONV"
../Experiments/$expt.sh ./${test_type}_CONV.exe 6 1 1 v >> $valid_log_file
echo " "
echo " "

echo " Testing Padding POOL"
../Experiments/$expt.sh ./${test_type}_POOL.exe 6 3 2 f > $padding_log_file
echo " "
echo " "

echo " Testing  Padding DW"
../Experiments/$expt.sh ./${test_type}_DW_CONV.exe 6 3 1 f >> $padding_log_file
echo " "

echo " "

echo " Testing  Padding DW"
../Experiments/$expt.sh ./${test_type}_DW_CONV.exe 6 3 2 f >> $padding_log_file
echo " "
echo " "


echo " Testing  Padding CONV"
../Experiments/$expt.sh ./${test_type}_CONV.exe 6 3 1 f >> $padding_log_file
echo " "
echo " "

echo " Testing  Padding CONV"
../Experiments/$expt.sh ./${test_type}_CONV.exe 6 3 2 f >> $padding_log_file
echo " "
echo " "

fi