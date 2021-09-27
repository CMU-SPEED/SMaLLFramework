// #include <torch/torch.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>
#include<cstring>
//Sample Convolution driver

#define RUNS 1
#define PARALLEL 1


#include "src/direct_convolution.h"

#include "src/utils.h"


int main(int argc, char ** argv)
{
  if(argc!=5)
  {
    printf("USAGE: torch_pool < 3x3 Input Channels> <3x3 Output Channels> <Output Height> <Output Width (multiple of 6)>\n");
    return 0;
  }

  // Setup Problem Size from command line variables
  int C_i = atoi(argv[1]);
  int C_o = atoi(argv[2]);
  // int C_o_1 = atoi(argv[3]);

  constexpr int kernel_size =   3;
  constexpr int stride = 1;
  constexpr int padding = 0;(kernel_size - 1)/2;
  int output_rows = atol(argv[3]);
  int output_cols = atol(argv[4]);
  // printf("%d %d", output_rows, output_cols);
  int N = (output_rows - 1) * stride + kernel_size;
  int M = (output_cols - 1) * stride + kernel_size;

 uint32_t in_dimensions = (C_i*N*M);
 uint32_t filter_dimensions = (C_i * C_o * kernel_size* kernel_size);
 uint32_t out_dimensions = (C_o * output_rows * output_cols);

  float * input_dc = alloc(in_dimensions);
  float * filter_dc = alloc(filter_dimensions);
  float * out_dc = alloc(out_dimensions);

//init 
init(input_dc, in_dimensions);
init(filter_dc, in_dimensions);

  #if PARALLEL
  uint32_t num_threads = atoi(std::getenv("OMP_NUM_THREADS"));
  #endif




  unsigned long long sum=ULLONG_MAX, sum_pool = ULLONG_MAX;
  volatile  unsigned long long sum_fused = ULLONG_MAX,
                               sum_conv = ULLONG_MAX;
  uint64_t t0, t1;
  // #if COMB == 1
  {
    // Initialize Outputs to 0
    memset(out_dc, 0, out_dimensions*sizeof(float));


    //3x3 unfused
    // copy_torch2dc(a, 'i', in_dimensions, input_dc);
    // copy_torch2dc(weights,'f',filter_dimensions,filter_dc);
    #if(L)
      volatile float check_sum = rand()/(1.0*RAND_MAX);
      for(uint32_t i = 0; i < bomb_size; i++){
        check_sum += cache_bomb_array[i];
      }
    #endif
    for (int run = 0; run < RUNS; run++){
      // Copy Inputs to their flat buffers

      t0 = rdtsc();
      direct_convolution<stride, kernel_size, kernel_size >(C_i, C_o, N, M, input_dc, filter_dc, out_dc);
      // direct_convolution_pooling_aware<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc, filter_dc, out_intermediate_dc);
      t1 = rdtsc();
      REDUCE(sum,(t1 - t0));
    }
    direct_convolution<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc, filter_dc, out_dc);
    // assert(equals(out,'o', out_dimensions, out_dc, 1e-3)==1);
    // print_flops(conv_ops, sum, RUNS);
    print_cycles(sum, RUNS);


  }



  #if(VERBOSE)
    printf("%d %d\n", inter_correctness, correctness);
  #endif



  // #endif
  // printf("%.4f \t",(1.0*(sum+sum_pool)/sum_fused)*100.0);
  printf("\n");

  free(input_dc);
  free(filter_dc);
  free(out_dc);


}
