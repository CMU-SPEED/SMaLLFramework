#include <torch/torch.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>
// Pooling driver

#define GEMM 0
#define L 0
#define RUNS 1000
#define VERBOSE 0
#define FUSION 1
#define STRIDE 1
#define PARALLEL 1
#define COMB 0
#ifndef BUFFER
  #define BUFFER 0
#endif
#define PREFETCH 1

#define H_TILE 0
#define POOLING 1
#include "src/direct_convolution.h"
#include "src/fused_conv_pooling.h"
#include "src/utils.h"
//Good Ol' Timing
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

// #define print_flops( ops,  time,  trials){\
//   printf("%lf\t", (ops)/(1.0 * time/trials));\
// }
// #define print_cycles(time,  trials){\
//   printf("%lf\t", 1.0*(time/trials));\
// }

#define print_flops( ops,  time,  trials){\
  printf("%.4lf\t", (ops)/(1.0 * time));\
}
#define print_cycles(time,  trials){\
  printf("%.0lf\t", 1.0*(time));\
}

#define MIN(a, b){\
  a = (b < a)?b:a;\
}
#define MEMORY_SIZES_LOG  {\
  printf("Testing %d runs, clearing the upto the L%d cache of the output each time\n", RUNS, L);\
  printf("WSS Size In_img : %.2f K/8K elements  dims: %u %u %u\n\
WSS Size In_filter 3x3: %.2f K/8K elements  dims: %u %u %u %u\n\
WSS Size Out_img 3x3 : %.2f K/8K elements  dims: %u %u %u\n\
WSS Size Out_img pool : %.2f K/8K elements  dims: %u %u %u\n\
",  a.numel()/1024.0, in_dimensions[1], in_dimensions[2], in_dimensions[3],\
       weights.numel()/1024.0, filter_dimensions[0], filter_dimensions[1], filter_dimensions[2], filter_dimensions[3],\
       out_intermediate.numel()/1024.0, out_intermediate_dimensions[1], out_intermediate_dimensions[2], out_intermediate_dimensions[3],\
       out.numel()/1024.0, out_dimensions[1], out_dimensions[2], out_dimensions[3]\
     );\
  }

int main(int argc, char ** argv)
{
  printf("%d \t %d\t ", BUFFER,PREFETCH);
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
  int output_rows = atol(argv[3]);
  int output_cols = atol(argv[4]);
  // printf("%d %d", output_rows, output_cols);
  int N = (output_rows - 1) * stride + kernel_size;
  int M = (output_cols - 1) * stride + kernel_size;
  // printf("%d %d", N, M);




  // Create and Initialize Pytorch tensors
  torch::manual_seed(1729);
  torch::Tensor a = torch::randn(C_i*N*M).reshape({1,C_i,N, M});
  torch::Tensor test_weights =  torch::randn(C_o*C_i*kernel_size*kernel_size).reshape({C_o,C_i,kernel_size,kernel_size});



  //Create PyTorch Convolution layers
  //set weights to generated values
  auto conv_3x3 = torch::nn::Conv2d( torch::nn::Conv2dOptions(C_i, C_o, kernel_size).
                                                stride(stride).
                                                padding(0).
                                                bias(false));

  conv_3x3->weight = test_weights;

  auto pool = torch::nn::MaxPool2d( torch::nn::MaxPool2dOptions(3).
                                                stride(2).
                                                padding(0));

  //Run Inference
  unsigned long long t0, t1;
  unsigned long long sum_pytorch = ULLONG_MAX;
  torch::Tensor out_intermediate, out;
  for(uint32_t r = 0; r < 100; r++)
  {
    t0 = rdtsc();
    out = conv_3x3(a);
    t1 = rdtsc();
    MIN(sum_pytorch, (t1 - t0));


  }

  uint64_t conv_ops = out.numel() * (kernel_size*kernel_size*C_i * 2.0);


  print_flops(conv_ops, sum_pytorch, (10));
  print_cycles(sum_pytorch, 10);


  uint64_t effective_conv_h = (out.size(2) - 1) * 2 + 3;
  uint64_t effective_conv_w = (out.size(3) - 1) * 2 + 3;

  uint64_t effective_conv_ops = effective_conv_h*effective_conv_w*C_o
                                                *
                                (kernel_size*kernel_size*C_i*2.0);
  //Direct Convolution Setup
  // Copy layer weights to temporaries
  torch::Tensor weights = test_weights;//conv_3x3->weight;

  std::vector<uint32_t> in_dimensions;
  std::vector<uint32_t> filter_dimensions;
  std::vector<uint32_t> out_dimensions;

  std::vector<uint32_t> intermediate_block_dimensions;
  float * input_dc = alloc_dc(a, in_dimensions);
  float * filter_dc = alloc_dc(weights, filter_dimensions);
  float * out_dc = alloc_dc(out, out_dimensions);


  #if PARALLEL
  uint32_t num_threads = atoi(std::getenv("OMP_NUM_THREADS"));
  #endif

  #if(VERBOSE)
  MEMORY_SIZES_LOG;
  #endif



  #if(L)
  uint32_t bomb_size = 16*1024*1024/4 *(L==3) + 512*1024/4 *(L==2) + 32*1024/4*(L==1);
  torch::Tensor cache_bomb= torch::randn(bomb_size); //L3
  float * cache_bomb_array = cache_bomb.data_ptr<float>();
  #endif



  unsigned long long sum=ULLONG_MAX, sum_pool = ULLONG_MAX;
  volatile  unsigned long long sum_fused = ULLONG_MAX,
                               sum_conv = ULLONG_MAX;
  // #if COMB == 1
  {
    // Initialize Outputs to 0
    memset(out_dc, 0, out.numel()*sizeof(float));


    //3x3 unfused
    copy_torch2dc(a, 'i', in_dimensions, input_dc);
    copy_torch2dc(weights,'f',filter_dimensions,filter_dc);
    #if(L)
      volatile float check_sum = rand()/(1.0*RAND_MAX);
      for(uint32_t i = 0; i < bomb_size; i++){
        check_sum += cache_bomb_array[i];
      }
    #endif
    for (int run = 0; run < RUNS; run++){
      // Copy Inputs to their flat buffers

      t0 = rdtsc();
      direct_convolution<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc, filter_dc, out_dc);
      // direct_convolution_pooling_aware<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc, filter_dc, out_intermediate_dc);
      t1 = rdtsc();
      MIN(sum,(t1 - t0));

       #if(L)
      { volatile float check_sum = rand()/(1.0*RAND_MAX);
         for(uint32_t i = 0; i < bomb_size; i++){
           check_sum += cache_bomb_array[i];
         }
       }
       #endif
    }
    assert(check_eqivalence(out,'o', out_dimensions, out_dc, 1e-3)==1);
    print_flops(conv_ops, sum, RUNS);
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
