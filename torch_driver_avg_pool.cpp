#include <torch/torch.h>
#include<math.h>
#include<assert.h>
#include<omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include<climits>
// Pooling driver
//average
#define GEMM 0
#define L 3
#define RUNS 1
#define VERBOSE 0
#define FUSION 1
#define STRIDE 1
#define PARALLEL 0
#define COMB 1
#ifndef BUFFER
  #define BUFFER 0
#endif
#define PREFETCH 1
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
  printf("%lf\t", (ops)/(1.0 * time));\
}
#define print_cycles(time,  trials){\
  printf("%lf\t", 1.0*(time));\
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
  printf("%d \t %d\t %f ", BUFFER,PREFETCH, weight);
  if(argc!=5)
  {
    printf("USAGE: torch_pool < 3x3 Input Channels> <3x3 Output Channels> <Output Height> <Output Width (multiple of 6)>\n");
    return 0;
  }

  // Setup Problem Size from command line variables
  int C_i = atoi(argv[1]);
  int C_o = atoi(argv[2]);
  // int C_o_1 = atoi(argv[3]);

  constexpr int kernel_size = 3;
  constexpr int stride = 1;
  int output_rows = atol(argv[3]);
  int output_cols = atol(argv[4]);
  // printf("%d %d", output_rows, output_cols);
  int N = (output_rows - 1) * stride + kernel_size;
  int M = (output_cols - 1) * stride + kernel_size;
  // printf("%d %d", N, M);

  if(atol(argv[4])%6 != 0 || atol(argv[4]) < 12){
    printf(" Please check that Output Width is a multiple of 6 >= 12\n");
    return 0;
  }


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

  auto pool = torch::nn::AvgPool2d( torch::nn::AvgPool2dOptions(3).
                                                stride(2).
                                                padding(0));

  //Run Inference
  unsigned long long t0, t1;
  unsigned long long sum_pytorch = ULLONG_MAX;
  torch::Tensor out_intermediate, out;
  for(uint32_t r = 0; r < 10; r++)
  {
    t0 = rdtsc();
    out_intermediate = conv_3x3(a);
    out = pool(out_intermediate);
    t1 = rdtsc();
    MIN(sum_pytorch, (t1 - t0));


  }

  uint64_t conv_ops = out_intermediate.numel() * (kernel_size*kernel_size*C_i * 2.0);
  uint64_t pool_ops = out.numel()              * (3*3);


  // print_flops(conv_ops+pool_ops, sum_pytorch, (10));
  // print_cycles(sum_pytorch, 10);


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
  std::vector<uint32_t> out_intermediate_dimensions;
  std::vector<uint32_t> out_dimensions;

  std::vector<uint32_t> intermediate_block_dimensions;
  float * input_dc = alloc_dc(a, in_dimensions);
  float * filter_dc = alloc_dc(weights, filter_dimensions);
  float * out_intermediate_dc = alloc_dc(out_intermediate, out_intermediate_dimensions);
  float * out_dc = alloc_dc(out, out_dimensions);


  #if PARALLEL
  uint32_t num_threads = atoi(std::getenv("OMP_NUM_THREADS"));
  #endif

  #if(BUFFER==0)
  float * out_intermediate_buffer;
  if(C_i > 16)
  {
    #if PARALLEL
    int ret = posix_memalign((void**)&out_intermediate_buffer, 4096, out_intermediate_dimensions[2]*out_intermediate_dimensions[3]*C_ob*sizeof(float)*(num_threads));
    #else
    int ret = posix_memalign((void**)&out_intermediate_buffer, 4096, out_intermediate_dimensions[2]*out_intermediate_dimensions[3]*C_ob*sizeof(float));
    #endif
  }
  else{
    printf("Output channels must be >= 32");
    return 0;
    printf("6x16 intermediate size\n");
    #if PARALLEL
    int ret = posix_memalign((void**)&out_intermediate_buffer, 4096, W_ob*C_ob*sizeof(float)*(num_threads));
    #else
    int ret = posix_memalign((void**)&out_intermediate_buffer, 4096, W_ob*C_ob*sizeof(float));
    #endif
  }
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
    memset(out_intermediate_dc, 0, out_intermediate.numel()*sizeof(float));
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
      direct_convolution_pooling_aware<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc, filter_dc, out_intermediate_dc);
      t1 = rdtsc();
      MIN(sum,(t1 - t0));
      t0 = rdtsc();
      pooling(C_o, out_intermediate_dimensions[2]-1, out_intermediate_dimensions[3] ,out_intermediate_dc, out_dc);
      t1 = rdtsc();
      MIN(sum_pool,(t1 - t0));


       #if(L)
      { volatile float check_sum = rand()/(1.0*RAND_MAX);
         for(uint32_t i = 0; i < bomb_size; i++){
           check_sum += cache_bomb_array[i];
         }
       }
       #endif
    }
    assert(check_eqivalence(out,'o', out_dimensions, out_dc, 1e-3)==1);
    print_flops(effective_conv_ops, sum, RUNS);
    print_cycles(sum, RUNS);

    print_flops(pool_ops, sum_pool, RUNS);
    print_cycles(sum_pool, RUNS);

    print_flops(effective_conv_ops+pool_ops, sum+sum_pool, RUNS);
    print_cycles(sum+sum_pool, RUNS);


  }
// #else
  {
    memset(out_intermediate_dc, 0, out_intermediate.numel()*sizeof(float));
    memset(out_dc, 0, out.numel()*sizeof(float));


    // fused
    copy_torch2dc(a, 'i', in_dimensions, input_dc);
    copy_torch2dc(weights,'f',filter_dimensions,filter_dc);
    #if(L)
    {
      volatile float check_sum = rand()/(1.0*RAND_MAX);
      for(uint32_t i = 0; i < bomb_size; i++){
        check_sum += cache_bomb_array[i];
      }
    }
    #endif
    for(int r = 0; r < RUNS; r++){

      t0 = rdtsc();
        #if BUFFER==1
          // printf("missing final update\n");
          fused_pooling_direct_convolution_not_buffered<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc,filter_dc, out_intermediate_dc, out_dc);
        #elif BUFFER==2
          // printf("full\n");
          fused_pooling_direct_convolution_complete<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc,filter_dc, out_intermediate_dc, out_dc);
        #else
          // printf("buffered\n");
          fused_pooling_direct_convolution<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc,filter_dc, out_intermediate_buffer, out_dc);
        #endif
      t1 = rdtsc();
      MIN(sum_fused,(t1 - t0));


      #if(L)
      // __asm__ __volatile__ ("cache_bomb:");
        volatile float check_sum = rand()/(1.0*RAND_MAX);
        for(uint32_t i = 0; i < bomb_size; i++){
          check_sum += cache_bomb_array[i];
        }
    #endif
  }
  print_flops(effective_conv_ops+pool_ops, sum_fused, RUNS);
  print_cycles( sum_fused, RUNS);
  bool correctness = check_eqivalence(out, 'o', out_dimensions, out_dc, 1e-3);


  #if(VERBOSE)
    printf("%d %d\n", inter_correctness, correctness);
  #endif


  //Make sure all outputs match pytorch
  assert(correctness==1);

  }
  // #endif
  printf("%.4f\t %.4f \t", (1.0*sum/sum_fused)*100.0, (1.0*(sum+sum_pool)/sum_fused)*100.0);
  printf("\n");

  free(input_dc);
  free(filter_dc);
  free(out_dc);
  free(out_intermediate_dc);
  #if BUFFER==0
  free(out_intermediate_buffer);
  #endif

}
