#include <torch/torch.h>
#include<math.h>
#include<assert.h>
#include<omp.h>
#include <stdio.h>
#include <stdlib.h>
// Pooling driver

#define GEMM 0
#define L 0
#define RUNS 1000
#define VERBOSE 0
#define FUSION 1
#define STRIDE 1
#define PARALLEL 1

#ifndef BUFFER
  #define BUFFER 2
#endif

#include "direct_convolution.h"
#include "fused_conv_pooling.h"
#include "utils.h"
//Good Ol' Timing
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


int main(int argc, char ** argv)
{
  printf("%d\t", BUFFER);
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
  if(atol(argv[4])%6 != 0 || atol(argv[4]) < 12){
    printf(" Please check that Output Width is a multiple of 6 >= 12\n");
    return 0;
  }


  // Create and Initialize Pytorch tensors
  torch::manual_seed(1729);
  torch::Tensor a = torch::randn(C_i*N*M).reshape({1,C_i,N, M});
  // a = torch::add(a, 0.1);
  // a = torch::mul(a, 0.1);
  torch::Tensor test_weights =  torch::randn(C_o*C_i*kernel_size*kernel_size).reshape({C_o,C_i,kernel_size,kernel_size});
  // test_weights = torch::mul(test_weights,1.0/9.0);
  // float *array;


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
  uint32_t t0, t1;
  uint64_t sum_pytorch = 1;
  torch::Tensor out_intermediate, out;
  for(uint32_t r = 0; r < RUNS/10; r++)
  {
    t0 = rdtsc();
    out_intermediate = conv_3x3(a);
    out = pool(out_intermediate);
    t1 = rdtsc();
    sum_pytorch += (t1 - t0);


  }

  printf("\t%lf\t%lf",
                      (
                        2.0*out_intermediate.numel()*(kernel_size*kernel_size*C_i)
                         + (out.numel()*(9))
                      )
                    /(sum_pytorch*1.0/(RUNS/10)),

                    (sum_pytorch*1.0/(RUNS/10))

                  );

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

  int m=out_intermediate_dimensions[2]*out_intermediate_dimensions[3], n=out_intermediate_dimensions[1], k = kernel_size*kernel_size*C_i;

  #if PARALLEL
  uint32_t num_threads = atoi(std::getenv("OMP_NUM_THREADS"));
  // printf("%d\n",num_threads);
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
    printf("6x16 intermediate size\n");
    #if PARALLEL
    int ret = posix_memalign((void**)&out_intermediate_buffer, 4096, W_ob*C_ob*sizeof(float)*(num_threads));
    #else
    int ret = posix_memalign((void**)&out_intermediate_buffer, 4096, W_ob*C_ob*sizeof(float));
    #endif

  }
  #endif
  #if(VERBOSE)
  printf("Testing %d runs, clearing the upto the L%d cache of the output each time\n", RUNS, L);
  printf("WSS Size In_img : %.2f K/8K elements  dims: %u %u %u\n\
WSS Size In_filter 3x3: %.2f K/8K elements  dims: %u %u %u %u\n\
WSS Size Out_img 3x3 : %.2f K/8K elements  dims: %u %u %u\n\
WSS Size Out_img pool : %.2f K/8K elements  dims: %u %u %u\n\
",  a.numel()/1024.0, in_dimensions[1], in_dimensions[2], in_dimensions[3],
       weights.numel()/1024.0, filter_dimensions[0], filter_dimensions[1], filter_dimensions[2], filter_dimensions[3],
       out_intermediate.numel()/1024.0, out_intermediate_dimensions[1], out_intermediate_dimensions[2], out_intermediate_dimensions[3],
       out.numel()/1024.0, out_dimensions[1], out_dimensions[2], out_dimensions[3]
                                                           );
  #endif



  #if(L)
  uint32_t bomb_size = 16*1024*1024/4 *(L==3) + 512*1024/4 *(L==2) + 32*1024/4*(L==1);
  torch::Tensor cache_bomb= torch::randn(bomb_size); //L3
  float * cache_bomb_array = cache_bomb.data_ptr<float>();
  #endif


  // Initialize Outputs to 0
  for (int i = 0; i != out_dimensions[2] * out_dimensions[3] * out_dimensions[1] ; ++i){
    out_dc[i] = 0.0;
    int j  = 1;
  }

  uint64_t sum=0, sum_pool = 0;
  //
  // 3x3 unfused
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
    // direct_convolution<stride*C_ob,kernel_size, kernel_size >(C_i, C_o, kernel_size, kernel_size, N, M, 1, input_dc, filter_dc, out_intermediate_dc);
    t1 = rdtsc();
    sum += (t1 - t0);
    t0 = rdtsc();
    pooling(C_o, out_intermediate_dimensions[2], out_intermediate_dimensions[3] ,out_intermediate_dc, out_dc);
    t1 = rdtsc();
    sum_pool += (t1 - t0);
     #if(L)
    { volatile float check_sum = rand()/(1.0*RAND_MAX);
       for(uint32_t i = 0; i < bomb_size; i++){
         check_sum += cache_bomb_array[i];
       }
     }
     #endif
  }

  // __volatile__ uint32_t c = 0;
  // for(c = 0; c < out.numel(); c++){
  //   out_dc[c] = 0.0;
  // }
  // // __volatile__
  // for(c = 0; c < out_intermediate.numel(); c++){
  //   out_intermediate_dc[c] = 0.0;
  // }
  memset(out_intermediate_dc, 0, out_intermediate.numel()*sizeof(float));
  memset(out_dc, 0, out.numel()*sizeof(float));
  assert(fabs(out_intermediate_dc[rand()%out_intermediate.numel()] - 0.0) < 1e-5);

  // %lf \t %lf\
  // \t %lf \t%lf
  printf("\t L%d \t %d\t%d\t %lf \t%lf\t%lf \t%lf\t%lf \t%lf\t", L, kernel_size,  C_i,
                          (
                          2.0*m*n*(k)
                          // +
                          //   out_dimensions[2]*out_dimensions[3]* out_dimensions[1]*9
                          )/((float)((sum)/(1.0*RUNS))), (float)((sum)/(1.0*RUNS)),
                          (
                          // 2.0*m*n*k
                          // +
                          out_dimensions[2]*out_dimensions[3]* out_dimensions[1]*9
                          )/((float)(sum_pool)/(1.0*RUNS)), (float)((sum_pool)/(1.0*RUNS)),
                          (
                          2.0*m*n*k
                          +
                          out_dimensions[2]*out_dimensions[3]* out_dimensions[1]*9
                        )/((float)((sum + sum_pool)/(1.0*RUNS))), (float)((sum+sum_pool)/(1.0*RUNS)));

  double torch_ops = 2.0*out_intermediate.numel()*(kernel_size*kernel_size*C_i) + (out.numel()*(9));
  double cpp_ops =   (2.0*m*n*k + out_dimensions[2]*out_dimensions[3]* out_dimensions[1]*9);
  assert(torch_ops == cpp_ops);
  uint64_t volatile sum_fused = 0;
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
    #if PARALLEL
      #if BUFFER==1
        // printf("missing final update\n");
        parallel_fused_pooling_direct_convolution_not_buffered<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc,filter_dc, out_intermediate_dc, out_dc);
      #elif BUFFER==2
        // printf("full\n");
        parallel_fused_pooling_direct_convolution_complete<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc,filter_dc, out_intermediate_dc, out_dc);
      #else
        // printf("buffered\n");
        parallel_fused_pooling_direct_convolution<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc,filter_dc, out_intermediate_buffer, out_dc);
      #endif
    #else
    fused_pooling_direct_convolution<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc,filter_dc, out_intermediate_buffer, out_dc);
    #endif

    t1 = rdtsc();
    sum_fused += (t1 - t0);

    #if(L)
    // __asm__ __volatile__ ("cache_bomb:");
      volatile float check_sum = rand()/(1.0*RAND_MAX);
      for(uint32_t i = 0; i < bomb_size; i++){
        check_sum += cache_bomb_array[i];
      }
    #endif
  }

  printf("%lf\t %lf\t", ( 2.0*m*n*(k)
                          + out_dimensions[2]*out_dimensions[3]* out_dimensions[1]*9
                        )/((sum_fused)/(1.0*RUNS)), ((sum_fused)/(1.0*RUNS)));
  assert((2.0*m*n*k + out_dimensions[2]*out_dimensions[3]* out_dimensions[1]*9) == cpp_ops);
  printf("\n");
  // printf("%lf %lf %lf\n", torch_ops, cpp_ops, 2.0*m*n*k + out_dimensions[2]*out_dimensions[3]* out_dimensions[1]*9);
  uint32_t block_size = out_intermediate_dimensions[2]*out_intermediate_dimensions[3]*C_ob;
  uint32_t offset = ((C_o - C_ob)/C_ob)*block_size; //last block

  // bool correctness = check_eqivalence(out_intermediate,'o', out_intermediate_dimensions, out_intermediate_dc, 1e-3);
  // printf("%d\n", correctness);
  bool inter_correctness = 1;
  #if BUFFER==2
    // inter_correctness = check_eqivalence(out_intermediate,'o', out_intermediate_dimensions, out_intermediate_dc, 1e-3);

  #endif
  //check_block_eqivalence(block_size, out_intermediate_buffer, out_intermediate_dc + offset,1e-3);
  bool correctness = 1;//check_eqivalence(out, 'o', out_dimensions, out_dc, 1e-3);


  #if(VERBOSE)
    printf("%d %d\n", inter_correctness, correctness);
  #endif
  // printf("%d %d\n", inter_correctness, correctness);

  //Make sure all outputs match pytorch
  assert(inter_correctness&correctness==1);





  free(input_dc);
  free(filter_dc);
  free(out_dc);
  free(out_intermediate_dc);
  #if BUFFER==0
  free(out_intermediate_buffer);
  #endif

}
