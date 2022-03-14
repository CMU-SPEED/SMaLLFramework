#include <torch/torch.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>
#include <vector>
#include <string>
#include <fstream>

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

// Problem size
#include "config.h"
#include "src/torch_utils.h"

// #include "src/direct_convolution.h"
#include "src/naive_direct_convolution.h"
// #include "src/fused_conv_dw.h"



// Good Ol' Timing
static __inline__ unsigned long long rdtsc(void)
{
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc"
                       : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

// #define print_flops( ops,  time,  trials){\
//   printf("%lf\t", (ops)/(1.0 * time/trials));\
// }
// #define print_cycles(time,  trials){\
//   printf("%lf\t", 1.0*(time/trials));\
// }

#define print_flops(ops, time)               \
  {                                          \
    printf("%.4lf\t", (ops) / (1.0 * time)); \
  }
#define print_cycles(time)             \
  {                                    \
    printf("%.0lf, \t", 1.0 * (time)); \
  }

#define AVG(accum, trials, avg)   \
  {                               \
    avg = (1.0 * accum) / trials; \
  }

#define MIN(a, b)        \
  {                      \
    a = (b < a) ? b : a; \
  }
#define ACCUM_time(a, b) \
  {                      \
    a += b;              \
  }
#define MEMORY_SIZES_LOG                                                                                                                      \
  {                                                                                                                                           \
    printf("Testing %d runs, clearing the upto the L%d cache of the output each time\n", RUNS, L);                                            \
    printf("WSS Size In_img : %.2f K/8K elements  dims: %u %u %u\n\
WSS Size In_filter 3x3: %.2f K/8K elements  dims: %u %u %u %u\n\
WSS Size Out_img 3x3 : %.2f K/8K elements  dims: %u %u %u\n\
WSS Size Out_img pool : %.2f K/8K elements  dims: %u %u %u\n\
",                                                                                                                                            \
           a.numel() / 1024.0, in_dimensions[1], in_dimensions[2], in_dimensions[3],                                                          \
           weights.numel() / 1024.0, filter_dimensions[0], filter_dimensions[1], filter_dimensions[2], filter_dimensions[3],                  \
           out_intermediate.numel() / 1024.0, out_intermediate_dimensions[1], out_intermediate_dimensions[2], out_intermediate_dimensions[3], \
           out.numel() / 1024.0, out_dimensions[1], out_dimensions[2], out_dimensions[3]);                                                    \
  }

#define LIMIT 1e-2

int main(int argc, char **argv)
{
  // printf("%d \t %d\t ", BUFFER, PREFETCH);
  if (argc < 5)
  {
    printf("USAGE: torch_pool < 3x3 Input Channels> <3x3 Output Channels> <Output Height> <Output Width (multiple of 6) <logfilename>>\n");
    return 0;
  }

  constexpr uint32_t W_ob = 6;
  constexpr uint32_t C_ob = 12;
  constexpr uint32_t C_ib = 12;
  constexpr int C_o_1 = C_ib;
  // constexpr uint32_t W_ob_dw W_ob
  // constexpr uint32_t W_ob_pool 3
  // constexpr uint32_t W_ob_g W_ob

  constexpr uint32_t kernel_size = config_kernel_size;
  constexpr uint32_t stride = config_stride;

  constexpr uint32_t channel_stride = config_channel_stride;

  // Setup Problem Size from command line variables
  int C_i = atoi(argv[1]);
  int C_o = atoi(argv[2]);
  //atoi(argv[2]);

  // uint32_t C_o_1 = atoi(argv[3]);

  int output_rows = atol(argv[3]);
  int output_cols = atol(argv[4]);

  uint32_t naive_kernel_size = atoi(argv[5]);
  constexpr uint32_t naive_stride = config_stride;
  atoi(argv[6]);

  uint32_t naive_channel_stride = atoi(argv[7]);

  printf(" %d %d %d (compile)\n %d %d %d (runtime)\n", kernel_size, stride, channel_stride, naive_kernel_size, naive_stride, naive_channel_stride);
  // fflush(0);
  assert(kernel_size == naive_kernel_size);
  assert(stride == naive_stride);
  assert(channel_stride == naive_channel_stride);
  assert(!(C_i%C_ib)*(!(C_o%C_ob)));
  // printf("%d %d", output_rows, output_cols);
  int N = (output_rows - 1) * stride + kernel_size;
  int M = (output_cols - 1) * stride + kernel_size;
  // printf("%d %d", N, M);

  //   if(atol(argv[4])%6 != 0 || atol(argv[4]) < 12){
  //     printf(" Please check that Output Width is a multiple of 6 >= 12\n");
  //     return 0;
  //   }

  // Create and Initialize Pytorch tensors
  torch::manual_seed(1729);
  torch::Tensor a = torch::randn(C_i * N * M).reshape({1, C_i, N, M});
  a = torch::mul(a, 1.0);
  torch::Tensor test_weights = torch::randn(C_o * C_i * kernel_size * kernel_size).reshape({C_o, C_i, kernel_size, kernel_size});

  test_weights = torch::mul(test_weights, 1.0 / (1.0 * kernel_size * kernel_size * C_i));

  std::vector<std::vector<uint64_t>> implementations;

  // Create PyTorch Convolution layers
  // set weights to generated values
  auto conv_3x3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(C_i, C_o, kernel_size).stride(stride).padding(0).bias(false));

  conv_3x3->weight = test_weights;

  // Run Inference with LibTorch
  unsigned long long t0, t1;
  unsigned long long sum_pytorch = ULLONG_MAX;
  torch::Tensor out_intermediate, out;
  float avg;
  std::vector<uint64_t> pytorch_timing;
  for (uint32_t r = 0; r < RUNS / 10 + 1; r++)
  {
    t0 = rdtsc();
    out_intermediate = conv_3x3(a);
    t1 = rdtsc();
    MIN(sum_pytorch, (t1 - t0));
    pytorch_timing.push_back((t1 - t0));
  }
  implementations.push_back(pytorch_timing);
  print_cycles(sum_pytorch);
  fflush(0);

  // Direct Convolution Setup
  //  Copy layer weights to temporaries
  torch::Tensor weights = test_weights; // conv_3x3->weight;
  std::vector<uint32_t> in_dimensions;
  std::vector<uint32_t> filter_dimensions;
  std::vector<uint32_t> out_intermediate_dimensions;

  std::vector<uint32_t> intermediate_block_dimensions;
  float *input_dc = alloc_dc(a, in_dimensions);
  float *filter_dc = alloc_dc(weights, filter_dimensions);
  float *out_intermediate_dc = alloc_dc(out_intermediate, out_intermediate_dimensions);

#if PARALLEL
  uint32_t num_threads = atoi(std::getenv("OMP_NUM_THREADS"));
#endif

#if (BUFFER == 0)
  float *out_intermediate_buffer;
  if (C_i > 16)
  {
#if PARALLEL
#if H_TILE == 1
    int ret = posix_memalign((void **)&out_intermediate_buffer, 4096, 3 * out_intermediate_dimensions[3] * C_ob * sizeof(float) * (num_threads));
#else
    int ret = posix_memalign((void **)&out_intermediate_buffer, 4096, out_intermediate_dimensions[2] * out_intermediate_dimensions[3] * C_ob * sizeof(float) * (num_threads));
#endif
#else
    int ret = posix_memalign((void **)&out_intermediate_buffer, 4096, out_intermediate_dimensions[2] * out_intermediate_dimensions[3] * C_ob * sizeof(float));
#endif
  }
  else
  {
    printf("Output channels must be >= 32");
    return 0;
  }
#endif

  unsigned long long sum = ULLONG_MAX, sum_pool = ULLONG_MAX;
  volatile unsigned long long sum_fused = ULLONG_MAX,
                              sum_conv = ULLONG_MAX;
  std::vector<uint64_t> unfused_timing;

  {
    // Initialize Outputs to 0

    // 3x3 unfused
    copy_torch2dc<C_ob, C_ib>(a, 'i', in_dimensions, input_dc);
    copy_torch2dc<C_ob,  C_ib>(weights, 'f', filter_dimensions, filter_dc);
    printf("init complete\n");
    fflush(0);
    sum = ULLONG_MAX;
    memset(out_intermediate_dc, 0.0, out_intermediate.numel() * sizeof(float));
    printf("init complete\n");
    fflush(0);
    direct_convolution_naive<W_ob, C_ob, C_ib, naive_stride, 'v','c'>(naive_channel_stride, naive_kernel_size, naive_kernel_size, C_i, C_o, 1, N, M, input_dc, filter_dc, out_intermediate_dc);
    // direct_convolution<W_ob, C_ob, C_o, stride, channel_stride, naive_kernel_size, naive_kernel_size>(C_i, C_o, C_o, N, M, input_dc, filter_dc, out_intermediate_dc);
    bool check = check_eqivalence<C_ob, C_ib>(out_intermediate, 'o', out_intermediate_dimensions, out_intermediate_dc, LIMIT);
    assert(check == 1);

    // for (int run = 0; run < RUNS; run++)
    // {
    //   // Copy Inputs to their flat buffers
    //   t0 = rdtsc();
    //   direct_convolution_naive<W_ob, C_ob, C_ib, naive_stride, 'c'>(naive_channel_stride, naive_kernel_size, naive_kernel_size, C_i, C_o, 1, N, M, input_dc, filter_dc, out_intermediate_dc);
    //   // direct_convolution_naive<W_ob, C_ob, C_ib, naive_stride, 'c'>(naive_channel_stride, naive_kernel_size, naive_kernel_size, C_i, C_o, 1, N, M, input_dc, filter_dc, out_intermediate_dc);
    //   t1 = rdtsc();
    //   MIN(sum, (t1 - t0));
    //   unfused_timing.push_back((t1 - t0));
    // }

    // print_cycles(sum);

    // printf("%2.3f \n", (100.0 * sum) / (sum_pool));

    fflush(0);
  }



  free(input_dc);
  free(filter_dc);
  free(out_intermediate_dc);
}
