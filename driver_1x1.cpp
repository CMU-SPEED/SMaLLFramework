// #include <torch/torch.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>
#include <cstring>
#include <fstream>

//Sample Convolution driver

#define RUNS 1
#define PARALLEL 1

#define LIMIT 1e-4
#include "src/direct_convolution.h"
#include "src/fused_conv_bottleneck.h"
#include "src/utils.h"


int main(int argc, char **argv)
{
  if (argc < 6)
  {
    printf("USAGE: torch_pool < 3x3 Input Channels> <3x3 Output Channels> <Output Height> <Output Width (multiple of 6)> <1x1 Output Channels>\n");
    return 0;
  }

  // Setup Problem Size from command line variables
  int C_i = atoi(argv[1]);
  int C_o = atoi(argv[2]);
  // int C_o_1 = atoi(argv[3]);

  constexpr int kernel_size = 3;
  constexpr int stride = 1;

  constexpr int bottleneck_kernel_size = 1;
  constexpr int bottleneck_stride = 1;

  constexpr int padding = 0;
  (kernel_size - 1) / 2;
  int output_rows = atol(argv[3]);
  int output_cols = atol(argv[4]);
  // printf("%d %d", output_rows, output_cols);
  int N = (output_rows - 1) * stride + kernel_size;
  int M = (output_cols - 1) * stride + kernel_size;

  int C_o_bottleneck = atoi(argv[5]);

// SET UP
  uint32_t pool_H = (output_rows - bottleneck_kernel_size)/bottleneck_stride + 1;
  uint32_t pool_W = (output_cols - bottleneck_kernel_size) / bottleneck_stride + 1;

#if PARALLEL
  uint32_t num_threads = atoi(std::getenv("OMP_NUM_THREADS"));
#else
  uint32_t num_threads = 1;
#endif

  uint32_t in_dimensions = (C_i * N * M);
  uint32_t filter_dimensions = (C_i * C_o * kernel_size * kernel_size);
  uint32_t out_intermediate_dimensions = (C_o * output_rows * output_cols);
  uint32_t out_intermediate_buffer_dimensions = (C_ob * output_rows * output_cols * num_threads);
  uint32_t filter_bottleneck_dimensions = (C_o * bottleneck_kernel_size* bottleneck_kernel_size);
  uint32_t out_dimensions = (C_o * pool_H * pool_W);

  float *input_dc = alloc(in_dimensions);
  float *filter_dc = alloc(filter_dimensions);

  float *out_intermediate_dc = alloc(out_intermediate_dimensions);
  float *out_intermediate_buffer = alloc(out_intermediate_buffer_dimensions);

  float * filter_bottleneck_dc = alloc(filter_bottleneck_dimensions);

  float *out_dc = alloc(out_dimensions);
  float *out_fused_dc = alloc(out_dimensions);
//END SET UP

  //init
  init(input_dc, in_dimensions);
  init_norm(filter_dc, filter_dimensions, C_o);
  init_norm(filter_bottleneck_dc, filter_bottleneck_dimensions, 1);



  unsigned long long sum = TIME_ZERO;

  //set up log file to capture all the timing
  constexpr int NUM_IMPLEMENTATIONS = 0;
  uint64_t combined_timing[(NUM_IMPLEMENTATIONS + 1) * RUNS];
  uint64_t * timing = combined_timing;
  uint64_t t0, t1;

  // Initialize Outputs to 0
  memset(out_dc, 0, out_dimensions * sizeof(float));

  for (int run = 0; run < RUNS; run++)
  {
    // Copy Inputs to their flat buffers

    t0 = rdtsc();
    direct_convolution<stride, kernel_size, kernel_size>(C_i, C_o, N, M, input_dc, filter_dc, out_intermediate_dc);
    bottleneck<bottleneck_stride, bottleneck_kernel_size, bottleneck_kernel_size>(C_o, C_o_bottleneck,  output_rows, output_cols, out_intermediate_dc, filter_bottleneck_dc, out_fused_dc);
    t1 = rdtsc();
    REDUCE(sum, (t1 - t0));
    timing[run] = t1-t0;

  }
  print_cycles(sum, RUNS);

  // direct_convolution<stride, kernel_size, kernel_size>(C_i, C_o, N, M, input_dc, filter_dc, out_dc);
  direct_convolution<stride, kernel_size, kernel_size>(C_i, C_o, N, M, input_dc, filter_dc, out_intermediate_dc);
  direct_convolution<bottle_neck_stride, bottleneck_kernel_size, bottleneck_kernel_size>(_o, C_o_bottleneck, output_rows, output_cols, out_intermediate_dc, filter_bottleneck_dc, out_dc);
  bottleneck<bottleneck_stride>(C_o, output_rows, output_cols, out_intermediate_dc, filter_bottleneck_dc, out_fused_dc);
  // assert(equals(out,'o', out_dimensions, out_dc, 1e-3)==1);
  // print_flops(conv_ops, sum, RUNS);

  //Test Fused implementations

  // uint64_t *  fused_timing = combined_timing + RUNS;
  // for (int implementation = 0; implementation < NUM_IMPLEMENTATIONS; implementation++)
  // {
  //   // Initialize Outputs to 0
  //   memset(out_intermediate_buffer, 0, out_intermediate_buffer_dimensions * sizeof(float));
  //   memset(out_fused_dc, 0, out_dimensions * sizeof(float));

  //   //3x3 unfused

  //   uint64_t sum_pool = TIME_ZERO;
  //   for (int run = 0; run < RUNS; run++)
  //   {
  //     // Copy Inputs to their flat buffers

  //     switch (implementation)
  //     {
  //     case 2:
  //       t0 = rdtsc();
  //       pixel_block_fused_bottleneck<stride,
  //                                 kernel_size, kernel_size,
  //                                 bottleneck_stride, bottleneck_kernel_size,
  //                                 bottleneck_kernel_size>(
  //           C_i,
  //           C_o,
  //           N,
  //           M,
  //           input_dc,
  //           filter_dc,
  //           out_intermediate_buffer,
  //           filter_bottleneck_dc,
  //           out_fused_dc);
  //       t1 = rdtsc();
  //       break;
  //     case 3:
  //       t0 = rdtsc();
  //       channel_block_fused_bottleneck<stride,
  //                                   kernel_size, kernel_size,
  //                                   bottleneck_stride, bottleneck_kernel_size,
  //                                   bottleneck_kernel_size>(
  //           C_i,
  //           C_o,
  //           N,
  //           M,
  //           input_dc,
  //           filter_dc,
  //           out_intermediate_buffer,
  //           filter_bottleneck_dc,
  //           out_fused_dc);
  //       t1 = rdtsc();
  //       break;
  //     case 1:
  //       t0 = rdtsc();
  //       row_full_fused_bottleneck<stride,
  //                              kernel_size, kernel_size,
  //                              bottleneck_stride, bottleneck_kernel_size,
  //                              bottleneck_kernel_size>(
  //           C_i,
  //           C_o,
  //           N,
  //           M,
  //           input_dc,
  //           filter_dc,
  //           out_intermediate_buffer,
  //           filter_bottleneck_dc,
  //           out_fused_dc);
  //       t1 = rdtsc();
  //       break;
  //       case 0:
  //         t0 = rdtsc();
  //         row_full_fused_bottleneck<stride,
  //                                kernel_size, kernel_size,
  //                                bottleneck_stride, bottleneck_kernel_size,
  //                                bottleneck_kernel_size>(
  //             C_i,
  //             C_o,
  //             N,
  //             M,
  //             input_dc,
  //             filter_dc,
  //             out_intermediate_buffer,
  //              filter_bottleneck_dc,
  //             out_fused_dc);
  //         t1 = rdtsc();
  //         break;
  //     }
  //     REDUCE(sum_pool, (t1 - t0));
  //     fused_timing[implementation*RUNS +  run] = (t1 - t0);
  //   }
  //   printf("%d\t", implementation);
  //   assert(equals(out_dimensions, out_dc, out_fused_dc, LIMIT) == 1);
  //   print_cycles(sum_pool, RUNS);
  // }



  printf("\n");

  //write detailed timing to stderr
  write_results<NUM_IMPLEMENTATIONS, RUNS>(combined_timing);


  free(input_dc);
  free(filter_dc);
  free(out_intermediate_dc);
  free(out_intermediate_buffer);
  free(filter_bottleneck_dc);
  free(out_dc);
  free(out_fused_dc);

}
