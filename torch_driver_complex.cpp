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

#define COMPLEX 2

#define H_TILE 0
#define POOLING 1
#include "src/direct_complex_convolution.h"
// #include "src/fused_conv_dw.h"
#include "src/torch_utils.h"

//Problem size
#include "config.h"

//Good Ol' Timing
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

#define print_flops(ops, time)                   \
    {                                            \
        printf("%.4lf\t", (ops) / (1.0 * time)); \
    }
#define print_cycles(time)               \
    {                                    \
        printf("%.0lf\t", 1.0 * (time)); \
    }

#define AVG(accum, trials, avg)       \
    {                                 \
        avg = (1.0 * accum) / trials; \
    }

#define MIN(a, b)            \
    {                        \
        a = (b < a) ? b : a; \
    }
#define ACCUM_time(a, b) \
    {                    \
        a += b;          \
    }
#define MEMORY_SIZES_LOG                                                                                                                          \
    {                                                                                                                                             \
        printf("Testing %d runs, clearing the upto the L%d cache of the output each time\n", RUNS, L);                                            \
        printf("WSS Size In_img : %.2f K/8K elements  dims: %u %u %u\n\
WSS Size In_filter 3x3: %.2f K/8K elements  dims: %u %u %u %u\n\
WSS Size Out_img 3x3 : %.2f K/8K elements  dims: %u %u %u\n\
WSS Size Out_img pool : %.2f K/8K elements  dims: %u %u %u\n\
",                                                                                                                                                \
               a.numel() / 1024.0, in_dimensions[1], in_dimensions[2], in_dimensions[3],                                                          \
               weights.numel() / 1024.0, filter_dimensions[0], filter_dimensions[1], filter_dimensions[2], filter_dimensions[3],                  \
               out_intermediate.numel() / 1024.0, out_intermediate_dimensions[1], out_intermediate_dimensions[2], out_intermediate_dimensions[3], \
               out.numel() / 1024.0, out_dimensions[1], out_dimensions[2], out_dimensions[3]);                                                    \
    }

#define LIMIT 1e-2

int main(int argc, char **argv)
{
    printf("%d \t %d\t ", BUFFER, PREFETCH);
    if (argc < 5)
    {
        printf("USAGE: torch_pool < 3x3 Input Channels> <3x3 Output Channels> <Output Height> <Output Width (multiple of 6) <logfilename>>\n");
        return 0;
    }

    // Setup Problem Size from command line variables
    int C_i = atoi(argv[1]);
    int C_o = atoi(argv[2]);

    // int C_o_1 = atoi(argv[3]);

    constexpr int kernel_size = config_kernel_size;
    constexpr int stride = config_stride;

    constexpr int channel_stride = 1;


    constexpr uint32_t W_ob  =  6;
    constexpr uint32_t C_ob  = 16;
    constexpr uint32_t C_ib  = 16;
    constexpr int C_o_1 = C_ib;
    // constexpr uint32_t W_ob_dw W_ob
    // constexpr uint32_t W_ob_pool 3
    // constexpr uint32_t W_ob_g W_ob

    int output_rows = atol(argv[3]);
    int output_cols = atol(argv[4]);
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
    torch::Tensor a_real = torch::ones(C_i * N * M).reshape({1, C_i, N, M});
    torch::Tensor a_imag = torch::ones(C_i * N * M).reshape({1, C_i, N, M});
  
    torch::Tensor test_weights_real = torch::ones(C_o * C_i * kernel_size * kernel_size).reshape({C_o, C_i, kernel_size, kernel_size});
    torch::Tensor test_weights_imag = torch::ones(C_o * C_i * kernel_size * kernel_size).reshape({C_o, C_i, kernel_size, kernel_size});
    a_real = torch::mul(a_real, 2.0);
    a_imag = torch::mul(a_imag, 1.0);

    test_weights_real = torch::mul(test_weights_real, 1.0 / (1.0 * kernel_size * kernel_size * C_i));
    test_weights_imag = torch::mul(test_weights_imag, 1.0 / (1.0 * kernel_size * kernel_size * C_i));

    test_weights_real = torch::mul(test_weights_real, 3.0);
    test_weights_imag = torch::mul(test_weights_imag, 4.0);


    std::vector<std::vector<uint64_t>> implementations;

    //Create PyTorch Convolution layers
    //set weights to generated values
    auto conv_3x3_real = torch::nn::Conv2d(torch::nn::Conv2dOptions(C_i, C_o, kernel_size).stride(stride).padding(0).bias(false));

    conv_3x3_real->weight = test_weights_real;

    auto conv_3x3_imag = torch::nn::Conv2d(torch::nn::Conv2dOptions(C_i, C_o, kernel_size).stride(stride).padding(0).bias(false));

    conv_3x3_imag->weight = test_weights_imag;

    //Run Inference with LibTorch
    unsigned long long t0, t1;
    unsigned long long sum_pytorch = ULLONG_MAX;
    torch::Tensor out_real, out_imag;
    float avg;
    std::vector<uint64_t> pytorch_timing;
    for (uint32_t r = 0; r < RUNS/10; r++)
    {
        t0 = rdtsc();
        out_real = conv_3x3_real(a_real);
        out_real += conv_3x3_imag(a_imag);
        out_imag = conv_3x3_real(a_imag);
        out_imag += conv_3x3_imag(a_real);
        t1 = rdtsc();
        MIN(sum_pytorch, (t1 - t0));
        pytorch_timing.push_back((t1 - t0));
    }
    implementations.push_back(pytorch_timing);
    print_cycles(sum_pytorch);
    fflush(0);
    // std::cout << out_real<< std::endl;
    // std::cout << out_imag << std::endl;
    //Direct Convolution Setup
    // Copy layer weights to temporaries
    //concatenate real and imaginary values
    auto combined_input = torch::cat({a_real, a_imag}, 0);
    auto combined_weight = torch::cat({test_weights_real, test_weights_imag}, 0);
    auto combined_output = torch::cat({out_real, out_imag},0);

    std::vector<uint32_t> in_dimensions;
    std::vector<uint32_t> filter_dimensions;
    std::vector<uint32_t> out_intermediate_dimensions;

    std::vector<uint32_t> intermediate_block_dimensions;
    float *input_dc = alloc_dc(combined_input, in_dimensions);
    float *filter_dc = alloc_dc(combined_weight, filter_dimensions);
    float *out_intermediate_dc = alloc_dc(combined_output, out_intermediate_dimensions);
    // std::cout <<"output: "<< out_intermediate_dimensions << std::endl
    //           << "input: " << in_dimensions << std::endl
    //           << "filter: "<<filter_dimensions << std::endl;
#if PARALLEL
    uint32_t num_threads = atoi(std::getenv("OMP_NUM_THREADS"));
#endif

// #if (BUFFER == 0)
//     float *out_intermediate_buffer;
//     if (C_i > 16)
//     {
// #if PARALLEL
// #if H_TILE == 1
//         int ret = posix_memalign((void **)&out_intermediate_buffer, 4096, 3 * out_intermediate_dimensions[3] * C_ob * sizeof(float) * (num_threads));
// #else
//         int ret = posix_memalign((void **)&out_intermediate_buffer, 4096, out_intermediate_dimensions[2] * out_intermediate_dimensions[3] * C_ob * sizeof(float) * (num_threads));
// #endif
// #else
//         int ret = posix_memalign((void **)&out_intermediate_buffer, 4096, out_intermediate_dimensions[2] * out_intermediate_dimensions[3] * C_ob * sizeof(float));
// #endif
//     }
//     else
//     {
//         printf("Output channels must be >= 32");
//         return 0;
//     }
// #endif


    unsigned long long sum = ULLONG_MAX, sum_pool = ULLONG_MAX;
    volatile unsigned long long sum_fused = ULLONG_MAX,
                                sum_conv = ULLONG_MAX;
    std::vector<uint64_t> unfused_timing;

    {
        // Initialize Outputs to 0

        //3x3 unfused
        copy_torch2dc<C_ob/COMPLEX, C_ib/COMPLEX>(combined_input, 'c', in_dimensions, input_dc);
        copy_torch2dc<C_ob/COMPLEX, C_ib/COMPLEX>(combined_weight, 'k', filter_dimensions, filter_dc);
        memset(out_intermediate_dc, 0.0, combined_output.numel() * sizeof(float));
        direct_convolution<W_ob, C_ob, C_ib, stride, channel_stride, kernel_size, kernel_size>(C_i, C_o, 1, N, M, input_dc, filter_dc, out_intermediate_dc);
        // bool check = check_complex_eqivalence<C_ob/COMPLEX, C_ib/COMPLEX>(combined_output, 'o', out_intermediate_dimensions, out_intermediate_dc, LIMIT);
        // assert(check == 1);
        // printf("op : %f", out_intermediate_dcop)
        for (int run = 0; run < RUNS; run++)
        {
            // Copy Inputs to their flat buffers
            t0 = rdtsc();
            direct_convolution<W_ob, C_ob, C_ib, stride, channel_stride, kernel_size, kernel_size>(C_i, C_o, 1,  N, M, input_dc, filter_dc, out_intermediate_dc);
            t1 = rdtsc();
            MIN(sum_pool, (t1 - t0));
            unfused_timing.push_back((t1 - t0));
        }
     
        // print_cycles(sum);

        print_cycles(sum_pool);

        printf("%2.3f %d \n", (100.0*sum_pytorch)/(sum_pool), 0);

        fflush(0);
    }

//     implementations.push_back(unfused_timing);
//     const int NUM_IMPLEMENTATIONS = 4;

//     printf("\n");

//     //output log file

//     std::string file;
//     if (argc == 6)
//     {
//         file = argv[5];
//     }
//     else
//     {
//         file = "log.txt";
//     }

//     write_results(file, implementations);

    free(input_dc);
    free(filter_dc);
    free(out_intermediate_dc);
}
