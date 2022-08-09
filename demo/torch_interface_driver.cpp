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


#include "config.h"

#if uarch == ZEN2
#include "src/kernels/zen2/params.h"

#elif uarch == REF
#include "src/kernels/reference/params.h"
#endif

#include "src/torch_utils.h"
#include "src/interface.h"

#include "test/interface.h"

// Problem size

// Timing Utils
static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc"
    : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}


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

#define LIMIT 1e-2

bool equals(uint32_t numel, float *unfused, float *fused, float tolerance = 1e-8)
{
    bool check = 1;
    float *unfused_ptr = unfused;
    float *fused_ptr = fused;

    for (uint32_t i = 0; i < numel; i++)
    {
        float diff = *(fused_ptr) - *(unfused_ptr);
        // printf("%d %.4f %.4f %.4f\n", i, *(fused_ptr), *(unfused_ptr), diff);

        if (fabs(diff) > tolerance)
        {
            printf("%d %.4f %.4f %.4f\n", i, *(fused_ptr), *(unfused_ptr), diff);
            check = 0;
        }
        unfused_ptr++;
        fused_ptr++;
    }
    return check;
}

#define CONV 0
#define PARTIAL_CONV 1 // under development
#define DW_CONV 2      // under development
#define GROUP_CONV 3   // under development
#define POOL 4
#define RELU 5


#ifndef LAYER
#define LAYER RELU
#endif

int main(int argc, char **argv)
{
  // printf("%d \t %d\t ", BUFFER, PREFETCH);
    if (argc < 5)
    {
        printf("USAGE: torch_pool <Input Channels> <Input Height> <Input Width> <kernel heightand width> <padding 'v' or 'f'> <Output Channels> ");
        return 0;
    }

    printf("layer %d %d %d \n", LAYER, uarch, W_ob);
    int C_i = atoi(argv[1]);

    int N = atol(argv[2]);
    int M = atol(argv[3]);

#if LAYER != RELU
    int kernel_size = atol(argv[4]);
    int stride = atol(argv[5]);
    char padding = argv[6][0] ;
    // int N, M; 

    int padding_elements;

    if (padding == 'v')
    {
        padding_elements = 0;

    }
    else if (padding == 'f')
    {
        padding_elements = (kernel_size) / (2);
    }
    else
    {
        printf("Unsupported padding type\n");
        exit();
    }
#else
    // int N = (output_rows);
    // int M = (output_cols);
#endif
#if LAYER == CONV
    uint32_t C_o = atol(argv[7]);
#endif 



    // Create and Initialize Pytorch tensors
    torch::manual_seed(1729);
    torch::Tensor a = torch::randn(C_i * N * M).reshape({1, C_i, N, M});
    a = torch::mul(a, 1.0);   
    
    
    std::vector<std::vector<uint64_t>> implementations;
    
    // Create PyTorch Convolution layers
    // set weights to generated values

#if LAYER == CONV
    torch::Tensor test_weights = torch::randn(C_o * C_i * kernel_size * kernel_size).reshape({C_o, C_i, kernel_size, kernel_size});
    auto layer = torch::nn::Conv2d(torch::nn::Conv2dOptions(C_i, C_o, kernel_size).stride(stride).padding(padding_elements).bias(false));
    layer->weight = test_weights;
#elif LAYER == POOL
    auto layer = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(kernel_size).stride(stride).padding(padding_elements));
#elif LAYER == RELU
    auto layer =  torch::relu;
#endif


    print_build_info_check();
   
    // #endif
    unsigned long long t0, t1;
    unsigned long long sum_pytorch = ULLONG_MAX;
    torch::Tensor out_intermediate, out;
    float avg;
    std::vector<uint64_t> pytorch_timing;
    for (uint32_t r = 0; r < RUNS / 10 + 1; r++)
    {
        t0 = rdtsc();
        out_intermediate = layer(a);
        t1 = rdtsc();
        MIN(sum_pytorch, (t1 - t0));
        pytorch_timing.push_back((t1 - t0));
    }
    implementations.push_back(pytorch_timing);
    print_cycles(sum_pytorch);
    fflush(0);
    
    // Direct Convolution Setup
    //  Copy layer weights to temporaries
    // torch::Tensor weights = test_weights; // layer->weight;
    std::vector<uint32_t> in_dimensions;
    std::vector<uint32_t> filter_dimensions;
    std::vector<uint32_t> out_intermediate_dimensions;
    
    std::vector<uint32_t> intermediate_block_dimensions;
    float *input_dc = alloc_dc(a, in_dimensions);
    #if LAYER < POOL
    torch::Tensor weights = test_weights; // conv_3x3->weight;
    float *filter_dc = alloc_dc(weights, filter_dimensions);
    #endif
        // float *filter_dc = NULL;
    float *out_intermediate_dc = alloc_dc(out_intermediate, out_intermediate_dimensions);
    float *output_dc = alloc_dc(out_intermediate, out_intermediate_dimensions);
    
#if PARALLEL
    uint32_t num_threads = atoi(std::getenv("OMP_NUM_THREADS"));
    #endif
    
    
    
    unsigned long long sum = ULLONG_MAX, sum_pool = ULLONG_MAX;
    volatile unsigned long long sum_fused = ULLONG_MAX,
    sum_conv = ULLONG_MAX;
    std::vector<uint64_t> unfused_timing;
    
    {
        // Initialize Outputs to 0

        // 3x3 unfused
        // Copy Inputs to their flat buffers
        copy_torch2dc<C_ob, C_ib>(a, 'i', in_dimensions, input_dc);
        #if LAYER==CONV
        copy_torch2dc<C_ob, C_ib>(weights, 'f', filter_dimensions, filter_dc);
        #endif
        bool check = 0;

       
        memset(out_intermediate_dc, 0.0, out_intermediate.numel() * sizeof(float));
        #if LAYER ==  RELU
        // direct_convolution_naive<W_ob, C_ob, C_o, 1 , 'v', 'a'>(1 , 1, 1, C_i, 1, C_i, N, M, input_dc, filter_dc, out_intermediate_dc);
        check_ReLUActivation(0, C_i, N, M, input_dc, out_intermediate_dc);
#elif LAYER == POOL
        Maxpool2D(0, kernel_size, stride, padding, C_i, N, M, input_dc, out_intermediate_dc);
#elif LAYER == CONV
        check_Conv2D(0, kernel_size, stride,  padding, C_o, C_i,  N, M, input_dc, filter_dc, out_intermediate_dc);
#endif
        check = check_eqivalence<C_ob, C_ib>(out_intermediate, 'o', out_intermediate_dimensions, out_intermediate_dc, LIMIT);
        assert(check == 1);

        // direct_convolution_naive<W_ob, C_ob, C_o, 1 , 'v', 'a'>(1 , 1, 1, C_i, 1, C_i, N, M, input_dc, filter_dc, out_intermediate_dc);
        #if LAYER == RELU
        ReLUActivation(0, C_i, N, M, input_dc, output_dc);
#elif LAYER == POOL
        Maxpool2D(0, kernel_size, stride, padding, C_i, N, M, input_dc, output_dc);
#elif LAYER == CONV
        Conv2D(0, kernel_size, stride,  padding, C_o, C_i, N, M, input_dc, filter_dc, output_dc);
#endif

        check = check_eqivalence<C_ob, C_ib>(out_intermediate, 'o', out_intermediate_dimensions, output_dc, LIMIT);
        assert(check == 1);
        // assert(equals(out_intermediate.numel(), out_intermediate_dc, output_dc, 1e-4));

        sum = ULLONG_MAX;
        for (int run = 0; run < RUNS; run++)
        {
            t0 = rdtsc();
            #if LAYER == RELU
                check_ReLUActivation(0, C_i, N, M, input_dc, out_intermediate_dc);
            #elif LAYER == POOL
            Maxpool2D(0, kernel_size, stride, padding, C_i, N, M, input_dc, out_intermediate_dc);
            #elif LAYER == CONV
            check_Conv2D(0, kernel_size, stride, padding, C_o, C_i, N, M, input_dc, filter_dc, output_dc);
            #endif
            t1 = rdtsc();
            MIN(sum, (t1 - t0));
            unfused_timing.push_back((t1 - t0));
        }

        print_cycles(sum);

        printf("%2.3f \n", (100.0 * sum_pytorch) / (sum));

        sum = ULLONG_MAX;
        for (int run = 0; run < RUNS; run++)
        {
            t0 = rdtsc();
            #if LAYER == RELU
            ReLUActivation(0, C_i, N, M, input_dc, out_intermediate_dc);
            #elif LAYER == POOL
            Maxpool2D(0, kernel_size, stride, padding, C_i, N, M, input_dc, output_dc);
            #elif LAYER == CONV
            Conv2D(0, kernel_size, stride,  padding, C_o, C_i, N, M, input_dc, filter_dc, output_dc); //     #endif
            #endif
            t1 = rdtsc();
            MIN(sum, (t1 - t0));
            unfused_timing.push_back((t1 - t0));
        }
        
        print_cycles(sum);
        
        printf("%2.3f \n", (100.0 * sum_pytorch) / (sum));

    fflush(0);
    }
    free(input_dc);
    #if LAYER < POOL
    free(filter_dc);
    #endif
    free(out_intermediate_dc);
    free(output_dc);
}
