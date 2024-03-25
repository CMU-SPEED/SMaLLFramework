#include <math.h>
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include<string.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

// Pooling driver

#include <small.h>
#include <small/utils/Timer.hpp>

#include "utils.h"

#ifndef PERFORMANCE
#define PERFORMANCE 1
#endif
#define TIME_LAYER 1

//if the compiler provides a def, ignore
#ifndef COMPUTE_BIAS
#define COMPUTE_BIAS false
#endif

#ifndef COMPUTE_RELU
#define COMPUTE_RELU false
#endif
#ifndef RUNS
#define RUNS 100
#endif

#ifndef TRIALS
#define TRIALS 100
#endif

#ifndef PARALLEL
#define PARALLEL 1
#endif

#define PREFETCH 1

#define H_TILE 0
#define POOLING 1

double layer_timers[3][15];
double min_layer_timers[3][15];
double avg_layer_timers[3][15];

// #include <params.h>  // SMaLL platform-specific includes

// #if uarch == ZEN2
// #include "/home/upasana/CMU/Quals/ActiveDevelopment/SMaLLFramework/src/kernels/zen2/params.h"
//
// #elif uarch == REF
// #include "/home/upasana/CMU/Quals/ActiveDevelopment/SMaLLFramework/src/kernels/reference/params.h"
// #endif

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

#define LIMIT 1e-2

// #include "test/interface.h"






template <class T>
inline bool almost_equal(T v1, T v2, float rtol = 5e-03, float atol = 1e-05)
{
    float abs_diff = fabs((float)(v1) - (float)(v2));
    float diff_tolerance = (atol + rtol * fabs(v2));
    return (abs_diff <= diff_tolerance);
}

#if defined(QUANTIZED)
#define CORRECTNESS_CHECK                                                                                         \
    (passing, calculated_output_dc, actual_output_dc) for (size_t ix = 0; ix < calculated_output_dc.size(); ++ix) \
    {                                                                                                             \
        if (actual_output_dc[ix] != calculated_output_dc[ix])                                                     \
        {                                                                                                         \
            passing = false;                                                                                      \
            std::cout << "FAIL: Conv2D_out(" << ix << ")-->"                                                      \
                      << std::setw(12) << std::setprecision(10)                                                   \
                      << actual_output_dc[ix] << "(computed) != "                                                 \
                      << std::setw(12) << std::setprecision(10)                                                   \
                      << calculated_output_dc[ix]                                                                 \
                      << std::endl;                                                                               \
        }                                                                                                         \
    }
#else
#define CORRECTNESS_CHECK(passing, calculated_output_dc, actual_output_dc) \
    for (size_t ix = 0; ix < calculated_output_dc.size(); ++ix)            \
    {                                                                      \
        if ((actual_output_dc[ix] != calculated_output_dc[ix]) &&          \
            !almost_equal(actual_output_dc[ix], calculated_output_dc[ix])) \
        {                                                                  \
            passing = false;                                               \
                                                                           \
            std::cout << "FAIL: Conv2D_out(" << ix << ")-->"               \
                      << std::setw(12) << std::setprecision(10)            \
                      << actual_output_dc[ix] << "(computed) != "          \
                      << std::setw(12) << std::setprecision(10)            \
                      << calculated_output_dc[ix]                          \
                      << std::endl;                                        \
        }                                                                  \
    }
#endif

#define CONV 0
#define PARTIAL_CONV 1
#define DW_CONV 2
#define GROUP_CONV 3 // under development
#define FC 4
#define LEAKY_RELU 5
#define MAX_POOL 6
#define RELU 7
#define UPSAMPLE 8
#define ACCUM 9
#define BIAS 10
#define AVERAGE_POOL 11
#define DROPOUT 12
#define SOFTMAX 13
#define PARTIAL_BIAS 14

#ifndef LAYER
#define LAYER SOFTMAX
#endif

#define CALC_PADDING(I_dim, K_dim, stride, padding_front, padding_back)              \
    {                                                                                \
        uint32_t padding;                                                            \
        if (I_dim % stride == 0)                                                     \
        {                                                                            \
            padding = (K_dim - stride > 0) ? K_dim - stride : 0;                     \
        }                                                                            \
        else                                                                         \
        {                                                                            \
            padding = (K_dim - (I_dim % stride) > 0) ? K_dim - (I_dim % stride) : 0; \
        }                                                                            \
        padding_front = padding / 2;                                                 \
        padding_back = padding - padding_front;                                      \
    }

#define C_ob FLOAT_C_ob
#define W_ob FLOAT_W_ob
#define C_ib FLOAT_C_ib




// SMaLL layer block
template <bool bias>
inline void small_layer_block(
    std::array<int32_t, 2> in_dims, uint32_t input_channels, // Input dimensions
    uint32_t kernel_size_conv,
    uint32_t stride_conv,          // Covolution parameters
    uint32_t output_channels_conv, // Convolution parameters
    uint8_t t_pad_conv,
    uint8_t b_pad_conv,
    uint8_t l_pad_conv,
    uint8_t r_pad_conv,
    uint32_t kernel_size_1,
    uint32_t stride_1,          // Covolution parameters
    uint32_t output_channels_1, // Convolution parameters
    uint8_t t_pad_1,
    uint8_t b_pad_1,
    uint8_t l_pad_1,
    uint8_t r_pad_1,
    small::FloatBuffer const &I,
    small::FloatBuffer const &F_conv,
    small::FloatBuffer const &Bias_conv,
    small::FloatBuffer const &F_layer,
    small::FloatBuffer const &Bias_layer,
    small::FloatBuffer &O_intermediate,
    small::FloatBuffer &O)
{
    small::Timer my_timer;

    uint32_t o_h, o_w;
    o_h = small::output_dim_new(in_dims[0] + t_pad_conv + b_pad_conv, stride_conv, kernel_size_conv);
    o_w = small::output_dim_new(in_dims[1] + l_pad_conv + r_pad_conv, stride_conv, kernel_size_conv);
    if constexpr (bias)
    {
        my_timer.start();
        small::Bias(output_channels_conv, o_h, o_w, Bias_conv, O_intermediate);
        my_timer.stop();
        layer_timers[0][4] = my_timer.elapsed();

        my_timer.start();
        small::PartialConv2D(kernel_size_conv, kernel_size_conv, stride_conv, t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv, output_channels_conv, input_channels, in_dims[0], in_dims[1], I, F_conv, O_intermediate);
        my_timer.stop();
        layer_timers[0][0] = my_timer.elapsed();
    }
    else
    {
        my_timer.start();
        small::Conv2D(kernel_size_conv, kernel_size_conv, stride_conv, t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv, output_channels_conv, input_channels, in_dims[0], in_dims[1], I, F_conv, O_intermediate);
        my_timer.stop();
        layer_timers[0][0] = my_timer.elapsed();
    }
    my_timer.start();
    if constexpr (COMPUTE_RELU)
    {
    small::ReLUActivation(output_channels_conv, o_h, o_w, O_intermediate, O_intermediate);
    }
    my_timer.stop();
    layer_timers[0][1] = my_timer.elapsed();
#if LAYER == MAX_POOL
    my_timer.start();
    small::MaxPool2D(kernel_size_1, kernel_size_1, stride_1, t_pad_1, b_pad_1, l_pad_1, r_pad_1, output_channels_conv, o_h, o_w, O_intermediate, O);
    my_timer.stop();
    layer_timers[0][2] = my_timer.elapsed();
#elif LAYER == DW_CONV

    uint32_t o_h_1, o_w_1;
    o_h_1 = small::output_dim_new(o_h + t_pad_1 + b_pad_1, stride_1, kernel_size_1);
    o_w_1 = small::output_dim_new(o_w + l_pad_1 + r_pad_1, stride_1, kernel_size_1);
    if constexpr (bias)
    {

        my_timer.start();
        small::Bias(output_channels_conv, o_h_1, o_w_1, Bias_layer, O);
        my_timer.stop();

        layer_timers[0][5] = my_timer.elapsed();
        my_timer.start();
        small::PartialDepthwiseConv2D(kernel_size_1, kernel_size_1, stride_1, t_pad_1, b_pad_1, l_pad_1, r_pad_1, output_channels_conv, o_h, o_w, O_intermediate, F_layer, O);
        my_timer.stop();
        layer_timers[0][2] = my_timer.elapsed();
    }
    else
    {
        my_timer.start();
        small::DepthwiseConv2D(kernel_size_1, kernel_size_1, stride_1, t_pad_1, b_pad_1, l_pad_1, r_pad_1, output_channels_conv, o_h, o_w, O_intermediate, F_layer, O);
        my_timer.stop();
        layer_timers[0][2] = my_timer.elapsed();
    }
    my_timer.start();
    if constexpr (COMPUTE_RELU)
    {   
    small::ReLUActivation(output_channels_conv, o_h_1, o_w_1, O, O);
    }
    my_timer.stop();
    layer_timers[0][3] = my_timer.elapsed();

#endif
}

// Fused SMaLL layer block

// Fusing just elementwise layers
template <bool bias>
inline void small_fused_ewise_layer_block(
    std::array<int32_t, 2> in_dims, uint32_t input_channels, // Input dimensions
    uint32_t kernel_size_conv,
    uint32_t stride_conv,          // Covolution parameters
    uint32_t output_channels_conv, // Convolution parameters
    uint8_t t_pad_conv,
    uint8_t b_pad_conv,
    uint8_t l_pad_conv,
    uint8_t r_pad_conv,
    uint32_t kernel_size_1,
    uint32_t stride_1,          // Covolution parameters
    uint32_t output_channels_1, // Convolution parameters
    uint8_t t_pad_1,
    uint8_t b_pad_1,
    uint8_t l_pad_1,
    uint8_t r_pad_1,
    small::FloatBuffer const &I,
    small::FloatBuffer const &F_conv,
    small::FloatBuffer const &Bias_conv,
    small::FloatBuffer const &F_layer,
    small::FloatBuffer const &Bias_layer,
    small::FloatBuffer &O_intermediate,
    small::FloatBuffer &O)
{
    small::Timer my_timer;
    uint32_t o_h, o_w;
    o_h = small::output_dim_new(in_dims[0] + t_pad_conv + b_pad_conv, stride_conv, kernel_size_conv);
    o_w = small::output_dim_new(in_dims[1] + l_pad_conv + r_pad_conv, stride_conv, kernel_size_conv);
#if LAYER == MAX_POOL
    if constexpr (bias)
    {
        my_timer.start();
        small::Conv2D_Bias_ReLU(kernel_size_conv, kernel_size_conv, stride_conv,
                                t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
                                output_channels_conv, input_channels,
                                in_dims[0], in_dims[1],
                                I,
                                F_conv,
                                Bias_conv,
                                O_intermediate);
        my_timer.stop();
        layer_timers[1][0] = my_timer.elapsed();

        my_timer.start();
        small::MaxPool2D(kernel_size_1, kernel_size_1, stride_1, t_pad_1, b_pad_1, l_pad_1, r_pad_1, output_channels_conv, o_h, o_w, O_intermediate, O);
        my_timer.stop();
        layer_timers[1][2] = my_timer.elapsed();
   
    }
    else
    {
        my_timer.start();
        if constexpr (COMPUTE_RELU)
        {
            small::Conv2D_ReLU(kernel_size_conv, kernel_size_conv, stride_conv,
                               t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
                               output_channels_conv, input_channels,
                               in_dims[0], in_dims[1],
                               I,
                               F_conv,
                               O_intermediate);
        }
        else
        {
            small::Conv2D(kernel_size_conv, kernel_size_conv, stride_conv,
                          t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
                          output_channels_conv, input_channels,
                          in_dims[0], in_dims[1],
                          I,
                          F_conv,
                          O_intermediate);
        }
        // small::Conv2D_ReLU(kernel_size_conv, kernel_size_conv, stride_conv,
        //                    t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
        //                    output_channels_conv, input_channels,
        //                    in_dims[0], in_dims[1],
        //                    I,
        //                    F_conv,
        //                    O_intermediate);
        my_timer.stop();
        layer_timers[1][0] = my_timer.elapsed();

        my_timer.start();
        small::MaxPool2D(kernel_size_1, kernel_size_1, stride_1, t_pad_1, b_pad_1, l_pad_1, r_pad_1, output_channels_conv, o_h, o_w, O_intermediate, O);
        my_timer.stop();
        layer_timers[1][2] = my_timer.elapsed();
   }
#elif LAYER == DW_CONV
    if constexpr (bias)
    {
        my_timer.start();
        small::Conv2D_Bias_ReLU(kernel_size_conv, kernel_size_conv, stride_conv,
                                t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
                                output_channels_conv, input_channels,
                                in_dims[0], in_dims[1],
                                I,
                                F_conv,
                                Bias_conv,
                                O_intermediate);

        my_timer.stop();
        layer_timers[1][0] = my_timer.elapsed();

        my_timer.start();
        small::DepthwiseConv2D_Bias_ReLU(kernel_size_1, kernel_size_1, stride_1,
                                         t_pad_1, b_pad_1, l_pad_1, r_pad_1,
                                         output_channels_conv,
                                         o_h, o_w,
                                         O_intermediate,
                                         F_layer,
                                         Bias_layer,
                                         O);
        my_timer.stop();
        layer_timers[1][2] = my_timer.elapsed();
    }
    else
    {
        my_timer.start();

        if constexpr (COMPUTE_RELU)
        {
            small::Conv2D_ReLU(kernel_size_conv, kernel_size_conv, stride_conv,
                               t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
                               output_channels_conv, input_channels,
                               in_dims[0], in_dims[1],
                               I,
                               F_conv,
                               O_intermediate);
        }
        else
        {
            small::Conv2D(kernel_size_conv, kernel_size_conv, stride_conv,
                          t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
                          output_channels_conv, input_channels,
                          in_dims[0], in_dims[1],
                          I,
                          F_conv,
                          O_intermediate);
        }
        // small::Conv2D_ReLU(kernel_size_conv, kernel_size_conv, stride_conv,
        //                    t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
        //                    output_channels_conv, input_channels,
        //                    in_dims[0], in_dims[1],
        //                    I,
        //                    F_conv,
                        //    O_intermediate);


        my_timer.stop();
        layer_timers[1][0] = my_timer.elapsed();

        my_timer.start();
        if constexpr(COMPUTE_RELU)
        {
            small::DepthwiseConv2D_ReLU(kernel_size_1, kernel_size_1, stride_1,
                                        t_pad_1, b_pad_1, l_pad_1, r_pad_1,
                                        output_channels_conv,
                                        o_h, o_w,
                                        O_intermediate,
                                        F_layer,
                                        O);
        }
        else
        {
            small::DepthwiseConv2D(kernel_size_1, kernel_size_1, stride_1,
                                    t_pad_1, b_pad_1, l_pad_1, r_pad_1,
                                    output_channels_conv,
                                    o_h, o_w,
                                    O_intermediate,
                                    F_layer,
                                    O);
        }
        // small::DepthwiseConv2D_ReLU(kernel_size_1, kernel_size_1, stride_1,
        //                             t_pad_1, b_pad_1, l_pad_1, r_pad_1,
        //                             output_channels_conv,
        //                             o_h, o_w,
        //                             O_intermediate,
        //                             F_layer,
        //                             O);
        my_timer.stop();
        layer_timers[1][2] = my_timer.elapsed();
    }
#endif
}

// Fusing a block of layers including non-elementwise layers
template <bool bias>
inline void fused_small_layer_block(
    std::array<int32_t, 2> in_dims, uint32_t input_channels, // Input dimensions
    uint32_t kernel_size_conv,
    uint32_t stride_conv,          // Covolution parameters
    uint32_t output_channels_conv, // Convolution parameters
    uint8_t t_pad_conv,
    uint8_t b_pad_conv,
    uint8_t l_pad_conv,
    uint8_t r_pad_conv,
    uint32_t kernel_size_1,
    uint32_t stride_1,          // Covolution parameters
    uint32_t output_channels_1, // Convolution parameters
    uint8_t t_pad_1,
    uint8_t b_pad_1,
    uint8_t l_pad_1,
    uint8_t r_pad_1,
    small::FloatBuffer const &I,
    small::FloatBuffer const &F_conv,
    small::FloatBuffer const &Bias_conv,
    small::FloatBuffer const &F_layer,
    small::FloatBuffer const &Bias_layer,
    small::FloatBuffer &O_intermediate,
    small::FloatBuffer &O)
{

    small::Timer my_timer;
    my_timer.start();
#if LAYER == MAX_POOL
    if constexpr (bias)
    {
        small::Conv2D_Bias_ReLU_Maxpool2D(kernel_size_conv, kernel_size_conv, stride_conv,
                                          t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
                                          kernel_size_1, kernel_size_1, stride_1,
                                          t_pad_1, b_pad_1, l_pad_1, r_pad_1,
                                          output_channels_conv, input_channels,
                                          in_dims[0], in_dims[1],
                                          I,
                                          F_conv,
                                          Bias_conv,
                                          O_intermediate,
                                          O);
    }
    else
    {
        if constexpr(COMPUTE_RELU)
        {
            small::Conv2D_ReLU_Maxpool2D(kernel_size_conv, kernel_size_conv, stride_conv,
                                          t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
                                          kernel_size_1, kernel_size_1, stride_1,
                                          t_pad_1, b_pad_1, l_pad_1, r_pad_1,
                                          output_channels_conv, input_channels,
                                          in_dims[0], in_dims[1],
                                          I,
                                          F_conv,
                                          O_intermediate,
                                          O);
        }
        else
        {
            small::Conv2D_Maxpool2D(kernel_size_conv, kernel_size_conv, stride_conv,
                                    t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
                                    kernel_size_1, kernel_size_1, stride_1,
                                    t_pad_1, b_pad_1, l_pad_1, r_pad_1,
                                    output_channels_conv, input_channels,
                                    in_dims[0], in_dims[1],
                                    I,
                                    F_conv,
                                    O_intermediate,
                                    O);
        }
        // small::Conv2D_ReLU_Maxpool2D(kernel_size_conv, kernel_size_conv, stride_conv,
        //                              t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
        //                              kernel_size_1, kernel_size_1, stride_1,
        //                              t_pad_1, b_pad_1, l_pad_1, r_pad_1,
        //                              output_channels_conv, input_channels,
        //                              in_dims[0], in_dims[1],
        //                              I,
        //                              F_conv,
        //                              O_intermediate,
        //                              O);
    }
#elif LAYER == DW_CONV
    if constexpr (bias)
    {
        small::Conv2D_Bias_ReLU_DepthwiseConv2D_Bias_ReLU(kernel_size_conv, kernel_size_conv, stride_conv,
                                                          t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
                                                          kernel_size_1, kernel_size_1, stride_1,
                                                          t_pad_1, b_pad_1, l_pad_1, r_pad_1,
                                                          output_channels_conv, input_channels,
                                                          in_dims[0], in_dims[1],
                                                          I, F_conv, Bias_conv, O_intermediate,
                                                          F_layer, Bias_layer, O);
    }
    else
    {
        if constexpr(COMPUTE_RELU)
        {
            small::Conv2D_ReLU_DepthwiseConv2D_ReLU(kernel_size_conv, kernel_size_conv, stride_conv,
                                                    t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
                                                    kernel_size_1, kernel_size_1, stride_1,
                                                    t_pad_1, b_pad_1, l_pad_1, r_pad_1,
                                                    output_channels_conv, input_channels,
                                                    in_dims[0], in_dims[1],
                                                    I, F_conv, O_intermediate,
                                                    F_layer, O);
        }
        else
        {
            small::Conv2D_DepthwiseConv2D(kernel_size_conv, kernel_size_conv, stride_conv,
                                          t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
                                          kernel_size_1, kernel_size_1, stride_1,
                                          t_pad_1, b_pad_1, l_pad_1, r_pad_1,
                                          output_channels_conv, input_channels,
                                          in_dims[0], in_dims[1],
                                          I, F_conv, O_intermediate,
                                          F_layer, O);
        }
        // small::Conv2D_ReLU_DepthwiseConv2D_ReLU(kernel_size_conv, kernel_size_conv, stride_conv,
        //                                         t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
        //                                         kernel_size_1, kernel_size_1, stride_1,
        //                                         t_pad_1, b_pad_1, l_pad_1, r_pad_1,
        //                                         output_channels_conv, input_channels,
        //                                         in_dims[0], in_dims[1],
        //                                         I, F_conv, O_intermediate,
        //                                         F_layer, O);
    }
#endif
my_timer.stop();
layer_timers[2][0] = my_timer.elapsed();
}

int main(int argc, char **argv)
{
    if (argc < 11)
    {
        std::cerr << "ERROR: too few arguments, got " << argc << std::endl;
        std::cerr << "USAGE: " << argv[0]
                  << " <Input Channels> <Input Height> <Input Width>"
                  << " Convolution parameters <kernel height/width> <stride> <padding 'v' or 'f'>"
                  << " Convolution <Output Channels> "
                  << " Second layer parameters <kernel height/width> <stride> <padding 'v' or 'f'>"
                  << " Second layer <Output Channels>"
                  << std::endl;
        return 0;
    }

    std::string base_fname;

    // Set up first (conv) layer

    // Input dims
    //  printf("layer %d %d %d \n", LAYER, uarch, W_ob);
    int C_i = atoi(argv[1]);
    base_fname += "Ci" + std::to_string(C_i) + "_";

    int input_height = atol(argv[2]);
    base_fname += "H" + std::to_string(input_height) + "_";

    int input_width = atol(argv[3]);
    base_fname += "W" + std::to_string(input_width) + "_";

    // #if LAYER != RELU
    int conv_kernel_size = atol(argv[4]);
    base_fname += "k" + std::to_string(conv_kernel_size) + "_";
    int conv_stride = atol(argv[5]);
    base_fname += "s" + std::to_string(conv_stride) + "_";
    char conv_padding = argv[6][0];
    base_fname += conv_padding;
    base_fname += "_";
    // int input_height, input_width;

    uint8_t t_pad_conv = 0, b_pad_conv = 0;
    uint8_t l_pad_conv = 0, r_pad_conv = 0;

    int conv_padding_front = 0, conv_padding_back = 0;

    if (conv_padding == 'f')
    {
        CALC_PADDING(input_height, conv_kernel_size, conv_stride, t_pad_conv, b_pad_conv);
        CALC_PADDING(input_width, conv_kernel_size, conv_stride, l_pad_conv, r_pad_conv);
    }

    uint32_t C_o_conv = atol(argv[7]);
    base_fname += "Co conv" + std::to_string(C_o_conv) + "_";

    // std::cout << "\n***TORCH FUSED INTERFACE DRIVER FIRST LAYER IS 3s1 conv SECOND LAYER: ***\n";
    std::string fused_layer_name;
    switch (LAYER)
    {
    case CONV:
        // std::cout << "LAYER = CONV\n";
        base_fname += "_conv2d_";
        break; // yes
    case PARTIAL_CONV:
        // std::cout << "LAYER = PARTIAL_CONV\n";
        base_fname += "_partial_conv_";
        break; // yes
    case DW_CONV:
        // std::cout << "LAYER = DW_CONV\n";
        base_fname += "_dw_conv_";
        break; // yes
    case GROUP_CONV:
        // std::cout << "LAYER = GROUP_CONV\n";
        base_fname += "_gp_conv_";
        break;
    case FC:
        // std::cout << "LAYER = FC\n";
        base_fname += "_fc_";
        break; // yes
    case MAX_POOL:
        // std::cout << "LAYER = MAX_POOL\n";
        base_fname += "_max_pool_";
        break; // yes
    case RELU:
        // std::cout << "LAYER = RELU\n";
        base_fname += "_relu_";
        break; // yes
    case UPSAMPLE:
        // std::cout << "LAYER = UPSAMPLE\n";
        base_fname += "_upsample_";
        break; // yes
    case AVERAGE_POOL:
        // std::cout << "LAYER = AVERAGE_POOL\n";
        base_fname += "_average_pool_";
        break; // yes
    case DROPOUT:
        // std::cout << "LAYER = DROPOUT\n";
        base_fname += "_dropout";
        break; // yes
    case SOFTMAX:
        // std::cout << "LAYER = SOFTMAX\n";
        base_fname += "_softmax";
        break; // yes
    case PARTIAL_BIAS:
        // std::cout << "LAYER = PARTIAL_BIAS\n";
        base_fname += "_partial_bias";
        break; // yes
    }

    int kernel_size = atol(argv[8]);
    base_fname += "k" + std::to_string(kernel_size) + "_";
    int stride = atol(argv[9]);
    base_fname += "s" + std::to_string(stride) + "_";
    char padding = argv[10][0];
    base_fname += padding;
    base_fname += "_";

    uint8_t t_pad = 0, b_pad = 0;
    uint8_t l_pad = 0, r_pad = 0;

    int padding_front = 0, padding_back = 0;

    auto conv_output_height = small::output_dim_new((input_height + t_pad_conv + b_pad_conv),
                                                    conv_stride, conv_kernel_size);
    auto conv_output_width = small::output_dim_new((input_width + l_pad_conv + r_pad_conv),
                                                   conv_stride, conv_kernel_size);

    if (padding == 'f')
    {
        CALC_PADDING(conv_output_height, kernel_size, stride, t_pad, b_pad);
        CALC_PADDING(conv_output_width, kernel_size, stride, l_pad, r_pad);
    }

    auto output_height = small::output_dim_new((conv_output_height + t_pad + b_pad),
                                                    stride, kernel_size);
    auto output_width = small::output_dim_new((conv_output_width + l_pad + r_pad),
                                                   stride, kernel_size);

#if LAYER == CONV || LAYER == PARTIAL_CONV
    uint32_t C_o = atol(argv[7]);

#elif LAYER == GROUP_CONV
    uint32_t G = atol(argv[7]);
#endif

#if LAYER == FC
    input_height = 1;
    input_width = 1;
    kernel_size = 1;
    stride = 1;
    uint32_t C_o = atol(argv[7]);
#endif

    // std::cout << "image dimensions  : " << input_height << " x " << input_width << std::endl;
    // std::cout << "stride:           " << stride << std::endl;
    // std::cout << "padding(t,b,l,r): ";
    // printf("%u %u %u %u \n", t_pad, b_pad, l_pad, r_pad);
    // std::cout << "conv padding(t,b,l,r): ";
    // printf("%u %u %u %u \n", t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv);

    uint32_t num_threads = 1;
#if PARALLEL
    if (NULL != std::getenv("OMP_NUM_THREADS"))
        num_threads = atoi(std::getenv("OMP_NUM_THREADS"));
#endif

    // Direct Convolution Setup
    //  Copy layer weights to temporaries
    // torch::Tensor weights = test_weights; // layer->weight;
    std::vector<uint32_t> in_dimensions;
    std::vector<uint32_t> conv_filter_dimensions;
    std::vector<uint32_t> conv_bias_dimensions;
    std::vector<uint32_t> filter_dimensions;
    std::vector<uint32_t> bias_dimensions;
    std::vector<uint32_t> out_intermediate_dimensions;
    std::vector<uint32_t> out_intermediate_unfused_dimensions;
    std::vector<uint32_t> out_dimensions;

    std::vector<uint32_t> intermediate_block_dimensions;

    size_t in_buffer_size(C_i*input_height*input_width);
    size_t conv_filter_buffer_size(C_i*C_o_conv*conv_kernel_size*conv_kernel_size);
    size_t conv_bias_buffer_size(C_o_conv*COMPUTE_BIAS);
#if LAYER == MAX_POOL
    size_t filter_buffer_size(0);
    size_t bias_buffer_size(0);
#elif LAYER == DW_CONV
    size_t filter_buffer_size(C_o_conv*kernel_size*kernel_size);
    size_t bias_buffer_size(C_o_conv*COMPUTE_BIAS);
#endif
    size_t out_intermediate_buffer_size(conv_output_height * conv_output_width * FLOAT_C_ob * num_threads);
    //@todo: work this out
    size_t out_intermediate_unfused_buffer_size(conv_output_height * conv_output_width * C_o_conv);
    size_t out_buffer_size(output_height*output_width*C_o_conv);


    // Allocate and init buffers
    small::FloatBuffer input_dc(in_buffer_size);
    small::init(input_dc, in_buffer_size);


    // Conv layer weights and intermediate

    small::FloatBuffer conv_filter_dc(conv_filter_buffer_size);
    small::init(conv_filter_dc, conv_filter_buffer_size);

    small::FloatBuffer conv_bias_dc(conv_bias_buffer_size);
    small::init(conv_bias_dc, conv_bias_buffer_size);

    // Fused layer weights

#if LAYER < MAX_POOL
    small::FloatBuffer filter_dc(filter_buffer_size);
    small::init(filter_dc, filter_buffer_size);

    small::FloatBuffer bias_dc(bias_buffer_size);
    small::init(bias_dc, bias_buffer_size);
#else
    small::FloatBuffer filter_dc(0);
    small::FloatBuffer bias_dc(0);
#endif

    small::FloatBuffer out_intermediate_unfused_dc(out_intermediate_unfused_buffer_size);

    small::FloatBuffer output_dc_unfused(out_buffer_size);
    small::FloatBuffer output_dc(out_buffer_size);








    unsigned long long sum = ULLONG_MAX, sum_pool = ULLONG_MAX;
    volatile unsigned long long sum_fused = ULLONG_MAX,
                                sum_conv = ULLONG_MAX;
    std::vector<uint64_t> unfused_timing;

    // printf("running unfused ");
    fflush(0);

    // Checking Fused SMaLL Framework implementation
    small_layer_block<COMPUTE_BIAS>(std::array<int32_t, 2>({input_height, input_width}), C_i, // Input dimensions
                                    conv_kernel_size,
                                    conv_stride, // Covolution parameters
                                    C_o_conv,    // Convolution parameters
                                    t_pad_conv,
                                    b_pad_conv,
                                    l_pad_conv,
                                    r_pad_conv,
                                    kernel_size,
                                    stride,   // Covolution parameters
                                    C_o_conv, // Convolution parameters
                                    t_pad,
                                    b_pad,
                                    l_pad,
                                    r_pad,
                                    input_dc,
                                    conv_filter_dc,
                                    conv_bias_dc,
                                    filter_dc,
                                    bias_dc,
                                    out_intermediate_unfused_dc,
                                    output_dc_unfused);

    // bool check_unfused;
    // check_unfused = check_eqivalence<C_ob, C_ib>(out_intermediate, 'o', out_intermediate_unfused_dimensions, out_intermediate_unfused_dc.data(), LIMIT);
    // assert(check_unfused == 1);
    // check_unfused = check_eqivalence<C_ob, C_ib>(out, 'o', out_dimensions, output_dc_unfused.data(), LIMIT);
    // assert(check_unfused == 1);
    // printf("check_unfused %d ", check_unfused);

    memset(out_intermediate_unfused_dc.data(), 0.0, out_intermediate_unfused_buffer_size * sizeof(float));

    // printf("running fused ewise ");
    // fflush(0);

    small_fused_ewise_layer_block<COMPUTE_BIAS>(std::array<int32_t, 2>({input_height, input_width}), C_i, // Input dimensions
                                                conv_kernel_size,
                                                conv_stride, // Covolution parameters
                                                C_o_conv,    // Convolution parameters
                                                t_pad_conv,
                                                b_pad_conv,
                                                l_pad_conv,
                                                r_pad_conv,
                                                kernel_size,
                                                stride,   // Covolution parameters
                                                C_o_conv, // Convolution parameters
                                                t_pad,
                                                b_pad,
                                                l_pad,
                                                r_pad,
                                                input_dc,
                                                conv_filter_dc,
                                                conv_bias_dc,
                                                filter_dc,
                                                bias_dc,
                                                out_intermediate_unfused_dc,
                                                output_dc);

    bool check_ewise_fusion = true;
    CORRECTNESS_CHECK(check_ewise_fusion, output_dc, output_dc_unfused);

    assert(check_ewise_fusion == 1);
    // printf("check_ewise_fusion %d ", check_ewise_fusion);
    // fflush(0);

    // Full Fused block
    memset(out_intermediate_unfused_dc.data(), 0.0, out_intermediate_unfused_buffer_size * sizeof(float));
    // printf("C_i %d, C_o_conv %d, C_o %d, kernel_size_conv %d, kernel_size %d, stride_conv %d, stride %d, t_pad_conv %d, t_pad %d, b_pad_conv %d, b_pad %d, l_pad_conv %d, l_pad %d, r_pad_conv %d, r_pad %d \n", C_i, C_o_conv, C_o_conv, conv_kernel_size, kernel_size, conv_stride, stride, t_pad_conv, t_pad, b_pad_conv, b_pad, l_pad_conv, l_pad, r_pad_conv, r_pad);
    fused_small_layer_block<COMPUTE_BIAS>(std::array<int32_t, 2>({input_height, input_width}), C_i, // Input dimensions
                                          conv_kernel_size,
                                          conv_stride, // Covolution parameters
                                          C_o_conv,    // Convolution parameters
                                          t_pad_conv,
                                          b_pad_conv,
                                          l_pad_conv,
                                          r_pad_conv,
                                          kernel_size,
                                          stride,   // Covolution parameters
                                          C_o_conv, // Convolution parameters
                                          t_pad,
                                          b_pad,
                                          l_pad,
                                          r_pad,
                                          input_dc,
                                          conv_filter_dc,
                                          conv_bias_dc,
                                          filter_dc,
                                          bias_dc,
                                          out_intermediate_unfused_dc,
                                          output_dc);

    bool check = true;
    CORRECTNESS_CHECK(check, output_dc, output_dc_unfused);
    assert(check == 1);
    // printf("check %d ", check);
    fflush(0);

#if PERFORMANCE == 0
    output_file(out_fname, output_dc.data(), out_buffer_size);
#endif
//_________________________________________________________________

    unsigned long long t0, t1;
// performance comparison
#if PERFORMANCE == 1

    // printf("runs %d, %d, \t ", RUNS, num_threads);
    // Unfused
    unsigned long long sum_small_conv = ULLONG_MAX;
    std::vector<unsigned long long> small_conv_timing;
    for (int r = 0; r < RUNS; r++)
    {
        t0 = rdtsc();
        for (int i = 0; i < TRIALS; i++)
        {
            small::Conv2D(conv_kernel_size, conv_kernel_size, conv_stride,
                          t_pad_conv,
                          b_pad_conv,
                          l_pad_conv,
                          r_pad_conv,
                          C_o_conv, C_i,
                          input_height, input_width,
                          input_dc,
                          conv_filter_dc,
                        //   conv_bias_dc,
                          out_intermediate_unfused_dc);
        }
        t1 = rdtsc();
        diff = (t1 - t0) / TRIALS;
        MIN(sum_small_conv, (diff));
        small_conv_timing.push_back((diff));
    }
    fflush(0);

    // Unfused
    unsigned long long sum_small = ULLONG_MAX, sum_small_conv_relu = ULLONG_MAX, sum_small_pool = ULLONG_MAX, sum_small_pool_relu = ULLONG_MAX;
    std::vector<unsigned long long> small_timing;
    for (int r = 0; r < RUNS; r++)
    {
        int impl = 0;
        t0 = rdtsc();
        for (int i = 0; i < TRIALS; i++)
        {
            small_layer_block<COMPUTE_BIAS>(std::array<int32_t, 2>({input_height, input_width}), C_i, // Input dimensions
                                            conv_kernel_size,
                                            conv_stride, // Covolution parameters
                                            C_o_conv,    // Convolution parameters
                                            t_pad_conv,
                                            b_pad_conv,
                                            l_pad_conv,
                                            r_pad_conv,
                                            kernel_size,
                                            stride,   // Covolution parameters
                                            C_o_conv, // Convolution parameters
                                            t_pad,
                                            b_pad,
                                            l_pad,
                                            r_pad,
                                            input_dc,
                                            conv_filter_dc,
                                            conv_bias_dc,
                                            filter_dc,
                                            bias_dc,
                                            out_intermediate_unfused_dc,
                                            output_dc);
        }

        t1 = rdtsc();
        diff = (t1-t0)/TRIALS;
        MIN(sum_small, (diff));
        small_timing.push_back((diff));


#if TIME_LAYER

        for (int i = 0; i < TRIALS; i++)
        {
            small_layer_block<COMPUTE_BIAS>(std::array<int32_t, 2>({input_height, input_width}), C_i, // Input dimensions
                                            conv_kernel_size,
                                            conv_stride, // Covolution parameters
                                            C_o_conv,    // Convolution parameters
                                            t_pad_conv,
                                            b_pad_conv,
                                            l_pad_conv,
                                            r_pad_conv,
                                            kernel_size,
                                            stride,   // Covolution parameters
                                            C_o_conv, // Convolution parameters
                                            t_pad,
                                            b_pad,
                                            l_pad,
                                            r_pad,
                                            input_dc,
                                            conv_filter_dc,
                                            conv_bias_dc,
                                            filter_dc,
                                            bias_dc,
                                            out_intermediate_unfused_dc,
                                            output_dc);

            for (int timer = 0; timer < 6; timer++)
            {
                if (0 < i)
                {
                    avg_layer_timers[impl][timer] += layer_timers[impl][timer];
                }
                else
                {
                    avg_layer_timers[impl][timer] = layer_timers[impl][timer];
                }
            }
        }


// #endif
        
       
// #if TIME_LAYER
        for (int timer = 0; timer < 6; timer++)
        {
            avg_layer_timers[impl][timer] /= TRIALS;
            if (0 < r)
            {
                min_layer_timers[impl][timer] = std::min<double>(min_layer_timers[impl][timer], avg_layer_timers[impl][timer]);
            }
            else
            {
                min_layer_timers[impl][timer] = avg_layer_timers[impl][timer];
            }
        }
#endif
    }
    fflush(0);


    // Fused implementations
    // Fused ewise
    unsigned long long sum_small_fused_ewise = ULLONG_MAX;

    unsigned long long sum_small_fused_ewise_conv = ULLONG_MAX;
    unsigned long long sum_small_fused_ewise_dw = ULLONG_MAX;

    std::vector<unsigned long long> small_timing_fused_ewise;
    for (int r = 0; r < RUNS; r++)
    {
        small::Timer my_timer;
        t0 = rdtsc();
        for(int trial = 0; trial < TRIALS; trial++)
        {
        small_fused_ewise_layer_block<COMPUTE_BIAS>(std::array<int32_t, 2>({input_height, input_width}), C_i, // Input dimensions
                                                    conv_kernel_size,
                                                    conv_stride, // Covolution parameters
                                                    C_o_conv,    // Convolution parameters
                                                    t_pad_conv,
                                                    b_pad_conv,
                                                    l_pad_conv,
                                                    r_pad_conv,
                                                    kernel_size,
                                                    stride,   // Covolution parameters
                                                    C_o_conv, // Convolution parameters
                                                    t_pad,
                                                    b_pad,
                                                    l_pad,
                                                    r_pad,
                                                    input_dc,
                                                    conv_filter_dc,
                                                    conv_bias_dc,
                                                    filter_dc,
                                                    bias_dc,
                                                    out_intermediate_unfused_dc,
                                                    output_dc);
        }
        t1 = rdtsc();
        MIN(sum_small_fused_ewise, (t1 - t0)/TRIALS);
        small_timing_fused_ewise.push_back((t1 - t0)/TRIALS);

        #if TIME_LAYER
        auto impl = 1;
        for (int trial = 0; trial < TRIALS; trial++)
        {
            small_fused_ewise_layer_block<COMPUTE_BIAS>(std::array<int32_t, 2>({input_height, input_width}), C_i, // Input dimensions
                                                        conv_kernel_size,
                                                        conv_stride, // Covolution parameters
                                                        C_o_conv,    // Convolution parameters
                                                        t_pad_conv,
                                                        b_pad_conv,
                                                        l_pad_conv,
                                                        r_pad_conv,
                                                        kernel_size,
                                                        stride,   // Covolution parameters
                                                        C_o_conv, // Convolution parameters
                                                        t_pad,
                                                        b_pad,
                                                        l_pad,
                                                        r_pad,
                                                        input_dc,
                                                        conv_filter_dc,
                                                        conv_bias_dc,
                                                        filter_dc,
                                                        bias_dc,
                                                        out_intermediate_unfused_dc,
                                                        output_dc);

            for (int timer = 0; timer < 6; timer++)
            {
                if (0 < trial)
                {
                    avg_layer_timers[impl][timer] += layer_timers[impl][timer];
                }
                else
                {
                    avg_layer_timers[impl][timer] = layer_timers[impl][timer];
                }
            }
        }
        for (int timer = 0; timer < 6; timer++)
        {
            avg_layer_timers[impl][timer] /= TRIALS;
            if (0 < r)
            {
                min_layer_timers[impl][timer] = std::min<double>(min_layer_timers[impl][timer], avg_layer_timers[impl][timer]);
            }
            else
            {
                min_layer_timers[impl][timer] = avg_layer_timers[impl][timer];
            }
        }
#endif

    }
  
    // print_cycles(sum_small_fused_ewise_dw + sum_small_fused_ewise_conv);

    fflush(0);

  
    printf("\t");

    // Fused block
    unsigned long long sum_small_fused = ULLONG_MAX;
    std::vector<unsigned long long> small_timing_fused;
    for (int r = 0; r < RUNS; r++)
    {
        t0 = rdtsc();
        for (int trial = 0; trial < TRIALS; trial++)
        {
            fused_small_layer_block<COMPUTE_BIAS>(std::array<int32_t, 2>({input_height, input_width}), C_i, // Input dimensions
                                                  conv_kernel_size,
                                                  conv_stride, // Covolution parameters
                                                  C_o_conv,    // Convolution parameters
                                                  t_pad_conv,
                                                  b_pad_conv,
                                                  l_pad_conv,
                                                  r_pad_conv,
                                                  kernel_size,
                                                  stride,   // Covolution parameters
                                                  C_o_conv, // Convolution parameters
                                                  t_pad,
                                                  b_pad,
                                                  l_pad,
                                                  r_pad,
                                                  input_dc,
                                                  conv_filter_dc,
                                                  conv_bias_dc,
                                                  filter_dc,
                                                  bias_dc,
                                                  out_intermediate_unfused_dc,
                                                  output_dc);
        }
        t1 = rdtsc();
        MIN(sum_small_fused, (t1 - t0)/TRIALS);
        small_timing_fused.push_back((t1 - t0)/TRIALS);

        #if TIME_LAYER 
        auto impl = 2;
        for (int trial = 0; trial < TRIALS; trial++)
        {
        
            fused_small_layer_block<COMPUTE_BIAS>(std::array<int32_t, 2>({input_height, input_width}), C_i, // Input dimensions
                                                  conv_kernel_size,
                                                  conv_stride, // Covolution parameters
                                                  C_o_conv,    // Convolution parameters
                                                  t_pad_conv,
                                                  b_pad_conv,
                                                  l_pad_conv,
                                                  r_pad_conv,
                                                  kernel_size,
                                                  stride,   // Covolution parameters
                                                  C_o_conv, // Convolution parameters
                                                  t_pad,
                                                  b_pad,
                                                  l_pad,
                                                  r_pad,
                                                  input_dc,
                                                  conv_filter_dc,
                                                  conv_bias_dc,
                                                  filter_dc,
                                                  bias_dc,
                                                  out_intermediate_unfused_dc,
                                                  output_dc);

            for (int timer = 0; timer < 6; timer++)
            {
                if (0 < trial)
                {
                    avg_layer_timers[impl][timer] += layer_timers[impl][timer];
                }
                else
                {
                    avg_layer_timers[impl][timer] = layer_timers[impl][timer];
                }
            }
        }

        for (int timer = 0; timer < 6; timer++)
        {
            avg_layer_timers[impl][timer] /= TRIALS;
            if (0 < r)
            {
                min_layer_timers[impl][timer] = std::min<double>(min_layer_timers[impl][timer], avg_layer_timers[impl][timer]);
            }
            else
            {
                min_layer_timers[impl][timer] = avg_layer_timers[impl][timer];
            }
        }
#endif
    }
    fflush(0);

#if TIME_LAYER
    double sum_unfused = 0, sum_ewise = 0, sum_fused_block = 0;
    printf("impl, conv , conv_relu, pool/dw, relu, conv_bias, dw_bias\n ");
    for (int impl = 0; impl < 3; impl++)
    {  printf("%d , ", impl);
        for (int layer = 0; layer < 6; layer++)
        {
            printf("%f ,", min_layer_timers[impl][layer]);
            if(impl == 0)
            {
            sum_unfused += min_layer_timers[0][layer];
            sum_ewise += min_layer_timers[1][layer];
            sum_fused_block += min_layer_timers[2][layer];
            }
        }
        printf("\n");

    }

    printf("sum , %f, %f, %f \n", sum_unfused, sum_ewise, sum_fused_block);
#endif

    print_cycles(sum_small_conv);
    print_cycles(sum_small);
    print_cycles(sum_small_fused_ewise);
    // print_cycles(sum_small_fused_ewise_conv);
    // print_cycles(sum_small_fused_ewise_dw);
    print_cycles(sum_small_fused);
    printf(" %.4f, ", (sum_small * 1.0) / (sum_small_fused_ewise * 1.0));
    printf("%.4f , %d", (sum_small * 1.0) / (sum_small_fused * 1.0), COMPUTE_BIAS);
    printf("\n");

    #if PARALLEL_DIST == ELEMENTAL
    printf("%d %d ELEMENTAL\n", RUNS, TRIALS);
    #else
    printf("%d %d BLOCK\n", RUNS, TRIALS);
    #endif
#endif


}