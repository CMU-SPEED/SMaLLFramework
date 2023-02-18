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
#include <algorithm> // std::min_element
#include <iterator>

#include <small.h>
#include "utils.h"

/// @todo Which of these defines are needed?
#ifndef RUNS
#define RUNS 1000
#endif
#ifndef PARALLEL
#define PARALLEL 0
#endif

#define PREFETCH 1

#define H_TILE 0
#define POOLING 1


#define LIMIT 1e-2

#define CONV 0
#define PARTIAL_CONV 1 // under development
#define DW_CONV 2      // under development
#define GROUP_CONV 3   // under development
#define POOL 4
#define RELU 5

#ifndef LAYER
#define LAYER DW_CONV
#endif

//****************************************************************************
// The output of the block is stored in O
// The weights must have been copied into F_conv beforehand
inline void yolo_block(
    std::array<uint32_t, 2> in_dims, uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride, // Covolution parameters
    uint32_t output_channels,
    uint8_t t_pad,
    uint8_t b_pad,
    uint8_t l_pad,
    uint8_t r_pad,
    uint32_t kernel_size_pool,
    uint32_t stride_pool, // MaxPooling parameters
    uint8_t t_pad_pool,
    uint8_t b_pad_pool,
    uint8_t l_pad_pool,
    uint8_t r_pad_pool,
    float *I,
    float *F_conv,
    float *O_intermediate,
    float *O)
{
    small::Conv2D(kernel_size, stride,
                  t_pad, b_pad, l_pad, r_pad,
                  output_channels, input_channels,
                  in_dims[0], in_dims[1],
                  I, F_conv, O_intermediate);

    uint32_t o_h = output_dim(in_dims[0] + t_pad + b_pad, stride, kernel_size);
    uint32_t o_w = output_dim(in_dims[1] + l_pad + r_pad, stride, kernel_size);
    small::ReLUActivation(output_channels,
                          o_h, o_w,
                          O_intermediate, O_intermediate);
    small::Maxpool2D(kernel_size_pool, stride_pool,
                     t_pad_pool, b_pad_pool, l_pad_pool, r_pad_pool,
                     output_channels,
                     o_h, o_w,
                     O_intermediate, O);
    // small::ReLUActivation(output_channels, o_h, o_w, O, O);
}

//****************************************************************************
inline void conv_block(
    std::array<uint32_t, 2> in_dims, uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride, // Covolution parameters
    uint32_t output_channels,
    uint8_t t_pad,
    uint8_t b_pad,
    uint8_t l_pad,
    uint8_t r_pad,
    float *I,
    float *F_conv,
    float *O_intermediate,
    float *O)
{
    small::Conv2D(kernel_size, stride,
                  t_pad, b_pad, l_pad, r_pad,
                  output_channels, input_channels,
                  in_dims[0], in_dims[1],
                  I, F_conv, O_intermediate);

    uint32_t o_h = output_dim(in_dims[0] + t_pad + b_pad, stride, kernel_size);
    uint32_t o_w = output_dim(in_dims[1] + l_pad + r_pad, stride, kernel_size);
    small::ReLUActivation(output_channels,
                          o_h, o_w,
                          O_intermediate, O_intermediate);
}

#define REDUCTION_C(layer_num) layer_params[layer_num][0]
#define GROUP_C(layer_num) layer_params[layer_num][1]
#define GROUPS(layer_num) layer_params[layer_num][2]
#define REDUCTION_HW(layer_num) layer_params[layer_num][3]
#define STRIDE(layer_num) layer_params[layer_num][4]
#define TYPE(layer_num) layer_params[layer_num][9]

#define SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad) layer_params[layer_num][5] = t_pad, layer_params[layer_num][6] = b_pad, layer_params[layer_num][7] = l_pad, layer_params[layer_num][8] = r_pad;
#define PADDING(layer_num) layer_params[layer_num][5], layer_params[layer_num][6], layer_params[layer_num][7], layer_params[layer_num][8]

#define PADDING_TORCH(layer_num) layer_params[layer_num][7], layer_params[layer_num][8], layer_params[layer_num][5], layer_params[layer_num][6]

#define I_WIDTH(layer_num) intermediate_dims[layer_num][0]
#define I_HEIGHT(layer_num) intermediate_dims[layer_num][1]

#define O_HEIGHT(layer_num) (((I_HEIGHT(layer_num - 1) + layer_params[layer_num - 1][5] + layer_params[layer_num - 1][6]) - REDUCTION_HW(layer_num - 1)) / STRIDE(layer_num - 1) + 1)
#define O_WIDTH(layer_num) (((I_WIDTH(layer_num - 1) + layer_params[layer_num - 1][7] + layer_params[layer_num - 1][8]) - REDUCTION_HW(layer_num - 1)) / STRIDE(layer_num - 1) + 1)

#define OUTPUT_DIMS(layer_num)                  \
    {                                           \
        O_HEIGHT(layer_num), O_WIDTH(layer_num) \
    }

#define INPUT_NUMEL(layer_num) \
    (O_HEIGHT(layer_num) * O_WIDTH(layer_num) * GROUP_C(layer_num - 1) * GROUPS(layer_num - 1))


//****************************************************************************
//****************************************************************************
int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("USAGE: torch_pool <Input Height> <Input Width> <Input Channels> <Output Classes>");
        return 0;
    }

    // printf("layer %d %d %d \n", LAYER, uarch, W_ob);
    int C_i = atoi(argv[1]);

    uint32_t N = atol(argv[2]);
    uint32_t M = atol(argv[3]);

    // int C_i = atol(argv[3]);

    int num_classes = atol(argv[4]);

    uint32_t check_blocks = atol(argv[5]);
    if (num_classes % 16 != 0)
    {
        printf("Number of output classes must be a multiple of 16\n");
        exit(-1);
    }

    //int C_o = 32;
    //int padding_elements = 0;
    //int kernel_size = 3;
    //int stride = 1;
    // Create and Initialize Input tensors
    uint32_t input_dimensions = C_i * N * M;
    float *input_dc = alloc(input_dimensions);
    init(input_dc, input_dimensions);
    // std::cout<<a<<std::endl;

    std::vector<std::vector<uint64_t>> implementations;

    // Set up all pytorch layers
    // calculate total number of weight elements
    //uint32_t total_num_weights = 0;
    int layer_num = 0;
    int pool_layer_num = 0, conv_layer_num = 0;
    uint16_t layer_params[30][10] = {1};
    uint32_t max_numel_inter_0 = 0, max_numel_inter_1 = 0;
    uint32_t inter_dim = 0;

    std::vector<std::array<uint32_t, 2>> intermediate_dims;

    uint8_t t_pad, b_pad, r_pad, l_pad;

    intermediate_dims.push_back(std::array<uint, 2>({N, M}));
    // conv
    REDUCTION_C(layer_num) = C_i; // input channels
    GROUP_C(layer_num) = 16;      // output channels
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 3; // kernel size
    STRIDE(layer_num) = 1;       // stride
    small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), t_pad, b_pad);
    small::calc_padding(I_WIDTH(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad)
    layer_num++; // 1
    conv_layer_num++;
    inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_1 = (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;
    intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));

    // Pool
    REDUCTION_C(layer_num) = 1; // input channels
    GROUP_C(layer_num) = 1;
    GROUPS(layer_num) = GROUP_C(layer_num - 1); // output channels
    REDUCTION_HW(layer_num) = 2;                // kernel size
    STRIDE(layer_num) = 2;                      // stride
    small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), t_pad, b_pad);
    small::calc_padding(I_WIDTH(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad)
    layer_num++; // 2
    pool_layer_num++;
    inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_0 = (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;
    intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));

    // common set up for model architecture
    uint32_t ds_blocks = (check_blocks < 6) ? check_blocks : 6;
    int layer_strides[] = {2, 2, 2, 2, 2, 1};
    //int num_channels = 16;
    int channel_multiplier = 2;
    // dwise 1
    for (uint32_t ds_layer = 1U; ds_layer < ds_blocks; ds_layer++)  ///@todo verify start at 1
    {
        REDUCTION_C(layer_num) = GROUPS(layer_num - 1);
        ;                                                                 // input channels
        GROUP_C(layer_num) = channel_multiplier * REDUCTION_C(layer_num); // output channels
        GROUPS(layer_num) = 1;
        REDUCTION_HW(layer_num) = 3; // kernel size
        STRIDE(layer_num) = 1;       // stride
        small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), t_pad, b_pad);
        small::calc_padding(I_WIDTH(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad)
        layer_num++; // 3
        conv_layer_num++;
        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_1 = (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;
        intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));
        // std::cout << intermediate_dims[layer_num - 1][0] << " " << intermediate_dims[layer_num - 1][1] << std::endl;

        // Pool
        REDUCTION_C(layer_num) = 1; // input channels
        GROUP_C(layer_num) = 1;
        GROUPS(layer_num) = GROUP_C(layer_num - 1);  // output channels
        REDUCTION_HW(layer_num) = 2;                 // kernel size
        STRIDE(layer_num) = layer_strides[ds_layer]; // stride
        if (STRIDE(layer_num) == 1)
        {
            SET_PADDING(layer_num, 0, 1, 0, 1)
        }
        else
        {
            small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), t_pad, b_pad);
            small::calc_padding(I_WIDTH(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), l_pad, r_pad);
            SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad)
        }
        layer_num++; // 2
        pool_layer_num++;
        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_0 = (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;
        intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));
    }

    REDUCTION_C(layer_num) = GROUPS(layer_num - 1);
    GROUP_C(layer_num) = GROUPS(layer_num - 1) * 2;
    GROUPS(layer_num) = 1;       // GROUP_C(layer_num - 1);
    REDUCTION_HW(layer_num) = 3; // I_HEIGHT(layer_num);
    STRIDE(layer_num) = 1;
    small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), t_pad, b_pad);
    small::calc_padding(I_WIDTH(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad)
    layer_num++;
    conv_layer_num++;
    inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_1 = (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;
    intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));

    REDUCTION_C(layer_num) = GROUP_C(layer_num - 1);
    GROUP_C(layer_num) = GROUP_C(layer_num - 1);
    GROUPS(layer_num) = 1;       // GROUP_C(layer_num - 1);
    REDUCTION_HW(layer_num) = 3; // I_HEIGHT(layer_num);
    STRIDE(layer_num) = 1;
    small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), t_pad, b_pad);
    small::calc_padding(I_WIDTH(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad)
    layer_num++;
    conv_layer_num++;
    inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_0 = (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;

    intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));

    REDUCTION_C(layer_num) = GROUP_C(layer_num - 1);
    GROUP_C(layer_num) = num_classes;
    GROUPS(layer_num) = 1;       // GROUP_C(layer_num - 1);
    REDUCTION_HW(layer_num) = 1; // I_HEIGHT(layer_num);
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0)
    layer_num++;
    conv_layer_num++;
    inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_1 = (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;
    intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));

    //__________Summarize Network____________________________________
    auto layer_num_total = layer_num;
    printf("layer num, H_in, W_in, F_c, K, G, F_h, stride, t_pad, b_pad, l_pad, r_pad\n");
    for (auto i = 0; i < layer_num_total; i++)
    {
        printf("%d, ", i);
        printf("%d, %d ", I_HEIGHT(i), I_WIDTH(i));
        for (auto j = 0; j < 10; j++)
        {
            printf("%d, ", layer_params[i][j]);
        }
        printf("\b");
        printf("\b");
        // printf("input dims: %d %d ", I_HEIGHT(i+1), I_WIDTH(i+1));
        printf("\n");
    }

    printf("set up %d pool layers and %d conv layers total: %d\n", pool_layer_num, conv_layer_num, layer_num_total);


    //====================================SMaLL===================================

    //_____________________setup______________________________________________

    //  Copy layer weights to temporaries
    std::vector<float *> filter_ptrs;

    for (int l = 0; l < conv_layer_num; l++)
    {
        float *filter_ptr;
        // weights = layers[l]->weight; // conv_1x1->weight;
        uint32_t filter_dimensions = REDUCTION_HW(l) * REDUCTION_HW(l) * REDUCTION_C(l) * GROUP_C(l) * GROUPS(l);
        filter_ptr = alloc(filter_dimensions);
        init(filter_ptr, filter_dimensions);
        filter_ptrs.push_back(filter_ptr);
    }
    printf("set up weights\n");
    //_______________________________________inference_________________________


    // allocate space for intermediate outputs (use the max sizes calculated previously)
    printf("Size of intermediate buffers: %d %d\n", max_numel_inter_0, max_numel_inter_1);
    float *inter_0_dc = alloc(max_numel_inter_0);
    float *inter_1_dc = alloc(max_numel_inter_1);

    //uint32_t inter_h, inter_w;
    layer_num = 0;
    yolo_block(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
               REDUCTION_HW(layer_num),
               STRIDE(layer_num), // Params for the  convolution
               GROUP_C(layer_num),
               PADDING(layer_num),
               REDUCTION_HW(layer_num + 1),
               STRIDE(layer_num + 1), // Params for the pooling
               PADDING(layer_num + 1),
               input_dc,
               filter_ptrs[layer_num],
               inter_1_dc,
               inter_0_dc);

    // printf("intermediate_output shape: ");print_shape(layer_num, inter_0);
    layer_num += 2;
    for (uint32_t ds_layer = 1U; ds_layer < ds_blocks; ds_layer++)
    {
        printf("layer %d\n", layer_num);
        yolo_block(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
                   REDUCTION_HW(layer_num),
                   STRIDE(layer_num), // Params for the convolution
                   GROUP_C(layer_num),
                   PADDING(layer_num),
                   REDUCTION_HW(layer_num + 1),
                   STRIDE(layer_num + 1), // Params for the pooling
                   PADDING(layer_num + 1),
                   inter_0_dc,
                   filter_ptrs[ds_layer],
                   inter_1_dc,
                   inter_0_dc);

        layer_num += 2;

    }

    // //Epilogue layers with no pooling

    for (int epilogue_layer = 0; epilogue_layer < 3; epilogue_layer++)
    {
        conv_block(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
                   REDUCTION_HW(layer_num),
                   STRIDE(layer_num), // Params for the  convolution
                   GROUP_C(layer_num),
                   PADDING(layer_num),
                   inter_0_dc,
                   filter_ptrs[ds_blocks + epilogue_layer],
                   inter_1_dc,
                   inter_0_dc);
        layer_num++;

        float *tmp = inter_0_dc;
        inter_0_dc = inter_1_dc;
        inter_1_dc = tmp;
    }

    //float *output_dc = inter_0_dc;

    //_______________________________________end inference_________________________

    // Free allocated weight buffers

    for (size_t l = 0; l < filter_ptrs.size(); l++)
    {
        free(filter_ptrs[l]);
    }

    //====================================End SMaLL===================================

    //____________________________Correctness check_________________________________
    // bool check = 1;
    // std::vector<uint32_t>
    //     inter_0_dims, inter_1_dims;
    // for (int tens_dim_i = 0; tens_dim_i < inter_1.dim(); tens_dim_i++)
    // {
    //     inter_1_dims.push_back(inter_1.size(tens_dim_i));
    // }
    // check = check_eqivalence<C_ob, C_ib>(inter_1, 'o', inter_1_dims, inter_1_dc, LIMIT);
    // std::cout << inter_1_dims << std::endl;

    // assert(check == 1);

    // inter_0_dims.clear();
    // for (int tens_dim_i = 0; tens_dim_i < inter_0.dim(); tens_dim_i++)
    // {
    //     inter_0_dims.push_back(inter_0.size(tens_dim_i));
    // }
    // check = check_eqivalence<C_ob, C_ib>(inter_0, 'o', inter_0_dims, inter_0_dc, LIMIT);
    // std::cout << inter_0_dims << std::endl;

    // assert(check == 1);

    // Free input and output buffers
    free(input_dc);
    free(inter_0_dc);
    free(inter_1_dc);
}
