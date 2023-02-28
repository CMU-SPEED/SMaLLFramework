#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define QUANTIZED 1
typedef uint8_t dtype;
typedef int32_t atype;

struct
{                           // Structure declaration
    dtype *tensor;          // Member (int variable)
    float scale = 0.752941; // Member (string variable)
    int32_t offset = 0;
    int32_t multiplier = 1616928864;
    int lshift = 0;
    int rshift = 3;
    int zero = 0;
    int min_val = 255;
    int max_val = 0;
    uint8_t b = 8;
} typedef qint32_t; // Structure variable

typedef qint32_t qdtype;

#include "include/small.h"
#include "include/utils.h"

/// @todo Which of these defines are needed?
#ifndef RUNS
#define RUNS 1
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
// The output of the block is stored in I
// The weights must have been copied into F_1x1 and F_dw beforehand
template <bool scale_channels>
inline void resnet_block(
    uint32_t in_dims[2], uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride,          // DWise Covolution parameters
    uint32_t output_channels, // 1x1 Convolution parameters
    uint8_t t_pad_0,
    uint8_t b_pad_0,
    uint8_t l_pad_0,
    uint8_t r_pad_0,
    uint8_t t_pad_1,
    uint8_t b_pad_1,
    uint8_t l_pad_1,
    uint8_t r_pad_1,
    qdtype *I,
    qdtype *F_conv0,
    qdtype *F_conv1,
    qdtype *F_conv_1x1,
    qdtype *O_intermediate,
    qdtype *O)
{

    Conv2D<dtype, qdtype>(2, kernel_size, stride, t_pad_0, b_pad_0, l_pad_0, r_pad_0, output_channels, input_channels, in_dims[0], in_dims[1], I, F_conv0, O_intermediate);
    uint32_t o_h = output_dim(in_dims[0] + t_pad_0 + b_pad_0, stride, kernel_size);
    uint32_t o_w = output_dim(in_dims[1] + l_pad_0 + r_pad_0, stride, kernel_size);

    ReLUActivation<dtype, qdtype>(1, input_channels, o_h, o_w, O_intermediate, O_intermediate);
    if (scale_channels)
    {
        Conv2D<dtype, qdtype>(2, 1, stride, 0, 0, 0, 0, output_channels, input_channels, in_dims[0], in_dims[1], I, F_conv_1x1, O);
    }

    PartialConv2D<dtype, qdtype>(0, kernel_size, 1, t_pad_1, b_pad_1, l_pad_1, r_pad_1, output_channels, output_channels, o_h, o_w, O_intermediate, F_conv1, O);
    ReLUActivation<dtype, qdtype>(1, output_channels, o_h, o_w, O, O);
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

qdtype *model_inference(uint32_t layer_num_total, uint16_t layer_params[30][10], uint32_t intermediate_dims[30][2], qdtype *q_filter_ptrs, qdtype *q_input, qdtype *q_inter_0, qdtype *q_inter_1, qdtype *q_inter_2)
{
    auto layer_num = 0;
    Conv2D<dtype, qdtype>(0, REDUCTION_HW(layer_num), STRIDE(layer_num), PADDING(layer_num), GROUP_C(layer_num), REDUCTION_C(layer_num), I_HEIGHT(layer_num), I_WIDTH(layer_num), q_input, &q_filter_ptrs[layer_num], q_inter_0);
    layer_num++;
    ReLUActivation<dtype, qdtype>(1, GROUP_C(0), I_HEIGHT(layer_num), I_WIDTH(layer_num), q_inter_0, q_inter_0);
    resnet_block<0>(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
                    REDUCTION_HW(layer_num),
                    STRIDE(layer_num), // Params for the first convolution
                    GROUP_C(layer_num),
                    PADDING(layer_num),
                    PADDING(layer_num + 1),
                    q_inter_0,
                    &q_filter_ptrs[layer_num],
                    &q_filter_ptrs[layer_num + 1],
                    NULL,
                    q_inter_1,
                    q_inter_0);

    layer_num += 2;
    auto resnet_blocks = 3;
    auto num_filters = layer_num_total - 1;
    for (int ds_layer = 1; ds_layer < resnet_blocks; ds_layer++)
    {
        qdtype *O_intermediate = q_inter_2;
        resnet_block<1>(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
                        REDUCTION_HW(layer_num),
                        STRIDE(layer_num), // Params for the first convolution
                        GROUP_C(layer_num),
                        PADDING(layer_num), PADDING(layer_num + 1),
                        q_inter_0,
                        &q_filter_ptrs[layer_num],
                        &q_filter_ptrs[layer_num + 1],
                        &q_filter_ptrs[layer_num + 2],
                        q_inter_1,
                        O_intermediate);
        layer_num += 3;

        // Since channels were scaled, switch the pointers between inter_2 and inter_0
        q_inter_2 = q_inter_0;
        q_inter_0 = O_intermediate;
    }

    Maxpool2D<dtype, qdtype>(0, REDUCTION_HW(layer_num), STRIDE(layer_num), PADDING(layer_num), GROUPS(layer_num), I_HEIGHT(layer_num), I_WIDTH(layer_num), q_inter_0, q_inter_1);
    Conv2D<dtype,qdtype>(0, 1, 1, 0, 0, 0, 0, GROUP_C(layer_num_total - 1), REDUCTION_C(layer_num_total - 1), 1, 1, q_inter_1, &q_filter_ptrs[num_filters - 1], q_inter_0);

    return q_inter_0;
}

//****************************************************************************
//****************************************************************************
void inference() {
    int C_i = 3;

    uint32_t N = 32;
    uint32_t M = 32;

    int num_classes = 16;

    uint32_t input_dimensions = C_i * N * M;

    dtype *input_dc = (dtype *) alloc<dtype>(input_dimensions);
    init(input_dc, input_dimensions);
    qdtype q_input;
    quantized_init(&q_input, input_dimensions);
    q_input.tensor = input_dc;

    uint16_t layer_params[30][10] = {1};
    uint32_t intermediate_dims[30][2];

    uint8_t t_pad, b_pad, r_pad, l_pad;

    // Set up model parameters
    int layer_num = 0;
    uint32_t max_numel_inter_0 = 0, max_numel_inter_1 = 0;

    intermediate_dims[layer_num][0] = M;
    intermediate_dims[layer_num][1] = N;

    // conv
    REDUCTION_C(layer_num) = C_i; // input channels
    GROUP_C(layer_num) = 16;      // output channels
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 3; // kernel size
    STRIDE(layer_num) = 1;       // stride
    CALC_PADDING(I_HEIGHT(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), t_pad, b_pad);
    CALC_PADDING(I_WIDTH(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad)
    layer_num++; // 1

    intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
    intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);
    auto inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_0 = (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;

    // common set up for model architecture
    auto resnet_blocks = 3;
    int layer_strides[] = {1, 2, 2};
    // dwise 1
    for (int ds_layer = 0; ds_layer < resnet_blocks; ds_layer++)
    {
        int channel_multiplier = (ds_layer > 0) ? 2 : 1;

        uint32_t in_channels = GROUP_C(layer_num - 1); // output channels from the previous block

        REDUCTION_C(layer_num) = in_channels; // input channels
        GROUP_C(layer_num) = in_channels * channel_multiplier;
        GROUPS(layer_num) = 1;                       // output channels
        REDUCTION_HW(layer_num) = 3;                 // kernel size
        STRIDE(layer_num) = layer_strides[ds_layer]; // stride
        CALC_PADDING(I_HEIGHT(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), t_pad, b_pad);
        CALC_PADDING(I_WIDTH(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);
        layer_num++; // 2,4,7
        intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
        intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);

        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_1 = (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;

        REDUCTION_C(layer_num) = GROUP_C(layer_num - 1);
        GROUP_C(layer_num) = GROUP_C(layer_num - 1);
        GROUPS(layer_num) = 1;
        REDUCTION_HW(layer_num) = 3;
        STRIDE(layer_num) = 1;
        CALC_PADDING(I_HEIGHT(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), t_pad, b_pad);
        CALC_PADDING(I_WIDTH(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);
        layer_num++; // 3,5,8
        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_0 = (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;
        if (channel_multiplier != 1)
        {

            intermediate_dims[layer_num][0] = O_WIDTH(layer_num - 2);
            intermediate_dims[layer_num][1] = O_HEIGHT(layer_num - 2);
            REDUCTION_C(layer_num) = in_channels; // input channels
            GROUP_C(layer_num) = in_channels * channel_multiplier;
            GROUPS(layer_num) = 1;       // output channels
            REDUCTION_HW(layer_num) = 1; // kernel size
            STRIDE(layer_num) = 2;       // stride
            SET_PADDING(layer_num, 0, 0, 0, 0);
            layer_num++; // 6,9
            inter_dim = INPUT_NUMEL(layer_num);
            max_numel_inter_0 = (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;
        }
        intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
        intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);
    }
    // pooling dims
    REDUCTION_C(layer_num) = 1;
    GROUP_C(layer_num) = 1;
    GROUPS(layer_num) = GROUP_C(layer_num - 1);
    REDUCTION_HW(layer_num) = I_HEIGHT(layer_num);
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0)
    layer_num++;

    intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
    intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);

    REDUCTION_C(layer_num) = GROUPS(layer_num - 1);
    GROUP_C(layer_num) = num_classes;
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 1;
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0)
    layer_num++;

    uint32_t layer_num_total = layer_num;
    auto num_filters = layer_num_total - 1;

    //  Copy layer weights to temporaries
    qdtype q_filter_ptrs[30];
    for (uint32_t l = 0; l < num_filters - 1; l++)
    {
        uint32_t filter_dimensions = REDUCTION_HW(l) * REDUCTION_HW(l) * REDUCTION_C(l) * GROUP_C(l) * GROUPS(l);
        dtype *filter_ptr = (dtype *) alloc<dtype>(filter_dimensions);
        init(filter_ptr, filter_dimensions);
        quantized_init(&q_filter_ptrs[l], filter_dimensions);
        q_filter_ptrs[l].tensor = filter_ptr;
    }

    uint32_t filter_dimensions = GROUP_C(layer_num_total - 1) * REDUCTION_C(layer_num_total - 1);
    dtype *filter_fc_dc = (dtype *) alloc<dtype>(filter_dimensions);
    init(filter_fc_dc, filter_dimensions);
    quantized_init(&q_filter_ptrs[num_filters - 1], filter_dimensions);
    q_filter_ptrs[num_filters - 1].tensor = filter_fc_dc;

    // copy input
    // allocate space for intermediate outputs (use the max sizes calculated previously)
    dtype *inter_0_dc = (dtype *) alloc<dtype>(max_numel_inter_0 + C_ob*16*16*3);
    dtype *inter_1_dc = (dtype *) alloc<dtype>(max_numel_inter_1 + C_ob*16*16*3);
    dtype *inter_2_dc = (dtype *) alloc<dtype>((max_numel_inter_0 / 2) + C_ob*16*16*3);
    qdtype *output_dc; //= (dtype *) alloc<dtype>(num_classes);

    qdtype q_inter_0;
    quantized_init(&q_inter_0, max_numel_inter_0);
    q_inter_0.tensor = inter_0_dc;

    qdtype q_inter_1;
    quantized_init(&q_inter_1, max_numel_inter_1);
    q_inter_1.tensor = inter_1_dc;

    qdtype q_inter_2;
    quantized_init(&q_inter_2, (max_numel_inter_0 / 2));
    q_inter_2.tensor = inter_2_dc;

    mbed::Timer t;
    t.start();
    for (int r = 0; r < RUNS; r++) {
        output_dc = model_inference(layer_num_total, layer_params, intermediate_dims, &(q_filter_ptrs[0]), &q_input, &q_inter_0, &q_inter_1, &q_inter_2);
    }
    t.stop();
    Serial.println(t.elapsed_time().count());

    free_all();
}
