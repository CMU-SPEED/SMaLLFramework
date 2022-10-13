#pragma once

void check_Conv2D(int layer_num, int kernel_size, int stride, uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad, int output_channels, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr);

void check_PartialConv2D(int layer_num, int kernel_size, int stride, uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad, int output_channels, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr);

void check_GroupConv2D(int layer_num, int kernel_size, int stride, uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr);

void check_DepthwiseConv2D(int layer_num, int kernel_size, int stride, uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr);

void check_Maxpool2D(int layer_num, int kernel_size, int stride, uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad, int input_channels, int input_height, int input_width, float *input_ptr, float *output_ptr);

void check_ReLUActivation(int layer_num, int input_channels, int input_height, int input_width, float *input_ptr, float *output_ptr);

void check_Dense(int layer_num, int output_elements, int input_elements, float *input_ptr, float *filter_ptr, float *output_ptr);

