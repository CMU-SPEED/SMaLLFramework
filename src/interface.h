

extern "C" {

void Conv2D(int layer_num, int kernel_size, int stride, char padding, int output_channels, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr);


void PartialConv2D(int layer_num, int kernel_size, int stride, char padding, int output_channels, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr);


void GroupConv2D(int layer_num, int kernel_size, int stride, char padding, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr);


void DepthwiseConv2D(int layer_num, int kernel_size, int stride, char padding, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr);


void Maxpool2D(int layer_num, int kernel_size, int stride, char padding, int input_channels, int input_height, int input_width, float *input_ptr, float *output_ptr);

void ReLUActivation(int layer_num, int input_channels, int input_height, int input_width, float *input_ptr, float *output_ptr);

}