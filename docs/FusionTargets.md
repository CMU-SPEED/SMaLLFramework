# Fusion Targets and their layer Equivalents

## General Pattern for fused operations
![Fused operator Pattern](Images/Fusion%Recipe.jpg)


## Fusion Targets


### Non elementwsie Operation + Ewise operations
1. Convolution Layer with ReLU Activation

    <b>Interface Abstract</b>
    ```c++
    void Conv2D_ReLU(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,
        int output_channels, int input_channels,
        int input_height, int input_width,
        BufferT const &input_buf,
        BufferT const &filter_buf,
        BufferT &output_buf)
    ```    

    <b>Program with Atomic Interface Abstract Operations</b>    
    ```c++
    Conv2D(
        conv_kernel_height, conv_kernel_width, conv_stride,
        conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
        output_channels, input_channels,
        input_height, input_width,
        input_buf,
        filter_buf,
        output_buf
        );
    intermediate_height = output_dim();
    intermediate_width  = output_dim();
    ReLUActivation(
        output_channels,
        intermediate_height, intermediate_width,
        output_buf, output_buf
    );    
    ```
    <b>Layer Object Level</b>
    ```
    Conv2DLayer
    ReLULayer
    ```
2. Depthwise Convolution with ReLU Activation  

    <b> Interface Abstract </b>
    ```c++
    void DepthwiseConv2D_ReLU(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,
    int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
    ```
    <b>Layer Object Level</b>
    ```
    DepthwiseConv2DLayer
    ReLULayer
    ```
3. Convolution with Bias and ReLU Activation

    <b> Interface Abstract </b>
    ```c++
    void Conv2D_Bias_ReLU(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,
    int output_channels, int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT const &bias_buf,
    BufferT &output_buf)
    ```

    <b>Program with Atomic Interface Abstract Operations</b>

    ```c++
    intermediate_height = output_dim()
    intermediate_width  = output_dim()
    Bias(
        output_channels, 
        intermediate_height, intermediate_width,
        bias_buf, output_buf
    );
    PartialConv2D(
        conv_kernel_size, conv_kernel_size, conv_stride, 
        conv_t_pad,conv_b_pad, conv_l_pad, conv_r_pad, 
        output_channels, input_channels, 
        input_height, input_width,
        input_buf, filter_buf, output_buf
    );
    ReLUActivation(
        output_channels,
        intermediate_height, intermediate_width,
        output_buf, output_buf
    ); 
    ```

    <b> Layer Object Level </b>

    ```c++
    Conv2DLayer //(with bias)
    ReLULayer
    ```

### Non elementwsie Operation + Non Elementwise operations
1. Convolution fused with Depthwise Convolution

    <b> Interface Abstract </b>
    ```c++
    void Conv2D_DepthwiseConv2D(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

    int pool_kernel_height, int pool_kernel_width, int pool_stride,
    uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,

    int output_channels, int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &inter_output_buf,
    BufferT const &dwise_filter_buf,
    BufferT &output_buf)
    ```

    <b>Program with Atomic Interface Abstract Operations</b>


    ```c++
    Conv2D(
        conv_kernel_size, conv_kernel_size, conv_stride, 
        conv_t_pad,conv_b_pad, conv_l_pad, conv_r_pad, 
        output_channels, input_channels, 
        input_height, input_width, 
        input_buf, filter_buf, inter_output_buf
    );
    intermediate_height = output_dim();
    intermediate_width  = output_dim();
    DepthwiseConv2D(
        pool_kernel_height, pool_kernel_width, pool_stride,
        pool_t_pad,pool_b_pad, pool_l_pad, pool_r_pad,
        output_channels, 
        intermediate_height, intermediate_width,
        inter_output_buf, dwise_filter_buf, output_buf
    )
    ```  
    <b> Layer Object Level </b>
    ```c++
    Conv2DLayer
    DepthwiseConv2DLayer
    ```
2. Convolution fused with MaxPooling

    <b> Interface Abstract </b>
    ```c++
    void Conv2D_Maxpool2D(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

    int pool_kernel_height, int pool_kernel_width, int pool_stride,
    uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,

    int output_channels, int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &inter_output_buf,
    BufferT &output_buf)
    ```
    <b> Layer Object Level </b>
    ```c++
    Conv2dLayer
    MaxPool2DLayer
    ```
