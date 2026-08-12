//********// Overloading to fuse partial spatial, single channel as well as partial spatial partial channel reductions
//********************************************************************
//****************************************************************************
// Fused abstract layer
//****************************************************************************
template <typename BufferT,
          dim_t _G_b,
          dim_t _K_b,
          dim_t _F_cb,
          dim_t _O_wb,
          dim_t _stride,
          dim_t _UNROLL,
          OpType op_type,
          int8_t op_class, //  2  (conv),  1  (dense,pool), or '0' (activation, upsample)
          bool rewrite_output,

         dim_t _G_b_1,
          dim_t _K_b_1,
          dim_t _F_cb_1,
          dim_t _O_wb_1,
          dim_t _stride_1,
          dim_t _UNROLL_1,
          OpType op_type_1,
          int8_t op_class_1, //  2  (conv),  1  (dense,pool), or '0' (activation, upsample)
          bool rewrite_output_1,

          OpType op_fused_single_element_before = OP_NONE,
          OpType op_fused_single_element_after = OP_NONE,
          dim_t _stride_before = 1,
          dim_t _stride_after = 1,

          OpType op_fused_single_element_before_1 = OP_NONE,
          OpType op_fused_single_element_after_1 = OP_NONE,
          dim_t _stride_before_1 = 1,
          dim_t _stride_after_1 = 1
          >
void fused_abstract_layer(
    Mapping<BufferT> *values_0, // Main Operation
    Mapping<BufferT> *values_1,
    dim_t I_h, // Input Height
    dim_t I_w, // Input Width

    BufferT const * /*__restrict__*/ I, // Data
    BufferT * /*__restrict__*/ O,
    BufferT *O_1) // 0 (partial conv, accum), 1 (otherwise)
{
    // Parameters for the 1st operation
    dim_t G = values_0->G;     // Output Channel Grouping
    dim_t K = values_0->K;     // Output Channels per group
    dim_t F_c = values_0->F_c; // Channel Reduction Dimension
    dim_t F_h = values_0->F_h; // Filter height
    dim_t F_w = values_0->F_w; // Filter width

    dim_t pad_top = values_0->pad_top; // Padding values
    dim_t pad_left = values_0->pad_left;
    dim_t pad_right = values_0->pad_right;
    dim_t pad_bottom = values_0->pad_bottom;

    BufferT const *__restrict__ F = values_0->F;               // Weight buffers 
    BufferT const *__restrict__ F_before = values_0->F_before; //(for fused elementwise operations)
    BufferT const *__restrict__ F_after = values_0->F_after;

    // Parameters for the second operations
    dim_t G_1 = values_1->G;     // Output Channel Grouping
    dim_t K_1 = values_1->K;     // Output Channels per group
    dim_t F_c_1 = values_1->F_c; // Channel Reduction Dimension
    dim_t F_h_1 = values_1->F_h; // Filter height
    dim_t F_w_1 = values_1->F_w; // Filter width

    dim_t pad_top_1 = values_1->pad_top; // Padding values
    dim_t pad_left_1 = values_1->pad_left;
    dim_t pad_right_1 = values_1->pad_right;
    dim_t pad_bottom_1 = values_1->pad_bottom;

    BufferT const *__restrict__ F_1 = values_1->F; // Weight buffers
    BufferT const *__restrict__ F_before_1 = values_1->F_before;
    BufferT const *__restrict__ F_after_1 = values_1->F_after;
    using ScalarT = typename BufferT::value_type;
    using AccumT = typename BufferT::accum_type;

    // Pointers to buffers inside Buffer class
    ScalarT const *I_buf = I->data(); //__restrict__ ?

    // Weight buffers for first operations
    ScalarT const *F_buf = nullptr;
    if constexpr (op_type == OP_CONV || op_type == OP_LEAKY_RELU || op_type == OP_MUL) // if (F != nullptr)
    {
        F_buf = F->data();
    }

    

    ScalarT const *F_before_buf = nullptr;
    if constexpr (op_fused_single_element_before == OP_UPSAMPLE || op_fused_single_element_before == OP_MUL || op_fused_single_element_before == OP_LEAKY_RELU)
    {
        F_before_buf = F_before->data();
        // printf("bias_buf: %f %f %f %f\n", F_before_buf[0], F_before_buf[1], F_before_buf[2], F_before_buf[3]);
    }
    ScalarT const *F_after_buf = nullptr;
    if constexpr (op_fused_single_element_after == OP_UPSAMPLE || op_fused_single_element_after == OP_MUL || op_fused_single_element_after == OP_LEAKY_RELU)
    {
        F_after_buf = F_after->data();
    }
    

    //Weights for second operations
    ScalarT const *F_buf_1 = nullptr;
    if constexpr (op_type_1 == OP_CONV || op_type_1 == OP_LEAKY_RELU || op_type_1 == OP_MUL) // if (F != nullptr)
    {
        F_buf_1 = F_1->data();
    }

    ScalarT const *F_before_buf_1 = nullptr;
    if constexpr (op_fused_single_element_before_1 == OP_UPSAMPLE || op_fused_single_element_before_1 == OP_MUL || op_fused_single_element_before_1 == OP_LEAKY_RELU)
    {
        F_before_buf_1 = F_before_1->data();
        // printf("bias_buf: %f %f %f %f\n", F_before_buf[0], F_before_buf[1], F_before_buf[2], F_before_buf[3]);
    }
    ScalarT const *F_after_buf_1 = nullptr;
    if constexpr (op_fused_single_element_after_1 == OP_UPSAMPLE || op_fused_single_element_after_1 == OP_MUL || op_fused_single_element_after_1 == OP_LEAKY_RELU)
    {
        F_after_buf_1 = F_after_1->data();
    }
    ScalarT *O_inter_buf = O->data();
    ScalarT *O_buf = O_1->data(); //__restrict__ ?

#if DEBUG == 1
    if (op_type == OP_CONV)
    {
        printf("conv class: %d \n", op_class);
    }
    else if (op_type == OP_MAX_POOL)
    {
        printf("pool class: %d \n", op_class);
    }
    else if (op_type == OP_RELU)
    {
        printf("activation class: %d \n", op_class);
    }
#endif

    // calculate output dimensions based on input params.
    constexpr dim_t _C_ib = _F_cb * _G_b;

    /*
     * Data layout (slowest to fastest changing dimensions):
     *    blocks of groups
     *       blocks of channels within groups
     *          blocks of weights in the same group
     *             spatial dimensions
     *                weights in the same group
     *                   weights across groups in a block
     *                      channels in a block
     *
     * I: [G/G_b,        F_c/F_cb, I_h, I_w, F_cb, G_b     ]
     * F: [G/G_b, K/K_b, F_c/F_cb, F_h, F_w, F_cb, G_b, K_b]
     * O: [G/G_b, K/K_b,           O_h, O_w,       G_b, K_b]
     */

    //************************************************************************
    // Deriving padding parameters for 1st operation

    //  To calculate offsets to next output row, next output block
    // @todo fix this in small::output_dim_new
    dim_t H_o_w_pad, W_o_w_pad;
    //@todo: fuse upsampling as an op_single_element_(before/after)
    //@todo: when fused, this computation goes into the kernel 
    if constexpr (op_type == OP_UPSAMPLE)
    {
        if constexpr (_stride == std::numeric_limits<dim_t>::max())
        {
            H_o_w_pad = I_h;
            W_o_w_pad = I_w;
        }
        else
        {
            H_o_w_pad = I_h * _stride;
            W_o_w_pad = I_w * _stride;
        }
    }
    else
    {
        H_o_w_pad = small::output_dim_new((I_h + pad_top + pad_bottom),
                                          _stride, F_h);
        W_o_w_pad = small::output_dim_new((I_w + pad_left + pad_right),
                                          _stride, F_w);
    }
    const dim_t O_h_w_pad = H_o_w_pad;
    const dim_t O_w_w_pad = W_o_w_pad;

    dim_t t_pad_el = pad_top / _stride + (pad_top % _stride != 0);
    dim_t l_pad_el = pad_left / _stride + (pad_left % _stride != 0);

    dim_t H_full_index = t_pad_el * _stride - pad_top;
    dim_t W_full_index = l_pad_el * _stride - pad_left;

    // Full kernel output elements
    dim_t H_o, W_o_full;
    if constexpr (op_type == OP_UPSAMPLE)
    {
        H_o = H_o_w_pad;
        W_o_full = W_o_w_pad;
    }
    else
    {
        H_o = small::output_dim_new((I_h - H_full_index), _stride, F_h);
        W_o_full = small::output_dim_new((I_w - W_full_index), _stride, F_w);
    }

    // back padding elements
    dim_t H_back_index = H_full_index + _stride * (H_o);
    dim_t W_back_index = W_full_index + _stride * (W_o_full);
    dim_t b_pad_el, r_pad_el;
    if constexpr (op_type == OP_UPSAMPLE)
    {
        b_pad_el = 0;
        r_pad_el = 0;
    }
    else
    {
        b_pad_el = small::output_dim_new((I_h + pad_bottom - H_back_index),
                                         _stride, F_h);
        r_pad_el = small::output_dim_new((I_w + pad_right - W_back_index),
                                         _stride, F_w);
    }

    const dim_t O_h = H_o;
    const dim_t O_w = W_o_full;
    //************************************************************************

    // setting up microkernel specific parameters
    const dim_t O_w_full = (O_w / _O_wb) * _O_wb;
    const dim_t O_w_left = O_w - O_w_full;
    const dim_t O_hxO_w = O_h_w_pad * O_w_w_pad;

    // Deriving padding parameters for 2nd operation

    //  To calculate offsets to next output row, next output block
    // @todo fix this in small::output_dim_new
    dim_t H_o_w_pad_1, W_o_w_pad_1;
    const dim_t I_h_1 = H_o_w_pad;
    const dim_t I_w_1 = W_o_w_pad;
    if constexpr (op_type_1 == OP_UPSAMPLE)
    {
        if constexpr (_stride_1 == std::numeric_limits<dim_t>::max())
        {
            H_o_w_pad_1 = I_h_1;
            W_o_w_pad_1 = I_w_1;
        }
        else
        {
            H_o_w_pad_1 = I_h_1 * _stride_1;
            W_o_w_pad_1 = I_h_1 * _stride_1;
        }
    }
    else
    {
        H_o_w_pad_1 = small::output_dim_new((I_h_1 + pad_top_1 + pad_bottom_1),
                                            _stride_1, F_h_1);
        W_o_w_pad_1 = small::output_dim_new((I_w_1 + pad_left_1 + pad_right_1),
                                            _stride_1, F_w_1);
    }
    const dim_t O_h_w_pad_1 = H_o_w_pad_1;
    const dim_t O_w_w_pad_1 = W_o_w_pad_1;

    dim_t t_pad_el_1 = pad_top_1 / _stride_1 + (pad_top_1 % _stride_1 != 0);
    dim_t l_pad_el_1 = pad_left_1 / _stride_1 + (pad_left_1 % _stride_1 != 0);

    dim_t H_full_index_1 = t_pad_el_1 * _stride_1 - pad_top_1;
    dim_t W_full_index_1 = l_pad_el_1 * _stride_1 - pad_left_1;

    // Full kernel output elements
    dim_t H_o_1, W_o_full_1;
    if constexpr (op_type_1 == OP_UPSAMPLE)
    {
        H_o_1 = H_o_w_pad_1;
        W_o_full_1 = W_o_w_pad_1;
    }
    else
    {
        H_o_1 = small::output_dim_new((I_h_1 - H_full_index_1), _stride_1, F_h_1);
        W_o_full_1 = small::output_dim_new((I_w_1 - W_full_index_1), _stride_1, F_w_1);
    }

    // back padding elements
    dim_t H_back_index_1 = H_full_index_1 + _stride_1 * (H_o_1);
    dim_t W_back_index_1 = W_full_index_1 + _stride_1 * (W_o_full_1);
    dim_t b_pad_el_1, r_pad_el_1;
    if constexpr (op_type == OP_UPSAMPLE)
    {
        b_pad_el_1 = 0;
        r_pad_el_1 = 0;
    }
    else
    {
        b_pad_el_1 = small::output_dim_new((I_h_1 + pad_bottom_1 - H_back_index_1),
                                           _stride_1, F_h_1);
        r_pad_el_1 = small::output_dim_new((I_w_1 + pad_right_1 - W_back_index_1),
                                           _stride_1, F_w_1);
    }

    const dim_t O_h_1 = H_o_1;
    const dim_t O_w_1 = W_o_full_1;
    //************************************************************************

    // setting up microkernel specific parameters
    const dim_t O_w_full_1 = (O_w_1 / _O_wb) * _O_wb;
    const dim_t O_w_left_1 = O_w_1 - O_w_full_1;
    const dim_t O_hxO_w_1 = O_h_w_pad_1 * O_w_w_pad_1;

#if DEBUG == 1
    printf("\t\t I_h %d I_w %d F_C %d G %d \n", I_h, I_w, F_c, G);
    printf("\t\t O_h_pad: %d O_w_w_pad %d \n", O_h_w_pad, O_w_w_pad);
    printf("O_h %d O_w %d O_w_left %d \n", O_h, O_w_full, O_w_left);

    printf("bottom padding index into input: %d \t bottom padding elements: %d \n",
           H_back_index, b_pad_el);
    printf("no padding index into input: %d \t top padding elements: %d \n",
           H_full_index, t_pad_el);
    printf("right padding index into input: %d \t right padding elements: %d \n",
           W_back_index, r_pad_el);
    printf("no padding index into input: %d \t left padding elements: %d \n",
           W_full_index, l_pad_el);
    printf("O_w_full: %d O_w_left: %d \n", O_w_full, O_w_left);
    printf("params: F_Cb %d G_b %d K_b %d\n", _F_cb, _G_b, _K_b);
    printf("rewrite output?: %d, op type/class:  %d/%d\n",
           rewrite_output, op_type, op_class);
#endif

    // Set up parallelism for the channel loops

    //  Get total available threads
    int N = 1;
#if PARALLEL == 1
    char const *env_nt(std::getenv("OMP_NUM_THREADS"));
    if (nullptr != env_nt)
    {
        N = atoi(std::getenv("OMP_NUM_THREADS"));
    }
#endif

    int T_channel = N, T_group = 1, T_height = 1;

    // If dwise, parallelize on groups
    if (K == 1)
    {
        T_channel = 1;
        T_group = N;
    }

    // create parallel region with all threads
#if PARALLEL == 1
#pragma omp parallel num_threads(N)
#endif
    {
#if PARALLEL == 1
        auto t_id = omp_get_thread_num();
#else
        auto t_id = 0;
#endif
        auto height_tid = t_id % T_height;
        auto channel_tid = ((t_id) / (T_height)) % T_channel;
        auto group_tid = ((t_id / (T_channel * T_height))) % T_group;

        // loops over output channels
        for (index_t g = group_tid; g < G / _G_b; g += T_group)
        {
            ScalarT const *I_group;
            if constexpr (op_type == OP_UPSAMPLE && _stride == std::numeric_limits<dim_t>::max())
            {
                I_group = I_buf + g * (F_c * 1 * 1 * _G_b);
            }
            else
            {
                I_group = I_buf + g * (F_c * I_h * I_w * _G_b);
            }
            ScalarT *O_group = O_inter_buf + g * (K * O_hxO_w * _G_b);
            // if leaky relu, the weight pointer does not change with the group id

            ScalarT const *F_group;
            if constexpr ((op_type == OP_LEAKY_RELU) || (op_type == OP_MUL))
            {
                F_group = F_buf;
            }
            else
            {
                F_group = F_buf + g * (K * F_c * F_h * F_w * _G_b);
            }

            // resuse O_group as a uint32_t array
            for (index_t k = channel_tid; k < K / _K_b; k += T_channel)
            {
                ScalarT const *I_channel_block_output =
                    I_group + 0;
                ScalarT const *F_channel_block_output =
                    F_group + k * (F_c * F_h * F_w * _G_b * _K_b);
//@todo: this indexing should change to save intermediate memory
                ScalarT *O_channel_block_output =
                    O_group + k * (O_hxO_w * _G_b * _K_b);

                //@todo fix the filter height and width as necessary (they should be one because it is a single element reduction)
                ScalarT const *F_before_buf_group = F_before_buf + (g * K + k) * (1 * 1 * _G_b * _K_b);
                ScalarT const *F_after_buf_group = F_after_buf + (g * K + k) * (1 * 1 * _G_b * _K_b);



                //************************************************************
                // Loop over input channel reduction
                for (index_t i = 0; i < (F_c / _F_cb) - 1; i++)
                {
                    bool first = rewrite_output && (i == 0);

                    ScalarT const *I_channel_block_input =
                        I_channel_block_output + i * (I_h * I_w * _F_cb * _G_b);
                    ScalarT const *F_channel_block_input =
                        F_channel_block_output + i * (F_h * F_w * _F_cb * _G_b * _K_b);
                    ScalarT *O_channel_block_input =
                        O_channel_block_output + 0;

                    // Loops over spatial dimensions of output

                    // Prologue with top padding
                    ScalarT const *I_row_top = I_channel_block_input;
                    ScalarT const *F_row_top = F_channel_block_input + 0;
                    AccumT *O_row_top = O_channel_block_input; // ScalarT --> AccumT

                    kernel_top<ScalarT, AccumT,
                               _G_b, _K_b, _F_cb, _O_wb, _stride,
                               _UNROLL, op_type, op_class>(
                        first,
                        F_h,
                        F_w,
                        I_w * _C_ib,
                        t_pad_el,
                        pad_top,
                        W_full_index,
                        l_pad_el,
                        pad_left,
                        O_w_w_pad,
                        O_w_full,
                        O_w_left,
                        r_pad_el,
                        pad_right,
                        I_row_top,
                        F_row_top,
                        O_row_top);

                    ScalarT const *I_row_full =
                        I_row_top + H_full_index * I_w * (_F_cb * _G_b);
                    AccumT *O_row_full =
                        O_row_top + t_pad_el * O_w_w_pad * (_G_b * _K_b); // ScalarT --> AccumT

                    // Steady State over rows
                    for (index_t j = height_tid; j < O_h; j += T_height)
                    {
                        ScalarT const *I_row;
                        // @todo cast index calculation as int and make stride a float value.
                        // I_x = I_x + (int)(j * _stride) * (<remaining dimensions>)
                        if constexpr (op_type == OP_UPSAMPLE)
                        {
                            I_row = I_row_full + (j / _stride) * (I_w * _F_cb * _G_b);
                        }
                        else
                        {
                            I_row = I_row_full + (j * _stride) * (I_w * _F_cb * _G_b);
                        }
                        ScalarT const *F_row = F_channel_block_input + 0;
                        AccumT *O_row =
                            O_row_full + j * (O_w_w_pad * _G_b * _K_b); // ScalarT --> AccumT
                        // Prologue with left padding
                        kernel_left<ScalarT, AccumT,
                                    _G_b, _K_b, _F_cb, _O_wb, _stride,
                                    _UNROLL, op_type, op_class>(
                            first,
                            F_h,
                            F_w,
                            I_w * _C_ib,
                            l_pad_el,
                            pad_left,
                            I_row,
                            F_row,
                            O_row,
                            0,
                            0);

                        ScalarT const *I_col_full =
                            I_row + W_full_index * (_F_cb * _G_b);
                        AccumT *O_col_full = O_row + l_pad_el * (_G_b * _K_b); // ScalarT --> AccumT
                        // Steady State with microkernel
                        for (index_t l = 0; l < O_w_full; l += _O_wb)
                        {
                            ScalarT const *I_col;
                            // @todo cast index calculation as int and make stride a float value.
                            // I_x = I_x + (int)(j * _stride) * (<remaining dimensions>)
                            if constexpr (op_type == OP_UPSAMPLE)
                            {
                                I_col = I_col_full + (l / _stride) * (_F_cb * _G_b);
                            }
                            else
                            {
                                I_col = I_col_full + (l * _stride) * (_F_cb * _G_b);
                            }
                            ScalarT const *F_col = F_row + 0;
                            AccumT *O_col = O_col_full + l * (_G_b * _K_b); // ScalarT --> AccumT

                            kernel<ScalarT, AccumT,
                                   _G_b, _K_b, _F_cb, _O_wb, _stride,
                                   _UNROLL, op_type, op_class>(
                                first,
                                F_h,
                                F_w,
                                I_w * _C_ib,
                                I_col,
                                F_col,
                                O_col,
                                0,
                                0,
                                0,
                                0);
                        }

#if DEBUG
                        printf(" end  kernel\n");
#endif

                        // Epilogue for microkernel + right padding elements
                        ScalarT const *I_col_left;
                        if constexpr (op_type == OP_UPSAMPLE)
                        {
                            I_col_left =
                                I_col_full + (O_w_full / _stride) * (_F_cb * _G_b);
                        }
                        else
                        {
                            I_col_left =
                                I_col_full + (O_w_full * _stride) * (_F_cb * _G_b);
                        }

                        ScalarT const *F_col_left = F_row + 0;
                        AccumT *O_col_left = O_col_full + O_w_full * (_G_b * _K_b); // ScalarT --> AccumT

#if DEBUG
                        printf(" calling right\n");
#endif
                        kernel_right<ScalarT, AccumT,
                                     _G_b, _K_b, _F_cb, _O_wb, _stride,
                                     _UNROLL, op_type, op_class>(
                            first,
                            F_h,
                            F_w,
                            I_w * _C_ib,
                            O_w_left,
                            r_pad_el,
                            pad_right,
                            I_col_left,
                            F_col_left,
                            O_col_left,
                            0,
                            0);
                    }
                    // Epilogue with bottom padding
                    ScalarT const *I_row_bot;
                    // @todo cast index calculation as int and make stride a float value.
                    // I_x = I_x + (int)(j * _stride) * (<remaining dimensions>)
                    if constexpr (op_type == OP_UPSAMPLE)
                    {
                        I_row_bot =
                            I_row_full + (O_h * _stride) * (I_w * _F_cb * _G_b);
                    }
                    else
                    {
                        I_row_bot =
                            I_row_full + (O_h * _stride) * (I_w * _F_cb * _G_b);
                    }
                    ScalarT const *F_row_bot = F_channel_block_input + 0;
                    AccumT *O_row_bot = O_row_full + O_h * (O_w_w_pad * _G_b * _K_b); // ScalarT --> AccumT

                    kernel_bottom<ScalarT, AccumT,
                                  _G_b, _K_b, _F_cb, _O_wb, _stride,
                                  _UNROLL, op_type, op_class>(
                        first,
                        F_h,
                        F_w,
                        I_w * _C_ib,
                        b_pad_el,
                        pad_bottom,
                        W_full_index,
                        l_pad_el,
                        pad_left,
                        O_w_w_pad,
                        O_w_full,
                        O_w_left,
                        r_pad_el,
                        pad_right,
                        I_row_bot,
                        F_row_bot,
                        O_row_bot);
                }

                // Last block of input channels:
                // Loop over input channel reduction
                for (index_t i = (F_c / _F_cb) - 1; i < (F_c / _F_cb); i++)
                {
                    bool first = rewrite_output && (i == 0);

                    bool first_1 = rewrite_output_1;
                    ScalarT const *I_channel_block_input =
                        I_channel_block_output + i * (I_h * I_w * _F_cb * _G_b);
                    ScalarT const *F_channel_block_input =
                        F_channel_block_output + i * (F_h * F_w * _F_cb * _G_b * _K_b);
                    ScalarT *O_channel_block_input =
                        O_channel_block_output + 0;

// Pointers to current group for second operation
                    ScalarT * F_output_channel_block_1 = F_1 + (g*K_1 + k) * (F_h_1 *F_w_1*_G_b_1*_K_b_1) ;
                    ScalarT * F_before_buf_group_1 = F_before_buf_1 + (g * K_1 + k) * ( 1 * 1 * _G_b_1 * _K_b_1 );
                    ScalarT *F_after_buf_group_1 = F_after_buf_1 + (g * K_1 + k) * (1 * 1 * _G_b_1 * _K_b_1);

                    // Pointer to output group for second operation
                    // pointer into fused array
                    ScalarT *O_channel_block_output_1 = O_buf + (g * K_1 + k) * (O_hxO_w_1 * _G_b_1 * _K_b_1);

                    // Loops over spatial dimensions of output

                    // Prologue with top padding
                    ScalarT const *I_row_top = I_channel_block_input;
                    ScalarT const *F_row_top = F_channel_block_input + 0;
                    AccumT *O_row_top = O_channel_block_input; // ScalarT --> AccumT

                    kernel_top<ScalarT, AccumT,
                               _G_b, _K_b, _F_cb, _O_wb, _stride,
                               _UNROLL, op_type, op_class, op_fused_single_element_before, op_fused_single_element_after, _stride_before, _stride_after>(
                        first,
                        F_h,
                        F_w,
                        I_w * _C_ib,
                        t_pad_el,
                        pad_top,
                        W_full_index,
                        l_pad_el,
                        pad_left,
                        O_w_w_pad,
                        O_w_full,
                        O_w_left,
                        r_pad_el,
                        pad_right,
                        I_row_top,
                        F_row_top,
                        O_row_top,
                        F_before_buf_group,
                        F_after_buf_group);

                    ScalarT const *I_row_full =
                        I_row_top + H_full_index * I_w * (_F_cb * _G_b);
                    AccumT *O_row_full =
                        O_row_top + t_pad_el * O_w_w_pad * (_G_b * _K_b); // ScalarT --> AccumT

                    // Peel (F_h_1 - S_h_1) -  t_pad_el + S_h_1
                    for (index_t j = 0; j < (F_h_1 - S_h_1) - t_pad_el + S_h_1; j++)
                    {
                        ScalarT const *I_row;
                        // tile over S_h_1
                        if constexpr (op_type == OP_UPSAMPLE)
                        {
                            I_row = I_row_full + (j / _stride) * (I_w * _F_cb * _G_b);
                        }
                        else
                        {
                            I_row = I_row_full + (j * _stride) * (I_w * _F_cb * _G_b);
                        }
                        ScalarT const *F_row = F_channel_block_input + 0;
                        AccumT *O_row =
                            O_row_full + j * (O_w_w_pad * _G_b * _K_b); // ScalarT --> AccumT
                        // Prologue with left padding
                        kernel_left<ScalarT, AccumT,
                                    _G_b, _K_b, _F_cb, _O_wb, _stride,
                                    _UNROLL, op_type, op_class, op_fused_single_element_before, op_fused_single_element_after, _stride_before, _stride_after>(
                            first,
                            F_h,
                            F_w,
                            I_w * _C_ib,
                            l_pad_el,
                            pad_left,
                            I_row,
                            F_row,
                            O_row,
                            0,
                            0,
                            F_before_buf_group,
                            F_after_buf_group);

                        ScalarT const *I_col_full =
                            I_row + W_full_index * (_F_cb * _G_b);
                        AccumT *O_col_full = O_row + l_pad_el * (_G_b * _K_b); // ScalarT --> AccumT
                        // Steady State with microkernel
                        for (index_t l = 0; l < O_w_full; l += _O_wb)
                        {
                            ScalarT const *I_col;
                            // @todo cast index calculation as int and make stride a float value.
                            // I_x = I_x + (int)(j * _stride) * (<remaining dimensions>)
                            if constexpr (op_type == OP_UPSAMPLE)
                            {
                                I_col = I_col_full + (l / _stride) * (_F_cb * _G_b);
                            }
                            else
                            {
                                I_col = I_col_full + (l * _stride) * (_F_cb * _G_b);
                            }
                            ScalarT const *F_col = F_row + 0;
                            AccumT *O_col = O_col_full + l * (_G_b * _K_b); // ScalarT --> AccumT

                            kernel<ScalarT, AccumT,
                                   _G_b, _K_b, _F_cb, _O_wb, _stride,
                                   _UNROLL, op_type, op_class, op_fused_single_element_before, op_fused_single_element_after, _stride_before, _stride_after>(
                                first,
                                F_h,
                                F_w,
                                I_w * _C_ib,
                                I_col,
                                F_col,
                                O_col,
                                0,
                                0,
                                0,
                                0,
                                F_before_buf_group,
                                F_after_buf_group);
                        }

#if DEBUG
                        printf(" end  kernel\n");
#endif

                        // Epilogue for microkernel + right padding elements
                        ScalarT const *I_col_left;
                        if constexpr (op_type == OP_UPSAMPLE)
                        {
                            I_col_left =
                                I_col_full + (O_w_full / _stride) * (_F_cb * _G_b);
                        }
                        else
                        {
                            I_col_left =
                                I_col_full + (O_w_full * _stride) * (_F_cb * _G_b);
                        }

                        ScalarT const *F_col_left = F_row + 0;
                        AccumT *O_col_left = O_col_full + O_w_full * (_G_b * _K_b); // ScalarT --> AccumT

#if DEBUG
                        printf(" calling right\n");
#endif
                        kernel_right<ScalarT, AccumT,
                                     _G_b, _K_b, _F_cb, _O_wb, _stride,
                                     _UNROLL, op_type, op_class, op_fused_single_element_before, op_fused_single_element_after, _stride_before, _stride_after>(
                            first,
                            F_h,
                            F_w,
                            I_w * _C_ib,
                            O_w_left,
                            r_pad_el,
                            pad_right,
                            I_col_left,
                            F_col_left,
                            O_col_left,
                            0,
                            0,
                            F_before_buf_group,
                            F_after_buf_group);
                    }
                    // Peel t_pad_el_1 and j_1
                    // Kernel top for second operation
                    ScalarT const *I_row_top_1 = O_channel_block_input;
                    ScalarT const *F_row_top_1 = F_channel_block_input_1 + 0;
                    AccumT *O_row_top_1 = O_channel_block_input_1; // ScalarT --> AccumT
                    kernel_top<ScalarT, AccumT,
                               _G_b_1, _K_b_1, _F_cb_1, _O_wb_1, _stride_1,
                               _UNROLL_1, op_type_1, op_class_1, op_fused_single_element_before_1, op_fused_single_element_after_1, _stride_before_1, _stride_after_1>(
                        first_1,
                        F_h_1,
                        F_w_1,
                        I_w_1 * _C_ib,
                        t_pad_el_1,
                        pad_top_1,
                        W_full_index_1,
                        l_pad_el_1,
                        pad_left_1,
                        O_w_w_pad_1,
                        O_w_full_1,
                        O_w_left_1,
                        r_pad_el_1,
                        pad_right_1,
                        I_row_top_1,
                        F_row_top_1,
                        O_row_top_1,
                        F_before_buf_group_1,
                        F_after_buf_group_1);

                    ScalarT const *I_row_full_1 =
                        I_row_top_1 + H_full_index_1 * I_w_1 * (_C_ib);
                    AccumT *O_row_full_1 =
                        O_row_top_1 + t_pad_el_1 * O_w_w_pad_1 * (_C_ob);

                    // First full reduction row of second operation
                    for (index_t j = 0; j < 1; j++)
                    {
                        // Upsample would be fused ( so the check can be removed)
                        ScalarT const *I_row_1 = I_row_full_1 + (j * _stride_1) * (I_w_1 * _F_cb_1 * _G_b_1);
                        ScalarT const *F_row_1 = F_channel_block_input_1 + 0;
                        AccumT *O_row_1 =
                            O_row_full_1 + j * (O_w_w_pad_1 * _C_ob);
                        // Prologue with left padding
                        kernel_left<ScalarT, AccumT,
                                    _G_b_1, _K_b_1, _F_cb_1, _O_wb, _stride_1,
                                    _UNROLL_1, op_type_1, op_class_1, op_fused_single_element_before_1, op_fused_single_element_after_1, _stride_before_1, _stride_after_1>(
                            first_1,
                            F_h_1,
                            F_w_1,
                            I_w_1 * _C_ib,
                            l_pad_el_1,
                            pad_left_1,
                            I_row_1,
                            F_row_1,
                            O_row_1,
                            0,
                            0,
                            F_before_buf_group_1,
                            F_after_buf_group_1);

                        ScalarT const *I_col_full_1 =
                            I_row_1 + W_full_index_1 * (_C_ib);
                        AccumT *O_col_full_1 = O_row_1 + l_pad_el_1 * (_C_ob); // ScalarT --> AccumT
                        // Steady State with microkernel
                        for (index_t l = 0; l < O_row_full_1; l += _O_wb)
                        {
                            ScalarT const *I_col_1 = I_col_full_1 + (l * _stride_1) * (_F_cb_1 * _G_b_1);

                            ScalarT const *F_col_1 = F_row_1 + 0;
                            AccumT *O_col_1 = O_col_full_1 + l * (_C_ob); // ScalarT --> AccumT

                            kernel<ScalarT, AccumT,
                                   _G_b_1, _K_b_1, _F_cb_1, _O_wb, _stride_1,
                                   _UNROLL_1, op_type_1, op_class_1, op_fused_single_element_after_1, _stride_before_1, _stride_after_1>(
                                first_1,
                                F_h_1,
                                F_w_1,
                                I_w_1 * _C_ib,
                                I_col_1,
                                F_col_1,
                                O_col_1,
                                0,
                                0,
                                0,
                                0,
                                F_before_buf_group_1,
                                F_after_buf_group_1);
                        }

#if DEBUG
                        printf(" end  kernel\n");
#endif

                        // Epilogue for microkernel + right padding elements
                        ScalarT const *I_col_left_1 = I_col_full_1 + (O_w_full_1 * _stride_1) * (_F_cb_1 * _G_b_1);
                        ScalarT const *F_col_left_1 = F_row_1 + 0;
                        AccumT *O_col_left_1 = O_col_full_1 + O_w_full_1 * (_C_ob); // ScalarT --> AccumT

#if DEBUG
                        printf(" calling right\n");
#endif
                        kernel_right<ScalarT, AccumT,
                                     _G_b_1, _K_b_1, _F_cb_1, _O_wb, _stride_1,
                                     _UNROLL_1, op_type_1, op_class_1, op_fused_single_element_after_1, _stride_before_1, _stride_after_1>(
                            first_1,
                            F_h_1,
                            F_w_1,
                            I_w_1 * _C_ib,
                            O_w_left_1,
                            r_pad_el_1,
                            pad_right_1,
                            I_col_left_1,
                            F_col_left_1,
                            O_col_left_1,
                            0,
                            0,
                            F_before_buf_group_1,
                            F_after_buf_group_1);
                    }
                    //
                    //@todo: implement more than one row in the single channel reduction
                    // Epilogue with bottom padding
                    ScalarT const *I_row_bot;
                    // @todo cast index calculation as int and make stride a float value.
                    // I_x = I_x + (int)(j * _stride) * (<remaining dimensions>)
                    if constexpr (op_type == OP_UPSAMPLE)
                    {
                        I_row_bot =
                            I_row_full + (O_h * _stride) * (I_w * _F_cb * _G_b);
                    }
                    else
                    {
                        I_row_bot =
                            I_row_full + (O_h * _stride) * (I_w * _F_cb * _G_b);
                    }
                    ScalarT const *F_row_bot = F_channel_block_input + 0;
                    AccumT *O_row_bot = O_row_full + O_h * (O_w_w_pad * _G_b * _K_b); // ScalarT --> AccumT

                    kernel_bottom<ScalarT, AccumT,
                                  _G_b, _K_b, _F_cb, _O_wb, _stride,
                                  _UNROLL, op_type, op_class, op_fused_single_element_before, op_fused_single_element_after, _stride_before, _stride_after>(
                        first,
                        F_h,
                        F_w,
                        I_w * _C_ib,
                        b_pad_el,
                        pad_bottom,
                        W_full_index,
                        l_pad_el,
                        pad_left,
                        O_w_w_pad,
                        O_w_full,
                        O_w_left,
                        r_pad_el,
                        pad_right,
                        I_row_bot,
                        F_row_bot,
                        O_row_bot,
                        F_before_buf_group,
                        F_after_buf_group);
                }
            }
        }
    }
}
