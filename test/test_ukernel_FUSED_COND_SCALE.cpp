#define PARALLEL 1

#include <acutest.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>

#include <small.h>
#include <small/utils/Timer.hpp>

#include "test_utils.hpp"


namespace small {
namespace float_detail {

void test_correctness_FLOAT_FUSED_COND_SCALE_TILE(void)
{
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    using BufferT = FloatBuffer;

    size_t const INPUT_SIZE = FLOAT_W_ob * FLOAT_C_ib;
    BufferT input_buf(INPUT_SIZE);
    for (size_t ix = 0; ix < INPUT_SIZE; ++ix) input_buf[ix] = 2.0 * ((float)rand() / RAND_MAX) - 1;

    float const negative_slope = 0.01;

    const float * negative_buf = &negative_slope;

    size_t const OUTPUT_SIZE = FLOAT_W_ob * FLOAT_C_ob;
    BufferT output_buf(OUTPUT_SIZE);
    for (size_t ix = 0; ix < OUTPUT_SIZE; ++ix) output_buf[ix] = input_buf[ix];

    std::cout << std::endl;


    //==================================================
    FLOAT_DEF_TILE_C;

    FLOAT_LOAD_TILE_C(output_buf.data());

    FLOAT_INPLACE_COND_SCALE_TILE_C(negative_buf);

    FLOAT_STORE_TILE_C(output_buf.data());
    //==================================================

    for (dim_t ii = 0; ii < FLOAT_W_ob; ++ii)
    {
        for (dim_t jj = 0; jj < FLOAT_C_ob; ++jj)
        {
            size_t ix = ii*FLOAT_C_ob + jj;
            std::cout << output_buf[ix] << " " << ((output_buf[ix] >= 0) ? (input_buf[ix]) : (negative_slope*input_buf[ix])) << " " << input_buf[ix] << " " << ix << std::endl;
            TEST_CHECK((output_buf[ix] >= 0) ? (output_buf[ix] == input_buf[ix]) : (output_buf[ix] == negative_slope*input_buf[ix]));
        }
    }
#endif 
}

void test_correctness_FLOAT_FUSED_COND_SCALE_END(void)
{
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    using BufferT = FloatBuffer;

    size_t const INPUT_SIZE = FLOAT_W_ob * FLOAT_C_ib;
    BufferT input_buf(INPUT_SIZE);
    for (size_t ix = 0; ix < INPUT_SIZE; ++ix) input_buf[ix] = 2.0 * ((float)rand() / RAND_MAX) - 1;

    float const negative_slope = 0.01;

    const float * negative_buf = &negative_slope;

    size_t const OUTPUT_SIZE = FLOAT_W_ob * FLOAT_C_ob;
    BufferT output_buf(OUTPUT_SIZE);
    for (size_t ix = 0; ix < OUTPUT_SIZE; ++ix) output_buf[ix] = input_buf[ix];

    std::cout << std::endl;


    //==================================================
    FLOAT_DEF_END_C(FLOAT_W_ob, FLOAT_C_ob);
    c_tile_t *c_cur = c_tile;
    FLOAT_LOAD_END_C(output_buf.data(), FLOAT_W_ob, FLOAT_C_ob);

    FLOAT_INPLACE_COND_SCALE_END_C(negative_buf, c_cur, FLOAT_W_ob, FLOAT_C_ob);

    FLOAT_STORE_END_C(output_buf.data(), FLOAT_W_ob, FLOAT_C_ob);
    //==================================================

    for (dim_t ii = 0; ii < FLOAT_W_ob; ++ii)
    {
        for (dim_t jj = 0; jj < FLOAT_C_ob; ++jj)
        {
            size_t ix = ii*FLOAT_C_ob + jj;
            std::cout << output_buf[ix] << " " << ((output_buf[ix] >= 0) ? (input_buf[ix]) : (negative_slope*input_buf[ix])) << " " << input_buf[ix] << " " << ix << std::endl;
            TEST_CHECK((output_buf[ix] >= 0) ? (output_buf[ix] == input_buf[ix]) : (output_buf[ix] == negative_slope*input_buf[ix]));
        }
    }
#endif 
}

}
}


TEST_LIST = {
    {"correctness FLOAT_FUSED_COND_SCALE_TILE",
     small::float_detail::test_correctness_FLOAT_FUSED_COND_SCALE_TILE},
     {"correctness FLOAT_FUSED_COND_SCALE_END",
     small::float_detail::test_correctness_FLOAT_FUSED_COND_SCALE_END},
    {NULL, NULL}
};
