#define PARALLEL 1

#include <acutest.h>

#include <cmath>
#include <cstdint>

#include <small.h>

namespace small {
namespace float_detail {

namespace {

float reference_exp(float x)
{
    return std::exp(x);
}

} // namespace


void test_correctness_range_reduced_exp(void)
{
    using BufferT = FloatBuffer;

    constexpr size_t num_samples = FLOAT_W_ob * FLOAT_C_ob;
    constexpr float min_input = 2.0f;
    constexpr float max_input = 10.0f;
    constexpr float step = (max_input - min_input) /
                           static_cast<float>(num_samples - 1);
    constexpr float abs_tolerance = 5.0e-3f;
    constexpr float rel_tolerance = 5.0e-3f;


    BufferT input_buf(num_samples);
    BufferT output_buf(num_samples);


    for (size_t ix = 0; ix < num_samples; ++ix)
    {
        float input = min_input + static_cast<float>(ix) * step;

        input_buf[ix] = input;
    }



    FLOAT_DEF_TILE_C(FLOAT_W_ob, FLOAT_C_ob);
    FLOAT_LOAD_TILE_C(input_buf.data(), FLOAT_W_ob, FLOAT_C_ob);
    
    FLOAT_EXP_RR_TILE_C(input_buf.data());

    FLOAT_STORE_TILE_C(output_buf.data(), FLOAT_W_ob, FLOAT_C_ob);
    
    float previous_output = 0.0f;
    for (size_t ix = 0; ix < num_samples; ++ix)
    {
        float expected = reference_exp(input_buf[ix]);
        float actual = output_buf[ix];//std::ldexp(output_buf[ix], exponent_buf[ix]);
        float abs_error = std::abs(actual - expected);
        float rel_error = abs_error / (std::abs(expected) + 1.0e-12f);

        TEST_CHECK(actual > 0.0f);
        printf("input: %f, expected: %f, actual: %f, abs_error: %f, rel_error: %f\n",
               input_buf[ix], expected, actual, abs_error, rel_error);
        TEST_CHECK(abs_error < abs_tolerance || rel_error < rel_tolerance);

        if (ix > 0)
        {
            TEST_CHECK(actual >= previous_output);
        }
        previous_output = actual;
    }
}

} // namespace float_detail
} // namespace small

TEST_LIST = {
    {"correctness range_reduced_exp",
     small::float_detail::test_correctness_range_reduced_exp},
    {NULL, NULL}
};
