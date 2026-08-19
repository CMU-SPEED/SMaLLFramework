#define PARALLEL 1

#include <acutest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <small.h>
#include <small/op_type.hpp>
#include <small/utils/Timer.hpp>

#include "test_utils.hpp"

namespace small {
namespace float_detail {

namespace {

float reference_exp(float x)
{
    return std::exp(x);
}

void init_exp_input(FloatBuffer &buffer)
{
    for (std::size_t ix = 0; ix < buffer.size(); ++ix)
    {
        // buffer[ix] = 0.000242f + static_cast<float>(ix % 1000U) * 0.01f;
        buffer[ix] = 0.000242f + static_cast<float>(ix ) * 1.0/static_cast<float>(buffer.size()); // Small positive values;

    }
}


std::vector<LayerParams> inplace_exp_performance_params()
{
    return {
        {  16,   48,  48, 1, 1, small::PADDING_V,   16},
        {  32,   24,  24, 1, 1, small::PADDING_V,   32},

        {  32,   48,  48, 1, 1, small::PADDING_V,   32},
        {  64,   24,  24, 1, 1, small::PADDING_V,   64},
        { 128,   12,  12, 1, 1, small::PADDING_V,  128},

        {  16,   48,  48, 1, 1, small::PADDING_V,   32},
        {  32,   24,  24, 1, 1, small::PADDING_V,   64},
        {  64,   12,  12, 1, 1, small::PADDING_V,  128},
        { 128,    6,   6, 1, 1, small::PADDING_V,  256},

        { 128,   24,  24, 1, 1, small::PADDING_V,  128},
        { 256,   12,  12, 1, 1, small::PADDING_V,  256},

        { 512,   12,  12, 1, 1, small::PADDING_V,  512},
        {1024,    6,   6, 1, 1, small::PADDING_V, 1024},
        {16,    1,   6, 1, 1, small::PADDING_V, 1024},

    };
}

void run_inplace_exp(std::size_t C_i,
                     std::size_t H,
                     std::size_t W,
                     FloatBuffer &buffer,
                     FloatBuffer &scratch)
{
    constexpr dim_t step = FLOAT_C_ob;
    uint32_t const spatial_elts = H * W;
    float *b_cur = scratch.data();
    float recip_fact_table[] = {1.0f/720.0f, 1.0f / 120.0f, 1.0f / 24.0f,               \
                                1.0f / 6.0f, 0.5f, 1.0f, 1.0f};                 \
    for (uint32_t channel = 0; channel < (C_i)/FLOAT_C_ob; channel ++)
    {
        uint32_t spatial = 0;
        FLOAT_DEF_TILE_C;
        for (; spatial + FLOAT_W_ob <= spatial_elts; spatial += FLOAT_W_ob)
        {
            float *c_cur = buffer.data() + spatial * FLOAT_C_ob + channel*spatial_elts*FLOAT_C_ob;
            // FLOAT_LOAD_TILE_C_strided(c_cur, C_i);

            FLOAT_LOAD_TILE_C(c_cur);
            // FLOAT_INPLACE_EXP_negative_horner_TILE_C(FLOAT_C_ob, c_cur, recip_fact_table);
            // FLOAT_BESPOKE_INPLACE_EXP_horner_TILE_C(FLOAT_C_ob, c_cur, recip_fact_table);
            FLOAT_EXP_RR_TILE_C(c_cur);
            // FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_EXP, 0, b_cur);
            FLOAT_STORE_TILE_C(c_cur);
        }

    }

    
}



void run_inplace_composed_exp(std::size_t C_i,
                     std::size_t H,
                     std::size_t W,
                     FloatBuffer &buffer,
                     FloatBuffer &scratch)
{
    constexpr dim_t step = FLOAT_C_ob;
    std::size_t const spatial_elts = H * W;
    float *b_cur = scratch.data();

    for (std::size_t channel = 0; channel < (C_i)/FLOAT_C_ob; channel ++)
    {
        std::size_t spatial = 0;
        FLOAT_DEF_TILE_C;
        for (; spatial + FLOAT_W_ob <= spatial_elts; spatial += FLOAT_W_ob)
        {
            float *c_cur = buffer.data() + spatial * FLOAT_C_ob + (channel)*spatial_elts*FLOAT_C_ob;
            // FLOAT_LOAD_TILE_C_strided(c_cur, C_i);
            FLOAT_LOAD_TILE_C(c_cur);
            FLOAT_EXP_RR_TILE_C(c_cur);
            FLOAT_STORE_TILE_C(c_cur);
        }

    }
}


} // namespace

bool check_inplace_exp_case(LayerParams const &p)
{
    constexpr float abs_tolerance = 5.0e-2f;
    constexpr float rel_tolerance = 5.0e-6f;

    std::size_t const num_input_elts = p.C_i * p.H * p.W;
    FloatBuffer input_buf(num_input_elts);
    FloatBuffer expected_buf(num_input_elts);
    FloatBuffer scratch_buf(1);
    scratch_buf[0] = 1.0f;

    init_exp_input(input_buf);
    /* Input is implicitly treated as packed: C_i/FLOAT_C_ob, H, W, FLOAT_C_ob */
    for (std::size_t ix = 0; ix < num_input_elts; ++ix)
    {
        expected_buf[ix] = reference_exp(input_buf[ix]);
    }

    run_inplace_exp(p.C_i, p.H, p.W, input_buf, input_buf);

    bool passed = true;

    for(int32_t channel_blocks = 0; channel_blocks < (p.C_i)/FLOAT_C_ob; channel_blocks++)
    {
        for(int32_t spatial = 0; spatial < p.H*p.W; spatial++)
        {
            for(int32_t channel = 0; channel < FLOAT_C_ob; channel++)
            {
                size_t ix = spatial * (FLOAT_C_ob) + channel_blocks*p.H*p.W*FLOAT_C_ob + channel;
                float const expected = expected_buf[ix];
                float const actual = input_buf[ix];
                float const abs_error = std::abs(actual - expected);
                float const rel_error = abs_error / (std::abs(expected) + 1.0e-12f);
                bool const close = (abs_error < abs_tolerance) ||
                                   (rel_error < rel_tolerance);
                if (!close)
                {
                    printf("inplace EXP correctness failure for C_i=%u H=%u W=%u at ix=%zu: expected %.9f, actual %.9f, abs_error %.9f, rel_error %.9f\n",
                           p.C_i, p.H, p.W, ix, expected, actual, abs_error,
                           rel_error);
                    TEST_MSG("EXP correctness failure for C_i=%u H=%u W=%u at ix=%zu: expected %.9f, actual %.9f, abs_error %.9f, rel_error %.9f",
                             p.C_i, p.H, p.W, ix, expected, actual, abs_error,
                             rel_error);
                    passed = false;
                    break;
                }
            }
        }
    }

    TEST_CHECK_(passed,
                "inplace EXP correctness for measured case C_i=%u H=%u W=%u",
                p.C_i, p.H, p.W);
    return passed;
}

void test_correctness_inplace_exp_performance_cases(void)
{
    for (LayerParams const &p : inplace_exp_performance_params())
    {
        check_inplace_exp_case(p);
    }
}

void measure_inplace_exp_performance(void)
{
    std::vector<LayerParams> const params = inplace_exp_performance_params();

    uint32_t const num_threads[] = {1, 2, 4};
    char const *str_num_threads[] = {"1", "2", "4"};
    uint32_t const num_runs(100);
    uint32_t const num_trials(1000);
    small::Timer t;

    printf("\ninplace_exp(float) func.\n");
    printf("\tC_i\tH\tW\tk\ts\tnthd\truns\tt_min\tt_max\tt_avg\n");

    std::ofstream csv_file("inplace_exp_timing.csv");
    if (!csv_file.is_open())
    {
        std::cerr << "Error: Could not open inplace_exp_timing.csv for writing" << std::endl;
        return;
    }

    csv_file << "C_i,H,W,k,s,num_threads,min_t,max_t,avg_t\n";

    for (LayerParams const &p : params)
    {
        if (!check_inplace_exp_case(p))
        {
            continue;
        }

        size_t const num_input_elts = p.C_i * p.H * p.W;
        FloatBuffer input_dc(num_input_elts);
        FloatBuffer scratch_dc(1);
        scratch_dc[0] = 1.0f;

        for (size_t ix = 0; ix < 1; ++ix)
        {
            setenv("OMP_NUM_THREADS", str_num_threads[ix], 1);

            for (size_t iy = 0; iy < 10; ++iy)
            {
                init_exp_input(input_dc);
                run_inplace_exp(p.C_i, p.H, p.W, input_dc, scratch_dc);
            }

            double tx(0.);
            double min_t = std::numeric_limits<double>::max();
            double max_t = 0.;

            for (size_t iy = 0; iy < num_runs; ++iy)
            {
                init_exp_input(input_dc);
                t.start();
                for(size_t iz = 0; iz < num_trials; ++iz)
                {
                    run_inplace_exp(p.C_i, p.H, p.W, input_dc, input_dc);
                }
                t.stop();
                double ts = t.elapsed()/static_cast<double>(num_trials);
                tx += ts;
                min_t = std::min(min_t, ts);
                max_t = std::max(max_t, ts);
            }

            csv_file << p.C_i << "," << p.H << "," << p.W << ","
                     << p.k << "," << p.s << "," << num_threads[ix] << ","
                     << min_t << "," << max_t << "," << (tx / num_runs) << "\n";
            printf("\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.0lf\t%.0lf\t%.0lf\n",
                   p.C_i, p.H, p.W, p.k, p.s,
                   num_threads[ix], num_runs,
                   min_t, max_t, (tx -(min_t+max_t)) / (num_runs-2));
        }
    }
}

} // namespace float_detail
} // namespace small

TEST_LIST = {
    {"correctness inplace_exp performance cases",
     small::float_detail::test_correctness_inplace_exp_performance_cases},
    {"inplace_exp_performance",
     small::float_detail::measure_inplace_exp_performance},
    {NULL, NULL}
};
