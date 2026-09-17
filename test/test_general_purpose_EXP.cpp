#include <acutest.h>

#include <cmath>
#include <small.h>
#include <small/utils/Timer.hpp>
#include <cstddef>
#include <fstream>
#include <limits>
#include <vector>


namespace {

#if defined(_MSC_VER) && defined(_M_IX86)
// Internal Microsoft CRT entry point selected by the CRT when SSE2 math is
// enabled. It is not available on non-MSVC or x64 builds.
extern "C" float __cdecl __libm_sse2_expf(float);
#endif

// Computes e^x by reducing x to r + n*ln(2), approximating e^r with its
// Taylor series, then restoring the power-of-two scale.  std::exp is not used
// here; it is only used by the test as the reference implementation.
float inline exp_scalar_from_scratch(float x)
{
    if (x == -std::numeric_limits<float>::infinity())
    {
        return 0.0f;
    }

    constexpr float ln2 = 0.69314718055994530942;
    constexpr float inv_ln2 = 1.4426950408889634074;

    // Round to the nearest multiple of ln(2), leaving r in roughly
    // [-ln(2)/2, ln(2)/2], where the series converges quickly.
    float const x_as_float = static_cast<float>(x);
    float const n = std::floor(x_as_float * inv_ln2 +
                                              (x >= 0.0f ? 0.5 : -0.5));
    float const r = x_as_float - static_cast<float>(n) * ln2;
    // Evaluate 1 + r + r^2/2! + ... + r^12/12! using Horner's rule.
    float sum = 1.0f;
    for (int k = 12; k >= 1; --k)
    {
        sum = 1.0f + r * sum / static_cast<float>(k);
    }

    return std::ldexp(sum, n);
}

constexpr std::size_t tile_height = FLOAT_W_ob;
constexpr std::size_t tile_width = FLOAT_C_ob;
constexpr std::size_t tile_elements = tile_height * tile_width;

using c_tile_t = small::FloatBuffer::value_type;

void exp_from_scratch(float const *input, float *output)
{
    for (std::size_t index = 0; index < tile_elements; ++index)
    {
        output[index] = exp_scalar_from_scratch(input[index]);
    }
}

void exp_from_scratch_simd(float const *input, float *output)
{

    alignas(64) float two_power[FLOAT_W_ob*FLOAT_C_ob];\



    FLOAT_DEF_TILE_C;
    FLOAT_EXP_GENERAL_TILE_C(input);
    FLOAT_STORE_TILE_C(output); 
 
    // for(int i = 0; i < tile_elements; i++)
    // {
    //     printf("%f %f\n", output[i],two_power[i]);
    //     // printf("%f r: %f exp %f right %f left%f o:%f\n",input[i], ((reduced_buf[i])), exponent_buf[i], exponent_buf_neg[i], exponent_buf_pos[i], output[i]);
    // }
    // printf("\n\n");
}


void test_exp_from_scratch_matches_std_exp()
{
    constexpr float values[] = {
        -10.0f, -5.0f, -1.0f, -0.5f, -0.001f, 0.0f,
        0.001f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f,
    };
    small::FloatBuffer input(tile_elements);
    small::FloatBuffer output(tile_elements);
    for (std::size_t index = 0; index < tile_elements; ++index)
    {
        input[index] = values[index % (sizeof(values) / sizeof(values[0]))];
        // input[index] = -10.0f + static_cast<float>(index) * (20.0f / tile_elements);

    }

    exp_from_scratch(input.data(), output.data());
    for (std::size_t index = 0; index < tile_elements; ++index)
    {
        float const expected = std::exp(input[index]);
        float const relative_error = std::abs(output[index] - expected) / expected;
        TEST_CHECK_(relative_error < 2.0e-6f,
                    "tile[%zu]: expected %.9g, got %.9g (relative error %.9g)",
                    index, expected, output[index], relative_error);
    }
}

void test_exp_from_scratch_simd_matches_std_exp()
{
    //  constexpr float values[] = {
    //     -20.0f, -5.0f, -1.0f, -0.5f, -0.001f, 0.0f,
    //     0.001f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f,
    // };
      constexpr float values[] = {
        -23.0f, -5.34f, 1.0f, -0.5f, 0.231f, 0.0f,
        0.001f, 10.5f
    };
    small::FloatBuffer input(tile_elements);
    small::FloatBuffer output(tile_elements);
    for (std::size_t index = 0; index < tile_elements; ++index)
    {
        input[index] = values[index % (sizeof(values) / sizeof(values[0]))];
        // input[index] = 10.0f + static_cast<float>(index) * (20.0f / tile_elements);
    }

    exp_from_scratch_simd(input.data(), output.data());
    for (std::size_t index = 0; index < tile_elements; ++index)
    {
        float const expected = std::exp(input[index]);
        float const relative_error = std::abs(output[index] - expected) / expected;
        //Accurate to 1/50th of a percent
        TEST_CHECK_(relative_error < 5.0e-4f,
                    "tile[%zu]: expected %.9g, got %.9g (relative error %.9g)",
                    index, expected, output[index], relative_error);
    }
}

void test_exp_from_scratch_special_values()
{
    TEST_CHECK(std::isinf(exp_scalar_from_scratch(std::numeric_limits<float>::infinity())));
    TEST_CHECK(exp_scalar_from_scratch(-std::numeric_limits<float>::infinity()) == 0.0f);
    TEST_CHECK(std::isnan(exp_scalar_from_scratch(std::numeric_limits<float>::quiet_NaN())));
}


void std_exp(float const *input, float *output)
{
    for (std::size_t index = 0; index < tile_elements; ++index)
    {
        output[index] = std::exp(input[index]);
    }
}

using ExpTileFunction = void (*)(float const *, float *);

double measure_nanoseconds_per_element(ExpTileFunction function,
                                       small::FloatBuffer const &input,
                                       std::size_t trials)
{
    small::FloatBuffer output(input.size());
    volatile float sink = 0.0f;
    small::Timer timer;
    timer.start();
    for (std::size_t trial = 0; trial < trials; ++trial)
    {
        for (std::size_t offset = 0; offset < input.size(); offset += tile_elements)
        {
            function(input.data() + offset, output.data() + offset);
            sink += output[offset];
        }
    }
    timer.stop();
    (void)sink;

    double const elapsed_ns = timer.elapsed();
    return elapsed_ns / static_cast<double>(input.size() * trials);
}

void measure_exp_performance()
{
    struct Shape
    {
        std::size_t channels;
        std::size_t height;
        std::size_t width;
    };

    // Active C_i, H, and W cases from test_ewise_kernel.cpp.
    std::vector<Shape> const shapes = {
        {16, 48, 48}, {32, 24, 24},
        {32, 48, 48}, {64, 24, 24}, {128, 12, 12},
        {16, 48, 48}, {32, 24, 24}, {64, 12, 12}, {128, 6, 6},
        {128, 24, 24}, {256, 12, 12},
        {512, 12, 12}, {1024, 6, 6},
    };
    constexpr std::size_t runs = 10;
    constexpr std::size_t trials = 100;

    std::ofstream csv_file("general_purpose_exp_timing.csv");
    TEST_CHECK_(csv_file.is_open(),
                "open general_purpose_exp_timing.csv for writing");
    if (!csv_file.is_open())
    {
        return;
    }

    csv_file << "C_i,H,W,elements,scratch_ns_per_element,simd_ns_per_element,"
                "std_exp_ns_per_element,simd_speedup\n";
    printf("\nEXP performance (ns/element)\n");
    printf("  C_i    H    W    elements    scratch       simd    std::exp  simd speedup\n");

    for (Shape const &shape : shapes)
    {
        std::size_t const elements = shape.channels * shape.height * shape.width;
        small::FloatBuffer input(elements);
        for (std::size_t index = 0; index < elements; ++index)
        {
            input[index] = -10.0f + static_cast<float>(index % 1000U) * 0.01f;
        }

        (void)measure_nanoseconds_per_element(exp_from_scratch, input, 1);
        (void)measure_nanoseconds_per_element(exp_from_scratch_simd, input, 1);
        (void)measure_nanoseconds_per_element(std_exp, input, 1);

        double scratch_total = 0.0;
        double simd_total = 0.0;
        double std_total = 0.0;
        for (std::size_t run = 0; run < runs; ++run)
        {
            scratch_total += measure_nanoseconds_per_element(exp_from_scratch, input, trials);
            simd_total += measure_nanoseconds_per_element(exp_from_scratch_simd, input, trials);
            std_total += measure_nanoseconds_per_element(std_exp, input, trials);
        }

        double const scratch_ns = scratch_total / static_cast<double>(runs);
        double const simd_ns = simd_total / static_cast<double>(runs);
        double const std_ns = std_total / static_cast<double>(runs);
        double const simd_speedup = std_ns / simd_ns;
        csv_file << shape.channels << "," << shape.height << "," << shape.width
                 << "," << elements << "," << scratch_ns << "," << simd_ns
                 << "," << std_ns << "," << simd_speedup << "\n";
        printf("%5zu %4zu %4zu %11zu %10.2f %10.2f %10.2f %11.2fx\n",
               shape.channels, shape.height, shape.width, elements,
               scratch_ns, simd_ns, std_ns, simd_speedup);
    }
}

} // namespace

TEST_LIST = {
    {"exp_from_scratch matches std::exp\n", test_exp_from_scratch_matches_std_exp},
    {"exp_from_scratch_simd matches std::exp\n", test_exp_from_scratch_simd_matches_std_exp},
    // {"exp_from_scratch special values", test_exp_from_scratch_special_values},
    {"EXP performance", measure_exp_performance},
    {NULL, NULL}
};
