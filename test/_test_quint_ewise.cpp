// The CMakeLists.txt file finds the path to this header file when -DCMAKE_UARCH=REF
// TODO: This file doesn't work because the intrinsics file is not in included correctly
// We should use this as a stub to test out quantized operations.

#include <intrinsics_quint8.h>
#include <stdio.h>
#include <acutest.h>
// #include <small.h>
#include <small/utils/Timer.hpp>

#if !defined(SMALL_HAS_QUINT8_SUPPORT)
#ERROR "ERROR: small does not have quint8 support"
#endif

// If the macros in the above header are used within the small namespace, these types would have been typedef-ed.
// Adding defines here so that you can get your tests up and running
typedef int32_t c_tile_t;
typedef uint8_t c_tile_out_t;

// These are defined in params_quint8.h
#ifdef QUINT8_W_ob
#undef QUINT8_W_ob
#define QUINT8_W_ob 8
#endif

#ifdef QUINT8_C_ob
#undef QUINT8_C_ob
#define QUINT8_C_ob 8
#endif

const int num_outer_runs = 100;
const int num_inner_runs = 1000;

void test_initialization()
{
    printf("\n");
    const small::QUInt8Buffer::accum_type x = 8;
    QUINT8_DEF_TILE_C(QUINT8_W_ob, QUINT8_C_ob);
    QUINT8_ZERO_TILE_C(QUINT8_W_ob, QUINT8_C_ob, x);

    // print the c_tile buffer to make sure the values were assigned
    for (int kk = 0; kk < QUINT8_W_ob; kk++)
    {
        for (int jj = 0; jj < QUINT8_C_ob; jj++)
        {
            TEST_CHECK(c_tile[kk * QUINT8_C_ob + jj] == x);
        }
        printf("\n");
    }
}

void test_add_correctness()
{
    printf("\n");

    // Initialize c_tile with random values
    // Initialize c_tile2 with the same values
    QUINT8_DEF_TILE_C(QUINT8_W_ob, QUINT8_C_ob);
    c_tile_t c_tile2[QUINT8_W_ob * QUINT8_C_ob];

    for (uint32_t kk = 0; kk < QUINT8_W_ob; kk++)
    {
        for (uint32_t jj = 0; jj < QUINT8_C_ob; jj++)
        {
            int r = rand();
            c_tile[kk * QUINT8_C_ob + jj] = r;
            c_tile2[kk * QUINT8_C_ob + jj] = r;
        }
    }

    // Initializing I with random values
    c_tile_out_t I[QUINT8_W_ob * QUINT8_C_ob];
    for (uint32_t kk = 0; kk < QUINT8_W_ob; kk++)
    {
        for (uint32_t jj = 0; jj < QUINT8_C_ob; jj++)
        {
            I[kk * QUINT8_C_ob + jj] = static_cast<c_tile_out_t>(rand());
        }
    }

    // Running old version of function being tested
    {
        OLD_QUINT8_ADD_TILE_C_G(I, QUINT8_W_ob, QUINT8_C_ob);
    }

    // Copy output of old version of function into ouput_ref
    c_tile_t output_ref[QUINT8_W_ob * QUINT8_C_ob];
    memcpy(output_ref, c_tile, sizeof(c_tile_t) * QUINT8_W_ob * QUINT8_C_ob);

    // Copy c_tile2 into c_tile and run the new version of the function
    memcpy(c_tile, c_tile2, sizeof(c_tile_t) * QUINT8_W_ob * QUINT8_C_ob);

    // Running new version of function being tested
    {
        QUINT8_ADD_TILE_C_G(I, QUINT8_W_ob, QUINT8_C_ob);
    }

    // Checking values
    for (int kk = 0; kk < QUINT8_W_ob; kk++)
    {
        for (int jj = 0; jj < QUINT8_C_ob; jj++)
        {
            TEST_CHECK(c_tile[kk * QUINT8_C_ob + jj] == output_ref[kk * QUINT8_C_ob + jj]);
        }
    }
}

void test_add_performance()
{
    printf("\n");
    printf("num_runs, min, max, avg");

    // Initialize c_tile with random values
    // Initialize c_tile2 with the same values
    QUINT8_DEF_TILE_C(QUINT8_W_ob, QUINT8_C_ob);
    QUINT8_ZERO_TILE_C(QUINT8_W_ob, QUINT8_C_ob, 0);

    // Initializing I with random values
    c_tile_out_t I[QUINT8_W_ob * QUINT8_C_ob];
    for (uint32_t kk = 0; kk < QUINT8_W_ob; kk++)
    {
        for (uint32_t jj = 0; jj < QUINT8_C_ob; jj++)
        {
            I[kk * QUINT8_C_ob + jj] = 1U;
        }
    }

    small::Timer t;
    double tx = 0.;
    double min_t = std::numeric_limits<double>::max();
    double max_t = 0.;
    for (int i = 0; i < num_outer_runs; ++i)
    {
        // Running old version of function being tested
        t.start();
        __asm__ volatile(
            "old_kernel:");
        for (int j = 0; j < num_inner_runs; ++j)
        {
            {
                OLD_QUINT8_ADD_TILE_C_G(I, QUINT8_W_ob, QUINT8_C_ob);
            }
        }
        t.stop();

        double ts = t.elapsed();
        tx += ts;
        min_t = std::min(min_t, ts);
        max_t = std::max(max_t, ts);
    }

    printf("num_runs: %d, c_tile value: %d\n", num_outer_runs * num_inner_runs, c_tile[0]);
    printf("Old macro\t%d\t%lf\t%lf\t%lf\n",
           num_outer_runs * num_inner_runs, min_t, max_t, (tx / (num_outer_runs * num_inner_runs)));

    // Running new version of function being tested
    {
        QUINT8_ZERO_TILE_C(QUINT8_W_ob, QUINT8_C_ob, 0);
    }
    tx = 0.;
    min_t = std::numeric_limits<double>::max();
    max_t = 0.;

    for (int i = 0; i < num_outer_runs; ++i)
    {
        // Running old version of function being tested
        t.start();
        __asm__ volatile(
            "new_kernel:");
        for (int j = 0; j < num_inner_runs; ++j)
        {
            QUINT8_ADD_TILE_C_G(I, QUINT8_W_ob, QUINT8_C_ob);
        }
        t.stop();

        double ts = t.elapsed();
        tx += ts;
        min_t = std::min(min_t, ts);
        max_t = std::max(max_t, ts);
    }

    printf("num_runs: %d, c_tile value: %d\n", (num_inner_runs * num_outer_runs), c_tile[0]);
    printf("New macro\t%d\t%lf\t%lf\t%lf\n",
           (num_inner_runs * num_outer_runs), min_t, max_t, (tx / (num_inner_runs * num_outer_runs)));
}

void test_zero_correctness()
{
    printf("\n");
    const small::QUInt8Buffer::accum_type x = 8;
    QUINT8_DEF_TILE_C(QUINT8_W_ob, QUINT8_C_ob);
    QUINT8_ZERO_TILE_C(QUINT8_W_ob, QUINT8_C_ob, x);
    // TODO check new test against old test

    // print the c_tile buffer to make sure the values were assigned
    for (int kk = 0; kk < QUINT8_W_ob; kk++)
    {
        for (int jj = 0; jj < QUINT8_C_ob; jj++)
        {
            TEST_CHECK(c_tile[kk * QUINT8_C_ob + jj] == x);
        }
    }
}

void test_load_correctness()
{
    printf("\n");

    // Initialize c_tile with zero values
    // Initialize O with random values
    QUINT8_DEF_TILE_C(QUINT8_W_ob, QUINT8_C_ob);
    {
        QUINT8_ZERO_TILE_C(QUINT8_W_ob, QUINT8_C_ob, 0);
    }
    c_tile_t O[QUINT8_W_ob * QUINT8_C_ob];

    for (uint32_t kk = 0; kk < QUINT8_W_ob; kk++)
    {
        for (uint32_t jj = 0; jj < QUINT8_C_ob; jj++)
        {
            O[kk * QUINT8_C_ob + jj] = rand();
        }
    }

    // running the function to be tested
    {
        QUINT8_LOAD_TILE_C(O, QUINT8_W_ob, QUINT8_C_ob);
    }

    // Checking values
    for (int kk = 0; kk < QUINT8_W_ob; kk++)
    {
        for (int jj = 0; jj < QUINT8_C_ob; jj++)
        {
            TEST_CHECK(c_tile[kk * QUINT8_C_ob + jj] == O[kk * QUINT8_C_ob + jj]);
        }
    }
}

void test_store_correctness()
{
    printf("\n");

    // Initialize c_tile with random values
    // Initialize O
    QUINT8_DEF_TILE_C(QUINT8_W_ob, QUINT8_C_ob);

    for (uint32_t kk = 0; kk < QUINT8_W_ob; kk++)
    {
        for (uint32_t jj = 0; jj < QUINT8_C_ob; jj++)
        {
            c_tile[kk * QUINT8_C_ob + jj] = rand();
        }
    }

    c_tile_t O[QUINT8_W_ob * QUINT8_C_ob];

    // running the function to be tested
    {
        QUINT8_STORE_TILE_C(O, QUINT8_W_ob, QUINT8_C_ob);
    }

    // Checking values
    for (int kk = 0; kk < QUINT8_W_ob; kk++)
    {
        for (int jj = 0; jj < QUINT8_C_ob; jj++)
        {
            TEST_CHECK(c_tile[kk * QUINT8_C_ob + jj] == O[kk * QUINT8_C_ob + jj]);
        }
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    // {"initialization of full block of data", test_initialization},
    {"zero function, correcteness test", test_zero_correctness},
    {"load function, correcteness test", test_load_correctness},
    {"store function, correcteness test", test_store_correctness},
    {NULL, NULL}};
