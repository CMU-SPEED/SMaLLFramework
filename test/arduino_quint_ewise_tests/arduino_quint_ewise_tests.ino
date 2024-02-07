#include "intrinsics_quint8.h"

// #if !defined(SMALL_HAS_QUINT8_SUPPORT)
// #ERROR "ERROR: small does not have quint8 support"
// #endif

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

void test_add_correctness()
{
    printf("Testing correctness\n");

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
    OLD_QUINT8_ADD_TILE_C_G(I, QUINT8_W_ob, QUINT8_C_ob);

    // Copy output of old version of function into ouput_ref
    c_tile_t output_ref[QUINT8_W_ob * QUINT8_C_ob];
    memcpy(output_ref, c_tile, sizeof(c_tile_t) * QUINT8_W_ob * QUINT8_C_ob);

    // Copy c_tile2 into c_tile and run the new version of the function
    memcpy(c_tile, c_tile2, sizeof(c_tile_t) * QUINT8_W_ob * QUINT8_C_ob);

    // Running new version of function being tested
    QUINT8_ADD_TILE_C_G(I, QUINT8_W_ob, QUINT8_C_ob);

    // Checking values
    for (int kk = 0; kk < QUINT8_W_ob; kk++)
    {
        for (int jj = 0; jj < QUINT8_C_ob; jj++)
        {
            if (c_tile[kk * QUINT8_C_ob + jj] != output_ref[kk * QUINT8_C_ob + jj]) {
              printf("TEST FAILED");
            }
        }
    }
}

void test_add_performance()
{
    printf("Testing performance\n");

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

    for (int i = 0; i < num_outer_runs; ++i)
    {
        // Running old version of function being tested
        __asm__ volatile(
            "old_kernel:");
        for (int j = 0; j < num_inner_runs; ++j)
        {
            OLD_QUINT8_ADD_TILE_C_G(I, QUINT8_W_ob, QUINT8_C_ob);
        }
    }

    printf("num_runs: %d, c_tile value: %d\n", num_outer_runs * num_inner_runs, c_tile[0]);
    printf("Old macro\t%d\t%lf\t%lf\t%lf\n",
           num_outer_runs * num_inner_runs, min_t, max_t, (tx / (num_outer_runs * num_inner_runs)));

    // Running new version of function being tested
    QUINT8_ZERO_TILE_C(QUINT8_W_ob, QUINT8_C_ob, 0);

    for (int i = 0; i < num_outer_runs; ++i)
    {
        // Running old version of function being tested
        __asm__ volatile(
            "new_kernel:");
        for (int j = 0; j < num_inner_runs; ++j)
        {
            QUINT8_ADD_TILE_C_G(I, QUINT8_W_ob, QUINT8_C_ob);
        }
    }

    printf("num_runs: %d, c_tile value: %d\n", (num_inner_runs * num_outer_runs), c_tile[0]);
    printf("New macro\t%d\t%lf\t%lf\t%lf\n",
           (num_inner_runs * num_outer_runs), min_t, max_t, (tx / (num_inner_runs * num_outer_runs)));
}

void setup() {
  test_add_correctness();
  test_add_performance();

}

void loop() {}