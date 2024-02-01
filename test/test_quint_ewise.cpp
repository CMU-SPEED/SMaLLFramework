// The CMakeLists.txt file finds the path to this header file when -DCMAKE_UARCH=REF
// #include <intrinsics_quint8.h>
#include <stdio.h>
#include <acutest.h>
#include <small.h>

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
            printf("%u ", c_tile[kk * QUINT8_C_ob + jj]);
        }
        printf("\n");
    }
}

void test_add()
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
            I[kk * QUINT8_C_ob + jj] = rand();
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
            TEST_CHECK(c_tile[kk * QUINT8_C_ob + jj] == output_ref[kk * QUINT8_C_ob + jj]);
        }
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"initialization of full block of data", test_initialization},
    {"add function", test_add},
    {NULL, NULL}};
