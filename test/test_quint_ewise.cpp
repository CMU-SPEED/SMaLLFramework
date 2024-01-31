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
    const small::QUInt8Buffer::accum_type x = 8;   // c_tile_t (32 bits)
    const small::QUInt8Buffer::value_type y = 255; // c_tile_out_t (8 bits)
    QUINT8_DEF_TILE_C(QUINT8_W_ob, QUINT8_C_ob);
    QUINT8_ZERO_TILE_C(QUINT8_W_ob, QUINT8_C_ob, x);

    c_tile_out_t I[QUINT8_W_ob * QUINT8_C_ob];

    for (uint32_t kk = 0; kk < QUINT8_W_ob; kk++)
    {
        for (uint32_t jj = 0; jj < QUINT8_C_ob; jj++)
        {
            I[kk * QUINT8_C_ob + jj] = y;
        }
    }

    QUINT8_ADD_TILE_C_G(I, QUINT8_W_ob, QUINT8_C_ob);

    // print the c_tile buffer to make sure the values were assigned
    for (int kk = 0; kk < QUINT8_W_ob; kk++)
    {
        for (int jj = 0; jj < QUINT8_C_ob; jj++)
        {
            TEST_CHECK(c_tile[kk * QUINT8_C_ob + jj] == x + y);
            printf("%d ", c_tile[kk * QUINT8_C_ob + jj]);
        }
        printf("\n");
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"initialization of full block of data", test_initialization},
    {"add function", test_add},
    {NULL, NULL}};
