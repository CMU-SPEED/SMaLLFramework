#include <arm_neon.h>
#define W_ob 6
#define C_ob 16
#define SIMD 4

#define UNROLL 16
#define C_ib C_ob

// not used for kernels, but used in throughput calculation.
#define NUM_FMA 2
#define NUM_MAX 1
#define NUM_LOAD 2
#define NUM_STORE 1

