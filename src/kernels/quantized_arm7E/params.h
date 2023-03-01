
#define QUANTIZED 1
typedef uint8_t dtype;
typedef int32_t atype;

struct
{                           // Structure declaration
    dtype *tensor;          // Member (int variable)
    float scale = 0.752941; // Member (string variable)
    int32_t offset = 0;
    int32_t multiplier = 1616928864;
    int lshift = 0;
    int rshift = 3;
    int zero = 0;
    int min_val = 255;
    int max_val = 0;
    uint8_t b = 8;
} typedef qint32_t; // Structure variable

typedef qint32_t qdtype;

#define W_ob 2
#define C_ob 1
#define SIMD 1
#define UNROLL 1
#define C_ib C_ob

//Potential blocking parameters for packing
#define NUM_FMA 1
#define NUM_MAX 1
#define NUM_LOAD 1
#define NUM_STORE 1
