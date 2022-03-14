
// #define config_stride 1
// #define config_padding 'v'

#define REF 0
#define ZEN2 1
#define ARM 2


#ifndef uarch
#define uarch ZEN2
#endif

#define RUNS 1000
#define PARALLEL 1

#define op_dim(IN_dim, stride, K_dim, OUT_dim)   \
    {                                            \
        OUT_dim = (IN_dim - K_dim) / stride + 1; \
    }
