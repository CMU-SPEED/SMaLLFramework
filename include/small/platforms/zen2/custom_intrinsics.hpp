// CUSTOM CELU microkernel
#define FLOAT_CELU_TILE_C(b) \
    c12 = _mm256_broadcast_ss(b);   \
    a_reg = _mm256_setzero_ps();   \
    b0 = _mm256_set1_ps(1.0f);      \
    b1 = _mm256_div_ps(c0, c12);    \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c0 = _mm256_max_ps(c0, a_reg);  \
    c0 = _mm256_add_ps(c0, b1);     \
    b1 = _mm256_div_ps(c1, c12);    \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c1 = _mm256_max_ps(c1, a_reg);  \
    c1 = _mm256_add_ps(c1, b1);     \
    b1 = _mm256_div_ps(c2, c12);    \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c2 = _mm256_max_ps(c2, a_reg);  \
    c2 = _mm256_add_ps(c2, b1);     \
    b1 = _mm256_div_ps(c3, c12);    \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c3 = _mm256_max_ps(c3, a_reg);  \
    c3 = _mm256_add_ps(c3, b1);     \
    b1 = _mm256_div_ps(c4, c12);    \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c4 = _mm256_max_ps(c4, a_reg);  \
    c4 = _mm256_add_ps(c4, b1);     \
    b1 = _mm256_div_ps(c5, c12);    \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c5 = _mm256_max_ps(c5, a_reg);  \
    c5 = _mm256_add_ps(c5, b1);     \
    b1 = _mm256_div_ps(c6, c12);    \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c6 = _mm256_max_ps(c6, a_reg);  \
    c6 = _mm256_add_ps(c6, b1);     \
    b1 = _mm256_div_ps(c7, c12);    \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c7 = _mm256_max_ps(c7, a_reg);  \
    c7 = _mm256_add_ps(c7, b1);     \
    b1 = _mm256_div_ps(c8, c12);    \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c8 = _mm256_max_ps(c8, a_reg);  \
    c8 = _mm256_add_ps(c8, b1);     \
    b1 = _mm256_div_ps(c9, c12);    \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c9 = _mm256_max_ps(c9, a_reg);  \
    c9 = _mm256_add_ps(c9, b1);     \
    b1 = _mm256_div_ps(c10, c12);   \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c10 = _mm256_max_ps(c10, a_reg); \
    c10 = _mm256_add_ps(c10, b1);   \
    b1 = _mm256_div_ps(c11, c12);   \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c11 = _mm256_max_ps(c11, a_reg); \
    c11 = _mm256_add_ps(c11, b1);  



#define FLOAT_SOFTMAX_TILE_C(step, a, c, d) \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD, c0);  \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 1 * FLOAT_SIMD, c1);  \
    _mm256_storeu_ps(c_tile + 1 * FLOAT_C_ob + 0 * FLOAT_SIMD, c2);  \
    _mm256_storeu_ps(c_tile + 1 * FLOAT_C_ob + 1 * FLOAT_SIMD, c3);  \
    _mm256_storeu_ps(c_tile + 2 * FLOAT_C_ob + 0 * FLOAT_SIMD, c4);  \
    _mm256_storeu_ps(c_tile + 2 * FLOAT_C_ob + 1 * FLOAT_SIMD, c5);  \
    _mm256_storeu_ps(c_tile + 3 * FLOAT_C_ob + 0 * FLOAT_SIMD, c6);  \
    _mm256_storeu_ps(c_tile + 3 * FLOAT_C_ob + 1 * FLOAT_SIMD, c7);  \
    _mm256_storeu_ps(c_tile + 4 * FLOAT_C_ob + 0 * FLOAT_SIMD, c8);  \
    _mm256_storeu_ps(c_tile + 4 * FLOAT_C_ob + 1 * FLOAT_SIMD, c9);  \
    _mm256_storeu_ps(c_tile + 5 * FLOAT_C_ob + 0 * FLOAT_SIMD, c10); \
    _mm256_storeu_ps(c_tile + 5 * FLOAT_C_ob + 1 * FLOAT_SIMD, c11); \
    c_tile_t *c_pixel = c_tile;                                \
    c_tile_t *d_pixel = d;                                     \
    c_tile_t const *a_pixel = a;                               \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                     \
    {                                                          \
        c_tile_t *c_channel = c_pixel;                         \
        c_tile_t *d_channel = d_pixel;                         \
        c_tile_t const *a_channel = a_pixel;                   \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                 \
        {                                                      \
            *(d_channel) = std::exp(*a_channel);               \
            *(c_channel) += *(d_channel);                      \
            std::cout << "exp: " << *(d_channel) << " sum: " << *(c_channel) << std::endl; \
            c_channel++;                                       \
            d_channel++;                                       \
            a_channel++;                                       \
        }                                                      \
        a_pixel += step;                                       \
        c_pixel += FLOAT_C_ob;                                       \
        d_pixel += FLOAT_C_ob;                                       \
    }                                                          \
    c0 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c2 = _mm256_loadu_ps(c_tile + 1 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c3 = _mm256_loadu_ps(c_tile + 1 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c4 = _mm256_loadu_ps(c_tile + 2 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c5 = _mm256_loadu_ps(c_tile + 2 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c6 = _mm256_loadu_ps(c_tile + 3 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c7 = _mm256_loadu_ps(c_tile + 3 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c8 = _mm256_loadu_ps(c_tile + 4 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c9 = _mm256_loadu_ps(c_tile + 4 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c10 = _mm256_loadu_ps(c_tile + 5 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    c11 = _mm256_loadu_ps(c_tile + 5 * FLOAT_C_ob + 1 * FLOAT_SIMD);
