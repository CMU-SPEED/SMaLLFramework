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


    #define FLOAT_CELU_TILE_C_UNROLLED_PAIR(r0, r1, off0, off1, b) \
    c12 = _mm256_setzero_ps(); \
    a_reg = _mm256_max_ps(r0, c12); \
    b0 = _mm256_max_ps(r1, c12); \
    _mm256_store_ps(c_tile + off0, a_reg); \
    b1 = _mm256_broadcast_ss(b); \
    _mm256_store_ps(c_tile + off1, b0); \
    b1 = _mm256_rcp_ps(b1); \
    r0 = _mm256_mul_ps(r0, b1); \
    r1 = _mm256_mul_ps(r1, b1); \
    r0 = _mm256_min_ps(r0, c12); \
    r1 = _mm256_min_ps(r1, c12); \
    a_reg = _mm256_sub_ps(c12, r0); \
    b0 = _mm256_sub_ps(c12, r1); \
    c12 = _mm256_set1_ps(1.0f/720.0f); \
    b1 = _mm256_set1_ps(-1.0f/120.0f); \
    r0 = _mm256_fmadd_ps(c12, a_reg, b1); \
    r1 = _mm256_fmadd_ps(c12, b0, b1); \
    c12 = _mm256_set1_ps(1.0f/24.0f); \
    r0 = _mm256_fmadd_ps(r0, a_reg, c12); \
    r1 = _mm256_fmadd_ps(r1, b0, c12); \
    c12 = _mm256_set1_ps(-1.0f/6.0f); \
    r0 = _mm256_fmadd_ps(r0, a_reg, c12); \
    r1 = _mm256_fmadd_ps(r1, b0, c12); \
    c12 = _mm256_set1_ps(0.5f); \
    r0 = _mm256_fmadd_ps(r0, a_reg, c12); \
    r1 = _mm256_fmadd_ps(r1, b0, c12); \
    c12 = _mm256_set1_ps(-1.0f); \
    r0 = _mm256_fmadd_ps(r0, a_reg, c12); \
    r1 = _mm256_fmadd_ps(r1, b0, c12); \
    c12 = _mm256_set1_ps(1.0f); \
    r0 = _mm256_fmadd_ps(r0, a_reg, c12); \
    r1 = _mm256_fmadd_ps(r1, b0, c12); \
    a_reg = _mm256_load_ps(c_tile + off0); \
    b1 = _mm256_load_ps(c_tile + off1); \
    r0 = _mm256_sub_ps(r0, c12); \
    r1 = _mm256_sub_ps(r1, c12); \
    c12 = _mm256_broadcast_ss(b); \
    r0 = _mm256_mul_ps(r0, c12); \
    r1 = _mm256_mul_ps(r1, c12); \
    r0 = _mm256_add_ps(r0, a_reg); \
    r1 = _mm256_add_ps(r1, b1);

#define FLOAT_CELU_TILE_C_UNROLLED(b) \
    FLOAT_CELU_TILE_C_UNROLLED_PAIR(c0, c1, 0 * FLOAT_C_ob, 0 * FLOAT_C_ob + FLOAT_SIMD, b); \
    FLOAT_CELU_TILE_C_UNROLLED_PAIR(c2, c3, 1 * FLOAT_C_ob, 1 * FLOAT_C_ob + FLOAT_SIMD, b); \
    FLOAT_CELU_TILE_C_UNROLLED_PAIR(c4, c5, 2 * FLOAT_C_ob, 2 * FLOAT_C_ob + FLOAT_SIMD, b); \
    FLOAT_CELU_TILE_C_UNROLLED_PAIR(c6, c7, 3 * FLOAT_C_ob, 3 * FLOAT_C_ob + FLOAT_SIMD, b); \
    FLOAT_CELU_TILE_C_UNROLLED_PAIR(c8, c9, 4 * FLOAT_C_ob, 4 * FLOAT_C_ob + FLOAT_SIMD, b); \
    FLOAT_CELU_TILE_C_UNROLLED_PAIR(c10, c11, 5 * FLOAT_C_ob, 5 * FLOAT_C_ob + FLOAT_SIMD, b);