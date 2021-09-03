//scalar versions of all the microkernels for platform portability

// Initializations


#define ZERO_TILE_C() \
  float c_tile[W_ob*C_ob] = {0};



// Loads


#define LOAD_TILE_C(O)\
  float c_tile[W_ob*C_ob];\
  for (int kk = 0; kk < W_ob; kk++) {\
      for (int jj = 0; jj < C_ob; jj++) {\
        c_tile[kk*C_ob + jj] = O[kk*C_ob + jj];\
      }\
  }\

#define LOAD_LAST_C(O, W_last)\
  float c_tile[W_ob*C_ob];\
  for (int kk = 0; kk < W_last; kk++) {\
      for (int jj = 0; jj < C_ob; jj++) {\
        c_tile[kk*C_ob + jj] = O[kk*C_ob + jj];\
      }\
  }\

//Stores

#define STORE_TILE_C(O) \
for (int kk = 0; kk < W_ob; kk++) {\
    for (int jj = 0; jj < C_ob; jj++) {\
        O[kk*C_ob + jj] = c_tile[kk*C_ob + jj];\
    }\
}\

#define STORE_END_C(O,  W_last) \
for (int kk = 0; kk < W_last; kk++) {\
    for (int jj = 0; jj < C_ob; jj++) {\
        O[kk*C_ob + jj] = c_tile[kk*C_ob + jj];\
    }\
}\
//Convolution Computation
//(Strided GEMM)
// Pointer to C defined in the outer scope
#define FMA_TILE_C(step, a, b, p_cur)  \
 {\
   float *c_pixel;\
   for(int kk = 0; kk < W_ob; kk++)\
   {\
     float a_val = *(a + p_cur + kk*step);\
     c_pixel = c_tile + kk*C_ob;\
     for(int jj = 0; jj < C_ob; jj++) \
     {\
       float b_val = *(b + p_cur + jj);\
       *(c_pixel+jj) += a_val*b_val;\
     }\
   }\
 }


#define FMA_END_C(step, a, b, p_cur, W_last)\
{\
  float *c_pixel;\
  for(int kk = 0; kk < W_last; kk++)\
  {\
    float a_val = *(a + p_cur + kk*step);\
    c_pixel = c_tile + kk*C_ob;\
    for(int jj = 0; jj < C_ob; jj++) \
    {\
      float b_val = *(b + p_cur + jj);\
      *(c_pixel+jj) += a_val*b_val;\
    }\
  }\
}
