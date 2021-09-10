//scalar versions of all the microkernels for platform portability

// Initializations


#define ZERO_TILE_C(W_ob, C_ob) \
  float c_tile[W_ob*C_ob] = {0};



// Loads


#define LOAD_TILE_C(O, W_ob, C_ob)\
  float c_tile[W_ob*C_ob];\
  for (uint32_t kk = 0; kk < W_ob; kk++) {\
      for (uint32_t jj = 0; jj < C_ob; jj++) {\
        c_tile[kk*C_ob + jj] = O[kk*C_ob + jj];\
      }\
  }\

#define LOAD_LAST_C(O,W_ob, C_ob, W_last)\
  float c_tile[W_ob*C_ob];\
  for (uint32_t kk = 0; kk < W_last; kk++) {\
      for (uint32_t jj = 0; jj < C_ob; jj++) {\
        c_tile[kk*C_ob + jj] = O[kk*C_ob + jj];\
      }\
  }\

// strided loads

#define LOAD_TILE_C_strided(O, step, W_ob, C_ob)\
  float c_tile[W_ob*C_ob];\
  for (uint32_t kk = 0; kk < W_ob; kk++) {\
      for (uint32_t jj = 0; jj < C_ob; jj++) {\
        c_tile[kk*C_ob + jj] = O[kk*step + jj];\
      }\
  }\

#define LOAD_LAST_C_strided(O, step, W_ob, C_ob, W_last)\
  float c_tile[W_ob*C_ob];\
  for (uint32_t kk = 0; kk < W_last; kk++) {\
      for (uint32_t jj = 0; jj < C_ob; jj++) {\
        c_tile[kk*C_ob + jj] = O[kk*step + jj];\
      }\
  }\

//Stores

#define STORE_TILE_C(O, W_ob, C_ob) \
for (uint32_t kk = 0; kk < W_ob; kk++) {\
    for (uint32_t jj = 0; jj < C_ob; jj++) {\
        O[kk*C_ob + jj] = c_tile[kk*C_ob + jj];\
    }\
}\

#define STORE_END_C(O, W_ob, C_ob, W_last) \
for (uint32_t kk = 0; kk < W_last; kk++) {\
    for (uint32_t jj = 0; jj < C_ob; jj++) {\
        O[kk*C_ob + jj] = c_tile[kk*C_ob + jj];\
    }\
}\
//Convolution Computation
//(Strided GEMM)
// Pouint32_ter to C defined in the outer scope
#define FMA_TILE_C(step, a, b, p_cur, W_ob, C_ob)  \
 {\
   float *c_pixel;\
   for(uint32_t kk = 0; kk < W_ob; kk++)\
   {\
     float a_val = *(a + p_cur + kk*step);\
     c_pixel = c_tile + kk*C_ob;\
     for(uint32_t jj = 0; jj < C_ob; jj++) \
     {\
       float b_val = *(b + p_cur*C_ob + jj);\
       *(c_pixel+jj) += a_val*b_val;\
     }\
   }\
 }


#define FMA_END_C(step, a, b, p_cur, W_ob, C_ob, W_last)\
{\
  float *c_pixel;\
  for(uint32_t kk = 0; kk < W_last; kk++)\
  {\
    float a_val = *(a + p_cur + kk*step);\
    c_pixel = c_tile + kk*C_ob;\
    for(uint32_t jj = 0; jj < C_ob; jj++) \
    {\
      float b_val = *(b + p_cur*C_ob + jj);\
      *(c_pixel+jj) += a_val*b_val;\
    }\
  }\
}


//Pooling 
//  Max pooling

#define MAX_TILE_C(step, a, W_ob, C_ob)\
{\
  float * c_pixel = c_tile;\
  float * a_pixel = a;\
  for(uint32_t kk = 0; kk < W_ob; kk++)\
  {\
    float * c_channel = c_pixel;\
    float * a_channel = a_pixel;\
    for(uint32_t jj = 0; jj < C_ob; jj++) \
    {\
      *(c_channel) = (*(a_channel)> *(c_channel))? *(a_channel) : *(c_channel) ;\
      c_channel++;\
      a_channel++;\
    }\
    a_pixel += step;\
    c_pixel += C_ob;\
  }\
}

#define MAX_END_C(step, a, W_ob, C_ob, W_last)\
{\
  float * c_pixel = c_tile;\
  float * a_pixel = a;\
  for(uint32_t kk = 0; kk < W_last; kk++)\
  {\
    float * c_channel = c_pixel;\
    float * a_channel = a_pixel;\
    for(uint32_t jj = 0; jj < C_ob; jj++) \
    {\
      *(c_channel) = (*(a_channel)> *(c_channel))? *(a_channel) : *(c_channel) ;\
      c_channel++;\
      a_channel++;\
    }\
    a_pixel += step;\
    c_pixel += C_ob;\
  }\
}
