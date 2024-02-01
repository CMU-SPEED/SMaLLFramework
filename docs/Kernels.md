# Description of Kernels required to support SMaLL on a new platform

## Elementwise Operations
Computation for each output element uses 1 input element.

### Operations with 1 Input Element
These operations reuse (broadcast) the input element over all elements in c_tile
 - GlobalMultiply [FLOAT_DIV_TILE_C](https://github.com/CMU-SPEED/SMaLLFramework/blob/dev/include/small/platforms/reference/intrinsics_float.h#L378)
   _Assumptions_:  all elements in c_tile are initialized with some initial value (LOAD or ZERO)

   _Input_: norm = a value to scale all values in c_tile by, \_W_ob and \_C_ob  = dimensions of c_tile (row major)

   _Operation_: ```c_tile[kk][jj] =  c_tile[kk][jj] * norm```

- Zero [FLOAT_ZERO_TILE_C](https://github.com/CMU-SPEED/SMaLLFramework/blob/dev/include/small/platforms/reference/intrinsics_float.h#L51)

### Operations with 1 Input Buffer (1D)

- Load -> [FLOAT_LOAD_TILE_C](https://github.com/CMU-SPEED/SMaLLFramework/blob/dev/include/small/platforms/reference/intrinsics_float.h#L74)

- Strided Load -> [FLOAT_LOAD_TILE_C_strided](https://github.com/CMU-SPEED/SMaLLFramework/blob/dev/include/small/platforms/reference/intrinsics_float.h#L99)

- Add [FLOAT_ACCUM_TILE_C](https://github.com/CMU-SPEED/SMaLLFramework/blob/dev/include/small/platforms/reference/intrinsics_float.h#L336)

  _Assumptions_:  all elements in c_tile are initialized with some initial value (LOAD or ZERO)

  _Input_: step= stride between rows in buffer a, a buffer a with (step x \_W_ob) rows and  \_C_ob columns (row major)

  _Operation_: ```c_tile[kk][jj] =  c_tile[kk][jj] + a[kk x step][jj]```

- ReLU ->[ FLOAT_MAX_TILE_C  ](https://github.com/CMU-SPEED/SMaLLFramework/blob/dev/include/small/platforms/reference/intrinsics_float.h#L209)
  _Assumptions_:  all elements in c_tile are initialized with "zero"

  _Input_: step= stride between rows in buffer a, a buffer a with (step x \_W_ob) rows and  \_C_ob columns (row major)

  _Operation_: ```c_tile[kk][jj]= (x > 0) ? x: 0; where x = a[kk x step][jj]```

 - Store -> [FLOAT_STORE_TILE_C](https://github.com/CMU-SPEED/SMaLLFramework/blob/dev/include/small/platforms/reference/intrinsics_float.h#L146)

### Operations with 1 Input Buffer (2D)
- Upsample2D  -->[FLOAT_LOAD_TILE_C_upsample](https://github.com/CMU-SPEED/SMaLLFramework/blob/dev/include/small/platforms/reference/intrinsics_float.h#L123)
  _Assumptions_: The values in c_tile can be over-written
  _Input_ :  input buffer I with \_W_ob rows and \_C_ib columns (row major), and _stride_ >=1, number of columns in the output \_C_ob == \_C_ib
  
  _Operation_: ```c_tile[kk][jj] = I[kk/stride][jj]```
  Load rows at a stride of 1/_stride_ and store them in the c_tile array. This has the effect of creating a buffer with the rows in I repeating stride times. [^1]

  
  [^1]: I repeats min(_W_ob, stride times)

### Operations with 1 Input Buffer and 1 Input Element
- LeakyReLU -> [FLOAT_COND_SCALE](https://github.com/CMU-SPEED/SMaLLFramework/blob/dev/include/small/platforms/reference/intrinsics_float.h#L295)


