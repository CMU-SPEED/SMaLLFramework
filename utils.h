// A set of functions to help interface between Libtorch and this codebase

//Utility functions to Interface between libtorch and C++



// allocate a page aligned flatbuffer of size Tensor.numel()
// Copy the dimensions of Tensor to a C++ vector to iterate over
float * alloc_dc(torch::Tensor t, std::vector<uint32_t> &dimensions)
{
  uint32_t total_elements = 1;
  for(int i = 0; i < t.dim(); i++){
    total_elements *= t.size(i);
    dimensions.push_back(t.size(i));
  }
  float * ptr_dc;

  int ret = posix_memalign((void**) &ptr_dc, 4096, t.numel() * sizeof(float));

  if(ret){
    return NULL;
  }
  return ptr_dc;
}

// copey elements of the torch tensor into the flat buffer allocated above in the
// direct conv data layout
int copy_torch2dc(torch::Tensor t,
                      char type, // 'i' = input, 'f' = filter, 'o'  = output
                      std::vector<uint32_t> dim_sizes,
                      float * dc_array, bool fused = false)
{
   // float * flat_t = t.data_ptr<float>();
   // uint32_t dim5, dim_4, dim_3, dim_2, dim_1, dim_0;
   uint32_t dim_3, dim_2, dim_1, dim_0;

   uint32_t ip_block, op_block;

   dim_3 = dim_sizes[0]; //C_o
   dim_2 = dim_sizes[1]; //C_i
   dim_1 = dim_sizes[2]; //H_f
   dim_0 = dim_sizes[3]; //W_f

   uint32_t total_elements = dim_3*dim_2*dim_1*dim_0;
   torch::Tensor reshape_tensor;
   if(type=='i'){
     //input
     ip_block = C_ib;
     op_block = 1;
     reshape_tensor = t.reshape({dim_sizes[1]/C_ib, C_ib, dim_sizes[2], dim_sizes[3]});
     reshape_tensor = reshape_tensor.permute({0,2,3,1});
   }
   else if(type=='o'){
     //output
     ip_block = C_ob;
     op_block = 1;
     reshape_tensor = t.reshape({dim_sizes[1]/C_ob, C_ob, dim_sizes[2], dim_sizes[3]});
     reshape_tensor = reshape_tensor.permute({0,2,3,1});

   }
   else if(type == 'f'){
     //filter
     if(fused){
       reshape_tensor = t.reshape({dim_sizes[0]/C_ob, C_ob, dim_sizes[1]/C_ib, C_ib, dim_sizes[2], dim_sizes[3]});
       reshape_tensor = reshape_tensor.permute({2,0,4,5,3,1});
     }
     else{
       reshape_tensor = t.reshape({dim_sizes[0]/C_ob, C_ob, dim_sizes[1]/C_ib, C_ib, dim_sizes[2], dim_sizes[3]});
       reshape_tensor = reshape_tensor.permute({0,2,4,5,3,1});
     }
   }
   else{
     printf("unsupported type. Use one of:\n  \'i\' = input \n \'f\' = filter \n \'o\'  = output \n");
     return 0;
   }
   // reshape_tensor.to(torch::kCPU);
   auto flattened_tensor = torch::flatten(reshape_tensor);
   // copying
   float * flat_t = flattened_tensor.data_ptr<float>();
   uint32_t offset = 0;
   for(uint32_t flattened_index = 0; flattened_index < total_elements; ++flattened_index){
      dc_array[flattened_index] = flat_t[flattened_index];
   }
   return 1;
}

// Tiling in the width dimenstion (Ignore for now)
int copy_torch2dc_gemm(torch::Tensor t,
                      char type,
                      std::vector<uint32_t> dim_sizes,
                      float * dc_array, bool fused = false)
{
   // float * flat_t = t.data_ptr<float>();
   // uint32_t dim5, dim_4, dim_3, dim_2, dim_1, dim_0;
   uint32_t dim_3, dim_2, dim_1, dim_0;

   uint32_t ip_block, op_block;

   dim_3 = dim_sizes[0]; //C_o
   dim_2 = dim_sizes[1]; //C_i
   dim_1 = dim_sizes[2]; //H_f
   dim_0 = dim_sizes[3]; //W_f

   uint32_t total_elements = dim_3*dim_2*dim_1*dim_0;
   torch::Tensor reshape_tensor;
   if(type=='i'){
     //input
     ip_block = C_ib;
     op_block = 1;
     reshape_tensor = t.reshape({dim_sizes[1]/C_ib, C_ib, dim_sizes[2], dim_sizes[3]/W_ob,W_ob});
     reshape_tensor = reshape_tensor.permute({0,2,3,1,4});

   }
   else if(type=='o'){
     //output
     ip_block = C_ob;
     op_block = 1;
     reshape_tensor = t.reshape({dim_sizes[1]/C_ob, C_ob, dim_sizes[2], dim_sizes[3]});
     reshape_tensor = reshape_tensor.permute({0,2,3,1});

   }
   else if(type == 'f'){
     //filter
     if(fused){
       reshape_tensor = t.reshape({dim_sizes[0]/C_ob, C_ob, dim_sizes[1]/C_ib, C_ib, dim_sizes[2], dim_sizes[3]});
       reshape_tensor = reshape_tensor.permute({2,0,4,5,3,1});
     }
     else{
       reshape_tensor = t.reshape({dim_sizes[0]/C_ob, C_ob, dim_sizes[1]/C_ib, C_ib, dim_sizes[2], dim_sizes[3]});
       reshape_tensor = reshape_tensor.permute({0,2,4,5,3,1});
     }
     ip_block = C_ib;
     op_block = C_ob;

   }
   else{
     return 0;
   }
   // reshape_tensor.to(torch::kCPU);
   auto flattened_tensor = torch::flatten(reshape_tensor);
   // copying
   float * flat_t = flattened_tensor.data_ptr<float>();
   uint32_t offset = 0;
   for(uint32_t flattened_index = 0; flattened_index < total_elements; ++flattened_index){
      dc_array[flattened_index] = flat_t[flattened_index];
   }




   return 1;
}

// check whether the pytorch output matches C++ output
bool check_eqivalence(torch::Tensor t,
                      char type,
                      std::vector<uint32_t> dim_sizes,
                      float * dc_array, float tolerance = 1e-8)
{
 bool check = 1;
 float * flat_t = t.data_ptr<float>();
 // uint32_t dim5, dim_4, dim_3, dim_2, dim_1, dim_0;
 uint32_t dim_3, dim_2, dim_1, dim_0;

 uint32_t ip_block, op_block;

 dim_3 = dim_sizes[0]; //C_o
 dim_2 = dim_sizes[1]; //C_i
 dim_1 = dim_sizes[2]; //H_f
 dim_0 = dim_sizes[3]; //W_f


 if(type=='i'){
   //input
   ip_block = C_ib;
   op_block = 1;
 }
 else if(type=='o'){
   //output
   ip_block = C_ob;
   op_block = 1;

 }

 else if(type == 'f'){
   //filter
   ip_block = C_ib;
   op_block = C_ob;

 }
 else{
   return 0;
 }
 // copying
 uint32_t offset = 0;
 for(uint32_t g = 0; g < dim_3; g+= op_block){
   uint32_t g_offset =  g*dim_2*dim_1*dim_0;
   for(uint32_t h = 0; h < dim_2; h+= ip_block){
     uint32_t h_offset = h*dim_1*dim_0;
     for(uint32_t i = 0; i < dim_1; i++){
       uint32_t i_offset = i*dim_0;
       for(uint32_t j = 0; j < dim_0; j++){
         uint32_t j_offset = j;
         for(uint32_t k = 0; k < ip_block; k++){
           uint32_t k_offset = k*dim_1*dim_0;
           for(uint32_t l = 0; l < op_block; l++){
             int l_offset = l*dim_2*dim_1*dim_0;
             // printf("offset: %d\n", offset);fflush(0);
             check &= (fabs(dc_array[offset++]
                          -
                       flat_t[g_offset + l_offset   +
                              h_offset + k_offset   +
                                         i_offset   +
                                         j_offset
                               ])

                           < tolerance);
          // if((fabs(dc_array[offset-1]
          //              -
          //           flat_t[g_offset + l_offset   +
          //                  h_offset + k_offset   +
          //                             i_offset   +
          //                             j_offset
          //                   ])
          //
          //               > tolerance))
          //   {
          //    printf("%d %.4f %.4f %f\n",offset-1, dc_array[offset-1] ,flat_t[g_offset + l_offset   +
          //                                                     h_offset + k_offset   +
          //                                                                i_offset   +
          //                                                                j_offset
          //                                                      ],
          //                                                      (fabs(dc_array[offset-1]
          //                                                                   -
          //                                                                flat_t[g_offset + l_offset   +
          //                                                                       h_offset + k_offset   +
          //                                                                                  i_offset   +
          //                                                                                  j_offset
          //                                                                        ]))
          //          );
          //   }
           }
         }
       }
     }
   }
 }
 return check;
}
//check whether the last block of the unfused 3x3 output matches the output buffer of the fused convolution
bool check_block_eqivalence(uint32_t block_size,
                      float * dc_buffer,
                      float * dc_array, float tolerance = 1e-8)
{
 bool check = 1;
 for(int i = 0; i < block_size; i++){
    check &= (fabs(dc_array[i] - dc_buffer[i])< tolerance);
    // printf(" %f  %f %d\n", dc_array[i], dc_buffer[i], check);
   }
 return check;
}

// bool interleave_arrays(float * a, float * b, uint32_t length, float * interleaved){
//
//   uint32_t offset = 0;
//
//   for(uint32_t i = 0; i < length; i+=16){
//     for(uint32_t j = 0; j < 16; j++){
//       interleaved[offset] = a[i+j];
//       offset++;
//     }
//     for(uint32_t j = 0; j < 16; j++){
//       interleaved[offset] = b[i+j];
//       offset++;
//     }
//   }
//
// }
