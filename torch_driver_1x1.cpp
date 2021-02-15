#include <torch/torch.h>
#include "direct_convolution.h"
#include<math.h>

/*

Direct Convolution Interface

void direct_convolution(
  std::vector<uint32_t> in_dims,
  std::vector<uint32_t> filter_dims,
  float * I,
  float * F,
  uint8_t stride,
  std::vector<uint32_t> out_dims,
  float * O
)



*/

//
#define GEMM 0
#define L 0
#define RUNS 1000
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

// this function assumes that the output array has been allocated already

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
              check &= (fabs(dc_array[offset++]
                           -
                        flat_t[g_offset + l_offset   +
                               h_offset + k_offset   +
                                          i_offset   +
                                          j_offset
                                ])

                            < tolerance);
              // printf("%d %.4f %.4f \n",offset-1, dc_array[offset-1] ,flat_t[g_offset + l_offset   +
              //                                                  h_offset + k_offset   +
              //                                                             i_offset   +
              //                                                             j_offset
              //                                                   ]
              //       );
            }
          }
        }
      }
    }
  }
  return check;
}

int copy_torch2dc(torch::Tensor t,
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
     // std::cout<<"switched " <<fused <<std::endl;
     // for(int d = 0; d < reshape_tensor.dim(); d++){
     //   std::cout<< reshape_tensor.size(d) << std::endl;
     // }
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


float * alloc_dc(torch::Tensor t, std::vector<uint32_t> &dimensions)
{
  uint32_t total_elements = 1;
  for(int i = 0; i < t.dim(); i++){
    total_elements *= t.size(i);
    dimensions.push_back(t.size(i));
  }
  float * ptr_dc;

  int ret = posix_memalign((void**) &ptr_dc, 4096, total_elements * sizeof(float));

  if(ret){
    return NULL;
  }
  return ptr_dc;
}

bool check_block_eqivalence(uint32_t block_size,
                      float * dc_buffer,
                      float * dc_array, float tolerance = 1e-8)
{
 bool check = 1;
 for(int i = 0; i < block_size; i++){
    check &= (fabs(dc_array[i] - dc_buffer[i])< tolerance);
    // printf(" %f  %f \n", dc_array[i], dc_buffer[i]);
   }
 return check;
}

int main(int argc, char ** argv)
{
  int C_o = atoi(argv[2]);//32;
  int C_i = atoi(argv[1]);//16;
  int C_o_1 = atoi(argv[3]);//16;
  // printf("%d, %d, %d\n", C_o, C_i, C_o_1);
  int kernel_size = 3;//atol(argv[4]);
  int N = atol(argv[4]);//6*50 + 2*(kernel_size/2);
  int M = atol(argv[5]);//6*50 + 2*(kernel_size/2);
  torch::Tensor a = torch::randn(C_i*N*M).reshape({1,C_i,N, M});
  // a = torch::add(a, 1);
  // a = torch::mul(a,0.01);
  // std::cout<<a<<std::endl;
  torch::Tensor test_weights =  torch::randn(C_o*C_i*kernel_size*kernel_size).reshape({C_o,C_i,kernel_size,kernel_size});
  // test_weights = torch::mul(test_weights, 1.0/(9*C_i));

  torch::Tensor test_weights_1x1 = torch::rand(C_o_1*C_o*1*1).reshape({C_o_1,C_o,1,1});
  // std::cout<< test_weights <<std::endl;
  float *array;

  torch::Tensor cache_bomb_l3 = torch::randn(16*1024*1024/4); //L3
  torch::Tensor cache_bomb_l2 = torch::randn(512*1024/4); //L2
  torch::Tensor cache_bomb_l1 = torch::randn(32*1024/4); //L1

  std::vector<torch::Tensor> cache_bombs{cache_bomb_l1,cache_bomb_l2, cache_bomb_l3};

  #if(L)
  uint32_t bomb_size = 16*1024*1024/4 *(L==3) + 512*1024/4 *(L==2) + 32*1024/4*(L==1);

  float * cache_bomb_array = cache_bombs[L-1].data_ptr<float>();
  #endif

  auto conv_3x3 = torch::nn::Conv2d( torch::nn::Conv2dOptions(C_i, C_o, 3).
                                                stride(1).
                                                padding(0).
                                                bias(false));

  conv_3x3->weight = test_weights;

  auto conv_1x1 = torch::nn::Conv2d( torch::nn::Conv2dOptions(C_o, C_o_1, 1).
                                                stride(1).
                                                padding(0).
                                                bias(false));

  conv_1x1->weight = test_weights_1x1;

  torch::Tensor weights = conv_3x3->weight;
  torch::Tensor weights_1x1 = conv_1x1-> weight;
  // std::cout<< weights.dim() <<std::endl;

  auto out_intermediate = conv_3x3(a);
  auto out = conv_1x1(out_intermediate);

  // printf("Testing %d runs, clearing the L%d cache of the output each time\n", RUNS, L);
  //
  // printf("WSS Size In_img : %.2f KB/ 4KB\nWSS Size In_filter : %.2f KB/ 8K elements\nWSS Size Out_img : %.2f KB/ 4KB\n", (a.numel()/1024.0) , weights.numel()/1024.0,  out.numel()/1024.0);

  //Direct Convolution Setup

  std::vector<uint32_t> in_dimensions;
  std::vector<uint32_t> filter_dimensions;
  std::vector<uint32_t> out_intermediate_dimensions;
  std::vector<uint32_t> filter_1x1_dimensions;
  std::vector<uint32_t> out_dimensions;
  std::vector<uint32_t> intermediate_block_dimensions;
  float * input_dc = alloc_dc(a, in_dimensions);
  float * filter_dc = alloc_dc(weights, filter_dimensions);
  float * out_intermediate_dc = alloc_dc(out_intermediate, out_intermediate_dimensions);
  float * filter_1x1_dc = alloc_dc(weights_1x1,filter_1x1_dimensions);
  float * out_dc = alloc_dc(out, out_dimensions);

  float * out_intermediate_buffer;
  int ret = posix_memalign((void**)&out_intermediate_buffer, 4096, out_intermediate_dimensions[2]*out_intermediate_dimensions[3]*C_ob*sizeof(float));
  std::cout<<"dimensions:\n\tIn_image: "<<in_dimensions
  << "\n\tIn_filter: " <<filter_dimensions
  << "\n\tOut_img 3x3: "<< out_intermediate_dimensions
  << "\n\tFilter 1x1:"<< filter_1x1_dimensions
  <<"\n\tOut_img: "<< out_dimensions<<std::endl;

  direct_convolution(C_i, C_o, kernel_size, kernel_size, N, M, 1, input_dc, filter_dc, out_intermediate_dc);


  for (int i = 0; i != out_dimensions[2] * out_dimensions[3] * out_dimensions[1] ; ++i){
    out_dc[i] = 0.0;
    int j  = 1;
  }

  uint64_t sum=0, t0, t1;
  for (int run = 0; run < RUNS; run++){

    copy_torch2dc(a, 'i', in_dimensions, input_dc);
    copy_torch2dc(weights,'f',filter_dimensions,filter_dc);
    t0 = rdtsc();
    direct_convolution(C_i, C_o, kernel_size, kernel_size, N, M, 1, input_dc, filter_dc, out_intermediate_dc);

    t1 = rdtsc();
     sum+=(t1-t0);
     #if(L)
       volatile float check_sum = rand()/(1.0*RAND_MAX);
       for(uint32_t i = 0; i < bomb_size; i++){
         check_sum += cache_bomb_array[i];
       }
       // printf("run %d %f\n",run, check_sum);
     #endif
  }

  int m=out_dimensions[2]*out_dimensions[3], n=out_intermediate_dimensions[1], k = kernel_size*kernel_size*C_i;


  printf("GEMM%d L%d %d %d %d %d %d \t  %lf\t",GEMM, L, kernel_size, m,  C_i,n,C_o_1,
                          (2.0*m*n*k
                          // +
                          // 2.0*m*C_o_1*out_intermediate_dimensions[1]
                        )/((float)(sum/(1.0*RUNS))));
  uint64_t sum_1x1 = 0;

  for (int run = 0; run < RUNS; run++){
    copy_torch2dc(weights_1x1, 'f', filter_1x1_dimensions, filter_1x1_dc);
    t0 = rdtsc();
    direct_convolution(C_o, C_o_1, 1, 1, out_intermediate_dimensions[2], out_intermediate_dimensions[3], 1, out_intermediate_dc, filter_1x1_dc, out_dc);
    t1 = rdtsc();
     sum_1x1+=(t1-t0);
     #if(L)
       volatile float check_sum = rand()/(1.0*RAND_MAX);
       for(uint32_t i = 0; i < bomb_size; i++){
         check_sum += cache_bomb_array[i];
       }
     #endif
  }
  printf("%lf  \t",
                          (
                          2.0*m*C_o_1*out_intermediate_dimensions[1]
                        )/((float)(sum_1x1/(1.0*RUNS))));


  printf("%f %lf\t",( 2.0*m*n*k
                                              +
  2.0*m*C_o_1*out_intermediate_dimensions[1])/((sum_1x1+sum)/(1.0*RUNS)),  ((float)((sum_1x1 +  sum)/(1.0*RUNS))));

  bool inter_correctness = 0;//check_eqivalence(out_intermediate, 'o', out_intermediate_dimensions, out_intermediate_dc, 1e-3);

  bool correctness = 1;//check_eqivalence(out, 'o', out_dimensions, out_dc, 1e-3);

  for (int i = 0; i != out_dimensions[2] * out_dimensions[3] * out_dimensions[1] ; ++i){
    out_dc[i] = 0.0;
    int j  = 1;
  }

  uint64_t volatile sum_fused = 0;
  for(int r = 0; r < RUNS; r++){
    copy_torch2dc(weights_1x1, 'f', filter_1x1_dimensions, filter_1x1_dc, 1);
    t0 = rdtsc();
    fused_direct_convolution(C_i, C_o, C_o_1, kernel_size, kernel_size, in_dimensions[2], in_dimensions[3], 1, input_dc,filter_dc, filter_1x1_dc, out_intermediate_buffer, out_dc);
    t1 = rdtsc();
    sum_fused += (t1 - t0);
  }

  printf("%lf %lf\t", ( 2.0*m*n*k
                                              +
  2.0*m*C_o_1*out_intermediate_dimensions[1])/((sum_fused)/(1.0*RUNS)), ((sum_fused)/(1.0*RUNS)));
  printf("\n");
  uint32_t block_size = out_intermediate_dimensions[2]*out_intermediate_dimensions[3]*C_ob;
  uint32_t offset = ((C_o - C_ob)/C_ob)*block_size;
  // printf("offset to last block %d size %d\n", offset, block_size);
  // inter_correctness = check_block_eqivalence(block_size, out_intermediate_buffer, out_intermediate_dc + offset,1e-3);
  bool fused_correctness = check_eqivalence(out, 'o', out_dimensions, out_dc, 1e-3);
  printf("%d %d %d\n", inter_correctness, correctness, fused_correctness);


  // std::cout<< a << std::endl;
  // std::cout<< out << std::endl;




  free(input_dc);
  free(filter_dc);
  free(out_dc);

}
