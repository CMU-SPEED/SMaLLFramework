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
#define L 0
#define RUNS 10
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
                      float * dc_array)
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
     ip_block = C_ib;
     op_block = C_ob;
     reshape_tensor = t.reshape({dim_sizes[0]/C_ob, C_ob, dim_sizes[1]/C_ib, C_ib, dim_sizes[2], dim_sizes[3]});
     reshape_tensor = reshape_tensor.permute({0,2,4,5,3,1});
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
   // for(uint32_t g = 0; g < dim_3; g+= op_block){
   //   uint32_t g_offset =  g*dim_2*dim_1*dim_0;
   //   for(uint32_t h = 0; h < dim_2; h+= ip_block){
   //     uint32_t h_offset = h*dim_1*dim_0;
   //     for(uint32_t i = 0; i < dim_1; i++){
   //       uint32_t i_offset = i*dim_0;
   //       for(uint32_t j = 0; j < dim_0; j++){
   //         uint32_t j_offset = j;
   //         for(uint32_t k = 0; k < ip_block; k++){
   //           uint32_t k_offset = k*dim_1*dim_0;
   //           for(uint32_t l = 0; l < op_block; l++){
   //             int l_offset = l*dim_2*dim_1*dim_0;
   //             dc_array[offset++] = flat_t[g_offset + l_offset   +
   //                                         h_offset + k_offset   +
   //                                                    i_offset   +
   //                                                    j_offset];
   //
   //
   //           }
   //         }
   //       }
   //     }
   //   }
   // }



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

  int ret = posix_memalign((void**) &ptr_dc, 64, total_elements * sizeof(float));

  if(ret){
    return NULL;
  }
  return ptr_dc;
}

int main(){
  int C_o = 16;
  int C_i = C_ib;
  int kernel_size = 1 ;
  int N = 6*11 + 2*(kernel_size/2);
  int M = 6*11 + 2*(kernel_size/2);
  torch::Tensor a = torch::randn(C_i*N*M).reshape({1,C_i,N, M});
  a = torch::mul(a,0.01);
  // std::cout<<a<<std::endl;
  torch::Tensor test_weights =  torch::randn(C_o*C_i*kernel_size*kernel_size).reshape({C_o,C_i,kernel_size,kernel_size});
  test_weights = torch::mul(test_weights, 1.0/(9*C_i));


  torch::Tensor cache_bomb_l3 = torch::randn(16*1024*1024/4); //L3
  torch::Tensor cache_bomb_l2 = torch::randn(512*1024/4); //L2
  torch::Tensor cache_bomb_l1 = torch::randn(32*1024/4); //L1

  std::vector<torch::Tensor> cache_bombs{cache_bomb_l1,cache_bomb_l2, cache_bomb_l3};
  uint32_t bomb_size = 16*1024*1024/4 *(L==3) + 512*1024/4 *(L==2) + 32*1024/4*(L==1);
  #if(L)
  float * cache_bomb_array = cache_bombs[L-1].data_ptr<float>();
  #endif

  auto conv_3x3 = torch::nn::Conv2d( torch::nn::Conv2dOptions(C_i, C_o, 3).
                                                stride(1).
                                                padding(0).
                                                bias(false));

  conv_3x3->weight = test_weights;



  torch::Tensor weights = conv_3x3->weight;
  std::cout<< weights.dim() <<std::endl;

  auto out = conv_3x3(a);

  printf("Testing %d runs, clearing the L%d cache of the output each time\n", RUNS, L);

  printf("WSS Size In_img : %.2f K elements/ 8K elements\nWSS Size In_filter : %.2f K elements/ 8K elements\nWSS Size Out_img : %.2f K elements/ 8K elements\n", (a.numel()/1024.0) , weights.numel()/1024.0,  out.numel()/1024.0);

  //Direct Convolution Setup

  std::vector<uint32_t> in_dimensions;
  std::vector<uint32_t> filter_dimensions;
  std::vector<uint32_t> out_dimensions;
  float * input_dc = alloc_dc(a, in_dimensions);
  float * filter_dc = alloc_dc(weights, filter_dimensions);
  float * out_dc = alloc_dc(out, out_dimensions);


  if(out_dc && input_dc && filter_dc != NULL){

    for (int i = 0; i != out_dimensions[2] * out_dimensions[3] * out_dimensions[1] ; ++i){
      out_dc[i] = 0.0;
      int j  = 1;
    }
  }


  std::cout<<"dimensions:\n\tIn_image: "<<in_dimensions << "\n\tIn_filter: " <<filter_dimensions << "\n\tOut_img: "<< out_dimensions<< std::endl;

  //copy inputs


  // direct_convolution(C_i, C_o, kernel_size, kernel_size, N, M, 1, input_dc, filter_dc, out_dc);
  // direct_convolution(C_i, C_o, kernel_size, kernel_size, N, M, 1, input_dc, filter_dc, out_dc);
  // direct_convolution(C_i, C_o, kernel_size, kernel_size, N, M, 1, input_dc, filter_dc, out_dc);
  // direct_convolution(C_i, C_o, kernel_size, kernel_size, N, M, 1, input_dc, filter_dc, out_dc);
  copy_torch2dc(a, 'i', in_dimensions, input_dc);
  copy_torch2dc(weights,'f',filter_dimensions,filter_dc);
  uint64_t sum = 0, t0, t1;
  for (int run = 0; run < RUNS; run++){
    t0 = rdtsc();
    direct_convolution(C_i, C_o, kernel_size, kernel_size, N, M, 1, input_dc, filter_dc, out_dc);
   // direct_convolution(in_dimensions, filter_dimensions, out_dimensions, 1, input_dc, filter_dc, out_dc);

    t1 = rdtsc();
     sum+=(t1-t0);
   // for (int i = 0; i != out_dimensions[2] * out_dimensions[3] * out_dimensions[1] ; ++i){
   //   out_dc[i] = 0.0;
   //   int j  = 1;
   // }
    #if(L)
     volatile float check_sum = rand()/(1.0*RAND_MAX);
     for(uint32_t i = 0; i < bomb_size; i++){
       check_sum += cache_bomb_array[i];
     }
     // printf("run %d %f\n",run, check_sum);
    #endif
  }
  //
  int m=out_dimensions[2]*out_dimensions[3], n=out_dimensions[1], k = kernel_size*kernel_size*C_i;
  //
  // printf("direct conv \n %lf\t", (2.0*m*n*k)/(sum/(1.0*(RUNS))));
  printf("direct : %.1f x %d x %d x %d ops in %f cycles\t%lf ops/cyc\n", 2.0, m, n, k, (sum/(1.0*RUNS)),
                          (2.0*m*n*k
                          // +
                          // 2.0*m*n*C_o
                        )/((float)(sum/(1.0*RUNS))));


  // direct_convolution(C_i, C_o, kernel_size, kernel_size, N, M, 1, input_dc, filter_dc, out_dc);
  //
  // for (int i = 0; i != out_dimensions[2] * out_dimensions[3] * out_dimensions[1] ; ++i){
  //   out_dc[i] /= RUNS;
  //   int j  = 1;
  // }
  bool correct = check_eqivalence(out, 'o', out_dimensions, out_dc, 1e-7);

  std::cout<<"correctness: "<< correct <<std::endl;


  // std::cout<< a << std::endl;
  // std::cout<< out << std::endl;




  free(input_dc);
  free(filter_dc);
  free(out_dc);

}
