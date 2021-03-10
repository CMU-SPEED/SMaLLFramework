#include <torch/torch.h>
// unfused
#include "src/direct_convolution.h"
// conv fused dw conv
#include "src/fused_conv_dw.h"

#include<math.h>
#include<assert.h>

#include "src/utils.h"

#define GEMM 0
#define L 0
#define RUNS 1000
#define VERBOSE 0
#define FUSION 0

//Good Ol' Timing
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


int main(int argc, char ** argv)
{
  if(argc!=5)
  {
    printf("USAGE: torch_1x1 < 3x3 Input Channels> <3x3 Output Channels> <Output Height> <Output Width (multiple of 6)>\n");
    return 0;
  }

  // Setup Problem Size from command line variables
  int C_i = atoi(argv[1]);
  int C_o = atoi(argv[2]);
  int C_o_1 = 1;
  constexpr int kernel_size = 1;
  constexpr int stride = 3;
  int N = (atol(argv[3]) - 1) * stride + kernel_size;
  int M = (atol(argv[4]) - 1) * stride + kernel_size;
  if(atol(argv[4])%6 != 0){
    printf(" Please check that Output Width is a multiple of 6");
    return 0;
  }


  // Create and Initialize Pytorch tensors
  torch::Tensor a = torch::randn(C_i*N*M).reshape({1,C_i,N, M});
                           //arange
  a = torch::mul(a, 0.001);
  torch::Tensor test_weights =  torch::randn(C_o*C_i*kernel_size*kernel_size).reshape({C_o,C_i,kernel_size,kernel_size});
  torch::Tensor test_weights_dw = torch::randn(C_o_1*C_o*3*3).reshape({C_o,C_o_1,3,3});

  // float *array;


  // Create PyTorch Convolution layers
  // set weights to generated values
  auto conv_3x3 = torch::nn::Conv2d( torch::nn::Conv2dOptions(C_i, C_o, kernel_size).
                                                stride(stride).
                                                padding(0).
                                                bias(false));

  conv_3x3->weight = test_weights;


  auto conv_dw = torch::nn::Conv2d( torch::nn::Conv2dOptions(C_o, C_o_1*C_o, 3).
                                                stride(1).
                                                padding(0).
                                                groups(C_o).
                                                bias(false));

  conv_dw->weight = test_weights_dw;
    // std::cout<< conv_dw -> weight << std::endl;

  //Run Inference
  uint64_t sum_pytorch = 0, st, et;
  torch::Tensor out, out_intermediate;
  for(uint32_t r = 0; r < RUNS/10; r++){
    st = rdtsc();
    out_intermediate = conv_3x3(a);
    out = conv_dw(out_intermediate);
    et = rdtsc();
    sum_pytorch +=(et - st);
  }

  double torch_ops = 2.0*out_intermediate.numel() * (kernel_size*kernel_size*C_i)
                    + (2.0*out.numel()              * (9));
  printf("%lf\t%lf\t",torch_ops/(1.0*sum_pytorch/(RUNS/10)), (1.0*sum_pytorch/(RUNS/10)));

  //Direct Convolution Setup

  // Copy layer weights to temporaries
  torch::Tensor weights = conv_3x3->weight;
  torch::Tensor weights_dw = conv_dw-> weight.reshape({C_o_1, C_o, 3, 3});
  // std::cout<<weights_dw<< std::endl;
  std::vector<uint32_t> in_dimensions;
  std::vector<uint32_t> filter_dimensions;
  std::vector<uint32_t> out_intermediate_dimensions;
  std::vector<uint32_t> filter_dw_dimensions;
  std::vector<uint32_t> out_dimensions;

  std::vector<uint32_t> intermediate_block_dimensions;
  float * input_dc = alloc_dc(a, in_dimensions);
  float * filter_dc = alloc_dc(weights, filter_dimensions);
  float * out_intermediate_dc = alloc_dc(out_intermediate, out_intermediate_dimensions);
  float * filter_dw_dc = alloc_dc(weights_dw,filter_dw_dimensions);
  float * out_dc = alloc_dc(out, out_dimensions);

  float * out_intermediate_buffer;
  if(C_i > 16)
  {
    int ret = posix_memalign((void**)&out_intermediate_buffer, 4096, out_intermediate_dimensions[2]*out_intermediate_dimensions[3]*C_ob*sizeof(float));
  }
  else{
    // printf("6x16 intermediate size\n");
    #if(FUSION == 1)
    printf("%dx16 tilefor 3x3 output\n",out_dimensions[3] );
    int ret = posix_memalign((void**)&out_intermediate_buffer, 4096, out_dimensions[3]*C_ob*sizeof(float));
    #else
    int ret = posix_memalign((void**)&out_intermediate_buffer, 4096, W_ob*C_ob*sizeof(float));
    #endif
  }
//
//   #if(VERBOSE)
//   printf("Testing %d runs, clearing the upto the L%d cache of the output each time\n", RUNS, L);
//   printf("WSS Size In_img : %.2f K/8K elements  dims: %u %u %u\n\
// WSS Size In_filter 3x3: %.2f K/8K elements  dims: %u %u %u %u\n\
// WSS Size Out_img 3x3 : %.2f K/8K elements  dims: %u %u %u\n\
// WSS Size In_filter 1x1: %.2f K/8K elements  dims: %u %u %u %u\n\
// WSS Size Out_img 1x1 : %.2f K/8K elements  dims: %u %u %u\n\
// ",  a.numel()/1024.0, in_dimensions[1], in_dimensions[2], in_dimensions[3],
//        weights.numel()/1024.0, filter_dimensions[0], filter_dimensions[1], filter_dimensions[2], filter_dimensions[3],
//        out_intermediate.numel()/1024.0, out_intermediate_dimensions[1], out_intermediate_dimensions[2], out_intermediate_dimensions[3],
//        weights_dw.numel()/1024.0, filter_dw_dimensions[0], filter_dw_dimensions[1], filter_dw_dimensions[2], filter_dw_dimensions[3],
//        out.numel()/1024.0, out_dimensions[1], out_dimensions[2], out_dimensions[3]
//                                                            );
//   #endif
//
//
//
//   #if(L)
//   uint32_t bomb_size = 16*1024*1024/4 *(L==3) + 512*1024/4 *(L==2) + 32*1024/4*(L==1);
//   torch::Tensor cache_bomb= torch::randn(bomb_size); //L3
//   float * cache_bomb_array = cache_bomb.data_ptr<float>();
//   #endif
//
//
//   // Initialize Outputs to 0
//   for (int i = 0; i != out_dimensions[2] * out_dimensions[3] * out_dimensions[1] ; ++i){
//     out_dc[i] = 0.0;
//     int j  = 1;
//   }
//
//   uint64_t sum=0, t0, t1;
//
//   copy_torch2dc(weights_dw, 'o', filter_dw_dimensions, filter_dw_dc);
//   copy_torch2dc(a, 'i', in_dimensions, input_dc);
//   copy_torch2dc(weights,'f',filter_dimensions,filter_dc);
//
//   // unfused
//   for (int run = 0; run < RUNS; run++){
//     // Copy Inputs to their flat buffers
//
//     t0 = rdtsc();
//     direct_convolution<stride, kernel_size, kernel_size>(C_i, C_o, N, M, input_dc, filter_dc, out_intermediate_dc);
//     depthwise_convolution(C_o, out_intermediate_dimensions[2], out_intermediate_dimensions[3], out_intermediate_dc, filter_dw_dc, out_dc);
//     t1 = rdtsc();
//      sum+=(t1-t0);
//      #if(L)
//        volatile float check_sum = rand()/(1.0*RAND_MAX);
//        for(uint32_t i = 0; i < bomb_size; i++){
//          check_sum += cache_bomb_array[i];
//        }
//      #endif
//   }
//
//   int m=out_dimensions[2]*out_dimensions[3], n=out_intermediate_dimensions[1], k = kernel_size*kernel_size*C_i;
//
//
//   printf("\tL%d %d\t%d\t%d\t%d\t%lf\t %lf",L, kernel_size,C_i,n,C_o_1,
//                           (2.0*m*n*k
//                           +
//                           2.0*m*C_o_1*out_intermediate_dimensions[1]
//                         )/((float)(sum/(1.0*RUNS))) , (float)(sum/(1.0*RUNS)));
//   uint64_t sum_dw = 0;
//
//   uint64_t volatile sum_fused = 0;
//   // fused
//   if(C_i > 16)
//   {
//     for(int r = 0; r < RUNS; r++){
//       copy_torch2dc(weights_dw, 'f', filter_dw_dimensions, filter_dw_dc, 1);
//       copy_torch2dc(a, 'i', in_dimensions, input_dc);
//       copy_torch2dc(weights,'f',filter_dimensions,filter_dc);
//       t0 = rdtsc();
//       fused_direct_convolution<stride, kernel_size, kernel_size>(C_i, C_o, C_o_1,N, M,input_dc,filter_dc, filter_dw_dc, out_intermediate_buffer, out_dc);
//       t1 = rdtsc();
//       sum_fused += (t1 - t0);
//       #if(L)
//         volatile float check_sum = rand()/(1.0*RAND_MAX);
//         for(uint32_t i = 0; i < bomb_size; i++){
//           check_sum += cache_bomb_array[i];
//         }
//       #endif
//     }
//   }
//   else{
//     for(int r = 0; r < RUNS; r++){
//       copy_torch2dc(weights_dw, 'f', filter_dw_dimensions, filter_dw_dc, 1);
//       copy_torch2dc(a, 'i', in_dimensions, input_dc);
//       copy_torch2dc(weights,'f',filter_dimensions,filter_dc);
//       t0 = rdtsc();
//       #if(FUSION==1)
//       fused_H_direct_convolution_c16<stride, kernel_size, kernel_size>(C_i, C_o, C_o_1,N, M,input_dc,filter_dc, filter_dw_dc, out_intermediate_buffer, out_dc);
//       #else
//       fused_direct_convolution_c16<stride, kernel_size, kernel_size>(C_i, C_o, C_o_1,N, M,input_dc,filter_dc, filter_dw_dc, out_intermediate_buffer, out_dc);
//       #endif
//       t1 = rdtsc();
//       sum_fused += (t1 - t0);
//       #if(L)
//         volatile float check_sum = rand()/(1.0*RAND_MAX);
//         for(uint32_t i = 0; i < bomb_size; i++){
//           check_sum += cache_bomb_array[i];
//         }
//       #endif
//     }
//   }
//
//   printf("\t%lf\t%lf\t", ( 2.0*m*n*k + 2.0*m*C_o_1*out_intermediate_dimensions[1])/((sum_fused)/(1.0*RUNS)), ((sum_fused)/(1.0*RUNS)));
//   printf("\n");
//
//   uint32_t block_size = out_intermediate_dimensions[2]*out_intermediate_dimensions[3]*C_ob;
//   uint32_t offset = ((C_o - C_ob)/C_ob)*block_size; //last block
//
//   // inter_correctness &= check_block_eqivalence(block_size, out_intermediate_buffer, out_intermediate_dc + offset,1e-3);
//   bool fused_correctness = check_eqivalence(out, 'o', out_dimensions, out_dc, 1e-2);
//   // printf("yo\n");fflush(0);
//   #if(VERBOSE)
//     printf("%d %d %d\n", inter_correctness, correctness, fused_correctness);
//   #endif
//
//   //Make sure all outputs match pytorch
//   assert(fused_correctness==1);





  free(input_dc);
  free(filter_dc);
  free(out_dc);
  free(filter_dw_dc);
  free(out_intermediate_dc);
  free(out_intermediate_buffer);

}
