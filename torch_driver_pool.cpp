#include <torch/torch.h>
#include<math.h>
#include<assert.h>
#include "direct_convolution.h"
#include "utils.h"
// Pooling driver

#define GEMM 0
#define L 3
#define RUNS 1000
#define VERBOSE 1

//Good Ol' Timing
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


int main(int argc, char ** argv)
{
  if(argc!=6)
  {
    printf("USAGE: torch_1x1 < 3x3 Input Channels> <3x3 Output Channels> <1x1 OutputChannels>  <Output Height> <Output Width (multiple of 6)>\n");
    return 0;
  }

  // Setup Problem Size from command line variables
  int C_i = atoi(argv[1]);
  int C_o = atoi(argv[2]);
  int C_o_1 = atoi(argv[3]);

  int kernel_size = 3;
  int stride = 1;
  int N = (atol(argv[4]) - 1) * stride + kernel_size;
  int M = atol(argv[5]) + 2*(kernel_size/2);
  if(M%6 != 2){
    printf(" Please check that Output Width is a multiple of 6");
    return 0;
  }


  // Create and Initialize Pytorch tensors
  torch::manual_seed(1729);
  torch::Tensor a = torch::randn(C_i*N*M).reshape({1,C_i,N, M});
  a = torch::add(a, 0.1);
  // for(int i = 0; i < C_i; i++){
  //   for(int j = 1; j < 4; j++){
  //     for(int k = 6; k < 12; k++){
  //       // auto cur = a.index({i, j, k});
  //       a.index_put_({0,i, j, k}, 1.0*(j*M+k));
  //     }
  //   }
  // }
  a = torch::mul(a, 0.1);
  torch::Tensor test_weights =  torch::ones(C_o*C_i*kernel_size*kernel_size).reshape({C_o,C_i,kernel_size,kernel_size});
  test_weights = torch::mul(test_weights,1.0/9.0);
  torch::Tensor test_weights_1x1 = torch::rand(C_o_1*C_o*1*1).reshape({C_o_1,C_o,1,1});
  // float *array;


  // Create PyTorch Convolution layers
  // set weights to generated values
  auto conv_3x3 = torch::nn::Conv2d( torch::nn::Conv2dOptions(C_i, C_o, 3).
                                                stride(1).
                                                padding(0).
                                                bias(false));

  conv_3x3->weight = test_weights;

  auto pool = torch::nn::MaxPool2d( torch::nn::MaxPool2dOptions(3).
                                                stride(2).
                                                padding(0));



  //Run Inference
  uint32_t t0, t1;
  uint64_t sum_pytorch = 0;
  torch::Tensor out_intermediate, out;
  for(uint32_t r = 0; r < RUNS; r++)
  {
    t0 = rdtsc();
    out_intermediate = conv_3x3(a);
    out = pool(out_intermediate);
    t1 = rdtsc();
    sum_pytorch += (t1 - t0);
  }


  //Direct Convolution Setup

  // Copy layer weights to temporaries
  torch::Tensor weights = conv_3x3->weight;

  std::vector<uint32_t> in_dimensions;
  std::vector<uint32_t> filter_dimensions;
  std::vector<uint32_t> out_intermediate_dimensions;
  std::vector<uint32_t> out_dimensions;

  std::vector<uint32_t> intermediate_block_dimensions;
  float * input_dc = alloc_dc(a, in_dimensions);
  float * filter_dc = alloc_dc(weights, filter_dimensions);
  float * out_intermediate_dc = alloc_dc(out_intermediate, out_intermediate_dimensions);
  float * out_dc = alloc_dc(out, out_dimensions);

  float * out_intermediate_buffer;
  int ret = posix_memalign((void**)&out_intermediate_buffer, 4096, out_intermediate_dimensions[2]*out_intermediate_dimensions[3]*C_ob*sizeof(float));

  #if(VERBOSE)
  printf("Testing %d runs, clearing the upto the L%d cache of the output each time\n", RUNS, L);
  printf("WSS Size In_img : %.2f K/8K elements  dims: %u %u %u\n\
WSS Size In_filter 3x3: %.2f K/8K elements  dims: %u %u %u %u\n\
WSS Size Out_img 3x3 : %.2f K/8K elements  dims: %u %u %u\n\
WSS Size Out_img pool : %.2f K/8K elements  dims: %u %u %u\n\
",  a.numel()/1024.0, in_dimensions[1], in_dimensions[2], in_dimensions[3],
       weights.numel()/1024.0, filter_dimensions[0], filter_dimensions[1], filter_dimensions[2], filter_dimensions[3],
       out_intermediate.numel()/1024.0, out_intermediate_dimensions[1], out_intermediate_dimensions[2], out_intermediate_dimensions[3],
       out.numel()/1024.0, out_dimensions[1], out_dimensions[2], out_dimensions[3]
                                                           );
  #endif



  #if(L)
  uint32_t bomb_size = 16*1024*1024/4 *(L==3) + 512*1024/4 *(L==2) + 32*1024/4*(L==1);
  torch::Tensor cache_bomb= torch::randn(bomb_size); //L3
  float * cache_bomb_array = cache_bomb.data_ptr<float>();
  #endif


  // Initialize Outputs to 0
  for (int i = 0; i != out_dimensions[2] * out_dimensions[3] * out_dimensions[1] ; ++i){
    out_dc[i] = 0.0;
    int j  = 1;
  }

  uint64_t sum=0;

  // 3x3 unfused
  for (int run = 0; run < RUNS; run++){
    // Copy Inputs to their flat buffers
    copy_torch2dc(a, 'i', in_dimensions, input_dc);
    copy_torch2dc(weights,'f',filter_dimensions,filter_dc);
    t0 = rdtsc();
    direct_convolution(C_i, C_o, kernel_size, kernel_size, N, M, stride, input_dc, filter_dc, out_intermediate_dc);
    pooling(C_o, out_intermediate_dimensions[2], out_intermediate_dimensions[3] ,out_intermediate_dc, out_dc);
    t1 = rdtsc();
     sum+=(t1-t0);
     #if(L)
       volatile float check_sum = rand()/(1.0*RAND_MAX);
       for(uint32_t i = 0; i < bomb_size; i++){
         check_sum += cache_bomb_array[i];
       }
     #endif
  }

  int m=out_dimensions[2]*out_dimensions[3], n=out_intermediate_dimensions[1], k = kernel_size*kernel_size*C_i;


  printf("GEMM %d L%d %d %d %d %d %d \t  %lf\t",GEMM, L, kernel_size, m,  C_i,n,C_o_1,
                          (2.0*m*n*k
                          +
                          out_dimensions[2]*out_dimensions[3]* out_dimensions[1]*9
                        )/((float)(sum/(1.0*RUNS))));
  uint64_t sum_pool = 0;


  // pooling unfused


  // for (int run = 0; run < RUNS; run++){
  //   // Copy Inputs to their flat buffers
  //   t0 = rdtsc();
  //   t1 = rdtsc();
  //    sum_pool+=(t1-t0);
  //    #if(L)
  //      volatile float check_sum = rand()/(1.0*RAND_MAX);
  //      for(uint32_t i = 0; i < bomb_size; i++){
  //        check_sum += cache_bomb_array[i];
  //      }
  //    #endif
  // }

  // printf("%lf\t",
  //                         (out_dimensions[2]*out_dimensions[3]* out_dimensions[1]*9
  //                         // +
  //                         // 2.0*m*C_o_1*out_intermediate_dimensions[1]
  //                       )/((float)(sum/(1.0*RUNS))));
 // printf("%lf\t",  2.0*m*n*k + out_dimensions[2]*out_dimensions[3]* out_dimensions[1]*9)/((sum_fused)/(1.0*RUNS)))
  uint64_t volatile sum_fused = 0;
  // fused
  for(int r = 0; r < RUNS; r++){
    t0 = rdtsc();
    fused_pooling_direct_convolution(C_i, C_o, kernel_size,kernel_size, N, M, stride, input_dc,filter_dc, out_intermediate_buffer, out_dc);
    t1 = rdtsc();
    sum_fused += (t1 - t0);
    #if(L)
      volatile float check_sum = rand()/(1.0*RAND_MAX);
      for(uint32_t i = 0; i < bomb_size; i++){
        check_sum += cache_bomb_array[i];
      }
    #endif
  }

  printf("%lf %lf\t", ( 2.0*m*n*k + out_dimensions[2]*out_dimensions[3]* out_dimensions[1]*9)/((sum_fused)/(1.0*RUNS)), ((sum_fused)/(1.0*RUNS)));
  printf("\n");

  uint32_t block_size = out_intermediate_dimensions[2]*out_intermediate_dimensions[3]*C_ob;
  uint32_t offset = ((C_o - C_ob)/C_ob)*block_size; //last block

  bool correctness = 1;//check_eqivalence(out_intermediate,'o', out_intermediate_dimensions, out_intermediate_dc, 1e-3);
  bool inter_correctness = 1;//check_block_eqivalence(block_size, out_intermediate_buffer, out_intermediate_dc + offset,1e-3);
  correctness &= check_eqivalence(out, 'o', out_dimensions, out_dc, 1e-3);


  #if(VERBOSE)
    printf("%d %d\n", inter_correctness, correctness);
  #endif

  //Make sure all outputs match pytorch
  assert(inter_correctness&correctness==1);





  free(input_dc);
  free(filter_dc);
  free(out_dc);
  free(out_intermediate_dc);
  free(out_intermediate_buffer);

}
