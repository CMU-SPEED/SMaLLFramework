#include <torch/torch.h>
#include "direct_convolution.h"
#include<math.h>
#include<assert.h>

#include "utils.h"
#define GEMM 0
#define L 3
#define RUNS 1000
#define VERBOSE 0
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
  int M = (atol(argv[5]) - 1) * stride + kernel_size;
  if(atol(argv[5])%6 != 0){
    printf(" Please check that Output Width is a multiple of 6");
    return 0;
  }


  // Create and Initialize Pytorch tensors
  torch::Tensor a = torch::randn(C_i*N*M).reshape({1,C_i,N, M});
  torch::Tensor test_weights =  torch::randn(C_o*C_i*kernel_size*kernel_size).reshape({C_o,C_i,kernel_size,kernel_size});
  torch::Tensor test_weights_1x1 = torch::rand(C_o_1*C_o*1*1).reshape({C_o_1,C_o,1,1});
  // float *array;


  // Create PyTorch Convolution layers
  // set weights to generated values
  auto conv_3x3 = torch::nn::Conv2d( torch::nn::Conv2dOptions(C_i, C_o, kernel_size).
                                                stride(stride).
                                                padding(0).
                                                bias(false));

  conv_3x3->weight = test_weights;

  auto conv_1x1 = torch::nn::Conv2d( torch::nn::Conv2dOptions(C_o, C_o_1, 1).
                                                stride(1).
                                                padding(0).
                                                bias(false));

  conv_1x1->weight = test_weights_1x1;


  //Run Inference
  uint64_t sum_pytorch = 0, st, et;
  torch::Tensor out, out_intermediate;
  for(uint32_t r = 0; r < 10; r++){
    st = rdtsc();
    out_intermediate = conv_3x3(a);
    out = conv_1x1(out_intermediate);
    et = rdtsc();
    sum_pytorch +=(et - st);
  }


  //Direct Convolution Setup

  // Copy layer weights to temporaries
  torch::Tensor weights = conv_3x3->weight;
  torch::Tensor weights_1x1 = conv_1x1-> weight;

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

  #if(VERBOSE)
  printf("Testing %d runs, clearing the upto the L%d cache of the output each time\n", RUNS, L);
  printf("WSS Size In_img : %.2f K/8K elements  dims: %u %u %u\n\
WSS Size In_filter 3x3: %.2f K/8K elements  dims: %u %u %u %u\n\
WSS Size Out_img 3x3 : %.2f K/8K elements  dims: %u %u %u\n\
WSS Size In_filter 1x1: %.2f K/8K elements  dims: %u %u %u %u\n\
WSS Size Out_img 1x1 : %.2f K/8K elements  dims: %u %u %u\n\
",  a.numel()/1024.0, in_dimensions[1], in_dimensions[2], in_dimensions[3],
       weights.numel()/1024.0, filter_dimensions[0], filter_dimensions[1], filter_dimensions[2], filter_dimensions[3],
       out_intermediate.numel()/1024.0, out_intermediate_dimensions[1], out_intermediate_dimensions[2], out_intermediate_dimensions[3],
       weights_1x1.numel()/1024.0, filter_1x1_dimensions[0], filter_1x1_dimensions[1], filter_1x1_dimensions[2], filter_1x1_dimensions[3],
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

  uint64_t sum=0, t0, t1;
  // 3x3 unfused
  for (int run = 0; run < RUNS; run++){
    // Copy Inputs to their flat buffers
    copy_torch2dc(a, 'i', in_dimensions, input_dc);
    copy_torch2dc(weights,'f',filter_dimensions,filter_dc);
    t0 = rdtsc();
    direct_convolution(C_i, C_o, kernel_size, kernel_size, N, M, stride, input_dc, filter_dc, out_intermediate_dc);

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
                          // +
                          // 2.0*m*C_o_1*out_intermediate_dimensions[1]
                        )/((float)(sum/(1.0*RUNS))));
  uint64_t sum_1x1 = 0;
  // 1x1 unfused
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
                        )/((float)(sum_1x1/(1.0*RUNS)))); \


  printf("%f %lf\t",( 2.0*m*n*k + 2.0*m*C_o_1*out_intermediate_dimensions[1])/((sum_1x1+sum)/(1.0*RUNS)),  ((float)((sum_1x1 +  sum)/(1.0*RUNS))));

  bool inter_correctness = check_eqivalence(out_intermediate, 'o', out_intermediate_dimensions, out_intermediate_dc, 1e-3);
  bool correctness = check_eqivalence(out, 'o', out_dimensions, out_dc, 1e-3);

  // printf("\n\nconv inter: %d %d\n\n", inter_correctness, correctness);


  uint64_t volatile sum_fused = 0;
  // fused
  for(int r = 0; r < RUNS; r++){
    copy_torch2dc(weights_1x1, 'f', filter_1x1_dimensions, filter_1x1_dc, 1);
    t0 = rdtsc();
    fused_direct_convolution(C_i, C_o, C_o_1, kernel_size,kernel_size,N, M, stride, input_dc,filter_dc, filter_1x1_dc, out_intermediate_buffer, out_dc);
    t1 = rdtsc();
    sum_fused += (t1 - t0);
    #if(L)
      volatile float check_sum = rand()/(1.0*RAND_MAX);
      for(uint32_t i = 0; i < bomb_size; i++){
        check_sum += cache_bomb_array[i];
      }
    #endif
  }

  printf("%lf %lf %lf %lf\t", ( 2.0*m*n*k + 2.0*m*C_o_1*out_intermediate_dimensions[1])/((sum_fused)/(1.0*RUNS)), ((sum_fused)/(1.0*RUNS)),
                                ( 2.0*m*n*k + 2.0*m*C_o_1*out_intermediate_dimensions[1])/((sum_pytorch)/(1.0*RUNS)), ((sum_pytorch)/(1.0*RUNS)));
  printf("\n");

  uint32_t block_size = out_intermediate_dimensions[2]*out_intermediate_dimensions[3]*C_ob;
  uint32_t offset = ((C_o - C_ob)/C_ob)*block_size; //last block

  inter_correctness &= check_block_eqivalence(block_size, out_intermediate_buffer, out_intermediate_dc + offset,1e-3);
  bool fused_correctness = check_eqivalence(out, 'o', out_dimensions, out_dc, 1e-3);

  #if(VERBOSE)
    printf("%d %d %d\n", inter_correctness, correctness, fused_correctness);
  #endif

  //Make sure all outputs match pytorch
  assert(inter_correctness&correctness&fused_correctness==1);





  free(input_dc);
  free(filter_dc);
  free(out_dc);
  free(filter_1x1_dc);
  free(out_intermediate_dc);
  free(out_intermediate_buffer);

}
