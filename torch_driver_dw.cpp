#include <torch/torch.h>
// unfused
#define PARALLEL 1
#include "src/direct_convolution.h"
#include "src/fused_conv_dw_pool.h"
// conv fused dw conv
// #include "src/fused_conv_dw.h"

#include<math.h>
#include<assert.h>

#include "src/utils.h"

#define GEMM 0
#define L 0
#define RUNS 1000
#define VERBOSE 0
#define FUSION 0

#ifndef BUFFER
  #define BUFFER 0
#endif
#define PREFETCH 1
#define H_TILE 0
//Good Ol' Timing
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

#define print_flops( ops,  time,  trials){\
  printf("%.4lf\t", (ops)/(1.0 * time));\
}
#define print_cycles(time,  trials){\
  printf("%.0lf\t", 1.0*(time));\
}

#define MIN(a, b){\
  a = (b < a)?b:a;\
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
  constexpr int kernel_size = 3;
  constexpr int stride = 1;
  int N = (atol(argv[3]) - 1) * stride + kernel_size;
  int M = (atol(argv[4]) - 1) * stride + kernel_size;
  if(atol(argv[4])%6 != 0){
    printf(" Please check that Output Width is a multiple of 6");
    return 0;
  }


  // Create and Initialize Pytorch tensors
  torch::Tensor a = torch::randn(C_i*N*M).reshape({1,C_i,N, M});
                           //arange
  // a = torch::mul(a, 0.001);
  torch::Tensor test_weights =  torch::randn(C_o*C_i*kernel_size*kernel_size).reshape({C_o,C_i,kernel_size,kernel_size});
  test_weights = torch::mul(test_weights, 1.0/(kernel_size*kernel_size*C_i));
  torch::Tensor test_weights_dw = torch::randn(C_o_1*C_o*DW_KERNEL*DW_KERNEL).reshape({C_o,C_o_1,3,3});

  // float * dw_w_ptr = test_weights_dw.data_ptr<float>();
  //
  // for(uint32_t b = 0; b < C_o_1;b++){
  //   for(uint32_t c = 0 ; c < C_o; c++){
  //     for(uint32_t h = 0; h < 3; h++){
  //       for(uint32_t w = 0; w < 3; w++){
  //         *dw_w_ptr=(1)*1.0 + (w+1)*0.1;
  //         dw_w_ptr++;
  //       }
  //     }
  //   }
  // }

  // float *array;


  // Create PyTorch Convolution layers
  // set weights to generated values
  auto conv_3x3 = torch::nn::Conv2d( torch::nn::Conv2dOptions(C_i, C_o, kernel_size).
                                                stride(stride).
                                                padding(0).
                                                bias(false));

  conv_3x3->weight = test_weights;


  auto conv_dw = torch::nn::Conv2d( torch::nn::Conv2dOptions(C_o, C_o_1*C_o, DW_KERNEL).
                                                stride(DW_STRIDE).
                                                padding(0).
                                                groups(C_o).
                                                bias(false));

  conv_dw->weight = test_weights_dw;
  // std::cout<< conv_dw -> weight << std::endl;

  //Run Inference
  uint64_t sum_pytorch = ULLONG_MAX, st, et;
  torch::Tensor out, out_intermediate;
  for(uint32_t r = 0; r < 10; r++){
    st = rdtsc();
    out_intermediate = conv_3x3(a);
    out = conv_dw(out_intermediate);
    et = rdtsc();
    MIN(sum_pytorch, et - st);
  }

  uint64_t conv_ops = out_intermediate.numel() * (kernel_size*kernel_size*C_i * 2.0);
  uint64_t pool_ops = out.numel()              * (3*3*2.0);
  uint64_t effective_conv_h = (out.size(2) - 1) * 2 + 3;
  uint64_t effective_conv_w = (out.size(3) - 1) * 2 + 3;

  uint64_t effective_conv_ops = effective_conv_h*effective_conv_w*C_o
                                                *
                                (kernel_size*kernel_size*C_i*2.0);
  // printf("%lf\t%lf\t",(conv_ops+pool_ops)/(1.0*sum_pytorch/(10)), (1.0*sum_pytorch/(10)));
  print_flops(conv_ops+pool_ops, sum_pytorch, RUNS);
  print_cycles(sum_pytorch, RUNS);
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

  #if (PARALLEL==1)
  uint32_t num_threads = atoi(std::getenv("OMP_NUM_THREADS"));
  #endif

  #if(BUFFER==0)
  float * out_intermediate_buffer;
  if(C_i > 16)
  {
    #if (PARALLEL==1)
    int ret = posix_memalign((void**)&out_intermediate_buffer, 4096, out_intermediate_dimensions[2]*out_intermediate_dimensions[3]*C_ob*sizeof(float)*(num_threads));
    #else
    int ret = posix_memalign((void**)&out_intermediate_buffer, 4096, out_intermediate_dimensions[2]*out_intermediate_dimensions[3]*C_ob*sizeof(float));
    #endif
  }
  else{
    printf("Output channels must be >= 32");
    return 0;
    printf("6x16 intermediate size\n");
    #if PARALLEL
    int ret = posix_memalign((void**)&out_intermediate_buffer, 4096, W_ob*C_ob*sizeof(float)*(num_threads));
    #else
    int ret = posix_memalign((void**)&out_intermediate_buffer, 4096, W_ob*C_ob*sizeof(float));
    #endif
  }
  #endif

  uint64_t t0, t1, sum=ULLONG_MAX, sum_pool=ULLONG_MAX;
  //unfused
  {
    memset(out_intermediate_dc, 0, out_intermediate.numel()*sizeof(float));
    memset(out_dc, 0, out.numel()*sizeof(float));


    //3x3 unfused
    copy_torch2dc(a, 'i', in_dimensions, input_dc);
    copy_torch2dc(weights,'f',filter_dimensions,filter_dc);
    copy_torch2dc(weights_dw,'d',filter_dw_dimensions,filter_dw_dc);
    // std::cout<<out_dimensions<<std::endl;
    #if(L)
      volatile float check_sum = rand()/(1.0*RAND_MAX);
      for(uint32_t i = 0; i < bomb_size; i++){
        check_sum += cache_bomb_array[i];
      }
    #endif

    for (int run = 0; run < RUNS; run++){
      // Copy Inputs to their flat buffers
      t0 = rdtsc();
      direct_convolution<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc, filter_dc, out_intermediate_dc);
      // direct_convolution_pooling_aware<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc, filter_dc, out_intermediate_dc);
      t1 = rdtsc();
      MIN(sum,(t1 - t0));
      t0 = rdtsc();
      dw_conv(C_o, out_intermediate_dimensions[2], out_intermediate_dimensions[3] ,out_intermediate_dc, filter_dw_dc, out_dc);
      t1 = rdtsc();
      MIN(sum_pool,(t1 - t0));

       #if(L)
      { volatile float check_sum = rand()/(1.0*RAND_MAX);
         for(uint32_t i = 0; i < bomb_size; i++){
           check_sum += cache_bomb_array[i];
         }
       }
       #endif
    }
    assert(check_eqivalence(out_intermediate,'o', out_intermediate_dimensions, out_intermediate_dc, 1e-3)==1);
    assert(check_eqivalence(out,'o', out_dimensions, out_dc, 1e-3)==1);

    print_flops(conv_ops, sum, RUNS);
    print_cycles(sum, RUNS);

    print_flops(conv_ops+pool_ops, sum+sum_pool, RUNS);
    print_cycles(sum+sum_pool, RUNS);
    printf("\t");

    memset(out_intermediate_dc, 0, out_intermediate.numel()*sizeof(float));
    memset(out_dc, 0, out.numel()*sizeof(float));
    sum_pool = ULLONG_MAX;
    for (int run = 0; run < RUNS; run++){
      // Copy Inputs to their flat buffers

      t0 = rdtsc();
      l_fused_dw_conv_direct_convolution<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc, filter_dc, out_intermediate_dc, filter_dw_dc, out_dc);
      // direct_convolution_pooling_aware<stride,kernel_size, kernel_size >(C_i, C_o, N, M, input_dc, filter_dc, out_intermediate_dc);
      t1 = rdtsc();
      MIN(sum,(t1 - t0));

       #if(L)
      { volatile float check_sum = rand()/(1.0*RAND_MAX);
         for(uint32_t i = 0; i < bomb_size; i++){
           check_sum += cache_bomb_array[i];
         }
       }
       #endif
    }
    assert(check_eqivalence(out,'o', out_dimensions, out_dc, 1e-3)==1);
    print_flops(effective_conv_ops+pool_ops, sum, RUNS);
    print_cycles(sum, RUNS);
    printf("%d %d %d\n",out_intermediate_dimensions[2], out_intermediate_dimensions[3] , C_o );



  }


  free(input_dc);
  free(filter_dc);
  free(out_dc);
  free(filter_dw_dc);
  free(out_intermediate_dc);
  free(out_intermediate_buffer);

}
