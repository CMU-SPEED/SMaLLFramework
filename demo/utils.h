// A set of functions to enable testing
#include <string>
// We have timing, initialization, allocation, logging and equivalence checking

//Timing
#include <sys/time.h>
// Change this to log min of all runs etc
#define TIME_ZERO 0

//Change this to log min of all runs etc


//TODO: Change when moving to different architectures

// static __inline__ unsigned long long rdtsc(void)
// {
//   unsigned hi, lo;
//   __asm__ __volatile__("rdtsc"
//                        : "=a"(lo), "=d"(hi));
//   return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
// }

#define REDUCE(a, b) \
  {                  \
    a += b;          \
  }

#define MIN(a, b)            \
    {                        \
        a = (b < a) ? b : a; \
    }

//logging
#define print_flops(ops, time)              \
  {                                                 \
    printf("%.4lf\t", (ops) / (1.0 * time); \
  }
#define print_cycles(time)          \
  {                                         \
    printf("%.2lf\t", 1.0 * (time)); \
  }
//Allocation

float * alloc (uint32_t numel)
{
  float *ptr_dc;

  int ret = posix_memalign((void **)&ptr_dc, 4096, numel * sizeof(float));

  if (ret)
  {
    return NULL;
  }
  return ptr_dc;
}


// Initialization Options
void init(float * ptr, uint32_t numel)
{
  float * cur_ptr = ptr;
  for(uint32_t i = 0 ; i < numel ; i++)
  {
    *(cur_ptr++) = 2.0*((float) rand()/ RAND_MAX) - 1;
  }
}

void init_ones(float *ptr, uint32_t numel)
{
  float *cur_ptr = ptr;
  for (uint32_t i = 0; i < numel; i++)
  {
    *(cur_ptr++) = 1.0;
    // printf("%.2f \n", *(cur_ptr - 1));
  }
}

template<uint32_t _C_ob>
void init_arange(float *ptr, uint32_t H, uint32_t W, uint32_t C)
{
  float *cur_ptr = ptr;
  for (uint32_t i = 0; i < C; i+=_C_ob)
  {
    for(uint32_t j = 0 ; j < H; j++)
    {
      for(uint32_t k = 0; k < W; k++)
      {
        for(uint32_t ii = 0; ii < _C_ob; ii++)
        {
           *(cur_ptr++) =  ii + i + k*(C) + j*(W*C);
          //  printf("%.2f \n", *(cur_ptr - 1));
        }
      }
    }
  }
}
void init_norm(float *ptr, uint32_t numel, uint32_t C_o)
{
  float *cur_ptr = ptr;
  float norm = (1.0*C_o)/(1.0*numel);
  for (uint32_t i = 0; i < numel; i++)
  {
    *(cur_ptr++) = norm;
  }
}
bool equals(uint32_t numel, float *unfused, float *fused, float tolerance = 1e-8)
{
  bool check = 1;
  float *unfused_ptr = unfused;
  float *fused_ptr = fused;

  for (uint32_t i = 0; i < numel; i++)
  {
    float diff = *(fused_ptr) - *(unfused_ptr);
    // printf("%d %.4f %.4f %.4f\n", i, *(fused_ptr), *(unfused_ptr), diff);

    if(fabs(diff) > tolerance)
    {
      printf("%d %.4f %.4f %.4f\n", i, *(fused_ptr), *(unfused_ptr), diff);
      check = 0;
    }
    unfused_ptr++;
    fused_ptr++;
  }
  return check;
}


template<uint32_t num_ops, uint32_t trials>
void write_results(uint64_t * fused_timing)
{
  // std::string path = "Results/logfile";
  // std::string path_to_log_file = path + file;
  // FILE *outfile = fopen(path.c_str(), "w");

  fprintf(stderr, "Unfused ");
  for (uint32_t j = 0; j < num_ops; j++)
  {
    fprintf(stderr, "Fused %d\t", j);
  }
  fprintf(stderr, "\n");
  for (uint32_t i = 0; i < trials; i++)
  {
    // fprintf(stderr, "%lu\t", timing[i]);
    for (uint32_t j = 0; j < num_ops + 1 ; j++)
    {
      fprintf(stderr, "%lu\t", fused_timing[j*trials + i]);
    }
    fprintf(stderr, "\n");
  }
}

timespec time1, time2;
long diff = 0;
long time_difference(timespec start, timespec end)
{
  timespec temp;
  if ((end.tv_nsec - start.tv_nsec) < 0)
  {
    temp.tv_sec = end.tv_sec - start.tv_sec - 1;
    temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
  }
  else
  {
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  }
  return (temp.tv_sec * 1000000000 + temp.tv_nsec);
}
