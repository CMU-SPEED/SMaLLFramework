// A set of functions to enable testing
// We have timing, initialization, allocation, logging and equivalence checking

#include <stdint.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include <iterator>
#include <fstream>
#include <algorithm> // std::min_element
#include <iterator>
#include <array>
#include <iostream>
// #include<functional>
#include <numeric>

//****************************************************************************
//Timing
//****************************************************************************

// Change this to log min of all runs etc
#define TIME_ZERO 0

//TODO: Change rdtsc(void) when moving to different architectures
#if uarch == ZEN2
static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc"
                         : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}
#elif uarch==ARM || uarch==REF
#include <time.h>
static __inline__ unsigned long long rdtsc(void)
{
    struct timespec gettime_now;
    clock_gettime(CLOCK_REALTIME, &gettime_now);
    return  gettime_now.tv_nsec ;
}
#endif

/// @todo unnecessary?
#define ACCUM_time(a, b) \
    {                    \
        a += b;          \
    }

/// @todo unnecessary? Fix duplication with ACCUM_time
#define REDUCE(a, b) \
  {                  \
    a += b;          \
  }

/// @todo change to use std::min<T> directly
#define MIN(a, b)            \
    {                        \
        a = (b < a) ? b : a; \
    }

#define AVG(accum, trials, avg)       \
    {                                 \
        avg = (1.0 * accum) / trials; \
    }

//****************************************************************************
//logging
//****************************************************************************

void print_build_info_check()
{
    //    string archs[] = {"reference", "zen2", "arm"};
    printf("W_ob =  %d \n C_ob = %d \n SIMD = %d \n", W_ob, C_ob, SIMD);
}

void print_stats(std::vector<unsigned long long> v, const char *benchmark)
{
    if (v.size() != 0)
    {
        unsigned long long sum = std::accumulate(v.begin(), v.end(), 0.0);
        unsigned long long mean = sum / v.size();
        unsigned long long min_elem = *min_element(v.begin(), v.end());
        unsigned long long max_elem = *max_element(v.begin(), v.end());
        printf("Average for %s: %llu, \t ", benchmark, mean);
        printf("Min for %s    : %llu, \t", benchmark, min_elem);
        printf("Max for %s    : %llu \n", benchmark, max_elem);
    }
}

#define print_flops(ops, time)                      \
    {                                               \
        printf("%.4lf\t", (ops) / (1.0 * time));    \
    }

#define print_cycles(time)                  \
    {                                       \
        printf("%.2lf\t", 1.0 * (time));    \
    }

//****************************************************************************
// Initialization Options
//****************************************************************************

//****************************************************************************
void init(small::Buffer<float> &ptr, uint32_t numel)
{
    if (numel > ptr.size())
    {
        throw std::invalid_argument("init ERROR: buffer too small.");
    }

    float *cur_ptr = ptr.data();
    for (size_t i = 0 ; i < numel ; i++)
    {
        *(cur_ptr++) = 2.0*((float) rand()/ RAND_MAX) - 1;
    }
}

//****************************************************************************
void init_ones(small::Buffer<float> &ptr, uint32_t numel)
{
    if (numel > ptr.size())
    {
        throw std::invalid_argument("init_ones ERROR: buffer too small.");
    }

    float *cur_ptr = ptr.data();
    for (size_t i = 0; i < numel; i++)
    {
        *(cur_ptr++) = 1.0;
        // printf("%.2f \n", *(cur_ptr - 1));
    }
}

//****************************************************************************
template<uint32_t _C_ob>
void init_arange(small::Buffer<float> &ptr, uint32_t H, uint32_t W, uint32_t C)
{
    if (C * H * W > ptr.size())
    {
        throw std::invalid_argument(
            "init_arange ERROR: buffer too small.");
    }

    float *cur_ptr = ptr.data();
    for (size_t i = 0; i < C; i+=_C_ob)
    {
        for (size_t j = 0 ; j < H; j++)
        {
            for (size_t k = 0; k < W; k++)
            {
                for (size_t ii = 0; ii < _C_ob; ii++)
                {
                    *(cur_ptr++) =  ii + i + k*(C) + j*(W*C);
                    //  printf("%.2f \n", *(cur_ptr - 1));
                }
            }
        }
    }
}

//****************************************************************************
void init_norm(small::Buffer<float> &ptr, uint32_t numel, uint32_t C_o)
{
    if (numel > ptr.size())
    {
        throw std::invalid_argument("init_norm ERROR: buffer too small.");
    }

    float *cur_ptr = ptr.data();
    float norm = (1.0*C_o)/(1.0*ptr.size());
    for (size_t i = 0; i < ptr.size(); i++)
    {
        *(cur_ptr++) = norm;
    }
}

//****************************************************************************
// tolerance based on absolute difference
bool equals(uint32_t numel,
            small::Buffer<float> const &unfused,
            small::Buffer<float> const &fused,
            float tolerance = 1e-8)
{
    float const *unfused_ptr = unfused.data();
    float const *fused_ptr   = fused.data();
    printf("begin correctness check\n");

    if ((unfused.size() > numel) ||
        (fused.size() > numel))
    {
        return false;
    }

    bool check = true;
    for (size_t i = 0; i < numel; i++)
    {
        float diff = *(fused_ptr) - *(unfused_ptr);
        // printf("equals      : %ld, fused/unfused %.4f/%.4f, diff %.4f\n",
        //        i, *(fused_ptr), *(unfused_ptr), diff);

        if (fabs(diff) > tolerance)
        {
            printf("equals ERROR: %ld, fused/unfused %.4f/%.4f, diff %.4f\n",
                   i, *(fused_ptr), *(unfused_ptr), diff);
            check = false;
        }
        unfused_ptr++;
        fused_ptr++;
    }
    printf("end of correctness check\n");
    return check;
}


//****************************************************************************
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

//****************************************************************************
//****************************************************************************
timespec time1, time2;
long diff = 0;

//****************************************************************************
long time_difference(timespec start, timespec end)
{
    timespec temp;
    if (end.tv_nsec < start.tv_nsec)
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
