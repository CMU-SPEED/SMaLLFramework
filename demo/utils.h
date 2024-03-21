//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM23-0126
//****************************************************************************

// A set of functions to enable testing
// We have timing, initialization, allocation, logging and equivalence checking

#pragma once

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

//****************************************************************************
//logging
//****************************************************************************

template <class BufferT>
void print_build_info_check()
{
    //    string archs[] = {"reference", "zen2", "arm"};
    printf("W_ob =  %d \n C_ob = %d \n SIMD = %d \n",
           BufferT::W_ob, BufferT::C_ob, BufferT::SIMD);
}

template <class T>
void print_stats(std::vector<T> v, const char *benchmark)
{
    if (v.size() != 0)
    {
        T sum = std::reduce(v.begin(), v.end());
        double mean = sum / (double)v.size();
        T min_elem = *min_element(v.begin(), v.end());
        T max_elem = *max_element(v.begin(), v.end());
        std::cout << benchmark << ": #runs = " << v.size()
                  << ", Min = " << min_elem
                  << ", Max = " << max_elem
                  << ", Avg = " << mean
                  << std::endl;
    }
}

#define print_flops(ops, time)                      \
    {                                               \
        printf("%.4lf FLOPS\t", (ops) / (1.0 * time));    \
    }

#define print_cycles(time)                  \
    {                                       \
        printf("%.2lf ", 1.0 * (time));    \
    }

//****************************************************************************
// Initialization Options
//****************************************************************************

// BUFFER INITIALIZATION MOVED TO include/small/buffers.hpp (for now)

//****************************************************************************
template<uint32_t num_ops, uint32_t trials>
void write_results(uint64_t *fused_timing)
{
    fprintf(stderr, "Unfused ");
    for (uint32_t j = 0; j < num_ops; j++)
    {
        fprintf(stderr, "Fused %d\t", j);
    }
    fprintf(stderr, "\n");
    for (uint32_t i = 0; i < trials; i++)
    {
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
