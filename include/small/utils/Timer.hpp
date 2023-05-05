/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DM20-0442
 */

#pragma once

namespace small
{

//****************************************************************************
#if defined(NANO33BLE)
//****************************************************************************
class Timer
{
public:
    using TimeType = decltype Timer::start();

    Timer() : m_timer() {}

    TimeType start() { m_timer.start(); return 0;}
    TimeType stop()  { m_timer.stop();  return m_timer.elapsed_time().count();}

    ///  %todo determine units
    double   elapsed() { return m_timer.elapsed_time().count(); }

private:
    mbed::Timer m_timer;
};

//****************************************************************************
//****************************************************************************
#else //(defined(UARCH_ARM) || defined(UARCH_ZEN2) || defined(UARCH_REF))
//****************************************************************************
#include <time.h>

class Timer
{
public:
    using TimeType = timespec;

    Timer() : start_time(), stop_time() {}

    TimeType start()
    {
        //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
        //clock_gettime(CLOCK_REALTIME, &start_time);
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        return start_time;  //.tv_nsec;
    }
    TimeType stop()
    {
        //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop_time);
        //clock_gettime(CLOCK_REALTIME, &stop_time);
        clock_gettime(CLOCK_MONOTONIC, &stop_time);
        return stop_time;  //.tv_nsec;
    }

    double elapsed() const
    {
        timespec temp;
        if (stop_time.tv_nsec < start_time.tv_nsec)
        {
            temp.tv_sec = stop_time.tv_sec - start_time.tv_sec - 1;
            temp.tv_nsec = 1000000000 + stop_time.tv_nsec - start_time.tv_nsec;
        }
        else
        {
            temp.tv_sec = stop_time.tv_sec - start_time.tv_sec;
            temp.tv_nsec = stop_time.tv_nsec - start_time.tv_nsec;
        }

        // microseconds
        // return (double)temp.tv_sec*1000000.0 + (double)temp.tv_nsec / 1000.0;

        // nanoseconds
        return (double)temp.tv_sec * 1000000000.0 + (double)temp.tv_nsec;
    }

private:
    TimeType start_time, stop_time;
};
#endif

//****************************************************************************
// Other Timer classes
//****************************************************************************
#if 0

//****************************************************************************
class Timer
{
public:
    using TimeType = unsigned long long;

    Timer() : start_time(), stop_time() {}

    TimeType start()
    {
        start_time = rdtsc();
        return start_time;
    }

    TimeType stop()
    {
        stop_time = rdtsc();
        return stop_time;
    }

    // units are clock cycles.
    double   elapsed() { return stop_time - start_time; }

private:
    static __inline__ unsigned long long rdtsc()
    {
        unsigned hi, lo;
        __asm__ __volatile__("rdtsc"
                             : "=a"(lo), "=d"(hi));
        return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
    }

    TimeType start_time, stop_time;
};

//****************************************************************************
#include <chrono>

template <class ClockT = std::chrono::steady_clock,
          class UnitsT = std::chrono::microseconds> //milliseconds>
class Timer
{
public:
    using TimeType = std::chrono::time_point<ClockT>;

    Timer() : start_time(), stop_time() {}

    TimeType start() { return (start_time = ClockT::now()); }
    TimeType stop()  { return (stop_time  = ClockT::now()); }

    // milliseconds
    double elapsed() const
    {
        //std::cout << "start, stop: " << start_time.count()
        //          << ", " << stop_time.count()
        //          << std::endl;
        return std::chrono::duration_cast<UnitsT>(
            stop_time - start_time).count();
    }

private:
    TimeType start_time, stop_time;
};
#endif

} // small namespace
