//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2026 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//****************************************************************************

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <small/utils/Timer.hpp>

namespace small
{
namespace test
{

struct TimingStats
{
    std::size_t count = 0;
    std::size_t trimmed_count = 0;
    double min = 0.0;
    double max = 0.0;
    double avg = 0.0;
    double std_dev = 0.0;
    double trimmed_avg = 0.0;
    double trimmed_std_dev = 0.0;
    double total = 0.0;
};

inline TimingStats summarize_times(std::vector<double> times,
                                   bool trim_extremes = true)
{
    if (times.empty())
    {
        throw std::invalid_argument("summarize_times requires at least one sample");
    }

    TimingStats stats;
    stats.count = times.size();
    stats.min = *std::min_element(times.begin(), times.end());
    stats.max = *std::max_element(times.begin(), times.end());
    stats.total = std::accumulate(times.begin(), times.end(), 0.0);
    stats.avg = stats.total / static_cast<double>(stats.count);

    double total_variance = 0.0;
    for (double const time : times)
    {
        double const delta = time - stats.avg;
        total_variance += delta * delta;
    }
    stats.std_dev = std::sqrt(total_variance / static_cast<double>(stats.count));

    std::sort(times.begin(), times.end());
    std::size_t begin = 0;
    std::size_t end = times.size();
    if (trim_extremes && times.size() > 2)
    {
        begin = 1;
        end = times.size() - 1;
    }

    stats.trimmed_count = end - begin;
    double const trimmed_total =
        std::accumulate(times.begin() + begin, times.begin() + end, 0.0);
    stats.trimmed_avg = trimmed_total / static_cast<double>(stats.trimmed_count);

    double trimmed_variance = 0.0;
    for (std::size_t ix = begin; ix < end; ++ix)
    {
        double const delta = times[ix] - stats.trimmed_avg;
        trimmed_variance += delta * delta;
    }
    stats.trimmed_std_dev =
        std::sqrt(trimmed_variance / static_cast<double>(stats.trimmed_count));

    return stats;
}

template <typename Callable>
std::vector<double> timed_trials(std::size_t trials, Callable &&run_once)
{
    if (trials == 0)
    {
        throw std::invalid_argument("timed_trials requires at least one trial");
    }

    std::vector<double> times;
    times.reserve(trials);

    small::Timer timer;
    timer.start();
    for (std::size_t trial = 0; trial < trials; ++trial)
    {
      
        run_once();
   
    }
    timer.stop();
    times.push_back(timer.elapsed()/trials);

    return times;
}

template <typename Callable>
TimingStats time_trials(std::size_t trials,
                        Callable &&run_once,
                        bool trim_extremes = true)
{
    return summarize_times(
        timed_trials(trials, std::forward<Callable>(run_once)),
        trim_extremes);
}

} // namespace test
} // namespace small
