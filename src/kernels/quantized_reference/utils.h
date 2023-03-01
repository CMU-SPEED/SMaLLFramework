// SMaLL, Software for Machine Learning Libraries
// Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM23-0126

// A set of functions to enable testing
// We have timing, initialization, allocation, logging and equivalence checking

#include <stdint.h>
//****************************************************************************
//Allocation
//****************************************************************************
// #define MAX_BUFF_SIZE 50000000
// #define MAX_BUFF_SIZE 10000000
// #define MAX_BUFF_SIZE 200000
#define MAX_BUFF_SIZE 208200
dtype memory_buffer[MAX_BUFF_SIZE];
void *current_free_ptr = memory_buffer;
uint32_t buf_offset = 0;
template <typename dtype>
void *alloc(uint32_t numel)
{
  size_t bytes_to_alloc = numel * sizeof(dtype);
  size_t used_bytes = buf_offset + bytes_to_alloc;
  void *next_free_ptr = memory_buffer + used_bytes;
    if (MAX_BUFF_SIZE < used_bytes) {
        // Serial.println("out of space\n");
        // Serial.println(used_bytes);
        // Serial.println();
        return NULL;
    }
    else
    {
      void *ret_ptr = memory_buffer + buf_offset;
      buf_offset = used_bytes;
      return ret_ptr;
    }
}


qdtype *quantized_init(qdtype *q_a, uint32_t numel)
{
    float max = 1.0;
    float min = -1.0;
    q_a->b = (sizeof(dtype) * 8);
    uint64_t max_q = (1 << q_a->b) - 1;
    int min_q = 0;
    double scale = 1.0;
    scale = (max - min) / ((max_q - min_q) * 1.0) + 1e-17;
    int shift;
    const double q = frexp(scale, &shift);
    auto q_fixed = static_cast<int64_t>(std::round(q * (1LL << 31)));
    if (q_fixed == (1LL << 31))
    {
      q_fixed /= 2;
      ++shift;
    }
    if (shift < -31)
    {
      shift = 0;
      q_fixed = 0;
    }
    if (shift > 30)
    {
      shift = 30;
      q_fixed = (1LL << 31) - 1;
    }
    int32_t quantized_multiplier = static_cast<int32_t>(q_fixed);

    int zero = rint((double)(max * min_q - min * max_q) / ((double)(max - min)));
    q_a->scale = scale;
    q_a->zero = zero;
    q_a->lshift = shift > 0 ? shift : 0;
    q_a->rshift = shift > 0 ? 0 : -shift;
    q_a->multiplier = quantized_multiplier;
    q_a->min_val = 255;
    q_a->max_val = 0;
    q_a->tensor = NULL;
    return q_a;
}

uint32_t free_all()
{
  auto freed_space = buf_offset;
  current_free_ptr = memory_buffer;
  buf_offset = 0;

  return freed_space;

}

//****************************************************************************
// Initialization Options
//****************************************************************************

void init(float * ptr, uint32_t numel)
{
  float * cur_ptr = ptr;
  for(uint32_t i = 0 ; i < numel ; i++)
  {
    *(cur_ptr++) = 2.0*((dtype) rand()/ RAND_MAX) - 1;
  }
}

// template <typename>
void init(uint8_t *ptr, uint32_t numel)
{
  uint8_t *cur_ptr = ptr;
  for (uint32_t i = 0; i < numel; i++)
  {
        *(cur_ptr++) = rand()  % 10;
  }
}

void init_ones(dtype *ptr, uint32_t numel)
{
  dtype *cur_ptr = ptr;
  for (uint32_t i = 0; i < numel; i++)
  {
    *(cur_ptr++) = 1.0;
  }
}

template<uint32_t _C_ob>
void init_arange(dtype *ptr, uint32_t H, uint32_t W, uint32_t C)
{
  dtype *cur_ptr = ptr;
  for (uint32_t i = 0; i < C; i+=_C_ob)
  {
    for(uint32_t j = 0 ; j < H; j++)
    {
      for(uint32_t k = 0; k < W; k++)
      {
        for(uint32_t ii = 0; ii < _C_ob; ii++)
        {
           *(cur_ptr++) =  ii + i + k*(C) + j*(W*C);
        }
      }
    }
  }
}
void init_norm(dtype *ptr, uint32_t numel, uint32_t C_o)
{
  dtype *cur_ptr = ptr;
  dtype norm = (1.0*C_o)/(1.0*numel);
  for (uint32_t i = 0; i < numel; i++)
  {
    *(cur_ptr++) = norm;
  }
}
