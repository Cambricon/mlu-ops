/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/

#ifndef KERNELS_GET_INDICE_PAIRS_GET_INDICE_PAIRS_STRUCTS_H_
#define KERNELS_GET_INDICE_PAIRS_GET_INDICE_PAIRS_STRUCTS_H_

#include "mlu_op.h"

#define MAX_PAD_DIM 6
#define MAX_STRIDE_DIM 6
#define MAX_DILATION_DIM 6
#define MAX_INPUT_DIM 3
#define MAX_FILTER_DIM 3
#define MAX_OUTPUT_DIM 3

struct FilterSpace {
  int k_d;
  int k_h;
  int k_w;
  FilterSpace(const int &k_d_, const int &k_h_, const int &k_w_)
      : k_d(k_d_), k_h(k_h_), k_w(k_w_) {}
};
struct InputSpace {
  int i_d;
  int i_h;
  int i_w;
  InputSpace(const int &i_d_, const int &i_h_, const int &i_w_)
      : i_d(i_d_), i_h(i_h_), i_w(i_w_) {}
};

struct OutputSpace {
  int o_d;
  int o_h;
  int o_w;
  OutputSpace(const int &o_d_, const int &o_h_, const int &o_w_)
      : o_d(o_d_), o_h(o_h_), o_w(o_w_) {}
};

struct Stride {
  int s_d;
  int s_h;
  int s_w;
  Stride(const int &s_d_, const int &s_h_, const int &s_w_)
      : s_d(s_d_), s_h(s_h_), s_w(s_w_) {}
};

struct Dilation {
  int d_d;
  int d_h;
  int d_w;
  Dilation(const int &d_d_, const int &d_h_, const int &d_w_)
      : d_d(d_d_), d_h(d_h_), d_w(d_w_) {}
};

struct Padding {
  int p_d;
  int p_h;
  int p_w;
  Padding(const int &p_d_, const int &p_h_, const int &p_w_)
      : p_d(p_d_), p_h(p_h_), p_w(p_w_) {}
};

struct mluOpSparseConvolutionStruct {
  int dimNb;
  int batch;
  int pad[MAX_PAD_DIM];
  int stride[MAX_STRIDE_DIM];
  int dilation[MAX_DILATION_DIM];
  int input_space[MAX_INPUT_DIM];
  int filter_space[MAX_FILTER_DIM];
  int output_space[MAX_OUTPUT_DIM];
  int sub_m = 0;
  int transpose = 0;
  int inverse = 0;
  int num_act_out = 0;
};

#endif  //  KERNELS_GET_INDICE_PAIRS_GET_INDICE_PAIRS_STRUCTS_H_
