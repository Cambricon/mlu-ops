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

#ifndef KERNELS_GET_INDICE_PAIRS_GET_INDICE_PAIRS_UTILS_H_
#define KERNELS_GET_INDICE_PAIRS_GET_INDICE_PAIRS_UTILS_H_

#include <algorithm>

#include "kernels/kernel.h"
#include "kernels/get_indice_pairs/normal_get_indice_pairs.h"

#if __BANG_ARCH__ >= 370
__mlu_func__ void assignTask(const int32_t num_total_task,
                             const int32_t &taskid, const int32_t &taskdim,
                             int32_t &task_offset, int32_t &num_cur_task) {
  int32_t num_per_task = num_total_task / taskdim;
  int32_t rem_idx = num_total_task % taskdim;
  if (taskid < rem_idx) {
    task_offset = taskid * (num_per_task + 1);
    num_cur_task = num_per_task + 1;
  } else {
    task_offset = taskid * num_per_task + rem_idx;
    num_cur_task = num_per_task;
  }
}

/*
func: init filter index kd * kh * kw
*/
__mlu_func__ void genFilterIndex(float *filter_kd_index, float *filter_kh_index,
                                 float *filter_kw_index, int32_t k_d,
                                 int32_t k_h, int32_t k_w) {
  // kw kh, kd loop init
  int32_t kdhw = k_d * k_h * k_w, khw = k_w * k_h, index_kd_count = 0,
          index_kh_count = 0;
  float index_kd = 0, index_kh = 0, index_kw = 0;
  for (int i = 0; i < kdhw; ++i) {
    filter_kw_index[i] = index_kw;
    index_kw++;
    if (index_kw >= k_w) index_kw = 0.0;
  }
  for (int i = 0; i < kdhw; ++i) {
    filter_kh_index[i] = index_kh;
    index_kh_count++;
    if (index_kh_count % k_w == 0) index_kh++;
    if (index_kh_count % khw == 0) index_kh = 0.0;
  }
  for (int i = 0; i < kdhw; ++i) {
    filter_kd_index[i] = index_kd;
    index_kd_count++;
    if (index_kd_count % khw == 0) index_kd++;
  }
}

/*
func: generate stage index from start_index
*/
__mlu_func__ void stepIndex(int32_t *dst_nram, int32_t start_index,
                            int32_t length) {
#if (__BANG_ARCH__ == 372 || __BANG_ARCH__ == 322 || __BANG_ARCH__ == 592)
  int32_t align_num = 128;
  int32_t repeat = (int32_t)(logf(length / align_num) / logf(2));
  int32_t remain = length / align_num - powf(2, repeat);
  int32_t global_remain = length % align_num;
  int32_t count = 1;
  for (int32_t i = 0; i < align_num; ++i) {
    dst_nram[i] = i + start_index;
    if (i == length - 1) {
      return;
    }
  }
  for (int i = 0; i < repeat; ++i) {
    __bang_add_scalar((int32_t *)dst_nram + count * align_num,
                      (int32_t *)dst_nram, count * align_num,
                      count * align_num);
    count *= 2;
  }
  if (remain > 0) {
    __bang_add_scalar((int32_t *)dst_nram + count * align_num,
                      (int32_t *)dst_nram, count * align_num,
                      remain * align_num);
  }
  if (global_remain > 0) {
    __bang_add_scalar(
        (int32_t *)dst_nram + count * align_num + remain * align_num,
        (int32_t *)dst_nram, count * align_num + remain * align_num,
        global_remain);
  }
  __asm__ volatile("sync;\n\t");
#endif
}

/*
input: nram_input  l
output: nram_output k,l
func: expand k nums input l
*/
__mlu_func__ void expandInput(int32_t *nram_input, int32_t deal_num,
                              int32_t k) {
  int offset = deal_num;
  for (int i = 0; i < k; ++i) {
    __bang_add_scalar((int32_t *)nram_input + offset, (int32_t *)nram_input,
                      (int32_t)0, deal_num);
    offset += deal_num;
  }
}

/*
input: input_pos, fliter_pos, stride, padding, dilation
output: do ho wo
func: generate do ho wo
*/
__mlu_func__ void computeOutputIndex(float *nram_output, float *nram_input,
                                     float *temp, float *filter_kd_index,
                                     float *filter_kh_index,
                                     float *filter_kw_index, int32_t deal_num,
                                     int32_t kdhw, Stride stride,
                                     Dilation dilation, Padding padding) {
  //  formula:  output_id = (input_id + padding - k_id * dilation) / stride
  float stride_sd = 1.0 / (float)stride.s_d;
  float stride_sh = 1.0 / (float)stride.s_h;
  float stride_sw = 1.0 / (float)stride.s_w;
  int32_t offset = deal_num;
  for (int i = 0; i < 3; ++i) {
    int32_t out_offset = offset - deal_num;
    float stride_s = i == 0 ? stride_sd : i == 1 ? stride_sh : stride_sw;
    int32_t padding_p =
        i == 0 ? padding.p_d : i == 1 ? (padding.p_h) : (padding.p_w);
    int32_t dilation_d =
        i == 0 ? dilation.d_d : i == 1 ? dilation.d_h : dilation.d_w;
    float *temp_filter_index =
        i == 0 ? filter_kd_index : (i == 1 ? filter_kh_index : filter_kw_index);
    __bang_add_scalar(nram_output + out_offset, nram_input + offset,
                      (float)(padding_p), deal_num);
    __bang_mul_scalar(temp, temp_filter_index, (float)(dilation_d), kdhw);
    __bang_cycle_sub(nram_output + out_offset, nram_output + out_offset, temp,
                     deal_num, kdhw);
    __bang_mul_scalar(nram_output + out_offset, nram_output + out_offset,
                      stride_s, deal_num);
    offset += deal_num;
  }
}

/*
input: nram_input  float  3 k l do ho wo
output: nram_output float  k l
func: generate mask represent output
*/
__mlu_func__ void computeMask(float *nram_output, float *nram_input,
                              float *temp, int32_t deal_num,
                              OutputSpace output_space) {
  int32_t o_d = output_space.o_d, o_h = output_space.o_h,
          o_w = output_space.o_w;
  int32_t offset = 0;
  int32_t offset_temp2 = deal_num;
  int32_t offset_temp3 = 2 * deal_num;
  __bang_write_value((float *)nram_output, deal_num, (float)1.0);
  for (int i = 0; i < 3; ++i) {
    int32_t output_dim = i == 0 ? o_d : i == 1 ? o_h : o_w;
    __bang_float2int32_tz((int32_t *)temp, (float *)nram_input + offset,
                          deal_num, 0);
    __bang_int322float_rn((float *)temp, (int32_t *)temp, deal_num, 0);
    __bang_sub((float *)temp + offset_temp2, (float *)temp,
              (float *)nram_input + offset, deal_num);
    __bang_le_scalar((float *)temp + offset_temp3, (float *)temp + offset_temp2,
              (float)0.000001, deal_num);  //  < 1e-6
    __bang_ge_scalar((float *)temp + offset_temp2, (float *)temp + offset_temp2,
              (float)-0.000001, deal_num);  //  > -1e-6
    __bang_and((float *)temp + offset_temp2, (float *)temp + offset_temp2,
               (float *)temp + offset_temp3, deal_num);
    __bang_ge_scalar((float *)temp + offset_temp3, (float *)temp, (float)0.0,
                     deal_num);
    __bang_and((float *)temp + offset_temp2, (float *)temp + offset_temp2,
               (float *)temp + offset_temp3, deal_num);
    __bang_le_scalar((float *)temp + offset_temp3, (float *)temp,
                     (float)(output_dim - 1), deal_num);
    __bang_and((float *)temp, (float *)temp + offset_temp3,
               (float *)temp + offset_temp2, deal_num);
    __bang_and((float *)nram_output, (float *)nram_output, (float *)temp,
               deal_num);
    offset += deal_num;
  }
}

/*
input: nram_input  int32_t l,4   n do ho wo
output: nram_output int32_t l indice_out_expand
func: generate  all_index from  n do ho wo index
*/
__mlu_func__ void genIndiceOutput(int32_t *nram_output, float *batch,
                                  float *nram_input, int32_t *temp,
                                  int32_t deal_num, OutputSpace output_space) {
  int32_t o_d = output_space.o_d, o_h = output_space.o_h,
          o_w = output_space.o_w;
  int32_t o_hw = o_h * o_w, o_dhw = o_d * o_h * o_w;
  __bang_float2int32_tz((int32_t *)temp + deal_num, (float *)batch,
                        deal_num, 0);  // n
  __bang_mul_scalar((int32_t *)temp, (int32_t *)temp + deal_num, (int32_t)o_dhw,
                    deal_num);  // n * odhw
  __bang_float2int32_tz((int32_t *)temp + 2 * deal_num, (float *)nram_input,
                        deal_num, 0);  // do
  __bang_mul_scalar((int32_t *)temp + deal_num, (int32_t *)temp + 2 * deal_num,
                    (int32_t)o_hw, deal_num);  // do * o_hw
  __bang_add((int32_t *)temp, (int32_t *)temp, (int32_t *)temp + deal_num,
             deal_num);
  __bang_float2int32_tz((int32_t *)temp + 2 * deal_num,
                        (float *)nram_input + deal_num, deal_num, 0);  // ho
  __bang_mul_scalar((int32_t *)temp + deal_num, (int32_t *)temp + 2 * deal_num,
                    (int32_t)o_w, deal_num);
  __bang_add((int32_t *)temp, (int32_t *)temp, (int32_t *)temp + deal_num,
             deal_num);
  __bang_float2int32_tz((int32_t *)temp + deal_num,
                        (float *)nram_input + 2 * deal_num, deal_num, 0);  // wo
  __bang_add((int32_t *)nram_output, (int32_t *)temp,
             (int32_t *)temp + deal_num, deal_num);
}

/*
input: nram_output  int32_t k,l indice_outout_expand
       mask_all     float   k,l mask_all
output nram_output  int32_t k,l  indice_output_expand
func: turn invalid index into int_max
*/
__mlu_func__ void genIndiceOutExpand(int32_t *nram_output, int32_t *mask_all,
                                     int32_t *nram_input, int32_t *temp,
                                     int32_t deal_num, int32_t output_size) {
  __bang_mul_scalar((int32_t *)temp, (int32_t *)mask_all, int(-1), deal_num);
  __bang_band((char *)nram_output, (char *)nram_input, (char *)temp,
              deal_num * sizeof(int32_t));
  // clost to intmax
  __bang_sub_scalar((int32_t *)temp, (int32_t *)mask_all, int(1), deal_num);
  __bang_mul_scalar((int32_t *)temp, (int32_t *)temp, int(-1 * output_size),
                    deal_num);
  __bang_bor((char *)nram_output, (char *)nram_output, (char *)temp,
             deal_num * sizeof(int32_t));
}

/*
input: nram_input  int32_t  L  indice_out_expand
output: nram_output int32_t  L,4  indice_out
func:  generate n,do,ho,wo index from input all_index
limits: imp on 300
*/
__mlu_func__ void genIndiceOutLast(int32_t *nram_output, int32_t *nram_input,
                                   int32_t *nram_aux, OutputSpace output_space,
                                   int32_t deal_num) {
#if __BANG_ARCH__ >= 590
  int32_t o_d = output_space.o_d, o_h = output_space.o_h,
          o_w = output_space.o_w;
  int32_t o_hw = o_h * o_w, o_dhw = o_d * o_h * o_w;
  __bang_div((int32_t *)nram_aux, (int32_t *)nram_input, (int)o_dhw,
             deal_num);  // n
  __bang_mul_scalar((int32_t *)nram_output, (int32_t *)nram_aux, (int)o_dhw,
                    deal_num);
  __bang_sub((int32_t *)nram_input, (int32_t *)nram_input, (int *)nram_output,
             deal_num);
  __bang_div((int32_t *)nram_aux + deal_num, (int32_t *)nram_input, (int)o_hw,
             deal_num);  // d
  __bang_mul_scalar((int32_t *)nram_output, (int32_t *)nram_aux + deal_num,
                    (int)o_hw, deal_num);
  __bang_sub((int32_t *)nram_input, (int32_t *)nram_input,
             (int32_t *)nram_output, deal_num);

  __bang_div((int32_t *)nram_aux + 2 * deal_num, (int32_t *)nram_input,
             (int)o_w, deal_num);  // h
  __bang_mul_scalar((int32_t *)nram_output, (int32_t *)nram_aux + 2 * deal_num,
                    (int)o_w, deal_num);
  __bang_sub((int32_t *)nram_aux + 3 * deal_num, (int32_t *)nram_input,
             (int32_t *)nram_output, deal_num);  //  w
  __bang_transpose((int32_t *)nram_output, (int32_t *)nram_aux, 4, deal_num);
#else
  int32_t o_d = output_space.o_d, o_h = output_space.o_h,
          o_w = output_space.o_w;
  int32_t o_hw = o_h * o_w, o_dhw = o_d * o_h * o_w;
  __bang_write_value((int32_t *)nram_aux + 4 * deal_num, deal_num, int(o_dhw));
  __cn_vector_div_s32(deal_num, (int32_t *)nram_aux, (int32_t *)nram_input,
                      (int32_t *)nram_aux + 4 * deal_num);
  __bang_mul_scalar((int32_t *)nram_output, (int32_t *)nram_aux, (int)o_dhw,
                    deal_num);
  __bang_sub((int32_t *)nram_input, (int32_t *)nram_input, (int *)nram_output,
             deal_num);
  __bang_write_value((int32_t *)nram_aux + 4 * deal_num, deal_num, int(o_hw));
  __cn_vector_div_s32(deal_num, (int32_t *)nram_aux + deal_num,
                      (int32_t *)nram_input,
                      (int32_t *)nram_aux + 4 * deal_num);
  __bang_mul_scalar((int32_t *)nram_output, (int32_t *)nram_aux + deal_num,
                    (int)o_hw, deal_num);
  __bang_sub((int32_t *)nram_input, (int32_t *)nram_input,
             (int32_t *)nram_output, deal_num);

  __bang_write_value((int32_t *)nram_aux + 4 * deal_num, deal_num, int(o_w));
  __cn_vector_div_s32(deal_num, (int32_t *)nram_aux + 2 * deal_num,
                      (int32_t *)nram_input,
                      (int32_t *)nram_aux + 4 * deal_num);
  __bang_mul_scalar((int32_t *)nram_output, (int32_t *)nram_aux + 2 * deal_num,
                    (int)o_w, deal_num);
  __bang_sub((int32_t *)nram_aux + 3 * deal_num, (int32_t *)nram_input,
             (int32_t *)nram_output, deal_num);  //  w
  __bang_transpose((int32_t *)nram_output, (int32_t *)nram_aux, 4, deal_num);
#endif
}

/*
input: nram_input  int32_t l,4   indice_in
output: nram_output int32_t l   indice_in_expand
func: generate  all_index from  n di hi wi index
*/
__mlu_func__ void genIndiceInExpand(int32_t *nram_output, int32_t *nram_input,
                                    int32_t *nram_aux, int32_t deal_num,
                                    InputSpace input_space) {
  __bang_transpose((int32_t *)nram_aux, (int32_t *)nram_input, deal_num, 4);
  int32_t i_d = input_space.i_d, i_h = input_space.i_h, i_w = input_space.i_w;
  int32_t i_hw = i_h * i_w, i_dhw = i_d * i_h * i_w;
  __bang_mul_scalar((int32_t *)nram_aux + 4 * deal_num,
                    (int32_t *)nram_aux + 2 * deal_num, int32_t(i_w), deal_num);
  __bang_add((int32_t *)nram_output, (int32_t *)nram_aux + 4 * deal_num,
             (int32_t *)nram_aux + 3 * deal_num, deal_num);
  __bang_mul_scalar((int32_t *)nram_aux + 4 * deal_num,
                    (int32_t *)nram_aux + deal_num, int32_t(i_hw), deal_num);
  __bang_add((int32_t *)nram_output, (int32_t *)nram_output,
             (int32_t *)nram_aux + 4 * deal_num, deal_num);
  __bang_mul_scalar((int32_t *)nram_aux + 4 * deal_num, (int32_t *)nram_aux,
                    int32_t(i_dhw), deal_num);
  __bang_add((int32_t *)nram_output, (int32_t *)nram_output,
             (int32_t *)nram_aux + 4 * deal_num, deal_num);
}
#endif
#endif  // KERNELS_GET_INDICE_PAIRS_GET_INDICE_PAIRS_UTILS_H_
