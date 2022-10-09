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

#ifndef BANGC_OPS_KERNELS_POLY_NMS_POLY_NMS_CORE_SET_H
#define BANGC_OPS_KERNELS_POLY_NMS_POLY_NMS_CORE_SET_H

/**
 * A util function to get the working set of current core, every core will
 * handle boxes in range of
 * [*o_begin, *o_begin + *o_box_num)
 *
 * @param input_boxes_num[in] the total box number
 * @param o_box_num[out] the number of boxes current core should processed
 * @param o_beg [out] the beginning box id of current core
 */
__mlu_func__ static void getCoreWorkingSet(int input_boxes_num, int *o_box_num,
                                           int *o_beg) {
  int core_box_num = input_boxes_num / taskDim;
  int rem = input_boxes_num % taskDim;
  int box_i_beg = 0;
  if (taskId < rem) {
    core_box_num += taskId < rem;
    box_i_beg = core_box_num * taskId;
  } else {
    box_i_beg = core_box_num * taskId + rem;
  }
  *o_box_num = core_box_num;
  *o_beg = box_i_beg;
}

#endif  // BANGC_OPS_KERNELS_POLY_NMS_POLY_NMS_CORE_SET_H
