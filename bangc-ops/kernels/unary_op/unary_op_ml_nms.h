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
 ************************************************************************/
#ifndef UNARY_OP_BLOCK_H_
#define UNARY_OP_BLOCK_H_
#include "kernels/kernel.h"
#define NRAM_SIZE 2 * 1024
#define UNION_OP_KERNEL_DECLARE(Op, DType, Prefer)           \
  __mlu_global__ void MLUBlockKernel##Op##DType##Prefer(\
    mluOpDataType_t data_type, void* boxes_data_ptr, \
    float nms_thres, int input_boxes_num, uint8_t* output_boxes_index);\

#define UNION_OP_KERNEL_IMPLE(Op, DType, Prefer)                 \
  __mlu_global__ void MLUOpKernel##Op##DType##Prefer(     \
    mluOpDataType_t data_type, void* boxes_data_ptr, \
    float nms_thres, int input_boxes_num, uint8_t* output_boxes_index) {\
    int offset, seg; \
    getOffsetNum##Op##Prefer(input_boxes_num, &offset); \
    getSegNumMlNmsFast(input_boxes_num, &seg); \
    unionImple<DType, compute##Op##Prefer>( \
    (DType*)boxes_data_ptr, (DType)nms_thres, \
    offset, seg, input_boxes_num, output_boxes_index);}

template <typename T, void (*OpFunc)(T*, T, int, int, int, uint8_t*)>
__mlu_device__ void unionImple(T* boxes_data_ptr, T nms_thres, int offset,
  int seg, int input_boxes_num, uint8_t* output_boxes_index) {
  __nram__ char worke_space[MAX_NRAM_SIZE / 16];
  __memcpy((T*)worke_space, boxes_data_ptr + (offset * 4), seg * 4 * sizeof(T), GDRAM2NRAM);
  __memcpy((T*)worke_space + (seg * 4), boxes_data_ptr, 4 * sizeof(T), GDRAM2NRAM); 
  OpFunc((T*)worke_space, nms_thres, input_boxes_num, offset,
    seg, output_boxes_index);
}

#endif
