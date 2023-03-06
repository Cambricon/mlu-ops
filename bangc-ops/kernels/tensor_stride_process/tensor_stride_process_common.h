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
#ifndef KERNELS_TENSOR_STRIDE_PROCESS_TENSOR_STRIDE_PROCESS_COMMON_H_
#define KERNELS_TENSOR_STRIDE_PROCESS_TENSOR_STRIDE_PROCESS_COMMON_H_
#include "mlu.h"
#include "kernels/tensor_stride_process/tensor_stride_process_mlu.h"
// b is always pow(2, n), no need to worry about int2float precision loss
#define OFFSET_SHIFT(a, b) (((int)a) << ((int)log2f(b)))

// sync load & store
#define TENSOR_STRIDE_LOAD(T, dst_nram, src_gdram, src_offset, data_num,       \
                           dtype_size, tensor_shape)                           \
  if (tensor_shape.is_contiguous) {                                            \
    __memcpy(dst_nram,                                                         \
             (int8_t *)src_gdram + OFFSET_SHIFT(src_offset, sizeof(T)),        \
             data_num * dtype_size, GDRAM2NRAM);                               \
  } else {                                                                     \
    tensorStrideLoad<T>(dst_nram, src_gdram, src_offset, data_num, dtype_size, \
                        tensor_shape);                                         \
  }

#define TENSOR_STRIDE_STORE(T, dst_gdram, dst_offset, src_nram, data_num, \
                            dtype_size, tensor_shape)                     \
  if (tensor_shape.is_contiguous) {                                       \
    __memcpy((int8_t *)dst_gdram + OFFSET_SHIFT(dst_offset, sizeof(T)),   \
             src_nram, data_num * dtype_size, NRAM2GDRAM);                \
  } else {                                                                \
    tensorStrideStore<T>(dst_gdram, dst_offset, src_nram, data_num,       \
                         dtype_size, tensor_shape);                       \
  }

#define TENSOR_STRIDE_LOAD_ASYNC(T, dst_nram, src_gdram, src_offset, data_num, \
                                 dtype_size, tensor_shape)                     \
  if (tensor_shape.is_contiguous) {                                            \
    __memcpy_async(dst_nram,                                                   \
                   (int8_t *)src_gdram + OFFSET_SHIFT(src_offset, sizeof(T)),  \
                   data_num * dtype_size, GDRAM2NRAM);                         \
  } else {                                                                     \
    tensorStrideLoad<T>(dst_nram, src_gdram, src_offset, data_num, dtype_size, \
                        tensor_shape);                                         \
  }

#define TENSOR_STRIDE_STORE_ASYNC(T, dst_gdram, dst_offset, src_nram,         \
                                  data_num, dtype_size, tensor_shape)         \
  if (tensor_shape.is_contiguous) {                                           \
    __memcpy_async((int8_t *)dst_gdram + OFFSET_SHIFT(dst_offset, sizeof(T)), \
                   src_nram, data_num * dtype_size, NRAM2GDRAM);              \
  } else {                                                                    \
    tensorStrideStore<T>(dst_gdram, dst_offset, src_nram, data_num,           \
                         dtype_size, tensor_shape);                           \
  }

#define TENSOR_STRIDE_LOAD_SRAM_ASYNC(T, dst_sram, src_gdram, src_offset,     \
                                      data_num, dtype_size, tensor_shape)     \
  if (tensor_shape.is_contiguous) {                                           \
    __memcpy_async(dst_sram,                                                  \
                   (int8_t *)src_gdram + OFFSET_SHIFT(src_offset, sizeof(T)), \
                   data_num * dtype_size, GDRAM2SRAM);                        \
  } else {                                                                    \
    tensorStrideLoadSram<T>(dst_sram, src_gdram, src_offset, data_num,        \
                            dtype_size, tensor_shape);                        \
  }

#define TENSOR_STRIDE_STORE_SRAM_ASYNC(T, dst_gdram, dst_offset, src_sram,    \
                                       data_num, dtype_size, tensor_shape)    \
  if (tensor_shape.is_contiguous) {                                           \
    __memcpy_async((int8_t *)dst_gdram + OFFSET_SHIFT(dst_offset, sizeof(T)), \
                   src_sram, data_num * dtype_size, GDRAM2GDRAM);             \
  } else {                                                                    \
    tensorStrideStoreSram<T>(dst_gdram, dst_offset, src_sram, data_num,       \
                             dtype_size, tensor_shape);                       \
  }

// There are some public mlu functions for load or store tensor that with
// strides parameters.
/******************************************************************************
 * Function API: int getTrueOffset(int offset, TensorShape &tensor_shape);
 *
 * This function will return the true address offset of the tensor that with
 *special stride parameters.
 *
 * 'offset': The address offset of the tensor you want.
 * 'tensor_shape': The shape of source tensor, which contain dimensions
 *                 and strides parameters.
 *
 * Function API: void tensorStrideLoad(T *dst_nram,
 *                                     const T *src_gdram,
 *                                     int src_offset,
 *                                     int data_num,
 *                                     int dtype_size,
 *                                     TensorShape tensor_shape);
 *
 * This function can load data from gdram to nram instead of __memcpy(...,
 *GDRAM2NRAM) when the tensor has special stride parameters.
 *
 * 'dst_nram': The destination where you want to load data to, which
 *             must be NRAM.
 * 'src_gdram': The source where you want to load data from, which
 *              must be GDRAM.
 * 'src_offset': The offset of source's address.
 * 'data_num': The total number of the data that you want to load.
 * 'dtype_size': sizeof(T).
 * 'tensor_shape': The shape of source tensor, which contain dimensions
 *                 and strides parameters.
 *
 * ----------------------------------------------------------------------------
 * Function API: void tensorStrideLoadSram(T *dst_sram,
 *                                         const T *src_gdram,
 *                                         int src_offset,
 *                                         int data_num,
 *                                         int dtype_size,
 *                                         TensorShape tensor_shape);
 *
 * This function is as the same with tensorStrideLoad except that the dst is
 *SRAM. It can load data from gdram to sram instead of __memcpy(..., GDRAM2SRAM)
 * when the tensor has special stride parameters.
 *
 * ----------------------------------------------------------------------------
 * Function API: void tensorStrideLoad(T *dst_nram,
 *                                     const T *src_gdram,
 *                                     int src_offset,
 *                                     int data_num,
 *                                     int dtype_size,
 *                                     int dst_stride,
 *                                     int src_stride,
 *                                     int count,
 *                                     TensorShape tensor_shape);
 *
 * This function can load data from gdram to nram with strideIO instead of
 * __memcpy(..., GDRAM2NRAM, ...) when the tensor has special stride parameters.
 *
 * 'dst_nram': The destination where you want to load data to, which
 *             must be NRAM.
 * 'src_gdram': The source where you want to load data from, which
 *              must be GDRAM.
 * 'src_offset': The offset of source's address.
 * 'data_num': The number of the data that you want to load once.
 * 'dtype_size': sizeof(T).
 * 'dst_stride': The stride parameter about destination address after once
 *loading. 'src_stride': The stride parameter about source address after once
 *loading. 'count': The total times of stride IO. 'tensor_shape': The shape of
 *source tensor, which contain dimensions and strides parameters.
 *
 * ----------------------------------------------------------------------------
 * Function API: void tensorStrideLoadSram(T *dst_sram,
 *                                         const T *src_gdram,
 *                                         int src_offset,
 *                                         int data_num,
 *                                         int dtype_size,
 *                                         int dst_stride,
 *                                         int src_stride,
 *                                         int count,
 *                                         TensorShape tensor_shape);
 *
 * This function is as the same with tensorStrideLoad except that the dst is
 *SRAM. It can load data from gdram to sram with strideIO instead of
 * __memcpy(..., GDRAM2SRAM, ...) when the tensor has special stride parameters.
 *
 * Function API: void tensorStrideStore(T *dst_gdram,
 *                                      int dst_offset,
 *                                      const T *src_nram,
 *                                      int data_num,
 *                                      int dtype_size,
 *                                      TensorShape &tensor_shape) {
 *
 * This function can store data from nram to gdram instead of __memcpy(...,
 *NRAM2GDRAM) when the tensor has special stride parameters.
 *
 * 'dst_gdram': The destination where you want to store data to, which
 *              must be GDRAM.
 * 'dst_offset': The offset of destination's address.
 * 'src_nram': The source where you want to store data from, which
 *              must be NRAM.
 * 'data_num': The total number of the data that you want to load.
 * 'dtype_size': sizeof(T).
 * 'tensor_shape': The shape of destination tensor, which contain dimensions
 *                 and strides parameters.
 *
 * ----------------------------------------------------------------------------
 * Function API: void tensorStrideStoreSram(T *dst_gdram,
 *                                          int dst_offset,
 *                                          const T *src_sram,
 *                                          int data_num,
 *                                          int dtype_size,
 *                                          TensorShape &tensor_shape) {
 *
 * This function can store data from sram to gdram instead of __memcpy(...,
 *SRAM2GDRAM) when the tensor has special stride parameters.
 *
 * ----------------------------------------------------------------------------
 * Function API: void tensorStrideStore(T *dst_gdram,
 *                                      int dst_offset,
 *                                      const T *src_nram,
 *                                      int data_num,
 *                                      int dtype_size,
 *                                      int dst_stride,
 *                                      int src_stride,
 *                                      int count,
 *                                      TensorShape &tensor_shape) {
 *
 * This function can store data from nram to gdram with strideIO instead of
 * __memcpy(..., NRAM2GDRAM, ...) when the tensor has special stride parameters.
 *
 * 'dst_gdram': The destination where you want to store data to, which
 *              must be GDRAM.
 * 'dst_offset': The offset of destination's address.
 * 'src_nram': The source where you want to store data from, which
 *              must be NRAM.
 * 'data_num': The total number of the data that you want to load.
 * 'dtype_size': sizeof(T).
 * 'dst_stride': The stride parameter about destination address after once
 *storing. 'src_stride': The stride parameter about source address after once
 *storing. 'count': The total times of stride IO. 'tensor_shape': The shape of
 *destination tensor, which contain dimensions and strides parameters.
 *
 * ----------------------------------------------------------------------------
 * Function API: void tensorStrideStoreSram(T *dst_gdram,
 *                                          int dst_offset,
 *                                          const T *src_sram,
 *                                          int data_num,
 *                                          int dtype_size,
 *                                          int dst_stride,
 *                                          int src_stride,
 *                                          int count,
 *                                          TensorShape &tensor_shape) {
 *
 * This function can store data from sram to gdram with strideIO instead of
 * __memcpy(..., SRAM2GDRAM, ...) when the tensor has special stride parameters.
 ******************************************************************************/
__mlu_func__ bool isUse64BitDiv(int offset, int *tensor_dims, int total_num) {
  if (offset > UINT32_MAX || total_num > UINT32_MAX) {
    return true;
  }
  for (int i = 0; i < MLUOP_DIM_MAX; ++i) {
    // as long as 1 dim of tensor_shape exceeds INT32_MAX, 64-bit should be used
    if (tensor_dims[i] > UINT32_MAX) {
      return true;
    }
  }
  return false;
}

template <typename U>
__mlu_func__ int getTrueOffsetInternel(U offset, TensorShape &tensor_shape) {
  if (tensor_shape.is_contiguous) {
    return offset;
  }
  U offset_temp = offset;
  int true_offset = 0;
  U total_num = tensor_shape.total_num;
  U temp = 0;
  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    total_num = total_num / (U)tensor_shape.tensor_dims[i];
    temp = offset_temp / total_num;
    true_offset += temp * tensor_shape.tensor_strides[i];
    offset_temp -= temp * total_num;
  }
  return true_offset;
}

__mlu_func__ int getTrueOffset(int offset, TensorShape &tensor_shape) {
  // write this because of the bad performance of 64-bit div
  if (isUse64BitDiv(offset, tensor_shape.tensor_dims, tensor_shape.total_num)) {
    return getTrueOffsetInternel<int>(offset, tensor_shape);
  } else {
    return getTrueOffsetInternel<uint32_t>(offset, tensor_shape);
  }
}

template <typename U>
__mlu_func__ void getLineOffsetIndexArrayInternel(
    const U origin_offset, const int length, int *offset_start, int *offset_end,
    int *offset_flag, int *tensor_dims, U origin_total_num) {
  // use template because of the bad performance of 64-bit div
  U offset_temp = origin_offset;
  U total_num = origin_total_num;
  int temp = 0;
  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    offset_flag[i] = 0;
    total_num = total_num / (U)tensor_dims[i];
    temp = offset_temp / total_num;
    offset_start[i] = temp;
    offset_temp -= temp * total_num;
  }

  offset_temp = origin_offset + length - 1;
  total_num = origin_total_num;
  temp = 0;
  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    total_num = total_num / (U)tensor_dims[i];
    temp = offset_temp / total_num;
    offset_end[i] = temp;
    offset_temp -= temp * total_num;
  }
}

__mlu_func__ void getLineOffsetIndexArray(const int origin_offset,
                                          const int length, int *offset_start,
                                          int *offset_end, int *offset_flag,
                                          int *tensor_dims,
                                          int origin_total_num) {
  // write this because of the bad performance of 64-bit div
  if (isUse64BitDiv(origin_offset, tensor_dims, origin_total_num)) {
    getLineOffsetIndexArrayInternel<int>(origin_offset, length, offset_start,
                                         offset_end, offset_flag, tensor_dims,
                                         origin_total_num);

  } else {
    getLineOffsetIndexArrayInternel<uint32_t>(
        origin_offset, length, offset_start, offset_end, offset_flag,
        tensor_dims, origin_total_num);
  }
}

__mlu_func__ void updateOffsetFlag(int dim, int *index, int *offset_start,
                                   int *offset_end, int *offset_flag,
                                   int *tensor_dims) {
  if (offset_flag[dim] == 0) {
    index[dim] = offset_start[dim];
    offset_flag[dim] = 1;
  }
  if (dim > 0) {
    if (index[dim - 1] == offset_end[dim - 1] && offset_flag[dim - 1] == 2) {
      offset_flag[dim] = 2;
    }
  }
}

template <typename T>
__mlu_func__ void strideLoad(int8_t *dst, const int8_t *src,
                             int num,  // tensor_shape.tensor_dims
                             int dtype_size, int dst_stride, int src_stride,
                             int seg_num) {  // tensor_shape.tensor_dims
  if (dst_stride == num && src_stride == num) {
    __memcpy(dst, src, seg_num * num * dtype_size, GDRAM2NRAM);
    return;
  }
  // The range if <count> in __memcpy is [0, 65535].
  int count_max = 65536;
  if (seg_num > count_max) {
    int repeat_time = seg_num >> 16;
    // __memcpy dtype of segment is int32_t (except G2G, which is int)
    int rem = seg_num % count_max;
    for (int i = 0; i < repeat_time; ++i) {
      __memcpy(dst + i * count_max * dst_stride * dtype_size,
               src + OFFSET_SHIFT(i * count_max * src_stride, sizeof(T)),
               num * dtype_size, GDRAM2NRAM, dst_stride * dtype_size,
               src_stride * dtype_size, count_max - 1);
    }
    if (rem) {
      __memcpy(
          dst + repeat_time * count_max * dst_stride * dtype_size,
          src + OFFSET_SHIFT(repeat_time * count_max * src_stride, sizeof(T)),
          num * dtype_size, GDRAM2NRAM, dst_stride * dtype_size,
          src_stride * dtype_size, rem - 1);
    }
  } else {
    __memcpy(dst, src, num * dtype_size, GDRAM2NRAM, dst_stride * dtype_size,
             src_stride * dtype_size, seg_num - 1);
  }
}

template <typename T>
__mlu_func__ void strideLoadSram(int8_t *dst, const int8_t *src, int num,
                                 int dtype_size, int dst_stride, int src_stride,
                                 int seg_num) {
#if MAX_SRAM_SIZE > 0  // TODO(sram): tp_520
  if (dst_stride == num && src_stride == num) {
    __memcpy(dst, src, seg_num * num * dtype_size, GDRAM2SRAM);
    return;
  }
  // The range if <count> in __memcpy is [0, 65535].
  int count_max = 65536;
  if (seg_num > count_max) {
    int repeat_time = seg_num >> 16;
    // __memcpy dtype of segment is int32_t (except G2G, which is int)
    int rem = seg_num % count_max;
    for (int i = 0; i < repeat_time; ++i) {
      __memcpy(dst + i * count_max * dst_stride * dtype_size,
               src + OFFSET_SHIFT(i * count_max * src_stride, sizeof(T)),
               num * dtype_size, GDRAM2SRAM, dst_stride * dtype_size,
               src_stride * dtype_size, count_max - 1);
    }
    if (rem) {
      __memcpy(
          dst + repeat_time * count_max * dst_stride * dtype_size,
          src + OFFSET_SHIFT(repeat_time * count_max * src_stride, sizeof(T)),
          num * dtype_size, GDRAM2SRAM, dst_stride * dtype_size,
          src_stride * dtype_size, rem - 1);
    }
  } else {
    __memcpy(dst, src, num * dtype_size, GDRAM2SRAM, dst_stride * dtype_size,
             src_stride * dtype_size, seg_num - 1);
  }
#endif
}

template <typename T>
__mlu_device__ void tensorStrideLoad(void *dst_nram, const void *src_gdram,
                                     int src_offset, int data_num,
                                     int dtype_size,
                                     TensorShape &tensor_shape) {
  if (data_num == 0) {
    return;
  }
  if (tensor_shape.is_contiguous) {
    __memcpy((int8_t *)dst_nram,
             (const int8_t *)src_gdram + OFFSET_SHIFT(src_offset, sizeof(T)),
             data_num * dtype_size, GDRAM2NRAM);
    return;
  }
  if (data_num == 1) {
    __memcpy(
        (int8_t *)dst_nram,
        (const int8_t *)src_gdram +
            OFFSET_SHIFT(getTrueOffset(src_offset, tensor_shape), sizeof(T)),
        dtype_size, GDRAM2NRAM);
    return;
  }
  int offset_0 = 0;
  int offset_1 = 0;
  int offset_2 = 0;
  int offset_3 = 0;
  int offset_4 = 0;
  int offset_5 = 0;
  int offset_6 = 0;
  int count = 0;
  int index[MLUOP_DIM_MAX];
  int offset_start[MLUOP_DIM_MAX];
  int offset_end[MLUOP_DIM_MAX];
  int offset_flag[MLUOP_DIM_MAX];
  int tensor_dims[MLUOP_DIM_MAX];
  int tensor_strides[MLUOP_DIM_MAX];
  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    tensor_dims[i] = tensor_shape.tensor_dims[i];
    tensor_strides[i] = tensor_shape.tensor_strides[i];
  }
  getLineOffsetIndexArray(src_offset, data_num, offset_start, offset_end,
                          offset_flag, tensor_dims, tensor_shape.total_num);
  for (index[0] = 0; index[0] < offset_end[0] + 1; ++index[0]) {
    updateOffsetFlag(0, index, offset_start, offset_end, offset_flag,
                     tensor_dims);
    offset_flag[0] = 2;
    offset_0 = index[0] * tensor_strides[0];
    for (index[1] = 0; index[1] < tensor_dims[1]; ++index[1]) {
      updateOffsetFlag(1, index, offset_start, offset_end, offset_flag,
                       tensor_dims);
      if (offset_flag[1] == 2 && index[1] > offset_end[1]) {
        break;
      }
      offset_1 = offset_0 + index[1] * tensor_strides[1];
      for (index[2] = 0; index[2] < tensor_dims[2]; ++index[2]) {
        updateOffsetFlag(2, index, offset_start, offset_end, offset_flag,
                         tensor_dims);
        if (offset_flag[2] == 2 && index[2] > offset_end[2]) {
          break;
        }
        offset_2 = offset_1 + index[2] * tensor_strides[2];
        for (index[3] = 0; index[3] < tensor_dims[3]; ++index[3]) {
          updateOffsetFlag(3, index, offset_start, offset_end, offset_flag,
                           tensor_dims);
          if (offset_flag[3] == 2 && index[3] > offset_end[3]) {
            break;
          }
          offset_3 = offset_2 + index[3] * tensor_strides[3];
          for (index[4] = 0; index[4] < tensor_dims[4]; ++index[4]) {
            updateOffsetFlag(4, index, offset_start, offset_end, offset_flag,
                             tensor_dims);
            if (offset_flag[4] == 2 && index[4] > offset_end[4]) {
              break;
            }
            offset_4 = offset_3 + index[4] * tensor_strides[4];
            for (index[5] = 0; index[5] < tensor_dims[5]; ++index[5]) {
              updateOffsetFlag(5, index, offset_start, offset_end, offset_flag,
                               tensor_dims);
              if (offset_flag[5] == 2 && index[5] > offset_end[5]) {
                break;
              }
              offset_5 = offset_4 + index[5] * tensor_strides[5];
              if (tensor_strides[7] == 1 && offset_start[6] == 0 &&
                  offset_end[6] == tensor_dims[6] && offset_start[7] == 0 &&
                  offset_end[7] == tensor_dims[7]) {
                strideLoad<T>((int8_t *)dst_nram + count * dtype_size,
                              (const int8_t *)src_gdram +
                                  OFFSET_SHIFT(offset_5, sizeof(T)),
                              tensor_dims[7], dtype_size, tensor_dims[7],
                              tensor_strides[6], tensor_dims[6]);
                count += tensor_dims[6] * tensor_dims[7];
              } else {
                for (index[6] = 0; index[6] < tensor_dims[6]; ++index[6]) {
                  updateOffsetFlag(6, index, offset_start, offset_end,
                                   offset_flag, tensor_dims);
                  if (offset_flag[6] == 2 && index[6] > offset_end[6]) {
                    break;
                  }
                  offset_6 = offset_5 + index[6] * tensor_strides[6];
                  if (offset_flag[7] == 0) {
                    offset_6 += offset_start[7] * tensor_strides[7];
                    if (index[6] == offset_end[6] && offset_flag[6] == 2) {
                      strideLoad<T>((int8_t *)dst_nram + count * dtype_size,
                                    (const int8_t *)src_gdram +
                                        OFFSET_SHIFT(offset_6, sizeof(T)),
                                    1, dtype_size, 1, tensor_strides[7],
                                    offset_end[7] - offset_start[7] + 1);
                      count += offset_end[7] - offset_start[7] + 1;
                    } else {
                      strideLoad<T>((int8_t *)dst_nram + count * dtype_size,
                                    (const int8_t *)src_gdram +
                                        OFFSET_SHIFT(offset_6, sizeof(T)),
                                    1, dtype_size, 1, tensor_strides[7],
                                    tensor_dims[7] - offset_start[7]);
                      count += tensor_dims[7] - offset_start[7];
                      offset_flag[7] = 1;
                    }
                  } else {
                    if (index[6] == offset_end[6] && offset_flag[6] == 2) {
                      strideLoad<T>((int8_t *)dst_nram + count * dtype_size,
                                    (const int8_t *)src_gdram +
                                        OFFSET_SHIFT(offset_6, sizeof(T)),
                                    1, dtype_size, 1, tensor_strides[7],
                                    offset_end[7] + 1);
                      count += offset_end[7] + 1;
                    } else {
                      strideLoad<T>((int8_t *)dst_nram + count * dtype_size,
                                    (const int8_t *)src_gdram +
                                        OFFSET_SHIFT(offset_6, sizeof(T)),
                                    1, dtype_size, 1, tensor_strides[7],
                                    tensor_dims[7]);
                      count += tensor_dims[7];
                      offset_flag[7] = 1;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
__mlu_device__ void tensorStrideLoadSram(void *dst_sram, const void *src_gdram,
                                         int src_offset, int data_num,
                                         int dtype_size,
                                         TensorShape &tensor_shape) {
#if MAX_SRAM_SIZE > 0  // TODO(sram): tp_520
  if (data_num == 0) {
    return;
  }
  if (tensor_shape.is_contiguous) {
    __memcpy((int8_t *)dst_sram,
             (const int8_t *)src_gdram + OFFSET_SHIFT(src_offset, sizeof(T)),
             data_num * dtype_size, GDRAM2SRAM);
    return;
  }
  if (data_num == 1) {
    __memcpy(
        (int8_t *)dst_sram,
        (const int8_t *)src_gdram +
            OFFSET_SHIFT(getTrueOffset(src_offset, tensor_shape), sizeof(T)),
        dtype_size, GDRAM2SRAM);
    return;
  }
  int offset_0 = 0;
  int offset_1 = 0;
  int offset_2 = 0;
  int offset_3 = 0;
  int offset_4 = 0;
  int offset_5 = 0;
  int offset_6 = 0;
  int count = 0;
  int index[MLUOP_DIM_MAX];
  int offset_start[MLUOP_DIM_MAX];
  int offset_end[MLUOP_DIM_MAX];
  int offset_flag[MLUOP_DIM_MAX];
  int tensor_dims[MLUOP_DIM_MAX];
  int tensor_strides[MLUOP_DIM_MAX];
  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    tensor_dims[i] = tensor_shape.tensor_dims[i];
    tensor_strides[i] = tensor_shape.tensor_strides[i];
  }
  getLineOffsetIndexArray(src_offset, data_num, offset_start, offset_end,
                          offset_flag, tensor_dims, tensor_shape.total_num);

  for (index[0] = 0; index[0] < offset_end[0] + 1; ++index[0]) {
    updateOffsetFlag(0, index, offset_start, offset_end, offset_flag,
                     tensor_dims);
    offset_flag[0] = 2;
    offset_0 = index[0] * tensor_strides[0];
    for (index[1] = 0; index[1] < tensor_dims[1]; ++index[1]) {
      updateOffsetFlag(1, index, offset_start, offset_end, offset_flag,
                       tensor_dims);
      if (offset_flag[1] == 2 && index[1] > offset_end[1]) {
        break;
      }
      offset_1 = offset_0 + index[1] * tensor_strides[1];
      for (index[2] = 0; index[2] < tensor_dims[2]; ++index[2]) {
        updateOffsetFlag(2, index, offset_start, offset_end, offset_flag,
                         tensor_dims);
        if (offset_flag[2] == 2 && index[2] > offset_end[2]) {
          break;
        }
        offset_2 = offset_1 + index[2] * tensor_strides[2];
        for (index[3] = 0; index[3] < tensor_dims[3]; ++index[3]) {
          updateOffsetFlag(3, index, offset_start, offset_end, offset_flag,
                           tensor_dims);
          if (offset_flag[3] == 2 && index[3] > offset_end[3]) {
            break;
          }
          offset_3 = offset_2 + index[3] * tensor_strides[3];
          for (index[4] = 0; index[4] < tensor_dims[4]; ++index[4]) {
            updateOffsetFlag(4, index, offset_start, offset_end, offset_flag,
                             tensor_dims);
            if (offset_flag[4] == 2 && index[4] > offset_end[4]) {
              break;
            }
            offset_4 = offset_3 + index[4] * tensor_strides[4];
            for (index[5] = 0; index[5] < tensor_dims[5]; ++index[5]) {
              updateOffsetFlag(5, index, offset_start, offset_end, offset_flag,
                               tensor_dims);
              if (offset_flag[5] == 2 && index[5] > offset_end[5]) {
                break;
              }
              offset_5 = offset_4 + index[5] * tensor_strides[5];
              if (tensor_strides[7] == 1 && offset_start[6] == 0 &&
                  offset_end[6] == tensor_dims[6] && offset_start[7] == 0 &&
                  offset_end[7] == tensor_dims[7]) {
                strideLoadSram<T>((int8_t *)dst_sram + count * dtype_size,
                                  (const int8_t *)src_gdram +
                                      OFFSET_SHIFT(offset_5, sizeof(T)),
                                  tensor_dims[7], dtype_size, tensor_dims[7],
                                  tensor_strides[6], tensor_dims[6]);
                count += tensor_dims[6] * tensor_dims[7];
              } else {
                for (index[6] = 0; index[6] < tensor_dims[6]; ++index[6]) {
                  updateOffsetFlag(6, index, offset_start, offset_end,
                                   offset_flag, tensor_dims);
                  if (offset_flag[6] == 2 && index[6] > offset_end[6]) {
                    break;
                  }
                  offset_6 = offset_5 + index[6] * tensor_strides[6];
                  if (offset_flag[7] == 0) {
                    offset_6 += offset_start[7] * tensor_strides[7];
                    if (index[6] == offset_end[6] && offset_flag[6] == 2) {
                      strideLoadSram<T>((int8_t *)dst_sram + count * dtype_size,
                                        (const int8_t *)src_gdram +
                                            OFFSET_SHIFT(offset_6, sizeof(T)),
                                        1, dtype_size, 1, tensor_strides[7],
                                        offset_end[7] - offset_start[7] + 1);
                      count += offset_end[7] - offset_start[7] + 1;
                    } else {
                      strideLoadSram<T>((int8_t *)dst_sram + count * dtype_size,
                                        (const int8_t *)src_gdram +
                                            OFFSET_SHIFT(offset_6, sizeof(T)),
                                        1, dtype_size, 1, tensor_strides[7],
                                        tensor_dims[7] - offset_start[7]);
                      count += tensor_dims[7] - offset_start[7];
                      offset_flag[7] = 1;
                    }
                  } else {
                    if (index[6] == offset_end[6] && offset_flag[6] == 2) {
                      strideLoadSram<T>((int8_t *)dst_sram + count * dtype_size,
                                        (const int8_t *)src_gdram +
                                            OFFSET_SHIFT(offset_6, sizeof(T)),
                                        1, dtype_size, 1, tensor_strides[7],
                                        offset_end[7] + 1);
                      count += offset_end[7] + 1;
                    } else {
                      strideLoadSram<T>((int8_t *)dst_sram + count * dtype_size,
                                        (const int8_t *)src_gdram +
                                            OFFSET_SHIFT(offset_6, sizeof(T)),
                                        1, dtype_size, 1, tensor_strides[7],
                                        tensor_dims[7]);
                      count += tensor_dims[7];
                      offset_flag[7] = 1;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
#endif
}

template <typename T>
__mlu_device__ void tensorStrideLoad(void *dst_nram, const void *src_gdram,
                                     int src_offset, int data_num,
                                     int dtype_size, int dst_stride,
                                     int src_stride, int count,
                                     TensorShape &tensor_shape) {
  for (int i = 0; i < count; i++) {
    tensorStrideLoad<T>(
        (int8_t *)dst_nram + OFFSET_SHIFT(i * dst_stride, sizeof(T)),
        (const int8_t *)src_gdram, src_offset + i * src_stride, data_num,
        dtype_size, tensor_shape);
  }
}

template <typename T>
__mlu_device__ void tensorStrideLoadSram(void *dst_sram, const void *src_gdram,
                                         int src_offset, int data_num,
                                         int dtype_size, int dst_stride,
                                         int src_stride, int count,
                                         TensorShape &tensor_shape) {
#if MAX_SRAM_SIZE > 0  // TODO(sram): tp_520
  for (int i = 0; i < count; i++) {
    tensorStrideLoadSram<T>(
        (int8_t *)dst_sram + OFFSET_SHIFT(i * dst_stride, sizeof(T)),
        (const int8_t *)src_gdram, src_offset + i * src_stride, data_num,
        dtype_size, tensor_shape);
  }
#endif
}

template <typename T>
__mlu_func__ void strideStore(const int8_t *src, int8_t *dst, int num,
                              int dtype_size, int src_stride, int dst_stride,
                              int seg_num) {
  if (dst_stride == num && src_stride == num) {
    __memcpy(dst, src, seg_num * num * dtype_size, NRAM2GDRAM);
    return;
  }
  // The range if <count> in __memcpy is [0, 65535].
  int count_max = 65536;
  if (seg_num > count_max) {
    int repeat_time = seg_num >> 16;
    int rem = seg_num % count_max;
    for (int i = 0; i < repeat_time; ++i) {
      __memcpy(dst + OFFSET_SHIFT(i * count_max * dst_stride, sizeof(T)),
               src + i * count_max * src_stride * dtype_size, num * dtype_size,
               NRAM2GDRAM, dst_stride * dtype_size, src_stride * dtype_size,
               count_max - 1);
    }
    if (rem) {
      __memcpy(
          dst + OFFSET_SHIFT(repeat_time * count_max * dst_stride, sizeof(T)),
          src + repeat_time * count_max * src_stride * dtype_size,
          num * dtype_size, NRAM2GDRAM, dst_stride * dtype_size,
          src_stride * dtype_size, rem - 1);
    }
  } else {
    __memcpy(dst, src, num * dtype_size, NRAM2GDRAM, dst_stride * dtype_size,
             src_stride * dtype_size, seg_num - 1);
  }
}

template <typename T>
__mlu_func__ void strideStoreSram(const int8_t *src, int8_t *dst, int num,
                                  int dtype_size, int src_stride,
                                  int dst_stride, int seg_num) {
#if MAX_SRAM_SIZE > 0  // TODO(sram): tp_520
  if (dst_stride == num && src_stride == num) {
    __memcpy(dst, src, seg_num * num * dtype_size, SRAM2GDRAM);
    return;
  }
  // The range if <count> in __memcpy is [0, 65535].
  int count_max = 65536;
  if (seg_num > count_max) {
    int repeat_time = seg_num >> 16;
    int rem = seg_num % count_max;
    for (int i = 0; i < repeat_time; ++i) {
      __memcpy(dst + OFFSET_SHIFT(i * count_max * dst_stride, sizeof(T)),
               src + i * count_max * src_stride * dtype_size, num * dtype_size,
               SRAM2GDRAM, dst_stride * dtype_size, src_stride * dtype_size,
               count_max - 1);
    }
    if (rem) {
      __memcpy(
          dst + OFFSET_SHIFT(repeat_time * count_max * dst_stride, sizeof(T)),
          src + repeat_time * count_max * src_stride * dtype_size,
          num * dtype_size, SRAM2GDRAM, dst_stride * dtype_size,
          src_stride * dtype_size, rem - 1);
    }
  } else {
    __memcpy(dst, src, num * dtype_size, SRAM2GDRAM, dst_stride * dtype_size,
             src_stride * dtype_size, seg_num - 1);
  }
#endif
}

template <typename T>
__mlu_device__ void tensorStrideStore(void *dst_gdram, int dst_offset,
                                      const void *src_nram, int data_num,
                                      int dtype_size,
                                      TensorShape &tensor_shape) {
  if (data_num == 0) {
    return;
  }
  if (tensor_shape.is_contiguous) {
    __memcpy((int8_t *)dst_gdram + OFFSET_SHIFT(dst_offset, sizeof(T)),
             (const int8_t *)src_nram, data_num * dtype_size, NRAM2GDRAM);
    return;
  }
  if (data_num == 1) {
    __memcpy(
        (int8_t *)dst_gdram +
            OFFSET_SHIFT(getTrueOffset(dst_offset, tensor_shape), sizeof(T)),
        (const int8_t *)src_nram, dtype_size, NRAM2GDRAM);
    return;
  }
  int offset_0 = 0;
  int offset_1 = 0;
  int offset_2 = 0;
  int offset_3 = 0;
  int offset_4 = 0;
  int offset_5 = 0;
  int offset_6 = 0;
  int count = 0;
  int index[MLUOP_DIM_MAX];
  int offset_start[MLUOP_DIM_MAX];
  int offset_end[MLUOP_DIM_MAX];
  int offset_flag[MLUOP_DIM_MAX];
  int tensor_dims[MLUOP_DIM_MAX];
  int tensor_strides[MLUOP_DIM_MAX];
  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    tensor_dims[i] = tensor_shape.tensor_dims[i];
    tensor_strides[i] = tensor_shape.tensor_strides[i];
  }
  getLineOffsetIndexArray(dst_offset, data_num, offset_start, offset_end,
                          offset_flag, tensor_dims, tensor_shape.total_num);

  for (index[0] = 0; index[0] < offset_end[0] + 1; ++index[0]) {
    updateOffsetFlag(0, index, offset_start, offset_end, offset_flag,
                     tensor_dims);
    offset_flag[0] = 2;
    offset_0 = index[0] * tensor_strides[0];
    for (index[1] = 0; index[1] < tensor_dims[1]; ++index[1]) {
      updateOffsetFlag(1, index, offset_start, offset_end, offset_flag,
                       tensor_dims);
      if (offset_flag[1] == 2 && index[1] > offset_end[1]) {
        break;
      }
      offset_1 = offset_0 + index[1] * tensor_strides[1];
      for (index[2] = 0; index[2] < tensor_dims[2]; ++index[2]) {
        updateOffsetFlag(2, index, offset_start, offset_end, offset_flag,
                         tensor_dims);
        if (offset_flag[2] == 2 && index[2] > offset_end[2]) {
          break;
        }
        offset_2 = offset_1 + index[2] * tensor_strides[2];
        for (index[3] = 0; index[3] < tensor_dims[3]; ++index[3]) {
          updateOffsetFlag(3, index, offset_start, offset_end, offset_flag,
                           tensor_dims);
          if (offset_flag[3] == 2 && index[3] > offset_end[3]) {
            break;
          }
          offset_3 = offset_2 + index[3] * tensor_strides[3];
          for (index[4] = 0; index[4] < tensor_dims[4]; ++index[4]) {
            updateOffsetFlag(4, index, offset_start, offset_end, offset_flag,
                             tensor_dims);
            if (offset_flag[4] == 2 && index[4] > offset_end[4]) {
              break;
            }
            offset_4 = offset_3 + index[4] * tensor_strides[4];
            for (index[5] = 0; index[5] < tensor_dims[5]; ++index[5]) {
              updateOffsetFlag(5, index, offset_start, offset_end, offset_flag,
                               tensor_dims);
              if (offset_flag[5] == 2 && index[5] > offset_end[5]) {
                break;
              }
              offset_5 = offset_4 + index[5] * tensor_strides[5];
              if (tensor_strides[7] == 1 && offset_start[6] == 0 &&
                  offset_end[6] == tensor_dims[6] && offset_start[7] == 0 &&
                  offset_end[7] == tensor_dims[7]) {
                strideStore<T>(
                    (const int8_t *)src_nram + count * dtype_size,
                    (int8_t *)dst_gdram + OFFSET_SHIFT(offset_5, sizeof(T)),
                    tensor_dims[7], dtype_size, tensor_dims[7],
                    tensor_strides[6], tensor_dims[6]);
                count += tensor_dims[6] * tensor_dims[7];
              } else {
                for (index[6] = 0; index[6] < tensor_dims[6]; ++index[6]) {
                  updateOffsetFlag(6, index, offset_start, offset_end,
                                   offset_flag, tensor_dims);
                  if (offset_flag[6] == 2 && index[6] > offset_end[6]) {
                    break;
                  }
                  offset_6 = offset_5 + index[6] * tensor_strides[6];
                  if (offset_flag[7] == 0) {
                    offset_6 += offset_start[7] * tensor_strides[7];
                    if (index[6] == offset_end[6] && offset_flag[6] == 2) {
                      strideStore<T>(
                          (const int8_t *)src_nram + count * dtype_size,
                          (int8_t *)dst_gdram +
                              OFFSET_SHIFT(offset_6, sizeof(T)),
                          1, dtype_size, 1, tensor_strides[7],
                          offset_end[7] - offset_start[7] + 1);
                      count += offset_end[7] - offset_start[7] + 1;
                    } else {
                      strideStore<T>(
                          (const int8_t *)src_nram + count * dtype_size,
                          (int8_t *)dst_gdram +
                              OFFSET_SHIFT(offset_6, sizeof(T)),
                          1, dtype_size, 1, tensor_strides[7],
                          tensor_dims[7] - offset_start[7]);
                      count += tensor_dims[7] - offset_start[7];
                      offset_flag[7] = 1;
                    }
                  } else {
                    if (index[6] == offset_end[6] && offset_flag[6] == 2) {
                      strideStore<T>(
                          (const int8_t *)src_nram + count * dtype_size,
                          (int8_t *)dst_gdram +
                              OFFSET_SHIFT(offset_6, sizeof(T)),
                          1, dtype_size, 1, tensor_strides[7],
                          offset_end[7] + 1);
                      count += offset_end[7] + 1;
                    } else {
                      strideStore<T>(
                          (const int8_t *)src_nram + count * dtype_size,
                          (int8_t *)dst_gdram +
                              OFFSET_SHIFT(offset_6, sizeof(T)),
                          1, dtype_size, 1, tensor_strides[7], tensor_dims[7]);
                      count += tensor_dims[7];
                      offset_flag[7] = 1;
                    }
                  }
                }
                if (tensor_strides[6] == 0) {
                  break;
                }
              }
              if (tensor_strides[5] == 0) {
                break;
              }
            }
            if (tensor_strides[4] == 0) {
              break;
            }
          }
          if (tensor_strides[3] == 0) {
            break;
          }
        }
        if (tensor_strides[2] == 0) {
          break;
        }
      }
      if (tensor_strides[1] == 0) {
        break;
      }
    }
    if (tensor_strides[0] == 0) {
      break;
    }
  }
}

template <typename T>
__mlu_device__ void tensorStrideStoreSram(void *dst_gdram, int dst_offset,
                                          const void *src_sram, int data_num,
                                          int dtype_size,
                                          TensorShape &tensor_shape) {
#if MAX_SRAM_SIZE > 0  // TODO(sram): tp_520
  if (data_num == 0) {
    return;
  }
  if (tensor_shape.is_contiguous) {
    __memcpy((int8_t *)dst_gdram + OFFSET_SHIFT(dst_offset, sizeof(T)),
             (const int8_t *)src_sram, data_num * dtype_size, SRAM2GDRAM);
    return;
  }
  if (data_num == 1) {
    __memcpy(
        (int8_t *)dst_gdram +
            OFFSET_SHIFT(getTrueOffset(dst_offset, tensor_shape), sizeof(T)),
        (const int8_t *)src_sram, dtype_size, SRAM2GDRAM);
    return;
  }
  int offset_0 = 0;
  int offset_1 = 0;
  int offset_2 = 0;
  int offset_3 = 0;
  int offset_4 = 0;
  int offset_5 = 0;
  int offset_6 = 0;
  int count = 0;
  int index[MLUOP_DIM_MAX];
  int offset_start[MLUOP_DIM_MAX];
  int offset_end[MLUOP_DIM_MAX];
  int offset_flag[MLUOP_DIM_MAX];
  int tensor_dims[MLUOP_DIM_MAX];
  int tensor_strides[MLUOP_DIM_MAX];
  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    tensor_dims[i] = tensor_shape.tensor_dims[i];
    tensor_strides[i] = tensor_shape.tensor_strides[i];
  }
  getLineOffsetIndexArray(dst_offset, data_num, offset_start, offset_end,
                          offset_flag, tensor_dims, tensor_shape.total_num);

  for (index[0] = 0; index[0] < offset_end[0] + 1; ++index[0]) {
    updateOffsetFlag(0, index, offset_start, offset_end, offset_flag,
                     tensor_dims);
    offset_flag[0] = 2;
    offset_0 = index[0] * tensor_strides[0];
    for (index[1] = 0; index[1] < tensor_dims[1]; ++index[1]) {
      updateOffsetFlag(1, index, offset_start, offset_end, offset_flag,
                       tensor_dims);
      if (offset_flag[1] == 2 && index[1] > offset_end[1]) {
        break;
      }
      offset_1 = offset_0 + index[1] * tensor_strides[1];
      for (index[2] = 0; index[2] < tensor_dims[2]; ++index[2]) {
        updateOffsetFlag(2, index, offset_start, offset_end, offset_flag,
                         tensor_dims);
        if (offset_flag[2] == 2 && index[2] > offset_end[2]) {
          break;
        }
        offset_2 = offset_1 + index[2] * tensor_strides[2];
        for (index[3] = 0; index[3] < tensor_dims[3]; ++index[3]) {
          updateOffsetFlag(3, index, offset_start, offset_end, offset_flag,
                           tensor_dims);
          if (offset_flag[3] == 2 && index[3] > offset_end[3]) {
            break;
          }
          offset_3 = offset_2 + index[3] * tensor_strides[3];
          for (index[4] = 0; index[4] < tensor_dims[4]; ++index[4]) {
            updateOffsetFlag(4, index, offset_start, offset_end, offset_flag,
                             tensor_dims);
            if (offset_flag[4] == 2 && index[4] > offset_end[4]) {
              break;
            }
            offset_4 = offset_3 + index[4] * tensor_strides[4];
            for (index[5] = 0; index[5] < tensor_dims[5]; ++index[5]) {
              updateOffsetFlag(5, index, offset_start, offset_end, offset_flag,
                               tensor_dims);
              if (offset_flag[5] == 2 && index[5] > offset_end[5]) {
                break;
              }
              offset_5 = offset_4 + index[5] * tensor_strides[5];
              if (tensor_strides[7] == 1 && offset_start[6] == 0 &&
                  offset_end[6] == tensor_dims[6] && offset_start[7] == 0 &&
                  offset_end[7] == tensor_dims[7]) {
                strideStoreSram<T>(
                    (const int8_t *)src_sram + count * dtype_size,
                    (int8_t *)dst_gdram + OFFSET_SHIFT(offset_5, sizeof(T)),
                    tensor_dims[7], dtype_size, tensor_dims[7],
                    tensor_strides[6], tensor_dims[6]);
                count += tensor_dims[6] * tensor_dims[7];
              } else {
                for (index[6] = 0; index[6] < tensor_dims[6]; ++index[6]) {
                  updateOffsetFlag(6, index, offset_start, offset_end,
                                   offset_flag, tensor_dims);
                  if (offset_flag[6] == 2 && index[6] > offset_end[6]) {
                    break;
                  }
                  offset_6 = offset_5 + index[6] * tensor_strides[6];
                  if (offset_flag[7] == 0) {
                    offset_6 += offset_start[7] * tensor_strides[7];
                    if (index[6] == offset_end[6] && offset_flag[6] == 2) {
                      strideStoreSram<T>(
                          (const int8_t *)src_sram + count * dtype_size,
                          (int8_t *)dst_gdram +
                              OFFSET_SHIFT(offset_6, sizeof(T)),
                          1, dtype_size, 1, tensor_strides[7],
                          offset_end[7] - offset_start[7] + 1);
                      count += offset_end[7] - offset_start[7] + 1;
                    } else {
                      strideStoreSram<T>(
                          (const int8_t *)src_sram + count * dtype_size,
                          (int8_t *)dst_gdram +
                              OFFSET_SHIFT(offset_6, sizeof(T)),
                          1, dtype_size, 1, tensor_strides[7],
                          tensor_dims[7] - offset_start[7]);
                      count += tensor_dims[7] - offset_start[7];
                      offset_flag[7] = 1;
                    }
                  } else {
                    if (index[6] == offset_end[6] && offset_flag[6] == 2) {
                      strideStoreSram<T>(
                          (const int8_t *)src_sram + count * dtype_size,
                          (int8_t *)dst_gdram +
                              OFFSET_SHIFT(offset_6, sizeof(T)),
                          1, dtype_size, 1, tensor_strides[7],
                          offset_end[7] + 1);
                      count += offset_end[7] + 1;
                    } else {
                      strideStoreSram<T>(
                          (const int8_t *)src_sram + count * dtype_size,
                          (int8_t *)dst_gdram +
                              OFFSET_SHIFT(offset_6, sizeof(T)),
                          1, dtype_size, 1, tensor_strides[7], tensor_dims[7]);
                      count += tensor_dims[7];
                      offset_flag[7] = 1;
                    }
                  }
                }
                if (tensor_strides[6] == 0) {
                  break;
                }
              }
              if (tensor_strides[5] == 0) {
                break;
              }
            }
            if (tensor_strides[4] == 0) {
              break;
            }
          }
          if (tensor_strides[3] == 0) {
            break;
          }
        }
        if (tensor_strides[2] == 0) {
          break;
        }
      }
      if (tensor_strides[1] == 0) {
        break;
      }
    }
    if (tensor_strides[0] == 0) {
      break;
    }
  }
#endif
}

template <typename T>
__mlu_device__ void tensorStrideStore(void *dst_gdram, int dst_offset,
                                      const void *src_nram, int data_num,
                                      int dtype_size, int dst_stride,
                                      int src_stride, int count,
                                      TensorShape &tensor_shape) {
  for (int i = 0; i < count; i++) {
    tensorStrideStore<T>(
        (int8_t *)dst_gdram, dst_offset + i * dst_stride,
        (const int8_t *)src_nram + OFFSET_SHIFT(i * src_stride, sizeof(T)),
        data_num, dtype_size, tensor_shape);
  }
}

template <typename T>
__mlu_device__ void tensorStrideStoreSram(void *dst_gdram, int dst_offset,
                                          const void *src_sram, int data_num,
                                          int dtype_size, int dst_stride,
                                          int src_stride, int count,
                                          TensorShape &tensor_shape) {
#if MAX_SRAM_SIZE > 0  // TODO(sram): tp_520
  for (int i = 0; i < count; i++) {
    tensorStrideStoreSram<T>(
        (int8_t *)dst_gdram, dst_offset + i * dst_stride,
        (const int8_t *)src_sram + OFFSET_SHIFT(i * src_stride, sizeof(T)),
        data_num, dtype_size, tensor_shape);
  }
#endif
}
#endif  // KERNELS_TENSOR_STRIDE_PROCESS_TENSOR_STRIDE_PROCESS_COMMON_H_
