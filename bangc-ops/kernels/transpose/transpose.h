/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef KERNELS_TRANSPOSE_KERNEL_TRANSPOSE_H_
#define KERNELS_TRANSPOSE_KERNEL_TRANSPOSE_H_

#include <vector>

#include "kernels/kernel.h"
#include "mlu_op.h"

#define BANG_TRANSPOSE_ALIGN_FP16 (32)
#define MEMCPY_NRAM2NRAM_ALIGN_FP16 (64)
#define BANG_TRANSPOSE_ALIGN_FP32 (16)
#define MEMCPY_NRAM2NRAM_ALIGN_FP32 (32)
#define BANG_TRANSPOSE_ALIGN_INT8 (64)
#define MEMCPY_NRAM2NRAM_ALIGN_INT8 (128)
#define BANG_TRANSPOSE_ALIGN_FP64 (8)

#define BANG_TRANSPOSE_ALIGN_BYTE (128)
#define MEMCPY_NRAM2NRAM_ALIGN_BYTE (128)

#define TRANSPOSE_NCHW2NHWC_ALIGN_FP16 (32)
#define TRANSPOSE_NCHW2NHWC_ALIGN_FP32 (16)
#define TRANSPOSE_NCHW2NHWC_ALIGN_INT8 (64)

#define RESERVED_NRAM (12 * 1024)
#define NRAM_LIMIT (MAX_NRAM_SIZE + REM_FOR_STACK - RESERVED_NRAM)

#define NRAM_NUM_LIMIT_FP32 (NRAM_LIMIT / 4 / 4)
#define NRAM_NUM_LIMIT_FP16 (2 * NRAM_NUM_LIMIT_FP32)
#define NRAM_BYTE_LIMIT (4 * NRAM_NUM_LIMIT_FP32)

#define NRAM_BYTE_1D (NRAM_BYTE_LIMIT * 4)
#define NRAM_BYTE_2D (NRAM_BYTE_LIMIT * 2)

#define TRANSPOSE_MAX_DIM (8)
#define TRANSPOSE_MAX_U1_JOB (4)
#define TRANSPOSE_MAX_BLOCK_JOB (4)

#define CEIL(A, B) ((((A)-1) / (B) + 1) * (B))
#define FLOOR(A, B) (((A) / (B)) * (B))

#define TRANS_SEGNUM_LIMIT 65536

#define TRANS_ALIGN_64 (64)
#define TRANS_ALIGN_128 (128)

// the variable is used to optimize TRANSPOSE_3D102.
// for transpose_3D102, (N, H, W)
// batch = N * H, batch_per_core = a
// when a / H > TRANS_SEGNUM_THRESHOLD, storing data from nram to gdram with
// nram to buffer. TRANS_SEGNUM_THRESHOLD is experimental data according to the
// cases of benchmark
#define TRANS_SEGNUM_THRESHOLD 15

struct mluOpTransposeStruct {
  int dim;
  std::vector<int> permute;
};

typedef enum mluOpTransposeStrategy {
  TRANSPOSE_INVALID,
  TRANSPOSE_1D,
  TRANSPOSE_2D,
  TRANSPOSE_3D_021,
  TRANSPOSE_3D_102,
  TRANSPOSE_3D_210,
  TRANSPOSE_4D_0213,
  TRANSPOSE_4D_0321,
  TRANSPOSE_4D_1032,
  TRANSPOSE_4D_1302,
  TRANSPOSE_4D_1320,
  TRANSPOSE_4D_2031,
  TRANSPOSE_4D_2103,
  TRANSPOSE_4D_2130,
  TRANSPOSE_4D_3021,
  TRANSPOSE_4D_3102,
  TRANSPOSE_4D_3210,
  TRANSPOSE_COMMON,
} mluOpTransposeStrategy_t;

typedef enum mluOpTranspose2DStrategy {
  TR_2D_ALLSPLIT,
  TR_2D_SMALL,
  TR_2D_ENORMOUS,
  TR_2D_INVALID,
} mluOpTranspose2DStrategy_t;
typedef enum mluOpTranspose3DStrategy {
  TR_3D_021_INVALID,
  TR_3D_021_TILING,
  TR_3D_021_LOOP,
  TR_3D_210,
  TR_3D_210_OPTIMIZE,
} mluOpTranspose3DStrategy_t;

#if defined(__BANG__)
// NRAM_BYTE_LIMIT * 2 == 253952 bytes
__nram__ char bank[NRAM_BYTE_1D];
#define TRANS_INFO_NUM_RESERVED 16
#define TRANS_SRAM_SIZE_AVAILABLE (MAX_SRAM_SIZE - 16 * sizeof(int))
#if __BANG_ARCH__ >= 300
#define TRANS_MIN_PERF_LOOP 4
#else
#define TRANS_MIN_PERF_LOOP 5
#endif
#if MAX_SRAM_SIZE > 0
__mlu_shared__ int last_core_info[TRANS_INFO_NUM_RESERVED];
__mlu_shared__ char sbuf[TRANS_SRAM_SIZE_AVAILABLE];  // sharedRAM
#endif

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel1D(void *input, void *output,
                                         const size_t sum_num);
template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel2DSmall(void *input, void *output,
                                              const size_t h, const size_t w,
                                              const bool split_h);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel2D(void *input, void *output,
                                         const size_t h, const size_t w,
                                         const bool split_h);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel2DEnormous(
    void *input, void *output, const size_t h, const size_t w,
    const size_t num_split0, const size_t num_split1,
    const size_t num_processed0, const size_t num_processed1,
    const size_t num_processed_ceil0, const size_t num_processed_ceil1);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel3D021(
    void *x, void *y, const size_t a, const size_t b, const size_t c,
    const size_t num_split0, const size_t num_split1, const size_t num_split2,
    const bool split_h, const size_t num_processed0,
    const size_t num_processed1, const size_t num_processed2,
    const size_t num_processed_ceil0, const size_t num_processed_ceil1,
    const size_t num_processed_ceil2, const size_t num_processed_limit0,
    const size_t num_processed_limit1, const size_t num_processed_limit2,
    const size_t num_processed_ceil_limit0,
    const size_t num_processed_ceil_limit1,
    const size_t num_processed_ceil_limit2);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel3D021Small(
    void *x, void *y, const size_t a, const size_t b, const size_t c,
    const size_t num_split, const bool split_h, const size_t num_processed,
    const size_t num_processed_ceil, const size_t num_processed_limit,
    const size_t num_processed_ceil_limit);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel3D021Tiling(void *x, void *y,
                                                  const size_t a,
                                                  const size_t b,
                                                  const size_t c);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel4D0231(
    void *x, void *y, const size_t a, const size_t b, const size_t c,
    const size_t d, const size_t num_split0, const size_t num_split1,
    const size_t num_split2, const size_t num_split3, const bool split_h,
    const size_t num_processed0, const size_t num_processed1,
    const size_t num_processed2, const size_t num_processed3,
    const size_t num_processed_ceil0, const size_t num_processed_ceil1,
    const size_t num_processed_ceil2, const size_t num_processed_ceil3,
    const size_t num_processed_limit0, const size_t num_processed_limit1,
    const size_t num_processed_limit2, const size_t num_processed_limit3,
    const size_t num_processed_ceil_limit0,
    const size_t num_processed_ceil_limit1,
    const size_t num_processed_ceil_limit2,
    const size_t num_processed_ceil_limit3);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel3D021Loop(void *x, void *y,
                                                const size_t a, const size_t b,
                                                const size_t c,
                                                const bool split_h);
template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel3D210(
    void *x, void *y, const size_t a, const size_t b, const size_t c,
    const size_t num_split0, const size_t num_split1, const size_t num_split2,
    const bool split_h, const size_t num_processed0,
    const size_t num_processed1, const size_t num_processed2,
    const size_t num_processed_ceil0, const size_t num_processed_ceil1,
    const size_t num_processed_ceil2, const size_t num_processed_limit0,
    const size_t num_processed_limit1, const size_t num_processed_limit2,
    const size_t num_processed_ceil_limit0,
    const size_t num_processed_ceil_limit1,
    const size_t num_processed_ceil_limit2);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel3D210Small(void *x, void *y,
                                                 const size_t a, const size_t b,
                                                 const size_t c);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel3D102(void *x, void *y, const size_t a,
                                            const size_t b, const size_t c);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel3D102Tower(
    void *x, void *y, const size_t a, const size_t b, const size_t c,
    const size_t num_split0, const size_t num_split1, const size_t num_split2,
    const bool split_h, const size_t num_processed0,
    const size_t num_processed1, const size_t num_processed2,
    const size_t num_processed_ceil0, const size_t num_processed_ceil1,
    const size_t num_processed_ceil2, const size_t num_processed_limit0,
    const size_t num_processed_limit1, const size_t num_processed_limit2,
    const size_t num_processed_ceil_limit0,
    const size_t num_processed_ceil_limit1,
    const size_t num_processed_ceil_limit2);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel4D0213(void *input, void *output,
                                             const size_t n, const size_t h,
                                             const size_t w, const size_t c);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel4D0213Tower(
    void *input, void *output, const size_t n, const size_t h, const size_t w,
    const size_t c, const size_t split_num0, const size_t split_num1,
    const size_t split_num2, const size_t split_h, const size_t per_core_num0_0,
    const size_t per_core_num0_1, const size_t per_core_num0_2,
    const size_t per_core_num1_0, const size_t per_core_num1_1,
    const size_t per_core_num1_2, const size_t once_num0_0,
    const size_t once_num0_1, const size_t once_num0_2,
    const size_t once_num1_0, const size_t once_num1_1,
    const size_t once_num1_2);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel4D0213Small(void *input, void *output,
                                                  const size_t n,
                                                  const size_t h,
                                                  const size_t w,
                                                  const size_t c);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel4D0321(
    void *input, void *output, const size_t n, const size_t h, const size_t w,
    const size_t c, const size_t split_num0, const size_t split_num1,
    const size_t split_num2, const size_t split_h, const size_t per_core_num0_0,
    const size_t per_core_num0_1, const size_t per_core_num0_2,
    const size_t per_core_num1_0, const size_t per_core_num1_1,
    const size_t per_core_num1_2, const size_t once_num0_0,
    const size_t once_num0_1, const size_t once_num0_2,
    const size_t once_num1_0, const size_t once_num1_1,
    const size_t once_num1_2);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel4D0321Small(void *input, void *output,
                                                  const size_t n,
                                                  const size_t h,
                                                  const size_t w,
                                                  const size_t c);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel4D1032(void *input, void *output,
                                             const size_t n, const size_t h,
                                             const size_t w, const size_t c);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel4D1302(void *input, void *output,
                                             const size_t n, const size_t h,
                                             const size_t w, const size_t c);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel4D1320(void *input, void *output,
                                             const size_t n, const size_t h,
                                             const size_t w, const size_t c);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel4D2031(void *input, void *output,
                                             const size_t n, const size_t h,
                                             const size_t w, const size_t c);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel4D2103(void *input, void *output,
                                             const size_t n, const size_t h,
                                             const size_t w, const size_t c);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel4D2130(void *input, void *output,
                                             const size_t n, const size_t h,
                                             const size_t w, const size_t c);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel4D3021(void *input, void *output,
                                             const size_t n, const size_t h,
                                             const size_t w, const size_t c);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel4D3102(void *input, void *output,
                                             const size_t n, const size_t h,
                                             const size_t w, const size_t c);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernel4D3210(void *input, void *output,
                                             const size_t n, const size_t h,
                                             const size_t w, const size_t c);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernelCommonSmall(
    void *input, void *output, const size_t x0, const size_t x1,
    const size_t x2, const size_t x3, const size_t x4, const size_t x5,
    const size_t x6, const size_t x7, const size_t p0, const size_t p1,
    const size_t p2, const size_t p3, const size_t p4, const size_t p5,
    const size_t p6, const size_t p7, const size_t sum, const size_t dims,
    const size_t split_dim, const size_t split_dim_y);

template <typename T, size_t REPEAT>
__mlu_global__ void MLUTransposeKernelCommonBig(
    void *input, void *output, const size_t x0, const size_t x1,
    const size_t x2, const size_t x3, const size_t x4, const size_t x5,
    const size_t x6, const size_t x7, const size_t p0, const size_t p1,
    const size_t p2, const size_t p3, const size_t p4, const size_t p5,
    const size_t p6, const size_t p7, const size_t sum, const size_t dims,
    const size_t split_dim, const size_t split_dim_y);

#endif  // defined(__BANG__)

#endif  // KERNELS_TRANSPOSE_KERNEL_TRANSPOSE_H_
