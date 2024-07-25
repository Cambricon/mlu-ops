/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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
#ifndef KERNELS_FFT_FFT_H_
#define KERNELS_FFT_FFT_H_

#include <string>
#include "core/context.h"
#include "core/logging.h"
#include "core/gen_case.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "core/tool.h"
#include "kernels/tensor_stride_process/tensor_stride_process_host.h"
#include "kernels/tensor_stride_process/tensor_stride_process_mlu.h"
#include "kernels/fft/common/fft_basic_ops.h"
#include "kernels/fft/common/fft_common_kernels.h"
#include "kernels/debug.h"
#include "kernels/kernel.h"

#ifndef FFT_DIM_MAX
#define FFT_DIM_MAX 3
#endif

#ifndef FFT_L_LIMIT
#define FFT_L_LIMIT 4096
#endif

#ifndef COMPLEX
#define COMPLEX 2
#endif

#ifndef FFT_HALF
#define FFT_HALF(x) ((x) / 2 + 1)
#endif

#ifndef FFT_MAXFACTORS
#define FFT_MAXFACTORS 278  // max length of factors[] in plan
#endif

#ifndef MAX_DFT_MATRIX_NR
#define MAX_DFT_MATRIX_NR 8
#endif

#ifndef DFT_TABLE_SIZE
#define DFT_TABLE_SIZE \
  (32 * 32 * (MAX_DFT_MATRIX_NR + 1) * 8 * 2 + (MAX_DFT_MATRIX_NR + 1) * 8)
// radix-16, 21-stages, double, complex
// + addrs size
#endif

// transform directions
#define FFT_FORWARD (0)
#define FFT_BACKWARD (+1)

#define FFT_PI (3.1415926535897932384626433832795)

struct dft_table_entry {
  int radix;
  int offset;
};

typedef enum {
  FFT_IFFT = 0,
  RFFT = 1,
  IRFFT = 2,
} FFTFlag;

typedef enum {
  CNFFT_FUNC_MATMUL =
      0,  // directly matmul strategy, specified for multiple batches of
          // transform, and output size is relatively small. Its structure is
          // suitable tensor computing-oriented machines.
  CNFFT_FUNC_STOCKHAM =
      1,  // an iterative FFT algorithm for n = r^l. It is self-sorting (does
          // not have a digit reversal permutation). Its structure is suitable
          // for long vector computing machines.
  CNFFT_FUNC_FOUR_STEP =
      2,  // a recursive FFT algorithm for n = km. It is built from two stages
          // of vector FFTs, the twiddle diagonal and a transposition. Its
          // structure is suitable for vector computers.
  CNFFT_FUNC_BLUESTEIN =
      3,  // a general-purpose algorithm (i.e., n is a prime number).

  CNFFT_FUNC_COOLEY_TUKEY =
      4,  // a recursive FFT algorithm for n = 2^m * L; It saves the space
          // occupied by the w matrix. And, compared to DFT, the time
          // complexity is reduced from o(n^2) to o(n * logn)
  CNFFT_FUNC_MANY_DIST1_2D =
      5,  // directly matmul strategy for [n0, n1, batch] pattern.
  CNFFT_FUNC_TWO_LEVEL_STOCKHAM = 6,  // an iterative FFT algorithm for n = r^l.
} FFTStrategy;

typedef enum {
  CNFFT_HALF2COMPLEX_HALF = 0,
  CNFFT_COMPLEX_HALF2HALF = 1,
  CNFFT_COMPLEX_HALF2COMPLEX_HALF = 2,
  CNFFT_FLOAT2COMPLEX_FLOAT = 3,
  CNFFT_COMPLEX_FLOAT2FLOAT = 4,
  CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT = 5,
} FFTType;

// struct for CNFFT_FUNC_MATMUL strategy.
struct cnfftMatmulAddrs {
  /* addrs set in the preprocess-stage */
  void *dft_matrix_addr;
  void *dft_re_matrix_addr;
  void *dft_im_matrix_addr;
  void *ifft_dft_matrix_addr;
  void *ifft_dft_re_matrix_addr;
  void *ifft_dft_im_matrix_addr;
  void *dft_pos_addr;
  void *dft_scale_addr;
  size_t dft_quantize_workspace_size;
  void *dft_quantize_workspace_addr;
  /* addrs set in the runtime stage */
  void *input_contiguous_addr;
  void *input_pad_addr;
  void *input_transed_addr;
  void *input_reversed_addr;
  void *input_merged_addr;
  void *input_re_addr;
  void *input_im_addr;
  void *input_pos_addr;
  void *input_scale_addr;
  void *matmul_re_mul_re_addr;
  void *matmul_re_mul_im_addr;
  void *matmul_im_mul_re_addr;
  void *matmul_im_mul_im_addr;
  void *output_re_addr;
  void *output_im_addr;
  void *output_contiguous_addr;
  void *internal_workspace_addr;
  size_t internal_workspace_size;
};

// struct for CNFFT_FUNC_MATMUL strategy.
struct cnfftButterflyAddrs {
  /* addrs set in the preprocess-stage */
  void *input;
  void *output;
  void *buffer;
  void *twiddles;
  void *twiddles_2d;
  void *twiddles_end;
  void *twiddles_2d_end;
  void *twiddles_inv;
  void *twiddles_inv_2d;
  void *twiddles_inv_end;
  void *twiddles_inv_2d_end;
  void *buffer_buf;
  void *buffer_in;
  void *buffer_out;
  void *dft_matrix;
  void *dft_matrix_2d;
  void *idft_matrix;
  void *idft_matrix_2d;
  int *factors;
  int *factors_2d;
};
struct mluOpFFTStruct {
  int rank;            // rank of FFT
  int n[FFT_DIM_MAX];  // FFT lengths on each dimension
  mluOpDataType_t input_dtype;
  mluOpDataType_t output_dtype;
  mluOpDataType_t execution_dtype;
  int idim;                  // the dimension size of input tensor
  int inembed[FFT_DIM_MAX];  // Pointer of size rank that indicates the storage
                             // dimensions of the input data in memory.
  int inum;                  // element num of input tensor
  int istride;  // distance between two successive input elements in the
                // innermost dimension
  int idist;    // distance between the first element of two consecutive signals
                // in a batch of the input data
  int odim;     // the dimension size of output tensor
  int onembed[FFT_DIM_MAX];  // Pointer of size rank that indicates the storage
                             // dimensions of the output data in memory
  int onum;                  // element num of output tensor
  int ostride;  // distance between two successive output elements in the
                // innermost dimension
  int odist;    // distance between the first element of two consecutive signals
                // in a batch of the output data
  int batch;    // batch size for this transform
  int L;        // n = L * 2^m, L size for this transform
  int m;        // n = L * 2^m, m size for this transform
  int s;        // The size that can be put down on NRAM: L * 2^s, only used by
                // Cooley-Tukey algorithm
  int L_sub;    // The size that can be put down on NRAM: L_sub * 2^m, only used
                // by  Stockham algorithm
  int prime;    // wether fft1d'size contains a prime number > 64
  bool is_input_contiguous;
  bool is_output_contiguous;
  bool is_batch_contiguous;
  size_t reservespace_size;
  size_t workspace_size;
  FFTType fft_type;  // types of fft
  FFTStrategy fft_strategy;
  mluOpTensorDescriptor_t input_desc;
  mluOpTensorDescriptor_t output_desc;
  void *reservespace_addr;
  cnfftMatmulAddrs matmul_addrs;
  int *factors;
  int *factors_2d;
  void *twiddles;
  void *twiddles_2d;
  void *twiddles_end;
  void *twiddles_2d_end;
  void *twiddles_inv;
  void *twiddles_inv_2d;
  void *twiddles_inv_end;
  void *twiddles_inv_2d_end;
  void *dft_matrix;
  void *dft_matrix_2d;
  void *idft_matrix;
  void *idft_matrix_2d;
  cnfftButterflyAddrs mlu_addrs;
};

struct ParamNode {
  int subgraph_size;
  int L_bytes;
  int L_align;
  int L_align_bytes;
  int op_size;
  int op_size_align;
  int op_size_align_via_L;
  int op_size_bytes;
  int op_size_bytes_align;
  int op_size_align_via_L_trans;
  int op_group_num_1_batch;
  int op_group_num_x_batch;
  int remain_layer_num;
};

template <class DT>
struct AddrNode {
  // GDRAM Addr Info:
  DT *wspace_r;
  DT *wspace_i;

  // NRAM Addr Info:
  // input addr:
  DT *y_in_r;
  DT *z_in_r;
  DT *y_in_i;
  DT *z_in_i;
  // output addr:
  DT *x_out1_r;
  DT *x_out2_r;
  DT *x_out1_i;
  DT *x_out2_i;
  // w_matrix addr:
  DT *w_r;
  DT *w_i;
  // temp addr reserved for vector generation w_matrix.
  DT *w_tmp1;
  DT *w_tmp2;
  DT *w_tmp3;
  // temp addr reserved for subgraph internal merge calculation, using the same
  // addr with w_tmp*.
  DT *wz_rr;
  DT *wz_ri;
  DT *wz_ir;
  DT *wz_ii;
  DT *wz_r;
  DT *wz_i;
};

mluOpStatus_t selectFFTStrategy(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                                const std::string make_plan_api);

mluOpStatus_t MLUOP_WIN_API kernelFFTCooleyTukey(cnrtDim3_t k_dim,
                                                 cnrtFunctionType_t k_type,
                                                 cnrtQueue_t queue,
                                                 mluOpFFTPlan_t fft_plan,
                                                 int direction, FFTFlag flag);

mluOpStatus_t MLUOP_WIN_API
kernelFFTStockham(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                  cnrtQueue_t queue, mluOpFFTPlan_t fft_plan, int direction,
                  const float scale_factor, FFTFlag flag);

// Sets the maximum parallel number for the FFT plan, factoring in the given
// buffer, stage, large radix, and row-major flag.
mluOpStatus_t MLUOP_WIN_API setMaxParallelNum(mluOpFFTPlan_t fft_plan,
                                              int *facbuf, int stage,
                                              const int large_radix,
                                              const int is_row_major);

// Factors the given FFT plan into two steps based on the size, factoring
// buffer, row-major flag, and FFT type.
mluOpStatus_t MLUOP_WIN_API fftTwoStepFactor(mluOpFFTPlan_t fft_plan,
                                             const int _n, int *facbuf,
                                             const int is_row_major,
                                             const int fft_type);

// Executes the 1D Butterfly FFT kernel for rows with the specified dimensions,
// function type, queue, FFT plan, direction, and flag.
mluOpStatus_t MLUOP_WIN_API kernelFFT1dButterflyRow(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpFFTPlan_t fft_plan, int direction, FFTFlag flag);

// Executes the 1D Butterfly FFT kernel for rows, converting complex to real,
// with the specified dimensions, function type, queue, FFT plan, and flag.
mluOpStatus_t MLUOP_WIN_API kernelFFT1dButterflyRowC2R(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpFFTPlan_t fft_plan, FFTFlag flag);

// Executes the 1D Butterfly FFT kernel for columns with the specified
// dimensions, function type, queue, FFT plan, direction, and flag.
mluOpStatus_t MLUOP_WIN_API kernelFFT1dButterflyColumn(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpFFTPlan_t fft_plan, int direction, FFTFlag flag);

// Executes the 2D Butterfly FFT kernel for columns with the specified
// dimensions, function type, queue, FFT plan, direction, and flag.
mluOpStatus_t MLUOP_WIN_API kernelFFT2dButterflyColumn(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpFFTPlan_t fft_plan, int direction, FFTFlag flag);

// Executes the inverse 2D Butterfly FFT kernel for columns with the specified
// dimensions, function type, queue, FFT plan, and flag.
mluOpStatus_t MLUOP_WIN_API kernelIRFFT2dButterflyColumn(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpFFTPlan_t fft_plan, FFTFlag flag);

// Executes the 2D Butterfly FFT kernel for rows with the specified dimensions,
// function type, queue, FFT plan, direction, and flag.
mluOpStatus_t MLUOP_WIN_API kernelFFT2dButterflyRow(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpFFTPlan_t fft_plan, int direction, FFTFlag flag);

// Executes the 1D Butterfly FFT kernel for real to complex conversion with the
// specified dimensions, function type, queue, FFT plan, and flag.
mluOpStatus_t MLUOP_WIN_API kernelFFT1dButterflyR2C(cnrtDim3_t k_dim,
                                                    cnrtFunctionType_t k_type,
                                                    cnrtQueue_t queue,
                                                    mluOpFFTPlan_t fft_plan,
                                                    FFTFlag flag);

// Executes the 2D Butterfly FFT kernel for real input, inverse operation,
// column-wise, with the specified dimensions, function type, queue, FFT plan,
// and flag.
mluOpStatus_t MLUOP_WIN_API kernelRFFT2dButterflyColumn(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpFFTPlan_t fft_plan, FFTFlag flag);

// Executes the 2D Butterfly FFT kernel for real input, inverse operation,
// row-wise, with the specified dimensions, function type, queue, FFT plan, and
// flag.
mluOpStatus_t MLUOP_WIN_API kernelRFFT2dButterflyRow(cnrtDim3_t k_dim,
                                                     cnrtFunctionType_t k_type,
                                                     cnrtQueue_t queue,
                                                     mluOpFFTPlan_t fft_plan,
                                                     FFTFlag flag);

// Executes the inverse 2D Butterfly FFT kernel for rows with the specified
// dimensions, function type, queue, FFT plan, and flag.
mluOpStatus_t MLUOP_WIN_API kernelIRFFT2dButterflyRow(cnrtDim3_t k_dim,
                                                      cnrtFunctionType_t k_type,
                                                      cnrtQueue_t queue,
                                                      mluOpFFTPlan_t fft_plan,
                                                      FFTFlag flag);

// Executes the complex-to-complex FFT/DFT matrix kernel with the specified
// dimensions, function type, queue, FFT plan, input real data type, and size.
mluOpStatus_t MLUOP_WIN_API kernelC2CFFTDFTMatrix(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpFFTPlan_t fft_plan, mluOpDataType_t in_r_dtype, int n);

// Searches for a large radix in the FFT plan, updating large radix and
// factoring buffer based on the stage ID, size, and row-major flag.
mluOpStatus_t MLUOP_WIN_API searchLargeRadix(mluOpFFTPlan_t fft_plan,
                                             int &large_radix, int *facbuf,
                                             int large_stage_id, int _n,
                                             const int is_row_major);

// Calculates the lower bound of the parallel number for the FFT plan, factoring
// in the stage, buffer, and row-major flag, updating parallel_num_lb.
mluOpStatus_t MLUOP_WIN_API calParallelNumLowBound(mluOpFFTPlan_t fft_plan,
                                                   int *facbuf, int stage,
                                                   int &parallel_num_lb,
                                                   const int is_row_major);

// Executes the kernel for conjugate merge FFT operation with the specified
// dimensions, function type, queue, output and input buffers, length, and data
// type.
mluOpStatus_t MLUOP_WIN_API kernelFFTConjMerge(cnrtDim3_t k_dim,
                                               cnrtFunctionType_t k_type,
                                               cnrtQueue_t queue, void *output,
                                               void *input, int len, int dtype);

// Executes the kernel for batched conjugate merge FFT operation with the
// specified dimensions, function type, queue, output and input buffers, length,
// batch size, and data type.
mluOpStatus_t MLUOP_WIN_API kernelFFTBatchConjMerge(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    void *output, void *input, int len, int batch, int dtype);

// Executes the kernel for batched conjugate merge FFT operation from real to
// complex with the specified dimensions, function type, queue, output and input
// buffers, length, batch size, and data type.
mluOpStatus_t MLUOP_WIN_API kernelFFTBatchConjMergeR2C(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    void *output, void *input, int len, int batch, int dtype);

// Executes the kernel for batched conjugate merge FFT operation from complex to
// real with the specified dimensions, function type, queue, output and input
// buffers, length, batch size, and data type.
mluOpStatus_t MLUOP_WIN_API kernelFFTBatchConjMergeC2R(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    void *output, void *input, int len, int batch, int dtype);

// Computes the 2D FFT followed by matrix multiplication for rows with the
// specified handle, FFT plan, scaling factor, and direction.
mluOpStatus_t computeFFT2dMatMulRow(mluOpHandle_t handle,
                                    mluOpFFTPlan_t fft_plan,
                                    const float scale_factor, int direction);

// Computes the 2D FFT followed by matrix multiplication for columns with the
// specified handle, FFT plan, scaling factor, and direction.
mluOpStatus_t computeFFT2dMatMulColumn(mluOpHandle_t handle,
                                       mluOpFFTPlan_t fft_plan,
                                       const float scale_factor, int direction);

// Computes the 2D FFT followed by matrix multiplication for rows from real to
// complex with the specified handle, FFT plan, and scaling factor.
mluOpStatus_t computeFFT2dMatMulRowR2C(mluOpHandle_t handle,
                                       mluOpFFTPlan_t fft_plan,
                                       const float scale_factor);

// Computes the 2D FFT followed by matrix multiplication for rows from complex
// to real with the specified handle, FFT plan, and scaling factor.
mluOpStatus_t computeFFT2dMatMulRowC2R(mluOpHandle_t handle,
                                       mluOpFFTPlan_t fft_plan,
                                       const float scale_factor);

// Computes the 2D FFT followed by matrix multiplication for columns from real
// to complex with the specified handle, FFT plan, and scaling factor.
mluOpStatus_t computeFFT2dMatMulColumnR2C(mluOpHandle_t handle,
                                          mluOpFFTPlan_t fft_plan,
                                          const float scale_factor);

// Computes the 2D FFT followed by matrix multiplication for columns from
// complex to real with the specified handle, FFT plan, and scaling factor.
mluOpStatus_t computeFFT2dMatMulColumnC2R(mluOpHandle_t handle,
                                          mluOpFFTPlan_t fft_plan,
                                          const float scale_factor);
#endif  // KERNELS_FFT_FFT_H_
