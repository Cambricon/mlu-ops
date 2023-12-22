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
#include "kernels/tensor_stride_process/tensor_stride_process.h"
#include "kernels/fft/common/fft_basic_ops.h"
#include "kernels/fft/common/fft_common_kernels.h"
#include "kernels/debug.h"
#include "kernels/kernel.h"
#include "kernels/utils/common.h"

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

struct mluOpFFTStruct {
  int rank;            // rank of FFT
  int n[FFT_DIM_MAX];  // FFT lengths on each dimension
  mluOpDataType_t input_dtype;
  mluOpDataType_t output_dtype;
  mluOpDataType_t execution_dtype;
  int idim;                  // the dimension size of input tensor
  int inembed[FFT_DIM_MAX];  // Pointer of size rank that indicates the storage
                             // dimensions of the input data in memory.
  int inum;     // element num of input tensor
  int istride;  // distance between two successive input elements in the
                // innermost dimension
  int idist;    // distance between the first element of two consecutive signals
              // in a batch of the input data
  int odim;                  // the dimension size of output tensor
  int onembed[FFT_DIM_MAX];  // Pointer of size rank that indicates the storage
                             // dimensions of the output data in memory
  int onum;     // element num of output tensor
  int ostride;  // distance between two successive output elements in the
                // innermost dimension
  int odist;    // distance between the first element of two consecutive signals
              // in a batch of the output data
  int batch;  // batch size for this transform
  int L;      // n = L * 2^m, L size for this transform
  int m;      // n = L * 2^m, m size for this transform
  int s;      // The size that can be put down on NRAM: L * 2^s, only used by
          // Cooley-Tukey algorithm
  int L_sub;  // The size that can be put down on NRAM: L_sub * 2^m, only used
              // by  Stockham algorithm
  bool is_input_contiguous;
  bool is_output_contiguous;
  size_t reservespace_size;
  size_t workspace_size;
  FFTType fft_type;  // types of fft
  FFTStrategy fft_strategy;
  mluOpTensorDescriptor_t input_desc;
  mluOpTensorDescriptor_t output_desc;
  void *reservespace_addr;
  cnfftMatmulAddrs matmul_addrs;
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

mluOpStatus_t selectStrategy(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                             const std::string make_plan_api);

#if TARGET_MLU_ARCH != 520
__mlu_global__ void MLUKernelFFTCooleyTukey(void *matmul_re_mul_re_addr,
                                            void *matmul_re_mul_im_addr,
                                            void *matmul_im_mul_re_addr,
                                            void *matmul_im_mul_im_addr,
                                            void *internal_workspace_addr,
                                            void *output, int fft_flag,
                                            int direction, int n, int batch,
                                            int L,  // less than 4096
                                            int m, int s, int output_dtype);

__mlu_global__ void MLUKernelFFTStockham(void *matmul_re_mul_re_addr,
                                         void *output, int fft_flag,
                                         int direction, int n, int batch, int L,
                                         int m, int L_sub, int output_dtype,
                                         const float scale_factor);
#endif  // #if TARGET_MLU_ARCH != 520
#endif  // KERNELS_FFT_FFT_H_
