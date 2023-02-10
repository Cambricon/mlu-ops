/*******************************************************************************
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
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************/

#include <algorithm>
#include <vector>
#include "carafe.h"
#include "kernels/tensor_stride_process/tensor_stride_process.h"
#include "kernels/tensor_stride_process/tensor_stride_process_mlu.h"

// 1.creat set destroy
mluOpStatus_t MLUOP_WIN_API
mluOpCreateCarafeDescriptor(mluOpCarafeDescriptor_t *carafe_desc) {
  PARAM_CHECK("[mluOpCreateCarafeDescriptor]", carafe_desc != NULL);
  *carafe_desc = new (std::nothrow) mluOpCarafeStruct();
  if (carafe_desc == NULL) {
    return MLUOP_STATUS_NOT_INITIALIZED;
  }
  return MLUOP_STATUS_SUCCESS;
}
int MLUOP_WIN_API test() {
  int a = 120;
  return a;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetCarafeDescriptor(
    mluOpCarafeDescriptor_t carafe_desc, const int dimNb, const int kernel_size,
    const int group_size, const int scale_factor) {
  PARAM_CHECK("[mluOpSetCarafeDescriptor]", carafe_desc != NULL);
  PARAM_CHECK("[mluOpSetCarafeDescriptor]",
              kernel_size >= 1 && (kernel_size - 1) % 2 == 0);
  PARAM_CHECK("[mluOpSetCarafeDescriptor]", group_size >= 1);
  PARAM_CHECK("[mluOpSetCarafeDescriptor]", scale_factor >= 1);
  PARAM_CHECK("[mluOpSetCarafeDescriptor]", dimNb == 4);

  carafe_desc->dimNb = dimNb;
  carafe_desc->kernel_size = kernel_size;
  carafe_desc->group_size = group_size;
  carafe_desc->scale_factor = scale_factor;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDestroyCarafeDescriptor(mluOpCarafeDescriptor_t carafe_desc) {
  PARAM_CHECK("[mluOpDestroyCarafeDescriptor]", carafe_desc != NULL);
  delete carafe_desc;
  return MLUOP_STATUS_SUCCESS;
}

void getNramUsage(int block_dimH, int block_dimW, int block_dimG,
                  int block_dimC, int grid_dimH, int grid_dimW, int grid_dimG,
                  int grid_dimC, int kernel_size, int scale_factor,
                  int align_size_NRAM, int align_size_NFU, int *nram_usage) {
  /*
   * Get total NRAM usage.
   */
  int kernel_size_sq = kernel_size * kernel_size;
  int block_c_stride = CEIL_ALIGN(block_dimC, align_size_NRAM);
  int sum_array_size_bang_add =
      CEIL_ALIGN(block_c_stride * block_dimG, align_size_NFU);

  // input_nram[block_H+K-1, block_W+K-1, block_dimC * block_dimG]
  int input_nram_stride_w = block_c_stride * block_dimG;
  int input_nram_stride_h =
      (block_dimW + kernel_size - 1) * input_nram_stride_w;
  int input_nram_size = (block_dimH + kernel_size - 1) * input_nram_stride_h;

  // mask_nram[sigma*block_H, sigma*block_W, K*K * block_dimG]
  int mask_nram_stride_w = kernel_size_sq * block_dimG;
  int mask_nram_stride_h = (scale_factor * block_dimW) * mask_nram_stride_w;
  int mask_nram_size = CEIL_ALIGN(
      (scale_factor * block_dimH) * mask_nram_stride_h, align_size_NRAM);

  // output_nram[sigma*D_H, sigma*D_W, block_dimC * block_dimG]
  int output_nram_stride_w = sum_array_size_bang_add;
  int output_nram_stride_h = (scale_factor * block_dimW) * output_nram_stride_w;
  int output_nram_size = (scale_factor * block_dimH) * output_nram_stride_h;

  // sum_array[block_dimC * block_dimG]
  int sum_array_size_bang_mul_const =
      (block_dimG - 1) * block_c_stride +
      CEIL_ALIGN(block_c_stride, align_size_NFU);
  int sum_array_size =
      std::max(sum_array_size_bang_add, sum_array_size_bang_mul_const);

  *nram_usage =
      input_nram_size + mask_nram_size + output_nram_size + sum_array_size;
}

mluOpStatus_t genPolicy(mluOpHandle_t handle,
                        const mluOpCarafeDescriptor_t carafe_desc,
                        const mluOpTensorDescriptor_t input_desc,
                        cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type,
                        int *block_dimH, int *block_dimW, int *block_dimG,
                        int *block_dimC, int *grid_dimH, int *grid_dimW,
                        int *grid_dimG, int *grid_dimC, int *job_num) {
  /*
   * policy function for carafe_forward.
   */
  VLOG(5) << CARAFE_FORWARD_API << "===== Start generating policy. =====";

  int union_number = mluop::runtime::getClusterLimitCapability(handle);
  int core_dim = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  int core_number = union_number * core_dim;

  const int input_dimN = mluOpGetTensordimN(input_desc);
  const int input_dimH = mluOpGetTensordimH(input_desc);
  const int input_dimW = mluOpGetTensordimW(input_desc);
  const int input_dimC = mluOpGetTensordimC(input_desc);

  const int kernel_size = carafe_desc->kernel_size;
  const int group_size = carafe_desc->group_size;
  const int scale_factor = carafe_desc->scale_factor;

  const int dtype_size = mluop::getSizeOfDataType(input_desc->dtype);

  const int align_size_NRAM = WRAM_ALIGN_SIZE / dtype_size;
  const int align_size_NFU = NFU_ALIGN_SIZE / dtype_size;

  // Determine maximum NRAM size in the number of <dtype>)
  int max_nram_size = int(handle->nram_size / dtype_size);

  // set block dimension on (H,W,C) within each group
  int channels_per_group = input_dimC / group_size;

  // initial values for block_h/w/g/c, grid_h/w/g/c
  int block_h = input_dimH;
  int block_w = input_dimW;
  int block_g = group_size;
  int block_c = channels_per_group;  // grid_c is number of block_c inside one
                                     // channel group.
  int grid_h = 1;
  int grid_w = 1;
  int grid_g = 1;
  int grid_c = 1;

  // decrease the block size to fit in the NRAM.
  int nram_usage = 0;
  while (true) {
    getNramUsage(block_h, block_w, block_g, block_c, grid_h, grid_w, grid_g,
                 grid_c, kernel_size, scale_factor, align_size_NRAM,
                 align_size_NFU, &nram_usage);

    if (nram_usage > max_nram_size) {
      // decrease block_h and block_w evenly to keep a block on h & w close to a
      // square shape.
      if (block_h > 1 && block_h >= block_w) {
        grid_h += 1;
        block_h = (input_dimH + grid_h - 1) / (grid_h);
      } else if (block_w > 1 && block_w > block_h) {
        grid_w += 1;
        block_w = (input_dimW + grid_w - 1) / (grid_w);
      } else if (block_g > 1) {
        grid_g += 1;
        block_g = (group_size + grid_g - 1) / (grid_g);
        // reset grid_h,w to maximize NRAM usage
        grid_h = 1;
        block_h = input_dimH;
        grid_w = 1;
        block_w = input_dimW;
      } else if (block_c > 1) {
        // decrease block_c in the last since c is the continuous dim (input
        // layout is NHWC) and large c can improves IO efficiency.
        grid_c += 1;
        block_c = (channels_per_group + grid_c - 1) / (grid_c);
        // reset grid_h,w
        grid_h = 1;
        block_h = input_dimH;
        grid_w = 1;
        block_w = input_dimW;
      } else {
        // block_h/w/g/c all have the value of one now, cannot decrease the
        // block size anymore!
        break;
      }
    } else {
      break;
    }
  }
  // maximum allowed scale_factor and kernel_size
  //   (scale_factor^2 + 1) * 128B/dtype_size_in_bytes
  // + kernel_size^2 * 64B/dtype_size_in_bytes
  // + CEIL_ALIGN((scale_factor^2 * kernel_size^2), 64B/dtype_size_in_bytes)
  // <= MAX_NRAM_IN_BYTES/dtype_size_in_bytes
  PARAM_CHECK_V2(
      CARAFE_FORWARD_API, nram_usage <= max_nram_size,
      "kernel_size and/or scale_factor are too large! Decrease the value."
          << " Maximum nram size (Nb. of dtype_size) = " << max_nram_size
          << ", but NRAM usage (Nb. of dtype_size) = " << nram_usage);

  // assert block_c == channels_per_group when block_g > 1
  if (block_g > 1) {
    PARAM_CHECK_V2(CARAFE_FORWARD_API,
                   grid_c == 1 && block_c == channels_per_group,
                   "block_g and block_c have wrong values!");
  }

  // set block dimensions
  *block_dimH = block_h;
  *block_dimW = block_w;
  *block_dimG = block_g;
  *block_dimC = block_c;
  *grid_dimH = grid_h;
  *grid_dimW = grid_w;
  *grid_dimG = grid_g;
  *grid_dimC = grid_c;

  *job_num =
      input_dimN * (*grid_dimH) * (*grid_dimW) * (*grid_dimG) * (*grid_dimC);

  VLOG(5) << CARAFE_FORWARD_API << "job_num = " << *job_num;
  VLOG(5) << CARAFE_FORWARD_API
          << "kernel_size,group_size,scale_factor = " << kernel_size << ","
          << group_size << "," << scale_factor;
  VLOG(5) << CARAFE_FORWARD_API << "input_dim(N,H,W,G,Cg) = " << input_dimN
          << "," << input_dimH << "," << input_dimW << "," << group_size << ","
          << channels_per_group;
  VLOG(5) << CARAFE_FORWARD_API << "block_dim(H,W,G,Cg) = " << *block_dimH
          << "," << *block_dimW << "," << *block_dimG << "," << *block_dimC;
  VLOG(5) << CARAFE_FORWARD_API << "grid_dim(H,W,G,Cg) = " << *grid_dimH << ","
          << *grid_dimW << "," << *grid_dimG << "," << *grid_dimC;
  VLOG(5) << CARAFE_FORWARD_API << "dtype_size = " << dtype_size;
  VLOG(5) << CARAFE_FORWARD_API
          << "Maximum nram size (Nb. of dtype_size) = " << max_nram_size;
  VLOG(5) << CARAFE_FORWARD_API
          << "NRAM usage (Nb. of dtype_size) = " << nram_usage;

  // determine task type and dims
  *k_type = CNRT_FUNC_TYPE_BLOCK;
  k_dim->x = core_dim;
  k_dim->y = union_number;
  k_dim->z = 1;
  if (*job_num < core_number) {
    k_dim->x = *job_num;
    k_dim->y = 1;
  }

  VLOG(5) << CARAFE_FORWARD_API << "===== End of generating policy. =====";

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t CarafeForwardParamCheck(
    mluOpHandle_t handle, const mluOpCarafeDescriptor_t carafe_desc,
    const mluOpTensorDescriptor_t input_desc, const void *input,
    const mluOpTensorDescriptor_t mask_desc, const void *mask,
    const mluOpTensorDescriptor_t output_desc, const void *output,
    bool *return_directly) {
  VLOG(5) << CARAFE_FORWARD_API << "===== Start checking parameters. =====";
  *return_directly = false;
  /*
   * descriptor param check
   */
  VLOG(5) << CARAFE_FORWARD_API << "Check for null descriptor.";
  PARAM_CHECK(CARAFE_FORWARD_API, handle != NULL);
  PARAM_CHECK(CARAFE_FORWARD_API, carafe_desc != NULL);
  PARAM_CHECK(CARAFE_FORWARD_API, input_desc != NULL);
  PARAM_CHECK(CARAFE_FORWARD_API, mask_desc != NULL);
  PARAM_CHECK(CARAFE_FORWARD_API, output_desc != NULL);
  /*
   * Check CarafeDescriptor
   */
  VLOG(5) << CARAFE_FORWARD_API << "Check carafe_descriptor.";
  PARAM_CHECK_V2(CARAFE_FORWARD_API, carafe_desc->dimNb == 4,
                 "dimNb should be 4, but the input value is "
                     << carafe_desc->dimNb << ".");
  PARAM_CHECK_V2(
      CARAFE_FORWARD_API,
      carafe_desc->kernel_size >= 1 && (carafe_desc->kernel_size - 1) % 2 == 0,
      "kernel_size should be an odd number, i.e., 2*k+1 (k>=0), "
      "but the input value is "
          << carafe_desc->kernel_size << ".");
  PARAM_CHECK_V2(
      CARAFE_FORWARD_API, carafe_desc->group_size >= 1,
      "group_size should be a positive integer, but the input value is "
          << carafe_desc->group_size << ".");
  PARAM_CHECK_V2(
      CARAFE_FORWARD_API, carafe_desc->scale_factor >= 1,
      "scale_factor should be a positive integer, but the input value is "
          << carafe_desc->scale_factor << ".");
  // kernel_size <= 45
  if (carafe_desc->kernel_size > 45) {
    LOG(ERROR) << CARAFE_FORWARD_API
               << "kernel_size > 45 is not supported! "
                  "The input is "
               << carafe_desc->kernel_size << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  // scale_factor <= 5
  if (carafe_desc->scale_factor > 5) {
    LOG(ERROR) << CARAFE_FORWARD_API
               << "scale_factor > 5 is not supported! "
                  "The input is "
               << carafe_desc->scale_factor << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  /*
   * dim check
   */
  VLOG(5) << CARAFE_FORWARD_API << "Check dimNb.";
  PARAM_CHECK_EQ(CARAFE_FORWARD_API, input_desc->dim, carafe_desc->dimNb);
  PARAM_CHECK_EQ(CARAFE_FORWARD_API, mask_desc->dim, carafe_desc->dimNb);
  PARAM_CHECK_EQ(CARAFE_FORWARD_API, output_desc->dim, carafe_desc->dimNb);
  /*
   * layout check
   */
  VLOG(5) << CARAFE_FORWARD_API << "Check tensor layout.";
  PARAM_CHECK_EQ(CARAFE_FORWARD_API, input_desc->layout, MLUOP_LAYOUT_NHWC);
  PARAM_CHECK_EQ(CARAFE_FORWARD_API, mask_desc->layout, MLUOP_LAYOUT_NHWC);
  PARAM_CHECK_EQ(CARAFE_FORWARD_API, output_desc->layout, MLUOP_LAYOUT_NHWC);
  /*
   * tensor contiguousness check
   */
  VLOG(5) << CARAFE_FORWARD_API << "Check data contiguousness.";
  PARAM_CHECK(CARAFE_FORWARD_API,
              !ifNeedTensorStrideProcess(input_desc),
              "The input tensor is not contiguous!");
  PARAM_CHECK(CARAFE_FORWARD_API,
              !ifNeedTensorStrideProcess(mask_desc),
              "The mask tensor is not contiguous!");
  PARAM_CHECK(CARAFE_FORWARD_API,
              !ifNeedTensorStrideProcess(output_desc),
              "The output tensor is not contiguous!");
  /*
   * off-chip data type check
   */
  VLOG(5) << CARAFE_FORWARD_API << "Check off-chip data type.";
  PARAM_CHECK_V2(
      CARAFE_FORWARD_API,
      (input_desc->dtype == MLUOP_DTYPE_HALF) ||
          (input_desc->dtype == MLUOP_DTYPE_FLOAT),
      "only half and float are supported, but the data type of input tensor is "
          << mluop::getNameOfDataType(input_desc->dtype) << ".");
  PARAM_CHECK_V2(CARAFE_FORWARD_API,
                 (input_desc->dtype == mask_desc->dtype) &&
                     (input_desc->dtype == output_desc->dtype),
                 "The input, mask and output tensors should have the same data "
                 "type, but the data types are: "
                     << mluop::getNameOfDataType(input_desc->dtype) << ", "
                     << mluop::getNameOfDataType(mask_desc->dtype) << ", "
                     << mluop::getNameOfDataType(output_desc->dtype) << ".");
  /*
   * shape param check
   *
   *    input[N,H,W,C]
   *    mask[N,\sigma*H,\sigma*W,G*k^2]
   *    output[N,\sigma*H,\sigma*W,C]
   */
  // check batch size: N
  VLOG(5) << CARAFE_FORWARD_API << "Check batch size.";
  int input_dimN = mluOpGetTensordimN(input_desc);
  int mask_dimN = mluOpGetTensordimN(mask_desc);
  int output_dimN = mluOpGetTensordimN(output_desc);
  PARAM_CHECK_V2(
      CARAFE_FORWARD_API,
      (input_dimN == mask_dimN) && (input_dimN == output_dimN),
      "The input, mask and output tensors should the same batch size, "
      "but the values are: "
          << input_dimN << "," << mask_dimN << "," << output_dimN << ".");
  // check H, W
  VLOG(5) << CARAFE_FORWARD_API << "Check dims H,W.";
  int input_dimH = mluOpGetTensordimH(input_desc);
  int input_dimW = mluOpGetTensordimW(input_desc);
  int mask_dimH = mluOpGetTensordimH(mask_desc);
  int mask_dimW = mluOpGetTensordimW(mask_desc);
  int output_dimH = mluOpGetTensordimH(output_desc);
  int output_dimW = mluOpGetTensordimW(output_desc);
  int scale_factor = carafe_desc->scale_factor;

  PARAM_CHECK_V2(
      CARAFE_FORWARD_API, mask_dimH == scale_factor * input_dimH,
      "mask_height should equal scale_factor*input_height, but the values are: "
          << mask_dimH << ", " << scale_factor << "*" << input_dimH << "="
          << scale_factor * input_dimH);
  PARAM_CHECK_V2(
      CARAFE_FORWARD_API, mask_dimW == scale_factor * input_dimW,
      "mask_width should equal scale_factor*input_width, but the values are: "
          << mask_dimW << ", " << scale_factor << "*" << input_dimW << "="
          << scale_factor * input_dimW);
  PARAM_CHECK_V2(CARAFE_FORWARD_API, output_dimH == mask_dimH,
                 "output_height should equal mask_height, but the values are: "
                     << output_dimH << ", " << mask_dimH << ".");
  PARAM_CHECK_V2(CARAFE_FORWARD_API, output_dimW == mask_dimW,
                 "output_width should equal mask_width, but the values are: "
                     << output_dimW << ", " << mask_dimW << ".");
  /*
   * channel check
   */
  VLOG(5) << CARAFE_FORWARD_API << "Check channels.";
  int input_dimC = mluOpGetTensordimC(input_desc);
  int mask_dimC = mluOpGetTensordimC(mask_desc);
  int output_dimC = mluOpGetTensordimC(output_desc);
  int group_size = carafe_desc->group_size;
  int kernel_size = carafe_desc->kernel_size;
  PARAM_CHECK_V2(
      CARAFE_FORWARD_API, input_dimC % group_size == 0,
      "Channel number of input tensor should be multiples of group_size, "
      "but the values are: input_dimC % group_size = "
          << input_dimC << " % " << group_size << " = "
          << input_dimC % group_size);
  PARAM_CHECK_V2(
      CARAFE_FORWARD_API, input_dimC == output_dimC,
      "The input and output tensors should have the same channel number, "
      "but the values are: "
          << input_dimC << ", " << output_dimC << ".");
  PARAM_CHECK_V2(
      CARAFE_FORWARD_API, mask_dimC == group_size * kernel_size * kernel_size,
      "Channel number of mask tensor should equal group_size*kernel_size^2, "
      "but the values are: "
          << mask_dimC << ", " << group_size << "*" << kernel_size
          << "^2=" << group_size * kernel_size * kernel_size);
  /*
   * Check 0 element
   */
  VLOG(5) << CARAFE_FORWARD_API << "Check for empty tensors.";
  size_t input_num = mluOpGetTensorElementNum(input_desc);
  size_t mask_num = mluOpGetTensorElementNum(mask_desc);
  size_t output_num = mluOpGetTensorElementNum(output_desc);
  if (input_num == 0 || mask_num == 0 || output_num == 0) {
    *return_directly = true;
    VLOG(5) << CARAFE_FORWARD_API << "empty tensor detected.";
    return MLUOP_STATUS_SUCCESS;
  }
  /*
   * null pointer check
   */
  VLOG(5) << CARAFE_FORWARD_API << "Check null data pointer.";
  PARAM_CHECK(CARAFE_FORWARD_API, input != NULL);
  PARAM_CHECK(CARAFE_FORWARD_API, mask != NULL);
  PARAM_CHECK(CARAFE_FORWARD_API, output != NULL);
  /*
   * Sanity check finished
   */
  VLOG(5) << CARAFE_FORWARD_API << "===== End of checking parameters. =====";
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t CarafeBackwardParamCheck(
    mluOpHandle_t handle, const mluOpCarafeDescriptor_t carafe_desc,
    const mluOpTensorDescriptor_t input_desc, const void *input,
    const mluOpTensorDescriptor_t mask_desc, const void *mask,
    const mluOpTensorDescriptor_t grad_output_desc, const void *grad_output,
    const mluOpTensorDescriptor_t grad_input_desc, void *grad_input,
    const mluOpTensorDescriptor_t grad_mask_desc, void *grad_mask,
    bool *return_directly) {
  VLOG(5) << "CarafeBackwardParamCheck start";
  *return_directly = false;
  /*
   * descriptor param check
   */
  PARAM_CHECK(CARAFE_BACKWARD_API, handle != NULL);
  PARAM_CHECK(CARAFE_BACKWARD_API, carafe_desc != NULL);
  PARAM_CHECK(CARAFE_BACKWARD_API, input_desc != NULL);
  PARAM_CHECK(CARAFE_BACKWARD_API, mask_desc != NULL);
  PARAM_CHECK(CARAFE_BACKWARD_API, grad_output_desc != NULL);
  PARAM_CHECK(CARAFE_BACKWARD_API, grad_input_desc != NULL);
  PARAM_CHECK(CARAFE_BACKWARD_API, grad_mask_desc != NULL);
  VLOG(5) << CARAFE_BACKWARD_API << "descriptor check end";

  /*
   * dim check
   */
  PARAM_CHECK_EQ(CARAFE_BACKWARD_API, input_desc->dim, carafe_desc->dimNb);
  PARAM_CHECK_EQ(CARAFE_BACKWARD_API, mask_desc->dim, carafe_desc->dimNb);
  PARAM_CHECK_EQ(CARAFE_BACKWARD_API, grad_output_desc->dim,
                 carafe_desc->dimNb);
  PARAM_CHECK_EQ(CARAFE_BACKWARD_API, grad_input_desc->dim, carafe_desc->dimNb);
  PARAM_CHECK_EQ(CARAFE_BACKWARD_API, grad_mask_desc->dim, carafe_desc->dimNb);
  VLOG(5) << CARAFE_BACKWARD_API << "dim check end.";

  /*
   * layout check
   */
  if (carafe_desc->dimNb == 4) {
    PARAM_CHECK_EQ(CARAFE_BACKWARD_API, input_desc->layout, MLUOP_LAYOUT_NHWC);
    PARAM_CHECK_EQ(CARAFE_BACKWARD_API, mask_desc->layout, MLUOP_LAYOUT_NHWC);
    PARAM_CHECK_EQ(CARAFE_BACKWARD_API, grad_output_desc->layout,
                   MLUOP_LAYOUT_NHWC);
    PARAM_CHECK_EQ(CARAFE_BACKWARD_API, grad_input_desc->layout,
                   MLUOP_LAYOUT_NHWC);
    PARAM_CHECK_EQ(CARAFE_BACKWARD_API, grad_mask_desc->layout,
                   MLUOP_LAYOUT_NHWC);
  }
  VLOG(5) << CARAFE_BACKWARD_API << "layout check end.";

  /*
   * shape param check
   * 1. batch check
   */
  int input_n = mluOpGetTensordimN(input_desc);
  int grad_input_n = mluOpGetTensordimN(grad_input_desc);
  int mask_n = mluOpGetTensordimN(mask_desc);
  int grad_mask_n = mluOpGetTensordimN(grad_mask_desc);
  int grad_output_n = mluOpGetTensordimN(grad_output_desc);
  bool n_check_invalid = input_n != grad_input_n;
  n_check_invalid = n_check_invalid || (input_n != mask_n);
  n_check_invalid = n_check_invalid || (input_n != grad_mask_n);
  n_check_invalid = n_check_invalid || (input_n != grad_output_n);
  if (n_check_invalid) {
    LOG(ERROR) << CARAFE_BACKWARD_API << "batch size mismatch. "
               << "The input batch is: " << input_n
               << ", the mask batch is: " << mask_n
               << ", the grad input batch is: " << grad_input_n
               << ", the grad mask batch is: " << grad_mask_n
               << ", the grad output batch is: " << grad_output_n;
    return MLUOP_STATUS_BAD_PARAM;
  }
  VLOG(5) << CARAFE_BACKWARD_API << "batch check end.";

  /*
   * 2. hw check
   */
  int input_h = mluOpGetTensordimH(input_desc);
  int input_w = mluOpGetTensordimW(input_desc);
  int grad_input_h = mluOpGetTensordimH(input_desc);
  int grad_input_w = mluOpGetTensordimW(input_desc);

  if (input_h != grad_input_h || input_w != grad_input_w) {
    LOG(ERROR) << CARAFE_BACKWARD_API
               << "The hw shapes of input and grad input mismatch. "
               << "The shape of input is (" << input_h << ", " << input_w
               << "), "
               << "the shape of grad input is (" << grad_input_h << ", "
               << grad_input_w << ").";
    return MLUOP_STATUS_BAD_PARAM;
  }

  int mask_h = mluOpGetTensordimH(mask_desc);
  int mask_w = mluOpGetTensordimW(mask_desc);
  int grad_mask_h = mluOpGetTensordimH(grad_mask_desc);
  int grad_mask_w = mluOpGetTensordimW(grad_mask_desc);

  if (mask_h != grad_mask_h || mask_w != grad_mask_w) {
    LOG(ERROR) << CARAFE_BACKWARD_API
               << "The hw shapes of mask and grad mask mismatch. "
               << "The shape of mask is (" << mask_h << ", " << mask_w << "), "
               << "the shape of grad mask is (" << grad_mask_h << ", "
               << grad_mask_w << ").";
    return MLUOP_STATUS_BAD_PARAM;
  }

  int grad_output_h = mluOpGetTensordimH(grad_output_desc);
  int grad_output_w = mluOpGetTensordimW(grad_output_desc);

  int derived_ho = input_h * carafe_desc->scale_factor;
  int derived_wo = input_w * carafe_desc->scale_factor;

  bool howo_invalid = false;
  howo_invalid = howo_invalid || (grad_output_h != mask_h);
  howo_invalid = howo_invalid || (grad_output_w != mask_w);

  if (howo_invalid) {
    LOG(ERROR) << CARAFE_BACKWARD_API
               << "The hw shapes of grad output, mask, grad mask mismatch. "
               << "The shape of grad output is (" << grad_output_h << ", "
               << grad_output_w << "), "
               << "the shape of mask is (" << mask_h << ", " << mask_w << "), "
               << "the shape of grad offset is (" << grad_mask_h << ", "
               << grad_mask_w << "), ";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (grad_output_h != derived_ho || grad_output_w != derived_wo) {
    LOG(ERROR)
        << CARAFE_BACKWARD_API
        << "The hw shape of output mismatch. The desired output hw shape is"
        << "(" << derived_ho << ", " << derived_wo
        << "), whereas the output shape is "
        << "(" << grad_output_h << ", " << grad_output_w << "). "
        << "the hw shape of input is (" << input_h << ", " << input_w << ").";
    return MLUOP_STATUS_BAD_PARAM;
  }
  VLOG(5) << CARAFE_BACKWARD_API << "hw check end.";

  /*
   * channel check
   */
  int input_ci = mluOpGetTensordimC(input_desc);
  int grad_input_ci = mluOpGetTensordimC(grad_input_desc);
  int grad_output_co = mluOpGetTensordimC(grad_output_desc);

  if (input_ci % carafe_desc->group_size != 0) {
    LOG(ERROR) << CARAFE_BACKWARD_API
               << "input channel should be divisible by group_size, "
               << "now input channel is:" << input_ci
               << ", group size is:" << carafe_desc->group_size << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (input_ci != grad_input_ci || input_ci != grad_output_co) {
    LOG(ERROR) << CARAFE_BACKWARD_API
               << "input channel, grad input channel and grad_output channel "
               << "mismatch. The input channel is " << input_ci
               << ", the grad input channel is " << grad_input_ci
               << ", and grad output channel is " << grad_output_co << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  int mask_ci = mluOpGetTensordimC(mask_desc);
  int grad_mask_ci = mluOpGetTensordimC(grad_mask_desc);
  int derived_mask_ci = carafe_desc->kernel_size * carafe_desc->kernel_size *
                        carafe_desc->group_size;
  if (mask_ci != derived_mask_ci || grad_mask_ci != derived_mask_ci) {
    LOG(ERROR) << CARAFE_BACKWARD_API
               << "mask channel, grad mask channel and derived mask channel "
               << "mismatch. The mask channel is " << mask_ci
               << ", the grad mask channel is " << grad_mask_ci
               << ", and derived mask channel is " << derived_mask_ci << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  VLOG(5) << CARAFE_BACKWARD_API << "channel check end.";

  /*
   * data type, offchip data type check
   */
  bool dtype_is_invalid = false;
  dtype_is_invalid =
      dtype_is_invalid || (input_desc->dtype != MLUOP_DTYPE_HALF &&
                           input_desc->dtype != MLUOP_DTYPE_FLOAT);
  if (dtype_is_invalid) {
    LOG(ERROR) << CARAFE_BACKWARD_API
               << "the data type of input only support half and float";
    return MLUOP_STATUS_BAD_PARAM;
  }
  dtype_is_invalid =
      dtype_is_invalid || (input_desc->dtype != mask_desc->dtype);
  dtype_is_invalid =
      dtype_is_invalid || (input_desc->dtype != grad_output_desc->dtype);
  dtype_is_invalid =
      dtype_is_invalid || (input_desc->dtype != grad_input_desc->dtype);
  dtype_is_invalid =
      dtype_is_invalid || (input_desc->dtype != grad_mask_desc->dtype);

  if (dtype_is_invalid) {
    LOG(ERROR) << CARAFE_BACKWARD_API
               << "the data type of input, mask, grad_output, "
                  "grad_input, grad_mask should be the same. "
               << "The data type of input is: "
               << mluop::getNameOfDataType(input_desc->dtype) << ", "
               << "The data type of mask is: "
               << mluop::getNameOfDataType(mask_desc->dtype) << ", "
               << "The data type of grad output is: "
               << mluop::getNameOfDataType(grad_output_desc->dtype) << ", "
               << "The data type of grad input is: "
               << mluop::getNameOfDataType(grad_input_desc->dtype) << ", "
               << "The data type of grad mask is: "
               << mluop::getNameOfDataType(grad_mask_desc->dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  VLOG(5) << CARAFE_BACKWARD_API << "offchip data type check end.";

  /*
   * tensor contiguous check
   */
  if (ifNeedTensorStrideProcess(input_desc)) {
    LOG(ERROR) << CARAFE_BACKWARD_API << "The input tensor is not contiguous.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (ifNeedTensorStrideProcess(mask_desc)) {
    LOG(ERROR) << CARAFE_BACKWARD_API << "The mask tensor is not contiguous.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (ifNeedTensorStrideProcess(grad_output_desc)) {
    LOG(ERROR) << CARAFE_BACKWARD_API
               << "The grad output tensor is not contiguous.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (ifNeedTensorStrideProcess(grad_input_desc)) {
    LOG(ERROR) << CARAFE_BACKWARD_API
               << "The grad input tensor is not contiguous.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (ifNeedTensorStrideProcess(grad_mask_desc)) {
    LOG(ERROR) << CARAFE_BACKWARD_API
               << "The grad mask tensor is not contiguous.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  VLOG(5) << CARAFE_BACKWARD_API << "contiguous check end.";
  /*
   * 0 element param check
   */
  size_t input_num = mluOpGetTensorElementNum(input_desc);
  size_t mask_num = mluOpGetTensorElementNum(mask_desc);
  size_t output_num = mluOpGetTensorElementNum(grad_output_desc);
  size_t grad_input_num = mluOpGetTensorElementNum(grad_input_desc);
  size_t grad_mask_num = mluOpGetTensorElementNum(grad_mask_desc);
  if (input_num == 0 || mask_num == 0 || output_num == 0 ||
      grad_input_num == 0 || grad_mask_num == 0) {
    VLOG(5) << CARAFE_BACKWARD_API
            << "zero element tensor encountered, return directly.";
    *return_directly = true;
    return MLUOP_STATUS_SUCCESS;
  }
  VLOG(5) << CARAFE_BACKWARD_API << "0 element check end.";

  /*
   * null pointer check
   */
  PARAM_CHECK(CARAFE_BACKWARD_API, input != NULL);
  PARAM_CHECK(CARAFE_BACKWARD_API, mask != NULL);
  PARAM_CHECK(CARAFE_BACKWARD_API, grad_output != NULL);
  PARAM_CHECK(CARAFE_BACKWARD_API, grad_input != NULL);
  PARAM_CHECK(CARAFE_BACKWARD_API, grad_mask != NULL);
  VLOG(5) << CARAFE_BACKWARD_API << "null pointer check end.";

  /*
   * k_up and scale check
   */
  if (carafe_desc->kernel_size > 137) {
    LOG(ERROR) << CARAFE_BACKWARD_API
               << "the kernel_size should less than or equal to 137, "
               << "but the kernel_size is: " << carafe_desc->kernel_size
               << ", ";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  VLOG(5) << "CarafeBackwardParamCheck end";
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpCarafeForward(
    mluOpHandle_t handle, const mluOpCarafeDescriptor_t carafe_desc,
    const mluOpTensorDescriptor_t input_desc, const void *input,
    const mluOpTensorDescriptor_t mask_desc, const void *mask,
    const mluOpTensorDescriptor_t output_desc, void *output) {
  // check param
  bool return_directly = true;

  MLUOP_CHECK_RETURN(
      CARAFE_FORWARD_API,
      CarafeForwardParamCheck(handle, carafe_desc, input_desc, input, mask_desc,
                              mask, output_desc, output, &return_directly),
      "Error occured in checking parameters");

  if (return_directly) {
    VLOG(5) << CARAFE_FORWARD_API << "return directly!";
    return MLUOP_STATUS_SUCCESS;
  }

  // generate policy
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  int block_dimH = 0;
  int block_dimW = 0;
  int block_dimG = 0;
  int block_dimC = 0;
  int grid_dimH = 0;
  int grid_dimW = 0;
  int grid_dimG = 0;
  int grid_dimC = 0;
  int job_num = 0;

  MLUOP_CHECK_RETURN(
      CARAFE_FORWARD_API,
      genPolicy(handle, carafe_desc, input_desc, &k_dim, &k_type, &block_dimH,
                &block_dimW, &block_dimG, &block_dimC, &grid_dimH, &grid_dimW,
                &grid_dimG, &grid_dimC, &job_num),
      "Error occured in generating policy.");

  // GEN_CASE
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("carafe_forward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", input, input_desc, 5.1, -5.3);
    GEN_CASE_DATA(true, "mask", mask, mask_desc, 0.0, 1.0);
    GEN_CASE_DATA(false, "output", output, output_desc, 1.7, -1.8);
    GEN_CASE_OP_PARAM_SINGLE(0, "carafe_forward", "dimnb", carafe_desc->dimNb);
    GEN_CASE_OP_PARAM_SINGLE(1, "carafe_forward", "kernel_size",
                             carafe_desc->kernel_size);
    GEN_CASE_OP_PARAM_SINGLE(1, "carafe_forward", "group_size",
                             carafe_desc->group_size);
    GEN_CASE_OP_PARAM_SINGLE(2, "carafe_forward", "scale_factor",
                             carafe_desc->scale_factor);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  // start kernel
  int input_dimN = mluOpGetTensordimN(input_desc);
  int input_dimH = mluOpGetTensordimH(input_desc);
  int input_dimW = mluOpGetTensordimW(input_desc);
  int input_dimC = mluOpGetTensordimC(input_desc);

  int kernel_size = carafe_desc->kernel_size;
  int group_size = carafe_desc->group_size;
  int scale_factor = carafe_desc->scale_factor;

  VLOG(5) << "Launch Kernel mluOpBlockKernelCarafeForward<<<k_type=" << k_type
          << ", k_dim=" << k_dim.x << "," << k_dim.y << "," << k_dim.z << ">>>";

  if (input_desc->dtype == MLUOP_DTYPE_HALF) {
    VLOG(5) << "Kernel mluOpBlockKernelCarafeForwardHalf";
    KERNEL_CHECK((mluOpBlockKernelCarafeForwardHalf(
        k_dim, k_type, handle->queue, input, mask, output, input_dimN,
        input_dimH, input_dimW, input_dimC, kernel_size, group_size,
        scale_factor, block_dimH, block_dimW, block_dimG, block_dimC, grid_dimH,
        grid_dimW, grid_dimG, grid_dimC)));
  } else {
    VLOG(5) << "Kernel mluOpBlockKernelCarafeForwardFloat";
    KERNEL_CHECK((mluOpBlockKernelCarafeForwardFloat(
        k_dim, k_type, handle->queue, input, mask, output, input_dimN,
        input_dimH, input_dimW, input_dimC, kernel_size, group_size,
        scale_factor, block_dimH, block_dimW, block_dimG, block_dimC, grid_dimH,
        grid_dimW, grid_dimG, grid_dimC)));
  }

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpCarafeBackward(
    mluOpHandle_t handle, const mluOpCarafeDescriptor_t carafe_desc,
    const mluOpTensorDescriptor_t input_desc, const void *input,
    const mluOpTensorDescriptor_t mask_desc, const void *mask,
    const mluOpTensorDescriptor_t grad_output_desc, const void *grad_output,
    const mluOpTensorDescriptor_t grad_input_desc, void *grad_input,
    const mluOpTensorDescriptor_t grad_mask_desc, void *grad_mask) {
  bool return_directly;
  mluOpStatus_t param_check_status = CarafeBackwardParamCheck(
      handle, carafe_desc, input_desc, input, mask_desc, mask, grad_output_desc,
      grad_output, grad_input_desc, grad_input, grad_mask_desc, grad_mask,
      &return_directly);

  if (return_directly || param_check_status != MLUOP_STATUS_SUCCESS) {
    return param_check_status;
  }

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("carafe_backward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", input, input_desc, 5.1, -5.3);
    GEN_CASE_DATA(true, "mask", mask, mask_desc, 0.0, 1.0);
    GEN_CASE_DATA(true, "grad_output", grad_output, grad_output_desc, 1.7,
                  -1.8);
    GEN_CASE_DATA(false, "grad_input", grad_input, grad_input_desc, 0, 0);
    GEN_CASE_DATA(false, "grad_mask", grad_mask, grad_mask_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "carafe_backward", "dimnb", carafe_desc->dimNb);
    GEN_CASE_OP_PARAM_SINGLE(1, "carafe_backward", "kernel_size",
                             carafe_desc->kernel_size);
    GEN_CASE_OP_PARAM_SINGLE(1, "carafe_backward", "group_size",
                             carafe_desc->group_size);
    GEN_CASE_OP_PARAM_SINGLE(2, "carafe_backward", "scale_factor",
                             carafe_desc->scale_factor);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  int n = mluOpGetTensordimN(input_desc);
  int hi = mluOpGetTensordimH(input_desc);
  int wi = mluOpGetTensordimW(input_desc);
  int c = mluOpGetTensordimC(input_desc);

  int k_up = carafe_desc->kernel_size;
  int group = carafe_desc->group_size;
  int scale = carafe_desc->scale_factor;

  const size_t fill_value = 0x0;
  PARAM_CHECK(CARAFE_BACKWARD_API,
              MLUOP_STATUS_SUCCESS ==
                  mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST, &fill_value,
                               grad_input_desc, grad_input));

  uint32_t task_dim_x, task_dim_y;
  task_dim_x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  task_dim_y = mluop::runtime::getClusterLimitCapability(handle);
  cnrtDim3_t k_dim = {task_dim_x, task_dim_y, 1};
  cnrtJobType_t k_type = CNRT_FUNC_TYPE_BLOCK;

  VLOG(5) << "Launch mluOpBlockKernelCarafeBackward<<<k_type=" << k_type << ", "
          << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";

  if (input_desc->dtype == MLUOP_DTYPE_HALF) {
    VLOG(5) << "Kernel mluOpBlockKernelCarafeBackwardHalf";
    KERNEL_CHECK((mluOpBlockKernelCarafeBackwardHalf(
        k_dim, k_type, handle->queue, (void *)input, (void *)mask,
        (void *)grad_output, grad_input, grad_mask, n, hi, wi, c, k_up, group,
        scale)));
  } else {
    VLOG(5) << "Kernel mluOpBlockKernelCarafeBackwardHalf";
    KERNEL_CHECK((mluOpBlockKernelCarafeBackwardFloat(
        k_dim, k_type, handle->queue, (void *)input, (void *)mask,
        (void *)grad_output, grad_input, grad_mask, n, hi, wi, c, k_up, group,
        scale)));
  }

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
