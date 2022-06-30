/*************************************************************************
 * Copyright (C) 2022 by Cambricon, Inc. All rights reserved.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include <string>
#include "core/context.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"

static void policyFunc(const mluOpHandle_t handle, int bin_num,
                       cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  uint32_t cluster_num = mluop::runtime::getClusterLimitCapability(handle);
  uint32_t core_in_cluster = handle->core_num_per_cluster;
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = core_in_cluster;
  uint32_t use_cluster = (bin_num + core_in_cluster - 1) / core_in_cluster;
  k_dim->y = use_cluster > cluster_num ? cluster_num : use_cluster;
  k_dim->z = 1;
}

/* user param check
 * step1:check desc and data ptr is not nullptr_t
 * step2:check shape and data type
 * step3:check zero element
 * */
mluOpStatus_t RoiCropBackwardParamCheck(
    const std::string &op_name, const mluOpHandle_t handle,
    const mluOpTensorDescriptor_t gradOutput_desc, const void *gradOutput,
    const mluOpTensorDescriptor_t grid_desc, const void *grid,
    const mluOpTensorDescriptor_t gradInput_desc, const void *gradInput) {
  // check descriptor and data
  PARAM_CHECK(op_name, handle != NULL);
  PARAM_CHECK(op_name, gradOutput_desc != NULL);
  PARAM_CHECK(op_name, grid_desc != NULL);
  PARAM_CHECK(op_name, gradInput_desc != NULL);
  // check data type
  PARAM_CHECK(op_name, gradOutput_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(op_name, gradOutput_desc->dim == 4);
  PARAM_CHECK(op_name, grid_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(op_name, grid_desc->dim == 4);
  PARAM_CHECK(op_name, gradInput_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(op_name, gradInput_desc->dim == 4);
  // check shape and layout
  PARAM_CHECK(op_name, gradOutput_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(op_name, gradInput_desc->layout == MLUOP_LAYOUT_NHWC);
  for (int i = 0; i < gradOutput_desc->dim - 1; ++i) {
    if (gradOutput_desc->dims[i] != grid_desc->dims[i]) {
      LOG(ERROR) << op_name << " Check failed: gradOutput_desc->dims[" << i
                 << "] should be equal to grid_desc->dims[" << i << "].";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  if (gradOutput_desc->dims[3] != gradInput_desc->dims[3]) {
    LOG(ERROR) << op_name
               << " Check failed: gradOutput_desc->dims[3] should be "
                  "equal to gradInput_desc->dims[3].";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (grid_desc->dims[3] != 2) {
    LOG(ERROR) << op_name
               << " Check failed: grid_desc->dims[3] should be equal to 2.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  const size_t max_input_num = 2147483648;  // 2^31 2G num
  if ((mluOpGetTensorElementNum(gradOutput_desc) >= max_input_num) ||
      (mluOpGetTensorElementNum(grid_desc) >= max_input_num) ) {
    LOG(ERROR) << op_name << " Overflow max tensor num."
               <<" Currently, MLU-OPS supports tensor num smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  // check zero element
  if ((mluOpGetTensorElementNum(gradInput_desc) == 0) ||
      (mluOpGetTensorElementNum(grid_desc) == 0) ||
      (mluOpGetTensorElementNum(gradOutput_desc) == 0)) {
    VLOG(5) << op_name << " skip zero element tensor.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  PARAM_CHECK(op_name, gradOutput != NULL);
  PARAM_CHECK(op_name, grid != NULL);
  PARAM_CHECK(op_name, gradInput != NULL);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpRoiCropBackward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t gradOutput_desc,
    const void *gradOutput, const mluOpTensorDescriptor_t grid_desc,
    const void *grid, const mluOpTensorDescriptor_t gradInput_desc,
    void *gradInput) {
  // check params
  mluOpStatus_t param_check = RoiCropBackwardParamCheck(
      "[mluOpRoiCropBackward]", handle, gradOutput_desc, gradOutput, grid_desc,
      grid, gradInput_desc, gradInput);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }

  uint32_t batch = gradInput_desc->dims[0];
  uint32_t height = gradInput_desc->dims[1];
  uint32_t width = gradInput_desc->dims[2];
  uint32_t channels = gradInput_desc->dims[3];
  uint32_t grid_n = grid_desc->dims[0];
  uint32_t output_h = gradOutput_desc->dims[1];
  uint32_t output_w = gradOutput_desc->dims[2];
  uint32_t bin_num = grid_n * output_h * output_w;

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  policyFunc(handle, bin_num, &k_dim, &k_type);
  VLOG(5) << " [mluOpRoiCropBackward] launch kernel policyFUnc[" << k_dim.x
          << ", " << k_dim.y << ", " << k_dim.z << "].";
  // gdram set zero
  int gd_num = channels * width * height * batch;
  KERNEL_CHECK((mluOpBlockKernelFreeZero(k_dim, k_type, handle->queue, gd_num,
                                         gradInput)));
  VLOG(5) << " kernel mluOpBlockKernelFreeZero";

  KERNEL_CHECK((mluOpBlockKernelRoiCropBackwardFloat(
      k_dim, k_type, handle->queue, gradOutput, grid, batch, height, width,
      channels, grid_n, output_h, output_w, gradInput)));
  VLOG(5) << " kernel mluOpBlockKernelRoiCropBackwardFloat";
  return MLUOP_STATUS_SUCCESS;
}
