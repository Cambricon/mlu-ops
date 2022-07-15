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
#include <string>

#include "core/context.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"

// policy function
static void policyFuncPsroipoolForward(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                                       cnrtFunctionType_t *k_type,
                                       const int rois_num) {
  size_t union_number = mluop::runtime::getClusterLimitCapability(handle);
  size_t core_in_cluster = handle->core_num_per_cluster;
  uint32_t use_cluster = (rois_num + core_in_cluster - 1) / core_in_cluster;
  *k_type = CNRT_FUNC_TYPE_UNION1;  // default func type
  k_dim->x = core_in_cluster;
  k_dim->y = use_cluster > core_in_cluster ? core_in_cluster : use_cluster;
  k_dim->z = 1;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetPsRoiPoolWorkspaceSize(mluOpHandle_t handle,
                                                           const int output_dim,
                                                           size_t *size) {
  PARAM_CHECK("[mluOpGetPsRoiPoolWorkspaceSize]", handle != NULL);
  PARAM_CHECK("[mluOpGetPsRoiPoolWorkspaceSize]", size != NULL);
  PARAM_CHECK("[mluOpGetPsRoiPoolWorkspaceSize]", output_dim >= 1);
  *size = output_dim * sizeof(uint32_t);  // the offset of each pixel

  VLOG(5) << "workspace size = " << *size << ".";
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t param_and_pointer_check(
    const std::string &api, const float spatial_scale, const int group_size,
    const void *input_data, const void *input_rois, const void *output_data,
    const void *mapping_channel, const mluOpTensorDescriptor_t input_data_desc,
    const mluOpTensorDescriptor_t input_rois_desc,
    const mluOpTensorDescriptor_t output_data_desc,
    const mluOpTensorDescriptor_t mapping_channel_desc, void *workspace,
    const size_t workspace_size) {
  PARAM_CHECK(api, input_data_desc != NULL);
  PARAM_CHECK(api, input_rois_desc != NULL);
  PARAM_CHECK(api, output_data_desc != NULL);
  PARAM_CHECK(api, mapping_channel_desc != NULL);
  PARAM_CHECK(api, input_data_desc->dim == 4);
  PARAM_CHECK(api, input_rois_desc->dim == 2);
  PARAM_CHECK(api, output_data_desc->dim == 4);
  PARAM_CHECK(api, mapping_channel_desc->dim == 4);
  // check the input and output datatype
  PARAM_CHECK(api, input_data_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(api, input_rois_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(api, output_data_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(api, mapping_channel_desc->dtype == MLUOP_DTYPE_INT32);
  // check layout
  PARAM_CHECK(api, input_data_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(api, output_data_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(api, mapping_channel_desc->layout == MLUOP_LAYOUT_NHWC);
  // param check
  // group_size == pooled_height
  PARAM_CHECK(api, group_size == output_data_desc->dims[1]);
  // pooled_height == pooled_width
  PARAM_CHECK(api, output_data_desc->dims[1] == output_data_desc->dims[2]);
  // group_size >= 1.
  PARAM_CHECK(api, group_size >= 1);
  // output_dim >= 1.
  PARAM_CHECK(api, output_data_desc->dims[3] >= 1);
  // spatial_scale > 0
  PARAM_CHECK(api, spatial_scale > 0);
  // rois_offset = 5.
  PARAM_CHECK(api, input_rois_desc->dims[1] == 5);
  // roi_num check
  PARAM_CHECK(api, output_data_desc->dims[0] == input_rois_desc->dims[0]);
  // channels == pooled_height * pooled_width * output_dim
  PARAM_CHECK(api, input_data_desc->dims[3] == output_data_desc->dims[1] *
                                                   output_data_desc->dims[2] *
                                                   output_data_desc->dims[3]);
  PARAM_CHECK(api, output_data_desc->dims[0] == mapping_channel_desc->dims[0]);
  PARAM_CHECK(api, output_data_desc->dims[1] == mapping_channel_desc->dims[1]);
  PARAM_CHECK(api, output_data_desc->dims[2] == mapping_channel_desc->dims[2]);
  PARAM_CHECK(api, output_data_desc->dims[3] == mapping_channel_desc->dims[3]);
  if (mluOpGetTensorElementNum(input_data_desc) == 0) {
    VLOG(5) << api << " Input_data skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
  if (mluOpGetTensorElementNum(input_rois_desc) == 0) {
    LOG(ERROR) << api << " Roi_data can not be zero element tensor.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (workspace_size > 0) {
    PARAM_CHECK(api, workspace != NULL);
  }
  PARAM_CHECK(api, input_data != NULL);
  PARAM_CHECK(api, input_rois != NULL);
  PARAM_CHECK(api, output_data != NULL);
  PARAM_CHECK(api, mapping_channel != NULL);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpPsRoiPoolForward(
    mluOpHandle_t handle, const float spatial_scale, const int group_size,
    const mluOpTensorDescriptor_t input_data_desc, const void *input_data,
    const mluOpTensorDescriptor_t input_rois_desc, const void *input_rois,
    void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t output_data_desc, void *output_data,
    const mluOpTensorDescriptor_t mapping_channel_desc, void *mapping_channel) {
  const std::string api = "[mluOpPsRoiPoolForward]";
  PARAM_CHECK(api, handle != NULL);
  const int batch_size = input_data_desc->dims[0];
  const int height = input_data_desc->dims[1];
  const int width = input_data_desc->dims[2];
  const int channels = input_data_desc->dims[3];
  const int rois_sum = output_data_desc->dims[0];
  const int pooled_height = output_data_desc->dims[1];
  const int pooled_width = output_data_desc->dims[2];
  const int output_dim = output_data_desc->dims[3];
  const int rois_offset = input_rois_desc->dims[1];

  mluOpStatus_t ret = param_and_pointer_check(
      api, spatial_scale, group_size, input_data, input_rois, output_data,
      mapping_channel, input_data_desc, input_rois_desc, output_data_desc,
      mapping_channel_desc, workspace, workspace_size);
  if (ret != MLUOP_STATUS_SUCCESS) {
    LOG(ERROR) << api
               << " Error found during element verification, please check.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFuncPsroipoolForward(handle, &k_dim, &k_type, rois_sum);
  VLOG(5) << api << " Launch [" << k_type << ", " << k_dim.x << ", " << k_dim.y
          << ", " << k_dim.z << "].";
  KERNEL_CHECK((mluOpBlockKernelPsRoiPoolForward(
      k_dim, k_type, handle->queue, (void *)input_data, (void *)input_rois,
      (void *)output_data, (void *)mapping_channel, channels, height, width,
      pooled_height, pooled_width, rois_sum, output_dim, group_size,
      rois_offset, spatial_scale, batch_size)));
  return MLUOP_STATUS_SUCCESS;
}
