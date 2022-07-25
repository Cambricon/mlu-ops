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

#include <cnrt.h>
#include "mlu_op.h"
#include "mlu_op_kernel.h"

#include <iostream>
#include <string>
#include "core/context.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "core/tool.h"
#include "core/gen_case.h"
#include "kernels/kernel.h"

static void policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type, const int input_boxes_num) {
  uint32_t job_limit = mluop::runtime::getJobLimitCapability(handle);
  // when current MLU arch only support Block type job
  if (job_limit == CN_KERNEL_CLASS_BLOCK) {
    *k_type = CNRT_FUNC_TYPE_BLOCK;
    k_dim->x = 1;
    k_dim->y = 1;
    k_dim->z = 1;
    return;
  }

  // union1 policy func
  *k_type = CNRT_FUNC_TYPE_UNION1;
  // dimx equals to num of ipu cores in each cluster
  k_dim->x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  // dimy equals to num of current available clusters
  k_dim->y = 1;
  k_dim->z = 1;
  return;
}

mluOpStatus_t MLUOP_WIN_API
mluOpPolyNms(mluOpHandle_t handle, const mluOpTensorDescriptor_t boxes_desc,
             const void *boxes, float iou_threshold, void *workspace,
             size_t workspace_size, const mluOpTensorDescriptor_t output_desc,
             void *output, void *result_num) {
  const std::string API = "[mluOpPolyNms]";
  // check inputs/outputs
  PARAM_CHECK(API, handle != NULL);
  PARAM_CHECK(API, boxes_desc != NULL);
  PARAM_CHECK(API, output_desc != NULL);
  PARAM_CHECK(API, result_num != NULL);
  // check inputs/outputs data type
  PARAM_CHECK(API, boxes_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, output_desc->dtype == MLUOP_DTYPE_INT32);
  // check inputs layout
  PARAM_CHECK(API, boxes_desc->layout == MLUOP_LAYOUT_ARRAY);
  // check inputs shape
  PARAM_CHECK_EQ(API, boxes_desc->dim, 2);
  PARAM_CHECK_EQ(API, boxes_desc->dims[1], 9);
  PARAM_CHECK(API, boxes_desc->dims[0] == output_desc->dims[0]);

  uint32_t input_boxes_num = boxes_desc->dims[0];
  uint32_t input_stride = boxes_desc->dims[1];

  if (input_boxes_num == 0) {
    VLOG(5) << API << " skip zero element tensor.";
    CNRT_CHECK(cnrtMemset(result_num, 0, sizeof(int32_t)));
    return MLUOP_STATUS_SUCCESS;
  }

  PARAM_CHECK(API, boxes != NULL);
  if (workspace_size > 0) {
    PARAM_CHECK(API, workspace != NULL);
  }

  // generate prototxt
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("poly_nms");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input1", boxes, boxes_desc, 10, 0);
    GEN_CASE_DATA(false, "output1", output, output_desc, 0, 0);
    GEN_CASE_DATA_UNFOLD(false, "output2", result_num, 1, {1},
                         MLUOP_DTYPE_INT32, MLUOP_LAYOUT_ARRAY, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "poly_nms", "iou_threshold", iou_threshold);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 3e-3, 3e-3, 0);
  }

  mluOpDataType_t data_type_input = boxes_desc->dtype;
  cnrtDim3_t k_dim;
  cnrtJobType_t k_type;
  policyFunc(handle, &k_dim, &k_type, input_boxes_num);

  VLOG(5) << "Launch Kernel MLUUnion1OrBlockPNMS<<<k_dim: " << k_type << ", "
          << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";

  KERNEL_CHECK((mluOpUnion1OrBlockKernelPolyNmsFloat(
      k_dim, k_type, handle->queue, (void *)boxes, input_boxes_num,
      input_stride, iou_threshold, output, result_num, workspace)));

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetPolyNmsWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t boxes_desc,
    size_t *size) {
  const std::string API = "[mluOpGetPolyNmsWorkspaceSize]";
  PARAM_CHECK(API, handle != NULL);
  PARAM_CHECK(API, boxes_desc != NULL);
  PARAM_CHECK("[mluOpGetPolyNmsWorkspaceSize]", size != NULL);

  PARAM_CHECK(API, boxes_desc->dim == 2);
  PARAM_CHECK(API, boxes_desc->dims[1] == 9);

  PARAM_CHECK(API, boxes_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, boxes_desc->layout == MLUOP_LAYOUT_ARRAY);
  // workspace stores the transposed input data[9, N].
  uint32_t input_boxes_num = boxes_desc->dims[0];  // N
  uint32_t input_stride = boxes_desc->dims[1];     // 9
#if __BANG_ARCH__ >= 300
  *size = input_boxes_num * input_stride * sizeof(float);
#else
  int align_num = 128 / sizeof(float);
  int align_box_num = CEIL_ALIGN(input_boxes_num, align_num);
  int align_stride = CEIL_ALIGN(input_stride, align_num);
  *size = align_box_num * align_stride * sizeof(float);
#endif

  VLOG(5) << "[mluOpGetPolyNmsWorkspaceSize] size = :" << *size;
  return MLUOP_STATUS_SUCCESS;
}
