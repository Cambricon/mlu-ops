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
#include <algorithm>
#include <string>
#include <vector>

#include "core/context.h"
#include "core/logging.h"
#include "core/mlu_env.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "kernels/sparse_conv/get_indice_pairs/get_indice_pairs_structs.h"
#include "kernels/sparse_conv/get_indice_pairs/normal_get_indice_pairs.h"
#include "kernels/kernel.h"
#include "kernels/utils/cnnl_helper.h"
#include "mlu_op.h"

static mluOpStatus_t getIndiceMaskAll(
    const mluOpTensorDescriptor_t indice_pairs_desc, const int kernel_volume,
    const int input_active_site, size_t *size) {
  size_t total_size = 0;
  total_size = kernel_volume * input_active_site *
               mluop::getSizeOfDataType(indice_pairs_desc->dtype);
  size[0] = total_size;
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t getIndiceIndexIn(
    const mluOpTensorDescriptor_t indice_pairs_desc, const int kernel_volume,
    const int input_active_site, size_t *size) {
  size_t total_size = 0;
  total_size = kernel_volume * input_active_site *
               mluop::getSizeOfDataType(indice_pairs_desc->dtype);
  size[0] = total_size;
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t getIndiceIndexOut(
    const mluOpTensorDescriptor_t indice_pairs_desc, const int kernel_volume,
    const int input_active_site, size_t *size) {
  size_t total_size = 0;
  total_size = kernel_volume * input_active_site *
               mluop::getSizeOfDataType(indice_pairs_desc->dtype);
  size[0] = total_size;
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t getIndiceOutExpand(
    const mluOpTensorDescriptor_t indice_pairs_desc, const int kernel_volume,
    const int input_active_site, size_t *size) {
  size_t total_size = 0;
  total_size = kernel_volume * input_active_site *
               mluop::getSizeOfDataType(indice_pairs_desc->dtype);
  size[0] = total_size;
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t getIndiceInExpand(
    const mluOpTensorDescriptor_t indice_pairs_desc,
    const int input_active_site, size_t *size) {
  size_t total_size = 0;
  total_size =
      input_active_site * mluop::getSizeOfDataType(indice_pairs_desc->dtype);
  size[0] = total_size;
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t getIndiceUnique(
    const mluOpTensorDescriptor_t indice_pairs_desc, const int kernel_volume,
    const int input_active_site, size_t *size) {
  size_t total_size = 0;
  total_size = (kernel_volume * input_active_site + 1) *
               mluop::getSizeOfDataType(indice_pairs_desc->dtype);
  size[0] = total_size;
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t getGridOut(const mluOpTensorDescriptor_t indice_pairs_desc,
                                int output_size, size_t *size) {
  size_t total_size = 0;
  total_size = output_size * mluop::getSizeOfDataType(indice_pairs_desc->dtype);
  size[0] = total_size;
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t getReduceOpWS(mluOpHandle_t handle,
                                   const std::string interface_name,
                                   const int kernel_volume,
                                   const int input_active_site, size_t *size) {
  size_t total_size = 0;
  mluOpTensorDescriptor_t reduce_in_desc, reduce_out_desc;
  CHECK_RETURN(interface_name, mluOpCreateTensorDescriptor(&reduce_in_desc));
  CHECK_RETURN(interface_name, mluOpCreateTensorDescriptor(&reduce_out_desc));
  std::vector<int> reduce_in_dims = {kernel_volume, input_active_site};
  CHECK_RETURN(interface_name,
               mluOpSetTensorDescriptor(
                   reduce_in_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                   reduce_in_dims.size(), reduce_in_dims.data()));
  reduce_in_dims[1] = 1;
  CHECK_RETURN(interface_name,
               mluOpSetTensorDescriptor(
                   reduce_out_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                   reduce_in_dims.size(), reduce_in_dims.data()));
  // reduce along lowest dimension
  int axis[1] = {1};
  int axis_num = 1;
  cnnlReduceDescriptor_t reduce_desc;
  CALL_CNNL(cnnlCreateReduceDescriptor(&reduce_desc));
  CALL_CNNL(cnnlSetReduceDescriptor(
      reduce_desc, axis, axis_num, CNNL_REDUCE_ADD,
      cnnlDataType_t(reduce_in_desc->dtype), CNNL_PROPAGATE_NAN,
      CNNL_REDUCE_NO_INDICES, CNNL_16BIT_INDICES));
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(reduce_in_desc,
                                                 cnnl_input_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(reduce_out_desc,
                                                 cnnl_output_desc);
    CALL_CNNL(cnnlGetReduceOpWorkspaceSize(cnnl_handle, cnnl_input_desc,
                                           cnnl_output_desc, reduce_desc,
                                           &total_size));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  CHECK_RETURN(interface_name, mluOpDestroyTensorDescriptor(reduce_in_desc));
  CHECK_RETURN(interface_name, mluOpDestroyTensorDescriptor(reduce_out_desc));
  CALL_CNNL(cnnlDestroyReduceDescriptor(reduce_desc));
  size[0] = total_size;
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t getUniqueOpWS(mluOpHandle_t handle,
                                   const std::string interface_name,
                                   const mluOpTensorDescriptor_t indices_desc,
                                   const int kernel_volume,
                                   const int input_active_site, size_t *size) {
  size_t total_size = 0;
  mluOpTensorDescriptor_t input_unique_desc;
  CHECK_RETURN(interface_name, mluOpCreateTensorDescriptor(&input_unique_desc));
  std::vector<int> unique_in_dims = {kernel_volume * input_active_site};
  CHECK_RETURN(interface_name,
               mluOpSetTensorDescriptor(
                   input_unique_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                   unique_in_dims.size(), unique_in_dims.data()));

  {
    cnnlUniqueSort_t unique_mode = CNNL_SORT_ASCEND;
    cnnlUniqueDescriptor_t unique_desc;

    CALL_CNNL(cnnlCreateUniqueDescriptor(&unique_desc));
    CALL_CNNL(
        cnnlSetUniqueDescriptor(unique_desc, unique_mode, 0, false, false));

    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_unique_desc,
                                                 cnnl_input_desc);
    CALL_CNNL(cnnlGetUniqueWorkspaceSize(cnnl_handle, unique_desc,
                                         cnnl_input_desc, &total_size));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);

    CALL_CNNL(cnnlDestroyUniqueDescriptor(unique_desc));
  }

  CHECK_RETURN(interface_name, mluOpDestroyTensorDescriptor(input_unique_desc));

  size[0] = total_size;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t getNormalGetIndicePairsWorkspaceSize(
    mluOpHandle_t handle, const std::string interface_name,
    mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
    const mluOpTensorDescriptor_t indices_desc,
    const mluOpTensorDescriptor_t indice_pairs_desc,
    const mluOpTensorDescriptor_t out_indices_desc,
    const mluOpTensorDescriptor_t indice_num_desc, size_t *return_ws) {
  // workspace for get_indice_pairs
  size_t total_size = 0;
  int sub_m = sparse_conv_desc->sub_m;
  int batch = sparse_conv_desc->batch;
  int kernel_volume = indice_pairs_desc->dims[0];
  int input_active_site = indice_pairs_desc->dims[2];
  int output_size = batch * sparse_conv_desc->output_space[0] *
                        sparse_conv_desc->output_space[1] *
                        sparse_conv_desc->output_space[2] +
                    1;
  size_t mask_all_ws = 0, indice_index_in_ws = 0, indice_index_out_ws = 0;
  size_t out_indices_expand_ws = 0, grid_out_ws = 0, reduce_op_ws = 0;
  CHECK_RETURN(interface_name,
               getIndiceMaskAll(indice_pairs_desc, kernel_volume,
                                input_active_site, &mask_all_ws));
  CHECK_RETURN(interface_name,
               getIndiceIndexIn(indice_pairs_desc, kernel_volume,
                                input_active_site, &indice_index_in_ws));
  CHECK_RETURN(interface_name,
               getIndiceIndexOut(indice_pairs_desc, kernel_volume,
                                 input_active_site, &indice_index_out_ws));
  CHECK_RETURN(interface_name,
               getIndiceOutExpand(indice_pairs_desc, kernel_volume,
                                  input_active_site, &out_indices_expand_ws));
  CHECK_RETURN(interface_name,
               getGridOut(indice_pairs_desc, output_size, &grid_out_ws));
  CHECK_RETURN(interface_name,
               getReduceOpWS(handle, interface_name, kernel_volume,
                             input_active_site, &reduce_op_ws));
  if (sub_m) {
    /*  workspace for subm mode
    | mask_all |indices_index_in | indices_index_out/ step_index |
    indices_in_expand |out_indices_expand| max(grid_out_ws, reduce_op_ws)|
    */
    size_t indice_in_expand_ws = 0;
    CHECK_RETURN(interface_name,
                 getIndiceInExpand(indice_pairs_desc, input_active_site,
                                   &indice_in_expand_ws));
    total_size = mask_all_ws + indice_index_in_ws + indice_index_out_ws +
                 out_indices_expand_ws + indice_in_expand_ws +
                 std::max(grid_out_ws, reduce_op_ws);
  } else {
    /* workspace for default mode
      | mask_all | indices_index_in | step_index/ indices_index_out |
      out_indices_expand  | | out_indices_unique | max(grid_out_ws, reduce_ws,
      unique_ws) |
    */
    size_t indice_unique_ws = 0, unique_op_ws = 0;
    CHECK_RETURN(
        interface_name,
        getUniqueOpWS(handle, interface_name, indices_desc, kernel_volume,
                      input_active_site, &unique_op_ws));
    CHECK_RETURN(interface_name,
                 getIndiceUnique(indice_pairs_desc, kernel_volume,
                                 input_active_site, &indice_unique_ws));
    total_size = mask_all_ws + indice_index_in_ws + indice_index_out_ws +
                 out_indices_expand_ws + indice_unique_ws +
                 std::max(grid_out_ws, std::max(reduce_op_ws, unique_op_ws));
  }
  return_ws[0] = total_size;
  return MLUOP_STATUS_SUCCESS;
}

/* DefaultKernel1
intput: indices  l,4  int
output: mask_all  k,l  int
        indice_index_in k,l int
        out_indices_expand k,l int
func:  gen mask_all, indice_index_in, out_indices_expand for next step.
*/
mluOpStatus_t launchDefaultKernel1(
    mluOpHandle_t handle,
    const mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
    const void *indices, void *mask_all_ws, void *indice_index_in_ws,
    void *out_indices_expand_ws, int batch, int kernel_volume,
    int input_active_site) {
  cnrtDim3_t kDim3;
  cnrtFunctionType_t func_type;
  int core_dim = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  int cluster_number = mluop::runtime::getClusterLimitCapability(handle);
  int core_nums = core_dim * cluster_number;
  int nram_size = handle->nram_size + REM_FOR_STACK - 12 * 1024;
  int nums = 19 * kernel_volume + 8;
  int core_num_l = (nram_size - 4 * 4096 * 3) / nums / sizeof(int);
  int jobs = (input_active_site + core_num_l - 1) / core_num_l;
  int job_num = jobs > core_nums ? core_nums : jobs;
  func_type = CNRT_FUNC_TYPE_BLOCK;
  kDim3.x = 1;
  kDim3.y = job_num;
  kDim3.z = 1;
  /*  nram_space */
  // |input| mask_all | indice_index_in | out_indices_expand |  l +  3 k l
  // |input| mask_all | indice_index_in | out_indices_expand |  l +  3 k l
  // | nram_aux_a  5 l k | nram_aux_b 8 l k
  // ping + pong + aux
  FilterSpace filter_space(sparse_conv_desc->filter_space[0],
                           sparse_conv_desc->filter_space[1],
                           sparse_conv_desc->filter_space[2]);
  InputSpace input_space(sparse_conv_desc->input_space[0],
                         sparse_conv_desc->input_space[1],
                         sparse_conv_desc->input_space[2]);
  OutputSpace output_space(sparse_conv_desc->output_space[0],
                           sparse_conv_desc->output_space[1],
                           sparse_conv_desc->output_space[2]);
  Stride stride(sparse_conv_desc->stride[0], sparse_conv_desc->stride[1],
                sparse_conv_desc->stride[2]);
  Dilation dilation(sparse_conv_desc->dilation[0],
                    sparse_conv_desc->dilation[1],
                    sparse_conv_desc->dilation[2]);
  Padding padding(sparse_conv_desc->pad[0], sparse_conv_desc->pad[1],
                  sparse_conv_desc->pad[2]);
  VLOG(5) << "[getIndicePairsDefault] Launch kernel "
             "KernelDefaultGetIndicePairKl1<<<U"
          << func_type / core_dim << ", " << kDim3.x << ", " << kDim3.y << ", "
          << kDim3.z << ">>>";
  CHECK_RETURN(
      "[getIndicePairsDefault]",
      KernelDefaultGetIndicePairKl1(
          kDim3, func_type, handle->queue, (void *)mask_all_ws,
          (void *)indice_index_in_ws, (void *)out_indices_expand_ws,
          (void *)indices, filter_space, input_space, output_space, stride,
          dilation, padding, core_num_l, input_active_site, batch));
  return MLUOP_STATUS_SUCCESS;
}

/* SubmKernel1
intput: indices  l,4  int
output: mask_all  k,l  int
        indice_index_in k,l int
        indice_in_expand l, int
        out_indices_expand k,l int
func:  gen mask_all, indice_index_in, indice_in_expand, out_indices_expand for
next step.
*/
mluOpStatus_t launchSubmKernel1(
    mluOpHandle_t handle,
    const mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
    const void *indices, void *mask_all_ptr, void *indice_index_in_ptr,
    void *indice_in_expand_ptr, void *out_indices_expand_ptr, int batch,
    int kernel_volume, int input_active_site) {
  cnrtDim3_t kDim3;
  cnrtFunctionType_t func_type;
  int core_dim = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  int cluster_number = mluop::runtime::getClusterLimitCapability(handle);
  int core_nums = core_dim * cluster_number;
  int nram_size = handle->nram_size + REM_FOR_STACK - 12 * 1024;
  int nums = 19 * kernel_volume + 10;
  int core_num_l = (nram_size - 4 * 4096 * 3) / nums / sizeof(int);
  int jobs = (input_active_site + core_num_l - 1) / core_num_l;
  int least_jobs = (input_active_site * sizeof(int) + 1024 - 1) / 1024;
  jobs = std::max(jobs, least_jobs);
  int job_num = jobs > core_nums ? core_nums : jobs;
  func_type = CNRT_FUNC_TYPE_BLOCK;
  kDim3.x = 1;
  kDim3.y = job_num;
  kDim3.z = 1;
  /*  nram_space
  |input| mask_all | indice_index_in |  out_indices_expand | indice_in_expand
  |4l + l + 3kl |input| mask_all | indice_index_in |  out_indices_expand |
  indice_in_expand |4l + l + 3kl | nram_aux_a  5lk | nram_aux_b 8lk |
 */
  FilterSpace filter_space(sparse_conv_desc->filter_space[0],
                           sparse_conv_desc->filter_space[1],
                           sparse_conv_desc->filter_space[2]);
  InputSpace input_space(sparse_conv_desc->input_space[0],
                         sparse_conv_desc->input_space[1],
                         sparse_conv_desc->input_space[2]);
  OutputSpace output_space(sparse_conv_desc->output_space[0],
                           sparse_conv_desc->output_space[1],
                           sparse_conv_desc->output_space[2]);
  Stride stride(sparse_conv_desc->stride[0], sparse_conv_desc->stride[1],
                sparse_conv_desc->stride[2]);
  Dilation dilation(sparse_conv_desc->stride[0], sparse_conv_desc->stride[1],
                    sparse_conv_desc->stride[2]);
  Padding padding(sparse_conv_desc->pad[0], sparse_conv_desc->pad[1],
                  sparse_conv_desc->pad[2]);
  VLOG(5) << "[getIndicePairsDefault] Launch kernel "
             "KernelSubmGetIndicePairKl1<<<U"
          << func_type / core_dim << ", " << kDim3.x << ", " << kDim3.y << ", "
          << kDim3.z << ">>>";
  CHECK_RETURN("[getIndicePairsDefault]",
               KernelSubmGetIndicePairKl1(
                   kDim3, func_type, handle->queue, (void *)mask_all_ptr,
                   (void *)indice_index_in_ptr, (void *)indice_in_expand_ptr,
                   (void *)out_indices_expand_ptr, (void *)indices,
                   filter_space, input_space, output_space, stride, dilation,
                   padding, core_num_l, input_active_site, batch));
  return MLUOP_STATUS_SUCCESS;
}

/* SubmKernel2
intput: indices  l,4  int
        out_indices_expand_ptr  k,l int
        mask_all_ptr  k,l  int
output: mask_all  k,l  int
        out_indices l,4 int
func:  gen out_indices from indices in subm mode;
       gen mask_all by and out_indices_expand_ptr/ mask_all_ptr.
*/
mluOpStatus_t launchSubmKernel2(mluOpHandle_t handle, const void *indices,
                                void *out_indices_index_ptr, void *mask_all_ptr,
                                void *out_indices, int kernel_volume,
                                int input_active_site) {
  cnrtDim3_t kDim3;
  cnrtFunctionType_t func_type;
  int core_dim = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  int cluster_number = mluop::runtime::getClusterLimitCapability(handle);
  int core_nums = core_dim * cluster_number;
  int nram_size = handle->nram_size + REM_FOR_STACK - 12 * 1024;
  int core_num_l_two = (nram_size - 4 * 4096 * 3) / 2 / sizeof(int);
  int core_num_l_one = (nram_size - 4 * 4096 * 3) / sizeof(int);
  int len_1_one = input_active_site * 4;
  int len_l_two = input_active_site * kernel_volume;
  int jobs_one = (len_1_one + core_num_l_one - 1) / core_num_l_one;
  int jobs_two = (len_l_two + core_num_l_two - 1) / core_num_l_two;
  int least_job_one = (len_1_one * sizeof(int) + 1024 - 1) / 1024;
  int least_job_two = (len_l_two * sizeof(int) + 1024 - 1) / 1024;
  int least_jobs = std::max(least_job_one, least_job_two);
  int jobs = std::max(std::max(jobs_one, jobs_two), least_jobs);
  int job_num = jobs > core_nums ? core_nums : jobs;
  func_type = CNRT_FUNC_TYPE_BLOCK;
  kDim3.x = 1;
  kDim3.y = job_num;
  kDim3.z = 1;
  VLOG(5) << "[getIndicePairsDefault] Launch kernel "
             "KernelSubmGetIndicePairKl2<<<U"
          << func_type / core_dim << ", " << kDim3.x << ", " << kDim3.y << ", "
          << kDim3.z << ">>>";
  CHECK_RETURN(
      "[getIndicePairsDefault]",
      KernelSubmGetIndicePairKl2(
          kDim3, func_type, handle->queue, (void *)out_indices,
          (void *)mask_all_ptr, (void *)out_indices_index_ptr, (void *)indices,
          len_1_one, len_l_two, core_num_l_one, core_num_l_two));
  return MLUOP_STATUS_SUCCESS;
}

// call reduce op
mluOpStatus_t launchReduceOp(mluOpHandle_t handle,
                             const std::string interface_name,
                             void *reduce_output_addr, void *reduce_input_addr,
                             void *reduce_workspace_ptr, size_t reduce_op_ws,
                             int kernel_volume, int input_active_site) {
  mluOpTensorDescriptor_t reduce_in_desc, reduce_out_desc;
  CHECK_RETURN(interface_name, mluOpCreateTensorDescriptor(&reduce_in_desc));
  CHECK_RETURN(interface_name, mluOpCreateTensorDescriptor(&reduce_out_desc));
  std::vector<int> reduce_in_dims = {kernel_volume, input_active_site};
  CHECK_RETURN(interface_name,
               mluOpSetTensorDescriptor(
                   reduce_in_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                   reduce_in_dims.size(), reduce_in_dims.data()));
  reduce_in_dims[1] = 1;
  CHECK_RETURN(interface_name,
               mluOpSetTensorDescriptor(
                   reduce_out_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                   reduce_in_dims.size(), reduce_in_dims.data()));
  // reduce along lowest dimension
  int axis[1] = {1};
  int axis_num = 1;
  cnnlReduceDescriptor_t reduce_desc;
  CALL_CNNL(cnnlCreateReduceDescriptor(&reduce_desc));
  CALL_CNNL(cnnlSetReduceDescriptor(
      reduce_desc, axis, axis_num, CNNL_REDUCE_ADD,
      cnnlDataType_t(reduce_in_desc->dtype), CNNL_PROPAGATE_NAN,
      CNNL_REDUCE_NO_INDICES, CNNL_16BIT_INDICES));
  void *alpha = NULL, *beta = NULL, *indices = NULL;
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(reduce_in_desc,
                                                 cnnl_input_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(reduce_out_desc,
                                                 cnnl_output_desc);
    CALL_CNNL(cnnlReduce(cnnl_handle, reduce_desc, reduce_workspace_ptr,
                         reduce_op_ws, alpha, cnnl_input_desc,
                         reduce_input_addr, 0, indices, beta, cnnl_output_desc,
                         reduce_output_addr));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  CHECK_RETURN(interface_name, mluOpDestroyTensorDescriptor(reduce_in_desc));
  CHECK_RETURN(interface_name, mluOpDestroyTensorDescriptor(reduce_out_desc));
  CALL_CNNL(cnnlDestroyReduceDescriptor(reduce_desc));
  return MLUOP_STATUS_SUCCESS;
}

// call unqiue_v2 op
mluOpStatus_t launchUniqueOp(mluOpHandle_t handle,
                             const std::string interface_name,
                             void *unique_output_addr, void *unique_input_addr,
                             void *unique_output_num_addr,
                             void *unique_workspace_ptr, size_t unique_op_ws,
                             int kernel_volume, int input_active_site,
                             int *return_num_act) {
  mluOpTensorDescriptor_t unique_input_desc, unique_output_desc;
  CHECK_RETURN(interface_name, mluOpCreateTensorDescriptor(&unique_input_desc));
  CHECK_RETURN(interface_name,
               mluOpCreateTensorDescriptor(&unique_output_desc));
  std::vector<int> unique_in_dims = {kernel_volume * input_active_site};
  CHECK_RETURN(interface_name,
               mluOpSetTensorDescriptor(
                   unique_input_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                   unique_in_dims.size(), unique_in_dims.data()));
  CHECK_RETURN(interface_name,
               mluOpSetTensorDescriptor(
                   unique_output_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                   unique_in_dims.size(), unique_in_dims.data()));
  {
    cnnlUniqueSort_t unique_mode = CNNL_SORT_ASCEND;
    cnnlUniqueDescriptor_t unique_desc;

    CALL_CNNL(cnnlCreateUniqueDescriptor(&unique_desc));
    CALL_CNNL(
        cnnlSetUniqueDescriptor(unique_desc, unique_mode, 0, false, false));

    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(unique_input_desc,
                                                 cnnl_input_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(unique_output_desc,
                                                 cnnl_output_desc);
    CALL_CNNL(cnnlUnique_v2(cnnl_handle, unique_desc, cnnl_input_desc,
                            unique_input_addr, unique_workspace_ptr,
                            unique_op_ws, (int *)unique_output_num_addr,
                            cnnl_output_desc, unique_output_addr, nullptr,
                            nullptr, nullptr, nullptr));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);

    CALL_CNNL(cnnlDestroyUniqueDescriptor(unique_desc));
  }
  cnrtQueueSync(handle->queue);
  cnrtMemcpy(return_num_act, unique_output_num_addr, sizeof(float),
             CNRT_MEM_TRANS_DIR_DEV2HOST);
  CHECK_RETURN(interface_name, mluOpDestroyTensorDescriptor(unique_input_desc));
  CHECK_RETURN(interface_name,
               mluOpDestroyTensorDescriptor(unique_output_desc));
  return MLUOP_STATUS_SUCCESS;
}

/*
DefaultKernel2
input: num_act_out
output: step_index
func: generate tensor incluing 0-num_act_out continuously
*/
mluOpStatus_t launchDefaultKernel2(mluOpHandle_t handle,
                                   void *step_index_output_ptr,
                                   int num_act_out) {
  cnrtDim3_t kDim3;
  cnrtFunctionType_t func_type;
  int core_dim = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  int cluster_number = mluop::runtime::getClusterLimitCapability(handle);
  int core_nums = core_dim * cluster_number;
  int nram_size = handle->nram_size + REM_FOR_STACK - 12 * 1024;
  int core_num_l = (nram_size - 4 * 4096 * 3) / sizeof(int);
  int jobs = (num_act_out + core_num_l - 1) / core_num_l;
  int job_num = jobs > core_nums ? core_nums : jobs;
  func_type = CNRT_FUNC_TYPE_BLOCK;
  kDim3.x = 1;
  kDim3.y = job_num;
  kDim3.z = 1;
  VLOG(5) << "[getIndicePairsDefault] Launch kernel "
             "KernelDefaultGetIndicePairKl2<<<U"
          << func_type / core_dim << ", " << kDim3.x << ", " << kDim3.y << ", "
          << kDim3.z << ">>>";
  CHECK_RETURN("[getIndicePairsDefault]",
               KernelDefaultGetIndicePairKl2(kDim3, func_type, handle->queue,
                                             step_index_output_ptr, num_act_out,
                                             core_num_l));
  return MLUOP_STATUS_SUCCESS;
}

/*
BalanceKernel
input: out_indices_expand_ptr
mask : mask_all_ptr
output: out_indices_expand_ptr
func: balance index distribution
*/
mluOpStatus_t launchBalanceKernel(mluOpHandle_t handle,
                                  const std::string interface_name,
                                  void *balance_input_addr,
                                  void *balance_output_addr,
                                  void *balance_mask_addr,
                                  int input_active_site, int kernel_volume,
                                  int output_size) {
  cnrtDim3_t kDim3;
  cnrtFunctionType_t func_type;
  int core_dim = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  int cluster_number = mluop::runtime::getClusterLimitCapability(handle);
  int core_nums = core_dim * cluster_number;
  int nram_size = handle->nram_size + REM_FOR_STACK - 12 * 1024;
  int core_num_l = (nram_size - 4 * 4096 * 3) / 8 / sizeof(int);
  int jobs = (input_active_site * kernel_volume + core_num_l - 1) / core_num_l;
  int job_num = jobs > core_nums ? core_nums : jobs;
  func_type = CNRT_FUNC_TYPE_BLOCK;
  kDim3.x = 1;
  kDim3.y = job_num;
  kDim3.z = 1;
  VLOG(5) << "[getIndicePairsDefault] Launch kernel "
             "KernelBalanceGetIndicePair<<<U"
          << func_type / core_dim << ", " << kDim3.x << ", " << kDim3.y << ", "
          << kDim3.z << ">>>";
  CHECK_RETURN("[getIndicePairsDefault]",
               KernelBalanceGetIndicePair(
                   kDim3, func_type, handle->queue, balance_input_addr,
                   balance_mask_addr, balance_output_addr, input_active_site,
                   kernel_volume, core_num_l, output_size));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t launchFillOp(mluOpHandle_t handle,
                           const std::string interface_name,
                           void *mluOp_fill_addr, int output_size,
                           int fill_value) {
  mluOpTensorDescriptor_t fill_tensor_desc;
  CHECK_RETURN(interface_name, mluOpCreateTensorDescriptor(&fill_tensor_desc));
  std::vector<int> fill_in_dims = {output_size};
  CHECK_RETURN(interface_name,
               mluOpSetTensorDescriptor(fill_tensor_desc, MLUOP_LAYOUT_ARRAY,
                                        MLUOP_DTYPE_INT32, fill_in_dims.size(),
                                        fill_in_dims.data()));
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(fill_tensor_desc,
                                                 cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &fill_value,
                          cnnl_output_desc, mluOp_fill_addr));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  CHECK_RETURN(interface_name, mluOpDestroyTensorDescriptor(fill_tensor_desc));
  return MLUOP_STATUS_SUCCESS;
}

// call scatter_nd op
mluOpStatus_t launchScatterNdOp(mluOpHandle_t handle,
                                const std::string interface_name,
                                void *scatter_output_addr,
                                void *scatter_input_addr,
                                void *scatter_indice_addr, int output_size,
                                int num_act_out) {
  VLOG(5) << interface_name << " call scatterNd";
  cnnlScatterNdMode_t scatter_mode = CNNL_SCATTERND_UPDATE;
  mluOpTensorDescriptor_t scatter_input_desc, scatter_output_desc,
      scatter_indice_desc;
  CHECK_RETURN(interface_name,
               mluOpCreateTensorDescriptor(&scatter_input_desc));
  CHECK_RETURN(interface_name,
               mluOpCreateTensorDescriptor(&scatter_output_desc));
  CHECK_RETURN(interface_name,
               mluOpCreateTensorDescriptor(&scatter_indice_desc));
  std::vector<int> scatter_in_dims = {num_act_out};
  std::vector<int> scatter_out_dims = {output_size};
  std::vector<int> scatter_indice_dims = {num_act_out, 1};
  CHECK_RETURN(interface_name,
               mluOpSetTensorDescriptor(
                   scatter_indice_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                   scatter_indice_dims.size(), scatter_indice_dims.data()));
  CHECK_RETURN(interface_name,
               mluOpSetTensorDescriptor(
                   scatter_input_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                   scatter_in_dims.size(), scatter_in_dims.data()));
  CHECK_RETURN(interface_name,
               mluOpSetTensorDescriptor(
                   scatter_output_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                   scatter_out_dims.size(), scatter_out_dims.data()));
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(scatter_indice_desc,
                                                 cnnl_indices_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(scatter_input_desc,
                                                 cnnl_updates_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(scatter_output_desc,
                                                 cnnl_input_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(scatter_output_desc,
                                                 cnnl_output_desc);

    CALL_CNNL(cnnlScatterNd_v2(
        cnnl_handle, scatter_mode, cnnl_indices_desc, scatter_indice_addr,
        cnnl_updates_desc, scatter_input_addr, cnnl_input_desc,
        scatter_output_addr, cnnl_output_desc, scatter_output_addr));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_indices_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_updates_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  CHECK_RETURN(interface_name,
               mluOpDestroyTensorDescriptor(scatter_input_desc));
  CHECK_RETURN(interface_name,
               mluOpDestroyTensorDescriptor(scatter_output_desc));
  CHECK_RETURN(interface_name,
               mluOpDestroyTensorDescriptor(scatter_indice_desc));
  return MLUOP_STATUS_SUCCESS;
}

// call gather_nd op
mluOpStatus_t launchGatherNdOp(mluOpHandle_t handle,
                               const std::string interface_name,
                               void *gather_input_addr,
                               void *gather_output_addr,
                               void *gather_indice_addr, int input_active_site,
                               int kernel_volume, int output_size) {
  VLOG(5) << interface_name << " call gatherNd";
  mluOpTensorDescriptor_t gather_input_desc, gather_output_desc,
      gather_indice_desc;
  CHECK_RETURN(interface_name, mluOpCreateTensorDescriptor(&gather_input_desc));
  CHECK_RETURN(interface_name,
               mluOpCreateTensorDescriptor(&gather_output_desc));
  CHECK_RETURN(interface_name,
               mluOpCreateTensorDescriptor(&gather_indice_desc));
  std::vector<int> gather_in_dims = {output_size};
  std::vector<int> gather_indices_dims = {input_active_site * kernel_volume, 1};
  std::vector<int> gather_out_dims = {input_active_site * kernel_volume};
  CHECK_RETURN(interface_name,
               mluOpSetTensorDescriptor(
                   gather_indice_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                   gather_indices_dims.size(), gather_indices_dims.data()));
  CHECK_RETURN(interface_name,
               mluOpSetTensorDescriptor(
                   gather_input_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                   gather_in_dims.size(), gather_in_dims.data()));
  CHECK_RETURN(interface_name,
               mluOpSetTensorDescriptor(
                   gather_output_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                   gather_out_dims.size(), gather_out_dims.data()));
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(gather_input_desc,
                                                 cnnl_params_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(gather_indice_desc,
                                                 cnnl_indices_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(gather_output_desc,
                                                 cnnl_output_desc);
    CALL_CNNL(cnnlGatherNd(cnnl_handle, cnnl_params_desc, gather_input_addr,
                           cnnl_indices_desc, gather_indice_addr,
                           cnnl_output_desc, gather_output_addr));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_params_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_indices_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  CHECK_RETURN(interface_name, mluOpDestroyTensorDescriptor(gather_input_desc));
  CHECK_RETURN(interface_name,
               mluOpDestroyTensorDescriptor(gather_output_desc));
  CHECK_RETURN(interface_name,
               mluOpDestroyTensorDescriptor(gather_indice_desc));
  return MLUOP_STATUS_SUCCESS;
}

/* DefaultKernel3
input: tensor1: kl  int32  indice_index_in
       tensor2: kl  int32  indice_index_out
       tensor3: kl  int32  mask
output: tensor: k2l
func: maskmove efficient data continuously by collect insts
*/
mluOpStatus_t launchDefaultKernel3(mluOpHandle_t handle, void *output_addr,
                                   void *input_addr, void *mask_addr,
                                   int input_active_site, int kernel_volume) {
  cnrtDim3_t kDim3;
  cnrtFunctionType_t func_type;
  int core_dim = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  int cluster_number = mluop::runtime::getClusterLimitCapability(handle);
  int core_nums = core_dim * cluster_number;
  int nram_size = handle->nram_size + REM_FOR_STACK - 12 * 1024;
  int core_num_l = (nram_size - 4 * 4096 * 3) / 4 / sizeof(int);
  int jobs = 2 * kernel_volume;
  int job_num = jobs > core_nums ? core_nums : jobs;
  func_type = CNRT_FUNC_TYPE_BLOCK;
  kDim3.x = 1;
  kDim3.y = job_num;
  kDim3.z = 1;
  VLOG(5) << "[getIndicePairsDefault] Launch kernel "
             "KernelDefaultGetIndicePairKl3<<<U"
          << func_type / core_dim << ", " << kDim3.x << ", " << kDim3.y << ", "
          << kDim3.z << ">>>";
  CHECK_RETURN("[getIndicePairsDefault]",
               KernelDefaultGetIndicePairKl3(
                   kDim3, func_type, handle->queue, output_addr, input_addr,
                   mask_addr, input_active_site, kernel_volume, core_num_l));
  return MLUOP_STATUS_SUCCESS;
}

/*
DefaultKernel4
input: tensor  num_act_out    int
output: tensor num_act_out,4  int
func: generate tensor incluing 0-num_act_out continuously
*/
mluOpStatus_t launchDefaultKernel4(
    mluOpHandle_t handle,
    const mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
    void *output_addr, void *input_addr, int num_act_out) {
  cnrtDim3_t kDim3;
  cnrtFunctionType_t func_type;
  int core_dim = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  int cluster_number = mluop::runtime::getClusterLimitCapability(handle);
  int core_nums = core_dim * cluster_number;
  int nram_size = handle->nram_size + REM_FOR_STACK - 12 * 1024;
  int core_num_split = 0;
  if (handle->arch >= MLUOP_MLU590) {
    core_num_split = 14;
  } else {
    core_num_split = 15;
  }
  int core_num_l = (nram_size - 4 * 4096 * 3) / core_num_split / sizeof(int);
  int jobs = (num_act_out + core_num_l - 1) / core_num_l;
  int job_num = jobs > core_nums ? core_nums : jobs;
  func_type = CNRT_FUNC_TYPE_BLOCK;
  kDim3.x = 1;
  kDim3.y = job_num;
  kDim3.z = 1;
  OutputSpace output_space(sparse_conv_desc->output_space[0],
                           sparse_conv_desc->output_space[1],
                           sparse_conv_desc->output_space[2]);

  VLOG(5) << "[getIndicePairsDefault] Launch kernel "
             "KernelDefaultGetIndicePairKl4<<<U"
          << func_type / core_dim << ", " << kDim3.x << ", " << kDim3.y << ", "
          << kDim3.z << ">>>";
  CHECK_RETURN("[getIndicePairsDefault]",
               KernelDefaultGetIndicePairKl4(
                   kDim3, func_type, handle->queue, output_addr, input_addr,
                   output_space, num_act_out, core_num_l));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t NormalGetIndicePairsKernel(
    mluOpHandle_t handle, const std::string interface_name,
    mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
    const mluOpTensorDescriptor_t indices_desc, const void *indices,
    void *workspace, const mluOpTensorDescriptor_t indice_pairs_desc,
    void *indice_pairs, const mluOpTensorDescriptor_t out_indices_desc,
    void *out_indices, const mluOpTensorDescriptor_t indice_num_desc,
    void *indice_num) {
  int sub_m = sparse_conv_desc->sub_m;
  int batch = sparse_conv_desc->batch;
  int kernel_volume = indice_pairs_desc->dims[0];
  int input_active_site = indice_pairs_desc->dims[2];
  int output_size = batch * sparse_conv_desc->output_space[0] *
                        sparse_conv_desc->output_space[1] *
                        sparse_conv_desc->output_space[2] +
                    1;

  if (sub_m) {
    /*  workspace for subm mode
    | mask_all |indices_index_in | indices_index_out/ step_index |
    indices_in_expand |out_indices_expand| | max(grid_out, reduce_op_ws)|
    */
    size_t mask_all_ws = 0, indice_index_in_ws = 0, indice_index_out_ws = 0;
    size_t indice_in_expand_ws = 0, out_indices_expand_ws = 0, grid_out_ws = 0;
    size_t reduce_op_ws = 0;
    CHECK_RETURN(interface_name,
                 getIndiceMaskAll(indice_pairs_desc, kernel_volume,
                                  input_active_site, &mask_all_ws));
    CHECK_RETURN(interface_name,
                 getIndiceIndexIn(indice_pairs_desc, kernel_volume,
                                  input_active_site, &indice_index_in_ws));
    CHECK_RETURN(interface_name,
                 getIndiceIndexOut(indice_pairs_desc, kernel_volume,
                                   input_active_site, &indice_index_out_ws));
    CHECK_RETURN(interface_name,
                 getIndiceOutExpand(indice_pairs_desc, kernel_volume,
                                    input_active_site, &out_indices_expand_ws));
    CHECK_RETURN(interface_name,
                 getIndiceInExpand(indice_pairs_desc, input_active_site,
                                   &indice_in_expand_ws));
    CHECK_RETURN(interface_name,
                 getGridOut(indice_pairs_desc, output_size, &grid_out_ws));
    CHECK_RETURN(interface_name,
                 getReduceOpWS(handle, interface_name, kernel_volume,
                               input_active_site, &reduce_op_ws));
    const void *compute_indices_ptr = indices;
    void *mask_all_ptr = (void *)((int8_t *)workspace);
    void *indice_index_in_ptr = (void *)((int8_t *)workspace + mask_all_ws);
    void *indice_in_expand_ptr =
        (void *)((int8_t *)workspace + mask_all_ws + indice_index_in_ws +
                 indice_index_out_ws);
    void *out_indices_expand_ptr =
        (void *)((int8_t *)workspace + mask_all_ws + indice_index_in_ws +
                 indice_index_out_ws + indice_in_expand_ws);
    CHECK_RETURN(
        interface_name,
        launchSubmKernel1(handle, sparse_conv_desc, compute_indices_ptr,
                          mask_all_ptr, indice_index_in_ptr,
                          indice_in_expand_ptr, out_indices_expand_ptr, batch,
                          kernel_volume, input_active_site));

    // call launchDefaultKernel2   gen step_index
    void *step_index_addr = NULL;
    step_index_addr = (void *)((int8_t *)(int8_t *)workspace + mask_all_ws +
                               indice_index_in_ws);
    CHECK_RETURN(interface_name, launchDefaultKernel2(handle, step_index_addr,
                                                      input_active_site));

    // call scatter_nd unique_output_addr + step_index_addr = grid_out_addr
    void *scatter_input_addr = NULL, *scatter_output_addr = NULL,
         *scatter_indice_addr = NULL;
    scatter_input_addr = step_index_addr;
    scatter_indice_addr = indice_in_expand_ptr;
    scatter_output_addr = (void *)((int8_t *)workspace + mask_all_ws +
                                   indice_index_in_ws + indice_index_out_ws +
                                   indice_in_expand_ws + out_indices_expand_ws);
    int fill_value = -1;
    CHECK_RETURN(interface_name,
                 launchFillOp(handle, interface_name, scatter_output_addr,
                              output_size, fill_value));
    CHECK_RETURN(interface_name,
                 launchScatterNdOp(handle, interface_name, scatter_output_addr,
                                   scatter_input_addr, scatter_indice_addr,
                                   output_size, input_active_site));

    // call gather_nd out_indices_expand + grid_out_addr = indice_index_out
    void *gather_input_addr = NULL, *gather_output_addr = NULL,
         *gather_indice_addr = NULL;
    gather_output_addr =
        (void *)((int8_t *)workspace + mask_all_ws + indice_index_in_ws);
    gather_input_addr = scatter_output_addr;
    gather_indice_addr = out_indices_expand_ptr;
    CHECK_RETURN(
        interface_name,
        launchGatherNdOp(handle, interface_name, gather_input_addr,
                         gather_output_addr, gather_indice_addr,
                         input_active_site, kernel_volume, output_size));

    // call sumb_kernel2 indice_index_out and  mask_all = mask_all
    // get out_indices from indices
    const void *kernel2_input1_addr = NULL;
    void *kernel2_input2_addr = NULL, *kernel2_output1_addr = NULL,
         *kernel2_output2_addr = NULL;
    kernel2_input1_addr = indices;
    kernel2_input2_addr = gather_output_addr;
    kernel2_output1_addr = mask_all_ptr;
    kernel2_output2_addr = out_indices;
    CHECK_RETURN(
        interface_name,
        launchSubmKernel2(handle, kernel2_input1_addr, kernel2_input2_addr,
                          kernel2_output1_addr, kernel2_output2_addr,
                          kernel_volume, input_active_site));

    // call reduceOp
    void *reduce_input_addr = NULL, *reduce_output_addr = NULL;
    reduce_input_addr = mask_all_ptr;
    reduce_output_addr = indice_num;
    void *reduce_workspace_ptr = NULL;
    if (reduce_op_ws > 0) {
      reduce_workspace_ptr =
          (void *)((int8_t *)workspace + mask_all_ws + indice_index_in_ws +
                   indice_index_out_ws + indice_in_expand_ws +
                   out_indices_expand_ws);
    }
    CHECK_RETURN(
        interface_name,
        launchReduceOp(handle, interface_name, reduce_output_addr,
                       reduce_input_addr, reduce_workspace_ptr, reduce_op_ws,
                       kernel_volume, input_active_site));

    // call launchDefaultKernel3 l k partition and sort
    void *kernel3_input_addr = NULL, *kernel3_output_addr = NULL,
         *kernel3_mask_addr = NULL;
    kernel3_input_addr = indice_index_in_ptr;
    kernel3_output_addr = indice_pairs;
    kernel3_mask_addr = mask_all_ptr;
    fill_value = -1;
    CHECK_RETURN(
        interface_name,
        launchFillOp(handle, interface_name, indice_pairs,
                     kernel_volume * 2 * input_active_site, fill_value));
    CHECK_RETURN(interface_name,
                 launchDefaultKernel3(handle, kernel3_output_addr,
                                      kernel3_input_addr, kernel3_mask_addr,
                                      input_active_site, kernel_volume));
  } else {
    /* workspace for default mode
    | mask_all | indices_index_in | step_index/ indices_index_out |
    out_indices_expand  | | out_indices_unique | max(grid_out_ws, reduce_ws,
    unique_ws) |
    */
    size_t mask_all_ws = 0, indice_index_in_ws = 0, indice_index_out_ws = 0;
    size_t out_indices_expand_ws = 0, indice_unique_ws = 0, grid_out_ws = 0;
    size_t reduce_op_ws = 0, unique_op_ws = 0;

    CHECK_RETURN(interface_name,
                 getIndiceMaskAll(indice_pairs_desc, kernel_volume,
                                  input_active_site, &mask_all_ws));
    CHECK_RETURN(interface_name,
                 getIndiceIndexIn(indice_pairs_desc, kernel_volume,
                                  input_active_site, &indice_index_in_ws));
    CHECK_RETURN(interface_name,
                 getIndiceIndexOut(indice_pairs_desc, kernel_volume,
                                   input_active_site, &indice_index_out_ws));
    CHECK_RETURN(interface_name,
                 getIndiceOutExpand(indice_pairs_desc, kernel_volume,
                                    input_active_site, &out_indices_expand_ws));
    CHECK_RETURN(interface_name,
                 getIndiceUnique(indice_pairs_desc, kernel_volume,
                                 input_active_site, &indice_unique_ws));
    CHECK_RETURN(interface_name,
                 getGridOut(indice_pairs_desc, output_size, &grid_out_ws));
    CHECK_RETURN(interface_name,
                 getReduceOpWS(handle, interface_name, kernel_volume,
                               input_active_site, &reduce_op_ws));
    CHECK_RETURN(
        interface_name,
        getUniqueOpWS(handle, interface_name, indices_desc, kernel_volume,
                      input_active_site, &unique_op_ws));
    const void *compute_indices_ptr = indices;
    void *mask_all_ptr = (void *)((int8_t *)workspace);
    void *indice_index_in_ptr = (void *)((int8_t *)workspace + mask_all_ws);
    void *out_indices_expand_ptr =
        (void *)((int8_t *)workspace + mask_all_ws + indice_index_out_ws +
                 indice_index_in_ws);
    CHECK_RETURN(interface_name,
                 launchDefaultKernel1(
                     handle, sparse_conv_desc, compute_indices_ptr,
                     mask_all_ptr, indice_index_in_ptr, out_indices_expand_ptr,
                     batch, kernel_volume, input_active_site));

    //  call reduce_sum mask_all to indice_num
    void *reduce_input_addr = NULL, *reduce_output_addr = NULL;
    reduce_input_addr = mask_all_ptr;
    reduce_output_addr = indice_num;
    void *reduce_workspace_ptr = NULL;
    if (reduce_op_ws > 0) {
      reduce_workspace_ptr = (void *)((int8_t *)workspace + mask_all_ws +
                                      indice_index_in_ws + indice_index_out_ws +
                                      out_indices_expand_ws + indice_unique_ws);
    }
    CHECK_RETURN(
        interface_name,
        launchReduceOp(handle, interface_name, reduce_output_addr,
                       reduce_input_addr, reduce_workspace_ptr, reduce_op_ws,
                       kernel_volume, input_active_site));

    // call unique_v2 out_indices_expand_ptr indice_unique_ws_ptr
    int num_act_out = 0;
    void *unique_input_addr = NULL, *unique_output_addr = NULL,
         *unique_output_num_addr = NULL;
    unique_input_addr = out_indices_expand_ptr;
    unique_output_addr =
        (void *)((int8_t *)workspace + mask_all_ws + indice_index_in_ws +
                 indice_index_out_ws + out_indices_expand_ws);
    unique_output_num_addr =
        (void *)((int8_t *)workspace + mask_all_ws + indice_index_in_ws);
    void *unique_workspace_ptr = NULL;
    if (unique_op_ws > 0) {
      unique_workspace_ptr = (void *)((int8_t *)workspace + mask_all_ws +
                                      indice_index_in_ws + indice_index_out_ws +
                                      out_indices_expand_ws + indice_unique_ws);
    }
    CHECK_RETURN(
        interface_name,
        launchUniqueOp(handle, interface_name, unique_output_addr,
                       unique_input_addr, unique_output_num_addr,
                       unique_workspace_ptr, unique_op_ws, kernel_volume,
                       input_active_site, &num_act_out));

    if (num_act_out != kernel_volume * input_active_site) {
      num_act_out = num_act_out - 1;
    }
    if (num_act_out <= 0) {
      // fill indice_pairs -1 indice_num 0
      int fill_value = -1;
      CHECK_RETURN(
          interface_name,
          launchFillOp(handle, interface_name, indice_pairs,
                       kernel_volume * 2 * input_active_site, fill_value));
      fill_value = 0;
      CHECK_RETURN(interface_name,
                   launchFillOp(handle, interface_name, indice_num,
                                kernel_volume, fill_value));
      return MLUOP_STATUS_SUCCESS;
    }
    sparse_conv_desc->num_act_out = num_act_out;
    // call launchDefaultKernel2   gen step_index
    void *step_index_addr = NULL;
    step_index_addr =
        (void *)((int8_t *)workspace + mask_all_ws + indice_index_in_ws);
    CHECK_RETURN(interface_name,
                 launchDefaultKernel2(handle, step_index_addr, num_act_out));

    // call balance out_indices_expand_ptr distr
    void *balance_input_addr = NULL, *balance_output_addr = NULL,
         *balance_mask_addr = NULL;
    balance_input_addr = out_indices_expand_ptr;
    balance_output_addr = out_indices_expand_ptr;
    balance_mask_addr = mask_all_ptr;
    CHECK_RETURN(
        interface_name,
        launchBalanceKernel(handle, interface_name, balance_input_addr,
                            balance_output_addr, balance_mask_addr,
                            input_active_site, kernel_volume, output_size));

    // call scatter_nd unique_output_addr + step_index_addr = grid_out_addr
    void *scatter_input_addr = NULL, *scatter_output_addr = NULL,
         *scatter_indice_addr = NULL;
    scatter_input_addr = step_index_addr;
    scatter_indice_addr = unique_output_addr;
    scatter_output_addr = (void *)((int8_t *)workspace + mask_all_ws +
                                   indice_index_in_ws + indice_index_out_ws +
                                   out_indices_expand_ws + indice_unique_ws);
    int fill_value = -1;
    CHECK_RETURN(interface_name,
                 launchFillOp(handle, interface_name, scatter_output_addr,
                              output_size, fill_value));
    CHECK_RETURN(interface_name,
                 launchScatterNdOp(handle, interface_name, scatter_output_addr,
                                   scatter_input_addr, scatter_indice_addr,
                                   output_size, num_act_out));

    // call gather_nd out_indices_expand + grid_out_addr = indice_index_out
    void *gather_input_addr = NULL, *gather_output_addr = NULL,
         *gather_indice_addr = NULL;
    gather_output_addr =
        (void *)((int8_t *)workspace + mask_all_ws + indice_index_in_ws);
    gather_input_addr = scatter_output_addr;
    gather_indice_addr = out_indices_expand_ptr;
    CHECK_RETURN(
        interface_name,
        launchGatherNdOp(handle, interface_name, gather_input_addr,
                         gather_output_addr, gather_indice_addr,
                         input_active_site, kernel_volume, output_size));

    // call launchDefaultKernel3 l k partition and sort
    void *kernel3_input_addr = NULL, *kernel3_output_addr = NULL,
         *kernel3_mask_addr = NULL;
    kernel3_input_addr = indice_index_in_ptr;
    kernel3_output_addr = indice_pairs;
    kernel3_mask_addr = mask_all_ptr;
    fill_value = -1;
    CHECK_RETURN(
        interface_name,
        launchFillOp(handle, interface_name, indice_pairs,
                     kernel_volume * 2 * input_active_site, fill_value));
    CHECK_RETURN(interface_name,
                 launchDefaultKernel3(handle, kernel3_output_addr,
                                      kernel3_input_addr, kernel3_mask_addr,
                                      input_active_site, kernel_volume));

    // get out_indices from indice unique
    void *kernel4_output_addr = NULL, *kernel4_input_addr = NULL;
    kernel4_input_addr = unique_output_addr;
    kernel4_output_addr = out_indices;
    CHECK_RETURN(
        interface_name,
        launchDefaultKernel4(handle, sparse_conv_desc, kernel4_output_addr,
                             kernel4_input_addr, num_act_out));
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t normalGetIndicePairs(
    mluOpHandle_t handle, const std::string interface_name,
    mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
    const mluOpTensorDescriptor_t indices_desc, const void *indices,
    void *workspace, size_t workspace_size,
    const mluOpTensorDescriptor_t indice_pairs_desc, void *indice_pairs,
    const mluOpTensorDescriptor_t out_indices_desc, void *out_indices,
    const mluOpTensorDescriptor_t indice_num_desc, void *indice_num,
    const bool is_get_workspace, size_t *return_ws) {
  if (is_get_workspace) {
    return getNormalGetIndicePairsWorkspaceSize(
        handle, interface_name, sparse_conv_desc, indices_desc,
        indice_pairs_desc, out_indices_desc, indice_num_desc, return_ws);
  } else {
    return NormalGetIndicePairsKernel(
        handle, interface_name, sparse_conv_desc, indices_desc, indices,
        workspace, indice_pairs_desc, indice_pairs, out_indices_desc,
        out_indices, indice_num_desc, indice_num);
  }
}
