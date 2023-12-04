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
#include "kernels/utils/cnnl_helper.h"

mluOpStatus_t MLUOP_WIN_API mluOpFill(mluOpHandle_t handle, float value,
                                      const mluOpTensorDescriptor_t output_desc,
                                      void *output) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpFill] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpFill", handle != NULL);
  PARAM_CHECK("mluOpFill", output_desc != NULL);
  PARAM_CHECK("mluOpFill", output != NULL);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, cnnl_output_desc);
  CHECK_FUNC_RETURN(cnnlFill(cnnl_handle, value, cnnl_output_desc, output),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpFill] Internal error accured in mluOpFill.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpFill_v2(mluOpHandle_t handle, const mluOpTensorDescriptor_t value_desc,
             const void *value, const mluOpTensorDescriptor_t output_desc,
             void *output) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpFill_v2] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpFill_v2", handle != NULL);
  PARAM_CHECK("mluOpFill_v2", value_desc != NULL);
  PARAM_CHECK("mluOpFill_v2", value != NULL);
  PARAM_CHECK("mluOpFill_v2", output_desc != NULL);
  PARAM_CHECK("mluOpFill_v2", output != NULL);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(value_desc, cnnl_value_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, cnnl_output_desc);
  CHECK_FUNC_RETURN(cnnlFill_v2(cnnl_handle, cnnl_value_desc, value,
                                cnnl_output_desc, output),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpFill_v2] Internal error accured in mluOpFill_v2.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_value_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpFill_v3(mluOpHandle_t handle, const mluOpPointerMode_t pointer_mode,
             const void *value, const mluOpTensorDescriptor_t output_desc,
             void *output) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpFill_v3] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpFill_v3", handle != NULL);
  PARAM_CHECK_GE("mluOpFill_v3", pointer_mode, 0);
  PARAM_CHECK("mluOpFill_v3", value != NULL);
  PARAM_CHECK("mluOpFill_v3", output_desc != NULL);
  PARAM_CHECK("mluOpFill_v3", output != NULL);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, cnnl_output_desc);
  CHECK_FUNC_RETURN(
      cnnlFill_v3(cnnl_handle, cnnlPointerMode_t(int(pointer_mode)), value,
                  cnnl_output_desc, output),
      CNNL_STATUS_SUCCESS,
      "[mluOpFill_v3] Internal error accured in mluOpFill_v3.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
