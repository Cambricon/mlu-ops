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
#include "chirpz.h"
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/unary_op/unary_op_host.h"

static void ChirpzPolicyFunc(const mluOpHandle_t &handle,
                               cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  // *k_type = cnrtFuncTypeBlock;
  // k_dim->x = 1;
  // k_dim->y = 1;
  // k_dim->z = 1;


  *k_type = cnrtFuncTypeUnion1;
  k_dim->x = 4;
  k_dim->y = 1;
  k_dim->z = 1;

  // *k_type = cnrtFuncTypeUnion1;
  // k_dim->x = handle->core_num_per_cluster;
  // k_dim->y = mluop::runtime::getClusterLimitCapability(handle);
  // k_dim->z = 1;
}

mluOpStatus_t MLUOP_WIN_API
mluOpChirpz(mluOpHandle_t handle, const int length, const int n, int pad_n, int type, bool chirpz,
              const mluOpTensorDescriptor_t output_desc, void *output) {

  // policy select
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  ChirpzPolicyFunc(handle, &k_dim, &k_type);
  VLOG(5) << "[ChirpzPolicyFunc] launch kernel policyFUnc[" << k_dim.x << ", "
          << k_dim.y << ", " << k_dim.z << "]";

  VLOG(5) << "kernel ChirpzPolicyFunc.";
  CHECK_RETURN("[ChirpzPolicyFunc] ", KernelChirpz(k_dim, k_type, handle->queue,
                                                  length, n, pad_n, type, chirpz, output));
  return MLUOP_STATUS_SUCCESS;
}
