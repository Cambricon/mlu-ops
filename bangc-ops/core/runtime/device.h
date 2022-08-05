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
#ifndef CORE_RUNTIME_DEVICE_H_
#define CORE_RUNTIME_DEVICE_H_

#include <pthread.h>
#include <string>
#include "core/mlu_op_core.h"
#include "cn_api.h"
#include "core/context.h"
#include "core/tensor.h"
#include "core/type.h"

typedef void *MLUaddr;
typedef void *HOSTaddr;

namespace mluop {
namespace runtime {

#define DEVICE_NAME_LENGTH (64)
inline int32_t getNumOfUnionCapability(mluOpHandle_t handle) {
  return handle->cluster_num;
}
inline int32_t getCoreNumOfEachUnionCapability(mluOpHandle_t handle) {
  return handle->core_num_per_cluster;
}
inline int32_t getNramSizeInBytes(mluOpHandle_t handle) {
  return handle->nram_size;
}
inline int32_t getWramSizeInBytes(mluOpHandle_t handle) {
  return handle->wram_size;
}
inline int32_t getSramSizeInBytes(mluOpHandle_t handle) {
  return handle->sram_size;
}
inline int32_t getClusterLimitCapability(mluOpHandle_t handle) {
  return handle->capability_cluster_num;
}
inline int32_t getJobLimitCapability(mluOpHandle_t handle) {
  return handle->capability_job_limit;
}
}  // namespace runtime
}  // namespace mluop

#endif  // CORE_RUNTIME_DEVICE_H_
