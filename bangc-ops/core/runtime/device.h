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
#include "cn_api.h"
#include "core/context.h"
#include "core/tensor.h"
#include "core/type.h"
#include "mlu_op.h"

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

/******************************************************************************
 * mluOp FUNC: getCoreNumOfJobLimitCapability
 * get mlu core number of every single CNRT_FUNC_TYPE with maximum job capacity.
 * param 'handle' is the handle of mluOpHandle_t.
 ******************************************************************************/
inline int32_t getCoreNumOfJobLimitCapability(mluOpHandle_t handle) {
  switch (handle->capability_job_limit) {
    default:
      return handle->core_num_per_cluster * handle->capability_job_limit;
    case CN_KERNEL_CLASS_BLOCK:
      return 1;
    case CN_KERNEL_CLASS_UNION:
      return handle->core_num_per_cluster;
    case CN_KERNEL_CLASS_UNION2:
      return handle->core_num_per_cluster * 2;
    case CN_KERNEL_CLASS_UNION4:
      return handle->core_num_per_cluster * 4;
    case CN_KERNEL_CLASS_UNION8:
      return handle->core_num_per_cluster * 8;
    case CN_KERNEL_CLASS_UNION16:
      return handle->core_num_per_cluster * 16;
  }
}

/******************************************************************************
 * mluOp FUNC: getClusterNumOfJobLimitCapability
 * get max cluster number of current job capacity.
 * param 'handle' is the handle of mluOpHandle_t.
 ******************************************************************************/
inline int32_t getClusterNumberOfJobLimitCapability(mluOpHandle_t handle) {
  switch (handle->capability_job_limit) {
    default:
      return getCoreNumOfJobLimitCapability(handle) /
             handle->core_num_per_cluster;
    case CN_KERNEL_CLASS_BLOCK:
      return 1;
    case CN_KERNEL_CLASS_UNION:
      return 1;
    case CN_KERNEL_CLASS_UNION2:
      return 2;
    case CN_KERNEL_CLASS_UNION4:
      return 4;
    case CN_KERNEL_CLASS_UNION8:
      return 8;
    case CN_KERNEL_CLASS_UNION16:
      return 16;
  }
}

/******************************************************************************
 * mluOp FUNC: castCnKernelClassToCnrtFuncType
 * cast KernelClass type into cnrtFunctionType_t
 * param 'jobType' is job type of KernelClass.
 ******************************************************************************/
inline cnrtFunctionType_t castCnKernelClassToCnrtFuncType(KernelClass jobType) {
  switch (jobType) {
    default:
      return CNRT_FUNC_TYPE_MUTABLE;
    case CN_KERNEL_CLASS_BLOCK:
      return CNRT_FUNC_TYPE_BLOCK;
    case CN_KERNEL_CLASS_UNION:
      return CNRT_FUNC_TYPE_UNION1;
    case CN_KERNEL_CLASS_UNION2:
      return CNRT_FUNC_TYPE_UNION2;
    case CN_KERNEL_CLASS_UNION4:
      return CNRT_FUNC_TYPE_UNION4;
    case CN_KERNEL_CLASS_UNION8:
      return CNRT_FUNC_TYPE_UNION8;
    case CN_KERNEL_CLASS_UNION16:
      return CNRT_FUNC_TYPE_UNION16;
  }
}

// get the max cnrtFunctionType on current CNdevice.
inline cnrtFunctionType_t getJobLimitCapabilityCnrtFuncType(
    mluOpHandle_t handle) {
  KernelClass job_type =
      static_cast<KernelClass>(getJobLimitCapability(handle));
  return castCnKernelClassToCnrtFuncType(job_type);
}

// get the max parallel job num of certain cnrtFunctionType on current CNdevice.
inline int getMaxParallelJobNum(mluOpHandle_t handle, cnrtFunctionType_t type) {
  return handle->getJobNum(type);
}

}  // namespace runtime
}  // namespace mluop

#endif  // CORE_RUNTIME_DEVICE_H_
