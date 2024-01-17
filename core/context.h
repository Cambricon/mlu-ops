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
#ifndef CORE_CONTEXT_H_
#define CORE_CONTEXT_H_

#include <string>
#include "mlu_op.h"
#include "cn_api.h"
#include "core/logging.h"

#define CONTEXT_DEVICENAME_BUFFER_SIZE 64
#define CONTEXT_DEVICENAME_LEAST_SIZE 6

// Tested version dependency: See MLUOP Release Note.
// Compatible with higher version CNRT/CNDRV by default.
#define MLUOP_DEP_CNRT_MIN_MAJOR 6
#define MLUOP_DEP_CNRT_MIN_MINOR 7
#define MLUOP_DEP_CNRT_MIN_PATCH 0

#define MLUOP_DEP_CNRT_MAX_MAJOR 999
#define MLUOP_DEP_CNRT_MAX_MINOR 999
#define MLUOP_DEP_CNRT_MAX_PATCH 999

#define MLUOP_DEP_CNDRV_MIN_MAJOR 2
#define MLUOP_DEP_CNDRV_MIN_MINOR 7
#define MLUOP_DEP_CNDRV_MIN_PATCH 0

#define MLUOP_DEP_CNDRV_MAX_MAJOR 999
#define MLUOP_DEP_CNDRV_MAX_MINOR 999
#define MLUOP_DEP_CNDRV_MAX_PATCH 999

// handle->arch
typedef enum {
  MLUOP_UNKNOWN_DEVICE = 0,
  // MLUOP_MLU100 = 100,
  MLUOP_MLU220 = 220,
  MLUOP_MLU270 = 270,
  MLUOP_MLU370 = 372,
  MLUOP_MLU590 = 592,
  MLUOP_MLU290 = 290,
} mluOpDevType_t;

// for handle->arch
struct deviceName {
  char name[CONTEXT_DEVICENAME_BUFFER_SIZE];
  mluOpDevType_t type;
};

struct mluOpContext {
  CNdev device;
  cnrtQueue_t queue;
  mluOpDevType_t arch;  // return arch type. e.g. MLUOP_MLU270
  /* device_name e.g.
   * "MLU590-M9"
   * "MLU370-X8", "MLU370-X4", "MLU370-S4", "MLU370-M8"
   * can also use printf(printf("name = %c\n", handle->device_name)) if have a
   * new MLUXXX. */
  char device_name[CONTEXT_DEVICENAME_BUFFER_SIZE] = "";
  int32_t cluster_num;
  int32_t core_num_per_cluster;
  int32_t nram_size;
  int32_t wram_size;
  int32_t sram_size;
  int32_t capability_cluster_num;
  int32_t capability_job_limit;  // the max job type you can launch, e.g.
                                 // CN_KERNEL_CLASS_UNION1
  int32_t clock_rate;  // the mlu clock frequency in kilohertz. 0 means that the
                       // frequency cannot be obtained on the current device
  int32_t l2cache_size;                // the size of L2 cache in bytes
  int32_t persisting_l2cache_maxsize;  // the maximum persisting cache size of
                                       // L2 cache in bytes
  double memory_band_width;            // the memory bandwidth in GB/s
  mluOpQuantizeRoundMode_t round_mode;
  mluOpAtomicsMode_t atomics_mode;
  int32_t getJobNum(cnrtFunctionType_t function_type) {
    switch (function_type) {
      default:
        return 0;
      case CNRT_FUNC_TYPE_BLOCK:
        return job_num[0];
      case CNRT_FUNC_TYPE_UNION1:
        return job_num[1];
      case CNRT_FUNC_TYPE_UNION2:
        return job_num[2];
      case CNRT_FUNC_TYPE_UNION4:
        return job_num[3];
      case CNRT_FUNC_TYPE_UNION8:
        return job_num[4];
      case CNRT_FUNC_TYPE_UNION16:
        return job_num[5];
    }
  }
  mluOpStatus_t initJobNum(const CNcontext drv_ctx,
                           const std::string& api_name) {
    int number = -1;
    INTERNAL_CHECK(api_name,
                   CN_SUCCESS == cnGetCtxMaxParallelUnionTasks(
                                     drv_ctx, CN_KERNEL_CLASS_BLOCK, &number));
    job_num[0] = number;
    INTERNAL_CHECK(api_name,
                   CN_SUCCESS == cnGetCtxMaxParallelUnionTasks(
                                     drv_ctx, CN_KERNEL_CLASS_UNION, &number));
    job_num[1] = number;
    INTERNAL_CHECK(api_name,
                   CN_SUCCESS == cnGetCtxMaxParallelUnionTasks(
                                     drv_ctx, CN_KERNEL_CLASS_UNION2, &number));
    job_num[2] = number;
    INTERNAL_CHECK(api_name,
                   CN_SUCCESS == cnGetCtxMaxParallelUnionTasks(
                                     drv_ctx, CN_KERNEL_CLASS_UNION4, &number));
    job_num[3] = number;
    INTERNAL_CHECK(api_name,
                   CN_SUCCESS == cnGetCtxMaxParallelUnionTasks(
                                     drv_ctx, CN_KERNEL_CLASS_UNION8, &number));
    job_num[4] = number;
    INTERNAL_CHECK(
        api_name, CN_SUCCESS == cnGetCtxMaxParallelUnionTasks(
                                    drv_ctx, CN_KERNEL_CLASS_UNION16, &number));
    job_num[5] = number;
    return MLUOP_STATUS_SUCCESS;
  }

 private:
  int32_t job_num[6] = {0};
};

typedef enum {
  WARNING = 1,
  ERROR = 2,
} DepCheckLevel;  // related to core/cnlog.hpp

mluOpStatus_t mluOpCheckDependency(bool need_check_min = true,
                                   bool need_check_max = false,
                                   DepCheckLevel level = WARNING);

#endif  // CORE_CONTEXT_H_
