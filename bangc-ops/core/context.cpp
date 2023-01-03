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
#include "cstring"

#include "core/context.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/tool.h"
#include "kernels/kernel.h"

#define DEP_CHECK_LOG(level)                                                 \
  cnlog::LogMessage(__FILE__, __LINE__, 4, level, "MLUOP", true, true, true, \
                    true)                                                    \
      .stream()

// see cnrt_function.c deviceCoreVersion for more info.
struct deviceName name_list_table[] = {
    {"MLU270", MLUOP_MLU270},
    {"MLU220", MLUOP_MLU220},
    {"MLU220 SOC", MLUOP_MLU220},
    {"MLU290", MLUOP_MLU290},
    {"MLU370", MLUOP_MLU370},
    {"MLU590", MLUOP_MLU590}
    // mluOp not support mlu100 only for error case.
    // {"MLU100", MLUOP_MLU100},
};

// update this funciton.
mluOpDevType_t convertDeviceName(char *name) {
  struct deviceName *pName = NULL;
  int num = sizeof(name_list_table) / sizeof(struct deviceName);
  if (CONTEXT_DEVICENAME_LEAST_SIZE > strlen(name)) {
    LOG(ERROR) << "get device name failed. device name too short. "
                  "device name = "
               << name << "\n";
    return MLUOP_UNKNOWN_DEVICE;
  }
  for (int i = 0; i < num; i++) {
    pName = &name_list_table[i];
    if (0 == strncmp(pName->name, name, strlen(pName->name))) {
      return pName->type;
    }
  }
  LOG(ERROR) << "get device name failed. return unknown device. device name = "
             << name << "\n";
  return MLUOP_UNKNOWN_DEVICE;
}

mluOpStatus_t mluOpCheckDependency(bool need_check_min, bool need_check_max,
                                   DepCheckLevel level) {
  int cnrt_major = 0, cnrt_minor = 0, cnrt_patch = 0;
  cnrtGetLibVersion(&cnrt_major, &cnrt_minor, &cnrt_patch);
  bool max_check = false;
  bool min_check = false;
  if (need_check_min) {
    min_check = (cnrt_major > MLUOP_DEP_CNRT_MIN_MAJOR) ||
                (cnrt_major == MLUOP_DEP_CNRT_MIN_MAJOR &&
                 cnrt_minor >= MLUOP_DEP_CNRT_MIN_MINOR);
    if (!min_check) {
      DEP_CHECK_LOG(level) << "Current CNRT version: " << cnrt_major << "."
                           << cnrt_minor << "." << cnrt_patch;
      DEP_CHECK_LOG(level) << "CNRT version is too low, please upgrade"
                              " CNRT to "
                           << MLUOP_DEP_CNRT_MIN_MAJOR << "."
                           << MLUOP_DEP_CNRT_MIN_MINOR
                           << " or higher. For more details, please check the"
                              " dependency rules in"
                              " Cambricon-MLUOP-Release-Notes.";
      if (level == ERROR) {
        return MLUOP_STATUS_NOT_INITIALIZED;
      }
    }
  }
  if (need_check_max) {
    max_check = (cnrt_major < MLUOP_DEP_CNRT_MAX_MAJOR) ||
                (cnrt_major == MLUOP_DEP_CNRT_MAX_MAJOR &&
                 cnrt_minor <= MLUOP_DEP_CNRT_MAX_MINOR);
    if (!max_check) {
      DEP_CHECK_LOG(level) << "Current CNRT version: " << cnrt_major << "."
                           << cnrt_minor << "." << cnrt_patch;
      DEP_CHECK_LOG(level) << "CNRT version is too high, please downgrade "
                              "CNRT to "
                           << MLUOP_DEP_CNRT_MAX_MAJOR << "."
                           << MLUOP_DEP_CNRT_MAX_MINOR
                           << " or lower. For more details, please check the"
                              " dependency rules in"
                              " Cambricon-MLUOP-Release-Notes.";
      if (level == ERROR) {
        return MLUOP_STATUS_NOT_INITIALIZED;
      }
    }
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpCreate(mluOpHandle_t *handle) {
  PARAM_CHECK("[mluOpCreate]", handle != NULL);

  if (MLUOP_STATUS_SUCCESS != mluOpCheckDependency(true, false, ERROR)) {
    LOG(ERROR) << "Check version dependency failed in mluOpCreate function.";
    return MLUOP_STATUS_NOT_INITIALIZED;
  }

  CNdev mlu_dev;
  int32_t cluster_num = 0;
  int32_t core_num_per_cluster = 0;
  int32_t nram_size = 0;
  int32_t wram_size = 0;
  int32_t sram_size = 0;
  char device_name[CONTEXT_DEVICENAME_BUFFER_SIZE] = "";
  mluOpContext *ctx = new mluOpContext();
  CNcontext drv_ctx;
  CNctxConfigParam ctx_conf_param;
  INTERNAL_CHECK("[mluOpCreate]", CN_SUCCESS == cnCtxGetCurrent(&drv_ctx));
  INTERNAL_CHECK("[mluOpCreate]", CN_SUCCESS == cnCtxGetDevice(&mlu_dev));
  INTERNAL_CHECK("[mluOpCreate]",
                 CN_SUCCESS == cnSharedContextAcquire(&drv_ctx, mlu_dev));
  INTERNAL_CHECK(
      "[mluOpCreate]",
      CN_SUCCESS == cnDeviceGetAttribute(&cluster_num,
                                         CN_DEVICE_ATTRIBUTE_MAX_CLUSTER_COUNT,
                                         mlu_dev));
  INTERNAL_CHECK(
      "[mluOpCreate]",
      CN_SUCCESS ==
          cnDeviceGetAttribute(&core_num_per_cluster,
                               CN_DEVICE_ATTRIBUTE_MAX_CORE_COUNT_PER_CLUSTER,
                               mlu_dev));
  INTERNAL_CHECK(
      "[mluOpCreate]",
      CN_SUCCESS == cnDeviceGetAttribute(
                        &nram_size,
                        CN_DEVICE_ATTRIBUTE_NEURAL_RAM_SIZE_PER_CORE, mlu_dev));
  INTERNAL_CHECK(
      "[mluOpCreate]",
      CN_SUCCESS == cnDeviceGetAttribute(
                        &wram_size,
                        CN_DEVICE_ATTRIBUTE_WEIGHT_RAM_SIZE_PER_CORE, mlu_dev));
  INTERNAL_CHECK(
      "[mluOpCreate]",
      CN_SUCCESS == cnDeviceGetAttribute(
                        &sram_size,
                        CN_DEVICE_ATTRIBUTE_MAX_SHARED_RAM_SIZE_PER_CLUSTER,
                        mlu_dev));
  INTERNAL_CHECK(
      "[mluOpCreate]",
      CN_SUCCESS == cnDeviceGetName(device_name, CONTEXT_DEVICENAME_BUFFER_SIZE,
                                    mlu_dev));
  //  ClusterLimitCapability and JobLimitCapability
  INTERNAL_CHECK("[mluOpCreate]",
                 CN_SUCCESS == cnGetCtxConfigParam(
                                   drv_ctx, CN_CTX_CONFIG_VISIBLE_CLUSTER_NUM,
                                   &ctx_conf_param));
  ctx->capability_cluster_num = (int32_t)ctx_conf_param.visibleClusterNumber;
  INTERNAL_CHECK(
      "[mluOpCreate]",
      CN_SUCCESS == cnGetCtxConfigParam(drv_ctx, CN_CTX_CONFIG_UNION_LIMIT,
                                        &ctx_conf_param));
  // set parallel job num
  if (MLUOP_STATUS_SUCCESS != ctx->initJobNum(drv_ctx, "[mluOpCreate]")) {
    return MLUOP_STATUS_INTERNAL_ERROR;
  }
  ctx->capability_job_limit = (int32_t)ctx_conf_param.unionLimit;
  ctx->arch =
      convertDeviceName(device_name);  // warning: possible return unknown.
  ctx->sram_size = sram_size - REM_FOR_STACK;
  ctx->device = mlu_dev;
  ctx->cluster_num = cluster_num;
  ctx->core_num_per_cluster = core_num_per_cluster;
  ctx->nram_size = nram_size - REM_FOR_STACK;
  if (ctx->arch == 290) {
#ifdef CONV_WARM_UP
    ctx->wram_size = wram_size - 8 * 1024;
#else
    ctx->wram_size = wram_size;
#endif
  } else {
    ctx->wram_size = wram_size;
  }
  if (ctx->arch < 322) {
    ctx->round_mode = MLUOP_ROUND_HALF_OFF_ZERO;
  } else {
    // round to even is supported by hardware since arch >= 322.
    // default usage mode is round to even when arch >= 322.
    // which change accuracy because of using round off zero beform.
    ctx->round_mode = MLUOP_ROUND_HALF_TO_EVEN;
  }
  ctx->atomics_mode =
      MLUOP_ATOMICS_NOT_ALLOWED;  // note : mluOp disallows atomics by defalut
  *handle = ctx;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpUpdateContextInformation(mluOpHandle_t handle) {
  PARAM_CHECK("[mluOpUpdateContextInformation]", handle != NULL);
  CNctxConfigParam ctx_conf_param;
  CNcontext drv_ctx;
  INTERNAL_CHECK(
      "[mluOpUpdateContextInformation]",
      CN_SUCCESS == cnQueueGetContext((CNqueue)(handle->queue), &drv_ctx));
  INTERNAL_CHECK("[mluOpUpdateContextInformation]",
                 CN_SUCCESS == cnGetCtxConfigParam(
                                   drv_ctx, CN_CTX_CONFIG_VISIBLE_CLUSTER_NUM,
                                   &ctx_conf_param));
  handle->capability_cluster_num = (int32_t)ctx_conf_param.visibleClusterNumber;
  INTERNAL_CHECK(
      "[mluOpUpdateContextInformation]",
      CN_SUCCESS == cnGetCtxConfigParam(drv_ctx, CN_CTX_CONFIG_UNION_LIMIT,
                                        &ctx_conf_param));
  handle->capability_job_limit = (int32_t)ctx_conf_param.unionLimit;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpSetAtomicsMode(mluOpHandle_t handle,
                                  mluOpAtomicsMode_t atomics_mode) {
  PARAM_CHECK("[mluOpSetAtomicsMode]", handle != NULL);

  handle->atomics_mode = atomics_mode;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetAtomicsMode(mluOpHandle_t handle,
                                  mluOpAtomicsMode_t *atomics_mode) {
  PARAM_CHECK("[mluOpGetAtomicsMode]", handle != NULL);
  PARAM_CHECK("[mluOpGetAtomicsMode]", atomics_mode != NULL);

  *atomics_mode = handle->atomics_mode;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpDestroy(mluOpHandle_t handle) {
  PARAM_CHECK("[mluOpDestroy]", handle != NULL);

  delete handle;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpSetQueue(mluOpHandle_t handle, cnrtQueue_t queue) {
  PARAM_CHECK("[mluOpSetQueue]", handle != NULL);
  PARAM_CHECK("[mluOpSetQueue]", queue != NULL);

  handle->queue = queue;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetQueue(mluOpHandle_t handle, cnrtQueue_t *queue) {
  PARAM_CHECK("[mluOpGetQueue]", handle != NULL);
  PARAM_CHECK("[mluOpGetQueue]", queue != NULL);

  *queue = handle->queue;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetDevice(mluOpHandle_t handle, CNdev *device) {
  PARAM_CHECK("[mluOpGetDevice]", handle != NULL);
  PARAM_CHECK("[mluOpGetDevice]", device != NULL);

  *device = handle->device;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetQuantizeRoundMode(
    mluOpHandle_t handle, mluOpQuantizeRoundMode_t round_mode) {
  PARAM_CHECK("[mluopSetQuantizeRoundMode]", handle != NULL);
  PARAM_CHECK("[mluopSetQuantizeRoundMode]",
              round_mode == MLUOP_ROUND_HALF_TO_EVEN ||
                  round_mode == MLUOP_ROUND_HALF_OFF_ZERO ||
                  round_mode == MLUOP_ROUND_HALF_UP);
  if (handle->arch < 322) {
    if (round_mode == MLUOP_ROUND_HALF_TO_EVEN) {
      LOG(ERROR)
          << "[mluopSetQuantizeRoundMode] Unsupported rounding mode on MLU200";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  handle->round_mode = round_mode;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetQuantizeRoundMode(
    mluOpHandle_t handle, mluOpQuantizeRoundMode_t *round_mode) {
  PARAM_CHECK("[mluopGetQuantizeRoundMode]", handle != NULL);
  PARAM_CHECK("[mluopGetQuantizeRoundMode]", round_mode != NULL);

  *round_mode = handle->round_mode;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetReservedMemSize(uint64_t *mem_size) {
  PARAM_CHECK("[mluOpGetReservedMemSize]", mem_size != NULL);

  uint64_t default_reserved_size = 2081ULL * 1024 * 1024;
  uint64_t env_size =
      getUintEnvVar("MLUOP_MEM_POOL_SIZE", default_reserved_size);
  *mem_size = static_cast<uint64_t>(env_size);

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetContextParam(mluOpHandle_t handle,
                                   CNctxConfigParamType type,
                                   CNctxConfigParam *param) {
  PARAM_CHECK("[mluOpGetReservedMemSize]", handle != NULL);
  PARAM_CHECK("[mluOpGetReservedMemSize]", param != NULL);
  PARAM_CHECK("[mluOpGetReservedMemSize]",
              type == CN_CTX_CONFIG_VISIBLE_CLUSTER_NUM ||
                  type == CN_CTX_CONFIG_UNION_LIMIT);

  if (type == CN_CTX_CONFIG_VISIBLE_CLUSTER_NUM) {
    param->visibleClusterNumber = handle->capability_cluster_num;
  } else if (type == CN_CTX_CONFIG_UNION_LIMIT) {
    param->unionLimit = static_cast<KernelClass>(handle->capability_job_limit);
  } else {
    LOG(ERROR) << "[mluOpGetContextParam] Unsupported type";
    return MLUOP_STATUS_BAD_PARAM;
  }
  return MLUOP_STATUS_SUCCESS;
}

void mluOpGetLibVersion(int *major, int *minor, int *patch) {
  *major = MLUOP_MAJOR;
  *minor = MLUOP_MINOR;
  *patch = MLUOP_PATCHLEVEL;
}
