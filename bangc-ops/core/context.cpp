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
#include "core/mlu_env.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/tool.h"
#include "kernels/kernel.h"

#define DEP_CHECK_LOG(level)                                              \
  mluop::logging::LogMessage(__FILE__, __LINE__, 4, level, "MLUOP", true, \
                             true, true, true)                            \
      .stream()

namespace mluop {
// see cnrt_function.c deviceCoreVersion for more info.
static struct deviceName name_list_table[] = {
    {"MLU270", MLUOP_MLU270},
    {"MLU290", MLUOP_MLU290},
    {"MLU370", MLUOP_MLU370},
    {"MLU500", MLUOP_MLU590}
    // {"MLU100", MLUOP_MLU100},  // mluop not support mlu100 only for error
    // case.
};

// once cnrtGetDeviceProperties() update and not use
// device_ordinal, update this funciton.
mluOpDevType_t convertDeviceName(char *name) {
  struct deviceName *pName = NULL;
  int num = sizeof(name_list_table) / sizeof(struct deviceName);
  if (CONTEXT_DEVICENAME_LEAST_SIZE > strlen(name)) {
    LOG(ERROR)
        << "get device name failed. device name too short. device name = "
        << name << "\n";
    return MLUOP_UNKNOWN_DEVICE;
  }
  for (int i = 0; i < num; i++) {
    pName = &name_list_table[i];
    if (0 == strncmp(pName->name, name, strlen(pName->name)) ||
        (i == num - 1 &&
         0 >= strncmp(pName->name, name, CONTEXT_DEVICENAME_LEAST_SIZE))) {
      return pName->type;
    }
  }
  LOG(ERROR) << "get device name failed. return unknown device. device name = "
             << name << "\n";
  return MLUOP_UNKNOWN_DEVICE;
}
}  // namespace mluop

mluOpStatus_t mluOpCheckDependency(bool need_check_min, bool need_check_max,
                                   DepCheckLevel level) {
  if (!IF_CHECK_DEP_VERSION) {
    VLOG(5) << "Skip check version dependency.";
    return MLUOP_STATUS_SUCCESS;
  }
  int mluop_major = 0, mluop_minor = 0, mluop_patch = 0;
  mluOpGetLibVersion(&mluop_major, &mluop_minor, &mluop_patch);
  int cnrt_major = 0, cnrt_minor = 0, cnrt_patch = 0;
  cnrtGetLibVersion(&cnrt_major, &cnrt_minor, &cnrt_patch);
  int cndrv_major = 0, cndrv_minor = 0, cndrv_patch = 0;
  cnGetLibVersion(&cndrv_major, &cndrv_minor, &cndrv_patch);
  bool cnrt_max_check = false;
  bool cnrt_min_check = false;
  bool cndrv_max_check = false;
  bool cndrv_min_check = false;
  if (need_check_min) {
    cnrt_min_check = (cnrt_major > MLUOP_DEP_CNRT_MIN_MAJOR) ||
                     (cnrt_major == MLUOP_DEP_CNRT_MIN_MAJOR &&
                      cnrt_minor >= MLUOP_DEP_CNRT_MIN_MINOR);
    if (!cnrt_min_check) {
      DEP_CHECK_LOG(level) << "Current CNRT version: " << cnrt_major << "."
                           << cnrt_minor << "." << cnrt_patch;
      DEP_CHECK_LOG(level) << "CNRT version is too low, please upgrade CNRT to "
                           << MLUOP_DEP_CNRT_MIN_MAJOR << "."
                           << MLUOP_DEP_CNRT_MIN_MINOR << "."
                           << MLUOP_DEP_CNRT_MIN_PATCH
                           << " or higher. For more details, "
                           << "please check the dependency rules in "
                              "Cambricon-MLUOP-Release-Notes.";
    }
    cndrv_min_check = (cndrv_major > MLUOP_DEP_CNDRV_MIN_MAJOR) ||
                      (cndrv_major == MLUOP_DEP_CNDRV_MIN_MAJOR &&
                       cndrv_minor >= MLUOP_DEP_CNDRV_MIN_MINOR);
    if (!cndrv_min_check) {
      DEP_CHECK_LOG(level) << "Current CNDRV version: " << cndrv_major << "."
                           << cndrv_minor << "." << cndrv_patch;
      DEP_CHECK_LOG(level)
          << "CNDRV version is too low, please upgrade CNDRV to "
          << MLUOP_DEP_CNDRV_MIN_MAJOR << "." << MLUOP_DEP_CNDRV_MIN_MINOR
          << "." << MLUOP_DEP_CNDRV_MIN_PATCH
          << " or higher. For more details, "
          << "please check the dependency rules in "
             "Cambricon-MLUOP-Release-Notes.";
    }
    if (((!cnrt_min_check) || (!cndrv_min_check)) && level == ERROR) {
      return MLUOP_STATUS_NOT_INITIALIZED;
    }
  }
  if (need_check_max) {
    cnrt_max_check = (cnrt_major < MLUOP_DEP_CNRT_MAX_MAJOR) ||
                     (cnrt_major == MLUOP_DEP_CNRT_MAX_MAJOR &&
                      cnrt_minor <= MLUOP_DEP_CNRT_MAX_MINOR);
    if (!cnrt_max_check) {
      DEP_CHECK_LOG(level) << "Current CNRT version: " << cnrt_major << "."
                           << cnrt_minor << "." << cnrt_patch;
      DEP_CHECK_LOG(level)
          << "CNRT version is too high, please downgrade CNRT to "
          << MLUOP_DEP_CNRT_MAX_MAJOR << "." << MLUOP_DEP_CNRT_MAX_MINOR << "."
          << MLUOP_DEP_CNRT_MAX_PATCH << " or higher. For more details, "
          << "please check the dependency rules in "
             "Cambricon-MLUOP-Release-Notes.";
    }
    cndrv_max_check = (cndrv_major < MLUOP_DEP_CNDRV_MAX_MAJOR) ||
                      (cndrv_major == MLUOP_DEP_CNDRV_MAX_MAJOR &&
                       cndrv_minor <= MLUOP_DEP_CNDRV_MAX_MINOR);
    if (!cndrv_max_check) {
      DEP_CHECK_LOG(level) << "Current CNDRV version: " << cndrv_major << "."
                           << cndrv_minor << "." << cndrv_patch;
      DEP_CHECK_LOG(level)
          << "CNDRV version is too high, please downgrade CNDRV to "
          << MLUOP_DEP_CNDRV_MAX_MAJOR << "." << MLUOP_DEP_CNDRV_MAX_MINOR
          << "." << MLUOP_DEP_CNDRV_MAX_PATCH
          << " or higher. For more details, "
          << "please check the dependency rules in "
             "Cambricon-MLUOP-Release-Notes.";
    }
    if (((!cnrt_max_check) || (!cndrv_max_check)) && level == ERROR) {
      return MLUOP_STATUS_NOT_INITIALIZED;
    }
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpCreate(mluOpHandle_t *handle) {
  PARAM_CHECK("[mluOpCreate]", handle != NULL);

  if (MLUOP_STATUS_SUCCESS != mluOpCheckDependency(true, false, ERROR)) {
    LOG(ERROR)
        << "Check version dependency failed in mluOpCreate function. "
        << "If don't want this check, set env MLUOP_CHECK_DEP_VERSION to 0, "
        << "but probably cause unexpected errors.";
    return MLUOP_STATUS_NOT_INITIALIZED;
  }

  CNdev mlu_dev;
  int32_t cluster_num = 0;
  int32_t core_num_per_cluster = 0;
  int32_t nram_size = 0;
  int32_t wram_size = 0;
  int32_t sram_size = 0;
  int32_t clock_rate = 0;
  int32_t memory_clock_rate = 0;
  int32_t memory_bus_width = 0;
  int32_t l2cache_size = 0;
  int32_t persisting_l2cache_maxsize = 0;
  double memory_band_width = 0;
  char device_name[CONTEXT_DEVICENAME_BUFFER_SIZE] = "";
  mluOpContext *ctx = new (std::nothrow) mluOpContext();
  CNcontext drv_ctx;
  CNctxConfigParam ctx_conf_param;
  int dev = 0;
  INTERNAL_CHECK("[mluOpCreate]", cnrtSuccess == cnrtGetDevice(&dev));
  INTERNAL_CHECK("[mluOpCreate]", cnrtSuccess == cnrtSetDevice(dev));
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
      CN_SUCCESS == cnDeviceGetAttribute(&nram_size,
                                         CN_DEVICE_ATTRIBUTE_NRAM_SIZE_PER_CORE,
                                         mlu_dev));
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
      CN_SUCCESS == cnDeviceGetAttribute(&clock_rate,
                                         CN_DEVICE_ATTRIBUTE_CLUSTER_CLOCK_RATE,
                                         mlu_dev) ||
          CN_OPS_ERROR_NOT_SUPPORTED ==
              cnDeviceGetAttribute(&clock_rate,
                                   CN_DEVICE_ATTRIBUTE_CLUSTER_CLOCK_RATE,
                                   mlu_dev));
  INTERNAL_CHECK(
      "[mluOpCreate]",
      CN_SUCCESS == cnDeviceGetAttribute(&memory_clock_rate,
                                         CN_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                                         mlu_dev));
  INTERNAL_CHECK(
      "[mluOpCreate]",
      CN_SUCCESS == cnDeviceGetAttribute(
                        &memory_bus_width,
                        CN_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, mlu_dev));
  INTERNAL_CHECK(
      "[mluOpCreate]",
      CN_SUCCESS == cnDeviceGetAttribute(&l2cache_size,
                                         CN_DEVICE_ATTRIBUTE_MAX_L2_CACHE_SIZE,
                                         mlu_dev));
  INTERNAL_CHECK(
      "[mluOpCreate]",
      CN_SUCCESS ==
          cnDeviceGetAttribute(&persisting_l2cache_maxsize,
                               CN_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
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
  // Set parallel job num
  if (MLUOP_STATUS_SUCCESS != ctx->initJobNum(drv_ctx, "[mluOpCreate]")) {
    return MLUOP_STATUS_INTERNAL_ERROR;
  }

  ctx->capability_job_limit = (int32_t)ctx_conf_param.unionLimit;
  ctx->arch = mluop::convertDeviceName(
      device_name);  // warning: possible return unknown.
  ctx->sram_size = sram_size - REM_FOR_STACK;

  strncpy(ctx->device_name, device_name, sizeof(device_name));
  memory_band_width = double(memory_bus_width) * double(memory_clock_rate) /
                      1000.0 / 1000.0 / 8.0 * 2.0;  // NOLINT
  ctx->device = mlu_dev;
  ctx->cluster_num = cluster_num;
  ctx->core_num_per_cluster = core_num_per_cluster;
  ctx->nram_size = nram_size - REM_FOR_STACK;
  ctx->clock_rate = clock_rate;
  ctx->l2cache_size = l2cache_size;
  ctx->persisting_l2cache_maxsize = persisting_l2cache_maxsize;
  ctx->memory_band_width = memory_band_width;
  if (ctx->arch == 290) {
#ifdef CONV_WARM_UP
    ctx->wram_size = wram_size - 8 * 1024;
#else
    ctx->wram_size = wram_size;
#endif
  } else {
    ctx->wram_size = wram_size;
  }
  if (ctx->arch < 372) {
    ctx->round_mode = MLUOP_ROUND_HALF_OFF_ZERO;
  } else {
    ctx->round_mode = MLUOP_ROUND_HALF_TO_EVEN;
  }
  ctx->atomics_mode =
      MLUOP_ATOMICS_NOT_ALLOWED;  // note: mluOp disallows atomics by defalut.
  *handle = ctx;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpUpdateContextInformation(mluOpHandle_t handle) {
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

  if (MLUOP_STATUS_SUCCESS !=
      handle->initJobNum(drv_ctx, "[mluOpUpdateContextInformation]")) {
    return MLUOP_STATUS_INTERNAL_ERROR;
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpSetAtomicsMode(mluOpHandle_t handle, mluOpAtomicsMode_t atomics_mode) {
  PARAM_CHECK("[mluOpSetAtomicsMode]", handle != NULL);

  handle->atomics_mode = atomics_mode;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpGetAtomicsMode(mluOpHandle_t handle, mluOpAtomicsMode_t *atomics_mode) {
  PARAM_CHECK("[mluOpGetAtomicsMode]", handle != NULL);
  PARAM_CHECK("[mluOpGetAtomicsMode]", atomics_mode != NULL);

  *atomics_mode = handle->atomics_mode;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpDestroy(mluOpHandle_t handle) {
  PARAM_CHECK("[mluOpDestroy]", handle != NULL);

  delete handle;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetQueue(mluOpHandle_t handle,
                                          cnrtQueue_t queue) {
  PARAM_CHECK("[mluOpSetQueue]", handle != NULL);

  // note, queue could be NULL
  handle->queue = queue;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetQueue(mluOpHandle_t handle,
                                          cnrtQueue_t *queue) {
  PARAM_CHECK("[mluOpGetQueue]", handle != NULL);
  PARAM_CHECK("[mluOpGetQueue]", queue != NULL);

  *queue = handle->queue;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetDevice(mluOpHandle_t handle,
                                           CNdev *device) {
  PARAM_CHECK("[mluOpGetDevice]", handle != NULL);
  PARAM_CHECK("[mluOpGetDevice]", device != NULL);

  *device = handle->device;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetQuantizeRoundMode(
    mluOpHandle_t handle, mluOpQuantizeRoundMode_t round_mode) {
  PARAM_CHECK("[mluOpSetQuantizeRoundMode]", handle != NULL);
  PARAM_CHECK("[mluOpSetQuantizeRoundMode]",
              round_mode == MLUOP_ROUND_HALF_TO_EVEN ||
                  round_mode == MLUOP_ROUND_HALF_OFF_ZERO ||
                  round_mode == MLUOP_ROUND_HALF_UP);
  if (handle->arch < 372) {
    if (round_mode == MLUOP_ROUND_HALF_TO_EVEN) {
      LOG(ERROR)
          << "[mluOpSetQuantizeRoundMode] Unsupported rounding mode on MLU200";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  handle->round_mode = round_mode;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetQuantizeRoundMode(
    mluOpHandle_t handle, mluOpQuantizeRoundMode_t *round_mode) {
  PARAM_CHECK("[mluOpGetQuantizeRoundMode]", handle != NULL);
  PARAM_CHECK("[mluOpGetQuantizeRoundMode]", round_mode != NULL);

  *round_mode = handle->round_mode;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetReservedMemSize(uint64_t *mem_size) {
  PARAM_CHECK("[mluOpGetReservedMemSize]", mem_size != NULL);
  uint64_t default_reserved_size = 2081ULL * 1024 * 1024;
  uint64_t env_size =
      mluop::getUintEnvVar("MLUOP_MEM_POOL_SIZE", default_reserved_size);
  *mem_size = static_cast<uint64_t>(env_size);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetContextParam(mluOpHandle_t handle,
                                                 CNctxConfigParamType type,
                                                 CNctxConfigParam *param) {
  PARAM_CHECK("[mluOpGetContextParam]", handle != NULL);
  PARAM_CHECK("[mluOpGetContextParam]", param != NULL);
  PARAM_CHECK("[mluOpGetContextParam]",
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

/*********************************************************************************************
 * @deprecate The function implementation is not recommended to use, it is
 *recommended to use the following function implementation: void
 *mluOpGetLibVersion(int* major, int* minor, int* patch);
 **********************************************************************************************/
size_t MLUOP_WIN_API mluOpGetVersion() {
  LOG_FIRST_N(WARNING, 1) << "[mluOpGetVersion] is deprecated and will be "
                             "removed in the future release,"
                          << " please use [mluOpGetLibVersion] instead.";
  return MLUOP_VERSION;
}
void MLUOP_WIN_API mluOpGetLibVersion(int *major, int *minor, int *patch) {
  *major = MLUOP_MAJOR;
  *minor = MLUOP_MINOR;
  *patch = MLUOP_PATCHLEVEL;
}
