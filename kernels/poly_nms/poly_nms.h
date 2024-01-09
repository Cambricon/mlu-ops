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
#ifndef KERNELS_POLY_NMS_POLY_NMS_H
#define KERNELS_POLY_NMS_POLY_NMS_H

#include <algorithm>

#include "core/runtime/device.h"
#include "mlu_op.h"

#define MASK_T_BITWIDTH 32  // mask will be stored in an uint32_t value

template <int MIN_BOX_NUM_PER_CORE>
struct BlockConfig {
  BlockConfig(mluOpHandle_t handle, int box_num, int core_num_limit = 0) {
    int core_num_to_use = mluop::runtime::getJobLimitCapability(handle);
    if (core_num_limit > 0) {
      core_num_to_use = std::min(core_num_limit, core_num_to_use);
    }
    core_num_to_use =
        std::min(core_num_to_use,
                 (box_num + MIN_BOX_NUM_PER_CORE - 1) / MIN_BOX_NUM_PER_CORE);
    dim.x = core_num_to_use;
    dim.y = 1;
    dim.z = 1;
  }
  cnrtFunctionType_t kernel_type = cnrtFunctionType_t::cnrtFuncTypeBlock;
  cnrtDim3_t dim;
};

/**
 * Generate launch config for mluCalcArea. By default, we will launch as many
 * BLOCKs as we can.
 */
struct MLUCalcAreaLaunchConfig : public BlockConfig<128> {
  using BlockConfig::BlockConfig;
};

/**
 * Generate launch config for mluCalcArea. By default, we will launch as many
 * BLOCKs as we can.
 */
struct MLUGenNmsMaskLaunchConfig : public BlockConfig<8> {
  using BlockConfig::BlockConfig;
};

/**
 * Generate launch config for MLUGenResult. By default, we will launch one BLOCK
 * task.
 */
struct MLUGenResultLaunchConfig {
  cnrtFunctionType_t kernel_type = cnrtFunctionType_t::cnrtFuncTypeBlock;
  cnrtDim3_t dim = {1, 1, 1};
};

void MLUOP_WIN_API KernelPolyNmsCalcArea(cnrtDim3_t k_dim,
                                         cnrtFunctionType_t k_type,
                                         cnrtQueue_t queue, const float *boxes,
                                         const int box_num,
                                         const int real_width, float *dev_area);

void MLUOP_WIN_API KernelPolyNmsGenMask(cnrtDim3_t k_dim,
                                        cnrtFunctionType_t k_type,
                                        cnrtQueue_t queue, const float *boxes,
                                        const int box_num, const int real_width,
                                        const float iou_threshold,
                                        float *dev_area, uint32_t *dev_mask,
                                        int *dev_sort_info);

void MLUOP_WIN_API KernelPolyNmsGenResult(cnrtDim3_t k_dim,
                                          cnrtFunctionType_t k_type,
                                          cnrtQueue_t queue, const int box_num,
                                          uint32_t *dev_mask,
                                          int *dev_sort_info, int *output,
                                          int *output_size);

#endif  // KERNELS_POLY_NMS_POLY_NMS_H
