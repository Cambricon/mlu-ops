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
#ifndef KERNELS_PRIOR_BOX_PRIOR_BOX_H
#define KERNELS_PRIOR_BOX_PRIOR_BOX_H

#include "mlu_op.h"

void MLUOP_WIN_API KernelPriorBox(
    cnrtDim3_t k_dim_box, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *min_sizes, const int min_sizes_num, const void *aspect_ratios,
    const int aspect_ratios_num, const void *variances, const int variances_num,
    const void *max_sizes, const int max_sizes_num, const int height,
    const int width, const int im_height, const int im_width,
    const float step_h, const float step_w, const float offset,
    const int num_priors, const bool clip,
    const bool min_max_aspect_ratios_order, void *output, const int output_size,
    void *var, const int var_size);

#endif  // KERNELS_PRIOR_BOX_PRIOR_BOX_H
