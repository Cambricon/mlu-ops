/*************************************************************************
 * Copyright (C) 2022 by Cambricon, Inc. All rights reserved.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef KERNELS_ROI_CROP_FORWARD_ROI_CROP_FORWARD_H_
#define KERNELS_ROI_CROP_FORWARD_ROI_CROP_FORWARD_H_
#include "core/mlu_op_core.h"
#include "kernels/kernel.h"

__mlu_global__ void MLUKernelRoiCropForward(
    const void *input, const int batch, const int height, const int width,
    const int channels, const void *grid, const int grid_n, void *output,
    const int output_h, const int output_w, const mluOpDataType_t data_type);
#endif  // KERNELS_ROI_CROP_FORWARD_ROI_CROP_FORWARD_H_

