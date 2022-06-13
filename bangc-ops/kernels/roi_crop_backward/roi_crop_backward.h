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
#ifndef KERNELS_ROI_CROP_BACKWARD_ROI_CROP_BACKWARD_H_
#define KERNELS_ROI_CROP_BACKWARD_ROI_CROP_BACKWARD_H_
#include "core/mlu_op_core.h"
#include "kernels/kernel.h"

__mlu_global__ void MLUKernelRoiCropBackward(
    const void *gradOutput, const int output_h, const int output_w,
    const void *grid, const int grid_n, void *gradInput, const int batch,
    const int height, const int width, const int channels,
    const mluOpDataType_t data_type);

__mlu_global__ void MLUKernelGdramSetZero(void *gradInput, const int batch,
                                          const int height, const int width,
                                          const int channels);
#endif  // KERNELS_ROI_CROP_BACKWARD_ROI_CROP_BACKWARD_H_
