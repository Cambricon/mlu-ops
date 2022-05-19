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
#include "kernels/kernel.h"
#include "core/mlu_op_core.h"
  
__mlu_global__ void MLUKernelRoiCropForward(const void *input,
                                            int batch,
                                            int height,
                                            int width,
                                            int channels,
                                            const void *grid,
                                            int grid_n,
                                            void *output,
                                            int output_h,
                                            int output_w,
                                            mluOpDataType_t data_type);
  
#endif  // KERNELS_ROI_CROP_FORWARD_ROI_CROP_FORWARD_H_


