/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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
#ifndef KERNELS_FFT_C2C_FFT_C2C_FFT_H_
#define KERNELS_FFT_C2C_FFT_C2C_FFT_H_

#include <string>
#include "kernels/fft/fft.h"

mluOpStatus_t makeFFT1dPolicy(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan);

mluOpStatus_t setFFT1dReserveArea(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                                  const std::string api);

mluOpStatus_t setFFT2dReserveArea(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                                  const std::string api);

mluOpStatus_t execFFT1d(mluOpHandle_t handle, const mluOpFFTPlan_t fft_plan,
                        const void *input, const float scale_factor,
                        void *workspace, void *output, int direction);

mluOpStatus_t execFFT2d(mluOpHandle_t handle, const mluOpFFTPlan_t fft_plan,
                        const void *input, const float scale_factor,
                        void *workspace, void *output, int direction);

#endif  // KERNELS_FFT_C2C_FFT_C2C_FFT_H_
