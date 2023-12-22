/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef KERNELS_FFT_RFFT_RFFT_H_
#define KERNELS_FFT_RFFT_RFFT_H_

#include <string>
#include "kernels/fft/fft.h"

mluOpStatus_t makeRFFT1dPolicy(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan);

mluOpStatus_t setRFFT1dReserveArea(mluOpHandle_t handle,
                                   mluOpFFTPlan_t fft_plan,
                                   const std::string api);

mluOpStatus_t execRFFT1d(mluOpHandle_t handle, const mluOpFFTPlan_t fft_plan,
                         const void *input, const float scale_factor,
                         void *workspace, void *output);

#endif  // KERNELS_FFT_RFFT_RFFT_H_
