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
#ifndef KERNELS_FFT_COMMON_FFT_COMMON_KERNELS_H_
#define KERNELS_FFT_COMMON_FFT_COMMON_KERNELS_H_

#include "kernels/fft/fft.h"

__mlu_global__ void generateRFFTHalfDFTMatrix(mluOpDataType_t data_type, int n, void *output);

__mlu_global__ void generateRFFTFullDFTMatrix(mluOpDataType_t data_type, int row, int n,
                                              void *output);

__mlu_global__ void generateIRFFTHalfDFTMatrix(mluOpDataType_t data_type, int n, void *output);

__mlu_global__ void generateIRFFTFullDFTMatrix(mluOpDataType_t data_type, int n, void *output);

__mlu_global__ void generateC2CFFTDFTMatrix(mluOpDataType_t data_type, int n, void *output);

#endif  // KERNELS_FFT_COMMON_FFT_COMMON_KERNELS_H_
