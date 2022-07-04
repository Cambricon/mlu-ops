/*************************************************************************
 * Copyright (C) 2022 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef MLU_OP_KERNEL_H_
#define MLU_OP_KERNEL_H_

#include <stdint.h>
#include "cnrt.h"

#ifndef MLUOP_WIN_API
#ifdef _WIN32
#define MLUOP_WIN_API __stdcall
#else
#define MLUOP_WIN_API
#endif  // _WIN32
#endif  // MLUOP_WIN_API

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus

/* Abs */
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineAbsHalfFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineAbsFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);

void MLUOP_WIN_API mluOpBlockKernel5StagePipelineAbsHalfFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);
void MLUOP_WIN_API mluOpBlockKernel5StagePipelineAbsFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);

/* Div */
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineDivHalfFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, const void *y, void *z, int num);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineDivHalfHighAcc(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, const void *y, void *z, int num);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineDivFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, const void *y, void *z, int num);

/* Log */
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineLogHalfFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num, float coef);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineLogHalfHighAcc(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num, float coef);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineLogFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num, float coef);

void MLUOP_WIN_API mluOpBlockKernel5StagePipelineLogHalfFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num, float coef);
void MLUOP_WIN_API mluOpBlockKernel5StagePipelineLogHalfHighAcc(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num, float coef);
void MLUOP_WIN_API mluOpBlockKernel5StagePipelineLogFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num, float coef);
    
/* RoICrop*/
void MLUOP_WIN_API mluOpBlockKernelRoiCropForwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, const void *grid, const int batch, const int height,
    const int width, const int channels, const int grid_n, const int output_h,
    const int output_w, void *output);

/* Sqrt */
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineSqrtHalfFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineSqrtHalfHighAcc(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineSqrtFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);

void MLUOP_WIN_API mluOpBlockKernel5StagePipelineSqrtHalfFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);
void MLUOP_WIN_API mluOpBlockKernel5StagePipelineSqrtHalfHighAcc(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);
void MLUOP_WIN_API mluOpBlockKernel5StagePipelineSqrtFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);

/* SqrtBackward */
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineSqrtBackwardHalfHighAcc(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *y, const void *diff_y, void *x, int num);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineSqrtBackwardFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *y, const void *diff_y, void *x, int num);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // MLU_OP_KERNEL_H_
