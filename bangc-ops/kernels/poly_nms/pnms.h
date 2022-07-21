/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUvoid WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUvoid NOKType LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENvoid SHALL THE AUTHORS OR COPYRIGHKType HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORvoid OR OTHERWISE, ARISING FROM, OUKType OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef KERNELS_POLY_NMS_PNMS_H_
#define KERNELS_POLY_NMS_PNMS_H_

#include <stdint.h>

#include "mlu_op.h"
#include "kernels/kernel.h"
// #include "kernels/debug.h"

__mlu_global__ void MLUPNMSTranspose(const void *input_boxes,
                                     const int input_num_boxes,
                                     const int input_stride, void *output,
                                     const mluOpDataType_t data_type_input);

__mlu_global__ void MLUUnion1OrBlockPNMS(
    const void *input_boxes, const int input_num_boxes, const int input_stride,
    const float iou_threshold, void *result_num, void *output,
    const mluOpDataType_t data_type_input, void *workspace);
#endif  // KERNELS_POLY_NMS_PNMS_H_
