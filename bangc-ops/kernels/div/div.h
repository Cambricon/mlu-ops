/*************************************************************************
 * Copyright (C) 2021 by Cambricon, Inc. All rights reserved.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef KERNELS_DIV_DIV_H_
#define KERNELS_DIV_DIV_H_

#include "kernels/binary_op/binary_op_3pipeline.h"

// declare div 5stage pipeline kernel, half:Fast or HighAcc mode, float:Fast
// mode
BINARY_OP_3PIPELINE_DECLARE(Div, half, HighAcc);
BINARY_OP_3PIPELINE_DECLARE(Div, half, Fast);
BINARY_OP_3PIPELINE_DECLARE(Div, float, Fast);
#endif  // KERNELS_DIV_DIV_H_
