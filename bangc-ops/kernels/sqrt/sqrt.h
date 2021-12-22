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
#ifndef KERNELS_SQRT_SQRT_H_
#define KERNELS_SQRT_SQRT_H_

#include "kernels/unary_op/unary_op_3pipeline.h"
#include "kernels/unary_op/unary_op_5pipeline.h"

// declare sqrt 3stage pipeline kernel, float:Fast mode, half:Fast or HighAcc mode
UNARY_OP_KERNEL_3PIPELINE_DECLARE(Sqrt, float, Fast);
UNARY_OP_KERNEL_3PIPELINE_DECLARE(Sqrt, half, Fast);
UNARY_OP_KERNEL_3PIPELINE_DECLARE(Sqrt, half, HighAcc);

// declare sqrt 5stage pipeline kernel, float:Fast mode, half:Fast or HighAcc mode
UNARY_OP_KERNEL_5PIPELINE_DECLARE(Sqrt, float, Fast);
UNARY_OP_KERNEL_5PIPELINE_DECLARE(Sqrt, half, Fast);
UNARY_OP_KERNEL_5PIPELINE_DECLARE(Sqrt, half, HighAcc);

#endif  // KERNELS_SQRT_SQRT_H_
