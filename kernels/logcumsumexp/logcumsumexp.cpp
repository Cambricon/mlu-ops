/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS for A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE for ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "logcumsumexp.h"
#include <string>
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"

mluOpStatus_t MLUOP_WIN_API
mluOpLogcumsumexp(mluOpHandle_t handle,
                const int32_t dim,
                const mluOpTensorDescriptor_t input_desc,
                const void *input,
                const mluOpTensorDescriptor_t result_desc,
                void *result) {
    const std::string API = "[mluOpLogcumsumexp]";

    cnrtFunctionType_t k_type;
    cnrtDim3_t k_dim;

    PARAM_CHECK(API, handle != NULL);
    PARAM_CHECK(API, input_desc != NULL);
    PARAM_CHECK(API, input != NULL);
    PARAM_CHECK(API, result_desc != NULL);
    PARAM_CHECK(API, result != NULL);
    PARAM_CHECK(API, input_desc->dtype == MLUOP_DTYPE_FLOAT ||
                input_desc->dtype == MLUOP_DTYPE_HALF);
    PARAM_CHECK(API, result_desc->dtype == MLUOP_DTYPE_FLOAT ||
                result_desc->dtype == MLUOP_DTYPE_HALF);
    PARAM_CHECK(API, input_desc->dim == result_desc->dim);
    PARAM_CHECK(API, input_desc->layout == MLUOP_LAYOUT_ARRAY);
    PARAM_CHECK(API, result_desc->layout == MLUOP_LAYOUT_ARRAY);


    if (dim < (-1) * input_desc->dim) {
        LOG(ERROR) << API
                << " dim is invalid. Received dim=["
                << dim << "]";
        return MLUOP_STATUS_BAD_PARAM;
    }
    if (dim >= input_desc->dim) {
        LOG(ERROR) << API
                << " dim beyonds the dimension of tensor. Received dim=["
                << dim << "]";
        return MLUOP_STATUS_BAD_PARAM;
    }

    int32_t axis_size = input_desc->dims[dim];
    int32_t higher_size = 1;
    int32_t lower_size = 1;

    for (int i = 0; i < dim; i++) {
        lower_size *= input_desc->dims[i];
    }

    for (int i = dim+1; i < input_desc->dim; i++) {
        higher_size *= input_desc->dims[i];
    }

    if (lower_size == 1 && higher_size == 1) {
        k_type = CNRT_FUNC_TYPE_UNION8;
        k_dim = {32, 1, 1};
    } else if (higher_size == 1) {
        k_type = CNRT_FUNC_TYPE_UNION8;
        k_dim = {32, 1, 1};
    } else if (lower_size == 1) {
        k_type = CNRT_FUNC_TYPE_UNION1;
        k_dim = {4, 1, 1};
    } else {
        k_type = CNRT_FUNC_TYPE_UNION8;
        k_dim = {32, 1, 1};
    }
    CHECK_RETURN(API, KernelLogcumsumexp(
                        k_dim, k_type, handle->queue, input_desc->dtype,
                        input, result, axis_size, higher_size, lower_size));
    GEN_CASE_END();
    return MLUOP_STATUS_SUCCESS;
}
