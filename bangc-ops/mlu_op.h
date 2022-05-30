/*************************************************************************
 * Copyright (C) 2021 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef MLUOP_EXAMPLE_H_
#define MLUOP_EXAMPLE_H_

#include <stdint.h>
#include "cnrt.h"
#include "core/mlu_op_core.h"

#ifndef MLUOP_WIN_API
#ifdef _WIN32
#define MLUOP_WIN_API __stdcall
#else
#define MLUOP_WIN_API
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/*!
 * @brief
 *
 * Enumeration variables describe the base that is used in the implementation
 * of the log function.
 *
 */
typedef enum {
  MLUOP_LOG_E  = 0, /*!< The base e is used.*/
  MLUOP_LOG_2  = 1, /*!< The base 2 is used.*/
  MLUOP_LOG_10 = 2, /*!< The base 10 is used.*/
} mluOpLogBase_t;

/*!
 * @brief Computes the absolute value for every element of the input tensor \b x and returns in \b
 y.
 *
 * @param[in] handle
 *   Input. Handle to a MLUOP context that is used to manage MLU devices and
 *   queues in the abs operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] x_desc
 *   Input. The descriptor of the input tensor. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[in] x
 *   Input. Pointer to the MLU memory that stores the input tensor.
 * @param[in] y_desc
 *   Input. The descriptor of the output tensor. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[out] y
 *   Output. Pointer to the MLU memory that stores the output tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *
 * @par Formula
 * - See "Abs Operator" section in "Cambricon MLUOP User Guide" for details.
 *
 * @par Data Type
 * - Date types of input tensor and output tensor should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float.
 *   - output tensor: half, float.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of the abs operation is as follows:
     @verbatim
      input arrays by 1 * 3 * 3 * 2 -->
          input: [[[[5, -11], [8, 1], [6, 4]],
                  [[3, 8], [2,6], [0, 6]],
                  [[8, 5], [7,4], [-9, 6]]]]

      output array by 1 * 3 * 3 * 2 -->
          output: [[[[5, 11], [8, 1], [6, 4]],
                   [[3, 8], [2,6], [0, 6]],
                   [[8, 5], [7,4], [9, 6]]]]
     @endverbatim
 *
 * @par Reference
 * - https://www.tensorflow.org/api_docs/python/tf/math/abs
 */
mluOpStatus_t MLUOP_WIN_API mluOpAbs(mluOpHandle_t handle,
                                  const mluOpTensorDescriptor_t x_desc,
                                  const void *x,
                                  const mluOpTensorDescriptor_t y_desc,
                                  void *y);

/*!
 * @brief Computes logarithm of input tensor \b x, and returns the results in the output tensor \b
 * y.
 *
 * Compared with ::mluOpLog, this function allows you to choose whether to perform log operation
 * with faster algorithm or higher precision.
 *
 * @param[in] handle
 *   Input. Handle to a MLUOP context that is used to manage MLU devices and queues in the log
 *   operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] prefer
 *   Input. The \b prefer modes defined in ::mluOpComputationPreference_t enum.
 * @param[in] base
 *    Input. A mluOpLogBase_t type value indicating which base (e, 2 or 10) to be used.
 * @param[in] x_desc
 *   Input. The descriptor of the input tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 *   Input. Pointer to the MLU memory that stores the input tensor \b x.
 * @param[in] y_desc
 *   Input. The descriptor of the output tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] y
 *   Output. Pointer to the MLU memory that stores the output tensor \b y.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Formula
 * - See "Log Operation" section in "Cambricon MLUOP User Guide" for details.
 *
 * @par Data Type
 * - Data type of input tensor and output tensor should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float.
 *   - output tensor: half, float.
 *
 * @par Scale Limitation
 * - The input tensor and output tensor have the same shape, and the input tensor must meet
 *   the following input data range:
 *   - float: [1e-20, 2e5].
 *   - half: [1, 60000].
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://www.tensorflow.org/api_docs/python/tf/math/log
 */
mluOpStatus_t MLUOP_WIN_API mluOpLog(mluOpHandle_t handle,
                                  const mluOpComputationPreference_t prefer,
                                  const mluOpLogBase_t base,
                                  const mluOpTensorDescriptor_t x_desc,
                                  const void *x,
                                  const mluOpTensorDescriptor_t y_desc,
                                  void *y);

/*!
 * @brief Computes division on input tensor \b x and \b y, and returns the results
 *        in the output tensor \b output.
 *
 * Compared with ::mluOpDiv, this function allows you to choose whether to perform div
 * operation with faster algorithm or higher precision.
 *
 * @param[in] handle
 *   Input. Handle to a MLUOP context that is used to manage MLU devices and queues in the division
 *   operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] prefer
 *   Input. The \b prefer modes defined in ::mluOpComputationPreference_t enum.
 * @param[in] x_desc
 *   Input. The descriptor of the input tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 *   Input. Pointer to the MLU memory that stores the dividend tensor.
 * @param[in] y_desc
 *   Input. The descriptor of the input tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] y
 *   Input. Pointer to the MLU memory that stores the divisor tensor.
 * @param[in] z_desc
 *   Input. The descriptor of the output tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] z
 *   Output. Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Formula
 * - See "Div Operation" section in "Cambricon MLUOP User Guide" for details.
 *
 * @par Data Type
 * - Data type of input tensors and output tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float.
 *   - output tensor: half, float.
 *
 * @par Scale Limitation
 * - The input tensors and output tensor must have the same shape.
 *
 * @note
 * - The inputs \b x and \b y are multi-dimensional array, supporting up to MLUOP_DIM_MAX dimensions.
 * - When input \b y data type is float, \b y data range is [-1e10,-1e-20] & [1e-20,1e10]. When \b y
 * data type is
 *   half, \b y data range is [-65504,-1e-4] & [1e-4,65504].
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://www.tensorflow.org/api_docs/python/tf/math/divide
 */
mluOpStatus_t MLUOP_WIN_API mluOpDiv(mluOpHandle_t handle,
                                  const mluOpComputationPreference_t prefer,
                                  const mluOpTensorDescriptor_t x_desc,
                                  const void *x,
                                  const mluOpTensorDescriptor_t y_desc,
                                  const void *y,
                                  const mluOpTensorDescriptor_t z_desc,
                                  void *z);

/*!
 * @brief Computes sqrt on input tensor \b x, and returns the results in the output tensor \b y.
 *
 * This function allows you to choose whether to perform sqrt operation with faster algorithm or
 * higher precision.
 *
 * @param[in] handle
 *   Input. Handle to a MLUOP context that is used to manage MLU devices and queues in the sqrt
 *   operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] prefer
 *   Input. The \b prefer modes defined in ::mluOpComputationPreference_t enum.
 * @param[in] x_desc
 *   Input. The descriptor of the input tensor. For detailed information, see
 ::mluOpTensorDescriptor_t.
 * @param[in] x
 *   Input. Pointer to the MLU memory that stores the input tensor.
 * @param[in] y_desc
 *   Input. The descriptor of the output tensor. For detailed information, see
 ::mluOpTensorDescriptor_t.
 * @param[out] y
 *   Output. Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Formula
 * - See "Sqrt Operation" section in "Cambricon MLUOP User Guide" for details.
 *
 * @par Data Type
 * - Data type of input tensor and output tensor should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float.
 *   - output tensor: half, float.
 *
 * @par Scale Limitation
 * - The input tensor and output tensor must have the same shape, and the input tensor must meet
 *   the following input data range:
 *   - float: [1e-10,1e10].
 *   - half: [1e-3,1e-2] & [1e-1,60000].
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.

 * @par Reference
 * - https://www.tensorflow.org/api_docs/python/tf/math/sqrt
 */
mluOpStatus_t MLUOP_WIN_API mluOpSqrt(mluOpHandle_t handle,
                                   const mluOpComputationPreference_t prefer,
                                   const mluOpTensorDescriptor_t x_desc,
                                   const void *x,
                                   const mluOpTensorDescriptor_t y_desc,
                                   void *y);

/*!
 * @brief Computes gradient of sqrt on input tensor \b y and \b diff_y, and returns the results
 *        in the output tensor \b diff_x.
 *
 * @param[in] handle
 *   Input. Handle to a MLUOP context that is used to manage MLU devices and queues in the sqrt
 * backward
 *   operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] y_desc
 *   Input. The descriptor of the tensors. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] y
 *   Input. Pointer to the MLU memory that stores the input tensor.
 * @param[in] dy_desc
 *   Input. The descriptor of the tensors. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] diff_y
 *   Input. Pointer to the MLU memory that stores the input tensor.
 * @param[in] dx_desc
 *   Input. The descriptor of the tensors. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] diff_x
 *   Output. Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Formula
 * - See "Sqrt Backward Operation" section in "Cambricon MLUOP User Guide" for details.
 *
 * @par Data Type
 * - Data types of input tensors and output tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensors: half, float.
 *   - output tensor: half, float.
 *
 * @par Scale Limitation
 * - The input tensors and output tensor must have the same shape, and the input tensor \b y must
 * meet
 *   the following input data range:
 *   - float: [1e-10,1e6].
 *   - half: [0.01,500].
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://www.tensorflow.org/api_docs/python/tf/raw_ops/SqrtGrad
 */
mluOpStatus_t MLUOP_WIN_API mluOpSqrtBackward(mluOpHandle_t handle,
                                           const mluOpTensorDescriptor_t y_desc,
                                           const void *y,
                                           const mluOpTensorDescriptor_t dy_desc,
                                           const void *diff_y,
                                           const mluOpTensorDescriptor_t dx_desc,
                                           void *diff_x);

#if defined(__cplusplus)
}
#endif

#endif  // MLUOP_EXAMPLE_H_
