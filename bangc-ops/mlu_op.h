/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc.
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
  MLUOP_LOG_E = 0,  /*!< The base e is used.*/
  MLUOP_LOG_2 = 1,  /*!< The base 2 is used.*/
  MLUOP_LOG_10 = 2, /*!< The base 10 is used.*/
} mluOpLogBase_t;

/*!
 * @brief Computes the absolute value for every element of the input tensor \b x
 *   and returns in \b y.
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
 * @brief Computes logarithm of input tensor \b x, and returns the results in
 *   the output tensor \b y.
 *
 * @param[in] handle
 *   Input. Handle to a MLUOP context that is used to manage MLU devices and
 *   queues in the log operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] prefer
 *   Input. The \b prefer modes defined in ::mluOpComputationPreference_t enum.
 * @param[in] base
 *   Input. A mluOpLogBase_t type value indicating which base (e, 2 or 10) to
 *   be used.
 * @param[in] x_desc
 *   Input. The descriptor of the input tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[in] x
 *   Input. Pointer to the MLU memory that stores the input tensor \b x.
 * @param[in] y_desc
 *   Input. The descriptor of the output tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
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
 * - The input tensor and output tensor have the same shape, and the input
 *   tensor must meet the following input data range:
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
mluOpStatus_t MLUOP_WIN_API
mluOpLog(mluOpHandle_t handle, const mluOpComputationPreference_t prefer,
         const mluOpLogBase_t base, const mluOpTensorDescriptor_t x_desc,
         const void *x, const mluOpTensorDescriptor_t y_desc, void *y);

/*!
 * @brief Computes division on input tensor \b x and \b y, and returns the
 *   results in the output tensor \b output.
 *
 * @param[in] handle
 *   Input. Handle to a MLUOP context that is used to manage MLU devices and
 *   queues in the division operation. For detailed information, see
 *   ::mluOpHandle_t.
 * @param[in] prefer
 *   Input. The \b prefer modes defined in ::mluOpComputationPreference_t enum.
 * @param[in] x_desc
 *   Input. The descriptor of the input tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[in] x
 *   Input. Pointer to the MLU memory that stores the dividend tensor.
 * @param[in] y_desc
 *   Input. The descriptor of the input tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[in] y
 *   Input. Pointer to the MLU memory that stores the divisor tensor.
 * @param[in] z_desc
 *   Input. The descriptor of the output tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
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
 * - The input tensors and output tensor have the same shape, and the input
 *   tensor \b y must meet the following input data range:
 *   - float: [-1e10,-1e-20] & [1e-20,1e10].
 *   - half: [-65504,-1e-4] & [1e-4,65504].
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
mluOpStatus_t MLUOP_WIN_API
mluOpDiv(mluOpHandle_t handle, const mluOpComputationPreference_t prefer,
         const mluOpTensorDescriptor_t x_desc, const void *x,
         const mluOpTensorDescriptor_t y_desc, const void *y,
         const mluOpTensorDescriptor_t z_desc, void *z);

/*!
 *  @brief Gets extra space size that is needed in poly_nms operation.
 *
 *  @param[in] handle
 *    Input. Handle to a MLUOP context that is used to manage MLU devices
 *    and queues in the psroipool_forward operation.
 *  @param[in] boxes_desc
 *    Input. The descriptor of the boxes tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t. 
 *  @param[out] size
 *    Output. A host pointer to the returned size of extra space in bytes.
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 */
mluOpStatus_t MLUOP_WIN_API mluOpGetPolyNmsWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t boxes_desc,
    size_t *size);

/*!
 *  @brief Polygon Non Maximum Suppression.
 *
 *  @param[in] handle
 *    Input. Handle to a MLUOP context that is used to manage MLU devices
 *    and queues in the poly_nms operation.
 *  @param[in] boxes_desc
 *    Input. The descriptor of the input tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
 *  @param[in] boxes
 *    Input. Pointer to the MLU memory that stores the input tensor.
 *  @param[in] iou_threshold
 *    Input. The iou_threshold data.
 *  @param[in] workspace
 *    Input. Pointer to the MLU memory that stores the extra workspace.
 *  @param[in] workspace_size
 *    Input. The size of extra space.
 *  @param[in] output_desc
 *    Input. The descriptor of the output tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
 *  @param[out] output
 *    Output. Pointer to the MLU memory that stores the output tensor.
 *  @param[in] output_size
 *    Output. Pointer to the MLU memory that stores the output tensor. Indicates
 *    the number of return values of output.
 * 
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *    ::MLUOP_STATUS_NOT_SUPPORTED
 * 
 *  @par Formula
 *  - See "poly_nms Operation" section in "Cambricon MLUOP User
 *    Guide" for details.
 * 
 *  @par Data Type
 *  - The supported data types of input and output tensors are as follows:
 *     - Input tensor: float.
 *     - iou_threshold: float.
 *     - Output tensor: int32.
 *     - output_size tensor: int32.
 * 
 *  @par Data Layout
 *  - The supported data layout of \b input, \b output, 
 *     \b output_size are as follows:
 * 
 *   - Input tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - Output tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - output_size tensor: \p MLUOP_LAYOUT_ARRAY.
 * 
 *  @par Scale Limitation
 *  - The dimension of \b input should be equal to 2.
 *  - The dimension of \b output should be equal to 1.
 *  - The dimension of \b output_size should be equal to 1.
 *  - The shape[0] of output should be equal input shape[0].
 *  - The shape[1] of input should be equal 9.
 *  - 
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 * 
 *  @par Note
 *  - This commit does not support nan/inf.
 *  - The coordinates of the input boxes must all be sorted clockwise or 
 *    counterclockwise. If the coordinates of the boxes are out of order,
 *    the calculation result is not guaranteed and is consistent with the
 *    calculation result of the competitor operator.
 *  - If there are cases with the same score in the input boxes, the output
 *    results may be inconsistent with the results of competing products.
 *  - The number of input boxes on mlu270, mlu290 and mlu370 does not exceed
 *    9770, and an error log will be printed if the size limit is exceeded.
 * 
 * @par Reference
 * - https://github.com/dingjiansw101/AerialDetection/tree/master/mmdet/ops/poly_nms
 */
mluOpStatus_t MLUOP_WIN_API
mluOpPolyNms(mluOpHandle_t handle, const mluOpTensorDescriptor_t boxes_desc,
             const void *boxes, const float iou_threshold, void *workspace,
             size_t workspace_size, const mluOpTensorDescriptor_t output_desc,
             void *output, void *output_size);

/*!
 *  @brief Generate fixed size feature map for each RoI(Regions of Interest).
 *
 *  @param[in] handle
 *    Input. Handle to a MLUOP context that is used to manage MLU devices
 *    and queues in the psroipool_forward operation. For detailed information,
 *    see::mluOpHandle_t.
 *  @param[in] spatial_scale
 *    Input. The spatial scale of each regions of interest in the output.
 *  @param[in] group_size
 *    Input. Used to set the number of \b rois to be divided equally in each direction.
 *  @param[in] pooled_height
 *    Input. The pooled_height data.
 *  @param[in] pooled_width
 *    Input. The pooled_width data.
 *  @param[in] output_dim
 *    Input. The output_dim data.
*  @param[in] input_desc
 *    Input. Descriptor of input tensor, containing dimension and the layout of input.
 *    For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[in] input
 *    Input. Pointer to the MLU memory that stores the input tensor. The shape of \b input is
 *    [batch_num, H, W, C].
 *  @param[in] rois_desc
 *    Input. Descriptor of rois tensor, containing dimension and the layout of rois.
 *    For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[in] rois
 *    Input. Pointer to the MLU memory that stores the rois tensor. \b rois[1] consists of
 *    [batch_id, roi_start_w, roi_start_h, roi_end_w, roi_end_h], where \p batch_id is the ID
 *    of the batch.
 *  @param[in] output_desc
 *    Input. Descriptor of output tensor, containing dimension and the layout of output.
 *    For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[out] output
 *    Output. Pointer to the MLU memory that stores the output tensor. The shape of \b output is
 *    [rois[0], pooled_height, pooled_width, output_dim].
 *  @param[in] mapping_channel_desc
 *    Input. Descriptor of the mapping_channel tensor, containing dimension and the layout of 
 *    mapping_channel. For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[out] mapping_channel
 *    Output. Pointer to the MLU memory that stores the mapping_channel tensor. The shape of 
 *    \b mapping_channel is [rois[0], pooled_height, pooled_width, output_dim].
 * 
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 * 
 *  @par Formula
 *  - See "psroipool_forward Operation" section in "Cambricon MLUOP User
 *    Guide" for details.
 * 
 *  @par Data Type
 *  - The supported data types of input and output tensors are as follows:
 *     - Input tensor: float.
 *     - Rois tensor: float.
 *     - Output tensor: float.
 *     - Mapping_channel tensor: int32.
 * 
 *  @par Data Layout
 *  - The supported data layout of \b input, \b rois, \b output, and \b mapping_channel 
 *    are as follows:
 *     - Input tensor: \p MLUOP_LAYOUT_NHWC.
 *     - Rois tensor: \p MLUOP_LAYOUT_ARRAY.
 *     - Output tensor: \p MLUOP_LAYOUT_NHWC.
 *     - Mapping_channel tensor: \p MLUOP_LAYOUT_NHWC.
 * 
 *  @par Scale Limitation
 *  - The input tensor, mapping_channel tensor and ouput tensor must have four dimensions.
 *  - The \b rois tensor should be 2-D array.
 *  - The shape of \b rois should be [rois_num, 5].
 *  - \p batch_id should be in the range of [0, \p batch_num - 1].
 *  - The spatial_scale should be greater than 0.
 *  - The group_size should be greater than 1.
 *  - THe output_dim should be greater than 1.
 *  - The group_size should be equal to pooled_height.
 *  - The pooled_height should be equal to pooled_width.
 *  - The fourth dimension of input tensor should be equal to pooled_height * pooled_width * output_dim.
 *  - The first dimension of output tensor and mapping_channel tensor must be the same size.
 *  - The second dimension of output tensor and mapping_channel tensor must be the same size.
 *  - The third dimension of output tensor and mapping_channel tensor must be the same size.
 *  - The fourth dimension of output tensor and mapping_channel tensor must be the same size.
 *  
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 * 
 *  @par Note
 *  - On MLU300 series, \b rois does not support NAN/INF.
 * 
 * @par Reference
 * - https://github.com/princewang1994/R-FCN.pytorch/tree/master/
 *   lib/model/psroi_pooling
 */
mluOpStatus_t MLUOP_WIN_API 
mluOpPsRoiPoolForward(mluOpHandle_t handle,
                      const int pooled_height, const int pooled_width,
                      const float spatial_scale, const int group_size,
                      const int output_dim,
                      const mluOpTensorDescriptor_t input_desc,
                      const void *input,
                      const mluOpTensorDescriptor_t rois_desc,
                      const void *rois,
                      const mluOpTensorDescriptor_t output_desc,
                      void *output,
                      const mluOpTensorDescriptor_t mapping_channel_desc,
                      void *mapping_channel);

/*!
 * @brief Generates fixed size feature map for each grid. Each value in the
 *   feature map is interpolated by bilinear sampling.
 *
 * @param[in] handle
 *   Input. Handle to a MLUOP context that is used to manage MLU devices and
 *   queues in ::mluOpRoiCropForward operation. For detailed information, see
 *   ::mluOpHandle_t.
 * @param[in] input_desc
 *   Input. The descriptor of the input tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[in] input
 *   Input. Pointer to the MLU memory that stores the input tensor.
 * @param[in] grid_desc
 *   Input. The descriptor of the grid tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[in] grid
 *   Input. Pointer to the MLU memory that stores the grid tensor. NaN and INF
 *   datas are not supported.
 * @param[in] output_desc
 *   Input. The descriptor of the output tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[out] output
 *   Output. Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Formula
 * - See "RoI Crop Operation" section in "Cambricon MLUOP User Guide" for
 *   details.
 *
 * @par Data Type
 * - Data types of input tensors and output tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - Input tensor: float.
 *   - Grid tensor: float.
 *   - Output tensor: float.
 * @par Data Layout
 * - The supported data layout of \b input , \b grid , \b output are as follows:
 *   - Input tensor: \p MLUOP_LAYOUT_NHWC.
 *   - Grid tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - Output tensor: \p MLUOP_LAYOUT_NHWC.
 *
 * @par Scale Limitation
 * - The input tensor, grid tensor and ouput tensor must have four dimensions.
 * - Size of the first dimension of input tensor is divisibled by size of the
 *   first dimension of grid tensor. 
 * - The second dimension of grid tensor and output tensor must be the same size.
 * - The third dimension of grid tensor and output tensor must be the same size.
 * - The fourth dimension of input tensor and output tensor must be the same size. 
 * - Size of the fourth dimension of grid tensor must be equal to 2.
 * - Grid tensor \b grid must meet the following data range:
 *   - Float: [-1.0,1.0].
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/princewang1994/R-FCN.pytorch/tree/master/lib/model/roi_crop
 */
mluOpStatus_t MLUOP_WIN_API mluOpRoiCropForward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const mluOpTensorDescriptor_t grid_desc,
    const void *grid, const mluOpTensorDescriptor_t output_desc, void *output);

/*!
 * @brief Computes the gradients of images \b grad_input based on the gradients
 *   \b grad_output and coordinate mapping parameter \b grid to perform the
 *   backpropagation.
 *
 * @param[in] handle
 *   Input. Handle to a MLUOP context that is used to manage MLU devices and
 *   queues in ::mluOpRoiCropBackward operation. For detailed information, see
 *   ::mluOpHandle_t.
 * @param[in] grad_output_desc
 *   Input. The descriptor of the grad_output tensor. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[in] grad_output
 *   Input. Pointer to the MLU memory that stores the gradient tensor \b grad_output
 *   in the backpropagation process.
 * @param[in] grid_desc
 *   Input. The descriptor of the grid tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[in] grid
 *   Input. Pointer to the MLU memory that stores the coordinate mapping
 *   tensor.
 * @param[in] grad_input_desc
 *   Input. The descriptor of the grad_input tensor. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[out] grad_input
 *   Output. Pointer to the MLU memory that stores the gradient tensor of the
 *   original images.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Formula
 * - See "RoI Crop Operation" section in "Cambricon MLUOP User Guide" for
 *   details.
 *
 * @par Data Type
 * - Data types of all tensors must be the same.
 * - The supported data types of all tensors are as follows:
 *   - Grad_input tensor: float.
 *   - Grad_output tensor: float.
 *   - Grid tensor: float.
 * @par Data Layout
 * - The supported data layout of \b grad_output , \b grid , \b grad_input are as
 *   follows.
 *   - Grad_output tensor: \p MLUOP_LAYOUT_NHWC.
 *   - Grid tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - Grad_input tensor: \p MLUOP_LAYOUT_NHWC.
 *
 * @par Scale Limitation
 * - The grad_output tensor, grid tensor and grad_input tensor must have four
 *   dimensions.
 * - Size of the first dimension of grad_input tensor is divisibled by size of
 *   the first dimension of grid tensor.
 * - The second dimension of grid tensor and grad_output tensor must be the same size.
 * - The third dimension of grid tensor and grad_output tensor must be the same size.
 * - The fourth dimension of grad_input \b grad_input tensor and grad_output tensor 
 *   \b grad_output must be the same size. 
 * - Size of the fourth dimension of grid tensor \b grid must be equal to 2.
 * - Grid tensor \b grid must meet the following data range:
 *   - Float: [-1.0,1.0].
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/princewang1994/R-FCN.pytorch/tree/master/lib/model/roi_crop
 */
mluOpStatus_t MLUOP_WIN_API mluOpRoiCropBackward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t grad_output_desc,
    const void *grad_output, const mluOpTensorDescriptor_t grid_desc,
    const void *grid, const mluOpTensorDescriptor_t grad_input_desc,
    void *grad_input);

/*!
 * @brief Computes sqrt on input tensor \b x, and returns the results in the
 *   output tensor \b y.
 *
 * @param[in] handle
 *   Input. Handle to a MLUOP context that is used to manage MLU devices and
 *   queues in the sqrt operation. For detailed information, see
 *   ::mluOpHandle_t.
 * @param[in] prefer
 *   Input. The \b prefer modes defined in ::mluOpComputationPreference_t enum.
 * @param[in] x_desc
 *   Input. The descriptor of the input tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[in] x
 *   Input. Pointer to the MLU memory that stores the input tensor.
 * @param[in] y_desc
 *   Input. The descriptor of the output tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
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
 * - The input tensor and output tensor must have the same shape, and the input
 *   tensor must meet the following input data range:
 *   - float: [1e-10,1e10].
 *   - half: [1e-3,1e-2] & [1e-1,60000].
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
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
 * @brief Computes gradient of sqrt on input tensor \b y and \b diff_y, and
 *   returns the results in the output tensor \b diff_x.
 *
 * @param[in] handle
 *   Input. Handle to a MLUOP context that is used to manage MLU devices and
 *   queues in the sqrt backward operation. For detailed information, see
 *   ::mluOpHandle_t.
 * @param[in] y_desc
 *   Input. The descriptor of the tensors. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[in] y
 *   Input. Pointer to the MLU memory that stores the input tensor.
 * @param[in] dy_desc
 *   Input. The descriptor of the tensors. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[in] diff_y
 *   Input. Pointer to the MLU memory that stores the input tensor.
 * @param[in] dx_desc
 *   Input. The descriptor of the tensors. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[out] diff_x
 *   Output. Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Formula
 * - See "Sqrt Backward Operation" section in "Cambricon MLUOP User Guide" for
 *   details.
 *
 * @par Data Type
 * - Data types of input tensors and output tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensors: half, float.
 *   - output tensor: half, float.
 *
 * @par Scale Limitation
 * - The input tensors and output tensor must have the same shape, and the input
 *   tensor \b y must meet the following input data range:
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
mluOpStatus_t MLUOP_WIN_API mluOpSqrtBackward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t y_desc, const void *y,
    const mluOpTensorDescriptor_t dy_desc, const void *diff_y,
    const mluOpTensorDescriptor_t dx_desc, void *diff_x);

#if defined(__cplusplus)
}
#endif

#endif  // MLUOP_EXAMPLE_H_
