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

/******************************************************************************
 * MLU-OPS: Cambricon Open Source operator library for Network
 ******************************************************************************/

#define MLUOP_MAJOR 1
#define MLUOP_MINOR 4
#define MLUOP_PATCHLEVEL 2
/*********************************************************************************
 * MLUOP_VERSION is deprecated and not recommended. To get the version of MLUOP, use
 * MLUOP_MAJOR, MLUOP_MINOR and MLUOP_PATCHLEVEL.
 ********************************************************************************/
#define MLUOP_VERSION (MLUOP_MAJOR * 1000 + MLUOP_MINOR * 100 + MLUOP_PATCHLEVEL)

#define MLUOP_DIM_MAX 8

#include <stdint.h>

#include "cn_api.h"
#include "cnrt.h"

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

/******************************************************************************
 * MLU-OPS Return Status
 ******************************************************************************/
/*! @brief Describes function return status.
 */
typedef enum {
  MLUOP_STATUS_SUCCESS         = 0, /*!< The operation is successfully completed. */
  MLUOP_STATUS_NOT_INITIALIZED = 1,
  /*!< MLU-OPS library is not initialized properly, which is usually caused by failing
       to call ::mluOpCreate, ::mluOpCreateTensorDescriptor or ::mluOpSetTensorDescriptor.
       Such error is usually due to incompatible MLU device or invalid driver environment.
       Notice that ::mluOpCreate should be called prior to any other MLU-OPS function. */
  MLUOP_STATUS_ALLOC_FAILED = 2,
  /*!< This error occurs when the resource allocation fails, which is usually caused by
       failing to call cnMallocHost due to exceeded memory usage. Make sure that
       the memory allocated previously is deallocated as much as possible. */
  MLUOP_STATUS_BAD_PARAM = 3,
  /*!< Invalid value or parameters are passed to the function, including data type, layout,
       dimensions, etc. */
  MLUOP_STATUS_INTERNAL_ERROR = 4,
  /*!< An error occurs inside of the function, which may indicate an internal error or bug in
       the library. This error is usually caused by failing to call cnrtMemcpyAsync.
       Check whether the memory passed to the function is deallocated before the completion
       of the routine. */
  MLUOP_STATUS_ARCH_MISMATCH = 5,
  /*!< Invalid MLU device which is not supported by current function. */
  MLUOP_STATUS_EXECUTION_FAILED = 6,
  /*!< An error occurs when the function fails to be executed on MLU device due to multiple reasons.
       You can check whether the hardware environment, driver version and other prerequisite
       libraries are correctly installed. */
  MLUOP_STATUS_NOT_SUPPORTED = 7,
  /*!< An error occurs when the requested functionality is not supported in this version but would
       be supported in the future. */
  MLUOP_STATUS_NUMERICAL_OVERFLOW = 8,
  /*!< A numerical overflow occurs when executing the function, which is usually due to large scale
       or inappropriate range of value of input tensor. */
} mluOpStatus_t;

/******************************************************************************
 * MLU-OPS Tensor Layout
 ******************************************************************************/
/*!
 * @brief Describes the data layouts in MLU-OPS.
 *
 * The data can be defined in three, four, or five dimensions.
 *
 * Take images for example, the format of the data layout can be NCHW:
 * - N: The number of images
 * - C: The number of image channels
 * - H: The height of images
 * - W: The weight of images
 *
 * Take sequence for example, the format of the data layout can be TNC:
 * - T: The timing steps of sequence
 * - N: The batch size of sequence
 * - C: The alphabet size of sequence
 */
typedef enum {
  MLUOP_LAYOUT_NCHW = 0,
  /*!< The data layout is in the following order: batch size, channel, height, and width. */
  MLUOP_LAYOUT_NHWC = 1,
  /*!< The data layout is in the following order: batch size, height, width, and channel. */
  MLUOP_LAYOUT_HWCN = 2,
  /*!< The data layout is in the following order: height, width, channel and batch size. */
  MLUOP_LAYOUT_NDHWC = 3,
  /*!< The data layout is in the following order: batch size, depth, height, width, and
   *   channel. */
  MLUOP_LAYOUT_ARRAY = 4,
  /*!< The data is multi-dimensional tensor. */
  MLUOP_LAYOUT_NCDHW = 5,
  /*!< The data layout is in the following order: batch size, channel, depth, height, and
   *   width. */
  MLUOP_LAYOUT_TNC = 6,
  /*!< The data layout is in the following order: timing steps, batch size, alphabet size. */
  MLUOP_LAYOUT_NTC = 7,
  /*!< The data layout is in the following order: batch size, timing steps, alphabet size. */
  MLUOP_LAYOUT_NC = 8,
  /*!< The data layout is in the following order: batch size, channel. */
  MLUOP_LAYOUT_NLC = 9,
  /*!< The data layout is in the following order: batch size, width, channel. */
  MLUOP_LAYOUT_NCL = 10,
  /*!< The data layout is in the following order: batch size, channel, length.*/
} mluOpTensorLayout_t;

/******************************************************************************
 * Cambricon MLU-OPS sequence data Layout
 ******************************************************************************/
/*!
 * @brief Enumeration variables describing the sequence data layouts.
 * N represents batch, B represents beam, T represents sequence length,
 * and C represents embedding size.
 */
typedef enum {
  MLUOP_SEQDATA_TNC        = 0,  /*!< Sequence data layout order: TNC. */
  MLUOP_SEQDATA_TNC_PACKED = 1,  /*!< Sequence data layout order: TNC_PACKED. */
  MLUOP_SEQDATA_NTC        = 2,  /*!< Sequence data layout order: NTC. */
  MLUOP_SEQDATA_NC         = 3,  /*!< Sequence data layout order:  NC. */
  MLUOP_SEQDATA_TNBC       = 4,  /*!< Sequence data layout order: TNBC. */
  MLUOP_SEQDATA_TBNC       = 5,  /*!< Sequence data layout order: TBNC. */
  MLUOP_SEQDATA_NBTC       = 6,  /*!< Sequence data layout order: NBTC. */
  MLUOP_SEQDATA_NTBC       = 7,  /*!< Sequence data layout order: NTBC. */
  MLUOP_SEQDATA_BNTC       = 8,  /*!< Sequence data layout order: BNTC. */
  MLUOP_SEQDATA_BTNC       = 9,  /*!< Sequence data layout order: BTNC. */
  MLUOP_SEQDATA_TN         = 10, /*!< Sequence data layout order: TN. */
  MLUOP_SEQDATA_NT         = 11, /*!< Sequence data layout order: NT. */
} mluOpSeqDataLayout_t;

/******************************************************************************
 * MLU-OPS Data Type
 ******************************************************************************/
/*! @brief Describes the data types in MLU-OPS. */
typedef enum {
  MLUOP_DTYPE_INVALID       = 0,  /*!< An invalid data type. */
  MLUOP_DTYPE_HALF          = 1,  /*!< A 16-bit floating-point data type. */
  MLUOP_DTYPE_FLOAT         = 2,  /*!< A 32-bit floating-point data type. */
  MLUOP_DTYPE_DOUBLE        = 14, /*!< A 64-bit floating-point data type. */
  MLUOP_DTYPE_INT8          = 3,  /*!< An 8-bit signed integer data type. */
  MLUOP_DTYPE_INT16         = 4,  /*!< A 16-bit signed integer data type. */
  MLUOP_DTYPE_INT31         = 5,  /*!< The data is a 31-bit signed integer data type. */
  MLUOP_DTYPE_INT32         = 6,  /*!< A 32-bit signed integer data type. */
  MLUOP_DTYPE_INT64         = 9,  /*!< A 64-bit signed integer data type. */
  MLUOP_DTYPE_UINT8         = 7,  /*!< An 8-bit unsigned integer data type. */
  MLUOP_DTYPE_UINT16        = 13, /*!< A 16-bit unsigned integer data type. */
  MLUOP_DTYPE_UINT32        = 11, /*!< A 32-bit unsigned integer data type. */
  MLUOP_DTYPE_UINT64        = 12, /*!< A 64-bit unsigned integer data type. */
  MLUOP_DTYPE_BOOL          = 8,  /*!< A boolean data type. */
  MLUOP_DTYPE_COMPLEX_HALF  = 15, /*!< A 32-bit complex number of two fp16. */
  MLUOP_DTYPE_COMPLEX_FLOAT = 16, /*!< A 64-bit complex number of two fp32. */
  MLUOP_DTYPE_BFLOAT16      = 17,
  /*!< The data is a 16-bit floating-point data type with one bit for sign,
   * 8 bits for exponent and 7 bits for fraction. */
} mluOpDataType_t;

/*!
 * @brief Describes whether to propagate NaN numbers.
 */
typedef enum {
  MLUOP_NOT_PROPAGATE_NAN = 0, /*!< The NaN numbers are not propagated . */
  MLUOP_PROPAGATE_NAN     = 1, /*!< The NaN numbers are propagated. */
} mluOpNanPropagation_t;

/*!
 * @brief Describes the options that can help choose the best suited algorithm used for
 * implementation of the activation and accumulation operations.
 **/
typedef enum {
  MLUOP_COMPUTATION_FAST = 0,
  /*!< Implementation with the fastest algorithm and lower precision. */
  MLUOP_COMPUTATION_HIGH_PRECISION = 1,
  /*!< Implementation with the high-precision algorithm regardless of the performance. */
  MLUOP_COMPUTATION_ULTRAHIGH_PRECISION = 2,
  /*!< Implementation with the ultrahigh-precision algorithm regardless of the performance. */
} mluOpComputationPreference_t;

/*!
 * @brief Describes the atomics modes in MLU-OPS.
 */
typedef enum {
  MLUOP_ATOMICS_NOT_ALLOWED = 1,
  /*!< The atomics is not allowed to cumulate results. */
  MLUOP_ATOMICS_ALLOWED = 2,
  /*!< The atomics is allowed to cumulate results. */
} mluOpAtomicsMode_t;

/*!
 * @brief Describes the rounding modes of quantization conversion.
 */
typedef enum {
  MLUOP_ROUND_HALF_TO_EVEN = 0,
  /*!< The rounding mode to round towards the nearest even neighbor is used for
   *   quantization conversion. */
  MLUOP_ROUND_HALF_UP = 1,
  /*!< The rounding mode to round up towards the nearest neighbor is used for
   *   quantization conversion. */
  MLUOP_ROUND_HALF_OFF_ZERO = 2,
  /*!< The rounding mode to round half away from zero is used for quantization
   *   conversion. */
} mluOpQuantizeRoundMode_t;

/*!
 * @brief Describes the modes of quantization method.
 */
typedef enum {
  MLUOP_QUANTIZE_POSITION = 0,
  /*!< Quantization method with position factor and without scale factor. */
  MLUOP_QUANTIZE_POSITION_SCALE = 1,
  /*!< Quantization method with position and scale factors. */
  MLUOP_QUANTIZE_POSITION_SCALE_OFFSET = 2,
  /*!< Asymmetric quantization method with position, scale, and offset factors. */
} mluOpQuantizeMode_t;

/*!
 * @brief Describes the bases that are used in the implementation of the log function.
 */
typedef enum {
  MLUOP_LOG_E  = 0, /*!< The base e is used. */
  MLUOP_LOG_2  = 1, /*!< The base 2 is used. */
  MLUOP_LOG_10 = 2, /*!< The base 10 is used. */
} mluOpLogBase_t;

/*!
 * @brief Describes the pointer modes that are used in the implementation of the fill function.
 */
typedef enum {
  MLUOP_POINTER_MODE_HOST = 0,
  /*!< A host pointer, which means that the values passed by reference are on the host. */
  MLUOP_POINTER_MODE_DEVICE = 1,
  /*!< A device pointer, which means that the values passed by reference are on the device. */
} mluOpPointerMode_t;

/*!
 * @brief Describes the input box modes that can be used to implement the Nms operation.
 */
typedef enum {
  MLUOP_NMS_BOX_DIAGONAL = 0, /*!< The box mode is [x1, y1, x2, y2]. */
  MLUOP_NMS_BOX_CENTER   = 1,
  /*!< The box mode is [x_center, y_center, width, height] where width > 0 and * height > 0. */
} mluOpNmsBoxPointMode_t;

/*!
 * @brief Describes the output modes that can be used to implement the Nms operation.
 */
typedef enum {
  MLUOP_NMS_OUTPUT_TARGET_INDICES = 0,
  /*!< Returns target indices, which are sorted in decreasing order of confidences. */
  MLUOP_NMS_OUTPUT_TARGET_CONFIDENCE_AND_POS_1 = 1,
  /*!< Returns target confidences and positions with the order of confidence_0, x_01, y_01, x_02, y_02,
   * confidence_1, x_11, y_11, x_12, y_12, ... ,
   * confidence_n, x_n1, y_n1, x_n2, and y_n2. The (x_01, y_01) and (x_02, y_02) represent the top left corner
   * and bottom right corner coordinates of the first box, respectively.
   */
  MLUOP_NMS_OUTPUT_TARGET_CONFIDENCE_AND_POS_2 = 2,
  /*!< Returns target confidences and positions with the order of confidence_0,
   * confidence_1, ... , confidence_n, x_01, x_11, ... , x_n1, y_01, y_11, ... , y_n1, x_02, x_12, ... , x_n2, y_02,
   * y_12, ... , and y_n2. The (x_01, y_01) and (x_02, y_02) represent the top left corner and
   * bottom right corner coordinates of the first box, respectively.
   */
  MLUOP_NMS_OUTPUT_TARGET_BATCH_AND_CLASS = 3,
  /*!< Returns batch indices, class indices, and positions with the order of batch_0, class_0, box_0,
   * ... , batch_0, class_0, box_m, batch_0, class_1, box_0, ... , batch_0, class_1, box_m, ... , ... ,
   * batch_s, class_n, and box_m.
   */
} mluOpNmsOutputMode_t;

/*!
 * @brief Describes the algorithms that can be used to implement the Nms operation.
 */
typedef enum {
  MLUOP_NMS_HARD_NMS = 0,
  /*!< A type of algorithm which updates confidence using hard Nms, for example
   *confidence = IOU < IOU_threshold ? confidence : 0.
   */
  MLUOP_NMS_SOFT_NMS_LINEAR = 1,
  /*!< A type of algorithm which updates confidence using linear method, for example
   * confidence = IOU < IOU_threshold ? confidence : confidence * (1 - IOU).
   */
  MLUOP_NMS_SOFT_NMS_GAUSSIAN = 2,
  /*!< A type of algorithm which updates confidence using Gaussian method, for example
   *confidence = confidence * exp{- \f$IOU^2\f$ / (2 * sigma)}.
   */
} mluOpNmsMethodMode_t;

/*!
 * @brief Describes the algorithms that can be used to implement the Nms operation.
 */
typedef enum {
  MLUOP_NMS_ALGO_EXCLUDE_BOUNDARY = 0,
  /*!< Implements Nms with boundary excluded. In this mode,
   * the height or width of boxes is ``(x2 - x1)``.
   */
  MLUOP_NMS_ALGO_INCLUDE_BOUNDARY = 1,
  /*!< Implements Nms with boundary included. In this mode,
   * the height or width of boxes is ``(x2 - x1 + offset)``.
   */
} mluOpNmsAlgo_t;

/******************************************************************************
 * MLU-OPS Data Structure: Customized Operation
 ******************************************************************************/

/*!
 * @brief Describes the data type of indices used in the reduce function.
 */
typedef enum {
  MLUOP_32BIT_INDICES = 0, /*!< The data type of indices is unsigned int. */
  MLUOP_16BIT_INDICES = 1, /*!< The data type of indices is unsigned short. */
} mluOpIndicesType_t;

/*!
 * @brief Describes the reduction applied to the output in the implementation of the loss function.
 */
typedef enum {
  MLUOP_LOSS_REDUCTION_NONE = 0,
  /*!< No reduction is applied in the operation.*/
  MLUOP_LOSS_REDUCTION_SUM = 1,
  /*!< The elements of output are summed in the operation.*/
  MLUOP_LOSS_REDUCTION_MEAN = 2,
  /*!< The weighted mean of the output is applied in the operation.*/
} mluOpLossReduction_t;

/*!
 * @brief Describes the modes that are used in the Reduce function.
 */
typedef enum {
  MLUOP_REDUCE_DSUM  = 0, /*!< Computes the sum value. */
  MLUOP_REDUCE_DMEAN = 1, /*!< Computes the mean value. */
  MLUOP_REDUCE_DMAX  = 2, /*!< Computes the maximum value. */
} mluOpReduceMode_t;

/*!
 * @brief Enumeration variables describing the pooling modes that can be used to
 * implement the pooling operation.
 */
typedef enum {
  MLUOP_POOLING_MAX                           = 0, /*!< The max pooling mode is implemented.*/
  MLUOP_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
  /*!< The average pooling with padding mode is implemented.*/
  MLUOP_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
  /*!< The average pooling without padding mode is implemented.*/
  MLUOP_POOLING_FIXED = 3,
  /*!< The fixed mode is implemented. This mode is used in the unpool operation.
   * In this mode, each input pixel will be put to the center of the pooling kernel
   * regardless of the index.*/
} mluOpPoolingMode_t;

/******************************************************************************
 * MLU-OPS Runtime Management
 ******************************************************************************/

/*!
 * @struct mluOpContext
 * @brief Describes the Cambricon MLU-OPS context.
 */
struct mluOpContext;

/*!
 * Pointer to ::mluOpContext struct that holds the Cambricon MLU-OPS context.
 *
 * MLU device resources cannot be accessed directly, so MLU-OPS uses
 * handle to manage MLU-OPS context including MLU device information
 * and queues.
 *
 * The MLU-OPS context is created with ::mluOpCreate and the returned
 * handle should be passed to all the subsequent function calls.
 * You need to destroy the MLU-OPS context at the end with ::mluOpDestroy.
 */
typedef struct mluOpContext *mluOpHandle_t;

/*!
 * The descriptor of the collection of tensor which is used in the RNN operation, such as weight,
 * bias.
 * You need to call ::mluOpCreateTensorSetDescriptor to create a descriptor, and
 * call ::mluOpInitTensorSetMemberDescriptor to set the information about each tensor in
 * the tensor set. If the data type of the tensor in the tensor set is in fixed-point data type,
 * call ::mluOpInitTensorSetMemberDescriptorPositionAndScale to set quantization
 * parameters.
 * At last, you need to destroy the descriptor at the end with
 * ::mluOpDestroyTensorSetDescriptor.
 */
typedef struct mluOpTensorSetStruct *mluOpTensorSetDescriptor_t;

// Group: Runtime Management
/*!
 * @brief Initializes the MLU-OPS library and creates a handle \b handle to a struct
 * that holds the MLU-OPS library context. It allocates hardware resources on the host
 * and device. You need to call this function before any other MLU-OPS function.
 *
 * You need to call ::mluOpDestroy to release the resources later.
 *
 * @param[out] handle
 * Pointer to a Cambricon MLU-OPS context that is used to manage MLU devices and queues.
 * For detailed information, see ::mluOpHandle_t.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreate(mluOpHandle_t *handle);

// Group: Runtime Management
/*!
 * @brief Updates the MLU-OPS context information that is held by \b handle. This function
 * should be called if you call CNDrv API cnSetCtxConfigParam to set the context information.
 * The related context information will be synchronized to MLU-OPS with this function. For
 * detailed information, see "Cambricon CNDrv Developer Guide".
 *
 * @param[in] handle
 * Pointer to a Cambricon MLU-OPS context that is used to manage MLU devices. For detailed information,
 * see ::mluOpHandle_t.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpUpdateContextInformation(mluOpHandle_t handle);

// Group: Runtime Management
/*!
 * @brief Releases the resources of the specified MLU-OPS handle \b handle that was
 * created by ::mluOpCreate. It is usually the last call to destroy
 * the handle to the MLU-OPS handle.
 *
 * @param[in] handle
 * Pointer to the MLU devices that holds information to be destroyed.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDestroy(mluOpHandle_t handle);

// Group: Runtime Management
/*!
 * @brief Sets the runtime queue \b queue in the handle \b handle. The queue is used to
 * launch kernels or to synchronize to this queue.
 *
 * Before setting a queue \b queue, you need to call ::mluOpCreate to initialize
 * MLU-OPS library, and call cnrtCreateQueue to create a queue \b queue.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues. For detailed information, see ::mluOpHandle_t.
 * @param[in] queue
 * The runtime queue to be set to the MLU-OPS handle.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetQueue(mluOpHandle_t handle, cnrtQueue_t queue);

// Group: Runtime Management
/*!
 * @brief Retrieves the queue \b queue that was previously set to the handle \b handle.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues. For
 * detailed information, see ::mluOpHandle_t.
 * @param[out] queue
 * Pointer to the queue that was previously set to the specified handle.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetQueue(mluOpHandle_t handle, cnrtQueue_t *queue);

// Group: Runtime Management
/*!
 * @brief Converts the MLU-OPS enumerated status code to ASCIIZ static string and returns
 * a pointer to the MLU memory that holds information about ASCIIZ static string with
 * the status name. For example, when the input argument is ::MLUOP_STATUS_SUCCESS, the
 * returned string is ::MLUOP_STATUS_SUCCESS. When an invalid status value is passed to
 * the function, the returned string is ::MLUOP_STATUS_BAD_PARAM.
 *
 * @param[in] status
 * The MLU-OPS enumerated status code.
 *
 * @par return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
const char *
mluOpGetErrorString(mluOpStatus_t status);

// Group: Tensor
/*!
 * @brief Gets the size of a data type in ::mluOpDataType_t.
 *
 * @param[in] data_type
 * For detailed information, see ::mluOpDataType_t.
 * @param[out] size
 * Host pointer to the size of the data type.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetSizeOfDataType(mluOpDataType_t data_type, size_t *size);

// Group: Version Management
/*!
 * @brief Retrieves the version of MLU-OPS library. The version of MLU-OPS
 * is composed of \b major, \b minor, and \b patch. For instance, major = 1,
 * minor = 2, patch = 3, the version of MLU-OPS library is 1.2.3.
 *
 * @param[in] major
 * Pointer to scale factor that gets the major version of MLU-OPS library.
 * @param[in] minor
 * Pointer to scale factor that gets the minor version of MLU-OPS library.
 * @param[in] patch
 * Pointer to scale factor that gets the patch version of MLU-OPS library.
 *
 * @par return
 * - None.
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
void
mluOpGetLibVersion(int *major, int *minor, int *patch);

// Group: QuantizeRoundMode
/*!
 * @brief Updates the specific rounding mode of MLU-OPS context information that is held by the \b
 * handle. This function should be called if you want to change the MLU-OPS rounding mode that
 * is used to cumulate the results. For detailed information, see "Cambricon CNDrv Developer
 * Guide".
 *
 * @param[in] handle
 * Pointer to a Cambricon MLU-OPS context that is used to manage MLU devices and queues. For detailed
 * information, see ::mluOpHandle_t.
 * @param[in] round_mode
 * The rounding mode of quantization conversion to be set to the MLU-OPS handle.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetQuantizeRoundMode(mluOpHandle_t handle, mluOpQuantizeRoundMode_t round_mode);

// Group: QuantizeRoundMode
/*!
 * @brief Retrieves the rounding mode of a specific MLU-OPS context.
 *
 * @param[in] handle
 * Pointer to a Cambricon MLU-OPS context that is used to manage MLU devices and queues. For detailed
 * information, see ::mluOpHandle_t.
 * @param[out] round_mode
 * The rounding mode of quantization conversion that was previously set to the specified handle.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The rounding mode of initialized ::mluOpHandle_t is MLUOP_ROUND_TO_EVEN.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetQuantizeRoundMode(mluOpHandle_t handle, mluOpQuantizeRoundMode_t *round_mode);

// Group: Runtime Management
/*!
 * @brief Updates the specific atomics mode of MLU-OPS context information that is held by the
 * \b handle. This function should be called if you want to change the atomics mode that is
 * used to cumulate the results. For detailed information, see "Cambricon CNDrv Developer Guide".
 *
 * @param[in] handle
 * Pointer to a Cambricon MLU-OPS context that is used to manage MLU devices and queues. For detailed
 * information, see ::mluOpHandle_t.
 * @param[in] atomics_mode
 * The atomics mode.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetAtomicsMode(mluOpHandle_t handle, mluOpAtomicsMode_t atomics_mode);

// Group: Runtime Management
/*!
 * @brief Retrieves the atomics mode of a specific MLU-OPS context.
 *
 * @param[in] handle
 * Pointer to a Cambricon MLU-OPS context that is used to manage MLU devices and queues. For
 * detailed information, see ::mluOpHandle_t.
 * @param[out] atomics_mode
 * The atomics mode.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The default atomics mode of default initialized ::mluOpHandle_t is ::MLUOP_ATOMICS_NOT_ALLOWED.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetAtomicsMode(mluOpHandle_t handle, mluOpAtomicsMode_t *atomics_mode);

/******************************************************************************
 * MLU-OPS Data Structure: Descriptor
 * The struct represent node, weight and the AI network layer
 ******************************************************************************/
/*!
 * The descriptor of a tensor that holds the information including tensor
 * layout, data type, the number of dimensions, shape and strides.
 *
 * You need to call ::mluOpCreateTensorDescriptor to create a descriptor,
 * and call ::mluOpSetTensorDescriptor or ::mluOpSetTensorDescriptorEx
 * to set the tensor information to the descriptor. Also, you need to destroy
 * the MLU-OPS context at the end with ::mluOpDestroyTensorDescriptor.
 */
typedef struct mluOpTensorStruct *mluOpTensorDescriptor_t;

/*! The descriptor of Sequence Data that holds the dimensions,
 * layout, data type, sequence length, padding fill, position, and scale.
 * The total size of the tensor descriptor supports up to 2 Giga elements.
 * Call ::mluOpCreateSeqDataDescriptor to create a descriptor, and
 * call ::mluOpSetSeqDataDescriptor_v2 to set the sequence data information to the descriptor.
 * If the sequence data is in fixed-point data type, call ::mluOpSetSeqDataDescriptorPositionAndScale
 * to set the position and scale of the sequence data.
 * To destroy the descriptor, call ::mluOpDestroySeqDataDescriptor.
 */
typedef struct mluOpSeqDataStruct *mluOpSeqDataDescriptor_t;

// Group: SeqData
/*!
 *  @brief Creates a sequence data instance \p seq_data_desc that holds the dimensions, data type,
 *  sequence lengths, padding fill and layout of sequence data on the host memory.
 *
 *  Use ::mluOpSetSeqDataDescriptor_v2 to configure the descriptor and ::mluOpDestroySeqDataDescriptor
 *  function to destroy the sequence data descriptor.
 *
 *  @param[out] seq_data_desc
 *  Pointer to the host memory that holds information about
 *  the struct of the sequence data descriptor.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - None.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 *
 *  @par Reference
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateSeqDataDescriptor(mluOpSeqDataDescriptor_t *seq_data_desc);

// Group: SeqData
/*!
 * @brief Sets the sequence data descriptor \p seq_data_desc that holds the dimensions,
 * data type, sequence lengths, padding fill and layout of the sequence data.
 *
 * The number of dimensions in the \p dimSize[] is defined by \p dimNb. For example,
 * if the layout of the sequence data is set to ::MLUOP_SEQDATA_NC, the \p dimNb is 2,
 * with \p dimSize={batch, embedding}.
 *
 * The ::mluOpSeqDataDescriptor_t container is a collection of fixed-length sequential
 * vectors, similar to the words constructing sentences. The T dimension described in the
 * ::mluOpSeqDataLayout_t is the time dimension. Different sequences are bundled together to a
 * batch. The beam dimension described in the ::mluOpSeqDataLayout_t is
 * different candidates presenting a similar meaning in a typical translation task. The original
 * sentence can be translated to several versions before picking the optimal one, and the number
 * of candidates is beam.
 *
 * Note that different sentences have different sequence lengths, even inside a beam.
 * \p seqLengthArray is to record the real sequence lengths before padding to the maximum sequence
 * length. The value of \p seqLengthArray should follow a batch-beam order, in despite of
 * sequence data layout. Take a sequence of batch=3, beam=2 for example, the \p seqLengthArray
 * should be as follows:
   @verbatim
   {batch_idx = 0, beam_idx = 0}
   {batch_idx = 0, beam_idx = 1}
   {batch_idx = 1, beam_idx = 0}
   {batch_idx = 1, beam_idx = 1}
   {batch_idx = 2, beam_idx = 0}
   {batch_idx = 2, beam_idx = 1}
   @endverbatim
 * If the real sequence lengths are not requested, pass NULL to \p seqLengthArray in this function.
 *
 * The \p seqLengthArraySize should be batch * beam, which is 6 in the example above.
 *
 * The \p PaddingFill describes whether the sequence data needs to be padded using a
 * specified value. In the multi-head attention operation, the padding part should be zero before
 * entering the attention part to ensure the result validity. If the sequence data is padding
 * zero in advance, pass NULL to \p PaddingFill in this function. Otherwise, pass a pointer to padding
 * value (e.g. float a = 0, &a) to \p PaddingFill to indicate this function that extra padding are
 * needed.
 *
 * @param[in,out] seq_data_desc
 *   Input/output. The descriptor of the sequence data. For detailed information,
 *   see ::mluOpSeqDataDescriptor_t.
 * @param[in] layout
 *   The layout of the sequence data. See ::mluOpSeqDataLayout_t for the description of the
 *   enumeration type.
 * @param[in] dtype
 *   The data type of the sequence data. See ::mluOpDataType_t for the description of the
 *   enumeration type.
 * @param[in] dimNb
 *   The number of dimensions of the sequence data.
 * @param[in] dimSize
 *   An array that contains the size of the sequence data for each dimension.
 * @param[in] seqLengthArraySize
 *   Number of elements in sequence length array, \p seqLengthArray[]. It should be
 *   batch * beam. The batch and beam are described in the ::mluOpSeqDataLayout_t.
 * @param[in] seqLengthArray
 *   An integer array recording the length of all sequences. Note that the array should be
 *   set in the batch-beam order, in despite of sequence data layout. Set this parameter to NULL
 *   when sequence length array is not requested.
 * @param[in] paddingFill
 *   A host pointer to the data type \p dtype to fill up the padding vectors within
 *   the valid length of each sequence. Use NULL when extra padding is not requested.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - Before calling this function, ::mluOpCreateSeqDataDescriptor should be called.
 *
 * @note
 * - dimSize[0] represents the highest dimension, and dimSize[dimNb - 1] represents
 *   the lowest dimension.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetSeqDataDescriptor_v2(mluOpSeqDataDescriptor_t seq_data_desc,
                             mluOpSeqDataLayout_t layout,
                             mluOpDataType_t dtype,
                             int dimNb,
                             const int64_t dimSize[],
                             int seqLengthArraySize,
                             const int seqLengthArray[],
                             void *paddingFill);

// Group: SeqData
/*!
 * @brief Destroys a sequence data descriptor \p seq_data_desc that was created by
 * ::mluOpCreateSeqDataDescriptor.
 *
 * @param[in] seq_data_desc
 * A sequence data descriptor created by ::mluOpCreateSeqDataDescriptor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 *  - None.
 *
 *  @par Reference
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDestroySeqDataDescriptor(mluOpSeqDataDescriptor_t seq_data_desc);

// Group: SeqData
/*!
 * @brief Sets the position \p position and scale \p scale factors used in fixed-point quantization.
 * It is only used if you have quantized the input data with the symmetric fixed-point
 * quantization with scale factor quantization method. For more information about quantization,
 * see "Cambricon MLU-OPS User Guide".
 *
 * @param[in] seq_data_desc
 * The descriptor of the sequence data. For detailed information,
 * see ::mluOpSeqDataDescriptor_t.
 * @param[in] position
 * An integer of fixed position factor that is used for quantization.
 * @param[in] scale
 * A scalar of scale factor that is used for quantization.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - Before calling this function, ::mluOpCreateSeqDataDescriptor, ::mluOpSetSeqDataDescriptor_v2
 *   should be called.
 *
 * @note
 * - If the sequence data is in fixed-point data type, you need to call this function.
 *   This function is only used in the inference mode.
 * - The \p position should be limited in [-128, 127], otherwise the result is undefined.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 *  - None.
 *
 *  @par Reference
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetSeqDataDescriptorPositionAndScale(mluOpSeqDataDescriptor_t seq_data_desc, int position, float scale);

// Group: SeqData
/*!
 * @brief Retrieves the sequence data descriptor \p seq_data_desc that holds the dimensions,
 * data type, layout, padding fill, and the sequence lengths of the input sequence data.
 *
 * @param[in] seq_data_desc
 *   The descriptor of the sequence data. For detailed information,
 *   see ::mluOpSeqDataDescriptor_t.
 * @param[out] layout
 *   The layout of the sequence data. See ::mluOpSeqDataLayout_t.
 * @param[out] dtype
 *   The data type of the sequence data. See ::mluOpDataType_t.
 * @param[out] dimNb
 *   The number of dimensions of the sequence data.
 * @param[out] dimSize
 *   An array containing the size of the sequence data for each dimension.
 * @param[out] seqLengthArraySize
 *   Number of elements in sequence length array, \p seqLengthArray[]. It is equal to
 *   batch * beam (N and B described in the ::mluOpSeqDataLayout_t).
 * @param[out] seqLengthArray
 *   An array of integers recording the length of all sequences. Note that the array is
 *   ordered in a batch-beam order, regardless of the sequence data layout. Return \p NULL
 *   when the sequence length array is not set.
 * @param[out] paddingFill
 *   A host pointer to the data type \p dtype to fill the padding vectors within
 *   the valid length of each sequence. Use NULL when extra padding is not required.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - Before calling this function, ::mluOpCreateSeqDataDescriptor, ::mluOpSetSeqDataDescriptor_v2
 *   should be called.
 *
 * @note
 * - ``dimSize[0]`` represents the highest dimension, and ``dimSize[dimNb - 1]`` represents
 *   the lowest dimension.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetSeqDataDescriptor_v2(const mluOpSeqDataDescriptor_t seq_data_desc,
                             mluOpSeqDataLayout_t *layout,
                             mluOpDataType_t *dtype,
                             int *dimNb,
                             int64_t dimSize[],
                             int64_t *seqLengthArraySize,
                             int64_t seqLengthArray[],
                             void *paddingFill);

/*!
 * The descriptor of the Nms function that holds compute type, bias type,
 * transpose flag.
 *
 * You need to call ::mluOpCreateNmsDescriptor to create a descriptor, and call
 * ::mluOpSetNmsDescriptor to set the information of the Nms
 * to the descriptor. Also, you need to destroy the MLU-OPS context at the end with
 * ::mluOpDestroyNmsDescriptor.
 */
typedef struct cnnlNmsStruct *mluOpNmsDescriptor_t;

/*!
 * The descriptor of a tensor that holds the information including tensor
 * shape, the number of dimensions, pad, strides, dilation, sub_m, and transpose.
 *
 * You need to call ::mluOpCreateSparseConvolutionDescriptor to create a descriptor,
 * and call ::mluOpSetSparseConvolutionDescriptor to set the tensor information to
 * the descriptor. Also, you need to destroy the MLU-OPS context at the end with
 * ::mluOpDestroySparseConvolutionDescriptor.
 */
typedef struct mluOpSparseConvolutionStruct *mluOpSparseConvolutionDescriptor_t;

/*! The descriptor of ::mluOpRoiAlignForward_v2 that holds parameter information.
 *
 *  You need to call ::mluOpCreateRoiAlignForwardDescriptor to create a descriptor,
 *  and call ::mluOpSetRoiAlignForwardDescriptor_v2 to set the information of
 *  ::mluOpRoiAlignForward_v2 operation to the descriptor. Also, you need to destroy the MLU-OPS context
 *  at the end with ::mluOpDestroyRoiAlignForwardDescriptor.
 */
typedef struct cnnlRoiAlignStruct *mluOpRoiAlignForwardDescriptor_t;

/*!
 * The descriptor of deformable convolution function that holds the deformable convolution
 * information including the number of input dimensions, padding, stride, dilation,
 * deformable group, convolution group, and img2col_step.
 *
 * You need to call the ::mluOpCreateDCNDescriptor function to create a descriptor, and call the
 * ::mluOpSetDCNDescriptor function to set the information of the deformable convolution operation
 * to the descriptor. Also, you need to destroy the Cambricon MLU-OPS context at the end with the
 * ::mluOpDestroyDCNDescriptor function.*/
typedef struct cnnlDCNStruct *mluOpDCNDescriptor_t;

/*!
 * The descriptor of CARAFE (Content-Aware ReAssembly of FEatures) operation that holds
 * CARAFE information including the number of input dimensions, kernel size, group size,
 * and scale factor.
 *
 * You need to call ::mluOpCreateCarafeDescriptor to create a descriptor,
 * and call ::mluOpSetCarafeDescriptor to set the information of the CARAFE operation
 * to the descriptor. Also, you need to destroy the MLU-OPS context at the end with
 * ::mluOpDestroyCarafeDescriptor.
 */
typedef struct mluOpCarafeStruct *mluOpCarafeDescriptor_t;

// Group: Tensor
/*!
 * @brief Creates a tensor descriptor pointed by \b desc that holds the dimensions, data type,
 * and layout of input tensor. If the input tensor is in fixed-point data type,
 * ::mluOpSetTensorDescriptorPositionAndScale or ::mluOpSetTensorDescriptorPosition
 * needs to be called to set quantization parameters.
 *
 * ::mluOpDestroyTensorDescriptor needs to be called to destroy the tensor descriptor
 * later.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateTensorDescriptor(mluOpTensorDescriptor_t *desc);

// Group: SparseConv
/*!
 * @brief Creates a tensor descriptor pointed by \b desc that holds the dimensions, pad, stride,
 * dilation, sub_m, transpose, inverse and layout of input filter and output tensor shape.
 * ::mluOpSetSparseConvolutionDescriptor needs to be called to set parameters.
 *
 * ::mluOpDestroySparseConvolutionDescriptor needs to be called to destroy the
 * tensor descriptor later.
 *
 * @param[in] desc
 * Pointer to the struct that holds information about the tensor descriptor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateSparseConvolutionDescriptor(mluOpSparseConvolutionDescriptor_t *desc);

// Group: SparseConv
/*!
 * @brief Destroys a convolution descriptor \b desc that was previously created with the
 * ::mluOpCreateSparseConvolutionDescriptor function.
 *
 * The sparse convolution descriptor is defined in ::mluOpSparseConvolutionDescriptor_t
 * and holds the information about the sparse convolution forward or backward operation.
 *
 * @param[in] desc
 * The sparse convolution descriptor to be destroyed.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - This function should be called to destroy the sparse convolution descriptor.
 *   Otherwise, the memory leak may occur.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDestroySparseConvolutionDescriptor(mluOpSparseConvolutionDescriptor_t desc);

// Group: Tensor
/*!
 * @brief Creates a group of tensor descriptor stored by \b group_desc that holds the
 * dimensions, data_type, and layout of input tensors. If the input tensor is in
 * fixed-point data type, ::mluOpSetTensorDescriptorPositionAndScale or
 * ::mluOpSetTensorDescriptorPosition needs to be called to set quantization
 * parameters.
 *
 * @param[in] group_desc
 * An array of pointers to the structs that hold information about the tensor descriptor.
 * @param[in] desc_num
 * The length of the input array \b group_desc.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - ::mluOpDestroyTensorDescriptor needs to be called for each descriptor
 *   to destroy all tensors in group_desc or ::mluOpDestroyGroupTensorDescriptors
 *   needs to be called to destroy the all tensor descriptors in group_desc later.
 *
 * @par Note
 * - None
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateGroupTensorDescriptors(mluOpTensorDescriptor_t *group_desc[], const int desc_num);

// Group: Tensor
/*!
 * @brief Initializes the tensor descriptor pointed by \b desc that was previously created
 * with ::mluOpCreateTensorDescriptor, and sets the information about the
 * dimensions, data type, and layout of the input tensor.
 *
 * If ::mluOpSetTensorDescriptor is called, you do not need to specify the strides of all
 * dimensions. The strides are inferred by parameters passed to this function. Also, the data
 * will be treated as contiguous in memory with no padding between dimensions. To specify the
 * strides of all dimensions, you can call ::mluOpSetTensorDescriptorEx. But the data might not
 * be treated as contiguous in memory.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] layout
 * The layout of the input tensor. For detailed information, see ::mluOpTensorLayout_t.
 * @param[in] dtype
 * The data type of the input tensor. For detailed information, see ::mluOpDataType_t.
 * @param[in] dimNb
 * The number of dimensions in the input tensor of the initialized operation.
 * @param[in] dimSize
 * An array that contains the size of the tensor for each dimension.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - dimSize[0] represents the highest dimension, dimSize[DIM_MAX - 1] represents the lowest
 *   dimension, and DIM_MAX represents the number of dimensions in the input tensor.
 * - This function cannot be called continuously. You need to call ::mluOpResetTensorDescriptor
 *   before calling ::mluOpSetTensorDescriptor to avoid memory leaks.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetTensorDescriptor(
    mluOpTensorDescriptor_t desc, mluOpTensorLayout_t layout, mluOpDataType_t dtype, int dimNb, const int dimSize[]);

// Group: Tensor
/*!
 * @brief Sets the pointer mode \p pointer_mode factor for the input tensor descriptor \p desc.
 *
 * @param[in,out] desc
 *   The descriptor of the tensor. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[in] pointer_mode
 *   The pointer mode of the input tensor. For detailed information, see ::mluOpPointerMode_t.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @note
 * - Currently, \p pointer_mode setting to MLUOP_POINTER_MODE_HOST is only supported when the number
 *   of dimensions of \p desc is 0.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetTensorDescriptorPointerMode(mluOpTensorDescriptor_t desc, mluOpPointerMode_t pointer_mode);

// Group: Tensor
/*!
 * @brief Retrieves the pointer mode of the input tensor descriptor \p desc set by
 * ::mluOpSetTensorDescriptorPointerMode.
 *
 * @param[in] desc
 *   The descriptor of the tensor. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[out] pointer_mode
 *   Pointer to the host memory holding information about the pointer mode of the input tensor.
 *   For detailed information, seee ::mluOpPointerMode_t.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorDescriptorPointerMode(mluOpTensorDescriptor_t desc, mluOpPointerMode_t *pointer_mode);

// Group: Tensor
/*!
 * @brief Initializes the tensor descriptor pointed by \b desc that was previously created
 * with ::mluOpCreateTensorDescriptor, and sets the information about the
 * dimensions, data type, and layout of the input tensor.
 *
 * If ::mluOpSetTensorDescriptor_v2 is called, you do not need to specify the strides of all
 * dimensions. The strides are inferred by parameters passed to this function. Also, the data
 * will be treated as contiguous in memory with no padding between dimensions. To specify the
 * strides of all dimensions, you can call ::mluOpSetTensorDescriptorEx_v2. But the data might not
 * be treated as contiguous in memory.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] layout
 * The layout of the input tensor. For detailed information, see ::mluOpTensorLayout_t.
 * @param[in] dtype
 * The data type of the input tensor. For detailed information, see ::mluOpDataType_t.
 * @param[in] dimNb
 * The number of dimensions in the input tensor of the initialized operation.
 * @param[in] dimSize
 * An array that contains the size of the tensor for each dimension.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - dimSize[0] represents the highest dimension, dimSize[DIM_MAX - 1] represents the lowest
 *   dimension, and DIM_MAX represents the number of dimensions in the input tensor.
 * - This function cannot be called continuously. You need to call ::mluOpResetTensorDescriptor
 *   before calling ::mluOpSetTensorDescriptor to avoid memory leaks.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetTensorDescriptor_v2(mluOpTensorDescriptor_t desc,
                            mluOpTensorLayout_t layout,
                            mluOpDataType_t dtype,
                            int dimNb,
                            const int64_t dimSize[]);

// Group: SparseConv
/*!
 * @brief Initializes the sparse convolution descriptor \b desc that was previously created
 * with ::mluOpCreateSparseConvolutionDescriptor, and sets the information
 * about the convolution forward and backward operation to the convolution descriptor
 * \b desc. The information includes the number of the convolution dimensions \b dimNb,
 * the padding size for each dimension \b pad, the stride of the sliding window for
 * each dimension \b stride, the dilation factor for each dimension \b dilation, and
 * the size of \b input , \b filter , and \b output.
 *
 * @param[in] desc
 * The descriptor of the sparse convolution operation. For detailed information,
 * see ::mluOpSparseConvolutionDescriptor_t.
 * @param[in] dimNb
 * The number of dimensions in the input tensor of the convolution operation.
 * Currently, the value of this parameter can only be set to 4 or 5. The value of this parameter
 * should be the same as the one you set in the input tensor descriptor.
 * @param[in] batch_size
 * The number of N-dimensions in the tensor.
 * @param[in] pad
 * An array that stores the zero-padding size for each dimension of the input tensor
 * used in the convolution operation.
 * For each dimension, the padding size represents the number of zeros to be concatenated at the
 * start and end of that dimension. If \b dimNb is set to 4, the padding is on top, bottom, left,
 * and right. If \b dimNb is set to 5, the padding is on front, back, top, bottom, left,
 * and right. The value of this parameter should be greater than or equal to 0.
 * @param[in] stride
 * An array that stores the filter stride for each dimension of the input tensor
 * used in the convolution operation. For each dimension, the filter stride represents
 * the number of elements to slide over the input tensor. If \b dimNb is set to 4,
 * the stride is in height and width. If \b dimNb is set to 5,
 * the stride is in depth_stride, height and width.
 * The value of this parameter should be greater than or equal
 * to 1.
 * @param[in] dilation
 * An array that stores the dilation factor for each dimension of the filter tensor
 * used in the convolution operation. For each dimension, the dilation factor represents
 * the spacing between the kernel points. If \b dimNb is set to 4, the dilation should be set in
 * height and width dimension. The value of this parameter
 * should be greater than or equal to 1. If \b dimNb is set to 5, the dilation should be set in
 * depth, height and width dimension. The value of this parameter should be greater than or equal to 1.
 * @param[in] input_space
 * An array that stores the input size for each dimension of the input tensor used in sparse
 * convolution operation. If \b dimNb is set to 4, the input_space should be set in height and width
 * dimension. If \b dimNb is set to 5, the input_space should be set in depth, height and width dimension.
 * @param[in] filter_space
 * An array that stores the filter size for each dimension of the input tensor used in sparse
 * convolution operation. if \b dimNb is set to 4, the filter_space should be set in height and width
 * dimension, If \b dimNb is set to 5, the filter_space should be set in depth, height and width dimension.
 * @param[in] output_space
 * An array that stores the output size for each dimension of the input tensor used in sparse
 * convolution operation. If \b dimNb is set to 4, the output_space should be set in height and width
 * dimension. If \b dimNb is set to 5, the output_space should be set in depth, height and width dimension.
 * @param[in] sub_m
 * A value that determine the algorithms for sparse convolution. If \b sub_m is set to 0, the
 * algorithms will be the default sparse convolution. If \b sub_m is set to 0, the algorithms will be the
 * submanifold sparse convolution.
 * @param[in] transpose
 * A value that determines transpose.
 * @param[in] inverse
 * A value that determines inverse.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_EXECUTION_FAILED
 *   ::MLUOP_STATUS_NOT_INITIALIZED
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - Currently, only 5D input tensors are supported for convolution
 *   forward or backward operation.
 * - The data width of compute_type must not be less than output tensor's data type.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetSparseConvolutionDescriptor(mluOpSparseConvolutionDescriptor_t desc,
                                    int dimNb,
                                    int batch_size,
                                    const int pad[],
                                    const int stride[],
                                    const int dilation[],
                                    const int input_space[],
                                    const int filter_space[],
                                    const int output_space[],
                                    const int sub_m,
                                    const int transpose,
                                    const int inverse);

// Group: SparseConv
/*!
 * @brief Obtains the parameter num_act_out from ::mluOpSparseConvolutionDescriptor_t.
 *
 * @param[in] desc
 * Pointer to the parameter num_act_out that holds information about the tensor descriptor.
 * @param[out] num_act_out
 * The active point number of output space in sparse convolution mode.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_NOT_INITIALIZED
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetSparseConvolutionNumActOut(mluOpSparseConvolutionDescriptor_t desc, int *num_act_out);

// Group: Tensor
/*!
 * @brief Initializes the group of tensor descriptors stored by \b group_desc that was
 * previously created with ::mluOpCreateTensorDescriptor or
 * ::mluOpCreateGroupTensorDescriptors, and sets the information about
 * the dimensions, data type, and layout of all the input tensors.
 *
 * If ::mluOpSetTensorDescriptor or ::mluOpSetGroupTensorDescriptors is called, you do
 * not need to specify the strides of all dimensions. The strides are inferred by parameters
 * passed to this function. Also, the data will be treated as contiguous in memory with
 * no padding between dimensions. To specify the strides of all dimensions, you can call
 * ::mluOpSetTensorDescriptorEx. But the data might not be treated as contiguous in memory.
 *
 * @param[in] group_desc
 * An array of pointers to the structs that hold information about the tensor descriptor.
 * @param[in] group_layout
 * An array that stores the layouts of all input tensors. For detailed information, see
 * ::mluOpTensorLayout_t.
 * @param[in] group_dtype
 * An array that stores the data types of all input tensors. For detailed information,
 * see ::mluOpDataType_t.
 * @param[in] group_dimNb
 * An array that stores the dimensions of all input tensors.
 * @param[in] group_dimSize
 * An array that stores the size of each dimension of all tensors.
 * @param[in] desc_num
 * The length of the input array \b group_desc.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The group_dimSize includes dimensions of all tensors. You need to store
 *   the dimension of each tensor one by one in order. For example, If there are
 *   three tensors, the first tensor dimension is [3,4,5,6], the second tensor
 *   dimension is [9,7,8], and the third tensor dimension is [4,7], the
 *   group_dimSize should be [3,4,5,6,9,7,8,4,7].
 * - For better performance, there is no overflow check in this function.
 *   Make sure that the size of each tensor is in the range of [0, 2^31].
 *   Otherwise, you will get wrong result.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetGroupTensorDescriptors(mluOpTensorDescriptor_t *group_desc[],
                               const mluOpTensorLayout_t group_layout[],
                               const mluOpDataType_t group_dtype[],
                               const int group_dimNb[],
                               const int group_dimSize[],
                               const int desc_num);

// Group: Tensor
/*!
 * @brief Resets the tensor descriptor pointed by \b desc that was previously created with
 *  ::mluOpCreateTensorDescriptor. If ::mluOpResetTensorDescriptor is called,
 *  all the information about the tensor will be reset to initial value, which means layout
 *  is MLUOP_LAYOUT_ARRAY, dtype is MLUOP_DTYPE_FLOAT, dimsNb is 0, and dimSize points to an
 *  \b MLUOP_DIM_MAX-dimension array.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - This function is used to avoid memory leaks when more than one ::mluOpSetTensorDescriptor
 *   function is called. You should call this function before calling
 *   ::mluOpSetTensorDescriptor.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpResetTensorDescriptor(mluOpTensorDescriptor_t desc);

// Group: Tensor
/*!
 * @brief Initializes the tensor descriptor pointed by \b desc that was previously created
 * with ::mluOpCreateTensorDescriptor, and sets the information about the
 * dimensions, strides, data type, and layout of the input tensor.
 *
 * Compared with ::mluOpSetTensorDescriptor, you can specify the strides of all dimensions with
 * this function. If ::mluOpSetTensorDescriptor is called, you do not need to specify the
 * strides of all dimensions and the strides are inferred by parameters passed to this function.
 *
 * This function does not support all the operations in this version. You can check if an
 * operation supports this function in the "note" section of the operation description.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] layout
 * The layout of the input tensor. For detailed information, see ::mluOpTensorLayout_t.
 * @param[in] dtype
 * The data type of the input tensor. For detailed information, see ::mluOpDataType_t.
 * @param[in] dimNb
 * The number of dimensions in the input tensor of the initialized operation.
 * @param[in] dimSize
 * An array that contains the size of the tensor for each dimension.
 * @param[in] dimStride
 * An array that contains the stride of the tensor for each dimension.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - dimSize[0] represents the highest dimension, and dimSize[DIM_MAX - 1] represents
 *   the lowest dimension.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetTensorDescriptorEx(mluOpTensorDescriptor_t desc,
                           mluOpTensorLayout_t layout,
                           mluOpDataType_t dtype,
                           int dimNb,
                           const int dimSize[],
                           const int dimStride[]);

// Group: Tensor
/*!
 * @brief Initializes the tensor descriptor pointed by \b desc that was previously created
 * with ::mluOpCreateTensorDescriptor, and sets the information about the
 * dimensions, strides, data type, and layout of the input tensor.
 *
 * Compared with ::mluOpSetTensorDescriptor_v2, you can specify the strides of all dimensions with
 * this function. If ::mluOpSetTensorDescriptor_v2 is called, you do not need to specify the
 * strides of all dimensions and the strides are inferred by parameters passed to this function.
 *
 * This function does not support all the operations in this version. You can check if an
 * operation supports this function in the "note" section of the operation description.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] layout
 * The layout of the input tensor. For detailed information, see ::mluOpTensorLayout_t.
 * @param[in] dtype
 * The data type of the input tensor. For detailed information, see ::mluOpDataType_t.
 * @param[in] dimNb
 * The number of dimensions in the input tensor of the initialized operation.
 * @param[in] dimSize
 * An array that contains the size of the tensor for each dimension.
 * @param[in] dimStride
 * An array that contains the stride of the tensor for each dimension.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - dimSize[0] represents the highest dimension, and dimSize[DIM_MAX - 1] represents
 *   the lowest dimension.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetTensorDescriptorEx_v2(mluOpTensorDescriptor_t desc,
                              mluOpTensorLayout_t layout,
                              mluOpDataType_t dtype,
                              int dimNb,
                              const int64_t dimSize[],
                              const int64_t dimStride[]);

// Group: Tensor
/*!
 * @brief Sets the \b dimNb and \b dimSize factors to the input tensor descriptor.
 * If ::mluOpSetTensorDescriptorDim is called, you do not need to specify the strides of all
 * dimensions. The strides are inferred by parameters passed to this function. Also, the data
 * will be treated as contiguous in memory with no padding between dimensions. To specify the
 * strides of all dimensions, you can call ::mluOpSetTensorDescriptorEx. But the data might not
 * be treated as contiguous in memory.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] dimNb
 * The number of dimensions in the input tensor of the initialized operation.
 * @param[in] dimSize
 * An array that contains the size of the tensor for each dimension.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - dimSize[0] represents the highest dimension, dimSize[DIM_MAX - 1] represents
 *   the lowest dimension, and DIM_MAX represents the number of dimensions in the input tensor.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t
mluOpSetTensorDescriptorDim(mluOpTensorDescriptor_t desc, int dimNb, const int *dimSize);

// Group: Tensor
/*!
 * @brief Sets the \b dimNb and \b dimSize factors to the input tensor descriptor.
 * If ::mluOpSetTensorDescriptorDim_v2 is called, you do not need to specify the strides of all
 * dimensions. The strides are inferred by parameters passed to this function. Also, the data
 * will be treated as contiguous in memory with no padding between dimensions. To specify the
 * strides of all dimensions, you can call ::mluOpSetTensorDescriptorEx_v2. But the data might not
 * be treated as contiguous in memory.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] dimNb
 * The number of dimensions in the input tensor of the initialized operation.
 * @param[in] dimSize
 * An array that contains the size of the tensor for each dimension.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - dimSize[0] represents the highest dimension, dimSize[DIM_MAX - 1] represents
 *   the lowest dimension, and DIM_MAX represents the number of dimensions in the input tensor.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t
mluOpSetTensorDescriptorDim_v2(mluOpTensorDescriptor_t desc, int dimNb, const int64_t *dimSize);

// Group: Tensor
/*!
 * @brief Sets the on-chip data type to the descriptor of a tensor \b desc. The on-chip
 * data type \b onchip_dtype can be different from the off-chip data type of the tensor.
 * This function is optional. If the on-chip data type is not set with this function, the
 * ::MLUOP_STATUS_BAD_PARAM data type is used by default.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] onchip_dtype
 * The on-chip data type of the tensor is used in the function that supports fixed-point
 * computing.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The on-chip data type is only used on the operations that support fixed-point computing. It
 *   has no effect on other functions. If you call this function to get on-chip data type for a
 *   function that does not support fixed-point computing, ::MLUOP_STATUS_BAD_PARAM is returned.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetTensorDescriptorOnchipDataType(mluOpTensorDescriptor_t desc, mluOpDataType_t onchip_dtype);

// Group: Tensor
/*!
 * @brief Sets the \b position factor to the descriptor \b desc of fixed-point data in
 * fixed-point quantization. It is used in ::MLUOP_QUANTIZE_POSITION mode.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] position
 * A scalar of fixed position factor that is used for quantization.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetTensorDescriptorPosition(mluOpTensorDescriptor_t desc, int position);

// Group: Tensor
/*!
 * @brief Sets the \b position and \b scale factors to the descriptor of fixed-point
 * data in fixed-point quantization. It is used in ::MLUOP_QUANTIZE_POSITION_SCALE mode.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] position
 * A scalar of fixed position factor that is used for quantization.
 * @param[in] scale
 * A scalar of scale factor that is used for quantization.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetTensorDescriptorPositionAndScale(mluOpTensorDescriptor_t desc, int position, float scale);

// Group: Tensor
/*!
 * @brief Sets the \b position , \b scale , and \b offset factors to the descriptor of fixed-point
 * data in fixed-point quantization. It is used in ::MLUOP_QUANTIZE_POSITION_SCALE_OFFSET mode.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] position
 * A scalar of fixed position factor that is used for quantization.
 * @param[in] scale
 * A scalar of scale factor that is used for quantization.
 * @param[in] offset
 * A scalar of offset factor that is used for quantization.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetTensorDescriptorPositionScaleAndOffset(mluOpTensorDescriptor_t desc, int position, float scale, int offset);

// Group: Tensor
/*!
 * @brief Retrieves a tensor descriptor \b desc that was previously created with
 * ::mluOpCreateTensorDescriptor, and sets the information about the dimensions,
 * data type, and layout of input tensor.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] layout
 * Pointer to the host memory that holds information about the layout of the input tensor.
 * For detailed information, see ::mluOpTensorLayout_t.
 * @param[out] dtype
 * Pointer to the host memory that holds information about the data type of the input tensor.
 * For detailed information, see ::mluOpDataType_t.
 * @param[out] dimNb
 * Pointer to the host memory that holds information about the dimension of input tensor.
 * @param[out] dimSize
 * An array that contains the size of the tensor for each dimension.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - dimSize[0] represents the highest dimension, and dimSize[DIM_MAX - 1] represents the lowest
 *   dimension.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorDescriptor(
    const mluOpTensorDescriptor_t desc, mluOpTensorLayout_t *layout, mluOpDataType_t *dtype, int *dimNb, int dimSize[]);

// Group: Tensor
/*!
 * @brief Retrieves a tensor descriptor \b desc that was previously created with
 * ::mluOpCreateTensorDescriptor, and sets the information about the dimensions,
 * data type, and layout of input tensor.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] layout
 * Pointer to the host memory that holds information about the layout of the input tensor.
 * For detailed information, see ::mluOpTensorLayout_t.
 * @param[out] dtype
 * Pointer to the host memory that holds information about the data type of the input tensor.
 * For detailed information, see ::mluOpDataType_t.
 * @param[out] dimNb
 * Pointer to the host memory that holds information about the dimension of input tensor.
 * @param[out] dimSize
 * An array that contains the size of the tensor for each dimension.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - dimSize[0] represents the highest dimension, and dimSize[DIM_MAX - 1] represents the lowest
 *   dimension.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorDescriptor_v2(const mluOpTensorDescriptor_t desc,
                            mluOpTensorLayout_t *layout,
                            mluOpDataType_t *dtype,
                            int *dimNb,
                            int64_t dimSize[]);

// Group: Tensor
/*!
 * @brief Retrieves a tensor descriptor \b desc that was previously created with the
 * ::mluOpCreateTensorDescriptor and sets the information about the dimensions, data type,
 * stride and layout of input tensor with ::mluOpSetTensorDescriptorEx.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] layout
 * Pointer to the host memory that holds information about the layout of the input tensor.
 * For detailed information, see ::mluOpTensorLayout_t.
 * @param[out] dtype
 * Pointer to the host memory that holds information about the data type of the input tensor.
 * For detailed information, see ::mluOpDataType_t.
 * @param[out] dimNb
 * Pointer to the host memory that holds information about the dimension of input tensor.
 * @param[out] dimSize
 * An array that contains the size of the tensor for each dimension.
 * @param[out] dimStride
 * An array that contains the stride of the tensor for each dimension.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - dimSize[0] represents the highest dimension, and dimSize[DIM_MAX - 1] represents the lowest
 *   dimension.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorDescriptorEx(const mluOpTensorDescriptor_t desc,
                           mluOpTensorLayout_t *layout,
                           mluOpDataType_t *dtype,
                           int *dimNb,
                           int dimSize[],
                           int dimStride[]);

// Group: Tensor
/*!
 * @brief Retrieves a tensor descriptor \b desc that was previously created with the
 * ::mluOpCreateTensorDescriptor and sets the information about the dimensions, data type,
 * stride and layout of input tensor with ::mluOpSetTensorDescriptorEx_v2.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] layout
 * Pointer to the host memory that holds information about the layout of the input tensor.
 * For detailed information, see ::mluOpTensorLayout_t.
 * @param[out] dtype
 * Pointer to the host memory that holds information about the data type of the input tensor.
 * For detailed information, see ::mluOpDataType_t.
 * @param[out] dimNb
 * Pointer to the host memory that holds information about the dimension of input tensor.
 * @param[out] dimSize
 * An array that contains the size of the tensor for each dimension.
 * @param[out] dimStride
 * An array that contains the stride of the tensor for each dimension.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - dimSize[0] represents the highest dimension, and dimSize[DIM_MAX - 1] represents the lowest
 *   dimension.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorDescriptorEx_v2(const mluOpTensorDescriptor_t desc,
                              mluOpTensorLayout_t *layout,
                              mluOpDataType_t *dtype,
                              int *dimNb,
                              int64_t dimSize[],
                              int64_t dimStride[]);

// Group: Tensor
/*!
 * @brief Retrieves the number of elements according to the input descriptor \b desc. You
 * need to call ::mluOpSetTensorDescriptor first to create a tensor descriptor
 * before calling this function.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
     @verbatim
      mluOpTensorDescriptor_t input_desc;
      mluOpCreateTensorDescriptor(&input_desc);
      mluOpSetTensorDescriptor(input_desc, MLUOP_LAYOUT_ARRAY,MLUOP_DTYPE_FLOAT, 2,{2, 3});
      size_t nums=mluOpGetTensorElementNum(input_desc);  // nums = 6
      input one array by 2 * 3
      input: [[1,2,3],[4,5,6]]
      output: 6
     @endverbatim
 *
 * @par Reference
 * - None.
 */
size_t MLUOP_WIN_API
mluOpGetTensorElementNum(const mluOpTensorDescriptor_t desc);

// Group: Tensor
/*!
 * @brief Retrieves the on-chip data type of a tensor descriptor \b desc set by
 * ::mluOpSetTensorDescriptorOnchipDataType. If the on-chip data type is not set
 * with ::mluOpSetTensorDescriptorOnchipDataType, the
 * ::MLUOP_STATUS_BAD_PARAM is returned.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] onchip_dtype
 * Pointer to the MLU memory that holds information about the on-chip data type of the tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The on-chip data type is only used on the operations that support fixed-point computing. It
 *   has no effect on other operations. If you call this function to get on-chip data type for an
 *   operation that does support fixed-point computing, ::MLUOP_STATUS_BAD_PARAM is returned.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorDescriptorOnchipDataType(const mluOpTensorDescriptor_t desc, mluOpDataType_t *onchip_dtype);

// Group: Tensor
/*!
 * @brief Gets the \b position factor to the descriptor \b desc of fixed-point data in
 * fixed-point quantization.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] position
 * A host pointer of fixed position factor that is used for quantization.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorDescriptorPosition(const mluOpTensorDescriptor_t desc, int *position);

// Group: Tensor
/*!
 * @brief Gets the position and scale factors of a tensor descriptor \b desc used in
 * fixed-point quantization.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] position
 * Pointer to the MLU memory that holds information about fixed position used for quantization.
 * @param[out] scale
 * Pointer to the MLU memory that holds information about scale factor used for quantization.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorDescriptorPositionAndScale(const mluOpTensorDescriptor_t desc, int *position, float *scale);

// Group: Tensor
/*!
 * @brief Gets the \b position, \b scale and \b offset factors to the descriptor \b desc of
 * fixed-point data in fixed-point quantization.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] position
 * A host pointer of fixed position factor that is used for quantization.
 * @param[out] scale
 * A host pointer of scale factor that is used for quantization.
 * @param[in] offset
 * A host pointer of offset factor that is used for quantization.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorDescriptorPositionScaleAndOffset(const mluOpTensorDescriptor_t desc,
                                               int *position,
                                               float *scale,
                                               int *offset);

// Group: Tensor
/*!
 * @brief Destroys a tensor descriptor that was created by ::mluOpCreateTensorDescriptor.
 *
 * @param[in] desc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDestroyTensorDescriptor(mluOpTensorDescriptor_t desc);

// Group: Tensor
/*!
 * @brief Destroys a group of tensor descriptors that were created by
 * ::mluOpCreateTensorDescriptor or ::mluOpCreateGroupTensorDescriptors.
 *
 * @param[in] group_desc
 * An array of pointers to the struct that holds information about the
 * tensor descriptor.
 * @param[in] desc_num
 * The length of the input array \b group_desc.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDestroyGroupTensorDescriptors(mluOpTensorDescriptor_t *group_desc[], const int desc_num);

// Group: TensorSet
/*!
 * @brief Creates a descriptor \b tensorSetDesc of tensor set that holds a series of tensors.
 * The number of tensors of tensor set is jointly determined by \b setDimNb and \b setDimSize.
 * Use ::mluOpInitTensorSetMemberDescriptor to set information for descriptor.
 *
 * @param[out] tensorSetDesc
 * Pointer to the memory that holds information about the descriptor of tensor set.
 * @param[in] setDimNb
 * The number of dimensions of the tensor set.
 * @param[in] setDimSize
 * An array that contains the number of the tensors for each dimension of the tensor set.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - After calling this function, you can call ::mluOpInitTensorSetMemberDescriptor
 *   to initialize and set the information to the tensor set descriptor.
 * - You need to call ::mluOpDestroyTensorSetDescriptor to destroy the
 *   descriptor.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateTensorSetDescriptor(mluOpTensorSetDescriptor_t *tensorSetDesc, const int setDimNb, const int setDimSize[]);

// Group: TensorSet
/*!
 * @brief Retrieves a tensor set descriptor \b tensorSetDesc that was previously created
 * with ::mluOpCreateTensorSetDescriptor.
 *
 * @param[in] tensorSetDesc
 * The descriptor of the tensor \b tensorSetDesc.
 * @param[out] setDimNb
 * The number of dimensions of the tensor set.
 * @param[out] setDimSize
 * An array that contains the number of the tensor for each dimension of the tensor set.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - Before calling this function, ::mluOpCreateTensorSetDescriptor should be called.
 *
 * @par Note
 * - setDimSize[0] represents the highest dimension, and dimSize[dimNb - 1] represents
 *   the lowest dimension.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorSetDescriptor(mluOpTensorSetDescriptor_t tensorSetDesc, int *setDimNb, int setDimSize[]);

// Group: TensorSet
/*!
 * @brief Destroys a tensor set descriptor \b tensorSetDesc that was previously created by
 * ::mluOpCreateTensorSetDescriptor.
 *
 * @param[in] tensorSetDesc
 * The descriptor of the tensor desc. For detailed information, see ::mluOpTensorDescriptor_t.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - This function should be called to destroy the tensor set descriptor.
 *   Otherwise, the memory leak may occur.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDestroyTensorSetDescriptor(mluOpTensorSetDescriptor_t tensorSetDesc);

// Group: TensorSet
/*!
 * @brief Initializes a member tensor in the tensor set descriptors pointed by
 * \b desc that was previously created with ::mluOpCreateTensorSetDescriptor,
 * and sets the information about the dimensions, data type, and
 * layout.
 *
 * @param[in] tensorSetDesc
 * The descriptor of the tensor \b tensorSetDesc. For detailed information,
 * see ::mluOpTensorSetDescriptor_t.
 * @param[in] setDimNb
 * The number of dimensions of the tensor set.
 * @param[in] tensorIndex
 * An array that contains the index of each dimension of a member tensor to be
 * initialized in the tensor set.
 * @param[in] layout
 * The layout of the member tensor. For detailed information, see ::mluOpTensorLayout_t.
 * @param[in] dtype
 * The data type of the member tensor. For detailed information, see ::mluOpDataType_t.
 * @param[in] dimNb
 * The number of dimensions in the member tensor.
 * @param[in] dimSize
 * An array that contains the size of the member tensor for each dimension.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - Before calling this function,
 *   You need to call ::mluOpCreateTensorSetDescriptor to create
 *   the tensor descriptor \b tensorSetDesc.
 * - All member tensors in the tensor set need to call this function to
 *   initialize related properties.
 * - dimSize[0] and dimSize[DIM_MAX - 1] represent the highest and lowest
 *   dimension respectively, where DIM_MAX is the number of dimensions in the
 *   input tensor.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpInitTensorSetMemberDescriptor(mluOpTensorSetDescriptor_t tensorSetDesc,
                                   const int setDimNb,
                                   const int tensorIndex[],
                                   mluOpTensorLayout_t layout,
                                   mluOpDataType_t dtype,
                                   const int dimNb,
                                   const int dimSize[]);

// Group: TensorSet
/*!
 * @brief Sets the position and scale factors used in fixed-point quantization.
 * It is only used if you have quantized the input data with the symmetric
 * fixed-point quantization with scale factor quantization method.
 *
 * @param[in] tensorSetDesc
 * The descriptor of the tensor \b tensorSetDesc. For detailed information,
 * see ::mluOpTensorSetDescriptor_t.
 * @param[in] setDimNb
 * The number of dimensions of the tensor set.
 * @param[in] tensorIndex
 * An array that contains the position index information of the member
 * tensor in the tensor set.
 * @param[in] position
 * A position of fixed position factor that is used for quantification.
 * @param[in] scale
 * A scalar of scale factor that is used for quantification.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - If the member tensor is in floating-point data type, and you need to call
 *   this function.
 * - If the member tensor is in fixed-point data type, and you need to call
 *   this function.
 * - Before calling this function,
 *   You need to call ::mluOpCreateTensorSetDescriptor to create
 *   the tensor descriptors \b tensorSetDesc.
 * - The \b position should be limited in [-128, 127], otherwise the result is
 *   undefined.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpInitTensorSetMemberDescriptorPositionAndScale(mluOpTensorSetDescriptor_t tensorSetDesc,
                                                   const int setDimNb,
                                                   const int tensorIndex[],
                                                   const int position,
                                                   const float scale);

// Group: TensorSet
/*!
 * @brief Retrieves the size of tensor set according to the input descriptor \b
 * tensorSetDesc. You need to call ::mluOpInitTensorSetMemberDescriptor
 * first to create a tensor set descriptor before calling this
 * function.
 *
 * @param[in] tensorSetDesc
 * The descriptor of the tensor desc. For detailed information,
 * see ::mluOpTensorSetDescriptor_t.
 * @param[out] sizeInBytes
 * The size in bytes of tensor set. You can allocate MLU memory for the
 * tensor set with this value.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorSetDescriptorSize(mluOpTensorSetDescriptor_t tensorSetDesc, int *sizeInBytes);

// Group: TensorSet
/*!
 * @brief Retrieves the tensor descriptor in the tensor set and the corresponding offset
 * address based on the entire block of MLU memory through the index \b tensorIndex.
 *
 * @param[in] tensorSetDesc
 * The descriptor of the tensor \b tensorSetDesc. For detailed information, see ::mluOpTensorSetDescriptor_t.
 * @param[in] setDimNb
 * The number of dimensions of the tensor set.
 * @param[in] tensorIndex
 * An array that contains the position information of the member tensor in the tensor set.
 * @param[in] data
 * Pointer to the MLU memory that is described by \b tensorSetDesc.
 * @param[out] tensorDesc
 * Pointer to the host member. It is member tensor descriptor that is indexed by \b tensorIndex
 * in the tensor set. \b *tensorDesc contains tensor member information about dimensions,
 * layout, data type, position and scale.
 * @param[out] dataAddrInDevice
 * Pointer to the MLU memory that is indexed by \b tensorIndex in the whole block of data
 * \b dataAddrInDevice.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorAndDataFromTensorSet(mluOpTensorSetDescriptor_t tensorSetDesc,
                                   const int setDimNb,
                                   const int tensorIndex[],
                                   void *data,
                                   mluOpTensorDescriptor_t *tensorDesc,
                                   void **dataAddrInDevice);

// Group: Abs
/*!
 * @brief Computes the absolute value for every element of the input tensor \b x
 * and returns results in \b y.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * abs operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] x_desc
 * The descriptor of the tensor \b x. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] y_desc
 * The descriptor of the tensor \b y. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] y
 * Pointer to the MLU memory that stores the output tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The date types of input tensor and output tensor should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float, bfloat16, int32, complex_float
 *   - output tensor: half, float, bfloat16, int32
 * - The data type bfloat16 is only supported on MLU500 series.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
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
mluOpStatus_t MLUOP_WIN_API
mluOpAbs(mluOpHandle_t handle,
         const mluOpTensorDescriptor_t x_desc,
         const void *x,
         const mluOpTensorDescriptor_t y_desc,
         void *y);

// Group: Log
/*!
 * @brief Computes logarithm of input tensor \b x, and returns the results in
 * the output tensor \b y.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in the log operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] prefer
 * The \b prefer modes defined in ::mluOpComputationPreference_t.
 * @param[in] base
 * An mluOpLogBase_t type value indicating the base (e, 2 or 10) to be used.
 * @param[in] x_desc
 * The descriptor of the tensor \b x. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor \b x.
 * @param[in] y_desc
 * The descriptor of the tensor \b y. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] y
 * Pointer to the MLU memory that stores the output tensor \b y.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - The data types of input tensor and output tensor should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The input tensor and output tensor have the same shape, and the input
 *   tensor must meet the following input data ranges:
 *   - float: [1e-20, 2e5]
 *   - half: [1, 60000]
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://www.tensorflow.org/api_docs/python/tf/math/log
 */
mluOpStatus_t MLUOP_WIN_API
mluOpLog(mluOpHandle_t handle,
         const mluOpComputationPreference_t prefer,
         const mluOpLogBase_t base,
         const mluOpTensorDescriptor_t x_desc,
         const void *x,
         const mluOpTensorDescriptor_t y_desc,
         void *y);

// Group: Log
/*!
 * @brief Returns a one-dimensional tensor of \b steps points logarithmically
 * spaced with base \b base between \b base^start and \b base^end.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in the log operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] start
 * The starting value for the set of points.
 * @param[in] end
 * The ending value for the set of points.
 * @param[in] steps
 * Number of points to sample between \b start and \b end.
 * @param[in] base
 * Base of the logarithm function.
 * @param[in] res_desc
 * The descriptor of the tensor \b res. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] res
 * Pointer to the MLU memory that stores the output tensor \b res.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - The supported data types of output tensor are as follows:
 *   - output tensor: half, float, int32
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - \b base cannot be NAN or infinity.
 * - \b steps should be greater than or equal to 0.
 * - \b steps should be less than or equal to the length of the output tensor \b res.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/pytorch/pytorch/blob/v2.1.0/aten/src/ATen/native/cuda/RangeFactories.cu#L123
 */
mluOpStatus_t MLUOP_WIN_API
mluOpLogspace(mluOpHandle_t handle,
              const float start,
              const float end,
              const int64_t steps,
              const float base,
              const mluOpTensorDescriptor_t res_desc,
              void *res);

// Group: Carafe
/*!
 * @brief Creates a descriptor pointed by \b carafe_desc for CARAFE upsampling forward and backward operations,
 * and allocates memory holding the configuration parameters. The information is defined in ::mluOpCarafeDescriptor_t.
 * For more information about descriptor, see "Cambricon MLU-OPS User Guide".
 *
 * @param[in] carafe_desc
 * A host pointer to the CARAFE descriptor that holds information about the CARAFE operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_NOT_INITIALIZED
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - After calling this function, you can call ::mluOpSetCarafeDescriptor to initialize
 *   and set the information to the CARAFE descriptor.
 * - You need to call ::mluOpDestroyCarafeDescriptor to destroy the descriptor.
 *
 * @par Note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateCarafeDescriptor(mluOpCarafeDescriptor_t *carafe_desc);

// Group: Carafe
/*!
 * @brief Initializes the CARAFE descriptor \b carafe_desc that was previously created with
 * ::mluOpCreateCarafeDescriptor, and sets the information about the
 * CARAFE forward and backward operations to the descriptor \b carafe_desc.
 *
 * @param[in] carafe_desc
 * The descriptor of the tensor \b carafe. For detailed information,
 * see ::mluOpCarafeDescriptor_t.
 * @param[in] dimNb
 * The number of dimensions in the input tensor of the CARAFE operation.
 * @param[in] kernel_size
 * The width of the upsampling kernel window.
 * @param[in] group_size
 * The number of channel groups. The channels in the same group share the same upsampling filter.
 * @param[in] scale_factor
 * The upsampling ratio by which the resolution of the input feature map will be
 * increased, i.e., the height and width of the output feature maps would be \b scale_factor times
 * of the height and width of the input feature maps, respectively.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - Before calling this function, ::mluOpCreateCarafeDescriptor should be called.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetCarafeDescriptor(mluOpCarafeDescriptor_t carafe_desc,
                         const int dimNb,
                         const int kernel_size,
                         const int group_size,
                         const int scale_factor);

// Group: Carafe
/*!
 * @brief Destroys a CARAFE descriptor \b carafe_desc that was previously created by
 * ::mluOpCreateCarafeDescriptor.
 *
 * The CARAFE descriptor is defined in ::mluOpCarafeDescriptor_t
 * and holds the information about the CARAFE forward or backward operations.
 *
 * @param[in] carafe_desc
 * The CARAFE descriptor to be destroyed. For detailed information,
 * see ::mluOpCarafeDescriptor_t.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - You need to call this function after calling ::mluOpCarafeForward,
 *   or ::mluOpCarafeBackward. Otherwise, \p MLUOP_STATUS_BAD_PARAM is returned.
 * - This function should be called to destroy the CARAFE descriptor. Otherwise, memory
 *   leak may occur.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDestroyCarafeDescriptor(mluOpCarafeDescriptor_t carafe_desc);

// Group: Carafe
/*!
 * @brief Performs the CARAFE upsampling operation on the input feature maps \b input using
 * weighted combination, where the filter is set with \b mask. The upsampled feature maps
 * are returned in the output tensor \b output.
 *
 * CARAFE performs upsampling at each output location by weighted summation in a nearby mask
 * window around the corresponding input location. The mask filters are defined separately
 * for each output location, which offers the ability of content-aware handling.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the carafe
 * forward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] carafe_desc
 * The descriptor of the tensor \b carafe. For detailed information,
 * see ::mluOpCarafeDescriptor_t.
 * @param[in] input_desc
 * The tensor descriptor of the input feature maps. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] mask_desc
 * The tensor descriptor of the mask applied to the input feature maps.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mask
 * Pointer to the MLU memory that stores the mask tensor.
 * @param[in] output_desc
 * The tensor descriptor of the output upsampled feature maps.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of \b input, \b mask and \b output tensors must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - mask tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - The data layouts of the \b input tensor, \b mask tensor, and \b output tensor should be \p MLUOP_LAYOUT_NHWC.
 *
 * @par Scale Limitation
 * - Parameters specified in \b carafe_desc should satisfy:
 * - The \b dimNb should be equal to 4.
 * - The \b kernel_size should be an odd number, i.e., 2*n+1 (n>=0), and \b kernel_size <= 45.
 * - The \b group_size is positive integers,
 * - The \b scale_factor is positive integers, and \b scale_factor <= 5.
 * - The shape of \b input_desc should be [N, H, W, C].
 * - The shape of \b mask_desc should be [N, Ho, Wo, Cm].
 * - The shape of \b output_desc should be [N, Ho, Wo, C].
 * - The length of all dimensions should be non-negative integers.
 * - The dimensions denoted by the same symbol should have the same value.
 * - The \b C should be divisible by \b group_size, i.e., C = N * group_size (N>=1).
 * - The formula of \b Cm is that \b Cm = \b group_size * \b kernel_size * \b kernel_size.
 * - The formula of \b Ho is that \b Ho = \b scale_factor * \b H.
 * - The formula of \b Wo is that \b Wo = \b scale_factor * \b W.
 *
 * @par API Dependency
 * - Before calling this function to implement CARAFE forward operation, you need to
 *   prepare all the parameters passed to this function. See each parameter description
 *   for details.
 *
 * @par Note
 * - If any dimension in \b input_desc, \b mask_desc, or \b output_desc is zero,
 *   which represents an empty tensor, ::MLUOP_STATUS_SUCCESS is returned without
 *   any changes to the \b output tensor.
 *
 * @par Example
 * - Example of CARAFE forward operation is as follows:
     @verbatim
      input tensor by 1 * 2 * 2 * 1 --> input:
        [[[[ 0.34064351], [-0.8246629 ]],
          [[-0.71797801], [-0.51707748]]]]

      mask tensor by 1 * 4 * 4 * 1 --> mask:
        [[[[ 0.97630979], [-0.06261992], [ 0.91232837], [-0.1598553 ]],
          [[-0.72060206], [ 0.48904262], [-0.65568251], [ 0.12801235]],
          [[-0.85134485], [-1.27589059], [ 3.00143314], [ 0.61258706]],
          [[-0.50308504], [-0.93015218], [-1.1125597 ], [ 0.67302385]]]]

      param:
        kernel_size: 3, group_size: 1, scale_factor: 2

      output tensor by 1 * 4 * 4 * 1 --> output:
        [[[[ 0.33257359], [-0.02133107], [-0.75236336], [ 0.13182674]],
          [[-0.24546842], [ 0.1665892 ], [ 0.54071704], [-0.10556703]],
          [[ 0.61124688], [ 0.91606138], [-1.55197348], [-0.31675497]],
          [[ 0.36120399], [ 0.66782881], [ 0.57527956], [-0.34800548]]]]
     @endverbatim
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/tree/master/mmcv/ops/carafe.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCarafeForward(mluOpHandle_t handle,
                   const mluOpCarafeDescriptor_t carafe_desc,
                   const mluOpTensorDescriptor_t input_desc,
                   const void *input,
                   const mluOpTensorDescriptor_t mask_desc,
                   const void *mask,
                   const mluOpTensorDescriptor_t output_desc,
                   void *output);

// Group: Carafe
/*!
 * @brief Performs the back-propagation of CARAFE.
 * operation to compute the gradient with respect to input \b grad_input and
 * mask \b grad_mask based on the gradient of response \b grad_output.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in the CARAFE backward operation. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in] carafe_desc
 * The descriptor of the tensor \b carafe. For detailed information,
 * see ::mluOpCarafeDescriptor_t.
 * @param[in] input_desc
 * The tensor descriptor of the input feature maps. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] mask_desc
 * The tensor descriptor of the mask applied to the input feature maps.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mask
 * Pointer to the MLU memory that stores the mask tensor.
 * @param[in] grad_output_desc
 * The tensor descriptor of the gradient with respect to the output feature maps.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] grad_output
 * Pointer to the MLU memory that stores the gradient with respect to the
 * upsampled feature maps.
 * @param[in] grad_input_desc
 * The tensor descriptor of the gradient with respect to the input feature maps.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] grad_input
 * Pointer to the MLU memory that stores the gradient with respect to the
 * input feature maps.
 * @param[in] grad_mask_desc
 * The descriptor of the gradient tensor with respect to the \b mask tensor.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] grad_mask
 * Pointer to the MLU memory that stores the gradient with respect to \b mask.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of \b input tensor, \b mask tensor, \b grad_output tensor, \b grad_input tensor, and \b grad_mask
 *   tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - mask tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - The data layouts of the \b input tensor, \b mask tensor, \b grad_output tensor, \b grad_input tensor, and \b
 * grad_mask tensor should be \p MLUOP_LAYOUT_NHWC.
 *
 * @par Scale Limitation
 * - Parameters specified in \b carafe_desc should satisfy:
 * - The \b dimNb = 4.
 * - The \b kernel_size should be an odd number, i.e., 2*n+1 (n>=0), and \b kernel_size <= 137.
 * - The \b group_size and \b scale_factor should be positive integers.
 * - The shape of \b input_desc should be [N, H, W, C].
 * - The shape of \b mask_desc should be [N, Ho, Wo, Cm].
 * - The shape of \b grad_output_desc should be [N, Ho, Wo, C].
 * - The shape of \b grad_input_desc should be [N, H, W, C].
 * - The shape of \b grad_mask_desc should be [N, Ho, Wo, Cm].
 * - The length of all dimensions should be non-negative integers.
 * - The dimensions denoted by the same symbol should have the same value.
 * - \b C should be divisible by \b group_size, i.e., C = n * group_size (n>=1).
 * - \b Cm = \b group_size * \b kernel_size * \b kernel_size.
 * - \b Ho = \b scale_factor * \b H.
 * - \b Wo = \b scale_factor * \b W.
 *
 * @par API Dependency
 * - Before calling this function to implement CARAFE backward operation, you need to
 *   prepare all the parameters passed to this function. See each parameter description
 *   for details.
 *
 * @par Note
 * - If any dimension in \b input_desc, \b mask_desc, \b grad_output_desc, \b grad_input_desc
 *   or \b grad_mask_desc is zero, which represents an empty tensor, ::MLUOP_STATUS_SUCCESS is
 *   returned without any changes to the \b grad_output and \b grad_input tensors.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/tree/master/mmcv/ops/carafe.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCarafeBackward(mluOpHandle_t handle,
                    const mluOpCarafeDescriptor_t carafe_desc,
                    const mluOpTensorDescriptor_t input_desc,
                    const void *input,
                    const mluOpTensorDescriptor_t mask_desc,
                    const void *mask,
                    const mluOpTensorDescriptor_t grad_output_desc,
                    const void *grad_output,
                    const mluOpTensorDescriptor_t grad_input_desc,
                    void *grad_input,
                    const mluOpTensorDescriptor_t grad_mask_desc,
                    void *grad_mask);

// Group: Div
/*!
 * @brief Computes division on input tensors \b x and \b y, and returns the
 * results in the output tensor \b output.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in the division operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] prefer
 * The \b prefer modes defined in ::mluOpComputationPreference_t.
 * @param[in] x_desc
 * The descriptor of the tensor \b x. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the dividend tensor.
 * @param[in] y_desc
 * The descriptor of the tensor \b y. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] y
 * Pointer to the MLU memory that stores the divisor tensor.
 * @param[in] z_desc
 * The descriptor of the tensor \b z. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] z
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - The data type of input tensors and output tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The input tensors and output tensor must have the same shape.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The input tensors and output tensor have the same shape, and the input
 *   tensor \b y must meet the following input data range:
 *   - float: [-1e10,-1e-20] & [1e-20,1e10]
 *   - half: [-65504,-1e-4] & [1e-4,65504]
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://www.tensorflow.org/api_docs/python/tf/math/divide
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDiv(mluOpHandle_t handle,
         const mluOpComputationPreference_t prefer,
         const mluOpTensorDescriptor_t x_desc,
         const void *x,
         const mluOpTensorDescriptor_t y_desc,
         const void *y,
         const mluOpTensorDescriptor_t z_desc,
         void *z);

// Group: DynamicPointToVoxel
/*!
 * @brief Gets extra space size for the DynamicPointToVoxelBackward operation.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context for MLU devices and queues management in the
 * DynamicPointToVoxelBackward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] reduce_type
 * Reduce op. Only the default 'max' is supported.
 * @param[in] grad_voxel_feats_desc
 * Pointer to the MLU memory that stores the gradient for \b voxel_feats.
 * @param[in] feats_desc
 * The descriptor of the tensor \b feats. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] voxel_feats_desc
 * The descriptor of the tensor \b voxel_feats. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] point2voxel_map_desc
 * The descriptor of the tensor \b point2voxel_map. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] voxel_points_count_desc
 * The descriptor of the tensor \b voxel_points_count. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] voxel_num_desc
 * The descriptor of the tensor \b voxel_num. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] workspace_size
 * A host pointer to the returned size of extra space in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetDynamicPointToVoxelBackwardWorkspaceSize(const mluOpHandle_t handle,
                                                 const mluOpReduceMode_t reduce_type,
                                                 const mluOpTensorDescriptor_t grad_voxel_feats_desc,
                                                 const mluOpTensorDescriptor_t feats_desc,
                                                 const mluOpTensorDescriptor_t voxel_feats_desc,
                                                 const mluOpTensorDescriptor_t point2voxel_map_desc,
                                                 const mluOpTensorDescriptor_t voxel_points_count_desc,
                                                 const mluOpTensorDescriptor_t voxel_num_desc,
                                                 size_t *workspace_size);

// Group: DynamicPointToVoxel
/*!
 * @brief Performs the back-propagation of DynamicPointToVoxelForward
 * operation to compute the gradient for input \b grad_voxel_feats
 * based on the gradient of response \b grad_feats.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context for MLU devices and queues management in the
 * DynamicPointToVoxelBackward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] reduce_type
 * Reduce op. Only the default 'max' is supported.
 * @param[in] grad_voxel_feats_desc
 * The descriptor of the tensor \b grad_voxel_feats. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] grad_voxel_feats
 * Pointer to the MLU memory that stores the gradient for \b voxel_feats.
 * @param[in] feats_desc
 * The descriptor of the tensor \b feats. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] feats
 * Pointer to the MLU memory that stores points features to be reduced into voxels.
 * @param[in] voxel_feats_desc
 * The descriptor of the tensor \b voxel_feats. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] voxel_feats
 * Pointer to the MLU memory that stores the voxel features.
 * @param[in] point2voxel_map_desc
 * The descriptor of the tensor \b point2voxel_map. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] point2voxel_map
 * Pointer to the MLU memory that stores the index to voxel.
 * @param[in] voxel_points_count_desc
 * The descriptor of the tensor \b voxel_points_count. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] voxel_points_count
 * Pointer to the MLU memory that stores the voxel points count.
 * @param[in] voxel_num_desc
 * The descriptor of the tensor \b voxel_num. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] voxel_num
 * Pointer to the MLU memory that stores the voxel coordinates num.
 * @param[in] workspace
 * Pointer to the MLU memory that stores the extra workspace.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in
 * ::mluOpDynamicPointToVoxelBackward.
 * @param[out] grad_feats_desc
 * The descriptor of the tensor \b grad_feats. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] grad_feats
 * Pointer to the MLU memory that stores the gradient with respect to \b feats.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS,        ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_ARCH_MISMATCH,  ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_INTERNAL_ERROR, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - grad_voxel_feats, feats, voxel_feats: float
 *   - point2voxel_map, voxel_points_count, voxel_num: int
 *   - reduce_type: mluOpReduceMode_t::MLUOP_REDUCE_DMAX
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The \b grad_voxel_feats tensor, \b feats tensor, \b voxel_feats tensor and \b grad_feats tensor
 *   must have two dimensions.
 * - The \b point2voxel_map tensor, \b voxel_points_count tensor, and \b voxel_num tensor
 *   must have one dimension.
 * - The first dimension of \b feats tensor, \b point2voxel_map tensor and \b grad_feats tensor
 *   must be equal to \b feats_desc[0].
 * - The first dimension of \b grad_voxel_feats tensor, \b voxel_feats tensor, and \b voxel_points_count tensor
 *   must be equal to \b grad_voxel_feats_desc[0].
 * - The second dimension of \b grad_voxel_feats tensor, \b feats tensor, \b voxel_feats tensor and \b grad_feats
 *   tensor must be equal to \b grad_voxel_feats[1].
 * - The first dimension of \b voxel_num tensor is one.
 * - The shape of \b feats is [N, C]:
 *   - 2C * sizeof(datatype of \b feats) + (N + 3C + 1) * sizeof(int) + N
 *     must be less than or equal to 640KB on MLU300 series.
 *   - 2C * sizeof(datatype of \b feats) + (N + 3C + 1) * sizeof(int) + N
 *     must be less than or equal to 380KB on series higher than MLU300 series.
 *
 * @par API Dependency
 * - Before calling this function, you need to get the size of workspace by
 *   ::mluOpGetDynamicPointToVoxelBackwardWorkspaceSize.
 *
 * @par Note
 * - This function is only supported on MLU300 series or above platforms.
 * - On MLU300 series and above, the inputs \b point2voxel_map, \b voxel_points_count, and \b voxel_num with NaN or
 *   infinity are not supported.
 * - On MLU300 series and above, the inputs \b grad_voxel_feats, \b feats and \b voxel_feats with NaN or infinity
 *   are supported.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/common/cuda/scatter_points_cuda_kernel.cuh
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDynamicPointToVoxelBackward(const mluOpHandle_t handle,
                                 const mluOpReduceMode_t reduce_type,
                                 const mluOpTensorDescriptor_t grad_voxel_feats_desc,
                                 const void *grad_voxel_feats,
                                 const mluOpTensorDescriptor_t feats_desc,
                                 const void *feats,
                                 const mluOpTensorDescriptor_t voxel_feats_desc,
                                 const void *voxel_feats,
                                 const mluOpTensorDescriptor_t point2voxel_map_desc,
                                 const void *point2voxel_map,
                                 const mluOpTensorDescriptor_t voxel_points_count_desc,
                                 const void *voxel_points_count,
                                 const mluOpTensorDescriptor_t voxel_num_desc,
                                 const void *voxel_num,
                                 void *workspace,
                                 const size_t workspace_size,
                                 const mluOpTensorDescriptor_t grad_feats_desc,
                                 void *grad_feats);

// Group: DynamicPointToVoxel
/*!
 * @brief Gets extra space size that is needed in the DynamicPointToVoxelForward operation.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices
 * and queues in the DynamicPointToVoxelForward operation.
 * @param[in] feats_desc
 * The descriptor of the tensor \b feats . For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] coors_desc
 * The descriptor of the tensor \b coors . For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] workspace_size
 * A host pointer to the returned size of extra space in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetDynamicPointToVoxelForwardWorkspaceSize(mluOpHandle_t handle,
                                                const mluOpTensorDescriptor_t feats_desc,
                                                const mluOpTensorDescriptor_t coors_desc,
                                                size_t *workspace_size);

// Group: DynamicPointToVoxel
/*!
 * @brief Scatters points features into voxels, used in the voxel encoder with
 * dynamic voxelization.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * the DynamicPointToVoxelForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] reduce_type
 * Reduce op. support 'max' and 'mean'. Default: 'max'.
 * @param[in] feats_desc
 * The descriptor of the tensor \b feats. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] feats
 * Pointer to the MLU memory that stores points features to be reduced into voxels.
 * @param[in] coors_desc
 * The descriptor of the tensor \b coors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] coors
 * Pointer to the MLU memory that stores corresponding voxel coordinates of each points.
 * @param[in] workspace
 * Pointer to the MLU memory that stores the extra workspace.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in
 * ::mluOpDynamicPointToVoxelForward.
 * @param[in] voxel_feats_desc
 * The descriptor of the tensor \b voxel_feats. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] voxel_feats
 * Pointer to the MLU memory that stores the voxel features.
 * @param[in] voxel_coors_desc
 * The descriptor of the tensor \b voxel_coors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] voxel_coors
 * Pointer to the MLU memory that stores the voxel coordinates.
 * @param[in] point2voxel_map_desc
 * The descriptor of the tensor \b point2voxel_map. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] point2voxel_map
 * Pointer to the MLU memory that stores the index which is point to voxel.
 * @param[in] voxel_points_count_desc
 * The descriptor of the tensor \b voxel_points_count. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] voxel_points_count
 * Pointer to the MLU memory that stores the voxel points count.
 * @param[in] voxel_num_desc
 * The descriptor of the tensor \b voxel_num. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] voxel_num
 * Pointer to the MLU memory that stores the voxel coordinates num.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - feats, voxel_feats: float
 *   - coors, voxel_coors, point2voxel_map, voxel_points_count, voxel_num: int
 *   - reduce_type: mluOpReduceMode_t
 *
 * @par Data Layout
 * - The supported layout of input and output tensors must be \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - The \b coors tensor, \b feats tensor, \b voxel_coors tensor and \b voxel_feats tensor
 *   must have two dimensions.
 * - The \b point2voxel_map tensor, \b voxel_points_count tensor, and \b voxel_num tensor
 *   must have one dimensions.
 * - The size of the dimension of tensor \b coors, \b feats, and \b point2voxel_map must be
 *   equal to \b feats_desc[0].
 * - The first dimension of \b voxel_feats tensor, \b voxel_coors tensor, and \b voxel_points_count tensor
 *   must be equal to \b voxel_feats_desc[0].
 * - The second dimension of \b coors tensor and \b voxel_coors tensor must be equal to \b coors_desc[1].
 * - The second dimension of \b feats tensor and \b voxel_feats tensor must be equal to \b feats_desc[1].
 * - The first dimension of \b voxel_num tensor must be equal to \b voxel_feats_desc[0].
 *
 * @par API Dependency
 * - Before calling this function to perform unique operator, you need to get
 *   the size of workspace by ::mluOpGetDynamicPointToVoxelForwardWorkspaceSize.
 *
 * @par Note
 * - This function is only supported on MLU300 series or above platforms.
 * - On MLU300 series and above, the input \b coors with NaN or infinity is not supported.
 * - On MLU300 series and above, the input \b feats with NaN or infinity is supported.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/scatter_points.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDynamicPointToVoxelForward(const mluOpHandle_t handle,
                                const mluOpReduceMode_t reduce_type,
                                const mluOpTensorDescriptor_t feats_desc,
                                const void *feats,
                                const mluOpTensorDescriptor_t coors_desc,
                                void *coors,
                                void *workspace,
                                const size_t workspace_size,
                                const mluOpTensorDescriptor_t voxel_feats_desc,
                                void *voxel_feats,
                                const mluOpTensorDescriptor_t voxel_coors_desc,
                                void *voxel_coors,
                                const mluOpTensorDescriptor_t point2voxel_map_desc,
                                void *point2voxel_map,
                                const mluOpTensorDescriptor_t voxel_points_count_desc,
                                void *voxel_points_count,
                                const mluOpTensorDescriptor_t voxel_num_desc,
                                void *voxel_num);

// Group: GenerateProposalsV2
/*!
 * @brief Gets extra space size that is needed in the GenerateProposalsV2 operation.
 *
 * @par Deprecated
 * - ::mluOpGetGenerateProposalsV2WorkspaceSize is deprecated and will be removed in the future
 *   release. It is recommended to use ::mluOpGetGenerateProposalsV2WorkspaceSize_v2 instead.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices
 * and queues in the GenerateProposalsV2 operation.
 * @param[in] scores_desc
 * The descriptor of the tensor \b scores. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] size
 * A host pointer to the returned size of extra space in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetGenerateProposalsV2WorkspaceSize(mluOpHandle_t handle, const mluOpTensorDescriptor_t scores_desc, size_t *size);

// Group: GenerateProposalsV2
/*!
 * @brief Gets extra space size that is needed in the GenerateProposalsV2 operation.
 *
 * Compared with ::mluOpGetGenerateProposalsV2WorkspaceSize, this function supports
 * parameter \p pre_nms_top_n.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices
 * and queues in the GenerateProposalsV2 operation.
 * @param[in] scores_desc
 * The descriptor of the tensor \b scores. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] pre_nms_top_n
 * The number of top scoring RPN proposals to keep before applying NMS.
 * @param[out] size
 * A host pointer to the returned size of extra space in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetGenerateProposalsV2WorkspaceSize_v2(mluOpHandle_t handle,
                                            const mluOpTensorDescriptor_t scores_desc,
                                            const int32_t pre_nms_top_n,
                                            size_t *size);

// Group: GenerateProposalsV2
/*!
 * @brief Generates bounding box proposals for Faster Region-CNN.
 * This operation is the second version of generate_proposals op.
 * The proposals are generated for a list of images based on image
 * score `Scores`, bounding box regression result `BboxDeltas` as
 * well as predefined bounding box shapes `anchors`. Greedy non-maximum
 * suppression is applied to generate the final bounding boxes.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices
 * and queues in the GenerateProposalsV2 operation.
 * @param[in] pre_nms_top_n
 * The number of top scoring RPN proposals to keep before applying
 * NMS.
 * @param[in] post_nms_top_n
 * The number of top scoring RPN proposals to keep after applying
 * NMS.
 * @param[in] nms_thresh
 * NMS threshold used on RPN proposals.
 * @param[in] min_size
 * The proposal height and width both need to be greater than this
 * min_size.
 * @param[in] eta
 * The parameter for adaptive NMS.
 * @param[in] pixel_offset
 * If true, im_shape pixel offset is 1.
 * @param[in] scores_desc
 * The descriptor of the tensor \b scores. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] scores
 * Pointer to the MLU memory that stores the input tensor. The
 * scores from conv is in shape (N, H, W, A), N is batch size, A is
 * number of anchors, H and W are height and width of the feature map.
 * @param[in] bbox_deltas_desc
 * The descriptor of the tensor \b bbox_deltas. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] bbox_deltas
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] im_shape_desc
 * The descriptor of the tensor \b im_shape. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] im_shape
 * Pointer to the MLU memory that stores the input tensor. Image
 * shape in shape (N, 2), in format (height, width)
 * @param[in] anchors_desc
 * The descriptor of the tensor \b anchors. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] anchors
 * Pointer to the MLU memory that stores the input tensor.
 * Bounding box anchors from anchor_generator_op is in shape (H, W, A, 4).
 * @param[in] variances_desc
 * The descriptor of the tensor \b variances. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] variances
 * Pointer to the MLU memory that stores the input tensor.
 * Bounding box variances with the same shape as `anchors`.
 * @param[in] workspace
 * Pointer to the MLU memory that stores the extra workspace.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in ::mluOpGenerateProposalsV2.
 * @param[in] rpn_rois_desc
 * The descriptor of the tensor \b rpn_rois. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] rpn_rois
 * Pointer to the MLU memory that stores the output tensor.
 * Output proposals with shape (N * post_nms_top_n, 4).
 * @param[in] rpn_roi_probs_desc
 * The descriptor of the tensor \b rpn_roi_probs. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] rpn_roi_probs
 * Pointer to the MLU memory that stores the output tensor.
 * Scores of proposals with shape (N * post_nms_top_n, 1).
 * @param[in] rpn_rois_num_desc
 * The descriptor of the tensor \b rpn_rois_num. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] rpn_rois_num
 * Pointer to the MLU memory that stores the output tensor. The
 * number of Rpn RoIs in each image.
 * @param[in] rpn_rois_batch_size
 * Pointer to the MLU memory that stores the output tensor, which indicates
 * the number of return values of output.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - scores: float
 *   - bbox_deltas: float
 *   - im_shape: float
 *   - anchors: float
 *   - variances: float
 *   - pre_nms_top_n: int32
 *   - post_nms_top_n: int32
 *   - nms_thresh: float
 *   - min_size: float
 *   - eta: float
 *   - pixel_offset: bool
 *   - rpn_rois: float
 *   - rpn_roi_probs: float
 *   - rpn_rois_num: int32
 *   - rpn_rois_batch_size: int32
 *
 * @par Data Layout
 * - The supported data layouts of \b input tensor, \b output tensor, and
 *   \b output_size tensor are as follows:
 *
 *   - input tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output_size tensor: \p MLUOP_LAYOUT_ARRAY
 *
 * @par Scale Limitation
 * - The dimension of \b scores should be equal to 4.
 * - The dimension of \b bbox_deltas should be equal to 4.
 * - The dimension of \b im_shape should be equal to 2.
 * - The dimension of \b anchors should be equal to 4.
 * - The dimension of \b variances should be equal to 4.
 * - The dimension of \b rpn_rois should be equal to 2.
 * - The dimension of \b rpn_roi_probs should be equal to 2.
 * - The dimension of \b rpn_rois_num should be equal to 1.
 * - The dimension of \b rpn_rois_batch_size should be equal to 1.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The operator does not support adaptive NMS.
 * - The attribute `eta` should not be less than 1.
 * - ``nms_thresh`` should be more than 0.
 * - On MLU300 series and above:
 *   - If \b pixel_offset is false, input \b scores with NaN/INF is not supported.
 *   - If \b pixel_offset is true, NaN/INF is not supported.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * -
 * https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/generate_proposals_kernel.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGenerateProposalsV2(mluOpHandle_t handle,
                         const int pre_nms_top_n,
                         const int post_nms_top_n,
                         const float nms_thresh,
                         const float min_size,
                         const float eta,
                         bool pixel_offset,
                         const mluOpTensorDescriptor_t scores_desc,
                         const void *scores,
                         const mluOpTensorDescriptor_t bbox_deltas_desc,
                         const void *bbox_deltas,
                         const mluOpTensorDescriptor_t im_shape_desc,
                         const void *im_shape,
                         const mluOpTensorDescriptor_t anchors_desc,
                         const void *anchors,
                         const mluOpTensorDescriptor_t variances_desc,
                         const void *variances,
                         void *workspace,
                         size_t workspace_size,
                         const mluOpTensorDescriptor_t rpn_rois_desc,
                         void *rpn_rois,
                         const mluOpTensorDescriptor_t rpn_roi_probs_desc,
                         void *rpn_roi_probs,
                         const mluOpTensorDescriptor_t rpn_rois_num_desc,
                         void *rpn_rois_num,
                         void *rpn_rois_batch_size);

// Group: PolyNms
/*!
 * @brief Gets extra space size that is needed in the poly_nms operation.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices
 * and queues in the poly_nms operation.
 * @param[in] boxes_desc
 * The descriptor of the tensor \b boxes. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] size
 * A host pointer to the returned size of extra space in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetPolyNmsWorkspaceSize(mluOpHandle_t handle, const mluOpTensorDescriptor_t boxes_desc, size_t *size);

// Group: PolyNms
/*!
 * @brief Computes the NMS (Non-Maximum Suppression) of polygon.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices
 * and queues in the poly_nms operation.
 * @param[in] boxes_desc
 * The descriptor of the tensor \b boxes. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] boxes
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] iou_threshold
 * The threshold of IOU.
 * @param[in] workspace
 * Pointer to the MLU memory that stores the extra workspace.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in ::mluOpPolyNms.
 * @param[in] output_desc
 * The descriptor of the tensor \b output. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 * @param[in] output_size
 * Pointer to the MLU memory that stores the output_size, which indicates
 * the actual output size of the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: float
 *   - iou_threshold: float
 *   - Output tensor: int32
 *   - output_size tensor: int32
 *
 * @par Data Layout
 * - The supported data layouts of \b input tensor, \b output tensor, and
 *   \b output_size tensor are as follows:
 *   - input tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output_size tensor: \p MLUOP_LAYOUT_ARRAY
 *
 * @par Scale Limitation
 * - The dimension of \b input should be equal to 2.
 * - The dimension of \b output should be equal to 1.
 * - The dimension of \b output_size should be equal to 1.
 * - The shape[0] of output should be equal to input shape[0].
 * - The shape[1] of input should be equal to 9.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The operator does not support NAN/INF.
 * - The coordinates of the input boxes must all be sorted clockwise or
 *   counterclockwise. If the coordinates of the boxes are out of order,
 *   the calculation result is not guaranteed and is consistent with the
 *   calculation result of the competitor operation.
 * - If there are cases with the same score in the input boxes, the output
 *   results may be inconsistent with the results of competing products.
 * - The number of input boxes should be less than 9770.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/dingjiansw101/AerialDetection/tree/master/mmdet/ops/poly_nms
 */
mluOpStatus_t MLUOP_WIN_API
mluOpPolyNms(mluOpHandle_t handle,
             const mluOpTensorDescriptor_t boxes_desc,
             const void *boxes,
             const float iou_threshold,
             void *workspace,
             size_t workspace_size,
             const mluOpTensorDescriptor_t output_desc,
             void *output,
             void *output_size);

// Group: Nms
/*!
 * @brief Creates a descriptor pointed to \b desc for ::mluOpNms, and allocates
 * memory for holding the information about the Nms function. The information
 * is defined in ::mluOpNmsDescriptor_t.
 *
 * @param[out] desc
 * A host pointer to the Nms descriptor that holds information about ::mluOpNms.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ALLOC_FAILED
 *
 * @par API Dependency
 * - After calling this function, you can call ::mluOpSetNmsDescriptor to
 *   initialize and set the information to Nms descriptor.
 * - You need to call ::mluOpDestroyNmsDescriptor to destroy the descriptor.
 *
 * @par Note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateNmsDescriptor(mluOpNmsDescriptor_t *desc);

// Group: Nms
/*!
 *
 * @brief Destroys an Nms descriptor \b desc that was previously created with
 * ::mluOpCreateNmsDescriptor.
 *
 * The Nms descriptor is defined in ::mluOpNmsDescriptor_t and holds the information
 * about ::mluOpNms.
 *
 * @param[in] desc
 * The Nms descriptor to be destroyed.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Note
 * - None
 * - This function should be called to destroy the Nms descriptor. Otherwise, the
 * memory leak may occur.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDestroyNmsDescriptor(mluOpNmsDescriptor_t desc);

// Group: Nms
/*!
 * @brief Initializes the Nms descriptor \b nms_desc that was previously created with
 * ::mluOpCreateNmsDescriptor, and sets the information about the Nms operation
 * to the Nms descriptor \b nms_desc.
 *
 * @param[in] nms_desc
 * The descriptor of the Nms operation. For detailed information, see ::mluOpNmsDescriptor_t.
 * @param[in] box_mode
 * The box mode struct mode of Nms descriptor to be set. For detailed information,
 * see ::mluOpNmsBoxPointMode_t.
 * @param[in] output_mode
 * The output mode of Nms descriptor to be set. For detailed information,
 * see ::mluOpNmsOutputMode_t.
 * @param[in] algo
 * The computation algorithm of Nms operation. For detailed information,
 * see ::mluOpNmsAlgo_t.
 * @param[in] method_mode
 * The confidence update method mode. For detailed information,
 * see ::mluOpNmsMethodMode_t.
 * Note that method modes 1 and 2 are not supported in current version,
 * and they will be supported in the future.
 * @param[in] iou_threshold
 * The IOU threshold used in Nms computation.
 * Boxes would be filtered out if the IOU is greater than or equal to \b iou_threshold.
 * @param[in] soft_nms_sigma
 * The parameter used in soft Nms with Gaussian method.
 * This value would be used when method_mode is ::MLUOP_NMS_SOFT_NMS_GAUSSIAN.
 * @param[in] max_output_size
 * The maximum number of output boxes. If the dimension of input box is 3, for example
 * [batches_num, boxes_num, 4] or [batches_num, 4, boxes_num], this parameter indicates
 * the maximum number of output boxes per class.
 * @param[in] confidence_threshold
 * The confidence threshold used in Nms computation.
 * Boxes would be filtered out directly if the confidence of boxes is no more than this
 * threshold.
 * @param[in] offset
 * The offset size of boundary used in Nms computation.
 * This value would be used when \b algo is ::MLUOP_NMS_ALGO_INCLUDE_BOUNDARY.
 * @param[in] input_layout
 * The supported values are 0 and 1. 0 represents
 * [boxes_num, 4], [boxes_num, 7] or [batches_num, boxes_num, 4] and 1 represents
 * [4, boxes_num], [7, boxes_num] or [batches_num, 4, boxes_num].
 * @param[in] pad_to_max_output_size
 * When the \b pad_to_max_output_size set is true, the \b output will be padded to max_output_size
 * with zero. The default value is false.
 * @par Returns
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 * *
 * @par Scale Limitation
 * - confidence_threshold should be in the range of [0, 1].
 * - soft_nms_sigma should not be less than 0.0.
 * - offset should be 0.0 or 1.0.
 * - input_layout should be 0 or 1.
 * - max_output_size should not be less than 0.
 * - pad_to_max_output_size should be true or false.
 * *
 * @par Note
 * - If the dimension of input box is 3 ([batches_num, boxes_num, 4] or
 * [batches_num, 4, boxes_num]), the dimension of input confidence is
 * 3 ([batches_num, classes_num, boxes_num]). If the dimension of input box
 * is 2 ([boxes_num, 4], [4, boxes_num], [boxes_num, 7] or [7, boxes_num]),
 * the dimension of input confidence is 1 ([boxes_num]).
 * - When the dimension of input confidence is 3, the box mode 1 is supported.
 * - When the dimension of input confidence is 3, the output modes of 0, 1 and 2
 * are supported only in the case of batches_num = 1 and classes_num = 1.
 * - Method modes 1 and 2 are not supported in current version,
 * and they will be supported in the future.
 * - When the input boxes are in Nms3D format ([boxes_num, 7] or [7, boxes_num]),
 * only parameters iou_threshold and layout are valid and other parameters
 * can be arbitrary. Besides, the mode is set as 0 during the computation.
 * *
 * @par Requirements
 * - None.
 * *
 * @par Example
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetNmsDescriptor(mluOpNmsDescriptor_t nms_desc,
                      const mluOpNmsBoxPointMode_t box_mode,
                      const mluOpNmsOutputMode_t output_mode,
                      const mluOpNmsAlgo_t algo,
                      const mluOpNmsMethodMode_t method_mode,
                      const float iou_threshold,
                      const float soft_nms_sigma,
                      const int max_output_size,
                      const float confidence_threshold,
                      const float offset,
                      const int input_layout,
                      const bool pad_to_max_output_size);

// Group: Nms
/*!
 * @brief Calculates the subset of input tensor \b boxes with the scores of \b confidence, and returns
 * the results in the output tensors \b output and \b output_size.
 * *
 * NMS function is a necessary procedure in detection networks. And this
 * function selects no more than \b max_output_size targets with high confidence, based on their
 * IOU. This function needs extra MLU memory, and you can get the size of workspace
 * \b workspace_size with ::mluOpGetNmsWorkspaceSize.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the Nms function.
 * For detailed information, see ::mluOpHandle_t.
 * @param[in] nms_desc
 * The descriptor of the Nms function. For detailed information, see ::mluOpNmsDescriptor_t.
 * @param[in] boxes_desc
 * The descriptor of the tensor \b boxes, including the information of dimension, data type, and
 * layout of input boxes. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] boxes
 * Pointer to the MLU memory that stores the input boxes tensor.
 * @param[in] confidence_desc
 * The descriptor of the tensor \b confidence , including the information of dimension, data type, and
 * layout of input confidence. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] confidence
 * Pointer to the MLU memory that stores the input confidence tensor.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the Nms function.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in the Nms function. You can
 * get the size of the workspace with ::mluOpGetNmsWorkspaceSize.
 * @param[in] output_desc
 * The descriptor of the output tensor, including the information of dimension, data type, and layout
 * of output. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 * @param[out] output_size
 * Pointer to the MLU memory that stores the number of output boxes.
 *
 * @par Returns
 * - ::MLUOP_STATUS_SUCCESS,   ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported combinations of data types for input and output tensors are as follows:
 * - input boxes tensor: half, float
 * - input confidence tensor: half, float
 * - output tensor: half, float, int32, uint32
 * - output size: int32, uint32
 * - If the output is the indices of boxes, the output data type should be int32 or uint32, otherwise
 * the output data type should be the same as input boxes data type. The data type of output size is int32 or uint32.
 * Note that when the shape of \b boxes is [boxes_num, 4] or [4, boxes_num],
 * the combinations of input boxes tensor and input confidence tensor can be float-half, otherwise the data
 * type of input boxes and input confidence tensor must be the same.
 *
 * @par Data Layout
 * - The input boxes tensor should be a 2D tensor, and the input confidence tensor should be a 1D tensor.
 * - The output tensor is a 1D tensor if the output result is the indices of boxes, otherwise it is a 2D
 * tensor, which contains the coordinates and confidence of output boxes.
 *
 * @par Scale Limitation
 * - For the input boxes tensor, if the shape is [boxes_num, 4], the order of coordinates is x_01, y_01, x_02, y_02,
 * x_11, y_11, x_12, y_12, ... x_n1, y_n1, x_n2, y_n2. And x_i1 must be less than x_i2, y_i1 must be less than y_i2.
 * The (x_i1, y_i1) and (x_i2, y_i2) represent the top left corner and bottom right corner coordinates, respectively.
 *
 * - For the input boxes tensor, if the shape is [4, boxes_num], the order of coordinates is x_01, x_11, ... x_n1,
 * y_01, y_11, ... y_n1, x_02, x_12, ... x_n2, x_01, x_11, ... x_n1. And x_i1 must be less than x_i2, y_i1 must be less
 * than y_i2. The (x_i1, y_i1) and (x_i2, y_i2) represent the top left corner and bottom right corner coordinates,
 * respectively.
 *
 * @par API Dependency
 * - Before calling this function to perform ::mluOpSetNmsDescriptor, you need to get
 *   the size of workspace by ::mluOpGetNmsWorkspaceSize.
 *
 * @par Performance Optimization
 * - For best practices, to have better performance, set the shape of input boxes tensor to [4, num_boxes].
 * The num_boxes represents the number of input boxes.
 * - When the dimension of input box is 2, it has better performance.
 *
 * @par Note
 * - When the input boxes is in Nms3D format ([boxes_num, 7] or [7, boxes_num]),
 *   both of confidence_desc and confidence should be provided as null pointer.
 *   - In Nms3D mode, when finding the point with minimum y and minimum x in convex-hull-graham,
 *     it performs min-pooling operation. If the input data of pooling contains NaN:
 * - On MLU300 series, if the last value in the kernel of the pooling is NaN, the \b output value is NaN.
 *   Otherwise, the \b output value is the minimum value after the last NaN.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://www.tensorflow.google.cn/api_docs/python/tf/
 */
mluOpStatus_t MLUOP_WIN_API
mluOpNms(mluOpHandle_t handle,
         const mluOpNmsDescriptor_t nms_desc,
         const mluOpTensorDescriptor_t boxes_desc,
         const void *boxes,
         const mluOpTensorDescriptor_t confidence_desc,
         const void *confidence,
         void *workspace,
         size_t workspace_size,
         const mluOpTensorDescriptor_t output_desc,
         void *output,
         void *output_size);

// Group: Nms
/*!
 * @brief Returns in \b size the size of the MLU memory that is used as an extra workspace
 * needed in Nms operation.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in the Nms operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] boxes_desc
 * The descriptor of the tensor \b boxes, which contains dimension, data type, and
 * data layout of input \b boxes. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] confidence_desc
 * The descriptor of the tensor \b confidence , which contains dimension, data type, and
 * data layout of input confidence. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] size
 * A host pointer to the returned size of the extra workspace in bytes that is used
 * in the Nms operation.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 * *
 * @par Data Type
 * - The supported combinations of data types for input and output tensors are as follows:
 * - input boxes tensor: half, float
 * - input confidence tensor: half, float
 * - output tensor: half, float, int32, uint32
 * - output size: int32, uint32
 * - If the output is the indices of boxes, the output data type should be int32 or uint32, otherwise
 * the output data type should be the same as input boxes data type. The data type of output size is int32 or uint32.
 * Note that when the shape of \b boxes is [boxes_num, 4] or [4, boxes_num],
 * the combinations of input boxes tensor and input confidence tensor can be float-half, otherwise the data
 * type of input boxes and input confidence tensor must be the same.
 *
 * @par Data Layout
 * - The input boxes tensor should be a 2D tensor, and the input confidence tensor should be a 1D tensor.
 * - The output tensor is a 1D tensor if the output result is the indices of boxes, otherwise it is a 2D
 * tensor, which containing the coordinates and confidence of output boxes.
 *
 * @par API Dependency
 * - The allocated extra workspace should be passed to ::mluOpNms to perform the Nms operation.
 * *
 * @par Note
 * - When the input boxes is in Nms3D format ([boxes_num, 7] or [7, boxes_num]),
 * the confidence_desc must be provided with null pointer.
 * *
 * @par Requirements
 * - None.
 * *
 * @par Example
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetNmsWorkspaceSize(mluOpHandle_t handle,
                         const mluOpTensorDescriptor_t boxes_desc,
                         const mluOpTensorDescriptor_t confidence_desc,
                         size_t *size);

// Group: PriorBox
/*!
 * @brief Generates prior boxes for SSD (Single Shot MultiBox Detector) algorithm.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices
 * and queues in the prior_box operation.
 * @param[in] min_sizes_desc
 * The descriptor of the tensor \b min_sizes. The minimum size of generated
 * prior boxes.
 * @param[in] min_sizes
 * Pointer to the MLU memory that stores the min_sizes tensor.
 * @param[in] aspect_ratios_desc
 * The descriptor of the tensor \b aspect_ratios. The aspect ratios of
 * generated prior boxes.
 * @param[in] aspect_ratios
 * Pointer to the MLU memory that stores the aspect_ratios tensor.
 * @param[in] variances_desc
 * The descriptor of the tensor \b variances. The variances to be
 * encoded in prior boxes.
 * @param[in] variances
 * Pointer to the MLU memory that stores the variances tensor.
 * @param[in] max_sizes_desc
 * The descriptor of the tensor \b max_sizes. The maximum size of generated
 * prior boxes.
 * @param[in] max_sizes
 * Pointer to the MLU memory that stores the max_sizes tensor.
 * @param[in] height
 * The height of the \b input feature_map.
 * @param[in] width
 * The width of the \b input feature_map.
 * @param[in] im_height
 * The height of the \b input image.
 * @param[in] im_width
 * The width of the \b input image.
 * @param[in] step_h
 * The prior box step in height.
 * @param[in] step_w
 * The prior box step in width.
 * @param[in] offset
 * The prior box center offset.
 * @param[in] clip
 * A Boolean value whether to clip out-of-boundary boxes.
 * @param[in] min_max_aspect_ratios_order
 * If the value is set as True, the \b output prior box is in
 * the order of [min, max, aspect_ratios]; otherwise the order is
 * [min, aspect_ratios, max].
 * @param[in] output_desc
 * The descriptor of the tensor \b output. The \b output prior boxes of
 * PriorBox.
 * @param[out] output
 * Pointer to the MLU memory that stores the \b output tensor.
 * @param[in] var_desc
 * The descriptor of the tensor \b var. The expanded variances of
 * PriorBox.
 * @param[out] var
 * Pointer to the MLU memory that stores the var tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - min_sizes tensor: float
 *   - aspect_ratios tensor: float
 *   - variances tensor: float
 *   - max_sizes tensor: float
 *   - height: int
 *   - width: int
 *   - im_height: int
 *   - im_width: int
 *   - step_h: float
 *   - step_w: float
 *   - offset: float
 *   - clip: bool
 *   - min_max_aspect_ratios_order: bool
 *   - output: float
 *   - var: float
 *
 * @par Data Layout
 * - The supported data layouts of \b input, and \b output are as follows:
 *   - input tensor:
 *     - min_sizes: \p MLUOP_LAYOUT_ARRAY
 *     - aspect_ratios: \p MLUOP_LAYOUT_ARRAY
 *     - variances: \p MLUOP_LAYOUT_ARRAY
 *     - max_sizes: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor:
 *     - output: \p MLUOP_LAYOUT_ARRAY
 *     - var: \p MLUOP_LAYOUT_ARRAY
 *
 * @par Scale Limitation
 * - The dimension of \b min_sizes should be equal to 1.
 * - The dimension of \b aspect_ratios should be equal to 1.
 * - The dimension of \b variances should be equal to 1.
 * - The dimension of \b max_sizes should be equal to 1.
 * - The dimension of \b output should be equal to 1.
 * - The dimension of \b var should be equal to 1.
 * - The shape[0] of \b variances should be equal to 4.
 * - The shape[0] of \b min_sizes should be larger than 0.
 * - The shape[0] of \b aspect_ratios should be larger than 0.
 * - The shape of \b output should be the same with \b var.
 * - The shape[0] of the \b output should be equal to the input height.
 * - The shape[1] of the \b output should be equal to the input width.
 * - The shape[2] of the \b output and \b var must be less than 2900 on MLU300 series.
 * - The shape[2] of \b output and \b var should be equal to
 *   the product of shape[0] of \b min_sizes and \b aspect_ratios
 *   plus shape[0] of \b max_sizes.
 * - The height should be greater than or equal to 0.
 * - The width should be greater than or equal to 0.
 * - The step_h should be greater than 0.
 * - The step_w should be greater than 0.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The shape[2] of the \b output and \b var must be
 *   less than 2900 on MLU300 series.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/prior_box_kernel.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpPriorBox(mluOpHandle_t handle,
              const mluOpTensorDescriptor_t min_sizes_desc,
              const void *min_sizes,
              const mluOpTensorDescriptor_t aspect_ratios_desc,
              const void *aspect_ratios,
              const mluOpTensorDescriptor_t variances_desc,
              const void *variances,
              const mluOpTensorDescriptor_t max_sizes_desc,
              const void *max_sizes,
              const int height,
              const int width,
              const int im_height,
              const int im_width,
              const float step_h,
              const float step_w,
              const float offset,
              const bool clip,
              const bool min_max_aspect_ratios_order,
              const mluOpTensorDescriptor_t output_desc,
              void *output,
              const mluOpTensorDescriptor_t var_desc,
              void *var);

// Group: PsRoiPool
/*!
 * @brief Generates fixed size feature map for each ROI (Regions of Interest).
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices
 * and queues in the psroipool_forward operation. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in] spatial_scale
 * The spatial scale of each ROI in the output.
 * @param[in] group_size
 * The number of \b rois to be divided equally in each direction.
 * @param[in] pooled_height
 * An integer value which is the height of the output after pooling.
 * @param[in] pooled_width
 * An integer value which is the width of the output after pooling.
 * @param[in] output_dim
 * An integer value which is the channel of the output after pooling.
 * @param[in] input_desc
 * The descriptor of the tensor, which contains dimension and the layout of input.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor. The shape of \b input is
 * [batch_num, H, W, C].
 * @param[in] rois_desc
 * The descriptor of \b rois tensor, which contains dimension and the layout of rois.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] rois
 * Pointer to the MLU memory that stores the rois tensor. \b rois[1] consists of
 * [batch_id, roi_start_w, roi_start_h, roi_end_w, roi_end_h], where \p batch_id is the ID
 * of the batch.
 * @param[in] output_desc
 * The descriptor of output tensor, which contains dimension and the layout of output.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor. The shape of \b output is
 * [rois[0], pooled_height, pooled_width, output_dim].
 * @param[in] mapping_channel_desc
 * The descriptor of the tensor \b mapping_channel, which contains dimension and the layout of
 * mapping_channel. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] mapping_channel
 * Pointer to the MLU memory that stores the mapping_channel tensor. The shape of
 * \b mapping_channel is [rois[0], pooled_height, pooled_width, output_dim].
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: float
 *   - Rois tensor: float
 *   - output tensor: float
 *   - Mapping_channel tensor: int32
 *
 * @par Data Layout
 * - The supported data layout of \b input tensor, \b rois tensor, \b output tensor, and \b mapping_channel
 *   tensor are as follows:
 *   - input tensor: \p MLUOP_LAYOUT_NHWC
 *   - Rois tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor: \p MLUOP_LAYOUT_NHWC
 *   - Mapping_channel tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The input tensors, mapping_channel and output must have four dimensions.
 * - The \b rois tensor should be 2D array.
 * - The shape of \b rois should be [rois_num, 5].
 * - \p batch_id should be in the range of [0, \p batch_num - 1].
 * - The spatial_scale should be greater than 0.
 * - The group_size should be greater than 1.
 * - THe output_dim should be greater than 1.
 * - The group_size should be equal to pooled_height.
 * - The pooled_height should be equal to pooled_width.
 * - The fourth dimension of input tensor should be equal to pooled_height * pooled_width *
 *   output_dim.
 * - The first dimension of output tensor and mapping_channel tensor must be the same size.
 * - The second dimension of output tensor and mapping_channel tensor must be the same size.
 * - The third dimension of output tensor and mapping_channel tensor must be the same size.
 * - The fourth dimension of output tensor and mapping_channel tensor must be the same size.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/princewang1994/R-FCN.pytorch/tree/master/lib/model/psroi_pooling
 */
mluOpStatus_t MLUOP_WIN_API
mluOpPsRoiPoolForward(mluOpHandle_t handle,
                      const int pooled_height,
                      const int pooled_width,
                      const float spatial_scale,
                      const int group_size,
                      const int output_dim,
                      const mluOpTensorDescriptor_t input_desc,
                      const void *input,
                      const mluOpTensorDescriptor_t rois_desc,
                      const void *rois,
                      const mluOpTensorDescriptor_t output_desc,
                      void *output,
                      const mluOpTensorDescriptor_t mapping_channel_desc,
                      void *mapping_channel);

// Group: PsRoiPool
/*!
 * @brief Computes the gradients of feature map \b bottom_grad based on the
 * inputs \b top_grad, \b rois, and \b mapping_channel to perform the backpropagation
 * of ::mluOpPsRoiPoolForward.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * psroipool_forward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] pooled_height
 * An integer value which is the height of the output after pooling.
 * @param[in] pooled_width
 * An integer value which is the width of the output after pooling.
 * @param[in] spatial_scale
 * A float value which is the scale factor of coordinates of rois.
 * @param[in] output_dim
 * An integer value which is the channel of the output after pooling.
 * @param[in] top_grad_desc
 * The descriptor of the tensor \b top_grad, which contains the dimension and the layout
 * of top_grad tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] top_grad
 * Pointer to the MLU memory that stores the top_grad tensor.
 * @param[in] rois_desc
 * The descriptor of the tensor \b rois, which contains the dimension and the layout
 * of rois tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] rois
 * Pointer to the MLU memory that stores the rois tensor.
 * @param[in] mapping_channel_desc
 * The descriptor of the tensor \b mapping_channel, which contains the dimension and the
 * layout of mapping_channel. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mapping_channel
 * Pointer to the MLU memory that stores the mapping_channel tensor.
 * @param[in] bottom_grad_desc
 * The descriptor of the \b bottom_grad tensor, which contains the dimension and the
 * layout of mapping_channel. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] bottom_grad
 * Pointer to the MLU memory that stores the bottom_grad tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - top_grad tensor: float
 *   - rois tensor: float
 *   - mapping_channel tensor: int
 *   - bottom_grad tensor: float
 *
 * @par Data Layout
 * - The supported data layouts of \b top_grad tensor,  \b rois tensor, \b mapping_channel tensor,
 *   and \b bottom_grad tensor are as follows:
 *   - top_grad tensor: \p MLUOP_LAYOUT_NHWC
 *   - rois tensor: \p MLUOP_LAYOUT_ARRAY
 *   - mapping_channel tensor: \p MLUOP_LAYOUT_NHWC
 *   - bottom_grad tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The top_grad tensor, mapping_channel tensor and bottom_grad tensor must be 4-D.
 * - Each dimension of the top_grad tensor and the mapping_channel tensor must be the same.
 * - The rois tensor be be 2D.
 * - The shape of \b top_grad should be [rois_num, pooled_height, pooled_width, output_dim].
 * - The shape of \b rois should be [rois_num, 5].
 * - The shape of \b mapping_channel should be [rois_num, pooled_height, pooled_width, output_dim].
 * - The shape of \b bottom_grad should be [batch_num, height, width, channels].
 * - \b rois[i] consists of [batch_id, roi_start_w, roi_start_h, roi_end_w, roi_end_h].
 *   \p batch_id should be in the range of [0, batch_num -1].
 * - The \b spatial_scale should be larger than 0.
 * - The \b output_dim should be larger than or equal to 1.
 * - The \b pooled_height should be equal to \b pooled_width.
 * - The \p channels should be equal to \b pooled_height * \b pooled_width * \b output_dim.
 *
 * @par API Dependency
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Reference
 * - https://github.com/princewang1994/R-FCN.pytorch/tree/master/lib/model/psroi_pooling
 */
mluOpStatus_t MLUOP_WIN_API
mluOpPsRoiPoolBackward(mluOpHandle_t handle,
                       const int pooled_height,
                       const int pooled_width,
                       const float spatial_scale,
                       const int output_dim,
                       const mluOpTensorDescriptor_t top_grad_desc,
                       const void *top_grad,
                       const mluOpTensorDescriptor_t rois_desc,
                       const void *rois,
                       const mluOpTensorDescriptor_t mapping_channel_desc,
                       const void *mapping_channel,
                       const mluOpTensorDescriptor_t bottom_grad_desc,
                       void *bottom_grad);

// Group: RoiAlign
/*!
 * @brief Creates a descriptor pointed by \b desc for ::mluOpRoiAlignForward_v2,
 * and allocates memory for holding the information about the function.
 * The information is defined in ::mluOpRoiAlignForwardDescriptor_t.
 *
 * @param[in] desc
 * A host pointer to the descriptor that holds information about ::mluOpRoiAlignForward_v2.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ALLOC_FAILED
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - ::mluOpSetRoiAlignForwardDescriptor_v2
 *   should be called to initialize the descriptor after calling this function.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */

mluOpStatus_t MLUOP_WIN_API
mluOpCreateRoiAlignForwardDescriptor(mluOpRoiAlignForwardDescriptor_t *desc);

// Group: RoiAlign
/*!
 * @brief Initializes the descriptor \b desc that was previously created with
 * ::mluOpCreateRoiAlignForwardDescriptor function, and sets RoiAlign information
 * to the descriptor \b desc. The information includes height \b pooled_height and
 * width \b pooled_width of RoiAlign feature map, sampling_ratio \b sampling_ratio
 * and spatial_scale \b spatial_scale for each boxes, shift mode \b aligned
 * and pooling mode \b pool_mode for RoiAlign operation.
 *
 * @param[in] roialign_desc
 * The descriptor of the RoiAlign operation. For detailed information,
 * see ::mluOpRoiAlignForwardDescriptor_t.
 * @param[in] pooled_height
 * The height of output feature map. The value of this parameter should be greater than 0.
 * @param[in] pooled_width
 * The width of output feature map. The value of this parameter should be greater than 0.
 * @param[in] sampling_ratio
 * The number of sampling points in grid used to compute output.
 * @param[in] spatial_scale
 * The spatial scale of each ROI in output.
 * @param[in] pool_mode
 * If \b pool_mode is 1, the average pooling mode is used.
 * If \b pool_mode is 0, the maximum pooling mode is used.
 * @param[in] aligned
 * A Boolean value which determines whether to shift the boxes by 0.5 pixel. If \b aligned
 * is true, the boxes is shifted by 0.5. If \b aligned is false, the boxes is not shifted.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - This function should be called after ::mluOpCreateRoiAlignForwardDescriptor.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetRoiAlignForwardDescriptor_v2(mluOpRoiAlignForwardDescriptor_t roialign_desc,
                                     const int pooled_height,
                                     const int pooled_width,
                                     const int sampling_ratio,
                                     const float spatial_scale,
                                     const int pool_mode,
                                     const bool aligned);

// Group: RoiAlign
/*!
 * @brief Destroys a RoiAlign descriptor \b desc that was previously created
 * with ::mluOpCreateRoiAlignForwardDescriptor function.
 *
 * The RoiAlign descriptor is defined in ::mluOpRoiAlignForwardDescriptor_t
 * and holds information about ::mluOpRoiAlignForward_v2.
 *
 * @param[in] desc
 * The RoiAlign descriptor to be destroyed.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - This function should be called after ::mluOpRoiAlignForward_v2 to
 *   destroy the descriptor, Otherwise, memory leak may occur.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */

mluOpStatus_t MLUOP_WIN_API
mluOpDestroyRoiAlignForwardDescriptor(mluOpRoiAlignForwardDescriptor_t desc);

// Group: RoiAlign
/*!
 * @brief Computes the output feature map \b output based on the input feature map \b input
 * and bounding boxes \b boxes to perform this function. This function supports
 * maximum pooling mode with two more output \b argmax_x and \b argmax_y.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpRoiAlignForward_v2. For detailed information, see ::mluOpHandle_t.
 * @param[in] roialign_desc
 * The descriptor of the RoiAlign operation. For detailed information,
 * see ::mluOpRoiAlignForwardDescriptor_t.
 * @param[in] input_desc
 * The descriptor of the input tensor in RoiAlign process. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the \b input tensor.
 * @param[in] boxes_desc
 * The descriptor of \b boxes tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] boxes
 * Pointer to the MLU memory that stores the \b boxes tensor.
 * @param[in] output_desc
 * The descriptor of \b output tensor of the original images. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the \b output tensor.
 * @param[in] argmax_x_desc
 * The descriptor of \b argmax_x tensor that stores the coordinate of x axis. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[out] argmax_x
 * Pointer to the MLU memory that stores the \b argmax_x tensor. When \b pool_mode is maximum
 * pooling mode, \b argmax_x represents \b output coordinate of x axis. When \b pool_mode is
 * average pooling mode, \b argmax_x is NULL.
 * @param[in] argmax_y_desc
 * The descriptor of the \b argmax_y tensor that stores the coordinate of y axis. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[out] argmax_y
 * Pointer to the MLU memory that stores the \b argmax_y tensor. When \b pool_mode is maximum
 * pooling mode, \b argmax_y represents \b output coordinate of y axis. When \b pool_mode is
 * average pooling mode, \b argmax_y is NULL.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS,   ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of all tensors should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - boxes tensor: half, float
 *   - output tensor: half, float
 *   - argmax_x tensor: half, float
 *   - argmax_y tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b input tensor, \b boxes tensor, \b output tensor, \b argmax_x tensor,
 *   and \b argmax_y tensor are as follows:
 *   - input tensor: \p MLUOP_LAYOUT_NHWC
 *   - boxes tensor: \p MLUOP_LAYOUT_ARRAY, which only supports 2D tensor
 *   - output tensor: \p MLUOP_LAYOUT_NHWC
 *   - argmax_x tensor: \p MLUOP_LAYOUT_NHWC
 *   - argmax_y tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The \b input tensor, \b output tensor, \b argmax_x tensor and \b argmax_y tensor must have four dimensions.
 * - \b input data type of half is not recommended due to low precision.
 * - The size of the lowest dimension of \b input tensor and \b output tensor must be the same.
 * - The size of the lowest dimension of \b input tensor and \b argmax_x tensor must be the same.
 * - The size of the lowest dimension of \b input tensor and \b argmax_y tensor must be the same.
 * - The \b boxes tensor must have two dimensions.
 * - The size of the highest dimension of \b output tensor and \b boxes tensor must be the same.
 * - The size of the highest dimension of \b argmax_x tensor and \b boxes tensor must be the same.
 * - The size of the highest dimension of \b argmax_y tensor and \b boxes tensor must be the same.
 * - The shape of \b boxes should be [boxes_num, 5].
 * - \b boxes[i] consists of [batch_id, x1, y1, x2, y2]. \p batch_id specifies which image this box
 *   is in, and should be in the range of [0, batch_num - 1]. \p x1 and \p y1 specify the starting
 *   coordinate of this box in origin image. \p x2 and \p y2 specify the ending coordinate of this box
 *   in origin image. \p x1 and \p y1 should be greater than or equal to 0. \p x2 should be greater
 *   than \p x1. \p y2 should be greater than \p y1.
 *
 * @par API Dependency
 * - This function should be called with ::mluOpSetRoiAlignForwardDescriptor_v2.
 *
 * @par Note
 * - When \b input contains NaN, if  \b pool_mode is maximum pooling_mode, \b output gets more NaN than
 *   IEEE 754 on MLU300 series.
 *
 * @par Example
 * - The example of ::mluOpRoiAlignForward_v2 is as follows:
     @verbatim
     input two arrays by 1 * 1 * 1 * 1 and 1 * 5 --> input: [[[[1.0]]]]

     --> boxes: [[1.0, 0.0, 0.0, 1.0, 1.0]]

     parameters:
            pooled_height: 1.0, pooled_width: 1.0, spatial_scale: 1.0,
            sampling_ratio: 0, aligned: false, pool_mode = 0

     output array by 1 * 1 * 1 * 1 -->
         output: [[[[1]]
     argmax_x array by 1 * 1 * 1 * 1 -->
         argmax_x: [[[[0.5]]
     argmax_y array by 1 * 1 * 1 * 1 -->
         argmag_y: [[[[0.5]]

     @endverbatim
 *
 * @par Reference
 * - github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/roi_align_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRoiAlignForward_v2(mluOpHandle_t handle,
                        const mluOpRoiAlignForwardDescriptor_t roialign_desc,
                        const mluOpTensorDescriptor_t input_desc,
                        const void *input,
                        const mluOpTensorDescriptor_t boxes_desc,
                        const void *boxes,
                        const mluOpTensorDescriptor_t output_desc,
                        void *output,
                        const mluOpTensorDescriptor_t argmax_x_desc,
                        void *argmax_x,
                        const mluOpTensorDescriptor_t argmax_y_desc,
                        void *argmax_y);

// Group: RoiAlignRotated
/*!
 * @brief Extracts the corresponding \b features information to \b output by bilinear interpolation
 * according to the \b rois with rotation.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpRoiAlignRotatedForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] features_desc
 * The descriptor of the tensor \b features.
 * @param[in] features
 * Pointer to the MLU memory that stores the features tensor. The shape of \b features
 * is [batch_num, H, W, C].
 * @param[in] rois_desc
 * The descriptor of \b rois tensor, which contains dimension and the layout of rois.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] rois
 * Pointer to the MLU memory that stores rois tensors. \b rois[i] consists of [batch_id,
 * x, y, w, h, theta], where \p batch_id is the ID of the batch, \p x and \p y are the coordinate
 * of center point, \p w and \p h are the width and height of rois, and \p theta is the rotated angle.
 * @param[in] pooled_height
 * The height of output.
 * @param[in] pooled_width
 * The width of output.
 * @param[in] sample_ratio
 * The number of sampling points in the bin which is used to compute the output.
 * @param[in] spatial_scale
 * The spatial scale of each ROI in the output.
 * @param[in] aligned
 * A Boolean value which determines whether to shift the ROI by 0.5 pixel. If the
 * value of \b aligned is set to true, the ROI is shifted by 0.5. If the value of \b aligned
 * is set to false, the ROI is not shifted.
 * @param[in] clockwise
 * A Boolean value which determines whether the rotation of ROI is clockwise.
 * @param[out] output_desc
 * The descriptor of output, which contains dimension and the layout of output.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of all tensors should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - rois tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b features tensor, \b rois tensor, and \b output tensor are as follows:
 *   - input tensor: \p MLUOP_LAYOUT_NHWC
 *   - rois tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The \b features tensor and \b output tensor should be 4D.
 * - The half data type is not recommended due to low precision.
 * - The size of the lowest dimension of \b features tensor and \b output tensor should be the same.
 * - The \b rois tensor should be 2D array.
 * - The size of the highest dimension of \b output tensor and \b rois tensor should be the same.
 * - The shape of \b rois should be [rois_num, 6].
 * - \p batch_id should be in the range of [0, \p batch_num - 1]; \p x and \p y should be greater than or
 *   equal to 0 and less than \p H and \p W respectively. Both of \p h and \p w should be greater than zero
 *   and less than \p H and \p W respectively.
 * - \p spatial_scale and \p sample_ratio should not be less than zero.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - NaN and infinity are not supported for all parameters in \b boxes, except for the \p x and \p y parameters
 *   that support infinity.
 * - The values of the parameters \p x , \p y, \p w and \p h in \b rois multiplied by \p spatial_scale cannot exceed
 *   the range that can be represented by the parameter type.
 *
 * @par Example
 * - The example of the roi_align_rotated_forward operation is as follows:
     @verbatim
      input two arrays by 1 * 3 * 3 * 1 and 1 * 6 -->
      input:[[[[1.0],[1.0],[1.0]],[[1.0],[1.0],[1.0]],[[1.0],[1.0],[1.0]]]]

      --> rois: [[0.0, 1.0, 1.0, 1.0, 1.0, 0.0]]

      param:
             pooled_height: 2, pooled_width: 2, spatial_scale: 1.0,
             sampling_ratio: 2, aligned: false, clockwise: false

      output array by 1 * 2 * 2 * 1 -->
          output: [[[[1],[1]],[[1],[1]]]]
     @endverbatim
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/roi_align_rotated.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRoiAlignRotatedForward(mluOpHandle_t handle,
                            const mluOpTensorDescriptor_t features_desc,
                            const void *features,
                            const mluOpTensorDescriptor_t rois_desc,
                            const void *rois,
                            const int pooled_height,
                            const int pooled_width,
                            const int sample_ratio,
                            const float spatial_scale,
                            const bool aligned,
                            const bool clockwise,
                            const mluOpTensorDescriptor_t output_desc,
                            void *output);

// Group: RoiAlignRotated
/*!
 * @brief Computes the gradients of feature map \b bottom_grad based on the input \b top_grad and
 * \b rois to perform the backpropagation of ::mluOpRoiAlignRotatedForward.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpRoiAlignRotatedBackward. For detailed information, see ::mluOpHandle_t.
 * @param[in] top_grad_desc
 * The descriptor of the tensor \b top_grad in the backpropagation process.
 * @param[in] top_grad
 * Pointer to the MLU memory that stores the top_grad tensor.
 * @param[in] rois_desc
 * The descriptor of the tensor \b rois, which contains dimension and the layout of rois.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] rois
 * Pointer to the MLU memory that stores rois tensors. \b rois[i] consists
 * of [batch_id, x, y, w, h, theta], where \p batch_id is the ID of the batch, \p x
 * and \p y are the coordinate of center point, \p w and \p h are the width and height
 * of rois, and \p theta is the rotated angle.
 * @param[in] pooled_height
 * The height of output.
 * @param[in] pooled_width
 * The width of output.
 * @param[in] sample_ratio
 * The number of sampling points in the bin which is used to compute the output.
 * @param[in] spatial_scale
 * The spatial scale of each ROI in the output.
 * @param[in] aligned
 * A Boolean value which determines whether to shift the ROI by 0.5 pixel.
 * If the value of \b aligned is set to true, the ROI is shifted by 0.5. If the value
 * of \b aligned is set to false, the ROI is not shifted.
 * @param[in] clockwise
 * A Boolean value which determines whether the rotation of ROI is clockwise.
 * @param[in] bottom_grad_desc
 * The descriptor of the tensor \b bottom_grad.
 * @param[out] bottom_grad
 * Pointer to the MLU memory that stores the bottom_grad tensor. The shape of
 * bottom_grad is [batch_num, H, W, C].
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of all tensors should be the same.
 * - The supported input and output tensors are as follows:
 *   - top_grad tensor: half, float
 *   - rois tensor: half, float
 *   - bottom_grad tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b top_grad tensor, \b rois tensor, and \b bottom_grad tensor are as follows:
 *   - top_grad tensor: \p MLUOP_LAYOUT_NHWC
 *   - rois tensor: \p MLUOP_LAYOUT_ARRAY
 *   - bottom_grad tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The \b bottom_grad tensor and \b top_grad tensor should be 4D.
 * - The half data type is not recommended due to low precision.
 * - The size of the lowest dimension of \b bottom_grad tensor and \b top_grad tensor should be the same.
 * - The \b rois tensor should be 2D array.
 * - The size of the highest dimension of \b top_grad tensor and \b rois tensor should be the same.
 * - \p batch_id should be in the range of [0, \p batch_num - 1], \p x and \p y should be greater than or
 *   equal to 0 and less than \p H and \p W respectively. Both of \p h and \p w should be greater than zero
 *   and less than \p H and \p W respectively.
 * - \p spatial_scale and \p sample_ratio should not be less than zero.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - NaN and infinity are not supported for all parameters in \b boxes, except for the \p x and \p y parameters
 *   that support infinity.
 * - The values of the parameters \p x , \p y , \p w , and \p h in \b rois multiplied by \p spatial_scale cannot exceed
 *   the range that can be represented by the parameter type.
 *
 * @par Example
 * - The example of the roi_align_rotated_backward operation is as follows:
     @verbatim
      input two arrays by 1 * 1 * 1 * 1 and 1 * 6 --> input: [[[[1.0]]]]

      --> rois: [[0.0, 0.0, 0.0, 1.0, 1.0, 0.0]]

      param:
             pooled_height: 1.0, pooled_width: 1.0, spatial_scale: 1.0,
             sampling_ratio: 2, aligned: false, clockwise: false

      output array by 1 * 2 * 2 * 1 -->
          output: [[[[0.25], [0.25]], [[0.25], [0.25]]]]
     @endverbatim
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/roi_align_rotated.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRoiAlignRotatedBackward(mluOpHandle_t handle,
                             const mluOpTensorDescriptor_t top_grad_desc,
                             const void *top_grad,
                             const mluOpTensorDescriptor_t rois_desc,
                             const void *rois,
                             const int pooled_height,
                             const int pooled_width,
                             const int sample_ratio,
                             const float spatial_scale,
                             const bool aligned,
                             const bool clockwise,
                             const mluOpTensorDescriptor_t bottom_grad_desc,
                             void *bottom_grad);

// Group: RoiCrop
/*!
 * @brief Generates fixed size feature map for each grid. Each value in the
 * feature map is interpolated by bilinear sampling.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in ::mluOpRoiCropForward operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] input_desc
 * The descriptor of the input tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] grid_desc
 * The descriptor of the tensor \b grid. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] grid
 * Pointer to the MLU memory that stores the grid tensor. NaN and INF
 * datas are not supported.
 * @param[in] output_desc
 * The descriptor of the output tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of input tensors and output tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: float
 *   - Grid tensor: float
 *   - output tensor: float
 * @par Data Layout
 * - The supported data layouts of \b input tensor, \b grid tensor, and \b output tensor are as follows:
 *   - input tensor: \p MLUOP_LAYOUT_NHWC
 *   - Grid tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The input tensor, grid tensor and output tensor must have four dimensions.
 * - The size of the first dimension of input tensor is divided by size of the
 *   first dimension of grid tensor.
 * - The second dimension of grid tensor and output tensor must be the same size.
 * - The third dimension of grid tensor and output tensor must be the same size.
 * - The fourth dimension of input tensor and output tensor must be the same size.
 * - The size of the fourth dimension of grid tensor must be equal to 2.
 * - Grid tensor \b grid must meet the following data range:
 *   - Float: [-1.0,1.0].
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - On MLU300, the input \b grid with NaN or infinity is not supported.
 * - On series higher than MLU300 series, the inputs \b grid and \b input with NaN or infinity are supported.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/princewang1994/R-FCN.pytorch/tree/master/lib/model/roi_crop
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRoiCropForward(mluOpHandle_t handle,
                    const mluOpTensorDescriptor_t input_desc,
                    const void *input,
                    const mluOpTensorDescriptor_t grid_desc,
                    const void *grid,
                    const mluOpTensorDescriptor_t output_desc,
                    void *output);

// Group: RoiCrop
/*!
 * @brief Computes the gradients of images \b grad_input based on the gradients
 * \b grad_output and coordinates mapping parameter \b grid to perform the
 * backpropagation.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in ::mluOpRoiCropBackward operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] grad_output_desc
 * The descriptor of the \b grad_output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] grad_output
 * Pointer to the MLU memory that stores the gradient tensor \b grad_output
 * in the backpropagation process.
 * @param[in] grid_desc
 * The descriptor of the tensor \b grid. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] grid
 * Pointer to the MLU memory that stores the coordinate mapping
 * tensor.
 * @param[in] grad_input_desc
 * The descriptor of the tensor \b grad_input. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] grad_input
 * Pointer to the MLU memory that stores the gradient tensor of the
 * original images.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of all tensors must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - Grad_input tensor: float
 *   - Grad_output tensor: float
 *   - Grid tensor: float
 *
 * @par Data Layout
 * - The supported data layouts of \b Grad_output tensor, \b Grid tensor tensor, and \b Grad_input
 *   tensor are as follows:
 *   - Grad_output tensor: \p MLUOP_LAYOUT_NHWC
 *   - Grid tensor: \p MLUOP_LAYOUT_ARRAY
 *   - Grad_input tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The grad_output tensor, grid tensor, and grad_input tensor must have four
 *   dimensions.
 * - The size of the first dimension of grad_input tensor is divided by size of
 *   the first dimension of grid tensor.
 * - The second dimension of grid tensor and grad_output tensor must be the same size.
 * - The third dimension of grid tensor and grad_output tensor must be the same size.
 * - The fourth dimension of grad_input \b grad_input tensor and grad_output tensor
 *   \b grad_output must be the same size.
 * - The size of the fourth dimension of grid tensor \b grid must be equal to 2.
 * - Grid tensor \b grid must meet the following data range:
 *   - Float: [-1.0,1.0]
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - On MLU300, the input \b grid with NaN or infinity is not supported.
 * - On series higher than MLU300 series, the inputs \b grid and \b grad_output with NaN or infinity are supported.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/princewang1994/R-FCN.pytorch/tree/master/lib/model/roi_crop
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRoiCropBackward(mluOpHandle_t handle,
                     const mluOpTensorDescriptor_t grad_output_desc,
                     const void *grad_output,
                     const mluOpTensorDescriptor_t grid_desc,
                     const void *grid,
                     const mluOpTensorDescriptor_t grad_input_desc,
                     void *grad_input);

// Group: RotatedFeatureAlign
/*!
 * @brief Uses the feature interpolation to obtain the position information corresponding to the
 * refined rotate anchors \b bboxes and reconstructs the feature maps \b output in pixel-wise
 * manner to achieve feature alignment.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpRotatedFeatureAlignForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] input_desc
 * The descriptor of the input tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] bboxes_desc
 * The descriptor of the tensor \b bboxes, which contains the dimension and layout of bboxes tensor.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] bboxes
 * Pointer to the MLU memory that stores the bboxes tensor.
 * @param[in] spatial_scale
 * A float value that is the scale factor of coordinates of bboxes.
 * @param[in] points
 * An int value that is the number of sample points. Only 1 and 5 are supported. The default value is 1.
 * @param[in] output_desc
 * The descriptor of the output tensor, which contains the dimension and layout of output tensor.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of all tensors should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - bboxes tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b input tensor, \b bboxes tensor, and \b output tensor are as follows:
 *   - input tensor: \p MLUOP_LAYOUT_NHWC
 *   - bboxes tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The input tensor and output tensor must be 4D.
 * - The size of each dimension of input tensor and output tensor must be the same.
 * - The bboxes tensor must be 4D.
 * - The 4D information of bboxes tensor consists of [y, x, w, h, a].
 * - The 1D sizes of input tensor and bboxes tensor must be the same.
 * - The 2D sizes of input tensor and bboxes tensor must be the same.
 * - The 3D sizes of input tensor and bboxes tensor must be the same.
 * - The shape of \b input should be [batch_num, height, width, channels].
 * - The shape of \b bboxes should be [batch_num, height, width, 5].
 * - The shape of \b output should be [batch_num, height, width, channels].
 * - \b points is the number of sample points. Only 1 and 5 are supported. The default value is 1.
 * - The value of \b spatial_scale is larger than zero.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The inputs \b bboxes and \b spatial_scale with NaN or infinity are not supported.
 *
 * @par Example
 * - The example of the rotated_feature_align_forward operation is as follows:
     @verbatim
      input two arrays by 1 * 2 * 2 * 1 and 1 * 2 * 2 * 5
      --> input: [[[[1.0], [2.0]], [[2.0], [4.0]]]]
      --> bboxes: [[0.0, 2.0, 2.0, 1.0, 1.0]]

      param:
             spatial_scale: 1.0, points: 1

      output array by 1 * 2 * 2 * 1 -->
          output: [[[[5.0], [6.0]], [[6.0], [8.0]]]]
     @endverbatim
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/rotated_feature_align.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRotatedFeatureAlignForward(const mluOpHandle_t handle,
                                const mluOpTensorDescriptor_t input_desc,
                                const void *input,
                                const mluOpTensorDescriptor_t bboxes_desc,
                                const void *bboxes,
                                const float spatial_scale,
                                const int points,
                                const mluOpTensorDescriptor_t output_desc,
                                void *output);

// Group: RotatedFeatureAlign
/*!
 * @brief Computes the gradients of feature map \b bottom_input based on the inputs \b top_output
 * and \b bboxes to perform the backpropagation of ::mluOpRotatedFeatureAlignForward.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpRotatedFeatureAlignBackward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] top_output_desc
 * The descriptor of the tensor \b top_output. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] top_output
 * Pointer to the MLU memory that stores the top_output tensor.
 * @param[in] bboxes_desc
 * The descriptor of the tensor \b bboxes, which contains the dimension and layout of bboxes tensor. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[in] bboxes
 * Pointer to the MLU memory that stores the bboxes tensor.
 * @param[in] spatial_scale
 * A float value that is the scale factor of coordinates of bboxes.
 * @param[in] points
 * An integer value that is the number of sample points. Only 1 and 5 are supported. The default value is 1.
 * @param[in] bottom_input_desc
 * The descriptor of the tensor \b bottom_input, which contains the dimension and layout of bottom_input tensor.
 * @param[out] bottom_input
 * Pointer to the MLU memory that stores the bottom_input tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of all tensors should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - top_output tensor: half, float
 *   - bboxes tensor: half, float
 *   - bottom_input tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b top_output tensor, \b bboxes tensor, and \b bottom_input tensor are as follows:
 *   - top_output tensor: \p MLUOP_LAYOUT_NHWC
 *   - bboxes tensor: \p MLUOP_LAYOUT_ARRAY
 *   - bottom_input tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The top_output tensor and bottom_input tensor must be 4D.
 * - Sizes of the each dimension of top_output tensor and bottom_input tensor must be the same.
 * - The bboxes tensor must be 4D.
 * - The 4D information of bboxes tensor consists of [y, x, w, h, a].
 * - The 1D sizes of top_output tensor and bboxes tensor must be the same.
 * - The 2D sizes of top_output tensor and bboxes tensor must be the same.
 * - The 3D sizes of top_output tensor and bboxes tensor must be the same.
 * - The shape of \b top_output should be [batch_num, height, width, channels].
 * - The shape of \b bboxes should be [batch_num, height, width, 5].
 * - The shape of \b bottom_input should be [batch_num, height, width, channels].
 * - \b points is the number of sample points. Only 1 and 5 are supported. The default value is 1.
 * - The value of \b spatial_scale is larger than zero.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The inputs \b bboxes and \b spatial_scale with NaN or infinity are not supported.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/rotated_feature_align.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRotatedFeatureAlignBackward(const mluOpHandle_t handle,
                                 const mluOpTensorDescriptor_t top_output_desc,
                                 const void *top_output,
                                 const mluOpTensorDescriptor_t bboxes_desc,
                                 const void *bboxes,
                                 const float spatial_scale,
                                 const int points,
                                 const mluOpTensorDescriptor_t bottom_input_desc,
                                 void *bottom_input);

// Group: Sqrt
/*!
 * @brief Computes sqrt on input tensor \b x, and returns the results in the
 * output tensor \b y.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in the sqrt operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] prefer
 * The \b prefer modes defined in ::mluOpComputationPreference_t.
 * @param[in] x_desc
 * The descriptor of the tensor \b x. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] y_desc
 * The descriptor of the tensor \b y. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] y
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of input tensor and output tensor should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The input tensor and output tensor must have the same shape, and the input
 *   tensor must meet the following input data range:
 *   - float: [1e-10,1e10]
 *   - half: [1e-3,1e-2] & [1e-1,60000]
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://www.tensorflow.org/api_docs/python/tf/math/sqrt
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSqrt(mluOpHandle_t handle,
          const mluOpComputationPreference_t prefer,
          const mluOpTensorDescriptor_t x_desc,
          const void *x,
          const mluOpTensorDescriptor_t y_desc,
          void *y);

// Group: Sqrt

/*!
 * @brief Computes gradient of sqrt on input tensor \b y and \b diff_y, and
 *  returns the results in the output tensor \b diff_x.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in the sqrt backward operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] y_desc
 * The descriptor of the tensor \b y. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] y
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] dy_desc
 * The descriptor of the tensor \b dy. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] diff_y
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] dx_desc
 * The descriptor of the tensor \b dx. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] diff_x
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of input tensors and output tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensors: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The input tensor and output tensor must have the same shape, and the input
 *   tensor \b y must meet the following input data ranges:
 *   - float: [1e-10, 1e6]
 *   - half: [0.01, 500]
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://www.tensorflow.org/api_docs/python/tf/raw_ops/SqrtGrad
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSqrtBackward(mluOpHandle_t handle,
                  const mluOpTensorDescriptor_t y_desc,
                  const void *y,
                  const mluOpTensorDescriptor_t dy_desc,
                  const void *diff_y,
                  const mluOpTensorDescriptor_t dx_desc,
                  void *diff_x);

// Group: Voxelization
/*!
 * @brief Gets extra space size that is needed in voxelization operation.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices
 * and queues in the voxelization operation.
 * @param[in] points_desc
 * The descriptor of the tensor \b points. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] voxel_size_desc
 * The descriptor of the tensor \b voxel_size. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] coors_range_desc
 * The descriptor of the tensor \b coors_range. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] max_points
 * An integer value which is the maximum number of points contained
 * in a voxel.
 * @param[in] max_voxels
 * An integer value which is the maximum number of voxels this
 * function has created.
 * @param[in] NDim
 * An integer value which is the second dimension of coors.
 * @param[in] deterministic
 * A Boolean value whether to invoke the non-deterministic
 * version of hard-voxelization implementations. Currently,
 * non-deterministic mode is not supported.
 * @param[in] voxels_desc
 * The descriptor of the tensor \b voxels. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] coors_desc
 * The descriptor of the tensor \b coors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] num_points_per_voxel_desc
 * The descriptor of the tensor \b num_points_per_voxel. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] voxel_num_desc
 * The descriptor of the tensor \b voxel_num. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] size
 * A host pointer to the returned size of extra space in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */

mluOpStatus_t MLUOP_WIN_API
mluOpGetVoxelizationWorkspaceSize(mluOpHandle_t handle,
                                  const mluOpTensorDescriptor_t points_desc,
                                  const mluOpTensorDescriptor_t voxel_size_desc,
                                  const mluOpTensorDescriptor_t coors_range_desc,
                                  const int32_t max_points,
                                  const int32_t max_voxels,
                                  const int32_t NDim,
                                  const bool deterministic,
                                  const mluOpTensorDescriptor_t voxels_desc,
                                  const mluOpTensorDescriptor_t coors_desc,
                                  const mluOpTensorDescriptor_t num_points_per_voxel_desc,
                                  const mluOpTensorDescriptor_t voxel_num_desc,
                                  size_t *size);

// Group: Voxelization
/*!
 * @brief Generates voxelization of input tensor \b points. Output tensor
 * \b voxels contains points in voxels; \b coors is the voxel coordinates;
 * \b num_points_per_voxel is the number of points per voxel; \b voxel_num
 * is the number of voxels.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in the voxelization operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] points_desc
 * The descriptor of the tensor \b points. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] points
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] voxel_size_desc
 * The descriptor of the tensor \b voxel_size. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] voxel_size
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] coors_range_desc
 * The descriptor of the tensor \b coors_range. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] coors_range
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] max_points
 * An integer value which is the maximum number of points contained
 * in a voxel.
 * @param[in] max_voxels
 * An integer value which is the maximum number of voxels this
 * function creates.
 * @param[in] NDim
 * An integer value which is the second dimension of coors.
 * @param[in] deterministic
 * A Boolean value whether to invoke the non-deterministic
 * version of hard-voxelization implementations. Currently,
 * non-deterministic mode is not supported.
 * @param[in] workspace
 * Pointer to the MLU memory that stores the extra workspace.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in ::mluOpVoxelization.
 * @param[in] voxels_desc
 * The descriptor of the tensor \b voxels. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] voxels
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] coors_desc
 * The descriptor of the tensor \b coors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] coors
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] num_points_per_voxel_desc
 * The descriptor of the tensor \b num_points_per_voxel. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] num_points_per_voxel
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] voxel_num_desc
 * The descriptor of the tensor \b voxel_num. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] voxel_num
 * Pointer to the MLU memory that stores the input tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - points, voxel_size, coors_range, voxels: float
 *   - coors, num_points_per_voxel, voxel_num: int
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - max_points and max_voxels must be greater than or equal to 0.
 * - NDim must be equal to 3, which means 3D.
 * - The value of the deterministic mode must be True. Currently,
 *   the non-deterministic mode is not supported.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/voxelize.py
 */

mluOpStatus_t MLUOP_WIN_API
mluOpVoxelization(mluOpHandle_t handle,
                  const mluOpTensorDescriptor_t points_desc,
                  const void *points,
                  const mluOpTensorDescriptor_t voxel_size_desc,
                  const void *voxel_size,
                  const mluOpTensorDescriptor_t coors_range_desc,
                  const void *coors_range,
                  const int32_t max_points,
                  const int32_t max_voxels,
                  const int32_t NDim,
                  const bool deterministic,
                  void *workspace,
                  size_t workspace_size,
                  const mluOpTensorDescriptor_t voxels_desc,
                  void *voxels,
                  const mluOpTensorDescriptor_t coors_desc,
                  void *coors,
                  const mluOpTensorDescriptor_t num_points_per_voxel_desc,
                  void *num_points_per_voxel,
                  const mluOpTensorDescriptor_t voxel_num_desc,
                  void *voxel_num);

// Group: YoloBox
/*!
 * @brief Computes bounding box information from the backbone output of the
 * detected network.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in the yolo_box operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] x_desc
 * The descriptor of the tensor \b x. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] img_size_desc
 * The descriptor of the tensor \b img_size. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] img_size
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] anchors_desc
 * The descriptor of the tensor \b anchors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] anchors
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] class_num
 * The number of classes.
 * @param[in] conf_thresh
 * The detection boxes with the confidence score below the threshold should be ignored.
 * @param[in] downsample_ratio
 * The downsample ratio from network input to yolo_box operation input,
 * so 32, 16, 8 should be set for the first, second, and third into yolo_box operation.
 * @param[in] clip_bbox
 * If the value is True, the bounding box is clipped in img_size boundary.
 * @param[in] scale
 * The scaling coefficient of the coordinate of the center point of the decoded bounding box.
 * @param[in] iou_aware
 * If the value is True, the parameter iou_aware_factor is used.
 * @param[in] iou_aware_factor
 * The IOU aware factor, the default value is 0.5.
 * @param[in] boxes_desc
 * The descriptor of the tensor \b boxes. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] boxes
 * Pointer to the MLU memory that stores the output tensor.
 * @param[in] scores_desc
 * The descriptor of the tensor \b scores. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] scores
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of input and output tensors must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input x tensor: float
 *   - input img_size and anchors tensors: int
 *   - output tensors: float
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The first dimension of x tensor, img_size tensor, boxes tensor and scores
 *   tensor must be the same size.
 * - The second dimension (the channel dimension) of x tensor, C should be equal to S * (5 +
 *   class_num) if \b iou_aware is false, otherwise C should be equal to S * (6 + class_num),
 *   the value S is equal to the anchors tensor size divided by 2.
 * - The first dimension of anchors tensor should be larger than 0.
 * - The second dimension of img_size tensor must be equal to 2.
 * - The second dimension of boxes tensor must be equal to S.
 * - The second dimension of scores tensor must be equal to S.
 * - The third dimension of boxes tensor must be equal to 4.
 * - The third dimension of scores tensor must be equal to \b class_num.
 * - The fourth dimension of boxes tensor and scores tensor must be equal to the
 *   multiplication result of the third dimension and the fourth dimension of input x tensor.
 * - The \b class_num should be larger than 0. On MLU300 series, the value cannot be greater than 2558.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - When the \b iou_aware is true, the \b iou_aware_factor should be between [0, 1].
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/PaddlePaddle/Paddle/blob/release/2.3/python/paddle/vision/ops.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpYoloBox(mluOpHandle_t handle,
             const mluOpTensorDescriptor_t x_desc,
             const void *x,
             const mluOpTensorDescriptor_t img_size_desc,
             const void *img_size,
             const mluOpTensorDescriptor_t anchors_desc,
             const void *anchors,
             const int class_num,
             const float conf_thresh,
             const int downsample_ratio,
             const bool clip_bbox,
             const float scale,
             const bool iou_aware,
             const float iou_aware_factor,
             const mluOpTensorDescriptor_t boxes_desc,
             void *boxes,
             const mluOpTensorDescriptor_t scores_desc,
             void *scores);

// Group: VoxelPooling
/*!
 * @brief Adds the eigenvalues of all the channels on the same x and y coordinates,
 * and then pools them to all the channels in the bev 2D area on the corresponding coordinates.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in the voxel_pooling_forward operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] batch_size
 * The number of the batch size.
 * @param[in] num_points
 * The number of voxels.
 * @param[in] num_channels
 * The number of channels for voxel features.
 * @param[in] num_voxel_x
 * The number of voxels for dimX.
 * @param[in] num_voxel_y
 * The number of voxels for dimY.
 * @param[in] num_voxel_z
 * The number of voxels for dimZ.
 * @param[in] geom_xyz_desc
 * The descriptor of the tensor \b geom_xyz. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] geom_xyz
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] input_features_desc
 * The descriptor of the tensor \b input_features. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] input_features
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] output_features_desc
 * The descriptor of the tensor \b output_features. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] output_features
 * Pointer to the MLU memory that stores the output tensor.
 * @param[in] pos_memo_desc
 * The descriptor of the tensor \b pos_memo. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] pos_memo
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - geom_xyz tensor: int
 *   - input_features tensor: float
 *   - output_features tensor: float
 *   - pos_memo tensor: int
 *
 * @par Data Layout
 * - The supported data layouts of input and output tensors are as follows:
 *   - input tensor:
 *     - geom_xyz tensor: \p MLUOP_LAYOUT_ARRAY
 *     - input_features tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor:
 *     - output_features tensor: \p MLUOP_LAYOUT_ARRAY
 *     - pos_memo tensor: \p MLUOP_LAYOUT_ARRAY
 *
 * @par Scale Limitation
 * - The geom_xyz tensor, input_features tensor and pos_memo tensor must be 3D.
 * - The output_features tensor must be 4D.
 * - The shape of \b geom_xyz should be [batch_size, num_points, 3].
 * - The shape of \b input_features should be [batch_size, num_points, num_channels].
 * - The shape of \b output_features should be [batch_size, num_voxel_y, num_voxel_x, num_channels].
 * - The shape of \b pos_memo should be [batch_size, num_points, 3].
 * - The \b batch_size, \b num_points, \b num_channels, \b num_voxel_x and \b num_voxel_y should be larger than zero.
 *
 * @par API Dependency
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Note
 * - You need to set the initial value for the output \b pos_memo before calling the funtion, and initialize it to a
 *   negative number.
 *
 * @par Reference
 * -
 * https://github.com/Megvii-BaseDetection/BEVDepth/blob/main/bevdepth/ops/voxel_pooling/src/voxel_pooling_forward_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpVoxelPoolingForward(mluOpHandle_t handle,
                         const int batch_size,
                         const int num_points,
                         const int num_channels,
                         const int num_voxel_x,
                         const int num_voxel_y,
                         const int num_voxel_z,
                         const mluOpTensorDescriptor_t geom_xyz_desc,
                         const void *geom_xyz,
                         const mluOpTensorDescriptor_t input_features_desc,
                         const void *input_features,
                         const mluOpTensorDescriptor_t output_features_desc,
                         void *output_features,
                         const mluOpTensorDescriptor_t pos_memo_desc,
                         void *pos_memo);

// Group: BoxIouRotated
/*!
 * @brief Computes the intersection-over-union (Jaccard index, IOU) of rotated
 * bounding-boxes. If \b aligned is false, then calculates the IOU
 * between each rotated bounding-box of \b bbox1 and \b bbox2, otherwise calculates
 * the IOU between each aligned pair of rotated bounding-box of \b bbox1
 * and \b bbox2.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in the box_iou_rotated operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] mode
 * An integer value which decides to return a result of
 * IOU (Intersection Over Union) or IOF (Intersection Over Foreground).
 * The integer 0 represents IOU and 1 represents IOF.
 * @param[in] aligned
 * A Boolean value. If it is false, then calculate the IOU[i][j]
 * or IOF[i][j] between the row i of \b bbox1 and the row j of \b bbox2,
 * otherwise calculate the IOU[i] or IOFs[i] between the row i of \b bbox1
 * and the row i of \b bbox2. Significantly, the numbers of rows of \b bbox1
 * and \b bbox2 must be equal when \b aligned is true.
 * @param[in] bbox1_desc
 * The descriptor of the tensor \b bbox1 (rotated bounding-box).
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] bbox1
 * Pointer to the MLU memory that stores the input tensor \b bbox1.
 * It has shape (n, 5), indicating (x, y, w, h, theta) for each row.
 * Note that theta is in radian.
 * @param[in] bbox2_desc
 * The descriptor of the tensor \b bbox2 (rotated bounding-box).
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] bbox2
 * Pointer to the MLU memory that stores the input tensor \b bbox2.
 * It has shape (m, 5), indicating (x, y, w, h, theta) for each row.
 * Note that theta is in radian.
 * @param[in] ious_desc
 * The descriptor of the tensor \b ious. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] ious
 * IOU or IOF of input rotated bounding-boxes. Pointer to the MLU
 * memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - By the order of \b bbox1 - \b bbox2 - \b ious, the supported data types of
 *   \b bbox1, \b bbox2 and \b ious tensors are as follows:
 *   - float - float - float
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The number of dimensions of \b bbox1 and \b bbox2 tensors must be 2.
 * - The length of lowest dimension of \b bbox1 and \b bbox2 tensors must be 5.
 * - Both sets of boxes are expected to be in
 *   (x_center, y_center, width, height, angle) format.
 *   - bbox1 (Tensor): shape [n, 5] in (x, y, w, h, theta) format.
 *   - bbox2 (Tensor): shape [m, 5] in (x, y, w, h, theta) format.
 * - When aligned mode is true, for input \b bbox1 and \b bbox2 with n-rows,
 *   the output \b ious must be a 1D array with n-elements. When
 *   \b aligned is false, for input \b bbox1 with n-rows and \b bbox2 with
 *   m-rows, the output \b ious must be a 2D matrix with shape n*m.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - When finding the point with minimum y and minimum x in convex-hull-graham,
 *   BoxIouRotated performs min-pooling operation. If the input data of pooling
 *   contains NaN:
 *   - On MLU300 series:
 *     - If the last value in the kernel of the pooling is NaN, the \b output
 *       value is NaN. Otherwise, the \b output value is the minimum value after
 *       the last NaN.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/box_iou_rotated.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpBoxIouRotated(mluOpHandle_t handle,
                   const int mode,
                   const bool aligned,
                   const mluOpTensorDescriptor_t bbox1_desc,
                   const void *bbox1,
                   const mluOpTensorDescriptor_t bbox2_desc,
                   const void *bbox2,
                   const mluOpTensorDescriptor_t ious_desc,
                   void *ious);

// Group: NmsRotated
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra
 * workspace to optimize ::mluOpNmsRotated.
 *
 * The size of extra workspace is based on the given information of ::mluOpNmsRotated,
 * including the input tensor descriptors \b boxes_desc.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpNmsRotated. For detailed information, see ::mluOpHandle_t.
 * @param[in] boxes_desc
 * The descriptor of the tensor \b boxes, which contains the dimension and layout of the boxes tensor.
 * @param[out] workspace_size
 * Pointer to the MLU global memory that stores the returned size of the extra workspace in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Scale Limitation
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetNmsRotatedWorkspaceSize(mluOpHandle_t handle, const mluOpTensorDescriptor_t boxes_desc, size_t *workspace_size);

// Group: NmsRotated
/*!
 * @brief Computes the index of nms with IOU of rotated bounding boxes.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in ::mluOpNmsRotated. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in] iou_threshold
 * The threshold of IOU.
 * @param[in] boxes_desc
 * The descriptor of the tensor \b boxes (rotated bounding boxes).
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] boxes
 * Pointer to the MLU memory that stores the input tensor \b boxes.
 * It has shape (n, 5) or (n, 6), indicating (x, y, w, h, theta) or
 * (x, y, w, h, theta, label) for each row. Note that theta is in radian.
 * @param[in] scores_desc
 * The descriptor of the tensor \b scores (rotated bounding boxes).
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] scores
 * Pointer to the MLU memory that stores the input tensor \b scores.
 * It has shape (n), indicating score of each box in \b boxes.
 * @param[in] workspace
 * Pointer to the MLU memory that stores the extra workspace.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in the Nms operation.
 * @param[in] output_desc
 * The descriptor of the tensor output. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor, which indicates
 * the index of each output box.
 * @param[out] result_num
 * Pointer to the MLU memory that stores the number of output boxes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS,   ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - By the order of \b boxes - \b scores - \b output, the supported data types of
 *   \b boxes, \b scores, and \b output tensors are as follows:
 *   - float - float - int32
 *
 * @par Scale Limitation
 * - The number of dimensions of \b boxes tensors must be 2.
 * - The number of dimensions of \b scores and \b output tensors must be 1.
 * - The highest dimension of \b boxes and \b scores must be equal.
 * - The lowest dimension of \b boxes tensors must be 5 or 6.
 *
 * @par note
 * - The input \b scores with NAN/INF are not supported currently.
 *
 * @par API Dependency
 * - You need to call::mluOpGetNmsRotatedWorkspaceSize to allocate extra
 *   workspace for \b workspace.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/nms.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpNmsRotated(mluOpHandle_t handle,
                const float iou_threshold,
                const mluOpTensorDescriptor_t boxes_desc,
                const void *boxes,
                const mluOpTensorDescriptor_t scores_desc,
                const void *scores,
                void *workspace,
                size_t workspace_size,
                const mluOpTensorDescriptor_t output_desc,
                void *output,
                int32_t *result_num);

// Group: BboxOverlaps
/*!
 * @brief Computes the IOUs or IOFs between two sets of
 * bounding-boxes. If \b aligned is false, this operation calculates the IOU of each row between each bounding-box
 * of \b bbox1 and \b bbox2, otherwise, it calculates the IOU of the corresponding row between each aligned
 * pair of \b bbox1 and \b bbox2. For input placed in the order of <x1, y1, x2, y2>, (x1, y1) and (x2, y2)
 * respectively represents the top-left and bottom-right corner coordinates of bounding-box.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * bounding-box overlaps operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] mode
 * An integer value which decides to return a result IOU or IOF.
 * The integer 0 represents IOU and 1 represents IOF.
 * @param[in] aligned
 * A Boolean value. If it is false, this operation calculates the IOUs[i][j] or IOFs[i][j] between
 * the row i of \b bbox1 and the row j of \b bbox2, otherwise the IOU[i] or IOF[i] between
 * the row i of \b bbox1 and the row i of \b bbox2 are calculated. The number of rows of \b bbox1
 * and \b bbox2 must be equal if \b aligned is true.
 * @param[in] offset
 * An integer value determines whether to increase the length and the width of the bounding-box by 0 or 1
 * before calculating the area.
 * @param[in] bbox1_desc
 * The descriptor of the tensor \b bbox1. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] bbox1
 * Pointer to the MLU memory that stores the input tensor \b bbox1.
 * @param[in] bbox2_desc
 * The descriptor of the tensor \b bbox2. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] bbox2
 * Pointer to the MLU memory that stores the input tensor \b bbox2.
 * @param[in] ious_desc
 * The descriptor of the tensor \b ious. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] ious
 * IOU or IOF. Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - By the order of \b bbox1 - \b bbox2 - \b ious, the supported data types of
 *   \b bbox1, \b bbox2 and \b ious are as follows:
 *   - float - float - float
 *   - half  - half  - half
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The number of dimensions of \b bbox1 and \b bbox2 tensors must be 2
 * - The lowest dimension of input tensor must be 4
 *   - bbox1 (Tensor): shape [m, 4] in <x1, y1, x2, y2> format
 *   - bbox2 (Tensor): shape [n, 4] in <x1, y1, x2, y2> format
 * - Input with NaN is not supported currently. Also you need to exclude the input with (inf - inf) or (inf - (-inf)),
 *   where inf represents infinity (because the result is NaN, the actual impact is that the input has NaN)
 * - For input in type <x1, y1, x2, y2>, the coordinates must satisfy x2 > x1, y2 > y1,
 *   otherwise incorrect results will be obtained
 * - When aligned mode is true, for input \b bbox1 and \b bbox2 with n-rows, if n is zero, the output IOU
 *   must be a 2D matrix with shape n * 1, otherwise the output IOU must be a 1D
 *   array with n-elements. When aligned mode is false, for input \b bbox1 with n-rows and
 *   \b bbox2 with m-rows, the output IOU must be a 2D matrix with shape n * m.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The input tensor \b x should be in the following range to guarantee the accuracy of output:
 *   If bbox_overlaps works on (m)tp_2xx :
 *   - half : [-300, 100]
 *   - float : [-300, 100]
 *
 * @par Example
 * - The example of the bounding-box overlaps operation is as follows:
     @verbatim
      input array by 3 * 4, type is float -->
          input: bbox1 = [
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [32, 32, 38, 42],
          ]
      input array by 3 * 4, type is float -->
          input: bbox2 = [
            [0, 0, 10, 20],
            [0, 10, 10, 19],
            [10, 10, 20, 20],
          ]
      param:
        mode = 0
        aligned = False
        offset = 0


      output array by 3 * 3, type is float -->
          output: [[0.5000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 1.0000],
                   [0.0000, 0.0000, 0.0000]]
     @endverbatim
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/cuda/bbox_overlaps_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpBboxOverlaps(mluOpHandle_t handle,
                  const int mode,
                  const bool aligned,
                  const int offset,
                  const mluOpTensorDescriptor_t bbox1_desc,
                  const void *bbox1,
                  const mluOpTensorDescriptor_t bbox2_desc,
                  const void *bbox2,
                  const mluOpTensorDescriptor_t ious_desc,
                  void *ious);

// Group: ThreeInterpolate
/*!
 * @brief Computes weighted linear interpolation on 3 points by using
 * 3 indices in \b indices to select 3 points in \b features, uses the
 * 3 points to multiply with corresponding 3 weights in \b weights,
 * adds the 3 multiplication results to get one interpolation result,
 * for each batch repeats the above process N times on each channel,
 * and returns the results in the output tensor \b output.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in the three_interpolate_forward operation. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in] features_desc
 * The descriptor of the tensor \b features. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] features
 * Pointer to the MLU memory that stores the input features tensor. The features'
 * shape (B, C, M), B is batch size, C is channel size, M is the number of
 * elements in one input channel.
 * @param[in] indices_desc
 * The descriptor of the tensor \b indices. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] indices
 * Pointer to the MLU memory that stores the input indices tensor. The indices'
 * shape (B, N, 3), B is batch size, N is the number of elements in one output channel.
 * @param[in] weights_desc
 * The descriptor of the tensor \b weights. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] weights
 * Pointer to the MLU memory that stores the input weights tensor. The weights'
 * shape (B, N, 3), B is batch size, N is the number of elements in one output channel.
 * @param[in] output_desc
 * The descriptor of the output tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output features tensor. The
 * output's shape (B, C, N), B is batch size, C is channel size, N is number
 * of elements in one output channel.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *  ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of features tensor, weights tensor and output tensor should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - features tensor: half, float
 *   - indices tensor: int
 *   - weights tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b features tensor, \b indices tensor, \b weights tensor, and \b output tensor are
 *   as follows:
 *   - features tensor: \p MLUOP_LAYOUT_ARRAY
 *   - indices tensor: \p MLUOP_LAYOUT_ARRAY
 *   - weights tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor: \p MLUOP_LAYOUT_ARRAY
 *
 * @par Scale Limitation
 * - The dimension of \b features, \b indices, \b weights, and \b output
 *   should be equal to 3.
 * - The shape[0] of \b features, \b indices, \b weights, and \b output
 *   should be the same.
 * - The shape[1] of \b features and \b output should be the same.
 * - The shape[1] of \b indices, \b weights, and the shape[2] of \b output
 *   should be the same.
 * - The shape[2] of \b indices and \b weights should be equal to 3.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The value of \b indices must be in the range of [0, M-1], otherwise the output result
 *   is meaningless and the corresponding output will be set to 0.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/three_interpolate.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpThreeInterpolateForward(mluOpHandle_t handle,
                             const mluOpTensorDescriptor_t features_desc,
                             const void *features,
                             const mluOpTensorDescriptor_t indices_desc,
                             const void *indices,
                             const mluOpTensorDescriptor_t weights_desc,
                             const void *weights,
                             const mluOpTensorDescriptor_t output_desc,
                             void *output);

// Group: ThreeInterpolate
/*!
 * @brief Computes the gradients of feature map \b grad_features based on the
 * inputs \b grad_output, \b indices, and \b weights to perform the backpropagation
 * of ::mluOpThreeInterpolateForward.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in the three_interpolate_forward operation. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in] grad_output_desc
 * The descriptor of the tensor \b grad_output. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] grad_output
 * Pointer to the MLU memory that stores the gradients of output tensor. The grad_output's
 * shape (B, C, N), B is batch size, C is channel size, N is the number of
 * elements in one output channel.
 * @param[in] indices_desc
 * The descriptor of the tensor \b indices. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] indices
 * Pointer to the MLU memory that stores the input indices tensor. The indices'
 * shape (B, N, 3), B is batch size, N is the number of elements in one output channel.
 * @param[in] weights_desc
 * The descriptor of the tensor \b weights. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] weights
 * Pointer to the MLU memory that stores the input weights tensor. The weights'
 * shape (B, N, 3), B is batch size, N is the number of elements in one output channel.
 * @param[in] grad_features_desc
 * The descriptor of the tensor \b grad_features. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] grad_features
 * Pointer to the MLU memory that stores the gradients of features tensor. The
 * grad_features' shape (B, C, M), B is batch size, C is channel size, M is number
 * of elements in one input channel.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of grad_output tensor, weights tensor and grad_features tensor should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - grad_output tensor: half, float
 *   - indices tensor: int
 *   - weights tensor: half, float
 *   - grad_features tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b grad_output tensor, \b indices tensor, \b weights tensor, and \b grad_features
 * tensor are as follows:
 *   - grad_output tensor: \p MLUOP_LAYOUT_ARRAY
 *   - indices tensor: \p MLUOP_LAYOUT_ARRAY
 *   - weights tensor: \p MLUOP_LAYOUT_ARRAY
 *   - grad_features tensor: \p MLUOP_LAYOUT_ARRAY
 *
 * @par Scale Limitation
 * - The dimension of \b grad_output should be equal to 3.
 * - The dimension of \b indices should be equal to 3.
 * - The dimension of \b weights should be equal to 3.
 * - The dimension of \b grad_features should be equal to 3.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The value of \b indices must be in the range of [0, M-1], otherwise the output result
 *   is meaningless and the corresponding output will be set to 0.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/three_interpolate.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpThreeInterpolateBackward(mluOpHandle_t handle,
                              const mluOpTensorDescriptor_t grad_output_desc,
                              const void *grad_output,
                              const mluOpTensorDescriptor_t indices_desc,
                              const void *indices,
                              const mluOpTensorDescriptor_t weights_desc,
                              const void *weights,
                              const mluOpTensorDescriptor_t grad_features_desc,
                              void *grad_features);

// Group: BallQuery
/*!
 * @brief Takes the point's index in the \b new_xyz set as the center of the sphere,
 * uses \b min_radius and \b max_radius as the radius, and returns the \b idx of
 * the first \n nsample points in the \b xyz set in the spherical domain.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in the Ballquery operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] new_xyz_desc
 * The descriptor of the tensor \b new_xyz, which indicates the center of the ball.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] new_xyz
 * Pointer to the MLU memory that stores the new_xyz tensor.
 * The shape of new_xyz is [B, M, 3]. B: the batch size; M: the number of elements in a batch;
 * 3: the dimension of the 3D coordinate.
 * @param[in] xyz_desc
 * The descriptor of the tensor \b xyz, which means cloud points. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] xyz
 * Pointer to the MLU memory that stores the xyz tensor.
 * The shape of xyz is [B, N, 3]. B: the batch size; N: the number of elements in a batch;
 * 3: the dimension of the 3D coordinate.
 * @param[in] min_radius
 * A float value which is the minimum radius.
 * @param[in] max_radius
 * A float value which is the maximum radius.
 * @param[in] nsample
 * An integer value which is the selected points index.
 * @param[in] idx_desc
 * The descriptor of the tensor \b idx, which contains output indexes. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] idx
 * Pointer to the MLU memory that stores the xyz tensor.
 * The shape of idx is [B, M, nsample]. B: the batch size; M: the number of elements in a batch;
 * nsample: the number of points selected in the sphere.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of new_xyz and xyz must be the same.
 * - The supported data types of input and output tensors as follows:
 *   - new_xyz tensor: float or half
 *   - xyz tensor: float or half
 *   - idx tensor: int
 *
 * @par Data Layout
 * - The supported data layouts of \b new_xyz tensor, \b xyz tensor, and \b idx tensor are
 *   as follows:
 *   - new_xyz tensor: \p MLUOP_LAYOUT_ARRAY
 *   - xyz tensor: \p MLUOP_LAYOUT_ARRAY
 *   - idx tensor: \p MLUOP_LAYOUT_ARRAY
 *
 * @par Scale Limitation
 * - The new_xyz tensor, xyz tensor and idx tensor must be 3D.
 * - The first dimension of the new_xyz tensor, xyz tensor and the idx tensor must be the same.
 * - The second dimension of the new_xyz tensor and the idx tensor must be the same.
 * - The third dimension of the new_xyz tensor and the xyz tensor must be the same and equal to 3.
 * - The third dimension of idx tensor must be equal to nsample.
 * - The \b min_radius should be greater or equal to 0.
 * - The \b max_radius should be greater or equal to 0.
 * - The \b nsample should be greater or equal to 0.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - Take the point in new_xyz as the center of the sphere, there may be no points in xyz within the
 *   sphere with min_radius and max_radius as diameters. At this time, the value of the
 *   corresponding position in idx is the value when it is passed into the kernel. Generally, before
 *   passing idx into the kernel, initialize all the values in idx to 0 or other const values.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/ball_query.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpBallQuery(mluOpHandle_t handle,
               const mluOpTensorDescriptor_t new_xyz_desc,
               const void *new_xyz,
               const mluOpTensorDescriptor_t xyz_desc,
               const void *xyz,
               const float min_radius,
               const float max_radius,
               const int nsample,
               const mluOpTensorDescriptor_t idx_desc,
               void *idx);

// Group: FocalLossSigmoid
/*!
 * @brief Computes cross entropy loss with weighting factor and focusing factor
 * to reduce the filter of samples which are easy to classify.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues
 * in ::mluOpFocalLossSigmoidForward. For detailed information, see ::mluOpHandle_t.
 * @param[in] prefer
 * The algorithm used to compute the output. For detailed information,
 * see ::mluOpComputationPreference_t. Currently, only \p MLUOP_COMPUTATION_FAST and
 * \p MLUOP_COMPUTATION_HIGH_PRECISION are supported.
 * @param[in] reduction
 * The reduction mode used to compute the operation, see ::mluOpLossReduction_t.
 * Currently, only \p MLUOP_LOSS_REDUCTION_NONE is supported.
 * @param[in] input_desc
 * The descriptor of input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] target_desc
 * The descriptor of the tensor \b target . For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] target
 * Pointer to the MLU memory that stores the target tensor, which is the target
 * of input.
 * @param[in] weight_desc
 * The descriptor of the tensor \b weight . For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] weight
 * Pointer to the MLU memory that stores the weight tensor, which is the weight
 * value of input.
 * @param[in] alpha
 * A float value which is the weighting factor of the focal loss sigmoid forward.
 * @param[in] gamma
 * A float value which is the focusing factor of the focal loss sigmoid forward.
 * @param[in] output_desc
 * The descriptor of output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input tensors \b input, \b target, \b weight, and output
 *   tensor \b output are as follows:
 *   - input: half, float
 *   - target: int32
 *   - weight: half, float
 *   - output: half, float
 *
 * @par Data Layout
 * - The supported data layout of the input tensors and output tensors must be \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - The shape of \b input must be [N, C].
 * - The shape of \b input and \b output must be equal.
 * - The shape of \b target is [N] when the shape of \b input is [N, C].
 * - The shape of \b weight is [C] when the shape of \b input is [N, C].
 * - \b input value should be in the range of [-20, 20] when the data type of \b input is float.
 * - \b input value should be in the range of [-5, 5] when the data type of \b input is half.
 * - \b target value should be in the range of [0, C] when \b weight is NULL and the shape of
 *   \b input is [N, C].
 * - \b target value should be in the range of [0, C-1] when \b weight is not NULL and the shape
 *   of \b input is [N, C].
 * - \b gamma should be greater than or equal to 0.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - When input data or parameter contains NaN/infinity:
 *   - If \b input is infinity, but \b weight, \b alpha and \b gamma are finite value,
 *     then \b output is NaN or finite value.
 *   - If \b weight is positive infinity, but \b input, \b alpha and \b gamma are finite value,
 *     then \b output is NAN or positive infinity.
 *   - If \b weight is negative infinity, but \b input, \b alpha and \b gamma are finite value,
 *     then \b output is NAN or negative infinity.
 *   - If \b alpha is infinity and data type of \b input is float, but \b input, \b weight and
 *     \b gamma are finite value, then \b output is NAN or infinity.
 *   - If \b alpha is infinity and data type of \b input is half, but \b input, \b weight and
 *     \b gamma are finite value, then \b output is NAN or finite value.
 *   - If \b gamma is positive infinity, but \b input, \b weight and \b alpha are finite value,
 *     then \b output is NAN or 0.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar. Proceedings of the IEEE
 *   International Conference on Computer Vision(ICCV), 2017, oo.2980-2988
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/focal_loss.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpFocalLossSigmoidForward(mluOpHandle_t handle,
                             const mluOpComputationPreference_t prefer,
                             const mluOpLossReduction_t reduction,
                             const mluOpTensorDescriptor_t input_desc,
                             const void *input,
                             const mluOpTensorDescriptor_t target_desc,
                             const void *target,
                             const mluOpTensorDescriptor_t weight_desc,
                             const void *weight,
                             const float alpha,
                             const float gamma,
                             const mluOpTensorDescriptor_t output_desc,
                             void *output);

// Group: FocalLossSigmoid
/*!
 * @brief Computes the gradients of ::mluOpFocalLossSigmoidBackward with \b input tensor,
 * \b target tensor, \b weight tensor, and returns the results in
 * the \b grad_input tensor.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in ::mluOpFocalLossSigmoidBackward. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in] prefer
 * The algorithm used to compute the output.
 * For detailed information, see ::mluOpComputationPreference_t. Currently, only
 * \p MLUOP_COMPUTATION_HIGH_PRECISION is supported.
 * @param[in] reduction
 * The reduction mode used to compute the operation.
 * For detailed information, see ::mluOpLossReduction_t. Currently, only
 * \p MLUOP_LOSS_REDUCTION_NONE is supported.
 * @param[in] input_desc
 * The descriptor of input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] target_desc
 * The descriptor of the tensor \b target . For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] target
 * Pointer to the MLU memory that stores the target tensor, which is the target
 * of input.
 * @param[in] weight_desc
 * The descriptor of the tensor \b weight . For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] weight
 * Pointer to the MLU memory that stores the weight tensor, which is the weight
 * value of input.
 * @param[in] alpha
 * A float value which is the weighting factor of the focal loss sigmoid backward.
 * @param[in] gamma
 * A float value which is the focusing factor of the focal loss sigmoid backward.
 * @param[in] grad_input_desc
 * The descriptor of \b grad_input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] grad_input
 * Pointer to the MLU memory that stores the \b grad_input tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input tensor \b input, \b target, \b weight, and output
 *   tensor \b output are as follows:
 *   - input: float, half
 *   - target: int32
 *   - weight: float, half
 *   - grad_input: float, half
 *
 * @par Data Layout
 * - The supported data layout of the input tensors and output tensors must be \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - The shape of \b input must be [N, C].
 * - The shape of \b input and \b grad_input must be consistent.
 * - The shape of \b target is [N] when the shape of \b input is [N, C].
 * - The shape of \b weight is [C] when the shape of \b input is [N, C].
 * - \b input value should be in the range of [-5, 5] when the data type of \b input is half.
 * - \b target value should be in the range of [0, C] when \b weight is NULL and the shape of
 *   \b input is [N, C].
 * - \b target value should be in the range of [0, C-1] when \b weight is not NULL and the
 *   shape of \b input is [N, C].
 * - prefer only supports MLUOP_COMPUTATION_HIGH_PRECISION currently.
 * - reduction only supports \p MLUOP_LOSS_REDUCTION_NONE currently.
 * - The layout of \b input, \b target, \b weight and \b grad_input must be ARRAY.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - If the shape of \b input is set to [N, C], the length of C should be in the range of [0, 16339] when
 *   \b weight is NULL on MLU300 series. The length of C should be in the range of [0, 14848] when
 *   \b weight is not NULL on MLU300 series.
 * - If the shape of \b input is set to [N, C], the length of C should be in the range of [0, 9785] when
 *   \b weight is NULL on series higher than MLU300 series. The length of C should be in the range of [0, 8864] when
 *   \b weight is not NULL on series higher than MLU300 series.
 * - \b weight does not support positive infinity and negative infinity currently.
 * - \b gamma should be in the range of [0, 10000].
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/focal_loss.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpFocalLossSigmoidBackward(mluOpHandle_t handle,
                              const mluOpComputationPreference_t prefer,
                              const mluOpLossReduction_t reduction,
                              const mluOpTensorDescriptor_t input_desc,
                              const void *input,
                              const mluOpTensorDescriptor_t target_desc,
                              const void *target,
                              const mluOpTensorDescriptor_t weight_desc,
                              const void *weight,
                              const float alpha,
                              const float gamma,
                              const mluOpTensorDescriptor_t grad_input_desc,
                              void *grad_input);

// Group: MaskedIm2col
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra workspace to
 * optimize ::mluOpMaskedIm2colForward.
 *
 * The size of the extra workspace is based on the given information of ::mluOpMaskedIm2colForward,
 * including the input tensor descriptor \b feature_desc and \b data_col_desc. For more information about the workspace,
 * see "Cambricon MLU-OPS User Guide".
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context to manage MLU devices and queues in
 * ::mluOpGetMaskedIm2colForwardWorkspaceSize. For detailed information, see ::mluOpHandle_t.
 * @param[in] feature_desc
 * The descriptor of the tensor \b feature. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mask_h_idx_desc
 * The descriptor of the tensor \b mask_h_idx. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mask_w_idx_desc
 * The descriptor of the tensor \b mask_w_idx. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] data_col_desc
 * The descriptor of the tensor \b data_col. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] kernel_h
 * The height of mask.
 * @param[in] kernel_w
 * The width of mask.
 * @param[out] workspace_size
 * A host pointer to the returned size of the extra workspace in bytes that is used in
 * the ::mluOpMaskedIm2colForward.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitations
 * - None.
 *
 * @par API Dependency
 * - This function must be called before ::mluOpMaskedIm2colForward.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetMaskedIm2colForwardWorkspaceSize(mluOpHandle_t handle,
                                         const mluOpTensorDescriptor_t feature_desc,
                                         const mluOpTensorDescriptor_t mask_h_idx_desc,
                                         const mluOpTensorDescriptor_t mask_w_idx_desc,
                                         const int kernel_h,
                                         const int kernel_w,
                                         const mluOpTensorDescriptor_t data_col_desc,
                                         size_t *workspace_size);

// Group: MaskedIm2col
/*!
 * @brief Copies the data of the input tensor \b feature covered by mask to the output tensor \b data_col.
 * The area of mask that is out of \b feature is padded with 0. This function firstly generates mask coordinates
 * by combining \b mask_h_idx tensor and \b mask_w_idx tensor, then copies the \b feature data covered by mask.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * ::mluOpMaskedIm2colForward. For detailed information, see ::mluOpHandle_t.
 * @param[in] feature_desc
 * The descriptor of the tensor \b feature. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] feature
 * Pointer to the MLU memory that stores \b feature.
 * @param[in] mask_h_idx_desc
 * The descriptor of the tensor \b mask_h_idx. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mask_h_idx
 * Pointer to the MLU memory that stores the tensor \b mask_h_idx which contains
 * the coordinates of mask in height direction.
 * @param[in] mask_w_idx_desc
 * The descriptor of the tensor \b mask_w_idx. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mask_w_idx
 * Pointer to the MLU memory that stores the tensor \b mask_w_idx which contains
 * the coordinates of mask in width direction.
 * @param[in] kernel_h
 * The height of mask.
 * @param[in] kernel_w
 * The width of mask.
 * @param[in] pad_h
 * The height of padding.
 * @param[in] pad_w
 * The width of padding.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the ::mluOpMaskedIm2colForward.
 * For more information about workspace, see "Cambricon MLU-OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes.
 * @param[in] data_col_desc
 * The descriptor of the tensor \b data_col. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] data_col
 * Pointer to the MLU memory that stores the tensor \b data_col that is the data copied from the tensor \b feature
 * based on mask coordinates. The mask area out of the tensor \b feature is padded with 0.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - This function supports the following data types for tensor \b feature, tensor \b mask_h_idx,
 *   tensor \b mask_w_idx, and tensor \b data_col.
 *   Data types of tensor \b feature and tensor \b data_col must be the same.
 *   - feature tensor: half, float.
 *   - mask_h_idx tensor: int32_t.
 *   - mask_w_idx tensor: int32_t.
 *   - data_col tensor: half, float.
 *
 * @par Data Layout
 * - The supported data layouts of \b feature, \b mask_h_idx, \b mask_w_idx, and \b data_col are as follows:
 *   - feature tensor: \p MLUOP_LAYOUT_NCHW.
 *   - mask_h_idx tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - mask_w_idx tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - data_col tensor: \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - The tensor \b mask_h_idx must be 1D.
 * - The tensor \b mask_w_idx must be 1D.
 * - The tensor \b data_col must be 2D.
 * - The sizes of the highest dimension of tensor \b feature must be 1.
 * - The sizes of the lowest dimension of tensor \b data_col, the element number of tensor \b mask_h_idx, and
 *   the element number of tensor \b mask_w_idx must be the same.
 * - When the element number of tensor \b feature equals zero, this function will return \b MLUOP_STATUS_BAD_PARAM.
 * - When size of the highest dimension of tensor \b data_col equals zero, this function will return
 *   \b MLUOP_STATUS_BAD_PARAM.
 *
 * @par API Dependency
 * - Before calling this function you need to call ::mluOpGetMaskedIm2colForwardWorkspaceSize
 *   to get the extra space size needed in ::mluOpMaskedIm2colForward.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - The example of the ::mluOpMaskedIm2colForward is as follows:
     @verbatim
     input three arrays by 1 * 1 * 3 * 3, 2 and 2
     --> feature: [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]
     --> mask_h_idx: [-1, 1]
     --> mask_w_idx: [-1, 1]

     param:
            kernel_h: 2, kernel_w: 2, pad_h: 1, pad_w: 1

     output array by 4 * 2 -->
         output: [[0.0, 1.0], [0.0, 2.0], [0.0, 4.0], [0.0, 5.0]]
     @endverbatim
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/cuda/masked_conv2d_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpMaskedIm2colForward(mluOpHandle_t handle,
                         const mluOpTensorDescriptor_t feature_desc,
                         const void *feature,
                         const mluOpTensorDescriptor_t mask_h_idx_desc,
                         const void *mask_h_idx,
                         const mluOpTensorDescriptor_t mask_w_idx_desc,
                         const void *mask_w_idx,
                         const int kernel_h,
                         const int kernel_w,
                         const int pad_h,
                         const int pad_w,
                         void *workspace,
                         const size_t workspace_size,
                         const mluOpTensorDescriptor_t data_col_desc,
                         void *data_col);

// Group: MoeDispatch
/*!
 * @brief Calculates the inverse gradient of \b input tensor, and returns the results in the output
 * tensor \b grad_input.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpMoeDispatchBackwardData. For detailed information, see ::mluOpHandle_t.
 * @param[in] gates_desc
 * The descriptor of the tensor \b gates, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] gates
 * Pointer to the MLU memory that stores the \b gates tensor.
 * @param[in] indices_desc
 * The descriptor of the tensor \b indices, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] indices
 * Pointer to the MLU memory that stores the \b indices tensor.
 * @param[in] locations_desc
 * The descriptor of the tensor \b locations, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] locations
 * Pointer to the MLU memory that stores the \b locations tensor.
 * @param[in] dispatch_desc
 * The descriptor of the tensor \b dispatch, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] dispatch
 * Pointer to the MLU memory that stores the \b dispatch tensor.
 * @param[in] samples
 * The number of elements in the \b gates tensor, the \b indices tensor, and the \b locations tensor.
 * @param[in] capacity
 * The maximum number of inputs that experts can process.
 * @param[in] hidden
 * The vector length of a single token.
 * @param[in] num_experts
 * The number of experts.
 * @param[in] grad_input_desc
 * The descriptor of the tensor \b grad_input, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] grad_input
 * Pointer to the MLU memory that stores the \b grad_input tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - gates tensor: float
 *   - indices tensor: int32
 *   - locations tensor: int32
 *   - dispatch tensor: float
 *   - grad_input tensor: float
 *
 * @par Data Layout
 * - The supported layout of the input tensors and output tensors must be \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - The first dimension of \b gates tensor, \b indices tensor, \b locations tensor, and \b grad_input
 *   tensor must be the same size and equal to \b samples .
 * - The second dimension of \b grad_input tensor and \b dispatch tensor must be equal to \b hidden .
 * - The first dimension of \b dispatch tensor must be equal to the multiplication result of
 *   the \b capacity and \b num_experts.
 * - The value of the input parameters \b samples, \b capacity , \b hidden , and \b num_experts
 *   must be greater than or equal to 0.
 * - The value range of the input parameter \b indices tensor must be greater than or equal to 0 and less than
 *   \b num_experts.
 * - The value range of the input parameter \b locations tensor must be greater than or equal to 0 and less than
 *   \b capacity.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - This function is only supported on MLU300 series or above platforms.
 * - The parameter \b samples, \b capacity , \b hidden , and \b num_experts should not be negative.
 *
 * @par Example
 * - The example of the function is as follows:
     @verbatim
      Dimension of gates tensor:  [samples]
      Dimension of indices tensor:  [samples]
      Dimension of locations tensor:  [samples]
      Dimension of dispatch tensor: [num_experts * capacity, hidden]
      Dimension of grad_input tensor: [samples, hidden]
     @endverbatim
 *
 * @par Reference
 * - https://github.com/microsoft/tutel/blob/v0.2.0/tutel/jit_kernels/sparse.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpMoeDispatchBackwardData(mluOpHandle_t handle,
                             const mluOpTensorDescriptor_t gates_desc,
                             const void *gates,
                             const mluOpTensorDescriptor_t indices_desc,
                             const void *indices,
                             const mluOpTensorDescriptor_t locations_desc,
                             const void *locations,
                             const mluOpTensorDescriptor_t dispatch_desc,
                             const void *dispatch,
                             const int samples,
                             const int capacity,
                             const int hidden,
                             const int num_experts,
                             const mluOpTensorDescriptor_t grad_input_desc,
                             void *grad_input);

// Group: MsDeformAttn
/*!
 * @brief Computes the gradient of the input tensors of ::mluOpMsDeformAttnForward.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues
 * in the ms_deform_attn_backward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] value_desc
 * The descriptor of the \b value tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] value
 * Pointer to the MLU memory that stores the input value.
 * @param[in] spatial_shapes_desc
 * The descriptor of the \b spatial_shapes tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] spatial_shapes
 * Pointer to the MLU memory that stores the shapes of multi-scale feature maps.
 * @param[in] level_start_index_desc
 * The descriptor of the \b level_start_index tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] level_start_index
 * Pointer to the MLU memory that stores the feature maps offset in \b value.
 * @param[in] sampling_loc_desc
 * The descriptor of the \b sampling_loc tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] sampling_loc
 * Pointer to the MLU memory that stores the normalized coordinates of sample points.
 * @param[in] attn_weight_desc
 * The descriptor of the \b attn_weight tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] attn_weight
 * Pointer to the MLU memory that stores the attention weight.
 * @param[in] grad_output_desc
 * The descriptor of the \b grad_output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] grad_output
 * Pointer to the MLU memory that stores the output gradient
 * of ::mluOpMsDeformAttnForward.
 * @param[in] im2col_step
 * The value of im2col_step.
 * @param[in] grad_value_desc
 * The descriptor of the \b grad_value tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] grad_value
 * Pointer to the MLU memory that stores the gradient of \b value tensor.
 * @param[in] grad_sampling_loc_desc
 * The descriptor of the \b grad_sampling_loc tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] grad_sampling_loc
 * Pointer to the MLU memory that stores the gradient of \b sampling_loc tensor.
 * @param[in] grad_attn_weight_desc
 * The descriptor of the \b grad_attn_weight tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] grad_attn_weight
 * Pointer to the MLU memory that stores the gradient of \b attn_weight tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - \b value: float
 *   - \b spatial_shapes: int32_t
 *   - \b level_start_index: int32_t
 *   - \b sampling_loc: float
 *   - \b attn_weight: float
 *   - \b grad_output: float
 *   - \b grad_value: float
 *   - \b grad_sampling_loc: float
 *   - \b grad_attn_weight: float
 *
 * @par Data Layout
 * - The supported layout of input and output tensors must be \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The input \b sampling_loc that contains NaN or infinity is not supported.
 * - The \b value, \b sampling_loc, \b with attn_weight and \b grad_output contain NaN or infinity are not
 *   supported on series higher than MLU300 series currently.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh
 */

mluOpStatus_t MLUOP_WIN_API
mluOpMsDeformAttnBackward(mluOpHandle_t handle,
                          const mluOpTensorDescriptor_t value_desc,
                          const void *value,
                          const mluOpTensorDescriptor_t spatial_shapes_desc,
                          const void *spatial_shapes,
                          const mluOpTensorDescriptor_t level_start_index_desc,
                          const void *level_start_index,
                          const mluOpTensorDescriptor_t sampling_loc_desc,
                          const void *sampling_loc,
                          const mluOpTensorDescriptor_t attn_weight_desc,
                          const void *attn_weight,
                          const mluOpTensorDescriptor_t grad_output_desc,
                          const void *grad_output,
                          const int32_t im2col_step,
                          const mluOpTensorDescriptor_t grad_value_desc,
                          void *grad_value,
                          const mluOpTensorDescriptor_t grad_sampling_loc_desc,
                          void *grad_sampling_loc,
                          const mluOpTensorDescriptor_t grad_attn_weight_desc,
                          void *grad_attn_weight);

// Group: MutualInformation
/*!
 * @brief Returns the size of the MLU memory as an extra workspace
 * to optimize ::mluOpMutualInformationBackward.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context for MLU devices and queues management in
 * ::mluOpMutualInformationBackward. For detailed information, see ::mluOpHandle_t.
 * @param[in] px_desc
 * The descriptor of the tensor \b px containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] py_desc
 * The descriptor of the tensor \b py containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] opt_boundary_desc
 * The descriptor of the tensor \b opt_boundary containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] p_desc
 * The descriptor of the tensor \b p containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] ans_grad_desc
 * The descriptor of the tensor \b ans_grad containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] overwrite_ans_grad
 * A Boolean value indicating whether to overwrite \b ans_grad.
 * @param[out] workspace_size
 * Pointer to the MLU memory that stores the returned size of the extra workspace in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitations
 * - None.
 *
 * @par API Dependency
 * - The allocated extra workspace must be passed to ::mluOpMutualInformationBackward.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetMutualInformationBackwardWorkspaceSize(mluOpHandle_t handle,
                                               const mluOpTensorDescriptor_t px_desc,
                                               const mluOpTensorDescriptor_t py_desc,
                                               const mluOpTensorDescriptor_t opt_boundary_desc,
                                               const mluOpTensorDescriptor_t p_desc,
                                               const mluOpTensorDescriptor_t ans_grad_desc,
                                               const bool overwrite_ans_grad,
                                               size_t *workspace_size);

// Group: MutualInformation
/*!
 * @brief Computes the gradients of tensor \b px and tensor \b py.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context for MLU devices and queues management in the
 * mutual_information_backward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] px_desc
 * The descriptor of the tensor \b px containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] px
 * Pointer to the MLU memory that stores the tensor \b px.
 * @param[in] py_desc
 * The descriptor of the tensor \b py containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] py
 * Pointer to the MLU memory that stores the tensor \b py.
 * @param[in] opt_boundary_desc
 * The descriptor of the input tensor \b opt_boundary containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] opt_boundary
 * Pointer to the MLU memory that stores the \b opt_boundary tensor.
 * @param[in] p_desc
 * The descriptor of the tensor \b p containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] p
 * Pointer to the MLU memory that stores the tensor \b p.
 * @param[in] ans_grad_desc
 * The descriptor of the tensor \b ans_grad containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] ans_grad
 * Pointer to the MLU memory that stores the tensor \b ans_grad.
 * @param[in] overwrite_ans_grad
 * A Boolean value indicating whether to overwrite \b ans_grad.
 * @param[in] workspace
 * Pointer to the MLU memory as an extra workspace for the mutual_information_backward operation.
 * For more information about the workspace, see "Cambricon MLU-OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes.
 * @param[in] px_grad_desc
 * The descriptor of the tensor \b px_grad containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] px_grad
 * Pointer to the MLU memory that stores the tensor \b px_grad.
 * @param[in] py_grad_desc
 * The descriptor of the tensor \b py_grad containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] py_grad
 * Pointer to the MLU memory that stores the tensor \b py_grad.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - tensor ``px`` : ``float``;
 *   - tensor ``py`` : ``float``;
 *   - tensor ``opt_boundary`` : ``int64``;
 *   - tensor ``p`` : ``float``;
 *   - tensor ``ans_grad`` : ``float``;
 *   - tensor ``px_grad`` : ``float``;
 *   - tensor ``py_grad`` : ``float``;
 *
 * @par Data Layout
 * - The supported layout of input and output tensors must be \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - The shape of \b px is 3D([B, S, T + 1]), where B is the batch size, S is the length of symbols
 *   and T is the length of the input sequence.
 * - The shape of \b py is 3D([B, S + 1, T]).
 * - The shape of \b opt_boundary is 2D([B, 4]), where each row contains
     [begin_symbol, begin_frame, end_symbol, end_frame] with
     "0 <= begin_symbol <= end_symbol <= S" and "0 <= begin_frame <= end_frame <= T".
     If \b opt_boundary is NULL, it will be treated as [0, 0, S, T].
 * - The shape of \b p is 3D([B, S + 1, T + 1]).
 * - The shape of \b ans_grad is 1D([B]).
 * - The shape of \b px and \b px_grad must be the same.
 * - The shape of \b py and \b py_grad must be the same.
 * - The size of each tensor must be in the range of [0, 2^31].
 *
 * @par API Dependency
 * - Before calling this function to perform ::mluOpMutualInformationBackward, you need to get
 *   the size of the workspace by ::mluOpGetMutualInformationBackwardWorkspaceSize.
 *
 * @par Note
 * - This function is only supported on MLU300 series or above platforms.
 * - If \b overwrite_ans_grad is true, \b ans_grad will be overwritten.
 *   If the computation worked correctly, the overwritten value should be the same as the original ans_grad.
 * - If B is zero, or S and T are both zero, ::MLUOP_STATUS_SUCCESS is returned without
 *   any changes to \b ans_grad, \b px_grad and \b py_grad tensor.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/k2-fsa/k2/blob/master/k2/python/csrc/torch/mutual_information_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpMutualInformationBackward(mluOpHandle_t handle,
                               const mluOpTensorDescriptor_t px_desc,
                               const void *px,
                               const mluOpTensorDescriptor_t py_desc,
                               const void *py,
                               const mluOpTensorDescriptor_t opt_boundary_desc,
                               const void *opt_boundary,
                               const mluOpTensorDescriptor_t p_desc,
                               const void *p,
                               const mluOpTensorDescriptor_t ans_grad_desc,
                               void *ans_grad,
                               const bool overwrite_ans_grad,
                               void *workspace,
                               const size_t workspace_size,
                               const mluOpTensorDescriptor_t px_grad_desc,
                               void *px_grad,
                               const mluOpTensorDescriptor_t py_grad_desc,
                               void *py_grad);

// Group: MutualInformation
/*!
 * @brief Returns the size of the MLU memory as an extra workspace
 * to optimize ::mluOpMutualInformationForward.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context for MLU devices and queues management in
 * ::mluOpMutualInformationForward. For detailed information, see ::mluOpHandle_t.
 * @param[in] px_desc
 * The descriptor of the tensor \b px containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] py_desc
 * The descriptor of the tensor \b py containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] opt_boundary_desc
 * The descriptor of the tensor \b opt_boundary containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] p_desc
 * The descriptor of the tensor \b p containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] ans_desc
 * The descriptor of the tensor \b ans containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] workspace_size
 * Pointer to the MLU memory that stores the returned size of the extra workspace in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitations
 * - None.
 *
 * @par API Dependency
 * - The allocated extra workspace must be passed to ::mluOpMutualInformationForward.
 *
 * @par Note
 * - Currently the \b workspace_size always returns 0.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetMutualInformationForwardWorkspaceSize(mluOpHandle_t handle,
                                              const mluOpTensorDescriptor_t px_desc,
                                              const mluOpTensorDescriptor_t py_desc,
                                              const mluOpTensorDescriptor_t opt_boundary_desc,
                                              const mluOpTensorDescriptor_t p_desc,
                                              const mluOpTensorDescriptor_t ans_desc,
                                              size_t *workspace_size);

// Group: MutualInformation
/*!
 * @brief Computes mutual information between tensor \b px and tensor \b py.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context for MLU devices and queues management in the
 * mutual_information_forward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] px_desc
 * The descriptor of the tensor \b px containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] px
 * Pointer to the MLU memory that stores the tensor \b px.
 * @param[in] py_desc
 * The descriptor of the tensor \b py containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] py
 * Pointer to the MLU memory that stores the tensor \b py.
 * @param[in] opt_boundary_desc
 * The descriptor of the input tensor \b opt_boundary containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] opt_boundary
 * Pointer to the MLU memory that stores the \b opt_boundary tensor.
 * @param[in] p_desc
 * The descriptor of the tensor \b p containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] p
 * Pointer to the MLU memory that stores the tensor \b p.
 * @param[in] workspace
 * Pointer to the MLU memory as an extra workspace for the mutual_information_forward operation.
 * For more information about the workspace, see "Cambricon MLU-OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes.
 * @param[in] ans_desc
 * The descriptor of the tensor \b ans containing dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] ans
 * Pointer to the MLU memory that stores the tensor \b ans.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - tensor ``px`` : ``float``;
 *   - tensor ``py`` : ``float``;
 *   - tensor ``opt_boundary`` : ``int64``;
 *   - tensor ``p`` : ``float``;
 *   - tensor ``ans`` : ``float``;
 *
 * @par Data Layout
 * - The supported layout of input and output tensors must be \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - The shape of \b px is 3D([B, S, T + 1]), where B is the batch size, S is the length of symbols
 *   and T is the length of the input sequence.
 * - The shape of \b py is 3D([B, S + 1, T]).
 * - The shape of \b opt_boundary is 2D([B, 4]), where each row contains
     [begin_symbol, begin_frame, end_symbol, end_frame] with
     "0 <= begin_symbol <= end_symbol <= S" and "0 <= begin_frame <= end_frame <= T".
     If \b opt_boundary is NULL, it will be treated as [0, 0, S, T].
 * - The shape of \b p is 3D([B, S + 1, T + 1]).
 * - The shape of \b ans is 1D([B]).
 * - The size of each tensor must be in the range of [0, 2^31].
 *
 * @par API Dependency
 * - Before calling this function to perform ::mluOpMutualInformationForward, you need to get
 *   the size of the workspace by ::mluOpGetMutualInformationForwardWorkspaceSize.
 *
 * @par Note
 * - This function is only supported on MLU300 series or above platforms.
 * - If B is zero, ::MLUOP_STATUS_SUCCESS is returned without any changes to tensor \b p and tensor \b ans.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/k2-fsa/k2/blob/master/k2/python/csrc/torch/mutual_information_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpMutualInformationForward(mluOpHandle_t handle,
                              const mluOpTensorDescriptor_t px_desc,
                              const void *px,
                              const mluOpTensorDescriptor_t py_desc,
                              const void *py,
                              const mluOpTensorDescriptor_t opt_boundary_desc,
                              const void *opt_boundary,
                              const mluOpTensorDescriptor_t p_desc,
                              void *p,
                              void *workspace,
                              const size_t workspace_size,
                              const mluOpTensorDescriptor_t ans_desc,
                              void *ans);

// Group: Deprecated APIs
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra
 * workspace to optimize ::mluOpRoiAwarePool3dForward.
 *
 * The size of extra workspace is based on the given information of ::mluOpRoiAwarePool3dForward,
 * including the input tensor descriptors \b pts_desc.
 *
 * @par Deprecated
 * - ::mluOpGetRoiawarePool3dForwardWorkspaceSize is deprecated and will be removed in the future
 *   release. It is recommended to use ::mluOpGetRoiAwarePool3dForwardWorkspaceSize instead.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpRoiAwarePool3dForward. For detailed information, see ::mluOpHandle_t.
 * @param[in] rois_desc
 * The descriptor of the tensor \b rois, which contains the dimension and layout of the rois tensor.
 * @param[in] pts_desc
 * The descriptor of the tensor \b pts, which contains the dimension and layout of the pts tensor.
 * @param[in] pts_feature_desc
 * The descriptor of the tensor \b pts_feature, which contains the dimension and layout of the pts tensor.
 * @param[out] workspace_size
 * Pointer to the returned size of the extra workspace in bytes that is used in the
 * ::mluOpRoiAwarePool3dForward operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetRoiawarePool3dForwardWorkspaceSize(mluOpHandle_t handle,
                                           const mluOpTensorDescriptor_t rois_desc,
                                           const mluOpTensorDescriptor_t pts_desc,
                                           const mluOpTensorDescriptor_t pts_feature_desc,
                                           size_t *workspace_size);

// Group: RoiAwarePool3d
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra
 * workspace to optimize ::mluOpRoiAwarePool3dForward.
 *
 * The size of extra workspace is based on the given information of ::mluOpRoiAwarePool3dForward,
 * including the input tensor descriptors \b pts_desc.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpRoiAwarePool3dForward. For detailed information, see ::mluOpHandle_t.
 * @param[in] rois_desc
 * The descriptor of the tensor \b rois, which contains the dimension and layout of the rois tensor.
 * @param[in] pts_desc
 * The descriptor of the tensor \b pts, which contains the dimension and layout of the pts tensor.
 * @param[in] pts_feature_desc
 * The descriptor of the tensor \b pts_feature, which contains the dimension and layout of the pts tensor.
 * @param[out] workspace_size
 * Pointer to the returned size of the extra workspace in bytes that is used in the
 * ::mluOpRoiAwarePool3dForward operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetRoiAwarePool3dForwardWorkspaceSize(mluOpHandle_t handle,
                                           const mluOpTensorDescriptor_t rois_desc,
                                           const mluOpTensorDescriptor_t pts_desc,
                                           const mluOpTensorDescriptor_t pts_feature_desc,
                                           size_t *workspace_size);

// Group: Deprecated APIs
/*!
 * @brief Returns \b argmax, \b pts_idx_of_voxels and \b pooled_features calculated by
 * this operator.
 *
 * The operator determines the points in each box based on input coordinates. The collection
 * of points in boxes are named as voxels and recorded as \b pts_idx_of_voxels. The operator
 * also performs max pooling or average pooling on the voxels and results in \b argmax
 * and \b pooled_features.
 *
 * @par Deprecated
 * - ::mluOpRoiawarePool3dForward is deprecated and will be removed in the future
 *   release. It is recommended to use ::mluOpRoiAwarePool3dForward instead.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpRoiAwarePool3dForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] pool_method
 * Pooling method of Roiaware, 0 is 'maxpool', 1 is 'avgpool'. The default value is 0.
 * @param[in] boxes_num
 * An integer value which is the number of the rois.
 * @param[in] pts_num
 * An integer value which is the number of the pts.
 * @param[in] channels
 * An integer value which is the number of the pts feature of channels.
 * @param[in] rois_desc
 * The descriptor of the tensor \b rois, which contains the dimension and layout of the rois tensor.
 * @param[in] rois
 * Pointer to the MLU memory that stores the rois tensor.
 * @param[in] pts_desc
 * The descriptor of the tensor \b pts, which contains the dimension and layout of the pts tensor.
 * @param[in] pts
 * Pointer to the MLU memory that stores the pts tensor.
 * @param[in] pts_feature_desc
 * The descriptor of the tensor \b pts_feature, which contains the dimension and layout of the pts_feature tensor.
 * @param[in] pts_feature
 * Pointer to the MLU memory that stores the pts_feature tensor.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the
 * ::mluOpRoiAwarePool3dForward operation.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in
 * ::mluOpRoiAwarePool3dForward. You can get the size of the workspace with
 * ::mluOpGetRoiAwarePool3dForwardWorkspaceSize.
 * @param[in] max_pts_each_voxel
 * The maximum number of points per each voxel. An integer value which is the dimension of the pts_idx_of_voxels.
 * @param[in] out_x
 * An integer value which is the dimension of the pooled_features.
 * @param[in] out_y
 * An integer value which is the dimension of the pooled_features.
 * @param[in] out_z
 * An integer value which is the dimension of the pooled_features.
 * @param[in] argmax_desc
 * The descriptor of the tensor \b argmax, which contains the dimension and layout of the argmax tensor.
 * @param[out] argmax
 * Pointer to the MLU memory that stores the argmax tensor.
 * @param[in] pts_idx_of_voxels_desc
 * The descriptor of the tensor \b pts_idx_of_voxels, which contains the dimension and layout of the pts_idx_of_voxels
 * tensor.
 * @param[out] pts_idx_of_voxels
 * Pointer to the MLU memory that stores the pts_idx_of_voxels tensor.
 * @param[in] pooled_features_desc
 * The descriptor of the tensor \b pooled_features, which contains the dimension and layout of the pooled_features
 * tensor.
 * @param[out] pooled_features
 * Pointer to the MLU memory that stores the pooled_features tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - rois tensor: half, float
 *   - pts tensor: half, float
 *   - pts_feature tensor: half, float
 *   - argmax tensor: int32
 *   - pts_idx_of_voxels tensor: int32
 *   - pooled_features tensor: half, float
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The value of \b boxes_num should be less than 65536.
 * - The value of \b channels should be less than 65536.
 * - The Product of \b boxes_num and \b pts_num should be less than 2G.
 * - When the data type is floating point, the value of \b max_pts_each_voxel cannot be
 *   greater than 2976, and when the data type is half, it cannot be greater than 2944.
 * - The shape of \b rois should be [boxes_num, 7].
 * - The shape of \b pts should be [pts_num, 3].
 * - The shape of \b pts_feature should be [pts_num, channels].
 * - The shape of \b argmax should be [boxes_num, out_x, out_y, out_z, channels].
 * - The shape of \b pts_idx_of_voxels should be [boxes_num, out_x, out_y, out_z, max_pts_each_voxel].
 * - The shape of \b pooled_features should be [boxes_num, out_x, out_y, out_z, channels].
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The inputs \b rois and \b pts with NaN or infinity are not supported on MLU300 series.
 * - The input \b pts_feature with NaN are not supported on MLU300 series.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/tree/master/mmcv/ops/roiaware_pool3d.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRoiawarePool3dForward(mluOpHandle_t handle,
                           const int pool_method,
                           const int boxes_num,
                           const int pts_num,
                           const int channels,
                           const mluOpTensorDescriptor_t rois_desc,
                           const void *rois,
                           const mluOpTensorDescriptor_t pts_desc,
                           const void *pts,
                           const mluOpTensorDescriptor_t pts_feature_desc,
                           const void *pts_feature,
                           void *workspace,
                           size_t workspace_size,
                           const int max_pts_each_voxel,
                           const int out_x,
                           const int out_y,
                           const int out_z,
                           const mluOpTensorDescriptor_t argmax_desc,
                           void *argmax,
                           const mluOpTensorDescriptor_t pts_idx_of_voxels_desc,
                           void *pts_idx_of_voxels,
                           const mluOpTensorDescriptor_t pooled_features_desc,
                           void *pooled_features);

// Group: RoiAwarePool3d
/*!
 * @brief Returns \b argmax, \b pts_idx_of_voxels and \b pooled_features calculated by
 * this operator.
 *
 * The operator determines the points in each box based on input coordinates. The collection
 * of points in boxes are named as voxels and recorded as \b pts_idx_of_voxels. The operator
 * also performs max pooling or average pooling on the voxels and results in \b argmax
 * and \b pooled_features.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpRoiAwarePool3dForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] pool_method
 * Pooling method of Roiaware, 0 is 'maxpool', 1 is 'avgpool'. The default value is 0.
 * @param[in] boxes_num
 * An integer value which is the number of the rois.
 * @param[in] pts_num
 * An integer value which is the number of the pts.
 * @param[in] channels
 * An integer value which is the number of the pts feature of channels.
 * @param[in] rois_desc
 * The descriptor of the tensor \b rois, which contains the dimension and layout of the rois tensor.
 * @param[in] rois
 * Pointer to the MLU memory that stores the rois tensor.
 * @param[in] pts_desc
 * The descriptor of the tensor \b pts, which contains the dimension and layout of the pts tensor.
 * @param[in] pts
 * Pointer to the MLU memory that stores the pts tensor.
 * @param[in] pts_feature_desc
 * The descriptor of the tensor \b pts_feature, which contains the dimension and layout of the pts_feature tensor.
 * @param[in] pts_feature
 * Pointer to the MLU memory that stores the pts_feature tensor.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the
 * ::mluOpRoiAwarePool3dForward operation.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in
 * ::mluOpRoiAwarePool3dForward. You can get the size of the workspace with
 * ::mluOpGetRoiAwarePool3dForwardWorkspaceSize.
 * @param[in] max_pts_each_voxel
 * The maximum number of points per each voxel. An integer value which is the dimension of the pts_idx_of_voxels.
 * @param[in] out_x
 * An integer value which is the dimension of the pooled_features.
 * @param[in] out_y
 * An integer value which is the dimension of the pooled_features.
 * @param[in] out_z
 * An integer value which is the dimension of the pooled_features.
 * @param[in] argmax_desc
 * The descriptor of the tensor \b argmax, which contains the dimension and layout of the argmax tensor.
 * @param[out] argmax
 * Pointer to the MLU memory that stores the argmax tensor.
 * @param[in] pts_idx_of_voxels_desc
 * The descriptor of the tensor \b pts_idx_of_voxels, which contains the dimension and layout of the pts_idx_of_voxels
 * tensor.
 * @param[out] pts_idx_of_voxels
 * Pointer to the MLU memory that stores the pts_idx_of_voxels tensor.
 * @param[in] pooled_features_desc
 * The descriptor of the tensor \b pooled_features, which contains the dimension and layout of the pooled_features
 * tensor.
 * @param[out] pooled_features
 * Pointer to the MLU memory that stores the pooled_features tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - rois tensor: half, float
 *   - pts tensor: half, float
 *   - pts_feature tensor: half, float
 *   - argmax tensor: int32
 *   - pts_idx_of_voxels tensor: int32
 *   - pooled_features tensor: half, float
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The value of \b boxes_num should be less than 65536.
 * - The value of \b channels should be less than 65536.
 * - The Product of \b boxes_num and \b pts_num should be less than 2G.
 * - When the data type is floating point, the value of \b max_pts_each_voxel cannot be
 *   greater than 2976, and when the data type is half, it cannot be greater than 2944.
 * - The shape of \b rois should be [boxes_num, 7].
 * - The shape of \b pts should be [pts_num, 3].
 * - The shape of \b pts_feature should be [pts_num, channels].
 * - The shape of \b argmax should be [boxes_num, out_x, out_y, out_z, channels].
 * - The shape of \b pts_idx_of_voxels should be [boxes_num, out_x, out_y, out_z, max_pts_each_voxel].
 * - The shape of \b pooled_features should be [boxes_num, out_x, out_y, out_z, channels].
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The inputs \b rois and \b pts with NaN or infinity are not supported on MLU300 series.
 * - The input \b pts_feature with NaN are not supported on MLU300 series.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/tree/master/mmcv/ops/roiaware_pool3d.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRoiAwarePool3dForward(mluOpHandle_t handle,
                           const int pool_method,
                           const int boxes_num,
                           const int pts_num,
                           const int channels,
                           const mluOpTensorDescriptor_t rois_desc,
                           const void *rois,
                           const mluOpTensorDescriptor_t pts_desc,
                           const void *pts,
                           const mluOpTensorDescriptor_t pts_feature_desc,
                           const void *pts_feature,
                           void *workspace,
                           size_t workspace_size,
                           const int max_pts_each_voxel,
                           const int out_x,
                           const int out_y,
                           const int out_z,
                           const mluOpTensorDescriptor_t argmax_desc,
                           void *argmax,
                           const mluOpTensorDescriptor_t pts_idx_of_voxels_desc,
                           void *pts_idx_of_voxels,
                           const mluOpTensorDescriptor_t pooled_features_desc,
                           void *pooled_features);

// Group: Deprecated APIs
/*!
 * @brief Returns \b pts_idx_of_voxels, \b argmax, \b grad_out and \b grad_in by
 * performing the backpropagation of ::mluOpRoiAwarePool3dForward.
 *
 * @par Deprecated
 * - ::mluOpRoiawarePool3dBackward is deprecated and will be removed in the future
 *   release. It is recommended to use ::mluOpRoiAwarePool3dBackward instead.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpRoiAwarePool3dBackward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] pool_method
 * Pooling method of Roiaware. 0 is maxpool and 1 is avgpool. The default value is 0.
 * @param[in] boxes_num
 * An integer value which is the dimension of the pts_idx_of_voxels and argmax.
 * @param[in] out_x
 * An integer value which is the dimension of the pts_idx_of_voxels and argmax.
 * @param[in] out_y
 * An integer value which is the dimension of the pts_idx_of_voxels and argmax.
 * @param[in] out_z
 * An integer value which is the dimension of the pts_idx_of_voxels and argmax.
 * @param[in] channels
 * An integer value which is the number of the argmax and grad_out of channels.
 * @param[in] max_pts_each_voxel
 * The maximum number of points per each voxel. An integer value which is the dimension of the pts_idx_of_voxels.
 * @param[in] pts_idx_of_voxels_desc
 * The descriptor of the tensor \b pts_idx_of_voxels, which contains the dimension and layout of the pts_idx_of_voxels
 * tensor.
 * @param[out] pts_idx_of_voxels
 * Pointer to the MLU memory that stores the pts_idx_of_voxels tensor.
 * @param[in] argmax_desc
 * The descriptor of the tensor \b argmax, which contains the dimension and layout of the argmax tensor.
 * @param[out] argmax
 * Pointer to the MLU memory that stores the argmax tensor.
 * @param[in] grad_out_desc
 * The descriptor of the tensor \b grad_out, which contains the dimension and layout of the grad_out tensor.
 * @param[out] grad_out
 * Pointer to the MLU memory that stores the grad_out tensor.
 * @param[in] grad_in_desc
 * The descriptor of the tensor \b grad_in, which contains the dimension and layout of the grad_in tensor.
 * @param[in] grad_in
 * Pointer to the MLU memory that stores the grad_in tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - pts_idx_of_voxels tensor: int32
 *   - argmax tensor: int32
 *   - grad_out tensor: half, float
 *   - grad_in tensor: half, float
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The value of \b boxes_num should be less than 65536.
 * - The value of \b channels should be less than 65536.
 * - The value of \b max_pts_each_voxel cannot be greater than 98240.
 * - The shape of \b pts_idx_of_voxels should be [boxes_num, out_x, out_y, out_z, max_pts_each_voxel].
 * - The shape of \b argmax should be [boxes_num, out_x, out_y, out_z, channels].
 * - The shape of \b grad_out should be [boxes_num, out_x, out_y, out_z, channels].
 * - The shape of \b grad_in should be [pts_num, channels].
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/tree/master/mmcv/ops/roiaware_pool3d.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRoiawarePool3dBackward(mluOpHandle_t handle,
                            const int pool_method,
                            const int boxes_num,
                            const int out_x,
                            const int out_y,
                            const int out_z,
                            const int channels,
                            const int max_pts_each_voxel,
                            const mluOpTensorDescriptor_t pts_idx_of_voxels_desc,
                            const void *pts_idx_of_voxels,
                            const mluOpTensorDescriptor_t argmax_desc,
                            const void *argmax,
                            const mluOpTensorDescriptor_t grad_out_desc,
                            const void *grad_out,
                            const mluOpTensorDescriptor_t grad_in_desc,
                            void *grad_in);

// Group: RoiAwarePool3d
/*!
 * @brief Returns \b pts_idx_of_voxels, \b argmax, \b grad_out and \b grad_in by
 * performing the backpropagation of ::mluOpRoiAwarePool3dForward.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpRoiAwarePool3dBackward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] pool_method
 * Pooling method of Roiaware. 0 is maxpool and 1 is avgpool. The default value is 0.
 * @param[in] boxes_num
 * An integer value which is the dimension of the pts_idx_of_voxels and argmax.
 * @param[in] out_x
 * An integer value which is the dimension of the pts_idx_of_voxels and argmax.
 * @param[in] out_y
 * An integer value which is the dimension of the pts_idx_of_voxels and argmax.
 * @param[in] out_z
 * An integer value which is the dimension of the pts_idx_of_voxels and argmax.
 * @param[in] channels
 * An integer value which is the number of the argmax and grad_out of channels.
 * @param[in] max_pts_each_voxel
 * The maximum number of points per each voxel. An integer value which is the dimension of the pts_idx_of_voxels.
 * @param[in] pts_idx_of_voxels_desc
 * The descriptor of the tensor \b pts_idx_of_voxels, which contains the dimension and layout of the pts_idx_of_voxels
 * tensor.
 * @param[out] pts_idx_of_voxels
 * Pointer to the MLU memory that stores the pts_idx_of_voxels tensor.
 * @param[in] argmax_desc
 * The descriptor of the tensor \b argmax, which contains the dimension and layout of the argmax tensor.
 * @param[out] argmax
 * Pointer to the MLU memory that stores the argmax tensor.
 * @param[in] grad_out_desc
 * The descriptor of the tensor \b grad_out, which contains the dimension and layout of the grad_out tensor.
 * @param[out] grad_out
 * Pointer to the MLU memory that stores the grad_out tensor.
 * @param[in] grad_in_desc
 * The descriptor of the tensor \b grad_in, which contains the dimension and layout of the grad_in tensor.
 * @param[in] grad_in
 * Pointer to the MLU memory that stores the grad_in tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - pts_idx_of_voxels tensor: int32
 *   - argmax tensor: int32
 *   - grad_out tensor: half, float
 *   - grad_in tensor: half, float
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The value of \b boxes_num should be less than 65536.
 * - The value of \b channels should be less than 65536.
 * - The value of \b max_pts_each_voxel cannot be greater than 98240.
 * - The shape of \b pts_idx_of_voxels should be [boxes_num, out_x, out_y, out_z, max_pts_each_voxel].
 * - The shape of \b argmax should be [boxes_num, out_x, out_y, out_z, channels].
 * - The shape of \b grad_out should be [boxes_num, out_x, out_y, out_z, channels].
 * - The shape of \b grad_in should be [pts_num, channels].
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/tree/master/mmcv/ops/roiaware_pool3d.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRoiAwarePool3dBackward(mluOpHandle_t handle,
                            const int pool_method,
                            const int boxes_num,
                            const int out_x,
                            const int out_y,
                            const int out_z,
                            const int channels,
                            const int max_pts_each_voxel,
                            const mluOpTensorDescriptor_t pts_idx_of_voxels_desc,
                            const void *pts_idx_of_voxels,
                            const mluOpTensorDescriptor_t argmax_desc,
                            const void *argmax,
                            const mluOpTensorDescriptor_t grad_out_desc,
                            const void *grad_out,
                            const mluOpTensorDescriptor_t grad_in_desc,
                            void *grad_in);

// Group: Psamask
/*!
 * @brief Moves the \b x tensor to \b y tensor according to \b h_mask, \b w_mask, and \b psa_type.
 *
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in ::mluOpPsamaskForward. For detailed information, see ::mluOpHandle_t.
 * @param[in] psa_type
 * The types of the psamask computation, including COLLECT and DISTRIBUTE.
 * @param[in] x_desc
 * The descriptor of data of input tensor \b x. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the data of input tensor.
 * @param[in] h_mask
 * An integer value which is the h_mask factor of the psamask.
 * @param[in] w_mask
 * An integer value which is the w_mask factor of the psamask.
 * @param[in] y_desc
 * The descriptor of the tensor \b y. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] y
 * Pointer to the MLU memory that stores the data of output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - x: float
 *   - y: float
 *
 * @par Data Layout
 * - The supported data layouts of input and output tensors are as follows:
 *   - x: NHWC
 *   - y: NHWC
 *
 * @par Scale Limitation
 * - The shape of \b x must be [N, H, W, C].
 * - The shape of \b y must be [N, H, W, C].
 * - All dimension sizes of \b x and \b y must be the same, except the C dimension.
 * - If the shape of \b x is set to [N, H, W, C], the size of C dimension should be \b h_mask * \b
 *   w_mask.
 * - If the shape of \b y is set to [N, H, W, C], the size of C dimension should be H * W.
 *   - On MLU300 series:
 *     - When psa_type is COLLECT, the size of \b x channels ci and \b y channels co should be
 *       satisfied: ci + co <= 10240.
 *     - When psa_type is DISTRIBUTE, the size of \b x channels ci and \b y channels co should be
 *       satisfied: ci + 2 * co <= 10240.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/psa_mask.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpPsamaskForward(mluOpHandle_t handle,
                    const int psa_type,
                    const mluOpTensorDescriptor_t x_desc,
                    const void *x,
                    const int h_mask,
                    const int w_mask,
                    const mluOpTensorDescriptor_t y_desc,
                    void *y);

// Group: Psamask
/*!
 * @brief Computes the gradients of input tensor \b dx with the gradients of output tensor \b dy
 * according to \b h_mask, \b w_mask, and \b psa_type.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in ::mluOpPsamaskBackward. For detailed information, see ::mluOpHandle_t.
 * @param[in] psa_type
 * The types of the psamask computation, including COLLECT and DISTRIBUTE.
 * @param[in] dy_desc
 * The descriptor of gradient of output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] dy
 * Pointer to the MLU memory that stores the gradient of output tensor.
 * @param[in] h_mask
 * An integer value which is the h_mask factor of the psamask.
 * @param[in] w_mask
 * An integer value which is the w_mask factor of the psamask.
 * @param[in] dx_desc
 * The descriptor of gradient of input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] dx
 * Pointer to the MLU memory that stores the gradient of input tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - dy: float
 *   - dx: float
 *
 * @par Data Layout
 * - The supported data layouts of input and output tensors are as follows:
 *   - dy: NHWC
 *   - dx: NHWC
 *
 * @par Scale Limitation
 * - The shape of \b dy must be [N, H, W, C].
 * - The shape of \b dx must be [N, H, W, C].
 * - All dimension sizes of \b dy and \b dx must be the same, except the C dimension.
 * - If the shape of \b dx is set to [N, H, W, C], the size of C dimension should be \b h_mask * \b
 *   w_mask .
 * - If the shape of \b dy is set to [N, H, W, C], the size of C dimension should be H * W.
 *   - On MLU300 series:
 *     - When psa_type is COLLECT, the size of \b dx channels ci and \b dy channels co should be
 *       satisfied: ci + co <= 10240.
 *     - When psa_type is DISTRIBUTE, the size of \b dx channels ci and \b dy channels co should be
 *       satisfied: ci + 2 * co <= 10240.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/psa_mask.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpPsamaskBackward(mluOpHandle_t handle,
                     const int psa_type,
                     const mluOpTensorDescriptor_t dy_desc,
                     const void *dy,
                     const int h_mask,
                     const int w_mask,
                     const mluOpTensorDescriptor_t dx_desc,
                     void *dx);

// Group: SparseConv
/*!
 * @brief Computes the get_indice_paris operation, then returns the results in the output
 * tensor \b out_indices, \b indice_pairs and \b ind, ice_num.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * get_indice_pairs operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] sparse_conv_desc
 * The descriptor of the tensor \b sparse_conv that needs convolution. For detailed information,
 * see ::mluOpSparseConvolutionDescriptor_t.
 * @param[in] indices_desc
 * The descriptor of the output grad. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] indices
 * Pointer to the MLU memory that stores the indices tensor.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the get_indice_pairs operation.
 * For more information about workspace, see "Cambricon MLU-OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in this
 * operation. You can get the size of the workspace with
 * ::mluOpGetIndicePairsWorkspaceSize.
 * @param[in] indice_pairs_desc
 * The descriptor of the tensor \b indice_pairs between input locations and output locations.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] indice_pairs
 * Pointer to the MLU memory that stores the indice_pairs tensor.
 * @param[in] indice_num_desc
 * The descriptor of the tensor \b indice_num including the number of input points while
 * calculating with every kernel points.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] indice_num
 * Pointer to the MLU memory that stores the indice_num tensor.
 * @param[in] out_indices_desc
 * The descriptor of the tensor \b out_indices including output locations. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] out_indices
 * Pointer to the MLU memory that stores the out_indices tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH,
 *   ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - This function supports the combinations of the following data types for
 *   input tensor \b indices and output tensor \b out_indices, \b indice_pairs and \b indice_num.
 * - \b indices, \b out_indices, \b indice_pairs, and \b indice_num data type: int32, int32, int32, int32
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The params inverse and transpose are not supported now.
 * - Get_indice_pairs only supports 3D.
 * - The input tensor and output tensor must meet the following requirements:
 *   - The \b indices must be two dimensions.
 *   - The \b indice_pairs must be three dimensions, and the first dimension value must be equal to kernel size,
 *     the second dimension must be 2, and the last dimension must be the same as the number of
 *     product of the first n-1 dimensions of the input tensor in sparse convolution.
 *   - The \b out_indices should be 2 dimensions. The first dimension of \b out_indices is the number effective
 *     output point. and the second dimension of must product of the first n-1 dimensions of the input tensor
 *     in sparse convolution.
 *   - The \b indice_num should be 1 dimension. The first dimension of \b indice_num is the kernel size.
 *
 * @par API Dependency
 * - Before calling this function, you need to prepare
 *   all the parameters passed to this function. See each parameter description for details.
 *
 * @par Note
 * - This function is only supported on MLU300 series or above platforms.
 * - The parameter num_act_out will be obtained from ::mluOpSparseConvolutionDescriptor_t.
 *
 * @par Example
 * - The example of the operation is as follows:
     @verbatim
      Dimension of indices tensor:  [input_active_in, dimnb -1]
      Dimension of out_indices tensor:  [output_active_num, dimnb - 1]
      Dimension of indice_pairs tensor: [kd * kh * kw, 2, input_active_in]
      Dimension of indice_num tensor: [kd * kh * kw]
     @endverbatim
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/cuda/spconv_ops_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetIndicePairs(mluOpHandle_t handle,
                    const mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
                    const mluOpTensorDescriptor_t indices_desc,
                    const void *indices,
                    void *workspace,
                    const size_t workspace_size,
                    const mluOpTensorDescriptor_t indice_pairs_desc,
                    void *indice_pairs,
                    const mluOpTensorDescriptor_t out_indices_desc,
                    void *out_indices,
                    const mluOpTensorDescriptor_t indice_num_desc,
                    void *indice_num);

// Group: SparseConv
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra workspace
 * to optimize the get_indice_pairs operation.
 *
 * The size of extra workspace is based on the given information of the get_indice_pairs
 * operation, including the input tensor descriptor \b sparse_conv_desc, and \b indices_desc, output
 * tensor descriptor \b out_indices_desc, \b indice_pairs_desc, and \b indice_num_desc.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * get_indice_pairs operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] sparse_conv_desc
 * The descriptor of the tensor \b sparse_conv that needs convolution. For detailed information,
 * see ::mluOpSparseConvolutionDescriptor_t.
 * @param[in] indices_desc
 * The descriptor of the tensor \b indices. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] out_indices_desc
 * The descriptor of the tensor \b out_indices including output locations. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] indice_pairs_desc
 * The descriptor of the tensor \b indice_pairs between input locations and output locations.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] indice_num_desc
 * The descriptor of the tensor \b indice_num, the number of input points while calculating with every kernel points.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] workspace_size
 * Pointer to the MLU memory that stores the returned size of the extra workspace in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_INTERNAL_ERROR,
 *   ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - You need to call ::mluOpCreateTensorDescriptor and ::mluOpSetTensorDescriptor to create and set
 *   tensor descriptors \b indices_desc, \b out_indices_desc, \b indice_pairs_desc, and \b indice_num_desc before
 *   calling this function.
 * - You need to call ::mluOpCreateSparseConvolutionDescriptor to create a descriptor,
 *   and call ::mluOpSetSparseConvolutionDescriptor to set the tensor information for
 *   the descriptor \b sparse_conv_desc.
 * - The allocated extra workspace should be passed to ::mluOpGetIndicePairs to
 *   perform the ge_indice_pairs operation.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetIndicePairsWorkspaceSize(mluOpHandle_t handle,
                                 const mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
                                 const mluOpTensorDescriptor_t indices_desc,
                                 const mluOpTensorDescriptor_t indice_pairs_desc,
                                 const mluOpTensorDescriptor_t out_indices_desc,
                                 const mluOpTensorDescriptor_t indice_num_desc,
                                 size_t *workspace_size);

// Group: ActiveRotatedFilter
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra
 * workspace to optimize ::mluOpActiveRotatedFilterForward. The size of the extra
 * workspace is based on the given information of the ActiveRotatedFilterForward operation,
 * including the input tensor descriptor \b input_desc. For more information about the workspace,
 * see "Cambricon MLU-OPS User Guide".
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpActiveRotatedFilterForward. For detailed information, see ::mluOpHandle_t.
 * @param[in] input_desc
 * The descriptor of the input data, which contains dimension, data type, and data layout.
 * @param[out] workspace_size
 * A host pointer to the returned size of the extra workspace in bytes that is used in
 * ::mluOpActiveRotatedFilterForward.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - This function must be called before ::mluOpActiveRotatedFilterForward.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetActiveRotatedFilterForwardWorkspaceSize(const mluOpHandle_t handle,
                                                const mluOpTensorDescriptor_t input_desc,
                                                size_t *workspace_size);

// Group: ActiveRotatedFilter
/*!
 * @brief Rotates \b input according to \b indices. This function encodes
 * the orientation information and generates orientation-sensitive features.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * the ActiveRotatedFilterForward operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] input_desc
 * The descriptor of input data, which contains dimension, data type, and data layout.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] indices_desc
 * The descriptor of input data \b indices, which contains dimension, data type, and data layout.
 * @param[in] indices
 * Pointer to the MLU memory that stores the indices tensor. It is used to
 * specify the position of each element of canonical filter after rotations.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the
 * ActiveRotatedFilterForward operation. For more information about workspace,
 * see "Cambricon MLU-OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that is used in
 * the ActiveRotatedFilterForward operation.
 * @param[in] output_desc
 * The descriptor of output data, which contains dimension, data type, and data layout.
 * @param[out] output
 * Pointer to the MLU memory that stores the \b output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of input tensor and output tensor should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - output tensor: half, float
 *   - indices tensor: int32
 *
 * @par Data Layout
 * - The supported data layouts of \b input tensor, \b indices tensor, and \b output tensor are as follows:
 *   - input tensor: MLUOP_LAYOUT_ARRAY
 *   - output tensor: MLUOP_LAYOUT_ARRAY
 *   - indices tensor: MLUOP_LAYOUT_ARRAY
 *
 * @par Scale Limitation
 * - The \b input is 5D array, and \b indices and \b output are 4D array.
 * - The dims[2] of \b input should be equal to the power of 2 and less than or
 *   equal to 128, dims[3] should be equal to 1 or 3, and dims[3] should be equal
 *   to dims[4].
 * - The dims[0] of \b indices should be equal to \b input dims[2], and dims[1]
 *   and dims[2] of \b indices should be equal to dims[3] and dims[4] of \b input
 *   respectively.
 * - The dims[3] of \b indices should be equal to 2, 4, or 8.
 * - The dims[0] of \b output should be equal to dims[0] of \b input times
 *   dims[3] of \b indices.
 * - The dims[1] of \b output should be equal to dims[1] of \b input times
 *   dims[2] of \b input.
 * - The dims[2] and dims[3] of \b output should be equal to dims[3] and dims[4]
 *   of \b input respectively.
 *
 * @par API Dependency
 * - Before calling this function, you need to call
 *   ::mluOpGetActiveRotatedFilterForwardWorkspaceSize to get the extra space size
 *   needed in ::mluOpActiveRotatedFilterForward.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * -
 * https://github.com/open-mmlab/mmcv/blob/v1.5.2/mmcv/ops/csrc/pytorch/cuda/active_rotated_filter_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpActiveRotatedFilterForward(const mluOpHandle_t handle,
                                const mluOpTensorDescriptor_t input_desc,
                                const void *input,
                                const mluOpTensorDescriptor_t indices_desc,
                                const void *indices,
                                void *workspace,
                                const size_t workspace_size,
                                const mluOpTensorDescriptor_t output_desc,
                                void *output);

/*!
 * @brief Enumeration variables describing the attributes of the AdamW computation.
 */
typedef enum {
  MLUOP_ADAMW_WEIGHT_DECAY = 0,
  /*!< Set the weight_decay attribute for the AdamW operation. */
  MLUOP_ADAMW_GRAD_SCALE = 1,
  /*!< Set the grad_scale attribute for the AdamW operation. */
  MLUOP_ADAMW_USE_NESTEROV = 2,
  /*!< Specifies whether to use nesterov on the AdamW operation. */
} mluOpAdamWDescAttribute_t;

typedef struct mluOpAdamWStruct *mluOpAdamWDescriptor_t;

// Group: AdamW
/*!
 * @brief Updates each attribute by using AdamW.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices
 * and queues in the AdamW operation. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in] adamw_desc
 * A host pointer to the AdamW descriptor that holds information about the AdamW operation.
 * @param[in] param_desc
 * The descriptor of the tensor, which contains the dimension and layout of param.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] param
 * Pointer to the MLU memory that stores the param tensor.
 * @param[in] paramh_desc
 * The descriptor of the tensor, which contains the dimension and layout of param_h.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] param_h
 * Pointer to the MLU memory that stores the param_h tensor.
 * @param[in] momentum_desc
 * The descriptor of the tensor, which contains the dimension and layout of momentum.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] momentum
 * Pointer to the MLU memory that stores the momentum tensor.
 * @param[in] velocity_desc
 * The descriptor of the tensor, which contains the dimension and layout of velocity.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] velocity
 * Pointer to the MLU memory that stores the velocity tensor.
 * @param[in] grad_desc
 * The descriptor of the tensor, which contains the dimension and layout of grad.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] grad
 * Pointer to the MLU memory that stores the grad tensor.
 * @param[in] lr
 * A scalar of lr factor that is used for AdamW.
 * @param[in] beta1
 * A scalar of beta1 factor that is used for AdamW.
 * @param[in] beta2
 * A scalar of beta2 factor that is used for AdamW.
 * @param[in] bias1
 * A scalar of bias1 factor that is used for AdamW.
 * @param[in] bias2
 * A scalar of bias2 factor that is used for AdamW.
 * @param[in] epsilon
 * A scalar of epsilon factor that is used for AdamW.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - param tensor: float
 *   - param_h tensor: bfloat16
 *   - momentum tensor: float
 *   - velocity tensor: float
 *   - grad tensor: bfloat16
 *
 * @par Data Layout
 * - The supported data layouts of \b param tensor, \b param_h tensor, \b momentum tensor, \b velocity tensor, and \b
 * grad tensor are as follows:
 *   - param tensor: \p MLUOP_LAYOUT_ARRAY
 *   - param_h tensor: \p MLUOP_LAYOUT_ARRAY
 *   - momentum tensor: \p MLUOP_LAYOUT_ARRAY
 *   - velocity tensor: \p MLUOP_LAYOUT_ARRAY
 *   - grad tensor: \p MLUOP_LAYOUT_ARRAY
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/OpenBMB/BMTrain/blob/6abcf772aa1e120192f7656e55c4adbcde53c886/csrc/cuda/adam_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpAdamW(mluOpHandle_t handle,
           mluOpAdamWDescriptor_t adamw_desc,
           const mluOpTensorDescriptor_t param_desc,
           void *param,
           const mluOpTensorDescriptor_t paramh_desc,
           void *param_h,
           const mluOpTensorDescriptor_t momentum_desc,
           void *momentum,
           const mluOpTensorDescriptor_t velocity_desc,
           void *velocity,
           const mluOpTensorDescriptor_t grad_desc,
           void *grad,
           const float lr,
           const float beta1,
           const float beta2,
           const float bias1,
           const float bias2,
           const float epsilon);

// Group: AdamW
/*!
 * @brief Creates a descriptor pointed by \p adamw_desc for AdamW operation.
 * The information is defined in ::mluOpAdamWDescriptor_t.
 * For more information about the descriptor, see "Cambricon MLU-OPS User Guide".
 *
 * @param[out] adamw_desc
 * A host pointer to the AdamW descriptor that holds information about the
 * AdamW operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ALLOC_FAILED
 *
 * @par API Dependency
 * - After calling this function, call ::mluOpSetAdamWDescAttr function to initialize
 *   and set the information to the AdamW descriptor.
 *
 * @par Note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateAdamWDescriptor(mluOpAdamWDescriptor_t *adamw_desc);

// Group: AdamW
/*!
 * @brief Initializes the descriptor \b adamw_desc that was previously created with
 * ::mluOpCreateAdamWDescriptor function, and sets AdamW information
 * to the descriptor \b adamw_desc. The information includes \b weight_decay, \b grad_scale
 * and \b use_nesterov for AdamW operation.
 *
 * @param[in] adamw_desc
 * The descriptor of the AdamW operation. For detailed information,
 * see ::mluOpAdamWDescriptor_t.
 * @param[in] attr
 * Attribute of AdamW descriptor to be set. For detailed information,
 * see ::mluOpAdamWDescAttribute_t.
 * @param[in] buf
 * A host pointer to the attribute value set by this function.
 * @param[in] size_in_bytes
 * Buffer in bytes for verification.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - This function should be called after ::mluOpCreateAdamWDescriptor.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetAdamWDescAttr(mluOpAdamWDescriptor_t adamw_desc,
                      mluOpAdamWDescAttribute_t attr,
                      const void *buf,
                      const size_t size_in_bytes);

// Group: AdamW
/*!
 * @brief Destroys the AdamW descriptor \p adamw_desc that was previously created by
 * ::mluOpCreateAdamWDescriptor.
 *
 * @param[in] adamw_desc
 *   The AdamW descriptor to be destroyed.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Note
 * - Call this function after calling ::mluOpAdamW.
 * - It is necessary to call this function to destroy the AdamW descriptor to avoid memory leak.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDestroyAdamWDescriptor(mluOpAdamWDescriptor_t adamw_desc);

// Group: DeformRoiPool
/*!
 * @brief Computes deformable roi pooling over \b input tensor. This function firstly divides the obtained
 * candidate region into regions with the same size according to the specified pooling width and pooling height,
 * then adds offsets to rois, and finally calculates the mean value of the sampling points in each bin as output.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpDeformRoiPoolForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] input_desc
 * The descriptor of input tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] rois_desc
 * The descriptor of the tensor \b rois, which contains the dimension and layout of rois tensor.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] rois
 * Pointer to the MLU memory that stores the rois tensor.
 * @param[in] offset_desc
 * The descriptor of the tensor \b offset, which contains the dimension and layout of offset tensor.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] offset
 * Pointer to the MLU memory that stores the offset tensor.
 * @param[in] pooled_height
 * An integer value which is the height of the output after pooling.
 * @param[in] pooled_width
 * An integer value which is the width of the output after pooling.
 * @param[in] spatial_scale
 * A float value which is the scale factor of coordinates of rois.
 * @param[in] sampling_ratio
 * An integer value which is the number of samples in one bin. This parameter
 * only works when it is greater than zero.
 * @param[in] gamma
 * A float value which is the scale factor of offset.
 * @param[in] output_desc
 * The descriptor of the output tensor, which contains the dimension and layout of output tensor.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - rois tensor: half, float
 *   - offset tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b input tensor, \b rois tensor, \b offset tensor, and \b output tensor are as
 follows:
 *   - input tensor: \p MLUOP_LAYOUT_NHWC
 *   - rois tensor: \p MLUOP_LAYOUT_ARRAY
 *   - offset tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The input tensor and output tensor must be 4D.
 * - The sizes of the lowest dimension of input tensor and output tensor must be the same.
 * - The rois tensor must be 2D.
 * - The offset tensor must be 4D.
 * - The sizes of the highest dimension of output tensor, rois tensor and offset tensor must be the same.
 * - The sizes of the middle two dimensions of output tensor and the sizes of the lower two dimensions of offset tensor
 *   must be the same.
 * - The shape of \b input should be [batch_num, height, width, channels].
 * - The shape of \b rois should be [rois_num, 5].
 * - The shape of \b offset should be [rois_num, 2, pooled_height, pooled_width].
 * - The shape of \b output should be [rois_num, pooled_height, pooled_width, channels].
 * - \b rois[i] consists of [batch_id, x1, y1, x2, y2]. \p batch_id should be in the range of [0, batch_num - 1].
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The inputs \b rois and \b offset with NaN or infinity are not supported.
 *
 * @par Example
 * - The example of the deform_roi_pool_forward operation is as follows:
     @verbatim
      input three arrays by 1  2  2  1, 1  5 and 1  2  1 * 1
      --> input: [[[[1.0], [2.0]], [[2.0], [4.0]]]]
      --> rois: [[0.0, 0.0, 0.0, 1.0, 1.0]]
      --> offset: [[[[0.5]], [[0.5]]]]
      param:
             pooled_height: 1.0, pooled_width: 1.0, spatial_scale: 1.0,
             sampling_ratio: 1, gamma: 1
      output array by 1  1  1 * 1 -->
          output: [[[[2.25]]]]
     @endverbatim
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/tree/master/mmcv/ops/deform_roi_pool.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDeformRoiPoolForward(const mluOpHandle_t handle,
                          const mluOpTensorDescriptor_t input_desc,
                          const void *input,
                          const mluOpTensorDescriptor_t rois_desc,
                          const void *rois,
                          const mluOpTensorDescriptor_t offset_desc,
                          const void *offset,
                          const int pooled_height,
                          const int pooled_width,
                          const float spatial_scale,
                          const int sampling_ratio,
                          const float gamma,
                          const mluOpTensorDescriptor_t output_desc,
                          void *output);

// Group: DeformRoiPool
/*!
 * @brief Computes the gradient of input \b grad_input and the gradient of offset \b grad_offset
 * based on the gradient of output \b grad_output, input \b input, ROI \b rois, and offset \b offset.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpDeformRoiPoolBackward. For detailed information, see ::mluOpHandle_t.
 * @param[in] grad_output_desc
 * The descriptor of the tensor \b grad_output. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] grad_output
 * Pointer to the MLU memory that stores the grad_output tensor.
 * @param[in] input_desc
 * The descriptor of the tensor \b input. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] rois_desc
 * The descriptor of the tensor \b rois, which contains the dimension and layout of rois tensor.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] rois
 * Pointer to the MLU memory that stores the rois tensor.
 * @param[in] offset_desc
 * The descriptor of the tensor \b offset, which contains the dimension and layout of offset tensor.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] offset
 * Pointer to the MLU memory that stores the offset tensor.
 * @param[in] pooled_height
 * An integer value which is the height of the output after pooling.
 * @param[in] pooled_width
 * An integer value which is the width of the output after pooling.
 * @param[in] spatial_scale
 * A float value which is the scale factor of coordinates of rois.
 * @param[in] sampling_ratio
 * An integer value which is the number of samples in one bin. This parameter
 * only works when it is greater than zero.
 * @param[in] gamma
 * A float value which is the scale factor of offset.
 * @param[in] grad_input_desc
 * The descriptor of the tensor \b grad_input, which contains the dimension and layout of grad_input tensor.
 * @param[out] grad_input
 * Pointer to the MLU memory that stores the gradient of the input tensor.
 * @param[in] grad_offset_desc
 * The descriptor of the tensor \b grad_offset, which contains the dimension and layout of grad_offset tensor.
 * @param[out] grad_offset
 * Pointer to the MLU memory that stores the gradient of the offset tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tentors are as follows:
 *   - grad_output tensor: half, float
 *   - input tensor: half, float
 *   - rois tensor: half, float
 *   - offset tensor: half, float
 *   - grad_input tensor: half, float
 *   - grad_offset tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b grad_output tensor, \b input tensor, \b rois tensor,
 *   \b offset tensor, \b grad_input tensor, and \b grad_offset tensor are as follows:
 *   - grad_output tensor: \p MLUOP_LAYOUT_NHWC
 *   - input tensor: \p MLUOP_LAYOUT_NHWC
 *   - rois tensor: \p MLUOP_LAYOUT_ARRAY
 *   - offset tensor: \p MLUOP_LAYOUT_ARRAY
 *   - grad_input tensor: \p MLUOP_LAYOUT_NHWC
 *   - grad_offset tensor: \p MLUOP_LAYOUT_ARRAY
 *
 * @par Scale Limitation
 * - The grad_output tensor, input tensor and grad_input tensor must be 4D.
 * - The sizes of the lowest dimension of grad_output tensor, input tensor and grad_input tensor must be the same.
 * - The rois tensor must be 2D.
 * - The offset tensor and grad_offset tensor must be 4D.
 * - The sizes of the highest dimension of output tensor, rois tensor, offset tensor and grad_offset tensor must be the
 *   same.
 * - The sizes of the middle two dimensions of grad_output tensor, the sizes of the lower two dimensions of offset
 *   tensor and the sizes of the lower two dimensions of grad_offset tensor must be the same.
 * - The shape of \b grad_output should be [rois_num, pooled_height, pooled_width, channels].
 * - The shape of \b input should be [batch_num, height, width, channels].
 * - The shape of \b rois should be [rois_num, 5].
 * - The shape of \b offset should be [rois_num, 2, pooled_height, pooled_width].
 * - The shape of \b grad_input should be [batch_num, height, width, channels].
 * - The shape of \b grad_offset should be [rois_num, 2, pooled_height, pooled_width].
 * - \b rois[i] consists of [batch_id, x1, y1, x2, y2]. \p batch_id should be in the range of [0, batch_num - 1].
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - The inputs \b rois and \b offset with NaN or infinity are not supported.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/tree/master/mmcv/ops/deform_roi_pool.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDeformRoiPoolBackward(const mluOpHandle_t handle,
                           const mluOpTensorDescriptor_t grad_output_desc,
                           const void *grad_output,
                           const mluOpTensorDescriptor_t input_desc,
                           const void *input,
                           const mluOpTensorDescriptor_t rois_desc,
                           const void *rois,
                           const mluOpTensorDescriptor_t offset_desc,
                           const void *offset,
                           const int pooled_height,
                           const int pooled_width,
                           const float spatial_scale,
                           const int sampling_ratio,
                           const float gamma,
                           const mluOpTensorDescriptor_t grad_input_desc,
                           void *grad_input,
                           const mluOpTensorDescriptor_t grad_offset_desc,
                           void *grad_offset);

// Group: BorderAlign
/*!
 * @brief Extracts the border features of \b input based on the bounding boxes
 * to compute the maximum border features of \b input with the maximum pooling.
 * The computing process of this operation is as follows:
 *  1. For each border line of each box (commonly four lines: top, left, bottom
 *     and right lines), uniformly samples pool_size + 1 positions on this line,
 *     involving the starting point and endpoint.
 *  2. Compute the corresponding features on these points by bilinear interpolation.
 *  3. Perform the max pooling over all pool_size + 1 positions to output the
 *     pooled features.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in ::mluOpBorderAlignForward operation. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in] input_desc
 * The descriptor of the input tensor.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor. The shape of \b input
 * is [N, H, W, 4C]. Channels ranged in [0,C), [C,2C), [2C,3C), [3C,4C) represent
 * the top, left, bottom and right features respectively.
 * @param[in] boxes_desc
 * Descriptor of bounding box tensor.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] boxes
 * Pointer to the MLU memory that stores boxes tensors. The shape of \b boxes is
 * [N, H * W, 4].
 * @param[in] pool_size
 * Number of positions sampled over the boxes' borders.
 * @param[in] output_desc
 * Descriptor of \b output, containing dimension and the layout of the output.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor. The shape of
 * argmax_idx is [N, H * W, 4, C].
 * @param[in] argmax_idx_desc
 * Descriptor of \b argmax_idx, containing dimension and the layout of \b argmax_idx .
 * @param[out] argmax_idx
 * Pointer to the MLU memory that stores the \b argmax_idx tensor, which is the
 * indices of maximum values after the max pooling. The shape of \b argmax_idx
 * is [N, H * W, 4, C].
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - boxes tensor: half, float
 *   - output tensor: half, float
 *   - argmax_idx tensor: int32_t
 *   Note that the data type of \b input, \b boxes, and \b output
 *   must be the same.
 *
 * @par Data Layout
 * - The supported data layout of \b input, \b boxes, \b output, and
 *   \b argmax_idx are as follows:
 *   - input tensor: \p MLUOP_LAYOUT_NHWC
 *   - boxes tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor: \p MLUOP_LAYOUT_NHWC
 *   - argmax_idx tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The \b input, \b output and \b argmax_idx are 4D tensor.
 * - The \b boxes tensor is 3D tensor.
 * - The dims[3] of \b boxes should be equal to 4.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - The example of the border_align_forward operation is as follows:
     @verbatim
     input: a tensor with shape by 1 * 4 * 3 *4  --> [[[[ 1.,  2.,  3.,  4.],
                                                        [ 5.,  6.,  7.,  8.],
                                                        [ 9., 10., 11., 12.]],

                                                       [[ 6.,  7.,  5.,  8.],
                                                        [ 2.,  1.,  3.,  4.],
                                                        [12.,  9., 11., 10.]],

                                                       [[-2., -3.,  2.,  0.],
                                                        [-4., -5.,  1., -1.],
                                                        [-1., -1., -1., -1.]],

                                                       [[ 0., -1.,  2.,  1.],
                                                        [-4., -3., -2., -1.],
                                                        [-1., -2., -3., -4.]]]]

     boxes: a tensor with shape by 1 * 12 * 4  --> [[[0., 0., 2., 1.],
                                                     [1., 0., 3., 1.],
                                                     [1., 0., 2., 1.],
                                                     [0., 0., 3., 1.],
                                                     [0., 0., 1., 2.],
                                                     [0., 0., 2., 2.],
                                                     [1., 0., 2., 1.],
                                                     [1., 0., 3., 1.],
                                                     [0., 1., 1., 2.],
                                                     [0., 0., 3., 2.],
                                                     [1., 0., 3., 2.],
                                                     [2., 0., 3., 2.]]]

     param: pool_size = 1.
     output: a tensor with shape by 1 * 1 * 12 * 4 --> [[[[ 3.,  6.,  1.,  2.],
                                                          [ 4.,  7., -1.,  1.],
                                                          [ 3.,  7.,  1.,  2.],
                                                          [ 4.,  6., -1.,  1.],
                                                          [ 2., 12., -1., -1.],
                                                          [ 3., 12., -1.,  2.],
                                                          [ 3.,  7.,  1.,  2.],
                                                          [ 4.,  7., -1.,  1.],
                                                          [ 6., 12., -1., -2.],
                                                          [ 4., 12., -1.,  1.],
                                                          [ 4.,  9., -1.,  1.],
                                                          [ 4., 11., -1.,  1.]]]]

     argmax_idx: a tensor with shape by 1 * 1 * 12 * 4 -->[[[[1, 0, 0, 1],
                                                             [1, 0, 0, 1],
                                                             [1, 0, 0, 1],
                                                             [1, 0, 0, 1],
                                                             [1, 1, 0, 1],
                                                             [1, 1, 0, 1],
                                                             [1, 0, 0, 1],
                                                             [1, 0, 0, 1],
                                                             [1, 1, 0, 0],
                                                             [1, 1, 0, 1],
                                                             [1, 1, 0, 1],
                                                             [1, 1, 0, 1]]]]
   @endverbatim
 *
 * @par Reference
 * - http://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/border_align.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpBorderAlignForward(mluOpHandle_t handle,
                        const mluOpTensorDescriptor_t input_desc,
                        const void *input,
                        const mluOpTensorDescriptor_t boxes_desc,
                        const void *boxes,
                        const int32_t pool_size,
                        const mluOpTensorDescriptor_t output_desc,
                        void *output,
                        const mluOpTensorDescriptor_t argmax_idx_desc,
                        void *argmax_idx);

// Group: BorderAlign
/*!
 * @brief Computes the gradient of the input tensor of ::mluOpBorderAlignForward
 * according to the output gradient \b grad_output, the maximum pooling index \b
 * argmax_idx and bounding boxes \b boxes .
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues
 * in ::mluOpBorderAlignBackward operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] grad_output_desc
 * The descriptor of the \b grad_output tensor.
 * @param[in] grad_output
 * Pointer to the MLU memory that stores the output gradient of ::mluOpBorderAlignForward.
 * The shape of \b grad_output is [N, K, 4, C], where N is the number of images,
 * K is the product of the width of images and the height of images, and C is the
 * number of image channels.
 * @param[in] boxes_desc
 * Descriptor of bounding box tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] boxes
 * Pointer to the MLU memory that stores \b boxes tensors. The shape of \b boxes is
 * [N, H * W, 4].
 * @param[in] argmax_idx_desc
 * Descriptor of \b argmax_idx, containing dimension and the layout of \b argmax_idx .
 * @param[in] argmax_idx
 * Pointer to the MLU memory that stores the \b argmax_idx tensor, which is the result
 * of max pooling index. The shape of argmax_idx is [N, K, 4, C].
 * @param[in] pool_size
 * Number of positions sampled over the boxes borders.
 * @param[in] grad_input_desc
 * Descriptor of \b grad_input, containing dimension and the layout of output.
 * @param[out] grad_input
 * Pointer to the MLU memory that stores the gradient of the input
 * tensor of ::mluOpBorderAlignForward. The shape of \b grad_input is [N, H, W, 4C],
 * where N is the number of images, H is the height of images, W is the width of
 * images, and C is the number of image channels.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - grad_output tensor: half, float
 *   - boxes tensor: half, float
 *   - argmax_idx tensor: int32_t
 *   - grad_input tensor: half, float
 *   Note that the data type of \b grad_output, \b boxes, and \b grad_input
 *   must be the same.
 *
 * @par Data Layout
 * - The supported data layout of \b grad_output, \b boxes, \b argmax_idx and,
 *   \b grad_input are as follows:
 *   - grad_output tensor: \p MLUOP_LAYOUT_NHWC
 *   - boxes tensor: \p MLUOP_LAYOUT_ARRAY
 *   - argmax_idx tensor: \p MLUOP_LAYOUT_NHWC
 *   - grad_input tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The \b grad_output, \b argmax_idx and \b grad_input are 4D tensor.
 * - The \b boxes is 3D tensor.
 * - The dims[3] of \b boxes should be equal to 4.
 * - The shape of \b grad_output and \b argmax_idx must be the same.
 * - The value of \b argmax_idx should be in the range of [0, pool_size].
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
     @verbatim
     input: a tensor with shape by 1 * 12 * 4 * 1  -->[[[[ 3.],[ 6.],[ 1.],[ 2.]],
                                                        [[ 4.],[ 7.],[-1.],[ 1.]],
                                                        [[ 3.],[ 7.],[ 1.],[ 2.]],
                                                        [[ 4.],[ 6.],[-1.],[ 1.]],
                                                        [[ 2.],[12.],[-1.],[-1.]],
                                                        [[ 3.],[12.],[-1.],[ 2.]],
                                                        [[ 3.],[ 7.],[ 1.],[ 2.]],
                                                        [[ 4.],[ 7.],[-1.],[ 1.]],
                                                        [[ 6.],[12.],[-1.],[-2.]],
                                                        [[ 4.],[12.],[-1.],[ 1.]],
                                                        [[ 4.],[ 9.],[-1.],[ 1.]],
                                                        [[ 4.],[11.],[-1.],[ 1.]]]]
     boxes: a tensor with shape by 1 * 12 * 4  --> [[[0., 0., 2., 1.],
                                                     [1., 0., 3., 1.],
                                                     [1., 0., 2., 1.],
                                                     [0., 0., 3., 1.],
                                                     [0., 0., 1., 2.],
                                                     [0., 0., 2., 2.],
                                                     [1., 0., 2., 1.],
                                                     [1., 0., 3., 1.],
                                                     [0., 1., 1., 2.],
                                                     [0., 0., 3., 2.],
                                                     [1., 0., 3., 2.],
                                                     [2., 0., 3., 2.]]]
     argmax_idx: a tensor with shape by 1 * 12 * 4 * 1 --> [[[[1],[0],[0],[1]],
                                                             [[1],[0],[0],[1]],
                                                             [[1],[0],[0],[1]],
                                                             [[1],[0],[0],[1]],
                                                             [[1],[1],[0],[1]],
                                                             [[1],[1],[0],[1]],
                                                             [[1],[0],[0],[1]],
                                                             [[1],[0],[0],[1]],
                                                             [[1],[1],[0],[0]],
                                                             [[1],[1],[0],[1]],
                                                             [[1],[1],[0],[1]],
                                                             [[1],[1],[0],[1]]]]
     param: pool_size = 1.
     output: a tensor with shape by 1 * 3 * 4 * 4 --> [[[[ 0, 12,  0,  0],
                                                         [ 2, 28,  0, -1],
                                                         [12,  0,  0,  8],
                                                         [24,  0,  0,  6]],

                                                        [[ 0,  0,  0,  0],
                                                         [ 6,  0,  0,  0],
                                                         [ 0,  0,  3,  0],
                                                         [ 0,  0, -3,  0]],

                                                        [[ 0, 48,  0,  0],
                                                         [ 0,  9, -2, -2],
                                                         [ 0, 11, -1,  0],
                                                         [ 0,  0, -3,  0]]]]

   @endverbatim
 *
 * @par Reference
 * - github.com/open-mmlab/mmcv/blob/master/mmcv/ops/border_align.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpBorderAlignBackward(mluOpHandle_t handle,
                         const mluOpTensorDescriptor_t grad_output_desc,
                         const void *grad_output,
                         const mluOpTensorDescriptor_t boxes_desc,
                         const void *boxes,
                         const mluOpTensorDescriptor_t argmax_idx_desc,
                         const void *argmax_idx,
                         const int32_t pool_size,
                         const mluOpTensorDescriptor_t grad_input_desc,
                         void *grad_input);

// Group: SparseConv
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as
 * an extra workspace to optimize the indice convolution backward data operation.
 *
 * The size of extra workspace is based on the given information of the indice
 * convolution backward data operation, including the input descriptor
 * \b input_grad_desc, the filter descriptor \b filter_desc, the indice pairs
 * descriptor \b indice_pairs_desc, the output descriptor \b indice_pairs_desc,
 * the array \b indice_num, and the scaler \b inverse. For more information
 * about the workspace, see "Cambricon MLU-OPS User Guide".
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpIndiceConvolutionBackwardData. For detailed
 * information, see ::mluOpHandle_t.
 * @param[in] output_grad_desc
 * The descriptor of the tensor \b output_grad. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] filters_desc
 * The descriptor of the tensor \b filters. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] indice_pairs_desc
 * The descriptor of the tensor \b indice_pairs. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] input_grad_desc
 * The descriptor of the tensor \b input_grad. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] indice_num
 * The array to describe the valid data number in the indice_pairs tensor.
 * @param[in] inverse
 * The parameter to describe whether the operation performs deconvolution logic.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that is used in
 * ::mluOpIndiceConvolutionBackwardData.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - This function must be called before ::mluOpIndiceConvolutionBackwardData.
 * - ::mluOpCreateTensorDescriptor and ::mluOpSetTensorDescriptor
 *   create and set the tensor descriptor \b output_grad_desc, \b filters_desc,
 *   \b indice_pairs_desc and \b input_grad_desc before this function is called.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetIndiceConvolutionBackwardDataWorkspaceSize(mluOpHandle_t handle,
                                                   const mluOpTensorDescriptor_t output_grad_desc,
                                                   const mluOpTensorDescriptor_t filters_desc,
                                                   const mluOpTensorDescriptor_t indice_pairs_desc,
                                                   const mluOpTensorDescriptor_t input_grad_desc,
                                                   const int64_t indice_num[],
                                                   const int64_t inverse,
                                                   size_t *workspace_size);

// Group: SparseConv
/*!
 * @brief Performs the back propagation of an indice convolution operation to
 * compute the gradient of input \b input_grad based on the gradient of response
 * \b output_grad, the filter tensor \b filter, the indice tensor \b indice_pairs,
 * and helper parameters: array \b indice_num, scaler \b inverse and \b sub_m.
 *
 * The tensors \b input_grad and \b output_grad are reordered from origin input
 * gradient and output gradient. The tensor \b indice_pairs describes the
 * calculation logic between the tensors \b input_grad and \b output_grad.
 * Every pair in \b indice_pairs points to part of \b filters and the single
 * line in \b input_grad and \b output_grad. Every single line in \b input_grad
 * is calculated from matmul calculation logic with data mentioned above.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpIndiceConvolutionBackwardData. For detailed
 * information, see ::mluOpHandle_t.
 * @param[in] output_grad_desc
 * The descriptor of input data \b output_grad, which contains dimension, data
 * type, and data layout.
 * @param[in] output_grad
 * Pointer to the MLU memory that stores the \b output_grad tensor. It is the
 * gradient data of output tensor after reordered.
 * @param[in] filters_desc
 * The descriptor of input data \b filters, which contains dimension, data type,
 * and data layout. It contains N, H, W, and C information when it is a 4D
 * tensor, or N, D, H, W and C information when it is a 5D tensor.
 * @param[in] filters
 * Pointer to the MLU memory that stores the filter tensor.
 * @param[in] indice_pairs_desc
 * The descriptor of input data \b indice_pairs, which contains dimension,
 * data type, and data layout.
 * @param[in] indice_pairs
 * Pointer to the MLU memory that stores the \b indice_pairs tensor. It is used to
 * specify the calculation pairs between \b input_grad and \b output_grad.
 * @param[in] indice_num
 * The array to describe the valid data number in \b indice_pairs.
 * @param[in] inverse
 * The parameter to describe whether the operation performs deconvolution logic.
 * @param[in] sub_m
 * The parameter to describe whether the operation performs sub_m convolution
 * logic or sparce convolution logic.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for
 * ::mluOpIndiceConvolutionBackwardData. For more information about
 * workspace, see "Cambricon MLU-OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that is used in
 * ::mluOpIndiceConvolutionBackwardData.
 * @param[in] input_grad_desc
 * The descriptor of the output data \b input_grad, which contains dimension, data
 * type, and data layout.
 * @param[out] input_grad
 * Pointer to the MLU memory that stores the \b output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH,
 *   ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - output_grad tensor: half, float
 *   - filters tensor: half, float
 *   - indice_pairs tensor: int32
 *   - input_grad tensor: half, float
 *
 * - The supported data type of array \b indice_num tensor, scaler \b inverse tensor, and
 *   \b sub_m tensor is int64.
 *
 * @par Data Layout
 * - The supported data layouts of \b output_grad tensor, \b filters tensor, and \b input_grad tensor
 *   are as follows:
 *   The layout MLUOP_LAYOUT_ARRAY of filters tensor is recognized as DHWCN,
 *   which is not defined yet.
 *   - output_grad tensor: MLUOP_LAYOUT_ARRAY
 *   - filters tensor: MLUOP_LAYOUT_ARRAY, MLUOP_LAYOUT_HWCN, MLUOP_LAYOUT_NCHW,
 *     MLUOP_LAYOUT_NHWC, MLUOP_LAYOUT_NCDHW, MLUOP_LAYOUT_NDHWC
 *   - input_grad tensor: MLUOP_LAYOUT_ARRAY
 *
 * @par Scale Limitation
 * - The \b output_grad and \b input_grad are 2D arrays.
 * - The \b indice_pairs is 3D array.
 * - The \b filter is 4D or 5D tensor.
 * - The dims[1] of \b indice_pairs should be equal to 2.
 * - When the \b filter is a 4D tensor, the dims[0] of \b indice_pairs should be
 *   equal to H * W corresponding to filter layout.
 * - When the \b filter is a 5D tensor, the dims[0] of \b indice_pairs should be
 *   equal to D * H * W corresponding to filter layout.
 * - The dims[1] of \b output_grad should be equal to N corresponding to filter layout.
 * - The dims[1] of \b input_grad should be equal to C corresponding to filter layout.
 * - The dims[0] of \b input_grad should be equal to the dims[2] of \b indice_pairs.
 * - Each value in the array \b indice_num should be no smaller than 0, no
 *   larger than the dims[0] of \b output_grad and no larger than the dims[2]
 *   of \b indice_pairs.
 * - The value \b sub_m should be 0 or 1.
 * - The value \b inverse should be 0.
 * - When the value of \b sub_m is 1, the dims D, H and W corresponding to
 *   filter layout should be odd numbers.
 * - When the value of \b sub_m is 1, the dims[0] of \b input_grad and the dims[0] of
 *   \b output_grad should be the same.
 * - When the value of \b sub_m is 1, the middle number of \b indice_num should be the
 *   maximum number of \b indice_num.
 *
 * @par API Dependency
 * - ::mluOpGetIndiceConvolutionBackwardDataWorkspaceSize should
 *   be called to get the extra space size before this function is called.
 *
 * @par Note
 * - When the \b filter is a 5D tensor, the layout MLUOP_LAYOUT_ARRAY represents
 *   the data layout of (D, H, W, C, N).
 * - The length of the array \b indice_num should be equal to the dims[0] of \b indice_pairs.
 * - The data values of \b indice_pairs should be no smaller than 0.
 * - The data values of tensor slices indice_pairs[:,0,:] should be no larger
 *   than the dims[0] of \b input_grad.
 * - The data values of tensor slices indice_pairs[:,1,:] should be no larger
 *   than the dims[0] of \b output_grad.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/v1.6.1/mmcv/ops/csrc/pytorch/cuda/spconv_ops_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpIndiceConvolutionBackwardData(mluOpHandle_t handle,
                                   const mluOpTensorDescriptor_t output_grad_desc,
                                   const void *output_grad,
                                   const mluOpTensorDescriptor_t filters_desc,
                                   const void *filters,
                                   const mluOpTensorDescriptor_t indice_pairs_desc,
                                   const void *indice_pairs,
                                   const int64_t indice_num[],
                                   const int64_t inverse,
                                   const int64_t sub_m,
                                   void *workspace,
                                   const size_t workspace_size,
                                   const mluOpTensorDescriptor_t input_grad_desc,
                                   void *input_grad);

// Group: SparseConv
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra workspace
 * to optimize the indice_convolution_backward_filter operation.
 *
 * The size of extra workspace is based on the given information of the indice_convolution_backward_filter
 * operation, including the input tensor descriptor \b features_desc, \b output_grad_desc, and \b indice_pairs_desc,
 * output tensor descriptor \b filters_grad_desc, and the array \b indice_num[].
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * indice_convolution_backward_filter operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] features_desc
 * The descriptor of the tensor \b features that need convolution. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] output_grad_desc
 * The descriptor of the tensor \b output_grad. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] indice_pairs_desc
 * The descriptor of the tensor \b indice_pairs between input locations and output locations. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] filters_grad_desc
 * The descriptor of the tensor \b filters_grad. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] indice_num
 * Pointer to the host memory that stores the indice pairs number.
 * @param[in] inverse
 * Currently it is not supported and should be set to 0.
 * @param[in] sub_m
 * The sub_m mode of convolution if the value is not 0.
 * @param[out] workspace_size
 * Pointer to the MLU memory that stores the returned size of the extra workspace in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - You need to call ::mluOpCreateTensorDescriptor and ::mluOpSetTensorDescriptor to create and set
 *   tensor descriptors \b features_desc, \b output_grad_desc, \b indice_pairs_desc, and \b filters_grad_desc before
 *   calling this function.
 * - The allocated extra workspace should be passed to ::mluOpIndiceConvolutionBackwardFilter to
 *   perform the indice_convolution_backward_filter operation.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetIndiceConvolutionBackwardFilterWorkspaceSize(mluOpHandle_t handle,
                                                     const mluOpTensorDescriptor_t features_desc,
                                                     const mluOpTensorDescriptor_t output_grad_desc,
                                                     const mluOpTensorDescriptor_t indice_pairs_desc,
                                                     const mluOpTensorDescriptor_t filters_grad_desc,
                                                     const int64_t indice_num[],
                                                     const int64_t inverse,
                                                     const int64_t sub_m,
                                                     size_t *workspace_size);

// Group: SparseConv
/*!
 * @brief Computes the indice_convolution_backward_filter operation, then returns the results in the output
 * tensor \b filters_grad.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * indice_convolution_backward_filter operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] features_desc
 * The descriptor of the tensor \b features that need convolution. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] features
 * Pointer to the MLU memory that stores the features tensor.
 * @param[in] output_grad_desc
 * The descriptor of the tensor \b output_grad. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] output_grad
 * Pointer to the MLU memory that stores the output grad tensor.
 * @param[in] indice_pairs_desc
 * The descriptor of the tensor \b indice_pairs between inputs locations and outputs locations. For detailed
 information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] indice_pairs
 * Pointer to the MLU memory that stores the indice pairs tensor.
 * @param[in] indice_num
 * Pointer to the host memory that stores the indice pairs num.
 * @param[in] inverse
 * Currently it is not supported and must be set to 0.
 * @param[in] sub_m
 * The sub_m mode of convolution if the value is not 0.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the indice_convolution_backward_filter operation.
 * For more information about workspace, see "Cambricon MLU-OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes.
 * @param[in] filters_grad_desc
 * The descriptor of the tensor \b filters_grad. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] filters_grad
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par Data Type
 * - This function supports the combinations of the following data types for
 *   input tensor \b features, \b output_grad, \b indice_pairs_num, and output tensor \b filters_grad.
 *   - \b features, \b output_grad, \b indice_pairs, \b filters_grad data type: half, half, int32, half
 *   - \b features, \b output_grad, \b indice_pairs, \b filters_grad data type: float, float, int32, float
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The input tensor and output tensor must meet the following requirements:
 * - The \b features and \b output_grad must be two dimensions.
 * - The \b indice_pairs must be three dimensions, and the first dimension value must be equal to the
 *   kernel size of \b filters_grad, the second dimension must be 2, and the last dimension must be
 *   equal to the number of \b features first dimension.
 * - The \b filters_grad must be four or five dimensions. The last dimension of \b filters_grad must
 *   be equal to the last dimension of \b output_grad, and the penultimate dimension of \b filters_grad
 *   must be equal to the last dimension of \b features.
 * - The array length of indice_num must be equal to the first dimension of \b indice_pairs.
 *
 * @par API Dependency
 * - Before calling this function to implement matrix multiplication, you need to prepare
 *   all the parameters passed to this function. See each parameter description for details.
 *
 * @par Note
 * - This function is only supported on MLU300 series or above platforms.
 * - This function does not support setting tensor onchip data type with fixed-point type.
 *
 * @par Example
 * - The example of the operation is as follows:
     @verbatim
      Dimension of features tensor:  [in_active_num, ci]
      Dimension of output_grad tensor:  [output_active_num, co]
      Dimension of indice_pairs tensor: [kd * kh * kw, 2, in_active_num]
      Dimension of filters_grad tensor: [kd, kh, kw, ci, co]
     @endverbatim
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/cuda/spconv_ops_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpIndiceConvolutionBackwardFilter(mluOpHandle_t handle,
                                     const mluOpTensorDescriptor_t features_desc,
                                     const void *features,
                                     const mluOpTensorDescriptor_t output_grad_desc,
                                     const void *output_grad,
                                     const mluOpTensorDescriptor_t indice_pairs_desc,
                                     const void *indice_pairs,
                                     const int64_t indice_num[],
                                     const int64_t inverse,
                                     const int64_t sub_m,
                                     void *workspace,
                                     const size_t workspace_size,
                                     const mluOpTensorDescriptor_t filters_grad_desc,
                                     void *filters_grad);

// Group: RoiPointPool3d
/*!
 * @brief Returns in \b size the size of the MLU memory in bytes that is used as
 * an extra workspace to optimize ::mluOpRoiPointPool3d.
 *
 * The size of extra workspace is based on the given information of the input
 * and output tensor descriptors, including \b points_desc , \b point_features_desc ,
 * \b boxes3d_desc , \b pooled_features_desc , and \b pooled_empty_flag_desc.
 * For more information about the workspace, see "Cambricon MLU-OPS User Guide".
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues
 * in ::mluOpRoiPointPool3d. For detailed information, see ::mluOpHandle_t.
 * @param[in] batch_size
 * The number of batches of the input.
 * @param[in] pts_num
 * The number of points of the input.
 * @param[in] boxes_num
 * The number of boxes of the input.
 * @param[in] feature_in_len
 * The number of features of the input.
 * @param[in] sampled_pts_num
 * The number of sampled points of the input.
 * @param[in] points_desc
 * The descriptor of the first input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] point_features_desc
 * The descriptor of the second input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] boxes3d_desc
 * The descriptor of the third input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] pooled_features_desc
 * The descriptor of the first output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] pooled_empty_flag_desc
 * The descriptor of the second output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] size
 * A host pointer to the returned size of the extra workspace in bytes that is
 * used in ::mluOpRoiPointPool3d.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None
 *
 * @par Data Layout
 * - None
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - The allocated extra workspace should be passed to ::mluOpRoiPointPool3d.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetRoiPointPool3dWorkspaceSize(mluOpHandle_t handle,
                                    const int batch_size,
                                    const int pts_num,
                                    const int boxes_num,
                                    const int feature_in_len,
                                    const int sampled_pts_num,
                                    const mluOpTensorDescriptor_t points_desc,
                                    const mluOpTensorDescriptor_t point_features_desc,
                                    const mluOpTensorDescriptor_t boxes3d_desc,
                                    const mluOpTensorDescriptor_t pooled_features_desc,
                                    const mluOpTensorDescriptor_t pooled_empty_flag_desc,
                                    size_t *size);

// Group: RoiPointPool3d
/*!
 * @brief Implements a linear interpolation of two tensors \b a and \b b based on
 * a scalar or tensor \b w and returns the results in \b d tensor.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues
 * in ::mluOpRoiPointPool3d. For detailed information, see ::mluOpHandle_t.
 * @param[in] batch_size
 * The number of batches of the input.
 * @param[in] pts_num
 * The number of points of the input.
 * @param[in] boxes_num
 * The number of boxes of the input.
 * @param[in] feature_in_len
 * The number of features of the input.
 * @param[in] sampled_pts_num
 * The number of sampled points of the input.
 * @param[in] points_desc
 * The descriptor of the first input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] points
 * Pointer to the MLU memory that stores the first input tensor.
 * @param[in] point_features_desc
 * The descriptor of the second input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] point_features
 * Pointer to the MLU memory that stores the second input tensor.
 * @param[in] boxes3d_desc
 * The descriptor of the third input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] boxes3d
 * Pointer to the MLU memory that stores the third input tensor.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the
 * ::mluOpRoiPointPool3d operation. For more information about workspace,
 * see "Cambricon MLU-OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in
 * ::mluOpRoiPointPool3d. You can get the size of the workspace with
 * ::mluOpGetRoiPointPool3dWorkspaceSize.
 * @param[in] pooled_features_desc
 * The descriptor of the first output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] pooled_features
 * Pointer to the MLU memory that stores the first output tensor.
 * @param[in] pooled_empty_flag_desc
 * The descriptor of the second output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] pooled_empty_flag
 * Pointer to the MLU memory that stores the second output tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types for input and output are as follows:
 *   Note that the data type of \b points, \b point_features, \b boxes3d , and
 *   \b pooled_features must be the same.
 *   - points: half, float
 *   - point_features: half, float
 *   - boxes3d: half, float
 *   - pooled_features: half, float
 *   - pooled_empty_flag: int32
 *
 * @par Data Layout
 * - None
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - Before calling this function to perform ::mluOpRoiPointPool3d, you need to
 *   get the size of workspace by ::mluOpGetRoiPointPool3dWorkspaceSize.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/v1.5.1/mmcv/ops/csrc/pytorch/cuda/roipoint_pool3d_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRoiPointPool3d(mluOpHandle_t handle,
                    const int batch_size,
                    const int pts_num,
                    const int boxes_num,
                    const int feature_in_len,
                    const int sampled_pts_num,
                    const mluOpTensorDescriptor_t points_desc,
                    const void *points,
                    const mluOpTensorDescriptor_t point_features_desc,
                    const void *point_features,
                    const mluOpTensorDescriptor_t boxes3d_desc,
                    const void *boxes3d,
                    void *workspace,
                    size_t workspace_size,
                    const mluOpTensorDescriptor_t pooled_features_desc,
                    void *pooled_features,
                    const mluOpTensorDescriptor_t pooled_empty_flag_desc,
                    void *pooled_empty_flag);

// Group: ThreeNN
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra
 * workspace to optimize ::mluOpThreeNNForward. The size of the extra workspace is
 * based on the given information of ::mluOpThreeNNForward, including the input
 * tensor descriptor \b known_desc. For more information about the workspace, see
 * "Cambricon MLU-OPS User Guide".
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpThreeNNForward. For detailed information, see ::mluOpHandle_t.
 * @param[in] known_desc
 * The descriptor of input data \b known, which contains dimension, data type, and data layout.
 * @param[out] workspace_size
 * A host pointer to the returned size of the extra workspace in bytes that is used in
 * ::mluOpThreeNNForward.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - This function must be called before ::mluOpThreeNNForward.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetThreeNNForwardWorkspaceSize(const mluOpHandle_t handle,
                                    const mluOpTensorDescriptor_t known_desc,
                                    size_t *workspace_size);

// Group: ThreeNN
/*!
 * @brief Finds the closest 3 points of \b unknown among \b known, and outputs \b dist and index
 * \b idx tensor. This function firstly computes dist of each known point to a unknown point, and
 * finds the closest 3 points, and outputs the dist and index of the known point in known dataset.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpThreeNNForward. For detailed information, see ::mluOpHandle_t.
 * @param[in] unknown_desc
 * The descriptor of input data \b unknown, which contains dimension, data type, and data layout.
 * @param[in] unknown
 * Pointer to the MLU memory that stores the unknown tensor.
 * @param[in] known_desc
 * The descriptor of input data \b known, which contains dimension, data type, and data layout.
 * @param[in] known
 * Pointer to the MLU memory that stores the known tensor.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for ::mluOpThreeNNForward.
 * For more information about workspace, see "Cambricon MLU-OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that is used in ::mluOpThreeNNForward.
 * @param[in] dist2_desc
 * The descriptor of the output data \b dist2, which contains dimension, data type, and data layout.
 * @param[out] dist2
 * Pointer to the MLU memory that stores the \b dist2 tensor.
 * @param[in] idx_desc
 * The descriptor of the output data \b idx, which contains dimension, data type, and data layout.
 * @param[out] idx
 * Pointer to the MLU memory that stores the \b idx tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types for unknown tensor \b unknown, known tensor \b known, dist2
 *   tensor \b dist2 and idx tensor \b idx. Data types of unknown tensor, known tensor and
 *   dist2 should be the same.
 *   - unknown tensor: half, float
 *   - known tensor: half, float
 *   - known tensor: half, float
 *   - idx tensor: int32
 *
 * @par Data Layout
 * - The supported data layouts of \b unknown tensor, \b known tensor, \b dist2 tensor, and \b idx tensor:
 *   \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - The shape of \b unknown, \b dist2 and \b idx should be [b, n, 3].
 * - The shape of \b known should be [b, m, 3].
 * - The shape of \b unknown , \b dist2 , \b idx, and \b known dims[0](b) should be equal.
 * - The shape of \b unknown , \b dist2 , \b idx, and \b known dims[2](3) should be equal to 3.
 * - The shape of \b unknown , \b dist2 , \b idx, and \b known dims[1](n) should be equal and larger
 *   than 0.
 *
 * @par API Dependency
 * - Before calling this function you need to call ::mluOpGetThreeNNForwardWorkspaceSize
 *   to get the extra space size needed in ::mluOpThreeNNForward.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/v1.5.2/mmcv/ops/csrc/pytorch/cuda/three_nn_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpThreeNNForward(const mluOpHandle_t handle,
                    const mluOpTensorDescriptor_t unknown_desc,
                    const void *unknown,
                    const mluOpTensorDescriptor_t known_desc,
                    const void *known,
                    void *workspace,
                    const size_t workspace_size,
                    const mluOpTensorDescriptor_t dist2_desc,
                    void *dist2,
                    const mluOpTensorDescriptor_t idx_desc,
                    void *idx);

// Group: SparseConv
/*!
 * @brief Returns in \b workspace_size of the MLU memory which is used as an extra workspace
 * to boost up indice_convolution_forward computation.
 *
 * The size of workspace is deduced from the input including input tensor descriptor
 * \b features_desc , \b filters_desc , \b indice_pairs_desc , output tensor descriptor
 * \b features_out_desc , and array indice_num[].
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * indice_convolution_forward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] features_desc
 * The descriptor of the tensor \b features. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] filters_desc
 * The descriptor of the tensor \b filters. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] indice_pairs_desc
 * The descriptor of the tensor \b indice_pair of features_in and filters.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] features_out_desc
 * The descriptor of the tensor \b features_out. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] indice_num
 * Pointer to the host memory that stores the indice pairs number.
 * @param[in] num_act_out
 * The number of non-zero element in output sparse tensor.
 * @param[in] inverse
 * Currently, it is not supported and should be set to 0.
 * @param[in] sub_m
 * The sub_m mode of convolution if the value is not 0.
 * @param[out] workspace_size
 * Pointer to the MLU memory that stores the returned size of the extra workspace in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH,
 *   ::MLUOP_STATUS_INTERNAL_ERROR, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - Call ::mluOpCreateTensorDescriptor and ::mluOpSetTensorDescriptor before this function
 *   to create and set tensor descriptor \b features_desc , \b filters_desc , \b indice_pairs_desc ,
 *   and \b features_out_desc.
 * - Output \b workspace_size should later be passed to ::mluOpIndiceConvolutionForward
 *   to complete computation.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetIndiceConvolutionForwardWorkspaceSize(mluOpHandle_t handle,
                                              const mluOpTensorDescriptor_t features_desc,
                                              const mluOpTensorDescriptor_t filters_desc,
                                              const mluOpTensorDescriptor_t indice_pairs_desc,
                                              const mluOpTensorDescriptor_t features_out_desc,
                                              const int64_t indice_num[],
                                              const int64_t num_act_out,
                                              const int64_t inverse,
                                              const int64_t sub_m,
                                              size_t *workspace_size);

// Group: SparseConv
/*!
 * @brief Performs convolution on input sparse tensor \b features with kernel \b filters,
 * then returns the output sparse tensor \b features_out.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * indice_convolution_forward operation. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in] features_desc
 * The descriptor of features that needs convolution. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] features
 * Pointer to the MLU memory that stores the features tensor.
 * @param[in] filters_desc
 * The descriptor of filters that convolves input. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] filters
 * Pointer to the MLU memory that stores the convolution kernel.
 * @param[in] indice_pairs_desc
 * The descriptor of the tensor \b indice_pairs of input indices and filters location.
 * For detailed informationm, see ::mluOpTensorDescriptor_t.
 * @param[in] indice_pairs
 * Pointer to the MLU memory that stores the indice pairs tensor.
 * @param[in] indice_num
 * Pointer to the host memory that stores the indice pairs number.
 * @param[in] num_act_out
 * The number of non-zero element in output sparse tensor.
 * @param[in] inverse
 * Currently it is not supported and should be set to 0.
 * @param[in] sub_m
 * The sub_m mode of convolution if the value is not 0.
 * @param[in] workspace
 * Pointer to the MLU memory that stores temporary tensor and extra computation space.
 * For more information about workspace, see "Cambricon MLU-OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes.
 * @param[in] features_out_desc
 * The descriptor of the the tensor \b features_out. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] features_out
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH,
 *   ::MLUOP_STATUS_INTERNAL_ERROR, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - This function supports the combination of the following data types:
 *   - input tensor \b features, \b filters, \b indice_pairs, and output tensor \b features_out: half, half, int32,
 * half
 *   - input tensor \b features, \b filters, \b indice_pairs, and output tensor \b features_out: float, float, int32,
 * float
 * - The supported data type of array \b indice_num , scalar \b inverse , and \b sub_m is int64.
 *
 * @par Data Layout
 * - This function supports the following tensor layouts:
 *   - features tensor: MLUOP_LAYOUT_ARRAY
 *   - filters tensor: MLUOP_LAYOUT_NDHWC, MLUOP_LAYOUT_NCDHW, MLUOP_LAYOUT_ARRAY
 *   - indice_pairs tensor: MLUOP_LAYOUT_ARRAY
 *   - features_out tensor: MLUOP_LAYOUT_ARRAY
 *
 * @par Scale Limitation
 * - The \b features and \b features_out are 2D tensor.
 * - The \b filters is 5D tensor.
 * - The \b indice_pairs is 3D tensor.
 * - The dims[1] of \b features equals to input channel of \b filters.
 * - The dims[1] of \b features_out equals to output channel of \b filters.
 * - The dims[0] of \b indice_pairs equals to D * H * W of \b filters.
 * - The dims[1] of \b indice_pairs equals to 2.
 * - The dims[2] of \b indice_pairs equals to dims[0] of \b features.
 * - The length of \b indice_num equals to D * H * W of \b filters.
 * - Values in \b indice_num should be no smaller than 0, no larger
 *   than dims[0] of \b features.
 * - The dims[0] of \b features_out equals to num_act_out.
 * - The value of \b inverse and \b sub_m should be 0 or 1.
 *
 * @par API Dependency
 * - ::mluOpGetIndiceConvolutionForwardWorkspaceSize should be
 *   called before this function to get extra space size.
 *
 * @par Note
 * - This function is only supported on MLU300 series or above platforms.
 * - This function does not support tensor onchip data type with fixed-point type.
 * - The input indices in \b indice_pairs tensor should be no larger than dims[0]
 *   of \b features. Such value is illegal and not checked, the output result is
 *   not guaranteed.
 * - The output indices in \b indice_pairs tensor should be no larger than dims[0]
 *   of \b features_out. Such value is illegal and not checked, the output result is
 *   not guaranteed.
 * - The input indices used to generate \b indice_pairs tensor should not point to
 *   the same location of \b features. Such value is illegal and not checked, the
 *   output result is not guaranteed.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/v1.6.1/mmcv/ops/csrc/pytorch/cuda/spconv_ops_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpIndiceConvolutionForward(mluOpHandle_t handle,
                              const mluOpTensorDescriptor_t features_desc,
                              const void *features,
                              const mluOpTensorDescriptor_t filters_desc,
                              const void *filters,
                              const mluOpTensorDescriptor_t indice_pairs_desc,
                              const void *indice_pairs,
                              const int64_t indice_num[],
                              const int64_t num_act_out,
                              const int64_t inverse,
                              const int64_t sub_m,
                              void *workspace,
                              size_t workspace_size,
                              const mluOpTensorDescriptor_t features_out_desc,
                              void *features_out);

// Group: MoeDispatch
/*!
 * @brief Dispatches the order of \b input tensor, and returns the
 * results in the output tensor \b dispatch in the MoE algorithm.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpMoeDispatchForward. For detailed information, see ::mluOpHandle_t.
 * @param[in] gates_desc
 * The descriptor of the tensor \b gates, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] gates
 * Pointer to the MLU memory that stores the \b gates tensor.
 * @param[in] indices_desc
 * The descriptor of the tensor \b indices, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] indices
 * Pointer to the MLU memory that stores the \b indices tensor.
 * @param[in] locations_desc
 * The descriptor of the tensor \b locations, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] locations
 * Pointer to the MLU memory that stores the \b locations tensor.
 * @param[in] input_desc
 * The descriptor of the tensor \b input, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the \b input tensor.
 * @param[in] samples
 * The number of elements in the \b gates tensor, the \b indices tensor, and the \b locations tensor.
 * @param[in] capacity
 * The maximum number of inputs that experts can process.
 * @param[in] hidden
 * The vector length of a single token.
 * @param[in] num_experts
 * The number of experts.
 * @param[in] dispatch_desc
 * The descriptor of \b dispatch tensor, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] dispatch
 * Pointer to the MLU memory that stores the \b dispatch tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - This function supports the following data types for input tensors \b gates, \b indices,
 *   \b locations, \b input , and \b dispatch.
 *   - gates tensor: float
 *   - indices tensor: int32
 *   - locations tensor: int32
 *   - input tensor: float
 *   - dispatch tensor: float
 *
 * @par Data Layout
 * - The supported layout of the input tensors and output tensors must be \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - The first dimension of \b gates tensor, \b indices tensor, \b locations tensor, and \b input
 *   tensor must be the same size and equal to \b samples.
 * - The second dimension of \b input tensor and \b dispatch tensor must be equal to \b hidden .
 * - The first dimension of \b dispatch tensor must be equal to the multiplication result of
 *   the \b capacity and \b num_experts.
 * - The samples must be less than or equal to the multiplication result of the \b capacity and \b
 *   num_experts.
 * - The values of indices must be between 0 and (num_experts-1) .
 * - The values of locations must be between 0 and (capacity-1) .
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - This function is only supported on MLU300 series or above platforms.
 * - The parameters \b samples, \b capacity , \b hidden , and \b num_experts should not be negative.
 *
 * @par Example
 * - The example of the function is as follows:
     @verbatim
      Dimension of gates tensor:  [samples]
      Dimension of indices tensor:  [samples]
      Dimension of locations tensor:  [samples]
      Dimension of input tensor: [samples, hidden]
      Dimension of dispatch tensor: [num_experts * capacity, hidden]
     @endverbatim
 *
 * @par Reference
 * - https://github.com/microsoft/tutel/blob/v0.2.0/tutel/jit_kernels/sparse.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpMoeDispatchForward(mluOpHandle_t handle,
                        const mluOpTensorDescriptor_t gates_desc,
                        const void *gates,
                        const mluOpTensorDescriptor_t indices_desc,
                        const void *indices,
                        const mluOpTensorDescriptor_t locations_desc,
                        const void *locations,
                        const mluOpTensorDescriptor_t input_desc,
                        const void *input,
                        const int samples,
                        const int capacity,
                        const int hidden,
                        const int num_experts,
                        const mluOpTensorDescriptor_t dispatch_desc,
                        void *dispatch);

// Group: MoeDispatch
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra workspace
 * to optimize the moe_dispatch_backward_gate operation.
 *
 * The size of extra workspace is based on the given information of the moe_dispatch_backward_gate
 * operation, including the input tensor descriptor \b input_desc.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * moe_dispatch_backward_gate operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] input_desc
 * The descriptor of the tensor \b input, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] workspace_size
 * Pointer to the MLU memory that stores the returned size of the extra workspace in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_ARCH_MISMATCH
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitations
 * - None.
 *
 * @par API Dependency
 * - The allocated extra workspace should be passed to ::mluOpMoeDispatchBackwardGate.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetMoeDispatchBackwardGateWorkspaceSize(mluOpHandle_t handle,
                                             const mluOpTensorDescriptor_t input_desc,
                                             size_t *workspace_size);

// Group: MoeDispatch
/*!
 * @brief Calculates the inverse gradient of \b gates tensor, and returns the results in the output
 * tensor \b grad_gates.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * moe_dispatch_backward_gate operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] indices_desc
 * The descriptor of the tensor \b indices, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] indices
 * Pointer to the MLU memory that stores the \b indices tensor.
 * @param[in] locations_desc
 * The descriptor of the tensor \b locations, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] locations
 * Pointer to the MLU memory that stores the \b locations tensor.
 * @param[in] input_desc
 * The descriptor of the tensor \b input, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] dispatch_desc
 * The descriptor of the tensor \b dispatch, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] dispatch
 * Pointer to the MLU memory that stores the \b dispatch tensor.
 * @param[in] samples
 * The number of elements in the \b indices tensor, \b locations tensor, and \b grad_gates tensor.
 * @param[in] capacity
 * The maximum number of inputs that experts can process.
 * @param[in] hidden
 * The vector length of a single token.
 * @param[in] num_experts
 * The number of experts.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the moe_dispatch_backward_gate operation.
 * For more information about workspace, see "Cambricon MLU-OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes.
 * @param[in] grad_gates_desc
 * The descriptor of the tensor \b grad_gates, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] grad_gates
 * Pointer to the MLU memory that stores the \b grad_gates tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - indices tensor: int32
 *   - locations tensor: int32
 *   - input tensor: float
 *   - dispatch tensor: float
 *   - grad_gates tensor: float
 *
 * @par Data Layout
 * - The supported layout of the input tensors and output tensors must be \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - The first dimension of \b indices tensor, \b locations tensor, \b input tensor, and \b grad_gates
 *   tensor must be the same size and equal to \b samples.
 * - The second dimension of \b input tensor and \b dispatch tensor must be equal to \b hidden.
 * - The first dimension of \b dispatch tensor must be equal to the multiplication result of
 *   the \b capacity and \b num_experts.
 * - The value of the input parameters \b samples, \b capacity , \b hidden , and \b num_experts
 *   must be greater than or equal to 0.
 * - The value range of the input parameter \b indices tensor must be greater than or equal to 0 and less than
 *   \b num_experts.
 * - The value range of the input parameter \b locations tensor must be greater than or equal to 0 and less than
 *   \b capacity.
 *
 * @par API Dependency
 * - Before calling this function to perform ::mluOpMoeDispatchBackwardGate, you need to get
 *   the size of workspace by ::mluOpGetMoeDispatchBackwardGateWorkspaceSize.
 *
 * @par Note
 * - This function is only supported on MLU300 series or above platforms.
 * - The parameters \b samples, \b capacity , \b hidden , and \b num_experts should not be negative.
 *
 * @par Example
 * - The example of the operation is as follows:
     @verbatim
      Dimension of indices tensor:  [samples]
      Dimension of locations tensor:  [samples]
      Dimension of input tensor: [samples, hidden]
      Dimension of dispatch tensor: [num_experts * capacity, hidden]
      Dimension of grad_gates tensor: [samples]
     @endverbatim
 *
 * @par Reference
 * - https://github.com/microsoft/tutel/blob/v0.2.0/tutel/jit_kernels/sparse.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpMoeDispatchBackwardGate(mluOpHandle_t handle,
                             const mluOpTensorDescriptor_t indices_desc,
                             const void *indices,
                             const mluOpTensorDescriptor_t locations_desc,
                             const void *locations,
                             const mluOpTensorDescriptor_t input_desc,
                             const void *input,
                             const mluOpTensorDescriptor_t dispatch_desc,
                             const void *dispatch,
                             const int samples,
                             const int capacity,
                             const int hidden,
                             const int num_experts,
                             void *workspace,
                             const size_t workspace_size,
                             const mluOpTensorDescriptor_t grad_gates_desc,
                             void *grad_gates);

// Group: PointsInBoxes
/*!
 * @brief Detects the first 3D box that each point belongs to in given points cloud data.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * points_in_boxes operation. For detailed information, see ::mluOpHandle_t.
 *
 * @param[in] points_desc
 * The descriptor of input tensor \b points, which contains dimension, data type and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 *
 * @param[in] points
 * Pointer to the MLU memory that stores the \b points tensor.
 *
 * @param[in] boxes_desc
 * The descriptor of input tensor \b boxes, which contains dimension, data type and data layout.
 *
 * @param[in] boxes
 * Pointer to the MLU memory that stores the \b boxes tensor.
 *
 * @param[out] points_indices_desc
 * The descriptor of input tensor \b points_indices, which contains dimension, data type and data layout.
 *
 * @param[out] points_indices
 * Pointer to the MLU memory that stores the \b points_indices tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - points tensor: float
 *   - boxes tensor: float
 *   - points_indices tensor: int32
 *
 * @par Data Layout
 * - The supported layout of input and output tensors must be \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - On MLU370, the number of boxes cannot exceed 23404;
 *   On series higher than MLU300 series, the number of boxes cannot exceed 14042.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - Differences between MLU and CPU/GPU may occur when the point is on the edge of the box.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/
 *   ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpPointsInBoxes(mluOpHandle_t handle,
                   const mluOpTensorDescriptor_t points_desc,
                   const void *points,
                   const mluOpTensorDescriptor_t boxes_desc,
                   const void *boxes,
                   const mluOpTensorDescriptor_t points_indices_desc,
                   void *points_indices);

// Group: RoiAlign
/*!
 * @brief Computes the gradients of images \b grads_image using the gradients \b grads and
 * bounding boxes \b boxes to perform the backpropagation of ::mluOpRoiAlignForward_v2
 * function.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * the roi_align_backward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] grads_desc
 * The descriptor of the tensor \b grads in the backpropagation process. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[in] grads
 * Pointer to the MLU memory that stores the gradient tensor.
 * @param[in] boxes_desc
 * The descriptor of the tensor \b boxes. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] boxes
 * Pointer to the MLU memory that stores the bounding boxes tensor.
 * @param[in] spatial_scale
 * A scaling factor that specifies how to map the box coordinates in the origin image to
 * the coordinates in the output.
 * @param[in] sampling_ratio
 * The number of sampling points in the grid used to compute the output.
 * @param[in] aligned
 * A Boolean value which determines whether to shift the boxes by 0.5 pixel.
 * @param[in] grads_image_desc
 * The descriptor of the tensor \b grads_image of the original images.
 * @param[out] grads_image
 * Pointer to the MLU memory that stores the gradients tensor of the original images.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of all tensors should be the same.
 *   The supported data types of input and output tensors are as follows:
 *   - gradient tensor: half, float
 *   - boxes tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b grads, \b boxes, and \b grads_images are as follows:
 *   - grads tensor: \p MLUOP_LAYOUT_NHWC
 *   - boxes tensor: \p MLUOP_LAYOUT_ARRAY, which only supports 2D tensor.
 *   - grads_image tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The gradient tensor and output tensor must have four dimensions.
 * - The size of the fourth dimension of gradient tensor and output tensor must be the same.
 * - The bounding boxes tensor \b boxes must have two dimensions.
 * - The size of the first dimension of gradient tensor and bounding boxes tensor must be the same.
 * - The shape of \b boxes should be [boxes_num, 5].
 * - \b boxes[i] consists of [image_id, x1, y1, x2, y2]. \p image_id specifies which image this box
 *   is in, and should be in the range of [0, batch_num - 1]. \p x1 and \p y1 specify the start
 *   coordinate of this box in origin image. \p x2 and \p y2 specify the end coordinate of this box
 *   in origin image. \p x1 and \p y1 should be greater than or equal to 0. \p x2 should be greater
 *   than \p x1. \p y2 should be greater than \p y1.
 * - \b spatial_scale should be in the range of (0, 1].
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - The example of the roi_align_backward operation is as follows:
     @verbatim
     input two arrays by 1 * 1 * 1 * 1 and 1 * 5 --> grads: [[[[1.0]]]]

     --> boxes: [[0.0, 0.0, 0.0, 1.0, 1.0]]

     param:
         spatial_scale: 1.0, sampling_ratio: 2, aligned: false

     output array by 1 * 2 * 2 * 1 -->
         output: [[[[0.25]], [[0.25]]], [[[0.25]], [[0.25]]]]
     @endverbatim
 *
 * @par Reference
 * - https://pytorch.org/vision/stable/ops.html#torchvision.ops.roi_align
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRoiAlignBackward(mluOpHandle_t handle,
                      const float spatial_scale,
                      const int sampling_ratio,
                      const bool aligned,
                      const mluOpTensorDescriptor_t grads_desc,
                      const void *grads,
                      const mluOpTensorDescriptor_t boxes_desc,
                      const void *boxes,
                      const mluOpTensorDescriptor_t grads_image_desc,
                      void *grads_image);

// Group: RoiAlign
/*!
 * @brief Computes the gradients of images \b grads_image based on the gradients \b grads,
 * bounding boxes \b boxes, the coordinate of x axis \b argmax_x, and the coordinate of y axis
 * \b argmax_y to perform this function. Compared with ::mluOpRoiAlignBackward, in addition to
 * supporting the average pooling mode, ::mluOpRoiAlignBackward_v2 also supports the maximum pooling mode
 * defined in \b pool_mode with two more inputs \b argmax_x and \b argmax_y.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpRoiAlignBackward_v2 function. For detailed information, see ::mluOpHandle_t.
 * @param[in] grads_desc
 * The descriptor of the tensor \b grads in the backpropagation process. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[in] grads
 * Pointer to the MLU memory that stores the gradient tensor.
 * @param[in] boxes_desc
 * The descriptor of the tensor \b boxes. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] boxes
 * Pointer to the MLU memory that stores the bounding boxes tensor.
 * @param[in] argmax_x_desc
 * The descriptor of the \b argmax_x tensor that stores the coordinate of x axis. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[in] argmax_x
 * Pointer to the MLU memory that stores the \b argmax_x tensor. \b argmax_x represents
 * \b output coordinate of x axis returned by ::mluOpRoiAlignForward_v2 when \b pool_mode is maximum
 * pooling mode. When \b pool_mode is average pooling mode, \b argmax_x is NULL.
 * @param[in] argmax_y_desc
 * The descriptor of the \b argmax_y tensor that stores the coordinate of y axis. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[in] argmax_y
 * Pointer to the MLU memory that stores the \b argmax_y tensor. \b argmax_y represents
 * \b output coordinate of y axis returned by ::mluOpRoiAlignForward_v2 when \b pool_mode is maximum
 * pooling mode. When \b pool_mode is average pooling mode, \b argmax_y is NULL.
 * @param[in] spatial_scale
 * A scaling factor that specifies how to map the box coordinates in the original image to
 * the coordinates in the output.
 * @param[in] sampling_ratio
 * The number of sampling points in the grid used to compute the output.
 * @param[in] aligned
 * A Boolean value which determines whether to shift the boxes by 0.5 pixel. If the value
 * of \b aligned is set to true, the boxes are shifted by 0.5. If the value of \b aligned is set
 * to false, the boxes are not shifted.
 * @param[in] pool_mode
 * The pooling mode which determines to use maximum pooling mode or average
 * pooling mode. If the value of \b pool_mode is set to 1, the average pooling mode is used. If
 * the value of \b pool_mode is set to 0, the maximum pooling mode is used.
 * @param[in] grads_image_desc
 * The descriptor of the tensor \b grads_image of the original images. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] grads_image
 * Pointer to the MLU memory that stores the \b grads_image tensor .
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The data types of all tensors should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - gradient tensor: half, float
 *   - boxes tensor: half, float
 *   - argmax_x tensor: half, float
 *   - argmax_y tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of gradient tensor \b grads, boxes tensor \b boxes, argmax_x tensor
 *   \b argmax_x, argmax_y tensor \b argmax_y and output tensor \b grads_images are as follows:
 *   - grads tensor: \p MLUOP_LAYOUT_NHWC
 *   - boxes tensor: \p MLUOP_LAYOUT_ARRAY, which only supports 2D tensor.
 *   - argmax_x tensor: \p MLUOP_LAYOUT_NHWC
 *   - argmax_y tensor: \p MLUOP_LAYOUT_NHWC
 *   - grads_image tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The gradient tensor \b grads, argmax_x tensor \b argmax_x , argmax_y tensor \b argmax_y, and
 *   output tensor \b grads_images must have four dimensions.
 * - The size of the fourth dimension of gradient tensor \b grads, argmax_x tensor \b argmax_x,
 *   argmax_y tensor \b argmax_y, and output tensor \b grads_images must be the same.
 * - The bounding boxes tensor \b boxes must have two dimensions.
 * - The size of the first dimension of gradient tensor \b grads, argmax_x tensor \b argmax_x, argmax_y
 *   tensor \b argmax_y and bounding boxes tensor \b boxes must be the same.
 * - The size of each dimension of gradient tensor \b grads, argmax_x tensor \b argmax_x and argmax_y
 *   tensor \b argmax_y must be the same.
 * - The shape of \b boxes should be [boxes_num, 5].
 * - \b boxes[i] consists of [image_id, x1, y1, x2, y2]. \p image_id specifies which image this box
 *   is in, and should be in the range of [0, batch_num - 1]. \p x1 and \p y1 specify the starting
 *   coordinate of this box in origin image. \p x2 and \p y2 specify the ending coordinate of this box
 *   in origin image. \p x1 and \p y1 should be greater than or equal to 0. \p x2 should be greater
 *   than \p x1. \p y2 should be greater than \p y1.
 *
 * @par API Dependency
 * - This function should be used with ::mluOpRoiAlignForward_v2.
 *
 * @par Note
 * - Set the values of \b argmax_x and \b argmax_y according to the result returned by
 *   ::mluOpRoiAlignForward_v2.
 *
 * @par Example
 * - The example of the RoiAlignBackward_v2 operation is as follows:
     @verbatim
     input four arrays by 1 * 1 * 1 * 1, 1 * 5, 1 * 1 * 1 * 1 and 1 * 1 *1 *1--> grads: [[[[1.0]]]]

     --> boxes: [[0.0, 0.0, 0.0, 1.0, 1.0]]

     --> argmax_x:[[[[0.5]]]]

     --> argmax_y:[[[[0.5]]]]

     param:
         spatial_scale: 1.0, sampling_ratio: 0, aligned: false

     output array by 1 * 1 * 1 * 1 -->
         output: [[[[1.0]]]]
     @endverbatim
 *
 * @par Reference
 * - http://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/cuda/roi_align_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRoiAlignBackward_v2(mluOpHandle_t handle,
                         const mluOpTensorDescriptor_t grads_desc,
                         const void *grads,
                         const mluOpTensorDescriptor_t boxes_desc,
                         const void *boxes,
                         const mluOpTensorDescriptor_t argmax_x_desc,
                         const void *argmax_x,
                         const mluOpTensorDescriptor_t argmax_y_desc,
                         const void *argmax_y,
                         const float spatial_scale,
                         const int sampling_ratio,
                         const bool aligned,
                         const int pool_mode,
                         const mluOpTensorDescriptor_t grads_image_desc,
                         void *grads_image);

// Group: MsDeformAttn
/*!
 * @brief Implements a multi-scale deformable attention module used in Deformable-Detr.
 * For detailed information about Deformable-Detr, see "Deformable DETR: Deformable
 * Transformers for End-to-End Object Detection".
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues
 * in ::mluOpMsDeformAttnForward function. For detailed information, see ::mluOpHandle_t.
 * @param[in] data_value_desc
 * The descriptor of the tensor \b data_value. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] data_value
 * Pointer to the MLU memory that stores the input multi-scale feature maps.
 * @param[in] data_spatial_shapes_desc
 * The descriptor of the tensor \b data_spatial_shapes. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] data_spatial_shapes
 * Pointer to the MLU memory that stores the shapes of multi-scale feature maps.
 * @param[in] data_level_start_index_desc
 * The descriptor of the tensor \b data_level_start_index. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] data_level_start_index
 * Pointer to the MLU memory that stores the feature maps offset in data_value.
 * @param[in] data_sampling_loc_desc
 * The descriptor of the tensor \b data_sampling_loc. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] data_sampling_loc
 * Pointer to the MLU memory that stores the normalized coordinates of sample points.
 * @param[in] data_attn_weight_desc
 * The descriptor of the tensor \b data_attn_weight. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] data_attn_weight
 * Pointer to the MLU memory that stores the attention weight.
 * @param[in] im2col_step
 * The value of im2col_step.
 * @param[in] data_col_desc
 * The descriptor of the tensor \b data_col. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] data_col
 * Pointer to the MLU memory that stores the output deformable attention feature.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - \b data_value: float
 *   - \b data_spatial_shapes: int32
 *   - \b data_level_start_index: int32
 *   - \b data_sampling_loc: float
 *   - \b data_attn_weight: float
 *   - \b data_col: float
 *
 * @par Note
 * - data_value.dims[1] depends on data_spatial_shapes:
 *   \f$data_value.dims[1]=\sum_{l=1}^{L}H_l*W_l\f$
 * - data_level_start_index value depends on data_spatial_shapes:
 *   \f$data_level_start_index[0]=0; data_level_start_index[i]=\sum_{l=1}^{i}H_l*W_l,i=1,...L-1 \f$
 * - The input \b data_sampling_loc with NaN or infinity is not supported.
 *
 * @par Reference
 * - Deformable DETR: Deformable Transformers for End-to-End Object Detection, Xizhou Zhu, 2020.
 * - https://arxiv.org/pdf/2010.04159
 */
mluOpStatus_t MLUOP_WIN_API
mluOpMsDeformAttnForward(mluOpHandle_t handle,
                         const mluOpTensorDescriptor_t data_value_desc,
                         const void *data_value,
                         const mluOpTensorDescriptor_t data_spatial_shapes_desc,
                         const void *data_spatial_shapes,
                         const mluOpTensorDescriptor_t data_level_start_index_desc,
                         const void *data_level_start_index,
                         const mluOpTensorDescriptor_t data_sampling_loc_desc,
                         const void *data_sampling_loc,
                         const mluOpTensorDescriptor_t data_attn_weight_desc,
                         const void *data_attn_weight,
                         const int im2col_step,
                         const mluOpTensorDescriptor_t data_col_desc,
                         void *data_col);

// Group: TinShift
/*!
 * @brief Shifts gradients from \b grad_output according to shift information in \b shifts and stores the
 * result into \b grad_input.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * ::mluOpTinShiftBackward. For detailed information, see ::mluOpHandle_t.
 * @param[in] grad_output_desc
 * The descriptor for the tensor \b grad_output, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] grad_output
 * Pointer to the MLU memory that stores the tensor \b grad_output.
 * @param[in] shifts_desc
 * The descriptor for the tensor \b shifts, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] shifts
 * Pointer to the MLU memory that stores the tensor \b shifts.
 * @param[in] grad_input_desc
 * The descriptor for the tensor \b grad_input, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] grad_input
 * Pointer to the MLU memory that stores the tensor \b grad_input.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - shifts tensor: int32
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - The supported layout of input and output tensors must be \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - The first dimension of tensor \b grad_output and tensor \b shifts must be the same size.
 * - The third dimension of tensor \b grad_output must be multiple of the second dimension of tensor \b shifts.
 *   For example, if the shape of \b grad_output is [N, T, C, HW], the shape of \b shifts is [N, G],
 *   C must be a multiple of G, and C and G cannot be zero.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - http://github.com/open-mmlab/mmcv/tree/master/mmcv/ops/tin_shift.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpTinShiftBackward(mluOpHandle_t handle,
                      const mluOpTensorDescriptor_t grad_output_desc,
                      const void *grad_output,
                      const mluOpTensorDescriptor_t shifts_desc,
                      const void *shifts,
                      const mluOpTensorDescriptor_t grad_input_desc,
                      void *grad_input);

// Group: TinShift
/*!
 * @brief Shifts datas from \b input according to shift information in \b shifts and stores the
 * result into \b output.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * ::mluOpTinShiftForward. For detailed information, see ::mluOpHandle_t.
 * @param[in] input_desc
 * The descriptor for the tensor \b input, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the tensor \b input.
 * @param[in] shifts_desc
 * The descriptor for the tensor \b shifts, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] shifts
 * Pointer to the MLU memory that stores the tensor \b shifts.
 * @param[in] output_desc
 * The descriptor for the tensor \b output, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the tensor \b output.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - shifts tensor: int32
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - The supported layout of input and output tensors must be \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - The first dimension of tensor \b input and tensor \b shifts must be the same size.
 * - The third dimension of tensor \b input must be multiple of the second dimension of tensor \b shifts.
 *   For example, if the shape of \b input is [N, T, C, HW], the shape of \b shifts is [N, G],
 *   C must be a multiple of G, and C and G cannot be zero.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - http://github.com/open-mmlab/mmcv/tree/master/mmcv/ops/tin_shift.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpTinShiftForward(mluOpHandle_t handle,
                     const mluOpTensorDescriptor_t input_desc,
                     const void *input,
                     const mluOpTensorDescriptor_t shifts_desc,
                     const void *shifts,
                     const mluOpTensorDescriptor_t output_desc,
                     void *output);

// Group: MaskedIm2col
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra workspace to
 * optimize the MaskedCol2imForward operation.
 *
 * The size of the extra workspace is based on the given information of the MaskedCol2imForward operation,
 * including the input tensor descriptor \b col_desc and \b im_desc. For more information about the workspace,
 * see "Cambricon MLU-OPS User Guide".
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * ::mluOpGetMaskedCol2imForwardWorkspaceSize operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] col_desc
 * The descriptor of the tensor \b col. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mask_h_idx_desc
 * The descriptor of the tensor \b mask_h_idx. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mask_w_idx_desc
 * Descriptor of input data \b mask_w_idx. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] im_desc
 * Descriptor of input data \b im. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] workspace_size
 * The size of the extra workspace in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitations
 * - None.
 *
 * @par API Dependency
 * - This function must be called before ::mluOpMaskedCol2imForward.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetMaskedCol2imForwardWorkspaceSize(mluOpHandle_t handle,
                                         const mluOpTensorDescriptor_t col_desc,
                                         const mluOpTensorDescriptor_t mask_h_idx_desc,
                                         const mluOpTensorDescriptor_t mask_w_idx_desc,
                                         const mluOpTensorDescriptor_t im_desc,
                                         size_t *workspace_size);

// Group: MaskedIm2col
/*!
 * @brief  Copies the data of the input tensor \b col to the special coordinates by combining \b mask_h_idx tensor
 * and \b mask_w_idx tensor of output tensor \b im.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * :mluOpMaskedCol2imForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] col_desc
 * The descriptor of the tensor \b col. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] col
 * Pointer to the MLU memory that stores the tensor \b col.
 * @param[in] mask_h_idx_desc
 * The descriptor of the tensor \b mask_h_idx. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mask_h_idx
 * Pointer to the MLU memory that stores the mask_h_idx tensor which contains
 * the coordinates of mask in height direction.
 * @param[in] mask_w_idx_desc
 * Descriptor of input data \b mask_w_idx. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mask_w_idx
 * Pointer to the MLU memory that stores the mask_w_idx tensor which contains
 * the coordinates of mask in width direction.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the MaskedCol2imForward
 * operation. For more information about workspace, see "Cambricon MLU-OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes.
 * @param[in] im_desc
 * The descriptor of the tensor \b im. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] im
 * Pointer to the MLU memory that stores the \b im tensor that is the data copied from \b col tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED,
 *   ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - This function supports the following data types for \b col tensor, \b mask_h_idx tensor,
 *   \b mask_w_idx tensor and \b im tensor.
 *   Data types of col tensor and im tensor must be the same.
 *   - col tensor: half, float
 *   - mask_h_idx tensor: int32_t
 *   - mask_w_idx tensor: int32_t
 *   - im tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b col, \b mask_h_idx, \b mask_w_idx and \b im are as follows:
 *   - col tensor: \p MLUOP_LAYOUT_ARRAY
 *   - mask_h_idx tensor: \p MLUOP_LAYOUT_ARRAY
 *   - mask_w_idx tensor: \p MLUOP_LAYOUT_ARRAY
 *   - im tensor: \p MLUOP_LAYOUT_NCHW
 *
 * @par Scale Limitation
 * - The \b col tensor must be 2D.
 * - The \b mask_h_idx tensor must be 1D.
 * - The \b mask_w_idx tensor must be 1D.
 * - The \b im tensor must be 4D.
 * - The highest dimension of \b im tensor must be 1.
 * - The sizes of the lowest dimension of \b col tensor, the element number of \b mask_h_idx tensor and
 *   the element number of \b mask_w_idx tensor must be the same.
 * - When the element number of \b im tensor equals to zero, this function will return MLUOP_STATUS_BAD_PARAM.
 * - When size of the highest dimension of \b col tensor equals to zero, this function will return
 *   MLUOP_STATUS_BAD_PARAM.
 *
 * @par API Dependency
 * - Before calling this function you need to call ::mluOpGetMaskedCol2imForwardWorkspaceSize
 *   to get the extra space size needed in ::mluOpMaskedCol2imForward operation.
 *
 * @par Note
 * - The data of \b mask_h_idx must be in the range of [0, h - 1], and h is the height of \b im tensor.
 * - The data of \b mask_w_idx must be in the range of [0, w - 1], and w is the width of \b im tensor.
 * - The coordinates by combining \b mask_h_idx tensor and \b mask_w_idx tensor can not be repeated.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/cuda/masked_conv2d_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpMaskedCol2imForward(mluOpHandle_t handle,
                         const mluOpTensorDescriptor_t col_desc,
                         const void *col,
                         const mluOpTensorDescriptor_t mask_h_idx_desc,
                         const void *mask_h_idx,
                         const mluOpTensorDescriptor_t mask_w_idx_desc,
                         const void *mask_w_idx,
                         const size_t workspace_size,
                         void *workspace,
                         const mluOpTensorDescriptor_t im_desc,
                         void *im);

// Group: DiffIouRotatedSortVertices
/*!
 * @brief Sorts the effective vertices of the polygon formed by the intersection of two boxes,
 * and outputs the sorted vertex index.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * diff_iou_rotated_sort_vertices_forward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] vertices_desc
 * The descriptor of input tensor \b vertices, which contains dimension, data type, and data layout.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] vertices
 * Pointer to the MLU memory that stores the tensor \b vertices.
 * @param[in] mask_desc
 * The descriptor of input tensor \b mask, which contains dimension, data type, and data layout.
 * @param[in] mask
 * Pointer to the MLU memory that stores the tensor \b mask.
 * @param[in] num_valid_desc
 * The descriptor of input tensor \b num_valid, which contains dimension, data type, and data layout.
 * @param[in] num_valid
 * Pointer to the MLU memory that stores the tensor \b num_valid.
 * @param[out] idx_desc
 * The descriptor of output tensor \b idx, which contains dimension, data type, and data layout.
 * @param[out] idx
 * Pointer to the MLU memory that stores the tensor \b idx.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - vertices tensor: float
 *   - mask tensor: bool
 *   - num_valid tensor: int32
 *   - idx tensor: int32
 *
 * @par Data Layout
 * - The supported layout of input and output tensors must be \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - The first dimension of \b vertices tensor, \b mask tensor, \b num_valid tensor, and \b idx
 *   tensor must be the same size.
 * - The second dimension of \b vertices tensor, \b mask tensor, \b num_valid tensor, and \b idx
 *   tensor must be the same size.
 * - The third dimension of \b vertices tensor and \b mask tensor must be the same size and equal to 24.
 * - The third dimension of \b idx must be equal to 9.
 * - The fourth dimension of \b vertices must be equal to 2.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/csrc/pytorch/cuda/diff_iou_rotated_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDiffIouRotatedSortVerticesForward(mluOpHandle_t handle,
                                       const mluOpTensorDescriptor_t vertices_desc,
                                       const void *vertices,
                                       const mluOpTensorDescriptor_t mask_desc,
                                       const void *mask,
                                       const mluOpTensorDescriptor_t num_valid_desc,
                                       const void *num_valid,
                                       const mluOpTensorDescriptor_t idx_desc,
                                       void *idx);

// Group: RoiPooling
/*!
 * @brief Generates a fixed size feature map and input feature index
 * of argmax for each ROI (Regions of Interest) to perform ::mluOpRoiPoolingForward operation.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpRoiPoolingForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] pooling_mode
 * The pooling mode of ROI Pooling Forward defined in ::mluOpPoolingMode_t.
 * @param[in] input_desc
 * The descriptor of the input tensor in the roipoolingforward process. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] rois_desc
 * The descriptor of the ROIs tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] rois
 * Pointer to the MLU memory that stores the ROIS tensor.
 * @param[in] spatial_scale
 * The spatial scale of each ROI in the input feature map.
 * @param[in] output_desc
 * The descriptor of the output tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 * @param[out] argmax
 * Pointer to the MLU memory that stores the argmax tensor. This pointer may be NULL. The tensor
 * \b argmax means input feature index of maximum for each ROI.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM.
 *
 * @par Data Type
 * - This function supports the following data types for input tensor \b input, rois tensor \b rois,
 *   output tensor \b output, and argmax tensor \b argmax. The data type of \b input, \b rois, and \b output
 *   must be the same.
 *   - input tensor: half, float.
 *   - rois tensor: half, float.
 *   - output tensor: half, float.
 *   - argmax tensor: int32.
 *
 * @par Data Layout
 * - The supported data layouts for \b input, \b rois, \b output, and \b argmax are as follows:
 *   - input tensor: \b MLUOP_LAYOUT_NHWC.
 *   - ROIs tensor: \b MLUOP_LAYOUT_ARRAY, only supports 2D tensor.
 *   - output tensor: \b MLUOP_LAYOUT_NHWC.
 *   - argmax tensor: \b MLUOP_LAYOUT_NHWC.
 *
 * @par Scale Limitation
 * - The value of \b pooling_mode only supports \p MLUOP_POOLING_MAX.
 * - The input tensor, output tensor, and argmax tensor must have four dimensions.
 * - The size of the lowest dimension of input tensors, output tensor, and argmax tensor must be the same.
 * - The total number of dimensions of ROIs tensor must be 2.
 * - The size of the highest dimension of output tensor and rois tensor must be the same.
 * - The shape of \b rois must be [rois_num, 5]. \b rois_num means the total number of \b rois.
 * - \b Rois[i] consists of [batch_id, x1, y1, x2, y2]. \b batch_id should be in the range of
 *   [0, batch_num - 1]. \b x1 and \b y1 should be greater than or equal to 0. \b x2 should be
 *   greater than \b x1. \b y2 must be greater than \b y1. \b batch_id represents ID of the batch of \b rois. \b x1, \b
 *   y1, \b x2 and \b y2 mean the coordinate values of rois in the input feature map. \b batch_num means the
 *   total number of the batch.
 * - \b Spatial_scale should be in the range of (0, 1].
 * - \b Output consists of [rois_num, pooled_h, pooled_w, channels]. In the dimensions of the h and w of the input
 *   and the output, (\b x2 - \b x1) * (\b y2 - \b y1) * \b spatial_scale * \b spatial_scale / (\b pooled_h * \b
 *   pooled_w) < (nram_limitation / 32). Nram_limitation means the limitation of the nram. On MLU300 series,
 *   the nram_limitation is (163804 - 4 * \b channels) / 2. \b pooled_h means height of output.
 *   \b pooled_w means width of output.
 *
 * @par API Dependency
 * - None
 *
 * @par Note
 * - When the input data or parameter contains NaN or infinity:
 *   - On MLU300 series, if the last value in the kernel of the pooling is NaN, \b argmax is
 *     the index of the last value, \b output is the last value, as shown in example 2 below.
 *     Otherwise, \b argmax is the index of the maximum value after the last NaN,
 *     \b output is the maximum value after the last NaN, as shown in example 3 below.
 *
 * @par Example
 * - The example 1 of the roipoolingforward operation is as follows:
     @verbatim
     input two arrays by 1 * 4 * 4 * 1 and 1 * 5 -->input: [[[1.0],[2.0],[3.0],[4.0]],
                                                            [[5.0],[6.0],[7.0],[8.0]],
                                                            [[9.0],[10.0],[11.0],[12.0]],
                                                            [[13.0],[14.0],[15.0],[16.0]]]

     --> rois: [[1.0, 0.0, 0.0, 3.0, 3.0]]

     params:
         pooling_modek: 0, spatial_scale: 1.0

     output array by 1 * 2 * 2 * 1 --> output: [[[6.0],[8.0]],
                                                [[14.0],[16.0]]]

     argmax array by 1 * 2 * 2 * 1 --> argmax: [[[5],[7]],
                                                [[13],[15]]]
     @endverbatim

   - The example 2 of the roipoolingforward operation is as follows:
     @verbatim
     input two arrays by 1 * 2 * 2 * 1 and 1 * 5 --> input: [[[1.0],[2.0]],
                                                             [[3.0],[NaN]]]

     --> rois: [[1.0, 0.0, 0.0, 1.0, 1.0]]

     params:
         pooling_mode: 0, spatial_scale: 1.0

     output array by 1 * 1 * 1 * 1 --> output: [[[NaN]]]

     argmax array by 1 * 1 * 1 * 1 --> argmax: [[[3]]]
     @endverbatim

   - The example 3 of the roipoolingforward operation is as follows:
     @verbatim
     input two arrays by 1 * 2 * 2 * 1 and 1 * 5 --> input: [[[1.0],[NaN]],
                                                             [[3.0],[4.0]]]

     --> rois: [[1.0, 0.0, 0.0, 1.0, 1.0]]

     params:
         pooling_mode: 0, spatial_scale: 1.0

     output array by 1 * 1 * 1 * 1 --> output: [[[4.0]]]

     argmax array by 1 * 1 * 1 * 1 --> argmax: [[[3]]]
     @endverbatim
 *
 * @par Reference
 * - https://github.com/pytorch/pytorch/caffe2/operators/roi_pool_op.cu
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRoiPoolingForward(mluOpHandle_t handle,
                       mluOpPoolingMode_t pooling_mode,
                       const mluOpTensorDescriptor_t input_desc,
                       const void *input,
                       const mluOpTensorDescriptor_t rois_desc,
                       const void *rois,
                       float spatial_scale,
                       const mluOpTensorDescriptor_t output_desc,
                       void *output,
                       int *argmax);

// Group: RoiPooling
/*!
 * @brief Computes the gradients of image \b grads_image based on the gradients \b grads and
 * region proposals \b rois to perform the backpropagation of ::mluOpRoiPoolingForward operation.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpRoiPoolingBackward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] pooling_mode
 * The pooling mode of ROI Pooling Forward defined in ::mluOpPoolingMode_t.
 * @param[in] grads_desc
 * The descriptor of the gradient tensor in the backpropagation process. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[in] grads
 * Pointer to the MLU memory that stores the gradient tensor.
 * @param[in] rois_desc
 * The descriptor of the region proposals tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] rois
 * Pointer to the MLU memory that stores the region proposals tensor.
 * @param[in] argmax_desc
 * The descriptor of the argmax tensor that stores the index returned by
 * ::mluOpRoiPoolingForward. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] argmax
 * Pointer to the MLU memory that stores the argmax tensor.
 * @param[in] spatial_scale
 * A scaling factor that specifies how to map the ROIs coordinates in the original image to
 * the coordinates in the output.
 * @param[in] grads_image_desc
 * The descriptor of the gradients tensor of the original images. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] grads_image
 * Pointer to the MLU memory that stores the gradients tensor of the original images.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_INTERNAL_ERROR.
 *
 * @par Data Type
 * - This function supports the following data types for gradient tensor \b grads, region proposals tensor \b rois,
 *   argmax_tensor \b argmax and output tensor \b grads_image. The data types of \b grads, \b rois, and \b grads_image
 *   must be the same.
 *   - grads tensor: half, float.
 *   - rois tensor: half, float.
 *   - argmax tensor: int32.
 *   - grads_image tensor: half, float.
 *
 * @par Data Layout
 * - The supported data layout of \b grads, \b rois, \b argmax, \b grads_image are as follows:
 *   - input tensor: \b MLUOP_LAYOUT_NHWC.
 *   - ROIs tensor: \b MLUOP_LAYOUT_ARRAY, only supports 2D tensor.
 *   - argmax tensor: \b MLUOP_LAYOUT_NHWC.
 *   - grads_image tensor: \b MLUOP_LAYOUT_NHWC.
 *
 * @par Scale Limitation
 * - The value of \b pooling_mode only supports \p MLUOP_POOLING_MAX.
 * - The \b grads tensor, \b argmax tensor and \b grads_image tensor must be 4-D tensor.
 * - The size of the lowest dimension of \b grads tensors, \b argmax tensor and \b grads_image tensor must be the same.
 * - The size of each dimension of \b grads tensor and \b argmax tensor must be the same.
 * - The \b rois tensor must be 2-D tensor.
 * - The size of the highest dimension of \b grads tensor, \b rois tensor and \b argmax tensor must be the same.
 * - The shape of \b rois must be [rois_num, 5].
 * - \b rois[i] consists of [rois_num, x1, y1, x2, y2]. \b rois_num specifies which image this ROI is in,
 *   and should be in the range of [0, batch_num - 1]. \b x1 and \b y1 specify the starting coordinate of the ROI
 *   in the original image. \b x2 and \b y2 specify the ending coordinate of this ROI in the original image. \b x1 and
 *   \b y1 should be greater than or equal to 0. \b x2 should be greater than \b x1. \b y2 should be greater than \b y1.
 * - \b Spatial_scale should be in the range of (0, 1].
 * - The value of argmax tensors with the data type is int32_t and should be in the range of [0, \f$2^(31)-1\f$].
 *
 * @par Note
 * - In general, set the values of \b argmax according to the result returned by ::mluOpRoiPoolingForward.
 *   Otherwise, these values may be regarded as invalid and will not be used in this operation.
 * - When \b rois contains NaN or infinity, it may cause undefined behavior.
 *
 * @par Example
 *   @verbatim
     input two arrays by 1 * 2 * 2 * 1 and 1 * 5 --> grads: [[[[1.0], [2.0]], [[3.0], [4.0]]]]

     --> rois: [[0.0, 0.0, 0.0, 2.0, 2.0]]
     --> argmax: [[[[0], [2]], [[8], [10]]]]

     param:
         spatial_scale: 1.0

     grads_image: [[[[1.0], [0.0], [2.0], [0.0]], [[0.0], [0.0], [0.0], [0.0]],
                    [[3.0], [0.0], [4.0], [0.0]], [[0.0], [0.0], [0.0], [0.0]]]]
     @endverbatim
 *
 * @par Reference
 * - https://pytorch.org/vision/stable/ops.html#torchvision.ops.roi_pool
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRoiPoolingBackward(mluOpHandle_t handle,
                        mluOpPoolingMode_t pooling_mode,
                        const mluOpTensorDescriptor_t grads_desc,
                        const void *grads,
                        const mluOpTensorDescriptor_t rois_desc,
                        const void *rois,
                        const mluOpTensorDescriptor_t argmax_desc,
                        const int *argmax,
                        const float spatial_scale,
                        const mluOpTensorDescriptor_t grads_image_desc,
                        void *grads_image);

// Group: SyncBatchNorm
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra
 * workspace to optimize ::mluOpSyncBatchNormStats_v2 operation.
 *
 * The size of extra workspace is based on the given information of ::mluOpSyncBatchNormStats_v2
 * operation, including the input tensor descriptor \b x_desc.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpSyncBatchNormStats_v2 operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] x_desc
 * The descriptor of the input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] workspace_size
 * Pointer to the returned size of the extra workspace in bytes that is used in the
 * ::mluOpSyncBatchNormStats_v2 operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - This API is only used along with ::mluOpSyncBatchNormStats_v2.
 * - ::mluOpSyncBatchNormStats does not require this API.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetSyncBatchNormStatsWorkspaceSize(mluOpHandle_t handle,
                                        const mluOpTensorDescriptor_t x_desc,
                                        size_t *workspace_size);

// Group: SyncBatchNorm
/*!
 * @brief Computes the local mean and the local inverse standard deviation for each channel
 * across a batch of data in the training scenario.
 *
 * ::mluOpSyncBatchNormStats_v2 is used in convolution network, including but not limited to
 * ResNet (Residual Network), Yolo (You Only Look Once) and R-CNN (Regions with CNN features).
 *
 * Compared with ::mluOpSyncBatchNormStats, this function allows you to allocate some extra
 * workspace as an input parameter. If you just set \b workspace to NULL and \b workspace_size
 * to 0, this function will perform as same as ::mluOpSyncBatchNormStats.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpSyncBatchNormStats_v2 operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] x_desc
 * The descriptor of the input tensor \b x. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor \b x.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for ::mluOpSyncBatchNormStats_v2.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in
 * ::mluOpSyncBatchNormStats_v2. You can get the size of the workspace with
 * ::mluOpGetSyncBatchNormStatsWorkspaceSize function.
 * @param[in] eps
 * A floating-point value added to the denominator for numerical stability.
 * @param[in] mean_desc
 * The descriptor of the output tensor \b mean. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] mean
 * Pointer to the MLU memory that stores the output tensor \b mean, which is the
 * local mean.
 * @param[in] invstd_desc
 * The descriptor of the output tensor \b invstd. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] invstd
 * Pointer to the MLU memory that stores the output tensor \b invstd, which is the
 * local inverse standard deviation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - The supported combinations of data types are shown below with the following order:
 *   - float(x) - float(eps) - float(mean) - float(invstd).
 *   - half(x) - float(eps) - float(mean) - float(invstd).
 *
 * @par Data Layout
 * - The supported data layout of the input tensor is shown as follows:
 *   - x tensor: \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NC and \p MLUOP_LAYOUT_NLC.
 * - The layout of the output tensors is shown as follows:
 *   - mean tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - invstd tensor: \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - Before calling this function to perform ::mluOpSyncBatchNormStats_v2, you need to get
 *   the size of workspace by ::mluOpGetSyncBatchNormStatsWorkspaceSize.
 *
 * @par note
 * - None.
 *
 * @par Example
 * - The example of ::mluOpSyncBatchNormStats_v2 operation is as follows:
     @verbatim
      input five arrays by 1 * 2 * 3 * 2
      --> x: [[[[1.0, 1.0],[1.0, 1.0],[1.0, 1.0]],
               [[1.0, 1.0],[1.0, 1.0],[1.0, 1.0]]]]
      param:
        eps: 0.00001
      output an array by 2
      --> mean: [1.0, 1.0]
      --> invstd: [316.221, 316.221]
     @endverbatim
 *
 * @par Reference
 * - https://pytorch.org/docs/1.6.0/jit_builtin_functions.html?highlight=batch_norm_stats
 *
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSyncBatchNormStats_v2(mluOpHandle_t handle,
                           const mluOpTensorDescriptor_t x_desc,
                           const void *x,
                           void *workspace,
                           size_t workspace_size,
                           const float eps,
                           const mluOpTensorDescriptor_t mean_desc,
                           void *mean,
                           const mluOpTensorDescriptor_t invstd_desc,
                           void *invstd);

// Group: SyncBatchNorm
/*!
 * @brief Computes the local mean and the local inverse standard deviation for each channel
 * across a batch of data in the training scenario.
 *
 * ::mluOpSyncBatchNormStats is used in CNN, including but not limited to
 * ResNet (Residual Network), Yolo (You Only Look Once) and R-CNN (Regions with CNN features).
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * ::mluOpSyncBatchNormStats operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] x_desc
 * The descriptor of the input tensor \b x. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor \b x.
 * @param[in] eps
 * A floating-point value added to the denominator for numerical stability.
 * @param[in] mean_desc
 * The descriptor of the output tensor \b mean. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] mean
 * Pointer to the MLU memory that stores the output tensor \b mean, which is the
 * local mean.
 * @param[in] invstd_desc
 * The descriptor of the output tensor \b invstd. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] invstd
 * Pointer to the MLU memory that stores the output tensor \b invstd, which is the
 * local inverse standard deviation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - The supported combinations of data types are shown below with the following order:
 *   - \b x - \b eps - \b mean - \b invstd
 *   - The supported data type combinations are:
 *     - float - float - float - float.
 *     - half - float - float - float.
 *
 * @par Data Layout
 * - The supported data layout of the input tensor is shown as follows:
 *   - x tensor: \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NC and \p MLUOP_LAYOUT_NLC.
 * - The layout of the output tensors is shown as follows:
 *   - mean tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - invstd tensor: \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par note
 * - None.
 *
 * @par Example
 * - The example of ::mluOpSyncBatchNormStats operation is as follows:
     @verbatim
      input five arrays by 1 * 2 * 3 * 2
      --> x: [[[[1.0, 1.0],[1.0, 1.0],[1.0, 1.0]],
               [[1.0, 1.0],[1.0, 1.0],[1.0, 1.0]]]]
      param:
        eps: 0.00001
      output an array by 2
      --> mean: [1.0, 1.0]
      --> invstd: [316.221, 316.221]
     @endverbatim
 *
 * @par Reference
 * - https://pytorch.org/docs/1.6.0/jit_builtin_functions.html?highlight=batch_norm_stats
 *
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSyncBatchNormStats(mluOpHandle_t handle,
                        const mluOpTensorDescriptor_t x_desc,
                        const void *x,
                        const float eps,
                        const mluOpTensorDescriptor_t mean_desc,
                        void *mean,
                        const mluOpTensorDescriptor_t invstd_desc,
                        void *invstd);

// Group: SyncBatchNorm
/*!
 * @brief Computes the global mean and the global inverse standard deviation across aggregation
 * of the local mean and local inverse standard deviation of multiple MLU devices.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpSyncBatchNormGatherStatsWithCounts. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in] mean_all_desc
 * The descriptor of the input tensor \b mean_all. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] mean_all
 * Pointer to the MLU memory that stores the input tensor \b mean_all, which is
 * the local mean of multiple MLU devices.
 * @param[in] invstd_all_desc
 * The descriptor of the input tensor \b invstd_all. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] invstd_all
 * Pointer to the MLU memory that stores the input tensor \n invstd_all, which
 * is the local inverse standard deviation of multiple MLU devices.
 * @param[in] moving_mean_desc
 * The descriptor of the input tensor \b moving_mean. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in,out] moving_mean
 * Pointer to the MLU memory that stores the input tensor \b moving_mean,
 * which is the moving average of mean computed over the dimensions of the input tensor
 * \b mean_all. The value of this pointer can be NULL.
 * @param[in] moving_var_desc
 * The descriptor of the input tensor \b moving_var. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in,out] moving_var
 * Pointer to the MLU memory that stores the tensor \b moving_var, which is
 * the moving average of inverse standard deviation computed over the dimensions of the input
 * tensor \b invstd_all. The value of this pointer can be NULL.
 * @param[in] momentum
 * A floating-point value used to do moving average of \b moving_mean and \b moving_var.
 * @param[in] eps
 * A floating-point value added to the denominator for numerical stability.
 * @param[in] count_all_desc
 * The descriptor of the input tensor \b count_all. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] count_all
 * Pointer to the MLU memory that stores an array, which stores the total size of
 * dimensions (except C dimension) of input for each MLU device.
 * @param[in] mean_desc
 * The descriptor of the output tensor \b mean. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] mean
 * Pointer to the MLU memory that stores the output tensor \b mean, which is the
 * global mean.
 * @param[in] invstd_desc
 * The descriptor of the output tensor \b invstd. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] invstd
 * Pointer to the MLU memory that stores the output tensor \b invstd, which is the
 * global inverse standard deviation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - The supported combinations of data types are shown as the following order:
 *   - mean_all - invstd_all - moving_mean - moving_var - momentum -  eps  - count_all - mean  - invstd
 *   -  float   -   float    -    float    -    float   -   float  - float -   float   - float -  float.
 *   -  float   -   float    -    half     -    half    -   float  - float -   half    - float -  float.
 *
 * @par Data Layout
 * - The supported data layout of the input tensors is shown as follows:
 *   - mean_all tensor: \p MLUOP_LAYOUT_NC.
 *   - invstd_all tensor: \p MLUOP_LAYOUT_NC.
 *   - moving_mean tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - moving_var tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - momentum: Scalar.
 *   - eps: Scalar.
 *   - count_all tensor: \p MLUOP_LAYOUT_ARRAY.
 * - The layout of the output tensors is shown as follows:
 *   - mean tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - invstd tensor: \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par note
 * - The input \b mean_all and the input \b invstd_all cannot be positive infinity or negative infinity
 *   at the same time on MLU300 series or above.
 *
 * @par Example
 * - The example of ::mluOpSyncBatchNormGatherStatsWithCounts operation is as follows:
     @verbatim
      --> mean_all: an array [8, 1024];
      --> invstd_all: an array [8, 1024];
      --> moving_mean: an array [1024];
      --> moving_var: an array [1024];
      --> count_all: an array [8];
      param:
      --> momentum: 0.1
      --> eps: 0.00001
      output:
      --> mean: an array [1024];
      --> invstd: [1024];
     @endverbatim
 *
 * @par Reference
 * - https://pytorch.org/docs/1.6.0/jit_builtin_functions.html?highlight=batch_norm_stats
 *
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSyncBatchNormGatherStatsWithCounts(mluOpHandle_t handle,
                                        const mluOpTensorDescriptor_t mean_all_desc,
                                        const void *mean_all,
                                        const mluOpTensorDescriptor_t invstd_all_desc,
                                        const void *invstd_all,
                                        const mluOpTensorDescriptor_t moving_mean_desc,
                                        void *moving_mean,
                                        const mluOpTensorDescriptor_t moving_var_desc,
                                        void *moving_var,
                                        float momentum,
                                        float eps,
                                        const mluOpTensorDescriptor_t count_all_desc,
                                        const void *count_all,
                                        const mluOpTensorDescriptor_t mean_desc,
                                        void *mean,
                                        const mluOpTensorDescriptor_t invstd_desc,
                                        void *invstd);

// Group: SyncBatchNorm
/*!
 * @brief Applies Batch Normalization for each channel across a batch of data with the given mean,
 *        inverse variance and scaling factors.
 *
 * Batch Normalization is used in artificial intelligence, including but not limited to
 * ResNet (Residual Network), Yolo (You Only Look Once) and R-CNN (Regions with CNN features).
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpSyncBatchNormElemt. For detailed information, see ::mluOpHandle_t.
 * @param[in] x_desc
 * The descriptor of the input tensor \b x. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor \b x.
 * @param[in] mean_desc
 * The descriptor of \b mean tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mean
 * Pointer to the MLU memory that stores the tensor \b mean, which is computed over the
 * batch and spatial dimensions by ::mluOpSyncBatchNormGatherStatsWithCounts.
 * @param[in] invstd_desc
 * The descriptor of \b invstd tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] invstd
 * Pointer to the MLU memory that stores the tensor \b invstd, which is the inverse variance
 * computed over the batch and spatial dimensions by ::mluOpSyncBatchNormGatherStatsWithCounts.
 * @param[in] filter_desc
 * The descriptor of \b filter tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * The descriptor can be NULL when \b filter pointer is NULL.
 * @param[in] filter
 * Pointer to the MLU memory that stores the input tensor \b filter for affine transformation
 * after batch normilization. The value of this pointer can be NULL.
 * @param[in] bias_desc
 * The descriptor of \b bias tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * The descriptor can be NULL when \b bias pointer is NULL.
 * @param[in] bias
 * Pointer to the MLU memory that stores the input tensor \b bias for affine transformation
 * after batch normalization. The value of this pointer can be NULL.
 * @param[in] y_desc
 * The descriptor of the sync batch normalization output tensor \b y. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] y
 * Pointer to the MLU memory that stores the output tensor \b y.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - The supported combinations of data types are shown below with the following order:
 *   - x_tensor - mean_tensor - invstd_tensor - filter_tensor - bias_tensor - y_tensor
 *   - float - float - float - float - float - float.
 *   - half - float - float - float - float - half.
 *
 * @par Data Layout
 * - The supported data layout of \b x, \b mean, \b invstd, \b filter, \b bias and \b y is as follows:
 *   - x tensor: \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NC and \p MLUOP_LAYOUT_NLC.
 *   - mean tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - invstd tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - filter tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - bias tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - y tensor: \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NC and \p MLUOP_LAYOUT_NLC.
 *     The layout of the \b y should be the same as \b x tensor.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par note
 * - The \b mean, \b invstd, \b filter and \b \b bias must be 1D tensors and the length of their dimensions
 *   should be the same as the length of the lowest dimension of \b x.
 * - The length of each dimension of \b x and \b y must be the same.
 *
 * @par Example
 * - The example of ::mluOpSyncBatchNormElemt operation is as follows:
     @verbatim
      input five arrays by 1 * 2 * 3 * 2, 2, 2, 2 and 2
      --> x: [[[[1.0, 1.0],[1.0, 1.0],[1.0, 1.0]],
               [[1.0, 1.0],[1.0, 1.0],[1.0, 1.0]]]]

      --> mean: [0.5, 0.5]

      --> invstd: [2.0, 2.0]

      --> filter: [0.5, 0.5]

      --> bias: [1.0, 1.0]

      output array by 1 * 2 * 3 * 2
      --> y: [[[[1.5, 1.5],[1.5, 1.5],[1.5, 1.5]],
               [[1.5, 1.5],[1.5, 1.5],[1.5, 1.5]]]]
     @endverbatim
 *
 * @par Reference
 * - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,
 *   Sergey Ioffe, 2015.
 *
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSyncBatchNormElemt(mluOpHandle_t handle,
                        const mluOpTensorDescriptor_t x_desc,
                        const void *x,
                        const mluOpTensorDescriptor_t mean_desc,
                        const void *mean,
                        const mluOpTensorDescriptor_t invstd_desc,
                        const void *invstd,
                        const mluOpTensorDescriptor_t filter_desc,
                        const void *filter,
                        const mluOpTensorDescriptor_t bias_desc,
                        const void *bias,
                        const mluOpTensorDescriptor_t y_desc,
                        void *y);

// Group: SyncBatchNorm
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra
 * workspace to optimize the sync_batchnorm_backward_reduce operation.
 *
 * The size of extra workspace is based on the given information of
 * ::mluOpSyncBatchNormBackwardReduce_v2 operation, including the input tensor descriptor \b x_desc.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the mse_loss
 * operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] x_desc
 * The descriptor of the input tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] workspace_size
 * Pointer to the returned size of the extra workspace in bytes that is used in
 * ::mluOpSyncBatchNormBackwardReduce_v2 operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par note
 * - This API is only used along with ::mluOpSyncBatchNormBackwardReduce_v2.
 * - ::mluOpSyncBatchNormBackwardReduce does not require this API.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetSyncBatchNormBackwardReduceWorkspaceSize(mluOpHandle_t handle,
                                                 const mluOpTensorDescriptor_t x_desc,
                                                 size_t *workspace_size);

// Group: Deprecated APIs
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra
 * workspace to optimize the sync_batchnorm_backward_reduce operation.
 *
 * The size of extra workspace is based on the given information of
 * ::mluOpSyncBatchNormBackwardReduce_v2 operation, including the input tensor descriptor \b x_desc.
 *
 * @par Deprecated
 * - ::mluOpGetSyncBatchnormBackwardReduceWorkspaceSize is deprecated and will be
 *   removed in the future release. It is recommended to use
 *   ::mluOpGetSyncBatchNormBackwardReduceWorkspaceSize instead.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the mse_loss
 * operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] x_desc
 * The descriptor of the input tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] workspace_size
 * Pointer to the returned size of the extra workspace in bytes that is used in
 * ::mluOpSyncBatchNormBackwardReduce_v2 operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par note
 * - This API is only used along with ::mluOpSyncBatchNormBackwardReduce_v2.
 * - ::mluOpSyncBatchNormBackwardReduce does not require this API.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetSyncBatchnormBackwardReduceWorkspaceSize(mluOpHandle_t handle,
                                                 const mluOpTensorDescriptor_t x_desc,
                                                 size_t *workspace_size);

// Group: SyncBatchNorm
/*!
 * @brief Applies Synchronized Batch Normalization Reduce operator to backwardly compute grad
 * filters, grad bias, sum_dy and sum_dy_xmu on each MLU device.
 *
 * Batch Normalization is used in convolution network, including but not limited to
 * ResNet (Residual Network), Yolo (You Only Look Once) and R-CNN (Regions with CNN features).
 *
 * Compared with ::mluOpSyncBatchNormBackwardReduce, this function allows you to allocate some extra
 * workspace as an input parameter. If you just set \b workspace to NULL and \b workspace_size to 0,
 * this function will perform as same as ::mluOpSyncBatchNormBackwardReduce.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpSyncBatchNormBackwardReduce_v2 operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] desc_dz
 * The descriptor of the input tensor \b dz. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] dz
 * Pointer to the MLU memory that stores the tensor \b dz, which denotes the partial
 * derivative of batch normalization forward output.
 * @param[in] desc_x
 * The descriptor of the input tensor \b x. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor \b x.
 * @param[in] desc_mean
 * The descriptor of \b mean tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mean
 * Pointer to the MLU memory that stores the tensor \b mean, which denotes the average
 * result of input \b x.
 * @param[in] desc_invstd
 * The descriptor of \b invstd tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] invstd
 * Pointer to the MLU memory that stores the tensor \b invstd, which denotes the inversed
 * standard deviation of input \b x.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for
 * ::mluOpSyncBatchNormBackwardReduce_v2.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in
 * the ::mluOpSyncBatchNormBackwardReduce_v2. You can get the size of the workspace with
 * the ::mluOpGetSyncBatchNormBackwardReduceWorkspaceSize function.
 * @param[out] desc_dfilter
 * The descriptor of \b dfilters tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] dfilter
 * Pointer to the MLU memory that stores the input tensor \b dfilters, which denotes
 * partial derivative of filter in sync batch normalization forward training. It will be computed
 * only if booleanvariable \b needs_input_grad1 is true.
 * @param[out] desc_dbias
 * The descriptor of the sync batch normalization output tensor \b dbias. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[out] dbias
 * Pointer to the MLU memory that stores the output tensor \b dbias, which denotes partial
 * derivative of bias in sync batch normalization forward training. It will be computed
 * only if \b needs_input_grad2 is true.
 * @param[out] desc_sum_dy
 * The descriptor of the sync batch normalization output tensor \b sum_dy. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[out] sum_dy
 * Pointer to the MLU memory that stores the output tensor \b sum_dy, which denotes the
 * summation of dz and is also an intermediate variable to compute the partial derivative of
 * input x. Moreover, it will be computed only if boolean variable \b needs_input_grad0 is true.
 * @param[out] desc_sum_dy_xmu
 * The descriptor of the sync batch normalization output tensor \b sum_dy_xmu. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[out] sum_dy_xmu
 * Pointer to the MLU memory that stores the output tensor \b sum_dy_xmu, which denotes
 * sum{dz(x-mean)}. It is also an intermediate variable to compute the partial derivative of
 * input \b x. Moreover, it will be computed only if boolean variable \b needs_input_grad0 is
 * true.
 * @param[in] needs_input_grad0
 * A boolean variable that determines whether to compute \b sum_dy and \b sum_dy_xmu.
 * When \b needs_input_grad0 is true, \b sum_dy and \b sum_dy_xmu will be computed.
 * When \b needs_input_grad0 is false, \b sum_dy and \b sum_dy_xmu will be NULL.
 * @param[in] needs_input_grad1
 * A boolean variable that determines whether to compute \b dfilters.
 * When \b needs_input_grad1 is true, \b dfilters will be computed.
 * When \b needs_input_grad1 is false, \b dfilter will be NULL.
 * @param[in] needs_input_grad2
 * A boolean variable that determines whether to compute \b dbias.
 * When \b needs_input_grad2 is true, \b dbias will be computed.
 * When \b needs_input_grad2 is false, \b dbias will be NULL.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - The supported combinations of data types are shown below with the following order:
 *   - dz_tensor - x_tensor - mean_tensor - invstd_tensor - dfilter_tensor - dbias_tensor -
 *   sum_dy_tensor - sum_dy_xmu_tensor
 *   - float - float - float - float - float - float - float - float.
 *   - half - half - float - float - float - float - float - float.
 *
 * @par Data Layout
 * - The supported data layout of \b dz, \b x, \b mean, \b invstd, \b dfilter, \b dbias, \b sum_dy
 *   and \b sum_dy_xmu is as follows:
 *   - dz tensor: \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NLC, \p MLUOP_LAYOUT_NC.
 *   - x tensor: \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NLC, \p MLUOP_LAYOUT_NC.
 *   - mean tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - invstd tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - dfilter tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - dbias tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - sum_dy tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - sum_dy_xmu tensor: \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - Before calling this function to perform ::mluOpSyncBatchNormBackwardReduce_v2, you need to get
 *   the size of workspace by ::mluOpGetSyncBatchNormBackwardReduceWorkspaceSize.
 *
 * @par note
 * - The \b mean, \b invstd, \b dfilter, \b bias, \b sum_dy and \b sum_dy_xmu must be 1D tensors
 *   and the length of the dimensions of these tensors should be the same as the length of
 *   the lowest dimension of \b x.
 * - The length of each dimension of \b x and \b dz must be the same.
 *
 * @par Example
 * - The example of ::mluOpSyncBatchNormBackwardReduce_v2 operation is as follows:
     @verbatim
      input four arrays by 1 * 2 * 3 * 2, 2, 2, 2 and 2
      --> dz: [[[[6.0, 6.0],[6.0, 6.0],[6.0, 6.0]],
               [[6.0, 6.0],[6.0, 6.0],[6.0, 6.0]]]]

      --> x: [[[[3.0, 3.0],[3.0, 3.0],[3.0, 3.0]],
               [[3.0, 3.0],[3.0, 3.0],[3.0, 3.0]]]]

      --> mean: [1, 1]

      --> invstd: [0.8, 0.8]

      output array by 2
      --> dfilter: [57.6, 57.6]

      --> dbias: [36.0, 36.0]

      --> sum_dy: [36.0, 36.0]

      --> sum_dy_xmu: [72.0, 72.0]
     @endverbatim
 *
 * @par Reference
 * - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,
 *   Sergey Ioffe, 2015.
 *
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSyncBatchNormBackwardReduce_v2(mluOpHandle_t handle,
                                    const mluOpTensorDescriptor_t desc_dz,
                                    const void *dz,
                                    const mluOpTensorDescriptor_t desc_x,
                                    const void *x,
                                    const mluOpTensorDescriptor_t desc_mean,
                                    const void *mean,
                                    const mluOpTensorDescriptor_t desc_invstd,
                                    const void *invstd,
                                    void *workspace,
                                    size_t workspace_size,
                                    const mluOpTensorDescriptor_t desc_dfilter,
                                    void *dfilter,
                                    const mluOpTensorDescriptor_t desc_dbias,
                                    void *dbias,
                                    const mluOpTensorDescriptor_t desc_sum_dy,
                                    void *sum_dy,
                                    const mluOpTensorDescriptor_t desc_sum_dy_xmu,
                                    void *sum_dy_xmu,
                                    const bool needs_input_grad0,
                                    const bool needs_input_grad1,
                                    const bool needs_input_grad2);

// Group: Deprecated APIs
/*!
 * @brief Applies Synchronized Batch Normalization Reduce operator to backwardly compute grad
 * filters, grad bias, sum_dy and sum_dy_xmu on each MLU device.
 *
 * Batch Normalization is used in convolution network, including but not limited to
 * ResNet (Residual Network), Yolo (You Only Look Once) and R-CNN (Regions with CNN features).
 *
 * Compared with ::mluOpSyncBatchNormBackwardReduce, this function allows you to allocate some extra
 * workspace as an input parameter. If you just set \b workspace to NULL and \b workspace_size to 0,
 * this function will perform as same as ::mluOpSyncBatchNormBackwardReduce.
 *
 * @par Deprecated
 * - ::mluOpSyncBatchnormBackwardReduce_v2 is deprecated and will be
 *   removed in the future release. It is recommended to use
 *   ::mluOpSyncBatchNormBackwardReduce_v2 instead.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpSyncBatchNormBackwardReduce_v2 operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] desc_dz
 * The descriptor of the input tensor \b dz. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] dz
 * Pointer to the MLU memory that stores the tensor \b dz, which denotes the partial
 * derivative of batch normalization forward output.
 * @param[in] desc_x
 * The descriptor of the input tensor \b x. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor \b x.
 * @param[in] desc_mean
 * The descriptor of \b mean tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mean
 * Pointer to the MLU memory that stores the tensor \b mean, which denotes the average
 * result of input \b x.
 * @param[in] desc_invstd
 * The descriptor of \b invstd tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] invstd
 * Pointer to the MLU memory that stores the tensor \b invstd, which denotes the inversed
 * standard deviation of input \b x.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for
 * ::mluOpSyncBatchNormBackwardReduce_v2.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in
 * the ::mluOpSyncBatchNormBackwardReduce_v2. You can get the size of the workspace with
 * the ::mluOpGetSyncBatchNormBackwardReduceWorkspaceSize function.
 * @param[out] desc_dfilter
 * The descriptor of \b dfilters tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] dfilter
 * Pointer to the MLU memory that stores the input tensor \b dfilters, which denotes
 * partial derivative of filter in sync batch normalization forward training. It will be computed
 * only if booleanvariable \b needs_input_grad1 is true.
 * @param[out] desc_dbias
 * The descriptor of the sync batch normalization output tensor \b dbias. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[out] dbias
 * Pointer to the MLU memory that stores the output tensor \b dbias, which denotes partial
 * derivative of bias in sync batch normalization forward training. It will be computed
 * only if \b needs_input_grad2 is true.
 * @param[out] desc_sum_dy
 * The descriptor of the sync batch normalization output tensor \b sum_dy. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[out] sum_dy
 * Pointer to the MLU memory that stores the output tensor \b sum_dy, which denotes the
 * summation of dz and is also an intermediate variable to compute the partial derivative of
 * input x. Moreover, it will be computed only if boolean variable \b needs_input_grad0 is true.
 * @param[out] desc_sum_dy_xmu
 * The descriptor of the sync batch normalization output tensor \b sum_dy_xmu. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[out] sum_dy_xmu
 * Pointer to the MLU memory that stores the output tensor \b sum_dy_xmu, which denotes
 * sum{dz(x-mean)}. It is also an intermediate variable to compute the partial derivative of
 * input \b x. Moreover, it will be computed only if boolean variable \b needs_input_grad0 is
 * true.
 * @param[in] needs_input_grad0
 * A boolean variable that determines whether to compute \b sum_dy and \b sum_dy_xmu.
 * When \b needs_input_grad0 is true, \b sum_dy and \b sum_dy_xmu will be computed.
 * When \b needs_input_grad0 is false, \b sum_dy and \b sum_dy_xmu will be NULL.
 * @param[in] needs_input_grad1
 * A boolean variable that determines whether to compute \b dfilters.
 * When \b needs_input_grad1 is true, \b dfilters will be computed.
 * When \b needs_input_grad1 is false, \b dfilter will be NULL.
 * @param[in] needs_input_grad2
 * A boolean variable that determines whether to compute \b dbias.
 * When \b needs_input_grad2 is true, \b dbias will be computed.
 * When \b needs_input_grad2 is false, \b dbias will be NULL.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - The supported combinations of data types are shown below with the following order:
 *   - dz_tensor - x_tensor - mean_tensor - invstd_tensor - dfilter_tensor - dbias_tensor -
 *   sum_dy_tensor - sum_dy_xmu_tensor
 *   - float - float - float - float - float - float - float - float.
 *   - half - half - float - float - float - float - float - float.
 *
 * @par Data Layout
 * - The supported data layout of \b dz, \b x, \b mean, \b invstd, \b dfilter, \b dbias, \b sum_dy
 *   and \b sum_dy_xmu is as follows:
 *   - dz tensor: \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NLC, \p MLUOP_LAYOUT_NC.
 *   - x tensor: \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NLC, \p MLUOP_LAYOUT_NC.
 *   - mean tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - invstd tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - dfilter tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - dbias tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - sum_dy tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - sum_dy_xmu tensor: \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - Before calling this function to perform ::mluOpSyncBatchNormBackwardReduce_v2, you need to get
 *   the size of workspace by ::mluOpGetSyncBatchNormBackwardReduceWorkspaceSize.
 *
 * @par note
 * - The \b mean, \b invstd, \b dfilter, \b bias, \b sum_dy and \b sum_dy_xmu must be 1D tensors
 *   and the length of the dimensions of these tensors should be the same as the length of
 *   the lowest dimension of \b x.
 * - The length of each dimension of \b x and \b dz must be the same.
 *
 * @par Example
 * - The example of ::mluOpSyncBatchNormBackwardReduce_v2 operation is as follows:
     @verbatim
      input four arrays by 1 * 2 * 3 * 2, 2, 2, 2 and 2
      --> dz: [[[[6.0, 6.0],[6.0, 6.0],[6.0, 6.0]],
               [[6.0, 6.0],[6.0, 6.0],[6.0, 6.0]]]]

      --> x: [[[[3.0, 3.0],[3.0, 3.0],[3.0, 3.0]],
               [[3.0, 3.0],[3.0, 3.0],[3.0, 3.0]]]]

      --> mean: [1, 1]

      --> invstd: [0.8, 0.8]

      output array by 2
      --> dfilter: [57.6, 57.6]

      --> dbias: [36.0, 36.0]

      --> sum_dy: [36.0, 36.0]

      --> sum_dy_xmu: [72.0, 72.0]
     @endverbatim
 *
 * @par Reference
 * - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,
 *   Sergey Ioffe, 2015.
 *
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSyncBatchnormBackwardReduce_v2(mluOpHandle_t handle,
                                    const mluOpTensorDescriptor_t desc_dz,
                                    const void *dz,
                                    const mluOpTensorDescriptor_t desc_x,
                                    const void *x,
                                    const mluOpTensorDescriptor_t desc_mean,
                                    const void *mean,
                                    const mluOpTensorDescriptor_t desc_invstd,
                                    const void *invstd,
                                    void *workspace,
                                    size_t workspace_size,
                                    const mluOpTensorDescriptor_t desc_dfilter,
                                    void *dfilter,
                                    const mluOpTensorDescriptor_t desc_dbias,
                                    void *dbias,
                                    const mluOpTensorDescriptor_t desc_sum_dy,
                                    void *sum_dy,
                                    const mluOpTensorDescriptor_t desc_sum_dy_xmu,
                                    void *sum_dy_xmu,
                                    const bool needs_input_grad0,
                                    const bool needs_input_grad1,
                                    const bool needs_input_grad2);

// Group: SyncBatchNorm
/*!
 * @brief Applies Synchronized Batch Normalization Reduce operator to backwardly compute grad filters,
 * grad bias, sum_dy and sum_dy_xmu on each MLU device.
 *
 * Batch Normalization is used in CNN, including but not limited to
 * ResNet (Residual Network), Yolo (You Only Look Once) and R-CNN (Regions with CNN features).
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * ::mluOpSyncBatchNormBackwardReduce operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] desc_dz
 * The descriptor of the input tensor \b dz. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] dz
 * Pointer to the MLU memory that stores the tensor \b dz, which denotes the partial derivative of
 * batch normalization forward output.
 * @param[in] desc_x
 * The descriptor of the input tensor \b x. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor \b x.
 * @param[in] desc_mean
 * The descriptor of \b mean tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mean
 * Pointer to the MLU memory that stores the tensor \b mean, which denotes the average result of
 * input \b x.
 * @param[in] desc_invstd
 * The descriptor of \b invstd tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] invstd
 * Pointer to the MLU memory that stores the tensor \b invstd, which denotes the inversed standard deviation
 * of input \b x.
 * @param[out] desc_dfilter
 * The descriptor of \b dfilter tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] dfilter
 * Pointer to the MLU memory that stores the input tensor \b dfilter, which denotes partial derivative
 * of filter in sync batch normalization forward training. It will be computed only if boolean variable
 * \b needs_input_grad1 is true.
 * @param[out] desc_dbias
 * The descriptor of the sync batch normalization output tensor \b dbias. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] dbias
 * Pointer to the MLU memory that stores the output tensor \b dbias, which denotes partial derivative of
 * bias in sync batch normalization forward training. It will be computed only if \b needs_input_grad2 is true.
 * @param[out] desc_sum_dy
 * The descriptor of the sync batch normalization output tensor \b sum_dy. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] sum_dy
 * Pointer to the MLU memory that stores the output tensor \b sum_dy, which denotes the summation of dz
 * and is also an intermediate variable to compute the partial derivative of input x. Moreover, it will be
 * computed only if boolean variable \b needs_input_grad0 is true.
 * @param[out] desc_sum_dy_xmu
 * The descriptor of the sync batch normalization output tensor \b sum_dy_xmu. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] sum_dy_xmu
 * Pointer to the MLU memory that stores the output tensor \b sum_dy_xmu, which denotes sum{dz(x-mean)}.
 * It is also an intermediate variable to compute the partial derivative of
 * input \b x. Moreover, it will be computed only if boolean variable \b needs_input_grad0 is true.
 * @param[in] needs_input_grad0
 * A boolean variable that determines whether to compute \b sum_dy and \b sum_dy_xmu.
 * When \b needs_input_grad0 is true, \b sum_dy and \b sum_dy_xmu will be computed.
 * When \b needs_input_grad0 is false, \b sum_dy and \b sum_dy_xmu will be NULL.
 * @param[in] needs_input_grad1
 * A boolean variable that determines whether to compute \b dfilters.
 * When \b needs_input_grad1 is true, \b dfilters will be computed.
 * When \b needs_input_grad1 is false, \b dfilter will be NULL.
 * @param[in] needs_input_grad2
 * A boolean variable that determines whether to compute \b dbias.
 * When \b needs_input_grad2 is true, \b dbias will be computed.
 * When \b needs_input_grad2 is false, \b dbias will be NULL.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - The supported combinations of data types are shown below with the following order:
 *   - dz_tensor - x_tensor - mean_tensor - invstd_tensor - dfilter_tensor - dbias_tensor - sum_dy_tensor
 *   - sum_dy_xmu_tensor
 *   - float - float - float - float - float - float - float - float.
 *   - half - half - float - float - float - float - float - float.
 *
 * @par Data Layout
 * - The supported data layout of \b dz, \b x, \b mean, \b invstd, \b dfilter, \b dbias, \b sum_dy and
 *   \b sum_dy_xmu is as follows:
 *   - dz tensor: \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NLC, \p MLUOP_LAYOUT_NC.
 *   - x tensor: \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NLC, \p MLUOP_LAYOUT_NC.
 *   - mean tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - invstd tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - dfilter tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - dbias tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - sum_dy tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - sum_dy_xmu tensor: \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par note
 * - The \b mean, \b invstd, \b dfilter, \b bias, \b sum_dy and \b sum_dy_xmu must be 1D tensors and the
 *   length of the dimensions of these tensors should be the same as the length of the lowest dimension of \b x.
 * - The length of each dimension of \b x and \b dz must be the same.
 *
 * @par Example
 * - The example of ::mluOpSyncBatchNormBackwardReduce operation is as follows:
     @verbatim
      input four arrays by 1 * 2 * 3 * 2, 2, 2, 2 and 2
      --> dz: [[[[6.0, 6.0],[6.0, 6.0],[6.0, 6.0]],
               [[6.0, 6.0],[6.0, 6.0],[6.0, 6.0]]]]

      --> x: [[[[3.0, 3.0],[3.0, 3.0],[3.0, 3.0]],
               [[3.0, 3.0],[3.0, 3.0],[3.0, 3.0]]]]

      --> mean: [1, 1]

      --> invstd: [0.8, 0.8]

      output array by 2
      --> dfilter: [57.6, 57.6]

      --> dbias: [36.0, 36.0]

      --> sum_dy: [36.0, 36.0]

      --> sum_dy_xmu: [72.0, 72.0]
     @endverbatim
 *
 * @par Reference
 * - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,
 *   Sergey Ioffe, 2015.
 *
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSyncBatchNormBackwardReduce(mluOpHandle_t handle,
                                 const mluOpTensorDescriptor_t desc_dz,
                                 const void *dz,
                                 const mluOpTensorDescriptor_t desc_x,
                                 const void *x,
                                 const mluOpTensorDescriptor_t desc_mean,
                                 const void *mean,
                                 const mluOpTensorDescriptor_t desc_invstd,
                                 const void *invstd,
                                 const mluOpTensorDescriptor_t desc_dfilter,
                                 void *dfilter,
                                 const mluOpTensorDescriptor_t desc_dbias,
                                 void *dbias,
                                 const mluOpTensorDescriptor_t desc_sum_dy,
                                 void *sum_dy,
                                 const mluOpTensorDescriptor_t desc_sum_dy_xmu,
                                 void *sum_dy_xmu,
                                 const bool needs_input_grad0,
                                 const bool needs_input_grad1,
                                 const bool needs_input_grad2);

// Group: Deprecated APIs
/*!
 * @brief Applies Synchronized Batch Normalization Reduce operator to backwardly compute grad filters,
 * grad bias, sum_dy and sum_dy_xmu on each MLU device.
 *
 * Batch Normalization is used in CNN, including but not limited to
 * ResNet (Residual Network), Yolo (You Only Look Once) and R-CNN (Regions with CNN features).
 *
 * @par Deprecated
 * - ::mluOpSyncBatchnormBackwardReduce is deprecated and will be
 *   removed in the future release. It is recommended to use
 *   ::mluOpSyncBatchNormBackwardReduce instead.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * ::mluOpSyncBatchNormBackwardReduce operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] desc_dz
 * The descriptor of the input tensor \b dz. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] dz
 * Pointer to the MLU memory that stores the tensor \b dz, which denotes the partial derivative of
 * batch normalization forward output.
 * @param[in] desc_x
 * The descriptor of the input tensor \b x. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor \b x.
 * @param[in] desc_mean
 * The descriptor of \b mean tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mean
 * Pointer to the MLU memory that stores the tensor \b mean, which denotes the average result of
 * input \b x.
 * @param[in] desc_invstd
 * The descriptor of \b invstd tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] invstd
 * Pointer to the MLU memory that stores the tensor \b invstd, which denotes the inversed standard deviation
 * of input \b x.
 * @param[out] desc_dfilter
 * The descriptor of \b dfilter tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] dfilter
 * Pointer to the MLU memory that stores the input tensor \b dfilter, which denotes partial derivative
 * of filter in sync batch normalization forward training. It will be computed only if boolean variable
 * \b needs_input_grad1 is true.
 * @param[out] desc_dbias
 * The descriptor of the sync batch normalization output tensor \b dbias. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] dbias
 * Pointer to the MLU memory that stores the output tensor \b dbias, which denotes partial derivative of
 * bias in sync batch normalization forward training. It will be computed only if \b needs_input_grad2 is true.
 * @param[out] desc_sum_dy
 * The descriptor of the sync batch normalization output tensor \b sum_dy. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] sum_dy
 * Pointer to the MLU memory that stores the output tensor \b sum_dy, which denotes the summation of dz
 * and is also an intermediate variable to compute the partial derivative of input x. Moreover, it will be
 * computed only if boolean variable \b needs_input_grad0 is true.
 * @param[out] desc_sum_dy_xmu
 * The descriptor of the sync batch normalization output tensor \b sum_dy_xmu. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] sum_dy_xmu
 * Pointer to the MLU memory that stores the output tensor \b sum_dy_xmu, which denotes sum{dz(x-mean)}.
 * It is also an intermediate variable to compute the partial derivative of
 * input \b x. Moreover, it will be computed only if boolean variable \b needs_input_grad0 is true.
 * @param[in] needs_input_grad0
 * A boolean variable that determines whether to compute \b sum_dy and \b sum_dy_xmu.
 * When \b needs_input_grad0 is true, \b sum_dy and \b sum_dy_xmu will be computed.
 * When \b needs_input_grad0 is false, \b sum_dy and \b sum_dy_xmu will be NULL.
 * @param[in] needs_input_grad1
 * A boolean variable that determines whether to compute \b dfilters.
 * When \b needs_input_grad1 is true, \b dfilters will be computed.
 * When \b needs_input_grad1 is false, \b dfilter will be NULL.
 * @param[in] needs_input_grad2
 * A boolean variable that determines whether to compute \b dbias.
 * When \b needs_input_grad2 is true, \b dbias will be computed.
 * When \b needs_input_grad2 is false, \b dbias will be NULL.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - The supported combinations of data types are shown below with the following order:
 *   - dz_tensor - x_tensor - mean_tensor - invstd_tensor - dfilter_tensor - dbias_tensor - sum_dy_tensor
 *   - sum_dy_xmu_tensor
 *   - float - float - float - float - float - float - float - float.
 *   - half - half - float - float - float - float - float - float.
 *
 * @par Data Layout
 * - The supported data layout of \b dz, \b x, \b mean, \b invstd, \b dfilter, \b dbias, \b sum_dy and
 *   \b sum_dy_xmu is as follows:
 *   - dz tensor: \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NLC, \p MLUOP_LAYOUT_NC.
 *   - x tensor: \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NLC, \p MLUOP_LAYOUT_NC.
 *   - mean tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - invstd tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - dfilter tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - dbias tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - sum_dy tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - sum_dy_xmu tensor: \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par note
 * - The \b mean, \b invstd, \b dfilter, \b bias, \b sum_dy and \b sum_dy_xmu must be 1D tensors and the
 *   length of the dimensions of these tensors should be the same as the length of the lowest dimension of \b x.
 * - The length of each dimension of \b x and \b dz must be the same.
 *
 * @par Example
 * - The example of ::mluOpSyncBatchNormBackwardReduce operation is as follows:
     @verbatim
      input four arrays by 1 * 2 * 3 * 2, 2, 2, 2 and 2
      --> dz: [[[[6.0, 6.0],[6.0, 6.0],[6.0, 6.0]],
               [[6.0, 6.0],[6.0, 6.0],[6.0, 6.0]]]]

      --> x: [[[[3.0, 3.0],[3.0, 3.0],[3.0, 3.0]],
               [[3.0, 3.0],[3.0, 3.0],[3.0, 3.0]]]]

      --> mean: [1, 1]

      --> invstd: [0.8, 0.8]

      output array by 2
      --> dfilter: [57.6, 57.6]

      --> dbias: [36.0, 36.0]

      --> sum_dy: [36.0, 36.0]

      --> sum_dy_xmu: [72.0, 72.0]
     @endverbatim
 *
 * @par Reference
 * - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,
 *   Sergey Ioffe, 2015.
 *
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSyncBatchnormBackwardReduce(mluOpHandle_t handle,
                                 const mluOpTensorDescriptor_t desc_dz,
                                 const void *dz,
                                 const mluOpTensorDescriptor_t desc_x,
                                 const void *x,
                                 const mluOpTensorDescriptor_t desc_mean,
                                 const void *mean,
                                 const mluOpTensorDescriptor_t desc_invstd,
                                 const void *invstd,
                                 const mluOpTensorDescriptor_t desc_dfilter,
                                 void *dfilter,
                                 const mluOpTensorDescriptor_t desc_dbias,
                                 void *dbias,
                                 const mluOpTensorDescriptor_t desc_sum_dy,
                                 void *sum_dy,
                                 const mluOpTensorDescriptor_t desc_sum_dy_xmu,
                                 void *sum_dy_xmu,
                                 const bool needs_input_grad0,
                                 const bool needs_input_grad1,
                                 const bool needs_input_grad2);

// Group: SyncBatchNorm
/*!
 * @brief Computes the gradients of input in the training scenario.
 *
 * This function is used in artificial intelligence, including but not limited
 * to ResNet (Residual Network), Yolo (You Only Look Once) and R-CNN (Regions with CNN features).
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in the
 * ::mluOpSyncBatchNormBackwardElemt operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] diff_y_desc
 * The descriptor of the backpropagated differential tensor \b diff_y. For
 * detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] diff_y
 * Pointer to the MLU memory that stores the backpropagated differential tensor.
 * @param[in] x_desc
 * The descriptor of the input tensor \b x. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] mean_desc
 * The descriptor of the input tensor \b mean. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] mean
 * Pointer to the MLU memory that stores the global mean.
 * @param[in] invstd_desc
 * The descriptor of the input tensor \b invstd. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] invstd
 * Pointer to the MLU memory that stores the global inverse standard deviation.
 * @param[in] filter_desc
 * The descriptor of the input tensor \b filter. For detailed information, see
 * ::mluOpTensorDescriptor_t. The descriptor can be NULL when \b filter pointer is NULL.
 * @param[in] filter
 * Pointer to the MLU memory that stores the input tensor \b filter for affine
 * transformation after batch normilization. The value of this pointer can be NULL.
 * @param[in] mean_dy_desc
 * The descriptor of the input tensor \b mean_dy. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] mean_dy
 * Pointer to the MLU memory that stores the mean of diff_y.
 * @param[in] mean_dy_xmu_desc
 * The descriptor of the input tensor \b mean_dy_xmu. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] mean_dy_xmu
 * Pointer to the MLU memory that stores the mean of the result of diff_y * (x - mean).
 * @param[in] diff_x_desc
 * The descriptor of the output tensor \b diff_x. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] diff_x
 * Pointer to the MLU memory that stores the derivative of input.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - The supported combinations of data types are shown below:
 *   - float(\b diff_y) - float(\b x) - float(\b mean) - float(\b invstd) - float(\b filter) -
 *     float(\b mean_dy) - float(\b mean_dy_xmu) - float(\b diff_x).
 *   - half(\b diff_y) - half(\b x) - float(\b mean) - float(\b invstd) - float(\b filter) -
 *     float(\b mean_dy) - float(\b mean_dy_xmu) - half(\b diff_x).
 *
 * @par Data Layout
 * - The supported data layout of \b diff_y, \b x, \b mean, \b invstd, \b filter, \b mean_dy,
 *   \b mean_dy_xmu and \b diff_x is as follows:
 *   - diff_y tensor: \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NC and
 *     \p MLUOP_LAYOUT_NLC.
 *   - x tensor: \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NC and \p MLUOP_LAYOUT_NLC.
 *   - mean tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - invstd tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - filter tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - mean_dy tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - mean_dy_xmu tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - diff_x tensor: \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NC and
 *     \p MLUOP_LAYOUT_NLC.
 * - The layouts of the \b diff_x \b x and \b diff_y should be the same.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par note
 * - The \b mean, \b invstd, \b filter, \b mean_dy and \b mean_dy_xmu must be 1D tensors and the
 *   length of the dimension of these tensors should be the same as the length of the lowest
 *   dimension of \b x.
 * - The length of each dimension of \b diff_y, \b x and \b diff_x must be the same.
 *
 * @par Example
 * - The example of ::mluOpSyncBatchNormBackwardElemt operation is as follows:
     @verbatim
      input seven arrays by 1, 1, 1, 1, 1, 1, 1 and 1
      --> diff_y: [[[[1.0]]]]
      --> x: [[[[2.0]]]]
      --> mean: [3.0]
      --> invstd: [4.0]
      --> filter: [5.0]
      --> mean_dy: [6.0]
      --> mean_dy_xmu: [7.0]

      output an array by 1
      --> mean: [[[[-8960.0]]]]
     @endverbatim
 *
 * @par Reference
 * - https://pytorch.org/docs/1.6.0/jit_builtin_functions.html?highlight=batch_norm_backward_elemt
 *
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSyncBatchNormBackwardElemt(mluOpHandle_t handle,
                                const mluOpTensorDescriptor_t diff_y_desc,
                                const void *diff_y,
                                const mluOpTensorDescriptor_t x_desc,
                                const void *x,
                                const mluOpTensorDescriptor_t mean_desc,
                                const void *mean,
                                const mluOpTensorDescriptor_t invstd_desc,
                                const void *invstd,
                                const mluOpTensorDescriptor_t filter_desc,
                                const void *filter,
                                const mluOpTensorDescriptor_t mean_dy_desc,
                                const void *mean_dy,
                                const mluOpTensorDescriptor_t mean_dy_xmu_desc,
                                const void *mean_dy_xmu,
                                const mluOpTensorDescriptor_t diff_x_desc,
                                void *diff_x);

// Group: SyncBatchNorm
/*!
 * @brief Computes the gradients of input in the training scenario.
 *
 * This function is used in ResNet (Residual Network), Yolo (You Only Look Once) and
 * R-CNN (Regions with CNN features).
 *
 * Compared with ::mluOpSyncBatchNormBackwardElemt, this function first computes the intermediate
 * results mean_dy and mean_dy_xmu based on \b sum_dy, \b sum_dy_xmu and \b count, and then
 * computes the gradient of \b x with the intermediate results.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and queues in
 * ::mluOpSyncBatchNormBackwardElemtV2 operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] diff_y_desc
 * The descriptor of the backpropagated differential tensor \b diff_y. For
 * detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] diff_y
 * Pointer to the MLU memory that stores the backpropagated differential tensor.
 * @param[in] x_desc
 * The descriptor of the input tensor \b x. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] mean_desc
 * The descriptor of the input tensor \b mean. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] mean
 * Pointer to the MLU memory that stores the global mean.
 * @param[in] invstd_desc
 * The descriptor of the input tensor \b invstd. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] invstd
 * Pointer to the MLU memory that stores the global inverse standard deviation.
 * @param[in] filter_desc
 * The descriptor of the input tensor \b filter. For detailed information, see
 * ::mluOpTensorDescriptor_t. The descriptor can be NULL when \b filter pointer is NULL.
 * @param[in] filter
 * Pointer to the MLU memory that stores the input tensor \b filter for affine
 * transformation after batch normalization. The value of this pointer can be NULL.
 * @param[in] sum_dy_desc
 * The descriptor of the input tensor \b sum_dy. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] sum_dy
 * Pointer to the MLU memory that stores the sum of diff_y.
 * @param[in] sum_dy_xmu_desc
 * The descriptor of the input tensor \b sum_dy_xmu. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] sum_dy_xmu
 * Pointer to the MLU memory that stores the sum of the result of diff_y * (x - mean).
 * @param[in] count_desc
 * The descriptor of the input tensor \b count. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] count
 * Pointer to the MLU memory that stores the number of the high dimensions (the dimensions
 * except the lowest dimension) of the input tensor \b x on all MLU devices.
 * @param[in] diff_x_desc
 * The descriptor of the output tensor \b diff_x.
 * @param[out] diff_x
 * Pointer to the MLU memory that stores the derivative of input.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_BAD_PARAM.
 *
 * @par Data Type
 * - The supported combinations of data types are shown below:
 *   - float(\b diff_y) - float(\b x) - float(\b mean) - float(\b invstd) - float(\b filter) -
 *     float(\b sum_dy) - float(\b sum_dy_xmu) - int32_t(\b count) - float(\b diff_x).
 *   - half(\b diff_y) - half(\b x) - float(\b mean) - float(\b invstd) - float(\b filter) -
 *     float(\b sum_dy) - float(\b sum_dy_xmu) - int32_t(\b count) - half(\b diff_x).
 *
 * @par Data Layout
 * - The supported data layouts of \b diff_y, \b x, \b mean, \b invstd, \b filter, \b sum_dy,
 *   \b sum_dy_xmu and \b diff_x is as follows:
 *   - diff_y tensor: \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NC and
 *     \p MLUOP_LAYOUT_NLC.
 *   - x tensor: \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NC and \p MLUOP_LAYOUT_NLC.
 *   - mean tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - invstd tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - filter tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - sum_dy tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - sum_dy_xmu tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - diff_x tensor: \p MLUOP_LAYOUT_NHWC, \p MLUOP_LAYOUT_NDHWC, \p MLUOP_LAYOUT_NC and
 *     \p MLUOP_LAYOUT_NLC.
 * - The layouts of the \b diff_x \b x and \b diff_y should be the same.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par note
 * - The \b mean, \b invstd, \b filter, \b sum_dy and \b sum_dy_xmu must be 1D tensors and the
 *   length of the dimension of these tensors should be the same as the length of the lowest
 *   dimension of \b x.
 * - The length of each dimension of \b diff_y, \b x and \b diff_x must be the same.
 *
 * @par Example
 * - The example of ::mluOpSyncBatchNormBackwardElemtV2 operation is as follows:
     @verbatim
      input seven arrays by 1, 1, 1, 1, 1, 1, 1 and 1
      --> diff_y: [[[[1.0]]]]
      --> x: [[[[2.0]]]]
      --> mean: [3.0]
      --> invstd: [4.0]
      --> filter: [5.0]
      --> sum_dy: [6.0]
      --> sum_dy_xmu: [7.0]
      --> count: [1]

      output an array by 1
      --> mean: [[[[-8960.0]]]]
     @endverbatim
 *
 * @par Reference
 * - https://pytorch.org/docs/1.11.0/jit_builtin_functions.html?highlight=batch_norm_backward_elemt
 *
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSyncBatchNormBackwardElemtV2(mluOpHandle_t handle,
                                  const mluOpTensorDescriptor_t diff_y_desc,
                                  const void *diff_y,
                                  const mluOpTensorDescriptor_t x_desc,
                                  const void *x,
                                  const mluOpTensorDescriptor_t mean_desc,
                                  const void *mean,
                                  const mluOpTensorDescriptor_t invstd_desc,
                                  const void *invstd,
                                  const mluOpTensorDescriptor_t filter_desc,
                                  const void *filter,
                                  const mluOpTensorDescriptor_t sum_dy_desc,
                                  const void *sum_dy,
                                  const mluOpTensorDescriptor_t sum_dy_xmu_desc,
                                  const void *sum_dy_xmu,
                                  const mluOpTensorDescriptor_t count_desc,
                                  const void *count,
                                  const mluOpTensorDescriptor_t diff_x_desc,
                                  void *diff_x);

// Group: Debugging
/*!
 * @brief Sets the mode of a Cambricon MLU-OPS debugging tool that can generate operator
 * information files for all the operators that are called. The generated file
 * contains the operator information including inputs shapes, outputs shapes,
 * parameters and inputs real data based on the setting of \b mode. For more
 * information, see "Cambricon MLU-OPS User Guide".
 *
 * @param[in] mode
 * The parameter determines what mode the Cambricon MLU-OPS debugging tool will turn on.
 *
 * @par Note
 * - When \b mode is set to 0, the Cambricon MLU-OPS debugging tool will turn off, and do not
 *   generate operator information files.
 *
 * - When \b mode is set to 1, the Cambricon MLU-OPS debugging tool will generate operator
 *   information files for all the operators that are called. And the inputs real
 *   data of the operators is not included in the files.
 *
 * - When \b mode is set to 2, the Cambricon MLU-OPS debugging tool will generate operator
 *   information files for all the operators that are called. Only part of the inputs real
 *   data of the operators is included in the files. If the environment variable
 *   MLUOP_GEN_CASE_DUMP_DATA is set to 1, all of the inputs real data of the operators will be
 *   included in the files. For more information about setting environment variable,
 *   see "Cambricon MLU-OPS User Guide".
 *
 * - When \b mode is set to 3, the Cambricon MLU-OPS debugging tool will print operator
 *   information on the screen without the inputs real data for all the operators
 *   that are called instead generating information files.
 *
 * - When \b mode is out of range [0, 3], the Cambricon MLU-OPS debugging tool will turn off,
 *   and do not generate operator information files.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
void MLUOP_WIN_API
mluOpSetGenCaseMode(int mode);

// Group: DeformConv
/*!
 * @brief Creates a descriptor pointed by \p dcn_desc for a deformable convolution forward
 * or backward operation, and allocates memory for holding the information about the
 * deformable convolution operation. The information is defined in ::mluOpDCNDescriptor_t.
 * For more information about descriptor, see "Cambricon MLU-OPS User Guide".
 *
 * @param[out] dcn_desc
 *   A host pointer to the deformable convolution descriptor that holds information about the
 *   deformable convolution operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_NOT_INITIALIZED
 *
 * @par API Dependency
 * - After calling this function, call ::mluOpSetDCNDescriptor function to initialize
 *   and set the information to the deformable convolution descriptor.
 * - Call ::mluOpDestroyDCNDescriptor function to destroy the descriptor.
 *
 * @par Note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateDCNDescriptor(mluOpDCNDescriptor_t *dcn_desc);

// Group: DeformConv
/*!
 * @brief Initializes the deformable convolution descriptor \p dcn_desc that was
 * created by ::mluOpCreateDCNDescriptor function, and sets the information about the
 * deformable convolution forward and backward operation to the deformable convolution descriptor
 * \p dcn_desc. The information includes the number of deformable convolution dimensions \p dimNb,
 * the padding size for each dimension \p pad, the stride of the sliding window for each dimension
 * \p stride, the dilation factor for each dimension \p dilation, the number of groups of input
 * offset to be split along the input channel \p deformable_group, the number of convolution group
 * \p conv_group, and the maximum image number per deformable convolution computing \p im2col_step.
 *
 * @param[in,out] dcn_desc
 *   Input/output. The descriptor of the deformable convolution operation. For detailed information,
 *   see ::mluOpDCNDescriptor_t.
 * @param[in] dimNb
 *   The number of dimensions in the input tensor of the deformable convolution operation.
 *   Currently, the value of this parameter can only be set to 4 and must be the
 *   same as the one you set in the input tensor descriptor.
 * @param[in] pad
 *   An array that stores the zero-padding size for each dimension of the input tensor
 *   used in the deformable convolution operation.
 *   For each dimension, the padding size represents the number of zeros to be concatenated at the
 *   start and end of that dimension. For 2D deformable convolution, the padding is
 *   on the top, bottom, left, and right.
 * @param[in] stride
 *   An array that stores the filter stride for each dimension of the input tensor
 *   used in the deformable convolution operation. For each dimension, the filter stride represents
 *   the number of elements to slide over the input tensor. For 2D deformable
 *   convolution, the stride must be set in height and width order.
 * @param[in] dilation
 *   An array that stores the dilation factor for each dimension of the filter tensor
 *   used in the deformable convolution operation. For each dimension, the dilation factor
 *   represents the spacing between the kernel points. For 2D deformable convolution,
 *   the dilation must be set in height and width order.
 * @param[in] deformable_group
 *   The number of deformable offset groups that split the input offset along the channel
 *   of input tensor. Each deformable group is deformed separately to detect different input parts.
 * @param[in] conv_group
 *   The number of groups that the input channel been divided.
 *   Each convolution group is convolved separately. The filter used for
 *   each group is the filter tensor divides \p conv_group. The result of
 *   the deformable convolution operation is the concatenation of all the group convolution results
 *   along the output channel dimension.
 * @param[in] im2col_step
 *   The maximum number of images per deformable convolution computing. This parameter
 *   affects both the workspace size and the computing efficiency.
 *   A larger \p im2col_step will consume a larger workspace size and have a higher performance,
 *   while a smaller one will consume a smaller workspace size and have a lower performance.
 * @param[in] compute_type
 *   The data type of the temporary result in the convolution operation. Only supports
 *   \p MLUOP_DTYPE_FLOAT type.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - Before calling this function, ::mluOpCreateDCNDescriptor must be called.
 *
 * @par Note
 * - Currently, only supports 4D input tensor for deformable convolution forward
 *   and backward operations.
 * - The values of \p pad must be greater than or equal to 0.
 * - The values of \p stride must be greater than or equal to 1.
 * - The values of \p dilation must be greater than or equal to 1.
 * - The value of \p deformable_group must be greater than or equal to 1 and
 *   less than or equal to the number of channels in the input tensor, and input channel must be
 *   divisible by \p deformable_group.
 *   - If \p deformable_group is set to 1, the same input offset is applied to all channels
 *   of one pixel.
 *   - If the value of \p deformable_group is between 1 and the number of channels of input tensor,
 *     the input channel will be split into \p deformable_group parts. Each part is responsible for
 *     detecting different input parts, which results in a more flexible geometric transformation.
 * - The value of \p conv_group must be greater than or equal to 1 and less than or equal to the
 *   number of channels in the input tensor, and input channels and output channels must both be
 *   divisible by \p conv_group.
 * - The value of \p im2col_step must be greater than or equal to 1 and less than or equal to
 *   the number of batch sizes in the input tensor, and the input batch must be divisible by
 *   \p im2col_step.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetDCNDescriptor(mluOpDCNDescriptor_t dcn_desc,
                      int dimNb,
                      const int pad[],
                      const int stride[],
                      const int dilation[],
                      int deformable_group,
                      int conv_group,
                      int im2col_step,
                      const mluOpDataType_t compute_type);

// Group: DeformConv
/*!
 * @brief Destroys a deformable convolution descriptor \p dcn_desc that was previously created by
 * ::mluOpCreateDCNDescriptor.
 *
 * @param[in] dcn_desc
 *   The deformable convolution descriptor to be destroyed.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Note
 * - Call this function after calling the ::mluOpDCNBackwardData, ::mluOpDCNForward,
 *   or ::mluOpDCNBackwardWeight. Otherwise, \p MLUOP_STATUS_BAD_PARAM is returned.
 * - It is necessary to call this function destroy the deformable convolution descriptor.
 *   to avoid the memory leaks.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDestroyDCNDescriptor(mluOpDCNDescriptor_t dcn_desc);

// Group: DeformConv
/*!
 * @brief Returns in \p workspace_size the size of the MLU memory that is used as an extra
 *        workspace to optimize the deformable convolution forward operation.
 *
 * The size of the extra workspace is determined by the deformable convolution
 * forward operation, including the deformable convolution descriptor \p dcn_desc,
 * input tensor descriptor \p input_desc, offset tensor
 * descriptor \p offset_desc, mask tensor descriptor \p mask_desc, filter tensor descriptor
 * \p filter_desc, bias tensor descriptor \p bias_desc, and output tensor descriptor \p output_desc.
 * For more information about the workspace, see "Cambricon MLUOP User Guide."
 *
 * @param[in] handle
 *   Handle to a Cambricon MLUOP context that is used to manage MLU devices and queues in the
 *   deformable convolution operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] dcn_desc
 *   The descriptor of the deformable convolution operation. For detailed information, see
 *   ::mluOpDCNDescriptor_t.
 * @param[in] input_desc
 *   The descriptor of the input tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[in] offset_desc
 *   The descriptor of the offset tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[in] mask_desc
 *   The descriptor of the mask tensor. Set this parameter to NULL if mask is not needed. For detailed
 *   information, see ::mluOpTensorDescriptor_t.
 * @param[in] filter_desc
 *   The descriptor of the filter tensor used as a filter in the deformable convolution
 *   operation. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] bias_desc
 *   The descriptor of the bias tensor. Set this parameter to NULL if bias is not needed.
 *   For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] output_desc
 *   The descriptor of the output tensor.
 *   For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] workspace_size
 *   Pointer to the returned size of the extra workspace in bytes that is used in the
 *   deformable convolution forward operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - You must call the ::mluOpCreateTensorDescriptor and ::mluOpSetTensorDescriptor functions
 *   to create and set the tensor descriptors \p input_desc, \p offset_desc, \p mask_desc (optional),
 *   \p filter_desc, and \p bias_desc (optional) before calling this function.
 * - The allocated extra workspace must be passed to the ::mluOpDCNForward function to perform
 *   the deformable convolution forward operation.
 *
 * @par Note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetDCNForwardWorkspaceSize(mluOpHandle_t handle,
                                const mluOpDCNDescriptor_t dcn_desc,
                                const mluOpTensorDescriptor_t input_desc,
                                const mluOpTensorDescriptor_t offset_desc,
                                const mluOpTensorDescriptor_t mask_desc,
                                const mluOpTensorDescriptor_t filter_desc,
                                const mluOpTensorDescriptor_t bias_desc,
                                const mluOpTensorDescriptor_t output_desc,
                                size_t *workspace_size);

// Group: DeformConv
/*!
 * @brief Performs a 2D deformable convolution forward operation. Compared with the standard
 *        convolution, the deformable convolution introduces 2D offsets and masks to make
 *        the convolution adapt to the geometric variation of objects.
 *        Offsets act on the regular grid sampling locations, which enables a free form
 *        deformation of the sampling grid. The mask is a modulation mechanism that improves the ability
 *        to focus on pertinent image regions. Both offsets and masks are
 *        learnable parameters obtained from additional convolutional layers.
 *
 *
 * @param[in] handle
 *   Handle to a Cambricon MLUOP context that is used to manage MLU devices and queues. For
 *   detailed information, see ::mluOpHandle_t.
 * @param[in] dcn_desc
 *   The descriptor of the deformable convolution. For detailed information, see
 *   ::mluOpDCNDescriptor_t.
 * @param[in] input_desc
 *   The descriptor of the input tensor. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[in] input
 *   Pointer to the MLU memory that stores the input tensor.
 * @param[in] offset_desc
 *   The descriptor of the offset tensor to be applied to each position in the convolution kernel.
    The shape of the offset should be (batch, out_height, out_width, 2 * deformable_group *
    filter_height, filter_width). For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] offset
 *   Pointer to the MLU memory that stores the offset tensor.
 * @param[in] mask_desc
 *   The descriptor of the scaling factor to be applied to each position in the convolution
 *   kernel. The shape of the mask must be (batch, out_height, out_width,
    deformable_group  filter_height * filter_width). Set this parameter to NULL when
 *  the mask is not requested. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mask
 *   Pointer to the MLU memory that stores the mask tensor. Set this parameter to NULL
 *   when mask is not requested.
 * @param[in] filter_desc
 *   The descriptor of the filter tensor used as a filter in the deformable convolution
 *   operation. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] filter
 *   Pointer to the MLU memory that stores the filter tensor.
 * @param[in] bias_desc
 *   The descriptor of the bias tensor. Set this parameter to NULL when bias is not
 *   requested. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] bias
 *   Pointer to the MLU memory that stores the bias tensor. Set this parameter to NULL when bias is not
 *   requested.
 * @param[in] workspace
 *   Pointer to the MLU memory that is used as an extra workspace for the
 *   deformable convolution operation. For more information about workspace, see
 *   "Cambricon MLUOP User Guide".
 * @param[in] workspace_size
 *   The size of the extra workspace in bytes needed for the deformable
 *   convolution operation. You can get the size of the workspace with the
 *   ::mluOpGetDCNForwardWorkspaceSize function.
 * @param[in] output_desc
 *   The descriptor of the output tensor. The shape of output is the same with the
 *   shape of output in the convolution.
 * @param[out] output
 *   Pointer to the MLU memory that stores the output tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_NUMERICAL_OVERFLOW
 * @par Formula
 * - See "Deformable Convolution Operator" section in "Cambricon MLUOP User Guide" for details.
 *
 * @par Data Type
 * - The off-chip data type of \p input, \p offset, \p mask, \p filter, \p bias, and \p output must be the same.
 * - The supported off-chip data types of the input tensor and output tensor are as follows:
 *   - input, offset, mask, filter, bias, output: half, float.
 * - \p input offchip data type can be combined with any supported onchip data types.
 * - \p filter offchip data type can be combined with any supported onchip data types.
 * - This function also supports floating-point computation on MLU300 series or above.
 *   To perform floating-point computation, the onchip data type of \p input and \p filter
 *   should be \p MLUOP_DTYPE_INVALID or the same as the corresponding offchip data type.
 *
 * @par Data Layout
 * - The supported data layouts of the input tensor, filter, bias tensor, and output tensor are
 *   as follows:
 *   - input, offset, mask, filter, output: \p MLUOP_LAYOUT_NHWC.
 *   - bias: \p MLUOP_LAYOUT_ARRAY
 *
 * @par Scale Limitation
 * - The input, offset, mask, filter, bias, output and the deformable convolution descriptor
 *   (including pad, stride, dilation, deformable_group, conv_group, im2col_step) must meet the
 *   following requirements:
 *   - input tensor: \p batch > 0, \p height > 0, \p width > 0, \p channel > 0
 *   - offset tensor: \p batch should be equal to the batch size of input tensor, \p height and \p width
 *     should be equal to the height and width of output tensor accordingly. \p channel should be equal to
 *     deformable_group  filter_height  filter_width  2.
 *   - mask tensor: When mask is needed, \p batch should be equal to the batch size of input tensor,
 *     \p height and \p width should be equal to the height and width of output tensor accordingly.
 *     \p channel should be equal to deformable_group  filter_height * filter_width.
 *   - The value of (im2col_step  out_height  out_filter  filter_h  filter_w  input_channel)
 *     should be less than or equal to the INT_MAX defined in limits.h.
 * @par API Dependency
 * - Before calling this function to implement deformable convolution, you need to prepare
 *   all the parameters passed to this function. See each parameter description
 *   for details.
 *
 * @par Performance Optimization
 * - To achieve better performance, set the im2col_step equal to the batch
 *   size of the input tensor.
 *
 * @par Note
 * - The alignment of \p input, \p offset, \p mask, \p filter, \p bias, \p output,
 *   should be contiguous in the MLU memory.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of the deformable convolution forward operation is as follows:
     @verbatim

     input tensor by 1  3  3 * 2 --> input:
     [[[[0.7944, 0.4922], [0.2008, 0.2081], [0.9998, 0.3053]],
       [[0.1815, 0.9210], [0.8463, 0.1819], [0.9159, 0.4917]],
       [[0.6668, 0.2843], [0.8364, 0.2765], [0.7150, 0.6780]]]]
     offset tensor by 1  3  3 * 2 --> offset:
     [[[[-0.6317, -1.4928], [-0.0696,  1.1910], [ 0.8778,  0.5145]],
       [[-0.9248, -0.9889], [ 0.6157,  0.2157], [-1.1540, -0.1283]],
       [[-0.5704,  1.0237], [ 0.7956,  1.1203], [-0.0129, -0.2686]]]]
     mask tensor by 1  3  3 * 1 --> mask:
     [[[[ 0.4581], [-1.1605], [ 0.5951]],
       [[ 0.4313], [ 0.1070], [ 0.0225]],
       [[ 0.7484], [ 0.6262], [ 1.1908]]]]
     filter tensor by 2  1  1 * 2 --> filter:
     [[[[0.8928, 0.9682]]], [[[0.9301, 0.6817]]]]
     bias tensor by 2 --> bias:
     [0.4356, 0.0840]

     param:
       pad: (0, 0, 0, 0), stride: (1, 1), dilation: (1, 1)

     output tensor by 1  3  3 * 2 --> output:
     [[[[ 0.4356,  0.0840], [-0.6024, -0.9101], [ 0.8056,  0.4252]],
       [[ 0.4412,  0.0890], [ 0.5478,  0.1898], [ 0.4562,  0.1037]],
       [[ 1.1652,  0.7876], [ 0.5814,  0.2109], [ 1.8874,  1.3752]]]]
     @endverbatim
 *
 * @par Reference
 * - https://github.com/msracver/Deformable-ConvNets
 * - Deformable Convolutional Networks, Jifeng Dai, et al., 2017.
 * - Deformable ConvNets v2: More Deformable, Better Results, Xizhou Zhu, et al., 2018.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDCNForward(mluOpHandle_t handle,
                const mluOpDCNDescriptor_t dcn_desc,
                const mluOpTensorDescriptor_t input_desc,
                const void *input,
                const mluOpTensorDescriptor_t offset_desc,
                const void *offset,
                const mluOpTensorDescriptor_t mask_desc,
                const void *mask,
                const mluOpTensorDescriptor_t filter_desc,
                const void *filter,
                const mluOpTensorDescriptor_t bias_desc,
                const void *bias,
                void *workspace,
                size_t workspace_size,
                const mluOpTensorDescriptor_t output_desc,
                void *output);

// Group: DeformConv
/*!
 * @brief Returns in \p workspace_size the size of the MLU memory that is used as an extra
 *        workspace to optimize the deformable convolution backward filter operation.
 *
 * The size of the extra workspace is determined by the deformable convolution
 * backward filter operation, including the deformable convolution descriptor \p dcn_desc,
 * input tensor descriptor \p input_desc, offset tensor
 * descriptor \p offset_desc, mask tensor descriptor \p mask_desc, gradient with respect to
 * the output tensor \p grad_output_desc, the gradient with respect to the filter tensor
 * \p grad_filter_desc, and the gradient with respect to the bias tensor \p grad_bias_desc.
 * For more information about the workspace, see "Cambricon MLUOP User Guide."
 *
 * @param[in] handle
 *   Handle to a Cambricon MLUOP context that is used to manage MLU devices and queues in the
 *   deformable convolution operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] dcn_desc
 *   The descriptor of the deformable convolution operation. For detailed information, see
 *   ::mluOpDCNDescriptor_t.
 * @param[in] input_desc
 *   The descriptor of the input tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[in] offset_desc
 *   The descriptor of the offset tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[in] mask_desc
 *   The descriptor of the mask tensor. Set this parameter to NULL if mask is not needed. For detailed
 *   information, see ::mluOpTensorDescriptor_t.
 * @param[in] grad_output_desc
 *   The descriptor of the gradient with respect to the output tensor.
 *   For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] grad_filter_desc
 *   The descriptor of the gradient with respect to the filter tensor.
 *   For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] grad_bias_desc
 *   The descriptor of the gradient with respect to the bias tensor.
 *   Set this parameter to NULL if the gradient with respect to bias is not needed.
 *   For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] workspace_size
 *   Pointer to the returned size of the extra workspace in bytes that is used in the
 *   deformable convolution backward filter operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - You must call the ::mluOpCreateTensorDescriptor and ::mluOpSetTensorDescriptor functions
 *   to create and set the tensor descriptors \p input, \p offset, \p mask (optional),
 *   \p grad_output, \p grad_filter, and \p grad_bias (optional) before calling this
 *   function.
 * - The allocated extra workspace must be passed to the ::mluOpDCNBackwardWeight function to
 *   perform the deformable convolution backward filter operation.
 *
 * @par Note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */

mluOpStatus_t MLUOP_WIN_API
mluOpGetDCNBackwardWeightWorkspaceSize(mluOpHandle_t handle,
                                       const mluOpDCNDescriptor_t dcn_desc,
                                       const mluOpTensorDescriptor_t input_desc,
                                       const mluOpTensorDescriptor_t offset_desc,
                                       const mluOpTensorDescriptor_t mask_desc,
                                       const mluOpTensorDescriptor_t grad_output_desc,
                                       const mluOpTensorDescriptor_t grad_filter_desc,
                                       const mluOpTensorDescriptor_t grad_bias_desc,
                                       size_t *workspace_size);

// Group: DeformConv
/*!
 * @brief Performs the back-propagation of a deformable convolution operation to compute
 *        the gradient with respect to filter \p grad_filter and bias \p grad_bias
 *        based on the gradient of response \p grad_output.
 *
 * This function needs extra MLU memory as the workspace to improve the performance.
 * You can get the size of the workspace \p workspace_size with the
 * ::mluOpGetDCNBackwardWeightWorkspaceSize function.
 *
 * @param[in] handle
 *   Handle to a Cambricon MLUOP context that is used to manage MLU devices and
 *   queues in the deformable convolution backward filter operation. For detailed information,
 *   see ::mluOpHandle_t.
 * @param[in] dcn_desc
 *   The descriptor of the deformable convolution operation. For detailed information,
 *   see ::mluOpDCNDescriptor_t.
 * @param[in] input_desc
 *   The descriptor of the input tensor. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[in] input
 *   Pointer to the MLU memory that stores the input tensor.
 * @param[in] offset_desc
 *   The descriptor of the offset to be applied to each position in the convolution kernel.
    The shape of offset should be (batch, out_height, out_width, 2  deformable_group *
    weight_height  filter_width). For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] offset
 *   Pointer to the MLU memory that stores the offset tensor.
 * @param[in] mask_desc
 *   The descriptor of the scaling factor to be applied to each position in the convolution
 *   kernel. The shape of the mask must be (batch, out_height, out_width,
    deformable_group  filter_height * filter_width). Set this parameter to NULL when
 *   mask is not requested. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mask
 *   Pointer to the MLU memory that stores the mask tensor. Set this parameter to NULL when mask is not
 *   requested.
 * @param[in] grad_output_desc
 *   The descriptor of the gradient with respect to the output tensor.
 *   For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] grad_output
 *   Pointer to the MLU memory that stores the gradient with respect to the output tensor.
 * @param[in] workspace
 *   Pointer to the MLU memory that is used as an extra workspace for the
 *   deformable convolution backward filter operation. For more information about workspace,
 *   see "Cambricon MLUOP User Guide".
 * @param[in] workspace_size
 *   The size of the extra workspace in bytes needed for
 *   the deformable convolution backward filter operation. You can get the size of the workspace
 *   with the ::mluOpGetDCNBackwardWeightWorkspaceSize function.
 * @param[in] grad_filter_desc
 *   The descriptor of the gradient with respect to the filter tensor.
 *   For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] grad_filter
 *   Pointer to the MLU memory that stores the gradient with respect to the filter tensor.
 * @param[in] grad_bias_desc
 *   The descriptor of the gradient with respect to the bias tensor. Set this parameter to NULL if the
 *   gradient of the bias tensor is not needed. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[out] grad_bias
 *   Pointer to the MLU memory that stores the gradient with respect to the bias tensor.
 *   Set this parameter to NULL if the gradient of the bias tensor is not needed.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_NUMERICAL_OVERFLOW
 *
 * @par Formula
 * - See "Deformable Convolution Operator" section in "Cambricon MLUOP User Guide" for details.
 *
 * @par Data Type
 * - The off-chip data type of \p input, \p offset, \p mask, \p grad_output, \p grad_filter,
 *   and \p grad_bias must be the same.
 * - The supported off-chip data types of the input tensor and output tensor are as follows:
 *   - input, offset, mask, grad_output, grad_filter, grad_bias, grad_mask: half, float.
 * - \p grad_output off-chip data type can be combined with any supported on-chip data types.
 * - \p input off-chip data type can be combined with any supported on-chip data types.
 * - This function also supports floating-point computation on MLU300 series or above. To perform
 *   floating-point computation, the on-chip data type of \p input and \p grad_output should be
 *   \p MLUOP_DTYPE_INVALID or the same as the corresponding off-chip data type.
 *
 * @par Data Layout
 * - The data layout of the input, offset, mask, grad_output, and grad_filter
 *   should be \p MLUOP_LAYOUT_NHWC.
 * - The data layout of grad_bias should be \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - The input, offset, mask, grad_output, grad_filter, grad_bias and
 *   the deformable convolution descriptor
 *   (including pad, stride, dilation, deformable_group, conv_group, im2col_step) must meet the
 *   following requirements:
 *   - input tensor: \p batch > 0, \p height > 0, \p width > 0, \p channel > 0
 *   - offset tensor: \p batch should be equal to the batch of input tensor, \p height and \p width
 *     should be equal to the height and width of output tensor. \p channel should be equal to
      deformable_group  filter_height  filter_width  2.
 *   - mask tensor: When mask is needed, \p batch should be equal to the batch of input tensor,
 *     \p height and \p width should be equal to the height and width of output tensor.
      \p channel should be equal to deformable_group  filter_height * filter_width.
 *   - grad bias tensor: When the gradient of bias is needed, the \p grad_bias should be a
 *     one-dimensional array with the length of \p out_channel.
    - The value of (im2col_step  out_height  out_filter  filter_h  filter_w  input_channel)
 *     should be less than or equal to the INT_MAX defined in limits.h.

 * @par API Dependency
 * - Before calling this function to implement the backward filter of deformable convolution,
 *   you need to prepare all the parameters passed to this function. See each parameter
 *   description for details.
 *
 * @par Performance Optimization
 * - To achieve better performance, set the im2col_step to the batch size.
 *
 * @par Note
 * - The alignment of \p input, \p offset, \p mask, \p grad_output, \p grad_filter, \p grad_bias
 *   should be contiguous in the MLU memory.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/msracver/Deformable-ConvNets
 * - Deformable Convolutional Networks, Jifeng Dai, et al., 2017.
 * - Deformable ConvNets v2: More Deformable, Better Results, Xizhou Zhu, et al., 2018.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDCNBackwardWeight(mluOpHandle_t handle,
                       const mluOpDCNDescriptor_t dcn_desc,
                       const mluOpTensorDescriptor_t input_desc,
                       const void *input,
                       const mluOpTensorDescriptor_t offset_desc,
                       const void *offset,
                       const mluOpTensorDescriptor_t mask_desc,
                       const void *mask,
                       const mluOpTensorDescriptor_t grad_output_desc,
                       const void *grad_output,
                       void *workspace,
                       size_t workspace_size,
                       const mluOpTensorDescriptor_t grad_filter_desc,
                       void *grad_filter,
                       const mluOpTensorDescriptor_t grad_bias_desc,
                       void *grad_bias);

// Group: DeformConv
/*!
 * @brief Returns in \p workspace_size the size of the MLU memory that is used as an extra
 * workspace to optimize the deformable convolution backward data operation.
 *
 * The size of the extra workspace is based on the information of the deformable convolution
 * backward data operation, including the deformable convolution descriptor \p dcn_desc,
 * input tensor descriptor \p input_desc, offset tensor
 * descriptor \p offset_desc, mask tensor descriptor \p mask_desc, filter tensor descriptor
 * \p filter_desc, gradient with respect to the output tensor \p grad_output_desc, the gradient
 * with respect to the input tensor \p grad_input_desc, the gradient with respect to the offset
 * tensor \p grad_offset, and the gradient with respect to the mask tensor \p grad_mask_desc.
 * For more information about the workspace, see "Cambricon MLUOP User Guide."
 *
 * @param[in] handle
 *   Handle to a Cambricon MLUOP context that is used to manage MLU devices and queues in the
 *   deformable convolution operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] dcn_desc
 *   The descriptor of the deformable convolution operation. For detailed information, see
 *   ::mluOpDCNDescriptor_t.
 * @param[in] input_desc
 *   The descriptor of the input tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[in] offset_desc
 *   The descriptor of the offset tensor. For detailed information, see
 *   ::mluOpTensorDescriptor_t.
 * @param[in] mask_desc
 *   The descriptor of the mask tensor. Set this parameter to NULL if mask is not needed. For detailed
 *   information, see ::mluOpTensorDescriptor_t.
 * @param[in] filter_desc
 *   The descriptor of the filter tensor as a filter in the deformable convolution
 *   operation. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] grad_output_desc
 *   The descriptor of the gradient with respect to the output tensor.
 *   For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] grad_input_desc
 *   The descriptor of the gradient with respect to the input tensor.
 *   This parameter is requested to be the same as \p input_desc. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[in] grad_offset_desc
 *   The descriptor of the gradient with respect to the offset tensor.
 *   This parameter is requested to be the same as \p offset_desc.
 *   For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] grad_mask_desc
 *   The descriptor of the gradient with respect to the mask tensor.
 *   This parameter is requested to be the same with \p mask_desc. Set this parameter to NULL when mask and
 *   grad_mask are not needed. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] workspace_size
 *   Pointer to the returned size of the extra workspace in bytes that is used in the
 *   deformable convolution backward data operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - You need to call the ::mluOpCreateTensorDescriptor and ::mluOpSetTensorDescriptor functions
 *   to create and set the tensor descriptors \p input, \p offset, \p mask (optional), \p filter,
 *   \p grad_output, \p grad_input, \p grad_offset, \p grad_mask (optional) before calling this
 *   function.
 * - The allocated extra workspace must be passed to the ::mluOpDCNBackwardData function to perform
 *   the deformable convolution backward data operation.
 *
 * @par Note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetDCNBakcwardDataWorkspaceSize(mluOpHandle_t handle,
                                     const mluOpDCNDescriptor_t dcn_desc,
                                     const mluOpTensorDescriptor_t input_desc,
                                     const mluOpTensorDescriptor_t offset_desc,
                                     const mluOpTensorDescriptor_t mask_desc,
                                     const mluOpTensorDescriptor_t filter_desc,
                                     const mluOpTensorDescriptor_t grad_output_desc,
                                     const mluOpTensorDescriptor_t grad_input_desc,
                                     const mluOpTensorDescriptor_t grad_offset_desc,
                                     const mluOpTensorDescriptor_t grad_mask_desc,
                                     size_t *workspace_size);

// Group: DeformConv
/*!
 * @brief Performs the back-propagation of a deformable convolution operation to compute
 * the gradient with respect to input \p grad_input, offset \p grad_offset, and mask
 * \p grad_mask based on the gradient of response \p grad_output.
 *
 * This function needs extra MLU memory as the workspace to improve the performance.
 * You can get the workspace size with the ::mluOpGetDCNBakcwardDataWorkspaceSize
 * function.
 *
 * @param[in] handle
 *   Handle to a Cambricon MLUOP context that is used to manage MLU devices and
 *   queues in the deformable convolution backward data operation. For detailed information,
 *   see ::mluOpHandle_t.
 * @param[in] dcn_desc
 *   The descriptor of the deformable convolution operation. For detailed information,
 *   see ::mluOpDCNDescriptor_t.
 * @param[in] input_desc
 *   The descriptor of the input tensor. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[in] input
 *   Pointer to the MLU memory that stores the input tensor.
 * @param[in] offset_desc
 *   The descriptor of the offset to be applied for each position in the convolution kernel.
 *   The shape of offset must be (batch, out_height, out_width, 2 * deformable_group *
 *   filter_height * filter_width). For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] offset
 *   Pointer to the MLU memory that stores the offset tensor.
 * @param[in] mask_desc
 *   The descriptor of the scaling factor to be applied to each position in the convolution
 *   kernel. The shape of mask must be (batch, out_height, out_width,
 *   deformable_group * filter_height * filter_width). Set this parameter to NULL when
 *   mask is not requested. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] mask
 *   Pointer to the MLU memory that stores the mask tensor. Set this parameter to NULL when mask is not
 *   requested.
 * @param[in] filter_desc
 *   The descriptor of the filter tensor used as a filter in the deformable convolution operation.
 *   For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] filter
 *   Pointer to the MLU memory that stores the filter tensor.
 * @param[in] grad_output_desc
 *   The descriptor of the gradient with respect to the output tensor.
 *   For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] grad_output
 *   Pointer to the MLU memory that stores the gradient with respect to the output.
 * @param[in] workspace
 *   Pointer to the MLU memory that is used as an extra workspace for the
 *   deformable convolution backward data operation. For more information about workspace,
 *   see "Cambricon MLU-OPS User Guide".
 * @param[in] workspace_size
 *   The size of the extra workspace in bytes needed for
 *   the deformable convolution backward data operation. You can get the size of the workspace
 *   with the ::mluOpGetDCNBakcwardDataWorkspaceSize function.
 * @param[in] grad_input_desc
 *   The descriptor of the gradient with respect to the input tensor.
 *   This parameter is requested to be the same as \p input_desc. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[out] grad_input
 *   Pointer to the MLU memory that stores the gradient with respect to \p input.
 * @param[in] grad_offset_desc
 *   The descriptor of the gradient with respect to the offset tensor.
 *   This parameter is requested to be the same as \p offset_desc.
 *   For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] grad_offset
 *   Pointer to the MLU memory that stores the gradient with respect to \p offset.
 * @param[in] grad_mask_desc
 *   The descriptor of the gradient with respect to the mask tensor.
 *   This parameter is requested to be the same with \p mask_desc. Set this parameter to NULL when mask and
 *   grad_mask are not needed. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] grad_mask
 *   Pointer to the MLU memory that stores the gradient with respect to \p mask.
 *   Set this parameter to NULL when mask and grad_mask are not needed.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_NUMERICAL_OVERFLOW
 *
 * @par Formula
 * - See "Deformable Convolution Operator" section in "Cambricon MLU-OPS User Guide" for details.
 *
 * @par Data Type
 * - Offchip data type of \p input, \p offset, \p mask, \p filter, \p grad_output,
 *   \p grad_input, \p grad_offset, and \p grad_mask must be the same.
 * - The supported offchip data types of the input tensor and output tensor are as follows:
 *   - input, offset, mask, filter, grad_output, grad_input, grad_offset, grad_mask: half, float.
 * - This function supports any combinations of the following onchip data types for input tensor
 * - \p grad_output offchip data type can be combined with any supported onchip data types.
 * - \p filter offchip data type can be combined with any supported onchip data types.
 * - This function also supports floating-point computation on MLU300 series or above. To perform
 *   floating-point computation, the onchip data type of \p grad_output and \p filter must be
 *   \p MLUOP_DTYPE_INVALID or the same as the corresponding offchip data type.
 *
 * @par Data Layout
 * - The data layout of the input, offset, mask, filter, grad_output, grad_input, grad_offset,
 *   and grad_mask must be \p MLUOP_LAYOUT_NHWC.
 *
 * @par Scale Limitation
 * - The input, offset, mask, filter, grad_output, grad_input, grad_offset, grad_mask and
 *   the deformable convolution descriptor
 *   (including pad, stride, dilation, deformable_group, conv_group, im2col_step) must meet the
 *   following requirements:
 *   - input tensor: \p batch > 0, \p height > 0, \p width > 0, \p channel > 0
 *   - offset tensor: \p batch must be equal to the batch size of input tensor, \p height and \p width
 *     must be equal to the height and width of output tensor accordingly. \p channel must be equal to
 *     deformable_group * filter_height * filter_width * 2.
 *   - grad offset tensor: the data type, layout, and shape of grad offset must be equal to the
 *     offset tensor.
 *   - mask tensor: When mask is needed, \p batch must be equal to the batch size of input tensor,
 *     \p height and \p width must be equal to the height and width of output tensor accordingly.
 *     \p channel must be equal to deformable_group * filter_height * filter_width.
 *   - grad mask tensor: the data type, layout and shape of the grad mask must be equal to
 *     the mask tensor. When mask is passed NULL, grad mask must be NULL.
 *   - The data bytes of (im2col_step * out_height * out_filter * filter_h * filter_w * input_channel)
 *     must be less than or equal to the INT_MAX defined in limits.h.
 *   - When mask is not needed, \p mask, \p mask_desc, \p grad_mask and \p grad_mask_desc must be
 *     set to NULL. When it is needed, any of \p mask, \p mask_desc, \p grad_mask and
 *     \p grad_mask_desc cannot be NULL.
 *
 * @par API Dependency
 * - Before calling this function to implement deformable convolution backward data,
 *   you need to prepare all the parameters passed to this function. See each parameter
 *   description for details.
 *
 * @par Performance Optimization
 * - For best practices, to have better performance, set the im2col_step of to the batch size.
 *
 * @par Note
 * - \p input, \p mask, \p filter, and \p grad_output must be smaller enough to prevent the result
 *   from data overflow especially when the data type is \p MLUOP_DTYPE_HALF.
 * - \p offset with NaN is not supported on MLU300 series and lower platforms.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/msracver/Deformable-ConvNets
 * - Deformable Convolutional Networks, Jifeng Dai, et al., 2017.
 * - Deformable ConvNets v2: More Deformable, Better Results, Xizhou Zhu, et al., 2018.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDCNBackwardData(mluOpHandle_t handle,
                     const mluOpDCNDescriptor_t dcn_desc,
                     const mluOpTensorDescriptor_t input_desc,
                     const void *input,
                     const mluOpTensorDescriptor_t offset_desc,
                     const void *offset,
                     const mluOpTensorDescriptor_t mask_desc,
                     const void *mask,
                     const mluOpTensorDescriptor_t filter_desc,
                     const void *filter,
                     const mluOpTensorDescriptor_t grad_output_desc,
                     const void *grad_output,
                     void *workspace,
                     const size_t workspace_size,
                     const mluOpTensorDescriptor_t grad_input_desc,
                     void *grad_input,
                     const mluOpTensorDescriptor_t grad_offset_desc,
                     void *grad_offset,
                     const mluOpTensorDescriptor_t grad_mask_desc,
                     void *grad_mask);

/*!
 * @brief The descriptor of FFT (Fast Fourier Transform) operation that holds FFT information including

 * the tensor descriptor of input tensor and output tensor, the rank of FFT, the FFT size on each
 * dimension, the size of reserved space and the size of workspace.
 *
 * You need to call the ::mluOpCreateFFTPlan function to create a descriptor for the FFT operation, and call
 * the ::mluOpMakeFFTPlanMany function to set the information of the FFT operation to the descriptor.
 * Then, you need to allocate the reserved space and set the space to the FFT descriptor by ::mluOpSetFFTReserveArea.
 * AT the end you need to destroy the Cambricon MLUOP context with the ::mluOpDestroyFFTPlan.
 */
typedef struct mluOpFFTStruct *mluOpFFTPlan_t;

// Group: FFT
/*!
 * @brief Creates a descriptor pointed by \p fft_plan for the FFT operation, and allocates memory
 * for holding the information about the FFT operation. The information is defined in ::mluOpFFTPlan_t.
 *
 * @param[out] fft_plan
 * Pointer to the FFT descriptor that holds information about the FFT operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ALLOC_FAILED
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - After calling this function, you can call the ::mluOpMakeFFTPlanMany function to initialize and set the
 *   information to the created descriptor.
 * - You need to call the ::mluOpDestroyFFTPlan to destroy the descriptor.
 *   Otherwise, the memory leak may occur.
 *
 * @par Note
 * - This function only supports 1D and 2D FFT currently. 3D FFT
 *   will be supported in the future.
 *
 * @par Example.
 * - None.
 *
 * @par Reference.
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateFFTPlan(mluOpFFTPlan_t *fft_plan);

// Group: FFT
/*!
 * @brief Initializes the FFT descriptor pointed by \p fft_plan that is previously created
 * with the ::mluOpCreateFFTPlan function, and sets the information about the
 * tensor descriptors of input tensor and output tensor, the rank of FFT, and the FFT size on each
 * dimension.
 * This function also gets the size of MLU memory buffers for FFT execution, including \p reservespace_size and
 * \p workspace_size. The size of extra workspace is based on the given information of the
 * \p fft_plan.
 *
 * @param[in] handle
 * Handle to a Cambricon MLUOP context that is used to manage MLU devices and queues
 * in the FFT operation. For detailed information, see ::mluOpHandle_t.
 * @param[in,out] fft_plan
 * The descriptor of FFT. For detailed information, see ::mluOpFFTPlan_t.
 * @param[in] input_desc
 * The descriptor of input signals. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] output_desc
 * The descriptor of output signals. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] rank
 * The dimensionality of the FFT operation. It can be 1D, 2D or 3D.
 * @param[in] n
 * An array of size \p rank describing the FFT size of each dimension. n[0]
 * is the size of the outermost dimension and n[rank - 1] is the innermost dimension
 * of FFT operation. If n[i] is greater than the size of input on dimension i, the input
 * signal will be zero-padded on that dimension. Otherwise, input signal is trimmed
 * on the dimension i.
 * @param[out] reservespace_size
 * The size of the extra reserved space in bytes that needs to be used in FFT operation.
 * @param[out] workspace_size
 * The size of the extra workspace in bytes that needs to be used in FFT operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_NOT_INITIALIZED
 *
 * @par Data Type
 * - The supported data types of \p input and \p output tensors are as follows:
 * - real-to-complex FFT:
 *     - half(input offchip)-complex_half(output offchip)-int16(input onchip)
 *     - half(input offchip)-complex_half(output offchip)-half(input onchip)
 *     - float(input offchip)-complex_float(output offchip)-float(input onchip)
 * - complex-to-real FFT:
 *     - complex_half(input offchip)-half(output offchip)-int16(input onchip)
 *     - complex_half(input offchip)-half(output offchip)-half(input onchip)
 *     - complex_float(input offchip)-float(output offchip)-float(input onchip)
 * - complex-to-complex FFT:
 *     - complex_half(input offchip)-complex_half(output offchip)-int16(input onchip)
 *     - complex_half(input offchip)-complex_half(output offchip)-half(input onchip)
 *     - complex_float(input offchip)-complex_float(output offchip)-float(input onchip)
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - Before calling this function, you need to call the ::mluOpCreateFFTPlan function to
 *   create an FFT descriptor, call the ::mluOpSetTensorDescriptor or
 *   ::mluOpSetTensorDescriptorEx function to set the input and output tensor descriptor,
 *   and then call the ::mluOpSetTensorDescriptorOnchipDataType to set the onchip data type
 *   of input tensor descriptor.
 *
 * @par Note
 * - The advanced data layout parameters including (i/o)nembed, (i/o)istride and (i/o)idist, are set through
 *   ::mluOpSetTensorDescriptorEx. If stride information is not needed, you can set the simple data layout
 *   through ::mluOpSetTensorDescriptor.
 * - The dimension size of input or output should be equal to \p rank or \p rank + 1. In the former case,
 *   the batch size is considered as 1. Otherwise, the outermost dimension is the batch size.
 * - For real-to-complex FFTs, the innermost dimension of FFT length and output arrays are not the same.
 *   For a x-length 1D real-to-complex FFT, the output is x/2 + 1 complex numbers (the non-redundant outputs).
 *   For a N-D real-to-complex FFT with n=[z, y, x], the output shape will be [z, y, x/2+1].
 * - For complex-to-real FFTs, the input tensor only holds the non-redundant part of the Fourier coefficients.
 *   And the output tensor stores the real output values.
 * - When n[0] is greater than 4096, the data type of input only supports float or complex_float.
 *
 * @par Example.
 * - None.
 *
 * @par Reference.
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpMakeFFTPlanMany(mluOpHandle_t handle,
                     mluOpFFTPlan_t fft_plan,
                     const mluOpTensorDescriptor_t input_desc,
                     const mluOpTensorDescriptor_t output_desc,
                     const int rank,
                     const int n[],
                     size_t *reservespace_size,
                     size_t *workspace_size);

// Group:FFT
/*!
 * @brief Bonds the \p reservespace to the \p fft_plan. The size of reserved space can be derived
 * through ::mluOpMakeFFTPlanMany.
 *
 * @param[in] handle
 * Handle to a Cambricon MLUOP context that is used to manage MLU devices and queues in the
 * ::mluOpExecFFT. For detailed information, see ::mluOpHandle_t.
 * @param[in, out] fft_plan
 * The descriptor of FFT. For detailed information, see ::mluOpFFTPlan_t.
 * @param[in] reservespace
 * Pointer to the MLU memory that is used as an extra memory space for saving
 * intermediate results of FFT operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - Before calling this function, you need to call the ::mluOpCreateFFTPlan function
 *   to create an FFT descriptor, call the ::mluOpMakeFFTPlanMany function to set the
 *   FFT descriptor and get the size of reserved space, and then call the
 *   cnrtMalloc function to create MLU memory according to the rservespace_size given.
 *
 * @par Note
 * - None.
 *
 * @par Example.
 * - None.
 *
 * @par Reference.
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetFFTReserveArea(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan, void *reservespace);

// Group:FFT
/*!
 * @brief Executes any FFT. In case of complex-to-real and real-to-complex
 * transforms, \p direction parameter is ignored. This function stores the Fourier coefficients
 * in the output array. If the address of input and output are the same, an in-place FFT
 * is adopted.
 *
 * @param[in] handle
 * Handle to a Cambricon MLUOP context that is used to manage MLU devices and queues
 * in the FFT operation. For detailed information, see ::mluOpHandle_t.
 * @param[in,out] fft_plan
 * Plan for the FFT operation. This parameter is used to store the configuration of the FFT operation.
 * @param[in,out] input
 * Input tensor for the FFT operation. This parameter is used to provide the data to be transformed.
 * @param[in,out] scale_factor
 * Scale factor applied to the FFT operation. This parameter is used to normalize the result.
 * @param[in,out] workspace
 * Workspace buffer used during the FFT operation. This parameter is used to store intermediate
 * results and other temporary data.
 * @param[in,out] output
 * Output tensor for the FFT operation. This parameter is used to store the result of the
 * FFT transformation.
 * @param[in,out] direction
 * Direction of the FFT operation. This parameter specifies whether to perform a
 * forward or inverse FFT transformation.
 * @par Note
 * - For in-place 1D real-to-complex FFTs, the input is a batch of n real numbers, and the
 *   output is n/2 + 1 non-redundant complex numbers. This requires a padding of input array.
 * - For in-place N-D real-to-complex FFTs, extra padding of the real-data array on the innermost
 *   dimension is necessary to accommodate the size of the complex-data output.
 * - For 2D FFTs, cases with strides that meet the following conditions have
 *   better performance:
 *     - real-to-complex:
 *       - n[0] < 200, n[0] == inembed[0], onembed[0] == n[0]
 *       - n[1] < 200, n[1] == inembed[1], onembed[1] == n[1]/2+1
 *       - input: dims[batch, n0, n1], strides[1, batch*n1, batch]
 *       - output: dims[batch, n0, n1/2+1], strides[1, batch*(n1/2+1), batch]
 *     - complex-to-complex:
 *       - n[0] < 200, n[0] == inembed[0], onembed[0] == n[0]
 *       - n[1] < 200, n[1] == inembed[1], onembed[1] == n[1]
 *       - input: dims[batch, n0, n1], strides[1, batch*n1, batch]
 *       - output: dims[batch, n0, n1], strides[1, batch*n1, batch]
 *     - complex-to-real:
 *       - n[0] < 200, n[0] == inembed[0], onembed[0] == n[0]
 *       - n[1] < 200, n[1]/2+1 == inembed[1], onembed[1] == n[1]
 *       - input: dims[batch, n0, n1/2+1], strides[1, batch*(n1/2+1), batch]
 *       - output: dims[batch, n0, n1], strides[1, batch*n1, batch]
 *
 * - When \p input contains NaN or infinity and the input onchip data type of FFT is not quantized
 *   data type, the output is computed through the FFT formula with computation rules of NaN or
 *   infinity based on IEEE 754.
 * - When \p input contains NaN or infinity and the input onchip data type of FFT is quantized
 *   data type such as int16, the output will be unpredictable.
 * - \p Input is recommended to be in range of [-10, 10] with uniform
 *   distribution for higher precision.
 * - \p Scale_factor is recommended to be in range of [-1, 1] to avoid exceeding
 *   the data representation range.
 * - Half data type of \p input is not recommended due to low precision. The first element of the
 *   FFT result is the sum of all input elements, and it is likely to overflow.
 * - This operation is not supported on the 1V platforms.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par Data Type
 * - The supported data types of \p input and \p output tensors are as follows:
 * - real-to-complex FFT:
 *     - half(input offchip)-complex_half(output offchip)-half(input onchip)
 *     - float(input offchip)-complex_float(output offchip)-float(input onchip)
 * - complex-to-real FFT:
 *     - complex_half(input offchip)-half(output offchip)-half(input onchip)
 *     - complex_float(input offchip)-float(output offchip)-float(input onchip)
 * - complex-to-complex FFT:
 *     - complex_half(input offchip)-complex_half(output offchip)-half(input onchip)
 *     - complex_float(input offchip)-complex_float(output offchip)-float(input onchip)
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - For float data types, FFT supports any combination of powers of i (i from 2 to 64), as well as \f$2^mL\f$.
 *   This means that for float data types, FFT can handle a wide range of sizes, allowing flexibility in choosing the
 *   dimensions of the input data. The values of i can be any integer from 2 to 64, enabling combinations such as 4, 8,
 * 16, etc., as well as sizes that are a product of a power of 2 and an additional integer L.
 *
 * - For half data types, FFT support is more limited. It only supports sizes of 2^m, where m is an integer. This
 * constraint means that the input size for half data types must be a power of 2. This restriction is important to note
 * when planning to use FFT with half-precision floating-point data, as it limits the flexibility compared to float data
 * types.
 *
 * - For FFT 2D:
 *     - real-to-complex FFT: Output numbers / 2 + 1 should not be less than input numbers.
 *     - complex-to-complex FFT: Output numbers should not be less than input numbers.
 *     - complex-to-real FFT: Output numbers should not be less than input numbers / 2 + 1.
 *
 * @par API Dependency
 * - Before calling this function, you need to call the ::mluOpCreateFFTPlan
 *   function to create an FFT descriptor, call the ::mluOpMakeFFTPlanMany
 *   function to set the FFT descriptor and the size of reserved space and work space,
 *   and then call the ::mluOpSetFFTReserveArea to bond the reservespace area to the descriptor.
 *
 * @par Note
 * - None.
 *
 * @par Example.
 * - None.
 *
 * @par Reference.
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpExecFFT(mluOpHandle_t handle,
             const mluOpFFTPlan_t fft_plan,
             const void *input,
             const float scale_factor,
             void *workspace,
             void *output,
             int direction);

// Group:FFT
/*!
 * @brief Destroys an FFT plan \p fft_plan that is created with the
 * ::mluOpCreateFFTPlan function. The FFT plan is defined in ::mluOpFFTPlan_t and
 * holds the information about the FFT operation.
 *
 * @param[in] fft_plan
 * The fft plan to be destroyed.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @par Data Type
 * - None.
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - You need to call this function after calling the ::mluOpExecFFT.
 *   Otherwise, memory leak may occur.
 *
 * @par Example.
 * - None.
 *
 * @par Reference.
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDestroyFFTPlan(mluOpFFTPlan_t fft_plan);

// Group:Lgamma
/*!
 * @brief Computes the lgamma value for every element of the input tensor \b x
 * and returns results in \b y.
 *
 * @param[in] handle
 * Handle to a Cambricon MLU-OPS context that is used to manage MLU devices and
 * queues in the lgamma operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] x_desc
 * The descriptor of the tensor \b x. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] y_desc
 * The descriptor of the tensor \b y. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] y
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH
 *
 * @par Data Type
 * - The data type of input tensor and output tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The input tensor and output tensor must have the same shape.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - Node.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://pytorch.org/docs/stable/generated/torch.lgamma.html#torch-lgamma
 */

mluOpStatus_t MLUOP_WIN_API
mluOpLgamma(mluOpHandle_t handle,
            const mluOpTensorDescriptor_t x_desc,
            const void *x,
            const mluOpTensorDescriptor_t y_desc,
            void *y);

#if defined(__cplusplus)
}
#endif

#endif  // MLUOP_EXAMPLE_H_
