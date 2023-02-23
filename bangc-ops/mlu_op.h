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

#define MLUOP_MAJOR 0
#define MLUOP_MINOR 5
#define MLUOP_PATCHLEVEL 0

#include <stdint.h>

#include "cnrt.h"
#define MLUOP_DIM_MAX 8

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
 * MLUOP Return Status
 ******************************************************************************/
/*! @brief Describes function return status.
 */
typedef enum {
  MLUOP_STATUS_SUCCESS         = 0, /*!< The operation is successfully completed. */
  MLUOP_STATUS_NOT_INITIALIZED = 1,
  /*!< MLUOP library is not initialized properly, which is usually caused by failing
       to call ::mluOpCreate, ::mluOpCreateTensorDescriptor or
       ::mluOpSetTensorDescriptor.
       Such error is usually due to incompatible MLU device or invalid driver environment.
       Notice that ::mluOpCreate should be called prior to any other MLUOP function.*/
  MLUOP_STATUS_ALLOC_FAILED = 2,
  /*!< This error occurs when the resource allocation fails, which is usually caused by
       failing to call cnMallocHost due to exceeded memory usage. Please make sure that
       the memory allocated previously is deallocated as much as possible.*/
  MLUOP_STATUS_BAD_PARAM = 3,
  /*!< Invalid value or parameters are passed to the function, including data type, layout,
       dimensions, etc.*/
  MLUOP_STATUS_INTERNAL_ERROR = 4,
  /*!< An error occurs inside of the function, which may indicate an internal error or bug in
       the library. This error is usually caused by failing to call cnrtMemcpyAsync.
       Please check whether the memory passed to the function is deallocated before the completion
       of the routine.*/
  MLUOP_STATUS_ARCH_MISMATCH = 5,
  /*!< Invalid MLU device which is not supported by current function.*/
  MLUOP_STATUS_EXECUTION_FAILED = 6,
  /*!< An error occurs when the function fails to be executed on MLU device due to multiple reasons.
       You can check whether the hardware environment, driver version and other prerequisite
       libraries are correctly installed.*/
  MLUOP_STATUS_NOT_SUPPORTED = 7,
  /*!< An error occurs when the requested functionality is not supported in
       this version but would be supported in the future. */
  MLUOP_STATUS_NUMERICAL_OVERFLOW = 8,
  /*!< A numerical overflow occurs when executing the function,
       which is usually due to large scale or inappropriate range of value of input tensor.*/
} mluOpStatus_t;

/******************************************************************************
 * MLUOP Tensor Layout
 ******************************************************************************/
/*!
 * @brief Describes the data layouts in MLUOP.
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
  /*!< The data layout is in the following order: batch size, depth, height, width, and channel.*/
  MLUOP_LAYOUT_ARRAY = 4,
  /*!< The data is multi-dimensional tensor. */
  MLUOP_LAYOUT_NCDHW = 5,
  /*!< The data layout is in the following order: batch size, channel, depth, height, and width.*/
  MLUOP_LAYOUT_TNC = 6,
  /*!< The data layout is in the following order: timing steps, batch size, alphabet size.*/
  MLUOP_LAYOUT_NTC = 7,
  /*!< The data layout is in the following order: batch size, timing steps, alphabet size.*/
  MLUOP_LAYOUT_NC = 8,
  /*!< The data layout is in the following order: batch size, channel.*/
  MLUOP_LAYOUT_NLC = 9,
  /*!< The data layout is in the following order: batch size, width, channel.*/
} mluOpTensorLayout_t;

/******************************************************************************
 * MLUOP Data Type
 ******************************************************************************/
/*! @brief Describes the data types in MLUOP. */
typedef enum {
  MLUOP_DTYPE_INVALID       = 0,  /*!< An invalid data type. */
  MLUOP_DTYPE_HALF          = 1,  /*!< A 16-bit floating-point data type. */
  MLUOP_DTYPE_FLOAT         = 2,  /*!< A 32-bit floating-point data type. */
  MLUOP_DTYPE_DOUBLE        = 14, /*!< A 64-bit floating-point data type. */
  MLUOP_DTYPE_INT8          = 3,  /*!< An 8-bit signed integer data type. */
  MLUOP_DTYPE_INT16         = 4,  /*!< A 16-bit signed integer data type. */
  MLUOP_DTYPE_INT32         = 6,  /*!< A 32-bit signed integer data type. */
  MLUOP_DTYPE_INT64         = 9,  /*!< A 64-bit signed integer data type. */
  MLUOP_DTYPE_UINT8         = 7,  /*!< An 8-bit unsigned integer data type. */
  MLUOP_DTYPE_UINT16        = 13, /*!< A 16-bit unsigned integer data type. */
  MLUOP_DTYPE_UINT32        = 11, /*!< A 32-bit unsigned integer data type. */
  MLUOP_DTYPE_UINT64        = 12, /*!< A 64-bit unsigned integer data type. */
  MLUOP_DTYPE_BOOL          = 8,  /*!< A boolean data type. */
  MLUOP_DTYPE_COMPLEX_HALF  = 15, /*!< A 32-bit complex number of two fp16. */
  MLUOP_DTYPE_COMPLEX_FLOAT = 16, /*!< A 64-bit complex number of two fp32. */
} mluOpDataType_t;

/*!
 * @brief Describes whether to propagate NaN numbers.
 */
typedef enum {
  MLUOP_NOT_PROPAGATE_NAN = 0,
  /*!< The NaN numbers are not propagated .*/
  MLUOP_PROPAGATE_NAN = 1,
  /*!< The NaN numbers are propagated.*/
} mluOpNanPropagation_t;

/*!
 * @brief Describes the options that can help choose
 * the best suited algorithm used for implementation of the activation
 * and accumulation operations.
 **/
typedef enum {
  MLUOP_COMPUTATION_FAST = 0,
  /*!< Implementation with the fastest algorithm and lower precision.*/
  MLUOP_COMPUTATION_HIGH_PRECISION = 1,
  /*!< Implementation with the high-precision algorithm regardless of the performance.*/
} mluOpComputationPreference_t;

/*!
 * @brief Describes the atomics modes in MLUOP.
 */
typedef enum {
  MLUOP_ATOMICS_NOT_ALLOWED = 1,
  /*!< The atomics is not allowed to cumulate results.*/
  MLUOP_ATOMICS_ALLOWED = 2,
  /*!< The atomics is allowed to cumulate results */
} mluOpAtomicsMode_t;

/*!
 * @brief Describes the rounding modes of quantization conversion.
 */
typedef enum {
  MLUOP_ROUND_HALF_TO_EVEN = 0,
  /*!< The rounding mode to round towards the nearest even neighbor
   *   is used for quantization conversion.*/
  MLUOP_ROUND_HALF_UP = 1,
  /*!< The rounding mode to round up towards the nearest neighbor is
   *   used for quantization conversion.*/
  MLUOP_ROUND_HALF_OFF_ZERO = 2,
  /*!< The rounding mode to round half away from zero is
   *   used for quantization conversion.*/
} mluOpQuantizeRoundMode_t;

/*!
 * @brief Describes the modes of quantization method.
 */
typedef enum {
  MLUOP_QUANTIZE_POSITION = 0,
  /*!< Quantization method with position factor and without scale factor.*/
  MLUOP_QUANTIZE_POSITION_SCALE = 1,
  /*!< Quantization method with position and scale factors.*/
  MLUOP_QUANTIZE_POSITION_SCALE_OFFSET = 2,
  /*!< Asymmetric quantization method with position, scale, and offset factors.*/
} mluOpQuantizeMode_t;

/*!
 * @brief Describes the bases that are used in the implementation
 * of the log function.
 */
typedef enum {
  MLUOP_LOG_E  = 0, /*!< The base e is used.*/
  MLUOP_LOG_2  = 1, /*!< The base 2 is used.*/
  MLUOP_LOG_10 = 2, /*!< The base 10 is used.*/
} mluOpLogBase_t;

/*!
 * @brief Describes the pointer modes that are used in the implementation
 * of the fill function.
 */
typedef enum {
  MLUOP_POINTER_MODE_HOST = 0,
  /*!< A host pointer, which means that the values passed by reference are on
     the host. */
  MLUOP_POINTER_MODE_DEVICE = 1,
  /*!< A device pointer, which means that the values passed by reference are on
     the device. */
} mluOpPointerMode_t;

/******************************************************************************
 * MLUOP Data Structure: Customized Operation
 ******************************************************************************/
/*!
 * @brief Describes the attributes of
 * the matrix multiplication computation.
 */
typedef enum {
  MLUOP_MATMUL_DESC_COMPUTE_TYPE = 0,
  /*!< Defines the data type used for multiplication and accumulation operations, and the
   *   accumulator for implementing matrix multiplication. It must be set before
   *   doing matrix multiplication. */
  MLUOP_MATMUL_DESC_SCALE_TYPE = 1,
  /*!< Defines the data type of the scaling factors \b alpha and \b beta. The default value
   *   is the same as ::MLUOP_MATMUL_DESC_COMPUTE_TYPE. It is not supported now. */
  MLUOP_MATMUL_DESC_POINTER_MODE = 2,
  /*!< Specifies whether \b alpha and \b beta are stored on the host or on the device.
   *   It is not supported now. */
  MLUOP_MATMUL_DESC_TRANSA = 3,
  /*!< Specifies whether the transpose should be performed on matrix A. The default
   *   value is 0 (false). */
  MLUOP_MATMUL_DESC_TRANSB = 4,
  /*!< Specifies whether the transpose should be performed on matrix B. The default
   *   value is 0 (false). */
  MLUOP_MATMUL_DESC_TRANSC = 5,
  /*!< Specifies whether the transpose should be performed on matrix C. The default
   *   value is 0 (false). It is not supported now. */
  MLUOP_MATMUL_DESC_EPILOGUE = 6,
  /*!< Specifies the epilogue function. It is not supported now. */
  MLUOP_MATMUL_DESC_BIAS_POINTER = 7,
  /*!< Pointer to bias vector on MLU device memory. Currently, it is only supported to set
   *   the attribute \b matmul_desc in ::mluOpMatMulInference. */
  MLUOP_MATMUL_DESC_EPILOGUE_TYPE = 8,
  /*!< Specifies matmul multiplication epilogue fusion type. */
  MLUOP_MATMUL_DESC_EPILOGUE_OPERAND = 9,
  /*!< Specifies matmul multiplication epilogue fusion operand. */
  MLUOP_MATMUL_ALLOW_TF32 = 10,
  /*!< Determines whether to enable TensorFloat-32 mode.
   *   TensorFloat-32 is enabled by default. */
  MLUOP_MATMUL_USE_BETA = 11,
  /*!< Specifies whether to use \b beta on matrix C. */
  MLUOP_MATMUL_CAST_MODE = 12,
  /*!< Specifies the quantization mode used for the matrix multiplication quantization. */
  MLUOP_MATMUL_USE_STRIDE = 13,
  /*!< Specifies whether stride should be performed on tensor. */
} mluOpMatMulDescAttribute_t;

/*!
 * @brief Describes the preference of matrix multiplication algorithm.
 */
typedef enum {
  MLUOP_MATMUL_FASTEST = 0,
  /*!< The high-speed preference is used. */
  MLUOP_MATMUL_LOW_MEMORY_OCCUPY = 1,
  /*!< The low-memory preference is used. This is not supported now. */
} mluOpMatMulPreference_t;

/*!
 * @brief Describes the unique modes that can be used to implement
 * the unique operation.
 */
typedef enum {
  MLUOP_UNSORT_FORWARD = 0,
  /*!< Returns the data in the same order as the input data after eliminating the
   * duplicated values.*/
  MLUOP_SORT_ASCEND = 1,
  /*!< Returns the data sorted in ascending order by input value after eliminating
   * the duplicated values.*/
  MLUOP_UNSORT_REVERSE = 2,
  /*!< Returns the data in the reversed order as the input data after eliminating
   * the duplicated values.*/
} mluOpUniqueSort_t;

/*!
 * @brief Describes the modes that are used in the
 * implementation of scatter_nd operation.
 */
typedef enum {
  MLUOP_SCATTERND_ADD = 0,
  /*!< The ADD operation is implemented.*/
  MLUOP_SCATTERND_SUB = 1,
  /*!< The SUB (subtraction) operation is implemented.
   * This mode is not supported currently.*/
  MLUOP_SCATTERND_MUL = 2,
  /*!< The MUL (multiplication) operation is implemented.
   * This mode is not supported currently.*/
  MLUOP_SCATTERND_UPDATE = 3,
  /*!< The replacement operation is implemented.*/
} mluOpScatterNdMode_t;

/*!
 * @brief Describes the modes that are used in the implementation of the Reduce function.
 */
typedef enum {
  MLUOP_REDUCE_ADD            = 0, /*!< The reduce addition operation is implemented.*/
  MLUOP_REDUCE_AVG            = 1, /*!< The reduce average operation is implemented.*/
  MLUOP_REDUCE_MUL            = 2, /*!< The reduce multiplication operation is implemented.*/
  MLUOP_REDUCE_MAX            = 3, /*!< The reduce maximum operation is implemented.*/
  MLUOP_REDUCE_MIN            = 4, /*!< The reduce minimum operation is implemented.*/
  MLUOP_REDUCE_AND            = 5, /*!< The reduce and operation is implemented.*/
  MLUOP_REDUCE_OR             = 6, /*!< The reduce or operation is implemented.*/
  MLUOP_REDUCE_NORM1          = 7, /*!< The sum of absolute values operation is implemented.*/
  MLUOP_REDUCE_NORM2          = 8, /*!< The square root of sum of squares operation is implemented.*/
  MLUOP_REDUCE_MAX_LAST_INDEX = 9,
  /*!< The operation of returning the index of the last maximum value is implemented.*/
  MLUOP_REDUCE_MIN_LAST_INDEX = 10,
  /*!< The operation of returning the index of the last minimum value is implemented.*/
  MLUOP_REDUCE_NORMP = 11, /*!< The 1/p power of sum of p power operation is implemented.*/
  MLUOP_REDUCE_ASUM  = 12,
  /*!< The sum of absolute values operation adapted to Caffe framework is implemented.*/
  MLUOP_REDUCE_SUMSQ = 13,
  /*!< The sum of the squared values operation adapted to Caffe framework is implemented.*/
} mluOpReduceOp_t;

/*!
 * @brief Describes whether the indices are computed in the implementation of the reduce function.
 */
typedef enum {
  MLUOP_REDUCE_NO_INDICES        = 0, /*!< The indices are not computed.*/
  MLUOP_REDUCE_FLATTENED_INDICES = 1, /*!< The indices and the corresponding values are computed.*/
  MLUOP_REDUCE_ONLY_INDICES      = 2, /*!< Only the indices are calculated.*/
} mluOpReduceIndices_t;

/*!
 * @brief Describes the data type of indices used in the reduce function.
 */
typedef enum {
  MLUOP_32BIT_INDICES = 0, /*!< The data type of indices is unsigned int.*/
  MLUOP_16BIT_INDICES = 1, /*!< The data type of indices is unsigned short.*/
} mluOpIndicesType_t;

/******************************************************************************
 * MLUOP Runtime Management
 ******************************************************************************/

/*!
 * @struct mluOpContext
 * @brief Describes the MLUOP context.
 */
struct mluOpContext;
/*!
 * A pointer to ::mluOpContext struct that holds the MLUOP context.
 *
 * MLU device resources cannot be accessed directly, so MLUOP uses
 * handle to manage MLUOP context including MLU device information
 * and queues.
 *
 * The MLUOP context is created with ::mluOpCreate and the returned
 * handle should be passed to all the subsequent function calls.
 * You need to destroy the MLUOP context at the end with ::mluOpDestroy.
 */
typedef struct mluOpContext *mluOpHandle_t;

/*! The descriptor of the collection of tensor which is used in the RNN operation, such as weight,
 *  bias.
 *  You need to call the ::mluOpCreateTensorSetDescriptor function to create a descriptor, and
 *  call the ::mluOpInitTensorSetMemberDescriptor to set the information about each tensor in
 *  the tensor set. If the data type of the tensor in the tensor set is in fixed-point data type,
 *  call ::mluOpInitTensorSetMemberDescriptorPositionAndScale function to set quantization
 *  parameters.
 *  At last, you need to destroy the descriptor at the end with the
 *  ::mluOpDestroyTensorSetDescriptor function.
 */
typedef struct mluOpTensorSetStruct *mluOpTensorSetDescriptor_t;

// Group:Runtime Management
/*!
 *  @brief Initializes the MLUOP library and creates a handle \b handle to a structure
 *  that holds the MLUOP library context. It allocates hardware resources on the host
 *  and device. You need to call this function before any other MLUOP function.
 *
 *  You need to call the ::mluOpDestroy function to release the resources later.
 *
 *  @param[out] handle
 *  Pointer to an MLUOP context that is used to manage MLU devices and
 *  queues. For detailed information, see ::mluOpHandle_t.
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
mluOpCreate(mluOpHandle_t *handle);

// Group:Runtime Management
/*!
 *  @brief Updates the MLUOP context information that is held by the \b handle. This function
 *  should be called if you call Cambriocn CNDrv API cnSetCtxConfigParam to set the context
 *  information. The related context information will be synchronized to MLUOP with this function.
 *  For detailed information, see "Cambricon CNDrv Developer Guide".
 *
 *  @param[in] handle
 *  Pointer to an MLUOP context that is used to manage MLU devices.
 *  For detailed information, see ::mluOpHandle_t.
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
mluOpUpdateContextInformation(mluOpHandle_t handle);

// Group:Runtime Management
/*!
 *  @brief Releases the resources of the specified MLUOP handle \b handle that was
 *  created by the ::mluOpCreate function.
 *  It is usually the last call to destroy the handle to the MLUOP handle.
 *
 *  @param[in] handle
 *  Pointer to the MLU devices that holds information to be destroyed.
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
mluOpDestroy(mluOpHandle_t handle);

// Group:Runtime Management
/*!
 *  @brief Sets the runtime queue \b queue in the handle \b handle. The queue is used to
 *  launch kernels or to synchronize to this queue.
 *
 *  Before setting a queue \b queue, you need to call the ::mluOpCreate function to initialize
 *  MLUOP library, and call the cnrtCreateQueue function to create a queue \b queue.
 *
 *  @param[in] handle
 *  Handle to an MLUOP context that is used to manage MLU devices and
 *  queues. For detailed information, see ::mluOpHandle_t.
 *  @param[in] queue
 *  The runtime queue to be set to the MLUOP handle.
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
mluOpSetQueue(mluOpHandle_t handle, cnrtQueue_t queue);

// Group:Runtime Management
/*!
 *  @brief Retrieves the queue \b queue that was previously set to the handle \b handle.
 *
 *  @param[in] handle
 *  Handle to an MLUOP context that is used to manage MLU devices and
 *  queues. For detailed information, see ::mluOpHandle_t.
 *  @param[out] queue
 *  Pointer to the queue that was previously set to the specified handle.
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
mluOpGetQueue(mluOpHandle_t handle, cnrtQueue_t *queue);

// Group:Runtime Management
/*!
 *  @brief Converts the MLUOP enumerated status code to ASCIIZ static string and returns
 *  a pointer to the MLU memory that holds information about ASCIIZ static string with the status
 *  name.
 *  For example, when the input argument is
 *  ::MLUOP_STATUS_SUCCESS, the returned string is MLUOP_STATUS_SUCCESS. When an invalid status
 *  value is passed to the function, the returned string is ::MLUOP_STATUS_BAD_PARAM.
 *
 *  @param[in] status
 *  The MLUOP enumerated status code.
 *  @return
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
 */
const char *
mluOpGetErrorString(mluOpStatus_t status);

// Group:Tensor
/*!
 *  @brief Gets the size of a data type in ::mluOpDataType_t.
 *
 *  @param[in] data_type
 *  The data type. For detailed information, see ::mluOpDataType_t.
 *  @param[out] size
 *  Host pointer to the size of the data type.
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
mluOpGetSizeOfDataType(mluOpDataType_t data_type, size_t *size);

// Group:Version Management
/*!
 *  @brief Retrieves the version of MLUOP library. The version of MLUOP
 *  is composed of \b major, \b minor and \b patch. For instance, major = 1,
 *  minor = 2, patch = 3, the version of MLUOP library is 1.2.3.
 *
 *  @param[in] major
 *  A pointer to scale factor that gets the major version of MLUOP
 *  library.
 *  @param[in] minor
 *  A pointer to scale factor that gets the minor version of MLUOP
 *  library.
 *  @param[in] patch
 *  A pointer to scale factor that gets the patch version of MLUOP
 *  library.
 *
 *  @note
 *  - None.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 * */
void
mluOpGetLibVersion(int *major, int *minor, int *patch);

// Group:QuantizeRoundMode
/*!
 *  @brief Updates the specific rounding mode of MLUOP context information that holds by the \b
 *  handle. This function should be called if you want to change the MLUOP rounding mode that is used
 *  to cumulate the results. For detailed information, see "Cambricon CNDrv Developer Guide".
 *
 *  @param[in] handle
 *  Pointer to an MLUOP context that is used to manage MLU devices and
 *  queues. For detailed information, see ::mluOpHandle_t.
 *  @param[in] round_mode
 *  The rounding mode of quantization conversion to be set to the MLUOP handle.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - On MLU200 series:
 *    You cannot set MLUOP_ROUND_HALF_TO_EVEN for the rounding mode because the hardware does not
 *    support it.
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
mluOpSetQuantizeRoundMode(mluOpHandle_t handle, mluOpQuantizeRoundMode_t round_mode);

// Group:QuantizeRoundMode
/*!
 *  @brief Retrieves the rounding mode of a specific MLUOP context.
 *
 *  @param[in] handle
 *  Pointer to an MLUOP context that is used to manage MLU devices and
 *  queues. For detailed information, see ::mluOpHandle_t.
 *
 *  @param[out] round_mode
 *  The rounding mode of quantization conversion that was previously set to the specified handle.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - The default round mode of default initialized ::mluOpHandle_t is MLUOP_ROUND_TO_EVEN.
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
mluOpGetQuantizeRoundMode(mluOpHandle_t handle, mluOpQuantizeRoundMode_t *round_mode);

// Group:Runtime Management
/*!
 *  @brief Updates the specific atomics mode of MLUOP context information that is held by the \b handle. This function
 *  should be called if you want to change the atomics mode that is used to cumulate the results.
 *  For detailed information, see "Cambricon CNDrv Developer Guide".
 *
 *  @param[in] handle
 *  Pointer to an MLUOP context that is used to manage MLU devices and
 *  queues. For detailed information, see ::mluOpHandle_t.
 *
 *  @param[in] atomics_mode
 *  The atomics mode.
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
mluOpSetAtomicsMode(mluOpHandle_t handle, mluOpAtomicsMode_t atomics_mode);

// Group:Runtime Management
/*!
 *  @brief Retrieves the atomics mode of a specific MLUOP context.
 *
 *  @param[in] handle
 *  Pointer to an MLUOP context that is used to manage MLU devices and
 *  queues. For detailed information, see ::mluOpHandle_t.
 *
 *  @param[out] atomics_mode
 *  The atomics mode.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - The default atomics mode of default initialized ::mluOpHandle_t is ::MLUOP_ATOMICS_NOT_ALLOWED.
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
mluOpGetAtomicsMode(mluOpHandle_t handle, mluOpAtomicsMode_t *atomics_mode);

/******************************************************************************
 * MLUOP Data Structure: Descriptor
 * The struct represent node, weight and the AI network layer
 ******************************************************************************/
/*! The descriptor of a tensor that holds the information including tensor
 *  layout, data type, the number of dimensions, shape and strides.
 *
 *  You need to call the ::mluOpCreateTensorDescriptor function to create a descriptor,
 *  and call the ::mluOpSetTensorDescriptor function or the ::mluOpSetTensorDescriptorEx
 *  function to set the tensor information to the descriptor. Also, you need to destroy
 *  the MLUOP context at the end with the ::mluOpDestroyTensorDescriptor function.
 */
typedef struct mluOpTensorStruct *mluOpTensorDescriptor_t;

/*! The descriptor of the matrix multiplication function that holds compute type, bias type,
 *  transpose flag, and other attributes defined in ::mluOpMatMulDescAttribute_t.
 *
 *  You need to call the ::mluOpMatMulDescCreate function to create a descriptor, and call
 *  the ::mluOpSetMatMulDescAttr function to set the information of the matrix multiplication
 *  to the descriptor. Also, you need to destroy the MLUOP context at the end with
 *  the ::mluOpMatMulDescDestroy function.
 */
typedef struct mluOpMatMulStruct *mluOpMatMulDescriptor_t;

/*! The descriptor of a tensor that holds the information including tensor
 *  shape, the number of dimensions, pad, strides, dalition, sub_m, transpose.
 *
 *  You need to call the ::mluOpCreateSparseConvolutionDescriptor function to create a descriptor,
 *  and call the ::mluOpSetSparseConvolutionDescriptor function to set the tensor information to
 *  the descriptor. Also, you need to destroy the MLUOP context at the end with
 *  the ::mluOpDestroySparseConvolutionDescriptor function.
 */
typedef struct mluOpSparseConvolutionStruct *mluOpSparseConvolutionDescriptor_t;

/*! The descriptor of the matrix multiplication that holds the configured matrix multiplication
 *  algorithm descriptor and its runtime properties.
 *
 *  You need to call the ::mluOpCreateMatMulHeuristicResult function to create a descriptor.
 *  Also, you need to destroy the MLUOP context at the end with
 *  the ::mluOpDestroyMatMulHeuristicResult function.
 */
typedef struct mluOpMatMulHeuristicResult *mluOpMatMulHeuristicResult_t;

/*! The descriptor of the matrix multiplication that holds the preferences for
 *  mluOpMatMulHeuristicResult_t configuration.
 */
typedef struct mluOpMatMulPrefer *mluOpMatMulPrefer_t;

/*! The descriptor of the matrix multiplication computation algorithm.
 *
 *  You need to call the ::mluOpMatMulAlgoCreate function to create a descriptor.
 *  Also, you need to destroy the MLUOP context at the end with
 *  the ::mluOpMatMulAlgoDestroy function.
 */
typedef struct mluOpMatMulAlgoStruct *mluOpMatMulAlgo_t;

/*! The descriptor of Reduce function that holds ::mluOpReduceOp_t,
 * ::mluOpDataType_t, ::mluOpNanPropagation_t, ::mluOpReduceIndices_t, and ::mluOpIndicesType_t.
 */
typedef struct mluOpReduceStruct *mluOpReduceDescriptor_t;

/*! The descriptor of the transpose operation that holds transpose information
 *  including \b dimensions and \b permute.
 *
 *  You need to call the ::mluOpCreateTransposeDescriptor function to create a descriptor,
 *  and call the ::mluOpSetTransposeDescriptor function to set the information of
 *  transpose operation to the descriptor. Also, you need to destroy the MLUOP context
 *  at the end with the ::mluOpDestroyTransposeDescriptor function.
 */
typedef struct mluOpTransposeStruct *mluOpTransposeDescriptor_t;

/*! The descriptor of Unique function that holds mluOpUniqueSort_t, dim, return_inverse,
 *  and return_counts.
 *
 *  You need to call the ::mluOpCreateUniqueDescriptor to create a descriptor,
 *  and call the ::mluOpSetUniqueDescriptor to set the information of the unique operation to
 *  the descriptor. At last, you need to destroy the descriptor at the end with the
 *  ::mluOpDestroyUniqueDescriptor function.*/
typedef struct mluOpUniqueStruct *mluOpUniqueDescriptor_t;

/*! The descriptor of CARAFE (Content-Aware ReAssembly of FEatures) operation that holds
 *  CARAFE information including the number of input dimensions, kernel size, group size,
 *  and scale factor.
 *
 *  You need to call the ::mluOpCreateCarafeDescriptor function to create a descriptor,
 *  and call the ::mluOpSetCarafeDescriptor function to set the information of the CARAFE operation
 *  to the descriptor. Also, you need to destroy the MLUOP context at the end with the
 *  ::mluOpDestroyCarafeDescriptor function.
 */
typedef struct mluOpCarafeStruct *mluOpCarafeDescriptor_t;

// Group:Tensor
/*!
 *  @brief Creates a tensor descriptor pointed by \b desc that holds the dimensions, data type,
 *  and layout of input tensor. If the input tensor is in fixed-point data type,
 *  the ::mluOpSetTensorDescriptorPositionAndScale function or the
 *  ::mluOpSetTensorDescriptorPosition
 *  function needs to be called to set quantization parameters.
 *
 *  The ::mluOpDestroyTensorDescriptor function needs to be called to destroy the
 *  tensor descriptor later.
 *
 *  @param[in] desc
 *  Pointer to the struct that holds information about the tensor descriptor.
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
mluOpCreateTensorDescriptor(mluOpTensorDescriptor_t *desc);

// Group:GetIndicePairs
/*!
 *  @brief Creates a tensor descriptor pointed by \b desc that holds the dimensions, pad,
 *  stride, dilation, sub_m, transpose, inverse and layout of input filter and output tensor shape.
 *  The::mluOpSetSparseConvolutionDescriptor function needs to be called to set parameters.
 *
 *  The ::mluOpDestroySparseConvolutionDescriptor function needs to be called to destroy the
 *  tensor descriptor later.
 *
 *  @param[in] desc
 *  Pointer to the struct that holds information about the tensor descriptor.
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
mluOpCreateSparseConvolutionDescriptor(mluOpSparseConvolutionDescriptor_t *desc);

// Group:GetIndicePairs
/*!
 * @brief Destroys a convolution descriptor \b desc that is previously created with the
 * ::mluOpCreateSparseConvolutionDescriptor function.
 *
 * The sparse convolution descriptor is defined in ::mluOpSparseConvolutionDescriptor_t
 * and holds the information about the sparse convolution forward or backward operation.
 *
 *
 * @param[in] desc
 * The sparse convolution descriptor to be destroyed.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_EXECUTION_FAILED
 *
 * @note
 * - This function should be called to destroy the sparse convolution descriptor.
 * Otherwise, the memory leak may occur.
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
mluOpDestroySparseConvolutionDescriptor(mluOpSparseConvolutionDescriptor_t desc);

// Group:Tensor
/*!
 *  @brief Creates a group of tensor descriptor stored by \b group_desc that
 *  holds the dimensions, data_type, and layout of input tensors. If the input
 *  tensor is in fixed-point data type, the
 *  ::mluOpSetTensorDescriptorPositionAndScale function or the
 *  ::mluOpSetTensorDescriptorPosition function need to be called to set
 *  quantization parameters.
 *
 *  @param[in] group_desc
 *  An array of pointers to the structs that hold information about the
 *  tensor descriptor.
 *  @param[in] desc_num
 *  The length of the input array \b group_desc.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @par API Dependency
 *  - The ::mluOpDestroyTensorDescriptor function needs to be called for each
 *    descriptor to destroy all tensors in group_desc or the
 *    ::mluOpDestroyGroupTensorDescriptors needs to be called to destroy the all
 *    tensor descriptors in group_desc later.
 *
 *  @note
 *  - None
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
mluOpCreateGroupTensorDescriptors(mluOpTensorDescriptor_t *group_desc[], const int desc_num);

// Group:Tensor
/*!
 *  @brief Initializes the tensor descriptor pointed by \b desc that is
 *  previously created with the ::mluOpCreateTensorDescriptor function, and sets
 *  the information about the dimensions, data type, and layout of the input
 *  tensor.
 *
 *  If ::mluOpSetTensorDescriptor is called, you do not need to specify the strides of all
 *  dimensions. The strides are inferred by parameters passed to this function. Also, the data
 *  will be treated as contiguous in memory with no padding between dimensions. To specify the
 *  strides of all dimensions, you can call ::mluOpSetTensorDescriptorEx. But the data might not
 *  be treated as contiguous in memory.
 *
 *  @param[in] desc
 *  The descriptor of the input tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[in] layout
 *  The layout of the input tensor. For detailed information, see ::mluOpTensorLayout_t.
 *  @param[in] dtype
 *  The data type of the input tensor. For detailed information, see ::mluOpDataType_t.
 *  @param[in] dimNb
 *  The number of dimensions in the input tensor of the initialized operation.
 *  @param[in] dimSize
 *  An array that contains the size of the tensor for each dimension.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - dimSize[0] represents the highest dimension, dimSize[DIM_MAX - 1] represents
 *    the lowest dimension, and DIM_MAX represents the number of dimensions in the input tensor.
 *  - This function cannot be called continuously. You need to call ::mluOpResetTensorDescriptor
 *    before calling another ::mluOpSetTensorDescriptor to avoid memory leaks.
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
mluOpSetTensorDescriptor(
    mluOpTensorDescriptor_t desc, mluOpTensorLayout_t layout, mluOpDataType_t dtype, int dimNb, const int dimSize[]);

// Group:GetIndicePairs
/*!
 * @brief Initializes the sparse convolution descriptor \b desc that is previously created
 * with the ::mluOpCreateSparseConvolutionDescriptor function, and sets the information
 * about the convolution forward and backward operation to the convolution descriptor
 * \b desc. The information includes the number of the convolution dimensions \b dimNb,
 * the padding size for each dimension \b pad, the stride of the sliding window for
 * each dimension \b stride, the dilation
 * factor for each dimension \b dilation, and the size of \b input , \b filter , \b output.
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
 * An value that determine the algorithms for sparse convolution. If \b sub_m is set to 0, the
 * algorithms will be the default sparse convolution. If \b sub_m is set to 0, the algorithms will be the
 * submanifold sparse convolution.
 * @param[in] transpose
 * An value that determines transpose.
 * @param[in] inverse
 * An value that determines inverse.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_EXECUTION_FAILED
 *   ::MLUOP_STATUS_NOT_INITIALIZED
 *
 * @note
 * - Currently, only 5D input tensors are supported for convolution
 * forward or backward operation.
 *
 * @par Requirements
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
                                    const int subm,
                                    const int transpose,
                                    const int inverse);

// Group:: GetIndicePairs
/*!
 *  @brief Obtains the parameter num_act_out from ::mluOpSparseConvolutionStruct.
 *
 *  @param[in] desc
 *  Pointer to the parameter num_act_out that holds information about the tensor descriptor.
 *  @param[out] num_act_out
 *  The active point number of output space in sparse convolution mode.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_NOT_INITIALIZED
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
mluOpGetSparseConvolutionNumActOut(mluOpSparseConvolutionDescriptor_t desc,
                                   int *num_act_out);

// Group:Tensor
/*!
 *  @brief Initializes the group of tensor descriptors stored by \b group_desc
 *  that is previously created with the ::mluOpCreateTensorDescriptor function or
 *  ::mluOpCreateGroupTensorDescriptors function, and sets the information about
 *  the dimensions, data type, and layout of all the input tensors.
 *
 *  If ::mluOpSetTensorDescriptor or ::mluOpSetGroupTensorDescriptors is called,
 *  you do not need to specify the strides of all dimensions. The strides are
 *  inferred by parameters passed to this function. Also, the data will be
 *  treated as contiguous in memory with no padding between dimensions. To
 *  specify the strides of all dimensions, you can call
 *  ::mluOpSetTensorDescriptorEx. But the data might not be treated as contiguous
 *  in memory.
 *
 *  @param[in] group_desc
 *  An array of pointers to the struct that hold information about the
 *  tensor descriptor.
 *  @param[in] group_layout
 *  An array that stores the layouts of all input tensors. For detailed
 *  information, see ::mluOpTensorLayout_t.
 *  @param[in] group_dtype
 *  An array that stores the data types of all input tensors. For
 *  detailed information, see ::mluOpDataType_t.
 *  @param[in] group_dimNb
 *  An array that stores the dimensions of all input tensors.
 *  @param[in] group_dimSize
 *  An array that stores the size of each dimension of all tensors.
 *  @param[in] desc_num
 *  The length of the input array \b group_desc.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - The group_dimSize includes dimensions of all tensors. You need to store
 *    the dimension of each tensor one by one in order. For example, If we have
 *    three tensors, the first tensor dimension is [3,4,5,6], the second tensor
 *    dimension is [9,7,8], and the third tensor dimension is [4,7], the
 *    group_dimSize should be [3,4,5,6,9,7,8,4,7].
 *  - For better performance, there is no overflow check in this function.
 *    Please make sure that the size of each tensor is in the range of [0, 2^31].
 *    Otherwise, you will get wrong result.
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
mluOpSetGroupTensorDescriptors(mluOpTensorDescriptor_t *group_desc[],
                               const mluOpTensorLayout_t group_layout[],
                               const mluOpDataType_t group_dtype[],
                               const int group_dimNb[],
                               const int group_dimSize[],
                               const int desc_num);

// Group:Tensor
/*!
 *  @brief Resets the tensor descriptor pointed by \b desc that is previously
 *  created with the ::mluOpCreateTensorDescriptor function.
 *  If ::mluOpResetTensorDescriptor is called, all the information about the tensor will be reset to
 *  initial value, which means layout is MLUOP_LAYOUT_ARRAY, dtype is MLUOP_DTYPE_FLOAT, dimsNb is
 *  0, and dimSize points to an \b MLUOP_DIM_MAX-dimension array.
 *
 *  @param[in] desc
 *  The descriptor of the tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - This function is used to avoid memory leaks when more than one ::mluOpSetTensorDescriptor
 *    function is called. You should call this function before calling another
 *    ::mluOpSetTensorDescriptor.
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
mluOpResetTensorDescriptor(mluOpTensorDescriptor_t desc);

// Group:Tensor
/*!
 *  @brief Initializes the tensor descriptor pointed by \b desc that is previously created
 *  with the ::mluOpCreateTensorDescriptor function, and sets the information about
 *  the dimensions, strides, data type, and layout of the input tensor.
 *
 *  Compare with ::mluOpSetTensorDescriptor, you can specify the strides of all dimensions with
 *  this function. If ::mluOpSetTensorDescriptor is called, you do not need to specify the
 *  strides of all dimensions and the strides are inferred by parameters passed to this function.
 *
 *  This function does not support all the operations in this version. You can check
 *  if an operation supports this function in the "note" section of the operation description.
 *
 *  @param[in] desc
 *  The descriptor of the input tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[in] layout
 *  The layout of the input tensor. For detailed information, see ::mluOpTensorLayout_t.
 *  @param[in] dtype
 *  The data type of the input tensor. For detailed information, see ::mluOpDataType_t.
 *  @param[in] dimNb
 *  The number of dimensions in the input tensor of the initialized operation.
 *  @param[in] dimSize
 *  An array that contains the size of the tensor for each dimension.
 *  @param[in] dimStride
 *  An array that contains the stride of the tensor for each dimension.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - dimSize[0] represents the highest dimension, and dimSize[DIM_MAX - 1] represents
 *    the lowest dimension.
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
mluOpSetTensorDescriptorEx(mluOpTensorDescriptor_t desc,
                           mluOpTensorLayout_t layout,
                           mluOpDataType_t dtype,
                           int dimNb,
                           const int dimSize[],
                           const int dimStride[]);

// Group:Tensor
/*!
 *  @brief Sets the \b dimNb and \b dimSize factors to the input tensor descriptor.
 *
 *  If ::mluOpSetTensorDescriptorDim is called, you do not need to specify the strides of all
 *  dimensions. The strides are inferred by parameters passed to this function. Also, the data
 *  will be treated as contiguous in memory with no padding between dimensions. To specify the
 *  strides of all dimensions, you can call ::mluOpSetTensorDescriptorEx. But the data might not
 *  be treated as contiguous in memory.
 *
 *  @param[in] desc
 *  The descriptor of the input tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[in] dimNb
 *  The number of dimensions in the input tensor of the initialized operation.
 *  @param[in] dimSize
 *  An array that contains the size of the tensor for each dimension.
 *
 *  @par Return
 *   - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *   - dimSize[0] represents the highest dimension, dimSize[DIM_MAX - 1] represents
 *    the lowest dimension, and DIM_MAX represents the number of dimensions in the input tensor.
 *
 *  @par Requirements
 *   - None.
 *
 *  @par Example
 *   - None.
 */
mluOpStatus_t
mluOpSetTensorDescriptorDim(mluOpTensorDescriptor_t desc, int dimNb, const int *dimSize);

// Group:Tensor
/*!
 *  @brief Sets the on-chip data type to the descriptor of a tensor \b desc.
 *  The on-chip data type \b onchip_dtype can be different from the off-chip data type of the
 *  tensor.
 *  This function is optional. If the on-chip data type is not set with this function, the
 *  ::MLUOP_STATUS_BAD_PARAM data type is used by default.
 *
 *  @param[in] desc
 *  The descriptor of input tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[in] onchip_dtype
 *  The on-chip data type of the tensor used in the operations that support fixed-point
 *  computing.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - The on-chip data type is only used on the operations that support fixed-point computing. It
 *    has no effect on other operations. If you call this function to get on-chip data type for an
 *    operation that does not support fixed-point computing, ::MLUOP_STATUS_BAD_PARAM is returned.
 *    To check if an operation supports fixed-point computing, see the detailed description of the
 *    operation.
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
mluOpSetTensorDescriptorOnchipDataType(mluOpTensorDescriptor_t desc, mluOpDataType_t onchip_dtype);

// Group:Tensor
/*!
 *  @brief Sets the \b position factor to the descriptor \b desc of fixed-point data in
 *  fixed-point quantization. It is used in ::MLUOP_QUANTIZE_POSITION mode.
 *
 *  @param[in] desc
 *  The descriptor of the tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[in] position
 *  A scalar of fixed position factor that is used for quantization.
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
mluOpSetTensorDescriptorPosition(mluOpTensorDescriptor_t desc, int position);

// Group:Tensor
/*!
 *  @brief Sets the \b position and \b scale factors to the descriptor of fixed-point data in
 *  fixed-point quantization. It is used in ::MLUOP_QUANTIZE_POSITION_SCALE mode.
 *
 *  @param[in] desc
 *  The descriptor of the tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[in] position
 *  A scalar of fixed position factor that is used for quantization.
 *  @param[in] scale
 *  A scalar of scale factor that is used for quantization.
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
mluOpSetTensorDescriptorPositionAndScale(mluOpTensorDescriptor_t desc, int position, float scale);
// Group:Tensor
/*!
 *  @brief Sets the \b position, \b scale and \b offset factors to the descriptor of fixed-point
 *  data in fixed-point quantization. It is used in ::MLUOP_QUANTIZE_POSITION_SCALE_OFFSET mode.
 *
 *  @param[in] desc
 *  The descriptor of the tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[in] position
 *  A scalar of fixed position factor that is used for quantization.
 *  @param[in] scale
 *  A scalar of scale factor that is used for quantization.
 *  @param[in] offset
 *  A scalar of offset factor that is used for quantization.
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
mluOpSetTensorDescriptorPositionScaleAndOffset(mluOpTensorDescriptor_t desc, int position, float scale, int offset);

// Group:Tensor
/*!
 *  @brief Retrieves a tensor descriptor \b desc that is previously created with the
 *  ::mluOpCreateTensorDescriptor function, and sets the information about the dimensions,
 *  data type, and layout of input tensor.
 *
 *  @param[in] desc
 *  The descriptor of the input tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[out] layout
 *  Pointer to the host memory that holds information about the layout of the input
 *  tensor.
 *  For detailed information, see ::mluOpTensorLayout_t.
 *  @param[out] dtype
 *  Pointer to the host memory that holds information about the data type of the input
 *  tensor.
 *  For detailed information, see ::mluOpDataType_t.
 *  @param[out] dimNb
 *  Pointer to the host memory that holds information about the dimension of input tensor.
 *  @param[out] dimSize
 *  An array that contains the size of the tensor for each dimension.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - dimSize[0] represents the highest dimension, and dimSize[DIM_MAX - 1] represents the lowest
 *    dimension.
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
mluOpGetTensorDescriptor(
    const mluOpTensorDescriptor_t desc, mluOpTensorLayout_t *layout, mluOpDataType_t *dtype, int *dimNb, int dimSize[]);

// Group:Tensor
/*!
 *  @brief Retrieves a tensor descriptor \b desc that is previously created with the
 *  ::mluOpCreateTensorDescriptor and sets the information about the dimensions, data type,
 *  stride and layout of input tensor with ::mluOpSetTensorDescriptorEx.
 *
 *  @param[in] desc
 *  The descriptor of the input tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[out] layout
 *  Pointer to the host memory that holds information about the layout of the input
 *  tensor.
 *  For detailed information, see ::mluOpTensorLayout_t.
 *  @param[out] dtype
 *  Pointer to the host memory that holds information about the data type of the input
 *  tensor.
 *  For detailed information, see ::mluOpDataType_t.
 *  @param[out] dimNb
 *  Pointer to the host memory that holds information about the dimension of input tensor.
 *  @param[out] dimSize
 *  An array that contains the size of the tensor for each dimension.
 *  @param[out] dimStride
 *  An array that contains the stride of the tensor for each dimension.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - dimSize[0] represents the highest dimension, and dimSize[DIM_MAX - 1] represents the lowest
 *    dimension.
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
mluOpGetTensorDescriptorEx(const mluOpTensorDescriptor_t desc,
                           mluOpTensorLayout_t *layout,
                           mluOpDataType_t *dtype,
                           int *dimNb,
                           int dimSize[],
                           int dimStride[]);

// Group:Tensor
/*!
 *  @brief Retrieves the number of elements according to the input descriptor \b desc. You
 *  need to call the ::mluOpSetTensorDescriptor function first to create a tensor descriptor
 *  before calling this function.
 *
 *  @param[in] desc
 *  The descriptor of input tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @return
 *  - ::MLUOP_STATUS_SUCCESS
 *
 *  @note
 *  - None.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
     @verbatim
      mluOpTensorDescriptor_t input_desc;
      mluOpCreateTensorDescriptor(&input_desc);
      mluOpSetTensorDescriptor(input_desc, MLUOP_LAYOUT_ARRAY,MLUOP_DTYPE_FLOAT, 2,{2, 3});
      size_t nums=mluOpGetTensorElementNum(input_desc);  // nums = 6
      input one array by 2 * 3
      input: [[1,2,3],[4,5,6]]
      output: 6
     @endverbatim
 */
size_t MLUOP_WIN_API
mluOpGetTensorElementNum(const mluOpTensorDescriptor_t desc);

// Group:Tensor
/*!
 *  @brief Retrieves the on-chip data type of a tensor descriptor \b desc set by
 *  ::mluOpSetTensorDescriptorOnchipDataType.
 *  If the on-chip data type is not set with the ::mluOpSetTensorDescriptorOnchipDataType function,
 *  the ::MLUOP_STATUS_BAD_PARAM is returned.
 *
 *  @param[in] desc
 *  The descriptor of input tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[in] onchip_dtype
 *  Pointer to the MLU memory that holds information about the on-chip data type of the
 *  tensor.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - The on-chip data type is only used on the operations that support fixed-point computing. It
 *    has no effect on other operations. If you call this function to get on-chip data type for an
 *    operation that does support fixed-point computing, ::MLUOP_STATUS_BAD_PARAM is returned. To
 *    check if an operation supports fixed-point computing, see the detailed description of the
 *    operation.
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
mluOpGetTensorDescriptorOnchipDataType(const mluOpTensorDescriptor_t desc, mluOpDataType_t *onchip_dtype);

// Group:Tensor
/*!
 *  @brief Gets the \b position factor to the descriptor \b desc of fixed-point data in
 *  fixed-point quantization.
 *
 *  @param[in] desc
 *  The descriptor of the tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[out] position
 *  A host pointer of fixed position factor that is used for quantization.
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
mluOpGetTensorDescriptorPosition(const mluOpTensorDescriptor_t desc, int *position);

// Group:Tensor
/*!
 *  @brief Gets the position and scale factors of a tensor descriptor \b desc used in
 *  fixed-point quantization.
 *
 *  @param[in] desc
 *  The descriptor of the input tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[out] position
 *  Pointer to the MLU memory that holds information about fixed position
 *  used for quantization.
 *  @param[out] scale
 *  Pointer to the MLU memory that holds information about scale factor
 *  used for quantization.
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
mluOpGetTensorDescriptorPositionAndScale(const mluOpTensorDescriptor_t desc, int *position, float *scale);
// Group:Tensor
/*!
 *  @brief Gets the \b position, \b scale and \b offset factors to the descriptor \b desc of
 *  fixed-point data in fixed-point quantization.
 *
 *  @param[in] desc
 *  The descriptor of the tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[out] position
 *  A host pointer of fixed position factor that is used for quantization.
 *  @param[out] scale
 *  A host pointer of scale factor that is used for quantization.
 *  @param[in] offset
 *  A host pointer of offset factor that is used for quantization.
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
mluOpGetTensorDescriptorPositionScaleAndOffset(const mluOpTensorDescriptor_t desc,
                                               int *position,
                                               float *scale,
                                               int *offset);

// Group:Tensor
/*!
 *  @brief Destroys a tensor descriptor that was created by
 *  ::mluOpCreateTensorDescriptor.
 *
 *  @param[in] desc
 *  A tensor descriptor created by ::mluOpCreateTensorDescriptor.
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
mluOpDestroyTensorDescriptor(mluOpTensorDescriptor_t desc);

// Group:Tensor
/*!
 *  @brief Destroys a group of tensor descriptors that was created by
 *  ::mluOpCreateTensorDescriptor or ::mluOpCreateGroupTensorDescriptors.
 *
 *  @param[in] group_desc
 *  An array of pointers to the struct that holds information about the
 *  tensor descriptor.
 *  @param[in] desc_num
 *  The length of the input array \b group_desc.
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
mluOpDestroyGroupTensorDescriptors(mluOpTensorDescriptor_t *group_desc[], const int desc_num);

// Group:TensorSet
/*!
 *  @brief Creates a descriptor \b tensorSetDesc of tensor set that holds a
 *  series of tensors. The number of tensors of tensor set is jointly determined
 *  by \b setDimNb and \b setDimSize. Use ::mluOpInitTensorSetMemberDescriptor to
 *  set information for descriptor and ::mluOpDestroySeqDataDescriptor function
 *  to destroy the tensor set descriptor.
 *
 *  @param[out] tensorSetDesc
 *  Pointer to the memory that holds information about the descriptor
 *  of tensor set.
 *  @param[in] setDimNb
 *  The number of dimensions of the tensor set.
 *  @param[in] setDimSize
 *  An array that contains the number of the tensors for each dimension
 *  of the tensor set.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - None.
 *
 *  @par Requirements
 *  - After calling this function, you can call the
 *    ::mluOpInitTensorSetMemberDescriptor function to initialize and set the
 *    information to the tensor set descriptor.
 *  - You need to call the ::mluOpDestroyTensorSetDescriptor function to destroy
 *    the descriptor.
 *
 *  @par Example
 *  - None.
 *
 *  @par Reference
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateTensorSetDescriptor(mluOpTensorSetDescriptor_t *tensorSet, const int setDimNb, const int setDimSize[]);

// Group:TensorSet
/*!
 *  @brief Retrieves a tensor set descriptor \b tensorSetDesc that is previously
 *  created with the ::mluOpCreateTensorSetDescriptor function.
 *
 *  @param[in] tensorSetDesc
 *  The descriptor of the tensor set.
 *  @param[out] setDimNb
 *  The number of dimensions of the tensor set.
 *  @param[out] setDimSize
 *  An array that contains the number of the tensor for each dimension
 *  of the tensor set.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @par API Dependency
 *  - Before calling this function, ::mluOpCreateTensorSetDescriptor should be
 *    called.
 *
 *  @note
 *  - setDimSize[0] represents the highest dimension, and dimSize[dimNb - 1]
 *    represents the lowest dimension.
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
mluOpGetTensorSetDescriptor(mluOpTensorSetDescriptor_t tensorSetDesc, int *setdimNb, int setDimSize[]);

// Group:TensorSet
/*!
 *  @brief Destroys a tensor set descriptor \b tensorSetDesc that is previously
 *  created by ::mluOpCreateTensorSetDescriptor.
 *
 *  @param[in] desc
 *  A tensor descriptor created by ::mluOpCreateTensorSetDescriptor.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - None.
 *
 *  @par Requirements
 *  - This function should be called to destroy the tensor set descriptor.
 *    Otherwise, the memory leak may occur.
 *
 *  @par Example
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDestroyTensorSetDescriptor(mluOpTensorSetDescriptor_t tensorSetDesc);

// Group:TensorSet
/*!
 *  @brief Initializes a member tensor in the tensor set descriptors pointed by
 *  \b desc that is previously created with the ::mluOpCreateTensorSetDescriptor
 *  function, and sets the information about the dimensions, data type, and
 *  layout.
 *
 *  @param[in] tensorSetDesc
 *  The descriptor of the tensor set. For detailed information,
 *  see ::mluOpTensorSetDescriptor_t.
 *  @param[in] setDimNb
 *  The number of dimensions of the tensor set.
 *  @param[in] tensorIndex
 *  An array that contains the index of each dimension of a member
 *  tensor to be initialized in the tensor set.
 *  @param[in] layout
 *  The layout of the member tensor. For detailed information, see
 *  ::mluOpTensorLayout_t.
 *  @param[in] dtype
 *  The data type of the member tensor. For detailed information, see
 *  ::mluOpDataType_t.
 *  @param[in] dimNb
 *  The number of dimensions in the member tensor.
 *  @param[in] dimSize
 *  An array that contains the size of the member tensor for each
 *  dimension.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - Before calling this function,
 *    You need to call the ::mluOpCreateTensorSetDescriptor functions to create
 *    the tensor descriptors \b tensorSetDesc.
 *  - All member tensors in the tensor set need to call this function to
 *    initialize related properties.
 *  - dimSize[0] and dimSize[DIM_MAX - 1] represent the highest and lowest
 *    dimension respectively, where DIM_MAX is the number of dimensions in the
 *    input tensor.
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
mluOpInitTensorSetMemberDescriptor(mluOpTensorSetDescriptor_t tensorSetDesc,
                                   const int setDimNb,
                                   const int tensorIndex[],
                                   mluOpTensorLayout_t layout,
                                   mluOpDataType_t dtype,
                                   const int dimNb,
                                   const int dimSize[]);

// Group:TensorSet
/*!
 *  @brief Sets the position and scale factors used in fixed-point quantization.
 *  It is only used if you have quantized the input data with the symmetric
 *  fixed-point quantization with scale factor quantization method.
 *
 *  @param[in] tensorSetDesc
 *  The descriptor of tensor set. For detailed information,
 *  see ::mluOpTensorSetDescriptor_t.
 *  @param[in] setDimNb
 *  The number of dimensions of the tensor set.
 *  @param[in] tensorIndex
 *  An array that contains the position index information of the member
 *  tensor in the tensor set.
 *  @param[in] position
 *  A position of fixed position factor that is used for
 *  quantification.
 *  @param[in] scale
 *  A scalar of scale factor that is used for quantification.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - If the member tensor is in floating-point data type, and  you need to call
 *    this function.
 *  - If the member tensor is in fixed-point data type, and  you need to call
 *    this function.
 *  - Before calling this function,
 *    You need to call the ::mluOpCreateTensorSetDescriptor functions to create
 *    the tensor descriptors \b tensorSetDesc.
 *  - The \b position should be limited in [-128, 127], otherwise the result is
 *    undefined.
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
mluOpInitTensorSetMemberDescriptorPositionAndScale(mluOpTensorSetDescriptor_t tensorSetDesc,
                                                   const int setDimNb,
                                                   const int tensorIndex[],
                                                   const int position,
                                                   const float scale);

// Group:TensorSet
/*!
 *  @brief Retrieves the size of tensor set according to the input descriptor \b
 *  tensorSetDesc. You need to call the ::mluOpInitTensorSetMemberDescriptor
 *  function first to create a tensor set descriptor before calling this
 *  function.
 *
 *  @param[in] desc
 *  The descriptor of tensor set. For detailed information,
 *  see ::mluOpTensorSetDescriptor_t.
 *  @param[Out] sizeInBytes
 *  Size in bytes of tensor set.
 *  You can allocate MLU memory for the tensor set with this value.
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
 *  @par Reference
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorSetDescriptorSize(mluOpTensorSetDescriptor_t tensorSetDesc, int *sizeInBytes);

// Group:TensorSet
/*!
 *  @brief Retrieves the tensor descriptor in the tensor set and the
 *  corresponding offset address based on the entire block of MLU memory through
 *  the index \b tensorIndex.
 *
 *  @param[in] tensorSetDesc
 *  The descriptor of tensor set. For detailed information,
 *  see ::mluOpTensorSetDescriptor_t.
 *  @param[in] tensorIndex
 *  An array that contains the position information of the member
 *  tensor in the tensor set.
 *  @param[in] data
 *  Pointer to the MLU memory that is described by \b tensorSetDesc.
 *  @param[out] tensorDesc
 *  Pointer to the host member. It is member tensor descriptor that
 *  indexed by \b tensorIndex in the tensor set. \b *tensorDesc contains tensor
 *  member information about dimensions, layout, data type, position and scale.
 *  @param[out] dataAddrInDevice
 *  Pointer to the MLU memory that indexed by \b tensorIndex in the
 *  whole block of data \b dataAddrInDevice.
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
 *  @par Reference
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorAndDataFromTensorSet(mluOpTensorSetDescriptor_t tensorSetDesc,
                                   const int setDimNb,
                                   const int tensorIndex[],
                                   void *data,
                                   mluOpTensorDescriptor_t *tensorDesc,
                                   void **dataAddrInDevice);

// Group:Abs
/*!
 * @brief Computes the absolute value for every element of the input tensor \b x
 * and returns results in \b y.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the abs operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] x_desc
 * The descriptor of the input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] y_desc
 * The descriptor of the output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] y
 * Pointer to the MLU memory that stores the output tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - Date types of input tensor and output tensor should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - output tensor: half, float
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
mluOpStatus_t MLUOP_WIN_API
mluOpAbs(mluOpHandle_t handle,
         const mluOpTensorDescriptor_t x_desc,
         const void *x,
         const mluOpTensorDescriptor_t y_desc,
         void *y);

// Group:AddN
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra
 * workspace to optimize the ::mluOpAddN_v2 operation.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the ::mluOpAddN
 * operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] input_descs[]
 * Array of descriptors for all input tensors. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] input_num
 * Number of tensors in array inputs[].
 * @param[in] output_desc
 * The descriptor of the output tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] workspace_size
 * Host pointer to the returned size of the extra workspace in bytes that is
 * used in ::mluOpAddN operation.
 *
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
 *
 *  @par Reference
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetAddNWorkspaceSize(mluOpHandle_t handle,
                          const mluOpTensorDescriptor_t input_descs[],
                          const uint32_t input_num,
                          const mluOpTensorDescriptor_t output_desc,
                          size_t *workspace_size);
// Group:AddN
/*!
 * @brief Computes the sum of input tensors.
 *
 * AddN operation is wildly used in artificial intelligence as a kind of basic mathematical
 * operations. Also, this operation is supported in almost all common frameworks, like
 * PyTorch and TensorFlow.
 * Compared with ::mluOpAddN, this function supports multidirectional broadcasting of input tensors.
 *
 * This function may need extra MLU memory as the workspace to support multidirectional broadcasting.
 * You can get the size of the workspace \b workspace_size with the ::mluOpGetAddNWorkspaceSize
 * function.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in ::mluOpAddN
 * operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] input_descs[]
 * Array of descriptors for all input tensors. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] inputs[]
 * Array of device pointers to the MLU memory for the input tensors.
 * @param[in] input_num
 * Number of tensors in array inputs[].
 * @param[in] output_desc
 * The descriptor of the output tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Device pointer to the MLU memory that stores the output tensor.
 * @param[in] workspace
 * Device pointer to the MLU memory that is used as an extra workspace for this
 * operation.
 * @param[in] workspace_size
 * Size of the extra workspace in bytes that needs to be used in this
 * operation. You can get the size of workspace with the ::mluOpGetAddNWorkspaceSize
 * function.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - Before calling this function to perform ::mluOpAddN operation, you need to get
 *   the size of workspace by the ::mluOpGetAddNWorkspaceSize function.
 *
 * @par Data Type
 * - This function supports the following data types for input and output tensors.
 * Note that the data types of output should be the same with that of input.
 *   - input tensor: float, half, int32, int16, int8, uint8
 *   - output tensor: float, half, int32, int16, int8, uint8
 *
 * @par Scale Limitation
 * - The maximum dimension of both input and output tensors is 8.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of this operation is as follows:
 *   @verbatim
 *     Input tensor  1 :   [[1, 2, 3]]
 *     Input tensor  2 :   [[1],
 *                          [4],
 *                          [7]]
 *     Input tensor  3 :   [[1, 2, 3],
 *                          [4, 5, 6],
 *                          [7, 8, 9]]
 *     Input num       :   3
 *     Output tensor   :   [[3,  5,  7],
 *                          [9, 11, 13],
 *                          [15,17, 19]]
 *   @endverbatim
 *
 *  @par Reference
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpAddN_v2(mluOpHandle_t handle,
             const mluOpTensorDescriptor_t input_descs[],
             const void *const inputs[],
             const uint32_t input_num,
             const mluOpTensorDescriptor_t output_desc,
             void *output,
             void *workspace,
             size_t workspace_size);
// Group:AddN
/*!
 * @brief Computes the sum of input tensors.
 *
 * @par Deprecated
 * - ::mluOpAddN is deprecated and will be removed in the future release. It is recommended to use
 *   ::mluOpAddN_v2 instead.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in this
 * operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] input_descs[]
 * Array of descriptor for all the input tensors. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] inputs[]
 * Array of device pointers to the MLU memory for the all input tensors.
 * @param[in] input_num
 * Number of tensors in array inputs[].
 * @param[in] output_desc
 * The descriptor of the output tensor, For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Device pointer to the MLU memory that stores the output tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - This function supports the following data types for input and output tensors.
 *   - input tensor: float, half
 *   - output tensor: float, half
 *   <b>Note that the data type of output should be same with inputs.</b>
 *
 * @par Data Layout
 * - Data layouts of all input tensors and output tensor must be the same.
 *
 * @par Scale Limitation
 * - The dimensions of input tensors and output tensor must be the same.
 * - The shape of input tensors and output tensor must be the same.
 * - The number of input tensors must be greater than or equal to one.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of this operation is as follows:
 *   @verbatim
 *     Input tensor  1 :   [[1, 2, 3],
 *                          [4, 5, 6],
 *                          [7, 8, 9]]
 *     Input tensor  2 :   [[1, 2, 3],
 *                          [4, 5, 6],
 *                          [7, 8, 9]]
 *     Input tensor  3 :   [[1, 2, 3],
 *                          [4, 5, 6],
 *                          [7, 8, 9]]
 *     Input num       :   3
 *     Output tensor   :   [[3,  6,  9],
 *                          [12, 15, 18],
 *                          [21, 24, 27]]
 *   @endverbatim
 *
 *  @par Reference
 *  - None.
 */

mluOpStatus_t MLUOP_WIN_API
mluOpAddN(mluOpHandle_t handle,
          const mluOpTensorDescriptor_t input_descs[],
          const void *const inputs[],
          uint32_t input_num,
          const mluOpTensorDescriptor_t output_desc,
          void *output);

// Group:Log
/*!
 * @brief Computes logarithm of input tensor \b x, and returns the results in
 * the output tensor \b y.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the log operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] prefer
 * The \b prefer modes defined in ::mluOpComputationPreference_t.
 * @param[in] base
 * An mluOpLogBase_t type value indicating the base (e, 2 or 10) to
 * be used.
 * @param[in] x_desc
 * The descriptor of the input tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor \b x.
 * @param[in] y_desc
 * The descriptor of the output tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] y
 * Pointer to the MLU memory that stores the output tensor \b y.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - Data type of input tensor and output tensor should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - output tensor: half, float
 *
 * @par Scale Limitation
 * - The input tensor and output tensor have the same shape, and the input
 *   tensor must meet the following input data ranges:
 *   - float: [1e-20, 2e5]
 *   - half: [1, 60000]
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
mluOpLog(mluOpHandle_t handle,
         const mluOpComputationPreference_t prefer,
         const mluOpLogBase_t base,
         const mluOpTensorDescriptor_t x_desc,
         const void *x,
         const mluOpTensorDescriptor_t y_desc,
         void *y);

// Group:Carafe
/*!
 * @brief Creates a descriptor pointed by \b carafe_desc for CARAFE upsampling forward and backward operations,
 * and allocates memory holding the configuration parameters.The information is defined in ::mluOpCarafeDescriptor_t.
 * For more information about descriptor, see "Cambricon BANGC OPS User Guide".
 *
 * @param[in] carafe_desc
 * A host pointer to the CARAFE descriptor that holds information about the
 * CARAFE operation.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_NOT_INITIALIZED
 *
 * @par API Dependency
 * - After calling this function, you can call the ::mluOpSetCarafeDescriptor function to initialize
 *   and set the information to the CARAFE descriptor.
 * - You need to call the ::mluOpDestroyCarafeDescriptor function to destroy the descriptor.
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
 *  @par Reference
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateCarafeDescriptor(mluOpCarafeDescriptor_t *carafe_desc);

// Group:Carafe
/*!
 * @brief Initializes the CARAFE descriptor \b carafe_desc that was previously created with
 * the ::mluOpCreateCarafeDescriptor function, and sets the information about the
 * CARAFE forward and backward operations to the descriptor \b carafe_desc.
 *
 * @param[in] carafe_desc
 * The descriptor of the CARAFE operation. For detailed information,
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
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - Before calling this function, ::mluOpCreateCarafeDescriptor should be called.
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
 *  @par Reference
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetCarafeDescriptor(mluOpCarafeDescriptor_t carafe_desc,
                         const int dimNb,
                         const int kernel_size,
                         const int group_size,
                         const int scale_factor);

// Group:Carafe
/*!
 * @brief Destroys a CARAFE descriptor \b carafe_desc that was previously created by
 * the ::mluOpCreateCarafeDescriptor function.
 *
 * The CARAFE descriptor is defined in ::mluOpCarafeDescriptor_t
 * and holds the information about the CARAFE forward or backward operations.
 *
 * @param[in] carafe_desc
 * The CARAFE descriptor to be destroyed. For detailed information,
 * see ::mluOpCarafeDescriptor_t.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @note
 * - You need to call this function after calling the ::mluOpCarafeForward,
 *   or ::mluOpCarafeBackward function. Otherwise, \p MLUOP_STATUS_BAD_PARAM is returned.
 * - This function should be called to destroy the CARAFE descriptor.
 *   Otherwise, memory leak may occur.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 *  @par Reference
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDestroyCarafeDescriptor(mluOpCarafeDescriptor_t carafe_desc);

// Group:Carafe
/*!
 * @brief Performs the CARAFE upsampling operation
 * on the input feature maps \b input using weighted combination, where the
 * filter is set with \b mask. The upsampled feature maps are returned in
 * the output tensor \b output.
 *
 * CARAFE performs upsampling at each output location by weighted summation in a nearby mask
 * window around the corresponding input location. The mask filters are defined separately
 * for each output location, which offers the ability of content-aware handling.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the carafe forward operation. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in] carafe_desc
 * The descriptor of the CARAFE operation. For detailed information,
 * see ::mluOpCarafeDescriptor_t.
 * @param[in] input_desc
 * The tensor descriptor of the input feature maps.
 * For detailed information, see ::mluOpTensorDescriptor_t.
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
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Formula
 * - See "CARAFE operation" section in "Cambricon BANGC OPS User Guide" for details
 *
 * @par Data Type
 * - Data types of \b input, \b mask and \b output tensors must be the same.
 * - The supported data types of input， output, and mask tensors are as follows:
 *   - input tensor: half, float
 *   - mask tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - Data layout of the \b input, \b mask, and \b output tensors should be \p MLUOP_LAYOUT_NHWC.
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
 * @par Performance Optimization
 * - None.
 *
 * @note
 * - If any dimension in \b input_desc, \b mask_desc, or \b output_desc is zero,
 *   which represents an empty tensor, ::MLUOP_STATUS_SUCCESS is returned without
 *   any changes to the \b output tensor.
 *
 * @par Requirements
 * - None.
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

// Group:Carafe
/*!
 * @brief Performs the back-propagation of CARAFE.
 * operation to compute the gradient with respect to input \b grad_input and
 * mask \b grad_mask based on the gradient of response \b grad_output.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the CARAFE backward operation. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in] carafe_desc
 * The descriptor of the CARAFE operation. For detailed information,
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
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Formula
 * - See "CARAFE operation" section in "Cambricon BANGC OPS User Guide" for details.
 *
 * @par Data Type
 * - Data types of \b input, \b mask, \b grad_output, \b grad_input and \b grad_mask
 *   tensors must be the same.
 * - For MLU200 series, it is not recommended to use half data type for tensors due to the
 *   low precision.
 * - The supported data types of input， mask and output tensors are as follows:
 *   - input tensor: half, float
 *   - mask tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - Data layout of the \b input, \b mask, \b grad_output, \b grad_input and \b grad_mask
 *   tensors should be \p MLUOP_LAYOUT_NHWC.
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
 * @par Performance Optimization
 * - None.
 *
 * @note
 * - If any dimension in \b input_desc, \b mask_desc, \b grad_output_desc, \b grad_input_desc
 *   or \b grad_mask_desc is zero, which represents an empty tensor, ::MLUOP_STATUS_SUCCESS is
 *   returned without any changes to the \b grad_output and \b grad_input tensors.
 *
 * @par Requirements
 * - None.
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
// Group:Div
/*!
 * @brief Computes division on input tensors \b x and \b y, and returns the
 * results in the output tensor \b output.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the division operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] prefer
 * The \b prefer modes defined in ::mluOpComputationPreference_t.
 * @param[in] x_desc
 * The descriptor of the input tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the dividend tensor.
 * @param[in] y_desc
 * The descriptor of the input tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] y
 * Pointer to the MLU memory that stores the divisor tensor.
 * @param[in] z_desc
 * The descriptor of the output tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] z
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - Data type of input tensors and output tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - output tensor: half, float
 *
 * @par Scale Limitation
 * - The input tensors and output tensor must have the same shape.
 *
 * @note
 * - The input tensors and output tensor have the same shape, and the input
 *   tensor \b y must meet the following input data range:
 *   - float: [-1e10,-1e-20] & [1e-20,1e10]
 *   - half: [-65504,-1e-4] & [1e-4,65504]
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
mluOpDiv(mluOpHandle_t handle,
         const mluOpComputationPreference_t prefer,
         const mluOpTensorDescriptor_t x_desc,
         const void *x,
         const mluOpTensorDescriptor_t y_desc,
         const void *y,
         const mluOpTensorDescriptor_t z_desc,
         void *z);

// Group:GenerateProposalsV2
/*!
 *  @brief Gets extra space size that is needed in poly_nms operation.
 *
 *  @param[in] handle
 *  Handle to an MLUOP context that is used to manage MLU devices
 *  and queues in the GenerateProposalsV2 operation.
 *  @param[in] scores_desc
 *  The descriptor of the scores tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[out] size
 *  A host pointer to the returned size of extra space in bytes.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @par Reference
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetGenerateProposalsV2WorkspaceSize(mluOpHandle_t handle, const mluOpTensorDescriptor_t scores_desc, size_t *size);

// Group:GenerateProposalsV2
/*!
 *  @brief Generates bounding box proposals for Faster Region-CNN.
 *  This operation is the second version of generate_proposals op.
 *  The proposals are generated for a list of images based on image
 *  score 'Scores', bounding box regression result `BboxDeltas` as
 *  well as predefined bounding box shapes `anchors`. Greedy non-maximum
 *  suppression is applied to generate the final bounding boxes.
 *
 *  @param[in] handle
 *  Handle to an MLUOP context that is used to manage MLU devices
 *  and queues in the poly_nms operation.
 *  @param[in] pre_nms_top_n
 *  Number of top scoring RPN proposals to keep before applying
 *  NMS.
 *  @param[in] post_nms_top_n
 *  Number of top scoring RPN proposals to keep after applying
 *  NMS.
 *  @param[in] nms_thresh
 *  NMS threshold used on RPN proposals.
 *  @param[in] min_size
 *  Proposal height and width both need to be greater than this
 *  min_size.
 *  @param[in] eta
 *  The parameter for adaptive NMS.
 *  @param[in] pixel_offset
 *  If true, im_shape pixel offset is 1.
 *  @param[in] scores_desc
 *  The descriptor of the input tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[in] scores
 *  Pointer to the MLU memory that stores the input tensor. The
 *  scores from conv is in shape (N, H, W, A), N is batch size, A is
 *  number of anchors, H and W are height and width of the feature map.
 *  @param[in] bbox_deltas_desc
 *  The descriptor of the input tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[in] bbox_deltas
 *  Pointer to the MLU memory that stores the input tensor.
 *  @param[in] im_shape_desc
 *  The descriptor of the input tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[in] im_shape
 *  Pointer to the MLU memory that stores the input tensor. Image
 *  shape in shape (N, 2), in format (height, width)
 *  @param[in] anchors_desc
 *  The descriptor of the input tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[in] anchors
 *  Pointer to the MLU memory that stores the input tensor.
 *  Bounding box anchors from anchor_generator_op is in shape (H, W, A, 4).
 *  @param[in] variances_desc
 *  The descriptor of the input tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[in] variances
 *  Pointer to the MLU memory that stores the input tensor.
 *  Bounding box variances with same shape as `anchors`.
 *  @param[in] workspace
 *  Pointer to the MLU memory that stores the extra workspace.
 *  @param[in] workspace_size
 *  The size of extra space.
 *  @param[in] rpn_rois_desc
 *  The descriptor of the output tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[out] rpn_rois
 *  Pointer to the MLU memory that stores the output tensor.
 *  Output proposals with shape (N * post_nms_top_n, 4).
 *  @param[in] rpn_roi_probs_desc
 *  The descriptor of the output tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[out] rpn_roi_probs
 *  Pointer to the MLU memory that stores the output tensor.
 *  Scores of proposals with shape (N * post_nms_top_n, 1).
 *  @param[in] rpn_rois_num_desc
 *  The descriptor of the output tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[out] rpn_rois_num
 *  Pointer to the MLU memory that stores the output tensor. The
 *  number of Rpn RoIs in each image.
 *  @param[in] rpn_rois_batch_size
 *  Pointer to the MLU memory that stores the output tensor. Indicates
 *  the number of return values of output.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *    ::MLUOP_STATUS_NOT_SUPPORTED
 *
 *  @par Data Type
 *  - The supported data types of input and output tensors are as follows:
 *     - scores: float
 *     - bbox_deltas: float
 *     - im_shape: float
 *     - anchors: float
 *     - variances: float
 *     - pre_nms_top_n: int32
 *     - post_nms_top_n: int32
 *     - nms_thresh: float
 *     - min_size: float
 *     - eta: float
 *     - pixel_offset: bool
 *     - rpn_rois: float
 *     - rpn_roi_probs: float
 *     - rpn_rois_num: int32
 *     - rpn_rois_batch_size: int32
 *
 *  @par Data Layout
 *  - The supported data layout of \b input, \b output,
 *     \b output_size are as follows:
 *
 *   - input tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output_size tensor: \p MLUOP_LAYOUT_ARRAY
 *
 *  @par Scale Limitation
 *  - The dimension of \b scores should be equal to 4.
 *  - The dimension of \b bbox_deltas should be equal to 4.
 *  - The dimension of \b im_shape should be equal to 2.
 *  - The dimension of \b anchors should be equal to 4.
 *  - The dimension of \b variances should be equal to 4.
 *  - The dimension of \b rpn_rois should be equal to 2.
 *  - The dimension of \b rpn_roi_probs should be equal to 2.
 *  - The dimension of \b rpn_rois_num should be equal to 1.
 *  - The dimension of \b rpn_rois_batch_size should be equal to 1.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 *
 *  @par Note
 *  - This commit does not support nan/inf or adaptive NMS.
 *  - The attribute `eta` should not be less than 1.
 *  - 'nms_thresh' should be more than 0.
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

// Group:PolyNms
/*!
 *  @brief Gets extra space size that is needed in poly_nms operation.
 *
 *  @param[in] handle
 *  Handle to an MLUOP context that is used to manage MLU devices
 *  and queues in the poly_nms operation.
 *  @param[in] boxes_desc
 *  The descriptor of the boxes tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[out] size
 *  A host pointer to the returned size of extra space in bytes.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @par Reference
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetPolyNmsWorkspaceSize(mluOpHandle_t handle, const mluOpTensorDescriptor_t boxes_desc, size_t *size);

// Group:PolyNms
/*!
 *  @brief Polygon Non Maximum Suppression.
 *
 *  @param[in] handle
 *  Handle to an MLUOP context that is used to manage MLU devices
 *  and queues in the poly_nms operation.
 *  @param[in] boxes_desc
 *  The descriptor of the input tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[in] boxes
 *  Pointer to the MLU memory that stores the input tensor.
 *  @param[in] iou_threshold
 *  The iou_threshold data.
 *  @param[in] workspace
 *  Pointer to the MLU memory that stores the extra workspace.
 *  @param[in] workspace_size
 *  The size of extra space.
 *  @param[in] output_desc
 *  The descriptor of the output tensor. For detailed information,
 *  see ::mluOpTensorDescriptor_t.
 *  @param[out] output
 *  Pointer to the MLU memory that stores the output tensor.
 *  @param[in] output_size
 *  Pointer to the MLU memory that stores the output tensor. Indicates
 *  the number of return values of output.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *    ::MLUOP_STATUS_NOT_SUPPORTED
 *
 *  @par Data Type
 *  - The supported data types of input and output tensors are as follows:
 *     - input tensor: float
 *     - iou_threshold: float
 *     - Output tensor: int32
 *     - output_size tensor: int32
 *
 *  @par Data Layout
 *  - The supported data layout of \b input, \b output,
 *     \b output_size are as follows:
 *
 *   - input tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output_size tensor: \p MLUOP_LAYOUT_ARRAY
 *
 *  @par Scale Limitation
 *  - The dimension of \b input should be equal to 2.
 *  - The dimension of \b output should be equal to 1.
 *  - The dimension of \b output_size should be equal to 1.
 *  - The shape[0] of output should be equal to input shape[0].
 *  - The shape[1] of input should be equal to 9.
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
 *    calculation result of the competitor operation.
 *  - If there are cases with the same score in the input boxes, the output
 *    results may be inconsistent with the results of competing products.
 *  - The number of input boxes on MLU270, MLU290 and MLU370 does not exceed
 *    9770.
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

// Group:PriorBox
/*!
 *  @brief Generates prior boxes for SSD (Single Shot MultiBox Detector) algorithm.
 *
 *  @param[in] handle
 *  Handle to an MLUOP context that is used to manage MLU devices
 *  and queues in the prior_box operation.
 *  @param[in] min_sizes_desc
 *  The descriptor of the min_sizes tensor. The minimum sizes of generated
 *  prior boxes.
 *  @param[in] min_sizes
 *  Pointer to the MLU memory that stores the min_sizes tensor.
 *  @param[in] aspect_ratios_desc
 *  The descriptor of the aspect_ratios tensor. The aspect ratios of
 *  generated prior boxes.
 *  @param[in] aspect_ratios
 *  Pointer to the MLU memory that stores the aspect_ratios tensor.
 *  @param[in] variances_desc
 *  The descriptor of the variances tensor. The variances to be
 *  encoded in prior boxes.
 *  @param[in] variances
 *  Pointer to the MLU memory that stores the variances tensor.
 *  @param[in] max_sizes_desc
 *  The descriptor of the max_sizes tensor. The maximum sizes of generated
 *  prior boxes.
 *  @param[in] max_sizes
 *  Pointer to the MLU memory that stores the max_sizes tensor.
 *  @param[in] height
 *  The height of the \b input feature_map.
 *  @param[in] width
 *  The width of the \b input feature_map.
 *  @param[in] im_height
 *  The height of the \b input image.
 *  @param[in] im_width
 *  The width of the \b input image.
 *  @param[in] step_h
 *  The prior box step in height.
 *  @param[in] step_w
 *  The prior box step in width.
 *  @param[in] offset
 *  The prior box center offset.
 *  @param[in] clip
 *  Whether to clip out-of-boundary boxes.
 *  @param[in] min_max_aspect_ratios_order
 *  If the value is set as true, the \b output prior box is in
 *  the order of [min, max, aspect_ratios]; otherwise the order is
 *  [min, aspect_ratios, max].
 *  @param[in] output_desc
 *  The descriptor of the \b output tensor. The \b output prior boxes of
 *  PriorBox.
 *  @param[out] output
 *  Pointer to the MLU memory that stores the \b output tensor.
 *  @param[in] var_desc
 *  The descriptor of the var tensor. The expanded variances of
 *  PriorBox.
 *  @param[out] var
 *  Pointer to the MLU memory that stores the var tensor.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *    ::MLUOP_STATUS_NOT_SUPPORTED
 *
 *  @par Data Type
 *  - The supported data types of \b input and \b output are as follows:
 *     - min_sizes tensor: float
 *     - aspect_ratios tensor: float
 *     - variances tensor: float
 *     - max_sizes tensor: float
 *     - height: int
 *     - width: int
 *     - im_height: int
 *     - im_width: int
 *     - step_h: float
 *     - step_w: float
 *     - offset: float
 *     - clip: bool
 *     - min_max_aspect_ratios_order: bool
 *     - output: float
 *     - var: float
 *
 *  @par Data Layout
 *  - The supported data layouts of \b input, \b output,
 *    are as follows:
 *
 *   - input tensor:
 *     - min_sizes: \p MLUOP_LAYOUT_ARRAY
 *     - aspect_ratios: \p MLUOP_LAYOUT_ARRAY
 *     - variances: \p MLUOP_LAYOUT_ARRAY
 *     - max_sizes: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor:
 *     - output: \p MLUOP_LAYOUT_ARRAY
 *     - var: \p MLUOP_LAYOUT_ARRAY
 *
 *  @par Scale Limitation
 *  - The dimension of \b min_sizes should be equal to 1.
 *  - The dimension of \b aspect_ratios should be equal to 1.
 *  - The dimension of \b variances should be equal to 1.
 *  - The dimension of \b max_sizes should be equal to 1.
 *  - The dimension of \b output should be equal to 1.
 *  - The dimension of \b var should be equal to 1.
 *  - The shape[0] of \b variances should be equal to 4.
 *  - The shape[0] of \b min_sizes should be larger than 0.
 *  - The shape[0] of \b aspect_ratios should be larger than 0.
 *  - The shape of \b output should be the same with \b var.
 *  - The shape[0] of the \b output should be equal to the input height.
 *  - The shape[1] of the \b output should be equal to the input width.
 *  - The shape[2] of the \b output and \b var must be less than 2100
 *     in MLU200 series, and less than 2900 in MLU300 series.
 *  - The shape[2] of \b output and \b var should be equal to
 *     the product of shape[0] of \b min_sizes and \b aspect_ratios
 *     plus shape[0] of \b max_sizes.
 *  - The height should be greater than or equal to 0.
 *  - The width should be greater than or equal to 0.
 *  - The step_h should be greater than 0.
 *  - The step_w should be greater than 0.
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 *
 *  @par Note
 *  - The shape[2] of the \b output and \b var must be
 *    less than 2100 in MLU200 series, while less than 2900 in MLU300 series.
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

// Group:PsRoiPool
/*!
 *  @brief Generates fixed size feature map for each ROI (Regions of Interest).
 *
 *  @param[in] handle
 *  Handle to an MLUOP context that is used to manage MLU devices
 *  and queues in the psroipool_forward operation. For detailed information,
 *  see::mluOpHandle_t.
 *  @param[in] spatial_scale
 *  The spatial scale of each ROI in the output.
 *  @param[in] group_size
 *  Sets the number of \b rois to be divided equally in each direction.
 *  @param[in] pooled_height
 *  The pooled_height data.
 *  @param[in] pooled_width
 *  The pooled_width data.
 *  @param[in] output_dim
 *  The output_dim data.
 *   @param[in] input_desc
 *  Descriptor of input tensor, which contains dimension and the layout of input.
 *  For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[in] input
 *  Pointer to the MLU memory that stores the input tensor. The shape of \b input is
 *  [batch_num, H, W, C].
 *  @param[in] rois_desc
 *  Descriptor of rois tensor, which contains dimension and the layout of rois.
 *  For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[in] rois
 *  Pointer to the MLU memory that stores the rois tensor. \b rois[1] consists of
 *  [batch_id, roi_start_w, roi_start_h, roi_end_w, roi_end_h], where \p batch_id is the ID
 *  of the batch.
 *  @param[in] output_desc
 *  Descriptor of output tensor, containing dimension and the layout of output.
 *  For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[out] output
 *  Pointer to the MLU memory that stores the output tensor. The shape of \b output is
 *  [rois[0], pooled_height, pooled_width, output_dim].
 *  @param[in] mapping_channel_desc
 *  Descriptor of the mapping_channel tensor, which contains dimension and the layout of
 *  mapping_channel. For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[out] mapping_channel
 *  Pointer to the MLU memory that stores the mapping_channel tensor. The shape of
 *  \b mapping_channel is [rois[0], pooled_height, pooled_width, output_dim].
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @par Data Type
 *  - The supported data types of input and output tensors are as follows:
 *    - input tensor: float
 *    - Rois tensor: float
 *    - output tensor: float
 *    - Mapping_channel tensor: int32
 *
 *  @par Data Layout
 *  - The supported data layout of \b input, \b rois, \b output, and \b mapping_channel
 *    are as follows:
 *     - input tensor: \p MLUOP_LAYOUT_NHWC
 *     - Rois tensor: \p MLUOP_LAYOUT_ARRAY
 *     - output tensor: \p MLUOP_LAYOUT_NHWC
 *     - Mapping_channel tensor: \p MLUOP_LAYOUT_NHWC
 *
 *  @par Scale Limitation
 *  - The input tensor, mapping_channel tensor and output tensor must have four dimensions.
 *  - The \b rois tensor should be 2D array.
 *  - The shape of \b rois should be [rois_num, 5].
 *  - \p batch_id should be in the range of [0, \p batch_num - 1].
 *  - The spatial_scale should be greater than 0.
 *  - The group_size should be greater than 1.
 *  - THe output_dim should be greater than 1.
 *  - The group_size should be equal to pooled_height.
 *  - The pooled_height should be equal to pooled_width.
 *  - The fourth dimension of input tensor should be equal to pooled_height * pooled_width *
 *    output_dim.
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

// Group:PsRoiPool
/*!
 *  @brief Computes the gradients of feature map \b bottom_grad based on the
 *  inputs \b top_grad , \b rois and \b mapping_channel to perform the backpropagation
 *  of the ::mluOpPsRoiPoolForward operation.
 *
 *  @param[in] handle
 *  Handle to an MLUOP context that is used to manage MLU devices and queues in the
 *  psroipool_forward operation. For detailed information, see ::mluOpHandle_t.
 *  @param[in] pooled_height
 *  An integer value which is the height of the output after pooling.
 *  @param[in] pooled_width
 *  An integer value which is the width of the output after pooling.
 *  @param[in] spatial_scale
 *  A float value which is the scale factor of coordinates of rois.
 *  @param[in] output_dim
 *  An integer value which is the channel of the output after pooling.
 *  @param[in] top_grad_desc
 *  Descriptor of the top_grad tensor, which contains the dimension and the layout
 *  of top_grad tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[in] top_grad
 *  Pointer to the MLU memory that stores the top_grad tensor.
 *  @param[in] rois_desc
 *  Descriptor of the rois tensor, which contains the dimension and the layout
 *  of rois tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[in] rois
 *  Pointer to the MLU memory that stores the rois tensor.
 *  @param[in] mapping_channel_desc
 *  Descriptor of the mapping_channel tensor, which contains the dimension and the
 *  layout of mapping_channel. For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[in] mapping_channel
 *  Pointer to the MLU memory that stores the mapping_channel tensor.
 *  @param[in] bottom_grad_desc
 *  Descriptor of the bottom_grad tensor, which contains the dimension and the
 *  layout of mapping_channel. For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[out] bottom_grad
 *  Pointer to the MLU memory that stores the bottom_grad tensor.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 *  @par Data Type
 *  - The supported data types of top_grad tensor \b top_grad, rois tensor \b rois,
 *    mapping_channel tensor \b mapping_channel and bottom_grad tensor \b bottom_grad
 *    are as follows:
 *    - top_grad tensor: float
 *    - rois tensor: float
 *    - mapping_channel tensor: int
 *    - bottom_grad tensor: float
 *
 *  @par Data Layout
 *  - The supported data layouts of top_grad tensor \b top_grad, rois tensor \b rois,
 *    mapping_channel tensor \b mapping_channel and bottom_grad tensor \b bottom_grad
 *    are as follows:
 *    - top_grad tensor: \p MLUOP_LAYOUT_NHWC
 *    - rois tensor: \p MLUOP_LAYOUT_ARRAY
 *    - mapping_channel tensor: \p MLUOP_LAYOUT_NHWC
 *    - bottom_grad tensor: \p MLUOP_LAYOUT_NHWC
 *
 *  @par Scale Limitation
 *  - The top_grad tensor, mapping_channel tensor and bottom_grad tensor must be 4-D.
 *  - Each dimension of the top_grad tensor and the mapping_channel tensor must be the same.
 *  - The rois tensor be be 2-D.
 *  - The shape of \b top_grad should be [rois_num, pooled_height, pooled_width, output_dim].
 *  - The shape of \b rois should be [rois_num, 5].
 *  - The shape of \b mapping_channel should be [rois_num, pooled_height, pooled_width, output_dim].
 *  - the shape of \b bottom_grad should be [batch_num, height, width, channels].
 *  - \b rois[i] consists of [batch_id, roi_start_w, roi_start_h, roi_end_w, roi_end_h].
 *    \p batch_id should be in the range of [0, batch_num -1].
 *  - The \b spatial_scale should be larger than 0.
 *  - The \b output_dim should be larger than or equal to 1.
 *  - The \b pooled_height should be equal to \b pooled_width.
 *  - The \p channels should be equal to \b pooled_height * \b pooled_width * \b output_dim.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 *
 *  @par Note
 *  - On MLU300 series, rois does not support NAN/INF.
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

// Group:RoiAlignRotated
/*!
 * @brief Extracts the corresponding \b features information to \b output by bilinear interpolation
 * according to the \b rois with rotation.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in
 * ::mluOpRoiAlignRotatedForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] features_desc
 * The descriptor of the features tensor.
 * @param[in] features
 * Pointer to the MLU memory that stores the features tensor. The shape of \b features
 * is [batch_num, H, W, C].
 * @param[in] rois_desc
 * The descriptor of rois tensor, which contains dimension and the layout of rois.
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
 * A boolean value which determines whether to shift the ROI by 0.5 pixel. If the
 * value of \b aligned is set to true, the ROI is shifted by 0.5. If the value of \b aligned
 * is set to false, the ROI is not shifted.
 * @param[in] clockwise
 * A boolean value which determines whether the rotation of ROI is clockwise.
 * @param[out] output_desc
 * The descriptor of output, which contains dimension and the layout of output.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - This function supports the following data types for input tensor \b features, \b rois,
 *   and output tensor \b output. Data types of all tensors should be the same.
 *   - input tensor: half, float
 *   - rois tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b features, \b rois, and \b output are as follows:
 *   - input tensor: \p MLUOP_LAYOUT_NHWC
 *   - rois tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The \b features tensor and \b output tensor should be 4D.
 * - The half data type is not recommended due to low precision.
 * - Size of the lowest dimension of \b features tensor and \b output tensor should be the same.
 * - The \b rois tensor should be 2D array.
 * - Size of the highest dimension of \b output tensor and \b rois tensor should be the same.
 * - The shape of \b rois should be [rois_num, 6].
 * - \p batch_id should be in the range of [0, \p batch_num - 1]; \p x and \p y should be greater than or
 *   equal to 0 and less than \p H and \p W respectively. Both of \p h and \p w should be greater than zero
 *   and less than \p H and \p W respectively.
 * - \p spatial_scale and \p sample_ratio should not be less than zero.
 *
 * @note
 * - NaN and infinity are not supported for all parameters in \b boxes, except for the \p x and \p y parameters
 *   that support infinity.
 * - The values of the parameters \p x , \p y, \p w and \p h in \b rois multiplied by \p spatial_scale cannot exceed
 *   the range that can be represented by the parameter type.
 *
 * @par Requirements
 * - None.
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

// Group:RoiAlignRotated
/*!
 * @brief Computes the gradients of feature map \b bottom_grad based on the input \b top_grad and
 * \b rois to perform the backpropagation of the ::mluOpRoiAlignRotatedForward operation.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in
 * ::mluOpRoiAlignRotatedBackward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] top_grad_desc
 * The descriptor of the gradient tensor in the backpropagation process.
 * @param[in] top_grad
 * Pointer to the MLU memory that stores the top_grad tensor.
 * @param[in] rois_desc
 * The descriptor of rois tensor, which contains dimension and the layout of rois.
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
 * A boolean value which determines whether to shift the ROI by 0.5 pixel.
 * If the value of \b aligned is set to true, the ROI is shifted by 0.5. If the value
 * of \b aligned is set to false, the ROI is not shifted.
 * @param[in] clockwise
 * A boolean value which determines whether the rotation of ROI is clockwise.
 * @param[in] bottom_grad_desc
 * The descriptor of the gradient tensor of the origin feature map.
 * @param[out] bottom_grad
 * Pointer to the MLU memory that stores the bottom_grad tensor. The shape of
 * bottom_grad is [batch_num, H, W, C].
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - This function supports the following Data types for input tensor \b top_grad, \b rois,
 *   and output tensor \b bottom_grad. Data types of all tensors should be the same.
 *   - top_grad tensor: half, float
 *   - rois tensor: half, float
 *   - bottom_grad tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b top_grad, \b rois, and \b bottom_grad are as follows:
 *   - top_grad tensor: \p MLUOP_LAYOUT_NHWC
 *   - rois tensor: \p MLUOP_LAYOUT_ARRAY
 *   - bottom_grad tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The \b bottom_grad tensor and \b top_grad tensor should be 4D.
 * - The half data type is not recommended due to low precision.
 * - Size of the lowest dimension of \b bottom_grad tensor and \b top_grad tensor should be the same.
 * - The \b rois tensor should be 2D array.
 * - Size of the highest dimension of \b top_grad tensor and \b rois tensor should be the same.
 * - \p batch_id should be in the range of [0, \p batch_num - 1], \p x and \p y should be greater than or
 *   equal to 0 and less than \p H and \p W respectively. Both of \p h and \p w should be greater than zero
 *   and less than \p H and \p W respectively.
 * - \p spatial_scale and \p sample_ratio should not be less than zero.
 *
 * @note
 * - NaN and infinity are not supported for all parameters in \b boxes, except for the \p x and \p y parameters
 *   that support infinity.
 * - The values of the parameters \p x , \p y, \p w and \p h in \b rois multiplied by \p spatial_scale cannot exceed
 *   the range that can be represented by the parameter type.
 *
 * @par Requirements
 * - None.
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

// Group:RoiCrop
/*!
 * @brief Generates fixed size feature map for each grid. Each value in the
 * feature map is interpolated by bilinear sampling.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in ::mluOpRoiCropForward operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] input_desc
 * The descriptor of the input tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] grid_desc
 * The descriptor of the grid tensor. For detailed information, see
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
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - Data types of input tensors and output tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: float
 *   - Grid tensor: float
 *   - output tensor: float
 * @par Data Layout
 * - The supported data layout of \b input , \b grid , \b output are as follows:
 *   - input tensor: \p MLUOP_LAYOUT_NHWC
 *   - Grid tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The input tensor, grid tensor and output tensor must have four dimensions.
 * - Size of the first dimension of input tensor is divided by size of the
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
mluOpStatus_t MLUOP_WIN_API
mluOpRoiCropForward(mluOpHandle_t handle,
                    const mluOpTensorDescriptor_t input_desc,
                    const void *input,
                    const mluOpTensorDescriptor_t grid_desc,
                    const void *grid,
                    const mluOpTensorDescriptor_t output_desc,
                    void *output);

// Group:RoiCrop
/*!
 * @brief Computes the gradients of images \b grad_input based on the gradients
 * \b grad_output and coordinates mapping parameter \b grid to perform the
 * backpropagation.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in ::mluOpRoiCropBackward operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] grad_output_desc
 * The descriptor of the grad_output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] grad_output
 * Pointer to the MLU memory that stores the gradient tensor \b grad_output
 * in the backpropagation process.
 * @param[in] grid_desc
 * The descriptor of the grid tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] grid
 * Pointer to the MLU memory that stores the coordinate mapping
 * tensor.
 * @param[in] grad_input_desc
 * The descriptor of the grad_input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] grad_input
 * Pointer to the MLU memory that stores the gradient tensor of the
 * original images.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - Data types of all tensors must be the same.
 * - The supported data types of all tensors are as follows:
 *   - Grad_input tensor: float
 *   - Grad_output tensor: float
 *   - Grid tensor: float
 * @par Data Layout
 * - The supported data layout of \b grad_output , \b grid , \b grad_input are as
 *   follows.
 *   - Grad_output tensor: \p MLUOP_LAYOUT_NHWC
 *   - Grid tensor: \p MLUOP_LAYOUT_ARRAY
 *   - Grad_input tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The grad_output tensor, grid tensor and grad_input tensor must have four
 *   dimensions.
 * - Size of the first dimension of grad_input tensor is divided by size of
 *   the first dimension of grid tensor.
 * - The second dimension of grid tensor and grad_output tensor must be the same size.
 * - The third dimension of grid tensor and grad_output tensor must be the same size.
 * - The fourth dimension of grad_input \b grad_input tensor and grad_output tensor
 *   \b grad_output must be the same size.
 * - Size of the fourth dimension of grid tensor \b grid must be equal to 2.
 * - Grid tensor \b grid must meet the following data range:
 *   - Float: [-1.0,1.0]
 * @par Requirements
 * - None.
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

// Group:RotatedFeatureAlign
/*!
 * @brief Uses the feature interpolation to obtain the position information corresponding to the
 * refined rotate anchors \b bboxes and reconstructs the feature maps \b output in pixel-wise
 * manner to achieve feature alignment.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in
 * ::mluOpRotatedFeatureAlignForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] input_desc
 * The descriptor of the input tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] bboxes_desc
 * The descriptor of bboxes, which contains the dimension and layout of bboxes tensor.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] bboxes
 * Pointer to the MLU memory that stores the bboxes tensor.
 * @param[in] spatial_scale
 * A float value that is the scale factor of coordinates of bboxes.
 * @param[in] points
 * An int value that is the number of sample points. Only 1 and 5 are supported. The default value is 1.
 * @param[in] output_desc
 * The descriptor of output tensor, which contains the dimension and layout of output tensor.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - This function supports the following data types for input tensor \b input, bboxes tensor \b
 *   bboxes, and output tensor \b output. Data types of all tensors should be the same.
 *   - input tensor: half, float
 *   - bboxes tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b input, \b bboxes and \b output are as follows:
 *   - input tensor: \p MLUOP_LAYOUT_NHWC
 *   - bboxes tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor: \p MLUOP_LAYOUT_NHWC
 *
 * @par Scale Limitation
 * - The input tensor and output tensor must be 4D.
 * - Size of the each dimension of input tensor and output tensor must be the same.
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
 * @note
 * - The inputs \b bboxes and \b spatial_scale with NaN or infinity are not supported.
 *
 * @par Requirements
 * - None.
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
mluOpRotatedFeatureAlignForward(const mluOpHandle_t handle_,
                                const mluOpTensorDescriptor_t input_desc,
                                const void *input_ptr,
                                const mluOpTensorDescriptor_t bboxes_desc,
                                const void *bboxes_ptr,
                                const float spatial_scale,
                                const int points,
                                const mluOpTensorDescriptor_t output_desc,
                                void *output_ptr);

// Group:RotatedFeatureAlign
/*!
 * @brief Computes the gradients of feature map \b bottom_input based on the inputs \b top_output
 * and \b bboxes to perform the backpropagation of the ::mluOpRotatedFeatureAlignForward operation.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in
 * ::mluOpRotatedFeatureAlignBackward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] top_output_desc
 * The descriptor of the top_output tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] top_output
 * Pointer to the MLU memory that stores the top_output tensor.
 * @param[in] bboxes_desc
 * The descriptor of bboxes, which contains the dimension and layout of bboxes tensor. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * @param[in] bboxes
 * Pointer to the MLU memory that stores the bboxes tensor.
 * @param[in] spatial_scale
 * A float value that is the scale factor of coordinates of bboxes.
 * @param[in] points
 * An int value that is the number of sample points. Only 1 and 5 are supported. The default value is 1.
 * @param[in] bottom_input_desc
 * The descriptor of bottom_input tensor, which contains the dimension and layout of bottom_input tensor.
 * @param[out] bottom_input
 * Pointer to the MLU memory that stores the bottom_input tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - This function supports the following data types for top_output tensor \b top_output, bboxes
 *   tensor \b
 *   bboxes, and bottom_input tensor \b bottom_input. Data types of all tensors should be the same.
 *   - top_output tensor: half, float
 *   - bboxes tensor: half, float
 *   - bottom_input tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b top_output, \b bboxes and \b bottom_input are as follows:
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
 * @note
 * - The inputs \b bboxes and \b spatial_scale with NaN or infinity are not supported.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/rotated_feature_align.py
 */
mluOpStatus_t MLUOP_WIN_API
mluOpRotatedFeatureAlignBackward(const mluOpHandle_t handle_,
                                 const mluOpTensorDescriptor_t top_output_desc,
                                 const void *top_output_ptr,
                                 const mluOpTensorDescriptor_t bboxes_desc,
                                 const void *bboxes_ptr,
                                 const float spatial_scale,
                                 const int points,
                                 const mluOpTensorDescriptor_t bottom_input_desc,
                                 void *bottom_input_ptr);

// Group:Sqrt
/*!
 * @brief Computes sqrt on input tensor \b x, and returns the results in the
 * output tensor \b y.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the sqrt operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] prefer
 * The \b prefer modes defined in ::mluOpComputationPreference_t.
 * @param[in] x_desc
 * The descriptor of the input tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] y_desc
 * The descriptor of the output tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] y
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - Data type of input tensor and output tensor should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensor: half, float
 *   - output tensor: half, float
 *
 * @par Scale Limitation
 * - The input tensor and output tensor must have the same shape, and the input
 *   tensor must meet the following input data range:
 *   - float: [1e-10,1e10]
 *   - half: [1e-3,1e-2] & [1e-1,60000]
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
mluOpStatus_t MLUOP_WIN_API
mluOpSqrt(mluOpHandle_t handle,
          const mluOpComputationPreference_t prefer,
          const mluOpTensorDescriptor_t x_desc,
          const void *x,
          const mluOpTensorDescriptor_t y_desc,
          void *y);
// Group:Sqrt

/*!
 * @brief Computes gradient of sqrt on input tensor \b y and \b diff_y, and
 *   returns the results in the output tensor \b diff_x.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the sqrt backward operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] y_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] y
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] dy_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] diff_y
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] dx_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] diff_x
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - Data types of input tensors and output tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensors: half, float
 *   - output tensor: half, float
 *
 * @par Scale Limitation
 * - The input tensor and output tensor must have the same shape, and the input
 *   tensor \b y must meet the following input data ranges:
 *   - float: [1e-10, 1e6]
 *   - half: [0.01, 500]
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
mluOpStatus_t MLUOP_WIN_API
mluOpSqrtBackward(mluOpHandle_t handle,
                  const mluOpTensorDescriptor_t y_desc,
                  const void *y,
                  const mluOpTensorDescriptor_t dy_desc,
                  const void *diff_y,
                  const mluOpTensorDescriptor_t dx_desc,
                  void *diff_x);

// Group:Voxelization
/*!
 * @brief Gets extra space size that is needed in voxelization operation.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices
 * and queues in the voxelization operation.
 * @param[in] points_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] voxel_size_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] coors_range_desc
 * The descriptor of the tensors. For detailed information, see
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
 * A bool value whether to invoke the non-deterministic
 * version of hard-voxelization implementations. Currently,
 * non-deterministic mode is not supported.
 * @param[in] voxels_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] coors_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] num_points_per_voxel_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] voxel_num_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 *  @param[out] size
 *  A host pointer to the returned size of extra space in bytes.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *    ::MLUOP_STATUS_NOT_SUPPORTED
 *
 *  @par Reference
 *  - None.
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

// Group:Voxelization
/*!
 * @brief Generates voxelization of input tensor \b points. Output tensor
 * \b voxels contains points in voxels; \b coors is the voxel coordinates;
 * \b num_points_per_voxel is the number of points per voxel; \b voxel_num
 * is the number of voxels.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the voxelization operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] points_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] points
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] voxel_size_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] voxel_size
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] coors_range_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] coors_range
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] max_points
 * An integer value which is the maximum number of points contained
 * in a voxel.
 * @param[in] max_voxels
 * An integer value which is the maximum number of voxels this
 * function create.
 * @param[in] NDim
 * An integer value which is the second dimension of coors.
 * @param[in] deterministic
 * A bool value whether to invoke the non-deterministic
 * version of hard-voxelization implementations. Currently,
 * non-deterministic mode is not supported.
 * @param[in] workspace
 * Pointer to the MLU memory that stores the extra workspace.
 * @param[in] workspace_size
 * The size of extra space.
 * @param[in] voxels_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] voxels
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] coors_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] coors
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] num_points_per_voxel_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] num_points_per_voxel
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] voxel_num_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] voxel_num
 * Pointer to the MLU memory that stores the input tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - points, voxel_size, coors_range, voxels: float
 *   - coors, num_points_per_voxel, voxel_num: int
 *
 * @par Scale Limitation
 * - max_points and max_voxels must be greater than or equal to 0.
 * - NDim must be equal to 3, which means 3D.
 * - The value of the deterministic mode must be True. Currently,
 *   the non-deterministic mode is not supported.
 *
 * @par Requirements
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

// Group:YoloBox
/*!
 * @brief Computes bounding box information from the backbone output of the
 * detected network.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the yolo_box operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] x_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] img_size_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] img_size
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] anchors_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] anchors
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] class_num
 * The number of classes.
 * @param[in] conf_thresh
 * The detection boxes with the confidence score below the threshold should be ignored.
 * @param[in] downsample_ratio
 * The downsample ratio from network input to yolo_box operation input,
 * so 32, 16, 8 should be set for the first, second, and thrid into yolo_box operation.
 * @param[in] clip_bbox
 * Whether clip output bounding box in img_size boundary.
 * @param[in] scale
 * Scale the center point of decoded bounding box.
 * @param[in] iou_aware
 * Whether use iou aware.
 * @param[in] iou_aware_factor
 * iou aware factor.
 * @param[in] boxes_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] boxes
 * Pointer to the MLU memory that stores the output tensor.
 * @param[in] scores_desc
 * The descriptor of the tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] scores
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - Data types of input tensors and output tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input x tensor: float
 *   - input img_size and anchors tensors: int
 *   - output tensors: float
 *
 * @par Scale Limitation
 * - The first dimension of x tensor, img_size tensor, boxes tensor and scores
 *   tensor must be the same size.
 * - The second dimension(the channel dimension) of x tensor , C should be equal to S * (5 +
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
 * - The \b class_num should be larger than 0. On MLU200, the value cannot be
 *   greater than 1534. On MLU300, the value cannot be greater than 2558.
 *
 * @par Requirements
 * - None.
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

// Group:VoxelPooling
/*!
 * @brief Computes bounding box information from the backbone output of the
 * detected network.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
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
 * The descriptor of the tensors. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] geom_xyz
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] input_features_desc
 * The descriptor of the tensors. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] input_features
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] output_features_desc
 * The descriptor of the tensors. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] output_features
 * Pointer to the MLU memory that stores the output tensor.
 * @param[in] pos_memo_desc
 * The descriptor of the tensors. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] pos_memo
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *
 *   - geom_xyz tensor: int
 *   - input_features tensor: float
 *   - output_features tensor: float
 *   - pos_memo tensor: int
 *
 * @par Data Layout
 *  - The supported data layouts of \b geom_xyz, \b input_features, \b output_features and \b pos_memo are
 *    as follows:
 *
 *   - input tensor:
 *     - geom_xyz tensor: \p MLUOP_LAYOUT_ARRAY
 *     - input_features tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor:
 *     - output_features tensor: \p MLUOP_LAYOUT_ARRAY
 *     - pos_memo tensor: \p MLUOP_LAYOUT_ARRAY
 *
 *  @par Scale Limitation
 *  - The geom_xyz tensor, input_features tensor and pos_memo tensor must be 3D.
 *  - The output_features tensor must be 4D.
 *  - The shape of \b geom_xyz should be [batch_size, num_points, 3].
 *  - The shape of \b input_features should be [batch_size, num_points, num_channels].
 *  - The shape of \b output_features should be [batch_size, num_voxel_y, num_voxel_x, num_channels].
 *  - The shape of \b pos_memo should be [batch_size, num_points, 3].
 *  - The \b batch_size, \b num_points, \b num_channels, \b num_voxel_x and \b num_voxel_y should be larger than zero.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 *
 *  @par Note
 *  - The operation does not support MLU200 series.
 *  - You need to set the initial value for the output \b pos_memo before calling the operation, and initialize it to a
 *    negative number.
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

// Group:BoxIouRotated
/*!
 * @brief Computes the intersection-over-union (Jaccard index, IOU) of rotated
 * bounding-boxes. If \b aligned is false, then calculate the IOUs
 * between each rotated bounding-box of \b bbox1 and \b bbox2, otherwise calculate
 * the IOUs between each aligned pair of rotated bounding-box of \b bbox1
 * and \b bbox2.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the box_iou_rotated operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] mode
 * An integer value which decides to return a result of
 * IOUs (Intersection Over Union) or IOFs (Intersection Over Foreground).
 * The integer 0 represents IOU and 1 represents IOF.
 * @param[in] aligned
 * A boolean value. If it is false, then calculate the IOUs[i][j]
 * or IOFs[i][j] between the row i of \b bbox1 and the row j of \b bbox2,
 * otherwise calculate the IOUs[i] or IOFs[i] between the row i of \b bbox1
 * and the row i of \b bbox2. Significantly, the numbers of rows of \b bbox1
 * and \b bbox2 must be equal when \b aligned is true.
 * @param[in] bbox1_desc
 * The descriptor of the input tensor \b bbox1 (rotated bounding-box).
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] bbox1
 * Pointer to the MLU memory that stores the input tensor \b bbox1.
 * It has shape (n, 5), indicating (x, y, w, h, theta) for each row.
 * Note that theta is in radian.
 * @param[in] bbox2_desc
 * The descriptor of the input tensor \b bbox2 (rotated bounding-box).
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] bbox2
 * Pointer to the MLU memory that stores the input tensor \b bbox2.
 * It has shape (m, 5), indicating (x, y, w, h, theta) for each row.
 * Note that theta is in radian.
 * @param[in] ious_desc
 * The descriptor of the output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] ious
 * IOUs or IOFs of input rotated bounding-boxes. Pointer to the MLU
 * memory that stores the output tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - By the order of \b bbox1 - \b bbox2 - \b ious, the supported data types of
 *    \b bbox1, \b bbox2 and \b ious are as follows:
 *   - float - float - float
 *
 * @par Scale Limitation
 * - The number of dimensions of \b bbox1 and \b bbox2 tensors must be 2.
 * - The length of lowest dimension of \b bbox1 and \b bbox2 tensors must be 5.
 * - Both sets of boxes are expected to be in
 *   (x_center, y_center, width, height, angle) format.
 *   - \b bbox1 (Tensor): shape [n, 5] in (x, y, w, h, theta) format.
 *   - \b bbox2 (Tensor): shape [m, 5] in (x, y, w, h, theta) format.
 * - When aligned mode is true, for input \b bbox1 and \b bbox2 with n-rows,
 *   the output \b ious must be a 1D array with n-elements. When
 *   \b aligned is false, for input \b bbox1 with n-rows and \b bbox2 with
 *   m-rows, the output \b ious must be a 2D matrix with shape n*m.
 *
 * @note
 * - When finding the point with minimum y and minimum x in convex-hull-graham,
 *   BoxIouRotated performs min-pooling operation. If the input data of pooling
 * contains NaN:
 *   - On MLU200 series:
 *    - The \b output value is the NaN.
 *   - On MLU300 series:
 *    - If the last value in the kernel of the pooling is NaN, the \b output
 *      value is NaN. Otherwise, the \b output value is the minimum value after
 *      the last NaN.
 *
 * @par API Dependency
 * - None.
 *
 * @par Requirements
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

// Group:BboxOverlaps
/*!
 * @brief Computes the IOUs or IOFs between two sets of
 * bounding-boxes. If \b aligned is false, this operation calculates the IOU of each row between each bounding-box
 * of \b bbox1 and \b bbox2, otherwise, it calculates the IOU of the corresponding row between each aligned
 * pair of \b bbox1 and \b bbox2. For input placed in the order of <x1, y1, x2, y2>, (x1, y1) and (x2, y2)
 * respectively represents the top-left and bottom-right corner coordinates of bounding-box.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the
 * bounding-box overlaps operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] mode
 * An integer value which decides to return a result IOU or IOF.
 * The integer 0 represents IOU and 1 represents IOF.
 * @param[in] aligned
 * A boolean value. If it is false, this operation calculates the IOUs[i][j] or IOFs[i][j] between
 * the row i of \b bbox1 and the row j of \b bbox2, otherwise the IOU[i] or IOF[i] between
 * the row i of \b bbox1 and the row i of \b bbox2 are calculated. The number of row of \b bbox1
 * and \b bbox2 must be equal if \b aligned is true.
 * @param[in] offset
 * An integer value determines whether to increase the length and the width of the bounding-box by 0 or 1
 * before calculating the area.
 * @param[in] bbox1_desc
 * The descriptor of the input tensor \b bbox1. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] bbox1
 * Pointer to the MLU memory that stores the input tensor \b bbox1.
 * @param[in] bbox2_desc
 * The descriptor of the input tensor \b bbox2. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] bbox2
 * Pointer to the MLU memory that stores the input tensor \b bbox2.
 * @param[in] ious_desc
 * The descriptor of the output tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] ious
 * IOU or IOF. Pointer to the MLU memory that stores the output tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Formula
 * - See "Bounding-Box Overlaps Operation" section in "Cambricon BANGC OPS User Guide" for details.
 *
 * @par Data Type
 * By the order of \b bbox1 - \b bbox2 - \b ious, the supported data types of
 * \b bbox1, \b bbox2 and \b ious are as follows:
 * - float - float - float
 * - half  - half  - half
 *
 * @par Scale Limitation
 * - The number of dimensions of \b bbox1 and \b bbox2 tensors must be 2
 * - The lowest dimension of input tensor must be 4
 *   \b bbox1 (Tensor): shape [m, 4] in <x1, y1, x2, y2> format
 *   \b bbox2 (Tensor): shape [n, 4] in <x1, y1, x2, y2> format
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
 * @note
 * - The input tensor \b x should be in the following range to guarantee the accuracy of output:
 *   If bbox_overlaps works on (m)tp_2xx :
 *    - half : [-300, 100]
 *    - float : [-300, 100]
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of the bounding-box overlaps operation is as follows:
 *   @verbatim
 *    input array by 3 * 4, type is float -->
 *        input: bbox1 = [
 *          [0, 0, 10, 10],
 *          [10, 10, 20, 20],
 *          [32, 32, 38, 42],
 *        ]
 *    input array by 3 * 4, type is float -->
 *        input: bbox2 = [
 *          [0, 0, 10, 20],
 *          [0, 10, 10, 19],
 *          [10, 10, 20, 20],
 *        ]
 *    param:
 *      mode = 0
 *      aligned = False
 *      offset = 0
 *
 *
 *    output array by 3 * 3, type is float -->
 *        output: [[0.5000, 0.0000, 0.0000],
 *                 [0.0000, 0.0000, 1.0000],
 *                 [0.0000, 0.0000, 0.0000]]
 *   @endverbatim
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
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the three_interpolate_forward operation. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in] features_desc
 * The descriptor of the features tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] features
 * Pointer to the MLU memory that stores the input features tensor. The features'
 * shape (B, C, M), B is batch size, C is channel size, M is the number of
 * elements in one input channel.
 * @param[in] indices_desc
 * The descriptor of the indices tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] indices
 * Pointer to the MLU memory that stores the input indices tensor. The indices'
 * shape (B, N, 3), B is batch size, N is the number of elements in one output channel.
 * @param[in] weights_desc
 * The descriptor of the weights tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] weights
 * Pointer to the MLU memory that stores the input weights tensor. The weights'
 * shape (B, N, 3), B is batch size, N is the number of elements in one output channel.
 * @param[in] output_desc
 * The descriptor of the output tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output features tensor. The
 * output's shape (B, C, N), B is batch size, C is channel size, N is number
 * of elements in one output channel.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - Data types of features tensor, weights tensor and output tensor should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - features tensor: half, float
 *   - indices tensor: int
 *   - weights tensor: half, float
 *   - output tensor: half, float
 *
 *  @par Data Layout
 *  - The supported data layouts of \b features, \b indices, \b weights, \b output are
 *    as follows:
 *
 *   - features tensor: \p MLUOP_LAYOUT_ARRAY
 *   - indices tensor: \p MLUOP_LAYOUT_ARRAY
 *   - weights tensor: \p MLUOP_LAYOUT_ARRAY
 *   - output tensor: \p MLUOP_LAYOUT_ARRAY
 *
 *  @par Scale Limitation
 *  - The dimension of \b features, \b indices, \b weights and \b output
 *    should be equal to 3.
 *  - The shape[0] of \b features, \b indices, \b weights and \b output
 *    should be the same.
 *  - The shape[1] of \b features and \b output should be the same.
 *  - The shape[1] of \b indices, \b weights and the shape[2] of \b output
 *    should be the same.
 *  - The shape[2] of \b indices and \b weights should be equal to 3.
 *
 * @par Requirements
 * - None.
 *
 *  @par Note
 *  - The value of \b indices must be in the range of [0, M-1], otherwise the output result
 *    is meaningless and the corresponding output will be set to 0.
 *  - In MLU200 series, the maximum value in the \b indices should be less than
 *    2^23, otherwise the output result is not guaranteed to be correct.
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
 *  inputs \b grad_output , \b indices and \b weights to perform the backpropagation
 *  of the ::mluOpThreeInterpolateForward operation.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the three_interpolate_forward operation. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in] grad_output_desc
 * The descriptor of the grad_output tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] grad_output
 * Pointer to the MLU memory that stores the gradients of output tensor. The grad_output's
 * shape (B, C, N), B is batch size, C is channel size, N is the number of
 * elements in one output channel.
 * @param[in] indices_desc
 * The descriptor of the indices tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] indices
 * Pointer to the MLU memory that stores the input indices tensor. The indices'
 * shape (B, N, 3), B is batch size, N is the number of elements in one output channel.
 * @param[in] weights_desc
 * The descriptor of the weights tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] weights
 * Pointer to the MLU memory that stores the input weights tensor. The weights'
 * shape (B, N, 3), B is batch size, N is the number of elements in one output channel.
 * @param[in] grad_features_desc
 * The descriptor of the grad_features tensors. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] grad_features
 * Pointer to the MLU memory that stores the gradients of features tensor. The
 * grad_features' shape (B, C, M), B is batch size, C is channel size, M is number
 * of elements in one input channel.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - Data types of grad_output tensor, weights tensor and grad_features tensor should be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - grad_output tensor: half, float.
 *   - indices tensor: int.
 *   - weights tensor: half, float.
 *   - grad_features tensor: half, float.
 *
 *  @par Data Layout
 *  - The supported data layouts of \b grad_output, \b indices, \b weights, \b grad_features are
 *    as follows:
 *
 *   - grad_output tensor: \p MLUOP_LAYOUT_ARRAY
 *   - indices tensor: \p MLUOP_LAYOUT_ARRAY
 *   - weights tensor: \p MLUOP_LAYOUT_ARRAY
 *   - grad_features tensor: \p MLUOP_LAYOUT_ARRAY
 *
 *  @par Scale Limitation
 *  - The dimension of \b grad_output should be equal to 3.
 *  - The dimension of \b indices should be equal to 3.
 *  - The dimension of \b weights should be equal to 3.
 *  - The dimension of \b grad_features should be equal to 3.
 *
 * @par Requirements
 * - None.
 *
 *  @par Note
 *  - The value of \b indices must be in the range of [0, M-1], otherwise the output result
 *    is meaningless and the corresponding output will be set to 0.
 *  - In MLU270 and MLU290, the maximum value in the \b indices should be less than
 *    2^23, otherwise the output result is not guaranteed to be correct.
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

// Group:Ballquery
/*!
 * @brief Takes the point's index in the \b new_xyz set as the center of the sphere,
 * uses \b min_radius and \b max_radius as the radius, and returns the \b idx of
 * the first \n nsample points in the \b xyz set in the spherical domain.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the sqrt backward operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] new_xyz_desc
 * The descriptor of the new_xyz tensors, which indicates the center of the ball.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] new_xyz
 * Pointer to the MLU memory that stores the new_xyz tensor.
 * The shape of new_xyz is [B, M, 3]. B: the batch size; M: the number of elements in a batch;
 * 3: the dimension of the 3D coordinate.
 * @param[in] xyz_desc
 * The descriptor of the xyz tensors, which means cloud points. For detailed information,
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
 * The descriptor of the idx tensors, which contains output indexes. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] idx
 * Pointer to the MLU memory that stores the xyz tensor.
 * The shape of idx is [B, M, nsample]. B: the batch size; M: the number of elements in a batch;
 * nsample: the number of points selected in the sphere.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - The data types of new_xyz and xyz must be the same. The supported data types of new_xyz
 *   tensor \b new_xyz, xyz tensor \b xyz and idx tensor \b idx are as follows:
 *   - new_xyz tensor: float or half
 *   - xyz tensor: float or half
 *   - idx tensor: int
 *
 *  @par Data Layout
 *  - The supported data layouts of \b new_xyz, \b xyz, \b idx are
 *    as follows:
 *
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
 * @note
 * - Take the point in new_xyz as the center of the sphere, there may be no points in xyz within the
 *   sphere with min_radius and max_radius as diameters. At this time, the value of the
 *   corresponding position in idx is the value when it is passed into the kernel. Generally, before
 *   passing idx into the kernel, initialize all the values in idx to 0 or other const values.
 *
 * @par Requirements
 * - None.
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

// Group:Copy
/*!
 * @brief Returns a copy of input tensor \b input in the output tensor \b output
 * on MLU device.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices
 * and queues in the copy operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] input_desc
 * The descriptor of the input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] output_desc
 * The descriptor of the output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par Data Type
 * - Data types of input tensor \b input and output tensor \b output must be the
 *   same. The supported data types are as follows:
 *   - input tensor: uint8, int8, uint16, int16, uint32, int32, uint64, int64,
 *     bool, half, float, double, complex_half, complex_float.
 *   - output tensor: uint8, int8, uint16, int16, uint32, int32, uint64, int64,
 *     bool, half, float, double, complex_half, complex_float.
 *
 * @note
 * - You can specify the stride of all dimensions for input_desc and output_desc
 *   with ::mluOpSetTensorDescriptorEx.
 *
 * @par Requirements
 * - Data type of input tensor and output tensor must be the same.
 * - Data layout of input tensor and output tensor must be the same.
 * - The shape of input tensor and output tensor must be the same.
 *
 * @par Scale Limitation
 * - When the input or output tensor is non-contiguous, for example with non-contiguous
 *   strides set in the tensor descriptor, the total number of bytes spanned by
 *   either of the input or output tensor should be less than or equal to
 *   \f$2^{23}-1\f$ (the maximum value for int32).
 *
 * @par Example
 * - The example of the copy operation is as follows:
 *   @verbatim
 *    input array by 2 * 2
 *    --> then: [[1, 8], [6, 4]]
 *
 *    output array by 2 * 2
 *    --> output: [[1, 8], [6, 4]]
 *   @endverbatim
 *
 * @par Reference
 * - https://www.tensorflow.org/api_docs/python/tf/raw_ops/Snapshot
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCopy(mluOpHandle_t handle,
          const mluOpTensorDescriptor_t input_desc,
          const void *input,
          const mluOpTensorDescriptor_t output_desc,
          void *output);

// Group:Expand
/*!
 * @brief Copies and expands the input tensor \b input to the shape of output
 * tensor \b output.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices
 * and queues in the expand operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] input_desc
 * The descriptor of the input tensor. For detailed information,
 * see::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] output_desc
 * The descriptor of the output tensor. For detailed information,
 * see::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par Data Type
 * - This function supports the following data types for input tensor \b input
 *   and output tensor \b output.
 *   Data type of both tensors should be the same.
 *   - input tensor: uint8, int8, uint16, int16, uint32, int32, uint64, int64,
 *     bool, half, float, complex_half, complex_float
 *   - output tensor: uint8, int8, uint16, int16, uint32, int32, uint64, int64,
 *     bool, half, float, complex_half, complex_float
 *
 * @par Data Layout
 * - None.
 *
 * @par Scale Limitation
 * - The input tensor and output tensor must meet the following requirements:
 *   - Every dimension of the input tensor should be divisible by the same
 *     dimension of the output tensor.
 *
 * @note
 * - The input tensor \b input and output tensor \b output are multi-dimensional
 *   array, supporting up to \p MLUOP_DIM_MAX dimensions.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of the expand operation is as follows:
 *   @verbatim
 *   input one array by 2 * 2 --> input: [[1, 2], [3, 4]]
 *   output array by 3 * 2 * 2 --> output: [[[1, 2], [3, 4]],
 *                                          [[1, 2], [3, 4]],
 *                                          [[1, 2], [3, 4]]]
 *   @endverbatim
 *
 * @par Reference
 * - https://pytorch.org/docs/stable/tensors.html#torch.Tensor.expand
 */
mluOpStatus_t MLUOP_WIN_API
mluOpExpand(mluOpHandle_t handle,
            const mluOpTensorDescriptor_t input_desc,
            const void *input,
            const mluOpTensorDescriptor_t output_desc,
            void *output);

// Group:Fill
/*!
 * @brief Fills the output tensor \b output with a scale \b value.
 *
 * @par Deprecated
 * - ::mluOpFill is deprecated and will be removed in the future release. It is recommended
 *   to use ::mluOpFill_v3 instead, which supports the parameter \b pointer_mode that sets \b value
 *   to host pointer or device pointer.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the fill
 * operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] value
 * A scale value to fill the output tensor.
 * @param[in] output_desc
 * The descriptor of the output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par Data Type
 * - This function supports the following data types for output tensor \b output.
 *   - output tensor: uint8, int8, uint16, int16, uint32, int32, uint64, int64, bool, half, float.
 *
 * @note
 * - You can specify the stride of all dimensions for \b output_desc with
 *   ::mluOpSetTensorDescriptorEx.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of the fill operation is as follows:
 *   @verbatim
 *    param:value: 5
 *
 *    output array by 2 * 3 * 2 --> output: [[[5,5],[5,5],[5,5]],
 *                                           [[5,5],[5,5],[5,5]]]
 *   @endverbatim
 *
 * @par Reference
 * - http://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Fill.cpp
 */
mluOpStatus_t MLUOP_WIN_API
mluOpFill(mluOpHandle_t handle, float value, const mluOpTensorDescriptor_t output_desc, void *output);

// Group:Fill
/*!
 * @brief Fills the output tensor \b output with the value in \b value tensor.
 *
 * @par Deprecated
 * - ::mluOpFill_v2 is deprecated and will be removed in the future release. It is recommended
 *   to use ::mluOpFill_v3 instead, which supports the parameter \b pointer_mode that sets \b value
 *   to host pointer or device pointer.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the fill
 * operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] value_desc
 * The descriptor of the \b value tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] value
 * Pointer to the MLU memory that stores the \b value tensor.
 * @param[in] output_desc
 * The descriptor of the output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in,out] output
 * Pointer to the MLU memory that stores the output tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - This function supports the following data types for value tensor \b value and output tensor \b output.
 *   - value tensor: uint8, int8, uint16, int16, uint32, int32, uint64, int64, bool, half, float
 *   - output tensor: uint8, int8, uint16, int16, uint32, int32, uint64, int64, bool, half, float
 *
 * @note
 * - Data types of value tensor \b value and output tensor \b output should be the same.
 * - The number of elements of value tensor \b value only supports one.
 * - You can specify the stride of all dimensions for \b output_desc with
 *   ::mluOpSetTensorDescriptorEx.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of the fill operation is as follows:
 *   @verbatim
 *    input array by 1 --> value: [1]
 *
 *    output array by 2 * 3 * 2 --> output: [[[5,5],[5,5],[5,5]],
 *                                           [[5,5],[5,5],[5,5]]]
 *   @endverbatim
 *
 * @par Reference
 * - http://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Fill.cpp
 */
mluOpStatus_t MLUOP_WIN_API
mluOpFill_v2(mluOpHandle_t handle,
             const mluOpTensorDescriptor_t value_desc,
             const void *value,
             const mluOpTensorDescriptor_t output_desc,
             void *output);

// Group:Fill
/*!
 * @brief Fills the output tensor \b output with \b value.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues
 * in the fill operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] pointer_mode
 * An enum value which indicates that the scalar value \b value is passed
 * by reference on the host or device. The information is defined in
 * ::mluOpPointerMode_t.
 * @param[in] value
 * Pointer to scaling factor of tensor input.
 * If the \b pointer_mode is \b MLUOP_POINTER_MODE_DEVICE, the \b value should
 * be a device pointer.
 * If the \b pointer_mode is \b MLUOP_POINTER_MODE_HOST, the \b value should
 * be a host pointer.
 * @param[in] output_desc
 * The descriptor of the output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - This function supports the following data types for \b value and output
 *   tensor \b output.
 *   - value: uint8, int8, uint16, int16, uint32, int32, uint64, int64, bool,
 *     half, float
 *   - output tensor: uint8, int8, uint16, int16, uint32, int32, uint64, int64,
 *     bool, half, float
 *
 * @note
 * - Data types of \b value and output tensor \b output should be the same.
 * - The number of elements of \b value can only be one.
 * - You can specify the stride of all dimensions for \b output_desc with
 *   ::mluOpSetTensorDescriptorEx.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of the fill operation is as follows:
 *   @verbatim
 *    param:value: 5
 *    output array by 2 * 3 * 2 --> output: [[[5,5],[5,5],[5,5]],
 *                                           [[5,5],[5,5],[5,5]]]
 *   @endverbatim
 *
 * @par Reference
 * - https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Fill.cpp
 */
mluOpStatus_t MLUOP_WIN_API
mluOpFill_v3(mluOpHandle_t handle,
             const mluOpPointerMode_t pointer_mode,
             const void *value,
             const mluOpTensorDescriptor_t output_desc,
             void *output);

// Group:RoiawarePool3d
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra
 * workspace to optimize the ::mluOpRoiawarePool3dForward operation.
 *
 * The size of extra workspace is based on the given information of the ::mluOpRoiawarePool3dForward
 * operation, including the input tensor descriptors \b pts_desc.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the
 * ::mluOpRoiawarePool3dForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] rois_desc
 * The descriptor of rois, which contains the dimension and layout of the rois tensor.
 * @param[in] pts_desc
 * The descriptor of pts, which contains the dimension and layout of the pts tensor.
 * @param[in] pts_feature_desc
 * The descriptor of pts, which contains the dimension and layout of the pts tensor.
 * @param[out] workspace_size
 * Pointer to the returned size of the extra workspace in bytes that is used in the
 * ::mluOpRoiawarePool3dForward operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Scale Limitation
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
mluOpGetRoiawarePool3dForwardWorkspaceSize(mluOpHandle_t handle,
                                           const mluOpTensorDescriptor_t rois_desc,
                                           const mluOpTensorDescriptor_t pts_desc,
                                           const mluOpTensorDescriptor_t pts_feature_desc,
                                           size_t *workspace_size);

// Group:RoiawarePool3d
/*!
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in
 * ::mluOpRoiawarePool3dForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] pool_method
 * Pooling method of Roiaware, 0 is 'maxpool', 1 is 'avgpool'. The default value is 0.
 * @param[in] boxes_num
 * An integer value which is the number of the rois.
 * @param[in] pts_num
 * An integer value which is the number of the pts.
 * @param[in] channels
 * An integer value which is the number of the pts feature of channels.
 * @param[in] rois_desc
 * The descriptor of rois, which contains the dimension and layout of the rois tensor.
 * @param[in] rois
 * Pointer to the MLU memory that stores the rois tensor.
 * @param[in] pts_desc
 * The descriptor of pts, which contains the dimension and layout of the pts tensor.
 * @param[in] pts
 * Pointer to the MLU memory that stores the pts tensor.
 * @param[in] pts_feature_desc
 * The descriptor of pts_feature, which contains the dimension and layout of the pts_feature tensor.
 * @param[in] pts_feature
 * Pointer to the MLU memory that stores the pts_feature tensor.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the
 * ::mluOpRoiawarePool3dForward operation.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in
 * the ::mluOpRoiawarePool3dForward operation. You can get the size of the workspace with
 * the ::mluOpGetRoiawarePool3dForwardWorkspaceSize function.
 * @param[in] max_pts_each_voxel
 * The maximum number of points per each voxel. An integer value which is the dimension of the pts_idx_of_voxels.
 * @param[in] out_x
 * An integer value which is the dimension of the pooled_features.
 * @param[in] out_y
 * An integer value which is the dimension of the pooled_features.
 * @param[in] out_z
 * An integer value which is the dimension of the pooled_features.
 * @param[in] argmax_desc
 * The descriptor of argmax, which contains the dimension and layout of the argmax tensor.
 * @param[out] argmax
 * Pointer to the MLU memory that stores the argmax tensor.
 * @param[in] pts_idx_of_voxels_desc
 * The descriptor of pts_idx_of_voxels, which contains the dimension and layout of the pts_idx_of_voxels tensor.
 * @param[out] pts_idx_of_voxels
 * Pointer to the MLU memory that stores the pts_idx_of_voxels tensor.
 * @param[in] pooled_features_desc
 * The descriptor of pooled_features, which contains the dimension and layout of the pooled_features tensor.
 * @param[out] pooled_features
 * Pointer to the MLU memory that stores the pooled_features tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - This function supports the following data types for input tensor \b rois , \b pts , \b pts_feature
 *   and output tensor \b argmax , \b pts_idx_of_voxels , \b pooled_features .
 *   - rois tensor: half, float.
 *   - pts tensor: half, float.
 *   - pts_feature tensor: half, float.
 *   - argmax tensor: int32.
 *   - pts_idx_of_voxels tensor: int32.
 *   - pooled_features tensor: half, float.
 *
 * @par Scale Limitation
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
 * @note
 * - The inputs \b rois and \b pts with NaN or infinity are not supported currently.
 * - The inputs \b pts_feature with NaN are not supported on MLU300 series.
 * - The operation does not support MLU200 series.
 *
 * @par Requirements
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

// Group:RoiawarePool3d
/*!
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in
 * ::mluOpRoiawarePool3dBackward operation. For detailed information, see ::mluOpHandle_t.
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
 * The descriptor of pts_idx_of_voxels, which contains the dimension and layout of the pts_idx_of_voxels tensor.
 * @param[out] pts_idx_of_voxels
 * Pointer to the MLU memory that stores the pts_idx_of_voxels tensor.
 * @param[in] argmax_desc
 * The descriptor of argmax, which contains the dimension and layout of the argmax tensor.
 * @param[out] argmax
 * Pointer to the MLU memory that stores the argmax tensor.
 * @param[in] grad_out_desc
 * The descriptor of grad_out, which contains the dimension and layout of the grad_out tensor.
 * @param[out] grad_out
 * Pointer to the MLU memory that stores the grad_out tensor.
 * @param[in] grad_in_desc
 * The descriptor of grad_in, which contains the dimension and layout of the grad_in tensor.
 * @param[in] grad_in
 * Pointer to the MLU memory that stores the grad_in tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *   ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - This function supports the following data types for input tensor \b pts_idx_of_voxels , \b argmax , \b grad_out
 *   and output tensor \b grad_in .
 *   - pts_idx_of_voxels tensor: int32.
 *   - argmax tensor: int32.
 *   - grad_out tensor: half, float.
 *   - grad_in tensor: half, float.
 *
 * @par Scale Limitation
 * - The shape of \b pts_idx_of_voxels should be [boxes_num, out_x, out_y, out_z, max_pts_each_voxel].
 * - The shape of \b argmax should be [boxes_num, out_x, out_y, out_z, channels].
 * - The shape of \b grad_out should be [boxes_num, out_x, out_y, out_z, channels].
 * - The shape of \b grad_in should be [pts_num, channels].
 *
 * @par API Dependency
 * - None.
 *
 * @note
 * - The operation does not support MLU200 series.
 *
 * @par Requirements
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

// Group:Psamask
/*!
 * @brief Moves the \b x tensor to \b y tensor according to \b h_mask,
 * \b w_mask and \b psa_type.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the ::mluOpPsamaskForward. For detailed information, see ::mluOpHandle_t.
 * @param[in] psa_type
 * Type of the psamask computation, including COLLECT and DISTRIBUTE.
 * @param[in] x_desc
 * The descriptor of data of input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the data of input tensor.
 * @param[in] h_mask
 * An integer value which is the h_mask factor of the psamask.
 * @param[in] w_mask
 * An integer value which is the w_mask factor of the psamask.
 * @param[in] y_desc
 * The descriptor of data of output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] y
 * Pointer to the MLU memory that stores the data of output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Formula
 * - See "Psamask Operation" section in "Cambricon BANGC OPS User Guide" for details.
 *
 * @par Data Type
 * - The supported data types of input tensor \b x and output tensor \b y are as follows:
 *   - x: float
 *   - y: float
 *
 * @par Data Layout
 * - The supported data layouts of input tensor \b x and output tensor \b y are as follows
 *   - x: NHWC
 *   - y: NHWC
 *
 * @par Scale Limitation
 * - The shape of \b x must be [N, H, W, C].
 * - The shape of \b y must be [N, H, W, C].
 * - All dimension size of \b x and \b y must be the same, except the C dimension.
 * - If the shape of \b x is set to [N, H, W, C], the size of C dimension should be \b h_mask * \b
 * w_mask .
 * - If the shape of \b y is set to [N, H, W, C], the size of C dimension should be H * W.
 * - On MLU200 series:
 *   - When psa_type is COLLECT, the size of \b x channels ci and \b y channels co should be
 * satisfied: ci + co <= 6144.
 *   - When psa_type is DISTRIBUTE, the size of \b x channels ci and \b y channels co should be
 * satisfied: ci + 2 * co <= 6144.
 * - On MLU300 series:
 *   - When psa_type is COLLECT, the size of \b x channels ci and \b y channels co should be
 * satisfied: ci + co <= 10240.
 *   - When psa_type is DISTRIBUTE, the size of \b x channels ci and \b y channels co should be
 * satisfied: ci + 2 * co <= 10240.
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

// Group:Psamask
/*!
 * @brief Computes the gradients of input tensor \b dx with the gradients of output tensor \b dy
 * according to \b h_mask , \b w_mask and \b psa_type.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the ::mluOpPsamaskBackward. For detailed information, see ::mluOpHandle_t.
 * @param[in] psa_type
 * Type of the psamask computation, including COLLECT and DISTRIBUTE.
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
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Formula
 * - See "Psamask Operation" section in "Cambricon BANGC OPS User Guide" for details.
 *
 * @par Data Type
 * - The supported data types of input tensor \b x and output tensor \b y are as follows
 *   - dy: float
 *   - dx: float
 *
 * @par Data Layout
 * - The supported data layouts of input tensor \b x and output tensor \b y are as follows:
 *   - dy: NHWC
 *   - dx: NHWC
 *
 * @par Scale Limitation
 * - The shape of \b dy must be [N, H, W, C].
 * - The shape of \b dx must be [N, H, W, C].
 * - All dimension size of \b dy and \b dx must be the same, except the C dimension.
 * - If the shape of \b dx is set to [N, H, W, C], the size of C dimension should be \b h_mask * \b
 * w_mask .
 * - If the shape of \b dy is set to [N, H, W, C], the size of C dimension should be H * W.
 * - On MLU200 series:
 *   - When psa_type is COLLECT, the size of \b dx channels ci and \b dy channels co should be
 *     satisfied: ci + co <= 6144.
 *   - When psa_type is DISTRIBUTE, the size of \b dx channels ci and \b dy channels co should be
 *     satisfied: ci + 2 * co <= 6144.
 * - On MLU300 series:
 *   - When psa_type is COLLECT, the size of \b dx channels ci and \b dy channels co should be
 * satisfied: ci + co <= 10240.
 *   - When psa_type is DISTRIBUTE, the size of \b dx channels ci and \b dy channels co should be
 *     satisfied: ci + 2 * co <= 10240.
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

// Group:MatMul
/*!
 * @brief Computes the matrix multiplication operation, then returns the results in the output
 * tensor \b c.
 *
 * @par Deprecated
 * - ::mluOpMatMul is deprecated and will be removed in the future release.
 *   Please use ::mluOpMatMul_v2 instead.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the
 * matrix multiplication operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] is_trans_a
 * Boolean value indicating whether \b a matrix is transposed.
 * @param[in] is_trans_b
 * Boolean value indicating whether \b b matrix is transposed.
 * @param[in] alpha
 * Host pointer to scaling factor of tensor \b a, the default value is 1.0.
 * @param[in] a_desc
 * The descriptor of the input tensor of left matrix. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] a
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] b_desc
 * The descriptor of the input tensor of right matrix. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] b
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] beta
 * Host pointer to scaling factor of tensor \b c, the default value is 0.0.
 * @param[in] c_desc
 * The descriptor of the output tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] c
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH
 *
 * @par Data Type
 * - On all hardware platforms, this function supports any combinations of the following data types for
 *   input tensor \b a, \b b and output tensor \b c.
 *   - \b a data type: int8, int16
 *   - \b b data type: int8, int16
 *   - \b c offchip data type: half, float
 *   - \b c onchip data type: half, float
 * - On MLU300 series or above, this function supports the combinations of the following data types for
 *   input tensor \b a, \b b and output tensor \b c.
 *   - \b a, \b b, \b c offchip data type, \b c onchip data type: half, half, half, half
 *   - \b a, \b b, \b c offchip data type, \b c onchip data type: half, half, half, float
 *   - \b a, \b b, \b c offchip data type, \b c onchip data type: float, float, float, float
 *
 * @note
 * - On all hardware platforms, the combinations of the data types should satisfy the following rules:
 *   - The data type bitwidth of \b c onchip data type for operation computing is not shorter than \b c
 *     offchip data type.
 *
 * @par Scale Limitation
 * - The input tensors and output tensor must meet the following requirements:
 *   - The \b a and \b b must be a 2D tensor.
 *   - The number of \b a matrix's columns must be equal to the number of \b b matrix's rows after both inputs
 *   perform transpose operations according to parameters.
 *
 * @par API Dependency
 * - Before calling this function to implement matrix multiplication, you need to prepare
 *   all the parameters passed to this function. See each parameter description for details.
 *
 * @par Performance Optimization
 * - For best practices, to have a better performance, matrix \b a does not need to transpose and matrix \b b
 *   needs to transpose.
 *
 * @par Example
 * - The example of the operation is as follows:
 *   @verbatim
 *    is_trans_a:                    false
 *    is_trans_b:                    false
 *    Dimension of input tensor a:  [99, 128]
 *    Dimension of input tensor b:  [128, 256]
 *    Dimension of output tensor c: [99, 256]
 *   @endverbatim
 *
 * @par Reference
 * - https://pytorch.org/docs/stable/torch.html?highlight=matmul#torch.matmul
 */
mluOpStatus_t MLUOP_WIN_API
mluOpMatMul(mluOpHandle_t handle,
            const bool is_trans_a,
            const bool is_trans_b,
            const void *alpha,
            const mluOpTensorDescriptor_t a_desc,
            const void *a,
            const mluOpTensorDescriptor_t b_desc,
            const void *b,
            const void *beta,
            const mluOpTensorDescriptor_t c_desc,
            void *c);

// Group:MatMul
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra workspace
 * to optimize the matrix multiplication operation.
 *
 * The size of extra workspace is based on the given information of the matrix multiplication
 * operation, including the matrix multiplication descriptor \b matmul_desc, input tensor
 * descriptor of left matrix \b a_desc, input tensor descriptor of right matrix \b b_desc, output
 * tensor  descriptor \b c_desc, and the matrix multiplication algorithm \b algo.
 *
 * @par Deprecated
 * - ::mluOpGetMatMulWorkspaceSize is deprecated and will be removed in the future release.
 *   Use ::mluOpGetMatMulHeuristicResult instead.

 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the
 * matrix multiplication operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] matmul_desc
 * The descriptor of the matrix multiplication operations. For detail information,
 * see ::mluOpMatMulDescriptor_t.
 * @param[in] a_desc
 * The descriptor of the input tensor of left matrix. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] b_desc
 * The descriptor of the input tensor of right matrix. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] c_desc
 * The descriptor of the output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] d_desc
 * The descriptor of the output tensor. Needed to be set in out-of-place matrix multiplication.
 * Currently not supported and should be set to NULL.
 * @param[in] algo
 * Host pointer to the most suitable algorithm to compute the matrix multiplication.
 * Currently not supported and should be set to NULL.
 * @param[out] workspace_size
 * Pointer to the MLU memory that stores the returned size of the extra workspace in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par API Dependency
 * - This function must be called after ::mluOpSetMatMulDescAttr function. You also need to
 *   call the ::mluOpCreateTensorDescriptor and ::mluOpSetTensorDescriptor functions to create and set
 *   tensor descriptors \b a_desc, \b b_desc, \b c_desc before calling this function.
 * - The allocated extra workspace should be passed to the ::mluOpMatMul_v2 function to
 *   perform the matrix multiplication operation.
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
mluOpGetMatMulWorkspaceSize(mluOpHandle_t handle,
                            mluOpMatMulDescriptor_t matmul_desc,
                            mluOpTensorDescriptor_t a_desc,
                            mluOpTensorDescriptor_t b_desc,
                            mluOpTensorDescriptor_t c_desc,
                            mluOpTensorDescriptor_t d_desc,
                            mluOpMatMulAlgo_t algo,
                            size_t *workspace_size);

// Group:MatMul
/*!
 * @brief Computes the matrix multiplication operation, then returns the results in the output
 * tensor \b d.
 *
 * Compared with ::mluOpMatMul, it supports the use of extra workspace size, the use of \b algo
 * to pass the algorithm information and the use of \b matmul_desc to pass parameters
 * like ::MLUOP_MATMUL_DESC_TRANSA.
 *
 * This function needs extra MLU memory as the workspace to improve the matrix multiplication
 * performance. You can get the size of the workspace \b workspace_size with the
 * ::mluOpGetMatMulAlgoHeuristic and ::mluOpGetMatMulHeuristicResult functions in turn.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the
 * matrix multiplication operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] matmul_desc
 * The descriptor of the matrix multiplication operations. For detail information,
 * see ::mluOpMatMulDescriptor_t.
 * @param[in] algo
 * Host pointer to the most suitable algorithm to compute the matrix multiplication.
 * @param[in] alpha
 * Host pointer to scaling factor of tensor \b a, the default value is 1.0.
 * @param[in] a_desc
 * The descriptor of the input tensor of left matrix. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] a
 * Pointer to the MLU memory that stores the input tensor of the left matrix.
 * @param[in] b_desc
 * The descriptor of the input tensor of right matrix. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] b
 * Pointer to the MLU memory that stores the input tensor of the right matrix.
 * @param[in] beta
 * Host pointer to scaling factor of tensor \b c, the default value is 0.0.
 * @param[in] c_desc
 * The descriptor of the input tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] c
 * Pointer to the MLU memory that stores the input tensor \b c in out-of-place matrix multiplication
 * where d = alpha * a * b + beta * c, or pointer to the MLU memory that stores the output tensor \b d in in-place
 * matrix multiplication where c == d = alpha * a * b + beta * c.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the matrix multiplication
 * operation.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in the matrix multiplication
 * operation. You can get the size of the workspace with the ::mluOpGetMatMulAlgoHeuristic and
 * ::mluOpGetMatMulHeuristicResult functions in turn.
 * @param[in] d_desc
 * The descriptor of the output tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] d
 * Pointer to the MLU memory that stores the output \b d.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par Data Type
 * - On all hardware platforms, this function supports any combinations of the following data types for
 *   input tensor \b a, \b b and output tensor \b d.
 *   - \b a data type: int8, int16
 *   - \b b data type: int8, int16
 *   - \b d offchip data type: half, float
 *   - \b d onchip data type: half, float
 * - On MLU300 series or above, this function supports the combinations of the following data types for
 *   input tensor \b a, \b b and output tensor \b d:
 *   - \b a, \b b, \b d offchip data type, \b d onchip data type: half, half, half, half
 *   - \b a, \b b, \b d offchip data type, \b d onchip data type: half, half, half, float
 *   - \b a, \b b, \b d offchip data type, \b d onchip data type: float, float, float, float
 *
 * @note
 * - The value of \b c_desc is the same as that of \b d_desc.
 * - On all hardware platforms, the combinations of the data types should satisfy the following rules:
 *   - The data type bitwidth of \b d onchip data type for operation computing is not shorter than \b d
 *     offchip data type.
 *
 * @par Scale Limitation
 * - The input tensors and output tensor must meet the following requirements:
 *   - The \b a and \b b must be a 2D tensor.
 *   - The number of \b a matrix's columns must be equal to the number of \b b matrix's rows after both inputs
 *   perform transpose operations according to parameters.
 *   - The product of the max size for \b a dimension and the size of \b a data type should be less than 2G.
 *   - The product of the max size for \b b dimension and the size of \b b data type should be less than 2G.
 *   - The product of the max size for \b d dimension and the size of \b d data type should be less than 2G.
 *
 * @par API Dependency
 * - Before calling this function to implement matrix multiplication operation, you need to prepare
 *   all the parameters passed to this function. See each parameter description for details.
 *
 * @par Performance Optimization
 * - For best practices, to have a better performance, matrix \b a should not be transposed and matrix \b b
 *   should be transposed.
 *
 * @par Example
 * - The example of the operation is as follows:
 *   @verbatim
 *    MLUOP_MATMUL_DESC_TRANSA:      false
 *    MLUOP_MATMUL_DESC_TRANSB:      false
 *    MLUOP_MATMUL_USE_BETA:         false
 *    Dimension of input tensor a:  [99, 128]
 *    Dimension of input tensor b:  [128, 256]
 *    Dimension of input tensor c:  [99, 256]
 *    Dimension of output tensor d: [99, 256]
 *   @endverbatim
 *
 * @par Reference
 * - https://pytorch.org/docs/stable/torch.html?highlight=matmul#torch.matmul
 */
mluOpStatus_t MLUOP_WIN_API
mluOpMatMul_v2(mluOpHandle_t handle,
               mluOpMatMulDescriptor_t matmul_desc,
               mluOpMatMulAlgo_t algo,
               const void *alpha,
               const mluOpTensorDescriptor_t a_desc,
               const void *a,
               const mluOpTensorDescriptor_t b_desc,
               const void *b,
               const void *beta,
               const mluOpTensorDescriptor_t c_desc,
               void *c,
               void *workspace,
               size_t workspace_size,
               const mluOpTensorDescriptor_t d_desc,
               void *d);

// Group:MatMul
/*!
 * @brief Creates a descriptor pointed by \b result for a matrix multiplication heuristic result,
 * and allocates memory for the result. The result is defined in ::mluOpMatMulHeuristicResult_t.
 *
 * @param[out] result
 * Host pointer to the struct of matrix multiplication heuristic result.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ALLOC_FAILED
 *
 * @par API Dependency
 * - You need to call the ::mluOpDestroyMatMulHeuristicResult function to destroy the descriptor.
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
mluOpStatus_t
mluOpCreateMatMulHeuristicResult(mluOpMatMulHeuristicResult_t *result);

// Group:MatMul
/*!
 * @brief Destroys a matrix multiplication heuristic result, that is previously created with
 * the ::mluOpCreateMatMulHeuristicResult.
 *
 * @param[in] result
 * The matrix multiplication heuristic result to be destroyed.
 *
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
mluOpStatus_t
mluOpDestroyMatMulHeuristicResult(mluOpMatMulHeuristicResult_t result);

// Group:MatMul
/*!
 * @brief Gets the matrix multiplication algorithm and workspace size from heuristic result,
 * that is previously selected with ::mluOpGetMatMulAlgoHeuristic.
 *
 * @param[in] result
 * The matrix multiplication heuristic result obtained by ::mluOpGetMatMulAlgoHeuristic.
 *
 * @param[out] algo
 * The matrix multiplication algorithm.
 *
 * @param[out] workspace_size
 * Pointer to the returned size of the extra workspace in bytes that is used in
 * the matrix multiplication operation.
 *
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
mluOpStatus_t
mluOpGetMatMulHeuristicResult(mluOpMatMulHeuristicResult_t result, mluOpMatMulAlgo_t algo, size_t *workspace_size);

// Group:MatMul
/*!
 * @brief Retrieves the possible algorithms can be used in the matrix multiplication.
 * The output is placed in result_array[] in the order of increasing estimated compute time.
 *
 * @param[in] matmul_desc
 * The descriptor of the matrix multiplication operations. For detail information,
 * see ::mluOpMatMulDescriptor_t.
 * @param[in] a_desc
 * The descriptor of the input tensor of left matrix. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] b_desc
 * The descriptor of the input tensor of right matrix. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] c_desc
 * The descriptor of the output tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] d_desc
 * The descriptor of the output tensor used in out-of-place matrix multiplication. For detailed
 * information, see ::mluOpTensorDescriptor_t.
 * Not supported currently and should be set to NULL.
 * @param[in] preference
 * The descriptor of the matrix multiplication that holds the preferences for ::mluOpMatMulHeuristicResult_t
 * configuration. Currently not supported and should be set to NULL.
 * @param[in] requested_algo_count
 * The number of requested algorithms. The maximum number of algorithms to be returned.
 * Currently this value only supports 1.
 * @param[out] result_array
 * Array containing the algorithm heuristics and associated runtime characteristics, returned by this function,
 * in the order of increasing estimated compute time.
 * @param[out] return_algo_count
 * Host pointer to the number of algorithms returned by this function.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @note
 * - Currently the maximum number of algorithms \b requested_algo_count only supports 1.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
mluOpStatus_t
mluOpGetMatMulAlgoHeuristic(mluOpHandle_t handle,
                            mluOpMatMulDescriptor_t matmul_desc,
                            mluOpTensorDescriptor_t a_desc,
                            mluOpTensorDescriptor_t b_desc,
                            mluOpTensorDescriptor_t c_desc,
                            mluOpTensorDescriptor_t d_desc,
                            mluOpMatMulPrefer_t preference,
                            int requested_algo_count,
                            mluOpMatMulHeuristicResult_t result_array[],
                            int *return_algo_count);

// Group:MatMul
/*!
 * @brief Creates a descriptor pointed by \b matmul_desc for a matrix multiplication operation,
 * and allocates memory for holding the information about the matrix multiplication operation.
 * The information is defined in ::mluOpMatMulDescriptor_t.
 *
 * @param[out] matmul_desc
 * The descriptor of matrix multiplication operation that holds information about matrix
 * multiplication operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ALLOC_FAILED
 *
 * @par API Dependency
 * - After calling this function, you can call the ::mluOpSetMatMulDescAttr function to initialize
 *   and set the information to the matrix multiplication descriptor.
 * - You need to call the ::mluOpMatMulDescDestroy function to destroy the descriptor.
 *
 * @note
 * - The default compute data type of c is c_desc->dtype, use ::mluOpSetTensorDescriptorOnchipDataType to
 *   set onchip data type if high accuracy of c is needed.
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
mluOpMatMulDescCreate(mluOpMatMulDescriptor_t *matmul_desc);

// Group:MatMul
/*!
 * @brief Destroys a matrix multiplication descriptor \b matmul_desc
 * that is previously created with the ::mluOpMatMulDescCreate.
 *
 * The matrix multiplication descriptor is defined in ::mluOpMatMulDescriptor_t
 * and holds the information about the matrix multiplication operation.
 *
 * @param[in] matmul_desc
 * The descriptor of matrix multiplication operation which needs to be destroyed.
 *
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
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpMatMulDescDestroy(mluOpMatMulDescriptor_t matmul_desc);

// Group:MatMul
/*!
 * @brief Initializes the matrix multiplication descriptor \b matmul_desc
 * that is previously created with the ::mluOpMatMulDescCreate function, and sets
 * the information about the matrix multiplication operation to the matrix multiplication
 * descriptor \b matmul_desc. The information includes the attribute defined in
 * ::mluOpMatMulDescAttribute_t \b attr, the host pointer to the attribute value \b buf, and
 * the size of buffer for verification.
 *
 * @param[in] matmul_desc
 * The descriptor of the matrix multiplication operation. For detailed
 * information, see ::mluOpMatMulDescriptor_t.
 * @param[in] attr
 * Attribute of matrix multiplication descriptor to be set. For detailed
 * information, see ::mluOpMatMulDescAttribute_t.
 * @param[out] buf
 * Host pointer to the attribute value set by this function.
 * @param[in] size_in_bytes
 * Buffer in bytes for verification.
 *
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
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetMatMulDescAttr(mluOpMatMulDescriptor_t matmul_desc,
                       mluOpMatMulDescAttribute_t attr,
                       const void *buf,
                       size_t size_in_bytes);

// Group:MatMul
/*!
 * @brief Returns the pointer to the \b buf and size of the buffer \b size_written of the attribute
 * retrieved with the given matmul multiplication descriptor \b matmul_desc, attribute \b attr.
 * And \b size_in_bytes is used to check whether the memory size is same with \b size_written.
 *
 * You can set the attribute in the matrix multiplication descriptor based on the return value
 * of this function.
 *
 * @param[in] matmul_desc
 * The descriptor of the matrix multiplication operation. For detailed
 * information, see ::mluOpMatMulDescriptor_t.
 * @param[in] attr
 * Attribute of matrix multiplication descriptor to be retrieved.
 * @param[out] buf
 * Host pointer to the attribute value to be retrieved by this function.
 * @param[in] size_in_bytes
 * Buffer in bytes for verification.
 * @param[out] size_written
 * Host pointer to the number of bytes actually written to the buffer.
 *
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
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetMatMulDescAttr(const mluOpMatMulDescriptor_t matmul_desc,
                       mluOpMatMulDescAttribute_t attr,
                       void *buf,
                       size_t size_in_bytes,
                       size_t *size_written);

// Group:MatMul
/*!
 * @brief Creates a descriptor pointed by \b algo for a matrix multiplication algorithm,
 * and allocates memory for holding the information about the algorithm.
 * The information is defined in ::mluOpMatMulAlgo_t.
 *
 * @param[out] algo
 * Host pointer to the matrix multiplication algorithm that holds information about the matrix
 * multiplication algorithm.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ALLOC_FAILED
 *
 * @par API Dependency
 * - After calling this function, you can call the ::mluOpGetQuantizeMatMulAlgorithm function to initialize
 *   and set the information to the matrix multiplication algorithm.
 * - You need to call the ::mluOpMatMulAlgoDestroy function to destroy the descriptor.
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
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpMatMulAlgoCreate(mluOpMatMulAlgo_t *algo);

// Group:MatMul
/*!
 * @brief Destroys a matrix multiplication algorithm descriptor \b algo
 * that is previously created with the ::mluOpMatMulAlgoCreate.
 *
 * The matrix multiplication descriptor is defined in ::mluOpMatMulAlgo_t
 * and holds the information about the matrix multiplication algorithm.
 *
 * @param[in] algo
 * The matrix multiplication algorithm descriptor to be destroyed.
 *
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
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpMatMulAlgoDestroy(mluOpMatMulAlgo_t algo);

// Group:Unique
/*!
 * @brief Creates a descriptor pointer by \b unique_desc for a unique operation, and allocates
 * memory for holding the information about the unique operation. The information is
 * defined in ::mluOpUniqueDescriptor_t.
 *
 * @param[in] unique_desc
 * Pointer to the unique descriptor that holds information about the unique operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - After calling this function, you can call the ::mluOpSetUniqueDescriptor function to initialize
 *   and set the information to the descriptor.
 * - You need to call the ::mluOpDestroyUniqueDescriptor function to destroy the descriptor.
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
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateUniqueDescriptor(mluOpUniqueDescriptor_t *unique_desc);

// Group:Unique
/*!
 *  @brief Destroys a unique descriptor \b unique_desc that is previously created with the
 *  ::mluOpCreateUniqueDescriptor function.
 *
 *  The unique descriptor is defined in ::mluOpUniqueDescriptor_t and holds the information
 *  about the unique operation.
 *
 * @param[in] unique_desc
 *   The unique descriptor to be destroyed.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_EXECUTION_FAILED, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @note
 * - You need to call this function after calling the ::mluOpUnique.
 * - This function should be called to destroy the unique descriptor. Otherwise, the memory
 *   leak may occur.
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
mluOpDestroyUniqueDescriptor(mluOpUniqueDescriptor_t unique_desc);

// Group:Unique
/*!
 * @brief Initializes the unique descriptor \b unique_desc that is previously created
 * with the ::mluOpCreateUniqueDescriptor function, and sets the information about the
 * unique operation to the unique descriptor \b unique_desc. The information includes
 * the sorted mode of the unique \b mode, the number of the unique dimensions \b dim,
 * whether to output index \b return_inverse, and whether to output counts \b return_counts.
 *
 * @param[in] unique_desc
 * The descriptor of the unique operation. For detailed information,
 * see ::mluOpUniqueDescriptor_t.
 * @param[in] mode
 * The sorted mode of unique operation. The sorted modes are define in the ::mluOpUniqueSort_t.
 * @param[in] dim
 * The number of dimensions in the input tensor of the unique operation.
 * Currently, only the unique of the flattened input is supported. It is recommended to set
 * the dimension \b dim to -1.
 * @param[in] return_inverse
 * A boolean value that specifies whether to return the index of input elements that
 * are in the returned unique elements.
 * @param[in] return_counts
 * A boolean value that specifies whether to return the number of duplicate values
 * for each unique element.
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
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetUniqueDescriptor(
    mluOpUniqueDescriptor_t unique_desc, mluOpUniqueSort_t mode, int dim, bool return_inverse, bool return_counts);

// Group:Unique
/*!
 * @brief Returns in \b size the size of the MLU memory that is used as an extra workspace
 * to store unique data.
 *
 * @par Deprecated
 * - ::mluOpGetUniqueWorkSpace is deprecated and will be removed in the future release. It is recommended
 *   to use ::mluOpGetUniqueWorkspaceSize instead, which supports better performance to unique.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the unique
 * operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] unique_desc
 * The descriptor of the unique operation. For detailed information, see ::mluOpUniqueDescriptor_t.
 * @param[in] input_desc
 * The descriptor of the input tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] size
 * Pointer to the returned size of the extra workspace in bytes that is used in the unique operation.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - You need to call the ::mluOpCreateTensorDescriptor and ::mluOpSetTensorDescriptor functions
 *   to create and set the tensor descriptor \b input_desc before calling this function, and
 *   call the ::mluOpCreateUniqueDescriptor and ::mluOpSetUniqueDescriptor functions to create
 *   and set the unique descriptor \b unique_desc.
 * - The allocated extra workspace should be passed to the ::mluOpUnique function to perform the
 *   unique operation.
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
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetUniqueWorkSpace(mluOpHandle_t handle,
                        const mluOpUniqueDescriptor_t unique_desc,
                        const mluOpTensorDescriptor_t input_desc,
                        size_t *size);
// Group:Unique
/*!
 * @brief Computes the length of unique data of input tensor, and returns the results
 * in \b output_len.
 *
 * @par Deprecated
 * - ::mluOpUniqueGetOutLen is deprecated and will be removed in the future release. It is recommended
 *   to use ::mluOpUnique_v2 instead, which supports better performance to unique.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the unique
 * operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] unique_desc
 * The descriptor of the unique operation. For detailed information, see ::mluOpUniqueDescriptor_t.
 * @param[in] input_desc
 * The descriptor of the input tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[out] unique_data
 * Pointer to the MLU memory that is used as an extra workspace for the unique operation.
 * @param[out] output_len
 * Pointer to the MLU memory that stores the length of unique data.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH
 *
 * @par API Dependency
 * - You need to call the ::mluOpGetUniqueWorkSpace function to allocate extra workspace for
 *   \b unique_data.
 *
 * @par Data Type
 * - Date types of input tensor \b input and output tensor \b unique_data must be the same.
 * - The supported data types of input tensor \b input and output tensor \b unique_data are as follows:
 *   - input tensor: float, int32
 *   - output tensor: float, int32
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
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpUniqueGetOutLen(mluOpHandle_t handle,
                     const mluOpUniqueDescriptor_t unique_desc,
                     const mluOpTensorDescriptor_t input_desc,
                     const void *input,
                     void *unique_data,
                     int *output_len);
// Group:Unique
/*!
 * @brief Retrieves unique elements in the input tensor.
 *
 * @par Deprecated
 * - ::mluOpUnique is deprecated and will be removed in the future release. It is recommended
 *   to use ::mluOpUnique_v2 instead, which supports better performance to unique.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in
 * the unique operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] unique_desc
 * The descriptor of the unique operation. For detailed information,
 * see ::mluOpUniqueDescriptor_t.
 * @param[in] input_desc
 * The descriptor of the input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] output_len
 * An integer value that is the length of unique data of input tensor.
 * @param[in] unique_data
 * Pointer to the MLU memory that is used as an extra workspace to store unique
 * data.
 * @param[out] output_data
 * Pointer to the MLU memory that stores the output tensor \b output_data.
 * @param[out] output_index
 * Pointer to the MLU memory that stores the index of input elements that are in
 * the returned unique elements \b output_data. This parameter only returns meaningful
 * value when \b return_inverse is set to true. If \b return_inverse is to false, this
 * parameter returns meaningless value. It is recommended to set this parameter to NULL
 * if \b return_inverse is to false.
 * @param[out] output_counts
 * Pointer to the MLU memory that stores the number of duplicate values for each
 * unique element \b output_data. This parameter only returns meaningful value when
 * \b return_counts is set to true. If \b return_counts is to false, this parameter
 * returns meaningless value. It is recommended to set this parameter to NULL if
 * \b return_counts is set to false.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par API Dependency
 * - You need to call the ::mluOpUniqueGetOutLen function to get the length of unique data
 *   of input tensor \b output_len and the unique data \b unique_data.
 *
 * @par Formula
 * - None.
 *
 * @par Data Type
 * - Date types of input tensor \b input and output tensor \b output_data must be the same.
 * - The supported data types of input tensor \b input and output tensors are as follows:
 *   - input tensor: float, int32
 *   - \b output_data: float, int32
 *   - \b output_index: int32
 *   - \b output_counts: int32
 *
 * @par Scale Limitation
 * - The input tensor \b input must meet the following requirement:
 *   - When the \b mode is set to \p MLUOP_UNSORT_FORWARD, the dimension of \b input must be
 *     one-dimensional.
 *
 * @note
 * - The \b input with NaN is not supported currently, and the data range of \b input should
 *   satisfy the following conditions:
 *   - (-inf, +inf), where inf represents infinity.
 * - You need to call the ::mluOpUniqueGetOutLen function to get the scale \b output_len and
 *   the tensor \b unique_data before calling this function.
 * - The tensor \b output_index is same shape as input tensor \b input, and the tensor
 *   \b output_counts is same shape as \b output_data.
 * - When the \b mode is set to \p MLUOP_UNSORT_FORWARD, the output \b output_counts is not
 *   supported yet.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of the unique operation is as follows:
 *   @verbatim
 *     Example 1:
 *     input array:
 *       input: [1, 1, 2, 4, 4, 9, 7, 8, 8]
 *     param:
 *       mode: MLUOP_UNSORT_FORWARD
 *     output array:
 *       output_data: [1, 2, 4, 9, 7, 8]
 *       output_index: [0, 0, 1, 2, 2, 3, 4, 5, 5]
 *     Example 2:
 *     input array:
 *       input: [1, 1, 2, 4, 4, 9, 7, 8, 8]
 *     param:
 *       mode: MLUOP_SORT_ASCEND, return_inverse: true, return_counts: true,
 *     output array:
 *       output_data: [1, 2, 4, 7, 8, 9]
 *       output_index: [0, 0, 1, 2, 2, 5, 3, 4, 4]
 *       output_counts: [2, 1, 2, 1, 2, 1]
 *     Example 3:
 *     input array:
 *       input: [1, 1, 2, 4, 4, 9, 7, 8, 8]
 *     param:
 *       mode: MLUOP_SORT_REVERSE, return_inverse: true, return_counts: true,
 *     output array:
 *       output_data: [8, 7, 9, 4, 2, 1]
 *       output_index: [5, 5, 4, 3, 3, 2, 1, 0, 0]
 *       output_counts: [2, 1, 1, 2, 1, 2]
 *  @endverbatim
 *
 * @par Reference
 * - http://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Unique.cpp
 *
 */
mluOpStatus_t MLUOP_WIN_API
mluOpUnique(mluOpHandle_t handle,
            const mluOpUniqueDescriptor_t unique_desc,
            const mluOpTensorDescriptor_t input_desc,
            const void *input,
            const int output_len,
            void *unique_data,
            void *output_data,
            int *output_index,
            int *output_counts);

// Group:Unique
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra workspace to
 * optimize the unique operation.
 *
 * Compared with ::mluOpGetUniqueWorkSpace, this function has a better performance for unique operation.
 *
 * The size of extra workspace is based on the given information of the unique operation,
 * including the input tensor descriptors \b input_desc, and the unique operation
 * descriptor \b unique_desc.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the unique
 * operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] unique_desc
 * The descriptor of the unique operation. For detailed information,
 * see ::mluOpUniqueDescriptor_t.
 * @param[in] input_desc
 * The descriptor of the input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] workspace_size
 * Pointer to the returned size of the extra workspace in bytes that is used in the
 * unique operation.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - You need to call the ::mluOpCreateTensorDescriptor and ::mluOpSetTensorDescriptor functions
 *   to create and set the tensor descriptor \b input_desc before calling this function, and
 *   call the ::mluOpCreateUniqueDescriptor and ::mluOpSetUniqueDescriptor functions to create
 *   and set the unique descriptor \b unique_desc.
 * - The allocated extra workspace should be passed to the ::mluOpUnique_v2 function to perform the
 *   unique operation.
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
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetUniqueWorkspaceSize(mluOpHandle_t handle,
                            const mluOpUniqueDescriptor_t unique_desc,
                            const mluOpTensorDescriptor_t input_desc,
                            size_t *workspace_size);

// Group:Unique
/*!
 * @brief Retrieves unique elements in the input tensor.
 *
 * Compared with ::mluOpUniqueGetOutLen and ::mluOpUnique, this function has a better performance.
 *
 * This function need extra MLU memory as the workspace to improve the unique
 * performance. You can get the size of the workspace \b workspace_size with the
 * ::mluOpGetUniqueWorkspaceSize function.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in
 * the unique operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] unique_desc
 * The descriptor of the unique operation. For detailed information,
 * see ::mluOpUniqueDescriptor_t.
 * @param[in] input_desc
 * The descriptor of the input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor \b input.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the unique
 * operation.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in the unique
 * operation. You can get the size of the workspace with the
 * ::mluOpGetUniqueWorkspaceSize function.
 * @param[out] output_num
 * Pointer to the MLU memory that stores the number of output unique data.
 * @param[in] output_desc
 * The descriptor of the output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor \b output.
 * @param[in] indices_desc
 * The descriptor of the inverse indices tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] inverse_indices
 * Pointer to the MLU memory that stores the index of input elements that are in
 * the returned unique elements \b output. This parameter only returns meaningful
 * value when \b return_inverse is set to true. If \b return_inverse is to false, this
 * parameter returns meaningless value. It is recommended to set this parameter to NULL
 * if \b return_inverse is to false.
 * @param[in] counts_desc
 * The descriptor of the counts tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] counts
 * Pointer to the MLU memory that stores the number of duplicate values for each
 * unique element \b output. This parameter only returns meaningful value when
 * \b return_counts is set to true. If \b return_counts is to false, this parameter
 * returns meaningless value. It is recommended to set this parameter to NULL if
 * \b return_counts is set to false.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH
 *
 * @par API Dependency
 * - You need to call the ::mluOpGetUniqueWorkspaceSize function to allocate extra
 *   workspace for \b workspace.
 *
 * @par Formula
 * - None.
 *
 * @par Data Type
 * - Date types of input tensor \b input and output tensor \b output must be the same.
 * - The supported data types of input tensor \b input and output tensors are as follows:
 *   - \b input: float, int32
 *   - \b output_num: int32
 *   - \b output: float, int32
 *   - \b inverse_indices: int32
 *   - \b counts: int32
 *
 * @par Scale Limitation
 * - The input tensor \b input must meet the following requirement:
 *   - When the \b mode is set to \p MLUOP_UNSORT_FORWARD, the dimension of \b input must be
 *     one-dimensional.
 * - Currently, the dimension \b dim do not support to apply unique, and the \b output is the unique
 *   of the flattened \b input. It is recommended to set the dimension \b dim to -1.
 *
 * @note
 * - The \b input with NaN is not supported currently, and the data range of \b input should
 *   satisfy the following conditions:
 *   - (-inf, +inf), where inf represents infinity.
 * - The tensor \b inverse_indices is same shape as input tensor \b input, and the tensor
 *   \b counts is same shape as \b output.
 * - When the \b mode is set to \p MLUOP_UNSORT_FORWARD, the output \b counts is not
 *   supported yet.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of the unique operation is as follows:
 *   @verbatim
 *     Example 1:
 *     input array:
 *       input: [1, 1, 2, 4, 4, 9, 7, 8, 8]
 *     param:
 *       mode: MLUOP_UNSORT_FORWARD
 *     output array:
 *       output: [1, 2, 4, 9, 7, 8]
 *       inverse_indices: [0, 0, 1, 2, 2, 3, 4, 5, 5]
 *     Example 2:
 *     input array:
 *       input: [1, 1, 2, 4, 4, 9, 7, 8, 8]
 *     param:
 *       mode: MLUOP_SORT_ASCEND, return_inverse: true, return_counts: true,
 *     output array:
 *       output: [1, 2, 4, 7, 8, 9]
 *       inverse_indices: [0, 0, 1, 2, 2, 5, 3, 4, 4]
 *       counts: [2, 1, 2, 1, 2, 1]
 *     Example 3:
 *     input array:
 *       input: [1, 1, 2, 4, 4, 9, 7, 8, 8]
 *     param:
 *       mode: MLUOP_SORT_REVERSE, return_inverse: true, return_counts: true,
 *     output array:
 *       output: [8, 7, 9, 4, 2, 1]
 *       inverse_indices: [5, 5, 4, 3, 3, 2, 1, 0, 0]
 *       counts: [2, 1, 1, 2, 1, 2]
 *  @endverbatim
 *
 * @par Reference
 * - http://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Unique.cpp
 */
mluOpStatus_t MLUOP_WIN_API
mluOpUnique_v2(mluOpHandle_t handle,
               const mluOpUniqueDescriptor_t unique_desc,
               const mluOpTensorDescriptor_t input_desc,
               const void *input,
               void *workspace,
               const size_t workspace_size,
               int *output_num,
               const mluOpTensorDescriptor_t output_desc,
               void *output,
               const mluOpTensorDescriptor_t indices_desc,
               void *inverse_indices,
               const mluOpTensorDescriptor_t counts_desc,
               void *counts);

// Group:GatherNd
/*!
 * @brief Gathers slices from \b params with shape specified by \b indices.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the gather_nd
 * operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] desc_params
 * The descriptor of the input tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] params
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] desc_indices
 * The descriptor of the index tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] indices
 * Pointer to the MLU memory that stores the index of each element of \b output in the
 * corresponding dimension of input tensor \b params.
 * @param[in] desc_output
 * The descriptor of the output tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par Formula
 * - None.
 *
 * @par Data Type
 * - The (I/O)function supports the following byte-width data types for \b params and \b output tensors.
 *   The byte width of a data type can be got with the ::mluOpGetSizeOfDataType function.
 *   <b>Note that the data type of input tensor \b params and output tensor \b output must be the same.</b>
 *   - params tensor: 1-byte, 2-byte, 4-byte, 8-byte
 *   - index tensor: int32, int64
 *   - output tensor: 1-byte, 2-byte, 4-byte, 8-byte
 *
 * @note
 * - The item in \b indices must be in the range of [-rank, rank), where rank is the element size
 *   of each dimension of \b params. E.g.,params.shape is [3,2], indices' first data item be in
 *   [-3, 3) and second item must be in [-2, 2).
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of the gather_nd operation is as follows:
 *   @verbatim
 *   input two arrays both by 3 * 2 --> params: [[1., 2.], [3., 4.], [5., 6.]]
 *   --> indices: [[-1, 0], [1, 1]]
 *   output array by 2 --> output: [5., 4.]
 *   @endverbatim
 *
 * @par Reference
 * - https://tensorflow.org/api_docs/python/tf/raw_ops/GatherNd
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGatherNd(mluOpHandle_t handle,
              const mluOpTensorDescriptor_t desc_params,
              const void *params,
              const mluOpTensorDescriptor_t desc_indices,
              const void *indices,
              const mluOpTensorDescriptor_t desc_output,
              void *output);

// Group:ScatterNd
/*!
 * @brief Distributes the elements in tensor \b updates to tensor \b output according to the coordinates
 * in tensor \b indices. If \b indices contains duplicates, then their \b updates are accumulated. This
 * operation is the inverse of the ::mluOpGatherNd operation which extracts values or slices from a given tensor.
 *
 * @par Deprecated
 * - ::mluOpScatterNd is deprecated and will be removed in the future release. It is recommended
 *   to use ::mluOpScatterNd_v2 instead.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the scatter_nd operation.
 * For detailed information, see ::mluOpHandle_t.
 * @param[in] indices_desc
 * The descriptor of the \b indices tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] indices
 * Pointer to the MLU memory that stores the index data. The value of the lowest dimension in \b indices
 * represents the coordinate of the element of \b updates in \b output tensor. If the coordinate contains negative
 * numbers or numbers that exceeds the range, then it will be deprecated.
 * @param[in] updates_desc
 * The descriptor of the \b updates tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] updates
 * Pointer to the MLU memory that stores the input data.
 * @param[in] output_desc
 * The descriptor of the output tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output data.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par Formula
 * - None.
 *
 * @par Data Type
 * - The ScatterNd operation supports the following data types for input tensor \b indices, \b updates,
 *   and output tensor \b output (except UPDATE mode on MLU590).
 *   The data type of \b updates and \b output must be the same.
 * - indices: int32, int64
 * - updates: int32, half, float
 * - output: int32, half, float
 * - When using the update mode in MLU590, The ScatterNd operation supports the following data types
 *   for input tensor \b indices, \b updates, and output tensor \b output.
 *   The data type of \b updates and \b output must be the same.
 * - indices: int32, int64
 * - updates: bool, int8, uint8, int16, uint16, half, int32, uint32, float, int64, uint64
 * - output: bool, int8, uint8, int16, uint16, half, int32, uint32, float, int64, uint64
 *
 * @par Scale Limitation
 * - If the rank of \b indices is n and indices[n-1] is ix, the shape of tensor \b indices, \b updates
 *   and \b output must meet the following restrictions:
 *
 *   indices.shape[0, n-2] = updates.shape[0, n-2]
 *   updates.rank - (n-1) = output.rank - ix
 *   updates.shape[n-1, updates.rank] = output.shape[ix, output.rank]
 *
 * @note
 * - This operation only supports TensorFlow framework.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of scatter_nd operation is as follows:
 * @verbatim
 *  The shape of output: [4,4,4].
 *  indices: [[0], [2]]
 *  updates: [[[5, 5, 5, 5], [6, 6, 6, 6],
 *             [7, 7, 7, 7], [8, 8, 8, 8]],
 *            [[5, 5, 5, 5], [6, 6, 6, 6],
 *             [7, 7, 7, 7], [8, 8, 8, 8]]]
 *  -->output: [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
 *              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
 *              [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
 *              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
 * @endverbatim
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpScatterNd(mluOpHandle_t handle,
               const mluOpTensorDescriptor_t indices_desc,
               const void *indices,
               const mluOpTensorDescriptor_t updates_desc,
               const void *updates,
               const mluOpTensorDescriptor_t output_desc,
               void *output);

// Group:ScatterNd
/*!
 * @brief Distributes the elements in tensor \b updates to tensor \b output according to the coordinates
 * in tensor \b indices. Compared with ::mluOpScatterNd, this function supports the \b mode parameter that contains
 * more calculation methods when \b indices contains duplicates. This operation is the inverse of the ::mluOpGatherNd
 * operation which extracts values or slices from a given tensor.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the scatter_nd operation.
 * For detailed information, see ::mluOpHandle_t.
 * @param[in] mode
 * The scatter_nd mode used when \b indices contains duplicates. For detailed information,
 * see ::mluOpScatterNdMode_t.
 * @param[in] indices_desc
 * The descriptor of the \b indices tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] indices
 * Pointer to the MLU memory that stores the index data. The value of the lowest dimension in \b indices
 * represents the coordinate of the element of \b updates in \b output tensor. If the coordinate contains negative
 * numbers or numbers that exceeds the range, then it will be ignored.
 * @param[in] updates_desc
 * The descriptor of the \b updates tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] updates
 * Pointer to the MLU memory that stores the data for updating.
 * @param[in] input_desc
 * The descriptor of the input tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input data.
 * @param[in] output_desc
 * The descriptor of the output tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output data.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Formula
 * - None.
 *
 * @par Data Type
 * - The ScatterNd operation supports the following data types for input tensor \b indices, \b updates,
 * and output tensor \b output. The data type of \b updates and \b output must be the same.
 * - indices: int32, int64, uint32, uint64.
 * - updates: int32, half, float.
 * - output: int32, half, float.
 *
 * @par Scale Limitation
 * - On MLU200 series, when the \b updates and \b output data type is int32 and \b mode = ::MLUOP_SCATTERND_ADD,
 *   the values of \b updates should be in the range of [\f$-2^{23}\f$, \f$2^{23}\f$].
 * - The number of dimensions is no more than \p MLUOP_DIM_MAX.
 * - If the rank of \b indices is n and indices[n-1] is ix, the shape of tensor \b indices, \b updates
 *   and \b output must meet the following restrictions:
 *   - indices.shape[0, n-2] = updates.shape[0, n-2]
 *   - updates.rank - (n-1) = output.rank - ix
 *   - updates.shape[n-1, updates.rank] = output.shape[ix, output.rank]
 *
 * @note
 * - When the \b input is NULL, it will be treated as zero vector. When the \b input is not NULL, the address
 *   can be equal to the \b output address, and in this case the performance is better.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of scatter_nd operation is as follows:
 * @verbatim
 *  Example 1:
 *  The shape of output: [4,4,4].
 *  mode:    MLUOP_SCATTERND_UPDATE
 *  indices: [[0], [0]]
 *  input:   [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
 *            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
 *            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
 *            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
 *  updates: [[[5, 5, 5, 5], [6, 6, 6, 6],
 *             [7, 7, 7, 7], [8, 8, 8, 8]],
 *            [[5, 5, 5, 5], [6, 6, 6, 6],
 *             [7, 7, 7, 7], [8, 8, 8, 8]]]
 *  -->output: [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
 *              [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
 *              [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
 *              [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
 *  Example 2:
 *  The shape of \b output: [4,4,4].
 *  mode:    MLUOP_SCATTERND_ADD
 *  indices: [[0], [0]]
 *  input:   NULL
 *  updates: [[[5, 5, 5, 5], [6, 6, 6, 6],
 *             [7, 7, 7, 7], [8, 8, 8, 8]],
 *            [[5, 5, 5, 5], [6, 6, 6, 6],
 *             [7, 7, 7, 7], [8, 8, 8, 8]]]
 *  -->output: [[[10, 10, 10, 10], [12, 12, 12, 12], [14, 14, 14, 14], [16, 16, 16, 16]],
 *              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
 *              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
 *              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
 * @endverbatim
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpScatterNd_v2(mluOpHandle_t handle,
                  mluOpScatterNdMode_t mode,
                  const mluOpTensorDescriptor_t indices_desc,
                  const void *indices,
                  const mluOpTensorDescriptor_t updates_desc,
                  const void *updates,
                  const mluOpTensorDescriptor_t input_desc,
                  const void *input,
                  const mluOpTensorDescriptor_t output_desc,
                  void *output);

/// Group:GetIndicePairs
/*!
 * @brief Computes the get_indice_paris operation, then returns the results in the output
 * tensor \b out_indices, \b indice_pairs and \b indice_num.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the
 * get_indice_pairs operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] sparse_conv_desc
 * The descriptor of sparse_conv parameter that needs convolution. For detailed information,
 * see ::mluOpSparseConvolutionDescriptor_t.
 * @param[in] indices_desc
 * The descriptor of output grad. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] indices
 * Pointer to the MLU memory that stores the indices tensor.
 * @param[in] out_indices_desc
 * The descriptor of out_indices including output locations. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] out_indices
 * Pointer to the MLU memory that stores the out_indices tensor.
 * @param[in] indice_pairs_desc
 * The descriptor of indice_pairs between input locations and output locations.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] indice_pairs
 * Pointer to the MLU memory that stores the indice_pairs tensor.
 * @param[in] indice_num_desc
 * The descriptor of indice_num including the number of input points while calculating with every kernel points.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] indice_num
 * Pointer to the MLU memory that stores the indice_num tensor.
 * @param[in] features_desc
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the get_indice_pairs operation.
 * For more information about workspace, see "Cambricon BANGC OPS User Guide".
 * @param[in] workspace_size
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH,
 *   ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - This function supports the combinations of the following data types for
 *   input tensor \b indices and output tensor \b out_indices, \b indice_pairs and \b indice_num.
 *   - \b indices, \b out_indices, \b indice_pairs and \b indice_num data type: int32, int32, int32, int32
 *
 * @note
 * - This function is only supported on MLU300 series or above platforms.
 * - The parameter num_act_out will be obtained from ::mluOpSparseConvolutionStruct.
 *
 * @par Scale Limitation
 * - The params inverse and transpose are not supported now.
 * - Get_indice_pairs only supported 3d.
 * - The input tensor and output tensor must meet the following requirements:
 *   - The \b indices must be two dimensions.
 *   - The \b indice_pairs must be three dimensions, and the first dimension value must be euqal to kernel size,
 *     the second dimension must be 2, and the last dimension must be the same as the number of
 *     product of the first n-1 dimensions of the input tensor in sparse convolution.
 *   - The \b out_indices should be 2 dimensions. The first dimension of \b out_indices is the number effective
 *     output point. and the second dimension of must product of the first n-1 dimensions of the input tensor
 *     in sparse convolution.
 *   - The \b indice_num should be 1 dimensions. The first dimension of \b indice_num is the kernel size.
 *
 * @par API Dependency
 * - Before calling this function, you need to prepare
 *   all the parameters passed to this function. See each parameter description for details.
 *
 * @par Example
 * - The example of the operation is as follows:
 *   @verbatim
 *    Dimension of indices tensor:  [input_active_in, dimnb -1]
 *    Dimension of out_indices tensor:  [output_active_num, dimnb - 1]
 *    Dimension of indice_pairs tensor: [kd * kh * kw, 2, input_active_in]
 *    Dimension of indice_num tensor: [kd * kh * kw]
 *   @endverbatim
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

// Group:GetIndicePairs
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra workspace
 * to optimize the get_indice_pairs operation.
 *
 * The size of extra workspace is based on the given information of the get_indice_pairs
 * operation, including the input tensor descriptor \b sparse_conv_desc, and \b indices_desc, output
 * tensor descriptor \b out_indices_desc, \b indice_pairs_desc and \b indice_num_desc.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the
 * get_indice_pairs operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] sparse_conv_desc
 * The descriptor of sparse_conv parameter that needs convolution. For detailed information,
 * see ::mluOpSparseConvolutionDescriptor_t.
 * @param[in] indices_desc
 * The descriptor of output grad. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] out_indices_desc
 * The descriptor of out_indices including output locations. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] indice_pairs_desc
 * The descriptor of indice_pairs between input locations and output locations.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] indice_num_desc
 * The descriptor of indice_num including the number of input points while calculating with every kernel points.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] workspace_size
 * Pointer to the MLU memory that stores the returned size of the extra workspace in bytes.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_INTERNAL_ERROR,
 *   ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par API Dependency
 * - You need to call the ::mluOpCreateTensorDescriptor and ::mluOpSetTensorDescriptor functions to create and set
 *   tensor descriptors \b indices_desc, \b out_indices_desc, \b indice_pairs_desc and \b indice_num_desc before
 *   calling this function.
 * - You need to call the ::mluOpCreateSparseConvolutionDescriptor function to create a descriptor,
 *   and call the ::mluOpSetSparseConvolutionDescriptor function to set the tensor information for
 *   the descriptor \b sparse_conv_desc.
 * - The allocated extra workspace should be passed to the ::mluOpGetIndicePairs function to
 *   perform the ge_indice_pairs operation.
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

// Group:Transpose
/*!
 * @brief Creates a descriptor pointed by \b desc for a transpose operation,
 * and allocated memory for holding the information about the transpose operation.
 *
 * The information is defined in ::mluOpTransposeDescriptor_t.
 *
 * @param[out] desc
 * Pointer to the transpose descriptor that holds information about
 * the transpose operation.
 * @par Return
 *   ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ALLOC_FAILED, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par API Dependency
 * - After calling this function, you can call the ::mluOpSetTransposeDescriptor
 *   function to initialize and set information to the transpose descriptor.
 * - You need to call the ::mluOpDestroyTransposeDescriptor function to destroy the descriptor.
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
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateTransposeDescriptor(mluOpTransposeDescriptor_t *desc);

// Group:Transpose
/*!
 * @brief Initializes the transpose descriptor \b desc that is previously created
 * with the ::mluOpCreateTransposeDescriptor function, and set the information
 * about the transpose operation to the transpose descriptor \b desc.
 * The information includes the permute dimensions \b dims and permute rules \b permute.
 *
 * @param[in] desc
 * The descriptor of the transpose operation. For detailed information,
 * see ::mluOpTransposeDescriptor_t.
 * @param[in] dims
 * The number of dimensions in the permute tensor of the transpose operation.
 * Currently, the value of this parameter should be less than or equal to 8.
 * @param[in] permute
 * The order of transpose. Currently, for each dimension, the value of permute
 * should be in the range of [0,...,dims -1], and should not be the same in each dimension.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_INTERNAL_ERROR
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
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetTransposeDescriptor(mluOpTransposeDescriptor_t desc, const int dims, const int permute[]);

// Group:Transpose
/*!
 * @brief Destroys a transpose descriptor \b desc that is previously created with the
 * ::mluOpCreateTensorDescriptor function.
 *
 * The transpose descriptor is defined in ::mluOpTransposeDescriptor_t and holds the information
 * about the transpose operation.
 *
 * @param[in] desc
 * The transpose descriptor to be destroyed. For detailed information,
 * see ::mluOpTransposeDescriptor_t.
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
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDestroyTransposeDescriptor(mluOpTransposeDescriptor_t desc);

// Group:Transpose
/*!
 * @brief Returns in \b size the size of the MLU memory that is used as an extra workspace
 * to optimize the transpose operation.
 *
 * The size of extra workspace is based on the given information of the transpose operation,
 * including the input tensor descriptor \b x_desc and transpose descriptor \b desc.
 * For more information about the workspace, see "Cambricon BANGC OPS User Guide".
 *
 * @param[in]  handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the transpose operation. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in]  x_desc
 * The descriptor of the input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] desc
 * The descriptor of the transpose operation. For detailed information,
 * see ::mluOpTransposeDescriptor_t.
 * @param[out] size
 * Host pointer to the returned size of the extra workspace in bytes that is used in
 * the transpose operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - This function must be called after the ::mluOpCreateTensorDescriptor and
 *   ::mluOpSetTensorDescriptor functions to create and set the tensor descriptors \b x_desc.
 * - The allocated extra workspace should be passed to the ::mluOpTranspose_v2 function
 *   to perform the transpose operation.
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
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTransposeWorkspaceSize(mluOpHandle_t handle,
                               const mluOpTensorDescriptor_t x_desc,
                               const mluOpTransposeDescriptor_t desc,
                               size_t *size);

// Group:Transpose
/*!
 * @brief Reorders the dimension according to the value of \b permute. To have better performance
 * for over 4D transpose with large-scale cases, call the * ::mluOpTranspose_v2 function.
 *
 * @par Deprecated
 * - ::mluOpTranspose is deprecated and will be removed in the further release. It is recommended
 *   to use ::mluOpTranspose_v2 instead.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues
 * in the transpose operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] desc
 * The descriptor of the transpose operation. For detailed information,
 * see ::mluOpTransposeDescriptor_t.
 * @param[in] x_desc
 * The descriptor of the input tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] y_desc
 * The descriptor of the output tensor. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[out] y
 * Pointer to the MLU memory that stores the output tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - This function supports the following data types for input tensor \b x and
 *   output tensor \b y.
 *   <b>Note that the data type of input tensor and output tensor should be same.</b>
 *   - input tensor: uint8, int8, uint16, int16, uint32, int32, uint64, int64, bool, half,
 *     float, complex_half, complex_float
 *   - output tensor: uint8, int8, uint16, int16, uint32, int32, uint64, int64, bool, half,
 *     float, complex_half, complex_float
 *
 * @par Data Layout
 * - The dimension of input tensor should be less than or equal to 8D.
 *
 * @par Scale Limitation
 * - The \b x, \b y and \b permute have the same shape.
 * - The dimension size of \b x, \b y and \b permute should be less than or equal to
 *   MLUOP_DIM_MAX.
 * - The \b permute i-th dimension is in the range [0,...n-1], where n is the rank of the \b x.
 * - The \b y i-th dimension will correspond to the \b x permute[i]-th dimension.
 * - The process of computing, the copy times of memcpy should be less than 65536.
 *
 * @par API Dependency
 * - Before calling this function to implement transpose, you need to prepare all the parameters
 *   passed to this function. See each parameter description for details.
 *
 * @note
 * - None.
 *
 * @par Example
 * - The example of the transpose operation is as follows:
 *   @verbatim
 *    input array by 3 * 2 -->
 *        input: [[1, 4],
 *                [2, 5],
 *                [3, 6]]
 *    param:
 *      dims: 2, permute: (1, 0),
 *    output array by 2 * 3 --> output: [[1, 2, 3],
 *                                       [4, 5, 6]]
 *   @endverbatim
 *
 * @par Reference
 * - https://www.tensorflow.org/api_docs/python/tf/transpose
 */
mluOpStatus_t MLUOP_WIN_API
mluOpTranspose(mluOpHandle_t handle,
               const mluOpTransposeDescriptor_t desc,
               const mluOpTensorDescriptor_t x_desc,
               const void *x,
               const mluOpTensorDescriptor_t y_desc,
               void *y);

// Group:Transpose
/*!
 * @brief Reorders the dimension according to the value of \b permute. Compared with
 * ::mluOpTranspose, ::mluOpTranspose_v2 provides better performance for above 4D
 * transpose with extra input space.
 *
 * This function needs extra MLU memory as the workspace to work.
 * You can get the size of the workspace \b workspace_size with the
 * ::mluOpGetTransposeWorkspaceSize function.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the transpose operation. For detailed information,
 * see ::mluOpHandle_t.
 * @param[in] desc
 * The descriptor of the transpose operation. For detailed information, see ::mluOpTransposeDescriptor_t.
 * @param[in] x_desc
 * The descriptor of the input tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] x
 * Pointer to the MLU memory that stores the input tensor.
 * @param[out] y_desc
 * The descriptor of the output tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] y
 * Pointer to the MLU memory that stores the output tensor.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the transpose operation.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in the transpose operation.
 * You can get the size of the workspace with the ::mluOpGetTransposeWorkspaceSize function.

 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par Scale Limitation
 * - The \b x, \b y and \b permute have the same shape.
 * - The dimension size of \b x, \b y and \b permute should be less than or equal to MLUOP_DIM_MAX.
 * - The \b permute i-th dimension is in the range [0,...n-1], where n is the rank of the \b x.
 * - The \b y i-th dimension will correspond to \b x permute[i]-th dimension.
 * - The process of computing, the copy times of memcpy should be less than 65536.
 *
 * @par Formula
 * - None.
 *
 * @par Data Type
 * - This function supports the following data types for input tensor \b x and
 *   output tensor \b y.
 *   <b>Note that the data type of input tensor and output tensor should be same.</b>
 *   - input tensor: uint8, int8, uint16, int16, uint32, int32, uint64, int64, bool, half,
 *     float, complex_half, complex_float
 *   - output tensor: uint8, int8, uint16, int16, uint32, int32, uint64, int64, bool, half,
 *     float, complex_half, complex_float
 *
 * @par Data Layout
 * - The dimension of input tensor should be less than or equal to 8D.

 * @par API Dependency
 * - Before calling this function to implement transpose, you need to prepare
 *   all the parameters passed to this function. See each parameter description
 *   for details.
 *
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of the transpose operation is as follows:
 *   @verbatim
 *    input array by 3 * 2 -->
 *         input: [[1, 4],
 *                 [2, 5],
 *                 [3, 6]]
 *     param:
 *       dims: 2, permute: (1, 0),
 *
 *     output array by 2 * 3 --> output: [[1, 2, 3],
 *                                        [4, 5, 6]]
 *    @endverbatim
 *
 * @par Reference
 * - https://www.tensorflow.org/api_docs/python/tf/transpose
 */
mluOpStatus_t MLUOP_WIN_API
mluOpTranspose_v2(mluOpHandle_t handle,
                  const mluOpTransposeDescriptor_t desc,
                  const mluOpTensorDescriptor_t x_desc,
                  const void *x,
                  const mluOpTensorDescriptor_t y_desc,
                  void *y,
                  void *workspace,
                  size_t workspace_size);

// Group:Reduce
/*!
 * @brief Creates a descriptor pointed by \b reduce_desc that holds the \b axis,\b reduce_op,
 * \b tensor_type, \b nan_propagation, \b tensor_indices, \b indices_type and \b p.
 * The information is defined in ::mluOpReduceDescriptor_t.
 *
 * @param[out] reduce_desc
 * Host pointer to the reduce descriptor that holds information about reduce.
 * For detailed information, see ::mluOpReduceDescriptor_t.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_ALLOC_FAILED, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par API Dependency
 * - You need to call the ::mluOpDestroyReduceDescriptor function to destroy the descriptor.
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
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateReduceDescriptor(mluOpReduceDescriptor_t *reduce_desc);

// Group:Reduce
/*!
 * @brief Initializes the reduce descriptor \b reduce_desc that is previously created with
 * the ::mluOpCreateReduceDescriptor function, and sets the information about the reduce
 * operation. To use p-norm in this operation, call ::mluOpSetReduceDescriptor_v2.
 *
 * @param[in] reduce_desc
 * The descriptor of the reduce operation. For detailed information,
 * see ::mluOpReduceDescriptor_t.
 * @param[in] axis[]
 * The axis dimension vector of the reduce operation.
 * @param[in] axis_num
 * The size of axis vector.
 * @param[in] reduce_op
 * The reduce mode. For detailed information, see ::mluOpReduceOp_t.
 * @param[in] tensor_type
 * The data type is used in computing the reduce operation. For detailed information,
 * see ::mluOpDataType_t.
 * @param[in] nan_propagation
 * The NaN propagation mode. The default value is NOT_PROPAGATE_NAN.
 * Now the reduce does not support this parameter.
 * For detailed information, see ::mluOpNanPropagation_t.
 * @param[in] tensor_indices
 * The reduce indices mode.
 * For detailed information, see ::mluOpReduceIndices_t.
 * @param[in]  indices_type
 * The bit width type of reduce indices.
 * At present, this parameter can only be set as MLUOP_32BIT_INDICES.
 * For detailed information, see ::mluOpIndicesType_t.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par API Dependency
 * - Before calling this function, you need to call ::mluOpCreateReduceDescriptor.
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
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetReduceDescriptor(mluOpReduceDescriptor_t reduce_desc,
                         int axis[],
                         int axis_num,
                         mluOpReduceOp_t reduce_op,
                         mluOpDataType_t tensor_type,
                         mluOpNanPropagation_t nan_propagation,
                         mluOpReduceIndices_t tensor_indices,
                         mluOpIndicesType_t indices_type);
// Group:Reduce
/*!
 * @brief Initializes the reduce descriptor \b reduce_desc that is previously created with
 * the ::mluOpCreateReduceDescriptor function, and sets the information about the reduce
 * operation. Compared with ::mluOpSetReduceDescriptor, this function supports \b
 * reduce_op == \p MLUOP_REDUCE_NORMP, and includes more information such as \b p.
 *
 * @param[in] reduce_desc
 * The descriptor of the reduce operation.
 * For detailed information, see ::mluOpReduceDescriptor_t.
 * @param[in] axis[]
 * The axis dimension vector of the reduce operation.
 * @param[in] axis_num
 * The size of axis vector.
 * @param[in] reduce_op
 * Enumeration to specify the reduce mode.
 * For detailed information, see ::mluOpReduceOp_t.
 * @param[in] tensor_type
 * The data type is used in computing the reduce operation.
 * For detailed information, see ::mluOpDataType_t.
 * @param[in] nan_propagation
 * The NaN propagation mode. Default
 * value is NOT_PROPAGATE_NAN. Now reduce does not support this parameter.
 * For detailed information, see ::mluOpNanPropagation_t.
 * @param[in] tensor_indices
 * The reduce indices mode.
 * For detailed information, see ::mluOpReduceIndices_t.
 * @param[in]  indices_type
 * The bit width type of reduce indices.
 * At present, this parameter can only be set as MLUOP_32BIT_INDICES.
 * For detailed information, see ::mluOpIndicesType_t.
 * @param[in] p
 * The exponent value in the norm formulation. \b p cannot be 1.0, 2.0, INF, and -INF.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - Before calling this function, you need to call ::mluOpCreateReduceDescriptor.
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
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetReduceDescriptor_v2(mluOpReduceDescriptor_t reduce_desc,
                            int axis[],
                            int axis_num,
                            mluOpReduceOp_t reduce_op,
                            mluOpDataType_t tensor_type,
                            mluOpNanPropagation_t nan_propagation,
                            mluOpReduceIndices_t tensor_indices,
                            mluOpIndicesType_t indices_type,
                            float p);
// Group:Reduce
/*!
 * @brief Destroys a reduce descriptor \b reduce_desc that is previously created with
 * the ::mluOpCreateReduceDescriptor.
 *
 * The reduce descriptor is defined in ::mluOpReduceDescriptor_t and holds the information
 * about the reduce.
 *
 * @param[in] reduce_desc
 * The reduce descriptor to be destroyed.
 * For detailed information, see ::mluOpReduceDescriptor_t.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - Before calling this function, you need to call ::mluOpCreateReduceDescriptor.
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
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpDestroyReduceDescriptor(mluOpReduceDescriptor_t reduce_desc);

// Group:Reduce
/*!
 * @brief Applies an operation of reduce to compute the sum value, mean value, maximum value,
 * maximum index, minimum value, and minimum index of tensor in the given dimension.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the reduce operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] reduce_desc
 * The descriptor of reduce operation. For detailed information,
 * see ::mluOpReduceDescriptor_t.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the reduce operation.
 * For more information about workspace, see "Cambricon BANGC OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that needs to be used in the reduce operation.
 * You can get the size of the workspace with the ::mluOpGetReduceOpWorkspaceSize function.
 * @param[in] alpha
 * Host pointer to scaling factor of tensor output. The value of this parameter can be NULL.
 * @param[in] beta
 * Host pointer to bias factor of tensor output. The value of this parameter can be NULL.
 * @param[in] input_desc
 * Descriptor of input data. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] indices_size_inbytes
 * The size in bytes of indices.
 * @param[out] indices
 * Pointer to the MLU memory that stores the indices tensor.
 * @param[in] output_desc
 * Descriptor of output data. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par Data Type
 * - Data types of input tensor and output tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - When \b reduce_op == \p MLUOP_REDUCE_MAX || \b reduce_op == \p MLUOP_REDUCE_MIN:
 *     - input: float, half, int32
 *     - output: float, half, int32
 *     - indices: uint32, int32
 *
 *   - When \b reduce_op == \p MLUOP_REDUCE_MUL:
 *     - input: float, half, int32
 *     - output: float, half, int32
 *
 *   - When \b reduce_op == \p MLUOP_REDUCE_AND || \b reduce_op == \p MLUOP_REDUCE_OR:
 *     - input: float, half, int8, uint8, bool
 *     - output: float, half, int8, uint8, bool
 *
 *   - When \b reduce_op == \p MLUOP_REDUCE_ADD || \b reduce_op == \p MLUOP_REDUCE_AVG:
 *     - input: float, half, int32
 *     - output: float, half, int32
 *
 *   - When \b reduce_op == \p MLUOP_REDUCE_ASUM || \b reduce_op == \p MLUOP_REDUCE_SUMSQ ||
 *          \b reduce_op == \p MLUOP_REDUCE_NORM1 || \b reduce_op == \p MLUOP_REDUCE_NORM2 ||
 *          \b reduce_op == \p MLUOP_REDUCE_NORMP:
 *     - input: float, half
 *     - output: float, half
 *
 *   - When \b reduce_op == \p MLUOP_REDUCE_MAX_LAST_INDEX ||
 *          \b reduce_op == \p MLUOP_REDUCE_MIN_LAST_INDEX:
 *     - input: float, half, int32
 *     - output: float, half, int32
 *     - indices: uint32, int32
 * - \b alpha and \b beta: If the data type of input tensor is float or half, the data type of \b alpha
 *   and \b beta should be float pointer. If the data type of input tensor is int32, the data type of \b alpha
 *   and \b beta should be int32 pointer.
 *
 * @par Data Layout
 * - The supported layout of the input tensors and output tensors must be \p MLUOP_LAYOUT_ARRAY.

 * @par API Dependency
 * - Before calling this function to implement reduce, you need to prepare all the parameters
 *   passed to this function. Call ::mluOpCreateReduceDescriptor to create the parameter \b reduce_desc.
 *   Then call ::mluOpSetReduceDescriptor_v2 to set information about the parameter \b reduce_desc and
 *   call ::mluOpGetReduceOpWorkspaceSize to get extra MLU memory size in reduce operation.
 * - After calling this function, the ::mluOpDestroyReduceDescriptor needs to be called to destroyed the
 *   parameter \b reduce_desc.
 *
 * @note
 * - The \b axis must meet the following requirements:
 *   - When the number of \b axis is greater than 1, the values of axis vector cannot be duplicated.
 *     For example, \b axis = [1,2,3] or \b axis = [1,2,4].
 *   - The size of \b axis cannot be greater than the size of input.
 * - The following reduce modes support multi-axis numbers including continuous and discontinuous axis numbers
 *   except for \b p is 0.0 when \b reduce_op == \p MLUOP_REDUCE_NORMP:
 *   - \p MLUOP_REDUCE_ADD, \p MLUOP_REDUCE_AVG, \p MLUOP_REDUCE_MUL, \p MLUOP_REDUCE_OR, \p MLUOP_REDUCE_AND,
 *     \p MLUOP_REDUCE_NORM1, \p MLUOP_REDUCE_NORM2, and \p MLUOP_REDUCE_NORMP
 * - When \b reduce_op == \p MLUOP_REDUCE_MAX || \b reduce_op == \p MLUOP_REDUCE_MIN:
 *   - The \b axis supports multi-axis numbers including continuous and discontinuous axis numbers,
 *     when \b tensor_indices is \p MLUOP_REDUCE_NO_INDICES.
 *   - The \b axis supports single-axis number and return the index of the first max or min value,
 *     when \b tensor_indices is \p MLUOP_REDUCE_ONLY_INDICES or \p MLUOP_REDUCE_FLATTENED_INDICES.
 * - When \b reduce_op == \p MLUOP_REDUCE_MAX_LAST_INDEX || \b reduce_op == MLUOP_REDUCE_MIN_LAST_INDEX:
 *   - The \b axis only support single-axis number.
 *   - The \b tensor_indices only support \p MLUOP_REDUCE_ONLY_INDICES or \p MLUOP_REDUCE_FLATTENED_INDICES,
 *     and return the index of the last max or min value.
 * - When \b reduce_op == \p MLUOP_REDUCE_ASUM || \b reduce_op == \p MLUOP_REDUCE_SUMSQ:
 *   - These two modes only support Caffe framework.
 *   - The mode \p MLUOP_REDUCE_ASUM refers to the cumulative reduction after taking the absolute value of
 *     a specified dimension.
 *   - The mode \p MLUOP_REDUCE_SUMSQ calculates the square of the specified dimension and performs cumulative
 *     reduction.
 *   - Whether you set \b tensor_type parameter or not, the data type used in computing the reduce operation
 *     of these two modes is fixed as float.
 * - The following modes support input with stride:
 *   - \p MLUOP_REDUCE_ADD, \p MLUOP_REDUCE_AVG, \p MLUOP_REDUCE_MUL, \p MLUOP_REDUCE_MAX, \p MLUOP_REDUCE_MIN,
 *     \p MLUOP_REDUCE_OR, \p MLUOP_REDUCE_AND, \p MLUOP_REDUCE_NORM1, \p MLUOP_REDUCE_NORM2 and MLUOP_REDUCE_NORMP.
 * - This function reduces tensor input by implementing the equation output = alpha * reduce(input) + beta,
 *   given tensors \b input and \b output and scaling factors \b alpha and \b beta.
 *   - The following modes support \b alpha and \b beta:
 *     - \p MLUOP_REDUCE_ADD, \p MLUOP_REDUCE_AVG, \p MLUOP_REDUCE_MUL, \p MLUOP_REDUCE_NORM1, \p MLUOP_REDUCE_NORM2,
 *       and \p MLUOP_REDUCE_NORMP.
 *   - The \b alpha and \b beta can set NULL, or the \b alpha float value is 1.0 and the \b beta float value
 *     is 0.0 for modes that not support \b alpha and \b beta.
 *
 * @par Scale Limitations
 * - When \b reduce_op == \p MLUOP_REDUCE_NORMP on MLU200 series:
 *   - The sum of p power of input absolute should be in range[7.2e-9, 507903] when data type is
 *     float and [6.1e-5,65504] when data type is half.
 *   - The p power of input absolute should be in range[-3.4e38, 16] when data type is float and
 *     [-65504,10.25] when data type is half.
 *   - The product of 1/p and sum of p power of input absolute should be in range[-3.4e38, 16]
 *     when data type is float and [-65504,10.25] when data type is half.
 * - When \b reduce_op == \p MLUOP_REDUCE_MAX_LAST_INDEX || \b reduce_op == \p MLUOP_REDUCE_MIN_LAST_INDEX:
 *   - The \b input with NaN or INFINITY is not supported.
 *   - The data range of \b input should satisfy the conditions: (-INFINITY, INFINITY).
 * - When input data contains NaN on MLU300 series:
 *   - The MLUOP_REDUCE_MIN and MLUOP_REDUCE_MAX results are different with IEEE754.
 *     - If the first operand is NaN and the second operand is finite value, then output is NaN.
 *     - If the first operand is finite value and the second operand is finite value, then output is finite value.
 *   - The \p MLUOP_REDUCE_NORMP results are different with IEEE754 when \b p is 0.0.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The examples of the layer normalization forward operation are as follows:
 *   @verbatim
 *   input dimension = [n,c,h,w,d],
 *   When axis = 0:
 *    output dimension = [1,c,h,w,d].
 *    (indices dimension = [1,c,h,w,d], reduce_op == MLUOP_REDUCE_MAX or MLUOP_REDUCE_MIN).
 *   When axis = 1:
 *    output dimension = [n,1,h,w,d].
 *    (indices dimension = [n,1,h,w,d], reduce_op == MLUOP_REDUCE_MAX or MLUOP_REDUCE_MIN).
 *   When axis = 2:
 *    output dimension = [n,c,1,w,d].
 *    (indices dimension = [n,c,1,w,d], reduce_op == MLUOP_REDUCE_MAX or MLUOP_REDUCE_MIN).
 *   When axis = 3:
 *    output dimension = [n,c,h,1,d].
 *    (indices dimension = [n,c,h,1,d], reduce_op == MLUOP_REDUCE_MAX or MLUOP_REDUCE_MIN).
 *   When axis = 4:
 *    output dimension = [n,c,h,w,1].
 *    (indices dimension = [n,c,h,w,1], reduce_op == MLUOP_REDUCE_MAX or MLUOP_REDUCE_MIN).
 *   When axis = -1:
 *    output dimension = [1,1,1,1,1].
 *    (indices dimension = [1,1,1,1,1], reduce_op == MLUOP_REDUCE_MAX or MLUOP_REDUCE_MIN).
 *   @endverbatim
 *
 * @par Reference
 * - https://tensorflow.google.cn/api_docs/python/tf/math/reduce_sum
 * - https://tensorflow.google.cn/api_docs/python/tf/math/reduce_mean
 * - https://tensorflow.google.cn/api_docs/python/tf/math/reduce_prod
 * - https://tensorflow.google.cn/api_docs/python/tf/math/reduce_max
 * - https://tensorflow.google.cn/api_docs/python/tf/math/reduce_min
 */
mluOpStatus_t MLUOP_WIN_API
mluOpReduce(mluOpHandle_t handle,
            const mluOpReduceDescriptor_t reduce_desc,
            void *workspace,
            size_t workspace_size,
            const void *alpha,
            const mluOpTensorDescriptor_t input_desc,
            const void *input,
            const size_t indices_size_inbytes,
            void *indices,
            const void *beta,
            const mluOpTensorDescriptor_t output_desc,
            void *output);
// Group:Reduce
/*!
 * @brief Returns in \b size the size of the MLU memory that is used to get
 * extra space size in reduce operation.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the reduce peration.
 * For detailed information, see ::mluOpHandle_t.
 * @param[in] input_desc
 * A descriptor of input tensor.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] output_desc
 * A descriptor of output tensor.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] reduce_op
 * An operation enum, which indicates a specific reduce operation.
 * For detailed information, see ::mluOpReduceDescriptor_t.
 * @param[out] workspace_size_inbytes
 * Pointer to the returned size of the extra workspace in bytes that is
 * used in the reduce operation.
 *
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
 *
 * @par Reference
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetReduceOpWorkspaceSize(mluOpHandle_t handle,
                              const mluOpTensorDescriptor_t input_desc,
                              const mluOpTensorDescriptor_t output_desc,
                              const mluOpReduceDescriptor_t reduce_op,
                              size_t *workspace_size_inbytes);

// Group:ActiveRotatedFilterForward
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra
 * workspace to optimize the mluOpActiveRotatedFilterForward operation. The size of the extra
 * workspace is based on the given information of the ActiveRotatedFilterForward operation,
 * including the input tensor descriptor \b input_desc. For more information about the workspace,
 * see "Cambricon BANGC OPS User Guide".
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the
 * ActiveRotatedFilterForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] input_desc
 * The descriptor of input data \b input, which contains dimension, data type and data layout.
 * @param[out] workspace_size
 * A host pointer to the returned size of the extra workspace in bytes that is used in
 * the ActiveRotatedFilterForward operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - This function must be called before the ::mluOpActiveRotatedFilterForward function.
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
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetActiveRotatedFilterForwardWorkspaceSize(const mluOpHandle_t handle,
                                                const mluOpTensorDescriptor_t input_desc,
                                                size_t *workspace_size);

// Group:ActiveRotatedFilterForward
/*!
 * @brief Rotates \b input according to \b indices. This function encodes
 * the orientation information and generates orientation-sensitive features.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in
 * the ActiveRotatedFilterForward operation. For detailed information, see
 * ::mluOpHandle_t.
 * @param[in] input_desc
 * The descriptor of input data \b input, which contains dimension, data type
 * and data layout.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] indices_desc
 * The descriptor of input data \b indices, which contains dimension, data type
 * and data layout.
 * @param[in] indices
 * Pointer to the MLU memory that stores the indices tensor. It is used to
 * specify the position of each element of canonical filter after rotations.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the
 * ActiveRotatedFilterForward operation. For more information about workspace,
 * see "Cambricon BANGC OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that is used in
 * the ActiveRotatedFilterForward operation.
 * @param[in] output_desc
 * The descriptor of output data \b output, which contains dimension, data type
 * and data layout.
 * @param[out] output
 * Pointer to the MLU memory that stores the \b output tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Formula
 * - None.
 *
 * @par Data Type
 * - The supported data types of \b input, \b indices and \b output are as follows:
 *   Data types of input tensor and output tensor should be the same.
 *   - input tensor: half, float
 *   - output tensor: half, float
 *   - indices tensor: int32
 *
 * @par Data Layout
 * - The supported data layouts of \b input, \b indices, and \b output are as follows:
 *   - input tensor: MLUOP_LAYOUT_ARRAY
 *   - output tensor: MLUOP_LAYOUT_ARRAY
 *   - indices tensor: MLUOP_LAYOUT_ARRAY
 *
 * @par Scale Limitation
 * - The \b input is 5D array, and \b indices and \b output are 4D array.
 * - The dims[2] of \b input should be equal to the power of 2 and less than or
 * equal to 128, dims[3] should be equal to 1 or 3, and dims[3] should be equal
 * to dims[4].
 * - The dims[0] of \b indices should be equal to \b input dims[2], and dims[1]
 * and dims[2] of \b indices should be equal to dims[3] and dims[4] of \b input
 * respectively.
 * - The dims[3] of \b indices should be equal to 2, 4, or 8.
 * - The dims[0] of \b output should be equal to dims[0] of \b input times
 * dims[3] of \b indices.
 * - The dims[1] of \b ouput should be equal to dims[1] of \b input times
 * dims[2] of \b input.
 * - The dims[2] and dims[3] of \b output should be equal to dims[3] and dims[4]
 * of \b input respectively.
 *
 * @par API Dependency
 * - Before calling this function, you need to call
 * ::mluOpGetActiveRotatedFilterForwardWorkspaceSize to get the extra space size
 * needed in ActiveRotatedFilterForward operation.
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

// Group:DeformRoiPool
/*!
 * @brief Computes deformable roi pooling over \b input tensor. This function firstly divides the obtained
 * candidate region into regions with the same size according to the specified pooling width and pooling height,
 * then adds offsets to rois, and finally calculates the mean value of the sampling points in each bin as output.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in
 * ::mluOpDeformRoiPoolForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] input_desc
 * The descriptor of input tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] rois_desc
 * The descriptor of rois tensor, which contains the dimension and layout of rois tensor.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] rois
 * Pointer to the MLU memory that stores the rois tensor.
 * @param[in] offset_desc
 * The descriptor of offset tensor, which contains the dimension and layout of offset tensor.
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
 * The descriptor of output tensor, which contains the dimension and layout of output tensor.
 * @param[out] output
 * Pointer to the MLU memory that stores the output tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Formula
 * - See "DeformRoiPoolForward Operation" section in "Cambricon BANGC OPS User Guide" for details.
 *
 * @par Data Type
 * - The supported data types of \b input, \b rois, \b offset and \b output are as follows:
 *   Data types of all tensors should be the same.
 *   - input tensor: half, float
 *   - rois tensor: half, float
 *   - offset tensor: half, float
 *   - output tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of \b input, \b rois, \b offset and \b output are as follows:
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
 * @note
 * - The inputs \b rois and \b offset with NaN or infinity are not supported.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - The example of the deform_roi_pool_forward operation is as follows:
 *   @verbatim
 *   input three arrays by 1  2  2  1, 1  5 and 1  2  1 * 1
 *   --> input: [[[[1.0], [2.0]], [[2.0], [4.0]]]]
 *   --> rois: [[0.0, 0.0, 0.0, 1.0, 1.0]]
 *   --> offset: [[[[0.5]], [[0.5]]]]
 *   param:
 *          pooled_height: 1.0, pooled_width: 1.0, spatial_scale: 1.0,
 *          sampling_ratio: 1, gamma: 1
 *   output array by 1  1  1 * 1 -->
 *       output: [[[[2.25]]]]
 *   @endverbatim
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

// Group:DeformRoiPool
/*!
 * @brief Computes the gradient of input \b grad_input and the gradient of offset \b grad_offset
 * based on the gradient of ouput \b grad_output, input \b input, ROI \b rois and offset \b offset.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in
 * ::mluOpDeformRoiPoolBackward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] grad_output_desc
 * The descriptor of grad_output tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] grad_output
 * Pointer to the MLU memory that stores the grad_output tensor.
 * @param[in] input_desc
 * The descriptor of input tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] input
 * Pointer to the MLU memory that stores the input tensor.
 * @param[in] rois_desc
 * The descriptor of rois tensor, which contains the dimension and layout of rois tensor.
 * For detailed information, see ::mluOpTensorDescriptor_t.
 * @param[in] rois
 * Pointer to the MLU memory that stores the rois tensor.
 * @param[in] offset_desc
 * The descriptor of offset tensor, which contains the dimension and layout of offset tensor.
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
 * The descriptor of grad_input tensor, which contains the dimension and layout of grad_input tensor.
 * @param[out] grad_input
 * Pointer to the MLU memory that stores the gradient of the input tensor.
 * @param[in] grad_offset_desc
 * The descriptor of grad_offset tensor, which contains the dimension and layout of grad_offset tensor.
 * @param[out] grad_offset
 * Pointer to the MLU memory that stores the gradient of the offset tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Formula
 * - None.
 *
 * @par Data Type
 * - The supported data types of grad_output tensor \b grad_output, input tensor \b input, rois tensor \b rois,
 *   offset tensor \b offset, grad_input tensor \b grad_input and grad_offset tensor \b grad_offset are as follows:
 *   Data types of all tensors should be the same.
 *   - grad_output tensor: half, float
 *   - input tensor: half, float
 *   - rois tensor: half, float
 *   - offset tensor: half, float
 *   - grad_input tensor: half, float
 *   - grad_offset tensor: half, float
 *
 * @par Data Layout
 * - The supported data layouts of grad_output tensor \b grad_output, input tensor \b input, rois tensor \b rois,
 *   offset tensor \b offset, grad_input tensor \b grad_input and grad_offset tensor \b grad_offset are as follows:
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
 * @note
 * - The inputs \b rois and \b offset with NaN or infinity are not supported.
 *
 * @par Requirements
 * - None.
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

// Group:IndiceConvolutionBackwardData
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as
 * an extra workspace to optimize the indice convolution backward data operation.
 *
 * The size of extra workspace is based on the given information of the indice
 * convolution backward data operation, including the input descriptor
 * \b input_grad_desc, the filter descriptor \b filter_desc, the indice pairs
 * descriptor \b indice_pairs_desc, the output descriptor \b indice_pairs_desc,
 * the array \b indice_num and the scaler \b inverse. For more information
 * about the workspace, see "Cambricon BANGC OPS User Guide".
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in
 * the ::mluOpIndiceConvolutionBackwardData operation. For detailed
 * information, see ::mluOpHandle_t.
 * @param[in] output_grad_desc
 * The descriptor of the output_grad tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] filters_desc
 * The descriptor of the filters tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] indice_pairs_desc
 * The descriptor of the indice_pairs tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] input_grad_desc
 * The descriptor of the input_grad tensor. For detailed information, see
 * ::mluOpTensorDescriptor_t.
 * @param[in] indice_num
 * The array to describe the valid data number in the indice_pairs tensor.
 * @param[in] inverse
 * The parameter to describe whether the operation performs deconvolution logic.
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that is used in the
 * ::mluOpIndiceConvolutionBackwardData operation.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - This function must be called before the ::mluOpIndiceConvolutionBackwardData
 *   function.
 * - The ::mluOpCreateTensorDescriptor and ::mluOpSetTensorDescriptor functions
 *   create and set the tensor descriptor \b output_grad_desc, \b filters_desc,
 *   \b indice_pairs_desc and \b input_grad_desc before this function is called.
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

// Group:IndiceConvolutionBackwardData
/*!
 * @brief Performs the back propagation of an indice convolution operation to
 * compute the gradient of input \b input_grad based on the gradient of response
 * \b output_grad, the filter tensor \b filter, the indice tensor \b indice_pairs
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
 * Handle to an MLUOP context that is used to manage MLU devices and queues in
 * the ::mluOpIndiceConvolutionBackwardData operation. For detailed
 * information, see ::mluOpHandle_t.
 * @param[in] output_grad_desc
 * The descriptor of input data \b output_grad, which contains dimension, data
 * type and data layout.
 * @param[in] output_grad
 * Pointer to the MLU memory that stores the output_grad tensor. It is the
 * gradient data of output tensor after reordered.
 * @param[in] filters_desc
 * The descriptor of input data \b filters, which contains dimension, data type
 * and data layout. It contains N, H, W and C information when it is a 4D
 * tensor, or N, D, H, W and C information when it is a 5D tensor.
 * @param[in] filters
 * Pointer to the MLU memory that stores the filter tensor.
 * @param[in] indice_pairs_desc
 * The descriptor of input data \b indice_pairs, which contains dimension,
 * data type and data layout.
 * @param[in] indice_pairs
 * Pointer to the MLU memory that stores the indice_pairs tensor. It is used to
 * specify the calculation pairs between \b input_grad and \b output_grad.
 * @param[in] indice_num
 * The array to describe the valid data number in \b indice_pairs.
 * @param[in] inverse
 * The parameter to describe whether the operation performs deconvolution logic.
 * @param[in] sub_m
 * The parameter to describe whether the operation performs sub_m convolution
 * logic or sparce convolution logic.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the
 * ::mluOpIndiceConvolutionBackwardData operation. For more information about
 * workspace, see "Cambricon BANGC OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that is used in the
 * ::mluOpIndiceConvolutionBackwardData operation.
 * @param[in] input_grad_desc
 * The descriptor of output data \b input_grad, which contains dimension, data
 * type and data layout.
 * @param[out] input_grad
 * Pointer to the MLU memory that stores the \b output tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH,
 *   ::MLUOP_STATUS_NOT_SUPPORTED, ::MLUOP_STATUS_INTERNAL_ERROR
 *
 * @par Formula
 * - None.
 *
 * @par Data Type
 * - The supported data types of \b output_grad, \b filters, \b indice_pairs and
 *   \b input_grad are as follows:
 *   Data types of output_grad tensor, filters tensor and input_grad tensor should
 *   be the same.
 *   - output_grad tensor: half, float
 *   - filters tensor: half, float
 *   - indice_pairs tensor: int32
 *   - input_grad tensor: half, float
 *
 * - The supported data type of array \b indice_num, scaler \b inverse and
 *   \b sub_m is int64.
 *
 * @par Data Layout
 * - The supported data layouts of \b output_grad, \b filters and \b input_grad
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
 * - The function ::mluOpGetIndiceConvolutionBackwardDataWorkspaceSize should
 *   be called to get the extra space size before this function is called.
 *
 * @note
 * - When the \b filter is a 5D tensor, the layout MLUOP_LAYOUT_ARRAY represents
 *   the data layout of (D, H, W, C, N).
 * - The length of the array \b indice_num should be equal to the dims[0] of \b indice_pairs.
 * - The data values of \b indice_pairs should be no smaller than 0.
 * - The data values of tensor slices indice_pairs[:,0,:] should be no larger
 *   than the dims[0] of \b input_grad.
 * - The data values of tensor slices indice_pairs[:,1,:] should be no larger
 *   than the dims[0] of \b output_grad.
 *
 * @par Requirements
 * - None.
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

// Group:IndiceConvolutionBackwardFilter
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra workspace
 * to optimize the indice_convolution_backward_filter operation.
 *
 * The size of extra workspace is based on the given information of the indice_convolution_backward_filter
 * operation, including the input tensor descriptor \b features_desc, \b output_grad_desc and \b indice_pairs_desc,
 * output tensor descriptor \b filters_grad_desc, and the array \b indice_num[].
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the
 * indice_convolution_backward_filter operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] features_desc
 * The descriptor of features that need convolution. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] output_grad_desc
 * The descriptor of output grad. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] indice_pairs_desc
 * The descriptor of indice pairs between input locations and output locations. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] filters_grad_desc
 * The descriptor of filters grad tensor. For detailed information,
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
 * @par API Dependency
 * - You need to call the ::mluOpCreateTensorDescriptor and ::mluOpSetTensorDescriptor functions to create and set
 *   tensor descriptors \b features_desc, \b output_grad_desc, \b indice_pairs_desc and \b filters_grad_desc before
 *   calling this function.
 * - The allocated extra workspace should be passed to the ::mluOpIndiceConvolutionBackwardFilter function to
 *   perform the indice_convolution_backward_filter operation.
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

// Group:IndiceConvolutionBackwardFilter
/*!
 * @brief Computes the indice_convolution_backward_filter operation, then returns the results in the output
 * tensor \b filters_grad.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the
 * indice_convolution_backward_filter operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] features_desc
 * The descriptor of features that need convolution. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] features
 * Pointer to the MLU memory that stores the features tensor.
 * @param[in] output_grad_desc
 * The descriptor of output grad. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] output_grad
 * Pointer to the MLU memory that stores the output grad tensor.
 * @param[in] indice_pairs_desc
 * The descriptor of indice pairs between inputs locations and outputs locations. For detailed information,
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
 * For more information about workspace, see "Cambricon BANGC OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes.
 * @param[in] filters_grad_desc
 * The descriptor of filters grad tensor. For detailed information,
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
 *   input tensor \b features, \b output_grad, \b indice_pairs_num and output tensor \b filters_grad.
 *   - \b features, \b output_grad, \b indice_pairs \b filters_grad data type: half, half, int32, half
 *   - \b features, \b output_grad, \b indice_pairs \b filters_grad data type: float, float, int32, float
 *
 * @note
 * - This function is only supported on MLU300 series or above platforms.
 * - This function does not support setting tensor onchip data type with fixed-point type.
 *
 * @par Scale Limitation
 * - The input tensor and output tensor must meet the following requirements:
 *   - The \b features and \b output_grad must be two dimensions.
 *   - The \b indice_pairs must be three dimensions, and the first dimension value must be euqal to the
 *     kernel size of \b filters_grad, the second dimension must be 2, and the last dimension must be
 *     equal to the number of \b features first dimension.
 *   - The \b filters_grad must be four or five dimensions. The last dimension of \b filters_grad must
 *     be euqal to the last dimension of \b output_grad, and the penultimate dimension of \b filters_grad
 *     must be equal to the last dimension of \b features.
 *   - The array length of indice_num must be euqal to the first dimension of \b indice_pairs.
 *
 * @par API Dependency
 * - Before calling this function to implement matrix multiplication, you need to prepare
 *   all the parameters passed to this function. See each parameter description for details.
 *
 * @par Example
 * - The example of the operation is as follows:
 *   @verbatim
 *    Dimension of features tensor:  [in_active_num, ci]
 *    Dimension of output_grad tensor:  [output_active_num, co]
 *    Dimension of indice_pairs tensor: [kd * kh * kw, 2, in_active_num]
 *    Dimension of filters_grad tensor: [kd, kh, kw, ci, co]
 *   @endverbatim
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

// Group:ThreeNNForward
/*!
 * @brief Returns in \b workspace_size the size of the MLU memory that is used as an extra
 * workspace to optimize the ::mluOpThreeNNForward operation. The size of the extra workspace is
 * based on the given information of the ::mluOpThreeNNForward operation, including the input
 * tensor descriptor \b known_desc. For more information about the workspace, see
 * "Cambricon BANGC OPS User Guide".
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the
 * ::mluOpThreeNNForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] known_desc
 * The descriptor of input data \b known, which contains dimension, data type and data layout.
 * @param[out] workspace_size
 * A host pointer to the returned size of the extra workspace in bytes that is used in
 * the ::mluOpThreeNNForward operation.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par API Dependency
 * - This function must be called before the ::mluOpThreeNNForward function.
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
 * - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetThreeNNForwardWorkspaceSize(const mluOpHandle_t handle,
                                    const mluOpTensorDescriptor_t known_desc,
                                    size_t *workspace_size);

// Group:ThreeNNForward
/*!
 * @brief Finds the closest 3 points of \b unknown among \b known, and outputs \b dist and index
 * \b idx tensor. This function firstly computes dist of each known point to a unknown point, and
 * finds the closest 3 points, and outputs the dist and index of the known point in known dataset.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the
 * ::mluOpThreeNNForward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] unknown_desc
 * The descriptor of input data \b unknown, which contains dimension, data type and data layout.
 * @param[in] unknown
 * Pointer to the MLU memory that stores the unknown tensor.
 * @param[in] known_desc
 * The descriptor of input data \b known, which contains dimension, data type and data layout.
 * @param[in] known
 * Pointer to the MLU memory that stores the known tensor.
 * @param[in] workspace
 * Pointer to the MLU memory that is used as an extra workspace for the ::mluOpThreeNNForward
 * operation. For more information about workspace, see "Cambricon BANGC OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes that is used in the ::mluOpThreeNNForward operation.
 * @param[in] dist2_desc
 * The descriptor of output data \b dist2, which contains dimension, data type and data layout.
 * @param[out] dist2
 * Pointer to the MLU memory that stores the \b dist2 tensor.
 * @param[in] idx_desc
 * The descriptor of output data \b idx, which contains dimension, data type and data layout.
 * @param[out] idx
 * Pointer to the MLU memory that stores the \b idx tensor.
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Formula
 * - None.
 *
 * @par Data Type
 * - The supported data types for unknown tensor \b unknown, known tensor \b known, dist2
 * tensor \b dist2 and idx tensor \b idx. Data types of unknown tensor, known tensor and
 * dist2 should be the same.
 *   - unknown tensor: half, float
 *   - known tensor: half, float
 *   - known tensor: half, float
 *   - idx tensor: int32
 *
 * @par Data Layout
 * - The supported data layouts of \b unknown, \b known, \b dist2 and \b idx:
 *   \p MLUOP_LAYOUT_ARRAY.
 *
 * @par Scale Limitation
 * - The shape of \b unknown, \b dist2 and \b idx should be [b, n, 3].
 * - The shape of \b known should be [b, m, 3].
 * - The shape of \b unknown, \b dist2, \b idx and \b known dims[0](b) should be equal.
 * - The shape of \b unknown, \b dist2, \b idx and \b known dims[2](3) should be equal to 3.
 * - The shape of \b unknown, \b dist2, \b idx and \b known dims[1](n) should be equal and larger
 *   than 0.
 *
 * @par API Dependency
 * - Before calling this function you need to call ::mluOpGetThreeNNForwardWorkspaceSize
 *   to get the extra space size needed in ::mluOpThreeNNForward operation.
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

// Group:IndiceConvolutionForward
/*!
 * @brief Returns in \b workspace_size of the MLU memory which is used as an extra workspace
 * to boost up indice_convolution_forward computation.
 *
 * The size of workspace is deduced from the input including input tensor descriptor
 * \b features_desc, \b filters_desc, \b indice_pairs_desc, output tensor descriptor
 * \b features_out_desc and array indice_num[].
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and queues in the
 * indice_convolution_forward operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] features_desc
 * The descriptor of input features. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] filters_desc
 * The descriptor of filters. For detailed information,
 * see ::mluOpTensorDescriptor_t.
 * @param[in] features_out_desc
 * The descriptor of features_out. For detailed information,
 * see ::mluOptensorDescriptor_t.
 * @param[in] indice_pairs_desc
 * The descriptor of indices mapping pairs of features_in and filters.
 * For detailed information, see ::mluOptensorDescriptor_t.
 * @param[in] features_out_desc
 * The descriptor of features_out. For detailed information,
 * see ::mluOptensorDescriptor_t.
 * @param[in] indice_num
 * Pointer to the host memory that stores the indice pairs number.
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
 * @par API Dependency
 * - Call ::mluOpCreateTensorDescripttor and ::mluOpSetTensorDescriptor before this function
 *   to create and set tensor descriptor \b features_desc, \b filters_desc, \b indice_pairs_desc
 *   and \b features_out_desc.
 * - Output \b workspace_size should later be passed to ::mluOpIndiceConvolutionForward function
 *   to complete computation.
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

// Group:IndiceConvolutionForward
/*!
 * @bried Performs convolution on input sparse tensor \b features with kernel \b filters,
 * then returns the output sparse tensor \b features_out.
 *
 * @param[in] handle
 * Handle to an MLU context that is used to manage MLU devices and queues in the
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
 * The descriptor of indices mappping pairs of input indices and filters location.
 * For deatailed informationm, see ::mluOpTensorDescriptor_t.
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
 * For more information about workspace, see "Cambricon BANGC OPS User Guide".
 * @param[in] workspace_size
 * The size of the extra workspace in bytes.
 * @param[in] features_out_desc
 * The descriptor of the output features tensor. For detailed information,
 * see ::mluOptensorDescriptor_t.
 * @param[out] features_out
 * Pointer to the MLU memory that stores the output tensor.
 *
 * @par Return
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_ARCH_MISMATCH,
 *   ::MLUOP_STATUS_INTERNAL_ERROR, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - This function supports the combination of the following data types:
 *   - input tensor \b featues, \b filters, \b indice_pairs and output tensor \b features_out: half, half, int32, half
 *   - input tensor \b featues, \b filters, \b indice_pairs and output tensor \b features_out: float, float, int32,
 * float
 * - The supported data type of array \b indice_num, scalar \b inverse and \b sub_m is int64.
 *
 * @par Data Layout
 * - This function supports the following tensor layouts:
 *   - features tensor: MLUOP_LAYOUT_ARRAY
 *   - filters tensor: MLUOP_LAYOUT_NDHWC, MLUOP_LAYOUT_NCDHW, MLUOP_LAYOUT_ARRAY
 *   - indice_pairs tensor: MLUOP_LAYOUT_ARRAY
 *   - features_out tensor: MLUOP_LAYOUT_ARRAY
 *
 * @note
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
 * @par Scale Limitation
 * - The \b features and \b features_out are 2D tensor.
 * - The \b filters is 5D tensor.
 * - The \b indice_pairs is 3D tensor.
 * - The dims[1] of \b features equals to input channel of \b filters.
 * - The dims[1] of \b features_out equals to onput channel of \b filters.
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
 * - The function ::mluOpGetIndiceConvolutionForwardWorkspaceSize should be
 *   called before this function to get extra space size.
 *
 * @par Requirements
 * - None.
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

#if defined(__cplusplus)
}
#endif

#endif  // MLUOP_EXAMPLE_H_
