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
#ifndef CORE_MLU_OP_CORE_H_
#define CORE_MLU_OP_CORE_H_

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
/*! @brief Enumeration variables describing function return status.
 */
typedef enum {
  MLUOP_STATUS_SUCCESS         = 0, /*!< The operation was successfully completed. */
  MLUOP_STATUS_NOT_INITIALIZED = 1,
  /*!< MLUOP library was not initialized properly, which is usually caused by the
       failure of calling ::mluOpCreate, ::mluOpCreateTensorDescriptor or
     ::mluOpSetTensorDescriptor.
       Such error is usually due to incompatible MLU device or invalid driver environment.
       Notice that ::mluOpCreate should be called prior to any other mluOp functions.*/
  MLUOP_STATUS_ALLOC_FAILED = 2,
  /*!< This error occurs when the resource allocation failed, usually caused by the failure
       of cnMallocHost, probably because of the exceeded memory usage. Please make sure that
       the memory allocated previously is deallocated as much as possible.*/
  MLUOP_STATUS_BAD_PARAM = 3,
  /*!< Invalid value or parameters passed to the function, including data type, layout,
       dimensions, etc.*/
  MLUOP_STATUS_INTERNAL_ERROR = 4,
  /*!< Error occurred inside of the function, which may indicate an internal error or bug in
       the library. This error is usually due to the failure of cnrtMemcpyAsync.
       Please check whether the memory passed to the function was deallocated before the completion
       of the routine.*/
  MLUOP_STATUS_ARCH_MISMATCH = 5,
  /*!< Invalid MLU device which was not supported by current function.*/
  MLUOP_STATUS_EXECUTION_FAILED = 6,
  /*!< Error occurred when the function failed to execute on MLU device due to multiple reasons.
       You can check whether the hardware environment, driver version and other prerequisite
       libraries are correctly installed. For more information about prerequisite libraries,
       see "Cambricon MLUOP User Guide".*/
  MLUOP_STATUS_NOT_SUPPORTED = 7,
  /*!< Error when the requested functionality was not supported in
       this version but would be supported in the future. */
  MLUOP_STATUS_NUMERICAL_OVERFLOW = 8,
  /*!< Numerical overflow occurred when executing the function,
       which is usually due to large scale or inappropriate range of value of input tensor.*/
} mluOpStatus_t;

/******************************************************************************
 * MLUOP Tensor Layout
 ******************************************************************************/
/*!
 * @brief Enumeration variables describing the data layouts in MLUOP.
 *
 * The data can be defined in three, four, or five dimensions.
 *
 * Take images for example, the format of the data layout can be NCHW:
 * - N: The number of images.
 * - C: The number of image channels.
 * - H: The height of images.
 * - W: The weight of images.
 *
 * Take sequence for example, the format of the data layout can be TNC:
 * - T: The timing steps of sequence.
 * - N: The batch size of sequence.
 * - C: The alphabet size of sequence.
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
/*! @brief Enumeration variables describing the data types in MLUOP. */
typedef enum {
  MLUOP_DTYPE_INVALID         = 0,   /*!< The data is an invalid data type. */
  MLUOP_DTYPE_HALF            = 1,   /*!< The data is a 16-bit floating-point data type. */
  MLUOP_DTYPE_FLOAT           = 2,   /*!< The data is a 32-bit floating-point data type. */
  MLUOP_DTYPE_DOUBLE          = 3,   /*!< The data is a 64-bit floating-point data type. */
  MLUOP_DTYPE_INT8            = 4,   /*!< The data is an 8-bit signed integer data type. */
  MLUOP_DTYPE_INT16           = 5,   /*!< The data is a 16-bit signed integer data type. */
  MLUOP_DTYPE_INT32           = 6,   /*!< The data is a 32-bit signed integer data type. */
  MLUOP_DTYPE_INT64           = 7,   /*!< The data is a 64-bit signed integer data type. */
  MLUOP_DTYPE_UINT8           = 8,   /*!< The data is an 8-bit unsigned integer data type. */
  MLUOP_DTYPE_UINT16          = 9,   /*!< The data is a 16-bit unsigned integer data type. */
  MLUOP_DTYPE_UINT32          = 10,  /*!< The data is a 32-bit unsigned integer data type. */
  MLUOP_DTYPE_UINT64          = 11,  /*!< The data is a 64-bit unsigned integer data type. */
  MLUOP_DTYPE_BOOL            = 12,  /*!< The data is a boolean data type. */
  MLUOP_DTYPE_COMPLEX_HALF    = 13,  /*!< The data is a 32-bit complex number of two fp16. */
  MLUOP_DTYPE_COMPLEX_FLOAT   = 14,  /*!< The data is a 64-bit complex number of two fp32. */
} mluOpDataType_t;

/*!
 * @brief Enumeration variables describing the options that can help choose
 *        the best suited algorithm used for implementation of the activation
 *        and accumulation operations.
 **/
typedef enum {
  MLUOP_COMPUTATION_FAST = 0,
  /*!< Implementation with the fastest algorithm and lower precision.*/
  MLUOP_COMPUTATION_HIGH_PRECISION = 1,
  /*!< Implementation with the high-precision algorithm regardless the performance.*/
} mluOpComputationPreference_t;

/*!
 * @brief Enumeration variables describing that the rounding mode of quantization conversion.
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

/******************************************************************************
 * MLUOP Runtime Management
 ******************************************************************************/

/*!
 * @struct mluOpContext
 * @brief The \b mluOpContext is a structure describing the MLUOP context.
 *
 *
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
 * For more information, see "Cambricon MLUOP User Guide".
 *
 */
typedef struct mluOpContext *mluOpHandle_t;

/*! The descriptor of the collection of tensor which is used in the RNN operation, such as weight,
 * bias, etc.
 *  You need to call the ::mluOpCreateTensorSetDescriptor function to create a descriptor, and
 *  call the ::mluOpInitTensorSetMemberDescriptor to set the information about each tensor in
 *  the tensor set. If the data type of the tensor in the tensor set is in fixed-point data type,
 *  call ::mluOpInitTensorSetMemberDescriptorPositionAndScale function to set quantization
 * parameters.
 *  At last, you need to destroy the descriptor at the end with the
 * ::mluOpDestroyTensorSetDescriptor
 *  function. */
typedef struct mluOpTensorSetStruct *mluOpTensorSetDescriptor_t;

/*!
 *  @brief Initializes the MLUOP library and creates a handle \b handle to a structure
 *  that holds the MLUOP library context. It allocates hardware resources on the host
 *  and device. You need to call this function before any other MLUOP functions.
 *
 *  You need to call the ::mluOpDestroy function to release the resources later.
 *
 *  @param[out] handle
 *    Output. Pointer to the MLUOP context that is used to manage MLU devices and
 *    queues. For detailed information, see ::mluOpHandle_t.
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
 */
mluOpStatus_t MLUOP_WIN_API mluOpCreate(mluOpHandle_t *handle);

/*!
 *  @brief Updates the MLUOP context information that holds by the \b handle. This function
 *  should be called if you call Cambriocn Driver API cnSetCtxConfigParam to set the context
 *  information. The related context information will be synchronized to MLUOP with this function.
 *  For detailed information, see Cambricon Driver API Developer Guide.
 *
 *  @param[in] handle
 *    Input. Pointer to the MLUOP context that is used to manage MLU devices and
 *    queues. For detailed information, see ::mluOpHandle_t.
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
 */
mluOpStatus_t MLUOP_WIN_API mluOpUpdateContextInformation(mluOpHandle_t handle);

/*!
 *  @brief Releases the resources of the specified MLUOP handle \b handle that was
 *  created by the ::mluOpCreate function.
 *  It is usually the last call to destroy the handle to the MLUOP handle.
 *
 *  @param[in] handle
 *    Input. Pointer to the MLU devices that holds information to be destroyed.
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
 */
mluOpStatus_t MLUOP_WIN_API mluOpDestroy(mluOpHandle_t handle);

/*!
 *  @brief Sets the runtime queue \b queue in the handle \b handle. The queue is used to
 *  launch kernels or to synchronize to this queue.
 *
 *  Before setting a queue \b queue, you need to call the ::mluOpCreate function to initialize
 *  MLUOP library, and call the cnrtCreateQueue function to create a queue \b queue.
 *
 *  @param[in] handle
 *    Input. Handle to a MLUOP context that is used to manage MLU devices and
 *    queues. For detailed information, see ::mluOpHandle_t.
 *  @param[in] queue
 *    Input. The runtime queue to be set to the MLUOP handle.
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
 */
mluOpStatus_t MLUOP_WIN_API mluOpSetQueue(mluOpHandle_t handle, cnrtQueue_t queue);

/*!
 *  @brief Retrieves the queue \b queue that was previously set to the handle \b handle.
 *
 *  @param[in] handle
 *    Input. Handle to a MLUOP context that is used to manage MLU devices and
 *    queues. For detailed information, see ::mluOpHandle_t.
 *  @param[out] queue
 *    Output. Pointer to the queue that was previously set to the specified handle.
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
 */
mluOpStatus_t MLUOP_WIN_API mluOpGetQueue(mluOpHandle_t handle, cnrtQueue_t *queue);

/*!
 *  @brief Converts the MLUOP enumerated status code to ASCIIZ static string and returns
 *  a pointer to the MLU memory that holds information about ASCIIZ static string with the status
 * name.
 *  For example, when the input argument is
 *  ::MLUOP_STATUS_SUCCESS, the returned string is MLUOP_STATUS_SUCCESS. When an invalid status
 * value
 * is passed
 *  to the function, the returned string is ::MLUOP_STATUS_BAD_PARAM.
 *
 *  @param[in] status
 *    Input. The MLUOP enumerated status code.
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
 *
 */
const char *mluOpGetErrorString(mluOpStatus_t status);

/******************************************************************************
 * MLUOP Data Structure: Descriptor
 * The struct represent neural, weight and the neural-network layer
 ******************************************************************************/
/*! The descriptor of a tensor that holds the information including tensor
 *  layout, data type, the number of dimensions, shape and strides.
 *
 *  You need to call the ::mluOpCreateTensorDescriptor function to create a descriptor,
 *  and call the ::mluOpSetTensorDescriptor function or the ::mluOpSetTensorDescriptorEx
 *  function to set the tensor information to the descriptor. Also, you need to destroy
 *  the MLUOP context at the end with the ::mluOpDestroyTensorDescriptor function.
 */
// Group:QuantizeRoundMode
/*!
 *  @brief Updates the specific rounding mode of MLUOP context information that holds by the \b handle. This function
 *  should be called if you want to change the mluop rounding mode that used to cumulate the results.
 *  For detailed information, see Cambricon Driver API Developer Guide.
 *
 *  @param[in] handle
 *    Input. Pointer to the MLUOP context that is used to manage MLU devices and
 *    queues. For detailed information, see ::mluopHandle_t.
 *  @param[in] round_mode
 *    Input. The rounding mode of quantization conversion to be set to the MLUOP handle.
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - On MLU200 series:
 *    You can't set MLUOP_ROUND_HALF_TO_EVEN for the rounding mode because the hardware does not support it.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API mluOpSetQuantizeRoundMode(mluOpHandle_t handle,
                                                      mluOpQuantizeRoundMode_t round_mode);

// Group:QuantizeRoundMode
/*!
 *  @brief Retrieves the rounding mode of a specific MLUOP context.
 *
 *  @param[in] handle
 *    Input. Pointer to the MLUOP context that is used to manage MLU devices and
 *    queues. For detailed information, see ::mluopHandle_t.
 *
 *  @param[out] round_mode
 *    Output. the rounding mode of quantization conversion that was previously set to the specified handle.
 *
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - The default round mode of default initialized mluopHandle_t is MLUOP_ROUND_TO_EVEN.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API mluOpGetQuantizeRoundMode(mluOpHandle_t handle,
                                                      mluOpQuantizeRoundMode_t *round_mode);

typedef struct mluOpTensorStruct *mluOpTensorDescriptor_t;

/*!
 *  @brief Creates a tensor descriptor pointed by \b desc that holds the dimensions, data type,
 *  and layout of input tensor. If the input tensor is in fixed-point data type,
 *  the ::mluOpSetTensorDescriptorPositionAndScale function or the
 * ::mluOpSetTensorDescriptorPosition
 *  function needs to be called to set quantization parameters.
 *
 *  The ::mluOpDestroyTensorDescriptor function needs to be called to destroy the
 *  tensor descriptor later.
 *
 *  @param[in] desc
 *    Input. Pointer to the struct that holds information about the tensor descriptor.
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
 */
mluOpStatus_t MLUOP_WIN_API mluOpCreateTensorDescriptor(mluOpTensorDescriptor_t *desc);

/*!
 *  @brief Initializes the tensor descriptor pointed by \b desc that is previously created
 *  with the ::mluOpCreateTensorDescriptor function, and sets the information about
 *  the dimensions, data type, and layout of the input tensor.
 *
 *  If ::mluOpSetTensorDescriptor is called, you do not need to specify the strides of all
 *  dimensions. The strides are inferred by parameters passed to this function. Also, the data
 *  will be treated as contiguous in memory with no padding between dimensions. To specify the
 *  strides of all dimensions, you can call ::mluOpSetTensorDescriptorEx. But the data might not
 *  be treated as contiguous in memory.
 *
 *  @param[in] desc
 *    Input. The descriptor of the input tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
 *  @param[in] layout
 *    Input. The layout of the input tensor. For detailed information, see ::mluOpTensorLayout_t.
 *  @param[in] dtype
 *    Input. The data type of the input tensor. For detailed information, see ::mluOpDataType_t.
 *  @param[in] dimNb
 *    Input. The number of dimensions in the input tensor of the initialized operation.
 *  @param[in] dimSize
 *    Input. An array that contains the size of the tensor for each dimension.
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
 */
mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptor(mluOpTensorDescriptor_t desc,
                                                     mluOpTensorLayout_t layout,
                                                     mluOpDataType_t dtype,
                                                     int dimNb,
                                                     const int dimSize[]);

/*!
 *  @brief Resets the tensor descriptor pointed by \b desc that is previously created
 *  with the ::mluOpCreateTensorDescriptor function.
 *
 *  If ::mluOpResetTensorDescriptor is called, all the information about the tensor will be reset to
 *  initial value, which means layout is MLUOP_LAYOUT_ARRAY, dtype is MLUOP_DTYPE_FLOAT, dimsNb is
 * 0,
 *  and dimSize points to an \b MLUOP_DIM_MAX-dimension array.
 *
 *  @param[in] desc
 *    Input. The descriptor of the tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - This function is used to avoid memory leaks when more than one ::mluOpSetTensorDescriptor
 *    function is called. You should call this function before calling another
 *    ::mluOpSetTensorDescriptor
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API mluOpResetTensorDescriptor(mluOpTensorDescriptor_t desc);

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
 *    Input. The descriptor of the input tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
 *  @param[in] layout
 *    Input. The layout of the input tensor. For detailed information, see ::mluOpTensorLayout_t.
 *  @param[in] dtype
 *    Input. The data type of the input tensor. For detailed information, see ::mluOpDataType_t.
 *  @param[in] dimNb
 *    Input. The number of dimensions in the input tensor of the initialized operation.
 *  @param[in] dimSize
 *    Input. An array that contains the size of the tensor for each dimension.
 *  @param[in] dimStride
 *    Input. An array that contains the stride of the tensor for each dimension.
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
 */
mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorEx(mluOpTensorDescriptor_t desc,
                                                       mluOpTensorLayout_t layout,
                                                       mluOpDataType_t dtype,
                                                       int dimNb,
                                                       const int dimSize[],
                                                       const int dimStride[]);

/*!
 *  @brief Sets the on-chip data type to the descriptor of a tensor \b desc.
 *  The on-chip data type \b onchip_dtype can be different from the off-chip data type of the
 * tensor.
 *  This function is optional. If the on-chip data type is not set with this function, the
 *  ::MLUOP_STATUS_BAD_PARAM data type is used by default.
 *
 *  @param[in] desc
 *    Input. The descriptor of input tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
 *  @param[in] onchip_dtype
 *    Input. The on-chip data type of the tensor used in the operations that support fixed-point
 * computing.
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - The on-chip data type is only used on the operations that support fixed-point computing. It
 *    has no effect on other operations. If you call this function to get on-chip data type for an
 *    operation that does not support fixed-point computing, ::MLUOP_STATUS_BAD_PARAM is returned.
 * To check
 *    if an operation supports fixed-point computing, see the detailed description of the operation.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorOnchipDataType(mluOpTensorDescriptor_t desc,
                                                                   mluOpDataType_t onchip_dtype);

/*!
 * @brief Sets the \b position factor to the descriptor \b desc of fixed-point data in
 * fixed-point quantization. It is used in ::MLUOP_QUANTIZE_POSITION mode.
 *
 * @param[in] desc
 *   Input. The descriptor of the tensor. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[in] position
 *   Input. A scalar of fixed position factor that is used for quantization.
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
*/
mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorPosition(mluOpTensorDescriptor_t desc,
                                                             int position);

/*!
 * @brief Sets the \b position and \b scale factors to the descriptor of fixed-point data in
 * fixed-point quantization. It is used in ::MLUOP_QUANTIZE_POSITION_SCALE mode.
 *
 * @param[in] desc
 *   Input. The descriptor of the tensor. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[in] position
 *   Input. A scalar of fixed position factor that is used for quantization.
 * @param[in] scale
 *   Input. A scalar of scale factor that is used for quantization.
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
*/
mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorPositionAndScale(mluOpTensorDescriptor_t desc,
                                                                     int position,
                                                                     float scale);
/*!
 * @brief Sets the \b position, \b scale and \b offset factors to the descriptor of fixed-point
 * data in fixed-point quantization. It is used in ::MLUOP_QUANTIZE_POSITION_SCALE_OFFSET mode.
 *
 * @param[in] desc
 *   Input. The descriptor of the tensor. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[in] position
 *   Input. A scalar of fixed position factor that is used for quantization.
 * @param[in] scale
 *   Input. A scalar of scale factor that is used for quantization.
 * @param[in] offset
 *   Input. A scalar of offset factor that is used for quantization.
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
*/
mluOpStatus_t MLUOP_WIN_API
mluOpSetTensorDescriptorPositionScaleAndOffset(mluOpTensorDescriptor_t desc,
                                               int position,
                                               float scale,
                                               int offset);

/*!
 *  @brief Retrieves a tensor descriptor \b desc that is previously created with the
 *  ::mluOpCreateTensorDescriptor function, and sets the information about the dimensions,
 *  data type, and layout of input tensor.
 *
 *  @param[in] desc
 *    Input. The descriptor of the input tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
 *  @param[out] layout
 *    Output. Pointer to the host memory that holds information about the layout of the input
 * tensor.
 *  For detailed information, see ::mluOpTensorLayout_t.
 *  @param[out] dtype
 *    Output. Pointer to the host memory that holds information about the data type of the input
 * tensor.
 *  For detailed information, see ::mluOpDataType_t.
 *  @param[out] dimNb
 *    Output. Pointer to the host memory that holds information about the dimension of input tensor.
 *  @param[out] dimSize
 *    Output. An array that contains the size of the tensor for each dimension.
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
 */
mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptor(const mluOpTensorDescriptor_t desc,
                                                     mluOpTensorLayout_t *layout,
                                                     mluOpDataType_t *dtype,
                                                     int *dimNb,
                                                     int dimSize[]);

/*!
 *  @brief Retrieves a tensor descriptor \b desc that is previously created with the
 *  ::mluOpCreateTensorDescriptor and sets the information about the dimensions, data type,
 *  stride and layout of input tensor with ::mluOpSetTensorDescriptorEx.
 *
 *  @param[in] desc
 *    Input. The descriptor of the input tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
 *  @param[out] layout
 *    Output. Pointer to the host memory that holds information about the layout of the input
 * tensor.
 *  For detailed information, see ::mluOpTensorLayout_t.
 *  @param[out] dtype
 *    Output. Pointer to the host memory that holds information about the data type of the input
 * tensor.
 *  For detailed information, see ::mluOpDataType_t.
 *  @param[out] dimNb
 *    Output. Pointer to the host memory that holds information about the dimension of input tensor.
 *  @param[out] dimSize
 *    Output. An array that contains the size of the tensor for each dimension.
 *  @param[out] dimStride
 *    Output. An array that contains the stride of the tensor for each dimension.
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
 */
mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptorEx(const mluOpTensorDescriptor_t desc,
                                                       mluOpTensorLayout_t *layout,
                                                       mluOpDataType_t *dtype,
                                                       int *dimNb,
                                                       int dimSize[],
                                                       int dimStride[]);

/*!
 *  @brief Retrieves the number of elements according to the input descriptor \b desc. You
 *  need to call the ::mluOpSetTensorDescriptor function first to create a tensor descriptor
 *  before calling this function.
 *
 *  @param[in] desc
 *    Input. The descriptor of input tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
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
size_t MLUOP_WIN_API mluOpGetTensorElementNum(const mluOpTensorDescriptor_t desc);

/*!
 *  @brief Retrieves the on-chip data type of a tensor descriptor \b desc set by
 *  ::mluOpSetTensorDescriptorOnchipDataType.
 *  If the on-chip data type is not set with the ::mluOpSetTensorDescriptorOnchipDataType function,
 *  the ::MLUOP_STATUS_BAD_PARAM is returned.
 *
 *  @param[in] desc
 *    Input. The descriptor of input tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
 *  @param[in] onchip_dtype
 *    Input. Pointer to the MLU memory that holds information about the on-chip data type of the
 * tensor.
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - The on-chip data type is only used on the operations that support fixed-point computing. It
 *    has no effect on other operations. If you call this function to get on-chip data type for an
 *    operation that does support fixed-point computing, ::MLUOP_STATUS_BAD_PARAM is returned. To
 * check
 *    if an operation supports fixed-point computing, see the detailed description of the operation.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 *
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorDescriptorOnchipDataType(const mluOpTensorDescriptor_t desc,
                                       mluOpDataType_t *onchip_dtype);

/*!
 * @brief Gets the \b position factor to the descriptor \b desc of fixed-point data in
 * fixed-point quantization.
 *
 * @param[in] desc
 *   Input. The descriptor of the tensor. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[out] position
 *   Output. A host pointer of fixed position factor that is used for quantization.
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
*/
mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptorPosition(const mluOpTensorDescriptor_t desc,
                                                             int *position);

/*!
 *  @brief Gets the position and scale factors of a tensor descriptor \b desc used in
 *  fixed-point quantization.
 *
 *  @param[in] desc
 *    Input. The descriptor of the input tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
 *  @param[out] position
 *    Output. Pointer to the MLU memory that holds information about fixed position
 *    used for quantization.
 *  @param[out] scale
 *    Output. Pointer to the MLU memory that holds information about scale factor
 *    used for quantization.
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
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorDescriptorPositionAndScale(const mluOpTensorDescriptor_t desc,
                                         int *position,
                                         float *scale);
/*!
 * @brief Gets the \b position, \b scale and \b offset factors to the descriptor \b desc of
 * fixed-point data in fixed-point quantization.
 *
 * @param[in] desc
 *   Input. The descriptor of the tensor. For detailed information,
 *   see ::mluOpTensorDescriptor_t.
 * @param[out] position
 *   Output. A host pointer of fixed position factor that is used for quantization.
 * @param[out] scale
 *   Output. A host pointer of scale factor that is used for quantization.
 * @param[in] offset
 *   Output. A host pointer of offset factor that is used for quantization.
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
*/
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorDescriptorPositionScaleAndOffset(const mluOpTensorDescriptor_t desc,
                                               int *position,
                                               float *scale,
                                               int *offset);

/*!
 *  @brief Destroies a tensor descriptor that was created by
 *         ::mluOpCreateTensorDescriptor.
 *
 *  @param[in] desc
 *    Input. A tensor descriptor created by ::mluOpCreateTensorDescriptor.
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
 */
mluOpStatus_t MLUOP_WIN_API mluOpDestroyTensorDescriptor(mluOpTensorDescriptor_t desc);

#if defined(__cplusplus)
}
#endif

#endif  // CORE_MLU_OP_CORE_H_
