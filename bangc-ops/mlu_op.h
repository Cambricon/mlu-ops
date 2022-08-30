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
#define MLUOP_MINOR 1
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
 * @brief Enumeration variables describing that the atomics mode in MLUOP.
 */
typedef enum {
  MLUOP_ATOMICS_NOT_ALLOWED = 1,
  /*!< The atomics is not allowed to cumulate results.*/
  MLUOP_ATOMICS_ALLOWED = 2,
  /*!< The atomics is allowed to cumulate results */
} mluOpAtomicsMode_t;

/*!
 * @brief Enumeration variables describing that the rounding mode of
 * quantization conversion.
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
 *  bias, etc.
 *  You need to call the ::mluOpCreateTensorSetDescriptor function to create a descriptor, and
 *  call the ::mluOpInitTensorSetMemberDescriptor to set the information about each tensor in
 *  the tensor set. If the data type of the tensor in the tensor set is in fixed-point data type,
 *  call ::mluOpInitTensorSetMemberDescriptorPositionAndScale function to set quantization
 *  parameters.
 *  At last, you need to destroy the descriptor at the end with the
 *  ::mluOpDestroyTensorSetDescriptor
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
 *  name.
 *  For example, when the input argument is
 *  ::MLUOP_STATUS_SUCCESS, the returned string is MLUOP_STATUS_SUCCESS. When an invalid status
 *  value is passed to the function, the returned string is ::MLUOP_STATUS_BAD_PARAM.
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

// Group:Tensor
/*!
 *  @brief Get the size of a data type in ::mluOpDataType_t.
 *
 *  @param[in] data_type
 *    Input. The data type. For detailed information, see ::mluOpDataType_t.
 *  @param[out] size
 *    Output. Host pointer to the size of the data type.
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
mluOpStatus_t MLUOP_WIN_API mluOpGetSizeOfDataType(mluOpDataType_t data_type,
                                                   size_t *size);

// Group:Version Management
/*!
 *  @brief Retrieves the version of MLUOP library. The version of MLUOP
 *  is composed of \b major, \b minor and \b patch. For instance, major = 1,
 *  minor = 2, patch = 3, the version of MLUOP library is 1.2.3.
 *
 *  @param[in] major
 *    Input. A pointer to scale factor that gets the major version of MLUOP
 *    library.
 *  @param[in] minor
 *    Input. A pointer to scale factor that gets the minor version of MLUOP
 *    library.
 *  @param[in] patch
 *    Input. A pointer to scale factor that gets the patch version of MLUOP
 *    library.
 *
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
void mluOpGetLibVersion(int *major, int *minor, int *patch);

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
 *  ::mluOpSetTensorDescriptorPosition
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
 *    Input. An array of pointers to the structs that hold information about the
 *    tensor descriptor.
 *  @param[in] desc_num
 *    Input. The length of the input array \b group_desc.
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @par API Dependency
 *  - The ::mluOpDestoryTensorDescriptor function needs to be called for each
 *    descriptor to destory all tensors in group_desc or the
 *    ::mluOpDestoryGroupTensorDescriptors needs to be called to destory the all
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
 */
mluOpStatus_t MLUOP_WIN_API mluOpCreateGroupTensorDescriptors(
    mluOpTensorDescriptor_t *group_desc[], const int desc_num);

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
 *    Input. An array of pointers to the struct that hold information about the
 *    tensor descriptor.
 *  @param[in] group_layout
 *    Input. An array that stores the layouts of all input tensors. For detailed
 *    information, see ::mluOpTensorLayout_t.
 *  @param[in] group_dtype
 *    Input. An array that stores the data types of all input tensors. For
 *    detailed information, see ::mluOpDataType_t.
 *  @param[in] group_dimNb
 *    Input. An array that stores the dimensions of all input tensors.
 *  @param[in] group_dimSize
 *    Input. An array that stores the size of each dimension of all tensors.
 *  @param[in] desc_num
 *    Input. The length of the input array \b group_desc.
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
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetGroupTensorDescriptors(mluOpTensorDescriptor_t *group_desc[],
                               const mluOpTensorLayout_t group_layout[],
                               const mluOpDataType_t group_dtype[],
                               const int group_dimNb[],
                               const int group_dimSize[],
                               const int desc_num);

/*!
 *  @brief Resets the tensor descriptor pointed by \b desc that is previously
 *  created with the ::mluOpCreateTensorDescriptor function.
 *  If ::mluOpResetTensorDescriptor is called, all the information about the tensor will be reset to
 *  initial value, which means layout is MLUOP_LAYOUT_ARRAY, dtype is MLUOP_DTYPE_FLOAT, dimsNb is
 *  0, and dimSize points to an \b MLUOP_DIM_MAX-dimension array.
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
 *  tensor.
 *  This function is optional. If the on-chip data type is not set with this function, the
 *  ::MLUOP_STATUS_BAD_PARAM data type is used by default.
 *
 *  @param[in] desc
 *    Input. The descriptor of input tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
 *  @param[in] onchip_dtype
 *    Input. The on-chip data type of the tensor used in the operations that support fixed-point
 *    computing.
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
 */
mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorOnchipDataType(mluOpTensorDescriptor_t desc,
                                                                   mluOpDataType_t onchip_dtype);
 
/*! 
 *  @brief Sets the \b position factor to the descriptor \b desc of fixed-point data in
 *  fixed-point quantization. It is used in ::MLUOP_QUANTIZE_POSITION mode.
 * 
 *  @param[in] desc
 *    Input. The descriptor of the tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
 *  @param[in] position
 *    Input. A scalar of fixed position factor that is used for quantization.
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
mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorPosition(mluOpTensorDescriptor_t desc,
                                                             int position);

/*! 
 *  @brief Sets the \b position and \b scale factors to the descriptor of fixed-point data in
 *  fixed-point quantization. It is used in ::MLUOP_QUANTIZE_POSITION_SCALE mode.
 * 
 *  @param[in] desc
 *    Input. The descriptor of the tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
 *  @param[in] position
 *    Input. A scalar of fixed position factor that is used for quantization.
 *  @param[in] scale
 *    Input. A scalar of scale factor that is used for quantization.
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
mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorPositionAndScale(mluOpTensorDescriptor_t desc,
                                                                     int position,
                                                                     float scale);
/*! 
 *  @brief Sets the \b position, \b scale and \b offset factors to the descriptor of fixed-point
 *  data in fixed-point quantization. It is used in ::MLUOP_QUANTIZE_POSITION_SCALE_OFFSET mode.
 * 
 *  @param[in] desc
 *    Input. The descriptor of the tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
 *  @param[in] position
 *    Input. A scalar of fixed position factor that is used for quantization.
 *  @param[in] scale
 *    Input. A scalar of scale factor that is used for quantization.
 *  @param[in] offset
 *    Input. A scalar of offset factor that is used for quantization.
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
 *    tensor.
 *    For detailed information, see ::mluOpTensorLayout_t.
 *  @param[out] dtype
 *    Output. Pointer to the host memory that holds information about the data type of the input
 *    tensor.
 *    For detailed information, see ::mluOpDataType_t.
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
 *    tensor.
 *    For detailed information, see ::mluOpTensorLayout_t.
 *  @param[out] dtype
 *    Output. Pointer to the host memory that holds information about the data type of the input
 *    tensor.
 *    For detailed information, see ::mluOpDataType_t.
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
 *    tensor.
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
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorDescriptorOnchipDataType(const mluOpTensorDescriptor_t desc,
                                       mluOpDataType_t *onchip_dtype);

/*!
 *  @brief Gets the \b position factor to the descriptor \b desc of fixed-point data in
 *  fixed-point quantization.
 * 
 *  @param[in] desc
 *    Input. The descriptor of the tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
 *  @param[out] position
 *    Output. A host pointer of fixed position factor that is used for quantization.
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
 *  @brief Gets the \b position, \b scale and \b offset factors to the descriptor \b desc of
 *  fixed-point data in fixed-point quantization.
 * 
 *  @param[in] desc
 *    Input. The descriptor of the tensor. For detailed information,
 *    see ::mluOpTensorDescriptor_t.
 *  @param[out] position
 *    Output. A host pointer of fixed position factor that is used for quantization.
 *  @param[out] scale
 *    Output. A host pointer of scale factor that is used for quantization.
 *  @param[in] offset
 *    Output. A host pointer of offset factor that is used for quantization.
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
mluOpGetTensorDescriptorPositionScaleAndOffset(const mluOpTensorDescriptor_t desc,
                                               int *position,
                                               float *scale,
                                               int *offset);

/*!
 *  @brief Destroies a tensor descriptor that was created by 
 *  ::mluOpCreateTensorDescriptor.
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
// Group:Tensor
/*!
 *  @brief Destroys a group of tensor descriptors that was created by
 *  ::mluOpCreateTensorDescriptor or ::mluOpCreateGroupTensorDescriptors.
 *
 *  @param[in] group_desc
 *    Input. An array of pointers to the struct that hold information about the
 *    tensor descriptor.
 *  @param[in] desc_num
 *    Input. The length of the input array \b group_desc.
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
mluOpStatus_t MLUOP_WIN_API mluOpDestroyGroupTensorDescriptors(
    mluOpTensorDescriptor_t *group_desc[], const int desc_num);

// Group:TensorSet
/*!
 *  @brief Creates a descriptor \b tensorSetDesc of tensor set that holds a
 *  series of tensors. The number of tensors of tensor set is jointly determined
 *  by \b setDimNb and \b setDimSize. Use ::mluOpInitTensorSetMemberDescriptor to
 *  set information for descriptor and ::mluOpDestroySeqDataDescriptor function
 *  to destroy the tensor set descriptor.
 *
 *  @param[out] tensorSetDesc
 *    Input. Pointer to the memory that holds information about the descriptor
 *    of tensor set.
 *  @param[in] setDimNb
 *    Input. The number of dimensions of the tensor set.
 *  @param[in] setDimSize
 *    Input. An array that contains the number of the tensors for each dimension
 *    of the tensor set.
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
 */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateTensorSetDescriptor(mluOpTensorSetDescriptor_t *tensorSet,
                               const int setDimNb,
                               const int setDimSize[]);

// Group:TensorSet
/*!
 *  @brief Retrieves a tensor set descriptor \b tensorSetDesc that is previously
 *  created with the ::mluOpCreateTensorSetDescriptor function.
 *
 *  @param[in] tensorSetDesc
 *    Input. The descriptor of the tensor set. For detailed information,
 *    see ::mluOpSeqDataDescriptor_t.
 *  @param[out] setDimNb
 *    Output. The number of dimensions of the tensor set.
 *  @param[out] setDimSize
 *    Output. An array that contains the number of the tensor for each dimension
 *    of the tensor set.
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
 */
mluOpStatus_t MLUOP_WIN_API mluOpGetTensorSetDescriptor(
    mluOpTensorSetDescriptor_t tensorSetDesc, int *setdimNb, int setDimSize[]);

// Group:TensorSet
/*!
 *  @brief Destroys a tensor set descriptor \b tensorSetDesc that is previously
 *  created by ::mluOpCreateTensorSetDescriptor.
 *
 *  @param[in] desc
 *    Input. A tensor descriptor created by ::mluOpCreateTensorSetDescriptor.
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
 *    Input. The descriptor of the tensor set. For detailed information,
 *    see ::mluOpTensorSetDescriptor_t.
 *  @param[in] setDimNb
 *    Input. The number of dimensions of the tensor set.
 *  @param[in] tensorIndex
 *    Input. An array that contains the index of each dimension of a member
 *    tensor to be initialized in the tensor set.
 *  @param[in] layout
 *    Input. The layout of the member tensor. For detailed information, see
 *    ::mluOpTensorLayout_t.
 *  @param[in] dtype
 *    Input. The data type of the member tensor. For detailed information, see
 *    ::mluOpDataType_t.
 *  @param[in] dimNb
 *    Input. The number of dimensions in the member tensor.
 *  @param[in] dimSize
 *    Input. An array that contains the size of the member tensor for each
 *    dimension.
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
 *  fixed-point quantization with scale factor quantization method. For more
 *  information about quantization, see "Cambricon mluOp User Guide".
 *
 *  @param[in] tensorSetDesc
 *    Input. The descriptor of tensor set. For detailed information,
 *    see ::mluOpTensorSetDescriptor_t.
 *  @param[in] setDimNb
 *    Input. The number of dimensions of the tensor set.
 *  @param[in] tensorIndex
 *    Input. An array that contains the position index information of the member
 *    tensor in the tensor set.
 *  @param[in] position
 *    Input. A position of fixed position factor that is used for
 *    quantification.
 *  @param[in] scale
 *    Input. A scalar of scale factor that is used for quantification.
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
 *    Input. The descriptor of tensor set. For detailed information,
 *    see ::mluOpTensorSetDescriptor_t.
 *  @param[Out] sizeInBytes
 *    Output. Size in bytes of tensor set.
 *    You can allocate MLU memory for the tensor set with this value.
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
 */
mluOpStatus_t MLUOP_WIN_API 
mluOpGetTensorSetDescriptorSize(mluOpTensorSetDescriptor_t tensorSetDesc, 
                                int *sizeInBytes);

// Group:TensorSet
/*!
 *  @brief Retrieves the tensor descriptor in the tensor set and the
 *  corresponding offset address based on the entire block of MLU memory through
 *  the index \b tensorIndex.
 *
 *  @param[in] tensorSetDesc
 *    Input. The descriptor of tensor set. For detailed information,
 *    see ::mluOpTensorSetDescriptor_t.
 *  @param[in] tensorIndex
 *    Input. An array that contains the position information of the member
 *    tensor in the tensor set.
 *  @param[in] data
 *    Input. Pointer to the MLU memory that is described by \b tensorSetDesc.
 *  @param[out] tensorDesc
 *    Output. Pointer to the host member. It is member tensor descriptor that
 *    indexed by \b tensorIndex in the tensor set. \b *tensorDesc contains tensor
 *    member information about dimensions, layout, data type, position and scale.
 *  @param[out] dataAddrInDevice
 *    Output. Pointer to the MLU memory that indexed by \b tensorIndex in the
 *    whole block of data \b dataAddrInDevice.
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
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorAndDataFromTensorSet(mluOpTensorSetDescriptor_t tensorSetDesc,
                                   const int setDimNb,
                                   const int tensorIndex[],
                                   void *data,
                                   mluOpTensorDescriptor_t *tensorDesc,
                                   void **dataAddrInDevice);

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
 *  - The coordinates of the input boxes must be sorted clockwise or
 *    counterclockwise. If the coordinates of the boxes are out of order,
 *    the calculation result is not guaranteed and is consistent with the
 *    calculation result of the competitor operator.
 *  - If there are cases with the same score in the input boxes, the output
 *    results may be inconsistent with the results of competing products.
 *  - The number of input boxes on mlu270, mlu290 and mlu370 should be less
 *    than 9770.
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
 *  @brief Computes the gradients of feature map \b bottom_grad based on the 
 *    inputs \b top_grad , \b rois and \b mapping_channel to perform the backpropagation 
 *    of the ::mluOpPsRoiPoolForward operator.
 *
 *  @param[in] handle
 *    Input. Handle to a MLUOP context that is used to manage MLU devices and queues in the 
 *    psroipool_forward operation. For detailed information, see ::mluOpHandle_t.
 *  @param[in] pooled_height
 *    Input. An integer value which is the height of the output after pooling.
 *  @param[in] pooled_width
 *    Input. An integer value which is the width of the output after pooling.
 *  @param[in] spatial_scale
 *    Input. A float value which is the scale factor of coordinates of rois.
 *  @param[in] output_dim
 *    Input. An integer value which is the channel of the output after pooling.
 *  @param[in] top_grad_desc
 *    Input. Descriptor of the top_grad tensor, which contains the dimension and the layout 
 *    of top_grad tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[in] top_grad
 *    Input. Pointer to the MLU memory that stores the top_grad tensor.
 *  @param[in] rois_desc
 *    Input. Descriptor of the rois tensor, which contains the dimension and the layout 
 *    of rois tensor. For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[in] rois
 *    Input. Pointer to the MLU memory that stores the rois tensor.
 *  @param[in] mapping_channel_desc
 *    Input. Descriptor of the mapping_channel tensor, which contains the dimension and the 
 *    layout of mapping_channel. For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[in] mapping_channel
 *    Input. Pointer to the MLU memory that stores the mapping_channel tensor. 
 *  @param[in] bottom_grad_desc
 *    Input. Descriptor of the bottom_grad tensor, which contains the dimension and the 
 *    layout of mapping_channel. For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[out] bottom_grad
 *    Output. Pointer to the MLU memory that stores the bottom_grad tensor. 
 * 
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED.
 * 
 *  @par Data Type
 *  - The supported data types of top_grad tensor \b top_grad, rois tensor \b rois, 
 *    mapping_channel tensor \b mapping_channel and bottom_grad tensor \b bottom_grad 
 *    are as follows:
 *    - top_grad tensor: float.
 *    - rois tensor: float.
 *    - mapping_channel tensor: int.
 *    - bottom_grad tensor: float.
 * 
 *  @par Data Layout
 *  - The supported data layouts of top_grad tensor \b top_grad, rois tensor \b rois, 
 *    mapping_channel tensor \b mapping_channel and bottom_grad tensor \b bottom_grad 
 *    are as follows:
 *    - top_grad tensor: \p MLUOP_LAYOUT_NHWC.
 *    - rois tensor: \p MLUOP_LAYOUT_ARRAY.
 *    - mapping_channel tensor: \p MLUOP_LAYOUT_NHWC.
 *    - bottom_grad tensor: \p MLUOP_LAYOUT_NHWC.
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
 * - https://github.com/princewang1994/R-FCN.pytorch/tree/master/
 *   lib/model/psroi_pooling
 */
mluOpStatus_t MLUOP_WIN_API 
mluOpPsRoiPoolBackward(mluOpHandle_t handle,
                       const int pooled_height, const int pooled_width,
                       const float spatial_scale, const int output_dim, 
                       const mluOpTensorDescriptor_t top_grad_desc,
                       const void *top_grad,
                       const mluOpTensorDescriptor_t rois_desc,
                       const void *rois,
                       const mluOpTensorDescriptor_t mapping_channel_desc,
                       const void *mapping_channel,
                       const mluOpTensorDescriptor_t bottom_grad_desc,
                       void *bottom_grad);

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
