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
#define MLUOP_MINOR 4
#define MLUOP_PATCHLEVEL 1

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
  /*!< An error occurrs inside of the function, which may indicate an internal error or bug in
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
/*! @brief Describes the data types in MLUOP. */
typedef enum {
  MLUOP_DTYPE_INVALID       = 0,  /*!< An invalid data type. */
  MLUOP_DTYPE_HALF          = 1,  /*!< A 16-bit floating-point data type. */
  MLUOP_DTYPE_FLOAT         = 2,  /*!< A 32-bit floating-point data type. */
  MLUOP_DTYPE_DOUBLE        = 3,  /*!< A 64-bit floating-point data type. */
  MLUOP_DTYPE_INT8          = 4,  /*!< An 8-bit signed integer data type. */
  MLUOP_DTYPE_INT16         = 5,  /*!< A 16-bit signed integer data type. */
  MLUOP_DTYPE_INT32         = 6,  /*!< A 32-bit signed integer data type. */
  MLUOP_DTYPE_INT64         = 7,  /*!< A 64-bit signed integer data type. */
  MLUOP_DTYPE_UINT8         = 8,  /*!< An 8-bit unsigned integer data type. */
  MLUOP_DTYPE_UINT16        = 9,  /*!< A 16-bit unsigned integer data type. */
  MLUOP_DTYPE_UINT32        = 10, /*!< A 32-bit unsigned integer data type. */
  MLUOP_DTYPE_UINT64        = 11, /*!< A 64-bit unsigned integer data type. */
  MLUOP_DTYPE_BOOL          = 12, /*!< A boolean data type. */
  MLUOP_DTYPE_COMPLEX_HALF  = 13, /*!< A 32-bit complex number of two fp16. */
  MLUOP_DTYPE_COMPLEX_FLOAT = 14, /*!< A 64-bit complex number of two fp32. */
} mluOpDataType_t;

/*!
 * @brief Describes the options that can help choose
 * the best suited algorithm used for implementation of the activation
 * and accumulation operations.
 **/
typedef enum {
  MLUOP_COMPUTATION_FAST = 0,
  /*!< Implementation with the fastest algorithm and lower precision.*/
  MLUOP_COMPUTATION_HIGH_PRECISION = 1,
  /*!< Implementation with the high-precision algorithm regardless the performance.*/
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
 * @brief Describes the rounding modes of
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
 *
 * @brief Describes the bases that are used in the implementation
 * of the log function.
 *
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
 * MLUOP Runtime Management
 ******************************************************************************/

/*!
 * @struct mluOpContext
 * @brief Describes the MLUOP context.
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

// Group:Runtime Management
/*!
 *  @brief Initializes the MLUOP library and creates a handle \b handle to a structure
 *  that holds the MLUOP library context. It allocates hardware resources on the host
 *  and device. You need to call this function before any other MLUOP functions.
 *
 *  You need to call the ::mluOpDestroy function to release the resources later.
 *
 *  @param[out] handle
 *  Pointer to the MLUOP context that is used to manage MLU devices and
 *  queues. For detailed information, see ::mluOpHandle_t.
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
mluOpCreate(mluOpHandle_t *handle);

// Group:Runtime Management
/*!
 *  @brief Updates the MLUOP context information that is held by the \b handle. This function
 *  should be called if you call Cambriocn CNDrv API cnSetCtxConfigParam to set the context
 *  information. The related context information will be synchronized to MLUOP with this function.
 *  For detailed information, see "Cambricon CNDrv Developer Guide".
 *
 *  @param[in] handle
 *  Pointer to the MLUOP context that is used to manage MLU devices.
 *  For detailed information, see ::mluOpHandle_t.
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
mluOpUpdateContextInformation(mluOpHandle_t handle);

// Group:Runtime Management
/*!
 *  @brief Releases the resources of the specified MLUOP handle \b handle that was
 *  created by the ::mluOpCreate function.
 *  It is usually the last call to destroy the handle to the MLUOP handle.
 *
 *  @param[in] handle
 *  Pointer to the MLU devices that holds information to be destroyed.
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
 *
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
 *  handle. This function should be called if you want to change the mluop rounding mode that used
 *  to cumulate the results. For detailed information, see Cambricon Driver API Developer Guide.
 *
 *  @param[in] handle
 *  Pointer to the MLUOP context that is used to manage MLU devices and
 *  queues. For detailed information, see ::mluopHandle_t.
 *  @param[in] round_mode
 *  The rounding mode of quantization conversion to be set to the MLUOP handle.
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 *  @note
 *  - On MLU200 series:
 *    You can't set MLUOP_ROUND_HALF_TO_EVEN for the rounding mode because the hardware does not
 *    support it.
 *
 *  @par Requirements
 *  - None.
 *
 *  @par Example
 *  - None.
 */
mluOpStatus_t MLUOP_WIN_API
mluOpSetQuantizeRoundMode(mluOpHandle_t handle, mluOpQuantizeRoundMode_t round_mode);

// Group:QuantizeRoundMode
/*!
 *  @brief Retrieves the rounding mode of a specific MLUOP context.
 *
 *  @param[in] handle
 *  Pointer to the MLUOP context that is used to manage MLU devices and
 *  queues. For detailed information, see ::mluopHandle_t.
 *
 *  @param[out] round_mode
 *  The rounding mode of quantization conversion that was previously set to the specified handle.
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
mluOpStatus_t MLUOP_WIN_API
mluOpGetQuantizeRoundMode(mluOpHandle_t handle, mluOpQuantizeRoundMode_t *round_mode);

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
mluOpStatus_t MLUOP_WIN_API
mluOpCreateTensorDescriptor(mluOpTensorDescriptor_t *desc);

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
mluOpStatus_t MLUOP_WIN_API
mluOpSetTensorDescriptor(mluOpTensorDescriptor_t desc,
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
 *  @par Return
 *   - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM.
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
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorDescriptor(const mluOpTensorDescriptor_t desc,
                         mluOpTensorLayout_t *layout,
                         mluOpDataType_t *dtype,
                         int *dimNb,
                         int dimSize[]);

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

// Group:Tensor
/*!
 *  @brief Destroies a tensor descriptor that was created by
 *  ::mluOpCreateTensorDescriptor.
 *
 *  @param[in] desc
 *  A tensor descriptor created by ::mluOpCreateTensorDescriptor.
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
mluOpStatus_t MLUOP_WIN_API
mluOpDestroyTensorDescriptor(mluOpTensorDescriptor_t desc);

// Group:Tensor
/*!
 *  @brief Destroys a group of tensor descriptors that was created by
 *  ::mluOpCreateTensorDescriptor or ::mluOpCreateGroupTensorDescriptors.
 *
 *  @param[in] group_desc
 *  An array of pointers to the struct that hold information about the
 *  tensor descriptor.
 *  @param[in] desc_num
 *  The length of the input array \b group_desc.
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
 *  The descriptor of the tensor set. For detailed information,
 *  see ::mluOpSeqDataDescriptor_t.
 *  @param[out] setDimNb
 *  The number of dimensions of the tensor set.
 *  @param[out] setDimSize
 *  An array that contains the number of the tensor for each dimension
 *  of the tensor set.
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
mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorSetDescriptor(mluOpTensorSetDescriptor_t tensorSetDesc,
                            int *setdimNb,
                            int setDimSize[]);

// Group:TensorSet
/*!
 *  @brief Destroys a tensor set descriptor \b tensorSetDesc that is previously
 *  created by ::mluOpCreateTensorSetDescriptor.
 *
 *  @param[in] desc
 *  A tensor descriptor created by ::mluOpCreateTensorSetDescriptor.
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
mluOpStatus_t MLUOP_WIN_API
mluOpAbs(mluOpHandle_t handle,
         const mluOpTensorDescriptor_t x_desc,
         const void *x,
         const mluOpTensorDescriptor_t y_desc,
         void *y);

// Group:Log
/*!
 * @brief Computes logarithm of input tensor \b x, and returns the results in
 * the output tensor \b y.
 *
 * @param[in] handle
 * Handle to an MLUOP context that is used to manage MLU devices and
 * queues in the log operation. For detailed information, see ::mluOpHandle_t.
 * @param[in] prefer
 * The \b prefer modes defined in ::mluOpComputationPreference_t enum.
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
 *   - input tensor: half, float.
 *   - output tensor: half, float.
 *
 * @par Scale Limitation
 * - The input tensor and output tensor have the same shape, and the input
 *   tensor must meet the following input data ranges:
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
mluOpLog(mluOpHandle_t handle,
         const mluOpComputationPreference_t prefer,
         const mluOpLogBase_t base,
         const mluOpTensorDescriptor_t x_desc,
         const void *x,
         const mluOpTensorDescriptor_t y_desc,
         void *y);

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
 * The \b prefer modes defined in ::mluOpComputationPreference_t enum.
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
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetGenerateProposalsV2WorkspaceSize(mluOpHandle_t handle,
                                         const mluOpTensorDescriptor_t scores_desc,
                                         size_t *size);

// Group:GenerateProposalsV2
/*!
 *  @brief Generates bounding box proposals for Faster Region-CNN.
 *  This operator is the second version of generate_proposals op.
 *  The proposals are generated for a list of images based on image
 *  score 'Scores', bounding box regression result 'BboxDeltas' as
 *  well as predefined bounding box shapes 'anchors'. Greedy non-maximum
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
 *     - scores: float.
 *     - bbox_deltas: float.
 *     - im_shape: float.
 *     - anchors: float.
 *     - variances: float.
 *     - pre_nms_top_n: int32.
 *     - post_nms_top_n: int32.
 *     - nms_thresh: float.
 *     - min_size: float.
 *     - eta: float.
 *     - pixel_offset: bool.
 *     - rpn_rois: float.
 *     - rpn_roi_probs: float.
 *     - rpn_rois_num: int32.
 *     - rpn_rois_batch_size: int32.
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
 *  - This commit does not support nan/inf.
 *  - Not support adaptive NMS. The attribute 'eta' should not less
 *    than 1.
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
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 */
mluOpStatus_t MLUOP_WIN_API
mluOpGetPolyNmsWorkspaceSize(mluOpHandle_t handle,
                             const mluOpTensorDescriptor_t boxes_desc,
                             size_t *size);

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
 *     - input tensor: float.
 *     - iou_threshold: float.
 *     - Output tensor: int32.
 *     - output_size tensor: int32.
 *
 *  @par Data Layout
 *  - The supported data layout of \b input, \b output,
 *     \b output_size are as follows:
 *
 *   - input tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - output tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - output_size tensor: \p MLUOP_LAYOUT_ARRAY.
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
 *    calculation result of the competitor operator.
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
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *    ::MLUOP_STATUS_NOT_SUPPORTED
 *
 *  @par Data Type
 *  - The supported data types of \b input and \b output are as follows:
 *     - min_sizes tensor: float.
 *     - aspect_ratios tensor: float.
 *     - variances tensor: float.
 *     - max_sizes tensor: float.
 *     - height: int.
 *     - width: int.
 *     - im_height: int.
 *     - im_width: int.
 *     - step_h: float.
 *     - step_w: float.
 *     - offset: float.
 *     - clip: bool.
 *     - min_max_aspect_ratios_order: bool.
 *     - output: float.
 *     - var: float.
 *
 *  @par Data Layout
 *  - The supported data layouts of \b input, \b output,
 *    are as follows:
 *
 *   - input tensor:
 *     - min_sizes: \p MLUOP_LAYOUT_ARRAY.
 *     - aspect_ratios: \p MLUOP_LAYOUT_ARRAY.
 *     - variances: \p MLUOP_LAYOUT_ARRAY.
 *     - max_sizes: \p MLUOP_LAYOUT_ARRAY.
 *   - output tensor:
 *     - output: \p MLUOP_LAYOUT_ARRAY.
 *     - var: \p MLUOP_LAYOUT_ARRAY.
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
 *  - The shape[0] of the \b ouput should be equal to the input height.
 *  - The shape[1] of the \b ouput should be equal to the input width.
 *  - The shape[2] of the \b ouput and \b var must be less than 2100
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
 *  - The shape[2] of the \b ouput and \b var must be
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
 *  Descriptor of input tensor, containing dimension and the layout of input.
 *  For detailed information, see ::mluOpTensorDescriptor_t.
 *  @param[in] input
 *  Pointer to the MLU memory that stores the input tensor. The shape of \b input is
 *  [batch_num, H, W, C].
 *  @param[in] rois_desc
 *  Descriptor of rois tensor, containing dimension and the layout of rois.
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
 *  Descriptor of the mapping_channel tensor, containing dimension and the layout of
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
 *    - input tensor: float.
 *    - Rois tensor: float.
 *    - output tensor: float.
 *    - Mapping_channel tensor: int32.
 *
 *  @par Data Layout
 *  - The supported data layout of \b input, \b rois, \b output, and \b mapping_channel
 *    are as follows:
 *     - input tensor: \p MLUOP_LAYOUT_NHWC.
 *     - Rois tensor: \p MLUOP_LAYOUT_ARRAY.
 *     - output tensor: \p MLUOP_LAYOUT_NHWC.
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
 *  of the ::mluOpPsRoiPoolForward operator.
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
 * Handle to a Cambricon MLUOP context that is used to manage MLU devices and queues in
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
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM.
 *
 * @par Data Type
 * - This function supports the following data types for input tensor \b features, \b rois,
 *   and output tensor \b output. Data types of all tensors should be the same.
 *   - input tensor: half, float.
 *   - rois tensor: half, float.
 *   - output tensor: half, float.
 *
 * @par Data Layout
 * - The supported data layouts of \b features, \b rois, and \b output are as follows:
 *   - input tensor: \p MLUOP_LAYOUT_NHWC.
 *   - rois tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - output tensor: \p MLUOP_LAYOUT_NHWC.
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
     input two arrays by 1 * 3 * 3 * 1 and 1 * 6 --> input: [[[[1.0],[1.0],[1.0]],[[1.0],[1.0],[1.0]],[[1.0],[1.0],[1.0]]]]

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
                            const void* features,
                            const mluOpTensorDescriptor_t rois_desc,
                            const void* rois,
                            const int pooled_height,
                            const int pooled_width,
                            const int sample_ratio,
                            const float spatial_scale,
                            const bool aligned,
                            const bool clockwise,
                            const mluOpTensorDescriptor_t output_desc,
                            void* output);

// Group:RoiAlignRotated
/*!
 * @brief Computes the gradients of feature map \b bottom_grad based on the input \b top_grad and
 * \b rois to perform the backpropagation of the ::mluOpRoiAlignRotatedForward operation.
 *
 * @param[in] handle
 * Handle to a Cambricon MLUOP context that is used to manage MLU devices and queues in
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
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM.
 *
 * @par Data Type
 * - This function supports the following Data types for input tensor \b top_grad, \b rois,
 *   and output tensor \b bottom_grad. Data types of all tensors should be the same.
 *   - top_grad tensor: half, float.
 *   - rois tensor: half, float.
 *   - bottom_grad tensor: half, float.
 *
 * @par Data Layout
 * - The supported data layouts of \b top_grad, \b rois, and \b bottom_grad are as follows:
 *   - top_grad tensor: \p MLUOP_LAYOUT_NHWC.
 *   - rois tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - bottom_grad tensor: \p MLUOP_LAYOUT_NHWC.
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
                             const void* top_grad,
                             const mluOpTensorDescriptor_t rois_desc,
                             const void* rois,
                             const int pooled_height,
                             const int pooled_width,
                             const int sample_ratio,
                             const float spatial_scale,
                             const bool aligned,
                             const bool clockwise,
                             const mluOpTensorDescriptor_t bottom_grad_desc,
                             void* bottom_grad);

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
 *   - input tensor: float.
 *   - Grid tensor: float.
 *   - output tensor: float.
 * @par Data Layout
 * - The supported data layout of \b input , \b grid , \b output are as follows:
 *   - input tensor: \p MLUOP_LAYOUT_NHWC.
 *   - Grid tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - output tensor: \p MLUOP_LAYOUT_NHWC.
 *
 * @par Scale Limitation
 * - The input tensor, grid tensor and ouput tensor must have four dimensions.
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
 * - Size of the first dimension of grad_input tensor is divided by size of
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
 * Handle to a Cambricon MLUOP context that is used to manage MLU devices and queues in
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
 *   - input tensor: half, float.
 *   - bboxes tensor: half, float.
 *   - output tensor: half, float.
 *
 * @par Data Layout
 * - The supported data layouts of \b input, \b bboxes and \b output are as follows:
 *   - input tensor: \p MLUOP_LAYOUT_NHWC.
 *   - bboxes tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - output tensor: \p MLUOP_LAYOUT_NHWC.
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
 * Handle to a Cambricon MLUOP context that is used to manage MLU devices and queues in
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
 *   - top_output tensor: half, float.
 *   - bboxes tensor: half, float.
 *   - bottom_input tensor: half, float.
 *
 * @par Data Layout
 * - The supported data layouts of \b top_output, \b bboxes and \b bottom_input are as follows:
 *   - top_output tensor: \p MLUOP_LAYOUT_NHWC.
 *   - bboxes tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - bottom_input tensor: \p MLUOP_LAYOUT_NHWC.
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
 * The \b prefer modes defined in ::mluOpComputationPreference_t enum.
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
 *
 * @par Data Type
 * - Data types of input tensors and output tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input tensors: half, float.
 *   - output tensor: half, float.
 *
 * @par Scale Limitation
 * - The input tensor and output tensor must have the same shape, and the input
 *   tensor \b y must meet the following input data ranges:
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
 * function create.
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
 *  @par Return
 *  - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM,
 *    ::MLUOP_STATUS_NOT_SUPPORTED.
 */

mluOpStatus_t MLUOP_WIN_API mluOpGetVoxelizationWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t points_desc,
    const mluOpTensorDescriptor_t voxel_size_desc,
    const mluOpTensorDescriptor_t coors_range_desc, const int32_t max_points,
    const int32_t max_voxels, const int32_t NDim, const bool deterministic,
    const mluOpTensorDescriptor_t voxels_desc,
    const mluOpTensorDescriptor_t coors_desc,
    const mluOpTensorDescriptor_t num_points_per_voxel_desc,
    const mluOpTensorDescriptor_t voxel_num_desc, size_t *size);

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
 *   ::MLUOP_STATUS_NOT_SUPPORTED.
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - points, voxel_size, coors_range, voxels: float.
 *   - coors, num_points_per_voxel, voxel_num: int.
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

mluOpStatus_t MLUOP_WIN_API mluOpVoxelization(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t points_desc,
    const void *points, const mluOpTensorDescriptor_t voxel_size_desc,
    const void *voxel_size, const mluOpTensorDescriptor_t coors_range_desc,
    const void *coors_range, const int32_t max_points, const int32_t max_voxels,
    const int32_t NDim, const bool deterministic, void *workspace,
    size_t workspace_size, const mluOpTensorDescriptor_t voxels_desc,
    void *voxels, const mluOpTensorDescriptor_t coors_desc, void *coors,
    const mluOpTensorDescriptor_t num_points_per_voxel_desc,
    void *num_points_per_voxel, const mluOpTensorDescriptor_t voxel_num_desc,
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
 * The downsample ratio from network input to yolo_box operator input,
 * so 32, 16, 8 should be set for the first, second, and thrid into yolo_box operator.
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
 * ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - Data types of input tensors and output tensor must be the same.
 * - The supported data types of input and output tensors are as follows:
 *   - input x tensor: float.
 *   - input img_size and anchors tensors: int.
 *   - output tensors: float.
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
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *
 *   - geom_xyz tensor: int.
 *   - input_features tensor: float.
 *   - output_features tensor: float.
 *   - pos_memo tensor: int.
 *
 * @par Data Layout
 *  - The supported data layouts of \b geom_xyz, \b input_features, \b output_features and \b pos_memo are
 *    as follows:
 *
 *   - Input tensor:
 *     - geom_xyz tensor: \p MLUOP_LAYOUT_ARRAY.
 *     - input_features tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - Output tensor:
 *     - output_features tensor: \p MLUOP_LAYOUT_ARRAY.
 *     - pos_memo tensor: \p MLUOP_LAYOUT_ARRAY.
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
 *  - The operator does not support MLU200 series.
 *  - You need to set the initial value for the output \b pos_memo before calling the operator, and initialize it to a negative number.
 *
 * @par Reference
 * - https://github.com/Megvii-BaseDetection/BEVDepth/blob/main/bevdepth/ops/voxel_pooling/src/voxel_pooling_forward_cuda.cu
 */
mluOpStatus_t MLUOP_WIN_API mluOpVoxelPoolingForward(mluOpHandle_t handle,
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
 *   - float - float - float.
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
 * Pointer to the MLU memory that stores the input indicies tensor. The indices'
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
 *   - features tensor: half, float.
 *   - indices tensor: int.
 *   - weights tensor: half, float.
 *   - output tensor: half, float.
 *
 *  @par Data Layout
 *  - The supported data layouts of \b features, \b indices, \b weights, \b output are
 *    as follows:
 *
 *   - features tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - indices tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - weights tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - output tensor: \p MLUOP_LAYOUT_ARRAY.
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
 * Pointer to the MLU memory that stores the input indicies tensor. The indices'
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
 *   - grad_output tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - indices tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - weights tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - grad_features tensor: \p MLUOP_LAYOUT_ARRAY.
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
mluOpStatus_t MLUOP_WIN_API mluOpThreeInterpolateBackward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t grad_output_desc,
    const void *grad_output, const mluOpTensorDescriptor_t indices_desc, const void *indices,
    const mluOpTensorDescriptor_t weights_desc, const void *weights,
    const mluOpTensorDescriptor_t grad_features_desc, void *grad_features);

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
 *   tensor \b new_xyz, xyz tensor \b xyz and idx tensor \b idx are as fllows:
 *   - new_xyz tensor: float or half.
 *   - xyz tensor: float or half.
 *   - idx tensor: int.
 *
 *  @par Data Layout
 *  - The supported data layouts of \b new_xyz, \b xyz, \b idx are
 *    as follows:
 *
 *   - new_xyz tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - xyz tensor: \p MLUOP_LAYOUT_ARRAY.
 *   - idx tensor: \p MLUOP_LAYOUT_ARRAY.
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
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
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
     @verbatim
      input array by 2 * 2
      --> then: [[1, 8], [6, 4]]

      output array by 2 * 2
      --> output: [[1, 8], [6, 4]]
     @endverbatim
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
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM
 *
 * @par Data Type
 * - This function supports the following data types for input tensor \b input
 *   and output tensor \b output.
 *   Data type of both tensors should be the same.
 *   - input tensor: uint8, int8, uint16, int16, uint32, int32, uint64, int64,
 *     bool, half, float, complex_half, complex_float.
 *   - output tensor: uint8, int8, uint16, int16, uint32, int32, uint64, int64,
 *     bool, half, float, complex_half, complex_float.
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
     @verbatim
     input one array by 2 * 2 --> input: [[1, 2], [3, 4]]
     output array by 3 * 2 * 2 --> output: [[[1, 2], [3, 4]],
                                            [[1, 2], [3, 4]],
                                            [[1, 2], [3, 4]]]
     @endverbatim
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
 * A pointer to scaling factor of tensor input.
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
 *     half, float.
 *   - output tensor: uint8, int8, uint16, int16, uint32, int32, uint64, int64,
 *     bool, half, float.
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
     @verbatim
      param:value: 5

      output array by 2 * 3 * 2 --> output: [[[5,5],[5,5],[5,5]],
                                             [[5,5],[5,5],[5,5]]]
     @endverbatim
 *
 * @par Reference
 * - https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Fill.cpp
 *
 */
mluOpStatus_t MLUOP_WIN_API mluOpFill(
    mluOpHandle_t handle, const mluOpPointerMode_t pointer_mode,
    const void *value, const mluOpTensorDescriptor_t output_desc, void *output);

// Group:Psamask
/*!
 * @brief Moves the \b x tensor to \b y tensor according to \b h_mask,
 * \b w_mask and \b psa_type.
 *
 * @param[in] handle
 * Handle to a Cambricon MLUOP context that is used to manage MLU devices and
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
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED.
 *
 * @par Formula
 * - See "Psamask Operator" section in "Cambricon BANGC OPS User Guide" for details.
 *
 * @par Data Type
 * - The supported data types of input tensor \b x and output tensor \b y are as follows:
 *   - x: float.
 *   - y: float.
 *
 * @par Data Layout
 * - The supported data layouts of input tensor \b x and output tensor \b y are as follows
 *   - x: NHWC.
 *   - y: NHWC.
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
 *
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
 * Handle to a Cambricon MLUOP context that is used to manage MLU devices and
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
 * - ::MLUOP_STATUS_SUCCESS, ::MLUOP_STATUS_BAD_PARAM, ::MLUOP_STATUS_NOT_SUPPORTED.
 *
 * @par Formula
 * - See "Psamask Operator" section in "Cambricon BANGC OPS User Guide" for details.
 *
 * @par Data Type
* - The supported data types of input tensor \b x and output tensor \b y are as follows
 *   - dy: float.
 *   - dx: float.
 *
 * @par Data Layout
 * - The supported data layouts of input tensor \b x and output tensor \b y are as follows:
 *   - dy: NHWC.
 *   - dx: NHWC.
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
 * satisfied: ci + co <= 6144.
 *   - When psa_type is DISTRIBUTE, the size of \b dx channels ci and \b dy channels co should be
 * satisfied: ci + 2 * co <= 6144.
 * - On MLU300 series:
 *   - When psa_type is COLLECT, the size of \b dx channels ci and \b dy channels co should be
 * satisfied: ci + co <= 10240.
 *   - When psa_type is DISTRIBUTE, the size of \b dx channels ci and \b dy channels co should be
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
 *
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

#if defined(__cplusplus)
}
#endif

#endif  // MLUOP_EXAMPLE_H_
