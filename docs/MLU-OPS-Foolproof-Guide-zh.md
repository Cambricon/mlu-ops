## MLU-OPS™ 防呆规范说明

防呆是一种预防矫正的行为约束手段。<br>
MLU-OPS™ 使用警告方法来防呆，即将不正常情形通过日志警告，以便用户即时修正错误。<br>
为了让日志清晰明确地输出错误信息，MLU-OPS™ 提供了一系列防呆宏来使用。<br>
以下是介绍常用防呆宏如何使用，以及自定义防呆的参考。<br>

## 常用防呆宏使用说明
### PARAM_CHECK 以及 PARAM_CHECK_V2、PARAM_CHECK_EQ 等
应用场景：单个变量的检查（或两个变量之间的简单关系检查），如：空指针检查，值大小的检查，tensor_dtype一致检查<br>
声明与定义位置：[core/logging.h](../core/logging.h)<br>
防呆返回值：MLUOP_STATUS_BAD_PARAM<br>
注意事项：不支持的状态检查不使用，例如：暂时不支持的规模。<br>
规范示例：
[example](../kernels/ball_query/ball_query.cpp)
```c++
const api_name = "[mluOp***]";
// PARAM_CHECK 检查指针
PARAM_CHECK(api_name, handle != NULL);
PARAM_CHECK(api_name, input_desc != NULL);
PARAM_CHECK(api_name, workspace_size != NULL);

// check tensor dim, dtype ...
PARAM_CHECK(api_name, input_desc->dim <= MLUOP_DIM_MAX);
PARMA_CHECK(api_name, input_desc->dtype == MLUOP_DTYPE_FLOAT ||
                      input_desc->dtype == MLUOP_DTYPE_HALF);
PARMA_CHECK(api_name, input_desc->dtype == output_desc->dtype);

// PARAM_CHECK 可以与 PARAN_CHECK_EQ，PARAM_CHECK_LE 互相替换
PARAM_CHECK_LE(api_name, input_desc->dim, MLUOP_DIM_MAX);
PARAM_CHECK_EQ(api_name, input_desc->dtype, output_desc->dtype);

// PARAM_CHECK_V2 允许添加更丰富的注释
PARAM_CHECK_V2(
      api_name, (input_desc->dtype == MLUOP_DTYPE_INT32),
      "Only int32 are supported in input tensor, but the data type of input tensor is "
          << mluOpGetNameOfDataType(input_desc->dtype) << ".");
```
常见错误示例：
```c++
// PARAM_CHECK 只会将判断条件作为字符串打印出来
// 1.错误使用 PARAM_CHECK 宏来检查自定义变量，用户无法得知其具体含义，如
PARAM_CHECK_EQ(api_name, grad_output_desc->dims[0], origin_n);
PARAM_CHECK_GE(api_name, pad[idx], 0);

// 2.使用 PARAM_CHECK 检查过多变量
PARAM_CHECK(api_name,
            (value_desc->dtype == MLUOP_DTYPE_FLOAT &&
            spatial_shapes_desc->dtype == MLUOP_DTYPE_INT32 &&
            level_start_index_desc->dtype == MLUOP_DTYPE_INT32 &&
            sampling_loc_desc->dtype == MLUOP_DTYPE_FLOAT &&
            attn_weight_desc->dtype == MLUOP_DTYPE_FLOAT &&
            grad_output_desc->dtype == MLUOP_DTYPE_FLOAT &&
            grad_value_desc->dtype == MLUOP_DTYPE_FLOAT &&
            grad_sampling_loc_desc->dtype == MLUOP_DTYPE_FLOAT &&
            grad_attn_weight_desc->dtype == MLUOP_DTYPE_FLOAT));
```

### INTERNAL_CHECK
应用场景：公共模块的检查<br>
声明与定义位置：[core/logging.h](../core/logging.h)<br>
防呆返回值：MLUOP_STATUS_INTERNAL_ERROR<br>
规范示例：无，算子开发不使用<br>
常见错误示例：
```c++
// 使用 INTERNAL_CHECK 检查 mlu_op.h 中的 api，这样的退出的状态会错误
INTERNAL_CHECK(
    api_name,
    MLUOP_STATUS_SUCCESS == mluOpCreateTensorDescriptor(&input_desc));
```

### CHECK_RETURN
应用场景：算子中使用，检查返回状态为 mluOpStatus_t 的函数<br>
声明与定义位置：core/logging.h<br>
防呆返回值：被检查函数的返回值<br>
注意事项：返回状态为 mluOpStatus_t 的函数中，
mlu_ops.h 中出现的函数需要带 MLUOP_WIN_API 修饰，其他的函数不带 MLUOP_WIN_API 修饰。
规范示例：
[example](../kernels/abs/abs.cpp)
```c++
mluOpStatus_t
Kernel3StagePipelineAbs(const cnrtDim3_t k_dim, const cnrtFunctionType_t k_type,
                        const cnrtQueue_t queue, const mluOpDataType_t d_type,
                        const void *x, void *y, const int num);

CHECK_RETURN("[mluOpAbs] ",
               Kernel3StagePipelineAbs(k_dim, k_type, handle->queue,
                                       x_desc->dtype, x, y, element_num));
```
常见错误示例：
无

### MLUOP_CHECK
应用场景：gtest 中使用，检查 mlu_op.h 中 的 api 函数。<br>
如果 kernel 中使用，在抛异常后程序仍会返回正常值，api_test会有影响。<br>
声明与定义位置：[core/logging.h](../core/logging.h)，[core/logging.h](../core/util.cpp)<br>
防呆返回值：无，只抛出异常<br>
规范示例：
[example](../test/mlu_op_gtest/pb_gtest/src/zoo/abs/abs.cpp)
```c++
// test/mlu_op_gtest/pb_gtest/src/zoo 目录下的MLUOP_CHECK
// 例如 abs.cpp > compute() 函数中
MLUOP_CHECK(mluOpAbs(handle_, tensor_x, dev_x, tensor_y, dev_y));
```
常见错误示例：
```c++
// kernel 目录下的MLUOP_CHECK
MLUOP_CHECK(mluOpCreate(&handle_));
```

### TENSOR_NUM_CHECK
应用场景：检查参数值的上阈值（该宏会打印出当前值和上阈值，而非变量名）<br>
声明与定义位置：[core/logging.h](../core/logging.h)<br>
防呆返回值：MLUOP_STATUS_NOT_SUPPORTED<br>
规范示例：
```c++
// masked_col2im_forward.cpp 中
const size_t col_element_num = mluOpGetTensorElementNum(col_desc);
TENSOR_NUM_CHECK("[mluOpMaskedCol2imForward]", col_element_num,
                   LARGE_TENSOR_NUM, "");

// indice_convolution_backward_data.cpp 中，可以补充打印信息
uint64_t input_grad_count = mluOpGetTensorElementNum(input_grad_desc);
TENSOR_NUM_CHECK(api_name, input_grad_count, LARGE_TENSOR_NUM,
                 "input_grad tensor num is too large. ");
```

常见错误示例：
无

### KERNEL_CHECK
应用场景：检查带有 __mlu_global 和 __mlu_entry 修饰的函数<br>
声明与定义位置：[core/logging.h](../core/logging.h)<br>
防呆返回值：MLUOP_STATUS_EXECUTION_FAILED<br>
注意事项：
1.主要是检查驱动和显存这些基础的状态。<br>
2.如果算子执行过程中出现错误抛出异常，由于 device 端异步执行，该防呆可能抓不到异常。<br>
只有同步执行时，才会使能。<br>
规范示例：
[example](../kernels/ball_query/ball_query_union1.mlu)
```c++
template <typename T>
__mlu_global__ void MLUUnion1KernelBallQuery(
    const uint32_t b, const uint32_t n, const uint32_t m,
    const float min_radius, const float max_radius, const int nsample,
    const T *new_xyz, const T *xyz, int32_t *idx) {
  ...
}

KERNEL_CHECK(MLUUnion1KernelBallQuery<float><<<k_dim, k_type, queue>>>(
          b, n, m, min_radius, max_radius, nsample, (float *)new_xyz,
          (float *)xyz, (int32_t *)idx));
```

常见错误示例：
```c++
// 使用 KERNEL_CHECK 检查带 mluOpStatus_t 返回状态的函数
mluOpStatus_t KernelRoiAlignRotatedBackward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, const void *top_grad, const void *rois,
    const int batch, const int height, const int width, const int channel,
    const int rois_num, const mluOpRoiAlignRotatedParams rroiAlignParams,
    void *bottom_grad);

KERNEL_CHECK((KernelRoiAlignRotatedBackward(
    k_dim, k_type, handle->queue, top_grad_desc->dtype, top_grad, rois, batch,
    height, width, channel, rois_nums, roiAlignRotatedParams, bottom_grad)));
```

### CALL_CNNL
应用场景：调用 CNNL 的 api<br>
声明与定义位置：[kernels/utils/cnnl_helper.h](../kernels/utils/cnnl_helper.h)，
[kernels/utils/cnnl_helper.cpp](../kernels/utils/cnnl_helper.cpp)<br>
防呆返回值：<br>
注意事项：v1.0 CHECK_FUNC_RETURN 待废弃
[example](../kernels/dynamic_point_to_voxel/dynamic_point_to_voxel_forward/dynamic_point_to_voxel_forward.cpp)
规范示例：
```c++
// dynamic_point_to_voxel_forward.cpp
CALL_CNNL(cnnlCreateUniqueDescriptor(***));
CALL_CNNL(cnnlSetUniqueDescriptor(...));

// dynamic_point_to_voxel_backward.cpp中
CALL_CNNL(cnnlFill_v3(...));
CALL_CNNL(cnnlScatterNd_v2(...));
```

常见错误示例：
```c++
// 遗留问题 CHECK_FUNC_RETURN 宏将会废弃
// dcn_backward_data.cpp 中
CHECK_FUNC_RETURN(
    cnnlDCNBackwardData(
        cnnl_handle, dcn_desc, cnnl_input_desc, input, cnnl_offset_desc,
        offset, cnnl_mask_desc, mask, cnnl_filter_desc, filter,
        cnnl_grad_output_desc, grad_output, workspace, workspace_size,
        cnnl_grad_input_desc, grad_input, cnnl_grad_offset_desc, grad_offset,
        cnnl_grad_mask_desc, grad_mask),
    CNNL_STATUS_SUCCESS,
    "[mluOpDcnBackwardData] Internal error accured in cnnlDCNBackwardData.",
    MLUOP_STATUS_INTERNAL_ERROR);
```

## 自定义防呆说明
应用场景：适用于各种情况，自定义防呆更加灵活，可以检查复杂的条件，更准确地解释防呆信息。
声明与定义位置：当前算子目录内
规范示例：
[example](../kernels/sparse_conv/indice_convolution_backward_data/indice_convolution_backward_data.cpp)
```c++
// 参数有关联关系，需要进行解释
// indice_convolution_backward_data 中部分参数有关联关系，需要进行解释
if (sub_m == 1 && K % 2 == 0) {
  LOG(ERROR) << api << " When sub_m value is 1, the filters dims (Kd, Kh & "
             << "Kw) should be odd numbers.";
  return MLUOP_STATUS_BAD_PARAM;
}

// 循环中的检查
// binary_op 中 dims 检查，涉及循环的变量检查
for (int i = 0; i < input1_desc->dim; ++i) {
  if (input1_desc->dims[i] != input2_desc->dims[i]) {
    LOG(ERROR) << op_name << ":Check failed: input1_desc->dims[" << i
               << "] should be equal to input2_desc->dims[" << i << "].";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (input1_desc->dims[i] != output_desc->dims[i]) {
    LOG(ERROR) << op_name << ":Check failed: input1_desc->dims[" << i
               << "] should be equal to output_desc->dims[" << i << "].";
    return MLUOP_STATUS_BAD_PARAM;
  }
}

// 自定义变量的检查
// indice_convolution_backward_data 中
int max_indice_num = getMaxNumInArray(indice_num, K);

  if (indice_pairs_desc->dims[2] < max_indice_num) {
    VLOG(5) << "indice_pairs_desc->dims[2] " << indice_pairs_desc->dims[2]
            << " max_indice_num " << max_indice_num;
    LOG(ERROR) << api
               << " The data in indice_num array should be smaller or equal to"
               << " the dims[2] of indice_pairs.";
    return MLUOP_STATUS_BAD_PARAM;
  }

// 环境限制的检查
// roiaware_pool3d 中
/* max_pts_each_voxel affects the allocation of NRAM memory space,
   so it's limited by the size of NRAM memory space. */
if (max_pts_each_voxel > THRESHOLD_OF_MAX_PTS_EACH_VOXEL_BACKWARD) {
  LOG(ERROR) << API << " Check failed: "
             << "max_pts_each_voxel cannot be greater than "
             << THRESHOLD_OF_MAX_PTS_EACH_VOXEL_BACKWARD << ".";
  return MLUOP_STATUS_NOT_SUPPORTED;
}
```

常见错误示例：
```c++
// 自定义的宏名字表意过大
// carafe 中
#define MLUOP_CHECK_RETURN(api, returned_status, ...)       \
  if (returned_status != MLUOP_STATUS_SUCCESS) {            \
    LOG(ERROR) << api << " BAD return value. " __VA_ARGS__; \
    return returned_status;                                 \
  }
```
