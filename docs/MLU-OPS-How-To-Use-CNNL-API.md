## MLU-OPS 如何使用 CNNL 基础算子
在MLU-OPS 开发算子时，如果需要用到某个基础功能，其功能已经由CNNL 算子实现，我们不妨直接在MLU-OPS 代码中调用CNNL 的API，快速达成算子开发的目的。下面3个示例介绍了如何在MLU-OPS 代码中实现调用CNNL 算子的方法。

### 示例1
在算子中调用CNNL tranpose 算子。
```c++
// 需要包含该文件
#include "utils/cnnl_helper.h"

// https://github.com/Cambricon/mlu-ops/blob/master/bangc-ops/kernels/three_nn_forward/three_nn_forward.cpp
static mluOpStatus_t transposeTensor(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const int *permute,
    const mluOpTensorDescriptor_t workspace_dst_desc, void *workspace_dst,
    void *transpose_workspace, size_t transpose_workspace_size) {
  const int input_dim = input_desc->dim;

  // step1. 将mluOpHandle、mluOpTensorDescriptor_t 等参数转换为 CNNL 公共参数类型

  // 宏 DEFINE_CREATE_AND_SET_CNNL_HANDLE 内部定义并创建 CNNL handle，
  // 并将 MLU-OPS handle 信息转换设置到 CNNL handle
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);

  // 宏 DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR 内部定义并创建 CNNL TensorDescriptor
  // (区别于各算子特有的descriptor，例如上述的 cnnlTranposeDescriptor，通用的TensorDescriptor通过宏操作更方便)，
  // 并将 MLU-OPS TensorDescriptor 信息转换设置到 CNNL TensorDescriptor
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(workspace_dst_desc,
                                               cnnl_workspace_dst_desc);

  // step2. 定义 cnnlTranspose_v2 cnnlTranposeDescriptor 类型入参

  // MLU-OPS代码中直接定义cnnl cnnl_trans_desc 变量
  cnnlTransposeDescriptor_t cnnl_trans_desc = NULL;

  // 直接调用 CNNL api 创建 cnnlTranposeDescriptor
  CALL_CNNL(cnnlCreateTransposeDescriptor(&cnnl_trans_desc));

  // 直接调用 CNNL api 设置 cnnlTranposeDescriptor
  CALL_CNNL(cnnlSetTransposeDescriptor(cnnl_trans_desc, input_dim, permute));

  // step3. 调用 cnnlTranspose_v2

  // 直接调用 CNNL 算子 api
  CALL_CNNL(cnnlTranspose_v2(cnnl_handle, cnnl_trans_desc, cnnl_input_desc,
                             input, cnnl_workspace_dst_desc, workspace_dst,
                             transpose_workspace, transpose_workspace_size));

  // step4. Destroy cnnlTranspose_v2 cnnlTranposeDescriptor 类型入参

  // 直接调用 CNNL api 销毁 cnnlTranposeDescriptor
  CALL_CNNL(cnnlDestroyTransposeDescriptor(cnnl_trans_desc));

  //step5. Destroy CNNL 公共类型参数 handle 及 tensor descriptor等.

  // 宏 DESTROY_CNNL_TENSOR_DESCRIPTOR 内部销毁新建的 CNNL TensorDescriptor
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_workspace_dst_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);

  // 宏 DESTROY_CNNL_TENSOR_DESCRIPTOR 内部销毁新建的 CNNL handle
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

```
### 示例2
在算子中调用CNNL addN 算子的GetWorkspaceSize, 与示例1不同的点在于，当输入的tensor是数组时的处理方法。
```c++
// 需要包含该文件
#include "utils/cnnl_helper.h"

mluOpStatus_t GetAddNWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_descs[],
    const uint32_t input_num, const mluOpTensorDescriptor_t output_desc,
    size_t *workspace_size) {);

  // 当对应的输入是 MLU-OPS TensorDescriptor 数组变量时
  // 可先定义并malloc 同样大小的CNNL TensorDescriptor数组变量
  cnnlTensorDescriptor_t *_input_descs = (cnnlTensorDescriptor_t *)malloc(
      sizeof(cnnlTensorDescriptor_t) * input_num);

  // 宏 CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR 内部创建 CNNL TensorDescriptor
  // (不需要再定义)，通过循环方式并将每个MLU-OPS TensorDescriptor 信息转换设置到对应的每个 
  // CNNL TensorDescriptor
  for (int i = 0; i < input_num; i++) {
    CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_descs[i], _input_descs[i]);
  }
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, _output_desc);
  CALL_CNNL(cnnlGetAddNWorkspaceSize(_handle, _input_descs, input_num,
                                     _output_desc, workspace_size);

  // 循环销毁每个 CNNL TensorDescriptor
  for (int i = 0; i < input_num; i++) {
    DESTROY_CNNL_TENSOR_DESCRIPTOR(_input_descs[i]);
  }

  // 释放malloc的空间
  free(_input_descs);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_output_desc);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}

```


### 示例3
在算子中调用CNNL matmul 算子的相关接口。
```c++
// https://github.com/Cambricon/mlu-ops/blob/master/bangc-ops/kernels/indice_convolution_backward_data/indice_convolution_backward_data.cpp
// 需要包含该文件
#include "utils/cnnl_helper.h"

mluOpStatus_t MLUOP_WIN_API mluOpGetIndiceConvolutionBackwardDataWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t output_grad_desc,
    const mluOpTensorDescriptor_t filters_desc,
    const mluOpTensorDescriptor_t indice_pairs_desc,
    const mluOpTensorDescriptor_t input_grad_desc, const int64_t indice_num[],
    const int64_t inverse, size_t *workspace_size) {
    ......
  // matmul workspace
  {
    mluOpTensorDescriptor_t sub_filters_desc;
    mluOpTensorDescriptor_t output_grad_condence_desc;
    mluOpTensorDescriptor_t input_grad_condence_desc;

    // 在 MLU-OPS 代码中直接定义 CNNL 变量
    cnnlMatMulDescriptor_t cnnl_matmul_desc;
    cnnlMatMulHeuristicResult_t cnnl_heuristic_result;
    cnnlMatMulAlgo_t cnnl_matmul_algo;

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&sub_filters_desc));
    int sub_filter_dims[2] = {(int)(dxc), (int)(dyc)};
    MLUOP_CHECK(mluOpSetTensorDescriptor(sub_filters_desc, MLUOP_LAYOUT_ARRAY,
                                         filters_desc->dtype, 2,
                                         sub_filter_dims));
    int is_trans_a = 0, is_trans_b = 1;
    int tf32_flag_int = 0;
    
    // 直接调用 CNNL api 对结构进行创建、设置
    CALL_CNNL(cnnlMatMulDescCreate(&cnnl_matmul_desc));
    CALL_CNNL(cnnlSetMatMulDescAttr(cnnl_matmul_desc, CNNL_MATMUL_DESC_TRANSA,
                                    &(is_trans_a), sizeof(is_trans_a)));
    CALL_CNNL(cnnlSetMatMulDescAttr(cnnl_matmul_desc, CNNL_MATMUL_DESC_TRANSB,
                                    &(is_trans_b), sizeof(is_trans_b)));
    CALL_CNNL(cnnlSetMatMulDescAttr(cnnl_matmul_desc, CNNL_MATMUL_ALLOW_TF32,
                                    &(tf32_flag_int), sizeof(tf32_flag_int)));
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_grad_condence_desc));
    int output_grad_condence_dims[2] = {(int)(max_indice_num), (int)(dyc)};
    MLUOP_CHECK(mluOpSetTensorDescriptor(
        output_grad_condence_desc, MLUOP_LAYOUT_ARRAY, output_grad_desc->dtype,
        2, output_grad_condence_dims));
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_grad_condence_desc));
    int input_grad_condence_dims[2] = {(int)(max_indice_num), (int)(dxc)};
    MLUOP_CHECK(mluOpSetTensorDescriptor(
        input_grad_condence_desc, MLUOP_LAYOUT_ARRAY, input_grad_desc->dtype, 2,
        input_grad_condence_dims));

    // 通过宏内部定义并创建 CNNL Handle，TensorDescritpor，
    // 并将 MLU-OPS Handle，TensorDescritpor 信息转换设置到对应 
    // CNNL Handle，TensorDescritpor
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(sub_filters_desc,
                                                 cnnl_sub_filters_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_grad_condence_desc,
                                                 cnnl_output_grad_condence_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_grad_condence_desc,
                                                 cnnl_input_grad_condence_desc);

    // 直接调用 CNNL api 对结构进行创建、设置
    CALL_CNNL(cnnlCreateMatMulHeuristicResult(&cnnl_heuristic_result));
    CALL_CNNL(cnnlMatMulAlgoCreate(&cnnl_matmul_algo));

    // set matmul heuristic_result & algorithm
    int requested_algo_count = 1, return_algo_count = 0;

    // 直接调用 CNNL api
    CALL_CNNL(cnnlGetMatMulAlgoHeuristic(
        cnnl_handle, cnnl_matmul_desc, cnnl_output_grad_condence_desc,
        cnnl_sub_filters_desc, cnnl_input_grad_condence_desc,
        cnnl_input_grad_condence_desc, NULL, requested_algo_count,
        &cnnl_heuristic_result, &return_algo_count));

    // launch matmul
    size_t workspace_size_matmul = 0;
    float alpha_gemm = 1.0f, beta_gemm = 0.0f;

    // 直接调用 CNNL api
    CALL_CNNL(cnnlGetMatMulHeuristicResult(
        cnnl_heuristic_result, cnnl_matmul_algo, &workspace_size_matmul));

    // destroy descriptors
    // 直接调用 CNNL api 对非宏生成的CNNL 变量销毁
    CALL_CNNL(cnnlDestroyMatMulHeuristicResult(cnnl_heuristic_result));
    CALL_CNNL(cnnlMatMulDescDestroy(cnnl_matmul_desc));
    CALL_CNNL(cnnlMatMulAlgoDestroy(cnnl_matmul_algo));

    // 销毁由宏内部定义并创建的变量
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_grad_condence_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_sub_filters_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_grad_condence_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);

    MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_grad_condence_desc));
    MLUOP_CHECK(mluOpDestroyTensorDescriptor(sub_filters_desc));
    MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_grad_condence_desc));
    matmul_workspace_size = (uint64_t)workspace_size_matmul;
  }
  ......
  return MLUOP_STATUS_SUCCESS;
}
```
