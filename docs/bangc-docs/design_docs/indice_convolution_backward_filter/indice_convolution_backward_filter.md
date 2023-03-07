# indice_convolution_backward_filter 算子开发设计方案

- #### 文档基本信息

| 算子名称    | indice_convolution_backward_filter       |
| ----------- | ---------------------------------------- |
| 编制人/日期 | 徐文明/2022-12-15                        |
| 审批人/日期 | 王远/2022-12-15                          |
| 审批人/日期 | 董成威/2022-12-15                        |

- #### 修改记录

| 修订人 | 修订日期   | 修订描述 |
| ------ | ---------- | -------- |
| 徐文明 | 2023-2-6   | 首次提交 |

- #### 内容描述

本文档为稀疏卷积算子的反向，`indice_convolution_backward_filter`的实现原理。

- #### 算子需求 checklist

* 算子接口描述
* 功能描述
* 框架版本 + 对应源码路径
* 需求对应网络
* 网络中用到的规模
* 是否需要支持原位
* 是否需要支持 stride 机制

## 1 需求分析

### 1.1 算子需求分析

该需求分析为框架原生算子实现功能的需求分析，对于框架原生支持但 MLU-OPS 当前版本不支持的功能，需要在`1.4算子限制` 章节中显式注明。未明确注明不支持的功能，默认 MLU-OPS 全部支持。

| 算子功能简介                | 根据感兴趣区域提取固定大小的输出特征|
| -------------------------- | ---------------------------------------- |
| 需求来源                    | PyTorch                                 |
| 应用网络                    | CenterPoint                             |
| 输入数据类型                | features and output_grad:half/float, indice_pairs: int32 |
| 输入 Shape                  | input:<br/>features: [in_active_num,ci] <br/> output_grad: [out_active_num, co] <br/> indice_pairs: [k, 2, in_active_num]<br/>|
| 输入 Layout                 |ARRAY, filters_grad的必须按照DHWCN的布局   |
| 输出数据类型                |filters_grad: half/float                   |
| 输出 Shape                  |filters_grad: [kd, kh, kw, ci, co]         |
| 输出 Layout                 |ARRAY, 不过filters_grad必须按照DHWCN的布局 |
| 模式(可选）                 |sub_m 和　inverse。　inverse暂不支持       |
| 是否含有 dim/axis 等类似语义的参数且该参数支持负数/其他特殊处理    |  sub_m, inverse|
| 是否含有 labels/index 等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | indice_pairs_num必须是host端的数组    |
| 是否需要支持原位            | 否                                        |
| 是否需要支持 stride 机制    | 否                                        |
| 是否需要支持广播            | 否                                        |
| 0 元素检查是否直接返回      | 是（内部不做计算）                        | 
| 其他特殊需求(在线量化，融合，转数提前等，可选)| 无                      |
| 本次开发优先支持的规模/模式 | 无规模限制                                |

### 1.2 算子功能和应用场景

indice_convolution_backward_filter算子用于在稀疏卷积网络中。是稀疏卷积的反向，根据输出的梯度，计算权值的梯度。

稀疏卷积常用于3D项目（如3D点云分割）中，由于点云数据是稀疏的，如果使用标准的卷积操作，会对大量无效点进行卷积，浪费算力。同理，2D任务中，如果只处理其中一部分像素，也可以使用稀疏卷积，这样有助于模型加速。
稀疏卷积的本质就是通过建立哈希表，保存输入和输出的有点点，以及输入输出有限点和卷积核进行计算时的对应关系，最终实现只对有效点进行计算。

### 1.3 算子输入输出参数要求

- 关键参数介绍

| 参数              | 语义                           | 支持类型    | 物理布局 | 规模限制 |
| -----------       | -----------------------------  | ----------- | -------- | -------- |
| handle            |MLU-OPS 上下文的指针            |mluOpHandle_t| /        | 无       |
| features_desc     |卷积输入，输入的有效点数据      |half/float   | ARRAY    | 无       |
| output_grad_desc  |卷积输出的梯度，输出有效点的梯度|half/float   | ARRAY    | 无       |
| indice_pairs_desc |输入，输出和卷积核的映射关系，  |int32        | ARRAY    | 无       |
| indice_pairs_num  |卷积核中的每个点进行计算的次数  |int64        | ARRAY    | 无       |
| filters_grad_desc |卷积权值的梯度                  |half/float   | ARRAY    | 无       |
| sub_m             |稀疏卷积的模式                  |int64        | /        | 无       |
| inverse           |稀疏卷积的算法，目前不支持      |int64        | /        | 无       |

### 1.4 算子限制
1. 虽然get_indice_pairs只支持conv2D, 但是indice_convolution_backward_filter为了增强兼容性，当前支持conv2D和conv3D的稀疏卷积。
2. mmcv原生框架中sparseConvolution中的filter只支持DHWCN或者HWCN的布局，但是当前mluops仓库中没有DHWCN的layout, 所以全部用ARRAY来代替，并在内部加防呆检查filter的shape是按DHWCN/HWCN的布局传进来的。
3. 当前不支持inverse参数不是０的情况。
4. indice_pairs_num在mmcv的接口中是device端的数据，但是为了性能考虑，在indice_convolution_backward_filter的接口设计中是host端的数组。

### 1.5 验收标准

#### 1.5.1 精度验收标准

按照[精度验收标准](../MLU-OPS-Accuracy-Acceptance-Standard.md)的要求明确本算子的精度标准。
- 算子精度验收标准：diff1、diff2；
- 算子精度阈值描述：half:diff1 <= 3e-3 && diff2 <=3e-3; float:diff1 <= 1e-5 && diff2 <=1e-5；

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../MLU-OPS-Performance-Acceptance-Standard.md)。


## 2 算子接口设计

### 2.1　参考接口

稀疏卷积对标的是mmcv仓库。mmcv中backward_filter和backward_data在同一个接口，mluops中是分开实现的。
```c++
std::vector<torch::Tensor> IndiceConvBackwardCUDAKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM)
```

### 2.2 接口设计

```c++
// getWorkspaceSize 接口
mluOpStatus_t MLUOP_WIN_API mluOpGetIndiceConvolutionBackwardFilterWorkspaceSize(mluOpHandle_t handle,
                                                                                 const mluOpTensorDescriptor_t features_desc,
                                                                                 const mluOpTensorDescriptor_t output_grad_desc,
                                                                                 const mluOpTensorDescriptor_t indice_pairs_desc,
                                                                                 const mluOpTensorDescriptor_t filters_grad_desc,
                                                                                 const int64_t indice_num[],
                                                                                 const int64_t inverse,
                                                                                 const int64_t sub_m,
                                                                                 size_t *workspace_size);

// 计算接口
mluOpStatus_t MLUOP_WIN_API mluOpIndiceConvolutionBackwardFilter(mluOpHandle_t handle,
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
```

## 3 实现方案设计

### 3.1 实现方案

indice_convolution_backward_filter是通过在host端调用gather_nd和matmul算子拼接实现。

1. 调用fill算子将filters_grad空间刷0, 防止部分点没有参与计算。
2. 调用gather_nd算子到输入中取出与卷积核中当前点参与加算的所有的输入点数据。记为[cur_active_num, ci]
3. 调用gather_nd算子到输出梯度中取出与卷积核中当前点对应的输出点的数据数据。记为[cur_active_num, co]
4. 调用matmul算子到完成[cur_active_num, ci] 和　[cur_active_num, co]的对位乘加，得到[ci, co]
5. 循环kd * kh * kw 次，　得到完整的filters_grad[kd, kh, kw, ci, co]  

### 3.2 伪代码实现

```c++
indiceConvBackwardFilter(filter_grad, output_grad, input, index_pair) {
  // 需要三个workspace，gather 输出梯度，gather输入
  gather_workspace = getWorkspaceForGather();
  input_gather_workspace = getWorkspaceForInputGather();
  input_transpose_workspace = getWorkspaceForInputTranspose();
  for (filter_element in filter) {
    // gather 输出梯度和输入
    gather(gather_workspace, output_grad, index_pair);
    gather(input_gather_workspace, input, index_pair);
    matmul(filter_grad[filter_id], input_transpose_workspace, gather_workspace);
  }
}

```
### 3.3 性能优化设计

由于indice_convolution_backward_filter是纯拼接实现，所以本身没什么性能优化设计。

## 4 性能问题/精度问题 ＆ 优化记录

indice_convolution_backward_filter的关键瓶颈在于gather_nd和matmul, 只能靠子算子gather_nd和matmul的性能优化来提升性能。当前测试，性能瓶颈主要在gather_nd算子。

## 5 方案实施

初步实施计划如下

- step1: 2022/12/05 - 2022/12/09: 完成host端的拼接逻辑
- step2: 2022/12/12 - 2022/12/16: 完成generator测试环境的搭建
- step3: 2022/12/19 - 2022/12/23: 本地功能调试
- step4: 2022/12/26 - 2022/12/30: 代码review和入库

