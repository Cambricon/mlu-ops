# IndiceConvolutionBackwardData 算子开发设计方案
* #### 文档基本信息

| 算子名称    | IndiceConvolutionBackwardData                             |
| ----------- | ------------------------------------------------------------|
| 编制人/日期 | 杜泽坤/2022-11-28                                            |
| 审批人/日期 | 董成威/2023-2-6                                              |
| 审批人/日期 | 王远/2023-2-6                                               |

* #### 修改记录

| 版本号| 修订人 | 修订日期 | 修订描述 |
| ----- | ------ | -------  | -------  |
| V1.0  | 杜泽坤  | 2022-11-28 | 首次提交 |

* #### 内容描述

本文档为`IndiceConvolutionBackwardData`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。该算子为`IndiceConvolutionForward`算子的反向算子

## 1 需求分析

### 1.1 算子需求分析

| 算子功能简介| 对稀疏的数据做稀疏卷积反向操作                     |
|-------------|--------------------------------------------------------------|
| 需求来源    | MMVC                                       |
| 应用网络    | CenterPoint                                                 |
| 输入数据类型| output_grad: half, float;<br>filter: half, float;<br>indices_pairs: int32;<br>indices_num: int32;<br>inverse: int64;<br>sub_m: int64 |
| 输入Shape   | output_grad: [Y, Co];<br>filter: [Co, Kh, Kw, Ci] 或 [Co, Kd, Kh, Kw, Ci];<br>indices_pairs: [K, 2, L];<br>indices_num: [K];<br>inverse: scalar;<br>sub_m: scalar  |
| 输入Layout  | output_grad: ARRAY;<br>filter: NHWC(HWCN/NCHW) 或 NDHWC (NCDHW);<br>indices_pairs: ARRAY;<br>indices_num: ARRAY;<br>inverse: scalar;<br>sub_m: scalar  |
| 输出数据类型| half, float                                                  |
| 输出Shape   | input_grad: [L, Ci]                                  |
| 输出Layout  | input_grad: ARRAY                                                         |
| 模式(可选） |                                                              |
| 是否含有dim/axis等类似语义的参数且该参数支持负数/其他特殊处理 | 无|
| 是否含有labels/index等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 有index参数，不支持负数 |
| 是否需要支持原位        | 否                                                  |
| 是否需要支持stride机制  | 否                                                  |
| 是否需要支持广播  | 否                                            |
| 0元素检查是否直接返回  | 是                                                  |
| 其他特殊需求(在线量化，融合，转数提前等，可选)|   |
| 本次开发优先支持的规模/模式|   |
### 1.2 算子功能和应用场景描述

#### 1.2.1 应用场景描述

在传统 convolution 计算过程中，如果遇到 input 中有效数据只占极小一部分，对所有数据进行卷积计算会有大量的无效计算，同理反向计算也有大量无效计算。针对上述特征的数据，使用 sparse convolution （稀疏卷积）对有效数据进行操作计算，可以得到传统 convolution 一样的计算结果。

把正向卷积拆解可以得出等效计算逻辑：同一个窗口 input 与 filter 对应 {d, h, w} 坐标的数据执行矩阵乘再累加可得到对应 output 数据，即 [Co, Ci] x 转置[1, Ci] -> [Co, 1]。转置后的数据 [1, Co] 即为对应 {d, h, w} 坐标的部分 output 计算结果。因此传统 convolution 计算可以拆解为多次矩阵乘：任一 {d, h, w} 坐标的 input 数据与任一 {d, h, w} 坐标的 filter 数据可以进行上述矩阵乘计算，并累加到相应 {d, h, w} 坐标的 output 数据中。

Indice convolution backward data 为 sparse convolution 的反向部分算子之一。

#### 1.2.2 算子功能

当 input 数据有大量不参与计算的数据（[1, Ci] 全部不参与计算，多表现为 0 值），这些数据可以直接不进入计算流程，节省操作。

Indice convolution 的关键点为找出需要参与计算的 input 数据块、filter 数据块与对应坐标的 output 数据块。Input 数据由 4 维或 5 维数据中可以提取出参与计算的 2 维矩阵，output 数据也可以提取出参与计算的 2 维矩阵。filter、2 维 input 与 2 维 output 的对应计算关系可以从 indices_pairs 中得到。

Indice convolution backward data 算子执行与上一段 space convolution 相反的操作。Output_grad 为 2 维输出梯度矩阵，filter 与前向相同，indices_pairs 与前向相同，indices_num 记录可索引的 indices_pairs 坐标终点。使用以上四个张量，可以计算出对应的 input_grad 数据。以下公式用 4 维数据作为举例，即等效 conv2d 计算。其中 filter layout 为 NHWC。

第一步的初始化为：

```math
\begin{aligned}
&\forall_{l\in [0,L)} \forall_{ci\in [0,Ci)} input\_grad[l,ci] = 0
\end{aligned}
```

计算公式为：

```math
\begin{aligned}
&\forall_{k\in [0,K)} \forall_{l\in[0,indices\_num[k])} \forall_{ci\in [0,Ci)} ~ input\_grad[indices\_pair[k,0,l],ci]\\
& += \sum_{co\in [0,Co)} (output\_grad[indices\_pair[k,1,l],co] \times filter[co,k/Kw,k\%Kw,ci])
\end{aligned}
```

其中：
- input_grad 与 output_grad 数据由原始张量数据处理得到，由稀疏原始数据处理成稠密 2 维张量。
- indices_pairs 中的坐标信息由原始数据信息处理得到，因此不允许此张量中的坐标信息指向不合理的张量数据位置。最终指向的张量数据位置来自上述 input_grad 与 output_grad 两个 2 维张量。

#### 1.3 算子输入输出参数要求

| 参数             | 语义                                                         | 类型（输入/输出）  | 支持类型        | 物理布局 | 规模限制  |
| ---------------- | ------------------------------------------------------------| ----------------- | ---------------| -------- | -------- |
| handle           | 操作句柄                                                     | 输入              |                | /        | 无       |
| output_grad_desc | 输入tensor output_grad 的描述符                              | 输入              |                | /        | 无       |
| output_grad      | 输入tensor output_grad，从三维点云压缩后的 output 梯度数据     | 输入              | half, float    | ARRAY    | 无       |
| filter_desc      | 输入tensor filter 的描述符                                   | 输入              |                | /        | 无       |
| filter           | 输入tensor filter，卷积操作的权值                             | 输入              | half, float    | NHWC/HWCN/NCHW/NDHWC/NCDHW | 无       |
| indices_pairs_desc| 输入tensor indices_pairs 的描述符                             | 输入              |                | /        | 无       |
| indices_pairs     | 输入tensor indices_pairs，input_grad 与 output_grad 的映射关系 | 输入              | int32          | ARRAY    | 无       |
| indices_num      | 输入数组 indices_num，indices_pairs 中各行的有效数据量          | 输入              | int32          | ARRAY    | 无       |
| inverse          | 输入标量 inverse，描述算子是否执行 inverse 行为                | 输入              | int64          | /        | 无       |
| sub_m            | 输入标量 sub_m，描述算子是否执行 sub_m 计算模式                | 输入              | int64          | /        | 无       |
| input_grad_desc  | 输出tensor input_grad 的描述符                               | 输入              |                | /        | 无       |
| input_grad       | 输出tensor input_grad，从三维点云压缩后的 input 梯度数据       | 输出              | half, float    | ARRAY    | 无       |

### 1.4 算子限制

| 限制类型    | 详细说明                                            |
| ----------- | ------------------------------------------------------------ |
| 数据类型限制| output_grad、filter 和 input_grad 的数据类型必须相同  |
| 布局限制    | 仅支持 filter 维度为 4 或 5                          |
| 规模限制    | 各个 tensor 的各维度大小不能超过 2^31-1（int32 的表示范围）          |
| 规模限制    | K = Kh * Kw 或 K = Kd * Kh * Kw |
| 规模限制    | 当 sub_m = 1 时，Kd * Kh * Kw 不可以为偶数 |
| 规模限制    | 当 sub_m = 1 时，L 与 Y 要相等 |
| 功能限制    | 此版本不支持 inverse = 1 |
| 数据范围限制| indices_pairs 数据必须为非负正数，且上限不能超过其指向的 output_grad 或 input_grad 的 shape 范围（Y 或 L）|
| 原位限制    | 不支持原位|
| stride限制  | 不支持stride机制|
| 广播限制    | 不支持广播|

### 1.5 验收标准

#### 1.5.1 精度验收标准

- output_grad、filter 和 input_grad dtype 为 float：[diff1, diff2], rate = [1e-5, 1e-5]
- output_grad、filter 和 input_grad dtype 为 half： [diff1, diff2], rate = [3e-3, 3e-3]
  
#### 1.5.2 性能验收标准

- 当前方案算子性能受到 scatter、gather 算子性能约束，预计不差于竞品性能的 1/10。待scatter、gather 算子性能优化后重新评估性能验收标准。

## 2 算子接口设计

### 2.1 参考接口- MMCV Pytorch

```c++
std::vector<torch::Tensor> IndiceConvBackwardCUDAKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM);
```

### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API
mluOpGetIndiceConvolutionBackwardDataWorkspaceSize(mluOpHandle_t handle,
                                                   const mluOpTensorDescriptor_t output_grad_desc,
                                                   const mluOpTensorDescriptor_t filters_desc,
                                                   const mluOpTensorDescriptor_t indice_pairs_desc,
                                                   const mluOpTensorDescriptor_t input_grad_desc,
                                                   const int64_t indice_num[],
                                                   const int64_t inverse,
                                                   size_t *workspace_size);
    
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
```

## 3 实现方案设计

### 3.1 实现方案

workspace 空间分成最多 5 份，分别取名为 filter_transpose (optional)、output_grad_condence、input_grad_condence、workspace_input_grad_tmp、workspace_addn。

step 1：若 filter 为 4 维且 layout 为 NHWC/NCHW，调用 transpose 转置为 HWCN；若 filter 为 5 维，调用 transpose 转置为 DHWCN。转置后的 HW/DHW 维度可以等效视作同一维度，即转置后的 filter_transpose shape 可视作 [K,Co,Ci]

step 2：对 K 做循环，每次循环计数为 k。

step 3：调用 gather_nd 算子从 output_grad 中取出数据存到 workspace 上，命名为 output_grad_condence。indices 的数据设置为 indices_pairs[k,1,0:Y']，Y' = indices_num[k]。output_grad_condence shape 为 [Y',Co]

step 4：循环调用 matmul 算子，循环次数为 Y’，循环计数为 y。输入 a 为 output_grad_condence[y,:]，输入 b 固定为 workspace 中转置完的 filter[k,:,:]。其中输入 b 需要配置 is_trans_b 为 true。输出根据循环顺序存至 workspace 的 input_grad_condence。input_grad_condence 整体 shape 为 [Y',Ci]。

step 5：调用 fill 算子，将 workspace_input_grad_tmp 刷 0。

step 6：调用 scatter_nd 算子将 input_grad_condence 更新到 workspace_input_grad_tmp 中。参数 input 与 output 均设置为 workspace_input_grad_tmp，indices 数据设置为 indices_pairs[k,0,0:indices_num[k]]，updates 设置为 input_grad_condence，mode 设置为 MLUOP_SCATTERND_UPDATE。

step 7：调用 addN 算子，将 workspace_input_grad_tmp 的数据加到最终的输出 input_grad 张量中。

step 8：若 k == K - 1，结束循环，否则回到 step 2，k = k + 1。

### 3.2 伪代码实现（可选）

### 3.3 拆分(任务拆分，多核拆分)

任务拆分体现在 filter DHW/HW 维度的循环以及循环调用 matmul 算子。此算子实现均为在 host 端调用多个 MLUOP 算子 API，所有多核拆分由各完整算子内部逻辑执行。

### 3.4 性能优化设计

#### 3.4.1 资源分配

| 表项            | 分配策略   |
| ----------------| -----------|
| NRAM            | 无 |
| WRAM            | 无 |
| SRAM            | 无 |
| DRAM(workspace) | filter_transpose: [K,Co,Ci]（特殊情况可不开）<br>output_grad_condence: [Y'',Co]<br>input_grad_condence: [Y'',Ci] |表格中 Y'' 值为 3.1 中 Y' 所有可能值的最大值。#### 3.4.2 流水设计此算子无 kernel 代码开发，无流水设计。

### 3.5 方案理论性能

理论性能为竞品的 1/5 到 1 倍。可以控制在硬件时间在同一个数量级上。

### 3.6 可维护性设计
1、对所有 workspace 计算的公式有具体的注释，有必要可以增加 LOG2、对每一个函数命名变量命名都有充分的注释3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代

### 3.7 测试用例设计- 框架在需求列表中给出的算子在网络中用到的规模：

| case | output_grad  | filter              | indices_pairs    | input_grad   |
| ---- | ------------ | ------------------- | --------------- | ------------ |
| 0    | [45406, 128] | [3, 1, 1, 128, 128] | [3, 2, 58838]   | [58838, 128] |
| 1    | [58838, 128] | [3, 3, 3, 64, 128]  | [27, 2, 149100] | [149100, 64] |
| 2    | [58838, 128] | [3, 3, 3, 128, 128] | [27, 2, 58838]  | [58838, 128] |
| 3    | [248636, 16] | [3, 3, 3, 5, 16]    | [3, 2, 58838]   | [248636, 5]  |

以上 case 为部分规模。其他可根据需要进行补充。算子开发完毕后，补充测试报告链接。

### 3.8 算子防呆检查

#### 3.8.1 空指针、硬件防呆

```c++
  PARAM_CHECK(api, handle != NULL);
  PARAM_CHECK(api, output_grad_desc != NULL);
  PARAM_CHECK(api, filters_desc != NULL);
  PARAM_CHECK(api, indice_pairs_desc != NULL);
  PARAM_CHECK(api, input_grad_desc != NULL);

  if (handle->arch != MLUOP_MLU370 && handle->arch != MLUOP_MLU590) {
    LOG(ERROR) << api << " Only support hardware over MLU300 .";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  PARAM_CHECK(api, output_grad != NULL);
  PARAM_CHECK(api, filters != NULL);
  PARAM_CHECK(api, indice_pairs != NULL);
  PARAM_CHECK(api, input_grad != NULL);
```

#### 3.8.2 0元素检查防呆

```c++
  if (input_grad_count == 0) {
    LOG(INFO) << "input_grad is a zero-element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
  if (output_grad_count == 0) {
    LOG(INFO) << "output_grad is a zero-element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
  if (filter_count == 0) {
    LOG(INFO) << "filters is a zero-element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
  if (indice_pairs_count == 0) {
    LOG(INFO) << "indice_pairs is a zero-element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
```

#### 3.8.3 workspace、workspace_size 的检查防呆

```c++
  if (workspace_size > 0) {
    PARAM_CHECK(api, workspace != NULL);
  }
```

#### 3.8.4 dtype、layout 以及 shape 防呆

```c++
  PARAM_CHECK_EQ(api, output_grad_desc->dim, 2);
  PARAM_CHECK(api, filters_desc->dim == 4 || filters_desc->dim == 5);
  PARAM_CHECK_EQ(api, indice_pairs_desc->dim, 3);
  PARAM_CHECK_EQ(api, input_grad_desc->dim, 2);

  PARAM_CHECK(api, indice_pairs_desc->dims[1] == 2);

  PARAM_CHECK(api, output_grad_desc->dtype == MLUOP_DTYPE_FLOAT ||
                   output_grad_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK(api, filters_desc->dtype == MLUOP_DTYPE_FLOAT ||
                   filters_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK(api, input_grad_desc->dtype == MLUOP_DTYPE_FLOAT ||
                   input_grad_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK(api, indice_pairs_desc->dtype == MLUOP_DTYPE_INT32);
  bool layout_check = filters_desc->layout == MLUOP_LAYOUT_NHWC ||
                      filters_desc->layout == MLUOP_LAYOUT_NCHW ||
                      filters_desc->layout == MLUOP_LAYOUT_HWCN ||
                      filters_desc->layout == MLUOP_LAYOUT_NCDHW ||
                      filters_desc->layout == MLUOP_LAYOUT_NDHWC ||
                      filters_desc->layout == MLUOP_LAYOUT_ARRAY;
  if (!layout_check) {
    LOG(ERROR) << api
               << " The filters tensor only supports "
                  "NHWC/NCHW/HWCN/NCDHW/NDHWC/ARRAY layout.";
    return MLUOP_STATUS_BAD_PARAM;
  }

```

#### 3.8.5 算子存在的自身的相关参数防呆

```c++
  int kd = 1, kh = 1, kw = 1, dyc = 1, dxc = 1;
  if (filters_desc->layout != MLUOP_LAYOUT_ARRAY) {
    kh = mluOpGetTensordimH(filters_desc);
    kw = mluOpGetTensordimW(filters_desc);
    dyc = mluOpGetTensordimN(filters_desc);
    dxc = mluOpGetTensordimC(filters_desc);
    if (filters_desc->dim == 5) {
      kd = mluOpGetTensordimD(filters_desc);
    }
  } else {
    if (filters_desc->dim == 5) {
      kd = filters_desc->dims[0];
    }
    int _dim = filters_desc->dim;
    kh = filters_desc->dims[_dim - 4];
    kw = filters_desc->dims[_dim - 3];
    dxc = filters_desc->dims[_dim - 2];
    dyc = filters_desc->dims[_dim - 1];
  }
  int K = kd * kh * kw;

  // check param
  PARAM_CHECK(api, inverse == 0 || inverse == 1);
  PARAM_CHECK(api, sub_m == 0 || sub_m == 1);
  for (int kk = 0; kk < K; ++kk) {
    PARAM_CHECK(api, indice_num[kk] >= 0);
  }
  if (inverse == 1) {
    LOG(ERROR) << api << " Not support inverse == 1 yet.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // check algorithm, relationship between params
  if (K != indice_pairs_desc->dims[0]) {
    LOG(ERROR) << api
               << " The dims[0] of indice_pairs should be equal to the "
                  "multiple of kd, kh and kw.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (output_grad_desc->dims[1] != dyc) {
    LOG(ERROR) << api
               << " The dims[1] of output_grad should be equal to dyc of "
                  "filters tensor.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (input_grad_desc->dims[1] != dxc) {
    LOG(ERROR) << api
               << " The dims[1] of input_grad should be equal to dxc of "
                  "filters tensor.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (input_grad_desc->dims[0] != indice_pairs_desc->dims[2]) {
    LOG(ERROR) << api
               << " The dims[0] of input_grad should be equal to the dims[2] "
                  "of indice_pairs.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  int max_indice_num = getMaxNumInArray(indice_num, K);

  if (indice_pairs_desc->dims[2] < max_indice_num) {
    VLOG(5) << "indice_pairs_desc->dims[2] " << indice_pairs_desc->dims[2]
            << " max_indice_num " << max_indice_num;
    LOG(ERROR) << api
               << " The data in indice_num array should be smaller or equal to"
               << " the dims[2] of indice_pairs.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (sub_m == 1) {
    if (input_grad_desc->dims[0] != output_grad_desc->dims[0]) {
      LOG(ERROR) << api
                 << " The dims[0] of input_grad should be equal to the dims[0]"
                 << " of output_grad when sub_m is 1.";
      return MLUOP_STATUS_BAD_PARAM;
    }

    if (indice_num[K / 2] < max_indice_num) {
      LOG(ERROR) << api
                 << " The middle number of the indice_num array should be the "
                 << "maximum of the array when sub_m is 1. Now the maximum is "
                 << max_indice_num << " while the middle number of the array "
                 << "is " << indice_num[K / 2] << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  if (output_grad_desc->dims[0] < max_indice_num) {
    LOG(ERROR)
        << api
        << " The dims[0] of output_grad should be larger than or equal to the"
        << " maximum number of indice_num.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (sub_m == 1 && K % 2 == 0) {
    LOG(ERROR) << api << " When sub_m value is 1, the filters dims (Kd, Kh & "
               << "Kw) should be odd numbers.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  PARAM_CHECK(api, output_grad_desc->dtype == input_grad_desc->dtype);
  PARAM_CHECK(api, output_grad_desc->dtype == filters_desc->dtype);

  // check constraints: not support large tensor
  uint64_t input_grad_count = mluOpGetTensorElementNum(input_grad_desc);
  TENSOR_NUM_CHECK(api, input_grad_count, LARGE_TENSOR_NUM,
                   "input_grad tensor num is too large. ");
  uint64_t output_grad_count = mluOpGetTensorElementNum(output_grad_desc);
  TENSOR_NUM_CHECK(api, output_grad_count, LARGE_TENSOR_NUM,
                   "output_grad tensor num is too large. ");
  uint64_t filter_count = mluOpGetTensorElementNum(filters_desc);
  TENSOR_NUM_CHECK(api, filter_count, LARGE_TENSOR_NUM,
                   "filters tensor num is too large. ");
  uint64_t indice_pairs_count = mluOpGetTensorElementNum(indice_pairs_desc);
  TENSOR_NUM_CHECK(api, indice_pairs_count, LARGE_TENSOR_NUM,
                   "indice_pairs tensor num is too large. ");

```

## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

### 4.2 已经过优化的规模说明