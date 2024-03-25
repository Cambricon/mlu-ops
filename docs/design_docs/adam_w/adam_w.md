# AdamW算子开发设计方案


* #### 文档基本信息

| 算子名称    | AdamW             |
| ----------- | ----------------- |
| 编制人/日期 | 龚恒嘉/2024-03-18 |

* #### 修改记录

| 版本号 | 修订人 | 修订日期   | 修订描述 |
| ------ | ------ | ---------- | -------- |
| V1.0   | 龚恒嘉 | 2024-03-18 | 首次提交 |

* #### 内容描述

本文档为`AdamW`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录。

- #### 算子需求 checklist

* 算子接口描述
* 功能描述
* 框架版本 + 对应源码路径
* 需求对应网络
* 网络中用到的规模
* 是否需要支持原位
* 是否需要支持 stride 机制
* 框架单元测试阈值指标（可选）

## 1 需求分析

### 1.1 算子需求分析

| 算子功能简介           | 一种基于梯度下降算法的优化算法。            |
| ---------------------- | ------------------------------------------- |
| 需求来源               | BMTrain                                     |
| 输入数据类型           | bfloat16, float                             |
| 输入Shape              | 所有的输入向量的形状规模一致，各维度均大于0 |
| 输入Layout             | ARRAY                                       |
| 输出数据类型           | bfloat16, float                             |
| 输出Shape              | 与输入规模一致                              |
| 输出Layout             | ARRAY                                       |
| 模式(可选)             | 无                                          |
| 是否需要支持原位       | 否                                          |
| 是否需要支持stride机制 | 否                                          |
| 是否需要支持广播       | 否                                          |



### 1.2 算子功能和应用场景描述

`Adam`是一种自适应学习率优化算法，它结合了动量梯度下降和RMSProp算法的思想。`Adam`使用指数移动平均来估计每个参数的动量和二次矩量，并将它们用于调整学习率。
`AdamW`是对`adam`算法的一个改进版本，它主要是为了解决`adam`算法中的权重衰减问题，`AdamW`引入了一种新的权重衰减方式，将权重衰减添加到损失函数中，提高了优化器的性能和稳定性。

grad为bfloat16时按以下公式进行更新：


$$
grad = \frac{bfloat162float(grad)}{scale}
$$

$$
m_{t} = \beta_1 * m_{t-1} + (1 - \beta_1) * grad
$$

$$
v_{t} = \beta_2 * v_{t-1} + (1 - \beta_2) * grad * grad
$$

$$
param_{t} = param_{t-1} - \frac{lr * m_{t}}{bias_1 * (\sqrt{\frac{v_{t}}{bias_2}} + epsilon)} - lr * weight\_decay * param_{t-1}
$$



### 1.3 算子输入输出参数要求

| 参数          | 语义                            | 类型（输入/输出） | 支持类型                | 物理布局 | 规模限制 |
| ------------- | ------------------------------- | ----------------- | ----------------------- | -------- | -------- |
| handle        | MLU-OPS句柄，保存运行上下文信息 | 输入              | mluOpHandle_t           | 无       | 无       |
| adamw_desc    | 输入参数adamw的描述信息         | 输入              | mluOpAdamwDescriptor_t  | 无       | 无       |
| param_desc    | 输入参数param的描述信息         | 输入              | mluOpTensorDescriptor_t | 无       | 无       |
| param         | 指向param数据的mlu地址的指针    | 输入/输出         | void*                   | 无       | 无       |
| paramh_desc   | 输入参数paramh的描述信息        | 输入              | mluOpTensorDescriptor_t | 无       | 无       |
| param_h       | 指向paramh数据的mlu地址的指针   | 输入/输出         | void*                   | 无       | 无       |
| momentum_desc | 输入参数momentum的描述信息      | 输入              | mluOpTensorDescriptor_t | 无       | 无       |
| momentum      | 指向momentum数据的mlu地址的指针 | 输入/输出         | void*                   | 无       | 无       |
| velocity_desc | 输入参数velocity的描述信息      | 输入              | mluOpTensorDescriptor_t | 无       | 无       |
| velocity      | 指向velocity数据的mlu地址的指针 | 输入/输出         | void*                   | 无       | 无       |
| grad_desc     | 输入参数grad的描述信息          | 输入              | mluOpTensorDescriptor_t | 无       | 无       |
| grad          | 指向grad数据的mlu地址的指针     | 输入              | void*                   | 无       | 无       |
| lr            | 浮点标量                        | 输入              | float                   | 无       | 无       |
| beta1         | 浮点标量                        | 输入              | float                   | 无       | 无       |
| beta2         | 浮点标量                        | 输入              | float                   | 无       | 无       |
| bias1         | 浮点标量                        | 输入              | float                   | 无       | 无       |
| bias2         | 浮点标量                        | 输入              | float                   | 无       | 无       |
| epsilon       | 浮点标量                        | 输入              | float                   | 无       | 无       |

### 1.4 算子限制

| 限制类型     | 详细说明                                                     |
| ------------ | ------------------------------------------------------------ |
| 数据类型限制 | grad和param_h类型均为bfloat16<br />param, momentum, velocity类型均为float<br />其余标量输入均为float |
| 布局限制     | ARRAY                                                        |
| 数据范围限制 | epsilon必须大于0,输入参数epsilon是一个极小值常量，是为了避免分母为0<br />beta1、beta2必须为[0, 1] |
| 原位限制     | 不支持原位                                                   |
| stride 限制  | 不支持 stride 机制                                           |
| 广播限制     | 不支持广播                                                   |
| shape 限制   | 输入向量的shape保持一致                                      |

### 1.5 验收标准

精度验收标准：diff1/diff2 cpu/gpu 3e-3。

性能验收标准：能达到理论分析数值，和GPU对比，分析差距。



## 2 算子接口设计

### 2.1 参考接口

- BMTrain

```c++
// CUDA(https://github.com/OpenBMB/BMTrain/blob/6abcf772aa1e120192f7656e55c4adbcde53c886/csrc/cuda/adam_cuda.cu#L39)
__global__ void adam_fp32_accum_bf16(
    int32_t n,
    const std::uintptr_t g_ptr,        // (n)
    float *m,        // (n)
    float *v,        // (n)
    float *param,   // (n)
    std::uintptr_t param_h_ptr,  // (n)
    float beta1,
    float beta2,
    float eps,
    float lr,
    float scale,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
)
```

### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API mluOpApplyAdamw(
    mluOpHandle_t handle,
    const mluOpAdamwDescriptor_t adamw_desc,
    const mluOpTensorDescriptor_t param_desc, void *param,
    const mluOpTensorDescriptor_t paramh_desc, void *param_h,
    const mluOpTensorDescriptor_t momentum_desc, void *momentum,
    const mluOpTensorDescriptor_t velocity_desc, void *velocity,
    const mluOpTensorDescriptor_t grad_desc, void *grad,
    const float lr,
    const float beta1, const float beta2, const float bias1,
    const float bias2, const float epsilon)

```

## 3 实现方案设计

adamw算子是element wise类型的算子，因此只需要按照数据量进行相应的分配即可，目前算子实现了多job并行机制，可按照每个Core上可用Nram空间对输入数据量进行拆分。

### 3.1 实现方案

- 使用Double buffer并行计算
- adamw算子是element wise类型的算子，因此只需要按照数据量进行相应的分配即可

### 3.2 伪代码实现（可选）

### 3.3 拆分（任务拆分，多核拆分）

1. 将数据量按核的数量进行拆分，可能存在剩余数据量，记为rem_for_all。
2. 由于NRAM的存储空间有限，导致每个Core分配到的数据量无法一次性处理完成，因此需要做多次处理，在这种平分中，也存在剩余数据量。为保证剩余数据为零，在拆分时需要考虑每个核处理数据对单次循环能处理的数据长度对齐。
3. 对每个核分配到的数据量进行循环处理之后，最后一个核处理rem_for_all的数据。由于大多数MLU存在多个计算核，因此将数据的计算拆分到不同的核上并行计算可以大幅提升性能，本算子拆分后的任务无需进行数据通信交互，任务类型为block。

多核间的拆分：由于软流水中的内存拷贝指令要求内存地址为128的整数倍，因此在拆分时就需要保证每个部分的首地址都是128的整数倍，具体的做法是先将整体向量以128字节为单位分解成小块，然后再将小块分配到核上，每个核计算一份，然后由第一个核计算分解为小块剩余的元素和分配给core时剩余的小块。

单核内的拆分：由于片上NRAM空间的限制，因此在数据量较大时不能一次处理完单核分配到的全部数据，这时就需要在单核内循环，每一次只处理一部分元素，元素的个数由NRAM上指定给该向量的空间大小决定。

### 3.4 性能优化设计

adamw算子为标准的load compute store三段式算子，且计算操作均为元素与元素之间一一对应的操作，不涉及到复杂的元素间交叉运算，可以排三级流水。

### 3.5 可维护性设计

bangc代码中加入必要的 log信息，比如输入的规模、数据类型、layout这些，以及如果出错会导致程序core dump的变量，比如IO指令的data_size、dim xyz的值等，这些信息都是有利于快速定位问题。

### 3.6 测试用例设计

- 该算子在网络中用到的规模：

- **input**

  ```c
  param: [999, 9, 16]
  param_h: [999, 9, 16]
  momentum: [999, 9, 16]
  velocity: [999, 9, 16]
  grad: [999, 9, 16]
  lr: 0.0008250000000000001
  beta1: 0.9
  beta2: 0.95
  bias1: 0.18999999999999995
  bias2: 0.09750000000000003
  epsilon: 1e-08
  weight_decay: 0.1
  scale: 1.0
  ```

  - **output**

    ```c
    param: [999, 9, 16]
    param_h: [999, 9, 16]
    momentum: [999, 9, 16]
    velocity: [999, 9, 16]
    ```

### 3.7 算子防呆检查

防呆报错使用错误码： `MLUOP_STATUS_BAD_PARAM, MLUOP_STATUS_ARCH_MISMATCH, MLUOP_STATUS_ALLOC_FAILED`。 

1. 检查handle/param_desc/paramh_desc/momentum_desc/velocity_desc/grad_desc是否为空。
2. 检查param, paramh, momentum, velocity, grad这几个输入参数的数据个数是否相同。

## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

（首次提交，暂无） 

### 4.2 已经过优化的规模说明

（首次提交，暂无）