# 1. Adamom 算子开发方案设计

* #### 文档基本信息

| 算子名称     | Adamom      |
| -----------  | -------   |
| 编制人/日期  | 汪凌峰/2025-03-16 |

* #### 修订记录

| 版本号 | 修订人 | 修订日期 | 修订描述 |
|------- | ----- | ------- | -------- |
| V1.0   | 汪凌峰 | 2025-03-16 | 最初提交 |

* #### 内容描述

本文档为`Adamom`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录。

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

| 算子功能简介           | 一种基于梯度下降算法的优化算法。 |
| ---------------------- | ------------------------------------------- |
| 需求来源               | 无                                     |
| 输入数据类型           | float                             |
| 输入Shape              | 任意多维数组 |
| 输入Layout             | ARRAY                                       |
| 输出数据类型           | 同输入                             |
| 输出Shape              | 同输入                              |
| 输出Layout             | 同输入                                       |
| 模式(可选)             | 无                                          |
| 是否需要支持原位       | 是                                       |
| 是否需要支持stride机制 | 否                                         |
| 是否需要支持广播       | 否                                          |


### 1.2 算子功能和应用场景描述
- `Adamom`是一种优化算法。按以下公式进行更新：
  $$
  dx = grad + weight\_decay*weight
  $$

  $$
  new\_v = beta2*v+dx*dx
  $$

  $$
  new\_v\_bias\_correction=beta2*v\_bias\_correction+1
  $$

  $$
  new\_m=m*beta1+dx*(1-beta1)
  $$

  $$
  eta=lr*\sqrt{\frac{1}{\frac{v}{new\_v\_bias\_correction}+epsilon}}
  $$

  $$
  new\_weight=weight-eta*m
  $$

当满足条件(isfinite(new\_m) &&isfinite(new_v)&&isfinite(new_weight)&&v>=0)时做以下更新 ：
$$
m=new\_m
$$

$$
v=new\_v
$$

$$
v\_bias\_correction=new\_v\_bias\_correction
$$

$$
weight=new\_weight
$$

### 1.3 算子输入输出参数要求

| 参数          | 语义                            | 类型（输入/输出） | 支持类型                | 物理布局 | 规模限制 |
| ------------- | ------------------------------- | ----------------- | ----------------------- | -------- | -------- |
| handle        | MLU-OPS句柄，保存运行上下文信息 | 输入              | mluOpHandle_t           | 无       | 无       |
| grad_desc | 输入参数grad的描述信息     | 输入              | mluOpTensorDescriptor_t  | 无       | 无       |
| grad | 指向输入数据grad的mlu地址的指针     | 输入              | void* | 无       | 无       |
| ms_desc        | 输入/输出参数ms的描述信息 | 输入/输出    | mluOpTensorDescriptor_t                   | 无       | 无       |
| ms | 指向输入/输出数据ms的mlu地址的指针 | 输入/输出         | void* | 无       | 无       |
| vs_desc | 输入/输出参数vs的描述信息 | 输入/输出 | mluOpTensorDescriptor_t | 无 | 无 |
| vs | 指向输入/输出数据vs的mlu地址的指针 | 输入/输出 | void* | 无 | 无 |
| v_bias_corrections_desc | 输入/输出参数v_bias_corrections的描述信息 | 输入/输出 | mluOpTensorDescriptor_t | 无 | 无 |
| v_bias_corrections | 指向输入/输出数据v_bias_corrections的mlu地址的指针 | 输入/输出 | void* | 无 | 无 |
| weights_desc | 输入/输出参数weights的描述信息 | 输入/输出 | mluOpTensorDescriptor_t | 无 | 无 |
| weights | 指向输入/输出数据weights的mlu地址的指针 | 输入/输出 | void* | 无 | 无 |
| nan_inf_found | 指向输入参数nan_inf_found的mlu地址的指针，预留参数，当前未实现相关功能，但不能为NULL | 输入 | void* | 无 | 无 |
| lr | 指向输入参数lr的mlu地址的指针 | 输入 | void* | 无 | 无 |
| beta1 | 指向输入参数beta1的mlu地址的指针 | 输入 | void* | 无 | 无 |
| beta2 | 指向输入参数beta2的mlu地址的指针 | 输入 | void* | 无 | 无 |
| weight_decay | 指向输入参数weight_decay的mlu地址的指针 | 输入 | void* | 无 | 无 |
| epsilon | 指向输入参数epsilon的mlu地址的指针 | 输入 | void* | 无 | 无 |


### 1.4 算子限制

| 限制类型     | 详细说明                    |
| ------------ | --------------------------- |
| 数据类型限制 | 输入输出数据仅支持float     |
| 布局限制     | ARRAY                       |
| 数据范围限制 | 无                          |
| 原位限制     | 支持原位                    |
| stride 限制  | 不支持 stride               |
| 广播限制     | 不支持广播                  |
| shape 限制   | 输入输出向量的shape保持一致 |

### 1.5 验收标准
精度验收标准：在相同数据类型的计算下，CPU和MLU计算结果所有数据单点误差的最大值，不得超过1E-7。
性能验收标准：性能IO/计算效率有一项不低于50%。

## 2 算子接口设计

### 2.1 参考接口

自定义算子，框架无对应实现




### 2.2 接口设计

```c
mluOpStatus_t MLUOP_WIN_API mluOpAdamom(
    mluOpHandle_t handle,
    const mluOpTensorDescriptor_t grad_desc, void *grad,
    const mluOpTensorDescriptor_t ms_desc, void *ms,
    const mluOpTensorDescriptor_t vs_desc, void *vs,
    const mluOpTensorDescriptor_t v_bias_corrections_desc, void *v_bias_corrections,
    const mluOpTensorDescriptor_t weights_desc, void *weights,
    const void *lr,
    const void *beta1, const void *beta2, const void *weight_decay,
    const void *epsilon)
```

## 3 实现方案设计

adamom算子是element wise类型的算子，因此只需要按照输入数据进行数据切分，目前算子实现了多job并行机制，可按照每个core上可用nram空间对输入数据量进行拆分。

### 3.1 实现方案
- 对输入数据按照数据大小进行拆分，计算每个core处理的数据大小per_core_data。
- 根据每个core的可用nram空间对per_core_data进行拆分，若nram放不下，则循环处理。

分析以上，具体代码可调用三级流水模板实现。

### 3.2 伪代码实现（可选）

### 3.3 拆分（任务拆分，多核拆分）

### 3.4 性能优化设计

调用三级流水模板实现

### 3.5 可维护性设计

bangc代码中加入必要的 log信息，比如输入的规模、数据类型、layout这些，以及如果出错会导致程序core dump的变量，比如IO指令的data_size、dim xyz的值等，这些信息都是有利于快速定位问题。

### 3.6 测试用例设计

- 该算子在网络中用到的规模：
无

### 3.7 算子防呆检查

防呆报错使用错误码： `MLUOP_STATUS_BAD_PARAM`，`MLUOP_STATUS_ARCH_MISMATCH`。 

1. 检查handle/grad_desc/ms_desc/vs_desc/v_bias_corrections_desc/weights_desc/grads/ms/vsv_bias_corrections/weights/nan_inf_found/lr/beta1/beta2/weight_decay/epsilon是否为空。
2. 检查各输入数据类型是否都是float。
3. 检查各输入的shape是否相同。

## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

（首次提交，暂无） 

### 4.2 已经过优化的规模说明

（首次提交，暂无）