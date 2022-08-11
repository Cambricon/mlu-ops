# HardShrink 算子开发设计方案

- #### 文档基本信息

| 算子名称      | HardShrink            |
| ----------- | ------------------ |
| 编制人/日期   | liuguoxiang/2022-03-06 |
| 审批人/日期   |                    |

- #### 修改记录

|   修订人  | 修订日期     | 修订描述  |
| -------- | ---------- | -------- |
| liuguoxiang | 2022-03-06 | 首次提交  |

- #### 内容描述

本文档为`HardShrink`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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

| 算子功能简介       |      激活函数，逐元素施加强制收缩    |
| ---------------- | ---------------------------------- |
| 需求来源          |      PyTorch                          |
| 应用网络          |         rarely used                 |
| 输入数据类型       |      half,float                    |
| 输入shape        |      无shape限制                         |
| 输入layout       |      无layout限制                    |
| 输出数据类型       |      同输入类型                         |
| 输出shape         |      同输入类型   |
| 输出layout        |      同输入类型      |

### 1.2 算子功能和应用场景描述

功能： HardShrink算子是一种激活函数，其功能是逐元素施加强制收缩。
运算公式描述如下：

$$Hardshrink(x)=\begin{cases}
 x,&x>\lambda \\
 x,&x<-\lambda \\
 0,&otherwise 
\end{cases}$$

**lambda** – the λ value for the Hardshrink formulation. Default: 0.5

来自[PyTorch HardShrink](https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#Hardshrink)。
例如： 

``` python
m = nn.Hardshrink(lambda=0.5)
inputTensor = torch.rand(2)
outputTensor = m(input)
# lambda : 0.5 inputTensor =  tensor([0.7286, 0.0830])
# expected outputTensor : tensor([0.7286, 0.0000])
```

应用场景： sparse coding等

### 1.3 算子输入输出参数要求

| 参数      |             语义         | 类型（输入/输出） | 支持类型    | 物理布局 | 规模限制 |
| -------- | ------------------------ | -------------- | ---------- | ------- | ------- |
| input    |输入Tensor                 |   输入         | half,float  |  Array | \  |
| lambda    | the \lambda value for the Hardshrink formulation. Default: 0.5| 输入           | float         | 无限制   |  0或1  |
| output   |输出Tensor                 |  输出          | 同input       | 同input  | 无限制 |

### 1.4 算子限制

| 限制类型     | 详细说明              |
| ----------- | ------------------- |
| 数据类型限制     | input 和 output 必须同时为同一数据类型 |
| 原位限制     | 支持原位            |
| stride 限制  | 不支持stride 机制      |

### 1.5 验收标准

#### 1.5.1 精度验收标准

按照[精度验收标准](../../../MLU-OPS精度验收标准.md)的要求明确本算子的精度标准。

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../../../MLU-OPS性能验收标准.md)。

## 2 算子接口设计

### 2.1 参考接口

- PyTorch

Pytorch 接口：
```python
m = nn.Hardshrink(lambda=0.5)
inputTensor = torch.rand(2)
outputTensor = m(input)
```


### 2.2 接口设计

- HardShrink计算接口：

```python
mluopHardShrink(inputTensor,lambda=0.5,shape[0],shape[1],shape[2],shape[3],outputTensor)
```

## 3 实现方案设计

### 3.1 实现方案

**1. HardShrink实现如下：**

目前主要是两种实现思路:
1.纯张量操作，先 abs ，然后 greater 生成0-1矩阵，然后根据0-1矩阵与原矩阵进行 multiply
优点：效率可能会高一些
缺点：受限于一些尺寸、对齐的要求
2.借助控制流语句如 for if 等 来实现elment-wise的操作
优点：灵活
缺点：效率可能会低，但是好像看选项，针对循环也是有一定的优化的



### 3.2 伪代码实现

1.纯张量操作

```python
tensor_in = tcp.match_buffer(shape, name = "INPUT", dtype = dtype, scope = "global")
tensor_out = tcp.match_buffer(shape, name = "OUTPUT" ,dtype = dtype, scope = "global")
scalar_lambda = value

tensor_in_n = tcp.alloc_buffer(shape, name="INPUT_N",dtype = dtype, scope = "nram")
tensor_out_n = tcp.alloc_buffer(shape, name = "OUTPUT" ,dtype = dtype, scope = "nram")
scalar_lambda_n = tcp.alloc_buffer(scalar_lambda, scope = "nram")

memcpy(gloabl, nram)

tensor_temp_abs = tcp.alloc_buffer(scope = "nram")
abs(tensor_in_n,tensor_temp_abs)
tensor_temp_01 = tcp.alloc_buffer(scope = "nram")
greater(tensor_temp_abs, scalar_lambda_n, tensor_temp_01)
multiply(tensor_input_n, tensor_temp_01, tensor_out_n)

memcpy(nram, global)
```

2.控制流语句element-wise
控制流语句按常规写法即可。

### 3.3 拆分（任务拆分，多核拆分）

1、把所有的数据均分给每个core，计算好每个core需要处理多少数据量以及起始位置，多余的数据有两种处理方式：(1)让最后一个core处理，实现core间并行，每个core自己算完后把结果store到对应位置（比如总共102个数据，有4个core，core0处理25个数，core1处理25个数，core2处理25个数，core3处理27个数）；(2)剩余的任务按序分配给core(core0处理26个数，core1处理26个数，core2处理25个数，core3处理25个数)，为了实现硬件负载均衡，采用第二种方案；
2、如果core自己需要处理的数据量很大，一次处理不完，这个时候需要分批处理，这时候就有多次的Load, compute,store，这部分可以实现流水

### 3.4 性能优化设计

1.自动流水；
2.内存复用

### 3.5 可维护性设计

1、对每一个函数命名变量命名都有充分的注释；
2、对算子进行模块化设计与拆分，确保模块具有复用性。

### 3.6 测试用例设计

包括三部分，分别是数据类型、数据规模和硬件支持。数据类型包括：[float16, float32]，数据规模测试任意范围数据（极大、极小、不规则（不对齐）等），包括(4, 16, 1024, 1024)，(4,16,1,64)，(3, 5, 197, 175)，硬件支持包括：["mlu370", "mlu290"]


## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

| 提交日期  | 问题规模 | 问题描述 | 是否已修复 |
| --------- | -------- | -------- | ---------- |
|           |          |          |            |

### 4.2 已经过优化的规模说明

| 提交日期  | 修复规模 | 修复问题 |
| --------- | -------- | -------- |
|           |          |          |

## 5 方案实施

### 5.1 开发测试计划

- 2022.4.30 算子入库

### 5.2 风险分析

暂无。



