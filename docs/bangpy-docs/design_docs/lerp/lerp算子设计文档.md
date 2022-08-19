# BANGPy Lerp 算子开发设计方案

- #### 文档基本信息

| 算子名称    | Lerp       |
| ----------- | -------------- |
| 编制人/日期 | 郑磊磊/2022-3-1 |
| 审批人/日期 |    |

- #### 修改记录

| 修订人 | 修订日期   | 修订描述 |
| ------ | ---------- | -------- |
| 郑磊磊    | 2022-3-1 | 首次提交 |
| liuguoxiang    | 2022-8-17 | 适配新版本bangpy2.0 |

- #### 内容描述

本文档为 `Lerp` 算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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


| 算子功能简介   | 基于权重的线性插值          |
| ------------ | ---------------------------------------------|
| 需求来源       | PyTorch                                     |
| 应用网络       | 未知                                  |
| 输入数据类型   | half, float                                 |
| 输入 Shape    | input_start.dim = 4, input_end.dim = 4, input_weight.dim = 4        |
| 输入 Layout   | 无layout限制                               |
| 输出数据类型    | 同输入类型                                 |
| 输出 Shape    | 同输入类型                  |
| 输出 Layout   | 同输入类型                                         |

### 1.2 算子功能和应用场景描述

功能：基于权重的线性插值。
运算公式描述如下：
    $$out_i = start_i + weight_i * (end_i - start_i)$$
	weight默认为标量，值为0.5

例如：
    start = [[1., 2.], [3., 4.]]
    end = [[5., 6.], [8., 9.]]
    weight = 0.5
    output = [[3., 4.], [5.5, 6.5]]

### 1.3 算子输入输出参数要求

| 参数        | 语义 | 类型（输入/输出） | 支持类型    | 物理布局 | 规模限制 |
| -----------| ---------------------------- | ----------------- | ----------- | -------- | -------- |
| input_start      | 输入张量1     | 输入              | half, float | Array     | 无       |
| input_end      | 输入张量2      | 输入      | half, float | Array     | 无       |
| input_weight      | 权重张量     | 输入              | half, float | Array     | 无       |
| output     | 输出张量     | 输出              | half, float | Array     | 无       |

### 1.4 算子限制

| 限制类型     | 详细说明                                                                                            |
| ------------ | ------------------------------------------------------------------------------------------------- |
| 数据类型限制 | input 和 output 必须同时为统一数据类型                                 |
| 原位限制     | 支持原位 |
| 广播限制 | 不支持广播 |
| stride限制 | 不支持stride机制 |
| 输入shape | 目前不支持任意shape，只支持dim = 4 |

### 1.5 验收标准

#### 1.5.1 精度验收标准

本算子属于 `算术` 类算子，验收标准为 diff3=0。

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../../../MLU-OPS性能验收标准.md)。

## 2 算子接口设计

### 2.1 参考接口

- PyTorch

```python
out = torch.lerp(input, end, weight)
```

### 2.2 接口设计

```python
MluOpLerp(input_start, input_end, input_weight, shape[0], shape[1], shape[2], shape[3], output)
```

## 3 实现方案设计

### 3.1 实现方案
目前主要是两种实现思路:
1.纯张量操作，先 input_end 与input_start 进行 subtract ，然后结果张量与权重张量进行 multiply ，得到的中间张量与 input_start 进行add得到结果张量
优点：效率可能会高一些
缺点：受限于一些尺寸、对齐的要求
2.借助控制流语句如 for if 等 来实现elment-wise的操作
优点：灵活
缺点：效率可能会低，但是好像看选项，针对循环也是有一定的优化的


### 3.2 伪代码实现

```python
tensor_in_start = tcp.match_buffer(shape, name = "INPUT_Start", dtype = dtype, scope = "global")
tensor_int_end = tcp.match_buffer(shape, name = "INPUT_End", dtype = dtype, scope = "global")
tensor_out = tcp.match_buffer(shape, name = "OUTPUT", dtype = dtype, scope = "global")
tensor_weight = tcp.match_buffer(shape, name = "WEIGHT", dtype = dtype, scope = "global")
tensor_in_start_n = tcp.alloc_buffer(shape, name="INPUT_Start_N", dtype = dtype, scope = "nram")
tensor_in_end_n = tcp.alloc_buffer(shape, name="INPUT_End_N", dtype = dtype, scope = "nram")
tensor_out_n = tcp.alloc_buffer(shape, name = "OUTPUT" , dtype = dtype, scope = "nram")
tensor_weight_n = tcp.alloc_buffer(shape, name = "WEIGHT_N", dtype = dtype, scope = "nram")
memcpy(gloabl, nram)
# tensor_in_end_n 复用
subtract(tensor_in_end_n, tensor_in_end_n, tensor_in_start_n)
multiply(tensor_in_end_n, tensor_weight_n, tensor_in_end_n)
add(tensor_out_n, tensor_in_start_n, tensor_in_end_n)
memcpy(nram, global)
```
### 3.3 拆分(任务拆分，多核拆分)

1、把所有的数据均分给每个core，计算好每个core需要处理多少数据量以及起始位置，多余的数据有两种处理方式：(1)让最后一个core处理，实现core间并行，每个core自己算完后把结果store到对应位置（比如总共102个数据，有4个core，core0处理25个数，core1处理25个数，core2处理25个数，core3处理27个数）；(2)剩余的任务按序分配给core(core0处理26个数，core1处理26个数，core2处理25个数，core3处理25个数)，为了实现硬件负载均衡，采用第二种方案；

2、如果core自己需要处理的数据量很大，一次处理不完，这个时候需要分批处理，这时候就有多次的Load, compute,store，这部分可以实现流水

### 3.4 性能优化设计

1.自动流水；
2.内存复用；

### 3.5 可维护性设计

1.对每一个函数命名变量都有充分的注释；

2.对算子进行模块化设计与拆分，确保模块具有复用性；

### 3.6 测试用例设计
包括三部分，分别是数据类型、数据规模和硬件支持。数据类型包括：[float16, float32]，数据规模测试任意范围数据（极大、极小、不规则（不对齐）等），包括(4, 16, 1024, 1024)，(4,16,1,64)，(3, 5, 197, 175)，硬件支持包括：["mlu370", "mlu290"]

### 3.7 算子防呆检查

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

### 4.2 已经过优化的规模说明

## 5 方案实施

### 5.1 开发测试计划

### 5.2 风险分析

暂无
