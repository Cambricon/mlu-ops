# BANGPy logaddexp2 算子开发设计方案

- #### 文档基本信息

| 算子名称    | Logaddexp2       |
| ----------- | -------------- |
| 编制人/日期 | alaskra/2022-05-31 |
| 审批人/日期 |    |

- #### 修改记录

| 修订人 | 修订日期   | 修订描述 |
| ------ | ---------- | -------- |
| alaskra    | 2022-05-31 | 首次提交 |

- #### 内容描述

本文档为 `Logaddexp2` 算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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

| 算子功能简介   | 逐元素计算表达式: log2(2^x+2^y)                                            |
| ------------ | -----------------------------------------------------------              |
| 需求来源       |  https://pytorch.org/docs/stable/generated/torch.logaddexp2.html        |
| 应用场景       |  https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html  |
| 输入数据类型   | half, float                                                              |
| 输入 Shape    | input0: [length]; input1: [length]                                       |
| 输入 Layout   | input0: ARRAY; input1: ARRAY;                                            |
| 输出数据类型    | half, float                                                             |
| 输出 Shape    | output: [length]                                                                 |
| 输出 Layout   | ARRAY                                                                    |

### 1.2 算子功能和应用场景描述

功能：逐元素计算 output = log2(2^input0 + 2^input1)

例如：input0=[1,2,3], input1=[4,5,6], logaddexp2(input0, input1) = [4.169925, 5.169925, 6.169925]

应用场景： 

参考https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html

为了处理基本数据类型无法表示的极小的数，通常取对数存储。这种方式储存的数进行加法运算，则需要调用logaddexp2算子。

比如 input0 = log2(1e-50), input1 = log2(2.5e-50), 则logaddexp2(input0, input1) == log2(3.5e-50)


### 1.3 算子输入输出参数要求

| 参数        | 语义 | 类型（输入/输出） | 支持类型    | 物理布局 | 规模限制 |
| ----------- | ---- | ----------------- | ----------- | -------- | -------- |
| input0      |  输入的形状为一维的buffer    | 输入              | half, float | ARRAY    | 无       |
| input1      |  输入的形状为一维的buffer    | 输入              | half, float | ARRAY    | 无       |
| length      |  输入的形状为标量           | 输入              | int32       | SCALAR   | 无       |
| output      |  输出的形状为一维的buffer    | 输出              | half, float | ARRAY    | 无       |

### 1.4 算子限制

| 限制类型     | 详细说明                                                                                            |
| ------------ | -------------------------------------------------------------------------- |
| 数据类型限制 | input 和 output 必须为同一类型                                             |
| 数据范围限制 | 由于exp2算子的限制，只支持整数值的输入，比如1.0, 2.0这种浮点数            |

### 1.5 验收标准

#### 1.5.1 精度验收标准

本算子属于 `激活` 类算子，验收标准为 diff1 <= 3e-3 && diff2 <= 3e-3。

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../MLU-OPS性能验收标准.md)。

## 2 算子接口设计

### 2.1 参考接口

- PyTorch

```python
torch.logaddexp2(input, other, *, out=None) → Tensor
```

### 2.2 接口设计

```python
MluOpLogaddexp2(input0, input1, output)
```

## 3 实现方案设计

### 3.1 实现方案

最简单的思路是直接实现log2(2^input0, 2^input1)，伪代码为：

```
input0 = exp2(input0)
input1 = exp2(input1)
output = add(input0, input1)
output = log2(output)
```

但是中间结果exp2(x)非常容易数据溢出，比如输入2^35是超出float的表示范围的，所以使用以下逻辑计算：

假设 input0 > input1:

if input0 - input1 <= 15, output = input1 + log2(1 + 2^(input0-input1))

if input0 - input1 > 15, output = input0

证明过程省略，其中input0-input1>15的情况使用泰勒展开进行近似。

### 3.2 伪代码实现（可选）

```bangpy
# swap in0 and in1 to make sure in0 >= in1
minimum(ex0, in0, in1)
maximum(in0, in0, in1)
in1, ex0 = ex0, in1 # equal to self.tcp.memcpy(in1, ex0), but reduce copy time

# ex0 = in0 - in1
# out = in1 + log2(1+2**(in0-in1))
subtract(ex0, in0, in1)
exp2(out, ex0)
add(out, out, 1)
log(out, out, high_precision=False)
multiply(out, out, 1/math.log(2))
add(out, out, in1)
# if in0-in1 > 15, out = in0
# in1: mask, if greater than 15, set 1
# ex0: ~in1
greater(in1, ex0, 15)
less_equal(ex1, ex0, 15)
multiply(in0, in0, in1)
multiply(out, out, ex1)
add(out, out, in0)
```

### 3.3 拆分(任务拆分，多核拆分)

采用BLOCK类型，64个task均分所有数据。

### 3.4 性能优化设计

1、资源分配

无

2、流水设计

使用`for_range`自动流水，使用`lcs`流水结构

3、Autotuning

无

4、其他优化

无

### 3.5 可维护性设计

1、对每一个函数命名变量命名都有充分的注释

2、避免魔鬼数字，对于确定的数字尽量使用全局变量来替代

3、代码风格遵守PEP8编码规范

### 3.6 测试用例设计

- 算子在网络中用到的规模：
  "shape", [(2**10), (2**18-1), (2**20), (2**26)]

- 数据范围：
  取值在[-180,200], 对应1e-60量级取log2后的结果

### 3.7 算子防呆检查

1、test 方法中过滤掉未支持的 target 。

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

无

### 4.2 已经过优化的规模说明

无

## 5 方案实施

### 5.1 开发测试计划

- 2022.6.1 PR设计文档
- 2022.6.15 PR代码和测试报告
- 2022.6.27 按意见完善设计方案&代码

### 5.2 风险分析

对功能、精度、性能问题的影响分析和后续解决计划。

1. 由于exp2算子的限制，不能支持非整数的输入，但实际上算子的应用场景小数情况居多
2. 只在mlu290上进行过测试，mlu3xx系列支持超越函数，exp2算子应该可以支持非整数输入了。
