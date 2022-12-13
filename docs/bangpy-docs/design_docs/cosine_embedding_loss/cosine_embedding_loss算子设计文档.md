# Cosine_embedding_loss 算子开发设计方案

- #### 文档基本信息

| 算子名称     | Cosine_embedding_loss       |
| ----------- | -------------- |
| 编制人/日期   | 胡煜霄/2022-6-1 |
| 审批人/日期   |                |

- #### 修改记录

| 修订人 | 修订日期   | 修订描述 |
| ------ | ---------- | -------- |
| 胡煜霄    | 2022-6-1 | 首次提交 |

- #### 内容描述

本文档为`Cosine_embedding_loss`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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

| 算子功能简介  | 计算两组输入数据之间的余弦损失   |
| --------------------------- | --------------------------- |
| 需求来源      | Pytorch          |
| 应用网络      | 非线性嵌入或半监督学习   |
| 输入数据类型  | half, float  |
| 输入 shape  | input_x1, input_x2: [N, D], input_y: [N]  |
| 输入 layout | input_x1: ARRAY, input_x2: ARRAY, input_y: ARRAY|
| 输出数据类型  | half, float |
| 输出 Shape  | [N]|
| 输出 Layout   | ARRAY |


### 1.2 算子功能和应用场景描述

功能：余弦相似度损失函数，用于判断输入的两个向量是否相似。\
对于包含N个样本的batch数据D(input_x1, input_x2, input_y)。input_x1 和 input_x2 代表输入的两个向量，y表示真实的类别标签，属于{-1, 1}，分别表示相似与不相似。

应用场景：常用于学习非线性嵌入或半监督学习。

### 1.3 算子输入输出参数要求

| 参数        | 语义 | 类型（输入/输出） | 支持类型    | 物理布局 | 规模限制 |
| ----------- | ---- | ----------------- | ----------- | -------- | -------- |
| input_x1 |  输入形状为[N, D]的张量    | 输入 | half, float |   ARRAY   | 无       |
| input_x2 |   输入形状为[N, D]的张量   | 输入 | half, float |  ARRAY       | 无       |
| input_y  |   输入的形状为N的张量  | 输入  | half, float | ARRAY     | {1, -1}       |
| margin   |   作为边界的标量，默认为0  | 输入   | half, float | 标量     | [-1, 1], 建议取值[0, 0.5]       |
| output   |   输出的形状为N的张量  | 输出  | half, float | ARRAY     | 无       |

### 1.4 算子限制

| 限制类型     | 详细说明   |
| ------------ | ------------ |
| 数据类型限制 | input 和 output 数据类型必须相同 |

### 1.5 验收标准

#### 1.5.1 精度验收标准

按照[精度验收标准](../../../MLU-OPS精度验收标准.md#精度验收标准)的要求明确本算子的精度标准。

本算子属于复合类算子，验收标准为 diff1 <= 3e-3 && diff2 <= 3e-3 。

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../../../MLU-OPS性能验收标准.md)。

## 2 算子接口设计

### 2.1 参考接口

- PyTorch

PyTorch接口：
```python
torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
```

### 2.2 接口设计

```python
MluOpCosineEmbeddingLoss(input_x1, input_x2, y, margin, output)
```

## 3 实现方案设计

### 3.1 实现方案
本算子的主要计算行为如下:
$$
output_i=
\begin{cases}
1 - cos(input\_x1_i, input\_x2_i), &if \ input\_y_i = 1 \\
max(0, cos(input\_x1_i, input\_x2_i) - margin), &if \ input\_y_i = -1
\end{cases}
$$
其中cos表示两个向量的余弦相似度:
$$
cos(input\_x1_i, input\_x2_i) = \frac{input\_x1_i * input\_x2_i}{||input\_x1_i||_2 * ||input\_x2_i||_2}
$$
实际计算向量的余弦相似度时采用如下方式:
$$
cos(input\_x1_i, input\_x2_i) = 
\frac 
{sum(multi(input\_x1_i, input\_x2_i))}
{sum(multi(input\_x1_i, input\_x1_i)) * sum(multi(input\_x2_i, input\_x2_i))}
$$

算子设计时主要考虑两个方面问题：
- bangpy内置sum函数基本上只能固定在128bytes的维度上进行求和运算，无法满足算子D维度动态变化的需求，因此考虑使用sumpool+sum共同实现求和规约的功能。
- 随着算子D维度尺寸的变化，mlu平台上的nram有可能装不下一行数据，有可能容纳有限数个数据，也有可能能容纳很多数据，在这种情况下，就需要对D的不同尺寸设计不同的处理逻辑。
由于bangpy平台需要静态分配内存，所以使用一个通用的双层循环构造主循环体：
```python
with tcp.for_range(0, outer_range)as i:
    with tcp.for_range(0, inner_range)as j: 
```
当nram无法容纳一行数据时使用外层循环在N维度上迭代，内层循环在D维度上进行迭代。
当nram能够容纳一行至多行数据时，外层循环不再使用，仅使用内层循环在D维度上进行迭代。

分三种情况对算子的计算逻辑进行设计：
1. nram中无法容纳一行数据,这种情况下无法一次性处理一行数据，所以只能采用多次访存分段计算，在将一整行的数据都计算完毕后得到$(input\_x1_i, input\_x2_i),(input\_x1_i, input\_x1_i), (input\_x2_i, input\_x2_i)$之间分别相乘后进行求和规约的结果，最后将这三个值进行如下运算，通过将input_y与1和-1分别求和的方式避免了算子原有的依据y的值进行条件判断的语义，减少了在循环计算过程中if语句的使用：
```python
with self.tcp.if_scope(tcp.all(lower1_sum != 0, lower2_sum != 0)):
    lower1_sum.assign(lower1_sum * lower2_sum)
    lower1_sum.assign(self.tcp.scalar_sqrt(lower1_sum))
    upper_sum.assign(upper_sum / lower1_sum)
with self.tcp.else_scope():
    upper_sum.assign(0)
lower1_sum.assign(0)
lower2_sum.assign(upper_sum - self.margin)
self.output[row] = ((self.input_y[row] + 1) * (1 - upper_sum) + (1 - self.input_y[row]) * self.tcp.scalar_max(lower1_sum, lower2_sum)) / 2
```
这里使用的求和规约也是结合了sumpool和sum, 先使用sumpool将数据规约成一个128bytes对齐的数组，随后使用sum求和得到最终结果：
```python
a = input_data.reshape(align_size, kernel_size)
tcp.sumpool(temp, a, (kernel_size, ), (kernel_size, ))
tcp.sum(result, temp)
```
2. nram中可以容纳多行数据，但是数据量无法满足使用sumpool的128bytes对齐的模式。这种情况下，对每次load进nram中的数据进行1中相同的运算。
3. nram中可以容纳多行数据，且数据量满足使用sumpool对齐的模式, 则可以直接将输入数据transpose成满足sumpool的(W, C)布局的数据结构，然后直接使用sumpool对每行数据进行规约：
```python
tcp.transpose(temp, input_data)
tcp.sumpool(result, temp, (D, ), (D, ))
```
在此步计算结束后需要结合y和margin对结果进行处理，这里使用向量的方式进行处理，避免逐元素计算和条件判断语句：
```python
self.tcp.multiply(lower1, lower1, lower2)# lower1 * lower2
self.tcp.sqrt(lower1, lower1)# (lower1 * lower2) ** 0.5
self.tcp.maximum(lower2, lower1, 0.004)
self.tcp.divide(upper, upper, lower2)# upper / (lower1 * lower2) ** 0.5
self.tcp.subtract(lower2, upper, self.margin) # upper / (lower1 * lower2) ** 0.5 - margin
# upper <- upper / (lower1 * lower2) ** 0.5
self.tcp.subtract(upper, 1, upper) #(1 - upper)
self.tcp.add(lower1, input_buffer_y, 1)# input_y + 1
self.tcp.multiply(upper, upper, lower1)# (input_y + 1) * (1 - upper)
self.tcp.subtract(input_buffer_y, 1, input_buffer_y) # 1 - input_y
self.tcp.maximum(lower1, lower2, 0) # max(lower1 * lower2, 0)

self.tcp.multiply(lower1, lower1, input_buffer_y)# (1 - input_y) * max(lower1 * lower2, 0)
self.tcp.add(upper, lower1, upper)
self.tcp.multiply(upper, upper, 0.5)
```
### 3.2 伪代码实现

### 3.3 拆分(任务拆分，多核拆分)
以输入数据的行为单位拆分数据，每次处理一行或者多行数据。

当的行方向尺寸在一定范围内使得nram一次性执行处理不足128bytes / data_type.bytes个数据时，就采用load进来后通过循环进行分别计算的方式进行处理。

当行方向尺寸使得nram能够处理128bytes / data_type.bytes个数据时，则每次load进来128bytes / data_type.bytes的整数倍个数据然后通过向量的方式进行计算。

当行方向尺寸过大，无法一次性装入nram时则需要考虑在水平方向上进行拆分，即使用双层循环。对一行数据进行分批计算。

### 3.4 性能优化设计
使用自动流水来进行IO与计算时间的相互掩盖。
  
对y进行变换进行判断行为：

 由于y的输入范围为1和-1，因此将y+1之后与目标向量相乘再除以2就可以将y=-1时对应位置上的元素抹除，反之将1-y之后与目标向量相乘再除以2就可将y=1对应位置上的元素抹除


### 3.5 可维护性设计

1、对每一个函数命名变量命名都有充分的注释。

2、对算子进行模块化设计与拆分，确保模块具有复用性。

### 3.6 测试用例设计

1、 对于不同的数据规模，包括$2^{20}, 2^{30}, 2^{30} * 2, 2^{30} * 4, 2^{30} * 8$进行测试。

2、 对于不同的数据行尺寸D，包括$2^{5}, 2^{6}, 2^{7}, ..., 2^{19}$进行测试。

2、 对于非对齐的数据行尺寸D，包括$2^{5} + 1, 2^{5} - 1, 2^{11} + 1, 2^{11} - 1$进行测试。


## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

| 提交日期  | 问题规模 | 问题描述 | 是否已修复 |
| --------- | -------- | -------- | ---------- |
|     2022.6.1      |          |   单向IO算子IO性能受限       |        未修复    |
|     2022.6.1      |     D方向尺寸小于32     |     无法动态管理内存，所以只能假设D方向上数据尺尺寸大于32，在分配内存时将需要尺寸为(N)方向的数据都声明为(N,D)类型数据尺寸的1/32      |        已修复    |
|     2022.6.1      |     nram能够装下128bytes/datatype.bytes时     |      向量除法包含0时无法得到正确结果     |        已修复    |
|     2022.6.1      |          |      无法处理字符串类型输入，暂未实现最终结果的规约操作     |        未修复    |
| 2022.6.24 | 非对齐情形| transpose无法处理非对齐数据| 已修复 |

### 4.2 已经过优化的规模说明

| 提交日期  | 修复规模 | 修复问题 |
| --------- | -------- | -------- |
|           |          |          |

## 5 方案实施

### 5.1 开发测试计划

- 2022.6.1 提交设计文档
- 2022.6.15 提交代码pr
- 2022.6.27 最终代码提交

### 5.2 风险分析

暂无。
