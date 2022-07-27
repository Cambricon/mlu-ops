# Cosine_similarity算子开发设计方案

- #### 文档基本信息

| 算子名称    | Cosine_similarity       |
| ----------- | -------------- |
| 编制人/日期 | 潘健行/2022-6-10 |
| 审批人/日期 |    |

- #### 修改记录

| 修订人 | 修订日期   | 修订描述 |
| ------ | ---------- | -------- |
| 潘健行    | 2022-6-20 | 首次提交 |
| 潘健行    | 2022-7-7  | 第二次提交

- #### 内容描述

本文档为 `Cosine_similarity` 算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

- #### 算子需求 checklist

* 算子接口描述
* 功能描述
* 框架版本 + 对应源码路径
* 需求对应网络
* 网络中用到的规模
* 是否需要支持原位
* 是否需要支持 stride 机制
* 框架单元测试阈值指标（可选）
## 1. 需求分析
### 1.1 算子需求分析

example:

| 算子功能简介   | 计算两个Buffer在指定维度上的余弦相似度           |
| ------------ | ---------------------------------------------|
| 需求来源       | PyTorch                                     |
| 应用网络       | resnet50等                                  |
| 输入数据类型   | float                                 |
| 输入 Shape    | input0: [batches, hi, wi, channels]    
|               |   input1: [batches, hi, wi, channels]   |
| 输入 Layout   | input0: NHWC   | 
|               | input1: NHWC          |
| 输出数据类型    |float                                 |
| 输出 Shape    | dim = 0, [ho, wo, channels]                  |
|               | dim = 1, [batches, wo, channels]                    
|               | dim = 2, [batches, ho, channels]                   
|               | dim = 3, [batches, ho, wo]                   
| 输出 Layout   | array                                         |

### 1.2 算子功能和应用场景描述

功能：计算两个输入的tenso在指定维度上的余弦相似度，得到的结果的tensor应该比输入的tensor少一个维度。

例如：输入2个tensor[1, 1]和[-1, 1]，其输出的tensor应该为[0]。

### 1.3 算子输入输出参数要求
| 参数 | 类型（输入/输出） | 支持类型 | 物理布局 | 规模限制
| ------ | ------ | ------ |------ | ------ |
| input0 | 输入 | half，float | NHWC | 无 |
| input1 | 输入 | half，float | NHWC | 无 |
| output | 输出 | float | ARRAY | 无 |

### 1.4 算子限制
| 限制类型 | 详细说明
| ------ | ------
| 原位限制 | 不支持原位
| stride限制 | 不支持stride机制
| 布局限制 |  input0.dim_n == input1.dim_n ; n = 0, 1, 2, 3
|          | output.dim_n == input0.dim_n ; n为除了指定归约维度外的维度
|数据类型限制 | 仅支持float

### 1.5 验收标准
#### 1.5.1 精度验收标准
按照精度验收标准的要求明确本算子的精度标准。具体可以参见MLU-OPS精度验收标准.md。

本算子属于复合类算子，验收标准为diff1 <= 3e-3 && diff2 <= 3e-3 。
#### 1.5.2 性能验收标准
具体可以参见MLU-OPS精度验收标准.md。

## 2. 算子接口设计
### 2.1 参考接口
PyTorch接口：
```
torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-08)
```
### 2.2 接口设计
```
MluOpCosineSimilarity(input0, input1, dim, output)
```

## 3. 实现方案设计
### 3.1 实现方案
对于输入的2个tensor，首先计算其在指定维度上的内积，得到buffer_mul，然后计算两个tensor在指定维度上的二阶范数，分别得到buffer_out_0和buffer_out_1。然后用buffer_mul除以两个二阶范数的成绩乘积，得到最终结果。

公式为：
$similarity = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}$
### 3.2 伪代码实现


```
import bangpy
from bangpy import tcp

self.bp.multiply(buffer_mul, buffer_in0_n, buffer_in1_n)
self.bp.square(buffer_in0_n, buffer_in0_n)
self.bp.square(buffer_in1_n, buffer_in1_n)
mul_reshape = buffer_mul.reshape((dim_h, dim_m, dim_l))
in_reshape0 = buffer_in0_n.reshape((dim_h, dim_m, dim_l))
in_reshape1 = buffer_in1_n.reshape((dim_h, dim_m, dim_l))
self.bp.sumpool(buffer_out_n, mul_reshape, dim_h, dim_m, 1, 0)
self.bp.sumpool(buffer_out0_n, in_reshape0, dim_h, dim_m, 1, 0)
self.bp.sumpool(buffer_out1_n, in_reshape1, dim_h, dim_m, 1, 0)
self.bp.sqrt(buffer_out0_n, buffer_out0_n)
self.bp.sqrt(buffer_out1_n, buffer_out1_n)
self.bp.multiply(buffer_out0_n, buffer_out0_n, buffer_out1_n)
self.bp.divide(buffer_out_n, buffer_out_n, buffer_out0_n)
```
### 3.3 拆分
将整体数据量（四维数据块）重新分成三部分，dim_h：指定参数维度的高维；dim_m：指定参数所在的维度；dim_l：指定参数维度的低维。

把输入的数据按task数量进行拆分。每个task处理自身分到的数据。如果数据相对于总的task数量有余数，那么把这些剩余的数据交给最后一个task来处理，具体实现时通过taskId这一变量实现。

对于每一个task内数据处理，应该按照每次加载dim_m * dim_l的最大整数倍。如果dim_m * dim_l 的规模很大，一次无法加载到偏上计算，那么就需要在每个task内按照nram空间的大小及划分，进行切分处理，计算repeat和rem。当总数据量不是每次加载数据的整数倍时，会有task内数据剩余，此时将所有的剩余数据全部加载，然后单独计算偏移，在nram中完成计算并放回。
### 3.4 性能优化设计
使用自动流水来进行IO与计算时间的相互掩盖。

使用内存复减少内存开辟和提升内存利用率。

尽量增大每个核单次加载的最大数据量，以此减少访存的次数。
### 3.5 可维护性设计
1、对每一个函数命名变量命名都有充分的注释。

2、对算子进行模块化设计与拆分，确保模块具有复用性。
### 3.6 测试用例设计
根据需要进行补充。详见算子测试文件。
### 3.7 算子防呆检查
除host端自动生成的部分参数防呆检查外，暂不需要进行其他的防呆检查。
## 4. 算子性能优化记录
### 4.1 当前存在问题的规模说明
无
### 4.2 已经过优化的规格说明
无

