# Frac算子开发设计方案

## 1. 需求分析
### 1.1 算子需求分析

算子功能简介	求一个数字的分数或小数部分

需求来源	Pytorch

应用网络	all

输入数据类型	half，float

输入 Shape	input: [batches, hi, wi, channels]

输入 Layout	input: NHWC

输出数据类型	float

输出 Shape	[batches, ho, wo, channels]

输出 Layout	NHWC

### 1.2 算子功能和应用场景描述

功能：计算输入的tensor的每一个元素的分数部分，并返回一个新的tensor，新的tensor中的元素保留原有元素的符号

例如：输入一个tensor[1, 1.5, -2.0300]，其输出的tensor应该为[0.0, 0.5, -0.03]

### 1.3 算子输入输出参数要求
| 参数 | 类型（输入/输出） | 支持类型 | 物理布局 | 规模限制
| ------ | ------ | ------ |------ | ------ |
| input | 输入 | half，float | NHWC | 无 |
| output | 输出 | float | NHWC | 无 |

### 1.4 算子限制
| 限制类型 | 详细说明
| ------ | ------
| 原位限制 | 不支持原位
| stride限制 | 不支持stride机制
| 广播限制 | 不支持广播

### 1.5 验收标准
#### 1.5.1 精度验收标准
按照精度验收标准的要求明确本算子的精度标准。

本算子属于复合类算子，验收标准为diff1 <= 3e-3 && diff2 <= 3e-3 。
#### 1.5.2 性能验收标准
见MLU-OPS性能验收标准。

## 2. 算子接口设计
### 2.1 参考接口
PyTorch接口：
```
torch.frac(input, *, out=None) 
```
### 2.2 接口设计
```
frac(input, output)
```

## 3. 实现方案设计
### 3.1 实现方案
对于输入的tensor的每一个元素，应减去其整数部分，并保留元素原有的符号。
公式为：
$out_i=input_i-\lfloor \vert input_i \vert \rfloor*sgn(input_i)$
### 3.2 伪代码实现


```
import bangpy
from bangpy import tcp

bp = tcp.TCP() 
input = bp.Tensor(shape=(n, h, w, c), name='input_tensor', dtype=bangpy.float16, scope="global, →")
memcpy(input_n, input)

tem1 = bp.Tensor(shape=input_n.shape)
bp.abs(tem1, input_n)
tem2 = bp.Tensor(shape=input_n.shape)
bp.sign(tem2, input_n)
tem3 = bp.Tensor(shape=input_n.shape)
bp.zeros(tem3, tem3)
with tcp_container.for_range(start, end) as i:
  tem3[i] = bp.scalar_floor(tem1[i])
tem4 = bp.Tensor(shape=input_n.shape)
bp.multiply(tem4, tem3, tem2)
output_n = bp.Tensor(shape=input_n.shape)
bp.substract(output_n, input_n, tem4)

memcpy(output, output_n)
```
### 3.3 拆分
把输入的数据按核数量进行拆分。每个核处理自身分到的数据。如果数据相对于总的核数量有余数，那么把这些剩余的数据交给最后一个核来处理，具体实现时通过taskId这一变量实现。

对于每一个核的核内数据处理，应该按照每次捞取最大设置捞取数据量来进行。当总数据量不是每次捞取数据的整数倍时，会有核内数据剩余，此时将所有的剩余数据全部捞取，然后单独计算偏移，在nram中完成计算并放回。
### 3.4 性能优化设计
使用自动流水来进行IO与计算时间的相互掩盖。

使用内存复减少内存开辟和提升内存利用率。

尽量增大每个核单次捞取的最大数据量，以此减少访存的次数。
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

