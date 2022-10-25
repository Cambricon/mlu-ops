# BANGPy Expm1 算子开发设计方案

- #### 文档基本信息

| 算子名称    | Expm1       |
| ----------- | -------------- |
| 编制人/日期 | 郑磊磊/2022-3-1 |
| 审批人/日期 |    |

- #### 修改记录

| 修订人 | 修订日期   | 修订描述 |
| ------ | ---------- | -------- |
| 郑磊磊    | 2022-3-1 | 首次提交 |
| 潘健行    | 2022-8-20 | 第二次提交

- #### 内容描述

本文档为 `Expm1` 算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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

example:

| 算子功能简介   | 计算buffer中数据以e为底的指数并减去1           |
| ------------ | ---------------------------------------------|
| 需求来源       | PyTorch                                     |
| 应用网络       | resnet50等                                  |
| 输入数据类型   | float32, half                                 |
| 输入 Shape    | input: [dim0, dim1, dim2, dim3]          |
| 输入 Layout   | ARRAY                               |
| 输出数据类型   | float32, half                                |
| 输出 Shape    | output：[dim0, dim1, dim2, dim3]                  |
| 输出 Layout   | ARRAY                                    |

### 1.2 算子功能和应用场景描述

功能：计算输入Tensor以e为底的指数，并减去1的结果。

例如：buffer_input = [[1., 2.], [3., 4.]]
     output = [[1.7183, 6.3891], [19.0855, 53.5982]]

### 1.3 算子输入输出参数要求
| 参数 | 类型（输入/输出） | 支持类型 | 物理布局 | 规模限制
| ------ | ------ | ------ |------ | ------ |
| input | 输入 | float, half | ARRAY | 无 |
| output | 输出 | float, half | ARRAY | 无 |

### 1.4 算子限制
| 限制类型 | 详细说明
| ------ | ------
| 原位限制 | 不支持原位
| stride限制 | 不支持stride机制
| 广播限制 | 不支持广播
|数据类型限制 | 无

### 1.5 验收标准
#### 1.5.1 精度验收标准
按照精度验收标准的要求明确本算子的精度标准。具体可以参见MLU-OPS精度验收标准.md。

本算子属于复合类算子，验收标准为diff1 <= 3e-3 && diff2 <= 3e-3 。
#### 1.5.2 性能验收标准
具体可以参见MLU-OPS精度验收标准.md。

## 2 算子接口设计

### 2.1 参考接口

- PyTorch

```python
torch.special.expm1(input,)
```

### 2.2 接口设计

```python
MluOp_Expm1(input, output, dim0, dim1, dim2, dim3)
```

## 3 实现方案设计

### 3.1 实现方案

在nram中开辟两个大小为 `单次最大空间大小` 的buffer分别用来存放一个输入buffer的数据及输出的结果，从gdram循环拷贝 `单次最大空间大小` 大小的数据至nram相应的buffer中。其中的单次最大空间大小计算方式由下面算式计算，两种数据剩余也标出。
```python
data_calculated_each_task = self.length // self.task_num #计算每个核的数据计算量
data_remain = self.length % self.task_num#总数据剩余
loop_num = data_calculated_each_task * self.dtype_sz // self.single_buffer_size
data_calculated_each_time = self.single_buffer_size // self.dtype_sz#计算每个核单次最大数据计算量
each_task_remain = data_calculated_each_task % data_calculated_each_time#单核计算数据剩余
```
然后进行求以e为底的指数运算，再将计算结果减去1（广播），循环次数为输入buffer长度在进行多核拆分后除以nram buffer长度的值。

### 3.2 伪代码实现

```python
# buffer_in和buffer_out分别位于GDRAM
buffer_in = tcp.match_buffer(buffer_in, [dim0, dim1, dim2, dim3],  
    dtype=self.dtype)
buffer_out = tcp.match_buffer(buffer_out, [dim0, dim1, dim2, dim3], 
    dtype=self.dtype)
# 此处是在NRAM空间上
buffer_in_n = tcp.alloc_buffer(
    [data_calculated_each_time,], dtype=self.dtype, scope="nram"
)
buffer_out_n = tcp.alloc_buffer(
    [data_calculated_each_time,], dtype=self.dtype, scope="nram"
)
buffer_one = tcp.alloc_buffer(
    shape=[data_calculated_each_time,],
    dtype=self.dtype,
    scope="nram",
)
buffer_tem = tcp.alloc_buffer(
    shape=[data_calculated_each_time,],
    dtype=self.dtype,
    scope="nram",
)
tcp.exp(buffer_tem, buffer_in_n)
tcp.subtract(buffer_out_n, buffer_tem, buffer_one)
```
### 3.3 拆分
把输入的数据按核数量进行拆分。每个核处理自身分到的数据。如果数据相对于总的核数量有余数，那么把这些剩余的数据交给最后一个核来处理，具体实现时通过taskId这一变量实现。

对于每一个核的核内数据处理，应该按照每次加载最大设置加载数据量来进行。当总数据量不是每次加载数据的整数倍时，会有核内数据剩余，此时将所有的剩余数据全部加载，然后单独计算偏移，在nram中完成计算并放回。
### 3.4 性能优化设计
使用自动流水来进行IO与计算时间的相互掩盖。

使用内存复用减少内存开辟和提升内存利用率。

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