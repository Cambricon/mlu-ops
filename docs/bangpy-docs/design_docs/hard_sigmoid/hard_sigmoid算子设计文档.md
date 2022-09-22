# BANGPy Hard_sigmoid 算子开发设计方案

- #### 文档基本信息

| 算子名称     | Hard_sigmoid         |
| ----------- | -------------------- |
| 编制人/日期  | pingmu123/2022-06-01 |
| 审批人/日期  |                      |

- #### 修改记录

|    修订人       | 修订日期    | 修订描述 |
| --------------- | ---------- | ------- |
|   pingmu123     | 2022-06-01 | 首次提交 |

- #### 内容描述

本文档为 `Hard_sigmoid` 算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

## 1 需求分析

### 1.1 算子需求分析

| 算子功能简介               | 按元素应用Hardsigmoid激活函数          |
| --------------------------| ------------------------------------- |
| 需求来源                   | Pytorch                               |
| 应用网络                   |                                       |
| 输入数据类型               | half, float                           |
| 输入 Shape                 | 任意shape                             |
| 输入 Layout                |  ARRAY                                |
| 输出数据类型               | half, float                            |
| 输出 Shape                 | 与输入张量的shape相同                  |
| 输出 Layout                |  ARRAY                                |


### 1.2 算子功能和应用场景描述

Hardsigmoid函数定义:  

$$  
\text{Hardsigmoid}(x)=\begin{cases}
0 & x \le -3, \\
1 & x \ge +3, \\
x/6+1/2 & \text{otherwise}.\end{cases}
$$

算子功能：对一个输入张量按元素应用Hardsigmoid激活函数后，得到一个激活后的张量。  
例如：
```
input = [[[[[[[[-3.5, 2.4, 4.0], [-2.4, 0.0, 1.8]]]]]]]]  
output = [[[[[[[[0.0, 0.9, 1.0], [0.1, 0.5, 0.8]]]]]]]]
```
应用场景：\

### 1.3 算子输入输出参数要求

| 参数    | 语义                          | 类型（输入/输出）  | 支持类型     | 物理布局         | 规模限制          |
| ------  | ----------------------------- | ------------------| ----------- | -----------------| ---------------- |
| input   | 输入任意shape的张量            |       输入        | half, float |  ARRAY           |  \               |
| output  | 输出同输入张量shape相同的张量  |        输出        | half, float |  ARRAY           |  \               |


### 1.4 算子限制

| 限制类型      | 详细说明               |
| ------------  | --------------------- |
| 数据类型限制   | 仅支持float、half     |
| 布局限制       | ARRAY                |
| 规模限制       |  \                   |

### 1.5 验收标准

#### 1.5.1 精度验收标准

按照[精度验收标准](../../../MLU-OPS精度验收标准.md)的要求明确本算子的精度标准。

本算子属于复合类算子：diff1 <= 3e-3 && diff2 <= 3e-3。

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../../../MLU-OPS性能验收标准.md)。

## 2 算子接口设计

### 2.1 参考接口

- Pytorch

```python
# https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
def hardsigmoid(input: Tensor,
                inplace: bool = False) -> Tensor:
    r"""Applies the element-wise function
    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}
    Args:
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    See :class:`~torch.nn.Hardsigmoid` for more details.
    """
```
### 2.2 接口设计

```python
# MluOpHardSigmoid(input, output, data_total)
tcp.BuildBANG(
    inputs=[buffer_in],
    outputs=[buffer_out],
    data_total, # data_total: len(data_in.flatten())
    kernel_name=KERNEL_NAME,
) 
```

## 3 实现方案设计

### 3.1 实现方案

主要步骤：  
（1）在host端将flatten后的张量数据传到device端。  
（2）均匀地分给各个IPU。  
（3）各个IPU计算时使用双缓冲技术进行优化，计算完毕后拷回GDRAM对应的位置。  
（4）将数据从device端传回host端并在host端进行reshape操作。

### 3.2 伪代码实现

```python

# host
# set I/O data
data_in_dev = bangpy.Array(data_in.flatten().astype(dtype.as_numpy_dtype), dev)
data_out_dev = bangpy.Array(np.zeros(data_out.flatten().shape, dtype.as_numpy_dtype), dev)


# device
# calculate split strategy
data_each_task = data_total // self.task_num
data_rem = data_total % self.task_num
data_each_time = self.nram_size_buffer // self.dtype_sz # float16: self.dtype_sz = 2, float32: self.dtype_sz = 4
loop_num = data_each_task // data_each_time
data_rem_n = data_each_task % data_each_time
if data_rem_n > 0 :
    loop_num = loop_num + 1

# calculate
memcpy:GDRAM-->NRAM
tcp.multiply(buffer_io_n, buffer_io_n, 1/6)          # x * 1/6
tcp.add(buffer_io_n, buffer_io_n, 1/2)               # x * 1/6 + 1/2
tcp.assign(buffer_temp_n, 1)
tcp.minimum(buffer_io_n, buffer_io_n, buffer_temp_n) # min(x * 1/6 + 1/2, 1)
tcp.zeros(buffer_temp_n)
tcp.maximum(buffer_io_n, buffer_io_n, buffer_temp_n) # max(x * 1/6 + 1/2, 0)
memcpy:NRAM-->GDRAM


# host
# data processing
data_out_dev2host = data_out_dev.numpy().reshape(shape)

```

### 3.3 拆分(任务拆分，多核拆分)

首先需要将张量flatten成一维传入，然后相关参数如下：  
data_total：张量的数据总个数  
self.task_num：MLU任务个数  
data_each_task：每个任务需要计算的数据个数(data_total // self.task_num)  
data_rem：平均分给所有IPU后的余数(data_total % self.task_num)  
data_each_time：每次NRAM计算的数据个数  
loop_num：每个data_each_task需要拷入NRAM进行计算的次数(data_each_task // data_each_time)  
data_rem_n：不足一次计算(data_each_task % data_each_time)  
loop_num = loop_num + 1：当data_rem_n > 0 时

### 3.4 性能优化设计

1.尽可能地给每个IPU分配了均匀的工作量，以达到负载均衡，并且数据也满足了对齐的要求，这样的话充分利用了MLU里面的资源。  
2.进行了向量优化，其中：访存均为向量访存，且计算也均是向量计算。  
3.使用pipeline参数来打开双缓冲，以实现访存指令与计算指令的并⾏。  
NRAM双缓冲对应的流水线如下(假设最后一块是G-->N2)：
|  time   |  t0_1   |  t0_2  |       t1        |       t2        |       t3        |       ...       |     t(n-1)      |  t(n)_1  |  t(n)_2  |
|:-------:|:-------:|:------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:--------:|:--------:|
|  NRAM1  | G-->N1  | C(N1)  | N1-->G   G-->N1 |      C(N1)      | N1-->G   G-->N1 |       ...       |      C(N1)      |  N1-->G  |          |
|  NRAM2  |         | G-->N2 |      C(N2)      | N2-->G   G-->N2 |      C(N2)      |       ...       | N2-->G   G-->N2 |  C(N2)   |  N2-->G  |

由以上的图可知，当流水稳定时，数据IO（一段拷出和一段拷进）和计算是并行的。此时，若IO时间大于等于计算时间，则计算时间便可被访存时间隐藏，这样GDRAM的带宽应该就已经被充分利用。  
并且，经测试，有以下结果：当`self.nram_size_each_buffer = ((self.nram_size - 4 * 1024) // 3) // 128 * 128`时，IO时间是大于计算时间的。  

另外，该算子每个cluster的4个ipu没有复用IO的部分，使用5级别流水（SRAM）无收益。

### 3.5 可维护性设计

详细注释。

### 3.6 测试用例设计

- 算子在测试时使用的规模：  
```
# dtype = half
case_0: input.shape = (2**23 + 1,), output.shape = (2**23 + 1,), dtype = half
case_1: input.shape = (1, 2**24 + 1), output.shape = (1, 2**24 + 1), dtype = half
case_2: input.shape = (1, 1, 2**25 + 1), output.shape = (1, 1, 2**25 + 1), dtype = half
case_3: input.shape = (1, 1, 1, 2**26 + 1), output.shape = (1, 1, 1, 2**26 + 1), dtype = half
case_4: input.shape = (1, 1, 1, 1, 2**27 + 1), output.shape = (1, 1, 1, 1, 2**27 + 1), dtype = half
case_5: input.shape = (1, 1, 1, 1, 1, 2**28 + 1), output.shape = (1, 1, 1, 1, 1, 2**28 + 1), dtype = half
case_6: input.shape = (1, 1, 1, 1, 1, 1, 2**29 + 1), output.shape = (1, 1, 1, 1, 1, 1, 2**29 + 1), dtype = half
case_7: input.shape = (1, 1, 1, 1, 1, 1, 1, 2**30 + 1), output.shape = (1, 1, 1, 1, 1, 1, 1, 2**30 + 1), dtype = half
# special test cases
case_8: input.shape = (66777500,), output.shape = (66777500,), dtype = half
case_9: input.shape = (67077500,), output.shape = (67077500,), dtype = half

# dtype = float
case_10: input.shape = (2**23 + 1,), output.shape = (2**23 + 1,), dtype = float
case_11: input.shape = (1, 2**24 + 1), output.shape = (1, 2**24 + 1), dtype = float
case_12: input.shape = (1, 1, 2**25 + 1), output.shape = (1, 1, 2**25 + 1), dtype = float
case_13: input.shape = (1, 1, 1, 2**26 + 1), output.shape = (1, 1, 1, 2**26 + 1), dtype = float
case_14: input.shape = (1, 1, 1, 1, 2**27 + 1), output.shape = (1, 1, 1, 1, 2**27 + 1), dtype = float
case_15: input.shape = (1, 1, 1, 1, 1, 2**28 + 1), output.shape = (1, 1, 1, 1, 1, 2**28 + 1), dtype = float
case_16: input.shape = (1, 1, 1, 1, 1, 1, 2**29 + 1), output.shape = (1, 1, 1, 1, 1, 1, 2**29 + 1), dtype = float
case_17: input.shape = (1, 1, 1, 1, 1, 1, 1, 2**30 + 1), output.shape = (1, 1, 1, 1, 1, 1, 1, 2**30 + 1), dtype = float
# special test cases
case_18: input.shape = (66777500,), output.shape = (66777500,), dtype = float
case_19: input.shape = (67077500,), output.shape = (67077500,), dtype = float
```
注：  
（1）测试用例覆盖了逻辑分支。  
（2）随着数据量的增加，对于dtype = half：case_3的IO效率突然降低；对于dtype = float：case_12的IO效率突然降低。  
（3）case_8与case_9规模相近，但后者的IO效率比前者低了很多。  
（4）只测试了1~8维的张量（理论上支持任意维度）。

### 3.7 算子防呆检查

暂无。

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

xx-xx-xx~2022-03-01 准备工作（学习白皮书、熟悉开发环境等）  
2022-03-01 算子调研与设计文档  
2022-03-14 开始编写代码  
2022-03-28 逻辑完善、性能优化与测试  
2022-05-30 编写与完善相关文档&代码  
2022-06-30 算子入库

### 5.2 风险分析

1.理论上随着数据规模的增大，流水线也越来越有效利用，对应的IO效率也应该越来越高，可一些数据规模的结果不符合。  
2.理论上规模相近的张量IO效率类似，可实际上却差别明显。  
注：二者可能有重叠，它们应该都是由于受到硬件的IO限制的影响，具体见3.6处。
