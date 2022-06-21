# BANGPy Hard_sigmoid 算子开发设计方案

- #### 文档基本信息

| 算子名称     |    Hard_sigmoid      |
| ----------- | --------------        |
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

| 算子功能简介               | 计算一个张量经该激活函数激活过后的张量  |
| --------------------------| ------------------------------------- |
| 需求来源                   | Pytorch                               |
| 应用网络                   |                                       |
| 输入数据类型               | half, float                           |
| 输入 Shape                 | 任意shape                             |
| 输入 Layout                |    \                                  |
| 输出数据类型               | half, float                            |
| 输出 Shape                 | 与输入张量的shape相同                   |
| 输出 Layout                |    \                                   |


### 1.2 算子功能和应用场景描述

hard_sigmoid激活函数定义:  
$hard_sigmoid(x)=\left\{\begin{matrix}0,  x<=3\\1/6*x+1/2,-3<x<3\\1,  x>=3\end{matrix}\right.$

算子功能：计算一个张量经该激活函数激活过后的张量。  
例如：  
     input= [[[[[[[[-3.5,2.4,4.0],  
                   [-2.4,0.0,1.8]]]]]]]]  
     output=[[[[[[[[0.0,0.9,1]  
                   [0.1,0.5,0.8]]]]]]]]

应用场景：\

### 1.3 算子输入输出参数要求

| 参数    | 语义                          | 类型（输入/输出）  | 支持类型     | 物理布局         |      规模限制                               |
| ------  | ----------------------------- | ------------------| ----------- | -----------------| ------------------------------------------ |
| input   | 输入任意shape的张量            |       输入         | half, float |   \             | 每次输入张量所占空间大小小于MLU的GDRAM的一半  |
| output  | 输出同输入张量shape相同的张量  |        输出         | half, float |   \             | 每次输出张量所占空间与输入张量的相同          |


### 1.4 算子限制

| 限制类型      | 详细说明                                        |
| ------------  | ---------------------------------------------- |
| 数据类型限制   | input 和 output 必须同时为同一数据类型          |
| 布局限制       |               \                               |
| 规模限制       | 输入/输出张量所占空间大小均小于MLU的GDRAM的一半  |

### 1.5 验收标准

#### 1.5.1 精度验收标准

按照[精度验收标准](../../../MLU-OPS精度验收标准.md)的要求明确本算子的精度标准。

本算子属于复合类算子：diff1 <= 3e-3 && diff2 <= 3e-3

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../../../MLU-OPS性能验收标准.md)。

## 2 算子接口设计

### 2.1 参考接口

- Pytorch

```python
nn.hard_sigmoid(input)
```

### 2.2 接口设计

```python
MluOpHard_sigmoid(input,output)
```

## 3 实现方案设计

### 3.1 实现方案

主要步骤：  
（1）在host端将flatten后的张量数据传到device端，均匀地分给各个IPU计算  
（2）各个IPU计算之后拷回对应的位置,计算时使用双缓冲技术进行优化  
（3）将数据从device端传回host端并在host端进行reshape

### 3.2 伪代码实现

```python

# host
# set I/O data
data_in_dev = bangpy.Array(data_in.flatten().astype(dtype.as_numpy_dtype), dev)
data_out_dev = bangpy.Array(np.zeros(data_out.flatten().shape, dtype.as_numpy_dtype), dev)


#device
# distribute
data_each_task = data_total // self.task_num
data_rem = data_total % self.task_num
data_each_time = self.nram_size_buffer // self.dtype_sz
loop = data_each_task // data_each_time
data_rem_n = data_each_task  % data_each_time

# calculate
memcpy:GDRAM-->NRAM
self.tcp.assign(buffer_temp_n,1/6)
self.tcp.multiply(buffer_io_n,buffer_io_n,buffer_temp_n) # x * 1/6
self.tcp.assign(buffer_temp_n,1/2)
self.tcp.add(buffer_io_n,buffer_io_n,buffer_temp_n)      # x * 1/6 + 1/2
self.tcp.assign(buffer_temp_n,1)
self.tcp.minimum(buffer_io_n,buffer_io_n,buffer_temp_n)  # min(x * 1/6 + 1/2 , 1)
self.tcp.zeros(buffer_temp_n)
self.tcp.maximum(buffer_io_n,buffer_io_n,buffer_temp_n)  # max(x * 1/6 + 1/2 , 0)
memcpy:NRAM-->GDRAM


# host
# data processing
data_out_dev2host = data_out_dev.numpy().reshape(shape)

```

### 3.3 拆分(任务拆分，多核拆分)

首先需要将张量压成flatten成一维传入，然后相关参数如下：  
data_total：张量的数据总个数  
task_num:任务个数  
data_each_task: 每个任务需要计算的数据个数（data_all // task_num）  
data_rem: 平均分给所有IPU后的余数(data_all % task_num)  
data_each_time: 每次NRAM计算的数据个数  
loop:每个task需要拷入NRAM进行计算的次数(data_each_task // data_each_time)  
data_rem_n: 不足一次计算(data_each_task % data_each_time)

### 3.4 性能优化设计

1.进行了向量优化，计算均是向量计算。  
2.使用for_range中的stage参数来打开双缓冲，以实现访存指令与计算指令的并⾏，即⽤访存时间来隐藏计算时间。  
NRAM双缓冲对应的流水线如下：
|       |        |        |        |          |        |        |        |       |        |
|-------| -------|------- |------- |--------- |------- |------- |------- |-------|------- |
|G-->N1 | C(N1)  | N1-->G | G-->N1 |  C(N1)   |        | N1-->G | G-->N1 | C(N1) | ...    |
|       | G-->N2 | C(N2)  |        |  N2-->G  | G-->N2 | C(N2)  |        |       | ...    |

由以上的图可知，计算时间被访存时间隐藏，并且DRAM的带宽应该已经充分利用。  
使用cnperf工具检测时，也基本符合上述结论。  
  
那么还有优化的空间吗？比如：引入SRAM/WRAM作为缓存？利用IO-DMA和Shared-DMA的并行？等等  
经过尝试，其它结果与分析如下：  
    （1） 该算子在NRAM中计算时，每次拷入NRAM的数据量较大，SRAM或WRAM的空间相对于每次拷入NRAM的数据量来说较小，单纯的引入它们作为缓存并不能起到缓存的作用，并且重复的数据拷贝反而使总的性能降低。   
    （2） 引入SRAM并且利用IO-DMA和Shared-DMA的并行时，该算子**理想的**流水线如下（其中：S//4=N1*3=N2\*3，即每计算3次后SRAM被填满，需拷出到GDRAM）：
|        |        |         |        |        |        |           |        |        |         |         |           |        |     |
|--------| -------|-------  |------- |--------|------- |---------- |------- |------- |-------- |-------- | --------- | ------ | --- |
| G-->N1 | C(N1)  |  N1-->S | G-->N1 | C(N1)  | N1-->S | **S-->G** | G-->N1 | C(N1)  | N1-->S  | G-->N1  |  C(N1)    | N1-->S | ... |
|        | G-->N2 |  C(N2)  | N2-->S | G-->N2 |  C(N2) |           | N2-->S | G-->N2 | C(N2)   | N2-->S  | **S-->G** | G-->N2 | ... |
  
    
    (a)NRAM双缓冲模型是先将数据拷入，然后拷出（计算时间被吃掉了），但是每次拷的数据较少，导致发起IO时产生延迟的次数较多  
    (b)IO-DMA和Shared-DMA并行的模型与NRAM双缓冲模型相比，流水稳定时，拷出时一次拷出的数据量较大，减少了延迟次数（减少了延迟时间），但每一轮S大小的数据多了一次计算时间  
    (c)由于for_range中的stage参数并不能实现上述逻辑，所以自己尝试手动实现了一下，在不考虑结果正确的情况下，可能由于其他的通信等开销，导致第二个模型的性能仍然略低于第一个模型，二者虽然性能接近，但第二个逻辑要复杂一些，并且实现较为困难，因此不再考虑

### 3.5 可维护性设计

详细注释。

### 3.6 测试用例设计

- 算子在测试时使用的规模：  

(2**23 + 1,), (1, 2**24 + 1), (1, 1, 2**25 + 1), (1, 1, 1, 2**26 + 1),  
(1, 1, 1, 1, 2**27 + 1), (1, 1, 1, 1, 1, 2**28 + 1),  
(1, 1, 1, 1, 1, 1, 2**29 + 1), (1, 1, 1, 1, 1, 1, 1, 2**30 + 1), 

（1）覆盖了逻辑分支  
（2）只测试了1~8维的张量（理论上支持任意维度）

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

1.目前只在MLU290上测试过。