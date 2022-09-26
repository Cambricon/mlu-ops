# BANGPy Cross 算子开发设计方案

- #### 文档基本信息

| 算子名称    | Cross             |
| ----------- | ----------------- |
| 编制人/日期 | 严嘉鹏/2022-06-01 |
| 审批人/日期 |                   |
| 审批人/日期 |                   |
| 审批人/日期 |                   |

- #### 修改记录

| 修订人 | 修订日期   | 修订描述    |
| ------ | ---------- | ----------- |
| 严嘉鹏 | 2022-06-01 | 首次提交    |
| 严嘉鹏 | 2022-09-21 | BANGPy2更新 |

- #### 内容描述

本文档为 `cross` 算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

## 1 需求分析

### 1.1 算子需求分析

| 算子功能简介                                                 | 实现Python的Pytorch包中的函数torch.cross的BANGPy版本（在给定的维度上进行向量叉乘计算） |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 需求来源                                                     | torch.cross(https://pytorch.org/docs/stable/generated/torch.cross.html) |
| 应用网络                                                     | 需要使用三维向量叉乘计算的网络                               |
| 输入数据类型                                                 | half, float                                                  |
| 输入 Shape                                                   | input1:[dim0,dim1,dim2,...,dim7]; <br/>input2: [dim0,dim1,dim2,...,dim7] |
| 输入 Layout                                                  | input1: 八维ARRAY; input2: 八维ARRAY                         |
| 输出数据类型                                                 | half, float                                                  |
| 输出 Shape                                                   | [dim0,dim1,dim2,...,dim7]                                    |
| 输出 Layout                                                  | 八维ARRAY                                                    |
| 是否含有 dim/axis 等类似语义的参数且该参数支持负数/其他特殊处理 | 有 dim参数 <br>dim指定叉乘计算发生的维度，所以dim的取值范围是[-8,7]且指定的维度的值必须是3<br>(例如shape=(1,2,3,4,5,4,3,2),则dim只可能取2或-6或6或-2) |
| 是否含有 labels/index 等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 否                                                           |
| 是否需要支持原位                                             | 否                                                           |
| 是否需要支持 stride 机制                                     | 否                                                           |
| 是否需要支持广播                                             | 否                                                           |
| 0 元素检查是否直接返回                                       | 否                                                           |
| 其他特殊需求(在线量化，融合，转数提前等，可选)               | BANGPy版本2.1.0以上                                          |

### 1.2 算子功能和应用场景描述

算子功能：对于给定的相同维度的输入input0和input1，在维度dim上计算它们的cross-product（三维向量的叉乘）。广泛应用于大批量的向量/张量的叉乘计算。

example:

```
>>> a
tensor([[[[[[[[ 0  1]
  	      [ 2  3]
  	      [ 4  5]]

 	     [[ 6  7]
  	      [ 8  9]
  	      [10 11]]]]]]]])
>>> b
tensor([[[[[[[[13 12]
  	      [15 14]
  	      [17 16]]

 	     [[19 20]
  	      [18 21]
  	      [23 22]]]]]]]])
>>> torch.cross(a, b, dim=6)
tensor([[[[[[[[-26, -22],
              [ 52,  44],
              [-26, -22]],

             [[  4, -33],
              [ 52,  66],
              [-44, -33]]]]]]]])
```

计算方法是

```
(0,2,4)x(13,15,17)=(-26,52,-26)
(1,3,5)x(12,14,16)=(-22,44,-22)
(6,8,10)x(19,18,23)=(4,52,-44)
(7,9,11)x(20,21,22)=(-33,66,-33)
```

如果dim不为6则报错，也就是说对于a\[1]\[1]\[1]\[1]\[1]\[2]\[3]\[2]，dim指定的维度[0,7]的维度值必须是6；dim也支持负数，可以是-2，但这个例子中不能是-1等。

### 1.3 算子输入输出参数要求

| 参数       | 语义                | 类型（输入/输出） | 支持类型    | 物理布局 | 规模限制       |
| ---------- | ------------------- | ----------------- | ----------- | -------- | -------------- |
| buffer_in0 | 参与计算的张量0     | 输入              | half, float | ARRAY    | 八维           |
| buffer_in1 | 参与计算的张量1     | 输入              | half, float | ARRAY    | 八维           |
| shape      | 张量的维度          | 输入              | int         | ARRAY    | 一维，八个元素 |
| dim        | cross计算所在的维度 | 输入              | int         | /        | 无             |
| buffer_out | 计算的结果          | 输出              | half, float | ARRAY    | 八维           |

### 1.4 算子限制

| 限制类型     | 详细说明                                                     |
| ------------ | ------------------------------------------------------------ |
| 数据类型限制 | buffer_in0,buffer_in1,buffer_out的数据类型必须相同           |
| 布局限制     | buffer_in0,buffer_in1,buffer_out的维度必须和输入的shape参数相同 |
| 规模限制     | /                                                            |
| 功能限制     | /                                                            |
| 数据范围限制 | 对输入的大规模数据的shape有一些限制，详见下文3.1              |
| 原位限制     | /                                                            |
| stride 限制  | /                                                            |
| 广播限制     | /                                                            |
| 参数输入限制 | dim必须在[-8,7]之间，shape[dim]必须为3                       |

### 1.5 验收标准

#### 1.5.1 精度验收标准

本算子属于算数类算子，验收标准为 diff3=0。

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../MLU-OPS性能验收标准.md)。

## 2 算子接口设计

### 2.1 参考接口

- PyTorch

```python
# 给出PyTorch接口
torch.cross(input,other,dim=None,*,out=None)
input(Tensor):the input tensors
other(Tensor):the second input tensor
dim(int,optional):the dimension to take the cross-product in
out(Tensor,optional):the output tensor
```

### 2.2 接口设计

```python
# 给出BANGPy MLU-OPS算子接口
# 算子接口名称(module参数)
MluOpCross(inputs=[
                buffer_in0,
                buffer_in1,
                shape,
                self.dim
            ],   
            outputs=[buffer_out],)
```


## 3 实现方案设计

### 3.1 实现方案

算子的输入包括两个参与计算的buffer_in0和buffer_in1，buffer的规模shape，cross计算所在的维度dim；输出则是一个结果buffer_out；

首先将八维的buffer映射成(group, 3, step)，3就是原来dim所在的维度，由于cross的计算是在这一维度发生的，所以dim之前的所有维度可以映射到一维（group），dim之后的所有维度映射到一维(step)，group就代表有多少组向量要进行叉乘，step代表从当前向量的第一维开始隔多少个数据到下一个向量的第一维。

在流水中设置九个buffer，分别代表三维叉乘中(a0, a1, a2) x (b0, b1, b2) = (c0, c1, c2)九个分量。

设流水线buffer能容纳的最大的数的量为data_each_buffer：

step <= data_each_buffer时，将buffer reshape成(group*3, step)，计算每组task要处理的group数group_each_task，然后将流水线buffer取[0:data_calculated_each_time]，data_calculated_each_time = data_each_buffer // step x step，也就是step最大的倍数，多余部分舍弃，流水线buffer[0:data_calculated_each_time] reshape 成(data_calculated_each_time/step, step)，然后流水线就可以按step为倍数进行strided copy。

step > data_each_buffer时，一次流水处理不了一整个step，只能每次尽可能地将buffer填满做计算，后续的处理和同步在BANGPy2的框架下实现非常困难，所以这种情况暂时不支持。

在nram大小为512KB的情况下，留出30KB，流水线buffer最大可以取27392B（计算详见3.5），也就是float16容纳13696个数（data_each_buffer=13696)，float32容纳6848个数，而且这只是对shape = (group, 3, step)中step的限制，group的大小不受算子功能限制。nram大小为768KB则流水线buffer最大取41984byte，可以容纳20992个float16或10496个float32。

流水的计算部分则直接按(c0, c1, c2) = (a0, a1, a2) x (b0, b1, b2)实现即可（注意是叉乘，也就是c0 = a1 \* b2 - a2 \* b1)。

### 3.2 伪代码实现（可选）

流水的计算部分则直接按(c0, c1, c2) = (a0, a1, a2) x (b0, b1, b2)实现即可（注意是叉乘，也就是c0 = a1 \* b2 - a2 \* b1)。

### 3.2 伪代码实现（可选）

```
Cross(buffer_in0, buffer_in1, shape, dim, buffer_out):

group = 1;

step = 1;

if (dim < -8 or dim >7):
	exit

else if (dim < 0):
	dim = dim+8

if (shape[dim] != 3):
	exit
	
for i in(0,dim):
	group = group * shape(i)

for i in(dim, 8):
	dim = dim * shape(i)
	
if (step > data_each_buffer):
	exit

if (step <= data_each_buffer):

	//计算流水线每次可以处理的step数
	step_each_time = data_each_buffer//step
	
	//计算每组task要处理的group数量
	group_each_task = group // task_num
	rest = group % self.task_num
	//计算每组task计算开始的位置start和结束位置stop
	if(task_id < rest):
		group_each_task = group_each_task+1
        start = group_each_task * task_id * 3
		stop = group_each_task * (task_id + 1) * 3
	else:
		start = 3 * (group_each_task * task_id + rest)
		stop = 3 * (group_each_task * (task_id+1) + rest)
	stop = stop - 2
	
	//计算流水线次数
	loop_num = group_each_task//step_each_time
	if(group_each_task%step_each_time!=0):
		loop_num = loop_num + 1
		last_loop = 1
		//计算最后一次余数循环操作的数据量
		data_calculated_each_time_last = ((stop - start - (loop_num-1)*3*step_each_time)//3) * step
		if ((stop - start - (loop_num-1)*3*step_each_time)%3 != 0):
			data_calculated_each_time_last=data_calculated_each_time_last+step
	
	进入流水:
	load: 以step为单位进行strided copy (GDRAM→NRAM)
	compute: axb=c
	store: 以step为单位进行strided copy (NRAM→GDRAM)
	
```

### 3.3 拆分(任务拆分，多核拆分)

多核任务分配：

首先计算总的组数group，然后计算每个task分到的group数量，余数从task_id=0开始往后各分一个直到分完，例如7个group分给3个task，那么需要计算的group_each_task分别是(3,2,2)；

循环次数：

计算每轮循环可以处理的step数量step_each_time，那么循环次数loop_num等于group_each_task整除以step_each_time，如果有余数则loop_num加1并且设置last_loop=1，提醒程序在处理最后一次循环时需要处理的数据量是和之前的一般循环不同的。

数据拆分：

流水中设置9个buffer，分别代表axb=c中a,b,c的三维，接下来的阐述就可以以其中一组流水线buffer为视角。

根据流水线buffer的大小计算能处理的step的最大组数step_each_time，然后放弃多余的部分将buffer reshape成(group*3,step)，每次流水按step_each_time进行strided copy。

数据store回去则是load的反操作，同上使用strided copy或者将buffer内容分一次（step中剩余的数据大于等于buffer）或两次（step中剩余的数据小于buffer，坐标发生跳跃）拷回。

### 3.4 性能优化设计

1、资源分配

| 表项            | 分配策略                           |
| --------------- | ---------------------------------- |
| NRAM            | 用于流水线存储该轮流水要计算的数据 |
| WRAM            | /                                  |
| SRAM            | /                                  |
| DRAM(workspace) | /                                  |

2、流水设计

使用 `for_range` 自动构建多级流水，流水的结构是`lcs`；

流水分为三个阶段，第一阶段load本轮流水要计算的数据，分为a0,a1,a2,b0,b1,b2六个部分，a=(a0,a1,a2)，b=(b0,b1,b2)；计算阶段，计算c=(c0,c1,c2)=axb；第三阶段将计算的结果store到buffer_out里。

使用strided copy按step对数据进行load/store，放弃掉流水线buffer的一小部分空间，将流水线buffer reshape成(group*3,step)，然后使用strided copy，代码上较为规律简洁，同时尽可能地将流水线buffer利用起来。

### 3.5 可维护性设计

变量：

shape：输入和输出的张量的shape

dim：cross计算所在的维度

group，step：将shape映射为(group, 3, step)，group是dim之前的维度的乘积，step是dim之后的维度的乘积，均不含dim

length：总元素数量

group_each_task：每个task要计算的group量

loop_num：每个task在流水线中循环的次数

data_each_buffer：流水线中每个buffer的大小

last_loop：是否因为数据量按data_each_buffer为倍数处理会有余数而需要特殊处理最后一轮循环（等于1需要）

step_each_time：根据流水线buffer的大小计算能处理的step的最大组数

data_calculated_each_time：每轮流水在a,b,c向量其中一个维度（总共六个）上的操作的数据量

data_calculated_each_time_last：最后一轮流水处理余数时在a,b,c向量其中一个维度（总共六个）上的操作的数据量

start：task要处理的数据的起始位置，在流水线过程中全程不变

stop：当前task中buffer索引允许的最大值，作为界标，防止该task的计算扩张到下个task的计算范围里（给余数循环使用）

其他变量可以参考代码内的注释。

流水线buffer大小：

一个NRAM的大小是512x1024byte，先空出30KB，而流水线至多需要9（axb=c，a,b,c都是三维）x2=18个buffer，其次乘法和减法要128byte对齐，128x18=2304，(512-30)x1024//2304x2304=493056，493056/18=27392，27392可以被128整除，当然也可以被4(byte)和2(byte)整除，所以流水线一个buffer的大小设置为27392byte，也就是13696个float16或6848个float32。NRAM为768KB时，则流水线一个buffer的大小设置为41984byte，可以容纳20992个float16或10496个float32。

### 3.6 测试用例设计

(shape,dim):

  ((1, 1, 1, 1, 2, 3, 4, 5), 5),	#小规模一般测试

  ((2, 1, 2, 1, 2, 2, 2, 3), 7),    #dim=7小规模一般测试

  ((2, 1, 2, 1, 2, 2, 3, 3), 6),	#多个维度为3时的小规模一般测试

  ((3, 2, 2, 1, 1, 1, 1, 1), 0),	#dim=0小规模一般测试

  ((1, 2, 2, 2, 3, 128, 1, 1), 4),	#step较大的小规模一般测试

  ((1, 2, 2, 2, 3, 128, 1, 1), -4),	#dim为合法负数的测试

  ((1024, 2, 2, 3, 3, 4, 4, 4), 4),	#group较大的测试

  ((1, 1024, 2, 4, 3, 2, 3, 1024), 4),   #group和step均较大的测试

  ((1, 1024, 2, 4, 3, 2, 3, 1024) ,6),	#group和step均较大的测试

### 3.7 算子防呆检查

需要检查输入的input0/input1的shape是否是八维且其中至少有一维数值为3；

需要检查输入的数据类型是否是float16/float32；

需要检查输入的(shape,dim)是否符合shape[dim]=3且dim∈[-8,7]。

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

只需列出在测试过程中发现的性能/精度异常的规模。

| 提交日期 | 问题规模 | 问题描述 | 是否已修复 |
| -------- | -------- | -------- | ---------- |
|          |          |          |            |

### 4.2 已经过优化的规模说明

| 提交日期   | 修复规模                            | 修复问题            |
| ---------- | ----------------------------------- | ------------------- |
| 2022.05.01 | ((2, 1024, 4, 4, 3, 2, 3, 1024), 4) | 性能提升            |
| 2022.09.21 | 将算子更新到BANGPy2                 | 算子代码改为BANGPy2 |
|            |                                     |                     |

## 5 方案实施

### 5.1 开发测试计划

- 2022.02.20 调研BANGPy，阅读样例代码，学习如何使用BANGPy进行开发
- 2022.03.01 设计方案：算子功能+接口设计
- 2022.03.08 test 代码开发
- 2022.03.15 设计方案：如何实现将存储不连续的数据组合在一起进行计算
- 2022.04.01 开发算子非流水的小规模数据的代码
- 2022.04.15 在原基础上实现流水方案
- 2022.05.01 在原基础上实现支持大规模数据的方案
- 2022.05.15 测试当前算子实现方案的鲁棒性和性能，增加每次流水处理的数据量
- 2022.05.20 测试当前算子实现方案的鲁棒性和性能，发现需要追加设计dim所在的维度太大时的分支2
- 2022.05.15 测试当前算子实现方案的鲁棒性和性能，发现数据量太大时分支2存在bug，效率提升至62%左右
- 2022.05.30 增加对输入的防呆处理
- 2022.06.01 基本完成设计文档，请公司提出意见进行完善
- 2022.06.15  PR代码&测试报告
- 2022.06.30 完善设计方案和代码，算子入库

### 5.2 风险分析

算子逻辑是尽可能将nram占满的，在数据规模较大或追加功能（算子融合等）时可能会产生一些问题。
test_cross.py中，只对mlu370-s4另外修改了nram大小和IO带宽的值，后续需要还要在里面手动追加设置。
