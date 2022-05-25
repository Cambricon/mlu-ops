# BANGPy Add 算子开发设计方案

- #### 文档基本信息

| 算子名称     | pairwise_distance              |
| ----------- | -------------- |
| 编制人/日期  | UniqueSquirrel/2022-5-18 |
| 审批人/日期  |              |

- #### 修改记录

| 修订人           | 修订日期    | 修订描述 |
| --------------- | ---------- | ------- |
| UniqueSquirrel  | 2022-5-18 | 首次提交 |

- #### 内容描述

本文档为 `pairwise_distance` 算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

## 1 需求分析

### 1.1 算子需求分析

| 算子功能简介               | 计算两个张量的pairwise_distance                   |
| ------------------------ | ----------------------------------------|
| 需求来源                  | 为bangpy-ops提供算子demo                  |
| 应用网络                  |                                  |
| 输入数据类型               | float                             |
| 输入 Shape                | input1: [ length ]; input2: [ length ]  |
| 输入 Layout               | input1: ARRAY; input2: ARRAY            |
| 输出数据类型               | float                              |
| 输出 Shape                | [ length ]                               |
| 输出 Layout               | ARRAY                                    |

### 1.2 算子功能和应用场景描述

功能：计算两个张量的pairwise_distance

例如：tensor1([[ 0.2135, -1.1229],
        [ 1.7612,  0.5365]])

tensor2([[-0.3812, -1.4980],
        [-1.0483,  0.5707]])

output([0.7031, 2.8097])

应用场景：ResNet等

### 1.3 算子输入输出参数要求

| 参数   | 语义                  | 类型（输入/输出）| 支持类型     | 物理布局 | 规模限制      |
| ------ | --------------------- | -------------    | -----------  | ------   | --------      |
| input1 | 多维buffer | 输入     |  float           | ARRAY        |  无      | --------      |
| input2 | 多维buffer | 输入     |  float           | ARRAY        |  无      | --------      |
| output | 多维buffer | 输出     |  float           | ARRAY        |  无      | --------      |

### 1.4 算子限制

| 限制类型       | 详细说明                    |
| ------------   | -----------------------     |
| 数据类型限制   | input 和 output 维度可以不同|
| 布局限制       | 仅支持ARRAY的layout         |
| 规模限制       |                             |

### 1.5 验收标准

#### 1.5.1 精度验收标准

本算子属于 `算术` 类算子，验收标准为 diff3=0。

#### 1.5.2 性能验收标准

待定。

## 2 算子接口设计

### 2.1 参考接口

- pytorch

```python
torch.nn.PairwiseDistance
```

### 2.2 接口设计

```python
MluOpPairwiseDistance(_mlu_input1, _mlu_input2,
                 _mlu_paras, 
                 get_total_size(_shape1), get_total_size(_shape2),
                 _pd_len, _pd_height, _pd_width, _output_len
                 , _mlu_border_output, _mlu_border_idx_output, _mlu_output)
				 
_mlu_input1, _mlu_input2, 为输入的两个向量
_mlu_paras 为 eps，p，keepdim 参数 
get_total_size(_shape1), get_total_size(_shape2), 为两个向量的长度，张量1的长度永远不小于张量2
_pd_len 为输入张量最后一个维度的长度
_pd_height, _pd_width 为对第一个张量进行reshape后的高度和宽度
具体算法为：
shp_len = len(_shape1)
dim_index = shp_len - 1

# mlu 输入参数
_pd_len = _shape1[shp_len - 1]
_pd_height = 1
_pd_width = 1

for i in range(0, dim_index + 1):
    _pd_height *= _shape1[i]

_output_len,是输出张量长度
_mlu_border_output, _mlu_border_idx_output, _mlu_output ，因为数据分散到多核中，这几个用于存放中间输出结果，
在mlu计算完毕后，用cpu加工，得到最终结果
                 

```

## 3 实现方案设计

### 3.1 实现方案

1 将输入数据转为一维向量后，传入mlu，将数据平均分配在多核中。
首先计算两个向量的差，如果其中a向量比b向量短，则将b连续拷贝多份，变成和a长度相等（a的长度必须是b的整数倍）


a: |-----------------------|
b: |-------|

==>

a: |-----------------------|
b: |-------|-------|-------|


2 按照最后一个维度，将该tensor划分，拿到所有子向量
比如最后一个维度长度为2，张量总长为30，那么子向量就有15个。

t: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 


3 检查一下子张量的长度是不是超过了nram的大小，如果超过了，跳转到4，否则，跳转到6

4 将该子张量的数据分段拷贝到nram中，计算distance，缓存，然后再拷贝下一段，直到一个子张量计算完毕

5 某些张量很长，可能跨越了核，单个核可能只会计算一个子张量的前半部分，中间，或者后半部分，将其缓存到 mlu_border_output 中，跳转到8

6 将若干子张量拷贝到nram中，计算长度

7 某些子张量可能跨越了核，将这部分数据缓存

t: 
                   子张量跨越core了，前后两部分分别在不同core上计算的，前半部分和后半部分要保存
                   /         \
                  /           \
   .-------------.-------------.-------------.-------------.-------------.-------------.-------------.
   |----------core1---------|----------core2--------|-----------core3-------|-------------core4------|



8 统一处理，将 mlu_border_output 中缓存的数据拼接起来，得到这个子张量最终的distance

9 拷贝回cpu，执行reshape操作。

```

### 3.3 拆分(任务拆分，多核拆分)

采用的tasktype固定为UNION1，数据拆分到多核内计算。

### 3.4 性能优化设计
### 3.2 伪代码实现

```python

subtract_tensor = input_tensor1 - intput_tensor2

sub_tensors = get_last_dim(subtract_tensor)  #按照最后一个维度，讲该tensor划分，拿到所有子向量

for t in sub_tensors:
    length = calc_distance(t)
    _mlu_output.append(length)


### 3.5 可维护性设计


### 3.6 测试用例设



### 3.7 算子防呆检查

除host端自动生成的部分参数防呆检查外，暂不需要进行其他的防呆检查。

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

2022.4.30 算子入库

### 5.2 风险分析

暂无。