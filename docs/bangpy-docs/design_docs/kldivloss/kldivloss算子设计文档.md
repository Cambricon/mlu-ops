# BANGPy KlDivLoss 算子开发设计方案

- #### 文档基本信息

| 算子名称     | KlDivLoss              |
| ----------- | -------------- |
| 编制人/日期  | 胡燕婷/2022-6-1 |
| 审批人/日期  |              |

- #### 修改记录

| 修订人           | 修订日期    | 修订描述 |
| --------------- | ---------- | ------- |
| 胡燕婷           | 2022-6-1 | 首次提交 |
| 胡燕婷           | 2022-8-9 | 修改 |

- #### 内容描述

本文档为 `KlDivLoss` 算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

- #### 算子需求 checklist

## 1 需求分析

### 1.1 算子需求分析

| 算子功能简介     |          衡量两个分布（离散分布和连续分布）之间的距离                             |
| ---------------|---------------------------------------------------------------------------|
| 需求来源        |  https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html        |
| 应用网络        |                                                                           |
| 输入数据类型     | input target :half, float        reduction log_target : int               |
| 输入 Shape     | input: [ length ,]; target: [ length ,]; reduction: [1]; log_target: [1]   |
| 输入 Layout    | input: ARRAY; target: ARRAY ; resuction: Scalar ; log_target:Scalar        |
| 输出数据类型     | half, float                                                               |
| 输出 Shape     | [ length, ]   / [1]                                                         |
| 输出 Layout    | ARRAY / Scalar                                                             |

### 1.2 算子功能和应用场景描述

功能：kldivloss算子是用来计算Kullback-Leibler Divergence(KL散度)，衡量两个分布的差异。该算子对标torch.nn.KLDivLoss算子

逐元素计算：
    如果target未进行过log操作： out[i] = target[i] * (target[i].log() - input[i])
    否则: out[i] = target[i].exp() * (target[i] - input[i])

然后根据reduction参数对结果进行归约：
    reduction == 'none' ： 不做归约，直接输出结果
    reduction == 'sum' ： loss = out.sum()  对结果进行求和
    reduction == 'mean' : loss = out.mean() 取结果的平均数
    reduction == 'batchmean' : loss = out.sum() / input.size(0) 

应用场景： 回归等

### 1.3 算子输入输出参数要求

| 参数        | 语义                         | 类型（输入/输出）| 支持类型     | 物理布局 | 规模限制      |
| ------     | ---------------------        | ------------- | ----------- | ------ | --------     |
| input      |  输入的形状为一维的buffer       | 输入           | half, float | ARRAY  | 无           |
| target     |  输入的形状为一维的buffer       | 输入           | half, float | ARRAY  | 无           |
| reduction  |  指定要应用于输出的归约值        | 输入           | int         | Scalar  | 无          |
| log_target |  指定target是否已经进行过log操作 | 输入           | int         | Scalar  | 无          |
| output     |  输出的形状为一维的buffer或者标量 | 输出           | half, float | ARRAY/Scalar |无           |

### 1.4 算子限制

| 限制类型      | 详细说明                 |
| ------------ | ----------------------- |
| 数据类型限制   | input、target和output的数据类型需要一致 |

### 1.5 验收标准

#### 1.5.1 精度验收标准

按照[精度验收标准](../../../MLU-OPS精度验收标准.md#精度验收标准)的要求明确本算子的精度标准

本算子属于 `复合` 类算子，验收标准为 diff1 <= 3e-3 && diff2 <= 3e-3。

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../../../MLU-OPS性能验收标准.md)。

## 2 算子接口设计

### 2.1 参考接口

- Pytorch

```python
torch.nn.KLDivLoss(
    size_average=None, reduce=None, reduction='mean', log_target=False
)
```

### 2.2 接口设计

```python
MluOpKldivloss(input, target, output, reduction, log_target)
```

## 3 实现方案设计

### 3.1 实现方案

1. 计算任务

计算部分对标pytorch中torch.nn.KLDivLoss算子的实现，默认input输入已经进行过log操作

```
if not log_target :
  output = target * ( log(target) - input)
else :
  output = exp(target) * (target - input)

``` 

2. 归约操作

初步计算结束之后会根据reduction参数值来进行归约
|  reduction  |    归约     |
|-------------|------------|
|      0      |    none    |  
|      1      |    sum     |  
|      2      |    mean    |
|      3      |  batchmean |


在对所有元素进行求和时，bangpy提供的sum函数只能进行逐128字节的求和，结果覆盖每128字节的第一个元素，需要手动去循环收集结果。因此选择设置合适大小的stride和kenerl大小，通过sumpool函数进行求和。
```
def compute_sum(in1, out_buf, out):
    self.tcp.sumpool(temp_buffer_pool, sum_input_pool, (sumpool_kernel_size, ), (sumpool_kernel_size, ))
    with self.tcp.for_range(begin=0, end=self.compute_row ) as i:
        self.tcp.sum(out_buf[0], temp_buffer[i * computed_size : (i + 1) * computed_size])
        out.assign(out + out_buf[0])
``` 

同样，由于需要128字节对齐以及sum的实现，

3. 数据同步

由于在最开始时，将数据尽量均匀分布在每块ipu上，在最终归约求和时需要选取一个ipu进行所有数据的统计。每块ipu在对所拥有数据的求和操作结束之后，会将sum值发送到gdram上，再统一拷贝到nram上，最终获取到sum的最终值。


### 3.2 伪代码实现（可选）

### 3.3 拆分(任务拆分，多核拆分)

一些基本的拆分原则：

1、拆分逻辑尽量保持均匀，避免出现负载不均衡，避免出现多核芯片单核工作，多核围观的情况出现。

2、尽可能保证拆分不会产生性能很差的 IO pattern，比如 stride 过大的访存，datasize 特别小的访存，非对齐的访存等等。

3、尽量保证拆分的时候不会造成重复的 IO，比如对 conv，如果对 HW 做拆分，由于有 kernel 的存在，hw 的 overlap 部分就会有重复的 IO。

4、拆分一定是和算子整体方案密切相关的，虽然模板把方案分成了几部分，但是这只是提醒大家关注这些重要的指标，并不是一部分一部分分开考虑的，最终方案肯定是拆分，资源分配，指令流水综合权衡得到的结果。

------------------------------------------------------------------------------------------

数据先均匀拆分到核内计算，剩余的数据添加到最后一个cluster。


### 3.4 性能优化设计

1. 通过bangpy自动流水功能对数据传输和计算任务进行自动流水生成
2. 通过BANGPy内存复用优化来减少NRAM内存申请并提升NRAM利用率。

### 3.5 可维护性设计

1、对每一个函数命名变量命名都有充分的注释。

2、对算子进行模块化设计与拆分，确保模块具有复用性。

3、代码风格遵守PEP8编码规范

### 3.6 测试用例设计

根据需要进行补充。详见算子测试文件。

### 3.7 算子防呆检查
1、在算子最开始会过滤掉不符合的数据类型和平台类型。

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

| 提交日期    | 问题规模  |       问题描述      |     是否已修复    |
| --------- | -------- | ------------------ | --------------- |
| 2022-6-1  |          | 单向IO算子IO性能受限  |    未修复        |

### 4.2 已经过优化的规模说明

| 提交日期  | 修复规模 | 修复问题 |
| --------- | -------- | -------- |
|           |          |          |

## 5 方案实施

### 5.1 开发测试计划

xx-xx-xx~2022-03-01 准备工作（学习白皮书，熟悉开发环境等）  
2022-03-01 算子调研与设计文档  
2022-03-14 开始编写代码  
2022-03-28 逻辑完善、性能优化与测试  
2022-05-31 编写与完善相关文档&代码  
2022-06-30 算子入库

### 5.2 风险分析

暂无。
