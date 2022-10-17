# NonZero 算子开发设计方案

- #### 文档基本信息

| 算子名称      | NonZero            |
| ----------- | ------------------ |
| 编制人/日期   | withfall/2021-12-27 |
| 审批人/日期   |                    |

- #### 修改记录

|   修订人  | 修订日期     | 修订描述  |
| -------- | ---------- | -------- |
| withfall | 2021-12-27 | 首次提交  |

- #### 内容描述

本文档为`NonZero`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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

| 算子功能简介       |      返回输入Tensor中非零元素的坐标    |
| ---------------- | ---------------------------------- |
| 需求来源          |      ONNX                          |
| 应用网络          |      翻译网络                        |
| 输入数据类型       |      half,float                     |
| 输入shape        |      4维输入                         |
| 输入layout       |      无layout限制                    |
| 输出数据类型       |      int64                         |
| 输出shape         |   2维输出 [输入维度数， NonZero数量]   |
| 输出layout        |       [输入维度数， NonZero数量]      |

### 1.2 算子功能和应用场景描述

功能： NonZero算子的功能是获取输入Tensor中非零元素的坐标。来自[ONNX NonZero](https://github.com/onnx/onnx/blob/master/docs/Operators.md#NonZero)。

例如： 

``` python
x = [[1, 0], [1, 1]]
y = NonZero(x)
# expected y: [[0, 1, 1], [0, 0, 1]]
```

应用场景： 翻译网络等

### 1.3 算子输入输出参数要求

| 参数      |             语义         | 类型（输入/输出） | 支持类型    | 物理布局 | 规模限制 |
| -------- | ------------------------ | -------------- | ---------- | ------- | ------- |
| input    |输入Tensor                 |   输入         | half,float  |  无限制 | 最后一个维度大小不能超过8192  |
| trans    |输出Tensor是否需要Transpose | 输入           | int         | 无限制   |  0或1  |
|output    |输出Tensor                 |  输出          | int64       | 无限制  | 无限制 |

### 1.4 算子限制

| 限制类型     | 详细说明              |
| ----------- | ------------------- |
| 数据类型     | input 只支持 half， float类型。 输出支持int64类型|
| 原位限制     | 不支持原位            |
|stride 限制  | 不支持stride 机制      |

### 1.5 验收标准

#### 1.5.1 精度验收标准

按照[精度验收标准](../../../MLU-OPS-Accuracy-Acceptance-Standard.md)的要求明确本算子的精度标准。

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../../../MLU-OPS-Performance-Acceptance-Standard.md)。

## 2 算子接口设计

### 2.1 参考接口

- ONNX

ONNX 接口：
```python
node = onnx.helper.make_node(
    'NonZero',
    inputs=['condition'],
    outputs=['result'],
)
```

ONNX cuda 接口：
```c++
// count nonzero elements in each block into counts_in_blocks,
// the counts_in_blocks buffer is pre-allocated on gpu first.
template<typename InputT>
cudaError_t NonZeroCountEachBlock(cudaStream_t stream, const InputT* input, int64_t x_size, int* counts_in_blocks);

// output nonzero positions using input x and prefix_counts for each blocks
template<typename InputT>
cudaError_t NonZeroOutputPositions(
    cudaStream_t stream, const InputT *x, int64_t size, int rank, const TArray<fast_divmod>& strides,
    const int* prefix_count, int nonzero_elements, int64_t* results
);
```


### 2.2 接口设计
为了高性能计算，对NonZero进行多核展开，在输出时需要知道每个核上NonZero的数量。
为此将NonZero计算前需要加上NonZeroCount计算步骤来对数据拆分到每个计算核上的NonZero数量。

- NonZeroCount计算接口：

```python
f_nonzero_count = build_module.build(
    NonZeroCount(dtype.name, align_size),
    target,
    "NonZeroCount",
)
```

- NonZero计算接口：

```python
f_nonzero = build_module.build(
    NonZero(
        target[:6],
        dtype.name,
        dtype.bytes,
        align_size,
    ),
    target,
    "NonZero",
)

```

## 3 实现方案设计

### 3.1 实现方案

**1. NonZeroCount实现如下：**

数据多核拆分后，由`tcp.count_nonzero()`接口来计算每个核上数据NonZero的数量，并且存储在core_count buffer中。

**2. NonZero实现如下：**

(1) 构造Tensor对应的坐标

NonZero需要获取对应NonZero的坐标值，而BANGPy当前没有直接获取Tensor中制定元素坐标的函数接口，
因此我们需要构造坐标Tensor。

一个张量输入Tensor对应的坐标Tensor可以如下所示：

```python
# 3维输入 shape: (2, 3, 4)
input:  [[[23, 21, 12, 34], [32, 22, 46, 54], [32, 22, 46, 54]], [[23, 21, 12, 34], [32, 22, 46, 54], [32, 22, 46, 54]]]

# 3维输入，input每个元素三个维度的坐标如下：
dim_2:  [[[ 0,  1,  2,  3], [ 0,  1,  2,  3], [ 0,  1,  2,  3]], [[ 0,  1,  2,  3], [ 0,  1,  2,  3], [ 0,  1,  2,  3]]]

dim_1:  [[[ 0,  0,  0,  0], [ 1,  1,  1,  1], [ 2,  2,  2,  2]], [[ 0,  0,  0,  0], [ 1,  1,  1,  1], [ 2,  2,  2,  2]]]

dim_0:  [[[ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0]], [[ 1,  1,  1,  1], [ 1,  1,  1,  1], [ 1,  1,  1,  1]]]

```

对于一个输入Tensor，可以按照上面Tensor元素坐标的规律来构造Tensor对应的坐标。

构造Tensor坐标由`tcp.assign()`接口来进行数据set。

(2) 计算Tensor对应的坐标

上面根据输入Tensor构造的坐标Tensor和输入input tensor，可以由`tcp.take()`来计算出非零元素的坐标。

这里需要注意`tcp.take()`只支持 half 和 float 数据类型。需求输出类型为int64类型，因为
`tcp.take()`接口只是对数据搬运。所以这里对坐标Tensor的数据类型设置为int32类型，在输入
`tcp.take()`时转为float类型，但是需要注意的是数据layout不做变化。

(3) 数据摆放

- 输出数据类型为int64

当前BANGPy不支持 int64 数据类型，并且坐标值没有负值，所以我们需要将int64的高32位置为0，低32位存储获取的坐标值。由此来获取int64数据类型结果。

- 输出结果tanspose

NonZero输出有是否进行transpose选项，如果不transpose可以直接将计算出的不同维
度的坐标输出，如果需要tranpose，则需要对输出坐标值进行transpose，顺序输出每个
非零元素的每个坐标值。

### 3.2 拆分
由上述设计描述，NonZeroCount 和 NonZero 可以归为elemwise计算。只需要根据
NRAM大小以及输入数据和构造的坐标数据大小来对input Tensor进行拆分。

### 3.3 性能优化
1.通过BANGPy 自动流水功能对NonZeroCount 和 NonZero 来进行自动三级流水
生成。

2.通过BANGPy 内存复用优化来减少NRAM内存申请并提升NRAM利用率。

### 3.4 可维护性设计

1、对每一个函数命名变量命名都有充分的注释。

2、对算子进行模块化设计与拆分，确保模块具有复用性。

### 3.5 测试用例设计

根据需要进行补充。详见算子测试文件。


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

- 2021.12.15 算子开发完成
- 2021.12.20 完善测例，增加压力测试完成测试
- 2021.12.24 完成代码合入

### 5.2 风险分析

暂无。



