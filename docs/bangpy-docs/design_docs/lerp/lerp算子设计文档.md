# BANGPy Lerp 算子开发设计方案

- #### 文档基本信息

| 算子名称    | Lerp       |
| ----------- | -------------- |
| 编制人/日期 | 郑磊磊/2022-3-1 |
| 审批人/日期 |    |

- #### 修改记录

| 修订人 | 修订日期   | 修订描述 |
| ------ | ---------- | -------- |
| 郑磊磊    | 2022-3-1 | 首次提交 |

- #### 内容描述

本文档为 `Lerp` 算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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
| 输入数据类型   | half, float                                 |
| 输入 Shape    | input1: [batches, hi, wi, channels]; input2: [batches, hi, wi, channels]; input3： Scalar or Tensor        |
| 输入 Layout   | input1: NHWC; input2: NHWC                                 |
| 输出数据类型    | half, float                                 |
| 输出 Shape    | [batches, ho, wo, channels]                  |
| 输出 Layout   | NHWC                                         |

### 1.2 算子功能和应用场景描述

功能：基于权重的线性插值，计算公式为：$out_i = start_i + wight_i * (end_i - start_i)$
	wight的默认为标量，值为0.5

例如：
    start = [[1., 2.], [3., 4.]]
    end = [[5., 6.], [8., 9.]]
    weight = 0.5
    output = [[3., 4.], [5.5, 6.5]]

### 1.3 算子输入输出参数要求

| 参数        | 语义 | 类型（输入/输出） | 支持类型    | 物理布局 | 规模限制 |
| -----------| ---------------------------- | ----------------- | ----------- | -------- | -------- |
| input1      | 输入形状为NHWC形状的buffer     | 输入              | half, float | NHWC     | 无       |
| input2      | 输入形状为NHWC形状的buffer     | 输入              | half, float | NHWC     | 无       |
| input3      | 标量或者输入形状为NHWC形状的buffer     | 输入              | half, float | NHWC     | 无       |
| output     | 输出形状为NHWC形状的buffer     | 输出              | half, float | NHWC     | 无       |

### 1.4 算子限制

example:

| 限制类型     | 详细说明                                                                                            |
| ------------ | ------------------------------------------------------------------------------------------------- |
| 数据类型限制 | input 和 output 必须同时为统一数据类型                                                                |
| 布局限制     | ？？？ |
| 规模限制     | ？？？                                                                               |

### 1.5 验收标准

#### 1.5.1 精度验收标准

本算子属于 `算术` 类算子，验收标准为 diff3=0。

#### 1.5.2 性能验收标准

待定

## 2 算子接口设计

### 2.1 参考接口

- PyTorch

```python
torch.lerp(input, end, weight, out=None)
```

### 2.2 接口设计

```python
MluOpLerp(start, end, weight, output)
```

## 3 实现方案设计

### 3.1 实现方案

在nram中开辟四个大小为 `1024 * 数据类型字节数` 的buffer分别用来存放输入start、输入end、输入weight的数据及输出output的结果，从gdram循环拷贝 `1024 * 数据类型字节数` 大小的数据至nram相应的buffer中，然后判断weight是否为标量，若为标量，则对weight进行广播，之后计算end-start，然后将结果与广播后的weight对应位相乘，再加上原先的start。

### 3.2 伪代码实现

```python
# buffer_in和buffer_out分别位于GDRAM
buffer_in0 = self.bp.Buffer(
	shape=(self.N, self.H, self.W, self.C),
	name="INPUT0",
	dtype=self.dtype,
	scope="global"
)
buffer_in1 = self.bp.Buffer(
	shape=(self.N, self.H, self.W, self.C),
	name="INPUT1",
	dtype=self.dtype,
	scope="global"
)
buffer_in2 = self.bp.Buffer(
	shape=(self.N, self.H, self.W, self.C),
	name="WEIGHT",
	dtype=self.dtype,
	scope="global"
)
buffer_out = self.bp.Buffer(
	shape=(self.N, self.H, self.W, self.C),
	name="OUTPUT",
	dtype=self.dtype,
	scope="global"
)

# 此处是在NRAM空间上
buffer_in0_n = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="INPUT0_N",
            dtype=self.dtype,
            scope="nram",
)
buffer_in1_n = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="INPUT1_N",
            dtype=self.dtype,
            scope="nram",
)
buffer_in2_n = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="WEIGHT",
            dtype=self.dtype,
            scope="nram",
)
buffer_out_n = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="OUTPUT_N",
            dtype=self.dtype,
            scope="nram",
)

# 额外空间
buffer_temp0_n = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="TEMP0",
            dtype=self.dtype,
            scope="nram",
)
buffer_temp1_n = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="TEMP1",
            dtype=self.dtype,
            scope="nram",
)
buffer_temp2_n = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="TEMP2",
            dtype=self.dtype,
            scope="nram",
)

with self.bp.for_range(0, loop_num) as i:
	start = i * data_calculated_each_time
	stop = start + data_calculated_each_time
	self.bp.memcpy(buffer_in0_n, buffer_in0[start, stop])
	self.bp.memcpy(buffer_in1_n, buffer_in1[start, stop])
	self.bp.memcpy(buffer_in2_n, buffer_in2[start, stop])

	self.bp.expand(buffer_temp0_n, buffer_in2_n)
	self.bp.subtract(buffer_temp1_n, buffer_in1_n, buffer_in0_n)
	self.bp.multiply(buffer_temp2_n, buffer_temp0_n,buffer_temp1_n)
	self.bp.add(buffer_out_n, buffer_in0_n, buffer_temp2_n)

	self.bp.memcpy(buffer_out[start:stop], buffer_out_n)
```
### 3.3 拆分(任务拆分，多核拆分)

### 3.4 性能优化设计

### 3.5 可维护性设计

### 3.6 测试用例设计

### 3.7 算子防呆检查

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

### 4.2 已经过优化的规模说明

## 5 方案实施

### 5.1 开发测试计划

### 5.2 风险分析

