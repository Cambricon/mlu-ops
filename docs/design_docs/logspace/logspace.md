# logspace 算子开发设计方案

- #### 文档基本信息

| 算子名称    | logspace     |
| ----------- | ------------ |
| 编制人/日期 | 薛浩然/2024-4-28 |

- #### 修改记录

| 版本号 | 修订人 | 修订日期  | 修订描述 |
| ------ | ------ | --------- | -------- |
| V1.0   | 薛浩然    | 2024-4-28 | 首次提交 |

- #### 内容描述

本文档为`logspace`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录。

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


| 算子功能简介                                  | 返回一个数组，数组的值为在区间 $[base^{start}, base^{stop}]$ 上指数均匀分布的`steps`个幂，输出数组的长度为`steps`                 |
| ----------------------------------------------| ----------------------------------------------------------- |
| 需求来源                                      | PyTorch                                    |
| 应用网络                                      | 科学计算                       |
| 输入数据类型                                  | start：float； </br>end: float； </br>steps: int; </br>base: float;          |
| 输入 Shape                                    | start：scalar； </br>end: scalar； </br>steps: scalar; </br>base: scalar;|
| 输入 Layout                                   | start：scalar； </br>end: scalar； </br>steps: scalar; </br>base: scalar;                                 |
| 输出数据类型                                  | out：float,bfloat16,half,int32                                          |
| 输出 Shape                                    | out：[steps]                                 |
| 输出 Layout                                   | out：ARRAY                                                        |                            |                                                             |
| 是否含有 dim/axis 等类似语义的参数且该参数支持负数/其他特殊处理| 不含带 dim/axis 语义的参数等 |
| 是否含有 labels/index 等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | steps参数和输出数组长度一致，不支持负数和0；当steps=1时，直接输出$base^{start}$ |
| 是否需要支持原位                               | 否                                                         |
| 是否需要支持 stride 机制                       | 否                                                       |
| 是否需要支持广播                               | 否                     |
| 0 元素检查是否直接返回                         | 否                                                       |
| 其他特殊需求(在线量化，融合，转数提前等，可选) |                                                             |
| 本次开发优先支持的规模/模式                    |                            |

### 1.2 算子功能和应用场景描述

算子功能：返回一个数组，数组的值为在区间 $[base^{start}, base^{stop}]$ 上指数均匀分布的`steps`个幂，输出数组的长度为`steps`。

目前并未找到合适的应用网络，torch中调用torch.logspace产生结果如下

```python
>>> torch.logspace(start=-10, end=10, steps=5)
tensor([1.0000e-10, 1.0000e-05, 1.0000e+00, 1.0000e+05, 1.0000e+10])
>>> torch.logspace(start=0.1, end=1.0, steps=5)
tensor([1.2589, 2.1135, 3.5481, 5.9566, 10.0000])
>>> torch.logspace(start=0.1, end=1.0, steps=1)
tensor([1.2589])
>>> torch.logspace(start=2, end=2, steps=1, base=2)
tensor([4.0])
```

对 inf/nan 输入的处理：对于float类型的三个参数`start`、`end`、`base`，均不支持 inf/nan 类型，在防呆检查中实现。

### 1.3 算子输入输出参数要求

| 参数             | 语义                               | 类型（输入/输出） | 支持类型    | 物理布局   | 规模限制 |
| ---------------- | ---------------------------------- | ----------------- | ----------- | ---------- | -------- |
| handle           | 当前可获得的资源信息                    | 输入                 | mluOpHandle_t           | /          | 无       |
| start  | 指定起始指数| 输入 | float |scalar     | 无       |
| end       | 指定结束指数               | 输入              |  float      | scalar       | 无       |
| steps  | 步数，指定指数的数量，同时也是输出数组长度| 输入 | int | scalar  | 无       |
| base  | 底数 | 输入 | float |scalar     | 无       |
| res_desc       | 对输出数据res的形状描述，包含res的数据类型等信息                  | 输入              | mluOpTensorDescriptor_t       |  /      | 无       |
| res       | 输出数组的地址                 | 输出              | float,bfloat16,half,int32       |  ARRAY      | 无       |


### 1.4 算子限制

| 限制类型    | 详细说明                                            |
| ----------- | ------------------------------------------------------------ |
| 数据类型限制| start、end、base为float，step为int，res为float/bfloat16/half/int32 |
| 布局限制    | 无 |
| 规模限制    | 无               |
| 功能限制    | 无                         |
| 数据范围限制|  无     |
| 原位限制    | 不支持原位|
| stride限制  | 不支持stride机制|
| 广播限制    | 不支持广播|

### 1.5 验收标准

#### 1.5.1 精度验收标准

本算子属于算数类算子，验收标准为 diff3=0。

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../MLU-OPS-Performance-Acceptance-Standard.md)。

## 2 算子接口设计

### 2.1 参考接口


- PyTorch

```python
//https://pytorch.org/docs/1.6.0/generated/torch.logspace.html
torch.logspace(start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

```


### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API mluOpLogspace(mluOpHandle_t handle, const float start, const float end, const int steps, const float base, const mluOpTensorDescriptor_t res_desc, void *res);
```

## 3 实现方案设计

### 3.1 实现方案

本函数参照pytorch实现，将[start,end]区间均匀划分，其划分间隔为step，之后向量化计算 $base^{start+step*index}$ 。

对于`steps`为1的情况，直接计算基底 `base` 的 `start` 次幂，并赋给输出数组的第一个元素。

对于`steps`大于1的情况，首先根据taskDim对任务进行拆分，每个kernel完成一部分结果的计算；

接着计算指数数列：先计算每个元素指数的间隔step，并生成一个从offset到offset+deal_num的递增数列inc_list，递增数列inc_list向量化乘step，并向量化加start，得到本core的指数数列；

最后，根据pow(x,y) = exp2(y * log2(x))公式，向量化计算得到本kernel的结果数组。



### 3.2 伪代码实现

输入：
- 指向输出数组的指针
- 起始值 `start`
- 结束值 `end`
- 步数 `steps`
- 基底 `base`

输出：
- 以base为底的指数均匀间隔的数组res[steps]

伪代码描述：
1. 如果步数 `steps` 为 1，那么：
   - 计算基底 `base` 的 `start` 次幂。
   - 将该值赋给输出数组的第一个元素。
   - 函数结束。

2. 如果步数 `steps` 大于1，首先计算本core需要处理的元素数量，具体见3.3拆分。

3. 之后计算指数间隔step，并生成offset到offset+deal_num的递增数列inc_list。

4. 递增数列inc_list向量化乘step，并向量化加start，得到本core的指数数列。

5. 根据pow(x,y) = exp2(y * log2(x))，计算log2(base)，并向量化乘指数数列，结果记为z。

6. 由于不存在load步骤，因此使用二级流水线，向量化计算exp2(z)，并根据res的数据类型进行类型转化，以及将结果存储到res对应位置。


### 3.3 拆分(任务拆分，多核拆分)
任务类型U1：对输出数组长度进行均匀拆分，每个核处理deal_num=steps/taskDim个数据。
对于不能被整除的情况，如果taskID<
steps%taskDim，那么当前core的deal_num+1。

### 3.4 性能优化设计

1、生成递增数列时，采用成倍增长的方式，先生成长度为64的递增数列0\~63，再通过向量化加的方式，生成64\~127，得到0\~127的数列。之后采用同样的方法，将向量长度翻倍，得到0\~255的数列，依此类推，得到deal_num长度的递增数列。经测试该方法快于直接生成deal_num长度的递增数列。

2、使用pow(x,y) = exp2(y * log2(x))的计算方法，通过调用__bang_mul_scalar()、__bang_pow2()的接口，快速计算幂。

3、使用二级流水线，用计算exp2(z)的compute时延覆盖store时延，经测试exp2(z)的计算步骤时延已经足以掩盖store的时延，因此没有将生成递增数列等步骤也放入二级流水线中的compute部分。

### 3.5 可维护性设计

1、bangc 代码中加入必要的 log 信息，比如输入的规模、数据类型、layout 这些，以及如果出错会导致程序 core dump 的变量，比如 IO 指令的 data_size、dim xyz 的值等，这些信息都是有利于快速定位问题；

2、对每一个函数命名变量命名都有充分的注释；

3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代。

### 3.6 测试用例设计

- 算子在网络中用到的规模：
  - res的数据类型: float32,half,bfloat16,int32
  - res的规模: [128],[65536],[131072],[98304],[262144]
  - 其他参数设置: 所有测试用例中，start、end、base均为1、3、10
- 边界case：
  测试了steps为1的情况，直接计算$base^{start}$。

  其他可根据需要进行补充。算子开发完毕后，补充测试报告，目前手工测试该算子，性能均在v100的1倍以内。

### 3.7 算子防呆检查

1、结果数组指针`res`不为空；

2、flaot类型参数`start`、`end`、`base`不为 inf/nan0，且`base`大于0；

3、长度参数`steps`大于0，且小于等于数组`res`的长度；

4、数据类型检查；

## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

无。

### 4.2 已经过优化的规模说明

无。
