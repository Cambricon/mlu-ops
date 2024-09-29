# logspace 算子开发设计方案

- #### 文档基本信息

| 算子名称    | logspace     |
| ----------- | ------------ |
| 编制人/日期 | 薛浩然/2024-8-27 |

- #### 修改记录

| 版本号 | 修订人 | 修订日期  | 修订描述 |
| ------ | ------ | --------- | -------- |
| V1.0   | 薛浩然    | 2024-8-27 | 首次提交 |

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


| 算子功能简介                                  | 该OP输出长度为`steps`的数组，其值为在区间 $[base^{start}, base^{stop}]$ 上指数均匀分布的幂。                 |
| ----------------------------------------------| ----------------------------------------------------------- |
| 需求来源                                      | PyTorch                                    |
| 应用网络                                      | 科学计算                       |
| 输入数据类型                                  | start：float； </br>end: float； </br>steps: int64; </br>base: float;          |
| 输入 Shape                                    | start：scalar； </br>end: scalar； </br>steps: scalar; </br>base: scalar;|
| 输入 Layout                                   | start：scalar； </br>end: scalar； </br>steps: scalar; </br>base: scalar;                                 |
| 输出数据类型                                  | out：float,half,int32                                          |
| 输出 Shape                                    | out：[steps]                                 |
| 输出 Layout                                   | out：ARRAY                                                        |                            |                                                             |
| 是否含有 dim/axis 等类似语义的参数且该参数支持负数/其他特殊处理| 不含带 dim/axis 语义的参数等 |
| 是否含有 labels/index 等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | steps 参数和输出数组长度一致，不支持负数；当 steps=0 时，直接返回；当 steps=1 时，直接输出 $base^{start}$ |
| 是否需要支持原位                               | 否                                                         |
| 是否需要支持 stride 机制                       | 否                                                       |
| 是否需要支持广播                               | 否                     |
| 0 元素检查是否直接返回                         | 否                                                       |
| 其他特殊需求(在线量化，融合，转数提前等，可选) |    无                                                         |
| 本次开发优先支持的规模/模式                    |                            |

### 1.2 算子功能和应用场景描述

**算子功能：** `logspace`输出长度为`steps`的数组，其值为在区间 $[base^{start}, base^{stop}]$ 上指数均匀分布幂。torch.logspace 输出如下。

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

**应用场景：** `logspace`算子无应用网络。

**对 inf/nan 输入的处理：** `logspace`输入参数`start`、`end`、`base`均支持 inf/nan 类型，输出均与 cuda 对齐。

### 1.3 算子输入输出参数要求

| 参数             | 语义                               | 类型（输入/输出） | 支持类型    | 物理布局   | 规模限制 |
| ---------------- | ---------------------------------- | ----------------- | ----------- | ---------- | -------- |
| handle           | 当前可获得的资源信息                    | 输入                 | mluOpHandle_t           | /          | 无       |
| start  | 指定起始指数| 输入 | float |scalar     | 无       |
| end       | 指定结束指数               | 输入              |  float      | scalar       | 无       |
| steps  | 步数，指定输出数组长度 | 输入 | int64 | scalar  | 无       |
| base  | 底数 | 输入 | float |scalar     | 无       |
| res_desc       | 对输出数据 res 的形状描述，包含 res 的数据类型等信息                  | 输入              | mluOpTensorDescriptor_t       |  /      | 无       |
| res       | 输出数组的地址                 | 输出              | float,half,int32       |  ARRAY      | 无       |


### 1.4 算子限制

| 限制类型    | 详细说明                                            |
| ----------- | ------------------------------------------------------------ |
| 输入限制     |  输入参数 steps >= 0 |
| 数据类型限制| start、end、base 只支持 float 输入 <br> step 只支持 int64 输入<br> res 支持 float/half/int32 类型指针输出 |
| 数据范围限制|  无     |
| 原位限制    | 不支持原位|
| stride 限制  | 不支持 stride 机制|
| 广播限制    | 不支持广播|

### 1.5 验收标准

#### 1.5.1 精度验收标准

按照精度验收标准的要求明确本算子的精度标准。 logspace 是计算类算子，采用动态阈值标准：diffs = [diff1, diff2]，threshold_rate = [10, 10] 。

#### 1.5.2 性能验收标准


- 良好： 算子 hw time 小于等于竞品 v100 的 5 倍 。
- 及格： 算子 hw time 小于等于竞品 v100 的 10 倍。

见 [MLU-OPS 性能验收标准](../MLU-OPS-Performance-Acceptance-Standard.md)。

## 2 算子接口设计

### 2.1 参考接口


- **PyTorch** ：https://github.com/pytorch/pytorch/blob/v1.13.0/aten/src/ATen/native/cuda/RangeFactories.cu#L124

```c++

Tensor& logspace_cuda_out(const Scalar& start, const Scalar& end, int64_t steps, double base, Tensor& result)

```


### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API mluOpLogspace(mluOpHandle_t handle, const float start, const float end, const int64_t steps, const float base, const mluOpTensorDescriptor_t res_desc, void *res);
```

## 3 实现方案设计

### 3.1 实现方案

`logspace`算子输出长度为`steps`的数组，其值为在区间 $[base^{start}, base^{stop}]$ 上指数均匀分布的幂。

- **参考算子的实现过程：**
1. 计算指数间隔 step = (end - start) / steps;
2. index 小于 steps / 2 时，计算 pow(base, start + step * index)；
3. index 大于等于 steps / 2 时，计算 pow(base, end - step * (steps - index - 1))。

- **MLU多核拆分策略**
1. 任务类型U1，均匀拆分输出数组，per_core_num = steps / taskDim。
2. 无法被整除时，若 taskId < remain_num ，则当前 core 数据量+1 。

#### 3.1.1 正底数实现

该分支计算`base`为非负数的情况，对应函数 dealNormalCase( ) 。

##### 3.1.1.1 正底数实现每个core上的数据量和偏移的计算
```c+
// 计算每个core上的计算量和偏移
const size_t per_core_num = steps / taskDim;
int32_t remain_num = steps % taskDim;
const size_t cur_core_num = taskId < remain_num ? per_core_num + 1 : per_core_num;
size_t cur_core_offset = taskId * per_core_num + (taskId < remain_num ? taskId : remain_num);

// 根据nram空间大小，计算core上需要循环的次数；
const int32_t max_deal_num = PAD_DOWN(LOGSPACE_NRAM_USED / 1 / sizeof(float), ALIGN_NUM_LOGSPACE);
const int32_t repeat_steps = cur_core_num / max_deal_num;
const int32_t remain_steps = cur_core_num % max_deal_num;

```

##### 3.1.1.2 正底数实现过程

1. 对`steps`进行拆分，均为为 taskdim 份，得到每个 core 上的计算量 cur_core_num

    ```c++
    per_core_num = steps / taskDim;
    remain_num = steps % taskDim;
    cur_core_num = taskId < remain_num ? per_core_num + 1 : per_core_num;
    ```

    计算每个 core 的偏移 cur_core_offset
    
    ```c++
    cur_core_offset = taskId * per_core_num + (taskId < remain_num ? taskId : remain_num
    ```

2. 计算指数间隔 step = (end - start) / steps，以及`base`的对数 base_log = log2(base)；

3. 若当前 core 上的计算量 cur_core_num 大于一次循环处理的长度 max_deal_num ，则对 cur_core_num 进行循环处理；

4. 重复外层循环，根据该 core 的偏置 cur_core_offset 以及当前循环次数 step_i ，计算当前循环在结果上的偏置 loop_offset 

    ```c++
    loop_offset = cur_core_offset + step_i * max_deal_num
    ```

5. 通过 __mluop_get_indices( ) ，生成从 loop_offset 开始，长度为 actual_deal_num，间隔为 1 的递增序列，即结果的下标 index ；

6. 根据下标 index 计算指数序列。 设 halfway = steps / 2 ，若 index < halfway ，则指数序列 index_y = start + step * index 。 index 与 index_y 均存储于数组 log2_result 

    ```c++
    __mluop_get_indices(log2_result, loop_offset, actual_deal_num);
    __bang_mul_scalar(log2_result, log2_result, step, actual_deal_num);
    __bang_add_scalar(log2_result, log2_result, start, actual_deal_num);
    ```

    若 index >= halfway ， 则指数序列 index_y = end + step * (index + 1 - steps) 。

    ```c++
    __mluop_get_indices(log2_result, loop_offset + 1 - steps, actual_deal_num);
    __bang_mul_scalar(log2_result, log2_result, step, actual_deal_num);
    __bang_add_scalar(log2_result, log2_result, end, actual_deal_num);
    ```

7. 使用 pow(base, index_y) = pow2(index_y * log2(base)) 计算结果 result_float 

    ```c++
    __bang_mul_scalar(log2_result, log2_result, base_log, actual_deal_num);
    __bang_pow2(result_float, log2_result, actual_deal_num);
    ```

    转化结果数据类型并拷贝结果。



##### 3.1.1.3 正底数实现的nram空间划分
由于`logspace`算子输入均为标量，无需nram加载数据；且该分支nram空间可进行复用，因此每轮循环使用整块nram，先后存储 index 、index_y 、result 。

#### 3.1.2 负底数实现
该分支`base`小于0，对应函数 dealBaseNegative( ) 。<br>指数为整数时，pow(x,y) = (-1)^y * pow2(y * log2|x|)；<br>指数为小数时，pow(x,y) = nan 。<br> 因此该分支需要对指数是否为整数，以及整数的奇偶进行判断。

##### 3.1.1.2 负底数实现每个core上的数据量和偏移的计算

与 3.1.1.1 正底数实现一致。

##### 3.1.1.2 负底数实现过程

前6步骤与 3.1.1.2 正底数实现一致，根据数组下标 index 计算得到指数序列 index_y 。

负数底数幂计算规则如下

- 指数为整数，则 pow(x,y) = (-1)^y * pow2(y * log2|x|)
- 指数为小数，则 pow(x,y) = nan

因此相比于正底数，本分支需要对指数是否为整数，以及整数的奇偶进行判断。计算得到 index_y 后步骤如下。

1. 判断指数是否为整数。将 index_y 取整得到 floor_y ，并与前者进行比较。floor_y 中非整数填 0 ，整数填 1

    ```c++
    __bang_floor(floor_y, log2_result, actual_deal_num);
    __bang_eq(floor_y, floor_y, log2_result, actual_deal_num);
    ```

    使用查找表，将 floor_y 中的非整数（0）刷为 0x7fffffff（float 类型的 nan），整数（1）刷为 0

    ```c++
    __bang_float2int32((int *)floor_y, floor_y, actual_deal_num, 0);
    table_for_integer_power[LUT_TABEL_LENGTH] = {0x7fffffff, 0};
    __bang_lut_s32((int *)floor_y, (int *)floor_y, (int *)table_for_integer_power, actual_deal_num, LUT_TABEL_LENGTH);
    ```

    将 floor_y 与结果按位或，完成指数为小数， pow(x,y) = nan 的处理

    ```c++
    __bang_bor((int *)log2_result, (int *)log2_result, (int *)floor_y, actual_deal_num);
    ```

2. 判断指数的奇偶性。复制 index_y 得到 y_copy ，转化为 int 类型，每个数据与 1 按位与。 y_copy 中偶数填 0 ，奇数填 1

    ```c++
    __bang_float2int32((int *)y_copy, log2_result, actual_deal_num, 0);
    __bang_band((int *)y_copy, (int *)y_copy, all_int_1, actual_deal_num);
    ```

    使用查找表，将 y_copy 中的偶数（0）刷为 0（数符+），奇数（1）刷为 0x80000000（数符-）

    ```c++
    table_for_odd_or_even_power[LUT_TABEL_LENGTH] = {0, 0x80000000};
    __bang_lut_s32((int *)y_copy, (int *)y_copy, (int *)table_for_odd_or_even_power, actual_deal_num, LUT_TABEL_LENGTH);
    ```

    将 y_copy 与结果按位或，完成指数为整数， pow(x,y) = (-1)^y * pow2(y * log2|x|) 的处理。

    ```c++
    __bang_bor((int *)result_float, (int *)result_float, (int *)y_copy, actual_deal_num);
    ```

3. 转化结果数据类型并拷贝结果。

##### 3.1.1.3 负底数实现的nram空间划分
该分支涉及到指数是否为整数，以及整数指数奇偶性的判断，将 nram 划分为 6 份：

1. result：长度为 max_deal_num ，存储中间量 index 、index_y ，以及计算结果 result；

2. floor_y：长度为 max_deal_num ，存储指数向下取整的结果，用于判断指数是否为整数；

3. y_copy：长度为 max_deal_num ，拷贝指数，并用于判断整数指数奇偶性；

4. all_int_1：长度为 max_deal_num ，内部全部填充为 1，通过与 y_copy 按位与，判断整数奇偶性。

5. 查找表 table_for_integer_power，长度为 64，用于是否为整数的刷数据。非整数指数（0）刷 0x7fffffff（float 类型的 nan），整数指数（1）刷 0。之后与结果进行按位或，将指数为小数的结果刷为 nan 。

6. 查找表 table_for_odd_or_even_power，长度为 64，用于整数奇偶的刷数据。偶整数（0）刷 0（符号位正），奇整数（1）刷 0x80000000（符号位负）。之后与结果按位或，奇数指数刷负，偶数指数刷正。

```c++
// nram: 划分为6份，前4块长度为 max_deal_num ，后2块长度为 64
// |  result   |  floor_y   |
// |  y_copy   |  all_int_1 |
// | table_for_integer_power  |  table_for_odd_or_even_power |
```

#### 3.1.3 其他计算逻辑

考虑到与 cuda 对齐，除上述 2 个主要分支外，还进行如下划分：

`steps`为 0，不调用 kernel ，直接返回。

kernel 内对分支进行如下划分。

1. `steps`为 1，直接计算 $base^{start}$；

2. `start`与`end`同时为 0 或同时为 inf，或者`base`为 1。结果均为 1 或 nan，填充数值；

3. `start`与`end`仅有 1 个为 inf，根据 cuda 的输出，结果填充 0，inf，nan 的组合；

4. `base`等于0，根据`start`与`end`的正负，在结果中填充 0 和 inf 的组合；

5. 间隔 step 等于 0，或在 half 类型下间隔过小。前一半结果为 $base^{start}$ ，后一半为 $base^{end}$；

6. 负底数分支，见 3.1.2；

7. 正底数以及底数为 nan 分支，见 3.1.1。

### 3.2 伪代码实现

输入：
- 指向输出数组的指针
- 起始值 `start`
- 结束值 `end`
- 步数 `steps`
- 基底 `base`

输出：
- 长度为`steps`的数组，其值为在区间 $[base^{start}, base^{stop}]$ 上指数均匀分布的幂

伪代码描述：
1. `steps`为 0，直接返回；

2. `steps`为 1，直接计算并返回 $base^{start}$；

3. `start`与`end`同时为 0 或同时为 inf，或者`base`为 1。结果均为 1 或 nan，填充数值；

4. `start`与`end`仅有 1 个为 inf，结果中填充 0，inf，nan 的组合；

5. `base`等于 0 ，此时根据`start`与`end`的正负，结果中填充 0 和 inf 的组合；

6. 间隔 step 等于 0，或在 half 类型下间隔过小。结果中前一半填充 $base^{start}$ ，后一半填充 $base^{end}$；

7. `base`小于 0。指数为整数时，pow(x,y) = (-1)^y * 2 ^(y * log2 |x|)；指数为小数时，pow(x,y) = nan；

8. `base`大于 0，以及输入中存在 nan。直接计算 pow(x,y) = 2 ^(y * log2 (x))。



### 3.3 拆分(任务拆分，多核拆分)

1. 任务类型U1，均匀拆分输出数组，per_core_num = steps / taskDim。

2. 无法被整除时，若 taskId < remain_num ，则当前 core 数据量+1 。

### 3.4 性能优化设计

1. 使用 pow(x,y) = exp2(y * log2(x)) 的计算策略快速计算幂。

2. 部分分支直接进行数值填充，减少计算量。

### 3.5 可维护性设计

1. bangc代码中加入必要的 log 信息，比如输入的规模、数据类型、layout，任务类型，以及如果出错会导致程序core dump的变量，比如IO指令的data_size、dim xyz的值等，这些信息都是有利于快速定位问题；

2. 对每一个函数命名变量命名都有充分的注释；

3. 避免魔鬼数字，对于确定的数字尽量使用公共宏来替代。

### 3.6 测试用例设计

1. 0元素测试

2. 给定的网络规模测试<br>
res的规模: [128],[65536],[131072],[98304],[262144]

3. inf/nan 测试


### 3.7 算子防呆检查

1. 指针为空防呆；

2. 对`steps`进行防呆，要求大于等于0；

3. output的数据类型须符合算子类型支持限制。


## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

无。

### 4.2 已经过优化的规模说明

无。
