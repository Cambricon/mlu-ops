# indice convolution forward 算子开发设计方案


* #### 文档基本信息

| 算子名称    | indiceConvolutionForward                                     |
| ----------- | ------------------------------------------------------------ |
| 编制人/日期 | 韩月恒/2022-12-28                                            |
| 审批人/日期 | 董成威/2023-2-10                                             |
| 审批人/日期 | 杜泽坤/2023-2-10                                             |
| 审批人/日期 | 王远/2023-2-10                                               |

* #### 修改记录

| 版本号| 修订人 | 修订日期 | 修订描述 |
| ----- | ------ | -------  | -------  |
| V1.0  | 韩月恒 | 2022-12-28 | 首次提交 |

* #### 内容描述

本文档为`indice convolution forward`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

* #### 需求分析checklist

## 1 需求分析

### 1.1 算子需求分析

| 算子功能简介| 完成对稀疏张量正向卷积操作                             |
|-------------|--------------------------------------------------------------|
| 需求来源    | pytorch, mmcv                                                |
| 应用网络    | CenterPoint/OpenPCDet                                        |
| 输入数据类型| half, float                                                  |
| 输入Shape   | features: [numActIn, inChannels] (numActIn：输入中非零元素个数)<br> filters: [co, d, h, w, ci] <br> indice_pairs: [numFilter, 2, numActIn] (numFilter：卷积核中元素个数) <br> indice_num：numFilter 标量数组 <br> num_activate_out：标量 |
| 输入Layout  | features: ARRAY <br> filters: NDHWC/NCDHW/ARRAY <br> indice_pairs: ARRAY |
| 输出数据类型| half, float (且与输入一致)                                   |
| 输出Shape   | features_out: [numActOut, outChannels] (numActOut：输出中非零元素个数) |
| 输出Layout  | features_out: ARRAY                                          |
| 模式(可选） | 							     |
| 是否含有dim/axis等类似语义的参数且该参数支持负数/其他特殊处理 | 无         |
| 是否含有labels/index等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | indice_pairs 有index参数，负数index表示此位置无效 |
| 是否需要支持原位        | 否                                               |
| 是否需要支持stride机制  | 否                                               |
| 是否需要支持广播  | 否                                                     |
| 0元素检查是否直接返回  | 是, 但filters不能为零元素                         |
| 其他特殊需求(在线量化，融合，转数提前等，可选)| 暂无                       |
| 本次开发优先支持的规模/模式| \                                             |


### 1.2 算子功能和应用场景描述

稀疏计算在业界实践中有着大量应用。比如，雷达扫描中的数据存在大量的0，稀疏程度可达千分之一至万分之一。传统卷积对大量0进行的计算会造成极大的浪费，因此引入稀疏卷积（Sparse Convolution）。
在经过算法变换拆分过后的卷积过程中，filters只与有效值做计算，减少算力资源浪费。同时，引入的特有的稀疏存储形式Position-Value只存储非零值的位置与数值，从而降低了存储与IO压力。

初始化：
```math
\begin{aligned}
&\forall_{l\in [0,L)} \forall_{co\in [0,Co)} features\_out[l,co] = 0
\end{aligned}
```

计算公式为：

```math
\begin{aligned}
&\forall_{k\in [0,K)} \forall_{l\in[0,indices\_num[k])} \forall_{co\in [0,Co)} ~ features\_out[indices\_pair[k,0,l],co]\\
& += \sum_{ci\in [0,Ci)} (features\_in[indices\_pair[k,1,l],ci] \times filters[co,k/Kw,k\%Kw,ci])
\end{aligned}
```

注：
1、Sparse Convolution Forward 正向过程分为 GetIndexPairs() 和 indiceConvForward()。本次开发indiceConvForward()。

### 1.3 算子输入输出参数要求

| 参数             | 语义                                                         | 类型（输入/输出） | 支持类型               | 物理布局 | 规模限制 |
| ---------------- | ------------------------------------------------------------ | ----------------- | ---------------------- | -------- | -------- |
| handle           |                                                              | 输入              |                        | /        | 无       |
| features_desc    |                                                              | 输入              |                        | /        | 无       |
| features         | 输入的数值信息（Position-Value存储形式输入由数值，位置两部分组成） | 输入              | half, float            | ARRAY    | [numActIn, Ci] |
| filters_desc     |                                                              | 输入              |                        | /        | 无       |
| filters          | 卷积核                                                       | 输入              | half, float            | NDHWC/NCDHW/ARRAY | [Co, D, H, W, Ci] |
| indice_pairs_desc |                                                             | 输入              |                        | /        | 无       |
| indice_pairs     | 根据位置信息得到的输入输出映射关系                           | 输入              | int32                  | ARRAY    | [numFilter, 2, numActIn] |
| indice_num       | filters中每个元素与输入非零值进行计算的次数                  | 输入              | int64                  | 标量数组 | [numFilter] |
| num_act_out      | 输出中非零值的个数                                           | 输入              | int64                  | 标量     | 等于numActOut |
| inverse          | inverse模式开关，暂不支持                                    | 输入              | int64                  | 标量     | 无       |
| sub_m            | subm模式开关                                                 | 输入              | int64                  | 标量     | 无       |
| output_desc      |                                                              | 输入              |                        | /        | 无       |
| output           | 输出的数值信息                                               | 输出              | half, float            | ARRAY    | [numActOut, Co] |

### 1.4 算子限制

| 限制类型    | 详细说明                                            |
| ----------- | ------------------------------------------------------------ |
| 维度限制    | 目前只支持3维 |
| 模式限制    | 不支持inverse模式                            |
| 原位限制    | 不支持原位|
| stride限制  | 不支持stride机制|
| 广播限制    | 参数不支持广播|
| 功能限制1   | get_indice_pair的输入indice_in中如果存在重复坐标数据，得到的indice_pairs中也会存在重复数据，传入到indice_conv影响到结果。这样的输入实际上不合法，框架采用去重后取第一出现的值处理，indice_conv不做结果保证 |
| 功能限制2   | indice_pairs中输入对应的index越界会导致读取内存随机值从而输出结果随机；输出对应的index越界会导致结果中数值丢失，从而为0。这点无法对齐，无法防呆 |
| 功能限制3   | num_act_out在mmcv中没有防呆，但设置比正常值大会导致输出多出很多0，设置比正常值小会导致输出截断，设置为0或负值会导致cuda报错。此处不对齐mmcv，将num_act_out防呆，只能设置为正确值 |

### 1.5 验收标准

#### 1.5.1 精度验收标准

  - half：diff1 <= 3e-3, diff2 <= 3e-3
  - float：diff1 <= 1e-5, diff2 <= 1e-5

#### 1.5.2 性能验收标准

  - 当前受到scatter算子性能影响，预计在竞品1/10。后续已有优化方案，待排期完成。

## 2 算子接口设计

### 2.1 参考接口

- MMCV PyTorch
```python
indice_conv(features,
            filters,
            indice_pairs,
            indice_pairs_num,
            num_active_out,
            inverser=False,
            subm=False) -> features_out
```

位置：
- sparse_convolution_forward: mmcv/mmcv/ops/sparse_conv.py
- indice_convolution_forward: mmcv/mmcv/ops/sparse_ops.py


### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API mluOpGetIndiceConvolutionForwardWorkspaceSize(mluOpHandle_t handle,
                                      		                          const mluOpTensorDescriptor_t features_desc,
                                      		                          const mluOpTensorDescriptor_t filters_desc,
                                      		                          const mluOpTensorDescriptor_t indice_pairs_desc,
                                      		                          const mluOpTensorDescriptor_t features_out_desc,
                                      		                          const int64_t indice_num[],
                                      		                          const int64_t num_act_out,
                                      		                          const int64_t inverse,
                                      		                          const int64_t sub_m,
                                      		                          const size_t workspace_size)
```

```c++
mluOpStatus_t MLUOP_WIN_API mluOpIndiceConvolutionForward(mluOpHandle_t handle,
                                      		          const mluOpTensorDescriptor_t features_desc,
                                      		          const void *features,
                                      		          const mluOpTensorDescriptor_t filters_desc,
                                      		          const void *filters,
                                      		          const mluOpTensorDescriptor_t indice_pairs_desc,
                                      		          const void *indice_pairs,
                                      		          const int64_t indice_num[],
                                      		          const int64_t num_act_out,
                                      		          const int64_t inverse,
                                      		          const int64_t sub_m,
                                      		          void *workspace,
                                      		          const size_t workspace_size,
                                      		          const mluOpTensorDescriptor_t features_out_desc,
                                      		          void *features_out)
```

## 3 实现方案设计

### 3.1 实现方案


由于 sparse convolution 是一系列算子，其中前向过程分为GetIndicePairs和indiceConvForward。

```python
# 将输出置零，为后面累加做准备
fill(features_out, 0)
# 将卷积核转置为DHWCN，为遍历做准备
transpose(filters，DHWCN)
# 遍历卷积核filters在空间维度(D, H, W)中的每一个元素切片
for filter_element in filters:
  # 对于filters中的每一个元素，取indice_pairs里面的会与该元素做计算的input的index_in，根据index_in从features中gather实际计算的数值
  input_seg = gather(features, index)
  # filter_element的shape为(co, ci)， input_seg的(numAct_in, ci)，matmul的输出output_seg的shape为(numAct_in, co)
  output_seg = matmul(input_seg, filter_element)
  # out_seg根据indice_pairs中index_in对应的index_out来scatter_add到features_out上
  scatter_add(features_out, output_seg, index_out)
```

注:
- indice_pairs的形状为[numFilter, 2 , numActIn]，numFilter为filters在空间维度(D, H, W)中filter_element的个数，numActIn为输入中非零值的个数。
indice_pairs由GetIndicPairs使用indices（输入的位置信息，Position-Value存储形式输入由数值，位置两部分组成）计算得出。
- indice_num[] 为长度为numFilter的数组，其记录了每一个filters中元素会进行计算的次数。

### 3.2 伪代码实现（可选）

见上一节

### 3.3 拆分(任务拆分，多核拆分)

算子在host端调用多个MLUOP算子，拆分按各个算子内部逻辑进行。

### 3.4 性能优化设计
1、资源分配

| 表项            | 分配策略   |
| ----------------| -----------|
| NRAM            | 无 |
| WRAM            | 无 |
| SRAM            | 无 |
| DRAM(workspace) | (transposed filters | matmul_result | gathered features | matmul_extra) |

2、流水设计

本算子在调用各个算子完成，无流水设计。

### 3.5 方案理论性能

理论性能在竞品的1/5～1倍，硬件时间在一个数量级上。

### 3.6 可维护性设计

1、bangc代码中加入必要的 log信息，比如输入的规模、数据类型、layout

2、对每一个函数命名变量命名都有充分的注释

3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代

### 3.7 测试用例设计- 网络规模

| features | filters | indice_pairs | indice_num | num_active_out | features_out |
|----------|---------|--------------|------------|----------------|--------------|
| [248636,16] | [3,3,3,16,32] | [27,2,248636] | [27] | 280511 | [280511,32] |
| [280511,32] | [3,3,3,32,64] | [27,2,280511] | [27] | 149100 | [149100,64] |
| [149100,64] | [3,3,3,64,128] | [27,2,149100] | [27] | 58838 | [58838,128] |
| [58838,128] | [3,1,1,128,128] | [3,2,58838] | [3] | 45406 | [45406,128] |

### 3.8 算子防呆检查

 1、指针为空防呆；

 2、支持的平台防呆；

 3、filters 0元素检查防呆，其他输入零元素直接返回，VLOG(5)打印信息；

 4、对于workspace_size的检查防呆；

 5、输入输出支持的dtype、layout以及shape防呆；

 6、暂不支持的模式inverse参数防呆。

## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

### 4.2 已经过优化的规模说明

