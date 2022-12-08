# mluopVoxelization 算子开发设计方案

- #### 文档基本信息

| 算子名称    | mluopVoxelization |
| ----------- | -------------- |
| 编制人/日期 | 张少鹏/2022-11-29 |
| 审批人/日期 | 王远/2022-11-29   |
| 审批人/日期 | 卜德飞/2022-11-29 |

- #### 修改记录

| 修订版本 | 修订人 | 修订日期   | 修订描述 |
| ------ | ------ | ---------- | -------- |
| v1.0   | 张少鹏  | 2022-11-29 | 首次提交 |

- #### 内容描述

本文档为`mluopVoxelization`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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

该需求分析为框架原生算子实现功能的需求分析，对于框架原生支持但 MLU-OPS 当前版本不支持的功能，需要在`1.4算子限制` 章节中显式注明。未明确注明不支持的功能，默认 MLU-OPS 全部支持。

example:

| 算子功能简介                    | 将输入点集转换为体素                           |
| ------------------------------| ------------------------------------------- |
| 需求来源                       | mmcv                                        |
| 应用网络                       | centerpoint                                 |
| 输入数据类型                    | points: float32<br>voxel_size: float32<br>coors_range: float32  |
| 输入标量参数                   | max_points: int32<br>max_voxels: int32<br>NDim: int32<br>deterministic: bool  |
| 输入 Shape                    | points: [num_points, num_features]<br>voxel_size: [3]<br>coors_range: [6]  |
| 输入 Layout                   | ARRAY                                       |
| 输出数据类型                   | voxels: float32<br>coors: int32<br>num_points_per_voxel: int32<br>voxel_num: int32 |
| 输出 Shape                    | voxels: [max_voxels, max_points, num_features]<br>coors: [max_voxels, 3]<br>num_points_per_voxel: [max_voxels]<br>voxel_num: [1] |
| 输出 Layout                   | ARRAY                                       |
| 模式(可选）                    | 否                                           |
| 是否含有 dim/axis 等类似语义的参数且该参数支持负数/其他特殊处理 | 否                 |
| 是否含有 labels/index 等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 否     |
| 是否需要支持原位                | 否                                          |
| 是否需要支持 stride 机制        | 否                                          |
| 是否需要支持广播                | 否                                          |
| 0 元素检查是否直接返回          | 输出tensor的维度max_points或max_voxels为0时，返回MLUOP_STATUS_SUCCESS；输入tensor points、voxel_size和coors_range若存在0元素，返回MLUOP_STATUS_BAD_PARAM |
| 其他特殊需求(在线量化，融合，转数提前等，可选)| 无                                 |
| 本次开发优先支持的规模/模式     | 优先支持 deterministic=True 模式                |

### 1.2 算子功能和应用场景描述

给定有限三维空间，由coors_range描述，具体为coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max, coors_z_max框定的长方体空间。给定体素尺寸，由voxel_size描述其长宽高，体素在coors_range空间内沿x、y、z正方向紧密排布。将输入点集转换为体素，输出voxels每个体素内所有点的坐标及特征值，输出coors每个体素的位置，输出num_points_per_voxel每个体素内的点数量，输出voxel_num所有体素的数量。

算子竞品实现拆分成5个kernel，分别为dynamic_voxelize_kernel、point_to_voxelidx_kernel、determin_voxel_num、assign_point_to_voxel、assign_voxel_coors，其中前三个kernel顺序执行，后两个kernel并行执行。各个kernel实现功能及拆分逻辑分别为：

1. dynamic_voxelize_kernel统计points各点所在体素位置，中间结果存放在temp_coors，规模为[num_points, 3]，多核拆分在num_points维度上拆分。

2. point_to_voxelidx_kernel对体素去重并计算各点和体素的映射关系。依次统计points中所有点，记当前点为p_idx，计算p_idx点前所有点，与p_idx在同一体素的第一个点的序号，中间结果存放在point_to_pointidx。计算p_idx点前所有点，与p_idx在同一体素的点的数量，中间结果存放在point_to_voxelidx。point_to_pointidx和point_to_voxelidx规模均为[num_points]，多核拆分在num_points维度上拆分。

3. determin_voxel_num依次统计points中所有点，记录各点在去重后的第几个体素内，中间结果存放在coor_to_voxelidx。统计总共有多少个体素，存放在输出结果voxel_num，统计各体素内有多少个点，存放在输出结果num_points_per_voxel。该kernel不做拆分，单核执行。

4. assign_point_to_voxel，根据各点所在体素序号coor_to_voxelidx，以及各点是所在体素内的第几个点point_to_voxelidx，将points点坐标及特征值映射到输出结果体素voxels中。算子输出voxels，规模为[max_voxels, max_points, num_features]，多核拆分在num_points * num_features维度上拆分。

5. assign_voxel_coors，根据各点所在体素序号coor_to_voxelidx，以及各点是所在体素内的第几个点point_to_voxelidx，将temp_coors体素位置映射到输出结果coors中。算子输出coors，规模为[max_voxels, 3]，多核拆分在num_points * 3维度上拆分。

备注： 坐标不支持nan/inf。

### 1.3 算子输入输出参数要求

| 参数             | 语义                                         | 类型（输入/输出）    | 支持类型                | 物理布局   | 规模限制  |
| ---------------- | ------------------------------------------- | ----------------- | ---------------------- | -------- | -------- |
| handle           | handle                                      | 输入              | mluOpHandle_t           | /        | 无       |
| points_desc      | 对输入points的描述                            | 输入              | mluOpTensorDescriptor_t | /        | 无       |
| points           | 输入点的坐标及特征值                           | 输入              | float32                 | ARRAY    | [num_points, num_features] |
| voxel_size_desc  | 对输入voxel_size的描述                        | 输入              | mluOpTensorDescriptor_t | /        | 无       |
| voxel_size       | 体素的尺寸                                    | 输入              | float32                 | ARRAY   | [3]      |
| coors_range_desc | 对输入coors_range的描述                       | 输入              | mluOpTensorDescriptor_t | /        | 无       |
| coors_range      | 体素空间的边界                                | 输入              | float32                 | ARRAY    | [6]      |
| max_points       | 一个体素中最多容纳的点数                        | 输入              | int32                   | /        | 无       |
| max_voxels       | 生成体素的最大数量                             | 输入              | int32                   | /        | 无       |
| NDim             | 输出coors的最低维度，固定值为3                  | 输入              | int32                   | /        | 无       |
| deterministic    | 模式选择，deterministic/non-deterministic模式 | 输入              | bool                    | /        | 无       |
| voxels_desc      | 对输出voxels的描述                            | 输入              | mluOpTensorDescriptor_t | /        | 无       |
| voxels           | 输出体素内各点的坐标及特征值                     | 输出              | float32                 | ARRAY    | [max_voxels, max_points, num_features] |
| coors_desc       | 对输出coors的描述                             | 输入              | mluOpTensorDescriptor_t  | /        | 无      |
| coors            | 输出体素的位置                                | 输出              | int32                    | ARRAY    | [max_voxels, 3]  |
| num_points_per_voxel_desc | 对输出num_points_per_voxel的描述     | 输入              | mluOpTensorDescriptor_t  | /        | 无      |
| num_points_per_voxel      | 输出体素内点的数量                    | 输出              | int32                    | ARRAY    | [max_voxels]     |
| voxel_num_desc   | 对输出voxel_num的描述                         | 输入              | mluOpTensorDescriptor_t  | /        | 无       |
| voxel_num        | 输出体素的数量                                | 输出              | int32                    | ARRAY    | [1]      |

### 1.4 算子限制

| 限制类型     | 详细说明                                                                                                        |
| ------------ | --------------------------------------------------------------------------------------------------------------- |
| 功能限制     | 不支持non-deterministic模式，不支持dynamic模式                                                                     |
| 数据范围限制 | max_points >= 0; max_voxels >= 0                                                                                |
| 原位限制     | 不支持原位                                                                                                      |
| stride 限制  | 不支持 stride 机制                                                                                              |
| 广播限制     | 不支持广播                                                                                                       |

### 1.5 验收标准

#### 1.5.1 精度验收标准

按照[精度验收标准](../MLU-OPS-Accuracy-Acceptance-Standard.md)的要求明确本算子的精度标准。

本算子属于算术类算子，精度验收标准：diff3 == 0。

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../MLU-OPS-Performance-Acceptance-Standard.md)。

- 竞品性能测试：

| 平台                 | 框架版本        | 数据类型  | 数据规模    | 计算效率   | IO效率     | Hardware time     |
| -------------------- | ------------- | -------- | --------  | --------- | --------- | ----------------- |
| Tesla V100-SXM2-16GB | Pytorch 1.9.0 | float32  | points: [253999, 5]<br>voxel_size: [3]<br>coors_range: [6]<br>voxels: [30000, 20, 5]<br>coors: [30000, 3]<br>num_points_per_voxel: [30000]<br>max_points = 20<br>max_voxels = 30000<br>NDim = 3<br>deterministic=True | dynamic_voxelize_kernel: 14.021749%<br>point_to_voxelidx_kernel: 90.053915%<br>determin_voxel_num: 0.015771%<br>assign_point_to_voxel: 25.195309%<br>assign_voxel_coors: 37.441100% | dynamic_voxelize_kernel: 47.321993%<br>point_to_voxelidx_kernel: 37.777134%<br>determin_voxel_num: 0.017807%<br>assign_point_to_voxel: 54.476897%<br>assign_voxel_coors: 33.271951% | dynamic_voxelize_kernel: 19.588800us<br>point_to_voxelidx_kernel: 64.612053ms<br>determin_voxel_num: 78.915080ms<br>assign_point_to_voxel: 34.353600us<br>assign_voxel_coors: 10.230400us |

## 2 算子接口设计

### 2.1 参考接口

- MMCV

```c++
// CUDA(https://github.com/open-mmlab/mmcv/blob/v1.7.0/mmcv/ops/csrc/pytorch/voxelization.cpp):
void hard_voxelize_forward(const at::Tensor &points,
                           const at::Tensor &voxel_size,
                           const at::Tensor &coors_range,
                           at::Tensor &voxels,
                           at::Tensor &coors,
                           at::Tensor &num_points_per_voxel,
                           at::Tensor &voxel_num,
                           const int max_points,
                           const int max_voxels,
                           const int NDim = 3,
                           const bool deterministic = true)
```

### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API mluOpVoxelization(mluOpHandle_t handle,
                                              const mluOpTensorDescriptor_t points_desc,
                                              const void *points,
                                              const mluOpTensorDescriptor_t voxel_size_desc,
                                              const void *voxel_size,
                                              const mluOpTensorDescriptor_t coors_range_desc,
                                              const void *coors_range,
                                              const int32_t max_points,
                                              const int32_t max_voxels,
                                              const int32_t NDim,
                                              const bool deterministic,
                                              void *workspace,
                                              size_t workspace_size,
                                              const mluOpTensorDescriptor_t voxels_desc,
                                              void *voxels,
                                              const mluOpTensorDescriptor_t coors_desc,
                                              void *coors,
                                              const mluOpTensorDescriptor_t num_points_per_voxel_desc,
                                              void *num_points_per_voxel,
                                              const mluOpTensorDescriptor_t voxel_num_desc,
                                              void *voxel_num)

mluOpStatus_t MLUOP_WIN_API mluOpGetVoxelizationWorkspaceSize(mluOpHandle_t handle,
                                              const mluOpTensorDescriptor_t points_desc,
                                              const mluOpTensorDescriptor_t voxel_size_desc,
                                              const mluOpTensorDescriptor_t coors_range_desc,
                                              const int32_t max_points,
                                              const int32_t max_voxels,
                                              const int32_t NDim,
                                              const bool deterministic,
                                              const mluOpTensorDescriptor_t voxels_desc,
                                              const mluOpTensorDescriptor_t coors_desc,
                                              const mluOpTensorDescriptor_t num_points_per_voxel_desc,
                                              const mluOpTensorDescriptor_t voxel_num_desc,
                                              size_t *size)
```

## 3 实现方案设计

### 3.1 实现方案

mlu实现将1.2小节竞品实现的5个kernel合并为4个kernel，多核拆分均在num_points维度上拆分，各个kernel具体实现方案为：

1. dynamic_voxelize_kernel

输入voxel_size中，voxel_size[0] ~ voxel_size[2]分别为voxel_x, voxel_y, voxel_z，表示体素在x、y、z方向上的长宽高。

输入coors_range中，coors_range[0] ~ coors_range[5]分别为coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max, coors_z_max，从而给定了体素所在三维空间的边界范围。

体素从点(coors_x_min, coors_y_min, coors_z_min)沿x、y、z正方向紧密排布，体素在x、y、z方向上切分出的网格数量分别为grid_x, grid_y, grid_z，采取四舍五入的方式计算。

```math
\begin{aligned}
grid\_x = round((coors\_x\_max - coors\_x\_min) / voxel\_x) \\
grid\_y = round((coors\_y\_max - coors\_y\_min) / voxel\_y) \\
grid\_z = round((coors\_z\_max - coors\_z\_min) / voxel\_z)
\end{aligned}
```

deterministic=True模式输出结果唯一，因此实现逻辑中计算顺序决定了最终输出结果体素及点的排列顺序。该kernel在统计每个点所在体素位置时，按输入points点的排列顺序依次统计。计算逻辑为，若该点在边界范围框定的体素内，则记录该点所在体素在网格中的位置（以体素在x、y、z方向上网格的index表示）；若该点不在边界范围框定的体素内，则记录网格位置为-1值。

输入points中，points[:, :3]表示点xyz坐标，points[:, 3:]表示其他特征值。则该点所在体素位置(c_x, c_y, c_z)计算公式为：

```math
\begin{aligned}
c\_x = floorf((points[p\_idx][0] - coors\_x\_min) / voxel\_x) \\
c\_y = floorf((points[p\_idx][1] - coors\_y\_min) / voxel\_y) \\
c\_z = floorf((points[p\_idx][2] - coors\_z\_min) / voxel\_z)
\end{aligned}
```

其中，p_idx = 0,1,...,num_points - 1。kernel函数分别矢量计算c_x、c_y、c_z，按计算公式使用bangc指令__bang_sub_scalar()、__bang_mul_scalar()、__bang_floor()实现：

```cpp
// x - coors_x_min
__bang_sub_scalar((float *)points_x, (float *)points_x, coors_x_min, deal_num);
// y - coors_y_min
__bang_sub_scalar((float *)points_y, (float *)points_y, coors_y_min, deal_num);
// z - coors_z_min
__bang_sub_scalar((float *)points_z, (float *)points_z, coors_z_min, deal_num);
// (x - coors_x_min) / voxel_x
__bang_mul_scalar((float *)points_x, (float *)points_x, 1.0 / voxel_x, deal_num);
// (y - coors_y_min) / voxel_y
__bang_mul_scalar((float *)points_y, (float *)points_y, 1.0 / voxel_y, deal_num);
// (z - coors_z_min) / voxel_z
__bang_mul_scalar((float *)points_z, (float *)points_z, 1.0 / voxel_z, deal_num);
// c_x = floor((x - coors_x_min) / voxel_x)
__bang_floor((float *)nram_auxa, (float *)points_x, deal_num);
__bang_float2int32((int32_t *)c_x, (float *)nram_auxa, deal_num, 0);
// c_y = floor((y - coors_y_min) / voxel_y)
__bang_floor((float *)nram_auxa, (float *)points_y, deal_num);
__bang_float2int32((int32_t *)c_y, (float *)nram_auxa, deal_num, 0);
// c_z = floor((z - coors_z_min) / voxel_z)
__bang_floor((float *)nram_auxa, (float *)points_z, deal_num);
__bang_float2int32((int32_t *)c_z, (float *)nram_auxa, deal_num, 0);
```

若点在边界范围外，体素位置记为(-1, -1, -1)；若点在边界范围内，体素位置记为(c_x, c_y, c_z)。中间结果存放在temp_coors，公式如下：

```math
\begin{aligned}
temp\_coors[p\_idx][3] = \begin{cases}
(c\_x, c\_y, c\_z) & 0 <= c\_x < grid\_x \&\& 0 <= c\_y < grid\_y \&\& 0 <= c\_z < grid\_z \\
(-1, -1, -1) & else
\end{cases}
\end{aligned}
```

kernel函数使用bangc指令__bang_ge_scalar()、__bang_lt_scalar()、__bang_mul()，统计的points各点所在体素位置，存放在中间结果temp_coors，layout按[3, num_points]摆放。伪代码如下：

```cpp
// c_x >= 0
__bang_ge_scalar((int32_t *)nram_auxb, (int32_t *)c_x, (int32_t)0, deal_num);
// c_x < grid_x
__bang_lt_scalar((int32_t *)nram_auxc, (int32_t *)c_x, grid_x, deal_num);
// 0 <= c_x < grid_x
__bang_mul((int32_t *)auxiliary_a, (int32_t *)nram_auxb, (int32_t *)nram_auxc, deal_num);
// c_y >= 0
__bang_ge_scalar((int32_t *)nram_auxb, (int32_t *)c_y, (int32_t)0, deal_num);
// c_y < grid_y
__bang_lt_scalar((int32_t *)nram_auxc, (int32_t *)c_y, grid_y, deal_num);
// 0 <= c_y < grid_y
__bang_mul((int32_t *)nram_auxb, (int32_t *)nram_auxb, (int32_t *)nram_auxc, deal_num);
// c_x >= 0 && c_x < grid_x && c_y >= 0 && c_y < grid_y
__bang_mul((int32_t *)auxiliary_a, (int32_t *)auxiliary_a, (int32_t *)nram_auxb, deal_num);
// c_z >= 0
__bang_ge_scalar((int32_t *)nram_auxb, (int32_t *)c_z, (int32_t)0, deal_num);
// c_z < grid_z
__bang_lt_scalar((int32_t *)nram_auxc, (int32_t *)c_z, grid_z, deal_num);
// 0 <= c_z < grid_z
__bang_mul((int32_t *)nram_auxb, (int32_t *)nram_auxb, (int32_t *)nram_auxc, deal_num);
// 0 <= c_x < grid_x && 0 <= c_y < grid_y && 0 <= c_z < grid_z
__bang_mul((int32_t *)auxiliary_a, (int32_t *)auxiliary_a, (int32_t *)nram_auxb, deal_num);
__bang_not((int32_t *)nram_auxc, (int32_t *)auxiliary_a, deal_num);

__bang_mul((int32_t *)c_x, (int32_t *)c_x, (int32_t *)auxiliary_a, deal_num);
__bang_mul_scalar((int32_t *)nram_auxb, (int32_t *)nram_auxc, (int32_t)(-1), deal_num);
__bang_add((int32_t *)temp_coors_x, (int32_t *)c_x, (int32_t *)nram_auxb, deal_num);
__bang_mul((int32_t *)c_y, (int32_t *)c_y, (int32_t *)auxiliary_a, deal_num);
__bang_add((int32_t *)temp_coors_y, (int32_t *)c_y, (int32_t *)nram_auxb, deal_num);
__bang_mul((int32_t *)c_z, (int32_t *)c_z, (int32_t *)auxiliary_a, deal_num);
__bang_add((int32_t *)temp_coors_z, (int32_t *)c_z, (int32_t *)nram_auxb, deal_num);
```

2. point_to_voxelidx_kernel

体素内可能存在若干个点，若点数量大于max_points则只取前max_points个点放到输出voxels中。points点所在体素的数量（去重后）若大于max_voxels则只取前max_voxels个体素放到输出voxels、coors、num_points_per_voxel中，体素排列顺序在输出voxels、coors、num_points_per_voxel中保持一致。temp_coors中统计的体素可能存在重复体素，因此这里要进行去重操作。

每次统计points中的一个点，记该点为p_idx。按points顺序遍历p_idx点前的所有点，记录p_idx点所在体素中，按points顺序第一个同在该体素内的点的index，中间结果存放到point_to_pointidx，若p_idx点不在边界范围框定的体素内，point_to_pointidx[p_idx]记为-1；记录p_idx点是其体素内的第几个点，中间结果存放到point_to_voxelidx，若p_idx点不在边界范围框定的体素或者体素内点数已达到max_points，point_to_voxelidx[p_idx]记为-1。伪代码如下：

```cpp
for (int32_t p_idx = 0; p_idx < num_points; ++p_idx) {
  int32_t c_x_cur = temp_coors_x[p_idx];
  int32_t c_y_cur = temp_coors_y[p_idx];
  int32_t c_z_cur = temp_coors_z[p_idx];

  if (c_x_cur == -1 || c_y_cur == -1 || c_z_cur == -1) {
    point_to_pointidx[p_idx] = -1;
    point_to_voxelidx[p_idx] = -1;
    continue;
  }

  num = 0;
  __bang_eq_scalar((int32_t *)temp_coors_x, (int32_t *)temp_coors_x, c_x_cur, deal_num);
  __bang_eq_scalar((int32_t *)temp_coors_y, (int32_t *)temp_coors_y, c_y_cur, deal_num);
  __bang_eq_scalar((int32_t *)temp_coors_z, (int32_t *)temp_coors_z, c_z_cur, deal_num);
  __bang_mul((int32_t *)coors_mask, (int32_t *)temp_coors_x, (int32_t *)temp_coors_y, deal_num);
  __bang_mul((int32_t *)coors_mask, (int32_t *)coors_mask, (int32_t *)temp_coors_z, deal_num);
  uint32_t first_point = __bang_findfirst1((float *)coors_mask, deal_num);
  num = (int32_t)__bang_count((float *)coors_mask, deal_num);
  if (num > 0) {
    point_to_pointidx[p_idx] = (int32_t)first_point;
  } else if (num == 0) {
    point_to_pointidx[p_idx] = p_idx;
  }

  if (num < max_voxels) {
    point_to_voxelidx[p_idx] = (int32_t)num;
  }
}
```

3. determin_voxel_num

根据point_to_pointidx、point_to_voxelidx统计每个点在去重后的第几个体素内，中间结果存放到coor_to_voxelidx。统计总共有多少个体素，存放在输出结果voxel_num，统计各体素内有多少个点，存放在输出结果num_points_per_voxel。该kernel不做拆分，单核执行。伪代码如下：

```cpp
int32_t voxel_num_temp = 0;
for (int32_t point_idx = 0; point_idx < num_points; ++point_idx) {
  int32_t point_pos_in_voxel = point_to_voxelidx[point_idx];
  if (point_pos_in_voxel == -1) {
    continue;
  } else if (point_pos_in_voxel == 0) {
    int32_t voxel_idx = voxel_num_temp;
    if (voxel_num_temp >= max_voxels) {
      continue;
    }
    voxel_num_temp += 1;
    coor_to_voxelidx[point_idx] = voxel_idx;
    num_points_per_voxel[voxel_idx] = 1;
  } else {
    int32_t point_idx_temp = point_to_pointidx[point_idx];
    int32_t voxel_idx = coor_to_voxelidx[point_idx_temp];
    if (voxel_idx != -1) {
      coor_to_voxelidx[point_idx] = voxel_idx;
      num_points_per_voxel[voxel_idx] += 1;
    }
  }
}
*voxel_num = voxel_num_temp;
```

4. assign_point_coors_to_voxel

此时已知各点和体素的映射关系，根据各点所在体素序号coor_to_voxelidx，以及各点是所在体素内的第几个点point_to_voxelidx，将points点坐标及特征值映射到输出结果体素voxels中。将temp_coors体素位置映射到输出结果coors中，temp_coors中体素位置(c_x, c_y, c_z)未经去重，还需判断当前点是否是所在体素的第一个点，若是其体素内第一个点则输出至coors，这样coors中存放的就是去重后的体素。伪代码如下：

```cpp
for (int32_t p_idx = points_start; p_idx < points_end; ++p_idx) {
  int32_t num = point_to_voxelidx[p_idx];
  int32_t voxel_idx = coor_to_voxelidx[p_idx];
  if (num > -1 && voxel_idx > -1) {
    float *voxels_offset = voxels + voxel_idx * max_points * num_features + num * num_features;
    float *points_offset = points + p_idx * num_features;
    __memcpy_async(voxels_offset, points_offset, num_features * sizeof(float), GDRAM2GDRAM);

    if (num == 0) {
      temp_coors_nram[0] = (int32_t)temp_coors[p_idx * 3];
      temp_coors_nram[1] = (int32_t)temp_coors[p_idx * 3 + num_points];
      temp_coors_nram[2] = (int32_t)temp_coors[p_idx * 3 + num_points * 2];
      int32_t *coors_offset = coors + voxel_idx * 3;
      __memcpy_async(coors_offset, temp_coors_nram, 3 * sizeof(int32_t), GDRAM2GDRAM);
    }
  }
}
__asm__ volatile("sync;");
```

### 3.2 伪代码实现

见3.1小节。

### 3.3 拆分(任务拆分，多核拆分)

**任务类型U1: **

多核拆分在中间结果temp_coors、point_to_pointidx、point_to_voxelidx，以及算子输出voxels、coors、num_points_per_voxel，均以num_points维度进行拆分，将num_points均分到每个core中。

### 3.4 性能优化设计

1、资源分配

| 表项            | 分配策略                                                          |
| --------------- | --------------------------------------------------------------- |
| NRAM            | 保存临时数据                                                      |
| DRAM(workspace) | 存储points转置结果points_xyz，存储中间计算结果temp_coors、point_to_pointidx、point_to_voxelidx、coor_to_voxelidx |

- workspace空间划分

points_xyz按转置前points规模[num_points, num_features]申请workspace空间，temp_coors按其规模[num_points, 3]申请workspace空间，point_to_pointidx、point_to_voxelidx和coor_to_voxelidx按其规模[num_points]申请workspace空间。

2、流水设计

实现方案的前两个步骤dynamicVoxelize和point2Voxel，在num_points值较大，不能一次性load到片上的场景排流水有一定收益。calcPointsPerVoxel和assignVoxelsCoors不排流水。

### 3.5 可维护性设计

1、bangc 代码中加入必要的 log 信息，比如输入的规模、数据类型、layout 这些，以及如果出错会导致程序 core dump 的变量，比如 IO 指令的 data_size、dim xyz 的值等，这些信息都是有利于快速定位问题；

2、对每一个函数命名变量命名都有充分的注释；

3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代。

### 3.6 测试用例设计

- 算子在网络中用到的规模：
  points: [253999, 5]<br>
  voxel_size: [3]<br>
  coors_range: [6]<br>
  voxels: [30000, 20, 5]<br>
  coors: [30000, 3]<br>
  num_points_per_voxel: [30000]<br>
  max_points = 20<br>
  max_voxels = 30000<br>
  NDim = 3<br>
  deterministic=True

其他可根据需要进行补充。算子开发完毕后，补充测试报告链接。

### 3.7 算子防呆检查

- 列出算子需要做的防呆，比如

1、指针为空防呆；

2、0 元素检查防呆，VLOG(5)打印信息，是否返回与框架沟通；

3、对输入输出支持的 dtype、layout 以及 shape 进行防呆；

4、算子存在的自身的相关参数防呆。

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

只需列出在测试过程中发现的性能/精度异常的规模。

### 4.2 已经过优化的规模说明

无

## 5 方案实施

### 5.1 开发测试计划

- 2022.11.14 调研源码+设计方案
- 2022.11.23 GTest 代码开发
- 2022.11.23 generator 代码开发
- 2022.11.24 算子主体框架开发
- 2022.11.30 批量测试+测试报告+代码调测
- 2022.12.08 提交 PR
- 2022.12.09 算子入库

### 5.2 风险分析

point_to_voxelidx_kernel去重功能在mlu上实现可能存在性能影响。
