# DynamicPointToVoxelForward 算子开发设计方案

* ## 文档基本信息

| 算子名称      | DynamicPointToVoxelForward                                        |
| ------------- | ------------------------------------------------------------ |
| 编制人 / 日期 | 涂德江 / 2023-04-06                                          |

* ## 修改记录

| 版本号 | 修订人 | 修订日期   | 修订描述 |
| ------ | ------ | ---------- | -------- |
| v0.1  | 涂德江 | 2023-04-06 | 首次提交 |

* ## 内容描述

本文档为 `DynamicPointToVoxelForward` 算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录。
- ## 算子需求 checklist

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

| 算子功能简介           | 将具有相同体素坐标的多个点云数据，在特征维度上利用 `mean` 或 `max` 方法去重为一个点 |
| ---------------------- | ------------------------------------------------------------ |
| 需求来源               | mmcv                                                         |
| 应用网络               | mvxnet                                                       |
| 输入数据类型           | feats: float32 <br> coors: int32                             |
| 输入标量参数           | reduce_type: 枚举类型                                        |
| 输入Shape              | feats: [num_points, num_feats] <br> coors: [num_points, 3]   |
| 输入Layout             | ARRAY                                                        |
| 输出数据类型           | voxel_feats: float32 <br> voxel_coors: int32 <br> point2voxel_map: int32 <br> voxel_points_count: int32 <br> voxel_num: int32|
| 输出shape              | voxel_feats: [num_voxels, num_feats] <br> voxel_coors: [num_voxels, 3] <br> point2voxel_map: [num_points] <br> voxel_points_count: [num_voxels] <br> voxel_num: [1]  |
| 输出Layout             |  ARRAY                                                        |
| 是否需要支持原位       | 否                                                           |
| 是否需要支持stride机制 | 否                                                           |
| 是否需要支持广播       | 否                                                           |
| 0元素检查是否直接返回  | 输入的维度 num_points 、num_feats 和输出的维度 num_voxels 为 0 时，返回 MLUOP_STATUS_SUCCESS |

### 1.2 算子功能和应用场景描述

- #### 先验知识

3D点云数据集在三维空间中涉及到的数据信息一般包括: 位置信息、特征信息、三维空间范围、体素大小、体素坐标等，其中根据三维空间范围和体素大小，可以将一个三维空间划分为一个个体素坐标，其计算公式如下：

1）voxel_size:[voxel_x, voxel_y, voxel_z]，表示一个体素在x、y、z方向上的长宽高，即体素大小；

2）coors_range:[coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max, coors_z_max]，给定的三维空间范围；

如图黄色长方体即是三维空间范围 coors_range, 灰色小方块即是体素大小 voxel_size;

![voxel_size](./voxel_size.png)

将体素从点(coors_x_min, coors_y_min, coors_z_min)沿x、y、z正方向紧密排布，体素在x、y、z方向上切分出的网格数量分别为grid_x, grid_y, grid_z，采取四舍五入的方式计算:

```math
\begin{aligned}
grid\_x = round((coors\_x\_max - coors\_x\_min) / voxel\_x) \\
grid\_y = round((coors\_y\_max - coors\_y\_min) / voxel\_y) \\
grid\_z = round((coors\_z\_max - coors\_z\_min) / voxel\_z)
\end{aligned}
```

从 [0- grid_x)、[0- grid_y)、[0- grid_z), 任意一组数据组合就是三维空间体素坐标。

利用点云的位置信息可进一步获取其在三维空间中对应的体素坐标。其计算公式如下:

1）`feats`: [N, C], 表示有 N 个点云数据，每个点云有 C 个特征；

2）`point_coors`: [N, 3], 表示 N 个点云数据对应在三维空间中具体坐标信息，`feats`与`coors` 中数据是一一对应的

将点云所在三维空间坐标信息转为三维体素网格坐标，则该点所在体素坐标 (c_x, c_y, c_z) 的计算公式为：

```math
\begin{aligned}
c\_x = floorf((point_coors[idx][0] - coors\_x\_min) / voxel\_x) \\
c\_y = floorf((point_coors[idx][1] - coors\_y\_min) / voxel\_y) \\
c\_z = floorf((point_coors[idx][2] - coors\_z\_min) / voxel\_z)
\end{aligned}
```
其中, idx的取值范围为:[0 - N), 这样就可以将点云的坐标信息 point_coors:[N, 3] 转为体素坐标 `coors`:[N, 3];

说明：根据转换公式计算得到的三维空间中体素坐标`coors`，其中每个体素坐标(c_x, c_y, c_z)，代表的是空间中一个小长方体位置，那么一个体素坐标位置可能包含多个点数据。

- #### 算子功能

`dynamic_point_to_voxel_forward`算子的主要功能就是将具有相同体素坐标的所有点数据，在 `num_feats` 特征维度上利用 `mean` 或 `max` 方法进行去重; 该算子包含三个输入:`feats`、`coors`、`reduce_type`，五个输出：`voxel_feats`、`voxel_coors`、`point2voxel_map`、`voxel_points_count`、`voxel_num`; 实现算子功能可以划分 4 个部分:

1）对 `coors` 体素坐标进行有效值检查，如果 `coors` 中 x、y、z 的值有一个小于 0 则都赋值为 -1;

```c++
coors: [[-3,5,2],[-2,1,4],[6,7,8],[6,7,8],[2,4,6]]
result:
coors: [[-1,-1,-1],[-1,-1,-1],[6,7,8],[6,7,8],[2,4,6]]
```

2）将体素坐标 `coors` 进行排序、去重，得到新的体素坐标 `voxel_coors`; 保存去重后体素的个数 num_voxels 到 `voxel_num`; 保存 `coors` 中每个体素坐标在 `voxel_coors` 中对应的索引到 `point2voxel_map`; 保存 `voxel_coors` 中每个体素坐标在 `coors` 中出现的个数到 `voxel_points_count`;

 ![point2voxel](./point2voxel.png) 

该步骤其实是 `unique` 操作：
- `mode = sort`
- `input` 为 `coors`
- `output` 为 `voxel_coors`
- `indices` 为 `point2voxel_map`
- `count` 为 `voxel_num`

 ```c++
 coors: [[-1,-1,-1],[-1,-1,-1],[6,7,8],[6,7,8],[2,4,6]]

 result:
 voxel_coors: [[-1,-1,-1],[2,4,6],[6,7,8]]  // 坐标 coors 进行排序、去重
 point2voxel_map:[0,0,2,1,1]                // 记录 coors 中元素在 voxel_coors 中的 idx
 voxel_points_count: [2,1,2]                // 记录 voxel_coors 每个元素的 count
 voxel_num:[3]
 ```

3）如果 `voxel_coors[i,:], i=0,1,2,...` 中包含小于 `0` 元素，则删除 `voxel_coors[i,:]` 以及 `voxel_points_count` 中对应 `count`，同时 `point2voxel_map -= 1`

```c++
result:
voxel_coors: [[2,4,6],[6,7,8]]
point2voxel_map:[-1,-1,1,1,0]
voxel_points_count: [1,2]
voxel_num:[2]
```

4）遍历 `feats` 中每个点，在特征维度上，对每个值根据 `reduce_type` 的方法进行计算，将结果保存到 `voxel_feats` 中; 当 `reduce_type` = `max`, 在特征维度上对每个值取最大的值; 当 `reduce_type` = `mean`, 将特征维度每个值都累加到 `voxel_feats` 对应位置中，再利用 `voxel_points_count` 获取该体素位置在原始体素中出现的个数，再对 `voxel_feats` 的特征维度求平均。

```c++
coors: [[-1,-1,-1],[-1,-1,-1],[6,7,8],[6,7,8],[2,4,6]]
voxel_coors: [[2,4,6],[6,7,8]]
point2voxel_map:[-1,-1,1,1,0]

// point2voxel_map 中 0,1 为-1，表示该位置 feats 数据无效
// 以 reduce_type=max 为例:
//    voxel_feats[0] 为 point2voxel_map 中 0 值对应的 feats 进行reduce
//    voxel_feats[1] 为 point2voxel_map 中 1 值对应的 feats 进行reduce
voxel_feats[0, i] = max(feats[2,i], feats[3,i]) 
voxel_feats[1]    = feats[4]                          
```

- #### nan/inf

1）feats 支持 nan/inf;

2）coors 不支持 nan/inf。

###  1.3 算子输入输出参数要求

| 参数          | 语义                       | 类型（输入/输出） | 支持类型              | 物理布局 | 规模限制 |
| ------------- | -------------------------- | ----------------- | --------------------- | -------- | -------- |
| handle        | mluOp 上下文的指针         | 输入              | mluOpHandle_t         | -        | 无       |
| reduce_type   | 多点reduce操作的枚举       | 输入              | mluOpReduceMode_t     | -        | 无       |
| feats_desc    | 输入数据 feats 的描述符    | 输入              | mluOpTensorDescriptor | -        | 无       |
| feats         | 输入点云数据特征的指针     | 输入              | float32               | ARRAY    |  见 1.4  |
| coors_desc    | 输入数据 coors 的描述符    | 输入              | mluOpTensorDescriptor | -        | 无       |
| coors         | 输入点云体素坐标数据的指针 | 输入              | int32                 | ARRAY    | 见 1.4   |
| workspace     | 算子所需额外的空间         | 输入              | void*                 | -        | 无       |
| workspace_size| workspace所需大小          | 输入              | size_t                | -        | 无       |
| voxel_feats_desc | 输出数据 voxel_feats的描述符 | 输入         | mluOpTensorDescriptor | -        | 无       |
| voxel_feats   | 输出去重后点云特征数据的指针| 输出             | float32               | ARRAY    | 无       |
| voxel_coors_desc | 输出 voxel_coors 的描述符| 输入             | mluOpTensorDescriptor | -        | 无       |
| voxel_coors   | 输出去重后点云体素坐标数据的指针| 输出         | int32                 | ARRAY    | 无       |
| point2voxel_map_desc | 输出 point2voxel_map 的描述符| 输入     | mluOpTensorDescriptor | -        | 无       |
| point2voxel_map | 输出原始点在去重点中索引数据的指针| 输出     | int32                 | ARRAY    | 无       |
| voxel_points_count_desc| 输出 voxel_points_count 的描述符| 输入| mluOpTensorDescriptor | -        | 无       |
| voxel_points_count | 输出去重点在原始点集中重复个数的指针| 输出| int32                 | ARRAY    | 无       |
| voxel_num_desc | 输出 voxel_num 的描述符     | 输入            | mluOpTensorDescriptor | -        | 无       |
| voxel_num      | 输出去重后点的个数的指针    | 输出            | int32                 | ARRAY    | 无       |

### 1.4 算子限制

| 限制类型         | 详细说明                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------- |
| 数据类型限制     | feats仅支持 float 类型; coors 仅支持 int32 类型                                                   |
| 布局限制         | 仅支持 layout 为 ARRAY                                                                            |
| 原位限制         | 不支持原位                                                                                        |
| stride限制       | 不支持 stride 机制                                                                                |
| 广播限制         | 不支持广播                                                                                        |
| shape 限制       | feats、coors、point2voxel_map的第一维度 dims[0] 都相等; voxel_num 的维度为 1                      |
| shape 限制       | voxel_feats、voxel_coors、voxel_points_count的第一维度 dims[0] 都相等，且<= feats的第一维度 dims[0] |
| shape 限制       | voxel_feats、feats的第二维度 dims[1] 都相等; coors、voxel_coors的第二维度 dims[1] 都相等且等于 3    |

### 1.5 验收标准

#### 1.5.1 精度验收标准

按照[MLU-OPS 算子精度验收标准](../../../MLU-OPS-Accuracy-Acceptance-Standard.md)的要求明确本算子的精度标准。
本算子为复合类算子：
- #### voxel_feats
- 算子精度验收标准：diff1、diff2;
- 算子精度阈值描述：diff1 <= 3e-3 && diff2 <=3e-3;

- #### voxel_coors、point2voxel_map、voxel_points_count、voxel_num
- 算子精度验收标准：diff3;
- 算子精度阈值描述：diff3 = 0;

综上，改算子为复合类算子，算子精度验收标准：diff1、diff2; 算子精度阈值描述：diff1 <= 3e-3 && diff2 <=3e-3;

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../../../MLU-OPS-Performance-Acceptance-Standard.md)：

## 2 算子接口设计

### 2.1 参考接口

- CUDA

```c++
std::vector<at::Tensor>
DynamicPointToVoxelForwardCUDAKernelLauncher(const at::Tensor &feats, 
                                             const at::Tensor &coors,
                                             const reduce_t reduce_type)
```

### 2.2 接口设计

```c++
typedef enum {
  MLUOP_REDUCE_DSUM  = 0, /*!< Computes the sum value. */
  MLUOP_REDUCE_DMEAN = 1, /*!< Computes the mean value. */
  MLUOP_REDUCE_DMAX  = 2, /*!< Computes the maximun value. */
} mluOpReduceMode_t;

mluOpStatus_t MLUOP_WIN_API
mluOpGetDynamicPointToVoxelForwardWorkspaceSize(mluOpHandle_t handle,
                                                const mluOpTensorDescriptor_t feats_desc,
                                                const mluOpTensorDescriptor_t coors_desc,
                                                size_t *workspace_size)

mluOpStatus_t MLUOP_WIN_API 
mluOpDynamicPointToVoxelForward(const mluOpHandle_t handle,
                                const mluOpReduceMode_t reduce_type,
                                const mluOpTensorDescriptor_t feats_desc,
                                const void *feats,
                                const mluOpTensorDescriptor_t coors_desc,
                                const void *coors,
                                void *workspace,
                                const size_t workspace_size,
                                const mluOpTensorDescriptor_t voxel_feats_desc,
                                void *voxel_feats，
                                const mluOpTensorDescriptor_t voxel_coors_desc,
                                void *voxel_coors，
                                const mluOpTensorDescriptor_t point2voxel_map_desc,
                                void *point2voxel_map，
                                const mluOpTensorDescriptor_t voxel_points_count_desc,
                                void *voxel_points_count，
                                const mluOpTensorDescriptor_t voxel_num_desc,
                                void *voxel_num)

```

## 3 实现方案设计

### 3.1 实现方案

#### 3.1.1 计算原理说明

`dynamic_point_to_voxel_forward` 算子包含三个输入:`feats`、`coors`、`reduce_type`，五个输出：`voxel_feats`、`voxel_coors`、`point2voxel_map`、`voxel_points_count`、`voxel_num`; 根据 1.2 节算子功能, 可将算子 4 个部分分为 4 个 kernel 来实现:

- #### 计算逻辑层面

- kernel1

对 `coors` 中体素坐标进行检查，对无效体素坐标都赋值为 -1;

- kernel2

对输入体素坐标 `coors` 排序、去重得到输出 `voxel_coors`、`point2voxel_map`、`voxel_points_count`、`voxel_num`;

- kernel3

若 `voxel_coors[0][0] < 0` , 则将 `voxel_coors` 第一个体素坐标(-1, -1, -1)删除，`voxel_num` 的第一个数值删除，`voxel_points_count` 所有值减 1, `voxel_num`;

- kernel4

当输入 `reduce_type` = `max`时, 对具有相同体素坐标的输入点云特征 `feats` 在特征维度 num_feats 上取最大值保存到 `voxel_feats` 中; 当输入 `reduce_type` = `mean`时, 对具有相同体素坐标的输入点云特征 `feats` 在特征维度 num_feats 上进行累加保存到 `voxel_feats` 中; 当输入 `reduce_type` = `mean`时, 根据 `voxel_points_count` 中每个位置的值对点云特征 `voxel_feats` 在特征维度 `num_feats` 上每个值求平均;

#### 3.1.2 实现方案

通过以上分析, 要实现 `dynamic_point_to_voxel_forward` 算子功能，可以通过以上 4 个 kernel 来实现，其详细实现方案如下: 

- #### host 端

在 host 端主要进行 kernel 的逻辑调用:

- kernel1: KernelMaskFillCoorsForward

该 kernel 用于完成 3.1.1节 第 1 点;

```c++
const int num_points = feats_desc->dims[0];
KERNEL_CHECK((KernelMaskFillCoorsForward(
      k_dim, k_type, handle->queue, num_points, coors)));
```

- kernel2: cnnlUnique_v2 

该 kernel 用于完成 3.1.1节 第 2 点;

```c++
  cnnlUniqueSort_t unique_mode = CNNL_SORT_ASCEND;
  cnnlUniqueDescriptor_t unique_desc;

  CALL_CNNL(cnnlCreateUniqueDescriptor(&unique_desc));
  CALL_CNNL(cnnlSetUniqueDescriptor(unique_desc, unique_mode, 0, true, true));

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(coors_desc, cnnl_input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(voxel_coors_desc, cnnl_output_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(point2voxel_map_desc, cnnl_indices_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(voxel_points_count_desc, cnnl_counts_desc);

  CALL_CNNL(cnnlUnique_v2(cnnl_handle, unique_desc, cnnl_input_desc,
                          coors, workspace, workspace_size, (int *)voxel_num,
                          cnnl_output_desc, voxel_coors, cnnl_indices_desc,
                          point2voxel_map, cnnl_counts_desc, voxel_points_count));
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_indices_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_counts_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  
  CALL_CNNL(cnnlDestroyUniqueDescriptor(unique_desc));
  int32_t num_voxels = 0;
  cnrtMemcpy(&num_voxels, voxel_num, sizeof(int), cnrtMemcpyDevToHost);
```
- kernel3: KernelRemoveFirstForward

该 kernel 用于完成 3.1.1节 第 3 点;
```c++
KERNEL_CHECK((KernelRemoveFirstForward(
      k_dim, k_type, handle->queue, num_points, voxel_coors,
      point2voxel_map, voxel_points_count)));
```

- kernel4: kernelDynamicPointToVoxelForward

该 kernel 用于完成 3.1.1节 第 4 点;

```c++
//reduce
  const int num_feats = feats_desc->dims[1];
  KERNEL_CHECK((KernelDynamicPointToVoxelForward(k_dim, k_type, handle->queue, reduce_type,
                                                 feats, num_points, num_feats, num_voxels,
                                                 point2voxel_map, voxel_points_count, voxel_feats)));
```

由于 unique kernel 是仓库已有算子，本设计文档不对其进行过多描述，下文主要针对 1、3、4的 kernel 在设备端的实现方案进行详细描述。

- #### device 端

- kernel1

  该 kernel 主要对 `num_points` 进行任务拆分，然后进行三级排流水，主要写下 compute 实现逻辑, 其步骤如下：

  step1: __bang_transpose() 将 `coors:[N, 3]` 转为 `coors:[3, N]`;

  step2: __bang_int322float() 对 `coors` 进行数据类型转换; 

  step3: __bang_gt_scalar() 对 `coors:[3, N]` 中 N 个数据于 0 比较得到三个 `mask_x:[N]`、`mask_y:[N]`、`mask_z:[N]`;

  step4: __bang_and() 对三个 mask 求并得到 `mask_x:[N]`;

  step5: __bang_cycle_mul() 对 `coors:[3, N]` 和 `mask_x:[N]` 进行计算;

  step6: `mask_x:[N]` 进行__bang_sub() 减 1 操作更新 `mask_x:[N]`;

  step7: __bang_cycle_add() 对 `coors:[3, N]` 和 `mask_x:[N]` 进行计算;

  step8: __bang_transpose()将 `coors:[3, N]` 转为 `coors:[N, 3]` 保存;

- kernel3

  step1: 判断 `voxel_coors[0][0]` 的值是否为 -1，如果是进行 step2, 如果不是 直接退出 kernel;

  step2: 删除 `voxel_coors` 中前 3 个值(-1，-1，-1), 采用 g2g 的拷贝方式;

  step3: 删除 `voxel_points_count` 中第 1 个值, 采用 g2g 的拷贝方式；

  step4: 对 `point2voxel_map` 中各值进行三级排流水，将每个值减去 1;

- kernel4

  该 kernel 内部主要分为两个部分进行实现功能:
  
- 第一部分

  该部分主要对具有相同体素坐标的 `feats` 在特征维度上进行求 `max` 和 `add` 操作，具体步骤如下: 

  step1: 对 `num_points` 进行任务拆分，计算每个 core 平均处理的点数量 `points_per_core` 和开始索引 `points_offset`;

  step2: nram空间划分，先计算 `deal_p` (nram 可以划分的份数)和 `deal_h` (每份的数据量)的大小; 当 deal_p = 1, deal_h = max_deal_h 时，  nram 空间只用于保存 max_deal_h 个 float 类型的数据，否者，nram 空间将划分 deal_h 个 float 类型数据，deal_p 份; 

  step3: 根据 `points_per_core` 和 `deal_p` 计算 `repeat_p` (单 core 需要循环处理的次数)和 `rem_p`(单 core 最后一次要处理的点数);

  step4: 根据 `num_feats` 和 `deal_h` 计算 `repeat_h` (单点可一次处理的特征个数)和 `rem_h`(单点最后一次被处理的特征个数);

  step5: 循环处理, 对每份数据进行 __bang_atomic_reduce_max() 和 __bang_atomic_reduce_add() 处理;

- 第二部分

  该部分主要对 `reduce_type = mean` 时， 在 `voxel_feats` 每个特征维度上做 `div`操作，求 `mean`，具体步骤如下: 

  step1: 对 `num_voxel` 进行任务拆分，计算每个 core 平均处理的点数量 `points_per_core` 和开始索引 `points_offset`;

  step2: nram空间划分，将 nram 分为三份，一份存储需要加载的 `voxel_points_count`, 另外两份存储 `voxel_feats` 的 ping 和 pong, 用于流水操作, 再计算 `deal_v` (nram 可以划分的份数)和 `deal_h` (每份的数据量)的大小;

  step3: 根据 `points_per_core` 和 `deal_v` 计算 `repeat_v` (单 core 需要循环处理的次数)和 `rem_v`(单 core 最后一次要处理的点数);

  step4: 根据 `num_feats` 和 `deal_h` 计算 `repeat_h` (单点可一次处理的特征个数)和 `rem_h`(单点最后一次被处理的特征个数);

  step5: 对 `voxel_feats` 的特征维度 num_feats 进行循环处理, 采用三级流水LCS

  说明：该部分代码可以通过 div 算子进行拼接完成，作为后期的优化点。

### 3.2 伪代码实现（可选）

- kernel4 

```c++
int remainder = num_points % taskDim;
int points_per_core = num_points / taskDim + (int)(taskId < remainder);
// offset of the point that core processes
int points_offset = taskId * (num_points / taskDim) + (taskId < remainder ? taskId : remainder);
// nram space
// |feats|
int max_deal_h = (MAX_NRAM_SIZE / sizeof(float));
int deal_h = 0;
int deal_p = 0;
if(num_feats > max_deal_h){
  deal_p = 1;
  deal_h = max_deal_h;
} else{
  deal_h = num_feats;
  deal_p = (MAX_NRAM_SIZE / (deal_h * sizeof(float)))
}
float *nram_feats = (float *)nram_buffer;
float *base_feats = feats + points_offset * num_feats;
int repeat_p = points_per_core / deal_p;
int rem_p = points_per_core % deal_p;
int repeat_h = num_feats / deal_h;
int rem_h = num_feats % deal_h;

for (int32_t p_iter = 0; p_iter < repeat_p + 1; p_iter++){
  int32_t deal_p_num = (p_iter < repeat_p) ? deal_p : rem_p;
  if (deal_p_num == 0) {
    break;
  }
  int32_t deal_p_num_offset = p_iter * deal_p * num_feats;
  for(int32_t h_iter = 0; h_iter < repeat_h + 1; h_iter++){
    int32_t deal_h_num = (h_iter < repeat_h) ? deal_h : rem_h;
    if (deal_h_num == 0) {
        break;
    }
    int32_t deal_h_num_offset = deal_p_num_offset + h_iter * deal_p * deal_h
    float *base_feats_addr = base_feats + deal_h_num_offset;
    // load
    __memcpy(nram_feats, base_feats_addr, deal_p_num * deal_h_num* sizeof(float), GDRAM2NRAM);
    // index and atomic
    for (int32_t i = 0 ; i < deal_p_num; i++){
        int32_t point_idx = points_offset + p_iter * deal_p + i;
        int32_t reduce_to = point2voxel_map[point_idx];
        int32_t count = 
        float * voxel_feats_offset = voxel_feats + reduce_to * num_feats + h_iter * deal_h;
        if (reduce_mode == REDUCE_MAX){
          __bang_atomic_reduce_max();
        } else{
          __bang_atomic_reduce_add();
        }
      }
  }
}
if (reduce_mode == REDUCE_MEAN){
  int remainder = num_voxel % taskDim;
  int points_per_core = num_voxel / taskDim + (int)(taskId < remainder);
  // offset of the point that core processes
  int points_offset = taskId * (num_voxel / taskDim) + (taskId < remainder ? taskId : remainder);
  // nram space
  // |voxel_points_count|
  // |voxel_feats_ping|voxel_feats_pong|
  int max_deal_h = (MAX_NRAM_SIZE - sizeof(int32_t)) / (2 * sizeof(float));
  int deal_h = 0;
  int deal_v = 0;
  if(num_feats > max_deal_h){
    deal_v = 1;
    deal_h = max_deal_h;
  } else{
    deal_h = num_feats;
    deal_v = (MAX_NRAM_SIZE - 2 * deal_h * sizeof(float)) / (sizeof(int32_t));
  }

  int *nram_points_count = (int *)nram_buffer;
  float *voxel_feats_ping = (float *)(nram_points_count + deal_v);
  float *voxel_feats_pong = voxel_feats_ping + deal_h;
  int *base_points_count = (int *)voxel_points_count + points_offset;
  float *base_voxel_feats = (float *)voxel_feats + points_offset * num_feats;
  int repeat_v = points_per_core / deal_v;
  int rem_v = points_per_core % deal_v;
  int repeat_h = num_feats / deal_h;
  int rem_h = num_feats % deal_h;
  for(int v_iter = 0; v_iter <= repeat_v; v_iter++){
    int deal_v_num = (v_iter < repeat_v) ? deal_v : rem_v;
    if (deal_v_num == 0) {
      break;
    }
    float * base_voxel_feats_addr = base_voxel_feats + v_iter * deal_v * num_feats;
    int * base_points_count_addr = base_points_count + v_iter * deal_v;
    __memcpy(nram_points_count, base_points_count_addr, deal_v_num * sizeof(int), GDRAM2NRAM);
    if (num_feats <= max_deal_h) {
      // L(vi=0)
      if (deal_v_num > 0) {
        load();
        __sync();
      }
      if (deal_v_num > 1) {
        // L(vi=1)
        load();
        // C(vi=0)
        compute();
        __sync();
      }
      for (int vi = 0; vi < deal_v_num - 2; vi++) {
        // S(vi)
        store();
        // C(vi+1)
        compute();
        // L(vi+2)
        load();
        __sync();
      }

      if (deal_v_num > 1) {
        // S(vi = deal_v_num - 2)
        store();
        __sync();
      }
      if (deal_v_num > 0) {
        // C[deal_v_num - 1]
        compute();
      }
      __sync();
      if (deal_v_num > 0) {
        // S[deal_v_num - 1]
        store();
      }
    }else {
      // vi = points_offset + v_iter
      lcs();
        }
      }
    
  }
```
### 3.3 拆分(任务拆分，多核拆分)

- 基本任务类型为UNION1的任务。

- 多核拆

由 3.1.2小节知, 主要对两个 kernel 进行拆分：

- kernel1:

输入`coors:[num_points, 3]`，因该 kernel 主要是将无效的体素坐标填充 -1，因此拆分`num_points`，将`num_points`平均拆分到所有task上处理。

- kernel3:

输入`point2voxel_map:[num_points, 1]`，因该 kernel 主要是将所有值减 1 ，因此拆分`num_points`，将`num_points`平均拆分到所有task上处理。

- kernel4:

第一部分：

输入`feats:[num_points, num_feats]`，因该 kernel 主要是对具有相同体素的点去重，因此拆分`num_points`，将`num_points`平均拆分到所有task上处理。

第二部分：

输入`vocel_feats:[num_voxels, num_feats]`, `voxel_points_count:[num_voxel]`, 因该 kernel 主要是对输入求均值，因此拆分`num_voxel`，将`num_voxel`平均拆分到所有task上处理。


### 3.4 性能优化设计

- 资源分配

本设计文档主要对 kernelDynamicPointToVoxelForward 中用到的资源进行分配，unique 算子的资源分配可对应参考 unique 算子的设计文档

| 表项            | 分配策略                                                                      |
| --------------- | ----------------------------------------------------------------------------- |
| NRAM            | 参考 3.1.2节                                                                  |
| WRAM            | 未使用                                                                        |
| SRAM            | 未使用                                                                        |
| DRAM(workspace) | 未使用，只在依赖算子 unique 中需要（不作为主要讨论内容）                      |

- 流水设计

在 kernel1、kernel3、kernel4 中都采用了三级流水设计，L C S之间排流水，即 GDRAM2NRAM、Compute、NRAM2GDRAM.

- 优化设计

1）将 kernel3 和 kernel4 融合，减少 launch kernel 的时间开销;

2）kernel4 的最后一步求均值，可以采用 div 算子进行;(后期仓库 div 算子支持广播计算再进行优化)。

### 3.5 可维护性设计

1、bangc 代码中加入必要的 log 信息，比如输入的规模、数据类型、layout 这些，以及如果出错会导致程序 core dump 的变量，比如 IO 指令的 data_size、dim xyz 的值等，这些信息都是有利于快速定位问题；

2、对每一个函数命名变量命名都有充分的注释；

3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代。

### 3.6 测试用例设计

算子在网络中用到的规模：
- #### case1
input:

feats: torch.Size([17563,4]) fp32

coors: torch.Size([17563,3]) int32

reduce_type: 'mean'

output:

voxel_feats: torch.Size([13757, 4]) fp32

voxels_coors: torch.Size([13757, 3]) fp32

point2voxel_map: torch.Size([17563]) int32

voxel_points_count: torch.Size([13757]) int32

voxel_num: torch.Size([1]) int32

- #### case2
input:
feats: torch.Size([17563,64]) fp32

coors: torch.Size([17563,3]) int32

reduce_type: 'max'

output:
voxel_feats: torch.Size([13757, 64]) fp32

voxels_coors: torch.Size([13757, 3]) int32

point2voxel_map: torch.Size([17563]) int32

voxel_points_count: torch.Size([13757]) int32

voxel_num: torch.Size([1]) int32

其他可根据需要进行补充。算子开发完毕后，补充测试报告链接。

### 3.7 算子防呆检查

- 描述符指针为空防呆：handle、feats_desc、coors_desc、voxel_feats_desc、voxels_coors_desc、point2voxel_map_desc、
                      voxel_points_count_desc、voxel_num_desc；
- 对输入输出支持的 dtype、layout 以及 shape 进行防呆
  1. dtype防呆：feats_desc、voxel_feats_desc：仅支持float；
  2. dtype防呆：coors_desc、voxels_coors_desc、point2voxel_map_desc、voxel_points_count_desc、voxel_num_desc：仅支持int；
  3. dim防呆：
     1. feats_desc、coors_desc、point2voxel_map_desc的第一个维度大小相等且等于 num_points;
     2. voxel_feats_desc、voxels_coors_desc、voxel_points_count_des的第一个维度大小相等且等于 num_voxels;
     3. feats_desc、voxel_feats_desc的第2个维度大小相等且等于 num_feats;
     4. coors_desc、voxels_coors_desc的第2个维度大小相等且等于 3;
     5. num_points >= num_feats;
     6. voxel_num_desc 的维度为 1;
- 0 元素检查防呆：返回 MLUOP_STATUS_SUCCESS;
- 指针为空防呆：对feats、coors、voxel_feats、voxels_coors、point2voxel_map、voxel_points_count、voxel_num指针为空防呆检查;
- large tensor防呆 ：对feats、coors的检查;

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

### 4.2 已经过优化的规模说明
