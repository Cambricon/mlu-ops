# dynamic_scatter_backward 算子开发设计方案

- #### 文档基本信息

| 算子名称    | `dynamic_scatter_backward`          |
| ----------- | --------------------- |
| 编制人/日期 | yangdian/2023          |
| 审批人/日期 | 袁梦，张双林/2023   |

- #### 修改记录

| 版本号 | 修订人  | 修订日期  | 修订描述 |
| ------ | ------- | --------- | -------- |
| V1.0   | yangdian | 2023 | 首次提交 |

- #### 内容描述

本文档为`dynamic_scatter_backward`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

- #### 算子需求 checklist

- 算子接口描述
- 功能描述
- 框架版本 + 对应源码路径
- 需求对应网络
- 网络中用到的规模
- 是否需要支持原位
- 是否需要支持 stride 机制
- 框架单元测试阈值指标（可选）

## 1 需求分析

### 1.1 算子需求分析
  
| 算子功能简介               | 简要填写算子功能，详细描述在 1.2 中进行说明                           |
| ------------------------ | -------------------------------------------------------------  |
| 需求来源                  | mmcv                                                           |
| 应用网络                  | mvxnet                                                         |
| 输入数据类型               | float/int32                                                    |
| 输入 Shape                | grad_reduced_feats:[M,C]; feats:[N,C]; reduced_feats:[M,C]; coors_map:[N];
                             reduce_count:[M]; reduce_type: reduce_t              |
| 输出数据类型               | float                                                          |
| 输出 Shape                | grad_feats:[N,C];                                                |
| 是否需要支持原位            | 是: grad_feats                                                  |
| 是否需要支持 stride 机制    | 否                                                              |
| 是否需要支持广播            | 否                                                              |
| 0 元素检查是否直接返回       | 是                                                              |

### 1.2 算子功能和应用场景描述

根据grad_reduced_feats，feats以及dynamic_scatter_forkward输出的reduced_feats，coors_map，reduce_count求出输入相应的梯度

### 1.3 算子输入输出参数要求

| 参数                    | 语义                               | 类型（输入/输出） | 支持类型                 | 物理布局        | 规模限制 |
| -----------            | ---------------------------------- | -------------- | ----------------------- | ------------- | -------- |
| handle                 | 操作句柄                            | 输入            | mluOpHandle_t           | /             | /        |
| grad_reduced_feats_desc| 输入数据，grad_reduced_feats的描述符  | 输入            | mluOpTensorDescriptor_t | /             | /        |
| grad_reduced_feats     | 输入数据，grad_reduced_feats 的坐标   | 输入            | float\*                 | [M,C]| /      |          |
| feats_desc             | 输入数据，feats 的描述符              | 输入            | mluOpTensorDescriptor_t | /             | /        |
| feats                  | 输入数据，feats 的大小                | 输入            | float\*                 | [N,C]         | /        |
| reduced_feats_desc     | 输入数据，reduced_feats 的描述符      | 输入            |mluOpTensorDescriptor_t  | /              | /        |
| reduced_feats          | 输入数据，reduced_feats 的大小        | 输入            | float\*                 | [M,C]         | /        |
| coors_map_desc         | 输入数据，coors_map 的描述符          | 输入            | mluOpTensorDescriptor_t | /             | /        |
| coors_map              | 输入数据，coors_map 的大小            | 输入            | int32\*                 | [N]           | /        |
| reduce_count_desc      | 输入数据，reduce_count 的描述符       | 输入            | mluOpTensorDescriptor_t | /             | /        |
| reduce_count           | 输入数据，reduce_count 的大小         | 输入            | int32\*                 | [M]           | /        |
| reduce_size            | 输入参数，reduce_size 的大小          | 输入            | int32\*                 | /             | /        |
| reduce_mode            | 输入参数，规约mode                    | 输入            | mluOpReduceMode_t \*    | /             | /        |
| workspace              | 输入数据，GDRAM上面的辅助空间           | 输入            | void\*                  | /             | /        |
| workspace_size         | 输入数据，辅助空间的大小                | 输入            | size_t\*                | /             | /        |
| grad_feats_desc        | 输出数据，输出 grad_feats 的描述符      | 输出            | mluOpTensorDescriptor_t | /             | /        |
| grad_feats             | 输出数据，输出 grad_feats 的数据       |  输出           | float \*                | [N,C]         | /        |

### 1.4 算子限制

| 限制类型     | 详细说明                                                                           |
| -----------  | -------------------------------------------------------------------------------  |
| 输入限制     | 输入 `feats` 的shape 必须满足要求: boxes[N,C] `reduced_feats`                        |
| 输入限制     | 输入 `fests`, `reduced_feats`不支持输入 nan 或 inf。                                 |
| 输入限制     | 仅支持输入float, coors_map ,reduce_count int32类型 不支持nan与inf                      |
| 输入参数限制  | 仅支持输入reduce_mode值为MLUOP_REDUCEMODE_MAX                                       |
| 输出限制     | 输出 `output`的shape 必须满足: [N，C]                                                |
| 数据类型限制 | 输入 `fests`，`reduced_feats` 输出 `grad_feats` 数据类型保持一致且仅支持float。`coors_map` `reduce_count`仅支持 int32 |
| 原位限制     | 不支持原位                                                                           |
| stride 限制  | 不支持 stride 机制                                                                 |
| 广播限制     | 不支持广播                                                                         |

### 1.5 验收标准

#### 1.5.1 精度验收标准

- 该算子max模式计算梯度所在最小的index，在根据index去scatter value。因此，该算子采用静态阈值，阈值标准：diff3 = 0.

#### 1.5.2 性能验收标准

- 暂无

## 2 算子接口设计

### 2.1 参考接口

- mmcv

```c++
// 给出接口
void DynamicPointToVoxelBackwardCUDAKernelLauncher(
    at::Tensor &grad_feats, const at::Tensor &grad_reduced_feats,
    const at::Tensor &feats, const at::Tensor &reduced_feats,
    const at::Tensor &coors_map, const at::Tensor &reduce_count,
    const reduce_t reduce_type);
```
```c++
// 给出cpu接口
void dynamic_point_to_voxel_backward(torch::Tensor &grad_feats,
                                     const torch::Tensor &grad_reduced_feats,
                                     const torch::Tensor &feats,
                                     const torch::Tensor &reduced_feats,
                                     const torch::Tensor &coors_idx,
                                     const torch::Tensor &reduce_count,
                                     const std::string &reduce_type);
```

### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API 
mluOpDynamicScatterBackward(const mluOpHandle_t handle,
                            const mluOpReduceMode_t reduce_type,
                            const mluOpTensorDescriptor_t grad_voxel_feats_desc,
                            const void *grad_voxel_feats,
                            const mluOpTensorDescriptor_t feats_desc,
                            const void *feats,
                            const mluOpTensorDescriptor_t voxel_feats_desc,
                            const void *voxel_feats,
                            const mluOpTensorDescriptor_t point2voxel_map_desc,
                            const void *point2voxel_map,
                            const mluOpTensorDescriptor_t voxel_points_count_desc,
                            const void *voxel_points_count,
                            const mluOpTensorDescriptor_t voxel_num_desc,
                            void *voxel_num,
                            void *workspace,
                            const size_t workspace_size,
                            const mluOpTensorDescriptor_t gard_feats_desc,
                            void *gard_feats);
```

```c++
mluOpStatus_t MLUOP_WIN_API 
mluOpGetDynamicScatterBackwardWorkspaceSize(const mluOpHandle_t handle,
                                            const mluOpReduceMode_t reduce_type,
                                            const mluOpTensorDescriptor_t feats_desc,
                                            size_t *workspace_size);
```


## 3 实现方案设计

### 3.1 实现方案

max模式有两个kernel
kernel1：
1.生成 0 - x 的 index 向量保存至 index_nram
2.将 index_nram 中每个index连续扩充C倍，保存至 index_mask_nram
3.根据输入的point2voxel_map取出对应的offset向量
4.生成mask1 在point2voxel_map[i] == -1 位置标志为 1
5.生成mask2 在feats[i] != voxel_feats[i]位置标志为 1
6.mask1，mask2做或操作生成mask3，mask3 做非操作生成mask4
7.tmp = mask3 * input_num +  mask4 * (index_mask_nram + i)
8.tmp与相应的 output_gdram 做atomicMin操作

kernel2：
输入的grad_voxel_feats根据kernel1输出的index结果把相应的value copy到对应的位置



### 3.2 伪代码实现（可选）

### 3.3 拆分(任务拆分，多核拆分)

- kernel1根据input_batch均拆到每一个core， kernel2根据voxel_num均拆到每一个core

### 3.4 性能优化设计

-首次提交暂无

### 3.5 方案理论性能

完成上述 3.1，3.2，3.3，3.4 几个步骤之后，基本可以给出一个理论性能，不需要每一个算子都有过于复杂的公式，但是一定要对自己的算子有一个心理的预期，最终实现之后的效率值是多少。

### 3.6 可维护性设计

- bangc 代码中加入必要的 log 信息，比如输入的规模、数据类型、layout 这些，以及如果出错会导致程序 core dump 的变量，比如 IO 指令的 data_size、dim xyz 的值等，这些信息都是有利于快速定位问题。

- 对每一个函数命名变量命名都有充分的注释

- 避免魔鬼数字，对于确定的数字尽量使用公共宏来替代 (宏的调用说明以及含义已经注释写在 kernels 代码中)

### 3.7 测试用例设计

- 框架在需求列表中给出的算子在网络中用到的规模
gard_feats	        [17176,128]	fp32
grad_voxel_feats	[13743, 128]	fp32
feats	            [17176,128]	fp32
voxel_feats	        [13743, 128]	fp32
point2voxel_map	    [17176]	int32
voxel_points_count	[13743]	int32
reduce_type	'max'

### 3.8 算子防呆检查

- 列出算子需要做的防呆，比如

1、指针为空防呆；

2、0 元素检查防呆，VLOG(5)打印信息，是否返回与框架沟通；

3、对输入输出支持的 dtype 以及 shape 进行防呆；

4、算子自身的`reduce_type`参数防呆。

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

### 4.2 已经过优化的规模说明

## 5 方案实施

### 5.1 开发测试计划

### 5.2 风险分析

原子操作存在性能问题，性能可能达不到预期
