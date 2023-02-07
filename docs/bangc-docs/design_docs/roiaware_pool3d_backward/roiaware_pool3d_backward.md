# roiaware_pool3d_backward 算子开发设计方案

- #### 文档基本信息

| 算子名称    | roiaware_pool3d_backward |
| ----------- | ------------------------ |
| 编制人/日期 | 张皓喆/2022-8-10         |
| 审批人/日期 | 张少鹏/2022-8-15         |
| 审批人/日期 | 王远/2022-8-15           |
| 审批人/日期 | 卜德飞/2022-8-15         |

- #### 修改记录

| 版本号 | 修订人 | 修订日期  | 修订描述 |
| ------ | ------ | --------- | -------- |
| V1.0   | 张皓喆 | 2022-8-10 | 首次提交 |

- #### 内容描述

本文档为`roiaware_pool3d_backward`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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

| 算子功能简介                                                                 | 本算子为 roiaware_pool3d_forward 的反向算子，输入体素中的 idx 以及前向的池化特征值，计算反向梯度值                                                                                                     |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 需求来源                                                                     | mmcv/cuda 自定义算子                                                                                                                                                                                   |
| 应用网络                                                                     | PartA2                                                                                                                                                                                                 |
| 输入数据类型                                                                 | `pts_idx_of_voxels` 、 `argmax` 为 int32 类型， `grad_out` 为 float/half                                                                                                                               |
| 输入 Shape                                                                   | `pts_idx_of_voxels` : [boxes_num, out_x, out_y, out_z, max_pts_each_voxel] <br> `argmax` : [boxes_num, out_x, out_y, out_z, channels] <br> `grad_out`: [boxes_num, out_x, out_y, out_z, channels] <br> |
| 输入 Layout                                                                  | 均为 ARRAY                                                                                                                                                                                             |
| 输出数据类型                                                                 | `grad_in` 为 float/half                                                                                                                                                                                |
| 输出 Shape                                                                   | `grad_in` : [pts_num, channels]                                                                                                                                                                        |
| 输出 Layout                                                                  | 均为 ARRAY                                                                                                                                                                                             |
| 是否含有 dim/axis 等类似语义的参数且该参数支持负数/其他特殊处理              | 否                                                                                                                                                                                                     |
| 是否含有 labels/index 等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 否                                                                                                                                                                                                     |
| 是否需要支持原位                                                             | 否                                                                                                                                                                                                     |
| 是否需要支持 stride 机制                                                     | 否                                                                                                                                                                                                     |
| 是否需要支持广播                                                             | 否                                                                                                                                                                                                     |
| 0 元素检查是否直接返回                                                       | 是，返回 MLUOP_STATUS_BAD_PARAM                                                                                                                                                                        |
| 其他特殊需求(在线量化，融合，转数提前等，可选)                               | 无                                                                                                                                                                                                     |

### 1.2 算子功能和应用场景描述

- 算子功能：

该算子属于 MMCV 中 3D 目标检测领域算子，应用于 PartA2 网络，是 roiaware_pool3d_forward 算子的反向实现，根据给定的每一个 box 的每一个体素和每一个通道上对应的点的索引，再根据点的索引值，找到给定的对应位置的特征值，计算反向特征值，并根据指定的最大或者平均模式，输出每一个点在每一个通道上的特征值。

- 说明：

1. 输入 `pts_idx_of_voxels` 和 `argmax` 对应前向算子的输出， `grad_out` 对应前向算子的输出 `pooled_feature`的梯度

2. 输出 `grad_in` 对应前向算子的输入 `pts_feature`的梯度

### 1.3 算子输入输出参数要求

| 参数                   | 语义                                                                                | 类型（输入/输出） | 支持类型                | 物理布局 | 规模限制                                         |
| ---------------------- | ----------------------------------------------------------------------------------- | ----------------- | ----------------------- | -------- | ------------------------------------------------ |
| handle                 | 句柄，用于获取当前资源                                                              | 输入              | mluOpHandle_t           | /        | /                                                |
| pool_method            | 指定进行池化计算的模式方式，默认为 maxpool                                          | 输入              | int32                   | /        | /                                                |
| boxes_num              | 所需要进行池化的 3dbox 的数量                                                       | 输入              | int32                   | /        | /                                                |
| out_x                  | 池化 3dbox 的体素在 x 维度上的数量                                                  | 输入              | int32                   | /        | /                                                |
| out_y                  | 池化 3dbox 的体素在 y 维度上的数量                                                  | 输入              | int32                   | /        | /                                                |
| out_z                  | 池化 3dbox 的体素在 z 维度上的数量                                                  | 输入              | int32                   | /        | /                                                |
| channels               | 点的特征的通道数量                                                                  | 输入              | int32                   | /        | /                                                |
| max_pts_each_voxel     | 池化 3dbox 中每个体素，所需要覆盖到的点的最大数量                                   | 输入              | int32                   | /        | /                                                |
| pts_idx_of_voxels_desc | 对 `pts_idx_of_voxels` 的 tensor 描述符                                             | 输入              | mluOpTensorDescriptor_t | /        | /                                                |
| pts_idx_of_voxels      | 在 box 中池化后的 box 中的点的索引                                                  | 输入              | int32                   | ARRAY    | [boxes_num,out_x,out_y,out_z,max_pts_each_voxel] |
| argmax_desc            | 对 `argmax` 的 tensor 描述符                                                        | 输入              | mluOpTensorDescriptor_t | /        | /                                                |
| argmax                 | 取得最大特征值的点的索引                                                            | 输入              | int32                   | ARRAY    | [boxes_num,out_x,out_y,out_z,channels]           |
| grad_out_desc          | 对 `grad_out` 的 tensor 描述符                                                      | 输入              | mluOpTensorDescriptor_t | /        | /                                                |
| grad_out               | 在每一个 box 池化之后的每一个体素中的点的特征，对应前向算子的 pooled_feature 的梯度 | 输入              | float/half              | ARRAY    | [boxes_num,out_x,out_y,out_z,channels]           |
| grad_in_desc           | 对 `grad_in` 的 tensor 描述符                                                       | 输入              | mluOpTensorDescriptor_t | /        | /                                                |
| grad_in                | 输出的每一个点的特征，对应前向算子的 pts_feature 的梯度                             | 输出              | float/half              | ARRAY    | [pts_num,channels]                               |

### 1.4 算子限制

| 限制类型     | 详细说明                                                                                                                                     |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 布局限制     | 仅支持 <1.3 算子输入输出参数要求> 中规模限制所描述的 shape 布局                                                                              |
| 池化方式限制 | 仅支持 `pool_method` = 0 的最大池化和 `pool_method` = 1 的平均池化方式                                                                       |
| 原位限制     | 不支持原位                                                                                                                                   |
| stride 限制  | 不支持 stride 机制                                                                                                                           |
| 广播限制     | 不支持广播                                                                                                                                   |
| 数据范围限制 | `pts_idx_of_voxels` 和 `argmax` 中点的索引 index 的数值需满足输出 `grad_in` 的所描述点的数量的数值范围，需要在[0, `pts_num` - 1]             |
| 数据范围限制 | `pts_idx_of_voxels` 和 `argmax` 中不支持含有 INF、NAN 的数值                                                                                 |
| 数据范围限制 | `pts_idx_of_voxels` 在最后一个维度`max_pts_each_voxel`中的第一个值，表示处于当前体素内的点的个数，该数值需要满足范围 [0, max_pts_each_voxel] |
| 数据范围限制 | 由于 NRAM 内存限制，`max_pts_each_voxel` 数值不能超过 2048，否则可能会发生内存踩踏导致 coredump                                              |

说明：需要注意，由于本算子与 mmcv 的 cuda 实现均采用 atomicAdd，存在乱序行为，当输出 `grad_in` 中的计算结果含有 INF 或者 NAN 时，计算结果可能与 mmcv cuda 不对齐。

### 1.5 验收标准

#### 1.5.1 精度验收标准

- 按照[精度验收标准](../MLU-OPS-Accuracy-Acceptance-Standard.md)的要求明确本算子的精度标准。：

  1. 输出 `grad_in` 适配动态阈值，在 MLU300 上采用动态阈值标准：diffs = [diff1, diff2, diff4]，threshold_rate = [10, 10, 1]。

#### 1.5.2 性能验收标准

- [MLU-OPS 性能验收标准](../MLU-OPS-Performance-Acceptance-Standard.md)

- 性能分析：

| 平台                 | 框架版本                                                 | 数据类型 | 数据规模                                                                                                                                       | cuda kernel                                                 | 计算效率          | IO 效率           | Hardware time(ms)     |
| -------------------- | -------------------------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------- | ----------------- | --------------------- |
| Tesla V100-SXM2 16GB | mmcv-1.3.16 + pytorch-1.9 + cuda-11.4 + Driver-440.64.00 | float    | pts_idx_of_voxels: [128, 12, 12, 12, 128] <br> argmax: [128, 12, 12, 12, 16] <br> grad_out: [128, 12, 12, 12, 16]<br> grad_in: [16000, 16]<br> | roiaware_maxpool3d_backward<br> roiaware_avgpool3d_backward | 38.24%<br> 27.21% | 34.80%<br> 20.79% | 74.46us<br> 157.86us  |
| Tesla V100-SXM2 16GB | mmcv-1.3.16 + pytorch-1.9 + cuda-11.4 + Driver-440.64.00 | half     | pts_idx_of_voxels: [128, 12, 12, 12, 128] <br> argmax: [128, 12, 12, 12, 16] <br> grad_out: [128, 12, 12, 12, 16]<br> grad_in: [16000, 16]<br> | roiaware_maxpool3d_backward<br> roiaware_avgpool3d_backward | 29.43%<br> 24.90% | 21.13%<br> 9.80%  | 110.40us<br> 238.53us |

## 2 算子接口设计

### 2.1 参考接口

- MMCV 接口

```c++
// CUDA(mmcv/ops/csrc/pytorch/cuda/roiaware_pool3d_cuda_kernel.cuh):
void RoiawarePool3dBackwardCUDAKernelLauncher(
    int boxes_num, int out_x, int out_y, int out_z, int channels,
    int max_pts_each_voxel, const Tensor pts_idx_of_voxels, const Tensor argmax,
    const Tensor grad_out, Tensor grad_in, int pool_method);
```

### 2.2 接口设计

```c++
// 给出MLUOP算子正向接口
mluOpStatus_t MLUOP_WIN_API
mluOpRoiawarePool3dBackward(mluOpHandle_t handle,
                            const int pool_method,
                            const int boxes_num,
                            const int out_x,
                            const int out_y,
                            const int out_z,
                            const int channels,
                            const int max_pts_each_voxel,
                            const mluOpTensorDescriptor_t pts_idx_of_voxels_desc,
                            const void *pts_idx_of_voxels,
                            const mluOpTensorDescriptor_t argmax_desc,
                            const void *argmax,
                            const mluOpTensorDescriptor_t grad_out_desc,
                            const void *grad_out,
                            const mluOpTensorDescriptor_t grad_in_desc,
                            void *grad_in)
```

## 3 实现方案设计

### 3.1 实现方案

- roiaware_pool3d_backward 算子 MLU 实现方案：

  0. 根据 `pool_method` 数值确定池化方式，选择不同的 kernel 并 launch，即最大池化选择 KernelRoiawareMaxPool3dBackward(以下称为 KernelMax )，平均池化选择 KernelRoiawareAvgPool3dBackward(以下称为 KernelAvg )，KernelMax 与 KernelAvg 的多核拆分方案是同样的

  1. 将所有 `boxes_num` 个 box 分别划分为`out_x` _ `out_y` _ `out_z`个体素，体素总数共有 voxels*nums = `boxes_num` * `out_x` \_ `out_y` \* `out_z` ，将所有的体素划分以 taskDim 为间隔的若干份，每个 core 将依次跳跃间隔 taskDim 个体素循环处理，根据体素索引 voxels_idx 计算 gdram 上的数据偏移，每一个 core 负责处理其所分到的 voxels；

  2. 当池化方式为最大池化时，执行 3 步骤，当池化方式为平均池化时，执行第 5 步；

  3. 依次 循环 load 当前 voxel_idx 对应的 `argmax` 的 nram_channels_limit 个点的索引，nram_channels_limit 表示一次性最多能处理多少个 channel 元素，执行第 4 步；

  4. 循环获取当前 load 到 nram 的 `channels` 中得到的点的 idx，计算输出地址偏移之后，使用 \_\_bang_atomic_reduce_add() 从 gdram 上取出当前 `grad_in` 的点的累加特征值，加上 `grad_out` 在当前 channel 的特征值后，再存回对应的 `grad_in` 的点的 channel 的位置，执行第 7 步；

  5. 依次循环 Load 当前体素中的点的特征，一次最多处理的 nram_channels_limit 个元素，执行第 6 步；

  6. 当前 core 中 Load 当前体素中点的 idx ，依次循环处理每一个点，使用 \_\_bang_atomic_reduce_add() 依次取出当前点的累加特征值，加上 `grad_out` 的特征值后，再存回对应的点的索引在 `grad_in`中对应 channel 的位置，执行第 7 步；

  7. 当前 voxel 计算完毕，返回 step2 处理下一个 voxels，直至当前 core 处理的索引达到最大数值 voxels_nums 的截止条件，之后当前 core 的计算结束。

### 3.2 伪代码实现（可选）

- 最大池化方式的伪代码：

```c++
  // nram_channels_limit 表示最多能放下 channel 维度的元素个数
  int channels_loop_times = channels / nram_channels_limit;
  // step.1
  for (int voxel_index = taskId; voxel_index < boxes_num * out_x * out_y * out_z; voxel_index += taskDim) {
    // step.2
    for (int channels_loop_idx = 0; channels_loop_idx <= channels_loop_times; channels_loop_idx++) {
      // LOAD argmax and grad_out GDRAM2NRAM
      __memcpy(argmax);
      __memcpy(grad_out);
      for (int channel_idx = 0; channel_idx < actual_channels_num; channel_idx++) {
        // trick : fast return
        if (nram_argmax_cur_channel[0] == -1) { continue; }
        // step.3
        T *grad_in_cur_channel = grad_in +
                                 nram_argmax_cur_channel[0] * channels +
                                 nram_channels_limit * channels_loop_idx +
                                 channel_idx;
        // STORE grad_in use atomic_add in GDRAM
        __bang_atomic_reduce_add((T *)nram_grad_in_cur_channel,
                                 (T *)grad_in_cur_channel,
                                 (T)(nram_grad_out_cur_channel[0] * 1));
      }
    }
  }
}
```

- 平均池化方式的伪代码：

```c++
  // nram_channels_limit 表示最多能放下 channel 维度的元素个数
  int channels_loop_times = channels / nram_channels_limit;
  // step.1
  for (int voxel_index = taskId; voxel_index < boxes_num * out_x * out_y * out_z; voxel_index += taskDim) {
    // LOAD pts_idx_cur_voxel GDRAM2NRAM
    __memcpy();
    int total_pts = nram_pts_idx_cur_voxel[0];
    // trick : fast return
    if (total_pts <= 0) {continue;}
    float cur_grad = 1.0 / ((float)total_pts);
    // step.2
    for (int channels_loop_idx = 0; channels_loop_idx <= channels_loop_times; channels_loop_idx++) {
        // LOAD grad_out GDRAM2NRAM
        __memcpy();
        __bang_mul_scalar(grad_out_cur_loop, cur_grad);
      // step.3
      for (int k = 1; k <= total_pts; k++) {
        T *grad_in_cur_loop = grad_in +
                              nram_pts_idx_cur_voxel[k] * channels +
                              nram_channels_limit * channels_loop_idx;
        // STORE grad_in use atomic_add in GDRAM
        __bang_atomic_reduce_add((T *)nram_grad_in_cur_channel,
                                 (T *)grad_in_cur_loop,
                                 (T *)nram_grad_out_cur_loop,
                                 actual_channels_num);
      }
    }
  }
```

### 3.3 拆分(任务拆分，多核拆分)

**任务类型 U1：**

当前实现根据不同模式，区分 2 个 kernel，进行拆分，平分到各个 core 中：

1. MLUUnion1KernelRoiawareMaxPool3dBackward 中输入 `argmax` 规模为 `[boxes_num, out_x, out_y, out_z, channels]`，多核拆分在 `boxes_num` _ `out_x` _ `out_y` \* `out_z` 维度上拆分；

2. MLUUnion1KernelRoiawareAvgPool3dBackward 中输入 `pts_idx_of_voxels` 规模为 `[boxes_num, out_x, out_y, out_z, max_pts_each_voxel]`，多核拆分在 `boxes_num` _ `out_x` _ `out_y` \* `out_z` 维度上拆分；

### 3.4 性能优化设计

1. 资源分配

1.1 MLUUnion1KernelRoiawareMaxPool3dBackward 中的空间划分如下

| 地址命名                 | 元素数量            | 数据类型   |
| ------------------------ | ------------------- | ---------- |
| nram_argmax_cur_loop     | nram_channels_limit | int32      |
| nram_grad_out_cur_loop   | nram_channels_limit | half/float |
| nram_grad_in_cur_channel | 1                   | half/float |

其中: </br> nram_channels_limit 为一次最多能放入 NRAM 的元素数量；</br>

1.2 MLUUnion1KernelRoiawareAvgPool3dBackward 中的空间划分如下

| 地址命名               | 元素数量                 | 数据类型   |
| ---------------------- | ------------------------ | ---------- |
| nram_pts_idx_cur_voxel | align_max_pts_each_voxel | int32      |
| nram_grad_out_cur_loop | nram_channels_limit      | half/float |
| nram_grad_in_cur_loop  | nram_channels_limit      | half/float |

其中: </br>
align_max_pts_each_voxel 为 `max_pts_each_voxel` 与 align_num 向上对齐后的元素数量；</br>
nram_channels_limit 为一次最多能放入 NRAM 的元素数量；</br>

1.3 暂不需要申请 workspace size

2. 流水设计

本算子采用 atomic_add 指令，属于 IO 瓶颈算子，不进行排流水。

### 3.5 可维护性设计

1、bangc 代码中加入必要的 log 信息，比如输入的规模 data_size、数据类型 dtype、维度 dim 等，以及如果出错会导致程序 core dump 的变量；

2、对每一个函数命名变量命名都有充分的注释；

3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代 (宏的调用说明以及含义已经注释写在 kernels 代码中)。

### 3.6 测试用例设计

- 框架在需求列表中给出的算子在网络中用到的规模(无，框架没提供)：
- partA2 网络规模： <br>
  输入： pts_idx_of_voxels(128, 12, 12, 12, 128) argmax(128, 12, 12, 12, 16) grad_out(128, 12, 12, 12, 16) <br>
  输出： grad_in(16000, 16)

- 边界 case：

1. 根据 job 类型, 1 mlu core(Block), 1 cluster, 4 cluster, 6 cluster, 8 cluster, 16 cluster, with/without remainder；
2. 根据 NRAM 空间计算单次处理的最大数量 m，小于 m, 大于 m，等于 m；
3. 根据各个维度是否对齐。

### 3.7 算子防呆检查

1. 检查 handle 以及所有输入的 tensor desc 是否为空；
2. 检查所有输入的 tensor desc 的 dim；
3. 检查所有输入的 tensor desc 的 shape，判断是否满足与其他输入参数的关联性；
4. 检查所有输入的 tensor desc 的 datatype，判断是否满足与其他输入参数的关联性；
5. 其他输入参数的防呆，如 pool_mode 仅能为 0、1；
6. 限制的防呆，比如算子限制 kernel 不能太大；
7. 检查 0 元素 （部分算子零元素返回 error 而不是 success，具体行为要和原生框架对齐） （零元素放在除了输入输出空间指针非空检查外，其余所有检查的后面）；
8. 检查 large tensor，tensor 规模限制在 2G；
9. 对于有 workspace 的算子：在 workspace_size > 0 且 workspace 指针为空时返回 error （默认用户传入的 workspace_size 大小满足需求，算子内无需使用 getOPWorkspaceSize() <= workspace_size 语句判断）；
10. 检查所有输入、输出的空间指针 input_ptr, output_ptr 是否为空。

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

无

### 4.2 已经过优化的规模说明

无

## 5 方案实施

### 5.1 开发测试计划

- 2022.08.01 ~ 2022.08.03 调研源码、开始设计方案
- 2022.08.04 ~ 2022.08.10 设计方案、伪代码
- 2022.08.11 ~ 2022.08.15 方案 review、generator、gtest 代码开发
- 2022.08.15 ~ 2022.08.19 算子主体框架开发
- 2022.08.22 ~ 2022.08.27 算子性能优化 debug
- 2022.08.28 ~ 2022.08.29 大规模测试、测试报告
- 2022.09.01 ~ 2022.09.02 提交 MR、代码 review
- 2022.09.03 ~ 2022.09.05 修改测试、算子入库

### 5.2 风险分析

无。
