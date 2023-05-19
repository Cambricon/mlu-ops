# dynamic_point_to_voxel_backward 算子开发设计方案

- #### 文档基本信息

| 算子名称 | `dynamic_point_to_voxel_backward` |
| ------- | -------------------------------  |
| 编制人/日期 | xuminjie/2023 |
| 审批人/日期 | 袁梦，张双林/2023 |

- #### 修改记录

| 版本号 | 修订人 | 修订日期 | 修订描述 |
| ----- | ----- | ------ | ------- |
| V1.0 | xuminjie | 2023 | 首次提交 |

- #### 内容描述

本文档为`dynamic_point_to_voxel_backward`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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
  
| 算子功能简介 | 找到特征维度上通过`max`方法去重后点的原始点，将体素坐标的梯度，回传给相应点 |
| ---------- | --------------------------------------------------------------- |
| 需求来源 | mmcv |
| 应用网络 | mvxnet |
| 输入数据类型 | grad_voxel_feats:float32 <br> feats:float32 <br> voxel_feats:float32 <br> point2voxel_map:int32 <br> voxel_points_count:int32 <br> voxel_num:int32 |
| 输入标量参数 | reduce_type:枚举类型 |
| 输入Shape | grad_voxel_feats:[M,C] <br> feats:[N,C] <br> voxel_feats:[M,C] <br> point2voxel_map:[N] <br> voxel_points_count:[M] <br> voxel_num:[1] |
| 输入Layout | 不限 |
| 输出数据类型 | grad_feats:float32 |
| 输出Shape | grad_feats:[N,C] |
| 是否需要支持原位 | 否 |
| 是否需要支持stride机制 | 否 |
| 是否需要支持广播 | 否 |
| 0元素检查是否直接返回 | 输入维度M, N, C任意一个为0时，返回 MLUOP_STATUS_SUCCESS |

### 1.2 算子功能和应用场景描述

max模式：根据point2voxel_map，分组找出feats和voxel_feats中值相同的点，从而将grad_voxel_feats中记录的梯度传给grad_feats。

### 1.3 算子输入输出参数要求

| 参数 | 语义 | 类型（输入/输出） | 支持类型 | 物理布局 | 规模限制 |
| --- | ---- | -------------- | ------ | ------- | ------- |
| handle | 操作句柄 | 输入 | mluOpHandle_t | / | / |
| reduce_type | 规约mode | 输入 | mluOpReduceMode_t \* | /  | / |
| grad_voxel_feats_desc | grad_voxel_feats的描述符 | 输入 | mluOpTensorDescriptor_t | / | / |
| grad_voxel_feats | grad_voxel_feats的坐标 | 输入 | float\* | / | [M,C] |
| feats_desc | feats的描述符 | 输入 | mluOpTensorDescriptor_t | / | /  |
| feats | feats的大小 | 输入 | float\* | / | [N,C] |
| voxel_feats_desc | voxel_feats的描述符 | 输入 |mluOpTensorDescriptor_t | / | / |
| voxel_feats | voxel_feats的大小 | 输入 | float\* | / | [M,C] |
| point2voxel_map_desc | point2voxel_map的描述符 | 输入 | mluOpTensorDescriptor_t | / | / |
| point2voxel_map | point2voxel_map的大小 | 输入 | int32\* | / | [N] |
| voxel_points_count_desc | voxel_points_count的描述符 | 输入 | mluOpTensorDescriptor_t | / | / |
| voxel_points_count | voxel_points_count的大小 | 输入 | int32\* | / | [M] |
| voxel_num_desc | voxel_num的描述符 | 输入 | mluOpTensorDescriptor_t | / | [1] |
| voxel_num | voxel_num的大小 | 输入 | int32\* | / | / |
| workspace | GDRAM上面的辅助空间 | 输入 | void\* | / | / |
| workspace_size | 辅助空间的大小 | 输入 | size_t\* | / | / |
| grad_feats_desc | grad_feats的描述符 | 输出 | mluOpTensorDescriptor_t | / | / |
| grad_feats | grad_feats的数据 | 输出 | float\* | / | [N,C] |

### 1.4 算子限制

| 限制类型 | 详细说明 |
| ----------- | ------------------------------------------------------------------------------- |
| 输入限制 | 输入 `grad_voxel_feats`, `feats`, `voxel_feats`支持输入 nan 或 inf |
| 输入参数限制 | 仅支持输入reduce_mode值为MLUOP_REDUCEMODE_MAX |
| 数据类型限制 | 输入 `grad_voxel_feats`, `feats`, `voxel_feats` 输出 `grad_feats` 数据类型保持一致;`point2voxel_map`, `voxel_points_count`, `voxel_num`数据类型保持一致 |
| 布局限制 | 无 |
| 原位限制 | 不支持原位 |
| stride 限制 | 不支持 stride 机制 |
| 广播限制 | 不支持广播 |

### 1.5 验收标准

#### 1.5.1 精度验收标准

按照[MLU-OPS 算子精度验收标准](../../../MLU-OPS-Accuracy-Acceptance-Standard.md)的要求明确本算子的精度标准。

- max模式计算梯度回传的点的最小index，根据index将grad_voxel_feats赋值给grad_feats，可以做到bit级一致
- 算子精度验收标准：diff3;
- 算子精度阈值描述：diff3 = 0;

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../../../MLU-OPS-Performance-Acceptance-Standard.md)：

## 2 算子接口设计

### 2.1 参考接口

- mmcv

```c++
// 给出接口
void DynamicPointToVoxelBackwardCUDAKernelLauncher(
    at::Tensor &grad_feats, const at::Tensor &grad_voxel_feats,
    const at::Tensor &feats, const at::Tensor &voxel_feats,
    const at::Tensor &point2voxel_map, const at::Tensor &voxel_points_count,
    const reduce_t reduce_type);
```
```c++
// 给出cpu接口
void dynamic_point_to_voxel_backward(torch::Tensor &grad_feats,
                                     const torch::Tensor &grad_voxel_feats,
                                     const torch::Tensor &feats,
                                     const torch::Tensor &voxel_feats,
                                     const torch::Tensor &coors_idx,
                                     const torch::Tensor &voxel_points_count,
                                     const std::string &reduce_type);
```

### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API mluOpDynamicPointToVoxelBackward(
    const mluOpHandle_t handle, const mluOpReduceMode_t reduce_type,
    const mluOpTensorDescriptor_t grad_voxel_feats_desc,
    const void *grad_voxel_feats, const mluOpTensorDescriptor_t feats_desc,
    const void *feats, const mluOpTensorDescriptor_t voxel_feats_desc,
    const void *voxel_feats, const mluOpTensorDescriptor_t point2voxel_map_desc,
    const void *point2voxel_map,
    const mluOpTensorDescriptor_t voxel_points_count_desc,
    const void *voxel_points_count,
    const mluOpTensorDescriptor_t voxel_num_desc, const void *voxel_num,
    void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t grad_feats_desc, void *grad_feats);
```

```c++
mluOpStatus_t MLUOP_WIN_API mluOpGetDynamicPointToVoxelBackwardWorkspaceSize(
    const mluOpHandle_t handle, const mluOpReduceMode_t reduce_type,
    const mluOpTensorDescriptor_t feats_desc, size_t *workspace_size);
```


## 3 实现方案设计

### 3.1 实现方案

#### 3.1.1 计算原理说明

#### 3.1.2 实现方案

max模式有两个kernel
kernel1：
阶段一：计算offset  捞数
1.生成 0 - x 的 index 向量保存至 index_nram
2.将 index_nram 中每个index连续扩充C倍，保存至 index_mask_nram
3.根据输入的point2voxel_map取出对应的offset向量
4.生成mask1 在point2voxel_map[i] == -1 位置标志为 1
5.生成mask2 在feats[i] != voxel_feats[i]位置标志为 1
6.mask1，mask2做或操作生成mask3，mask3 做非操作生成mask4
7.tmp = mask3 * input_num +  mask4 * (index_mask_nram + i)

阶段二：做min比较，写数
方案一：
tmp直接与相应的 output_gdram 做atomicMin操作
方案二：
申请num_reduced * sizeof(bool)的共享内存is_occupy， 初始赋值false
根据index写值时，先判端 is_occupy[offset[index]]是否为false，是的话将值写为true将对应的gdram value load到片上，
作bang_min,然后写会gdram，最后将is_occupy[offset[index]]写为true
1Union1任务 is_occupy用sram申请， UnionX任务用gram申请
方案三：
根据 taskId == offset % taskDim 保证每个core分到不同的offset

kernel2：
输入的grad_voxel_feats根据kernel1输出的index结果把相应的value copy到对应的位置：VAA实现



### 3.2 伪代码实现（可选）
cuda源码

```c++
//kernel1：
template <typename T>
__global__ void max_reduce_traceback_scatter_idx_kernel(
    const T *feats, const T *reduced_feats, int32_t *reduce_from,
    const int32_t *coors_map, const int num_input, const int num_feats) {
  CUDA_1D_KERNEL_LOOP(x, num_input) {
    int32_t reduce_to = coors_map[x];

    const int input_offset = x * num_feats;
    const T *feats_offset = feats + input_offset;

    if (reduce_to == -1) {
      continue;
    }

    const int reduced_offset = reduce_to * num_feats;
    const T *reduced_feats_offset = reduced_feats + reduced_offset;
    int32_t *reduce_from_offset = reduce_from + reduced_offset;

    for (int i = 0; i < num_feats; i++) {
      if (feats_offset[i] == reduced_feats_offset[i]) {
        atomicMin(&reduce_from_offset[i], static_cast<int32_t>(x));
      }
    }
  }
}

//kernel2
template <typename T>
__global__ void max_reduce_scatter_grad_kernel(T *grad_feats,
                                               const T *grad_reduced_feats,
                                               const int32_t *reduce_from,
                                               const int num_reduced,
                                               const int num_feats) {
  CUDA_1D_KERNEL_LOOP(x, num_reduced) {
    const int reduced_offset = x * num_feats;
    const int32_t *scatter_to_offset = reduce_from + reduced_offset;
    const T *grad_reduced_feats_offset = grad_reduced_feats + reduced_offset;

    for (int i = 0; i < num_feats; i++) {
      grad_feats[scatter_to_offset[i] * num_feats + i] =
          grad_reduced_feats_offset[i];
    }
  }
}
```
mlu实现

```c++
// kernel1
// 每次loop可以处理 n 个 num_feats
// 生成index mask， 只需要生成一次

//方案二：
for (int i = 0; i < n, n++) {
    __bang_write_value(index_mask + i * num_feats, num_feats, i);
}

//loop
__memcpy(reduce_offset, coors_map + n_start, n * sizeof(int), GDRAM2NRAM);
//load reduce_feats
int index_offset = reduce_offset[0];
int ddr_offset = reduce_offset[0] * num_feats;

// get mask
__memcpy_async(feats_nram, feats_gdram + n_start * num_feats, n * num_feats * sizeof(T), GDRAM2NRAM);
for (int i = 0; i < n; i ++) {
    if (reduce_offset[i] != -1) {
      __memcpy_async(reduce_feats_nram + i * num_feats, reduce_feats_gdram + reduce_offset[i] * num_feats, num_feats * sizeof(T), GRRAM2NRAM);
    }
}
__sync_io();
__bang_equal(value_mask, reduce_feats_nram, feats_nram, n * num_feats);
__bang_not(tmp_mask, value_mask, n * num_feats);
__bang_mul(value_mask, value_mask, index_mask, n * num_feats);
__bang_mul_saclar(tmp_mask, tmp_mask, num_reduced, n * num_feats);
__bang_add(value_mask, value_mask, tmp_mask, n * num_feats);

bool is_go_lcoal = false;
for (int i = 0; i < n; i ++) {
    if (offset > 0) {
        if (is_occupy[index_offset] == false) {
          pvLock();
          is_occupy[index_offset] == true;
          pvUnlock();
        }
        while(1) {
            pvLock();
            if (is_occupy[index_offset] == false) {
              is_occupy[index_offset] = true;
              is_go_lcoal = true;
            }
            pvUnlock();
            if (is_go_lcoal) {
              __memcpy(result_nram, result_gdram + ddr_offset, num_feats * sizeof(T), GDRAM2NRAM);
              __bang_minimum(result_nram, result_nram, value_mask + i * num_feats, num_feats);
              __memcpy(result_gdram + ddr_offset, result_nram, num_feats * sizeof(T), NRAM2GDRAM);
              is_occupy[index_offset] == false;
              is_go_lcoal = false;
              break;
            }
          }
        }
    if (i < n - 1) {
        index_offset = reduce_offset[i + 1];
        ddr_offset = reduce_offset[i + 1] * num_feats;
    }
}

//方案三：
// 考虑先将所有的offset load sram上，然后广播到每个ipu core的nram上，
// 根据网络拆解下来的数规模 offset的size为 17176 * sizeof(int), 大约68k
// 根据nram_size拆解算出一次每个core能处理的c的数量记作n
int n_start = 0;
while (n_start < N) {
  int real_n = 0;
  int *offset_real_nram;
  // laod data
  for (int i = n_start; i < N; i ++) {
    int offset = offset_nram[i];
    if (taskId == offset % taskDim) {
      if (offset != -1) {
        __memcpy_async(result_nram + real_n * num_feats, result_gdram + offset * num_feats, num_feats *sizeof(T), GDRAM2NRAM);
        __memcpy_async(feats_nram + real_n * num_feats, feats_gdram + i * num_feats, num_feats *sizeof(T), GDRAM2NRAM);
        __memcpy_async(reduce_feats_nram + real_n * num_feats, reduce_feats_gdram + offset * num_feats, num_feats *sizeof(T), GDRAM2NRAM);
        __bang_write_value(index_mask + real_n * num_feats, num_feats, i);
        offset_real_nram[real_n] = offset;
        real_n++;
      }
      
    }
    if (real_n == n) break;
  }
  __sync_io();
  
  // compute
  __bang_equal(value_mask, reduce_feats_nram, feats_nram, n * num_feats);
  __bang_not(tmp_mask, value_mask, n * num_feats);
  __bang_mul(value_mask, value_mask, index_mask, n * num_feats);
  __bang_mul_saclar(tmp_mask, tmp_mask, num_reduced, n * num_feats);
  __bang_add(value_mask, value_mask, tmp_mask, n * num_feats);
  __bang_minimum(result_nram, result_nram, value_mask, n * num_feats);
  
  // store
  for (int i = 0; i < real_n; i++) {
    int offset = offset_real_nram[i];
    __memcpy_async(result_nram_gdram + offset * num_feats, result_nram + i * num_feats, num_feats * sizeof(T), NRAM2GDRAM);
  }
  __sync_io();
  n_start += real_n;
}
//kernel 2


```


### 3.3 拆分(任务拆分，多核拆分)

- kernel1根据input_num_reduced均拆到每一个core， kernel2根据voxel_num均拆到每一个core

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

|            | dtype | torch.Size() | | | | | | | | | |
| ---------- | ----- | ------------ |-|-|-|-|-|-|-|-|-|
| gard_feats | fp32	| [17176,128]	| [17398,128]	| [18726,128]	|[17398,64]	| [18726,64] | [19922,128] | [19440,128] | [19922,64]	| [19440,64] | [18601,128] |
| grad_voxel_feats | fp32	| [13743, 128] | [15130, 128]	| [14525, 128] | [15130, 64] | [14525, 64] | [16849,128] | [14676, 128]	| [16849,64] | [14676, 64] | [16768, 128] |
| feats	| fp32 | [17176,128] | [17398,128] | [18726,128] | [17398,64] | [18726,64] | [19922,128] | [19440,128] | [19922,64] | [19440,64] | [18601,128] |
| voxel_feats	| fp32 | [13743, 128] | [15130, 128] | [14525, 128] | [15130, 64] | [14525, 64] | [16849,128] | [14676, 128] | [16849,64] | [14676, 64] | [15737, 128] |
| point2voxel_map	| int32	| [17176] | [17398] | [18726] | [17398] | [18726] | [19922] | [19440] | [19922] | [19440] | [18607] |
| voxel_points_count | int32 | [13743] | [15130] | [14525] | [15130] | [14525] | [16849] | [14676] | [16849] | [14676] | [15737] |
| reduce_type	| | max |

### 3.8 算子防呆检查

在网络中，由于正向算子的实际输出规模无法提前预知，因此反向算子允许输入tensor中应该用M的地方用N代替, 实际的M值通过voxel_num获取。

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
