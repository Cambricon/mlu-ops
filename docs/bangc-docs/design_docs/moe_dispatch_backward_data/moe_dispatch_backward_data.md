# mluopMoeDispatchBackwardData 算子开发设计方案

- #### 文档基本信息

| 算子名称    | mluopMoeDispatchBackwardData |
| ----------- | ---------------------------- |
| 编制人/日期 | 张皓喆/2023-2-20             |
| 审批人/日期 | 吴少强/2023-2-24             |
| 审批人/日期 | 董成威/2023-2-24             |
| 审批人/日期 | 胡永安/2023-2-24             |

- #### 修改记录

| 版本号 | 修订人 | 修订日期  | 修订描述 |
| ------ | ------ | --------- | -------- |
| V1.0   | 张皓喆 | 2023-2-20 | 首次提交 |

- #### 内容描述

本文档为`mluopMoeDispatchBackwardData`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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

| 算子功能简介                                                                 | 属于moe_dispatch_forward的反向算子                                                                                 |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 需求来源                                                                     | tutel                                                                                                              |
| 应用网络                                                                     | swin-transformer                                                                                                   |
| 输入数据类型                                                                 | float                                                                                                              |
| 输入标量参数                                                                 | samples: int32<br />capacity: int32<br />hidden: int32<br />num_experts: int32                                     |
| 输入 Shape                                                                   | gates: [samples]<br />indices: [samples]<br />locations: [samples]<br />dispatch: [num_experts * capacity, hidden] |
| 输入 Layout                                                                  | ARRAY                                                                                                              |
| 输出数据类型                                                                 | float                                                                                                              |
| 输出 Shape                                                                   | grad_input: [samples, hidden]                                                                                      |
| 输出 Layout                                                                  | ARRAY                                                                                                              |
| 模式(可选）                                                                  | /                                                                                                                  |
| 是否含有 dim/axis 等类似语义的参数且该参数支持负数/其他特殊处理              | 否                                                                                                                 |
| 是否含有 labels/index 等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 否                                                                                                                 |
| 是否需要支持原位                                                             | 否                                                                                                                 |
| 是否需要支持 stride 机制                                                     | 否                                                                                                                 |
| 是否需要支持广播                                                             | 否                                                                                                                 |
| 0 元素检查是否直接返回                                                       | 是，返回 MLUOP_STATUS_SUCCESS                                                                                      |
| 其他特殊需求(在线量化，融合，转数提前等，可选)                               | /                                                                                                                  |
| 本次开发优先支持的规模/模式                                                  | /                                                                                                                  |

### 1.2 算子功能和应用场景描述

算子功能：本算子属于moe_dispatch_forward的反向算子，属于MoE系统中的一个算子，主要用于对 dispatch 计算输入input的反向梯度。

应用场景：本算子用于MoE库中的tutel框架中的 swin-transformer 网络。

example：

```python
import numpy as np
import torch
from tutel.jit_kernels import sparse as jit_kernel

print(torch.__version__)
def moe_dispatch_bwd_data():
    samples=8192
    capacity=8192
    hidden=2048
    num_experts=2

    gates_t = np.asarray(np.random.randn(samples)*10, dtype=np.float32)
    indices_t = np.asarray(np.random.randint(0, num_experts, size = samples))
    locations_t = np.asarray(np.random.randint(0, capacity, size = samples))
    dispatch_t = np.asarray(np.random.randn(num_experts*capacity*hidden)*10, dtype=np.float32)

    gates_gpu = torch.from_numpy(gates_t).cuda()
    indices_gpu = torch.from_numpy(indices_t).cuda()
    locations_gpu = torch.from_numpy(locations_t).cuda()
    dispatch_gpu = torch.from_numpy(dispatch_t).cuda()

    grad_input = torch.zeros([samples, hidden], dtype=dispatch_gpu.dtype, device=dispatch_gpu.device)
    moe_dispatch_bwd_data = jit_kernel.create_backward_data(torch.float32, grad_input.is_cuda)
    moe_dispatch_bwd_data(gates_gpu, indices_gpu, locations_gpu, grad_input, dispatch_gpu, extra=[samples, hidden, capacity])
    print(grad_input)

if __name__ == '__main__':
    moe_dispatch_bwd_data()
```

### 1.3 算子输入输出参数要求

| 参数            | 语义                                           | 类型（输入/输出） | 支持类型                | 物理布局 | 规模限制                         |
| --------------- | ---------------------------------------------- | ----------------- | ----------------------- | -------- | -------------------------------- |
| handle          | 当前可获得的资源（context）                    | 输入              | mluOpHandle_t           | /        | 无                               |
| gates_desc      | 输入数据 gates 的描述符                        | 输入              | mluOpTensorDescriptor_t | /        | 无                               |
| gates           | 输入数据 gates 的指针                          | 输入              | float                   | ARRAY    | [samples]                        |
| indices_desc    | 输入数据 indices 的描述符                      | 输入              | mluOpTensorDescriptor_t | /        | 无                               |
| indices         | 输入数据 indices 的指针                        | 输入              | int32                   | ARRAY    | [samples]                        |
| locations_desc  | 输入数据 locations 的描述符                    | 输入              | mluOpTensorDescriptor_t | /        | 无                               |
| locations       | 输入数据 locations 的指针                      | 输入              | int32                   | ARRAY    | [samples]                        |
| dispatch_desc   | 输入数据 dispatch 的描述符                     | 输入              | mluOpTensorDescriptor_t | /        | 无                               |
| dispatch        | 输入数据 dispatch 的指针                       | 输入              | float                   | ARRAY    | [num_experts * capacity, hidden] |
| samples         | 输入标量数据，表示输入的个数，等效于batch-size | 输入              | int32                   | /        | 无                               |
| capacity        | 输入标量数据，表述需要处理的最大输入个数       | 输入              | int32                   | /        | 无                               |
| hidden          | 输入标量数据，表述单个 token 的向量长度        | 输入              | int32                   | /        | 无                               |
| num_experts     | 输入标量数据，表述专家数量                     | 输入              | int32                   | /        | 无                               |
| grad_input_desc | 输入数据 grad_input 的描述符                   | 输入              | mluOpTensorDescriptor_t | /        | 无                               |
| grad_input      | 输出数据 grad_input 的指针                     | 输出              | float                   | ARRAY    | [samples, hidden]                |

### 1.4 算子限制

| 限制类型         | 详细说明                                                                      |
| ---------------- | ----------------------------------------------------------------------------- |
| 数据类型限制     | gates: float <br /> indices: int32<br />locations: int32<br />dispatch: float |
| 布局限制         | ARRAY                                                                         |
| 原位限制         | 不支持原位                                                                    |
| stride 限制      | 不支持 stride 机制                                                            |
| 广播限制         | 不支持广播                                                                    |
| 输入 shape 限制  | gates、indices、locations、grad_input的第1维度大小等于samples                 |
| 输入 shape 限制  | grad_input、dispatch 的第2维度大小等于 hidden                                 |
| 输入 shape 限制  | dispatch的第1维度大小等于 num_experts * capacity                              |
| 输入数据范围限制 | indices 数据范围为 [0, num_experts)                                           |
| 输入数据范围限制 | locations 数据范围为 [0, capacity)                                            |

### 1.5 验收标准

#### 1.5.1 精度验收标准

按照[MLU-OPS 算子精度验收标准](../../../MLU-OPS-Accuracy-Acceptance-Standard.md)的要求明确本算子的精度标准：

本算子属于算术类算子，验收标准为 diff3=0。

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../../../MLU-OPS-Performance-Acceptance-Standard.md)：

本算子属于算术类算子，在适当规模 case 下，compute_efficiency 或者 io__efficiency 应达到 50% 以上。

## 2 算子接口设计

### 2.1 参考接口实现

- tutel cpu

```c
for (int i = 0; i < samples; ++i) {
  if (locations1_s[i] < capacity && indices1_s[i] >= 0) {
    for (int j = 0; j < hidden; ++j) {
      reshaped_input[i * hidden + j] = gates1_s[i] * dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j];
    }
  } else {
    for (int j = 0; j < hidden; ++j) {
      reshaped_input[i * hidden + j] = 0;
    }
  }
}
```

- tutel cuda

```c
// 函数定义
def create_backward_data(param_dtype, is_cuda=True):
  if not is_cuda:
    return JitCompiler.generate_cpu_kernel(kernel_type=1)

  return JitCompiler.generate_kernel({'dtype': get_kernel_dtype(param_dtype), 'IS_FLOAT': 1 if param_dtype == torch.float32 else 0}, '''
    #define __dtype @dtype@
    extern "C" __global__ __launch_bounds__(1024) void execute(__dtype* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ grad_reshaped_input, __dtype* __restrict__ dispatched_input, int samples, int hidden, int capacity) {
      // [thread_extent] blockIdx.x = 512
      // [thread_extent] threadIdx.x = 1024
      for (int i = blockIdx.x; i < samples; i += gridDim.x)
          if (locations1_s[i] < capacity && indices1_s[i] >= 0) {
              #pragma unroll
              for (int j = threadIdx.x; j < hidden; j += 1024)
                  grad_reshaped_input[i * hidden + j] = gates1_s[i] * dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j];
          } else {
              #pragma unroll
              for (int j = threadIdx.x; j < hidden; j += 1024)
    #if @IS_FLOAT@
                  grad_reshaped_input[i * hidden + j] = __dtype(0);
    #else
                  grad_reshaped_input[i * hidden + j] = __dtype(0, 0);
    #endif
          }
    }
  ''')

// 函数挂钩
self.func_bwd_data = jit_kernel.create_backward_data(self.dtype, indices_[0].is_cuda)

// 函数调用
ctx.config.func_bwd_data(g, i, l, grad_data, dispatched_input, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
```

### 2.2 接口设计

```c
// 给出MLU-OPS算子接口
mluOpStatus_t MLUOP_WIN_API
mluOpMoeDispatchBackwardData(mluOpHandle_t handle,
                             const mluOpTensorDescriptor_t gates_desc,
                             const void *gates,
                             const mluOpTensorDescriptor_t indices_desc,
                             const void *indices,
                             const mluOpTensorDescriptor_t locations_desc,
                             const void *locations,
                             const mluOpTensorDescriptor_t dispatch_desc,
                             const void *dispatch,
                             const int samples,
                             const int capacity,
                             const int hidden,
                             const int num_experts,
                             const mluOpTensorDescriptor_t grad_input_desc,
                             void *grad_input);
```

## 3 实现方案设计

### 3.1 实现方案

该算子用于MoE库中tutel网络，属于 moe_dispatch_forward 的反向算子，用于计算输入数据 input 的反向梯度`grad_input`。

该算子有四个输入tensor，包括`gates`维度`[samples]`，`indices`维度`[samples]`，`locations`维度`[samples]`，`dispatch`维度`[num_experts * capacity, hidden]`，
一个输出tensor，`grad_input` 维度`[samples, hidden]`，
以及四个标量参数`samples`， `capacity`， `hidden`， `num_experts`。

该算子根据输入 `gates`、`indices`、`locations`、`dispatch` 计算 input 的反向梯度`grad_input`，
主要涉及到 `gates` 和 `dispatch` 中的值，其中 `dispatch` 所取值的位置由 `indices`、`locations` 所决定，具体计算过程可参考伪代码。
当samples与hidden均不为0时，先调用 fill 算子初始化 GDRAM 上输出空间`grad_input`的内存初始化为 0，然后根据 samples 大小，分两个kernel计算，具体实现步骤实现步骤：

**Kernel 1 实现：**

1. 当 samples <= taskDim 时，调用`MLUKernelMoeDispatchBwdData_1`计算，主要实现步骤：

- 对`samples`进行拆分：根据samples大小，计算每个sample由多少个task并行计算；

```c
// 每个sample由多少个task来并行计算
int one_sample_task_num = taskDim / samples;
// 剩余task数，均分前n个sample
int rem_task = taskDim % samples;
int sample_idx = 0;
// 根据taskId，计算起始索引 sample_idx
if ((rem_task > 0) && (taskId < (one_sample_task_num + 1) * rem_task)) {
    sample_idx = (int)(taskId / (one_sample_task_num + 1));
    one_sample_task_num = one_sample_task_num + 1;
} else {
    sample_idx = (int)((taskId - rem_task) / one_sample_task_num);
}
```

- 获取当前 sample_idx 对应的 indices 和 locations 值，判断如果不合法时提前返回，伪代码如下：

```c
int indices_value = indices[sample_idx];
int location_value = locations[sample_idx];
if ( indices_value < 0 || indices_value >= num_experts ||
    location_value < 0 || location_value >= capacity) {
  return;
}
```

- 根据`taskId` 计算起始索引`sample_idx` ，并计算每个task处理的hidden的大小和偏移，伪代码如下：

```c
int logic_tid = taskId % one_sample_task_num;
int hidden_per_task = hidden / one_sample_task_num;
int rem_hidden_num = hidden % one_sample_task_num;
int hidden_seg_num = hidden_per_task + (int)(logic_tid < rem_hidden_num);
int hidden_data_offset = logic_tid * hidden_per_task + ((logic_tid < rem_hidden_num) ? logic_tid : rem_hidden_num);
```

2. 初始化阶段

2.1 计算得到`deal_h`的大小；

```c
  const int max_nram_num = MAX_NRAM_SIZE / sizeof(T);
  const int deal_h = max_nram_num / 4;
  const int pingpong_num = 2 * deal_h;
  T *nram_grad_input = (T *)nram_buffer;
  T *nram_dispatch = nram_grad_input + deal_h;
```

2.2 根据`hidden_seg_num` 和 `deal_h` 计算 repeat_h 和 rem_h；

```c
int logic_tid = taskId % one_sample_task_num;
int hidden_per_task = hidden / one_sample_task_num;
int rem_hidden_num = hidden % one_sample_task_num;
int hidden_seg_num = hidden_per_task + (int)(logic_tid < rem_hidden_num);
int hidden_data_offset =
    logic_tid * hidden_per_task +
    ((logic_tid < rem_hidden_num) ? logic_tid : rem_hidden_num);
int repeat_h = hidden_seg_num / deal_h;
int rem_h = hidden_seg_num % deal_h;
```

3. 处理阶段，可以排三级流水：

```c
  int grad_input_addr_offset = sample_idx * hidden + hidden_data_offset;
  T *base_grad_input_addr = (T *)grad_input + grad_input_addr_offset;
  int dispatch_idx_offset = (indices_value * capacity + location_value) * hidden;
  T *base_dispatch_addr = (T *)dispatch + dispatch_idx_offset + hidden_data_offset;
  T gates_si_value = gates[sample_idx];
  lcs(base_grad_input_addr, base_dispatch_addr, nram_grad_input, nram_dispatch,
      gates_si_value, repeat_h, rem_h, deal_h, pingpong_num);

```

**Kernel 2 实现：**

- 当 samples >= taskDim 时，调用`MLUKernelMoeDispatchBwdData_2`计算，主要实现步骤：

1. 根据`samples` 计算每个`taskId` 要处理的数量`samples_num`，以及起始索引`sample_idx` ，伪代码如下；

  ```c
  // 一个task需要处理多个sample
  int per_task_sample_num = samples / taskDim;
  int rem_sample_num = samples % taskDim;
  // 根据taskId，计算当前task，需要处理的sample的数量 sample_num
  int samples_num = per_task_sample_num + (int)((taskId < rem_sample_num));
  // 根据taskId，计算起始索引 sample_idx
  int sample_idx = taskId * per_task_sample_num + ((taskId < rem_sample_num) ? taskId : rem_sample_num);
  ```

2. 初始化阶段

- 根据当前架构的nram空间计算得到deal_s和deal_h的大小；

```c
  int max_deal_h = (MAX_NRAM_SIZE - 4 * sizeof(int) - 1 * sizeof(T)) / 2 / sizeof(T);
  int deal_h = 0;
  int deal_s = 0;
  if (hidden > max_deal_h) {
    deal_s = 1;
    deal_h = max_deal_h;
  } else {
    deal_h = hidden;
    deal_s = (MAX_NRAM_SIZE - 2 * deal_h * sizeof(T)) / (1 * sizeof(T) + 4 * sizeof(int));
  }
```

- 根据 samples_num 和 deal_s 计算 repeat_s 和 rem_s ；
计算余数部分，根据 `hidden` 和 deal_h 计算 repeat_h 和 rem_h ；
计算各输入tensor的GDRAM地址偏移：

```c
int repeat_s = samples_num / deal_s;
int rem_s = samples_num % deal_s;
int repeat_h = hidden / deal_h;
int rem_h = hidden % deal_h;

T *base_gates = (T *)gates + sample_idx;
T *base_indices = (T *)indices + sample_idx;
T *base_locations = (T *)locations + sample_idx;
uint32_t input_addr_offset = sample_idx * hidden；
T *base_grad_input = (T *)grad_input + input_addr_offset;
```

3. 循环处理

```c
  T *nram_gates = (T *)nram_buffer;
  int *nram_dispatch_idx_offset = (int *)(nram_gates + deal_s);
  int *nram_mask = nram_dispatch_idx_offset + deal_s;
  int *nram_indices = nram_mask + deal_s;
  int *nram_locations = nram_indices + deal_s;
  T *nram_grad_input = (T *)(nram_locations + deal_s);
  T *nram_dispatch = nram_grad_input + deal_h;
  int repeat_s = samples_num / deal_s;
  int rem_s = samples_num % deal_s;
  int repeat_h = hidden / deal_h;
  int rem_h = hidden % deal_h;
  // get gdram input gates indices locations offset
  T *base_gates = (T *)gates + sample_idx;
  int *base_indices = (int *)indices + sample_idx;
  int *base_locations = (int *)locations + sample_idx;
  // get gdram output grad_input offset
  int grad_input_offset = sample_idx * hidden;
  T *base_grad_input = (T *)grad_input + grad_input_offset;
  for (int s_iter = 0; s_iter <= repeat_s; ++s_iter) {
    int deal_s_num = (s_iter == repeat_s)
                            ? rem_s
                            : deal_s;
    if (deal_s_num == 0) {
      break;
    }
    // load gates indices locations
    T *base_gates_s = base_gates + s_iter * deal_s;
    int *base_indices_s = base_indices + s_iter * deal_s;
    int *base_locations_s = base_locations + s_iter * deal_s;
    __memcpy(nram_gates, base_gates_s, deal_s_num * sizeof(T), GDRAM2NRAM);
    __memcpy(nram_indices, base_indices_s, deal_s_num * sizeof(int), GDRAM2NRAM);
    __memcpy(nram_locations, base_locations_s, deal_s_num * sizeof(int), GDRAM2NRAM);
    // dispatch idx = (nram_indices * capacity + nram_locations) * hidden
    __bang_mul_scalar(nram_dispatch_idx_offset, nram_indices, capacity, deal_s_num);
    __bang_add(nram_dispatch_idx_offset, nram_dispatch_idx_offset, nram_locations, deal_s_num);
    __bang_mul_scalar(nram_dispatch_idx_offset, nram_dispatch_idx_offset, hidden, deal_s_num);
    // 0 <= nram_locations < capacity
    __bang_ge_scalar(nram_mask, nram_locations, (int)0, deal_s_num);
    __bang_lt_scalar(nram_locations, nram_locations, capacity, deal_s_num);
    __bang_and(nram_locations, nram_locations, nram_mask, deal_s_num);
    // 0 <= nram_indices < num_experts
    __bang_ge_scalar(nram_mask, nram_indices, (int)0, deal_s_num);
    __bang_lt_scalar(nram_indices, nram_indices, num_experts, deal_s_num);
    __bang_and(nram_indices, nram_indices, nram_mask, deal_s_num);
    __bang_and(nram_mask, nram_indices, nram_locations, deal_s_num);
    // get output grad_input s offset
    T *base_grad_input_s = base_grad_input + s_iter * deal_s * hidden;
    for (int si = 0; si < deal_s_num; ++si) {
      if (nram_mask[si] != 1) {
        continue;
      }
      T *base_dispatch_si = (T *)dispatch + nram_dispatch_idx_offset[si];
      T *base_grad_input_s_si = base_grad_input_s + si * hidden;
      for (int h_iter = 0; h_iter <= repeat_h; ++h_iter) {
        int deal_h_num = (h_iter == repeat_h)
                                ? rem_h
                                : deal_h;
        if (deal_h_num == 0) {
          break;
        }
        // get input dispatch h offset
        T *base_dispatch_si_h = base_dispatch_si + h_iter * deal_h;
        // get output grad_input s h offset
        T *base_grad_input_s_si_h = base_grad_input_s_si + h_iter * deal_h;
        __memcpy(nram_dispatch, base_dispatch_si_h, deal_h_num * sizeof(T), GDRAM2NRAM);
        __bang_mul_scalar(nram_grad_input, nram_dispatch, nram_gates[si], deal_h_num);
        // store grad_input
        __memcpy(base_grad_input_s_si_h, nram_grad_input, deal_h_num * sizeof(T), NRAM2GDRAM);
      }  // repeat h
    } // repeat deal_s_num
  } // repeat s
```

### 3.2 伪代码实现

```c
for (int i = 0; i < samples; ++i) {
  uint64_t index_i = i * hidden;
  if (locations[i] < capacity && indices[i] >= 0) {
    int64_t dispatch_index_i = (indices[i] * capacity + locations[i]) * (hidden);
    for (int j = 0; j < hidden; ++j) {
      grad_input[index_i + j] = gates[i] * dispatch[dispatch_index_i + j];
    }
  } else {
    for (int j = 0; j < hidden; ++j) {
      grad_input[index_i + j] = 0;
    }
  }
}
```

### 3.3 拆分(任务拆分，多核拆分)

基本任务类型为UNION1。总体思路是在核间拆分 `samples` ，核内拆分 `hidden`：

根据输出 grad_input 规模 [samples, hidden]，多核拆分在 samples 维度上拆分，

输入`input` 维度`[samples, hidden]`，输出`grad_input`的维度`[samples, hidden]`，分两种情况：

- 情况 1：当 samples <= taskDim 时，则表示一个sample由多个task处理，由 Kernel1 实现，拆分伪代码如下：

```c
// 每个sample由多少个task来计算
int one_sample_task_num = taskDim / samples;
// 剩余task数，可将剩余task均分给前n个sample
int rem_task = taskDim / samples;
```

说明：前rem_task个sample中，每个sample可以由（one_sample_task_num + 1）个task处理，最后（samples - rem_task）个sample，每个sample由one_sample_task_num个task处理。
在实际计算时，根据taskId，计算起始索引`sample_idx`，伪代码如下：

```c
// 根据taskId，计算起始索引 sample_idx
int sample_idx = 0;
if ((rem_task > 0) && (taskId < (one_sample_task_num + 1) * rem_task)) {
    sample_idx = (int)(taskId / (one_sample_task_num + 1));
    one_sample_task_num = one_sample_task_num + 1:
} else {
    sample_idx = (int)((taskId - rem_task) / one_sample_task_num);
}
```

- 情况 2：当 samples > taskDim 时，则表示一个task需要处理多个sample，由 Kernel2 实现 拆分伪代码如下：

```c
int per_task_sample_num = samples / taskDim;
int rem_sample_num = samples % taskDim;
// 根据taskId，计算当前task，需要处理的sample的数量 sample_num
int samples_num = per_task_sample_num + (int)((taskId < rem_sample_num));
// 根据taskId，计算起始索引 sample_idx
int sample_idx = taskId * per_task_sample_num + ((taskId < rem_sample_num) ? taskId : rem_sample_num);
```

### 3.4 性能优化设计

1、资源分配

| 表项            | 分配策略 |
| --------------- | -------- |
| NRAM            | 见下述   |
| WRAM            | 暂未使用 |
| SRAM            | 暂未使用 |
| DRAM(workspace) | 暂未使用 |

NRAM 空间划分如下：

- 当 samples <= taskDim 时，`nram_gates`,`nram_indices`,`nram_locations`,`nram_idx`,`nram_mask` 空间大小为 deal_s = 1：

| nram空间指针含义 | 大小   | 类型  |
| ---------------- | ------ | ----- |
| nram_gates       | 1      | float |
| nram_indices     | 1      | int32 |
| nram_locations   | 1      | int32 |
| nram_idx         | 1      | int32 |
| nram_mask        | 1      | int32 |
| nram_grad_input  | deal_h | float |
| nram_dispatch    | deal_h | float |


- 当 samples > taskDim 时，nram空间划分如下：

| nram空间指针含义 | 大小   | 类型  |
| ---------------- | ------ | ----- |
| nram_gates       | deal_s | float |
| nram_indices     | deal_s | int32 |
| nram_locations   | deal_s | int32 |
| nram_idx         | deal_s | int32 |
| nram_mask        | deal_s | int32 |
| nram_grad_input  | deal_h | float |
| nram_dispatch    | deal_h | float |

说明：

- `deal_s` 表示一次可load多少个`sample`维度的数据，当`deal_s`小于`samples`时，计算`repeat_s`和`rem_s`；
- `deal_h`表述一次可以load的`hidden`维度的数据大小，当`deal_h`小于`hidden`时，计算`repeat_h`和`rem_h`；

2、流水设计

- 经过实际测试验证，排流水的收益为负，因此不再排流水。

### 3.5 可维护性设计

1、bangc 代码中加入必要的 log 信息，输入的规模、数据类型、layout ，以及如果出错会导致程序 core dump 的变量，比如 IO 指令的 data_size、dim xyz 的值等，这些信息都是有利于快速定位问题；

2、对每一个函数命名变量命名都有充分的注释；

3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代。

### 3.6 测试用例设计

- 算子在网络中用到的规模：

  samples: 18432, hidden: 512, capacity: 11520, num_experts: 2

  samples: 4608, hidden: 1024, capacity: 2880, num_experts: 2

  这里详细列举出其中一种case：

    | 参数        | shape                            | 网络规模     | 类型  | 范围             |
    | ----------- | -------------------------------- | ------------ | ----- | ---------------- |
    | gates       | [samples]                        | [18432]      | float |                  |
    | indices     | [samples]                        | [18432]      | int   | [0, num_experts) |
    | locations   | [samples]                        | [18432]      | int   | [0, capacity)    |
    | dispatch    | [num_experts * capacity, hidden] | [23040, 512] | float |                  |
    | samples     | SCALAR                           | 18432        |       |                  |
    | capacity    | SCALAR                           | 11520        |       |                  |
    | hidden      | SCALAR                           | 512          |       |                  |
    | num_experts | SCALAR                           | 2            |       |                  |
    | grad_input  | [samples, hidden]                | [18432, 512] | float |                  |

- 边界 case：需要注意测试 samples = 1 or samples = (taskdim - 1) or samples = (taskdim + 1) 的情况。

- 0元素测试 case1:
- int型变量：samples: 0 ， capacity: 8192， hidden: 2048， num_experts: 2
- tensor：gates: [0]，float；
        indices: [0]，int类型
        locations: [0]， int类型
        grad_input: [0, 2048]，float类型
        dispatch: [16384, 2048]， float类型

- 0元素测试 case2:
- int型变量：samples: 8192 ， capacity: 8192， hidden: 0， num_experts: 2
- tensor：gates: [8192]，float；
        indices: [8192]，int类型，数据范围为 [0, num_experts)
        locations: [8192]， int类型，数据范围为 [0, capacity)
        grad_input: [8192, 0]，float类型
        dispatch: [16384, 0]， float类型

- 0元素测试 case3:
- int型变量：samples: 8192 ， capacity: 0， hidden: 2048， num_experts: 2
- tensor：gates: [8192]，float；
        indices: [8192]，int类型，数据范围为 [0, num_experts)
        locations: [8192]， int类型，数据范围为 [0, capacity)
        grad_input: [8192, 2048]，float类型
        dispatch: [0, 2048]， float类型

### 3.7 算子防呆检查

- 列出算子需要做的防呆：

  1、描述符指针为空防呆，对 handle、gates_desc、indices_desc、locations_desc、dispatch_desc、grad_input_desc 的检查；

  2、指针为空防呆，对 gates、indices、locations、dispatch、grad_input 的检查；

  3、0 元素检查防呆，VLOG(5)打印信息，返回MLUOP_STATUS_SUCCESS；

  4、对输入输出支持的 dtype、layout 以及 shape 进行防呆：

  - grad_input_desc、dispatch_desc、gates_desc：仅支持float，indices_desc、locations_desc 仅支持int；
  - indices_desc、locations_desc、gates_desc、grad_input_desc的第1个维度大小相等且等于samples；
  - grad_input_desc、dispatch_desc的第2个维度大小相等且等于hidden；
  - dispatch_desc的第1个维度大小等于num_experts * capacity;

  5、large tensor防呆：对 gates_desc、indices_desc、locations_desc、dispatch_desc、grad_input_desc 的检查；

## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

新算子首次提交，暂无

### 4.2 已经过优化的规模说明

新算子首次提交，暂无

## 5 方案实施

### 5.1 开发测试计划

- 2023.2.16-2023.2.17：算子功能调研，需求沟通和确认
- 2023.2.20-2023.2.24：算子方案设计及评审，gtest开发、generator开发
- 2023.2.27-2023.3.3：算子device开发调试，大规模测试
- 2023.3.6-2023.3.8：整理测试报告，review，PR合入

### 5.2 风险分析

根据 MoE 仓库 v0.2.0 版本 JitCompiler.generate_kernel create_backward_data 的实现，包括但不限于以下情况不对齐：
1. 当 num_experts * capacity = 0时，会出现 RuntimeError，取决于内存是否越界，但 mlu 返回成功。
2. 当 输入tensor indices 的数据范围超过 [0, num_experts) 或者 locations 的数据范围超过 [0, capacity) 时会出现 RuntimeError，取决于内存是否越界，但 mlu 跳过该点进行下一个计算。
