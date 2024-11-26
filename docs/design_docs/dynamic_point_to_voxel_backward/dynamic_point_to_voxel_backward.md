# dynamic_point_to_voxel_backward 算子开发设计方案

- #### 文档基本信息

| 算子名称 | `dynamic_point_to_voxel_backward` |
| ------- | -------------------------------  |
| 编制人/日期 | xuminjie/2023 |

- #### 修改记录

| 版本号 | 修订人 | 修订日期 | 修订描述 |
| ----- | ----- | ------ | ------- |
| V1.0 | xuminjie | 2023 | 首次提交 |
| V1.1 | wangyuan | 2024.11.08 | 修复sync、算法时序引入的潜在缺陷 |

- #### 内容描述

本文档为`dynamic_point_to_voxel_backward`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录。

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
| 布局限制     | 无               |
| 原位限制     | 不支持原位         |
| stride 限制 | 不支持 stride 机制 |
| 广播限制     | 不支持广播        |

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
typedef enum {
  MLUOP_REDUCE_DSUM  = 0, /*!< Computes the sum value. */
  MLUOP_REDUCE_DMEAN = 1, /*!< Computes the mean value. */
  MLUOP_REDUCE_DMAX  = 2, /*!< Computes the maximun value. */
} mluOpReduceMode_t;
```

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
    const mluOpTensorDescriptor_t grad_voxel_feats_desc,
    const mluOpTensorDescriptor_t feats_desc,
    const mluOpTensorDescriptor_t voxel_feats_desc,
    const mluOpTensorDescriptor_t point2voxel_map_desc,
    const mluOpTensorDescriptor_t voxel_points_count_desc,
    const mluOpTensorDescriptor_t voxel_num_desc, size_t *workspace_size);
```


## 3 实现方案设计

### 3.1 实现方案

#### 3.1.1 计算原理说明

`dynamic_point_to_voxel_backward` 算子包含7个输入，1个输出。

input:
1. `grad_feats`，`shape=[N,C]`
2. `grad_voxel_feats`，`shape=[M,C]`
3. `voxel_feats`，`shape=[M,C]`
4. `point2voxel_map`，`shape=[N]`
5. `voxel_point2_count`，`shape=[M]`
6. `voxel_num`，`hape=[1]`
7. `reduce_type`

output：
1. `grad_feats`，`shape=[N,C]` 

`dynamic_point_to_voxel_forward` 中:
- `coor` 表示 N 个点云数据对应在三维（三维体素网格坐标）具体坐标信息，`feats` 表示有 N 个点云数据，每个点云有 C 个特征
- `feats`、`coors` 中数据是一一对应的
- 该算子将全正坐标外坐标刷-1、去重、排序后，通过 `reduce_mode` ，对 `feats` 中数据进行处理

根据 1.2 节算子功能，可将`dynamic_point_to_voxel_forward` 分为2个kernel来实现:

- #### 计算逻辑层面

输入`reduce_type` = `max`时
- kernel1

先将`voxel_from`初始化成最大值N;

根据`point2voxel_map`中记录的“特征与体素特征的映射关系”。对比输入的特征`feats`和体素特征 `voxel_feats`：
1. 对于第 `i` 个体素特征中 `c(c=0,1,2,...,C-1)` 维特征 `voxel_feats[i,c]`，若与第 `j` 个特征的 `c` 维特征的 `feats[j, c]`相等，则认为这个体素特征是由该特征得到的
2. 若 `voxel_feats[i,c]` 与多个特征 `feats[j, c]`、`feats[k, c]` 相等，此时认为下标靠前的特征`feats[j, c]`（`j<k`）是 `voxel_feats[i,c]` 对应点
3. 新建 `voxel_from` 保存上述两者的下标关系
4. 举个简单例子
```c++
// feats 中数据在 voxel_feats 中的映射如下
point2voxel_map = [0, 0, 1, 1, 2, 3, ...]

// 依据映射关系 load feats,voxel_feats
deal_feats       = [feats[0], feats[1], feats[2], feats[3],...,]
dead_voxel_feats = [voxel_feats[0], voxel_feats[0], voxel_feats[1], voxel_feats[1], voxel_feats[2],...]

// 计算映射下标 voxel_from，对于多个 feats[j] 相同，取其中最小值
voxel_from = [0, 2, 4, 5, ...]
```

- kernel2

在正向计算时，如果某一体素特征对应的若干特征都是nan,那么`voxel_feats`中该值等于-inf；反向比较时，由于`voxel_feats`该点对应的`feats`中的点都是nan，与-inf不相等，所以`voxel_from`该点没有被更新成有效值，依旧是初始值。

构造mask, `voxel_from[i]==N`时，mask[i] = 0， `voxel_from[i]!=N`时，mask[i] = 1。
执行scatter操作，update = `grad_voxel_feats`, indices = `voxel_from`, output = `grad_feats`

#### 3.1.2 实现方案
通过以上分析, 要实现 `dynamic_point_to_voxel_backward` 算子功能，可以通过以上2个kernel来实现，其详细实现方案如下: 
- #### host 端

在host端主要进行kernel的逻辑调用:

- kernel1: KernelDynamicPointToVoxelBackward

该kernel用于获取梯度传播的目标坐标;
```c++
// 1. get scatter indices
KERNEL_CHECK((KernelDynamicPointToVoxelBackward(
    k_dim, k_type, handle->queue, feats, voxel_feats, grad_feats, workspace,
    point2voxel_map, voxel_num, N, C)));
```

- kernel2: KernelMaxReduceScatterGrad

该kernel用于执行离散拷贝
```c++
// 2. scatter
KERNEL_CHECK((KernelMaxReduceScatterGrad(k_dim, k_type, handle->queue,
                                             grad_feats, grad_voxel_feats,
                                             workspace, voxel_num, N, C)))
```

在300系列平台上缺少__scatter指令，在300系列以上平台上scatter_nd的性能优于kernel2，因此使用scatter_nd算子代替kernel2；scatter_nd是仓库已有算子，本设计文档不对其进行过多描述，下文主要对本算子的kernel在设备端的实现方案进行详细描述。

- #### device 端

- kernel1

  该kernel主要根据`point2voxel_map`比较`feats`和`voxel_feats`的值，从而确定`voxel_from`的值，其步骤如下：
  
  1. load <br>
    每个core只处理point2voxel_map[x] == taskId的数据 <br>
    如果point2voxel_map[x] == -1， 跳过 <br>
    load `feats`, `voxel_feats`, `voxel_from` 的值 <br>
    index_mask记录feats的坐标x <br>
    point2voxel_map_real保存去掉-1之后的point2voxel_map <br>
  2. compute <br>
    如果feats[i] == voxel_feats[i]，{mask1[i] = 1}，否则{mask1[i] = 0} <br>
    mask1的数据类型转换成int32，参与后续指令运算 <br>
    令mask2 = NOT mask1 <br>
    使用mask1筛选出"feats[i] == voxel_feats[i]"的点的下标 <br>
    mask2乘N，表示非法坐标 <br>
    将mask1和mask2相加，混合到同一块内存index_mask中，不同位置的值都得到了保留 <br>
    index_mask和voxel_from取较小值，更新voxel_from <br>
  3. store <br>
    由于正向reduce max时，可能存在多个值同时等于最大值的情况，因此反向比较时，会有多个feats[i] == voxel_feats[i]，我们只取最小的i <br>
    voxel_from_flag记录目标index是否已被赋值，true表示被赋值，false表示未被赋值 <br>
    当voxel_from_flag[i] == false时，直接store index_mask到voxel_from; <br>
    当voxel_from_flag[i] == true时，load voxel_from[i]，与index_mask比较取较小值，将较小值store回去。 <br>

- kernel2

  该kernel主要执行scatter操作，update = `grad_voxel_feats`, indices = `voxel_from`, output = `grad_feats`

  1. 处理nan/inf
    根据3.1.1节的描述，voxel_from中部分点可能未被更新，保持这初始值N，超过了output的index范围，需要去除 <br>
    如果voxel_from_nram[i] < N，那么mask[i] = 1，否则mask[i] = 0 <br>
    mask类型从int转成float，再转成bitindx，供scatter指令使用
  2. scatter
    执行scatter指令，update = `grad_voxel_feats`, indices = `voxel_from`, output = `grad_feats`, mask取上个步骤中构造的mask。


  说明：scatter指令的性能较差，后期需要优化。

### 3.2 伪代码实现（可选）
cuda源码

https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/cuda/scatter_points_cuda.cu#L100

https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/common/cuda/scatter_points_cuda_kernel.cuh

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

// 考虑先将所有的 offset load sram 上，然后广播到每个 mlu core 的 nram 上，
// 根据网络拆解下来的数规模offset的size为17176 * sizeof(int), 大约68k
// 根据nram_size拆解算出一次每个core能处理的C的数量记作n_limit
__memset_nram(voxel_from_flag_nram, M, false);
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
    if (real_n == n_limit) break;
  }
  __sync_io();
  
  // compute
  __bang_equal(value_mask, reduce_feats_nram, feats_nram, n_limit * num_feats);
  __bang_not(tmp_mask, value_mask, n_limit * num_feats);
  __bang_mul(value_mask, value_mask, index_mask, n_limit * num_feats);
  __bang_mul_saclar(tmp_mask, tmp_mask, num_reduced, n_limit * num_feats);
  __bang_add(value_mask, value_mask, tmp_mask, n_limit * num_feats);
  __bang_minimum(result_nram, result_nram, value_mask, n_limit * num_feats);
  
  // store
  for (int i = 0; i < real_n; i++) {
    int offset = offset_real_nram[i];
    if (voxel_from_flag_nram[offset] == false) {
      __memcpy_async(result_nram_gdram + offset * num_feats, result_nram + i * num_feats, num_feats * sizeof(T), NRAM2GDRAM);
      voxel_from_flag_nram[offset] = true;
    else {
      __sync_io();
      __memcpy(index_mask_nram, voxel_from + offset_real * C, size_feats_idx,
               GDRAM2NRAM);
      __bang_minequal(index_mask_nram, index_mask_nram, voxel_from_nram + i * C,
                      C);
      __memcpy(voxel_from + offset_real * C, index_mask_nram, size_feats_idx,
               NRAM2GDRAM);
    }
  }
  __sync_io();
  n_start += real_n;
}

// kernel 2

// 每次处理m_limit个C
// 先获得mask，再调用scatter
template <typename T>
__mlu_global__ void MLUKernelMaxReduceScatterGrad(T *grad_feats,
                                                  const T *grad_voxel_feats,
                                                  const int *voxel_from,
                                                  const int *voxel_num,
                                                  const int N, const int C) {
  const int M = *voxel_num;
  int size_feats = C * sizeof(T);
  int size_feats_idx = C * sizeof(int);

  int m_per_core = M / taskDim;
  int rem = M % taskDim;
  m_per_core += (int)(taskId < rem);
  int m_start = taskId * m_per_core + ((taskId < rem) ? 0 : rem);
  int m_start_offset = m_start * C;
  if (m_per_core <= 0) {
    return;
  }
  int nram_size = MAX_NRAM_SIZE;
  int m_limit =
      nram_size / (size_feats + 2 * size_feats_idx + C * sizeof(float));
  int stride = FLOOR_ALIGN(m_limit * C, 128 / sizeof(int));
  m_limit = stride / C;

  T *grad_voxel_feats_nram = (T *)nram_buffer;  // [m_limit, C]
  int *voxel_from_nram =
      (int *)(grad_voxel_feats_nram + stride);                // [m_limit, C]
  int *mask_nram = voxel_from_nram + stride;                  // [m_limit, C]
  float *mask_bitindex_nram = (float *)(mask_nram + stride);  // [m_limit, C]

  int m_repeat = m_per_core / m_limit;
  int m_rem = m_per_core % m_limit;

  // record index up bound
  __bang_write_value(mask_bitindex_nram, m_limit * C, (float)0.0f);
  for (int i = 0; i <= m_repeat; i++) {
    int m_real = (i == m_repeat) ? m_rem : m_limit;
    if (m_real <= 0) {
      break;
    }
    int data_num = m_real * C;
    __memcpy_async(grad_voxel_feats_nram, grad_voxel_feats + m_start_offset,
                   m_real * size_feats, GDRAM2NRAM);
    __memcpy_async(voxel_from_nram, voxel_from + m_start_offset,
                   m_real * size_feats_idx, GDRAM2NRAM);
    __sync();
    // if (voxel_from_nram[i] < N * C) {mask[i] = 1} else {mask[i] = 0}
    __bang_lt_scalar(mask_nram, voxel_from_nram, N * C, data_num);
    // mask change to float
    __bang_int322float((float *)mask_nram, (int *)mask_nram, data_num, 0);
    // mask change to bit
    __bang_gt_bitindex((float *)mask_nram, (float *)mask_nram,
                       (float *)mask_bitindex_nram, CEIL_ALIGN(data_num, 8));
    // indices change to bytes
    __bang_mul_scalar(voxel_from_nram, voxel_from_nram, sizeof(int),
    data_num);
#if __BANG_ARCH__ >= 592
    __scatter((void *)grad_feats, (const void *)grad_voxel_feats_nram,
              (const unsigned int *)voxel_from_nram, mask_nram, sizeof(T),
              NRAM2GDRAM, sizeof(T), data_num);
#endif
    m_start += m_real;
    m_start_offset = m_start * C;
  }
}

void MLUOP_WIN_API KernelMaxReduceScatterGrad(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    void *grad_feats, const void *grad_voxel_feats, const void *voxel_from,
    const void *voxel_num, const int N, const int C) {
  MLUKernelMaxReduceScatterGrad<<<k_dim, k_type, queue>>>(
      (float *)grad_feats, (const float *)grad_voxel_feats,
      (const int *)voxel_from, (const int *)voxel_num, N, C);
}
```


其他备选方案：
```c++
// 每次loop可以处理 n_limit 个 C
// 生成index mask， 只需要生成一次
//kernel1 备选方案：
for (int i = 0; i < n_limit, i++) {
    __bang_write_value(index_mask + i * num_feats, num_feats, i);
}

//loop
__memcpy(reduce_offset, coors_map + n_start, n_limit * sizeof(int), GDRAM2NRAM);
//load reduce_feats
int index_offset = reduce_offset[0];
int ddr_offset = reduce_offset[0] * num_feats;

// get mask
__memcpy_async(feats_nram, feats_gdram + n_start * num_feats, n_limit * num_feats * sizeof(T), GDRAM2NRAM);
for (int i = 0; i < n_limit; i ++) {
    if (reduce_offset[i] != -1) {
      __memcpy_async(reduce_feats_nram + i * num_feats, reduce_feats_gdram + reduce_offset[i] * num_feats, num_feats * sizeof(T), GRRAM2NRAM);
    }
}
__sync_io();
__bang_equal(value_mask, reduce_feats_nram, feats_nram, n_limit * num_feats);
__bang_not(tmp_mask, value_mask, n_limit * num_feats);
__bang_mul(value_mask, value_mask, index_mask, n_limit * num_feats);
__bang_mul_saclar(tmp_mask, tmp_mask, num_reduced, n_limit * num_feats);
__bang_add(value_mask, value_mask, tmp_mask, n_limit * num_feats);

bool is_go_lcoal = false;
for (int i = 0; i < n_limit; i ++) {
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
    if (i < n_limit - 1) {
        index_offset = reduce_offset[i + 1];
        ddr_offset = reduce_offset[i + 1] * num_feats;
    }
}
```


### 3.3 拆分(任务拆分，多核拆分)

- 执行UX任务，kernel1根据feats_desc->dims[0]均拆到每一个core， kernel2根据voxel_num[0]均拆到每一个core

### 3.4 性能优化设计

- 资源分配

本设计文档主要对本算子kernel中用到的资源进行分配，scatter_nd算子的资源分配可对应参考该算子的设计文档

| 表项 | 分配策略 |
| --- | ------- |
| NRAM | 参考3.1.2节                                                                  |
| WRAM | 未使用                                                                        |
| SRAM | 未使用                                                                        |
| GDRAM(workspace) | voxel_from使用[N, C]*sizeof(int)|

- 流水设计

  暂无

- 优化设计

1）kernel1加流水;

2）kernel2中调用的scatter指令性能较差，且跟真值相关，需详细测试，找出优化点。

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
| grad_feats | fp32	| [17176,128]	| [17398,128]	| [18726,128]	|[17398,64]	| [18726,64] | [19922,128] | [19440,128] | [19922,64]	| [19440,64] | [18601,128] |
| grad_voxel_feats | fp32	| [13743, 128] | [15130, 128]	| [14525, 128] | [15130, 64] | [14525, 64] | [16849,128] | [14676, 128]	| [16849,64] | [14676, 64] | [16768, 128] |
| feats	| fp32 | [17176,128] | [17398,128] | [18726,128] | [17398,64] | [18726,64] | [19922,128] | [19440,128] | [19922,64] | [19440,64] | [18601,128] |
| voxel_feats	| fp32 | [13743, 128] | [15130, 128] | [14525, 128] | [15130, 64] | [14525, 64] | [16849,128] | [14676, 128] | [16849,64] | [14676, 64] | [15737, 128] |
| point2voxel_map	| int32	| [17176] | [17398] | [18726] | [17398] | [18726] | [19922] | [19440] | [19922] | [19440] | [18607] |
| voxel_points_count | int32 | [13743] | [15130] | [14525] | [15130] | [14525] | [16849] | [14676] | [16849] | [14676] | [15737] |
| reduce_type	| | max |

### 3.8 算子防呆检查

在网络中，由于正向算子的实际输出规模无法提前预知，因此反向算子允许输入tensor中应该用M的地方用N代替, 实际的M值通过voxel_num获取。

1. handle为空防呆；
2. 平台检查，不支持300以下平台；
3. 检查输入输出支持的dtype以及shape；
4. 算子自身参数检查：reduce_type只支持max；
5. large tensor检查；
6. 指针为空检查；
7. 0元素检查；

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

### 4.2 已经过优化的规模说明
