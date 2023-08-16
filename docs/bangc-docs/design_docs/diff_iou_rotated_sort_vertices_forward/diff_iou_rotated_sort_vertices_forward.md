# diff_iou_rotated_sort_vertices_forward 算子开发设计方案

- #### 文档基本信息

| 算子名称    | diff_iou_rotated_sort_vertices_forward |
| ----------- | -------------------------------------- |
| 编制人/日期 | 吴少强/2023-4-11                       |

- #### 修改记录

| 版本号 | 修订人 | 修订日期  | 修订描述 |
| ------ | ------ | --------- | -------- |
| V1.0   | 吴少强 | 2023-4-11 | 首次提交 |

- #### 内容描述

本文档为`diff_iou_rotated_sort_vertices_forward`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录。

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

| 算子功能简介                                                 | 对两个box相交构成的多边形有效顶点排序，输出排序后顶点索引 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 需求来源                                                     | MMCV                                                         |
| 应用网络                                                     | imvoxelnet                                                   |
| 输入数据类型                                                 | vertices: float32<br />mask: bool<br />num_valid: int32_t    |
| 输入 Shape                                                   | vertices: [B, N, 24, 2]<br />mask: [B, N, 24]<br />num_valid: [B, N] |
| 输入 Layout                                                  | ARRAY                                                        |
| 输出数据类型                                                 | int32_t                                                      |
| 输出 Shape                                                   | idx: [B, N, 9]                                               |
| 输出 Layout                                                  | ARRAY                                                        |
| 模式(可选）                                                  | 无                                                           |
| 是否含有 dim/axis 等类似语义的参数且该参数支持负数/其他特殊处理 | 无                                                           |
| 是否含有 labels/index 等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 无                                                           |
| 是否需要支持原位                                             | 否                                                           |
| 是否需要支持 stride 机制                                     | 否                                                           |
| 是否需要支持广播                                             | 否                                                           |
| 0 元素检查是否直接返回                                       | N=0 返回 MLUOP_STATUS_SUCCESS，其余情况直接返回MLUOP_STATUS_BAD_PARAM |
| 其他特殊需求(在线量化，融合，转数提前等，可选)               | 无                                                           |
| 本次开发优先支持的规模/模式                                  |                                                              |

### 1.2 算子功能和应用场景描述

1. 算子功能：该算子为 `diff_iou_rotated` 整个计算过程中的一步，即完成对两个box相交构成的多边形有效顶点排序（参考算法按逆时针排序所有有效顶点），输出排序后顶点索引。

   - 输入输出 tensor 说明：

     1. 输入 `vertices: [B, N, 24, 2]` ： 保存两个 box 的交点坐标，其中 dim[2] = 24 表示交点数量，在实际`diff_iou_rotated` 计算时，

        24 等于 24，dim[3] = 2表示每个交点的坐标`x`和`y`；

     2. 输入 `mask:[B, N, 24]`：`24` 个交点是否为构成多边形的有效顶点，bool类型，0：无效，1：有效；

     3. 输入 `num_valid:[B, N]`：对`mask`的dim[2]求和的结果，表示多边形由几个顶点构成；

     4. 输出 `idx: [B, N, 9]`：两个矩形框相交，最大构成八边形，因此封闭八边形由8个顶点且重复第一个顶点构成，总共9个顶点的索引；

2. `diff_iou_rotated`的计算过程，分如下几步：

   1. 计算旋转后的 `box` 4个顶点坐标；
   2. 计算两个 `box` 交点及其有效性：计算两个旋转 `box` 的交点（共16个交点），并判断交点是否有效（是否有效：是否为构成相交多边形的有效顶点） ，得到 `valid_mask`；
   3. 计算 `box1` 和 `box2` 的顶点关系，即判断 `box1` 的4个角点是否在 `box2` 内 或者 `box2` 的4个点是否在 `box1` 内，输出 mask：`c1_in_2` 和 `c2_in_1`；
   4. 计算所有交点个数（ box1 和 box2 的8个角点 + 第2步中两个 `box` 交点（16个）），共24个点，及其有效性 `mask`（mask =  c1_in_2 + c2_in_1 + valid_mask ）；
   5. 对两个 `box` 相交构成的多边形有效顶点排序，输出排序后顶点索引；调用算子`diff_iou_rotated_sort_vertices_forward` 实现；
   6. 计算两个 `box` 交集面积: 根据第5步的计算结果 `idx`，计算交集面积 `intersection_area`；
   7. 计算 `iou`：两个 box 的面积为 `area1` 和 `area2`，则 `iou = intersection_area / (area1 + area2 - intersection_area)`；

3. nan/inf行为

   1. 输入vertices 存在nan/inf 的处理：只包含nan、只包含inf、同时包含nan/inf的情况。

4. example：

   ```python
   import mmcv
   import torch
   import sys
   
   from mmcv.utils import ext_loader
   ext_module = ext_loader.load_ext(
           '_ext',['diff_iou_rotated_sort_vertices_forward'])
   
   print(torch.__version__)
   def diff_iou_rotated_sort_vertices_forward():
       b = 1
       n = 4
   
       vertices = torch.randn((b, n, 24, 2), dtype=torch.float).cuda()
       mask = torch.randint(0, 2, (b, n, 24), dtype=torch.bool).cuda()
       num_valid = num_valid = torch.sum(mask.int(), dim=2).int().cuda()  # (B, N)
   
       print(vertices)
       print(mask)
       print(num_valid)
   
       idx = ext_module.diff_iou_rotated_sort_vertices_forward(
               vertices, mask, num_valid)
       
       print(idx)
   
   if __name__ == '__main__':
       diff_iou_rotated_sort_vertices_forward()
       
   """
   # vertices[1, 1, 24, 2]
   tensor([[[[    inf,  1.2343],
             [-1.7780,     inf],
             [ 0.3783,     inf],
             [    nan, -0.3525],
             [ 0.5128, -1.3507],
             [ 0.8502, -0.6566],
             [ 1.5277,  0.1336],
             [-0.3684,  1.5274],
             [-0.3581, -1.0142],
             [ 0.8423,  0.8622],
             [ 1.6489, -0.7533],
             [    inf,  0.7217],
             [-0.1898,     inf],
             [    nan,  0.3361],
             [ 1.1861,  0.2382],
             [ 2.7012,     inf],
             [ 0.3704, -0.6879],
             [ 1.1565, -0.5807],
             [    nan,  0.1900],
             [ 0.4597,     nan],
             [ 1.3887,     inf],
             [    inf,     nan],
             [ 1.3208,  1.2068],
             [-1.0437,     nan]]]], device='cuda:0')
   # mask[1, 1, 24]
   tensor([[[False,  True, False, False,  True, False, False,  True, False,  True,
             False,  True,  True, False, False, False, False, False, False,  True,
              True, False, False, False]]], device='cuda:0')
   # num_valid[1, 1]
   tensor([[8]], device='cuda:0', dtype=torch.int32)
   
   # idx[1, 1, 9]
   tensor([[[9, 1, 7, 4, 0, 4, 0, 4, 9]]], device='cuda:0', dtype=torch.int32)
   """
   
   """
   # B = 0
   tensor([], device='cuda:0', size=(0, 1, 24, 2))
   tensor([], device='cuda:0', size=(0, 1, 24), dtype=torch.bool)
   torch.Size([0, 1])
   tensor([], device='cuda:0', size=(0, 1), dtype=torch.int32)
   Traceback (most recent call last):
     File "diff_iou_rotated_kernel.py", line 48, in <module>
       diff_iou_rotated_sort_vertices_forward()
     File "diff_iou_rotated_kernel.py", line 43, in diff_iou_rotated_sort_vertices_forward
       vertices, mask, num_valid)
   RuntimeError: CUDA error: invalid configuration argument
   """
   
   """
   # N = 0
   tensor([], device='cuda:0', size=(1, 0, 24, 2))
   tensor([], device='cuda:0', size=(1, 0, 24), dtype=torch.bool)
   torch.Size([1, 0])
   tensor([], device='cuda:0', size=(1, 0), dtype=torch.int32)
   torch.Size([1, 0, 9])
   tensor([], device='cuda:0', size=(1, 0, 9), dtype=torch.int32)
   """
   ```

   

### 1.3 算子输入输出参数要求

| 参数           | 语义                                                         | 类型（输入/输出） | 支持类型                | 物理布局 | 规模限制 |
| -------------- | ------------------------------------------------------------ | ----------------- | ----------------------- | -------- | -------- |
| handle         | 当前可获得的资源（context）                                  | 输入              | mluOpHandle_t           | /        | 无       |
| vertices_desc  | 输入数据 vertices 的描述符，包含了 vertices 的数据类型、数据维度和布局等信息 | 输入              | mluOpTensorDescriptor_t | /        | 见1.4    |
| vertices       | 输入数据 vertices 的指针                                     | 输入              | float32                 | ARRAY    | 无       |
| mask_desc      | 输入数据 mask 的描述符，包含了 mask 的数据类型、数据维度和布局等信息 | 输入              | mluOpTensorDescriptor_t | /        | 见1.4    |
| mask           | 输入数据 mask 的指针                                         | 输入              | bool                    | ARRAY    | 无       |
| num_valid_desc | 输入数据 num_valid 的描述符，包含了 num_valid 的数据类型、数据维度和布局等信息 | 输入              | mluOpTensorDescriptor_t | /        | 见1.4    |
| num_valid      | 输入数据 num_valid 的指针                                    | 输入              | int32_t                 | ARRAY    | 无       |
| idx_desc       | 输出数据 idx 的描述符，包含了 idx 的数据类型、数据维度和布局等信息 | 输入              | mluOpTensorDescriptor_t | /        | 见1.4    |
| idx            | 输出数据 idx 的指针                                          | 输出              | int32_t                 | ARRAY    | 无       |

### 1.4 算子限制

| 限制类型     | 详细说明                                                     |
| ------------ | ------------------------------------------------------------ |
| 数据类型限制 | vertices: float32<br />mask: bool<br />num_valid: int32_t    |
| 布局限制     | ARRAY                                                        |
| 数据范围限制 | mask[B, N, 24]：bool类型，取值为0或者1<br />num_valid 取值范围为[0, 8] ：num_valid[B, N] = sum(mask[B, N, 24], dim=2)，即对mask最后一个维度求和<br />num_valid[B, N] <= 8：表示相交构成的多边形最多有8个有效顶点，否则输入不合法，不保证计算结果与参考接口一致 |
| 原位限制     | 不支持原位                                                   |
| stride 限制  | 不支持 stride 机制                                           |
| 广播限制     | 不支持广播                                                   |
| shape 限制   | B == vertices_desc.dims[0] == mask_desc.dims[0] == num_valid_desc.dims[0] == idx_desc.dims[0] |
| shape 限制   | N == vertices_desc.dims[1] == mask_desc.dims[1] == num_valid_desc.dims[1] == idx_desc.dims[1] |
| shape 限制   | vertices_desc.dims[2] == mask_desc.dims[2] == 24             |
| shape 限制   | vertices_desc.dims[3] == 2                                   |
| shape 限制   | idx_desc.dims[2] == 9                                        |

### 1.5 验收标准

#### 1.5.1 精度验收标准

按照[精度验收标准](../MLU-OPS-Accuracy-Acceptance-Standard.md)的要求明确本算子的精度标准。

本算子属于算术类算子，验收标准为 diff3=0。

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../MLU-OPS-Performance-Acceptance-Standard.md)。

## 2 算子接口设计

### 2.1 参考接口

- MMCV

```c++
at::Tensor DiffIoURotatedSortVerticesCUDAKernelLauncher(at::Tensor vertices,
                                                        at::Tensor mask,
                                                        at::Tensor num_valid);
```

### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API
mluOpDiffIouRotatedSortVerticesForward(mluOpHandle_t handle,
                                       const mluOpTensorDescriptor_t vertices_desc,
                                       const void *vertices,
                                       const mluOpTensorDescriptor_t mask_desc,
                                       const void *mask,
                                       const mluOpTensorDescriptor_t num_valid_desc,
                                       const void *num_valid,
                                       const mluOpTensorDescriptor_t idx_desc,
                                       void *idx);
```

## 3 实现方案设计

### 3.1 实现方案

#### 3.1.1 计算原理说明

1. 该算子为 `diff_iou_rotated` 整个计算过程中的一步，主要完成对两个box相交构成的多边形有效顶点坐标排序，输出排序后顶点坐标索引。

2. 该算子有三个输入tensor，一个输出tensor，输入`vertices` 维度`[B, N, 24, 2]`，输入`mask` 维度`[B, N, 24]`，输入`num_valid` 维度`[B, N]`，输出`idx` 维度`[B, N, 9]`。

3. 参考接口计算原理说明：

   1. 从输入`mask[B, N, 8:]` 开始获取第一个mask值为0的索引，记为 `pad`，用于给输出`idx`填充无效索引，参考接口代码实现如下：

      ```c
      #define INTERSECTION_OFFSET 8
      int pad;  // index of arbitrary invalid intersection point (not box corner!)
      int m = 24;
      for (int j = INTERSECTION_OFFSET; j < m; ++j) {
      	if (!mask[i * m + j]) {
      		pad = j;
      		break;
      	}
      }
      ```

   2. 判断有效顶点数是否可以构成三角形，即判断 `num_valid[i]` 是否大于3（`num_valid` 合法取值范围`[0, 8]`）
      1. 若`num_valid[i]<3` 则将输出`idx[i][9]` 全部初始化为 `pad`，参考接口代码实现如下：

         ```c
         if (num_valid[i] < 3) {
           // not enough vertices, take an invalid intersection point
           // (zero padding)
           for (int j = 0; j < MAX_NUM_VERT_IDX; ++j) {
             idx[i * MAX_NUM_VERT_IDX + j] = pad;
           }
         }
         ```

      2. 若`num_valid[i]>=3` 则循环`num_valid[i]`次，从小到大（按顶点与坐标原点的夹角从小到大排序）依次保存有效顶点索引到`idx`中。顶点排序原理见下文 “4 顶点排序原理说明”；

      3. 将`idx[i][num_valid[i]]` 赋值为 `idx[i][0]`，构成封闭多边形 ，其余输出`idx[i][num_valid[i]+1:9]` 赋值为 `pad`；

   3. 特殊情况处理：当两个`box` 完全相同时，`idx`输出其中一个 `box` 的四个角点索引，其余赋值为 `pad`；

4. 顶点排序原理说明：

   - 通过分析参考接口的算法实现，算法按逆时针排序所有有效顶点（类似Graham扫描法），如下图所示：

     ![diff_iou_rotated_sort_vertices_forward](./res/sort_vertices.png)

     排序结果（按夹角从小到大排序）：`[p0, p1, ..., p7]`

     算子输出：将排序后顶点索引保存到 `idx` 中。

     参考接口代码实现（部分代码）如下：

     ```c
     // sort the valid vertices
     // note the number of valid vertices is known
     // note: check that num_valid[i] < MAX_NUM_VERT_IDX
     #define MAX_NUM_VERT_IDX 9
     #define EPSILON 1e-8
     // 依次遍历所有顶点
     int m = 24;
     for (int j = 0; j < num_valid[i]; ++j) {
       // initialize with a "big" value
       float x_min = 1;
       float y_min = -EPSILON;
       int i_take = 0;
       int i2;
       float x2, y2;
       if (j != 0) {
         i2 = idx[i * MAX_NUM_VERT_IDX + j - 1];
         x2 = vertices[i * m * 2 + i2 * 2 + 0];
         y2 = vertices[i * m * 2 + i2 * 2 + 1];
       }
       // 找到夹角最小的点，保存到idx[i * MAX_NUM_VERT_IDX + j] 中
       for (int k = 0; k < m; ++k) {
         float x = vertices[i * m * 2 + k * 2 + 0];
         float y = vertices[i * m * 2 + k * 2 + 1];
         if (mask[i * m + k] && compare_vertices(x, y, x_min, y_min)) {
           if ((j == 0) || (j != 0 && compare_vertices(x2, y2, x, y))) {
             x_min = x;
             y_min = y;
             i_take = k;
           }
         }
       }
       idx[i * MAX_NUM_VERT_IDX + j] = i_take;
     }
     ```
   
     其中 compare_vertices 接口用于比较两个点的大小关系。
   
   -  compare_vertices 接口说明：通过比较两个顶点 `cosθ`平方的大小（保留正负号），来判断两个点的大小关系。
   
     如下图中 p0 和 p1 点 的大小关系，调用 compare_vertices(p0_x, p0_y, p1_x, p1_y) ，结果返回 true，表示p0 < p1。
     
     ![diff_iou_rotated_sort_vertices_forward](./res/sort_vertices1.png)
     
     参考接口代码实现如下：
     
     ```c
     /*
     compare normalized vertices (vertices around (0,0))
     if vertex1 < vertex2 return true.
     order: minimum at x-aixs, become larger in anti-clockwise direction
     */
     #define EPSILON 1e-8
     bool compare_vertices(float x1, float y1, float x2, float y2) {
       if (fabs(x1 - x2) < EPSILON && fabs(y2 - y1) < EPSILON)
         return false;  // if equal, return false
     
       if (y1 > 0 && y2 < 0) return true;
       if (y1 < 0 && y2 > 0) return false;
     
       // 通过计算 cosθ 的平方的大小，来判断两个点的大小关系
       float n1 = x1 * x1 + y1 * y1 + EPSILON;
       float n2 = x2 * x2 + y2 * y2 + EPSILON;
       float diff = fabs(x1) * x1 / n1 - fabs(x2) * x2 / n2;
     
       if (y1 > 0 && y2 > 0) {
         if (diff > EPSILON)
           return true;
         else
           return false;
       }
       if (y1 < 0 && y2 < 0) {
         if (diff < EPSILON)
           return true;
         else
           return false;
       }
       return false;
     }
     ```
     
     
     

#### 3.1.2 nram 空间划分

- 计算阶段所需额外的nram空间 `nram_pub_space`，大小 `3 * deal_num * 24`；

- 采用三级流水实现，因此将nram空间划分为两份：每一份的大小`max_nram_size/2`，其中一份的空间划分如下：![diff_iou_rotated_sort_vertices_forward](./res/nram_space.png)

  `max_nram_size` 等于 `MAX_NRAM_SIZE - nram_pub_space`的nram空间大小：

  ```c
  int max_nram_size = MAX_NRAM_SIZE -  3 * deal_num * 24 * sizeof(T);
  ```

- `deal_num` 计算如下：

  ```c
  int deal_num = MAX_NRAM_SIZE /(7 * 24 * sizeof(T) + 2 * 24 * sizeof(bool) + 2 * (1 + 9) * sizeof(int));
  int deal_vertices_num = deal_num * 24 * 2;
  int deal_mask_num = deal_num * 24;
  int deal_num_valid_num = deal_num;
  int max_nram_size = MAX_NRAM_SIZE -  3 * deal_num * 24 * sizeof(T);
  int pingpong_offset = max_nram_size / 2;
  
  T *nram_pub_space = (T *)nram_buffer;
  
  // ping/pong
  char *nram_vertices = (char *)(nram_pub_space + 3 * deal_num * 24);
  char *nram_mask = nram_vertices + deal_vertices_num * sizeof(T);
  char *nram_num_valid = nram_mask + deal_mask_num * sizeof(bool);
  char *nram_idx = nram_num_valid + deal_num_valid_num * sizeof(int);
  ```

#### 3.1.3 实现方案

- 采用三级流水实现

- 实现过程：

  1. 任务拆分：根据3.3 拆分(任务拆分，多核拆分)章节介绍，对`B*N`进行拆分；

     ```c
     // total_bn_num
     int total_bn_num = B * N;
     ```
  
  2. 初始化阶段
  
     1. nram空间划分：根据3.1.2 nram空间划分，计算得到`deal_num`的大小；
  
     2. 根据 `total_bn_num` 和 `deal_num` 计算 repeat_n 和 rem_num
  
        此处采用核间访存连续的方案，数据切分如下：

        ```c
          int repeat_n = total_bn_num / (deal_num * taskDim);
          int rem_num_device = total_bn_num % (deal_num * taskDim);
          int rem_num_per_task = rem_num_device / taskDim;
          int rem_bn_num = rem_num_device % taskDim;
          int rem_num = rem_num_per_task + (int)((taskId < rem_bn_num));
          int rem_offset =
              taskId * rem_num_per_task + ((taskId < rem_bn_num) ? taskId : rem_bn_num);
        ```
  
  3. 处理阶段：三级流水LCS
  
     1. 循环处理
  
        ```c
        // 三级流水计算过程
        if (repeat_n > 0) {
            // L[0]：load
            // load vertices数据，地址 vertices_addr = base_vertices，大小 deal_num * 24 * 2
            // load mask，地址 mask_addr = base_mask，大小 deal_num * 24
            // load num_valid，地址 mask_addr = base_num_valid，大小 deal_num
            __sync();
        }
        if (repeat_n > 1) {
            // L[1]
            // C[0]：计算
            __sync();
        }
        for (int n_iter = 0; n_iter < repeat_n - 2; ++n_iter) {
            // S[n_iter]store
            // L[n_iter + 2]
            // C[n_iter + 1]
            __sync();
        }
        if (repeat_n >= 2) {
        	// S[repeat_n - 2]
        }
        if (rem_num > 0) {
        	// L[repeat_n]
        }
        if (repeat_n > 0) {
        	// C[repeat_n - 1]
        }
        __sync();
        
        if (repeat_n > 0) {
        	// S[repeat_n - 1]
        }
        if (rem_num > 0) {
            // C[repeat_n]
            __sync();
            // S[repeat_n]
        }
        ```
     
     2. Load/Store 时GDRAM地址计算
     
        采用核间连续访存，load和store时，各个输入输出的GDRAM地址计算需加上`deal_num_device`的偏移，伪代码如下：
     
        ```c
        const uint32_t deal_num_device = taskDim * deal_num;
        const uint32_t task_data_offset = taskId * deal_num;
        if (repeat_n > 0) {
            // L[0]：load
            const T *addr_vertices = base_vertices + task_data_offset * dim_m * 2;
            const bool *addr_mask = base_mask + task_data_offset * dim_m;
            const int *addr_num_valid = base_num_valid + task_data_offset;
            load(addr_vertices, addr_mask, addr_num_valid, nram_vertices, nram_mask,
                 nram_num_valid, dim_m, deal_num, pingpong_offset, 0);
            __sync();
        }
        if (repeat_n > 1) {
            // L(1)
            const T *addr_vertices =
                base_vertices + (deal_num_device + task_data_offset) * dim_m * 2;
            const bool *addr_mask =
                base_mask + (deal_num_device + task_data_offset) * dim_m;
            const int *addr_num_valid =
                base_num_valid + deal_num_device + task_data_offset;
            load(addr_vertices, addr_mask, addr_num_valid, nram_vertices, nram_mask,
                 nram_num_valid, dim_m, deal_num, pingpong_offset, 1);
        
            // Compute(0)
            __sync();
        }
        // ...
        if (repeat_n >= 2) {
            // S
            int *addr_idx =
                base_idx + ((repeat_n - 2) * deal_num_device + task_data_offset) *
                MAX_NUM_VERT_IDX;
            store(addr_idx, nram_idx, deal_idx_num, pingpong_offset, (repeat_n - 2));
        }
        if (rem_num > 0) {
            // L
            const T *addr_vertices =
                base_vertices + (repeat_n * deal_num_device + rem_offset) * dim_m * 2;
            const bool *addr_mask =
                base_mask + (repeat_n * deal_num_device + rem_offset) * dim_m;
            const int *addr_num_valid =
                base_num_valid + repeat_n * deal_num_device + rem_offset;
            load(addr_vertices, addr_mask, addr_num_valid, nram_vertices, nram_mask,
                 nram_num_valid, dim_m, rem_num, pingpong_offset, repeat_n);
        }
        ```
     
        
     
     3. 计算过程详细说明：
     
        1. 提前计算所有顶点的cosθ 的平方
     
           设点`p0(x, y)`， 计算 `f(p0) = abs(x) * x / (x * x + y * y)`：
           代码实现：compute_cosine 函数伪代码
     
           ```c
           template <typename T>
           static __mlu_func__ void compute_cosine(T *nram_vertices,
                                                   const bool *nram_vertices_mask,
                                                   T *nram_mask0, T *nram_temp0,
                                                   T *nram_temp1, const int point_num) {
             // 坐标处理让所有x连续，y连续：[x0,y0,x1,y1,...] -> [x0,x1,...]，[y0,y1,...]
             T *nram_vertices_x = nram_vertices;
             T *nram_vertices_y = nram_vertices + point_num;
             __bang_transpose(nram_temp0, nram_vertices, point_num, 2);
             __bang_move(nram_vertices_x, nram_temp0, 2 * point_num * sizeof(T));
           
             // 计算cosine 
             // 计算 x * x + y * y + EPSILON
             __bang_square(nram_temp0, nram_vertices_x, point_num);
             __bang_square(nram_temp1, nram_vertices_y, point_num);
             __bang_add(nram_temp0, nram_temp0, nram_temp1, point_num);
             __bang_add_scalar(nram_temp0, nram_temp0, (T)EPSILON, point_num);
             // 计算 fabs(x) * x;
             __bang_abs(nram_temp1, nram_vertices_x, point_num);
             __bang_mul(nram_temp1, nram_temp1, nram_vertices_x, point_num);
           
             // 计算cosine = fabs(x) * x / (x * x + y * y + EPSILON)
             computeDiv(nram_temp0, nram_temp1, nram_temp0, point_num);
           }
           ```
     
        2. 预处理 `mask` tensor，以便后续获取 pad
     
           首先将`nram_pad` 的前8个值填1，然后取反，以便在遍历 `deal_num` 时，可以调用 __bang_findfirst1 快速的找到pad，伪代码如下：
     
           ```c
           // set nram_temp0[24] = [1,1,1,1,1,1,1,1,0,0,0,...]
           int dim_m = 24;
           __bang_write_zero(nram_temp0, dim_m);
           __bang_write_value(nram_temp0, INTERSECTION_OFFSET, (T)1.0);
           __bang_int82float(nram_pad, (int8_t *)(nram_mask_p), deal_num * dim_m, 0);
           // 调用 cycle_maxequal 将 nram_pad 的前8个值set为1，其余保持不变，结果存 nram_pad 中
           __bang_cycle_maxequal(nram_pad, nram_pad, nram_temp0, deal_num * dim_m, dim_m);
           // 取反 nram_pad：以便后续调用 __bang_findfirst1 找到第一个不为0的值的索引
           __bang_not(nram_pad, nram_pad,  deal_num * dim_m);
           
           // 遍历
           for (int i = 0; i < deal_num; i++) {
           	int pad = (int)__bang_findfirst1(nram_pad + i * dim_m, dim_m);
           	...
           }
           ```
     
           
     
        3. 获取排序结果：循环 `deal_num` 次，依次找到每个多边形有效顶点排序后的索引，排序时加入提前算好的cos值
     
           ```c
            template <typename T>
            static __mlu_func__ bool compare_vertices(T x1, T y1, T diff_1, T x2, T y2, T diff_2) {
              ...
              //diff 值计算
              T diff = diff_1 - diff_2;
              ...
            }

            //排序计算
            for (int j = 0; j < num_valid_points; ++j) {
              T x_min = 1;
              T y_min = -EPSILON;
              T diff_min = x_min/(x_min + y_min * y_min + EPSILON);
              int i_take = 0;
              int i2;
              T x2, y2;
              T diff_x2;
              if (j != 0) {
                  i2 = idx[j - 1];
                  x2 = vertice_x[i2];
                  y2 = vertice_y[i2];
                  diff_x2 = nram_cos[i2];
              }
              for (int k = 0; k < dim_m; ++k) {
                  T x = vertice_x[k];
                  T y = vertice_y[k];
                  T diff_x = nram_cos[k];
                  if (mask_p[k] &&
                      compare_vertices(x, y, diff_x, x_min, y_min, diff_min)) {
                      if ((j == 0) || (j != 0 && compare_vertices(x2, y2, diff_x2,
                                                                  x, y, diff_x))) {
                          x_min = x;
                          y_min = y;
                          i_take = k;
                          diff_min = diff_x;
                      }
                  }
              }
              idx[j] = i_take;
            }
           ```
     
        4. 额外空间 ：完成上述计算过程，需要额外的nram空间，大小`4 * deal_num * 24`
     
           ```c
           // nram_pub_space size = 3 * deal_num * 24
           T *nram_pad = nram_pub_space;
           T *nram_mask0 = nram_pad + deal_num * 24;
           T *nram_temp1 = nram_temp0 + deal_num * 24;
           ```
     
           

### 3.2 伪代码实现

### 3.3 拆分(任务拆分，多核拆分)

- 基本任务类型为BLOCK的任务。
- 输入`vertices` 维度 `[B, N, 24, 2]`，拆分`B * N`，将`B * N`平均拆分到所有task上处理。

### 3.4 性能优化设计

1、资源分配

| 表项 | 分配策略                                    |
| ---- | ------------------------------------------- |
| NRAM | 分 ping pong 两部分，ping 计算的时，pong IO |

2、流水设计

- 采用三级流水设计，L C S 之间排流水，即 GDRAM2NRAM、Compute、NRAM2GDRAM

  - `L:GDRAM2NRAM`
  - `S:NRAM2GDRAM`
  - `C:Compute`

  ![diff_iou_rotated_sort_vertices_forward](./res/pingpong.png)

### 3.5 可维护性设计

1、bangc 代码中加入必要的 log 信息，比如输入的规模、数据类型、layout 这些，以及如果出错会导致程序 core dump 的变量，比如 IO 指令的 data_size、dim xyz 的值等，这些信息都是有利于快速定位问题；

2、对每一个函数命名变量命名都有充分的注释；

3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代。

### 3.6 测试用例设计

- 算子在网络中用到的规模：

  - **input**

    ```c
    vertices: [1, 211, 24, 2]
    mask: [1, 211, 24]
    num_valid: [1, 211]
    ```

  - **output**

    ```c
    idx: [1, 211, 9]
    ```

- 边界 case：

  - 0元素测试

    ```c
    case1：B = 0
    tensor: vertices: [0, 211, 24, 2]
            mask: [0, 211, 24]
            num_valid: [0, 211]
            idx: [0, 211, 9]
    case2：N = 0
    tensor: vertices: [1, 0, 24, 2]
            mask: [1, 0, 24]
            num_valid: [1, 0]
            idx: [1, 0, 9]
    ```


其他可根据需要进行补充。算子开发完毕后，补充测试报告链接。

### 3.7 算子防呆检查

- 描述符指针为空防呆：handle、vertices_desc、 mask_desc、num_valid_desc、idx_desc；
- 对输入输出支持的 dtype、layout 以及 shape 进行防呆
  1. dtype防呆：vertices_desc：仅支持float32；
  2. dtype防呆：mask_desc：仅支持bool；
  3. dtype防呆：num_valid_desc、idx_desc：仅支持int32_t；
  4. dim防呆：
     1. B = vertices_desc.dims[0] == mask_desc.dims[0] == num_valid_desc.dims[0] == idx_desc.dims[0]；
     2. N = vertices_desc.dims[1] == mask_desc.dims[1] == num_valid_desc.dims[1] == idx_desc.dims[1]；
     3. vertices_desc.dims[2] == mask_desc.dims[2] == 24；
     4. vertices_desc.dims[3] == 2；
     5. idx_desc.dims[2] == 9；
- 0 元素检查防呆：N=0 返回 MLUOP_STATUS_SUCCESS，其余情况直接返回 MLUOP_STATUS_BAD_PARAM；
- 指针为空防呆：对vertices、mask、num_valid、idx指针为空防呆检查；
- large tensor防呆 （参考接口不支持large tensor）：对vertices_desc、 mask_desc、num_valid_desc、idx_desc 进行large tensor 防呆检查；

## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

暂无。

### 4.2 已经过优化的规模说明

暂无。
