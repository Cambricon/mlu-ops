# mluOpVoxelPoolingForward 算子开发设计方案

- #### 文档基本信息

| 算子名称    | voxel_pooling_forward |
| ----------- | --------------------- |
| 编制人/日期 | 张皓喆/2022-11-23     |
| 审批人/日期 | 卜德飞/2022-11-28     |
| 审批人/日期 | 涂德江/2022-11-28     |
| 审批人/日期 | 王远/2022-11-28       |

- #### 修改记录

| 版本号 | 修订人 | 修订日期   | 修订描述 |
| ------ | ------ | ---------- | -------- |
| V1.0   | 张皓喆 | 2022-11-18 | 首次提交 |

- #### 内容描述

本文档为`voxel_pooling_forward`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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

| 算子功能简介                                                                 | 用于 BEVDepth 网络，将相同 x,y 坐标上的特征值相加，再投射到对应坐标上的 bev 2D 区域内           |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 需求来源                                                                     | PyTorch                                                                                         |
| 应用网络                                                                     | BEVDepth                                                                                        |
| 输入数据类型                                                                 | geom_xyz:int<br />input_features:float<br />output_features:float<br />pos_memo:int             |
| 输入标量参数                                                                 | batch_size<br />num_points<br />num_channels<br />num_voxel_x<br />num_voxel_y<br />num_voxel_z |
| 输入 Shape                                                                   | geom_xyz:[B,N,3]<br />input_features:[B,N,C]                                                    |
| 输入 Layout                                                                  | geom_xyz:ARRAY<br />input_features:ARRAY                                                        |
| 输出数据类型                                                                 | output_features:float<br />pos_memo:int                                                         |
| 输出 Shape                                                                   | output_features:[B,H,W,C]<br />pos_memo:[B,N,3]                                                 |
| 输出 Layout                                                                  | output_features:ARRAY<br />pos_memo:ARRAY                                                       |
| 模式(可选）                                                                  | 无                                                                                              |
| 是否含有 dim/axis 等类似语义的参数且该参数支持负数/其他特殊处理              | 不含带 dim/axis 语义的参数等                                                                    |
| 是否含有 labels/index 等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 不含带 labels/index 语义的参数等                                                                |
| 是否需要支持原位                                                             | 否                                                                                              |
| 是否需要支持 stride 机制                                                     | 否                                                                                              |
| 是否需要支持广播                                                             | 否                                                                                              |
| 0 元素检查是否直接返回                                                       | 是 (返回 MLUOP_STATUS_BAD_PARAM)                                                                |
| 其他特殊需求(在线量化，融合，转数提前等，可选)                               | 无                                                                                              |
| 本次开发优先支持的规模/模式                                                  | 无                                                                                              |

### 1.2 算子功能和应用场景描述

**算子功能：** `voxel_pooling_forward`算子属于 Bev 网络，Bev 网络是把各个相机的 2D 图片先转成 3D，中心思想是把 N 个相机得到的 3D 特征，转化成鸟瞰的 2D 特征，即把相同 x,y 坐标上的特征值相加，再投射到对应坐标上的 bev 2D 区域内。
**应用场景：** `voxel_pooling_forward`算子应用于`BEVDepth`网络。算子来源自：BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection(https://arxiv.org/abs/2206.10092)
备注：

1、输入 `geom_xyz` 数据类型为 int ，不涉及 nan,inf，`input_features` 支持 nan 或 inf

### 1.3 算子输入输出参数要求

| 参数                     | 语义                                              | 类型（输入/输出） | 支持类型                | 物理布局  | 规模限制 |
| ------------------------ | ------------------------------------------------- | ----------------- | ----------------------- | --------- | -------- |
| **handle**               | 操作句柄                                          | 输入              | mluOpHandle_t           | /         | /        |
| **batch_size**           | 输入数据，batch_size 大小                         | 输入              | int                     | scalar    | /        |
| **num_points**           | 输入数据，体素的数量                              | 输入              | int                     | scalar    | /        |
| **num_channels**         | 输入数据，体素的特征的通道数                      | 输入              | int                     | scalar    | /        |
| **num_voxel_x**          | 输入数据，体素在 X 轴上的坐标最大范围，对应的是 W | 输入              | int                     | scalar    | /        |
| **num_voxel_y**          | 输入数据，体素在 Y 轴上的坐标最大范围，对应的是 H | 输入              | int                     | scalar    | /        |
| **num_voxel_z**          | 输入数据，体素在 Z 轴上的坐标最大范围，对应的是 1 | 输入              | int                     | scalar    | /        |
| **geom_xyz_desc**        | 输入数据，体素的坐标的描述符                      | 输入              | mluOpTensorDescriptor_t | /         | /        |
| **geom_xyz**             | 输入数据，体素的坐标                              | 输入              | int\*                   | [B,N,3]   | /        |
| **input_features_desc**  | 输入数据，体素的特征值的描述符                    | 输入              | mluOpTensorDescriptor_t | /         | /        |
| **input_features**       | 输入数据，体素的特征值                            | 输入              | float\*                 | [B,N,C]   | /        |
| **output_features_desc** | 输入数据，体素的池化后的特征图的描述符            | 输入              | mluOpTensorDescriptor_t | /         | /        |
| **output_features**      | 输出数据，体素的池化后的特征图                    | 输出              | float\*                 | [B,H,W,C] | /        |
| **pos_memo_desc**        | 输入数据，体素的位置信息的描述符                  | 输入              | mluOpTensorDescriptor_t | /         | /        |
| **pos_memo**             | 输出数据，体素的位置信息                          | 输出              | int\*                   | [B,N,3]   | /        |

### 1.4 算子限制

| 限制类型     | 详细说明                                                                                                   |
| ------------ | ---------------------------------------------------------------------------------------------------------- |
| 输入限制     | 输入 `geom_xyz` 必须满足 dims=3，并且 dim[0]=batch_size, dim[1]=num_points, dim[2]=3                       |
| 输入限制     | 输入 `input_features` 必须满足 dims=3，并且 dim[0]=batch_size, dim[1]=num_points, dim[2]=num_channels      |
| 输入限制     | 输入 `batch_size`,`num_points`,`num_channels`,`num_voxel_x`,`num_voxel_y` 对应输入 tensor 的维度必须大于 0 |
| 输入限制     | 输入 `geom_xyz` 属于 int 类型坐标值，不涉及 nan 或 inf，`input_features` 支持 nan 或 inf                   |
| 输入限制     | 输出 `pos_memo` 需要调用算子之前对输出空间置初始值，且初始化为负数                                         |
| 数据类型限制 | `geom_xyz` 只支持 int 输入，`input_features` 只支持 float 输入                                             |
| 精度限制     | 由于采用 atomicAdd，存在乱序计算，可能存在精度问题                                                         |
| 原位限制     | 不支持原位                                                                                                 |
| stride 限制  | 不支持 stride                                                                                              |
| 广播限制     | 不支持广播                                                                                                 |
| 架构限制     | 不支持 mlu 200 系列                                                                                        |

### 1.5 验收标准

#### 1.5.1 精度验收标准

该算子属于 atomicAdd 类算子，多次运行结果可能不一致，采用阈值标准：diff1<=3e-3 && diff2 <= 3e-3

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../MLU-OPS-Performance-Acceptance-Standard.md)。

## 2 算子接口设计

### 2.1 参考接口

- PyTorch

```c++
int voxel_pooling_forward_wrapper(int batch_size, int num_points,
                                  int num_channels, int num_voxel_x,
                                  int num_voxel_y, int num_voxel_z,
                                  at::Tensor geom_xyz_tensor,
                                  at::Tensor input_features_tensor,
                                  at::Tensor output_features_tensor,
                                  at::Tensor pos_memo_tensor);
```

### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API mluOpVoxelPoolingForward(mluOpHandle_t handle,
                                                     const int batch_size,
                                                     const int num_points,
                                                     const int num_channels,
                                                     const int num_voxel_x,
                                                     const int num_voxel_y,
                                                     const int num_voxel_z,
                                                     const mluOpTensorDescriptor_t geom_xyz_desc,
                                                     const void *geom_xyz,
                                                     const mluOpTensorDescriptor_t input_features_desc,
                                                     const void *input_features,
                                                     const mluOpTensorDescriptor_t output_features_desc,
                                                     void *output_features,
                                                     const mluOpTensorDescriptor_t pos_memo_desc,
                                                     void *pos_memo);
```

## 3 实现方案设计

### 3.1 实现方案

0. 首先明确几个变量含义：

​​nram_limit_pt_num 表示一次最多可以 load 到 nram 上的点的数量；

nram_limit_channels 表示一次 load nram_limit_pt_num 个点的通道数量；

具体计算过程如下：

```c
// 将nram空间划分为9份 nram_limit_pt_num 大小，处理完毕 nram_limit_pt_num 个点后，剩余 6 份空间可以用来 load 该点的通道。
int nram_limit_pt_num = MAX_NRAM_SIZE / sizeof(int) / 9;
int nram_limit_channels = nram_limit_pt_num * 6;
```

1. 将 `geom_xyz(batch_size, num_points, 3)` 一次 load nram_limit_pt_num 个点到 nram 上

再 transpose 成 `(3, batch_size, num_points)`，这样就可以分别得到 xyz 维度的坐标：

x 维度坐标： `geom_xyz_offset_x = geom_xyz`；

y 维度坐标： `geom_xyz_offset_y = geom_xyz + batch_size * num*points`；

z 维度坐标： `geom_xyz_offset_z = geom_xyz + batch_size * num_points * 2`；

2. 分别向量化 计算 x y z 坐标 在 (0,`num_voxel_x`) 、(0,`num_voxel_y`) 、(0,`num_voxel_z`) 范围内得到 mask_temp_x 、mask_temp_y 、mask_temp_z ：

```c
// x > 0 , y > 0 , z > 0
__bang_gt_scalar(mask_temp_x0, geom_xyz_offset_x, 0, nram_limit_pt_num);
__bang_gt_scalar(mask_temp_y0, geom_xyz_offset_y, 0, nram_limit_pt_num);
__bang_gt_scalar(mask_temp_z0, geom_xyz_offset_z, 0, nram_limit_pt_num);
// x < num_voxel_x , y < num_voxel_y, z < num_voxel_z
__bang_lt_scalar(mask_temp_x1, geom_xyz_offset_x, num_voxel_x, nram_limit_pt_num);
__bang_lt_scalar(mask_temp_y1, geom_xyz_offset_y, num_voxel_y, nram_limit_pt_num);
__bang_lt_scalar(mask_temp_z1, geom_xyz_offset_z, num_voxel_z, nram_limit_pt_num);
// 0 < x < num_voxel_x , 0 < y < num_voxel_y, 0 < z < num_voxel_z
__bang_and(mask_temp_x, mask_temp_x0, mask_temp_x1, nram_limit_pt_num);
__bang_and(mask_temp_y, mask_temp_y0, mask_temp_y1, nram_limit_pt_num);
__bang_and(mask_temp_z, mask_temp_z0, mask_temp_z1, nram_limit_pt_num);
```

3. 然后做向量化的与操作 \_\_bang_and() 得到 pt_in_voxel_mask ：

```c
__bang_and(pt_in_voxel_mask, mask_temp_x, mask_temp_y, nram_limit_pt_num);
__bang_and(pt_in_voxel_mask, pt_in_voxel_mask, mask_temp_z, nram_limit_pt_num);
```

- 说明： 第 2，3 步中所用到的 temp 非实际需要的数量，这里仅为了清晰代码逻辑。实际仅使用 2 个 temp 完成。

4. 处理 `pos_memo` 输出：

for 循环 nbatch( = nram_limit_pt_num / num_points ) 次，循环索引 batch_idx :

```c
for (batch_idx = 1, batch_idx < nram_limit_pt_num / num_points, ++batch_idx) {
  bang_write_value(pos_memo, num_points, batch_idx);
}
// 处理余数段
rem_points = nram_limit_pt_num % num_points,
if (rem_points > 0) {
  bang_write_value(pos_memo, rem_points, batch_idx + 1);
}
// copy x y 坐标
memcpy(pos_memo_offset_x, geom_xyz_offset_x, nram_limit_pt_num, nram2nram);
memcpy(pos_memo_offset_y, geom_xyz_offset_y, nram_limit_pt_num, nram2nram);
```

从 nram 上 copy geom_xyz 的 nram_limit_pt_num 个点的 y 坐标以及 x 坐标到 pos_memo nram 空间部分上，

至此， 得到 pos_memo_temp(3, nram_limit_pt_num) 所有坐标的索引值，接下来进行第 5 步处理非体素范围内的点为初始默认值；

5. 调用 \_\_bang_mul() 将 pt_in_voxel_mask 与 pos_memo 中的 batch、x、y 依次做乘法，

调用 \_\_bang_not() 将 pt_in_voxel_mask 取反、\_\_bang_mul_scalar() 乘 initial_value ，

调用 \_\_bang_cycle_add() 将 pt_in_voxel_mask 与 pos_memo 中的 batch、x、y 依次做加法，

使得 pt_in_voxel_mask 为 0 的位置置为初始值(由框架指定，默认为-1)，这时得到 `pos_memo(3, nram_limit_pt_num)`，

再调用 \_\_bang_transpose() 后，将 `pos_memo` 转置成 (nram_limit_pt_num, 3)，

每个 batch 循环完毕后 store 至 gdram 输出空间:

```c
  __memcpy(pos_memo + pt_idx_cur_loop * 3,
          nram_pos_memo,
          actual_pt_num * 3 * sizeof(int),
          NRAM2GDRAM);
```

6. 处理 `output_features` 输出：

6.1 计算每一点的在输出 tensor 中的偏移量

```c
// 使用 第 4 步 的 pos_memo_temp 计算在 `output_features` 上的点的地址偏移
// output_features_pt_offset_addr = (batch_idx * num_voxel_y * num_voxel_x + y * num_voxel_x + x) * num_channels
__bang_mul_scalar(nram_buffer_temp1, nram_pos_memo_batch, num_voxel_y * num_voxel_x, actual_pt_num); // batch_idx * num_voxel_y * num_voxel_x
__bang_mul_scalar(nram_buffer_temp2, nram_pos_memo_y, num_voxel_x, actual_pt_num);
__bang_add(nram_buffer_temp1, nram_buffer_temp1, nram_buffer_temp2, actual_pt_num);  // y * num_voxel_x
__bang_add(nram_buffer_temp1, nram_buffer_temp1, nram_pos_memo_x, actual_pt_num); // x
__bang_mul_scalar(nram_buffer_temp1, nram_buffer_temp1, num_channels, actual_pt_num);
// 这里 nram_buffer_temp1 就是 点在输出 tensor 中的偏移地址
int *output_features_pt_offset_addr = nram_buffer_temp1;
```

- 说明： 这里所用到的 temp 非实际需要的数量，这里仅为了清晰代码逻辑，实际可以复用 buffer_temp 完成。

  6.2 循环 load 每一个在体素内的点的 nram_limit_channels 个特征到 nram 上，actual_channels_num 表示实际 load 的 channel 数量

```c
int actual_channels_num = (channels_loop_idx == channels_loop_times)
                              ? rem_channels
                              : nram_limit_channels;
for(int pt_idx = 0; pt_idx < actual_pt_num; ++pt_idx){
  //从 feature_offset[pt_idx] 中取特征值的偏移量
  int output_features_pt_offset = output_features_pt_offset_addr[pt_idx];
  // 若 output_features_pt_offset < 0 表示该点不在体素内
  if( output_features_pt_offset < 0 ){ continue; }
  // 拆分channels，共 load channels_loop_times 次
  for (int channels_loop_idx = 0; channels_loop_idx <= channels_loop_times; ++channels_loop_idx) {
      // load input_features
      __memcpy(nram_input_features,
              input_features_pt_addr + channels_offset,
              actual_channels_num * sizeof(float),
              GDRAM2NRAM);
      // 调用 __bang_atomic_reduce_add，完成特征值累加计算
      __bang_atomic_reduce_add(output_features_pt_addr + channels_offset,
                              nram_input_features,
                              actual_channels_num);
    }
  }
```

### 3.2 伪代码实现

见 3.1 实现方案小节。

这里给出 cpu 实现供理解和参考：

```c
// 输入 tensor : (int)geom_xyz: (batch_size, num_points, 3)
// 输入 tensor : (float)input_features: (batch_size, num_points, channels)
// 输出 tensor : (float)output_features: (batch_size, num_voxel_y, num_voxel_x, channels)
// 输出 tensor : (int)pos_memo: (batch_size, num_points, 3)

for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
  for (int pt_idx = 0; pt_idx < num_points; ++pt_idx) {
    int x = geom_xyz[batch_idx * num_points * 3 + pt_idx * 3];
    int y = geom_xyz[batch_idx * num_points * 3 + pt_idx * 3 + 1];
    int z = geom_xyz[batch_idx * num_points * 3 + pt_idx * 3 + 2];
    // if coord of current voxel is out of boundary, return.
    if (x < 0 || x >= num_voxel_x || y < 0 || y >= num_voxel_y || z < 0 ||
        z >= num_voxel_z) {
      continue;
    }
    pos_memo[batch_idx * num_points * 3 + pt_idx * 3] = batch_idx;
    pos_memo[batch_idx * num_points * 3 + pt_idx * 3 + 1] = y;
    pos_memo[batch_idx * num_points * 3 + pt_idx * 3 + 2] = x;

    int in_offset = (batch_idx * num_points + pt_idx) * num_channels;
    int out_offset = (batch_idx * num_voxel_y * num_voxel_x + y * num_voxel_x + x) * num_channels;

    for (int c_idx = 0; c_idx < num_channels; ++c_idx) {
      output_features[out_offset + c_idx] += input_features[in_offset + c_idx];
    }
  }
}
```

### 3.3 拆分(任务拆分，多核拆分)

**任务类型 union1**

多核拆分以 batch_size \* num_points 进行均匀拆分，将点的数量的总和 total_num_points 均分到每个 core 中。

### 3.4 性能优化设计

1、资源分配

| 表项 | 分配策略                                               |
| ---- | ------------------------------------------------------ |
| NRAM | geom_xyz[nram_limit_pt_num,3]                          |
|      | output_features[nram_limit_pt_num,nram_limit_channels] |
|      | pos_memo [nram_limit_pt_num,3]                         |
|      | buffer_temp[nram_limit_pt_num] \* 3                    |

- 具体 nram 空间划分见 device 端代码注释。

2、流水设计

该算子主要时间耗费在 atomic_add 属于 IO 瓶颈，暂不考虑排流水。

### 3.5 可维护性设计

1、bangc 代码中加入必要的 log 信息，比如输入的规模、数据类型、layout 这些，以及如果出错会导致程序 core dump 的变量，比如 IO 指令的 data_size、dim xyz 的值等，这些信息都是有利于快速定位问题；

2、对每一个函数命名变量命名都有充分的注释；,

3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代。

### 3.6 测试用例设计

- 算子在网络中用到的规模：
  <br /> batch_size=2
  <br /> num_points=473088
  <br /> num_channels=80
  <br /> voxel_num[0]=128
  <br /> voxel_num[1]=128
  <br /> voxel_num[2]=1
  <br /> geom_xyz=[2,473088,3]
  <br /> input_features=[2,473088,80]
  <br /> output_features=[2,128,128,80],
  <br /> pos_memo=[2,473088,3]

其他可根据需要进行补充。算子开发完毕后，补充测试报告链接。

### 3.7 算子防呆检查

1、对当前运行架构平台进行防呆，不支持 200 系列。

2、对输入的 tensor 描述符以及空指针进行防呆；

3、对输入输出支持的 dtype、layout 以及 shape 进行防呆；

4、对 large tensor 进行防呆，暂不支持 2G 以上 num 的 tensor；

5、算子存在的自身的相关参数防呆，需要保证传入的标量参数与 tensor 参数的含义一致。

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

### 4.2 已经过优化的规模说明

## 5 方案实施

### 5.1 开发测试计划

- **总体计划**：2022.11.15~2022.12.09 算子开发 共 3 周
- **拆解计划**：2022.11.15~2022.11.18 需求分析以及设计文档撰写 4 天
- 2022.11.21~2022.11.23 设计文档 review 评审、generator 以及 gtest 开发 3 天
- 2022.11.24~2022.11.29 主体代码实现 4 天
- 2022.11.30~2022.12.05 大规模测试，撰写测试报告 4 天
- 2022.12.06~2022.12.08 提交 MR，代码 review 意见修改、算子入库 3 天

### 5.2 风险分析
