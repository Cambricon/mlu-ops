# roiaware_pool3d_forward 算子开发设计方案

- #### 文档基本信息

| 算子名称    | roiaware_pool3d_forward |
| ----------- | ----------------------- |
| 编制人/日期 | 张皓喆/2022-5-30        |
| 审批人/日期 | 张少鹏/2022-6-15        |
| 审批人/日期 | 王远/2022-6-15          |
| 审批人/日期 | 周晨阳/2022-6-15        |

- #### 修改记录

| 版本号 | 修订人 | 修订日期  | 修订描述 |
| ------ | ------ | --------- | -------- |
| V1.0   | 张皓喆 | 2022-5-30 | 首次提交 |

- #### 内容描述

本文档为`roiaware_pool3d_forward`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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

| 算子功能简介                                                                 | 给定一组点和点的特征值，以及一组长方体框，将框中的点的特征进行池化，输出指定数量的体素中的最大或者平均特征值以及点在对应体素中的索引                                                          |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 需求来源                                                                     | mmcv 自定义算子                                                                                                                                                                               |
| 应用网络                                                                     | PartA2                                                                                                                                                                                        |
| 输入数据类型                                                                 | rois、pts、pts_feature 为 float/half                                                                                                                                                          |
| 输入 Shape                                                                   | rois:[boxes_num, 7] <br> pts:[pts_num, 3] <br> pts_feature:[pts_num, channels] <br>                                                                                                           |
| 输入 Layout                                                                  | 均为 ARRAY                                                                                                                                                                                    |
| 输出数据类型                                                                 | pooled_features 为 float/half <br> argmax 为 int32 <br> pts_idx_of_voxels 为 int32                                                                                                            |
| 输出 Shape                                                                   | pooled_features:[boxes_num, out_x, out_y, out_z, channels] <br> argmax:[boxes_num, out_x, out_y, out_z, channels] <br> pts_idx_of_voxels:[boxes_num, out_x, out_y, out_z, max_pts_each_voxel] |
| 输出 Layout                                                                  | 均为 ARRAY                                                                                                                                                                                    |
| 是否含有 dim/axis 等类似语义的参数且该参数支持负数/其他特殊处理              | 否                                                                                                                                                                                            |
| 是否含有 labels/index 等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 否                                                                                                                                                                                            |
| 是否需要支持原位                                                             | 否                                                                                                                                                                                            |
| 是否需要支持 stride 机制                                                     | 否                                                                                                                                                                                            |
| 是否需要支持广播                                                             | 否                                                                                                                                                                                            |
| 0 元素检查是否直接返回                                                       | 是，返回 MLUOP_STATUS_BAD_PARAM，即不支持输入 0 元素                                                                                                                                          |
| 其他特殊需求(在线量化，融合，转数提前等，可选)                               | 无                                                                                                                                                                                            |

### 1.2 算子功能和应用场景描述

该算子属于 MMCV 中 3D 目标检测领域的算子，由 PartA2 网络提出，常用于自动驾驶激光点云 3D 目标检测中，该算子的输入数据的点云是来自雷达 LiDAR 坐标系下的点的数据集。点包含了丰富的信息，包括三维坐标 X，Y，Z、颜色、分类值、强度值、时间等等。本算子的输入数据的点仅涉及到点的坐标。<br>
该算子的主要功能是根据给定的点云数据找出其中关心的 roi 框中的点，若点的数量较多，仅取其中指定数量的若干个点，并根据这些点的最大或者平均特征进行池化，输出池化之后的特征值。

pts 低维度的 3 个数字包括的信息为：X, Y, Z，代表点的三维坐标；boxes_num 代表物体 3D 框 roi 的数量，rois 低维度的 7 个数字用(cx, cy, cz, dx, dy, dz, heading)来表示, 其中 (cx, cy, cz) 为物体 3D 框的`z轴底部`几何中心的位置，(dx, dy, dz)分别为物体 3D 框在 heading 角度为 0 时沿着 x-y-z 三个方向的长度，heading 为物体在俯视图下的朝向角 (沿着 x 轴方向为 0 度角，逆时针 x 到 y 角度增加)。<br>

该算子的实现主要参考 pytorch 实现的点云 3D 目标检测代码库 OpenPCDet。

对于给定的 pt(x, y, z), roi(cx, cy, cz, dx, dy, dz, rz), 检测 pt 是否在 roi 内的公式如下：

```math
in\_flag = \lvert (z - cz) \rvert <= \frac{dz}{2} \ \& \\
\lvert (x - cx) * cos(-rz) - (y - cy) * sin(-rz)\rvert < \frac{dx}{2} \ \& \\
\lvert (x - cx) * sin(-rz) + (y - cy) * cos(-rz)\rvert < \frac{dy}{2}
```

### 1.3 算子输入输出参数要求

| 参数                   | 语义                                                   | 类型（输入/输出） | 支持类型                | 物理布局 | 规模限制                                         |
| ---------------------- | ------------------------------------------------------ | ----------------- | ----------------------- | -------- | ------------------------------------------------ |
| handle                 | 句柄，用于获取当前资源                                 | 输入              | mluOpHandle_t           | /        | /                                                |
| pool_method            | 指定进行池化计算的模式方式，默认为 maxpool             | 输入              | int32                   | /        | /                                                |
| boxes_num              | 所需要进行 roipool 池化的数量                          | 输入              | int32                   | /        | /                                                |
| pts_num                | 点的数量                                               | 输入              | int32                   | /        | /                                                |
| channels               | 点的特征的通道数量                                     | 输入              | int32                   | /        | /                                                |
| rois_desc              | 对选择框 roi 的信息描述，包含维度、布局和数据类型信息  | 输入              | mluOpTensorDescriptor_t | /        | /                                                |
| rois                   | 需要进行池化的选择框，雷达坐标系下的 3D 坐标的指针     | 输入              | float/half              | /        | [boxes_num,7]                                    |
| pts_desc               | 对点云 pts 的信息描述，包含维度、布局和数据类型信息    | 输入              | mluOpTensorDescriptor_t | /        | /                                                |
| pts                    | 存储点云在雷达坐标系下的 3D 坐标的指针                 | 输入              | float/half              | /        | [pts_num,3]                                      |
| pts_feature_desc       | 对点云的特征的信息描述，包含维度、布局和数据类型信息   | 输入              | mluOpTensorDescriptor_t | /        | /                                                |
| pts_feature            | 保存点云的特征的指针                                   | 输入              | float/half              | /        | [pts_num,channels]                               |
| workspace              | 算子所需 workspace 空间                                | 输入              | void \*                 | /        | /                                                |
| workspace_size         | workspace 空间大小                                     | 输入              | size_t                  | /        | /                                                |
| max_pts_each_voxel     | 池化 3dbox 中每个体素，所需要覆盖到的点的最大数量      | 输入              | int32                   | /        | /                                                |
| out_x                  | 池化 3dbox 的体素在 x 维度上的数量                     | 输入              | int32                   | /        | /                                                |
| out_y                  | 池化 3dbox 的体素在 y 维度上的数量                     | 输入              | int32                   | /        | /                                                |
| out_z                  | 池化 3dbox 的体素在 z 维度上的数量                     | 输入              | int32                   | /        | /                                                |
| argmax_desc            | 对取得最大特征值的点的信息描述(仅在最大池化时需要输入) | 输入              | mluOpTensorDescriptor_t | /        | /                                                |
| argmax                 | 保存取得最大特征值的点的索引的指针(仅在最大池化时输出) | 输出              | int32                   | /        | [boxes_num,out_x,out_y,out_z,channels]           |
| pts_idx_of_voxels_desc | 在 roi 中的池化后的 box 中的点的索引的信息描述         | 输入              | mluOpTensorDescriptor_t | /        | /                                                |
| pts_idx_of_voxels      | 在 roi 中池化后的 box 中的点的索引的指针               | 输出              | int32                   | /        | [boxes_num,out_x,out_y,out_z,max_pts_each_voxel] |
| pooled_features_desc   | 对进行池化之后的 box 的特征块的信息描述                | 输入              | mluOpTensorDescriptor_t | /        | /                                                |
| pooled_features        | 保存进行池化之后的 box 的特征 box 的指针               | 输出              | float/half              | /        | [boxes_num,out_x,out_y,out_z,channels]           |

### 1.4 算子限制

| 限制类型     | 详细说明                                                                                                                     |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| 布局限制     | 仅支持上述输入数据 shape 说明的 shape 排列                                                                                   |
| 池化方式限制 | 仅支持 pool_method=0 的最大池化和 pool_method=1 的平均池化方式                                                               |
| 数据类型限制 | `rois`, `pts`, `pts_feature`, `pooled_features`数据类型应为 float/half <br> `argmax`, `pts_idx_of_voxels` 数据类型应为 int32 |
| 原位限制     | 不支持原位                                                                                                                   |
| stride 限制  | 不支持 stride 机制                                                                                                           |
| 广播限制     | 不支持广播                                                                                                                   |
| 输入数据限制 | `rois` 和 `pts` 不支持输入数值为 nan、inf                                                                                    |
| 输入数据限制 | 输入`pts_feature` 在 MLU300 系列上：a. 仅含 INF 时，与 mmcv cuda 结果对齐；b. 当含有 NAN 输入时，结果不对齐                  |
| 数据规模限制 | 数据维数和 dim 信息需要满足 1.3 中规模限制所列含义和维度                                                                     |
| 数据范围限制 | `max_pts_each_voxel`数值在 float 类型下不能超过 2880，half 类型不能超过 2930，否则可能导致 coredump                          |

- 说明: 由于硬件限制，当前计算除法无超越函数指令，计算精度不足，可能与 mmcv 在特定数据的计算结果不一致。

### 1.5 验收标准

#### 1.5.1 精度验收标准

- 按照[精度验收标准](../MLU-OPS-Accuracy-Acceptance-Standard.md)的要求明确本算子的精度标准：

  1. 浮点型输出`pooled_features`适配动态阈值，采用动态阈值标准：diffs = [diff1, diff2, diff4]，threshold_rate = [10, 10, 1]。

  2. 整型输出`argmax`、`pts_idx_of_voxels`适配静态阈值，采用阈值标准 DIFF3 == 0，在 unittest 中配置为 diffs = [diff1, diff2, diff4]，threshold_rate = [0, 0, 1]。

#### 1.5.2 性能验收标准

- 见 [MLU-OPS 性能验收标准](../MLU-OPS-Performance-Acceptance-Standard.md)

- 性能分析：

| 平台                 | 框架版本                                                 | 数据类型 | 数据规模                                                                                                                                                                                                                  | cuda kernel                                                                                                | 计算效率                                   | IO 效率                                | Hardware time(ms)                            |
| -------------------- | -------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------ | -------------------------------------- | -------------------------------------------- |
| Tesla V100-SXM2 16GB | mmcv-1.3.16 + pytorch-1.9 + cuda-11.4 + Driver-440.64.00 | float    | pts: [16000, 3]<br> pts_feature: [16000, 16]<br>rois: [128, 7]<br>max_pts_each_voxel: 128<br>pooled_features: [128, 12, 12, 12, 16]<br> argmax: [128, 12, 12, 12, 16] <br> pts_idx_of_voxels: [128, 12, 12, 12, 128] <br> | generate_pts_mask_for_box3d<br> collect_inside_pts_for_box3d<br> roiaware_maxpool3d<br> roiaware_avgpool3d | 70.29%<br> 0.05%<br> 15.02%<br> 26.57%<br> | 24.28%<br> 0.33%<br> 60.14%<br> 61.63% | 47.33us<br>7.54ms<br>278.40us<br>202.40us    |
| Tesla V100-SXM2 16GB | mmcv-1.3.16 + pytorch-1.9 + cuda-11.4 + Driver-440.64.00 | half     | pts: [16000, 3]<br> pts_feature: [16000, 16]<br>rois: [128, 7]<br>max_pts_each_voxel: 128<br>pooled_features: [128, 12, 12, 12, 16]<br> argmax: [128, 12, 12, 12, 16] <br> pts_idx_of_voxels: [128, 12, 12, 12, 128] <br> | generate_pts_mask_for_box3d<br> collect_inside_pts_for_box3d<br> roiaware_maxpool3d<br> roiaware_avgpool3d | 83.71%<br> 0.04%<br> 16.95% <br>31.50%     | 17.19% <br>0.30%<br> 60.43% <br>61.26% | 69.73us<br> 7.84ms<br> 259.84us<br> 173.28us |

## 2 算子接口设计

### 2.1 参考接口

- MMCV 接口

```c++
// CUDA(mmcv/ops/csrc/pytorch/cuda/roiaware_pool3d_cuda_kernel.cuh):
void RoiawarePool3dForwardCUDAKernelLauncher(
  int boxes_num, int pts_num, int channels, int max_pts_each_voxel, int out_x,
  int out_y, int out_z, const Tensor rois, const Tensor pts,
  const Tensor pts_feature, Tensor argmax, Tensor pts_idx_of_voxels,
  Tensor pooled_features, int pool_method);
```

### 2.2 接口设计

```c++
// 给出MLUOP算子正向接口
mluOpStatus_t MLUOP_WIN_API
mluOpRoiawarePool3dForward(mluOpHandle_t handle,
                           const int pool_method,
                           const int boxes_num,
                           const int pts_num,
                           const int channels,
                           const mluOpTensorDescriptor_t rois_desc,
                           const void *rois,
                           const mluOpTensorDescriptor_t pts_desc,
                           const void *pts,
                           const mluOpTensorDescriptor_t pts_feature_desc,
                           const void *pts_feature,
                           void *workspace,
                           size_t workspace_size,
                           const int max_pts_each_voxel,
                           const int out_x,
                           const int out_y,
                           const int out_z,
                           const mluOpTensorDescriptor_t argmax_desc,
                           void *argmax,
                           const mluOpTensorDescriptor_t pts_idx_of_voxels_desc,
                           void *pts_idx_of_voxels,
                           const mluOpTensorDescriptor_t pooled_features_desc,
                           void *pooled_features);

mluOpStatus_t MLUOP_WIN_API
mluOpGetRoiawarePool3dForwardWorkspaceSize(mluOpHandle_t handle,
                                           const mluOpTensorDescriptor_t rois_desc,
                                           const mluOpTensorDescriptor_t pts_desc,
                                           const mluOpTensorDescriptor_t pts_feature_desc,
                                           size_t *size)
```

## 3 实现方案设计

### 3.1 实现方案

- roiaware_pool3d_forward 算子算法：

  1. 根据 `rois` 坐标，计算 `pts_num` 个点是否在该 roi 中，得到 flag[pts_num,1]，1 为在其中，0 为不在；

  2. 根据每一个 `rois` 坐标和尺寸旋转角，按照 `out_x`、`out_y`、`out_z` 来切分相同数量的 voxels，得到其的切分粒度大小 x_res、y_res、z_res；

  3. 计算每个点在上述粒度下的所归属的 voxels，将该 voxels 的索引称为 voxels_idx，每个 voxel 仅包含前 `max_pts_each_voxel` 个点，超过则丢弃；

  4. 根据每个 voxel 中的点的特征 `pts_feature`，取所有的点中的最大的特征值或者平均特征值，输出在`pooled_features`；

  5. 将每个 voxel 中所取的点的索引输出在 `pts_idx_of_voxels`，将最大池化时在 channels 个通道上分别所取的点的索引输出在 `argmax` 。

- MLU 实现方案如下：

  - 第一阶段 launch pts_idx_of_voxels_kernel：

    1. 将 boxes_num 个 `rois` 均分为 taskDim 份，每个 core 将循环处理 boxes_num_seg 个 roi，读取一个 roi 至片上；

    2. 计算当前数据规模下的 nram 空间一次最多处理`nram_limit_max_pts` 个点，分批次判断这些点是否在 roi 中，计算公式如下：

    ```math
    in\_flag = \lvert (z - cz) \rvert <= \frac{dz}{2} \ \& \\
    \lvert (x - cx) * cos(-rz) - (y - cy) * sin(-rz)\rvert < \frac{dx}{2} \ \& \\
    \lvert (x - cx) * sin(-rz) + (y - cy) * cos(-rz)\rvert < \frac{dy}{2}
    ```

    3. 在每个 core 内循环处理所有的点 `pts_num`，共循环 (`pts_num`/nram*limit_max_pts + 1) 次，每次处理 nram_limit_max_pts 个点，依次 load 点的坐标到 nram 上，将当前 roi 根据 `out_x`、`out_y`、`out_z` 的大小划分成 each_voxel_nums 个 voxels，其值为 `out_x` * `out_y` \_ `out_z` ，根据所有的点的坐标值，计算得到归属在哪一个 voxels ，得到该 voxels 在三个维度的 idx，store `pts_idx_of_voxels`，直到循环完毕所有 `pts_num` 或者达到最大保存数量 `max_pts_each_voxel` 后 break，执行 step4；

    4. 该 roi 计算完毕，返回 step1 处理下一次 roi，直至当前 core 处理完 boxes_num_seg 个 roi。

  - 第二阶段 launch roiaware_pool3d_forward_kernel：

    1. 将 voxels*nums 均分为 taskDim 份，每个 core 将循环处理 voxels_nums_seg 个 voxels，总共有 voxels_nums 个 voxels，其值为 `boxes_num` * `out_x` \_ `out_y` \* `out_z`，将 roi 中按照 `out_x`、`out_y`、`out_z` 划分之后的每一个体素的索引称为 voxels_idx ；

    2. 加载当前 voxel 对应的 `pts_idx_of_voxels` 到片上，获取当前 voxel 中的点数量 `pts_num_in_voxels` 和 点的`idx`，每个 core 每次处理一个 voxels_idx；

    3. 循环处理 pts_num_in_voxels 次，根据 `pts_idx_of_voxels` 依次 load 对应点的 `pts_feature` 到片上，得到当前 voxel 中的点在所有通道的特征值 pts_feature_cur_voxels，总共 pts_num_in_voxels \* channels 个特征值，若 `channels` 过大，无法一次性放入所有数据，则在 `channels` 通道上拆分，循环放入 C_seg 个数据；

    4. 计算这些点在所有通道上的最大或者平均的特征值 `pooled_features`，并得到 `argmax`, 若没有点在当前 voxel 中时，argmax 的值置为-1；

    5. 至此当前 voxel 计算完毕，store `pooled_features`[voxels_idx]和 `argmax`[voxels_idx]，返回 step1 处理下一个 voxels，直至当前 core 处理完 voxels_nums_seg 个。

- 根据实际网络中的数据规模:

  输入为 <br>
  rois[128,7] 3.5kB <br>
  pts[16000,3] 187kB <br>
  pts*feature[16000,16] 1000kB <br>
  输出为 <br>
  pooled_features[128,12,12,12,16] 128 * 108kB <br>
  pts*idx_of_voxels[128,12,12,12,128] 128 * 864kB <br>
  argmax[128,12,12,12,16] 128 \*108kB

- 以上述规模为例，实现方案可以描述如下，对于每一个 `rois`[1, 7]，在所有点 `pts`[16000, 3] 中找到在这个 roi 中的点，对于每一个 roi 等分成 相同大小的`out_x`、`out_y`、`out_z`，即 12 _ 12 _ 12 个 voxels，对于在每个 voxel 中的前 `max_pts_each_voxel`个点，即前 128 个点，取其在当前 voxel 中的点的每一个通道上的最大特征值或者平均特征值，得到`pooled_features`[1, 1, 1, 1, 16]，给出取得这个特征值的点的 idx，得到`argmax`[1, 1, 1, 1, 16] , 给出在每个 voxel 中的点的 idx `pts_idx_of_voxels`[1, 1, 1, 1, 128]。

### 3.2 伪代码实现（可选）

```c++

__global__ pts_idx_of_voxels_kernel(){
  memcpy(pts, ..., pts_num, GDRAM2NRAM);
  nram pts_flag_of_cur_roi = check_point_in_roi(pts_num, cur_roi);
  if (pts_flag_of_cur_roi > 0) {
    pts_idx_of_voxels = get_pts_idx_of_voxels(pts_num, cur_roi);
  }
  memcpy(pts_idx_of_voxels, ..., pts_num, NRAM2GDRAM);
}

__global__ roiaware_pool3d_forward_kernel(){
  memcpy(pts_idx_of_voxels[cur_voxels], ..., GDRAM2NRAM)
  memcpy(pts_feature[pts_in_cur_voxels], ..., GDRAM2NRAM);
  if (pool_method == 0) {
    pooled_features[voxels_idx], argmax[voxels_idx]
          = roiaware_maxpool3d(pts_idx_of_voxels[cur_voxels], pts_feature[pts_in_cur_voxels]);
    memcpy(pooled_features[voxels_idx], argmax[voxels_idx], ..., NRAM2GDRAM);
  } else if (pool_method == 1) {
    pooled_features[voxels_idx] = roiaware_avgpool3d(pts_idx_of_voxels[cur_voxels], pts_feature[pts_in_cur_voxels]);
    memcpy(pooled_features[voxels_idx], ..., NRAM2GDRAM);
  }
}

// cx,cy,cz,x_size,y_size,z_size,rz为一个roi的坐标
__device__ check_point_in_roi(float *X, float *Y, float *Z,
      float *local_X, float *local_Y, float *local_Z,
      float cx, float cy, float cz, float x_size, float y_size, float z_size, float rz, int num) {
  __bang_sub_const(local_Z, Z, cz, num);  //  local_Z
  __bang_active_abs(tmp, local_Z, num);
  __bang_write_value(tmp1, num, 0.5 * dz);
  __bang_le(flag, tmp, tmp1, num);     // Z in_flag
  float cosa = std::cos(-rz);
  float sina = std::sin(-rz);
  __bang_sub_const(tmp, X, cx, num);
  __bang_sub_const(tmp1, Y, cy, num);
  __bang_mul_const(tmp2, tmp, cosa, num);
  __bang_mul_const(tmp3, tmp1, sina, num);
  __bang_sub_const(local_X, tmp2, tmp3, num);  //  local_X

  __bang_mul_const(tmp, tmp, sina, num);
  __bang_mul_const(tmp1, tmp1, cosa, num);
  __bang_add_const(local_Y, tmp, tmp1, num);  //  local_Y

  __bang_active_abs(tmp2, tmp2, num);
  __bang_write_value(tmp1, num, 0.5 * dx);
  __bang_lt(flag_tmp, tmp2, tmp1, num);    // X in_flag
  __bang_mul(flag, flag, flag_tmp, num);

  __bang_active_abs(tmp, tmp, num);
  __bang_write_value(tmp1, num, 0.5 * dy);
  __bang_lt(flag_tmp, tmp, tmp1, num);   // Y in_flag
  __bang_mul(flag, flag, flag_tmp, num);  // in_flag

  // pts_flag_of_roi = flag;
}

__device__ get_pts_idx_of_voxels(int max_pts_each_voxel, int out_x, int out_y, int out_z,
    float cx, float cy, float cz, float x_size, float y_size, float z_size, float rz,
    int pts_num, float *local_X, float *local_Y, float *local_Z,
    const float *pts_flag_of_roi,
    int *pts_idx_of_voxels) {
  // input: pts_flag_of_roi(pts_num, channels)  NRAM
  // output: pts_idx_of_voxels(x_size, y_size, z_size, max_pts_each_voxel)  GDRAM
  float voxels_x_res = x_size / out_x;
  float voxels_y_res = y_size / out_y;
  float voxels_z_res = z_size / out_z;

  nram float X_idx = float2int((local_X + x_size / 2) / voxels_x_res);
  nram float Y_idx = float2int((local_Y + y_size / 2) / voxels_y_res);
  nram float Z_idx = float2int(local_Z                / voxels_z_res);
  nram voxels_index = X_idx * out_y * out_z +
                      Y_idx * out_z +
                      Z_idx ;

  X_flag = X_idx >= 0 & X_idx <= out_x - 1;
  Y_flag = Y_idx >= 0 & Y_idx <= out_y - 1;
  Z_flag = Z_idx >= 0 & Z_idx <= out_z - 1;

  nram pts_in_voxels_flag = X_flag & Y_flag & Z_flag;

  set nram pts_idx[pts_num] = 0 : pts_num - 1;
  set nram pts_idx_in_cur_voxels[pts_num] = 0;
  __bang_select(pts_idx_in_cur_voxels, pts_idx, pts_in_voxels_flag, pts_num);

  uint pts_num_in_voxels = (uint)pts_idx_in_cur_voxels[0];
  pts_num_in_voxels = std::min(pts_num_in_voxels, max_pts_each_voxel - 1);

  pts_idx_of_voxels_cur_voxels = pts_idx_of_voxels + voxels_index * max_pts_each_voxel;

  // pts_idx_of_voxels_cur_voxels[0] = pts_num_in_voxels;
  __memcpy(pts_idx_of_voxels_cur_voxels, pts_num_in_voxels, 1 * sizeof(int), NRAM2GDRAM)

  // memcpy pts_idx_in_cur_voxels[32:32 + pts_num_in_voxels] to pts_idx_of_voxels_cur_voxels[1:max_pts_each_voxel - 1];
  __bang_float2int(pts_idx_in_cur_voxels + 32, pts_num_in_voxels);
  __memcpy(pts_idx_of_voxels_cur_voxels, pts_idx_in_cur_voxels + 32, pts_num_in_voxels * sizeof(float), NRAM2GDRAM);
}

__device__ roiaware_maxpool3d(int max_pts_each_voxel, int out_x, int out_y, int out_z, int voxels_index,  // voxels
    int pts_num, int channels, float *local_X, float *local_Y, float *local_Z,   // pts
    const float *pts_idx_of_voxels, const float *pts_feature,
    int *argmax, float *pooled_features) {
  // voxels_nums = boxes_num * out_x * out_y * out_z;
  // input: pts_feature(pts_num, channels)  GDRAM
  // input: pts_idx_of_voxels(x_size, y_size, z_size, max_pts_each_voxel)  GDRAM
  // output: pooled_features(voxels_nums, channels) GDRAM
  // output: argmax(voxels_nums, channels) GDRAM

  nram pooled_feature_cur_voxels[channels];
  pooled_features_cur_voxels = pooled_features + voxels_index * channels;
  pts_idx_of_voxels_cur_voxels = pts_idx_of_voxels + voxels_index * max_pts_each_voxel;
  __bang_collect(pooled_feature_cur_voxels, pts_idx_of_voxels_cur_voxels);

  nram nram_pts_feature_cur_voxels[max_pts_each_voxel];
  nram nram_pooled_feature_cur_voxels[max_pts_each_voxel];
  for (pts_idx_of_voxels_cur_voxels){
    memcpy(pts_feature, ..., GDRAM2NRAM);
    __bang_max(nram_pooled_feature_cur_voxels, nram_pts_feature_cur_voxels);
    argmax = nram_pooled_feature_cur_voxels[0];
    pooled_features = nram_pooled_feature_cur_voxels[1];
    memcpy(argmax, ..., NRAM2GDRAM);
    memcpy(pooled_features, ..., NRAM2GDRAM);
  }
}

__device__ roiaware_avgpool3d(int max_pts_each_voxel, int out_x, int out_y, int out_z, int voxels_index,  // voxels
    int pts_num, int channels, float *local_X, float *local_Y, float *local_Z,   // pts
    const float *pts_idx_of_voxels, const float *pts_feature,
    int *argmax, float *pooled_features) {
  // voxels_nums = boxes_num * out_x * out_y * out_z;
  // input: pts_feature(pts_num, channels)  GDRAM
  // input: pts_idx_of_voxels(x_size, y_size, z_size, max_pts_each_voxel)  GDRAM
  // output: pooled_features(voxels_nums, channels) GDRAM

  nram pooled_feature_cur_voxels[channels];
  pooled_features_cur_voxels = pooled_features + voxels_index * channels;
  pts_idx_of_voxels_cur_voxels = pts_idx_of_voxels + voxels_index * max_pts_each_voxel;
  __bang_collect(pooled_feature_cur_voxels, pts_idx_of_voxels_cur_voxels);

  nram nram_pts_feature_cur_voxels[max_pts_each_voxel];
  nram nram_pooled_feature_cur_voxels[max_pts_each_voxel];
  for (pts_idx_of_voxels_cur_voxels){
    memcpy(pts_feature, ..., GDRAM2NRAM);
    __bang_sum(nram_pooled_feature_cur_voxels, nram_pts_feature_cur_voxels);
    float pooled_features = nram_pooled_feature_cur_voxels[1] / pts_num_in_cur_voxels;
    memcpy(pooled_features, ..., NRAM2GDRAM);
  }
}
```

### 3.3 拆分(任务拆分，多核拆分)

**任务类型 U1：**

2 个 kernel 根据其 output 规模不同按照不同维度进行拆分，平分到各个 core 中：

1. pts_idx_of_voxels_kernel 中，输出 pts_idx_of_voxels 规模为 [boxes_num, out_x, out_y, out_z, max_pts_each_voxel]，多核拆分在 boxes_num 维度上拆分；
2. roiaware_pool3d_forward_kernel 中，输出 pooled_features 规模为 [boxes_num, out_x, out_y, out_z, channels]，输出 argmax 规模为[boxes_num, out_x, out_y, out_z, channels]，多核拆分在 boxes_num * out*x * out_y \* out_z 上拆分。

### 3.4 性能优化设计

1. 资源分配

1.1 pts_idx_of_voxels_kernel 中的空间划分如下

| 地址命名            | 元素数量     | 数据类型 |
| ------------------- | ------------ | -------- |
| X                   | nram_pts_num | T        |
| Y                   | nram_pts_num | T        |
| Z                   | nram_pts_num | T        |
| local_X             | nram_pts_num | T        |
| local_Y             | nram_pts_num | T        |
| local_Z             | nram_pts_num | T        |
| pts_in_flag         | nram_pts_num | T        |
| nram_temp_buffer1   | nram_pts_num | float    |
| nram_temp_buffer2   | nram_pts_num | float    |
| nram_temp_buffer3   | nram_pts_num | float    |
| nram_temp_buffer4   | nram_pts_num | float    |
| nram_temp_buffer5   | nram_pts_num | float    |
| nram_voxel_offset   | nram_pts_num | float    |
| nram_pts_idx_seq    | nram_pts_num | int32    |
| fp_nram_pts_in_flag | nram_pts_num | float    |

其中: </br>
nram_pts_num 为一次最多能放入 nram 的点的数量；</br>

1.2 roiaware_pool3d_forward_kernel 中的空间划分如下

| 地址命名                       | 元素数量                                        | 数据类型 |
| ------------------------------ | ----------------------------------------------- | -------- |
| nram_pts_idx_cur_voxel         | align_max_pts_each_voxel                        | int32    |
| nram_max_pts_feature_tmp       | align_max_pts_each_voxel                        | T        |
| nram_pts_feature_in_voxel      | nram_channels_limit \* align_max_pts_each_voxel | T        |
| nram_pooled_features_cur_voxel | nram_channels_limit                             | T        |
| nram_argmax_cur_voxel          | nram_channels_limit                             | int32    |
| one_pooled_feature             | 128Byte                                         | T        |

其中: </br>
align*max_pts_each_voxel = PAD_UP(max_pts_each_voxel, align_num); </br>
nram_channels_limit = PAD_DOWN((MAX_NRAM_SIZE - 128 - align_max_pts_each_voxel * (sizeof(int) + sizeof(T))) / ((align*max_pts_each_voxel + 1) * sizeof(T) + sizeof(int)), align_num); </br>

1.3 需要申请 workspace size 大小为 `pts_num` _ (3 + `channels` + `channels`(or 3)) _ sizeof(T) Byte, 用来存放转置之后的`pts_num`和`pts_feature`。

DRAM(workspace)中的空间划分如下:

| 地址命名              | 空间 shape                | 数据类型 |
| --------------------- | ------------------------- | -------- |
| transpose_pts         | [3, pts_num]              | T        |
| transpose_pts_feature | [channels, pts_num]       | T        |
| transpose_tmp         | [channels(or 3), pts_num] | T        |

2. 流水设计

根据实际测试，该算子排流水收益不大，暂不进行排流水。

### 3.5 可维护性设计

1、bangc 代码中加入必要的 log 信息，比如输入的规模 data_size、数据类型 dtype、维度 dim 等，以及如果出错会导致程序 core dump 的变量；

2、对每一个函数命名变量命名都有充分的注释；

3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代 (宏的调用说明以及含义已经注释写在 kernels 代码中)。

### 3.6 测试用例设计

- 框架在需求列表中给出的算子在网络中用到的规模(无，框架没提供)：
- partA2 网络规模： <br>
  输入：rois(128, 7) pts(16000, 3) pts_feature(16000, 16) <br>
  输出：argmax(128, 12, 12, 12, 16) pts_idx_of_voxels(128, 12, 12, 12, 128) pooled_features(128, 12, 12, 12, 16) <br>

- 边界 case：

1. 根据 job 类型, 1 mlu core(Block), 1 cluster, 4 cluster, 6 cluster, 8 cluster, 16 cluster, with/without remainder；
2. 根据 nram 空间计算单次处理的最大数量 m，小于 m, 大于 m，等于 m；
3. 根据各个维度是否对齐。

### 3.7 算子防呆检查

1. 检查 handle 以及所有输入的 tensor desc 是否为空；
2. 检查所有输入的 tensor desc 的 dim；
3. 检查所有输入的 tensor desc 的 shape，判断是否满足与其他输入参数的关联性；
4. 检查所有输入的 tensor desc 的 datatype，只支持 fp16 和 fp32，判断是否满足与其他输入参数的关联性；
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

- 2022.05.27 ~ 2022.06.02 调研源码、开始设计方案
- 2022.06.06 ~ 2022.06.10 设计方案+伪代码
- 2022.06.13 ~ 2022.06.15 方案 review
- 2022.06.15 ~ 2022.06.17 generator、gtest 代码开发
- 2022.06.17 ~ 2022.07.01 算子主体框架开发
- 2022.07.04 ~ 2022.07.08 大规模测试+测试报告
- 2022.07.11 ~ 2022.07.14 提交 MR+代码 review
- 2022.07.15 算子入库

### 5.2 风险分析

无。
