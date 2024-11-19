# roi_align_rotated_forward 向量化实现设计方案

* #### 文档基本信息

| 算子名称    | roi_align_rotated_forward 向量化实现 |
| ---------- | ---------------------------------- |
| 编制人/日期 | 陈其阳/2024-11-4                     |

* #### 修改记录
| 版本号 | 修订人  | 修订日期   | 修订描述 |
| ----- | ------ | -------   | ------- |
| V 1.0 | 陈其阳  | 2024-11-4 | 首次提交 |

* #### 内容描述

本文档为`roi_align_rotated_forward`算子向量化实现的设计文档，包括需求分析、接口设计、方案设计、性能优化记录。

## 1 需求分析

### 1.1 算子需求分析

| 算子功能简介| 以双线性插值的方式提取非整数大小且带有旋转的roi的特征图|
|-------------|--------------------------------------------------------------|
| 需求来源    | mmcv                                     |
| 应用网络    | FOTS                                                 |
| 输入数据类型| half, float                                                  |
| 输入Shape   | input1: [batch, hi, wi, channels]; input2: [roi_nums, 6]    |
| 输入Layout  | input1: NHWC; input2: ARRAY |
| 输出数据类型| half, float |
| 输出Shape   | [roi_nums, ho, wo, channels] |
| 输出Layout  | NHWC |
|是否含有dim/axis等类似语义的参数且该参数支持负数/其他特殊处理 | 否|
|是否含有labels/index等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 否|
|是否需要支持原位        | 否   |
| 是否需要支持stride机制  | 否       |
| 是否需要支持广播  | 否                       |
| 0元素检查是否直接返回  | 是                   |

### 1.2 算子功能和应用场景描述

在FOTS网络中，roi_align_rotated算子用于统一检测和识别到端到端的pipeline中，输入检测分支中得到的带有旋转角度的bounding boxes，提取对应的特征图用于后续的识别。

![通道的二维展开](fots_framework.png)

### 1.3 算子输入输出参数要求
#### 1.3.1 roi_align_rotated_forward

| 参数          | 语义 | 类型（输入/输出） | 支持类型    | 物理布局 | 规模限制 |
| ------------- | ---- | ----------------- | ----------- | -------- | -------- |
| handle        | MLUOP句柄，保存运行的上下文信息     | 输入              |             | /        | 无       |
| features_desc | 输入特征图数据的描述信息     | 输入              |             | /        | features的维度必须是4       |
| features      | 输入数据，指向输入特征图数据的mlu首地址     | 输入              | half, float | NHWC     | 无       |
| rois_desc     | roi数据的描述信息     | 输入              |             | /        | rois的维度必须是2，且第二维的大小必须是6       |
| rois          |  输入数据，指向rois的mlu地址    | 输入              | half, float | ARRAY    | 无       |
| pooled_height | 输出output的height     | 输入              | int         | /        | 无       |
| pooled_width  | 输出output的width    | 输入              | int         | /        | 无       |
| sample_ratio  | 一个bin的采样率     | 输入              | int         | /     | 无       |
| spatial_scale | rois在feature map上的缩放比例     | 输入              | float       | /        | 无       |
| aligned       | 决定rois中的像素是否需要偏移     | 输入              | bool        | /        | 无       |
| clockwise     | 是否顺时针旋转     | 输入              | bool        | /        | 无       |
| output_desc   | 输出数据的描述信息     | 输入              |             | /        | output的维度必须是4，且第一维大小与rois的第一维大小一致，第二维大小与pooled_height一致，第三维大小与pooled_width一致，第四维大小与features的第四维大小一致       |
| output        | 指向输出数据的mlu首地址     | 输出              | half, float | NHWC     | 无       |

### 1.4 算子限制
#### 1.4.1 roi_align_rotated_forward
- rois是一个二维的Tensor，其中第一维与output的第一维相同，最后一维必须等于6。每个roi包含（batch_id，x，y, w, h, θ），其中，x和y表示的是roi中心点的坐标，w和h分别是roi的宽和高，θ表示边框逆时针旋转的角度。

- rois中batch_id的值在[0, batch-1]范围内，其中batch是features的第一维的大小，rois中参数x，y，w和h与spatial_scale的乘积值不能超过参数类型可表示的范围；rois中包含NaN和infinity数据时，只有x和y支持infinity数据，其它都不支持。

- output的最高维与rois的最高维相等，最后一维大小与features的最后一维相等。

- features, rois, output数据类型要相同。

### 1.5 验收标准

#### 1.5.1 精度验收标准

- 采用动态阈值：
  diff=[diff1, diff2], threshold_rate=[10, 10]。

## 2 算子接口设计

### 2.1 参考接口

- MMCV
```c++
// forward
template <typename scalar_t>
__global__ void roi_align_rotated_forward_cuda_kernel(
    const int nthreads, const scalar_t *bottom_data,
    const scalar_t *bottom_rois, const scalar_t spatial_scale,
    const int sample_num, const bool aligned, const bool clockwise,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, scalar_t *top_data);
```

### 2.2 接口设计

```c++
// forward
mluOpStatus_t MLUOP_WIN_API mluOpRoiAlignRotatedForward(mluOpHandle_t handle,
                                                        const mluOpTensorDescriptor_t features_desc,
                                                        const void *features,
                                                        const mluOpTensorDescriptor_t rois_desc,
                                                        const void *rois,
                                                        const int pooled_height,
                                                        const int pooled_width,
                                                        const int sample_ratio,
                                                        const float spatial_scale,
                                                        const bool aligned,
                                                        const bool clockwise,
                                                        const mluOpTensorDescriptor_t output_desc,
                                                        void *output);
```

## 3 实现方案设计

### 3.1 实现方案
在支持 gather.vector 后，相比于一次处理单个输出点，一次处理多个输出点的方案有更高的 IO 效率，能够显著提升性能。<br>


该算子的MMCV 实现有 6 层循环，分别是[roi_nums, pooled_height, pooled_width, channels, roi_bin_grid_h, roi_bin_grid_w]。<br>
其中[roi_nums, pooled_height, pooled_width, channels]的一个点（bin）对应一个线程，每个线程中的循环是[roi_bin_grid_h, roi_bin_grid_w]。<br>
该算子涉及大量的坐标计算，坐标计算与 channels 维度无关，因此将该维度放置于最后处理，可以复用坐标信息。<br>
注：有考虑过 [roi_bin_grid_h, roi_bin_grid_w] 的顺序放置在 [pooled_height, pooled_width] 之前，但是单个 bin 内有一个累加计算，这个累加计算拆给多个单元去计算，就需要 atomic_add，且控制逻辑会更复杂。<br>


MLU 将6层循环分成4个部分，第一循环[roi_nums, pooled_height, pooled_width]，第二循环[channels]（channels累加结果进行缓存），第三循环[roi_bin_grid_h, roi_bin_grid_w]，第四循环[channels_cache]（复用坐标信息）。<br>


（1）第一循环
- 核间拆分，记第一循环中总数量为 n1，使用最简单拆分逻辑即可。
```c++
  for (uint32_t i = taskId; i < n1; i+=taskDim)
```
- [roi_info 计算](#321-roi_info-计算)，roi_info 信息只与 roi_idx 有关，若roi_idx 与上次相同时，则跳过该步。<br>


（2）第二循环
- 由于 NRAM 大小限制，设置 channels 缓存量为1024，如果 channels 大于 1024，会进行循环。记 channels 缓存空间为 output_channels。<br>
- 在第三循环开始，对 output_channels 刷 0。<br>
- 在第三循环结束，将 output_channels 存到 output 。（前两层循环与 output 一一对应）<br>
注：channels 在坐标计算外循环，会导致坐标重复计算；channels 过大时，gather.vector 需要多次，IO 开销增大。即 channels 过大时，向量化实现的性能会下降。<br>


（3）第三循环（向量化的重点）
- h_idx, w_idx 序列构造 <br>
roi_bin_grid_h,roi_bin_grid_w 对应两个方向的采样率。<br>
sample_ratio>0 时，采样率都等于 sample_ratio，通常值大小在 2~9 之间。<br>
sample_ratio<0 时，采样率为 roi_height(width)/pooled_height(width)，参考 roi_align，采样率范围通常在30以内。<br>
根据采样率，向上选择序列长度（包括：8，16，32），序列长度记为 bin_order_num。为方便起见，h,w方向处理的数量一致。如果采样率超过32，则需要循环处理。<br>
使用vv.index构造自增序列（从 .5f 开始，步长为 1），该序列会缓存，方便多次构建二维序列。<br>
二维序列 w_idx 构造，相当于长度为 roi_bin_grid_w 自增序列复制 roi_bin_grid_h 次，使用 stride=0 的 memcpy2d 可以实现。<br>
二维序列 h_idx 构造，相当于长度为 roi_bin_grid_h 自增序列每个位置扩展至 roi_bin_grid_w 次，使用 __extension 实现。<br>
- 计算 x, y 序列 <br>
先计算 h_idx_in_bin, w_idx_in_bin。<br>
h_idx_in_bin = roi_start_h + ph * bin_size_h + (h_idx + .5f) * bin_size_h / roi_bin_grid_h; <br>
w_idx_in_bin = roi_start_w + pw * bin_size_w + (w_idx + .5f) * bin_size_w / roi_bin_grid_w; <br>
注：除法部分不能用乘法代替，否则坐标计算出现误差，而后续又有筛选操作，会使得精度严重下降。<br>
y = h_idx_in_bin * cosscalar_theta - w_idx_in_bin * sinscalar_theta + roi_center_h;<br>
x = h_idx_in_bin * sinscalar_theta + w_idx_in_bin * cosscalar_theta + roi_center_w;<br>
- 筛选有效点 <br>
if (y < -1.0 || y > height || x < -1.0 || x > width) return 0; <br>
mmcv 有上行处理，无效点的val为0，+= 0 等价于不做处理。<br>
bangC 实现为 [筛选有效点](#322-筛选有效点)。<br>
注：mmcv和MLU实现时，nan 都不会被筛选出去。<br>
- 计算双线性插值四个点的坐标(pos)和权值(w)<br>
[双线性插值前半部](#323-双线性插值)，由于需要处理边界情况，使用更为直接的标量处理。<br>
- 坐标去重，权重相加 <br>
根据坐标在GDRAM上取值是对性能影响最大的一步，通常来说有坐标重复的点，去重后可以减少IO次数，进而提升性能。<br>
去重的逻辑较为复杂也不便向量化实现，在[双线性插值后半部](#323-双线性插值)中实现。<br>
注:经测试，去重后点的数量（unique_num）一般变少，有时能取得几倍的性能提升。然而，权重提前相加，改变了计算顺序，使得 nan/inf 不对齐。<br>


（4）第四循环
- 在[channels_cache]中循环，记单次最大取 max_once_c 个 channel。<br>
在不支持 gather.vector 的机器上，一次最大只能取一个 channel，max_once_c = 1。<br>
在支持 gather.vector 的机器上，unique * max_once_c 要不超过 NRAM 空间限制，详细见[拆分](#33-拆分)。<br>
- 从 input 中根据坐标信息去取值（对性能影响最大的一步） <br>
pos 需要变成字节偏移，即乘以 channels * sizeof(T)。<br>
input 可能不按 64B 对齐，需要做对齐处理，pos 还需加上 input 对齐的偏移。<br>
- 计算双线性插值结果
取数后 v([unique_num, once_c])，w([unique_num])，要进行广播乘法（目前只能先转置再调用__bang_cycle_mul），得到 val([unique_num, once_c])。<br>
使用 __bang_sumpool 对 val 做累加得到 val_sum([once_c]) 。val_sum 加到 output_channels 中。<br>
注：mmcv 的累加顺序一定是从前往后，而 sumpool 累加顺序不是，会使得精度有偏差，inf/nan 无法对齐。<br>


### 3.2 伪代码实现

#### 3.2.1 roi_info 计算
```c++
template <typename T, bool sr_gt0>
__mlu_func__ void getRoiInfo(const T *rois_dram, int roi_idx,
                             const mluOpRoiAlignRotatedParams &params,
                             int &roi_batch_ind, T &roi_center_h,
                             T &roi_center_w, T &bin_size_h, T &bin_size_w,
                             int &roi_bin_grid_h, int &roi_bin_grid_w,
                             T &roi_start_h, T &roi_start_w, T &cos_theta,
                             T &sin_theta, T &count) {
  const T *roi_info = rois_dram + roi_idx * ROI_OFFSET;
  roi_batch_ind = (int)roi_info[0];
  T offset = params.aligned ? (T)0.5 : (T)0.0;
  roi_center_w = roi_info[1] * (T)params.spatial_scale - offset;
  roi_center_h = roi_info[2] * (T)params.spatial_scale - offset;
  T roi_width = roi_info[3] * (T)params.spatial_scale;
  T roi_height = roi_info[4] * (T)params.spatial_scale;
  T theta = roi_info[5];
  if (params.clockwise) {
    theta = -(theta);
  }
  if (!params.aligned) {
    roi_width = fmaxf(roi_width, (T)1.0);
    roi_height = fmaxf(roi_height, (T)1.0);
  }

  bin_size_h = roi_height / (T)params.pooled_height;
  bin_size_w = roi_width / (T)params.pooled_width;

  if constexpr (sr_gt0) {
    roi_bin_grid_h = params.sample_ratio;
    roi_bin_grid_w = params.sample_ratio;
  } else {
    if constexpr (std::is_same<T, half>::value) {
      roi_bin_grid_h = __half2int_up(bin_size_h);
      roi_bin_grid_w = __half2int_up(bin_size_w);
    } else {
      roi_bin_grid_h = __float2int_up(bin_size_h);
      roi_bin_grid_w = __float2int_up(bin_size_w);
    }
  }

  roi_start_h = roi_height / (T)-2.0;
  roi_start_w = roi_width / (T)-2.0;

  if constexpr (std::is_same<T, half>::value) {
    cos_theta = __cn_scalar_cos_f16(theta);
    sin_theta = __cn_scalar_sin_f16(theta);
  } else {
    cos_theta = __cn_scalar_cos_f32(theta);
    sin_theta = __cn_scalar_sin_f32(theta);
  }

  count = fmaxf(T(roi_bin_grid_h * roi_bin_grid_w), (T)1.0);
}
```

#### 3.2.2 筛选有效点
```c++
template <typename T>
__mlu_func__ void selectValidPoint(const int height, const int width, T *nram_y,
                                   T *nram_x, const uint32_t deal_num, T *aux1,
                                   T *aux2, T *aux3, uint32_t &valid_num) {
  // y < -1.0
  __bang_lt_scalar(aux1, nram_y, (T)-1, deal_num);
  // || y > height
  __bang_gt_scalar(aux2, nram_y, (T)height, deal_num);
  __bang_or(aux3, aux1, aux2, deal_num);
  // || x < -1
  __bang_lt_scalar(aux1, nram_x, (T)-1, deal_num);
  __bang_or(aux3, aux3, aux1, deal_num);
  // || x > width
  __bang_gt_scalar(aux2, nram_x, (T)width, deal_num);
  __bang_or(aux3, aux3, aux2, deal_num);
  __bang_not(aux3, aux3, deal_num);
  __bang_filter(nram_y, nram_y, aux3, deal_num);
  valid_num = __bang_filter(nram_x, nram_x, aux3, deal_num);
}
```

#### 3.2.3 双线性插值
```c++
template <typename T>
__mlu_func__ void bilinearInterpolatePosWeight(
    const int height, const int width, T *nram_y, T *nram_x,
    const uint32_t valid_num, uint32_t *pos1, uint32_t *pos2, uint32_t *pos3,
    uint32_t *pos4, T *w1, T *w2, T *w3, T *w4, uint32_t &unique_num) {
  for (uint32_t i = 0; i < valid_num; ++i) {
    T y = nram_y[i];
    T x = nram_x[i];

    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    int y_low, x_low, y_high, x_high;
    if constexpr (std::is_same<T, half>::value) {
      y_low = __half2int(y);
      x_low = __half2int(x);
    } else {
      y_low = __float2int(y);
      x_low = __float2int(x);
    }

    if (y_low >= height - 1) {
      y_high = y_low = height - 1;
      y = (T)y_low;
    } else {
      y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
      x_high = x_low = width - 1;
      x = (T)x_low;
    } else {
      x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;
    pos1[i] = y_low * width + x_low;
    pos2[i] = y_low * width + x_high;
    pos3[i] = y_high * width + x_low;
    pos4[i] = y_high * width + x_high;
    w1[i] = hy * hx;
    w2[i] = hy * lx;
    w3[i] = ly * hx;
    w4[i] = ly * lx;
  }
  // unique
  unique_num = 0;
  for (int i = 0; i < valid_num; ++i) {
    if (w1[i] < 0) {
      continue;
    }
    for (int j = i + 1; j < valid_num; ++j) {
      if (pos1[i] == pos1[j]) {
        // Points at the same position
        w1[i] += w1[j];
        w2[i] += w2[j];
        w3[i] += w3[j];
        w4[i] += w4[j];
        w1[j] = -1;
      }
    }
    if (unique_num != i) {
      pos1[unique_num] = pos1[i];
      pos2[unique_num] = pos2[i];
      pos3[unique_num] = pos3[i];
      pos4[unique_num] = pos4[i];
      w1[unique_num] = w1[i];
      w2[unique_num] = w2[i];
      w3[unique_num] = w3[i];
      w4[unique_num] = w4[i];
    }
    unique_num += 1;
  }
}
```

### 3.3 拆分

#### 3.3.1 核间拆分
目前使用简单拆分的逻辑：
```c++
uint32_t n1 = rois_num * pooled_height * pooled_width;
for (uint32_t i = taskId; i < n1; i+=taskDim) {
  ...
}
```
总逻辑分为 6 个维度，如果针对特定规模的话，可以采用别的拆分方法，目前没有这样的需求。

#### 3.3.2 核内拆分
记原始自增序列长度为 bin_order_num，二维扩展后长度为 bin_hw_order_num。<br>
bin_hw_order_num = bin_order_num ^ 2。<br>


固定NRAM空间划分为：
| name | size | 用途 |
| ------ | ------ | ---------- |
| order | 128  |  起点 0.5，步长为 1 的自增序列 |
| output_channels | sizeof(T) * 1024 | 存储 output 结果 |
| bin_h | sizeof(T) * bin_hw_order_num | h_idx 二维序列 |
| bin_w | sizeof(T) * bin_hw_order_num | w_idx 二维序列 |
| y | sizeof(T) * bin_hw_order_num | y 序列 |
| x | sizeof(T) * bin_hw_order_num | x 序列 |
| w1 | sizeof(T) * bin_hw_order_num | 计算 y,x 时的缓冲区。w1 权重 |
| w2 | sizeof(T) * bin_hw_order_num | 计算 y,x 时的缓冲区。w2 权重 |
| w3 | sizeof(T) * bin_hw_order_num | 计算 y,x 时的缓冲区。w3 权重 |
| w4 | sizeof(T) * bin_hw_order_num | w4 权重 |
| pos1 | sizeof(uint) * bin_hw_order_num | pos1 坐标 |
| pos2 | sizeof(uint) * bin_hw_order_num | pos2 坐标 |
| pos3 | sizeof(uint) * bin_hw_order_num | pos3 坐标 |
| pos4 | sizeof(uint) * bin_hw_order_num | pos4 坐标 |


剩余空间对齐均分为三份 vi, vi_t, val，记空间大小为 max_v_size。<br>
其中 vi 复用多次，最终的 val_sum 也存储于 vi 中。<br>
此时 max_once_c = max_v_size / unique_num / sizeof(T)。 <br>
以float 类型为例：
- 若 bin_order_num 为 32，固定的 size 为 53376, max_vi_size 为 113280。
unique_num 最大可到 bin_hw_order_num(1024)，此时 max_once_c = 27。
- 若 bin_order_num 为 8，固定的 size 为 7296, max_vi_size 为 128640。
unique_num 最大可到 bin_hw_order_num(64)，此时 max_once_c = 502。


### 3.4 性能优化设计
1.向量化加速。
2.减少重复计算，例如:roi_info 计算，bin_h、bin_w 二维序列构建等。
3.使用 fuse.nram 融合三条以上的乘加法。
4.双线性插值坐标进行查重，减少 IO 的数量。


### 3.5 可维护性设计

1、每个函数都有相应的注释，表明该函数的功能以及参数信息；

2、算子对应的feature提交，bug修复等，均应记录在对应的维护表中；

3、在kernel入口处应用参数检查，log打印，kernel出口应有相应的debug信息打印；

4、不支持的数据类型与物理布局应有相应的检查报错;

### 3.6 算子防呆检查
 1、指针为空防呆；

 2、0元素检查防呆，VLOG(5)打印信息；

 3、features和output必须为4维，rois必须要为2维，且rois的第二维大小必须是6；

 4、features和output的layout必须相同，且都为NHWC；

 5、output和rois的第一维必须相等，features和output的第四维必须相等；

 6、output的HW维度需要分别与参数中的pooled_height和pooled_width保持一致。


## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

暂无

### 4.2 已经过优化的规模说明

1.适用于向量化的规模：(sample_ratio >=3 || sample_ratio <= 0) && channels < 1024
