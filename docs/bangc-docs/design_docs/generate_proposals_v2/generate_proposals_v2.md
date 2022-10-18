# generate_proposals_v2 算子开发设计方案
* #### 文档基本信息
| 算子名称    | generate_proposals_v2     |
| ----------- | ----------------- |
| 编制人/日期 |  谷中豪/2022-08-22 |
| 审批人/日期	|  杜泽坤/2022-08-22|
| 审批人/日期	|  董成威/2022-08-22|
| 审批人/日期	|  王远/2022-08-22 |

* #### 修改记录
| 版本号| 修订人 | 修订日期 | 修订描述 |
| ----- | ------ | -------  | -------  |
| V1.0  | 谷中豪   | 2022-08-22 | 首次提交 |

* #### 内容描述
本文档为`generate_proposals_v2`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

* #### 算子需求checklist
算子需求提出者需要`提供`的信息如下：
- 框架负责人
- 算子接口描述
- 功能描述
- 框架版本 + 对应源码路径
- 需求对应网络
- 网络中用到的规模
- 常用规模下的竞品性能（可选）
- 是否需要支持原位
- 是否需要支持stride机制
- 框架单元测试阈值指标（可选）
- 其他特殊需求（在线量化/融合/转数提前等，可选）
- 确认算子需求是否已经过框架层review（滤除mluOp已支持的算子）

算子需求提出者需要`check`的部分如下：
- 1.1 算子需求分析
- 1.2 算子功能和应用场景描述
- 1.3 算子输入输出参数要求
- 1.4 算子限制
- 1.5 验收标准
- 2.2 接口设计
- 3.5 测试用例（需求提出者check算子需求表中所给规模是否列出）

## 1 需求分析
### 1.1 算子需求分析
| 算子功能简介| 该OP根据每个检测框为foreground对象的概率，推选生成用于后续检测网络的RoIs。|
| :------------- | :-------------------------------------------------------------- |
| 需求来源    | TensorFlow                                     |
| 应用网络    | maskrcnn                                        |
| 输入数据类型| scores: float <br>bbox_deltas: float  <br>im_shape: float  <br>anchors: float <br>variances: float <br>pre_nms_top_n: int <br>post_nms_top_n: int <br>nms_thresh: float <br>min_size: float <br>eta: float <br>pixel_offset:bool                                            |
| 输入Shape   | scores: [N, H, W, A]<br>bbox_deltas: [N，H，W, A*4]<br>im_shape: [N, 2]<br>anchors: [H, W, A, 4] <br>variances: [H, W, A, 4]<br>pre_nms_top_n: scalar <br>post_nms_top_n: scalar <br>nms_thresh: scalar <br>min_size: scalar <br>eta: scalar <br>pixel_offset: scalar|
| 输入Layout  | scores:ARRAY<br>bbox_deltas:ARRAY<br>im_shape:ARRAY<br>anchors:ARRAY<br>variances:ARRAY <br>pre_nms_top_n: scalar <br>post_nms_top_n: scalar <br>nms_thresh: scalar <br>min_size: scalar <br>eta: scalar <br>pixel_offset: scalar                             |
| 输出数据类型 | rpn_rois:float <br> rpn_roi_probs:float  <br>rpn_rois_num: int <br>rpn_rois_batch_size：int         |
| 输出Shape   | rpn_rois: [B, 4]<br>rpn_roi_probs: [B, 1]<br>rpn_rois_num: [N] <br>rpn_rois_batch_size:[1], dim=1,shape[0]=1  |
| 输出Layout  | rpn_rois: ARRAY <br>rpn_roi_probs: ARRAY<br> rpn_rois_num: ARRAY  <br> rpn_rois_batch_size: ARRAY                                    |
| 模式(可选） | 否 |
| 是否含有dim/axis等类似语义的参数且该参数支持负数/其他特殊处理 | 否 |
| 是否含有labels/index等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 否 |
| 是否需要支持原位        | 否                                                  |
| 是否需要支持stride机制  | 否                                                 |
| 是否需要支持广播  | 否                       |
| 0元素检查是否直接返回  |  对于scores、bbox_deltas、im_shape、anchors、variances参数<br>N=0 时返回 MLUOP_STATUS_SUCCESS <br>A、W、H 任一为0时返回 MLUOP_BAD_PARAM  |
| 其他特殊需求(在线量化，融合，转数提前等，可选)|        无                                                |
| 本次开发优先支持的规模/模式|   |

### 1.2 算子功能和应用场景描述
**算子功能：** `generate_proposals_v2`根据每个检测框为 foreground 对象的概率，推选生成用于后续检测网络的 RoIs。其中的检测框根据`anchors`和`bbox_deltas`计算得到。<br>
-  `anchors` 是在 feature_map 的每一个位置生成多个不同大小不同长宽比的矩形框。每个 anchor 以（xmin，ymin，xmax，ymax）的格式表示，其中， (xmin, ymin) 为左上角的坐标，(xmax, ymax) 为右下角的坐标。
- 在检测网络中， anchor 为 foreground 类型的对象表示可能有一个目标存在在 anchor box 中，前景 anchor 可能并没有完美的位于目标中心， 需使用`bbox_deltas`对其位置和尺寸进行精调， 使得anchor box能更好的拟合目标， 如果有多个 anchor 互相重叠，将保留拥有最高前景分数的 anchor，并舍弃余下的anchor（非极大值抑制)， 最终输出用于后续检测网络的 RoIs。

- 每张图片生成 proposals 的计算过程描述:
  1. 取topK： 对`scores`降序排序，取前`pre_nms_top_n`个`scores`值，并根据`scores`的index位置取对应的`bbox_deltas`、`anchors`、`variances`的值，每个`score`对应的`bbox_delta`为(xmin，ymin，xmax，ymax)，对应的`anchor`为(xmin，ymin，xmax，ymax) ，对应的`variance`值为(xcenter，ycenter，w，h)，此处w，h取值与shape 中 H，W 无关；
  2. 创建 proposals: 根据`bbox_deltas`、`anchors`、`variances`的取值来计算得到每个 proposal 的左上角和右下角点的坐标(xmin，ymin，xmax，ymax)；
  3. 对第二步创建好的 porposals 进行筛选, 根据 proposal 的坐标值计算宽和高， 移除宽或者高小于`min_size`的 proposal；
  4. 对剩余的 proposals 进行nms筛选，nms阈值设为`nms_thresh`, nms筛选输出`post_nms_top_n`个proposals及其对应的scores值(实际输出的proposals个数可能比`post_nms_top_n`少);
  5. 此时，一张图片上对应的 proposals 已经生成完毕，将第4步输出的 proposals 保存到`rpn_rois`，scores保存到`rpn_roi_probs`，proposals的数量保存到rpn_rois_num中,并在`rpn_rois_batch_size`中累加proposals的数量。

- 创建 proposals 的具体计算过程
  1. 根据anchor 两个点坐标 (xmin，ymin，xmax，ymax) 计算 box_anchor的中心点坐标 (cx， cy) 及 anchor的宽高；
  ```c++
    offset = pixes_offset? 1.0 : 0;
    w = xmax -xmin + offset;
    h = ymax -ymin + offset;
    cx = xmin + 0.5 * w;
    cy = ymin + 0.5 * h;
  ```
  2. 根据 (cx， cy) 和 deltal 的两点的坐标 (xmin，ymin，xmax，ymax) 计算的 box_deltal 中心点坐标和宽高 (d_cx，d_cy，d_w，d_h)；
  ```c++
    bbox_clip_default = std::log(1000.0 / 16.0);
    d_cx = cx + dxmin * w * var[0];
    d_cy = cy + dymin * h * var[1];
    d_w = exp(Min(dxmax * var[2], bbox_clip_default)) * w;
    d_h = exp(Min(dymax * var[3], bbox_clip_default)) * h;
  ```
  3. 根据box_deltal中心点坐标和宽高计算proposal的两个点的坐标 (oxmin，oymin，oxmax，oymax)；
   ```c++
    oxmin = d_cx - d_w * 0.5;
    oymin = d_cy - d_h * 0.5;
    oxmax = d_cx + d_w * 0.5 - offset;
    oymax = d_cy + d_h * 0.5 - offset;
   ```
  4. 通过min，max把proposal的坐标约束到[im_shape.w], [im_shape.h]
   ```c++
    proposals[0] = Max(Min(oxmin, im_shape[1] - offset), 0.);
    proposals[1] = Max(Min(oymin, im_shape[0] - offset), 0.);
    proposals[2] = Max(Min(oxmax, im_shape[1] - offset), 0.);
    proposals[3] = Max(Min(oymax, im_shape[0] - offset), 0.);
   ```

**应用场景：** `generate_proposals_v2`算子应用于`maskrcnn`。

**paddle 示例：**
```py
import paddle.fluid as fluid
import numpy as np
import paddle
print(paddle.__version__)
paddle.disable_static()

N = 2
A = 15
H = 54
W = 40

scores = paddle.rand((N, A, H, W), dtype=paddle.float32)
bbox_deltas = paddle.rand((N, 4*A, H, W), dtype=paddle.float32)
img_size = paddle.to_tensor([[10.0, 10.0], [5.0, 5.0]])
anchors = paddle.rand((H, W, A, 4), dtype=paddle.float32)
variances = paddle.rand((H, W, A, 4), dtype=paddle.float32)

pre_nms_top_n = 2000
post_nms_top_n = 1000
nms_thresh = 0.5
min_size = 0.
eta = 1.0
pixel_offset = False
return_rois_num = True

rpn_rois, rpn_roi_probs, rpn_rois_num = paddle.vision.ops.generate_proposals(
        scores,
        bbox_deltas,
        img_size,
        anchors,
        variances,
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        eta=eta,
        pixel_offset=pixel_offset,
        return_rois_num=return_rois_num)
print(rpn_rois, rpn_roi_probs, rpn_rois_num)

output:
rpn_rois： Tensor(shape=[2, 4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            [[0.        , 0.        , 0.        , 0.        ],
            [0.27896869, 0.27956927, 1.38735509, 1.82488048]])
rpn_roi_probs：Tensor(shape=[2, 1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            [[0.        ],
            [0.96731776]])
rpn_rois_num：Tensor(shape=[2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [1, 1])
##--------------------------------------------------------------------
# 0元素行为分析
# N=0 正常返回
Tensor(shape=[0, 4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [])
ensor(shape=[0, 1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [])
Tensor(shape=[0], dtype=int32, place=Place(gpu:0), stop_gradient=True,
       [])

# A,W,H 任意为0, core dump
Error Message Summary:
FatalError: `Erroneous arithmetic operation` is detected by the operating system.
  [TimeInfo: *** Aborted at 1660893422 (unix time) try "date -d @1660893422" if you are using GNU date ***]
  [SignalInfo: *** SIGFPE (@0x7fa513bc2049) received by PID 3171 (TID 0x7fa5cdf56700) from PID 331096137 ***]
Floating point exception (core dumped)

# N=0  且 A,W,h 任意为0时, core dump
Error Message Summary:
FatalError: `Erroneous arithmetic operation` is detected by the operating system.
  [TimeInfo: *** Aborted at 1660893422 (unix time) try "date -d @1660893422" if you are using GNU date ***]
  [SignalInfo: *** SIGFPE (@0x7fa513bc2049) received by PID 3171 (TID 0x7fa5cdf56700) from PID 331096137 ***]
Floating point exception (core dumped)
##--------------------------------------------------------------------
# nan和inf行为分析
# img_size的图片宽高存在nan，inf行为, nan和inf按最大值处理，-inf按最小值处理, 正常返回

## scores中nan和inf按最大值处理，-inf按最小值处理
N = 2
A = 1
H = 1
W = 2

scores = paddle.to_tensor(np.array([[[[0.2, np.nan]]], [[[0.5, -np.inf]]]]), dtype=paddle.float32)
bbox_deltas = paddle.to_tensor(np.array([[[[0.0, 0.0, 2.0, 2.0],[0.0, 0.0, 2.0, 2.0]]], [[[0.0, 0.0, 2.0, 2.0],[0.0, 0.0, 2.0, 2.0]]]]), dtype=paddle.float32)
img_size = paddle.to_tensor([[5.0, 5.0], [5.0, 5.0]])
anchors = paddle.to_tensor(np.array([[[[1.0, 1.0, 3.0, 3.0], [1.0, 1.0, 3.0, 3.0]]], [[[1.0, 1.0, 3.0, 3.0], [1.0, 1.0, 3.0, 3.0]]]]), dtype=paddle.float32)
variances = paddle.to_tensor(np.array([[[[1.0, 1.0, 3.0, 3.0],[1.0, 1.0, 3.0, 3.0]]], [[[1.0, 1.0, 3.0, 3.0],[1.0, 1.0, 3.0, 3.0]]]]), dtype=paddle.float32)

Tensor(shape=[2, 4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[0., 0., 5., 5.],
        [0., 0., 5., 5.]]) Tensor(shape=[2, 1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[nan       ],
        [0.50000000]]) Tensor(shape=[2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
       [1, 1])
```
### 1.3 算子输入输出参数要求
| 参数             | 语义                           | 类型（输入/输出） | 支持类型               | 物理布局 | 规模限制 |
| ---------------- | ------------------------------ | ----------------- | ---------------------- | -------- | -------- |
| **handle**           |        操作句柄                        | 输入              |    mluOpHandle_t   | /        | 无       |
| **pre_nms_top_n**         |  每张图在 NMS 操作之前要保留的总框数，数据类型仅支持int       | 输入              | int            | scalar   | |
| **post_nms_top_n**         |  每个图在 NMS 后要保留的总框数，数据类型仅支持int      | 输入              | int             | scalar   | |
| **nms_thresh**         |   NMS中的阈值，数据类型仅支持 float       | 输入              | float             | scalar   | |
| **min_size**         |   根据宽和高过滤候选框的阈值，宽或高小于该阈值的候选框将被过滤掉，数据类型仅支持 float       | 输入              | float             | scalar   |  |
| **eta**         |   自适应阈值的衰减系数，仅在自适应NMS中且自适应阈值大于0.5时生效，在每次迭代中 adaptive_threshold = adaptive_treshold * eta ，**自适应nms当前不支持**        | 输入              | float             | scalar   | |
| **pixel_offset**         | pixel_offset 默认为 true，表示 img_size 的像素偏移，offset = pixel_offset ？1：0        | 输入              | bool             |    | |
| **scores_desc**      |    输入 scores 的形状描述             | 输入              |           /             | /        | 无       |
| **scores**         |   表示每个框包含 object 的概率，shape是[N, H ,W, A,]，N是batch大小，A是achors数量，H、W是 feature map 的高和宽        | 输入              | float             | ARRAY   | [N, H, W, A]|
| **bbox_deltas_desc**      |    输入 bbox_deltas 的形状描述             | 输入              |           /             |        | 无       |
| **bbox_deltas**         |   表示预测出的候选框的位置和 anchor 的位置之间的距离        | 输入              | float             | ARRAY    | [N, H, W, A*4] |
| **img_size_desc**      |    输入 img_size 的形状描述           | 输入              |           /             | /        | 无       |
| **img_size**         |    表示原始图像的大小信息，每个img_size以（height, width）表示    | 输入              | float             | ARRAY  | [N, 2]|
| **anchors_desc**      |    输入 anchors 的形状描述             | 输入              |           /             | /        | 无       |
| **anchors**         | anchor 是在 feature_map 的 每一个位置生成多个不同大小不同长宽比的矩形框，shape是[H ,W, A, 4]。每个 anchor 以（xmin，ymin，xmax，ymax）的格式表示，其中，xmin 和 ymin 为左上角的坐标，xmax 和 ymax 为右下角的坐标       | 输入              | float             | ARRAY    | [H ,W, A, 4] |
| **variances_desc**      |    输入variances的形状描述             | 输入              |           /             | /        | 无       |
| **variances**         |    表示 anchors 的方差，shape是[H, W, A, 4],每个 anchor 的方差都是(xcenter，ycenter，w，h)的格式表示    | 输入              | float    | ARRAY    | [H ,W, A, 4]|
| **workspace**        |   指向额外GDRAM空间的指针         | 输入             |  void *                  | /          | 无       |
| **workspace_size**   |   输入参数，workspace 的空间大小   | 输入             |  size_t                  | /          | 无       |
| **rpn_rois_desc**      |    输入 rpn_rois 的形状描述             | 输出              |           /             | /        | 无       |
| **rpn_rois**         |    表示产出的 RoIs，shape 是[B, 4]，B表示Rois的数量，传入的B等于 N *`post_nms_top_n`的大小，实际的计算返回的batch等于`rpn_rois_batch_size`       | 输出             | float             | ARRAY   | [B, 4]|
| **rpn_roi_probs_desc**      |    输入rpn_roi_probs的形状描述            | 输出              |           /             | /        | 无       |
| **rpn_roi_probs**         |   RoIs的得分，shape是[B, 1]，B表示Rois的数量，传入的B等于 N * `post_nms_top_n`的大小，实际的计算返回的batch等于`rpn_rois_batch_size`       | 输出              | float             | ARRAY    | [B, 1]|
| **rpn_rois_num_desc**      |    输入rpn_rois_num的形状描述             | 输出              |           /             | /        | 无       |
| **rpn_rois_num**         | 每张图片对应的RoIs的数量，数组中每个值的累加和等于rpn_rois的dim[0]，shape是[N]，N是batch的大小，表示输入图片个数        | 输出              | int             | ARRAY    | [N]|
| **rpn_rois_batch_size**  | 表示rpn_rois、rpn_roi_probs实际输出的batch大小| 输出              | int             | ARRAY    |dim=1, shape[0]=1|


### 1.4 算子限制
| 限制类型     | 详细说明                                                     |
| ------------ | ------------------------------------------------------------ |
| 输入限制     |  输入参数shape必须满足要求:<br>scores：[N, H, W, A]<br>bbox_deltas:[N, H, W, A*4]<br>img_size: [N, 2]  <br>anchors[H, W, A, 4] <br>variances[H, W, A, 4] |
| 输入限制     |  输出参数shape必须满足要求:<br>rpn_rois:[N * post_nms_top_n, 4]，实际输出的维度信息为[rpn_rois_batch_size, 4] <br>rpn_roi_probs:[N * post_nms_top_n, 1], 实际输出的维度信息为[rpn_rois_batch_size, 1] <br>rpn_rois_num:[N] <br>rpn_rois_batch_size:[1], dim=1, shape[0]=1|
| 输入限制     |  输入参数eta表示自适应NMS，当前不支持，和竞品保持一致，参数保留，输入满足 eta >=1.0 |
| 输入限制     |  输入参数nms_thresh > 0 |
| 输入限制     |  输入参数scores、bbox_deltas、anchors、variances、img_size中包含nan或者inf时，行为不保证与竞品一致。 |
| 数据类型限制 | scores、bbox_deltas、anchors、variances只支持 float 输入 <br> pre_nms_top_n、post_nms_top_n只支持int类型输入 <br >nms_thresh、min_size只支持 float 输入|
| 数据范围限制 | 无 |
|  原位限制     | 不支持原位                                                 |
| stride限制   | 不支持stride                                      |
| 广播限制     | 不支持广播                                                   |


### 1.5 验收标准
#### 1.5.1 精度验收标准

按照[精度验收标准](../../../MLU-OPS-Accuracy-Acceptance-Standard.md)的要求明确本算子的精度标准。
`generate_proposals_v2` 是复合类算子。<br>
`rpn_rois`参数精度设为 diff1 <= 3e-3 && diff2 <=3e-3。<br>
`rpn_roi_probs`、`rpn_rois_num`、`rpn_rois_batch_size`的精度设置为 diff3=0。

#### 1.5.2 性能验收标准
- 网络中使用到的规模性能优于或至少与竞品性能持平。
- 部分与竞品差距过大的规模在4.算子性能优化记录中进行说明。
- 附上算子测试报告链接，测试报告必须包括框架给出的网络中规模的性能数据以及对应效率值。

[给定的网络规模](./network_scale.txt)

## 2 算子接口设计
### 2.1 参考接口
- **paddlepaddle** :https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/generate_proposals_v2_kernel.h#L22
```c++
template <typename T, typename Context>
void GenerateProposalsV2Kernel(const Context& ctx,
                               const DenseTensor& scores,
                               const DenseTensor& bbox_deltas,
                               const DenseTensor& im_shape,
                               const DenseTensor& anchors,
                               const DenseTensor& variances,
                               int pre_nms_top_n,
                               int post_nms_top_n,
                               float nms_thresh,
                               float min_size,
                               float eta,
                               bool pixel_offset,
                               DenseTensor* rpn_rois,
                               DenseTensor* rpn_roi_probs,
                               DenseTensor* rpn_rois_num);
```

### 2.2 接口设计
#### 2.2.1 `generate_proposals_v2` 获取额外申请空间大小
```c++
mluOpStatus_t MLUOP_WIN_API
mluOpGetGenerateProposalsV2WorkspaceSize(mluOpHandle_t handle,
                             const mluOpTensorDescriptor_t scores_desc,
                             size_t *size);
```
**参数描述：**
- `handle`：输入参数。操作句柄，内部绑定device和对应的queue。
- `size`：输入参数。需要用户申请的额外的空间大小，通过`mluOpGetGenerateProposalsV2WorkspaceSize`获取。
- `mluOpTensorDescriptor_t`: 输入tensor的形状描述。

#### 2.2.2 `generate_proposals_v2` 计算接口
```c++
mluOpStatus_t MLUOP_WIN_API mluOpGenerateProposalsV2(mluOpHandle_t handle,
                                                    const int pre_nms_top_n,
                                                    const int post_nms_top_n,
                                                    const float nms_thresh,
                                                    const float min_size,
                                                    const float eta,
                                                    bool pixel_offset,
                                                    const mluOpTensorDescriptor_t scores_desc,
                                                    const void *scores,
                                                    const mluOpTensorDescriptor_t bbox_deltas_desc,
                                                    const void *bbox_deltas,
                                                    const mluOpTensorDescriptor_t im_shape_desc,
                                                    const void *im_shape,
                                                    const mluOpTensorDescriptor_t anchors_desc,
                                                    const void *anchors,
                                                    const mluOpTensorDescriptor_t variances_desc,
                                                    const void *variances,
                                                    void *workspace,
                                                    size_t workspace_size,
                                                    const mluOpTensorDescriptor_t rpn_rois_desc,
                                                    void *rpn_rois,
                                                    const mluOpTensorDescriptor_t rpn_roi_probs_desc,
                                                    void *rpn_roi_probs,
                                                    const mluOpTensorDescriptor_t rpn_rois_num_desc,
                                                    void *rpn_rois_num,
                                                    void *rpn_rois_batch_size);
```

## 3 实现方案设计
### 3.1 实现方案

`generate_proposals_v2`根据每个检测框为 foreground 对象的概率，推选生成用于后续检测网络的 RoIs。其中的检测框根据`anchors`和`bbox_deltas`计算得到。

- **参考算子的实现过程：**
1. 通过转置操作将 scores 和 bbox_deltas 的大小分别调整为 [H * W * A，1] 和 [H * W * A，4]；
2. 根据 anchors 和 bbox_deltas 计算出候选框的位置；
3. Clip boxes to image；
4. 删除面积较小的候选框；
5. 通过NMS选出满足条件的候选框作为结果。

- **MLU多核拆分策略**
1. 网络规模中，一般情况下，`N`的值小于`HWA`，因此这里 MLU core 间选择不对`N`进行拆分，MLU core 间选择拆分`HWA`；
2. `generate_proposals_v2` job类型设为Ubest，尽可能的launch多个 MLU cluster 去参与计算;
3. MLU core 内以 batch 作为最外层循环，每次循环计算一个batch， 因第1步中的 MLU core 间拆分， 每个 MLU core上计算 HWA / taskDim 数量的数据。

- **输入数据预处理**
1. scores shape为[N, H, W, A]， 计算过程中按照[N, H, W， A]方式取数；
2. bbox_deltas shape为[N, H, W, A*4]， 计算过程中按照[N, 4, 1, H * W * A]方式取数，需要转置；
3. anchors shape为 [H, W, A, 4]，计算过程中按照[1，4, 1, H * W * A]方式取数，需要转置；
4. variances shape为 [H, W, A, 4]， 计算过程中按照[1，4, 1, H * W * A]方式取数，需要转置。

每次循环计算一个batch，即每次循环生成一张图片的 proposals ，对每张图片生成 proposals 的步骤总结为以下三步：
1. topK 计算, 获取到scores中 第`pre_nms_top_n`大的score值 k_score;
2. createAndRemoveSmallBox: 根据计算规则创建 proposals 并移除其中不满足条件的 proposals;
3. nms筛选: 对第二步的生成 proposals 进行 nms 筛选后输出;

#### 3.1.1 topK实现

首先比较`pre_nms_top_n`和`HWA`的值，`pre_nms_top_n` >= `HWA`时，跳过 topK 步骤; `pre_nms_top_n` < `HWA`时， 对 `scores` 进行 向量topK 操作，获取第 `pre_nms_top_n` 大的 score 值 k_score。<br>

##### 3.1.1.1 topK实现时中每个core上的数据量和偏移的计算
```c++
// 计算每个cluster上的数据量
int rem_num = HWA % taskDimY;
int cluster_deal_num = HWA / takDimY + (int)(taskIdY < rem_num);
int n_start = taskIdY * cluster_deal_num + ((taskIdY < rem_num) ? 0: rem_num);

//  计算每个core上的计算量和偏移
int rem_core_num = cluster_deal_num % coreDim;
int per_core_num = (coreId < rem_core_num) ? cluster_deal_num / coreDim + 1 : cluster_deal_num / coreDim;
int core_offset = (coreId < rem_core_num) ? coreId * per_core_num : coreId * per_core_num + rem_core_num;

// 根据nram空间大小，计算core上需要循环的次数；
int seg_pad_k = CEIL_ALIGN(max_nram_size / 2, align_num);
int repeat =  per_core_num / seg_pad_k;
int rem_num = per_core_num % seg_pad_k;
```
##### 3.1.1.2 topK实现过程
1. 对`HWA`按照 core 的数量平均分配到每个 core 上，每个 core 上计算 per_core_num 份，在 core 上循环 repeat 次，每次循环加载 seg_pad_k 份 `scores` 数据，用 bang_max 找出单个 core 上最大值，并进行规约，找到 cluster 间的 global_max_score;<br>

2. 设置 up_score = global_max_score, down_score = -FLT_MAX, mid_score = 0.5 * (up_score - down_score);<br>

3. 循环加载每个 core 上的所有`scores`数据，每次循环中用 bang_ge 获取 scores 中大于 mid_score 的 mask，使用 bang_count 统计大于 mid_score 的个数count，把每次循环统计到的 count 累加起来，得到单 core 上所有 `scores` 大于 mid_score 的数量， 再把每个 cluster 上每个 core 上的 count 规约累加计算出 HWA 个 `scores` 中大于 mid_score 的数量 totol_count;<br>

4. 比较 totol_count 是否等于 pre_nms_top_n，并更新 up_score 、down_score、 mid_score, 重复setp 3、 4，直到 totol_count 等于 pre_nms_top_n，此时，第K大的 score 值 k_score 等于 mid_score， topK结束。<br>

##### 3.1.1.3 topK实现nram空间和workspace的划分
```c++
// nram: 从 GDRAM 的 input 数据中load scores，seg_pad_k = max_nram_size / 2
// |  scores   |  ge_mask   |
// | seg_pad_k |  seg_pad_k |

// workspace: topK时存放每个core上计算出来的最大值和index
// | input_scores | max_score | max_index |
// |  HWA         | taskDim   |  taskDim  |
```

#### 3.1.2 createAndRemoveBoxes 实现
##### 3.1.2.1 createAndRemoveBoxes 过程中每个core上的数据量和偏移的计算
```c+
// 计算每个cluster上的数据量
int rem_num = pre_nms_top_n % taskDimY;
int cluster_deal_num = pre_nms_top_n / takDimY + (int)(taskIdY < rem_num);
int n_start = taskIdY * cluster_deal_num + ((taskIdY < rem_num) ? 0: rem_num);

// 计算每个core上的计算量和偏移
int rem_core_num = cluster_deal_num % coreDim;
int per_core_num = (coreId < rem_core_num) ? cluster_deal_num / coreDim + 1 : cluster_deal_num / coreDim;
int core_offset = (coreId < rem_core_num) ? coreId * per_core_num : coreId * per_core_num + rem_core_num;

// 根据nram空间大小，计算core上需要循环的次数；
int seg_pad_1 = CEIL_ALIGN(max_nram_size / (13 + X), align_num);
int repeat =  per_core_num / seg_pad_1;
int rem_num = per_core_num % seg_pad_1;
```

##### 3.1.2.2 createAndRemoveBox 实现过程
1. 从 GDRAM 上load scores、anchors、bbox_deltas、variances数据，平分到每个 core 上的 nram 空间，每个core上 load 的大小为 per_core_num, core 上每次循环load seg_pad_1 个数据;

2. 单次循环load完数据后，使用bang_ge 获取 nram 上 scores 大于等于 k_score 的mask;

3. 使用 bang_collect，根据 第2步的mask， 把 mask 等于1位置的`scores`、`anchors`、`bbox_deltas`、`variances`值collect到一起, `scores` 需要collect一次, `anchors`、`bbox_deltas`、`variances`需要对四个值分别进行collect, 每次循环 collect 数量为seg_pad_1;

4. 用 collect 后的数据，根据 createbox 计算过程创建 proposals;

5. 根据 removeSmallBox 的计算方法，生成新的 mask2， 用 bang_collect 操作移除proposal中宽和高小于 min_size 的 proposal，把有效的 proposals 集中到一起，此时，单次循环内的计算过程结束；

6. 把单次循环时创建好 proposal 数据，保存到 workspace 空间内， 若单 core 内数据未处理完，回到第 2 步；<br>

##### 3.1.2.3 createAndRemoveBox 的nram空间和workspace划分
```c++
// nram：重新从 worksapce 上 load scores、anchors、bbox_deltas、variance， seg_pad_1 = max_nram_size / (13 + X)
// |  scores   | anchors       | bbox_deltas   | variances     | nram_temp     |
// | seg_pad_1 | 4 * seg_pad_1 | 4 * seg_pad_1 | 4 * seg_pad_1 | X * seg_pad_1 |

// workspace : createAndRemoveBox 后的 proposals 存放到 worksapce 中, 其对应的 scores 也存放在 worksapce 中
// | scores | proposals | scores_tmp | proposals_tmp | collect_num |
// | AHW    | AHW * 4   | AHW        | AHW * 4       | taskDim     |
```

##### 3.1.2.4 createbox 计算过程
a. 根据anchor 两个点坐标 (xmin，ymin，xmax，ymax) 计算 box_anchor的中心点坐标 (cx， cy) 及 anchor的宽高；<br>
```c++
offset = pixes_offset? 1.0 : 0;
w = xmax -xmin + offset;
h = ymax -ymin + offset;
cx = xmin + 0.5 * w;
cy = ymin + 0.5 * h;
```

b. 根据 (cx， cy) 和 deltal 的两点的坐标 (xmin，ymin，xmax，ymax) 计算的 box_deltal 中心点坐标和宽高 (d_cx，d_cy，d_w，d_h)；
```c++
bbox_clip_default = std::log(1000.0 / 16.0);
d_cx = cx + dxmin * w * var[0];
d_cy = cy + dymin * h * var[1];
d_w = exp(Min(dxmax * var[2], bbox_clip_default)) * w;
d_h = exp(Min(dymax * var[3], bbox_clip_default)) * h;
```

c. 根据box_deltal中心点坐标和宽高计算proposal的两个点的坐标 (oxmin，oymin，oxmax，oymax)；
```c++
oxmin = d_cx - d_w * 0.5;
oymin = d_cy - d_h * 0.5;
oxmax = d_cx + d_w * 0.5 - offset;
oymax = d_cy + d_h * 0.5 - offset;
```

d. 通过min，max把proposal的坐标约束到[im_shape.w], [im_shape.h]；
```c++
proposals[0] = Max(Min(oxmin, im_shape[1] - offset), 0.);
proposals[1] = Max(Min(oymin, im_shape[0] - offset), 0.);
proposals[2] = Max(Min(oxmax, im_shape[1] - offset), 0.);
proposals[3] = Max(Min(oymax, im_shape[0] - offset), 0.);
```
##### 3.1.2.5 removeSmallBoxs 计算过程
1. 通过proposals的两点坐标计算 proposal的宽 box_w和高 box_h；

2. 用bang_ge方法，分别获取box_w 和 box_h 和 min_size 比较的mask，记为mask_w 和 mask_h;

3. 用bang_and 计算 mask_w 和 mask_h 的与的结果 mask_res；

4. 根据mask_res，用bang_collect，把proposals中对应位置的值取出集中到一起；

5. 把 collect 后的 proposal 数据存放到 workspace 上， 先在 workspace 上开辟 coreNum 大小的空间，每个 core 在对应 taskId 位置存放自己当前的 collect 数量，sync_all 同步后，每个 core 上计算自己存放在 workspace 上的数据偏移，按照这个偏移往 workspace 上存放 collect 后的数值（由于3.1.3 nms筛选中会对乱序数据进行排序操作，本节中存放在 workspace 中的数据相对顺序与 input tensors 可能会不同，但不影响最终算子结果）。

#### 3.1.3 nms筛选
对剩余的 proposal_num 个 proposals 进行nms筛选，nms阈值设为 nms_thresh，nms筛选按照 scores 从大到小顺序输出输出 min(proposal_num，post_nms_top_n) 个proposals及其对应的scores值。<br>
##### 3.1.3.1 nms过程中每个core上的数据量和偏移的计算
```c++
// nms前计算proposal的总数，作为nms的循环次数
int proposal_num = min(proposal_num, post_nms_top_n);
// 计算每个cluster上的数据量
int rem_num = proposal_num % taskDimY;
int cluster_deal_num = proposal_num / takDimY + (int)(taskIdY < rem_num);
int n_start = taskIdY * cluster_deal_num + ((taskIdY < rem_num) ? 0: rem_num);

//  计算每个core上的计算量和偏移
int rem_core_num = cluster_deal_num % coreDim;
int per_core_num = (coreId < rem_core_num) ? cluster_deal_num / coreDim + 1 : cluster_deal_num / coreDim;
int core_offset = (coreId < rem_core_num) ? coreId * per_core_num : coreId * per_core_num + rem_core_num;

// 根据nram空间大小，计算core上需要循环的次数；
int seg_pad_2 = CEIL_ALIGN(max_nram_size / (5 + X), align_num);
int repeat =  per_core_num / seg_pad_2;
int rem_num = per_core_num % seg_pad_2;
```
##### 3.1.3.2 nms实现
1. load workspace上全部的 proposal 到片上，计算出 box_area 并将结果存放到 workspace 上，box_area 在计算iou时会用到，（如果 nram 空间不足，正常在 MLU core 内循环处理）；

2. load scores 到 nram 上，通过 bang_max 获取当前 core 上 scores 的最大值 local_max_score， 再利用 workspace（max_score 这块空间，详见 3.1.3.3） 对每个 core 上的 local_max_score 进行规约，计算得到 global_max_score，并将 workspace 上 scores global_max_score 对应位置的score置为 -FLT_MAX；<br>

3. 根据 global_max_score 的 global_max_score_index，从 workspace 的 proposals 中拿到 global_max_score_box 的坐标，及对应的box_area的值, 并保存 global_max_score 和 global_max_score_box 到 nram 的output rois，roi_probs 空间内;<br>

4. 把 worksapce 上的 scores、proposals、box_area 数据load到 nram，计算 global_max_score_box 与 其余 boxes 的 iou, 整个过程为向量运算;

5. 通过 bang_ge 获取 iou 比 iou_thresh 大的 mask，并将 mask 为 1 的位置的scores置为 -FLT_MAX（如果 nram 空间不足，4、5 两步需要增加循环处理）；<br>

6. 重复循环 2,3,4,5步，循环 min(proposal_num，post_nms_top_n) 次或者单次循环取到的 global_max_score 的值等于 -FLT_MAX 时循环结束;<br>

7. copy nram 上的output_rois、output_roi_probs 数据到 GDARAM的 output 空间。

##### 3.1.3.3 nms的nram空间和workspace划分
```c++
// nram： 从workspace中loadscores和proposals, seg_pad_2 = max_nram_size / (5 + X)
// |  output_rois   | output_roi_probs | scores    | proposals     | nram_temp     |
// | post_nms_top_n | post_nms_top_n   | seg_pad_2 | 4 * seg_pad_2 | X * seg_pad_2 |

// workspace：用于规约nms过程中每个core上的最大score值及其index
// | max_score | scores       | proposals      | box_ares     | max_index |
// |  taskDim  | proposal_num | proposal_num*4 | proposal_num | taskDim   |
```
### 3.2 伪代码实现

```c++
...
// kernel 实现逻辑
// 每个cluster上每次循环计算一个batch，即每次循环生成一张图片的 proposals 。
template <typename T>
__mlu_func__ void mluOpsGeneratorProposalsV2Kernel(){
  if (coreId == 0x80){
    return;
  }

  // split batch for cluster
  int rem_num = N % taskDimY;
  int n_deal = N / takDimY + (int)(taskIdY < rem_num);
  int n_start = taskIdY * n_deal + ((taskIdY < rem_num) ? 0: rem_num);
  if(n_deal <= 0){
    return;
  }

  for(int idx_n = n_start; idx_n < n_start + n_deal; ++idx_n){
    int rem_core_num = HWA % coreDim;
    int per_core_num = (coreId < rem_core_num) ? HWA / coreDim + 1 : HWA / coreDim;
    int core_offset =(coreId < rem_core_num) ? coreId * per_core_num : coreId * per_core_num + rem_core_num;
    ...
    getTopKVal();
    createBox();
    removeSmallBox();
    nms();
    ...
  }
}

// TopK 实现
__mul_func__ void getTopKVal(T * scores, T * bbox_deltas, T *anchors, T *variances, const int topk_num, const int scores_num){
    if(scores_num <= topk_num){
    return;
  }
#if __BANG_ARCH__ >= 300
  __bang_argmax(result, scores, scores_num);
#else
 __bang_max(result, scores, scores_num);
#endif

  T dn = NE_TNF;
  T up = result[0];
  T mid = dn + (up - dn) * 0.5;
  int count  = 0

  while(1){
    __bang_ge_scalar(tmp, scores, mid, scores_num);

    // 获取当前大于mid的数量
    count = __bang_count(tmp, scores_num);

    if(count == topk_num){
      break;
    } else if (count > topk_num && (mid == up || mid =dn)){
      break;
    }
    // update up, dn, mid
    if(count > topk_num){
      dn = (dn == mid) ? up : mid;
    } else if(count < topk_num){
      up = (up == mid) ? dn : mid;
    }
    mid = up + (up - dn) * 0.5;
  }
}

// createbox实现
// `createbox` 根据输入anchor、bbox_deltas、variances的坐标，生成proposals;

// output = exp(input)
__mlu__func void calcExp(T * output, const T * input, cpnst int length){
#if __BANG_ARCH__ >=322
#define LOG_2_E (1.44269504088f)
  __bang_mul_scalar(output, input, (float)LOG_2_E, length);
  __bang_pow2(output, output, length);
#else
  __bang_active(output, input, length);
#endif
}
// 生成proposals
__mlu__func void createBox(const T* anchor, const T *deltas, const T *var, const int deal_size, T * proposals, T *nram_temp, bool pixes_offset = true){
  T *axmin = anchor;
  T *aymin = anchor + deal_size;
  T *axmax= anchor + 2 * deal_size;
  T *aymax = anchor + 3 * deal_size;

  T offset = pixes_offset? static_cast<T>(1.0) : 0;

  T *w = nram_temp;
  T *h = nram_temp + deal_size;
  T *cx = nram_temp + 2 * deal_size;
  T *cy = nram_temp + 3 * deal_size;

  // w = axmax - axmin + offset
  __bang_sub(w, axmax, axmin, deal_size);
  // h = aymax - aymin + offset；
  __bang_sub(h, aymax, aymin, deal_size);
  if(pixes_offset){
    __bang_add_scalar(w, w, T(1.0), deal_size);
    __bang_add_scalar(h, h, T(1.0), deal_size);
  }
  // T cx = axmin + 0.5 * w;
  __bang_mul_scalar(cx, w, T(0.5), deal_size);
  __bang_add(cx, cx, axmin, deal_size);

  // T cy = aymin + 0.5 * h;
  __bang_mul_scalar(cy, h, T(0.5), deal_size);
  __bang_add(cy, cy, aymin, deal_size);

  T *dxmin = deltas;
  T *dymin = deltas + deal_size;
  T *dxmax= deltas + 2 * deal_size;
  T *dymax = deltas + 3 * deal_size;

  T *d_w = nram_temp + 4 * deal_siz
  T *d_h = nram_temp + 5 * deal_size;
  T *d_cx =nram_temp + 6 * deal_size;
  T *d_cy = nram_temp + 7 * deal_size;
  T *tmp_exp = nram_temp + 8 * deal_size;

 __bang_mul(d_w, dxmin, w, deal_size);
 __bang_mul(d_h, dymin, h, deal_size);

 if(var){
  __bang_mul(d_w, d_w, var, deal_size);
  __bang_mul(d_h, d_h, var + deal_size, deal_size);

  __bang_mul(dxmax, dxmax, var + 2 * deal_size, deal_size);
  __bang_mul(dymax, dymax, var + 3 * deal_size, deal_size);

  static const double bbox_clip_default = std::log(1000.0 / 16.0);
  __bang_mineq_scalar(dxmax, tmp_exp, (T)bbox_clip_default, deal_size);
  __bang_mineq_scalar(dymax, dymax, (T)bbox_clip_default, deal_size);
 }

  __bang_add(d_cx, cx, d_w, deal_size)
  __bang_add(d_cy, cy, d_h, deal_size)

  calcExp(d_w, dxmax, deal_size);
  calcExp(d_h, dymax, deal_size);

  __bang_mul(d_w, d_w, w, deal_size);
  __bang_mul(d_h, d_h, h, deal_size);

  T *oxmin = nram_temp + 9 * deal_siz;
  T *oymin = nram_temp + 10 * deal_size;
  T *oxmax = nram_temp + 11 * deal_size;
  T *oymax = nram_temp + 12 * deal_size;
  T *tmp_o = nram_temp + 13 * deal_size;

  __bang_mul_scalar(tmp_o, d_w, (T)0.5, deal_size);
  __bang_sub(oxmin, d_cx, tmp_o, deal_size);
  __bang_add(oxmax, d_cx, tmp_o, deal_size);

  __bang_mul_scalar(tmp_o, d_h, (T)0.5, deal_size);
  __bang_sub(oymin, d_cy, tmp_o, deal_size);
  __bang_add(oymax, d_cy, tmp_o, deal_size);

   if(pixes_offset){
    __bang_sub_scalar(oxmax, oxmax, T(1.0), deal_size);
    __bang_sub_scalar(oymax, oymax, T(1.0), deal_size);
  }

  __bang_mineq_scalar(oxmin, oxmin, (T)(im_info[1] - offset), deal_size);
  __bang_mineq_scalar(oymin, oymin, (T)(im_info[0] - offset), deal_size);
  __bang_mineq_scalar(oxmax, oxmax, (T)(im_info[1] - offset), deal_size);
  __bang_mineq_scalar(oymax, oymax, (T)(im_info[0] - offset), deal_size);

  __bang_relu(box, oxmin, deal_size);
  __bang_relu(box + deal_size, oymin, deal_size);
  __bang_relu(box + 2 * deal_size, oxmax, deal_size);
  __bang_relu(box + 3 * deal_size, oymax, deal_size);
}

// removeSmallBox
// `removeSmallBox`： 移除box长和宽比min_size小的box，pixel_offset=1时需要计算偏移。
__mlu_func__ void removeSmallBox(T * boxes, T *scores, const T *im_size,
        const T min_size, T *nram_buffer, const  int deal_size, unsigned int * count, bool pixel_offset){
  T *w = nram_buffer;
  T *h = nram_buffer + deal_size;
  T *cx = nram_buffer + deal_size;
  T *cy = nram_buffer + deal_size;

  // w = box[2] - box[0];
  __bang_sub(w, boxes + 2 * deal_size, boxes, deal_size);
  // h = box[3] - box[1];
  __bang_sub(h, boxes + 3 * deal_size, boxes + 1 * deal_size, deal_size);

  if(pixel_offset){
    T offset = pixel_offset ? 1.0 : 0;
    // w = w - offset
    __bang_add_scalar(w, w, offset, deal_size);
    // h = h - offset
    __bang_add_scalar(h, h, offset, deal_size);
  }

  // cx = box[0] + 0.5 * w
  __bang_mul_scalar(cx, w, (T)0.5, deal_size);
  __bang_add(cx, boxes, cx, dea;_size);

  // cy = box[1] + 0.5 * h
  __bang_mul_scalar(cy, h, (T)0.5, deal_size);
  __bang_add(cy, boxes + deal_size, cy, deal_size);

  // mask_tmp1 = w >= min_size ? 1 : 0;
 __bang_ge_scalar(mask_tmp1, w, min_size, deal_size);
  // mask_tmp2 = h >= min_size ? 1 : 0;
 __bang_ge_scalar(mask_tmp2, h, min_size, deal_size);
  // mask_result = mask_tmp1 & mask_tmp2
 __bang_and(mask_result, mask_tmp1, mask_tmp2, deal_size);

 if(pixel_offset){
  T im_h = im_size[0];
  T im_w = im_size[1];
  // mask_tmp1 = cx <= im_w ? 1 : 0;
  __bang_le_scalar(mask_tmp1, cx, im_w, deal_size);
  // mask_tmp2 = cy <= im_h ? 1 : 0;
  __bang_le_scalar(mask_tmp2, cy, im_h, deal_size);
  // mask_result = mask_result & mask_tmp1 & mask_tmp2
  __bang_and(mask_result, mask_result, mask_tmp1, deal_size);
  __bang_and(mask_result, mask_result, mask_tmp2, deal_size);
 }

 // count nan-zero value in mask_result
 *count = __bang_count(mask_result, deal_size);

// collect and store box and scores
 __bang_collect(box + 0 * deal_size, box + 0 * deal_size, mask_result, deal_size);
 __bang_collect(box + 1 * deal_size, box + 1 * deal_size, mask_result, deal_size);
 __bang_collect(box + 2 * deal_size, box + 2 * deal_size, mask_result, deal_size);
 __bang_collect(box + 3 * deal_size, box + 3 * deal_size, mask_result, deal_size);
 __bang_collect(scores, scores, mask_result, deal_size);
}
```

### 3.3 拆分(任务拆分，多核拆分)
**拆分策略**
1. 网络规模中，一般情况下，`N`的值小于`HWA`，因此这里 MLU core 间选择不对`N`进行拆分，MLU core 间选择拆分`HWA`；
2. `generate_proposals_v2` job类型设为Ubest，尽可能的launch多个 MLU cluster 去参与计算;
3. MLU core 内以 batch 作为最外层循环，每次循环计算一个batch， 因第1步中的 MLU core 间拆分， 每个 MLU core上计算 HWA / taskDim 数量的数据；

### 3.4 性能优化设计
1. 流水设计
 无

### 3.5 方案理论性能

### 3.6 可维护性设计
1. bangc代码中加入必要的 log 信息，比如输入的规模、数据类型、layout，任务类型，以及如果出错会导致程序core dump的变量，比如IO指令的data_size、dim xyz的值等，这些信息都是有利于快速定位问题；
2. 对每一个函数命名变量命名都有充分的注释；
3. 避免魔鬼数字，对于确定的数字尽量使用公共宏来替代。

### 3.7 测试用例设计
1. 0元素测试
2. 给定的网络规模测试
3. 不同规模的N,H,W,A测试

### 3.8 算子防呆检查
1. 指针为空防呆；
2. 0元素检查防呆，VLOG(5)打印信息；
3. input，output的数据类型须保持一致，且符合算子类型支持限制；
4. 对shape进行防呆，需要保证输入boxes满足要求。
5. workspace防呆

## 4 算子性能/精度问题 & 优化记录
### 4.1 当前存在问题的规模说明
无
### 4.2 已经过优化的规模说明

### 4.3 优化记录

## 5 方案实施
### 5.1 开发测试计划
- 2022.8.16-2022.8.19：算子调研，竞品行为分析，方案设计撰写
- 2022.8.22-2022.8.26：方案设计评审，generator和gtest代码开发
- 2022.8.29-2022.9.2：算子host/device代码实现、功能调试，大规模测试
- 2022.9.5-2022.9.9：输出测试报告，PR
