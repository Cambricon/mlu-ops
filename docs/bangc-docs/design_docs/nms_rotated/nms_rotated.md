# nms rotated 算子开发设计方案

- #### 文档基本信息

| 算子名称    | `nms_rotated`          |
| ----------- | --------------------- |
| 编制人/日期 | liuyuan1/2022-02-06     |
| 审批人/日期 | 袁梦，张双林/2022-02-17   |

- #### 修改记录

| 版本号 | 修订人  | 修订日期  | 修订描述 |
| ------ | ------- | --------- | -------- |
| V1.0   | liuyuan1 | 2022-2-6 | 首次提交 |

- #### 内容描述

本文档为`nms_rotated`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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
| 应用网络                  | fcos3d/pointpillar                                             |
| 输入数据类型               | float                                                          |
| 输入 Shape                | boxes:[N,5]; scores:[N]; iou_threshold:float;                  |
| 输出数据类型               | int32                                                          |
| 输出 Shape                | output:[N]; result_num: int32, 表示输出box的个数                 |
| 是否需要支持原位            | 否                                                              |
| 是否需要支持 stride 机制    | 否                                                              |
| 是否需要支持广播            | 否                                                              |
| 0 元素检查是否直接返回       | 是                                                              |

### 1.2 算子功能和应用场景描述

NmsRotated 算子有 2 个输入 Tensor，分别为 `boxes`[N,5] or [N,6], `scores`[N]，`boxes`输入低维度的数字包括的信息为： `x_ctr`, `y_ctr`, `width`, `height`, `radian` 或者为： `x_ctr`， `y_ctr`， `width`， `height`， `radian`, `label`.

计算过程简单描述：

1. 对scores进行由大到小的排序，将最大score对应的index输出到`output`。
2. 计算最大score的box与其他box的`IOU`，`IOU`的计算参考 [box_iou_rotated设计文档](../box_iou_rotated/box_iou_rotated.md)，其计算过程如下：
    1. 根据中心点计算旋转后的 rotatedbox1/box2.
    2. 根据 rotatedbox1/box2 的坐标点，计算每条边相交与否，是否有互相包含的情况，得到交点坐标（总共 24 种可能性）。
    3. 如果当前 box pair 相交的点数大于 2 个，则计算交叠面积，否则返回当前的为 0。
    4. 按照 Convex-hull-graham 顶点扫描法，排序、筛选得出凸包形状的顶点集合。
    5. 计算有效的交叠面积 intersection 后，计算结果`IOU`。
3. 将`IOU`大于`iou_threshold`的box移除，回到step1直到对所有的`scores`完成遍历。

### 1.3 算子输入输出参数要求

| 参数           | 语义                       | 类型（输入/输出） | 支持类型                 | 物理布局        | 规模限制 |
| -----------   | -------------------------- | -------------- | ----------------------- | ------------- | -------- |
| handle        | 操作句柄                    | 输入            | mluOpHandle_t           | /             | /        |
| boxes_desc    | 输入数据，box 的描述符        | 输入            | mluOpTensorDescriptor_t | /             | /        |
| boxes         | 输入数据，box 的坐标          | 输入            | float\*                 | [N, 5] or [N, 6]  | /    |
| scores_desc   | 输入数据，scores 的描述符     | 输入            | mluOpTensorDescriptor_t | /             | /        |
| scores        | 输入数据，scores 的大小       | 输入            | float\*                 | [N]           | /        |
| workspace     | 输入数据，GDRAM上面的辅助空间  | 输入            | void\*                  | /             | /        |
| workspace_size| 输入数据，辅助空间的大小       | 输入            | size_t\*                | /             | /        |
| iou_threshold | 输入数据，IOU 的阈值          | 输入            | float                   | scalar        | /        |
| output_desc   | 输入数据，输出 index 的描述符  | 输入            | mluOpTensorDescriptor_t | /             | /        |
| output        | 输出数据，输出 index 的数据    | 输出            | int32 \*                | [N]           | /        |
| result_num    | 输出数据，输出 box 的个数      | 输出            | int32 \*                | /             | /        |

### 1.4 算子限制

| 限制类型     | 详细说明                                                                           |
| -----------  | ------------------------------------------------------------------------------- |
| 输入限制     | 输入 `boxes`，`scores`的shape 必须满足要求: boxes[N, 5]或[N, 6] 和 scores:[N]         |
| 输入限制     | 输入 `boxes`, `scores`不支持输入 nan 或 inf。                                               |
| 输入限制     | 输入参数 `iou_threshold` 仅支持输入float, 可支持nan与inf                           |
| 输出限制     | 输出 `output`的shape 必须满足: [N]                                                 |
| 输出限制     | 输出 `result_num` 仅支持 int32\* 类型                                              |
| 数据类型限制 | 输入 `boxes`，`scores` 数据类型保持一致且仅支持float。`output` 仅支持 int32        |
| 原位限制     | 不支持原位                                                                         |
| stride 限制  | 不支持 stride 机制                                                                 |
| 广播限制     | 不支持广播                                                                         |

限制说明：

- 不支持`scores`中存在相同score的情况。参考接口和mlu所选择框的index不一致。
- `scores`不支持nan/inf。出现多个inf时不满足上述说明。不支持nan，因为参考接口nan score等同于inf，会被选择。mlu中`__bang_max` 选择正常数据，不选择nan。
- `boxes`不支持nan/inf。sin/cos和参考接口的bit级无法保持一致。

### 1.5 验收标准

#### 1.5.1 精度验收标准

- 该算子输出`output`为所选择的box的索引数据。`output`是 int32_t 数据类型。因此，该算子采用静态阈值，阈值标准：diff3 = 0.

- 注意：MLU200 系列精度需要限定数值范围和规模大小，避免计算IOU时出现大规模随机错误。

#### 1.5.2 性能验收标准

## 2 算子接口设计

### 2.1 参考接口

- mmcv

```c++
// 给出cuDNN接口
__global__ void nms_rotated_cuda_kernel (
    const int n_boxes, const float iou_threshold, const T* dev_boxes,
    unsigned long long *dev_mask, const int multi_label);
```
```c++
// 给出cpu接口
Tensor nms_rotated_cpu_kernel (
    const Tensor dets, const Tensor scores, const float iou_threshold);
```

### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API mluOpNmsRotated(mluOpHandle_t handle,
                                            const float iou_threshold,
                                            const mluOpTensorDescriptor_t boxes_desc,
                                            const void *boxes,
                                            const mluOpTensorDescriptor_t scores_desc,
                                            const void *scores,
                                            void *workspace,
                                            size_t workspace_size,
                                            const mluOpTensorDescriptor_t output_desc,
                                            void *output,
                                            int32_t *result_num);
```

## 3 实现方案设计

### 3.1 实现方案

1. 将`boxes`中N个box拆分到一个core或一个cluster，分别对应block和union1任务类型。
2. 分别计算每个core上的最大score。如果是union1任务类型，则还需通过SRAM取得全局最大score以及对应的index和box信息。
3. 将得到的全局最大score的box信息传到每个core的NRAM中，每个core分别计算最大score的box和其他box的iou，将iou大于iou_threshold的box的score置0, 等价于移除box。
4. 重复步骤2,直至每个box都被遍历。

### 3.2 伪代码实现（可选）

### 3.3 拆分(任务拆分，多核拆分)

- 由于需要得到全局最大score的box信息，因此只能用一个job完成任务。目前仅支持N个box拆分到一个core(BLOCK任务)或者一个cluster(UNION1任务)。

### 3.4 性能优化设计

1、资源分配

| 表项 | 分配策略                                          |
| ---- | -----------------------------------------------   |
| NRAM | 存储每个box的数据以及计算所使用的中间结果         |
| SRAM | 存储每个core的最大score;存储box数据和score数据    |
| DRAM | 输入输出数据的存储                                |

2、流水设计

- 由于计算部分远远超过于 IO 部分的时间，片上 RAM 每次分配需要的空间太大，所以不划分乒乓空间，不做软流水。

### 3.5 方案理论性能

完成上述 3.1，3.2，3.3，3.4 几个步骤之后，基本可以给出一个理论性能，不需要每一个算子都有过于复杂的公式，但是一定要对自己的算子有一个心理的预期，最终实现之后的效率值是多少。

### 3.6 可维护性设计

- bangc 代码中加入必要的 log 信息，比如输入的规模、数据类型、layout 这些，以及如果出错会导致程序 core dump 的变量，比如 IO 指令的 data_size、dim xyz 的值等，这些信息都是有利于快速定位问题。

- 对每一个函数命名变量命名都有充分的注释

- 避免魔鬼数字，对于确定的数字尽量使用公共宏来替代 (宏的调用说明以及含义已经注释写在 kernels 代码中)

### 3.7 测试用例设计

- 框架在需求列表中给出的算子在网络中用到的规模：`boxes` 为 [34,5], [78,5]

- 单核BLOCK任务下：
  1. MLU590: box_num < 3000 不超时
  2. MLU370: box_num < 3200 不超时
  2. MLU290: box_num < 1100 不超时

### 3.8 算子防呆检查

- 列出算子需要做的防呆，比如

1、指针为空防呆；

2、0 元素检查防呆，VLOG(5)打印信息，是否返回与框架沟通；

3、对输入输出支持的 dtype 以及 shape 进行防呆；

4、算子自身的`iou_threshold`参数防呆。

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

### 4.2 已经过优化的规模说明

## 5 方案实施

### 5.1 开发测试计划

### 5.2 风险分析

关于IOU计算时的精度和性能问题，可参考[box_iou_rotated设计方案](../box_iou_rotated/box_iou_rotated.md)
