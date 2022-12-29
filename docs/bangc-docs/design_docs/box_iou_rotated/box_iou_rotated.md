# box iou rotated 算子开发设计方案

- #### 文档基本信息

| 算子名称    | `box_iou_rotated`     |
| ----------- | --------------------- |
| 编制人/日期 | songjin/2021-11-8     |
| 审批人/日期 | sifengyang/2021-12-12 |

- #### 修改记录

| 版本号 | 修订人  | 修订日期  | 修订描述 |
| ------ | ------- | --------- | -------- |
| V1.0   | songjin | 2021-11-8 | 首次提交 |

- #### 内容描述

本文档为`box_iou_rotated`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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

| 算子功能简介             | 简要填写算子功能，详细描述在 1.2 中进行说明                  |
| ------------------------ | ------------------------------------------------------------ |
| 需求来源                 | mmcv                                                         |
| 应用网络                 | mmdet                                                        |
| 输入数据类型             | float                                                        |
| 输入 Shape               | boxes1: [N, 5]; boxes2: [M, 5]; mode: iou/iof; aligned: bool |
| 输出数据类型             | float                                                        |
| 输出 Shape               | [N, M] if aligned == False, else [N]                         |
| 模式(可选）              | IOU or IOF                                                   |
| 是否需要支持原位         | 否                                                           |
| 是否需要支持 stride 机制 | 否                                                           |
| 是否需要支持广播         | 否                                                           |
| 0 元素检查是否直接返回   | 是                                                           |

### 1.2 算子功能和应用场景描述

BoxIouRotated 算子有 2 个输入 Tensor，分别为 Box1[N,5], Box2[M,5]，其中每个输入低维度的 5 个数字包括的信息为： `x_ctr`, `y_ctr`, `height`, `width`, `radian`.
根据参数`aligned`判断输出的交并比`IOUS`是对位计算 2 个输入 Box 的 IOUS 还是两两相交计算 IOUS。
参数`mode`为 0 时，结果为`IOU`(intersection/(area1+area2))，为 1 时，结果为`IOF`(intersection/area1).
计算过程简单描述：

1. 根据`aligned`参数判断待计算的当前 box pair 的中心点，再计算旋转后的 rotatedbox1/box2.
2. 根据 rotatedbox1/box2 的坐标点，计算每条边相交与否，是否有互相包含的情况，得到交点坐标（总共 24 种可能性）。
3. 如果当前 box pair 相交的点数大于 2 个，则计算交叠面积，否则返回当前的`IOUS`为 0.
4. 按照 Convex-hull-graham 顶点扫描法，排序、筛选得出凸包形状的顶点集合
5. 计算有效的交叠面积 intersection 后，按照`mode`参数，计算结果`IOUS`.

### 1.3 算子输入输出参数要求

| 参数        | 语义                           | 类型（输入/输出） | 支持类型                | 物理布局      | 规模限制 |
| ----------- | ------------------------------ | ----------------- | ----------------------- | ------------- | -------- |
| handle      | 操作句柄                       | 输入              | mluOpHandle_t           | /             | /        |
| boxes1_desc | 输入数据，box1 的描述符        | 输入              | mluOpTensorDescriptor_t | /             | /        |
| boxes1      | 输入数据，box1 的坐标          | 输入              | float\*                 | [N, 5]        | /        |
| boxes2_desc | 输入数据，box1 的描述符        | 输入              | mluOpTensorDescriptor_t | /             | /        |
| boxes2      | 输入数据，box2 的坐标          | 输入              | float\*                 | [M, 5]        | /        |
| mode        | 输入数据，选择交并比的计算模式 | 输入              | int                     | scalar        | /        |
| aligned     | 输入数据，选择是否对齐         | 输入              | bool                    | scalar        | /        |
| output_desc | 输入数据，输出结果的描述符     | 输入              | mluOpTensorDescriptor_t | /             | /        |
| output      | 输出数据，交并比的计算结果     | 输出              | float\*                 | [N, M] or [N] | /        |

### 1.4 算子限制

| 限制类型     | 详细说明                                                                 |
| ------------ | ------------------------------------------------------------------------ |
| 输入限制     | 输入参数`boxes1` `boxes2` shape 必须满足要求: boxes1[N, 5] boxes2:[M, 5] |
| 输入限制     | 输入 `boxes1` `boxes2` 不支持输入 nan 或 inf                             |
| 输出限制     | 输出参数`output` shape 必须满足: [N, M] or [N]                           |
| 输入限制     | 输入 `mode` 仅支持输入 0 或 1                                            |
| 输入限制     | 输入 `aligned` 仅支持输入 True 或 False                                  |
| 输入限制     | 当输入参数`aligned`为 Ture 时，需要保证`boxes1`和`boxes2`中的第一维相等  |
| 数据类型限制 | 输入 `boxes1` `boxes2` 仅支持 float 类型                                 |
| 原位限制     | 不支持原位                                                               |
| stride 限制  | 不支持 stride 机制                                                       |
| 广播限制     | 不支持广播                                                               |

### 1.5 验收标准

#### 1.5.1 精度验收标准

该算子使用了 sin/cos 三角函数，计算旋转后的顶点坐标，还涉及 div 除法操作， MLU200 系列上使用 reciphp 激活计算，MLU300 上使用超越函数指令。
mmcv 框架中使用了 numpy.allclose 函数，参数 atol=1e-4, 绝对误差小于 1e-4 的精度阈值标准。

综上，该算子采用静态阈值，采用阈值标准：diff1<=3e-3 && diff2 <= 3e-3.

- 注意：MLU200 系列精度需要限定数值范围和规模大小，避免出现大规模随机错误导致 Diff2 挂掉。

#### 1.5.2 性能验收标准

## 2 算子接口设计

### 2.1 参考接口

- mmcv

```c++
// 给出cuDNN接口
__global__ void box_iou_rotated_cuda_kernel (
    const int n_boxes1, const int n_boxes2, const T* dev_boxes1,
    const T* dev_boxes2, T* dev_ious, const int mode_flag, const bool aligned);
```

### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API mluOpBoxIouRotated(mluOpHandle_t handle,
                                               const int mode,
                                               const bool aligned,
                                               const mluOpTensorDescriptor_t bbox1_desc,
                                               const void *bbox1,
                                               const mluOpTensorDescriptor_t bbox2_desc,
                                               const void *bbox2,
                                               const mluOpTensorDescriptor_t ious_desc,
                                               void *ious);
```

## 3 实现方案设计

### 3.1 实现方案

参考文档：[伪代码实现](./pseudo_code)

- 整体逻辑，任务拆分思路：

1. aligned=true, box1 和 box2 是做对位的`box_iou_rotated`计算，参考 element-wise 的算子方案，输出维度为(N)。 (N==M)
2. aligned=false, 则 box1 每次只取一个标量，与片上长向量 box2 做“向量标量计算”，存储连续的 iou-pairs 在 box2 的维度(M)，输出维度为 (N, M)。
   每个 ipu core 做 box2 的全循环，存储完整的 M 维度的输出，不同的 ipu core、cluster 以不同的 box1 作为任务划分。

- 计算步骤：

1. load boxes1、boxes2 到片上，进行转置，区分 MLU arch 进行转置，因为原始输入的 shape 是 (N, 5), (M, 5)，低维度是 5，无法向量化计算，且不对齐。
2. 计算 Area1、Area2，如果是 aligned=false，则在第一次 load 该 box1 的时候计算，判断 Area<1e-14；如果小于，则该 box1 对应的所有的 ious 赋值为 0. 另外向量化判断 Area2，不符合条件的设置`valid_box`为 false，表示该 box-pair 的 iou 值是 0，无需后续计算（后续计算过程仍然存在冗余的向量计算）。
3. 计算`new_box_pts`，向量化地计算新的`x_ctr`, `y_ctr`，但是 width/height/theta 都可以来源于输入，无需复制
4. 通过`new_pts`的数据，计算旋转后的顶点`rotated_vertices`，通过旋转后的顶点，计算每条边的`vector`的表示，通过 3 个过程，计算 24 种可能的交点。由于向量化计算，每组 box-pair 的有效交点个数和类型都不同，需要设置`valid_pts`对应位置是否为 true，标记该位置是否为有效交点。
5. 通过上述过程得到的有效交点，如果`nums_in`对应大于 2，代表有 2 个以上的交点，可以计算交叠面积，否则设置对应位置的`valid_box`为 false，设置 ious=0.
6. 用 Convex-hull-graham 顶点扫描法，标量计算每一组 24 个交点，进行排序、筛选得出凸包形状的顶点集合，最后使用`polygon_area`函数计算有效 box 的 iou 面积。然后再根据`mode`的不同，做不同分母的除法（区分 arch）。

### 3.2 伪代码实现（可选）

参考文档：[伪代码实现](./pseudo_code)

### 3.3 拆分(任务拆分，多核拆分)

参考文档：[伪代码实现](./pseudo_code)
拆分建议分几个维度来写，job 间拆分，cluster 间拆分，cluster 内拆分：

1、基本任务类型是什么：U1（Block 类型也支持，如果该 MLU arch 只支持 Block 任务类型，或者 1 个 ipu core 就足够了的话）
如果 aligned=false, job 间拆分 & cluster 间拆分 & cluster 内拆分：num of boxes1, 每个 IPU CORE 对 box2 做完整的循环运算；
如果 aligned=true， 则对 num of boxes 划分，每个 IPU CORE 内做 element-wise 的计算；

### 3.4 性能优化设计

1、资源分配

| 表项 | 分配策略                                                                         |
| ---- | -------------------------------------------------------------------------------- |
| NRAM | 260xM(aligned),258xM(non-aligned), 查看 kernel 代码中，开头注释图，对 RAM 的划分 |
| DRAM | 不需要额外的 DRAM 空间，直接存放 output 的结果                                   |

2、流水设计
由于计算部分远远超过于 IO 部分的时间，片上 RAM 每次分配需要的空间太大，所以不划分乒乓空间，不做软流水。

### 3.5 方案理论性能

完成上述 3.1，3.2，3.3，3.4 几个步骤之后，基本可以给出一个理论性能，不需要每一个算子都有过于复杂的公式，但是一定要对自己的算子有一个心理的预期，最终实现之后的效率值是多少。

由于 伪代码 part 8. Convex-Hull-Graham 的算法目前设计为标量循环实现，所以性能暂无估计，片上时间复杂度 O(24x24xM)，其他部分已做了向量化的部分优化，时间复杂度为 O(M).
在 aligned=false 的情况，片外循环的时间复杂度是 O(NxM)，否则则为 O(M).

- aligned = true 的情况： O(60000xN)
- aligned = false 的情况： O(66000xNxM)

实际由于标量计算占比时间很大，会有额外的寄存器换入换出操作，以及额外的间接寻址时间，造成理论预估时间不准确。

### 3.6 可维护性设计

1、bangc 代码中加入必要的 log 信息，比如输入的规模、数据类型、layout 这些，以及如果出错会导致程序 core dump 的变量，比如 IO 指令的 data_size、dim xyz 的值等，这些信息都是有利于快速定位问题。

2、对每一个函数命名变量命名都有充分的注释

3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代 (宏的调用说明以及含义已经注释写在 kernels 代码中)

### 3.7 测试用例设计

- 框架在需求列表中给出的算子在网络中用到的规模(无，框架没提供)：

- 边界 case：

  1. aligned=true, 1 ipu core(Block), 1 cluster, 4 cluster, 6 cluster, 8 cluster, 16 cluster, with/without remainder.
  2. aligned=false, box1 once, box2 once ...
  3. area1/2 has/are all 1e-14.
  4. intersection points, all 24 conditions, rect1 in 2, 2 in 1, `nums_in` is 1/2/more points.
  5. `convex_hull_graham` func, 24 points exist same `min_y_value`, but different `min_x_value`...
  6. 排序、扫描部分的代码行覆盖率。

- NOTE: 可以考虑在每组生成 box 坐标之后，遍历旋转角度 theta 生成 N 组 boxes 输入。

### 3.8 算子防呆检查

- 列出算子需要做的防呆，比如

1、指针为空防呆；

2、0 元素检查防呆，VLOG(5)打印信息，是否返回与框架沟通；

3、对输入输出支持的 dtype 以及 shape 进行防呆；

4、算子自身的`mode`、`aligned`参数防呆。

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

### 4.2 已经过优化的规模说明

## 5 方案实施

### 5.1 开发测试计划

### 5.2 风险分析

性能：`convex_hull_graham`函数中对于交点的角度排序、入栈出栈的扫描过程由标量实现，性能较差。

- 当 aligned=True 的时候，` num_box1`与`num_box2 `在 100k 量级时，也不会超时（迫于 prototxt 大小限制，更大的 case 没有测试）。
- 当 aligned=False 的时候，`num_box1`与`num_box2`都在 3k 量级的时候，就有超时风险了。

精度(MLU200 系列)：

1. `sin` `cos` 三角函数精度不足，采用了`__bang_taylor4_sin/cos`进行精度优化。在计算 box 旋转顶点(rotatedPts)的时候，会引入初始数据误差，导致后续的结果误差。
2. `active_reciphp` 倒数激活函数精度不足，且后续需要将倒数后的结果 `t1`、`t2`，与`0`和`1`做大小比较，在极端情况下，可能会出现边界判断错误的情况，导致交点个数不准确，后续的面积计算偏差巨大，造成单点误差巨大。
3. nan/inf 与 mmcv cuda 实现无法对齐，由于在 convex-hull-graham 求左下角最小点时，使用了 minpool 操作，硬件指令功能限制无法与 mmcv 结果对齐。已在 mlu_op.h 中说明。

精度(MLU300 系列）：

1. 在 MLU 300 系列硬件上，sin/cos 三角函数无法与 mmcv 框架中的 CPU、Cuda 源码 double 实现 bit 级一致，即使在该例子中，引入的绝对误差在 1e-9 量级，但是当多个交点在 dist 计算时的前后大小关系不同，可能会发生后续逻辑中排序的执行分支不同，导致有效交点个数不同，当交点数小于等于 2 的时候，会导致最终该 box-pair 的 IOU 结果为 0，从而导致结果误差巨大，无法通过测试。
