# box_overlap_bev 算子开发设计方案


* #### 文档基本信息

| 算子名称    | box_overlap_bev                                            |
| ----------- | ------------------------------------------------------------ |
| 编制人/日期 | lijiawei/2026-8-17                                           |

* #### 修改记录

| 版本号| 修订人 | 修订日期 | 修订描述 |
| ----- | ------ | -------  | -------  |
| V1.0  |   lijiawei  | 2026-8-17 | 首次提交 |

* #### 内容描述

本文档为`box_overlap_bev`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

* #### 算子需求checklist

算子需求提出者需要`提供`的信息如下：


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

| 算子功能简介| 计算两个旋转框边界框之间的重叠面积（overlap）                              |
|-------------|--------------------------------------------------------------|
| 需求来源    | PyTorch//...                                       |
| 应用网络    | pointopillar/...                                                 |
| 输入数据类型|  float                                                  |
| 输入Shape   | boxes1: [N,7]; boxes2: [M,7]    |
| 输入Layout  | boxes1: ARRAY; boxes2: ARRAY                                  |
| 输出数据类型|  float                                                  |
| 输出Shape   | [N,M]                                  |
| 输出Layout  | ARRAY                                                         |
| 模式(可选） |                      无                                        |
| 是否含有dim/axis等类似语义的参数且该参数支持负数/其他特殊处理 | 否 |
| 是否含有labels/index等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 否|
| 是否需要支持原位        | 否                                                  |
| 是否需要支持stride机制  | 否                                                  |
| 是否需要支持广播  |  否                        |
| 0元素检查是否直接返回  | 是，return MLUOPS_STATUS_SUCCESS                                                    |
| 其他特殊需求(在线量化，融合，转数提前等，可选)|                                                        |
| 本次开发优先支持的规模/模式|                     无                                                  |


### 1.2 算子功能和应用场景描述

算子功能： 根据输入 boxes1(N 个 box)，boxes2(M 个 box)，计算 box 两两之间重叠的面积，输入 tensor.shape=[N, M]

应用场景： 该算子应用于pointpillar网络。

example:

    boxes1:  shape is [2, 7]
             tensor([[1., 2., 3., 4., 5., 6., 7.], 
                     [2., 3., 4., 5., 6., 7., 8.]],
                    dtype=np.float32)

    boxes2: shape is [3, 7]
            tensor([[1., 2., 3., 4., 5., 6., 20.],
                    [2., 3., 4., 5., 6., 7., 21.],
                    [3., 4., 5., 6., 7., 8., 22.]],
                    dtype=np.float32)
    
    outout: shape is [2, 3]
            tensor([[17.0636, 17.1154, 12.4745],
                    [16.0011, 25.6561, 24.7464]],
                    device='cuda:0')


### 1.3 算子输入输出参数要求

| 参数             | 语义                           | 类型（输入/输出） | 支持类型               | 物理布局 | 规模限制 |
| ---------------- | ------------------------------ | ----------------- | ---------------------- | -------- | -------- |
| handle           |        操作句柄                        | 输入              |                        | /        | 无       |
| boxes1_desc      |    输入数据boxes1的形状描述             | 输入              |                        | /        | 无       |
| boxes1           |    指向boxes1数据的mlu地址的指针        | 输入              | float             | ARRAY    | dim = 2,shape[1] =7   |
| boxes2_desc      |    输入数据boxes2的形状描述             | 输入              |                        | /        | 无       |
| boxes2           |    指向boxes2数据的mlu地址的指针        | 输入              | float             | ARRAY    | dim = 2,shape[1] =7   |
| output_desc      |    输出数据output的形状描述             | 输入              |                        | /        | 无       |
| output           |    指向output数据的mlu地址的指针        | 输出              | float              | ARRAY    | 无       |

### 1.4 算子限制

| 限制类型    | 详细说明                  |
| ----------- | ------------------------- |
| 原位限制    | 不支持原位|
| stride限制  | 不支持stride机制|
| 广播限制    | 不支持广播|
| 输入限制    | boxes1、boxes2为二维张量，须符合boxes1.shape[1]==boxes2.shape[1]==7，boxes1/2.shape[0]取值范围[1,1000],hw取值范围需满足[0,1]以上 |

### 1.5 验收标准

#### 1.5.1 精度验收标准

- 设置静态阈值，采用阈值标准：diff1<=3e-3 && diff2 <= 3e-3。

#### 1.5.2 性能验收标准

## 2 算子接口设计

### 2.1 参考接口

- PyTorch
```c++
__global__ void boxes_overlap_kernel(const int num_a, const float *boxes_a,
                                     const int num_b, const float *boxes_b,
                                     float *ans_overlap);
```

### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API
mluOpBoxOverlapBev(mluOpHandle_t handle,
                   const mluOpTensorDescriptor_t boxes1_desc,
                   const void *boxes1,
                   const mluOpTensorDescriptor_t boxes2_desc,
                   const void *boxes2,
                   const mluOpTensorDescriptor_t overlaps_desc,
                   void *overlaps);
```


## 3 实现方案设计

### 3.1 实现方案

BoxOverlapBev 算子有2个输入Tensor，分别为 Box1[N,7], Box2[M,7]，其中每个输入低维度的7个数字包括的信息为： `x`, `y`, `z`, `dx`, `dy`, `dz`, `heading`.

cuda中overlap计算过程简单描述：
1. Box1/Box2分别根据中心点坐标（x，y）以及宽高dx、dy计算得到四个顶点坐标box1_corners[4]/box2_corners[4]。
2. Box1/Box2根据中心点坐标（x，y）、旋转角（heading）、四个顶点坐标分别计算出逆时针旋转heading度后的四个顶点坐标rotated_box1_corner[s4]/rotated_box2_corners[4]。
3. 根据旋转后顶点坐标rotated_box1_corner[s4]/rotated_box2_corners[4]计算出两个box四条边之间的交点坐标cross_point[16]，更新最后得到重叠面积中心点坐标poly_center。
4. 分别计算各个顶点rotated_box1_corner[s4]/rotated_box2_corners[4]在另一个box的包含情况，如果包含则将顶点增加到cross_point[16]中，更新多边形中心点坐标poly_center，至此多边形所有顶点坐标统计完成。
5. 根据atan2(顶点坐标-poly_center)降序对所有顶点进行排序，得到排序后的顶点sort_cross_point[16]。
6. 以sort_cross_point[0]为起始点以及相邻两个顶点遍历求和，所有面积之和即为overlap重叠面积。

设计方案如下：

1. 加载分到每个task的boxes1以及boxes2到片上，此处加载到narm只加载每个输入低维度的7个参数的5个参数：`x`, `y`,  `dx`, `dy`, `heading`.
2. 计算的当前box pair的中心点，再计算旋转后的rotatedbox1/box2。
3. 根据rotatedbox1/box2的坐标点，计算每条边相交与否，是否有互相包含的情况，得到交点坐标（总共24种可能性）。
4. 如果当前box pair相交的点数大于2个，则计算交叠面积，否则返回当前的`overlap`为0。
5. 按照 Convex-hull-graham 顶点扫描法，排序、筛选得出凸包形状的顶点集合。
6. 计算有效的交叠面`overlap`.

### 3.2 伪代码实现（可选）

triple-cycle:
(load N and M, M is large, N is same as M, loop N onchip;
loop N offchip, reuse large M compute result onchip, mark first_box2;
loop M offchip, compute large M compute, restore N offchip offset back to 0)

For MLU core computing algorithm:
1. load input box1 and box2. 
2. transpose (distinguish MLU arch >= 300)
3. calculate new points and area, set which area < 1e-14 valid_box = false
4. calculate rotated vertices
5. calculate intersection points, set which nums_in <= 2 valid_box = false
6. convex-hull-graham, set which num_convex <= 2 valid_box = false
7. compute polygon area

### 3.3 拆分(任务拆分，多核拆分)
  每个MLU core做box2的全循环，存储完整的M维度的输出，不同的MLU core、cluster以不同的boxes1作为任务划分。
  boxes1每次只取一个标量，与片上长向量boxes2做“向量标量计算”，存储连续的overlaps-pairs在boxes2的维度(M)，输出维度为 (N, M)。
  - 计算步骤：
  1. load boxes1、boxes2到片上，进行转置，区分MLU arch进行转置，因为原始输入的shape是 (N, 7), (M, 7)，低维度是7，无法向量化计算，且不对齐。
  2. 计算Area1、Area2，在第一次load该boxes1的时候计算，判断Area<1e-14；如果小于，则该boxes1对应的所有的overlaps赋值为0. 另外向量化判断Area2，不符合条件的设置`valid_box`为false，表示该box-pair的overlap值是0，无需后续计算（后续计算过程仍然存在冗余的向量计算）。
  3. 计算`new_box_pts`，向量化地计算新的`x_ctr`, `y_ctr`，但是width/height/theta都可以来源于输入，无需复制
  4. 通过`new_pts`的数据，计算旋转后的顶点`rotated_vertices`，通过旋转后的顶点，计算每条边的`vector`的表示，通过3个过程，计算24种可能的交点。由于向量化计算，每组box-pair的有效交点个数和类型都不同，需要设置`valid_pts`对应位置是否为true，标记该位置是否为有效交点。
  5. 通过上述过程得到的有效交点，如果`nums_in`对应大于2，代表有2个以上的交点，可以计算交叠面积，否则设置对应位置的`valid_box`为false，设置ioverlas=0.
  6. 用 Convex-hull-graham 顶点扫描法，标量计算每一组24个交点，进行排序、筛选得出凸包形状的顶点集合，最后使用`polygon_area`函数计算有效box的overlap面积。

### 3.4 性能优化设计
1、资源分配

 * NRAM buffer
 * BOXES1 cannot be over-written inside loop of box1_onchip
 * BOX2_TRANS cannot be over-written inside loop of box2_loop
                      Total: 258 copies of max_box_pair ram


---------------------------------------------------------------------------
|final data |  box1_onchip  | box2_onchip |   box1_trans  |   box2_trans  |
| ----------| --------------| ------------| --------------|---------------| 
|    2xN    |     64 x N    |   64 x N    |      64 x N   |     64 x N    |
| valid_box |   5xN  BOXES1   |  2x4*2 x/y  | 5xN broadcast |5xN BOX2_TRANS |
|    area2  |   temp 1~5    | rotated_vert|  17~64 24*2   | 6~10 new_pts2 |
|           |    nums_in    |  2*4*2 x/y  |   intersectPts | 17~64 24*2   |
|           |   temp 6~10   |   vec1/2    |               |   orderedPts  |
|           |  dist 17~40   |             |               |               |
|            |valid_pts 41~64|            |               |                |

2、流水设计

由于计算部分远远超过于IO部分的时间，片上RAM每次分配需要的空间太大，所以不划分乒乓空间，不做软流水(后期如果对性能有要求可以加上)。

### 3.5 方案理论性能

Convex-Hull-Graham排序顶点算法目前设计为标量循环实现，所以性能暂无估计，片上时间复杂度O(24x24xM)，其他部分已做了向量化的部分优化，时间复杂度为O(M).
片外循环的时间复杂度是O(NxM).
实际由于标量计算占比时间很大，会有额外的寄存器换入换出操作，以及额外的间接寻址时间，造成理论预估时间不准确。

### 3.6 可维护性设计

1、bangc代码中加入必要的 log信息，比如输入的规模、数据类型、layout这些，以及如果出错会导致程序core dump的变量，比如IO指令的data_size、dim xyz的值等，这些信息都是有利于快速定位问题。   (待整理完善，对log进行规范)

2、对每一个函数命名变量命名都有充分的注释

3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代   (待提供公共宏文档)

### 3.7 测试用例设计

- 框架在需求列表中给出的算子在网络中用到的规模：
   
1. boxes1:(188,7)、 boxes2(23,7)。
   
  尽量覆盖：boxes1:(1~1000,7) boxes2(1~1000,7)

- 边界case：

1. area1/2 has/are all 1e-14.

2. intersection points, all 24 conditions, rect1 in 2, 2 in 1, `nums_in` is 1/2/more points.

3. `convex_hull_graham` func, 24 points exist same `min_y_value`, but different `min_x_value`...

4. 排序、扫描部分的代码行覆盖率。
  NOTE: 可以考虑在每组生成box坐标之后，遍历旋转角度生成N组boxes输入。

其他可根据需要进行补充。算子开发完毕后，补充测试报告链接。

### 3.8 算子防呆检查

  1、指针为空防呆；

  2、0元素检查防呆，VLOG(5)打印信息；

  3、对输入输出支持的dtype以及shape进行防呆；

## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

### 4.2 已经过优化的规模说明
