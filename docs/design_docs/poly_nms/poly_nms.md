# poly_nms 算子开发设计方案
* #### 文档基本信息
| 算子名称    | poly_nms     |
| ----------- | ----------------- |
| 编制人/日期 |  谷中豪/2022-05-31 |

* #### 修改记录
| 版本号| 修订人 | 修订日期 | 修订描述 |
| ----- | ------ | -------  | -------  |
| V1.0  | 谷中豪   | 2022-05-31 | 首次提交 |
| V2.0  | 谷中豪   | 2022-08-09 | 取消输入boxes顶点坐标全是顺时针或者全是逆时针的限制 |

* #### 内容描述
本文档为`poly_nms`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录。

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
| 算子功能简介| 多边形的非极大值抑制，用于删除高度冗余的多边形输入框 |
|-------------|--------------------------------------------------------------|
| 需求来源    | PyTorch                                      |
| 应用网络    | FasterRCNN trans obb                                         |
| 输入数据类型| float                                                  |
| 输入Shape   | boxes:dim[N, 9], iou_threshold:float |
| 输入Layout  | boxes:ARRAY, iou_threshold:标量                              |
| 输出数据类型 | int32_t                                         |
| 输出Shape   | output:dim[N]，输出实际长度为result_num; result_num:dim=1      |
| 输出Layout  | output:ARRAY，result_num::ARRAY                                     |
| 模式(可选） | 否 |
| 是否含有dim/axis等类似语义的参数且该参数支持负数/其他特殊处理 | 否 |
| 是否含有labels/index等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 否 |
| 是否需要支持原位        | 否                                                  |
| 是否需要支持stride机制  | 否                                                 |
| 是否需要支持广播  | 否                       |
| 0元素检查是否直接返回  | 是 (返回MLUOP_STATUS_SUCCESS)                                              |
| 其他特殊需求(在线量化，融合，转数提前等，可选)|        无                                                |
| 本次开发优先支持的规模/模式|   |

### 1.2 算子功能和应用场景描述
**算子功能：** `poly_nms`(polygon nms)算子用于计算多边形非极大值抑制，删除高度冗余的多边形框。
**应用场景：** `poly_nms`算子应用于`FasterRCNN`，`trans obb`网络。

**AerialDetection 示例：**
```py
def pnms_test():
    nms_thresh=0.1
    dets = np.asarray(np.random.randn(70，9)， dtype=np.float32)
    boxes_gpu=torch.from_numpy(dets).cuda()
    keep = poly_nms(boxes_gpu， nms_thresh)
    print("output:")
    print(keep[1])

output:
# 输出为[0]
dets = [[0， 0， 1， 0， 1， 1， 0， 1， 3]]

# box2 和 box0，box1 没有交集，box1 和 box0 有交集，且iou>0.1，输出结果为[1，2]
dets = [[0， 0， 1， 0， 1， 1， 0， 1， 1]， [0.5， 0.5， 1.5， 0.5， 1.5， 1.5， 0.5， 1.5， 2]，[0， 0， -0.5， 0， -0.5， -0.5， 0， -0.5， 3]]

# box0 和 box1，box2 均有交集，且iou>0.1，输出为[0]
dets =  [[0， 0， 1， 0， 1， 1， 0， 1， 3]， [0.5， 0.5， 1.5， 0.5， 1.5， 1.5， 0.5， 1.5， 2]，[0， 0， 0.5， 0， 0.5， 0.5， 0， 0.5， 1]]

# box0 和 box1 有交集，且iou=0.1428 > nms_thresh(0.1)，box0，box1 都和 box2 没交集，输出为[0，2]
dets = [[0， 0， 1， 0， 1， 1， 0， 1， 3]， [0.5， 0.5， 1.5， 0.5， 1.5， 1.5， 0.5， 1.5， 2]，[0， 0， -0.5， 0， -0.5， -0.5， 0， -0.5， 1]]

# 以下为nan，inf，-inf行为
# 输入scores包含inf: box0 和 box1 相交， box2 和 box0，box1 不相交，输出[0，2]
dets = [[0， 0， 2， 0， 2， 2， 0， 2， np.inf]， [1.5， 1.5， 2.5， 1.5， 2.5， 2.5， 1.5， 2.5， 1]，[0， 0， -0.5， 0， -0.5， -0.5， 0， -0.5， 3]]

# 输入scores包含-inf: box0 和 box1 相交， box2 和 box0，box1 不相交，输出[1，2]
dets = [[0， 0， 2， 0， 2， 2， 0， 2， -np.inf]， [1.5， 1.5， 2.5， 1.5， 2.5， 2.5， 1.5， 2.5， 1]，[0， 0， -0.5， 0， -0.5， -0.5， 0， -0.5， 3]]

# 输入scores有nan，box0 和box1 相交且iou > iout_thresh，输出较大score的box，最终输出[0,2]
dets = [[0, 0, 2, 0, 2, 2, 0, 2, np.nan], [1.5, 1.5, 2.5, 1.5, 2.5, 2.5, 1.5, 2.5, 1],[0, 0, -0.5, 0, -0.5, -0.5, 0, -0.5, 3]]

# 输入box1坐标包含inf，认为该box和其他boxes不相交， box1， box2 也不相交， 输出[0,1,2] 
dets = [[np.inf, 0, 2, 0, 2, 2, np.inf, 2, 2], [1.5, 1.5, 2.5, 1.5, 2.5, 2.5, 1.5, 2.5, 1],[0, 0, -0.5, 0, -0.5, -0.5, 0, -0.5, 3]]

# 输入boxes坐标包含inf，认为该box和其他boxes不相交, 输出[0,1,2] 
dets = [[0, 0, np.inf, np.inf, 2, 2, 0, 2, 2], [1.5, 1.5, np.inf, np.inf, 2.5, 2.5, 1.5, 2.5, 1],[0, 0, -0.5, 0, -0.5, -0.5, 0, -0.5, 3]]

# 输入box1坐标包含nan，认为和其他boxes不相交， box1 和 box2 之间也不相交，输出[0,1,2] 
dets = [[0, 0, 2, 0, 2, 2, 0, np.nan, 2], [1.5, 1.5, 2.5, 1.5, 2.5, 2.5, 1.5, 2.5, 1],[0, 0, -0.5, 0, -0.5, -0.5, 0, -0.5, 3]]
```

### 1.3 算子输入输出参数要求
| 参数             | 语义                           | 类型（输入/输出） | 支持类型               | 物理布局 | 规模限制 |
| ---------------- | ------------------------------ | ----------------- | ---------------------- | -------- | -------- |
| **handle**           |        操作句柄                        | 输入              |    mluOpHandle_t   | /        | 无       |
| **boxes_desc**      |    输入boxes的形状描述             | 输入              |           /             | /        | 无       |
| **boxes**         |    计算所需的输入框        | 输入              | float             | ARRAY    | dim=2， shape[1] =9 |
| **iou_threshold**   |   计算所需的阈值             | 输入              |    float                   | /       | 标量       |
| **workspace**        |   指向额外GDRAM空间的指针          | 输入             |  void *                  | /          | 无       |
| **workspace_size**   |   输入参数，workspace的空间大小   | 输入             |  size_t                  | /          | 无       |
| **output_desc**      |  输出数据output的形状描述       | 输出       |   /     | /          | 无       |
| **output**          |  指向output数据的mlu地址的指针    | 输出       |  int32_t      | ARRAY     |dim=1，shape[0]==boxes.shape[0]       |
| **result_num**      |  指向result_num数据的mlu地址的指针,表示output实际输出index的个数    | 输出       |  int32_t      | ARRAY     |dim=1，shape[0]=1 |

### 1.4 算子限制
| 限制类型     | 详细说明                                                     |
| ------------ | ------------------------------------------------------------ |
| 输入限制     |  输入boxes必须满足dim=2，shape[1]=9；输入input1必须满足格式:[[x1， y1， x2， y2， x3， y3， x4， y4， score]，..]  |
| 输入限制     |  输入boxes的的每个box顶点坐标必须是顺时针排序或者逆时针排序，boxes顶点坐标有乱序情况不保证计算结果且与竞品算子计算结果一致。 |
| 输入限制     |  输入不支持nan,inf |
| 输入限制     |  输入boxes中有相同的score的case，输出结果可能与竞品结果不一致。比如输入两个box且其score值相同，假定其iou不大于给定的iou阈值，此时两个box都是满足输出条件，则两个box的index都会输出，此时竞品只输出一个box的index。 |
| 数据类型限制 | 只支持float输入  |
| 数据范围限制 | 无 |
|  原位限制     | 不支持原位                                                 |
| stride限制   | 不支持stride                                      |
| 广播限制     | 不支持广播                                                   |
| 规模限制     | mlu270,mlu290及mlu370上输入boxes个数不超过9770个，超过规模限制会有打印报错日志。|


### 1.5 验收标准
#### 1.5.1 精度验收标准
该算子为算术类算子，采用当前的 diff3 评价公式，验收标准为：
- 静态阈值 diff3 == 0

#### 1.5.2 性能验收标准
- 网络中使用到的规模性能优于或至少与竞品性能持平。
- 部分与竞品差距过大的规模在4.算子性能优化记录中进行说明。
- 附上算子测试报告链接，测试报告必须包括框架给出的网络中规模的性能数据以及对应效率值。

**竞品性能测试**
在Tesla V100-SXM2-16GB平台上测试poly_nms算子性能；
需求未提供的网络中算子规模， 借鉴nms算子规模([70，9]，[119，9])，并补充规模（[500，9] [1000，9] [2000，9]）进行性能分析，循环调用算子100次取得平均性能结果如下：
测试规模[70，9]，[119，9] [500，9] [1000，9] [2000，9]，iou_thresh=1.0
测试环境： Tesla V100-SXM2-16GB +  PyTorch 1.6.0

| 平台                 | 框架版本      | 数据类型 | 规模     | 计算效率  | IO效率    | Hardware time(us) |
| -------------------- | ------------- | -------- | --------  | --------- | --------- | ----------------- |
| Tesla V100-SXM2-16GB | Pytorch 1.6.0 | float    | [70， 9]   | 0.118567% | 0.157380% | 9348.685          |
|                      |               | float    | [119， 9]  | 0.288131% | 0.493981% | 9319.030     |
|                      |               | float    | [500， 9]  | 5.471360% | 10.057026% | 9324.388            |
|                      |               | float    | [1000， 9] | 20.845317% |35.143975% | 10798.381          |
|                      |               | float    | [2000， 9] | 21.691825%    | 45.542830% |  41368.463          |

## 2 算子接口设计
### 2.1 参考接口
- **AerialDetection** https://github.com/dingjiansw101/AerialDetection/blob/master/mmdet/ops/poly_nms/src/poly_nms_cuda.cpp
```c++
at::Tensor poly_nms_cuda(const at::Tensor boxes， float nms_overlap_thresh);
```

### 2.2 接口设计
#### 2.2.1 poly_nms获取额外申请空间大小
```c++
mluOpStatus_t MLUOP_WIN_API
mluOpGetPolyNmsWorkspaceSize(mluOpHandle_t handle,
                             const mluOpTensorDescriptor_t boxes_desc,
                             size_t *size);
```
**参数描述：**
- `handle`：输入参数。操作句柄，内部绑定device和对应的queue。
- `size`：输入参数。需要用户申请的额外的空间大小，通过`mluOpGetPnmsWorkspaceSize`获取。
- `mluOpTensorDescriptor_t`: 输入tensor的形状描述。

#### 2.2.2 poly_nms计算接口
```c++
mluOpStatus_t MLUOP_WIN_API mluOpPolyNms(mluOpHandle_t handle,
                                        const mluOpTensorDescriptor_t boxes_desc,
                                        const void *boxes,
                                        float iou_threshold,
                                        void *workspace,
                                        size_t workspace_size,
                                        const mluOpTensorDescriptor_t output_desc,
                                        void *output,
                                        void *result_num);
```

## 3 实现方案设计
### 3.1 实现方案
`poly_nms`(polygon nms)算子用于计算多边形非极大值抑制， 删除冗余的多边形框。
poly_nms算子有两个输入，input1是2维Tensor，包含四边形的四个顶点坐标及其对应的score，具体信息为：
[[x1， y1， x2， y2， x3， y3， x4， y4， score]，...];
input2 是float数，是给定的iou的阈值iou_thresh。
- **poly_nms算子CPU实现**
1. 将scores降序排序；
2. 用score最大的box分别和其余的box做iou计算， 如果iou大于iou_thresh，认为这两个box相交，删除score值小的box；
3. 再选取次大的score，重复第二步计算；
4. 输出剩于box的index(升序输出)。

- **MLU实现步骤**
1. 实现过程主要分为三个kernel进行。
2. MLUCalcArea：用来计算输入不规则四边形的面积，计算结果保到workspace中；
2. MLUGenNMSMask： 用来计算每两个box之间iou，和给定的iou_threshold进行比较，对比结果生成N*N的mask矩阵。
3. MLUGenNMSResult： 根据输入boxes的score顺序，从mask中选取符合阈值条件的box，并输出对应的index。

- **计算不规则四边形IOU**
1. 计算overlap：参考竞品计算两个四边形overlap的计算方法：https://github.com/dingjiansw101/AerialDetection/blob/master/mmdet/ops/poly_nms/src/poly_nms_kernel.cu#L144；
2. 计算四边形面积box1_area1，box2_area：不规则四边形面积计算使用叉乘方法计算；
3. iou = overlap / (box1_area + box2_area - overlap)。

- **不规则四边形面积计算**
  已知四边形四个顶点坐标(x1，y1)， (x2，y2)， (x3，y3)， (x4，y4)
```c++
// 向量计算
box_area = 1/2 * ((x1*y2 - y1*x2) + (x2*y3-y2*x3) + (x3*y4 - y3*x4) + (x4*y1 - y4*x1))

// 标量计算
p[4] = p[0];
for(int i = 0;i<4;i++)
{
  ret += p[i].x * p[i+1].y - p[i].y * p[i+1].x；
}
box_area = ret/2;
```
### 3.2 伪代码实现
```c++
...
// 1. 计算四边形面积
_mlu_func__ static void polyArea(float *__restrict__ nram_tile_beg,
                            int i_tile_size,
                            float *__restrict__ area_buffer) {
  float *ptrx = nram_tile_beg;
  float *ptry = nram_tile_beg + i_tile_size;
  float *ptr0 = nram_tile_beg + 2 * i_tile_size;
  float *ptr1 = nram_tile_beg + 3 * i_tile_size;
  float *ptr2 = nram_tile_beg + 4 * i_tile_size;
  float *ptr3 = nram_tile_beg + 5 * i_tile_size;
  int stride = 2 * i_tile_size;

  for (int i = 1; i < 3; i++) {
    if (i == 1) {
      __bang_sub(ptr0, ptr0, ptrx, i_tile_size);
      __bang_sub(ptr1, ptr1, ptry, i_tile_size);
    }
    __bang_sub(ptr2, ptr2, ptrx, i_tile_size);
    __bang_sub(ptr3, ptr3, ptry, i_tile_size);

    __bang_mul(ptr0, ptr0, ptr3, i_tile_size);
    __bang_mul(ptr1, ptr1, ptr2, i_tile_size);
    if (i == 1) {
      __bang_sub(area_buffer, ptr0, ptr1, i_tile_size);
    } else {
      __bang_sub(ptr0, ptr0, ptr1, i_tile_size);
      __bang_add(area_buffer, ptr0, area_buffer, i_tile_size);
    }
    ptr0 = ptr2;
    ptr1 = ptr3;
    ptr2 = ptr2 + stride;
    ptr3 = ptr3 + stride;
  }
  __bang_mul_scalar(area_buffer, area_buffer, 0.5, i_tile_size);
  __bang_abs(area_buffer, area_buffer, i_tile_size);
}

// 2. 计算不规则四边形overlap，并生成每个box的iou是否大于iou_threshold的mask矩阵
template <PointDirection POINT_DIR>
__mlu_func__ static void MLUGenNmsMaskImpl(
    const float *__restrict__ input_boxes, int input_boxes_num, int real_width,
    float threshold, const float *__restrict__ boxes_area, uint32_t *mask,
    int *sort_info) {
  // TODO(ZW): support larger size.
  int mask_col_num = (input_boxes_num + MASK_T_BITWIDTH - 1) / MASK_T_BITWIDTH;

  // nram: | box_buffer    | area_buffer | mask_buffer  | mask_buffer_swap |
  // size: | ipt_box_num*9 | ipt_box_num | mask_col_num |  mask_col_num    |

  // load boxes data into nram and remove padding
  float *box_buffer = nram_gen_mask;
  __memcpy_async(box_buffer, input_boxes, 9 * sizeof(float), GDRAM2NRAM,
                 9 * sizeof(float), real_width * sizeof(float),
                 input_boxes_num - 1);

  int box_buffer_num = input_boxes_num * 9;
  int mask_buffer_num = mask_col_num;
  int area_buffer_num = input_boxes_num;

  // load box area into nram
  float *area_buffer = box_buffer + box_buffer_num;
  __memcpy_async(area_buffer, boxes_area, input_boxes_num * sizeof(float),
                 GDRAM2NRAM);

  // create mask_buffer for a single row
  constexpr uint32_t allones = 0xFFFFFFFF;
  constexpr int default_mask_v = allones;
  uint32_t *mask_buffer = (uint32_t *)area_buffer + area_buffer_num;
  uint32_t *mask_buffer_swap = mask_buffer + mask_buffer_num;
  __bang_write_value(mask_buffer, mask_buffer_num * 2, default_mask_v);

  // get the rows this core should handle
  int core_box_num = 0;
  int box_i_beg = 0;
  GetCoreWorkingSet(input_boxes_num, &core_box_num, &box_i_beg);
  int box_i_end = box_i_beg + core_box_num;
  box_i_end = box_i_end < input_boxes_num ? box_i_end : input_boxes_num;

  __sync_io();

  for (int i = box_i_beg; i < box_i_end; i += 1) {
    int i_pos = 0;
    float *box_i = &box_buffer[i * 9];
    QuadClipBox<POINT_DIR> clip_box;
    clip_box.AddLines(reinterpret_cast<const Point2D *>(box_i));

    float score_i = box_i[8];
    for (int j = 0; j < input_boxes_num; ++j) {
      if (i == j) {
        continue;
      }

      float *box_j = &box_buffer[j * 9];
      float score_j = box_j[8];
      if (score_i < score_j) {
        i_pos += 1;
      } else {
        if (score_i == score_j) {
          i_pos += (j < i);
        } else {
          float iou = IOU(&clip_box, box_j, area_buffer[i], area_buffer[j]);
          if (iou > threshold) {
            MaySuppress(mask_buffer, j);
          }
        }
      }
    }
    __memcpy(mask + i * mask_col_num, mask_buffer,
             mask_col_num * sizeof(uint32_t), NRAM2GDRAM);
    __memcpy_async(sort_info + i_pos, &i, sizeof(int), NRAM2GDRAM);
    uint32_t *tmp = mask_buffer;
    mask_buffer = mask_buffer_swap;
    mask_buffer_swap = tmp;
    __bang_write_value(mask_buffer_swap, mask_buffer_num,
                       default_mask_v);  // reset to all 1
  }
}

// 3. 根据输入boxes的score顺序，从mask中选取符合阈值条件的box，并输出对应的index。
template <OutputOrder OUTPUT_ORDER>
__mlu_global__ void MLUGenNmsResult(int input_boxes_num,
                                    const uint32_t *__restrict__ p_mask,
                                    const int *__restrict__ p_sort_info,
                                    int *o_index, int *o_num) {
  // nram: | final_mask_buffer | mask_row_buffer | sort_buffer(o_index_buffer)|
  int mask_col_num = (input_boxes_num + MASK_T_BITWIDTH - 1) / MASK_T_BITWIDTH;
  int mas_col_num_align = mask_col_num;

  uint32_t *final_mask_buffer = (uint32_t *)nram_gen_result;
  __bang_write_value(final_mask_buffer, mas_col_num_align, (int)0xFFFFFFFF);

  uint32_t *mask_row_buffer = (uint32_t *)final_mask_buffer + mas_col_num_align;
  int *sort_buffer = (int *)mask_row_buffer +
                     mas_col_num_align;  // len of input_boxes_num will be used
  int *o_index_buffer = sort_buffer;     // reuse sort buffer
  __memcpy(sort_buffer, p_sort_info, sizeof(int) * input_boxes_num, GDRAM2NRAM);
  int n = 0;
  for (int i = 0; i < input_boxes_num; ++i) {
    int box_id =
        sort_buffer[i];  // i is the ith large, sort_buffer[i] is its id
    if (IsSuppressed(final_mask_buffer, box_id)) {
      continue;
    } else {
      if (OUTPUT_ORDER == OutputOrder::HIGH_SCORE_FIRST) {
        o_index_buffer[n] = box_id;
      }
      ++n;
    }
    __memcpy(mask_row_buffer, (uint32_t *)p_mask + box_id * mask_col_num,
             sizeof(uint32_t) * (mask_col_num), GDRAM2NRAM);
    __bang_band((char *)final_mask_buffer, (char *)final_mask_buffer,
                (char *)mask_row_buffer, 4 * mas_col_num_align);
  }
}

```

### 3.3 拆分(任务拆分，多核拆分)
**拆分策略**
计算过程使用Block任务，launch尽可能多core进行计算。

### 3.4 性能优化设计
1. 流水设计
  计算overlap过程复杂，暂不划分乒乓空间，不做流水。

### 3.5 方案理论性能

### 3.6 可维护性设计
1、bangc代码中加入必要的 log信息，比如输入的规模、数据类型、layout，任务类型，以及如果出错会导致程序core dump的变量，比如IO指令的data_size、dim xyz的值等，这些信息都是有利于快速定位问题；
2、对每一个函数命名变量命名都有充分的注释；
3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代。

### 3.7 测试用例设计
- 测试输入boxes的顶点坐标全是顺时针情况的case；
- 测试输入boxes的顶点坐标全是逆时针情况的case;
- 测试规模：测试不同规模下的计算结果；

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
