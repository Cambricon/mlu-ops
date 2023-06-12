# border_align_backward算子开发设计方案


* #### 文档基本信息

| 项目名称    | Training Solution                                            |
| ----------- | ------------------------------------------------------------ |
| 算子名称    | border_align_backward                                                        |
| 编制人/日期 |    郑斌/2022-4-18                                            |
| 审批人/日期 |    谷中豪/2022-4-18                                             |
| 审批人/日期 |    卜德飞/2022-4-18                                              |
| 审批人/日期 |     王远/2022-4-18                                             |
| 审批人/日期 |     周晨阳/2022-4-18                                             |

* #### 修改记录

| 版本号 | 修订人 | 修订日期 | 修订描述 |
| ------ | ------ | -------- | -------- |
| V1.0   | 郑斌   | 2022-4-18 | 首次提交 |

* #### 内容描述

本文档为`border_align_backward`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

* #### 算子需求checklist


算子需求提出者需要`check`的部分如下：

- 1.1 算子需求分析
- 1.2 算子功能和应用场景描述
- 1.3 算子输入输出参数要求
- 1.4 算子限制
- 1.5 验收标准
- 2.2 接口设计
- 3.5 测试用例（需求提出者`check`算子需求表中所给规模是否列出）

## 1 需求分析

### 1.1 算子需求分析

| 算子功能简介                                                 | border_align的反向算子，根据输入grad_output、boxes和argmax_idx求取grad_input。 |
| ------------------------------------------------------------ | ----------------------------------------- |
| 需求来源                                                     | mmcv                        |
| 应用网络                                                     |  BorderDet                   |
| 输入数据类型                                                 |  grad_output和boxes的类型为half/float，pool_size的类型为int32_t                         |
| 输入Shape                                                    |  grad_output: [N, K, 4, C], boxes: [N, K, 4], argmax_idx: [N, K, 4, C]
| 输入Layout                                                   | input: NHWC             |
| 输出数据类型                                                 | grad_input的类型为half/float                          |
| 输出Shape                                                    |  [N, H, W, 4C]
| 输出Layout                                                   | NHWC                                    |
| 模式(可选）                                                  |   无                                       |
| 是否含有dim/axis等类似语义的参数且该参数支持负数/其他特殊处理 | 否                                        |
| 是否含有labels/index等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 否                                        |
| 是否需要支持原位                                             | 否                                   |
| 是否需要支持stride机制                                       | 否                                        |
| 是否需要支持广播                                             | 否                                        |
| 0元素检查是否直接返回                                        | 是（返回MLUOP_STATUS_BAD_PARAM）                                        |
| 其他特殊需求(在线量化，融合，转数提前等，可选)               |        无                                   |
| 本次开发优先支持的规模/模式                                  |        无                                   |


### 1.2 算子功能和应用场景描述

​    `border_align_backward`是border_align的反向算子，根据输入grad_output、boxes和argmax_idx求取grad_input。

​    该算子的应用场景：`BorderDet`网络。 
例：
```python
grad_output = torch.tensor([[[[ 3.,  6.,  1.,  2.],
                              [ 4.,  7., -1.,  1.],
                              [ 3.,  7.,  1.,  2.],
                              [ 4.,  6., -1.,  1.],
                              [ 2., 12., -1., -1.],
                              [ 3., 12., -1.,  2.],
                              [ 3.,  7.,  1.,  2.],
                              [ 4.,  7., -1.,  1.],
                              [ 6., 12., -1., -2.],
                              [ 4., 12., -1.,  1.],
                              [ 4.,  9., -1.,  1.],
                              [ 4., 11., -1.,  1.]]]], device='cuda:0')

boxes_arr=torch.tensor([[[0., 0., 2., 1.],
                         [1., 0., 3., 1.],
                         [1., 0., 2., 1.],
                         [0., 0., 3., 1.],
                         [0., 0., 1., 2.],
                         [0., 0., 2., 2.],
                         [1., 0., 2., 1.],
                         [1., 0., 3., 1.],
                         [0., 1., 1., 2.],
                         [0., 0., 3., 2.],
                         [1., 0., 3., 2.],
                         [2., 0., 3., 2.]]], device='cuda:0')
argmax_idx=torch.tensor([[[[1, 0, 0, 1],
                           [1, 0, 0, 1],
                           [1, 0, 0, 1],
                           [1, 0, 0, 1],
                           [1, 1, 0, 1],
                           [1, 1, 0, 1],
                           [1, 0, 0, 1],
                           [1, 0, 0, 1],
                           [1, 1, 0, 0],
                           [1, 1, 0, 1],
                           [1, 1, 0, 1],
                           [1, 1, 0, 1]]]], device='cuda:0', dtype=torch.int32)
 grad_input = grad_output.new_zeros(1, 4, 3, 4).cuda() 
 ext_module.border_align_backward(grad_output, boxes, argmax_idx, grad_input, 1)

>>> grad_input
tensor([[[[ 0.,  4., 24., 48.],
          [ 0., 12.,  0.,  0.],
          [ 0.,  0.,  0.,  0.]],

         [[24., 56.,  0.,  0.],
          [ 0.,  0.,  0.,  0.],
          [96., 18., 22.,  0.]],

         [[ 0.,  0.,  0.,  0.],
          [ 0.,  0.,  6., -6.],
          [ 0., -4., -2., -6.]],

         [[ 0., -2., 16., 12.],
          [ 0.,  0.,  0.,  0.],
          [ 0., -4.,  0.,  0.]]]], device='cuda:0')
```
### 1.3 算子输入输出参数要求

| 参数        | 语义               | 类型（输入/输出） | 支持类型     | 物理布局 | 规模约束 |
| ----------- | ------------------ | ----------------- | ------------ | -------- | -------- |
| handle      | MLU-OPS 句柄，保存运行的上下文信息   | 输入              | mluOpHandle_t | /        | /       |
| grad_output_desc | 输入grad_output的描述信息 | 输入              |  /  | /      | grad_output的维度必须为4       |
| grad_output     | 输入数据，指向grad_output的mlu地址的指针           | 输入              | half, float | NHWC |   /    |
| boxes_desc | 输入bounding box的描述信息   | 输入              | /  | /        | boxes的维度必须为3且最后一维必须为4       |
| boxes      | 输入数据，指向boxes的mlu地址的指针          | 输入              |half, float  | ARRAY | /       |
| argmax_idx_desc | 前向算子计算出最大值对应的idx的描述信息   | 输入              | /  | /        | /       |
| argmax_idx      | 输入数据，指向argmax_idx的mlu地址的指针          | 输入              |int32_t  | NHWC | /       |
| pool_size        | 池化核尺寸                          |输入                 |int32_t       |      scalar|/  |
|grad_input_desc | 输出的描述信息 | 输入              |  /  | /      | grad_input的维度必须为4且最后一个维度为4的倍数      |
| grad_input     | 输出数据，指向grad_input的mlu地址的指针           | 输出             | half, float | NHWC | /       |
  

### 1.4 算子限制
| 限制类型   | 详细说明       |
| ---------- | -------------- |
| 原位限制   | 不支持原位     |
| stride限制 | 不支持`stride`机制 |
| 广播限制   | 不支持广播     |
|数据范围|算子不支持nan和inf|
| 数据类型 |支持`half`、`float`，且`grad_input`、`grad_output`和`boxes`须保持一致，`argmax_idx`必须为int类型|
|规模限制|grad_output的维度必须为4 ,第三个维度的dim为4，第一个维度、第二个维度 分别和boxes的第一维度、第二维度一致。boxes的维度必须为3且最后一维必须为4。argmax_idx维度必须和grad_output一致。grad_input的维度必须为4且最后。一个维度为4的倍数，第一个维度和grad_output的第一个维度保持一致。|

### 1.5 验收标准

#### 1.5.1 精度验收标准

- MLU-OPS精度验收标准：该算子为pool算子。
- 300系列的评价公式为`diff1、diff2、diff4`，验收标准采用动态阈值[10,10,1]。
- 因为fma不对齐问题，如果测例在300系列上不通过，需要将测例更改为cpu模式，此时采用静态阈值：half：1e-3，float：1e-5。

#### 1.5.2 性能验收标准

- IO效率或计算效率至少有一个不低于50%。
- 部分效率比较低的规模在4.算子性能优化记录中进行说明。
- 附上算子测试报告链接，测试报告必须包括框架给出的网络中规模的性能数据以及对应效率值。

mmcv性能测试

| 平台                 |  数据类型 | grad_ouput规模            | boxes规模            |argmax_idx规模            |计算效率(%)  | IO效率(%)    | Hardware time(us) |
| -------------------- |  -------- | --------------- | --------------- |--------------- |---------- | ---------- | ----------------- |
| Tesla V100-SXM2-16GB |  float16  | [2,256,70,4]  | [2,70,4]  |[2,256,70,4]  |37.51 | 11.51 | 55.74        |
|                      |  float32  | [2,256,70,4]  | [2,70,4]  |[2,256,70,4]  |31.77 | 13.86 |18.91        |
|                      |  float16  | [2,256,950,4] | [2,950,4] |[2,256,950,4] |47.4 | 14.3 | 1500        |
|                      |  float32  | [2,256,950,4] | [2,950,4] |[2,256,950,4] |48.04 | 16.57  | 189.22       |
|                      |  float16  | [2,128,70,4] | [2,70,4]   |[2,128,70,4]  |22.64 |7.01 | 46        |
|                      |  float32  | [2,128,70,4] | [2,70,4]   |[2,128,70,4]  |31.64|9.39 | 13.73         |

## 2 算子接口设计

### 2.1 参考接口

- MMCV

```c++
void BorderAlignBackwardCUDAKernelLauncher(const Tensor &grad_output,
                                           const Tensor &boxes,
                                           const Tensor &argmax_idx,
                                           Tensor grad_input,
                                           const int32_t pool_size);

```

### 2.2 接口设计

```c++
// 给出mlu-ops算子接口 
mluOpStatus_t mluOp_WIN_API 
mluOpBorderAlignBackward(mluOpHandle_t handle,
                        const mluOpTensorDescriptor_t grad_output_desc,
                        const void *grad_output,
                        const mluOpTensorDescriptor_t boxes_desc,
                        const void *boxes,
                        const mluOpTensorDescriptor_t argmax_idx_desc,
                        const void *argmax_idx,
                        const int32_t pool_size,
                        const mluOpTensorDescriptor_t grad_input_desc,
                        void *grad_input);
```
## 3 实现方案设计

### 3.1 实现方案
 
**计算原理说明：**
 
`grad_output`的维度为[N, K, 4, C]，`boxes`的维度为[N, K, 4]，argmax_idx的维度为[N, K, 4, C]。 在计算`grad_input`时，将每组`bounding box`，坐标为[x0, y0, x1,y1]，`height`或者`width`均分为`pool_size + 1`段（`height`为`y1 - y0`的结果，`width`为`x1 - x0`的结果），得到`height`或者`width`对应的`x_stride`和`y_stride`，根据`bouding box` 的初始值`x0`和`y0`或者`x1`和`y1`、argmax_idx对应`idx`和`stride`可以得到前向在计算最大池化结果时对应的坐标 `x`和`y`（x  = x0 + x_stride * (*argmax_idx),  y  = y0 + y_stride * (*argmax_idx) ），通过双线性插值和`x`和`y`计算得到每个像素点四邻域对应的权重和`grad_input`对应的坐标，最后通过原子操作累加给 `grad_input`。
 
**实现方案：** 

1、首先根据输入`grad_output`计算出所有需要处理的`border`数量，将`border`的数量均分给所有的核，剩余部分依次分给每个`core`，对每个core需要处理的`border`数量做遍历。

2、对每条`border`做`pool_size + 1`次遍历。由于每个`C`上的`argmax_idx`是不一样的，需要通过遍历的方法才能获取。

3、对每条`border`中`C`做遍历，计算得到`C`通道像素点的值。考虑到`C`比较大的情况，无法一次处理全部的`C`，需要对`C`做一次遍历以及对余数段的处理。

nram划分：
- ![figure1](./figure1.png)


### 3.2 伪代码实现（可选）

伪代码表示如下：

```c++
// 计算一次能够处理的数据量
deal_num = PAD_DOWN((MAX_NRAM_SIZE - NRAM_BBOX_SIZE - NFU_ALIGN_SIZE) / (4 * sizeof(T) + sizeof(int32_t)), NFU_ALIGN_NUM);
// core间任务拆分，把bounding box拆分给所有core并计算每个core待处理的bounding box的起始位置
int32_t N = boxes_desc->dims[0];
int32_t K = boxes_desc->dims[1];
const int32_t total_num = N * K * 4;
int32_t num_per_core = total_num / taskDim;
const int32_t num_rem = total_num % taskDim;
num_per_core = num_per_core + int32_t(taskId < num_rem);
int32_t start_per_core = num_rem > taskId ? (taskId * num_per_core)
          : ((num_per_core + 1) * num_rem + (taskId - num_rem) * num_per_core);
int32_t end_per_core = start_per_core + num_per_core;
// 依次处理每条border
bool empty = true;
for (int32_t num_loop = start_per_core; num_loop < end_per_core; ++num_loop) {
    int32_t n = num_loop / K / 4;
    bbox_offset = num_loop;
    __memcpy(bbox_nram, bbox_gdram + bbox_offset, 4 * sizeof(T), GDRAM2NRAM);
    box_width = *(bbox_nram + 2) - *bbox_nram;
    box_height = *(bbox_nram + 3) - *(bbox_nram + 1);

    for (int32_t c = 0; c < C / deal_num; ++c) {      
      computeImpl();     
    } 
    // compute c_tail
    c_tail = C % deal_num;
    if (c_tail != 0) {       
      deal_num = PAD_UP(c_tail, NFU_ALIGN_NUM);
      computeImpl();     
    }   
}
     
computeImpl() { 
    int32_t border = num_loop / N / K % 4;
    x = *(bbox_nram + border % 4 / 2 * 2);
    y = *(bbox_nram + 1 + border % 4 / 2 * 2);
    switch (border) {       
     case 0:{x_stride = box_width / pool_size; y_stride = 0;} break;
     case 1:{x_stride = 0; y_stride = box_height / pool_size;} break;
     case 2:{x_stride = -box_width / pool_size; y_stride = 0;} break;
     case 3:{x_stride = 0; y_stride = -box_height / pool_size;} break;
     default:{ MLULOG("Invalid Border Type."); }; break;
    }
    int32_t x_tl, x_br, y_tl, y_br;
    bool first_time_flag = true;
    bool empty = false;
    // 每条border处理pool_size + 1次
    for (int32_t pool_loop = 0; pool_loop < pool_size + 1; ++pool_loop) {
      if (empty) {       
        bilinear_interpolate_gradient(H, W, x, y, w1, w2, w3, w4, x_tl, x_br, y_tl, y_br, empty);  
        if (empty) {  // empty是双线性插值是否越界的标志位   
          pool_loop = pool_loop + 1;
          x = *(bbox_nram + border % 4 / 2 * 2) + x_stride * pool_loop;
          y = *(bbox_nram + 1 + border % 4 / 2 * 2) + y_stride * pool_loop;
          continue;
          } 
        LoadInput();   
        __bang_eq_scalar(argmax_idx_temp, argmax_idx_nram, (T)0, deal_num); // 为避免第pool_loop + 1次的结果覆盖第pool_loop次的结果，此处保存bool结果，在计算最终结果时把不需要修改的`C`通道置0。
      computeGradInput();
      pool_loop = pool_loop + 1;
      x = *(bbox_nram + border % 4 / 2 * 2) + x_stride * pool_loop;
      y = *(bbox_nram + 1 + border % 4 / 2 * 2) + y_stride * pool_loop;           
    }
  }
}
void computeGradInput()  {   
  __bang_mul_scalar((T *)grad_input, (T *)nram_ping, w1, deal_num);   
  __bang_mul((T *)grad_output_temp, (T *)(grad_output + (y_tl * W + x_tl) * C + c * deal_num), argmax_idx_temp, deal_num);
  __bang_atomic_add_reduce((T *)grad_input, (T *)(grad_output_temp), deal_num);
  __bang_mul_scalar((T *)grad_input, (T *)nram_ping, w2, deal_num);
  __bang_mul((T *)grad_output_temp, (T *)(grad_output + (y_tl * W + x_br) * C + c * deal_num), argmax_idx_temp, deal_num);
  __bang_atomic_add_reduce((T *)grad_input, (T *)(grad_output_temp), deal_num);
  __bang_mul_scalar((T *)grad_input, (T *)nram_ping, w3, deal_num);
  __bang_mul((T *)grad_output_temp, (T *)(grad_output + (y_br * W + x_tl) * C + c * deal_num), argmax_idx_temp, deal_num);
  __bang_atomic_add_reduce((T *)grad_input, (T *)(grad_output_temp),  deal_num);  
  __bang_mul_scalar((T *)grad_input, (T *)nram_ping, w4, deal_num);   
  __bang_mul((T *)grad_output_temp, (T *)(grad_output + (y_br * W + x_br) * C + c * deal_num), argmax_idx_temp, deal_num);   
  __bang_atomic_add_reduce((T *)grad_input, (T *)(grad_output_temp),  deal_num);
}
 
void loadInput() {   
  src_offset = n * H * W * C * 4 + h * C * 4 + w * C + c * deal_num;   
  __memcpy(argmax_idx_nram, argmax_idx + src_offset, GDRAM2NRAM); // 需要把int类型转换成float或者half。
  __memcpy((void *)(nram_ping), (void *)(grad_output + src_offset), deal_num * sizeof(T), GDRAM2NRAM);
}
 
```
### 3.3 拆分
### (任务拆分，多核拆分)

1、首先根据输入`grad_output`计算出所有需要处理的`border`数量，将`border`数量均分给所有的核，对于剩余的`border`依次分配给所有核。

### 3.4 性能优化设计

1、暂无。（`atomic_add`是io指令，无法使用流水来做优化。）

### 3.5 方案理论性能
### 3.6 可维护性设计

1、变量、函数和类名按照MLUOPS命名规范，尽力做到只读名称而不需读注释就能读懂代码。

2、每个函数确保只实现一个功能，尽可能缩短函数的长度。

3、合理的防呆设计。

4、关键信息打印到log中。

### 3.7 测试用例设计

- 框架在需求列表中给出的算子在网络中用到的规模：

grad_output:[2, 70, 4, 256] boxes:[2, 70, 4] argmax_idx:[2, 70, 4, 256] pool_size:10

grad_output:[2, 950, 4, 256] boxes:[2, 950, 4] argmax_idx:[2, 950, 4, 256] pool_size:10

grad_output:[2, 70, 4, 128] boxes:[2, 70, 4] argmax_idx:[2, 70, 4, 128] pool_size:10

- 随机测例：

grad_output:[3, 5, 4, 1] boxes:[3, 5, 4] argmax_idx:[3, 5, 4, 1] pool_size:8

grad_output:[10, 80, 4, 10] boxes:[3, 80, 4] argmax_idx:[10, 80, 4, 10] pool_size:2

- 反向测试：生成一些随机input、boxes和argmax_idx，校验防呆能够报error（后续补充）。

### 3.8 算子防呆检查

以下情形防呆报错并返回错误码MLUOP_STATUS_BAD_PARAM：

 1、输入和输出指针为空。

 2、输入为0元素。

 3、对数据类型做检查，grad_output和boxes数据类型不为half且不为float类型，argmax_idx类型不为int32_t。

 4、对输入boxes的shape做防呆检查，维度不为3或者最后一个维度不为4。

 5、对输入grad_output的shape做防呆检查，维度不为4。

 6、对输入argmax_idx的shape做防呆检查，维度不为4。

 7、对输出grad_input的shape做防呆检查，维度不为4或者最后一个维度不为4的倍数。

 8、对输入argmax_idx和grad_output的shape做防呆检查，维度不为4或者dim不相等。

 9、对输入grad_output和boxes的前两个维度做防呆检查，不相等返回MLUOP_STATUS_BAD_PARAM。

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

### 4.2 已经过优化的规模说明

此项仅填写未在4.1中列出的规模，否则填入4.1.

## 5 方案实施

### 5.1 开发测试计划

- 2022.4.18 ~ 2022.6.7 调研源码，设计方案：算子功能+接口设计+方案review
- 2022.6.8 ~  2022.6.8 generator代码开发
- 2022.6.9 ~ 2022.6.13 gtest代码开发
- 2022.6.14 ~ 2022.6.17 算子主体框架开发
- 2022.6.20 ~ 2022.6.24 批量测试+测试报告+提交MR+代码review
- 2022.6.27 ~ 2022.7.2 提交MR+代码review+算子入库


### 5.2 风险分析

暂无
