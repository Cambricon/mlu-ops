# mutual_information_forward算子开发设计方案

- #### 文档基本信息

| 算子名称    | mutual_information_forward |
| ---------- | -------------- |
| 编制人/日期 | 吴奇 & 唐成达/2023-4-14 |
| 审批人/日期 | 黄启元/2023-5-9           |
| 审批人/日期 | 徐文明/2023-4-17           |
| 审批人/日期 | 光昀轶/2023-4-17 |
| 审批人/日期 | 聂文昌/2023-4-17 |
| 审批人/日期 | 于宝印/2023-4-17 |

- #### 修改记录

| 版本号 | 修订人 | 修订日期   | 修订描述 |
| ------| ------| --------- | --------|
| V1.0  | 唐成达 | 2023-5-9 | 首次提交 |

- #### 内容描述

本文档为`mutual_information_forward`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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

该需求分析为框架原生算子实现功能的需求分析，对于框架原生支持但 MLU-OPS 当前版本不支持的功能，需要在`1.4算子限制` 章节中显式注明。未明确注明不支持的功能，默认 MLU-OPS 全部支持。


| 算子功能简介                  | 简要填写算子功能，详细描述在 1.2 中进行说明 |
| ------------------------------| ------------------------------------------- |
| 需求来源                      | k2                    |
| 应用网络                      | RNNT                                                         |
| 输入数据类型                  | px: float<br>py : float<br>opt_boundary(可选): int<br>p: float |
| 输入 Shape                    | px : [B, S, T+1]<br/>py : [B, S+1, T]<br/>opt_boundary : [B, 4]<br/>p: [B, S+1, T+1] |
| 输入 Layout                   | px : ARRAY<br/>py: ARRAY<br>opt_boundary : ARRAY |
| 输出数据类型                  | float                              |
| 输出 Shape                    | p : [B,S+1,T+1]<br>ans : [B]|
| 输出 Layout                   | p : ARRAY<br>ans : ARRAY|
| 模式(非输入, 算子内判断）           | 当前仅支持!modified模式，即rnnt_type=regular                |
| 是否含有 dim/axis 等类似语义的参数且该参数支持负数/其他特殊处理 | 无 |
| 是否含有 labels/index 等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | opt_boundary为标签和语音输入的边界，shape为[B, 4]其中B为batch，4的含义为[begin_symbol, begin_frame, end_symbol, end_frame]<br/>要求：（**python层已有相关防呆**）<br/>0<= begin_symbol <= end_symbol <= S<br/>0<= begin_frame <= end_frame <= T |
| 是否需要支持原位                | 是 (输入数据p需要支持原位)                    |
| 是否需要支持 stride 机制        | 否                                     |
| 是否需要支持广播                | 否    |
| 0 元素检查是否直接返回          | 是(当B为0，内部不做计算，直接返回，其余情况需要计算) |
| 其他特殊需求(在线量化，融合，转数提前等，可选)| 无 |
| 本次开发优先支持的规模/模式     | !modified模式 |

### 1.2 mutual_information_forward 算子功能和应用场景描述

1. 算子功能：mutual_information算子中的正向算子，用于计算px和py之间的互信息。mutual_information可用于rnnt网络(一种基于RNN的序列到序列方案，可以将任意长度的输入序列转换到任意长度的输出序列)。

   **输入参数说明：**

   `px`: 表示当前输出标签的概率(log形式)。在!modified模式下，shape为[B,S,T+1]；在modified模式下。shape为[B,S,T]。当前仅支持!modified模式。其中B为batch，S为输出标签symbol的长度，T为输入语音片段的长度

   `py`: 表示当前输出终止符号(termination_symbol)的概率(log形式)，shape为[B,S+1,T]

   `opt_boundary`: 输入(可选)，表示每个batch输入和输出的边界，shape为[B,4]，其中4的含义为[begin_symbol, begin_frame, end_symbol, end_frame]。当opt_boundary为空时，默认使用B*[0, 0, S, T]进行计算

   `p`: 默认输入是一个空tensor。在计算过程中会被复写，作为反向算子的输入。表示输入T个序列时，输出S个标签的概率，shape为[B,S+1,T+1]

   **输出参数说明：**

   `ans`: 表示每对序列之间的互信息，可从p中直接得到

2. !modified模式的计算公式如下：

   ```math
   \begin{array}{lcl}
      p(b,s,t) = ln(e^{p(b,s-1,t) + px(b,s-1,t)} + e^{p(b,s,t-1) + py(b,s,t-1)}) \\
      ans(b) = p(b,s\_end,t\_end)
   \end{array}
   ```
   
   * 当opt_boundary为空时，s_end=S，t_end=T
   * 当opt_boundary不为空时，s_end=opt_boundary[:, 2]，t_end=opt_boundary[:, 3]

3. nan/inf行为

   mlu实现需与参考接口对齐。参考接口代码的LogAdd函数对nan有特殊处理，如下所示：
   
   * 当输入x=nan, y=1时，返回nan
   * 当输入x=nan, y=nan时，返回nan
   * 当输入x=1, y=nan时，返回1
   
   当输入包含nan/inf时，参考接口cpu输出和cuda输出结果不一致，mlu实现与参考接口cuda输出对齐


### 1.3 算子输入输出参数要求

|参数|语义|类型(输入/输出)|支持类型|物理布局|规模说明|
|---|----|-------------|-------|------|-------|
|handle| MLU-OPS上下文指针                                            |输入|mluOpHandle_t | / |无|
|px_desc | 输入数据px的描述符号，包含px的数据类型、数据维度和布局等信息 | 输入  |         	mluOpTensorDescriptor_t   |	/     |  	无    |
| px  | 指向px数据的mlu地址指针                                      |  输入  | float |  ARRAY  | 见1.4	|
|py_desc | 输入数据py的描述符号，包含py的数据类型、数据维度和布局等信息 | 输入 |          	mluOpTensorDescriptor_t   |	/ |      	无       	|
|py| 指向py数据的mlu地址指针                                      |  输入 |  float | ARRAY   |	见1.4	|
|opt_boundary_desc | 输入数据opt_boundary的描述符号，包含opt_boundary的数据类型、数据维度和布局等信息 |      	输入  |         	mluOpTensorDescriptor_t   |	/  |     	无      |
| opt_boundary | 指向opt_boundary数据的mlu地址指针                            |   输入 | int | ARRAY   |	见1.4 |
| p_desc | 输入数据p的描述符号，包含p的数据类型、数据维度和布局等信息   | 输入 | mluOpTensorDescriptor_t | / |	见1.4 |
| p | 指向p数据的mlu地址指针 | 输入 | float | ARRAY |	无 |
|ans_desc | 输出数据ans的描述符号，包含ans的数据类型、数据维度和布局等信息 | 输入 | mluOpTensorDescriptor_t|	/     |  	无    |
|ans| 指向ans数据的mlu地址指针                                     |  输出 | 	float|  ARRAY   |	见1.4 |


### 1.4 算子限制

|限制类型     |详细说明                                                     	|
|------------|------------------------------------------------------------|
|数据类型限制	|px: float<br/>py: float<br/>opt_boundary: int<br/>p: float<br/>ans: float |
|布局限制    | 仅支持 layout 为 ARRAY                                       |
|	规模限制    | 不支持large tensor;<br/>在MLU370中，<br/>T * (S + 1) + (T + 1) * S + (T + 1) * (S + 1) + 4 * std::min(S, T) + 4 <= 163840;<br/>在MLU590中，<br/>T * (S + 1) + (T + 1) * S + (T + 1) * (S + 1) + 4 * std::min(S, T) + 4 <= 98304;	|
|功能限制     | 仅支持!modified模式，即输入参数px的shape为 [B, S, T+1]       |
| 数据范围限制 |opt_boundary的shape为[B, 4]，其中B为batch，4的含义为[begin_symbol, begin_frame, end_symbol, end_frame]<br/>要求： （**python层已有相关防呆**）<br/>0<= begin_symbol <= end_symbol <= S<br/>0<= begin_frame <= end_frame <= T|
|功能限制     |	不支持stride机制|
|功能限制     |	不支持广播|

### 1.5 验收标准

#### 1.5.1 精度验收标准
本算子包含动态规划和指数运算，属于复合类算子，验收标准采用动态阈值为diff1, diff2

#### 1.5.2 性能验收标准

本次优先交付功能，后续视情况再优化性能；

## 2 算子接口设计

### 2.1 参考接口

* K2 (release v1.23.4)

```C++
torch::Tensor MutualInformationCuda(torch::Tensor px, torch::Tensor py,
                                    torch::optional<torch::Tensor> opt_boundary,
                                    torch::Tensor p)
```

### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API
mluOpGetMutualInformationForwardWorkspaceSize(mluOpHandle_t handle,
                                              const mluOpTensorDescriptor_t px_desc,
                                              const mluOpTensorDescriptor_t py_desc,
                                              const mluOpTensorDescriptor_t opt_boundary_desc,
                                              const mluOpTensorDescriptor_t p_desc,
                                              const mluOpTensorDescriptor_t ans_desc,
                                              size_t *workspace_size)

mluOpStatus_t MLUOP_WIN_API
mluOpMutualInformationForward(mluOpHandle_t handle,
                              const mluOpTensorDescriptor_t px_desc,
                              const void *px,
                              const mluOpTensorDescriptor_t py_desc,
                              const void *py,
                              const mluOpTensorDescriptor_t opt_boundary_desc,
                              const void *opt_boundary,
                              const mluOpTensorDescriptor_t p_desc,
                              void *p,
                              void *workspace,
                              const size_t workspace_size,
                              const mluOpTensorDescriptor_t ans_desc, 
                              void *ans)
```

备注：mluOpGetMutualInformationForwardWorkspaceSize接口：workspace_size固定返回0。预留此接口，当解除规模限制后，算子需要用到workspace

## 3 实现方案设计

当前方案存在规模限制，即nram空间一次可处理一个batch的数据

### 3.1 实现方案

由于p计算为动态规划算法，并且当opt_boundary不为空时，每个batch的boundary不一样。因此下在core间拆分batch，core内对一个batch的数据循环计算。

* step1：将当前batch的px和py全都加载到NRAM上

* step2：计算p

  p的计算公式：

  ```math
  p(b,s,t) = ln(e^{p(b,s-1,t) + px(b,s-1,t)} + e^{p(b,s,t-1) + py(b,s,t-1)})
  ```

  从p计算公式可分析出，p(b,s,t) 依赖 p(b,s-1,t) 和p(b,s,t-1) ，因此p计算可在对角线方向进行并行计算。

  以B=1, T=2，S=2举例，在batch=0时，p的计算过程如下：

  - 第一步: 计算p(0, 0, 0)
  - 第二步: 计算p(0, 0, 1)，p(0, 1, 0)
  - 第三步: 计算p(0, 0, 2)，p(0, 1, 1)，p(0, 2, 0)
  - 第四步: 计算p(0, 1, 2)，p(0, 2, 1)
  - 第五步: 计算p(0, 2, 2)

  如下表所示，上述每一步计算的位置都是在对角线方向

  | S\T  | t0         | t1         | t2         |
  | ---- | ---------- | ---------- | ---------- |
  | s0   | p(0, 0, 0) | p(0, 0, 1) | p(0, 0, 2) |
  | s1   | p(0, 1, 0) | p(0, 1, 1) | p(0, 1, 2) |
  | s2   | p(0, 2, 0) | p(0, 2, 1) | p(0, 2, 2) |

  在nram从对角线方向读取px和py，逐行计算p

* step3: 将p的结果存回gdram，并将p(b,s_end,t_end)存回到ans中



### 3.2 伪代码实现

以下伪代码，以opt_boundary=nullptr举例

```c++
#define MIN_LOG_DIFF_FLOAT -15.9423847198486328125f 
__mlu_func__ void logAddVector(float *dst, float *src1, float *src2, float *max_value, float *mask, float *temp, int data_num) {
    __bang_nan_minimum(dst, src1, src2, data_num);
    __bang_maximum(max_value, src1, src2, data_num);

    // if src1 is nan, then max_value = src1
    int nan_mask_i = 0x7fffffff;
    float nan_mask_f = *(float *)&nan_mask_i;
    __bang_band_scalar(mask, src1, nan_mask_f, data_num);
    __bang_ge_scalar((int *)mask, (int *)mask, MIN_POS_NAN, data_num);
    __bang_mul_scalar((int *)mask, (int *)mask, -1, data_num);
    __bang_band((char *)temp, (char *)src1, (char *)mask,
                data_num * sizeof(float));
    __bang_add(max_value, max_value, temp, data_num);

    // compute log sum exp
    __bang_sub(dst, dst, max_value, data_num);
    __bang_ge_scalar(mask, dst, MIN_LOG_DIFF_FLOAT, data_num);
    __mluop_exp(dst, dst, nullptr, 0, data_num);
    __bang_add_scalar(dst, dst, 1.f, data_num);
    computeLog(dst, dst, data_num);
    __bang_add(dst, dst, max_value, data_num);

    // if min_value - max_value < MIN_POS_NAN, return the larger one
    __bang_float2int32_rn((int *)mask, mask, data_num, 0);
    __bang_mul_scalar((int *)mask, (int *)mask, -1, data_num);
    __bang_band((char *)dst, (char *)dst, (char *)mask, data_num * sizeof(float));
    __bang_add_scalar((int *)mask, (int *)mask, 1, data_num);
    __bang_mul_scalar((int *)mask, (int *)mask, -1, data_num);
    __bang_band((char *)max_value, (char *)max_value, (char *)mask,
                data_num * sizeof(float));
    __bang_add(dst, dst, max_value, data_num);
}


__mlu_func__ void computeMutualInformation() {
    /* ***************************************nram space split**************************************** */
    /* |    px   |   py    |      p      |   cur_px   |   cur_py   |   cur_p    | max_value/mask/temp |*/
    /* | S*(T+1) | (S+1)*T | (S+1)*(T+1) | min(S,T)+1 | min(S,T)+1 | min(S,T)+1 |    3*min(S,T)+3     |*/
    float *px = (float *)buffer;
    float *py = px + S*(T+1);
    float *p = py + (S+1)*T;
    float *cur_px = p + (S+1)*(T+1);
    float *cur_py = cur_px + min(S,T)+1;
    float *cur_p = cur_py + min(S,T)+1;
    float *max_value = cur_p + min(S,T)+1;
    float *mask = max_value + min(S,T)+1;
    float *temp = mask + min(S,T)+1;
    for (int i = 1; i < S + T + 1; ++i) {
        int data_num = 1;
        if (i < T) {
            data_num = std::min(i + 1, S);
        } else {
            data_num = T + S - i;
        }
        __memcpy(); // load cur_px
        __memcpy(); // load cur_py
        __memcpy(); // load cur_p
        __bang_add(cur_px, cur_px, cur_p, data_num);
        __bang_add(cur_py, cur_py, cur_p, data_num);
        logAddVector(cur_p, cur_px, cur_py, max_value, mask, temp, data_num);
        __memcpy(); // store cur_p
    }
}

__mlu_global__ void MLUKernelMutualInformationForward(void *px,
                                                      void *py,
                                                      void *opt_boundary,
                                                      void *p,
                                                      void *ans) {
    int num_per_core = batches / taskDim;
    int num_rem = batches % taskDim;
    int num_cur_core = num_per_core + (taskId < num_rem);
    int b_offset = taskId * num_cur_core + (taskId >= num_rem) * num_rem;

    for (int i = b_offset; i < b_offset + num_cur_core; ++i) {
        computeMutualInformation();
    }
}
```

### 3.3 拆分（任务拆分，多核拆分）

基本任务类型为Block任务，core间按照对batch维度进行拆分，core内在S-T矩阵的对角线方向进行循环计算

### 3.4 性能优化设计

1、资源分配

NRAM空间划分参考3.2 伪代码实现中的描述

2、流水设计

bang_maximum等指令在370上无法异步执行，因此暂不排流水。

### 3.5 可维护性设计

1、bangc 代码中加入必要的 log 信息，比如输入的规模、数据类型、layout 这些，以及如果出错会导致程序 core dump 的变量，比如 IO 指令的 data_size、dim xyz 的值等，这些信息都是有利于快速定位问题；

2、对每一个函数命名变量命名都有充分的注释；

3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代，或添加注释。

### 3.6 测试用例设计

- 算子在网络中用到的规模：

  未提供


- 边界 case：
  - T=0; S=0; T= 1; S= 1
  - opt_boundary中存在begin_symbol = end_symbol
  - opt_boundary中存在begin_frame = end_frame


其他可根据需要进行补充。算子开发完毕后，补充测试报告链接。

### 3.7 算子防呆检查

1、检查handle/px_desc/py_desc/p_desc/ans_desc是否为空

2、检查输入输出空间指针px/py/p/ans是否为空

3、检查opt_boundary_desc和opt_boundary是否同时为空，或者同时不为空

3、检查0元素，如果batch_size为=0，直接返回MLUOP_STATUS_SUCCESS

3、涉及 workspace 算子对于 workspace_size 的检查防呆；

4、检查px的维度为[B, S,T+1]；py的维度为[B, S+1,T]；p的维度为[B, S+1,T+1]；ans的维度为[B]；如果opt_boundary不为空，opt_boundary为维度为[B，4]

5、检查px/py/p/ans的数据类型为float；如果opt_boundary不为空，opt_boundary的数据类型为int

6、检查输入输出是否包含large tensor，或者输入规模是否超出限制


## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

暂无

### 4.2 已经过优化的规模说明

暂无

## 5 方案实施

### 5.1 开发测试计划

- 2023.03.30 算法原理调研与参考接口分析
- 2023.04.10 设计方案：算子功能+接口设计
- 2023.05.09 generator开发
- 2023.05.10 host端开发
- 2023.05.11 正向算子gtest、device端开发
- 2023.05.18 功能调试与优化
- 2023.05.22 批量测试与撰写测试报告
- 2023.05.24 提交PR与代码review
- 2023.05.26 算子入库


### 5.2 风险分析
* LogAdd函数中nan/inf状态对齐会影响算子性能