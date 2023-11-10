# mutual_information_backward算子开发设计方案

- #### 文档基本信息

| 算子名称    | mutual_information_backward |
| ----------- | --------------------------- |
| 编制人/日期 | 唐成达/2023-4-14            |

- #### 修改记录

| 版本号 | 修订人 | 修订日期  | 修订描述 |
| ------ | ------ | --------- | -------- |
| V1.0   | 唐成达 | 2023-4-14 | 首次提交 |
| V1.1  | 宋琎 | 2023-10-26 | 取消因为nram大小导致的对于S, T的规模限制 |

- #### 内容描述

本文档为`mutual_information_backward`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录。

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

| 算子功能简介                                                 | 求输入px和py的梯度                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 需求来源                                                     | K2                                                           |
| 应用网络                                                     | rnnt                                                         |
| 输入数据类型                                                 | px: float<br/>py: float<br/>opt_boundary(可选): int64<br/>p: float<br/>ans_grad: float |
| 输入标量参数                                                 | overwrite_ans_grad: bool                                     |
| 输入 Shape                                                   | px: [B, S, T+1]<br/>py: [B, S+1, T]<br/>opt_boundary: [B, 4] or None<br/>p: [B, S+1, T+1]<br/>ans_grad: [B] |
| 输入 Layout                                                  | ARRAY                                                        |
| 输出数据类型                                                 | float                                                        |
| 输出 Shape                                                   | px_grad: [B, S, T+1]<br/>py_grad: [B, S+1, T]                |
| 输出 Layout                                                  | ARRAY                                                        |
| 模式(可选）                                                  | 当前仅支持!modified模式，即rnnt_type=regular                 |
| 是否含有 dim/axis 等类似语义的参数且该参数支持负数/其他特殊处理 | 无                                                           |
| 是否含有 labels/index 等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | opt_boundary为标签和语音输入的边界，shape为[B, 4]其中B为batch，4的含义为[begin_symbol, begin_frame, end_symbol, end_frame]<br/>要求：（**python层已有相关防呆**）<br/>0<= begin_symbol <= end_symbol <= S<br/>0<= begin_frame <= end_frame <= T |
| 是否需要支持原位                                             | 是，当overwrite_ans_grad为true时，ans_grad需要原位操作       |
| 是否需要支持 stride 机制                                     | 否                                                           |
| 是否需要支持广播                                             | 否                                                           |
| 0 元素检查是否直接返回                                       | 是(当B为0 或 S和T都为0时，内部不做计算，直接返回，其余情况需要计算)                        |
| 其他特殊需求(在线量化，融合，转数提前等，可选)               | 无                                                           |
| 本次开发优先支持的规模/模式                                  | !modified模式                                                |

### 1.2 算子功能和应用场景描述

1. 算子功能：mutual_information算子中的反向算子，用于计算px和py的梯度。mutual_information可用于rnnt网络(一种基于RNN的序列到序列方案，可以将任意长度的输入序列转换到任意长度的输出序列)

   **输入参数说明：**

   `px`: 表示当前输出标签的概率(log形式)。在!modified模式下，shape为[B,S,T+1]；在modified模式下。shape为[B,S,T]。当前仅支持!modified模式。其中B为batch，S为输出标签symbol的长度，T为输入语音片段的长度

   `py`: 输入，表示当前输出终止符号(termination_symbol)的概率(log形式)，shape为[B,S+1,T]

   `opt_boundary`: 输入(可选)，表示每个batch输入和输出的边界，shape为[B,4]，其中4的含义为[begin_symbol, begin_frame, end_symbol, end_frame]。当opt_boundary为空时，默认使用B*[0, 0, S, T]进行计算

   `p`: 输入，为前向算子的输出，表示输入T个序列时，输出S个标签的概率，shape为[B,S+1,T+1]

   `ans_grad`: 输入，前向输出`ans`对应的梯度，shape为[B]，为全1矩阵

   `overwrite_ans_grad`: 输入，表示输入`ans_grad`是否需要被复写

   **输出参数说明：**

   `px_grad`: 表示px对应的梯度，shape和px相同

   `py_grad`: 表示py对应的梯度，shape和py相同



2. !modified模式的计算公式如下：

   ```math
   \begin{array}{lcl}
   term1(b,s,t) = e^{p(b,s,t) + px(b,s,t) - p(b,s+1,t)} \\
   term2(b,s,t) = e^{p(b,s,t) + py(b,s,t) - p(b,s,t+1)} \\
   p\_grad(b,s,t) = p\_grad(b,s+1,t) * term1(b,s,t) + p\_grad(b,s,t+1) * term2(b,s,t) \\
   px\_grad(b,s,t) = p\_grad(b,s+1,t) * term1(b,s,t) \\
   py\_grad(b,s,t) = p\_grad(b,s,t+1) * term2(b,s,t)
   \end{array}
   ```



3. nan/inf行为

   通过分析，参考接口代码实现中有对中间结果为nan/inf时进行特殊处理。当p(b,s,t)、term1(b,s,t)和term2(b,s,t)为nan或inf时，需将其置为0，即safeExp函数中包括的功能。（当输入包含nan/inf时，参考接口cpu和cuda输出结果不一致，mlu实现与参考接口cuda对齐）



### 1.3 算子输入输出参数要求

| 参数               | 语义                                                         | 类型（输入/输出） | 支持类型                | 物理布局 | 规模限制 |
| ------------------ | ------------------------------------------------------------ | ----------------- | ----------------------- | -------- | -------- |
| handle             | MLU-OPS上下文指针                                            | 输入              | mluOpHandle_t           | /        | 无       |
| px_desc            | 输入数据px的描述符号，包含px的数据类型、数据维度和布局等信息 | 输入              | mluOpTensorDescriptor_t | /        | 见1.4    |
| px                 | 指向px数据的mlu地址指针                                      | 输入              | float                   | ARRAY    | 无       |
| py_desc            | 输入数据py的描述符号，包含py的数据类型、数据维度和布局等信息 | 输入              | mluOpTensorDescriptor_t | /        | 见1.4    |
| py                 | 指向py数据的mlu地址指针                                      | 输入              | float                   | ARRAY    | 无       |
| opt_boundary_desc  | 输入数据opt_boundary的描述符号，包含opt_boundary的数据类型、数据维度和布局等信息 | 输入              | mluOpTensorDescriptor_t | /        | 见1.4    |
| opt_boundary       | 指向opt_boundary数据的mlu地址指针                            | 输入              | int64                   | ARRAY    | 无       |
| p_desc             | 输入数据p的描述符号，包含p的数据类型、数据维度和布局等信息   | 输入              | mluOpTensorDescriptor_t | /        | 见1.4    |
| p                  | 指向p数据的mlu地址指针                                       | 输入              | float                   | ARRAY    | 无       |
| ans_grad_desc      | 输入数据ans_grad的描述符号，包含ans_grad的数据类型、数据维度和布局等信息 | 输入              | mluOpTensorDescriptor_t | /        | 见1.4    |
| ans_grad           | 指向ans_grad数据的mlu地址指针                                | 输入              | float                   | ARRAY    | 无       |
| overwrite_ans_grad | 标识ans_grad是否需要复写                                     | 输入              | bool                    | /        | 无       |
| px_grad_desc       | 输出数据px_grad的描述符号，包含px_grad的数据类型、数据维度和布局等信息 | 输入              | mluOpTensorDescriptor_t | /        | 见1.4    |
| px_grad            | 指向px_grad数据的mlu地址指针                                 | 输出              | float                   | ARRAY    | 无       |
| py_grad_desc       | 输出数据py_grad的描述符号，包含py_grad的数据类型、数据维度和布局等信息 | 输入              | mluOpTensorDescriptor_t | /        | 见1.4    |
| py_grad            | 指向py_grad数据的mlu地址指针                                 | 输出              | float                   | ARRAY    | 无       |

### 1.4 算子限制

| 限制类型     | 详细说明                                                     |
| ------------ | ------------------------------------------------------------ |
| 数据类型限制 | px: float<br/>py: float<br/>opt_boundary: int64<br/>p: float<br/>ans_grad: float<br/>overwrite_ans_grad: bool<br/>px_grad: float<br/>py_grad: float |
| 布局限制     | ARRAY                                                        |
| 规模限制     | 不支持large tensor; |
| 功能限制     | 仅支持!modified模式，即输入参数px的shape为 [B, S, T+1]       |
| 数据范围限制 | opt_boundary的shape为[B, 4]其中B为batch，4的含义为[begin_symbol, begin_frame, end_symbol, end_frame]<br/>要求：（**python层已有相关防呆**）<br/>0<= begin_symbol <= end_symbol <= S<br/>0<= begin_frame <= end_frame <= T |
| 原位限制     | 仅当overwrite_ans_grad为true时，ans_grad支持原位操作         |
| stride 限制  | 不支持 stride 机制                                           |
| 广播限制     | 不支持广播                                                   |



### 1.5 验收标准

#### 1.5.1 精度验收标准

本算子包含动态规划和指数运算，属于复合类算子，验收标准采用动态阈值为diff1, diff2



#### 1.5.2 性能验收标准

本次优先交付功能，后续视情况再优化性能



## 2 算子接口设计

### 2.1 参考接口

- K2

```c++
std::vector<torch::Tensor> MutualInformationBackwardCuda(
    torch::Tensor px, torch::Tensor py,
    torch::optional<torch::Tensor> opt_boundary, torch::Tensor p,
    torch::Tensor ans_grad, bool overwrite_ans_grad)
```

### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API
mluOpGetMutualInformationBackwardWorkspaceSize(mluOpHandle_t handle,
                                               const mluOpTensorDescriptor_t px_desc,
                                               const mluOpTensorDescriptor_t py_desc,
                                               const mluOpTensorDescriptor_t opt_boundary_desc,
                                               const mluOpTensorDescriptor_t p_desc,
                                               const mluOpTensorDescriptor_t ans_grad_desc,
                                               const bool overwrite_ans_grad,
                                               size_t *workspace_size)

mluOpStatus_t MLUOP_WIN_API
mluOpMutualInformationBackward(mluOpHandle_t handle,
                               const mluOpTensorDescriptor_t px_desc,
                               const void *px,
                               const mluOpTensorDescriptor_t py_desc,
                               const void *py,
                               const mluOpTensorDescriptor_t opt_boundary_desc,
                               const void *opt_boundary,
                               const mluOpTensorDescriptor_t p_desc,
                               const void *p,
                               const mluOpTensorDescriptor_t ans_grad_desc,
                               void *ans_grad,
                               const bool overwrite_ans_grad,
                               void *workspace,
                               const size_t workspace_size,
                               const mluOpTensorDescriptor_t px_grad_desc,
                               void *px_grad,
                               const mluOpTensorDescriptor_t py_grad_desc,
                               void *py_grad)
```

## 3 实现方案设计

原方案存在规模限制，即nram空间一次可处理一个batch的数据。本次1.1版本提交后，将原方案kernel重命名为3PipelineKernel，新的取消S T规模限制的kernel作为DefaultKernel.

### 3.1 实现方案

#### 3Pipeline Kernel方案

由于p_grad计算为动态规划算法，并且当opt_boundary不为空时，每个batch的boundary不一样。因此下面3Pipeline方案是在core间拆分batch，core内对一个batch的数据循环计算。

* step1:  计算term1和term2

  将p,px,py加载到nram上，逐行计算term1和term2，结果存储在nram(term1和term2分别复用px和py的空间)

  ```math
  term1(b,s,t) = e^{p(b,s,t) + px(b,s,t) - p(b,s+1,t)}
  term2(b,s,t) = e^{p(b,s,t) + py(b,s,t) - p(b,s,t+1)}
  ```

* step2：计算p_grad

  p_grad计算公式：

  ```math
  p_{grad}(b,s,t) = p_{grad}(b,s+1,t) * term1(b,s,t) + p_{grad}(b,s,t+1) * term2(b,s,t)
  ```

  从p_grad计算公式可分析出，p_grad(b,s,t) 依赖 p_grad(b,s+1,t) 和p_grad(b,s,t+1) ，因此p_grad计算可在对角线方向进行并行计算。

  以B=1, T=2，S=2举例，在batch=0时，p_grad的计算过程如下：

  * 第一步: 计算p_grad(0, 2, 2)
  * 第二步: 计算p_grad(0, 1, 2)，p_grad(0, 2, 1)
  * 第三步: 计算p_grad(0, 0, 2)，p_grad(0, 1, 1)，p_grad(0, 2, 0)
  * 第四步: 计算p_grad(0, 0, 1)，p_grad(0, 1, 0)
  * 第五步: 计算p_grad(0, 0, 0)

  如下表所示，上述每一步计算的位置都是在对角线方向

  | S\T  | t0              | t1              | t2              |
  | ---- | --------------- | --------------- | --------------- |
  | s0   | p_grad(0, 0, 0) | p_grad(0, 0, 1) | p_grad(0, 0, 2) |
  | s1   | p_grad(0, 1, 0) | p_grad(0, 1, 1) | p_grad(0, 1, 2) |
  | s2   | p_grad(0, 2, 0) | p_grad(0, 2, 1) | p_grad(0, 2, 2) |


  在nram从对角线方向读取term1和term2，逐行计算p_grad，结果也存储到NRAM上

* step3：计算px_grad和py_grad

  在nram从读取term1、term2和p_grad，逐行计算px_grad和py_grad，结果存回GDRAM

#### Default Kernel方案

当一次性无法将一个batch的px和py全部加载到片上，无法在片上完成全部的数据运算的时候，需要在S和T维度进行切分（核间同时也拆batch维度），多次load到片上进行计算，再将中间结果`p_grad`存回片外，由有数据依赖的block完成后续的计算。

由于数据依赖是基于对角线方向的，所以可以将划分的任务Block在对角线方向进行并行化Launch，一次性Launch的job之间没有数据依赖，可以并行。

所以在一次job step中，除了在Batch上划分的不同block，还可能会Launch多个不同S T位置的Block。

由于每个batch的boundary有可能不同，所以每个单独的job Kernel只处理1个batch的Block计算。

沿着对角线分步骤（job step）进行Launch Kernel，每一步Launch的Kernel数量也不同，第一步Launch Batch个job计算第(0,0)的Block，第二步Launch 2×Batch个job分别计算第(0,1)和(1,0)的Block：
```
 |-----------------|
 | 0 | 1 | 2 | ... |
 | 1 | 2 | 3 | ... |
 | 2 | 3 | 4 | ... |
 | 3 | 4 | ...     |
 | ...             |
 |-----------------|
```

在每一个block中，先初始化Load`px`,`py`,`p`,`p_grad`的数据。注意`px``py`两者的shape不同，导致在GDRAM上的src_stride也不同。
当超出边界时，对`px`,`py`的数据块写-inf,因为后续需要对其进行exp操作，使得计算之后的结果是0，与公式相符；
而对`p`的数据块需要补的是-1.0e+30的大负数，根据竞品GPU代码，对p中 < -1.0e+30 的数，都赋值为了该值，避免 -inf -(-inf) 导致出现nan的问题。
对`p_grad`则需要补0,因为计算公式中后续参与计算的为乘法，而term1,term2的结果不会出现inf和nan的值，因为已经被刷0了，所以补0.

在计算时，也按照对角线方向并行化计算：

* step1:
先计算`term1`和`term2`：

```
term1(s, t) = exp(p(s, t) + px(s, t) - p(s + 1, t));

term2(s, t) = exp(p(s, t) + py(s, t) - p(s, t + 1));
```

先 memcpy 对应坐标的 px 和 py，到cur_px和cur_py上；

将本次计算对应坐标的p(s,t)，memcpy到cur_p上，以及在term1和term2计算中，所需依赖的p(s+1,t), p(s,t+1)memcpy到next_p上；
在空间分配上，每次并行化计算中的`cur_p` 和 `next_p` 使用了2块空间，且next_p占用多1个float空间。

为了避免出现 -inf - (-inf) = nan 的问题，将 `cur_p` 和 `next_p` 中 < -1e+30 的数，设置为 -1e+30; 通过 `__bang_nan_maximum(p, large_neg_value)` 指令完成；

再使用 SafeExp 函数计算，在前序步骤中分别 add、sub 后的数值（需要额外分配一块 mask 空间）；

原方案中，(px_buf)term1 复用 px，(py_buf)term2 复用 py，而从反向算子的整体角度来看，其实并不需要term1和term2数据，只是中间计算过程中需要，所以仅仅在每次对角线计算的过程中，暂存在cur_px, cur_py，而无需memcpy拷回；

至此，cur_term1, cur_term2的结果，暂存在cur_px, cur_py的NRAM空间上，无需拷回；

* step2:
在当前支持的!modified模式下，发现计算`p_grad`的加法的两个操作数就是在分别计算`px_grad`和`py_grad`的公式；
所以先计算`px_grad`和`py_grad`，并将结果写回。
空间分配上，(px_buf)px_grad 复用 cur_term1，(py_buf)py_grad 复用 cur_term2；而计算所需的`next_p_grad`则复用next_p的空间。

```
px_grad(s, t) = p_grad(s + 1, t) * term1(s, t);

py_grad(s, t) = p_grad(s, t + 1) * term2(s, t);
```

该步骤的计算结果，通过 memcpy NRAM2NRAM 存储到原`px`和`py`的位置上，因为后续计算不再使用，可以复用。
并在对角线计算结束后，整体Store至算子输出`px_grad`和`py_grad`；

* step3:

最后再继续计算p_grad，(p_buf)cur_p_grad复用cur_p的空间。

```
p_grad(s, t) = p_grad(s + 1, t) * term1(s, t) + p_grad(s, t + 1) * term2(s, t);
```

因为前序步骤中，已经计算了`px_grad`和`py_grad`，本计算步骤中，只做加法即可。
其中，当本次计算的s_end和t_end均为该batch的s_end和t_end时，对`p_grad`的初始化 (p_buf) `p_grad[s_end][t_end] = ans_grad[b]`

不同Block之间的数据依赖，可以通过device端计算当前job的S和T的begin和end，与边界进行对比，超出边界的部分，由于前面已经通过将px和py的数值设置为-INF，公式中的`p(s+1,t)`和`p(s,t+1)`超出范围的部分，填充为-1e+30，所以前面步骤的中间结果是0，在计算`p_grad`时，可以直接使用；否则则从其他已经计算过的Block的结果处Load所需的`p_grad`数据。

最终，该步骤的计算结果，通过 memcpy 存储到output 的 `p_grad` 也就是Workspace申请的空间上，以便在不同的计算block中，有数据依赖的地方进行load；

`overwrite_ans_grad`：如果设置为true，则在反向计算`p_grad` 之后，同时也更新`ans_grad`，复写该输入 `ans_grad[b] = p_grad[s_begin][t_begin];`


对于如何进行S和T的划分，本算子的优化目标是：所有job的计算对角线数量总和越少，在算子内的计算并行度越高。根据划分后所有Block的计算对角线之和，选择`S_block_size`和`T_block_size`。


### 3.2 伪代码实现

以下伪代码，以opt_boundary=nullptr举例

```c++
__mlu_func__ void computeTerm1AndTerm2() {
    /* *********************nram space split********************** */
    /* |  term1  |  term2  | cur_p | next_p | large_neg | mask |*/
    /* | S*(T+1) | (S+1)*T |  T+1  |  T+1   |  2*(T+1)  | T+1  |*/
    float *term1 = (float *)buffer;
    float *term2 = term1 + S*(T+1);
    float *cur_p = term2 + (S+1)*T;
    float *next_p = cur_p + T + 1;
    float *large_neg = next_p + T + 1;
    float *mask = large_neg + 2*(T+1);
    __bang_write_value(large_neg, 2*(T+1), -1.0e+30);

    for (int i = 0; i < S + 1; ++i) {
        if (i == S) {
            // compute term2[S][:]
            __memcpy(term2 + S * T, py + S * T, T * sizeof(float), GDRAM2NRAM);
            __memcpy(cur_p, p + S * (T+1), (T+1) * sizeof(float), GDRAM2NRAM);
            __bang_add(term2 + S * T, term2 + S * T, cur_p, T);
            __bang_sub(term2 + S * T, term2 + S * T, cur_p + 1, T);
            safeExp(term2 + S * T, term2 + S * T, mask, T);
            break;
        }
        // load px->term1, py->term2, p->cur_p/next_p
        __memcpy(term1 + i * (T+1), px + i * (T+1), (T+1) * sizeof(float), GDRAM2NRAM);
        __memcpy(term2 + i * T, py + i * T, T * sizeof(float), GDRAM2NRAM);
        __memcpy(cur_p, p + i * (T+1), 2 * (T+1) * sizeof(float), GDRAM2NRAM);

        __bang_nan_maximum(cur_p, cur_p, large_neg, 2*(T+1));

        // term1(b,s,t) = exp(p(b,s,t) + px(b,s,t) - p(b,s+1,t))
        __bang_add(term1 + i * (T+1), term1 + i * (T+1), cur_p, T + 1);
        __bang_sub(term1 + i * (T+1), term1 + i * (T+1), next_p, T + 1);
        safeExp(term1 + i * (T+1), term1 + i * (T+1), mask, T + 1);

        // term2(b,s,t) = exp(p(b,s,t) + py(b,s,t) - p(b,s,t+1))
        __bang_add(term2 + i * T, term2 + i * T, cur_p, T);
        __bang_sub(term2 + i * T, term2 + i * T, cur_p + 1, T);
        safeExp(term2 + i * T, term2 + i * T, mask, T);
    }
}

__mlu_func__ void safeExp(float *dst, float *src, float *mask, const int num) {
    setNanInfToZero(src, mask, num);
    __mluop_exp(dst, src, NULL, 0, num);
      __bang_band((char *)dst, (char *)dst, (char *)mask, num * sizeof(float));
    setNanInfToZero(dst, mask, num);
}

__mlu_func__ void setNanInfToZero(float *src, float *mask, const int num) {
    __bang_sub(mask, src, src, num);
    __bang_eq_scalar(mask, mask, 0., num);
    __bang_float2int32_rn((int *)mask, mask, num, 0);
    __bang_mul_scalar((int *)mask, (int *)mask, int(-1), num);
    __bang_band((char *)src, (char *)src, (char *)mask, num * sizeof(float));
}

__mlu_func__ void computePGrad() {
    /* ******************************nram space split****************************** */
    /* |  term1  |  term2  |     p_grad    | cur_term1  | cur_term2  | cur_p_grad | */
    /* | S*(T+1) | (S+1)*T |  (S+1)*(T+1)  | min(S,T)+1 | min(S,T)+1 |  min(S,T)  | */
    float *term1 = (float *)buffer;
    float *term2 = term1 + S*(T+1);
    float *p_grad = term2 + (S+1)*T;
    float *cur_term1 = p_grad + (S+1)*(T+1);
    float *cur_term2 = cur_term1 + min(S,T)+1;
    float *cur_p_grad = cur_term2 + min(S,T)+1;
    __bang_write_value(cur_p_grad, min(S,T)+2, 0.0);
    for (int i = 1; i < S + T + 1; ++i) {
        int data_num = 1;
        if (i < T) {
            data_num = std::min(i + 1, S);
        } else {
            data_num = T + S - i;
        }
        // p_grad(b,s,t) = p_grad(b,s+1,t) * term1(b,s,t) + p_grad(b,s,t+1) * term2(b,s,t)
        __memcpy(); // load cur_term1
        __memcpy(); // load cur_term2
        __memcpy(); // load cur_p_grad
        __bang_mul(cur_term1, cur_term1, cur_p_grad, data_num);
        __bang_mul(cur_term2, cur_term2, cur_p_grad, data_num);
        __bang_add(cur_p_grad, cur_term1, cur_term2, data_num);
        __memcpy(); // store cur_p_grad
    }
}

__mlu_func__ void computePxGradAndPyGrad() {
    /* ***********nram space split********** */
    /* |  term1  |  term2  |     p_grad    | */
    /* | S*(T+1) | (S+1)*T |  (S+1)*(T+1)  | */
    float *term1 = (float *)buffer;
    float *term2 = term1 + S*(T+1);
    float *p_grad = term2 + (S+1)*T;
    for (int i = 0; i < S + 1; ++i) {
        if (i == S) {
            // compute py_grad[S][:]
            __bang_mul(term2 + S * T, term2 + S * T, p_grad + S * T + 1, T);
            break;
        }
        // compute py_grad：px_grad(b,s,t) = p_grad(b,s+1,t) * term1(b,s,t)
        __bang_mul(term1 + i * (T+1), term1 + S * (T+1), p_grad + S * (T+1) , T + 1);
        // compute py_grad：py_grad(b,s,t) = p_grad(b,s,t+1) * term2(b,s,t)
        __bang_mul(term2 + i * T, term2 + i * T, p_grad + i * T + 1, T);
    }
    __memcpy(px_grad + i * (T+1), term1 + S * (T+1), (T+1) * sizeof(float), NRAM2GDRAM);
    __memcpy(py_grad + i * T, term2 + i * T, T * sizeof(float), NRAM2GDRAM);
}

__mlu_global__ void MLUKernelMutualInformationBackward(void *px,
                                                       void *py,
                                                       void *opt_boundary,
                                                       void *p,
                                                       void *ans_grad,
                                                       bool overwrite_ans_grad,
                                                       void *workspace,
                                                       void *px_grad,
                                                       void *py_grad) {
    int num_per_core = batches / taskDim;
    int num_rem = batches % taskDim;
    int num_cur_core = num_per_core + (taskId < num_rem);
    int b_offset = taskId * num_cur_core + (taskId >= num_rem) * num_rem;

    for (int i = b_offset; i < b_offset + num_cur_core; ++i) {
        computeTerm1AndTerm2();
        computePGrad();
        computePxGradAndPyGrad();
    }
}
```

### 3.3 拆分(任务拆分，多核拆分)

3Pipeline Kernel：基本任务类型为Block任务，core间按照对batch维度进行拆分，core内在S-T矩阵的对角线方向进行循环计算

Default Kernel：基本任务类型为Block任务，core间按照对角线对S和T维度进行划分，每一次Launch Batch乘以并行数量的job，在对角线并行计算。core内计算一个batch内的Block

### 3.4 性能优化设计

1、资源分配

3Pipeline kernel NRAM空间划分参考3.2 伪代码实现中的描述；

Default kernel NRAM空间划分：

```
  /******************************** NRAM SPACE ******************************/
  /* Load Init */
  /*|---------------------------------------------------------------------|*/
  /*| px,py |  p, p_grad  |large_neg    |         |         |             |*/
  /*| 2*S*T |2*(S+1)*(T+1)| 2*min_len+1 | min_len | min_len | 2*min_len+1 |*/
  /*|---------------------------------------------------------------------|*/
  /* Compute term1 and term2 */
  /*|------------------------------------------------------------------|*/
  /*| px,py |  p          |large_neg,mask|cur_term1,2| cur_p | next_p  |*/
  /*| 2*S*T |2*(S+1)*(T+1)| 2*min_len+1  | 2*min_len |min_len|min_len+1|*/
  /*|------------------------------------------------------------------|*/
  /* Compute px_grad, py_grad, p_grad */
  /*|------------------------------------------------------------------------|*/
  /*|px/y_grad|     p_grad  |           | cur_term1,2 |cur_p_grad|next_p_grad|*/
  /*|         |             |           |cur_px/y_grad|          |           |*/
  /*|  2*S*T  |2*(S+1)*(T+1)|2*min_len+1|  2*min_len  | min_len  | min_len+1 |*/
  /*|------------------------------------------------------------------------|*/
```

2、流水设计

在B=4, T=104, S=15的规模下，排流水的收益不大，因此暂不排流水。待后续明确支持规模后，视情况进行优化

在default kernel中，由于为了将每个job中的计算并行度提升至最大，所以并未切乒乓进行计算。而在代码中，可以考虑前后没有数据依赖的memcpy NRAM2NRAM与计算进行指令级的并行，进行细粒度流水。

### 3.5 可维护性设计

1、bangc 代码中加入必要的 log 信息，比如输入的规模、数据类型、layout 这些，以及如果出错会导致程序 core dump 的变量，比如 IO 指令的 data_size、dim xyz 的值等，这些信息都是有利于快速定位问题；

2、对每一个函数命名变量命名都有充分的注释；

3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代。

### 3.6 测试用例设计

- 算子在网络中用到的规模：
  暂无

- 边界 case：
  * T=0; S=0; T= 1; S= 1
  * opt_boundary中存在begin_symbol = end_symbol
  * opt_boundary中存在begin_frame = end_frame

其他可根据需要进行补充。算子开发完毕后，补充测试报告链接。

### 3.7 算子防呆检查

1、检查handle/px_desc/py_desc/p_desc/ans_grad_desc/px_grad_desc/py_grad_desc是否为空

2、检查输入输出空间指针px/py/p/ans_grad/px_grad/py_grad是否为空

3、检查opt_boundary_desc和opt_boundary是否同时为空，或者同时不为空

3、检查0 元素，如果输入输出中包含0元素直接返回MLUOP_STATUS_SUCCESS

3、涉及 workspace 算子对于 workspace_size 的检查防呆；

4、检查px的维度为[B, S,T+1]；py的维度为[B, S+1,T]；p的维度为[B, S+1,T+1]；ans_grad的维度为[B]；px_grad的维度为[B, S,T+1]；py_grad的维度为[B, S+1,T]；如果opt_boundary不为空，opt_boundary为维度为[B，4]

5、检查px/py/p/ans_grad/px_grad/py_grad的数据类型为float；如果opt_boundary不为空，opt_boundary的数据类型为int64

6、检查输入输出是否包含large tensor，或者输入规模是否超出限制



## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

暂无

### 4.2 已经过优化的规模说明

暂无
