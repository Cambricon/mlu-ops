# mluopGetIndicePairs 算子开发设计方案

* #### 文档基本信息

| 算子名称    | `mluopGetIndicePairs`                                  |
| ----------- | ---------------------------------------------------- |
| 编制人/日期 | 杜军/2022-12-15                                         |
| 审批人/日期 | 王远/2022-12-15                                        |
| 审批人/日期 | 董成威/2022-12-15                                       |

* #### 修改记录

| 版本号| 修订人 | 修订日期 | 修订描述 |
| ----- | ------ | -------  | -------  |
|  v1.0 | dujun | 2023-02-07| 首次提交 |

* #### 内容描述

本文档为`mluopGetIndicePairs`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

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
- 确认算子需求是否已经过框架层review（滤除mluop已支持的算子）

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


| 算子功能简介| 用于生成sparse_conv计算的input,filter和output的坐标                           |
|-------------|--------------------------------------------------------------------------|
| 需求来源      | mmcv                                                                    |
| 应用网络      | centerPoint / OpenPCDet                                                 |
| 输入数据类型   | int                                                                     |
| 输入Shape    | indices: [L, 4]                                                          |
| 输出数据类型   | int                                                                     |
| 输出Shape   | indice_pairs:[ k,2,l ]; indices_out: [ num_act_out, 4]; indices_num [ k ] |
| 模式(可选） | default(sparse) or subm                                                   |
| 是否需要支持原位        | 否                                                              |
| 是否需要支持stride机制  | 否                                                              |
| 是否需要支持广播        | 否                                                              |
| 0元素检查是否直接返回   | 是                                                               |


### 1.2 算子功能和应用场景描述
get_indices_pair 算子有一个输入tensor, 为indices[L,4], 表示conv的输入input中有L个active input site,对应的维度大小分别为n,di,hi,wi保存在第二维中; conv_desc保存conv的stride,dalition,pad,filter_desc用于获取kd,kh,kw, y_desc用于获取do,ho,wo等信息;  
sparse conv的核心在于找到filter各个点计算的input active site的index， 组成indices_pair;  
对于default模式,filter在滑动时存在active input site的output site即认为时有效输出，记录对应的index;  
而对于subm模式, 基本模式与default一致，但对于active output site的会要求其对应的input site是active的  
(subm 模式 要求input size与output size一致);
简易算法流程：
1. 遍历每个active input site, 找到能生成output site点的数量及其坐标index；
2. 根据算法模式要求,确定有效输出点；
3. 将有点输出点的index 分别填入indices_pair, indices_out与indices_num;


### 1.3 算子输入输出参数要求

| 参数             | 语义                           | 类型（输入/输出） | 支持类型               | 物理布局 | 规模限制 |
| ---------------- | ------------------------------ | ----------------- | ---------------------- | -------- | -------- |
| handle           |                                | 输入              |                        | /        | 无       |
| sparse_conv_desc | sparse_conv 描述符             | 输入              |                        | /        | 无       |
| indices_desc     | indices tensor描述符        | 输入              |                        | ARRAY    | 无       |
| indices          | active input site index        | 输入              | int32                  | /        | 无       |
| workspace        | workspace 指针                 | 输入              |                        | ARRAY    | 无       |
| workspace_size   | workspace size 大小            | 输入              |                        | /        | 无       |
| indice_pairs_desc | indices_pair tensor 描述符     | 输出              |                        | /        | 无       |
| indice_pairs      |                                | 输出              | int32                  | ARRAY    | 无       |
| out_indices_desc  |                                | 输出              |                        | ARRAY    | 无       |
| out_indices       |                                | 输出              | int32                  | ARRAY    | 无       |
| indice_num_desc  |                                | 输出              |                        | ARRAY    | 无       |
| indice_num       |                                | 输出              | int32                  | ARRAY    | 无       |


### 1.4 算子限制

该小节为limitation.md的补充。详细描述与框架需求相比，算子尚有哪些功能/模式/范围/数据类型/xxx不支持。
使用表格或分段列出均可。

| 限制类型    | 详细说明                  |
| ----------- | ------------------------- |
| 原位限制    | 不支持原位|
| stride限制  | 不支持stride机制|
| 广播限制    | 不支持广播|

1. 目前 get_indice_pairs 只支持3d.
2. 目前 transpose 和inverse 只支持为0的情况. 
3. get_indice_pairs 支持int输入
4. get_indice_pairs 的输入indices只能为2维, [input_active_in, 4], 
5. get_indice_pairs 的输入indice_paris只能为3维, [kernel_size, 2, input_active_in]; indice_num 为1维, [ kernel_size ]; out_indices 为2维, [num_act_out, 4];

### 1.5 验收标准

#### 1.5.1 精度验收标准

- mluop精度验收标准: mmcv中实现get_indices_pair使用int, 主要是针对sparse conv获取的index，用于表示准确性; 故mluop对于index的精度要求应是diff3=0;

#### 1.5.2 性能验收标准

- 附上算子测试报告链接，测试报告必须包括框架给出的网络中规模的性能数据以及对应效率值。

## 2 算子接口设计

### 2.1 参考接口

- mmcv
```c++
// 给出mmcv接口
get_indice_pairs_func(indices, batch_size, out_shape, spatial_shape, ksize, stride, padding, dilation, out_padding, int(subm),int(transpose))

https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/cuda/spconv_ops_cuda.cu

```

### 2.2 接口设计

首先定义sparse结构体, 主要包含get_indice_pairs所需要的各参数
```c++
struct mluOpSparseConvolutionStruct {
  int dimNb;
  int batch_size;
  int pad[MAX_PAD_DIM];
  int stride[MAX_STRIDE_DIM];
  int dilation[MAX_DILATION_DIM];
  int input_space[MAX_INPUT_DIM];
  int filter_space[MAX_FILTER_DIM];
  int output_space[MAX_OUTPUT_DIM];
  int sub_m = 0;
  int transpose = 0;
  int inverse = 0;
};
```

```c++
// 给出mluop算子接口
mluopStatus_t MLUOP_WIN_API mluopGetIndicePairs(mluOpHandle_t handle,
                                                const mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
                                                const mluOpTensorDescriptor_t indices_desc,
                                                const void *indices,
                                                void *workspace,
                                                size_t workspace_size,
                                                const mluOpTensorDescriptor_t indice_pairs_desc,
                                                void *indice_pairs,
                                                const mluOpTensorDescriptor_t out_indices_desc,
                                                void *out_indices,
                                                const mluOpTensorDescriptor_t indice_num_desc,
                                                void *indice_num);

// 定义该算子workspace接口
mluopStatus_t MLUOP_WIN_API mluopGetIndicePairsWorkspaceSize(mluOpHandle_t handle,
                                                             const mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
                                                             const mluOpTensorDescriptor_t indices_desc,
                                                             const mluOpTensorDescriptor_t indice_pairs_desc,
                                                             const mluOpTensorDescriptor_t out_indices_desc,
                                                             const mluOpTensorDescriptor_t indice_num_desc,
                                                             size_t *workspace_size);

```
## 3 实现方案设计

首先假设nram空间足够大, 一次处理完所有得数据得算法方案如下
输入数据维度为L,4  4表示ni di hi wi
### 3.1 实现方案
整体方案如下（default）
  1. load数据 indices 并做transpose; (L,4 -> 4,L)
  2. 生成indices数据得index为indices_index长度为L;(4,L -> 5,L)
  3. 将输入数据扩展k份, 对应filter得每个点; (5,L -> 5,L,K)  k得大小为 kd * kh * kw
  4. 针对filter生成对应得index,维度为k
  5. 根据公式由输入坐标di,hi,wi和filter点算出对应得do,ho,wo  (5,L,k -> 8,L,k)
  6. 计算除do,ho,wo 是否有效，生成对应mask_valid and mask_bound;
  7. mask_all得维度为L, K, 规约为k维度极为输出indice_num;
  8. 将n,do,ho,wo扩展为输出空间点得index，极为indices_out_expand,  剔除无效值,且调用mluopUnique算子做去重排序;  indices_out_unique
  9. 得到num_act_out, 阶梯化, step_index;
  10. bang_scatter(dst=grid_out src=step_index, offset=indices_out_unique); //获取gride_out
  11. 展开indices_out_unique, num_act_out, 变为num_act_out, 4  indice_out获取；
  12. 根据grid_out替换indices_out_expand的值,得到indices_out_index;
  13. 整合indices_index和indices_out_index 为2,l,k d transpose为k,2,l; indices_pair 获取完成；

对于subm模式, 在步骤6中mask_all的判断需对应output active site;

### 3.2 伪代码实现（可选）

``` c++

computeOutputIndex: output = ((input_index + padding) - k_id * dilation ) / stride;

step1:
void  get_indices_pair_kernel_1() {
    split L  among jobs;  ( L_num_job L_offset_job) 
    split L_job among cores;  ( L_num_cores , L_offset_cores)
    int L_limit_cores; 
    int repeat  =  PAD_UP(L_num_cores, L_limit_cores);
    filter_index init 
    
    for (int i = 0; i < repeat; i++) {
        int L_start = i * L_limit_cores + L_offset_job + L_offset_cores;
        void *core_start_addr = L_start * (Ndim+ 1) * sizeof(float) +  indices;
        load_async(input, core_start_addr, L_limit_cores);  // L_limit_cores, 4
        transpose(tmp1, input)  //  4 L_limit_cores;     ops : 4 * L
        step_index(tmp2,  L_start);   //  5  L_limit_cores   ops : L 
        bang_add(tmp3, tmp2); //  5,L_limit_cores -> k,5,L_limit_cores
        transpose(tmp4, tmp3);  // k,5,L_limit_cores - > 5,L_limit_cores,K   ops: 5*L* K
        cast()   //  ops  5*L*K
        computeOutputIndex(tmp5, tmp4, filter_index);  //  5,L_limit_cores,K   8,L_limit_cores,K;
        /*  ops =  3 * 4 * L * K =12*L*K */
        computeMask(tmp6, tmp5); // mask_all = mask_valid + mask_bound  + mask_subm    L_limit_cores,K;  store 到indices_num
        /* ops  3 * 3 * L*K + 2 * L *K = 11*L*K */

        computeIndicesOuput(tmp7, tmp6); //  indicec_output_expand   L_limit_cores,K
        /* ops L* K  *6 */
        store(mask_all);  // indices_num_ws
        store(indicec_output_expand); // indices_out_expand_ws
        store(indice_index_in)
    }
}

step2:  
mluop_reduce(mask_all)  -> indice_num 

step3:
mluop_unique(out_indices_expand)  ->  out_indices_unique

step4: 
defaultkernel2(out_indices_unique+len)  ->  generate index  

step5: 
scatter_nd() -> grid_out

step6:
gather_nd()  -> out_index

step7:  -> out_indices

step8:  -> indice_pairs
int  num_act_out = genLen(out_indices_unique);
int  output_num =  n * do * ho * wo;
// 涉及到index 建议起block任务  
defaualtkernel4() {
    // int 
    load(out_indices_unique)  // num_act_out
    load(indicesdex_in)  //  l*K
    step_index(tmp1, num_act_out);  // num_act_out
    bang_scatter(dst=grid_out src=step_index, offset=indices_out_unique);  // grid_out  output_num;
    computeIndicesOuput(indices_out, indices_out_unique); // indices_out_unique -indics_output num_act_out  - num_act_out,4
    computeIndicesOutIndex(out_indicesdex,  grid_out, indices_out_expand);  // L,K  bang_gather
    computeIndicesPair(indices_pair, indicesput_index,out_indicesdex);  // 2,L,K, -> K,2,L
    indice_nums_redcuce(); //  mask_all   load  L * K    -> K  
    store(indices_pair); // k* 2*L 
    store(indices_num);  // k
    store(indices_out);  // num_act_out, 4
}

```

### 3.3 拆分(任务拆分，多核拆分)
拆分建议分几个维度来写，job间拆分，cluster间拆分，cluster内拆分：
对于step kernel 1 
可拆分维度 indices 的L维度, 与filter的各个维度和 K(生成 filter index）
起u1任务
1. job间/cluster间均拆分L(潜在问题: 1. L不够大时拆分不均匀会导致很多core空转; 2. 需要对原有数据完成扩展K)
2. job间拆分L, cluster间拆分K;(避免上述第一个问题但仍需要扩展k/core_dim)
3. job间拆分k, cluster间拆分L,  (避免第二个问题, 但拆分会出现core拆分不均, 部分core存在空转);
4. job间/cluster间均拆分K, 因K的维度不会很大, 拆分容易出现core拆分不均， 不太合理;

当前方案设计起block任务, 拆分按照1来完成;

step2: 略；

step3:  block任务,

### 3.4 性能优化设计
1. 资源分配

对于default(sparse)模式 dram空间划分如下
''' c++
    | mask_all | indices_index_in | step_index/ indices_index_out |
    out_indices_expand  | | out_indices_unique | max(grid_out_ws, reduce_ws,
    unique_ws) |
'''

对于subm模式 dram 空间划分如下
''' c++
    | mask_all |indices_index_in | indices_index_out/ step_index |
    indices_in_expand |out_indices_expand| | max(grid_out, reduce_op_ws)|
'''

2. 流水设计

由于计算部分远远超过于IO部分的时间，片上RAM每次分配需要的空间太大，所以不划分乒乓空间，不做软流水。

### 3.5 方案理论性能

完成上述3.1，3.2，3.3，3.4几个步骤之后，基本可以给出一个理论性能，不需要每一个算子都有过于复杂的公式，但是一定要对自己的算子有一个心理的预期，最终实现之后的效率值是多少。
按照cpu端/gpu端实现算法，该算法由于是关于index的计算, 对于mlu向量化实现并不友好, 性能初步评估不会比gpu好;  
按照基本性能评估如下：590 m9  带宽 2765 GB/s    CT  14.7 * 1024 GFLOPS  int32 9.8GTOPS;  
L = 149100   K = 27   num_act_out= 58838  
框架给定时间：  三次kernel：
上述方案性能评估如下
``` c++ 
kernel_1 :
theory_ops =  15*k*L(4*L+L+(k-1)*5*L+5*k*L+5*L*K)+ 12*L*K+ 11*L*K+ 5*L*K= 43KL (fp32) 173105100 + 6KL (int32) 24154200    
theory_ios =  4*L + 3*k*L = (3*k+4)*L  = 11331600 * 4 =  45326400
theory_compute_time:  173105100 * 10^6 /  14.7 * 1024 * 10 ^ 9  (11.7)  +  24154200 / 9.8 * 10^-6 (2.46us) =  15.16us
theory_io_time =  45326400 * 10^6 / 2765 * 0.8 * 10^9 = 1.875us

unique  L * K =   149100 * 27 = 4025700     int32   0-int32_max   测试所得4614us   (unique算子开发者：该规模可优化至2739us)
        
kernel2 :  block 任务 无法拆分
theory_ops: num_act_out + num_act_out  + num_act_out * 4 + num_act_out * 4 + L * K  + 2 * L * K + L * K= 10*num_act_out +4*L*K =  16691180
theory_ios =  num_act_out + L *K + L * K + L * K  + k * 2 * L + num_act_out * 4  + k = 5 *L*K + num_act_out*5 + K = 20422717 * 4 =  81690868
theory_compute_time: 16691180 10^6 /  14.7 * 1024 * 10 ^ 9  * 48 = 54us  涉及到gather scatter
theory_io_time: 81690868 10^6 / 2765 * 0.8 * 10^9 * 48  =    1772us

all_time = 15.16us + 1.875us  + 4614us   + 54us + 1772us = 6457.053us
         
```

### 3.6 可维护性设计

1. bangc代码中加入必要的 log信息，比如输入的规模、数据类型、layout这些，以及如果出错会导致程序core dump的变量，比如IO指令的data_size、dim xyz的值等，这些信息都是有利于快速定位问题。

2. 对每一个函数命名变量命名都有充分的注释

3. 避免魔鬼数字，对于确定的数字尽量使用公共宏来替代 (宏的调用说明以及含义已经注释写在kernels代码中)

4. 同一文件下,同含义命名变量保持一致

### 3.7 测试用例设计

- 框架在需求列表中给出的算子在网络中用到的规模 
``` c++  
对于sub_m模式而言具体的case
输入：
indice_in:  [248636, 4]   int32    
batch_size:   4
spatail shape： 41, 1440, 1440
kernel_size: 3,3,3
stride: 1,1,1
padding: 1,1,1
dilation: 1,1,1
output_padding: 0,0,0
sub_m: true;
transposed : false
grid : none 
输出：
outids :  248636, 4   int32
indice_pairs: 27,2,248636   int32
indice_pair_num: 27   int32
```

``` c++
非sub_m模式 具体case     sparse_conv
indice_in:  [149100, 4]   int32    
batch_size:   4
spatail shape： 11, 360, 360
kernel_size: 3,3,3
stride: 2,2,2
padding: 0,1,1
dilation: 1,1,1
output_padding: 0,0,0
sub_m: false;
transposed : false
grid : none 
输出：
outids :  58838, 4   int32
indice_pairs: 27,2,149100   int32
indice_pair_num: 27   int32
``` 

- 边界case：
  1. case的输出index较大
  2. indices 中各维度点的index接近上下边界
  

### 3.8 算子防呆检查

- 无

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

### 4.2 已经过优化的规模说明

## 5 方案实施

### 5.1 开发测试计划

### 5.2 风险分析