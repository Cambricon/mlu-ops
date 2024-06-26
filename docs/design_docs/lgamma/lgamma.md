# lgamma 算子开发设计方案


- #### 文档基本信息

| 算子名称    | lgamma                 |
| --------- | -------------------------------------|
| 编制人/日期 | 代韵涛/2024-04-25                      |

- #### 修改记录

| 版本号| 修订人 | 修订日期 | 修订描述 |
| ----- | ------ | -------  | -------  |
| V1.0  | 代韵涛    | 2024-04-25   | 首次提交 |
| V2.0  | 代韵涛    | 2024-06-24   | 修改设计方案 |

- #### 内容描述

本文档为`lgamma`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录。

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

该需求分析为 FIXME


| 算子功能简介| 该算子根据lgamma函数计算输入张量input中每个元素作为输出，输出相同Shape的张量                  |
|-------------|--------------------------------------------------------------|
| 需求来源    | Pytorch                                                       |
| 应用网络    | LDA模型（隐式狄利克雷分布）                                     |
| 输入数据类型|     half, float                                               |
| 输入Shape   | input: 任意Tensor                                             |
| 输入Layout  | Array                                                         |
| 输出数据类型| half, float                                                   |
| 输出Shape   | Array                                                        |
| 输出Layout  | 无要求                                                        |
| 模式(可选） |                                                               |
| 是否含有dim/axis等类似语义的参数且该参数支持负数/其他特殊处理 | 否              |
| 是否含有labels/index等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 否  |
| 是否需要支持原位        | 是                                                  |
| 是否需要支持stride机制  | 否                                                  |
| 是否需要支持广播  | 否                      |
| 0元素检查是否直接返回  | 输入为空 Tensor 直接返回  |

### 1.2 算子功能和应用场景描述


- 算子功能：该算子根据lgamma函数计算输入张量中每个元素作为输出，输出相同Shape的张量

    lgamma函数计算输入的绝对值的gamma函数的自然对数，其公式如下

    $lgamma(x) = ln | \Gamma (x)|$

   gamma函数计算公式如下：
   $\Gamma(x) =  \int_{0}^{+\infty} t^{x-1} e^{-t} dt (x>0)$

- 备注：

- 1、输入存在nan/inf的，不影响其他位置计算结果，原nan/inf在输出后不变

### 1.3 算子输入输出参数要求

| 参数             | 语义                                                         | 类型（输入/输出） | 支持类型               | 物理布局 | 规模限制 |
| ---------------- | ------------------------------------------------------------ | ----------------- | ---------------------- | -------- | -------- |
| handle           |  操作句柄                  | 输入              |    mluOpHandle_t      | /        | 无       |
| x_desc    | 输入向量 x 的描述信息         | 输入              | mluOpTensorDescriptor_t  | 无       | 无       |
| x         | 指向 x 的 mlu 地址的指针    | 输入       | const void*                   | Array       | 无       |
| y_desc    | 输出向量 y 的描述信息         | 输出              | mluOpTensorDescriptor_t  | 无       | 无       |
| y         | 指向 y 的 mlu 地址的指针    | 输入       | void*                   | Array       | 无       |

### 1.4 算子限制

| 限制类型    | 详细说明                                            |
| ----------- | ------------------------------------------------------------ |
| 数据类型限制| input仅支持half和float |
| 布局限制    | Array             |
| 规模限制    | 无               |
| 功能限制    | 无                         |
| 数据范围限制|  无     |
| 原位限制    | 支持原位|
| stride限制  | 不支持stride机制|
| 广播限制    | 不支持广播|

### 1.5 验收标准

#### 1.5.1 精度验收标准

本算子属于算数类算子，验收标准为 diff3=0。

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../MLU-OPS-Performance-Acceptance-Standard.md)。

## 2 算子接口设计

### 2.1 参考接口

- torch
```python
torch.lgamma(input, *, out=None) → Tensor
```

### 2.2 接口设计

```c++
mluOpStatus_t MLUOP_WIN_API mluOpLgamma(mluOpHandle_t handle,
                                        const mluOpTensorDescriptor_t x_desc,
                                        const void *x,
                                        const mluOpTensorDescriptor_t y_desc,
                                        void *y)；
```


## 3 实现方案设计

### 3.1 实现方案

- step1 
将输入张量 input 看做是一个一维数组, 设其长度为 total, 将 total 均分给每个core处理, 使用 total 整除 taskDim 得到结果 core_n，余数记为 rem_n，前 rem_n 个 task 多计算一个元素，以处理不能平均分配到每个 task 的部分
```C++
  int core_n = total / taskDim;
  int rem_n = total % taskDim;    
```
- step2 
计算每个 core 需要处理的数据的位置: 输入向量起始地址为 x, 输出向量起始地址为 y,根据 taskId 计算偏移
```C++
  if (taskId <= rem_n) {
    core_n += 1;
    x += core_n*taskId;
    y += core_n*taskId;
  } else {
    x += (core_n + 1) * rem_n + core_n * (taskId - rem_n);
    y += (core_n + 1) * rem_n + core_n * (taskId - rem_n);
  }
```
 - step3 根据 NRAM 大小计算一次最多能处理的数量 arr_size, 若处理不下, 循环处理, 每个循环中可以并行处理的数据量为处理的 num_deal, 将从偏移处开始的 num_deal 个数据从 GDRAM 拷贝至 NRAM, 如果类型为 half, 需要转换为 float 进行计算
```C++
  for (size_t i = 0; i < core_n; i += arr_size) {
    int num_deal = i + arr_size > core_n ? core_n - i : arr_size;

    if (sizeof(T) == sizeof(float)) {
      __memcpy(buf1, x + i, num_deal*sizeof(float), GDRAM2NRAM);  
    } else if (sizeof(T) == sizeof(half)) {  // half
      __memcpy(buf0, x + i, num_deal*sizeof(half), GDRAM2NRAM);  
      __bang_half2float(buf1, (half *)buf0, num_deal);
    }
    ...
  }

```
- step4 使用数值逼近算法进行计算：关于 Gamma 函数的数值逼近算法有 

  (1) [Lanczos approximation](https://en.wikipedia.org/wiki/Lanczos_approximation): $\Gamma \left(z + 1\right) = \sqrt{2\pi }\left(z + a + 0.5\right)^{z + 0.5}e^{ - \left(z + a + 0.5\right)}\left(c_{0} + \sum \limits_{k = 1}^{N}\frac{c_{k}}{z + k} + \varepsilon _{a}\left(z\right)\right) $

  (2) [Spouge's approximation](https://en.wikipedia.org/wiki/Spouge%27s_approximation): $ \Gamma \left(z + 1\right)  = (z + a)^{z + 0.5}e^{ - z - a}\left(c_{0} + \sum \limits_{k = 1}^{a - 1}\frac{c_{k}}{z + k} + \varepsilon _{a}\left(z\right)\right) $

  其中整数 a 的取值决定了 $c_{i}$ 的值，具体计算参考链接网页；且要求 $z > 0$

  计算 Lgamma 函数则是在 Gamma 函数数值逼近算法上取 Log 后，先进行公式上的化简再计算值。同时考虑到以上 Gamma 函数的逼近算法只对输入大于 0 的情况有效，需要通过 [Euler's reflection formula](https://en.wikipedia.org/wiki/Reflection_formula) 计算 $z <= 0$ 的情况：
  
  $ \Gamma(1-z) \Gamma(z) = \frac{\pi}{sin(\pi z)}$ => $ \Gamma(z) = \frac{\pi}{sin(\pi z) \Gamma(1-z) }$

  以 xx approximation 为例，最终计算方法为 FIXME -- 根据最终方案选取例子进行阐述


- step5 将计算完成后的数据从 NRAM 拷贝回 GDRAM 
```C++
  for (size_t i = 0; i < core_n; i += arr_size) {
    ...
    if (sizeof(T) == sizeof(float)) {
      __memcpy(y + i, buf1, num_deal*sizeof(float), NRAM2GDRAM);  
    } else if (sizeof(T) == sizeof(half)) {  // half
      __mluop_float2half((half *)buf1, buf1, num_deal);
      __memcpy(y + i, buf1, num_deal*sizeof(half), NRAM2GDRAM);  
    }
  }
```

### 3.2 拆分(任务拆分，多核拆分)

1、基本任务类型是Block

2、对 input 进行数据拆分，当前可获得 MLU 核心数决定了任务数量 taskDim，将 input 平均分为 $\lfloor len(input) / taskDim \rfloor$ 个，每次任务分配给核心进行计算

3、对不能均匀拆分的情况下，前 $ len(input) $% taskDim$ 个 task 每个多处理一个数据 


### 3.4 性能优化设计
1、资源分配

| 表项            | 分配策略   |
| ----------------| -----------|
| NRAM            | 存放一次 task 中输入数据、中间数据、结果数据 |
| WRAM            |  |
| SRAM            |  |
| DRAM(workspace) | 用于存放输入张量 input 的内存空间 |

2、NRAM 分配策略

本算子计算过程中需要存放多个临时变量，至少需要 5 个 buffer 同时存放临时变量，再加上软件流水 input output 各需要 1 个 buffer，所以将最大 NRAM 使用空间平均划分为 7 份
```C++
buf_size = FLOOR_ALIGN(LGAMMA_NRAM_USED / sizeof(T) / AUX_N, UNARY_ALIGN_NUM), AUX_N = 7
```

3、流水设计
本算子采用三级软件流水，将 buffer 从 0 到 6 编号，其中 0 1 号 buffer 作为 input 的 ping-pong buffer，5 6 号 buffer 作为 input 的 ping-pong buffer，3 4 5 固定为计算过程使用的 buffer；在流水过程中，每个周期 input output 只需要使用一个 buffer 做输入/输出缓冲，即可以空余各 1 个 buffer 给计算过程使用，具体分配方法如下：

对第 C 个周期处于 input 阶段的数据，其应该被 load 进入 buffer[(C % 2) ? 1 : 0]，结果写入 buffer[(C % 2) ? 6 : 5]，计算时使用的 5 个 buffer 为： { （C % 2) ? 1 : 0, 2, 3, 4, (C % 2) ? 6 : 5 }

### 3.5 可维护性设计

1、bangc 代码中加入必要的 log 信息，比如输入的规模、数据类型、layout 这些，以及如果出错会导致程序 core dump 的变量，比如 IO 指令的 data_size、dim xyz 的值等，这些信息都是有利于快速定位问题；

2、对每一个函数命名变量命名都有充分的注释；

3、避免魔鬼数字，对于确定的数字尽量使用公共宏来替代。

### 3.6 测试用例设计

- 算子在网络中用到的规模：
  - case0: float，input: [ 128, 748, 80 ]
  - case1: float，input: [ 8, 65536, 10 ]
  - case2: float，input: [ 273600, 4 ]
  - case3: float，input: [ 8, 32768, 1 ]
  - case4: float，input: [ 1179648 ]
  - case5: half, input: [ 16646144 ]
  - case6: half, input: [ 18, 48000 ]
  - case7: half, input: [ 10, 65536, 1]
  - case8: half, input: [ 7020, 1000]
  - case9: half, input: [ 57145500 ]

- 边界case：
  - case0: 负整数产生 inf 值 - pass
  - case1: 输入 nan\inf，正确产生输出 - pass 
  - case2: 输入空向量，正确产生输出 - not test 

  
  
### 3.7 算子防呆检查

- 列出算子需要做的防呆，比如

1、指针为空防呆；

2、0 元素检查防呆，VLOG(5)打印信息，是否返回与框架沟通；

3、涉及 workspace 算子对于 workspace_size 的检查防呆；

4、是否需要对输入输出支持的 dtype、layout 以及 shape 进行防呆；

5、elementwise 算子则需要保证输入输出的每一个维度都要一样；

6、算子存在的自身的相关参数防呆。

主要是列出 4,5,6 防呆内容，方便 review。

## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

需要支持 stride 机制

### 4.2 已经过优化的规模说明

新算子暂无
