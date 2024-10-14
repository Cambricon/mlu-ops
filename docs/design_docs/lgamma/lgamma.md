# lgamma 算子开发设计方案

- #### 文档基本信息

| 算子名称    | lgamma            |
| ----------- | ----------------- |
| 编制人/日期 | 代韵涛/2024-08-30 |

- #### 修改记录

| 版本号 | 修订人 | 修订日期   | 修订描述     |
| ------ | ------ | ---------- | ------------ |
| V1.0   | 代韵涛 | 2024-08-30 | 首次提交     |

- #### 内容描述

本文档为 `lgamma`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录。

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

| 算子功能简介                                                               | 该算子根据lgamma函数计算输入张量input中每个元素作为输出，输出相同Shape的张量 |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 需求来源                                                                   | Pytorch                                                                      |
| 应用网络                                                                   | LDA模型（隐式狄利克雷分布）                                                  |
| 输入数据类型                                                               | half, float                                                                  |
| 输入Shape                                                                  | 任意Tensor                                                            |
| 输入Layout                                                                 | Array                                                                        |
| 输出数据类型                                                               | half, float                                                                  |
| 输出Shape                                                                  | 任意Tensor                                                                        |
| 输出Layout                                                                 | Array                                                                       |
| 模式(可选）                                                                |                                                                              |
| 是否含有dim/axis等类似语义的参数且该参数支持负数/其他特殊处理              | 否                                                                           |
| 是否含有labels/index等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 否                                                                           |
| 是否需要支持原位                                                           | 是                                                                           |
| 是否需要支持stride机制                                                     | 是                                                                           |
| 是否需要支持广播                                                           | 否                                                                           |
| 0元素检查是否直接返回                                                      | 输入为空 Tensor 直接返回                                                     |

### 1.2 算子功能和应用场景描述

- 算子功能：该算子根据lgamma函数计算输入张量中每个元素作为输出，输出相同Shape的张量

  lgamma函数计算输入的绝对值的gamma函数的自然对数，其公式如下

  $lgamma(x) = ln | \Gamma (x)|$

  gamma函数计算公式如下：
  $\Gamma(x) =  \int_{0}^{+\infty} t^{x-1} e^{-t} dt (x>0)$
- 备注：
- 1、输入存在nan/inf的，不影响其他位置计算结果，原nan/inf在输出后不变

### 1.3 算子输入输出参数要求

| 参数   | 语义                     | 类型（输入/输出） | 支持类型                | 物理布局 | 规模限制 |
| ------ | ------------------------ | ----------------- | ----------------------- | -------- | -------- |
| handle | 操作句柄                 | 输入              | mluOpHandle_t           | /        | 无       |
| x_desc | 输入向量 x 的描述信息    | 输入              | mluOpTensorDescriptor_t | 无       | 无       |
| x      | 指向 x 的 mlu 地址的指针 | 输入              | const void*             | Array    | 无       |
| y_desc | 输出向量 y 的描述信息    | 输出              | mluOpTensorDescriptor_t | 无       | 无       |
| y      | 指向 y 的 mlu 地址的指针 | 输入              | void*                   | Array    | 无       |

### 1.4 算子限制

| 限制类型     | 详细说明               |
| ------------ | ---------------------- |
| 数据类型限制 | input仅支持half和float |
| 布局限制     | Array                  |
| 规模限制     | 无                     |
| 功能限制     | 无                     |
| 数据范围限制 | 无                     |
| 原位限制     | 支持原位               |
| stride限制   | 支持stride机制       |
| 广播限制     | 不支持广播             |

### 1.5 验收标准

#### 1.5.1 精度验收标准

动态阈值 diff1 和 diff2。

#### 1.5.2 性能验收标准

见 [MLU-OPS 性能验收标准](../MLU-OPS-Performance-Acceptance-Standard.md)。

## 2 算子接口设计

### 2.1 参考接口

- cuda

```C++
CONSTEXPR_EXCEPT_WIN_CUDA char lgamma_name[] = "lgamma_kernel";
void lgamma_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "lgamma_cuda", [&]() {
      jitted_gpu_kernel</*name=*/lgamma_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 1>(iter, lgamma_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "lgamma_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ::lgamma(a);
      });
    });
  #endif
}```

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

lgamma 算子是 element wise 类型的算子，因此只需要按照输入数据进行数据切分，目前算子实现了多job并行机制，可按照每个core上可用nram空间对输入数据量进行拆分。

- step1 对输入数据按照数据大小进行拆分，计算每个core处理的数据大小per_core_data。
- step2 根据每个core的可用nram空间对per_core_data进行拆分，若nram放不下，则循环处理


- step3 使用 [Spouge's approximation](https://en.wikipedia.org/wiki/Spouge%27s_approximation) 算法进行计算，考虑到精度，half 需升格为 float 完成计算再转换回 half

$$
\Gamma \left(z + 1\right)  = (z + a)^{z + 0.5}e^{ - z - a}\left(c_{0} + \sum \limits_{k = 1}^{a - 1}\frac{c_{k}}{z + k} + \varepsilon _{a}\left(z\right)\right)
$$

  其中整数 a 的取值决定了 $c_{i}$ 的值，具体计算参考链接网页；且要求 $z > 0$

  计算 Lgamma 函数则是在 Gamma 函数数值逼近算法上取 Log 后，先进行公式上的化简再计算值。同时考虑到以上 Gamma 函数的逼近算法只对输入大于 0 的情况有效，需要通过 [Euler's reflection formula](https://en.wikipedia.org/wiki/Reflection_formula) 计算 $z <= 0$ 的情况：

$$ 
\Gamma(1-z) \Gamma(z) = \frac{\pi}{sin(\pi z)} =>  \Gamma(z) = \frac{\pi}{sin(\pi z) \Gamma(1-z) }
$$

### 3.2 伪代码实现
```
// spouge 算法计算方式为
z < 0:
  lgamma(z) = log(pi) + abs(log(sin(pi * z))) + lgamma(1 - z)
z > 0:
  lgamma(z+1) = (z+0.5)*log(z+a) - (z+a) + log(accm)
  ==>
  lgamma(z) = (z+0.5)*log(z+a) - (z+a) + log(accm/z)


// 为了方便 SIMD 计算，将两种情况合并计算

bool need_reflect = x < 0;
float reflect_x = need_reflect ? 1-x : x;

float accm = coeffs[0];
int numCoeff = coeffs.size();   // aka a
for (size_t k = 1; k < numCoeff; k++) {
  accm += coeffs[k] / (reflect_x + k);
}
float lgamma_x = (reflect_x+0.5)*log(reflect_x+numCoeff) - (reflect_x+numCoeff) + log(accm/reflect_x);

// 为保证 abs(log(sin(pi * z)) 计算精度
float abs_input = std::abs(x);
float abs_frac_input = abs_input - std::floor(abs_input);
float reduced_frac_input = (abs_frac_input > 0.5) ? 1 - abs_frac_input : abs_frac_input;
float reflection_denom = std::log(std::sin(M_PI * reduced_frac_input));

// if x < 0, lgamma(x) = log(pi) - lgamma(1-x) - log(abs(sin(pi * x))) 
float reflection = std::isfinite(reflection_denom) ? log(M_PI) - lgamma_x - reflection_denom: -reflection_denom;
float result = need_reflect ? reflection : lgamma_x;

// 检查输入是否为 inf
return std::isinf(x) ? std::numeric_limits<float>::infinity() : result;

```

### 3.3 拆分(任务拆分，多核拆分)

1、基本任务类型是Block

2、使用三级流水模板对任务平均拆分并分配至各个 core 进行计算

### 3.4 性能优化设计

1、资源分配

| 表项            | 分配策略                                     |
| --------------- | -------------------------------------------- |
| NRAM            | 将 NRAM 平均分为 7 份，前四份作为 input output 的 ping-pong buffer, 后三份作为计算时使用的辅助数组：input-ping - output-ping - input-pong - output-pong - aux1 - aux2 - aux3 |
| WRAM            |                                              |
| SRAM            |                                              |
| DRAM(workspace) |          |

2、NRAM 分配策略

如上表

3、流水设计

采用三级流水模板实现

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
  - case2: 输入范围很窄的向量测试精度 - pass

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

新算子暂无

### 4.2 已经过优化的规模说明

新算子暂无
