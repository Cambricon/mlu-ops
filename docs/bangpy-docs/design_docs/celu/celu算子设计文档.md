# BANGPy Celu 算子开发设计方案

- #### 文档基本信息

| 算子名称   | Celu              |
|--------|-------------------|
| 编制人/日期 | SS7D631/2022-5-18 |   
| 审批人/日期 |                   |

- #### 修改记录

| 修订人     | 修订日期      | 修订描述 |
|---------|-----------|------|
| SS7D631 | 2022-5-18 | 首次提交 |  

- #### 内容描述

本文档为 `Celu` 算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

## 1 需求分析

### 1.1 算子需求分析

| 算子功能简介    | 根据给定的α值对张量每个元素计算max(0, x) + min(0, α ∗ (exp(x / α) − 1)) |
|-----------|----------------------------------------------------------|
| 需求来源      | 为bangpy-ops提供算子demo                                      |
| 应用网络      |                                                          |
| 输入数据类型    | float                                                    |
| 输入 Shape  | buffer_in0[任意维度]   buffer_alpha[1]  inplace:bool         |
| 输入 Layout | buffer_in0:Array     buffer_alpha:Array  inplace:bool    |
| 输入        | input：Array   shape为任意维度                                 |
| 输出数据类型    | float                                                    |
| 输出        | buffer_out:Array      shape同输入                           |


### 1.2 算子功能和应用场景描述

功能：Celu为Elu激活函数的变体，采用参数在负数区间为指数计算min(0, α ∗ (exp(x / α) − 1))，参数在正数区间为线性计算max(0, x) ，是一种更为平滑的激活函数。  
该函数公式为celu(x) = max(0, x) + min(0, α ∗ (exp(x / α) − 1))。

例如：  
data_x = [  85.20301468, -442.93308314 , 128.46804217]  
fun = Celu(2)  
fun(data_x) == [ 85.20302 , -1.9999999, 128.46805  ]  


### 1.3 算子输入输出参数要求
| 参数           | 语义                  | 类型（输入/输出） | 支持类型  | 物理布局     | 规模限制 |
|--------------|---------------------|-----------|-------|----------|------|
| buffer_in0   | 输入的任意shape的buffer   | 输入        | float | ARRAY    | 无    | --------      |
| buffer_alpha | CELU公式的α值。默认值为1.0   | 输入        | float | ARRAY    | 无    | --------      |
| inplace      | 是否原位替换              | 输入        | bool  | -------- | 无    | --------      |
| buffer_out   | 与输入shape一致的输出buffer | 输出        | float | ARRAY    | 无    | --------      |

### 1.4 算子限制

| 限制类型   | 详细说明                   |
|--------|------------------------|
| 数据类型限制 | inplace为bool值 其余为float |
| 布局限制   | 仅支持ARRAY的layout        |
| 规模限制   | 无                      |

### 1.5 验收标准

#### 1.5.1 精度验收标准

本算子属于 `激活` 类算子，验收标准为  diff1 <= 3e-3 && diff2 <= 3e-3   

#### 1.5.2 性能验收标准

待定。

## 2 算子接口设计

### 2.1 参考接口

- pytorch
- torch.nn.CELU(alpha=1.0, inplace=False)
[torch.nn.CELU文档](https://pytorch.org/docs/stable/generated/torch.nn.CELU.html?highlight=celu#torch.nn.CELU)
```python
# https://github.com/pytorch/pytorch/blob/master/torch/nn/quantized/functional.py
def celu(input: Tensor,
         scale: float,
         zero_point: int, 
         alpha: float = 1.) -> Tensor:
input = torch.randn(2)
output = m(input)
```

### 2.2 接口设计

```python
m = Celu()
input = np.random.uniform(low = -1000, high = 1000, size = shape)
output = m(input)
```

## 3 实现方案设计

### 3.1 实现方案
计算max(0, x) + min(0, α ∗ (exp(x / α) − 1))。
x为输入张量中的值，α为celu公式的参数。  
将计算分为max(0, x) 与 min(0, α ∗ (exp(x / α) − 1)) 两部分。    
min(0, α ∗ (exp(x / α) − 1))中根据α是否为0分别讨论：    
当α为0时，min直接返回0。  
不为0时正常计算min(0, α ∗ (exp(x / α) − 1))。    
计算max(0, x)。  
将max和min相加。  
拷贝至cpu端。  

### 3.2 伪代码实现

```python
# 变量说明
# buffer_alpha      celu表达式中的a
# const_one         常量1
# const_zero        常量0
# once_loop_start   当前的开始索引
# calc_size         本次计算的实际长度
# nram_middle_value nram中存放计算中间过程的buffer
# nram_buffer_in0   nram中存放gram中拷贝数据的buffer
# nram_min          nram中存放计算最小值的buffer
# nram_max          nram中存放计算最大值的buffer
# buffer_out        输出buffer
with self.bp.if_scope(buffer_alpha != 0):
    self.bp.divide(nram_middle_value, nram_buffer_in0, buffer_alpha)
    self.bp.exp(nram_middle_value, nram_middle_value)
    self.bp.subtract(nram_middle_value, nram_middle_value, const_one)
    self.bp.multiply(nram_middle_value, nram_middle_value, alpha)
    self.bp.minimum(nram_min, nram_middle_value,const_zero)
with self.bp.else_scope():
    self.bp.zeros(nram_min)         
self.bp.maximum(nram_max, nram_buffer_in0, const_zero)
self.bp.add(nram_buffer_in0, nram_max, nram_min)         
self.bp.memcpy(buffer_out[once_loop_start:once_loop_start + calc_size], nram_buffer_in0[:calc_size])
```
### 3.3 拆分(任务拆分，多核拆分)

使用当前设备的所有核心，并尽可能的将数据均摊到各核。计算公式近似为：数据总量 /（cluster数量*每个cluster的核心数）。

### 3.4 性能优化设计
尽可能将数据分摊至各核。  
计算均采用向量api。

### 3.5 可维护性设计

添加变量及函数的相关注释，代码风格遵守PEP8编码规范。


### 3.6 测试用例设计

- 算子在测试时使用的规模：
  固定测试规模0元素、单个元素、两个元素，128字节对齐，128字节对齐边界，nram空间满占用，nram空间满占用边界。
  通过shape随机生成函数 生成若干二维及以上shape。
  并通过bangpy提供的测试接口比较每次计算后cpu计算结果和mlu结算结果得误差是否在精度得误差范围内。

## 3.7 算子防呆检查  

### 3.7 算子防呆检查    
| 测试点      | 验收标准 | 测试结果（出错信息） |
|----------|------|------------|
| 公式参数α设为0 | 正常计算 | 通过         |

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

| 提交日期 | 问题规模 | 问题描述 | 是否已修复 |
|------|------|------|-------|
|      |      |      |       |

### 4.2 已经过优化的规模说明

| 提交日期 | 修复规模 | 修复问题 |
|------|------|------|
|      |      |      |

## 5 方案实施

### 5.1 开发测试计划

2022.4.30 算子入库

### 5.2 风险分析

暂无。
