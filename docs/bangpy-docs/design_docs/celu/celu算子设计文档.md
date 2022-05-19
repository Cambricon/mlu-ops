# BANGPy Celu 算子开发设计方案

- #### 文档基本信息

| 算子名称     | Celu              |
| ----------- | -------------- |
| 编制人/日期  | UniqueSquirrel/2022-5-18 |   
| 审批人/日期  |              |

- #### 修改记录

| 修订人           | 修订日期    | 修订描述 |
| --------------- | ---------- | ------- |
| UniqueSquirrel  | 2022-5-18 | 首次提交 |  

- #### 内容描述

本文档为 `Celu` 算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

## 1 需求分析

### 1.1 算子需求分析

| 算子功能简介               | 激活函数                      |
| ------------------------ | ---------------------------------------- |
| 需求来源                  | 为bangpy-ops提供算子demo                  | 
| 应用网络                  | ResNet等                                 |
| 输入数据类型               | float                                   |
| 输入 Shape                | buffer_in0[任意维度]   buffer_alpha[1]  inplace:bool |
| 输入 Layout               |buffer_in0:Array     buffer_alpha:Array  inplace:bool |
| 输入                      | input：Array   shape为任意维度             |
| 输出数据类型               | float                                    |
| 输出                      | buffer_out:Array      shape同输入                |


### 1.2 算子功能和应用场景描述

功能：Celu为Elu激活函数的变体。采用负数区间为指数计算，整数区间为线性计算,是更为平滑的激活函数，而不是像 ReLU 这样过渡不够平滑的函数。针对输入的每个元素进行Element-wise操作，计算max(0,x)+min(0,α∗(exp(x/α)−1)) 。

例如：
data_x = [  85.20301468, -442.93308314 , 128.46804217]
fun = Celu(2)
fun(data_x) == [ 85.20302 , -1.9999999, 128.46805  ]

应用场景：ResNet等

### 1.3 算子输入输出参数要求

| 参数          | 语义                                | 类型（输入/输出）| 支持类型     | 物理布局 | 规模限制 |
| ------ -------| ------------------------------------| ----------------| ----------- | ------ | -------- |
| buffer_in0    | 输入的任意shape的buffer               | 输入             | float      | ARRAY  | 无        |
| buffer_alpha  | CELU公式的α值。 默认值:1.0            | 输入             | float      | /      | 无        |
| inplace       | bool值                               | 输入             | float      | /      | 无        |
| buffer_out    | 输出的shape与输入一致的buffer         | 输出             | float      | ARRAY  | 无        |

### 1.4 算子限制

| 限制类型      | 详细说明                 |
| ------------ | ----------------------- |
| 数据类型限制   | input 和 output 必须同时为同一数据类型  |
| 布局限制      | 仅支持ARRAY的layout |
| 规模限制      | 无 |

### 1.5 验收标准

#### 1.5.1 精度验收标准

本算子属于 `激活` 类算子，验收标准为  diff1 <= 3e-3 && diff2 <= 3e-3   

#### 1.5.2 性能验收标准

待定。

## 2 算子接口设计

### 2.1 参考接口

- torch

```python
m = torch.nn.CELU()
input = torch.randn(2)
output = m(input)
```

### 2.2 接口设计

```python
m = Celu()
input = np.random.uniform(low=-1000, high=1000, size=shape)
output = m(input)
```

## 3 实现方案设计

### 3.1 实现方案
计算max(0,x)+min(0,α∗(exp(x/α)−1))
将计算分为max(0,x) 与 min(0,α∗(exp(x/α)−1)) 两部分
min(0,α∗(exp(x/α)−1))中根据alpha是否为0分别讨论
    当alpha为0时，min直接返回0    
    不为0时：
        从min(0,α∗(exp(x/α)−1))可知只有当x小于0时，才会取α∗(exp(x/α)−1)的值作为最小值 ，所以将x大于等于0的直接返回0。
        当x<0时 max为0 
        因为x < 0 且 x/a < 0 时 α∗(exp(x/α)−1)才有意义，所以不存在上溢的情况。
        而exp(x/a)的结果必然大于零，则当exp(x/a)下溢时，返回0，α∗(exp(x/α)−1)此时结果为-a。    
    计算max(0,x)
    将max和min相加
    拷贝至输出

### 3.2 伪代码实现

```python
#计算min
with self.bp.if_scope(alpha != 0):
    self.bp.less_equal(nram_marked_zero,nram_buffer_in0,const_zero,'elemwise') #大于等于0的标记为0 小于的标记为1
    self.bp.divide(nram_middle_value,nram_buffer_in0,alpha)#获得x/a 
    self.mark_the_out_of_range_vlaue(nram_middle_value,nram_marked_exp_overrun_the_upper_limit,nram__marked_exp_beyond_the_lower_limit)#标记出所有超出运算范围的值的位置并分别在两个buffer中用0标注
    #前期准备基本完成 开始常规计算
    self.bp.exp(nram_middle_value,nram_middle_value)#计算exp(x/a)
    self.bp.subtract(nram_middle_value, nram_middle_value, const_one)#-1
    self.bp.multiply(nram_middle_value, nram_middle_value, alpha)#*a
    self.bp.minimum(nram_min,nram_middle_value,const_zero)#min(0,...)
    #开始替换
    self.bp.multiply(nram_middle_value, nram_middle_value,nram_marked_zero)#将所有x>=0得位置全部替换成0
    #另一种情况  当（x/a）< e 的最小次方值时   将所有标记位替换成 -a和0中小的那个
    with self.bp.if_scope(alpha * -1 > 0):
        replace_value.assign(0)
    with self.bp.else_scope():
        replace_value.assign(alpha * -1)
    self.replace_the_marked_position_with_the_value_of_the_same_position(nram_middle_value,nram__marked_exp_beyond_the_lower_limit,replace_value)             
with self.bp.else_scope():#当alpha为0时  min全为0
    self.bp.zeros(nram_min)   
#这里开始计算max           
self.bp.maximum(nram_max,nram_buffer_in0,const_zero)
#计算max+min
self.bp.add(nram_buffer_in0,nram_max,nram_min)         
self.bp.memcpy(buffer_out[once_loop_start:once_loop_start + calc_size], nram_buffer_in0[:calc_size])
```
```python
#标记会造成溢出得指数位置
def mark_the_out_of_range_vlaue(self,input,x,y):     
        max_threshold_valu = self.bp.Scalar(self.dtype,"max_threshold_valu",10)
        min_threshold_valu = self.bp.Scalar(self.dtype,"min_threshold_valu",-7.5)
        self.mark_the_value_compare_with_threshold_value(input,x,1,min_threshold_valu)
        self.mark_the_value_compare_with_threshold_value(input,y,0,max_threshold_valu)
```
```python
#采用何种标记办法
def mark_the_value_compare_with_threshold_value(self,input,nram_bool_mark,is_min,threshold_value):
         if  is_min == 1:       
              self.bp.greater_equal(nram_bool_mark,input,threshold_value,'elemwise') #大于等于阈值返回1
         else :  
              self.bp.less_equal(nram_bool_mark,input,threshold_value,'elemwise') #小于等于阈值返回1        
```
```python
#相同位置进行替换

 def replace_the_marked_position_with_the_value_of_the_same_position(self,waiting_to_be_changed_buffer,value_buffer,marked_bool_buffer):
        self.bp.multiply(waiting_to_be_changed_buffer,waiting_to_be_changed_buffer,marked_bool_buffer)
        self.bp.logical_not(marked_bool_buffer,marked_bool_buffer) 
        self.bp.multiply(marked_bool_buffer,value_buffer,marked_bool_buffer) 
        self.bp.add(waiting_to_be_changed_buffer,waiting_to_be_changed_buffer,marked_bool_buffer)
```
### 3.3 拆分(任务拆分，多核拆分)

采用的tasktype固定为UNION16，数据拆分到64个核内计算。

### 3.4 性能优化设计
使用290所有核心
尽可能将数据分摊至各核
计算均采用向量api

### 3.5 可维护性设计

添加变量及函数的相关注释，代码风格遵守PEP8编码规范，支持的target有290


### 3.6 测试用例设计

- 算子在测试时使用的规模：
  固定测试规模(1,),(2,),128字节对齐,128字节对齐边界,满buffer,满buffer边界
  通过shape随机生成函数 生成若干二维及以上shape 
  并通过bangpy提供得测试接口比较每次计算后cpu计算结果和mlu结算结果得误差是否在精度得误差范围内 

## 3.7 算子防呆检查  

除host端自动生成的部分参数防呆检查外，暂不需要进行其他的防呆检查。

## 4 算子性能优化记录

### 4.1 当前存在问题的规模说明

| 提交日期  | 问题规模 | 问题描述 | 是否已修复 |
| --------- | -------- | -------- | ---------- |
|           |          |          |            |

### 4.2 已经过优化的规模说明

| 提交日期  | 修复规模 | 修复问题 |
| --------- | -------- | -------- |
|           |          |          |

## 5 方案实施

### 5.1 开发测试计划

2021.12.28 算子入库   

### 5.2 风险分析

暂无。