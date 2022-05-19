# BANGPy LogAddExp 算子开发设计方案

- #### 文档基本信息

| 算子名称     | LogAddExp              |
| ----------- | -------------- |
| 编制人/日期  | UniqueSquirrel/2022-5-18| 
| 审批人/日期  |              |

- #### 修改记录

| 修订人           | 修订日期    | 修订描述 |
| --------------- | ---------- | ------- |
| UniqueSquirrel  | 2022-5-18 | 首次提交 |  

- #### 内容描述

本文档为 `LogAddExp` 算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

## 1 需求分析

### 1.1 算子需求分析

| 算子功能简介               | 输入的幂和的对数。                        |
| ------------------------ | ---------------------------------------- |
| 需求来源                  | 为bangpy-ops提供算子demo                  |  
| 应用网络                  | ResNet等                                 |
| 输入数据类型               | float                                   |
| 输入                      | input1,input2:ARRAY     如果shape不相等 则它们必须可以广播到一个公共形状(成为输出的形状)。|
| 输出数据类型               | float                                    |
| 输出                      | out:Array    shape为输入的公共形状         |


### 1.2 算子功能和应用场景描述

功能：计算 log(exp(x1) + exp(x2)) 。此函数在计算的事件概率可能小到超出正常浮点数范围的统计中很有用。在这种情况下，存储计算概率的对数。此函数允许添加以这种方式存储的概率。

例如：
data_x = [[-744.38378411  , 32.08532465 , 259.21401044],[ -65.55983881 ,-783.89169849 , 692.46914092]]
data_y = [ 205.4972709 , -982.95625446 , 731.07663893]
logaddexp(data_x,data_y) == [[ 205.49727  , 32.085323 , 731.07666 ] , [ 205.49727 , -783.8917 , 731.07666 ]]


应用场景：ResNet等

### 1.3 算子输入输出参数要求

| 参数    | 语义                                | 类型（输入/输出）| 支持类型     | 物理布局 | 规模限制 |
| ------ | ------------------------------------| ----------------| ----------- | ------ | -------- |
| input1 | 输入的任意shape的buffer              | 输入             | float      | ARRAY  | 无        |
| input2 | 输入的任意shape的buffer              | 输入             | float      | ARRAY  | 无        |
| output | 输出的shape为输入的公共shape的buffer  | 输出             | float      | ARRAY  | 无        |

### 1.4 算子限制

| 限制类型      | 详细说明                 |
| ------------ | ----------------------- |
| 数据类型限制   | input 和 output 必须同时为同一数据类型  |
| 布局限制      | 仅支持ARRAY的layout |
| 规模限制      | 无 |

### 1.5 验收标准

#### 1.5.1 精度验收标准

本算子属于 `算术` 类算子，验收标准为 diff3=0。   

#### 1.5.2 性能验收标准

待定。

## 2 算子接口设计

### 2.1 参考接口

- numpy

```python
numpy.logaddexp(data_x,data_y)
```

### 2.2 接口设计

```python
logaddexp(input1, input2, output)
```

## 3 实现方案设计

### 3.1 实现方案
为防止数据溢出  将公式变形为 x+log(exp(y-x) +1) 当y-x小于exp计算范围的最小值时返回x 大于计算范围最大值时返回y 
将nram空间开辟出5个尽可能大的满足128字节对齐且大小相等的buffer ,nram_buffer_in0、nram_buffer_in1、nram_x_bool、nram_y_bool、nram_middle_value
将输入从gram中循环拷贝至nram_buffer_in0 与 nram_buffer_in1 中 其中 nram_buffer_in0存储的时x  nram_buffer_in1存储的是y
nram_middle_value用来存储中间结果
nram_x_bool用以存储 y - x 结果小于exp计算范围最小值的真值  符合判断条件的结果为1 其余为0
nram_y_bool用以存储 y - x 结果大于exp计算范围最大值的真值  符合判断条件的结果为1 其余为0
计算y-x 的差值 存入 nram_middle_value
与阈值比较 将结果分别存入 nram_x_bool nram_y_bool 
计算 x+log(exp(y-x) +1) 结果存入 nram_middle_value
将对nram_x_bool 与 nram_y_bool 分别取反 再与nram_middle_value 做乘法 将溢出位的数据归0
再将对nram_x_bool 与 nram_y_bool 分别取反 恢复其标记功能
nram_x_bool 与 nram_buffer_in0 做乘法后 并于 nram_middle_value 相加 完成下溢替换
nram_y_bool 与 nram_buffer_in1 做乘法后 并于 nram_middle_value 相加 完成上溢替换
将nram_middle_value拷贝至输出计算完成
### 3.2 伪代码实现

```python
self.bp.subtract(nram_middle_value, nram_buffer_in1, nram_buffer_in0) #y-x
self.mark_the_out_of_range_vlaue(nram_middle_value,nram_x_bool,nram_y_bool) #标记溢出
self.bp.exp(nram_middle_value, nram_middle_value) #指数计算   
self.bp.add(nram_middle_value,nram_middle_value,const_one) # +1  
self.bp.log(nram_middle_value, nram_middle_value) #取对数       
self.bp.add(nram_middle_value,nram_buffer_in0,nram_middle_value) # +x
self.replace_the_marked_position_with_the_value_of_the_same_position(nram_middle_value,nram_buffer_in1,nram_y_bool) #上溢替换
self.replace_the_marked_position_with_the_value_of_the_same_position(nram_middle_value,nram_buffer_in0,nram_x_bool) #下溢替换
self.bp.memcpy(buffer_out[once_loop_start:once_loop_start + calc_size], nram_middle_value[:calc_size]) #拷贝至输出    
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
  通过shape随机生成函数 生成若干二维及以上shape 并随机将input2的规模随机成input1的子集 以测试不同规模的计算
  并通过bangpy提供得测试接口比较每次计算后cpu计算结果和mlu结算结果得误差是否在精度得误差范围内 

### 3.7 算子防呆检查    

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