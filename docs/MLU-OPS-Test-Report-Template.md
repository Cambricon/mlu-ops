**_本测试报告模板是希望能帮助算子开发者在完成算子开发后进行有效充分的自检，开发出功能、性能都满足要求的高质量算子。_**

# 1. 修改描述

添加算子描述
- 影响范围/算子：op_name
- 影响版本/分支：master

### 1.1 精度验收标准

根据算子需求给出算子分类及其对应的精度验收标准

如：算子采用静态阈值标准：diffs=[diff1, diff2], diff1<=3e-3 && diff2 <= 3e-3

详细见 [MLU-OPS 精度验收标准](./MLU-OPS-Accuracy-Acceptance-Standard.md)

### 1.2 算子方案CHECKLIST

|      序号      |           需求            |            需求详情            |
|----------------|---------------------------|--------------------------------|
|        1       |          支持硬件         |      MLU370 <br> MLU590        |
|        2       |          job类型          |  block <br> U1 <br> U2 <br> U4 |
|        3       |         layout            | 支持NHWC 、NCHW、ARRAY等layout |
|        4       |         多维              |         是否支持多维           |
|        5       |         0元素             |         是否支持0元素          |
|        6       |         数据类型          |         half / float 等        |
|        7       |        规模限制           |     如果有请说明限制和原因     |

### 1.3 新特性测试

- [ ] 数据类型测试
- [ ] 多维张量测试
- [ ] Layout 测试
- [ ] 不同规模 / 整数余数端段 / 对齐不对齐测试
- [ ] 零维张量测试/ 0 元素测试
- [ ] 稳定性测试
- [ ] 多平台测试
- [ ] gen_case模块测试
- [ ] nan / inf测试  
- [ ] 内存泄漏检查, 详见[GTest-User-Guide-zh](./GTest-User-Guide-zh.md)
- [ ] 代码覆盖率检查，详见[GTest-User-Guide-zh](./GTest-User-Guide-zh.md)
- [ ] IO计算效率检查，详见[MLU-OPS性能验收标准](./MLU-OPS-Performance-Acceptance-Standard.md) 


### 1.4 参数检查

提交新算子时，给出测试点，并说明测试结果。

| 测试点         | 验收标准 | 测试结果（出错信息） |
| -------------- | -------- | -------------------- |
| 不符合算子限制 | 正常报错 |                      |
| 非法参数传递   | 正常报错 |                      |

# 2. 功能测试

对于 New Feature Test 部分中使用的案例，此处记录了特征、案例数量和结果。当测试多个操作时，需要多个表来包含这些操作的详细信息。

|    测试点       |        描述                      | 数量或结果 |  备注    |
|-----------------|----------------------------------|------------|----------|
|  数据类型测试   |    half/float/int8               |            |          |
|  多维张量测试   |    支持 1-8 dims                 |            |          |
|  Layout 测试    |    支持 NCHW/NHWC                |            |          |
|  0 元素测试     |    是否支持 0 元素测试           |            |          |
|  稳定性测试     |--gtest_repeat=NUM<br>--thread=NUM|            |          |
|  多平台测试     |     MLU370/MLU590                |            |          |
|  nan / inf 测试 |     是否支持 nan / inf 测试      |            |          |
|  内存泄漏测试   |      测试结果                    |            |          |
|  代码覆盖率测试 |      测试结果                    |            |          |

# 3. 性能测试

详见：[MLU-OPS性能验收标准](./MLU-OPS-Performance-Acceptance-Standard.md)

平台：MLU370

|operator|mlu_hardware_time(us)|mlu_interface_time(us)|mlu_io_efficiency|mlu_compute_efficiency|mlu_workwpace_size(Bytes)|data_type|shape|
|-------|----|----|----|----|----|----|-----|
|op_name|    |    |    |    |    |    |     |
|op_name|    |    |    |    |    |    |     |

平台：MLU590

|operator|mlu_hardware_time(us)|mlu_interface_time(us)|mlu_io_efficiency|mlu_compute_efficiency|mlu_workwpace_size(Bytes)|data_type|shape|
|-------|----|----|----|----|----|----|-----|
|op_name|    |    |    |    |    |    |     |
|op_name|    |    |    |    |    |    |     |

# 4. 总结分析

总结分析主要需要考虑以下几点：

1. 需要对功能、性能测试结果有一个总结性的一句话描述；

2. 对于功能测试中发现的问题，例如精度不达标、规模受限等问题，需要显式列出；

3. 对于性能测试中 efficiency 异常、与对标硬件 latency 或 efficiency 相比差别过大（好于 / 坏于 都要包括）等情况，给出分析解释；
