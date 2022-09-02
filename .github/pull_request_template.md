Thanks for your contribution and we appreciate it a lot. Please state your motivation and modification for this pull_request here.

## 1. Motivation

Please describe the motivation of this PR and the goal you want to achieve through this PR.

## 2. Modification

Please briefly describe what modification is made in this PR ，and indicate where to modify it.

## 3. Test Report

If you want to know how to do operator testing, you can see [GTest-User-Guide-zh](../docs/GTest-User-Guide-zh.md)

### 3.1 Modify The Description

#### 3.1.1 Accuracy Acceptance Standard

- Static threshold standard
  Detailed see: [MLU-OPS Accuracy Acceptance Standard](../docs/MLU-OPS精度验收标准.md)
- [ ] diff1, diff1 <= 3e-3
- [ ] diff2, diff2 <= 3e-3

#### 3.1.2 Operator Scheme checklist

|      Serial Number     |           Demand            |      Demand For Details       |
|----------------|---------------------------|---------------------|
|        1       |          support hardware         | MLU270 <br> MLU290 <br>MLU370|
|        2       |          job type          |    block <br> U1 <br> U4    |
|        3       |         layout            |  NHWC 、NCHW、ARRAY etc    |
|        4       |         multidimensional              |       Whether multi-dimensions are supported         |
|        5       |         zero elements             |       Whether element zero is supported         |
|        6       |         data type       |         half / float etc           |
|        7      |        size limit           |       If so, please explain the restrictions and reasons      |

#### 3.1.3 New Feature Testing

- [ ] Data type testing
- [ ] Multidimensional tensor testing
- [ ] Layout test
- [ ] Different size / integer remainder end segment / Alignment misalignment test
- [ ] Zero dimensional tensor test / zero element test
- [ ] Test of stability
- [ ] Multiple platform testing
- [ ] Gen_case module test
- [ ] Nan / INF tests 
- [ ] Memory leak checking, detailed see: [GTest-User-Guide-zh](../docs/GTest-User-Guide-zh.md)
- [ ] Code coverage check, detailed see: [GTest-User-Guide-zh](../docs/GTest-User-Guide-zh.md)
- [ ] I/O calculation efficiency check, detailed see: [MLU-OPS Performance Acceptance Criteria](../docs/MLU-OPS性能验收标准.md) 

#### 3.1.4 Parameter Check

When a new operator is submitted, the test points are given and the test results are stated.

| Test Point         | Acceptance Criteria | Test result (error message) |
| -------------- | -------- | -------------------- |
| Don't conform to the operator restriction | normal error |                      |
|  Illegal parameter passing  | normal error |                      |

### 3.2 Performance Test

Detailed see：[MLU-OPS Performance Acceptance Criteria](../docs/MLU-OPS性能验收标准.md)

Platform ：MLU270

|operator|mlu_hardware_time(us)|mlu_interface_time(us)|mlu_io_efficiency|mlu_compute_efficiency|mlu_workwpace_size(Bytes)|data_type|shape|
|-----|----|----|----|----|----|------|-----|
|op_name|   |    |     |    |    |    |     |
|op_name|   |    |     |    |    |    |     |

Platform ：MLU290

|operator|mlu_hardware_time(us)|mlu_interface_time(us)|mlu_io_efficiency|mlu_compute_efficiency|mlu_workwpace_size(Bytes)|data_type|shape|
|-----|----|----|----|----|----|------|-----|
|op_name|   |    |     |    |    |    |     |
|op_name|   |    |     |    |    |    |     |

Platform：MLU370

|operator|mlu_hardware_time(us)|mlu_interface_time(us)|mlu_io_efficiency|mlu_compute_efficiency|mlu_workwpace_size(Bytes)|data_type|shape|
|-----|----|----|----|----|----|------|-----|
|op_name|   |    |     |    |    |    |     |
|op_name|   |    |     |    |    |    |     |

### 3.3 Summary Analysis
