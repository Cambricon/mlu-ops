Thanks for your contribution and we appreciate it a lot. 

## 1. Motivation

Please describe the motivation of this pull request and the goal you want to achieve through this pull request.

## 2. Modification

Please briefly describe what modification is made in this pull request, and indicate where to make the modification.

## 3. Test Report

If you want to know how to do operator testing, you can see [GTest-User-Guide-zh](../docs/GTest-User-Guide-zh.md).

### 3.1 The Description of modification

#### 3.1.1 Accuracy Acceptance Standard

For static threshold standard details, see: [MLU-OPS Accuracy Acceptance Standard](../docs/MLU-OPS精度验收标准.md)

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

#### 3.1.3 New Feature Test

If you have checked the following items, please put a check mark in front of the corresponding items.

- [ ] Data type test
- [ ] Multidimensional tensor test
- [ ] Layout test
- [ ] Different size/integer remainder end segment/alignment misalignment test
- [ ] Zero dimensional tensor test/zero element test
- [ ] stability test
- [ ] Multiple platform test
- [ ] Gen_case module test
- [ ] Nan/INF tests 
- [ ] For memory leak check details, see[GTest-User-Guide-zh](../docs/GTest-User-Guide-zh.md).
- [ ] For code coverage check details, see: [GTest-User-Guide-zh](../docs/GTest-User-Guide-zh.md).
- [ ] For I/O calculation efficiency check details see: [MLU-OPS Performance Acceptance Criteria](../docs/MLU-OPS性能验收标准.md).

#### 3.1.4 Parameter Check

When a new operator is submitted, the test points are given and the test results are stated.

| Test Point         | Acceptance Criteria | Test Result (Error Message) |
| -------------- | -------- | -------------------- |
| Whether it conforms to the operator restriction | Normal error |                      |
| Whether illegal parameters are passed  | Normal error |                      |

### 3.2 Performance Test

See [MLU-OPS Performance Acceptance Criteria](../docs/MLU-OPS性能验收标准.md) for details.

Platform ：MLU270

|Operator|Mlu_hardware_time(us)|Mlu_interface_time(us)|Mlu_io_efficiency|Mlu_compute_efficiency|Mlu_workwpace_size(Bytes)|Data_type|Shape|
|-----|----|----|----|----|----|------|-----|
|op_name|   |    |     |    |    |    |     |
|op_name|   |    |     |    |    |    |     |

Platform ：MLU290

|Operator|Mlu_hardware_time(us)|Mlu_interface_time(us)|Mlu_io_efficiency|Mlu_compute_efficiency|Mlu_workwpace_size(Bytes)|Data_type|Shape|
|-----|----|----|----|----|----|------|-----|
|op_name|   |    |     |    |    |    |     |
|op_name|   |    |     |    |    |    |     |

Platform：MLU370

|Operator|Mlu_hardware_time(us)|Mlu_interface_time(us)|Mlu_io_efficiency|Mlu_compute_efficiency|Mlu_workwpace_size(Bytes)|Data_type|Shape|
|-----|----|----|----|----|----|------|-----|
|op_name|   |    |     |    |    |    |     |
|op_name|   |    |     |    |    |    |     |

### 3.3 Summary Analysis

Please give a brief overview here, if any need to note and summarize the content.
