Thanks for your contribution and we appreciate it a lot. 

## 1. Motivation

Please describe your motivation and the goal you want to achieve through this pull request.

## 2. Modification

Please briefly describe what modification is made in this pull request, and indicate where to make the modification.

## 3. Test Report

If you want to know how to do operator testing, you can see [GTest-User-Guide-zh](../docs/GTest-User-Guide-zh.md).

### 3.1 Modification Details

#### 3.1.1 Accuracy Acceptance Standard

For static threshold standard details, see: [MLU-OPS Accuracy Acceptance Standard](../docs/MLU-OPS-Accuracy-Acceptance-Standard.md).

- [ ] diff1, diff1 <= 3e-3
- [ ] diff2, diff2 <= 3e-3

#### 3.1.2 Operator Scheme checklist

|     No.      |           Details            |      Check Results       |
|----------------|---------------------------|---------------------|
|        1       |          Supported hardware         | MLU270 <br> MLU290 <br>MLU370<br>MLU590|
|        2       |          Job types          |    block <br> U1 <br> U4    |
|        3       |         Layouts            |  NHWC 、NCHW、ARRAY etc    |
|        4       |         Whether multi-dimensions are supported              |                |
|        5       |          Whether element zero is supported             |               |
|        6       |         Data type(half/float)       |         half / float etc           |
|        7      |        Whether there is size limit           |             |

#### 3.1.3 New Feature Test

If you have checked the following items, please tick the relevant box.

- [ ] Data type test
- [ ] Multidimensional tensor test
- [ ] Layout test
- [ ] Different size/integer remainder end segment/alignment misalignment test
- [ ] Zero dimensional tensor test/zero element test
- [ ] stability test
- [ ] Multiple platform test
- [ ] Gen_case module test
- [ ] Nan/INF tests 
- [ ] Bug fix tests
- [ ] For memory leak check details, see[GTest-User-Guide-zh](../docs/GTest-User-Guide-zh.md).
- [ ] For code coverage check details, see: [GTest-User-Guide-zh](../docs/GTest-User-Guide-zh.md).
- [ ] For I/O calculation efficiency check details, see: [MLU-OPS Performance Acceptance Standard](../docs/MLU-OPS-Performance-Acceptance-Standard.md).

#### 3.1.4 Parameter Check

When a new operator is submitted, the test points are given and the test results are stated.

| Test Point         | Acceptance Standard | Test Result (Error Message) |
| -------------- | -------- | -------------------- |
| Whether it conforms to the operator restriction | Normal error |                      |
| Whether illegal parameters are passed  | Normal error |                      |

### 3.2 Performance Test

See [MLU-OPS Performance Acceptance Standard](../docs/MLU-OPS-Performance-Acceptance-Standard.md) for details.

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

Platform：MLU590

|Operator|Mlu_hardware_time(us)|Mlu_interface_time(us)|Mlu_io_efficiency|Mlu_compute_efficiency|Mlu_workwpace_size(Bytes)|Data_type|Shape|
|-----|----|----|----|----|----|------|-----|
|op_name|   |    |     |    |    |    |     |
|op_name|   |    |     |    |    |    |     |

### 3.3 Summary Analysis

Please give a brief overview here, if you want to note and summarize the content.
