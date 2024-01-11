Thanks for your contribution and we appreciate it a lot. :rocket::rocket:

## 1. Motivation

Please describe your motivation and the goal you want to achieve through this pull request.

## 2. Modification

Please briefly describe what modification is made in this pull request, and indicate where to make the modification.

Are new test cases added? If so, please post the corresponding generator-PR link here.

## 3. Test Report

If you want to know how to do operator testing, you can see [GTest-User-Guide-zh](https://github.com/Cambricon/mlu-ops/blob/master/docs/GTest-User-Guide-zh.md).

### 3.1 Modification Details

#### 3.1.1 Accuracy Acceptance Standard

For static threshold standard details, see: [MLU-OPS™ Accuracy Acceptance Standard](https://github.com/Cambricon/mlu-ops/blob/master/docs/MLU-OPS-Accuracy-Acceptance-Standard.md).

- static threshold
  - diff1
    - [ ] float32 mlu diff1 <= 1e-5
    - [ ] float32 mlu diff1 <= 3e-3
    - [ ] float16 mlu diff1 <= 3e-3
  - diff2
    - [ ] float32 mlu diff2 <= 1e-5
    - [ ] float32 mlu diff2 <= 3e-3
    - [ ] float16 mlu diff2 <= 3e-3
  - diff3
    - [ ] mlu diff3 == 0
    - [ ] mlu diff3_1 == 0
    - [ ] mlu diff3_2 == 0
- dynamic threshold
  - [ ] diff1: mlu diff1 <= max(baseline diff1 * 10, static threshold)
  - [ ] diff2: mlu diff2 <= max(baseline diff2 * 10, static threshold)
  - [ ] diff3: mlu diff3 <= max(baseline diff3 * 10, static threshold)
    - float32, threshold = 1e-5
    - float16, threshold = 1e-3

#### 3.1.2 Operator Scheme checklist

- Supported hardware
  - [ ] MLU370
  - [ ] MLU590
- Job types
  - [ ] BLOCK
  - [ ] UNION1
  - [ ] UNION2
  - [ ] UNION4
  - [ ] The operator will dynamically select the most suitable task type, for example, UNION8

### 3.2 Accuracy Test

#### 3.2.1 Accuracy Test

If you have checked the following items, please tick the relevant box.

- [ ] Data type test (e.g. float32/int8)
- [ ] Multi-dimensional tensor test
- [ ] Layout test
- [ ] Different size/integer remainder end segment/alignment misalignment test
- [ ] Zero dimensional tensor test/zero element test
- [ ] stability test
- [ ] Multiple platform test
- [ ] Gen_case module test, see: [Gencase-User-Guide-zh](https://github.com/Cambricon/mlu-ops/blob/master/docs/Gencase-User-Guide-zh.md)
- [ ] Nan/INF tests 
- [ ] Bug fix tests
- [ ] For memory leak check details, see: [GTest-User-Guide-zh](https://github.com/Cambricon/mlu-ops/blob/master/docs/GTest-User-Guide-zh.md)
- [ ] For code coverage check details, see: [GTest-User-Guide-zh](https://github.com/Cambricon/mlu-ops/blob/master/docs/GTest-User-Guide-zh.md)
- [ ] For I/O calculation efficiency check details, see: [MLU-OPS™-Performance-Acceptance-Standard](https://github.com/Cambricon/mlu-ops/blob/master/docs/MLU-OPS-Performance-Acceptance-Standard.md)

#### 3.2.2 Parameter Check

Test Point-1: `When a new operator is submitted, the test points are given and the test results are stated`. Acceptance Standard: `Normal error`.
```bash
Please fill your test results(Error Message) in here, ...
```

Test Point-2: `Whether illegal parameters are passed`. Acceptance Standard: `Normal error`.
```bash
Test results...
```


### 3.3 Performance Test

See [MLU-OPS™ Performance Acceptance Standard](https://github.com/Cambricon/mlu-ops/blob/master/docs/MLU-OPS-Performance-Acceptance-Standard.md) for details.

Platform：MLU370

```bash
# The test results should contain Op name, Shape, Data type,  
#   MLU Hardware Time(us), MLU Interface Time(us), MLU IO Efficiency, 
#   MLU Compute Efficiency, and Mlu Workspace Size(Bytes)
# 
# for example:
#
# ----------- case0 -----------
# case0
# [Op name                ]: abs
# [Shape                  ]: input.shape=[1024,1024,3,4], output.shape=[1024,1024,3,4]
# [Data type]             ]: float32
# [MLU Hardware Time      ]: 15728 (us)
# [MLU Interface Time     ]: 369.008 (us)
# [MLU IO Efficiency      ]: 0.23275
# [MLU Compute Efficiency ]: 0.5
# [Mlu Workspace Size     ]: -1 (Bytes)
# 
# ----------- case1 -----------
# ...
```

Platform：MLU590
```bash
# ----------- case0 -----------
# ----------- case1 -----------
# ...
```

### 3.4 Summary Analysis

Please give a brief overview here, if you want to note and summarize the content.