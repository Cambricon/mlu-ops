# MLU-OPS™ 算子开发流程

- [概述](#1-概述)
- [文档开发](#2-文档开发)
   - [算子设计文档](#21-算子设计文档)
   - [其它](#22-其它)
- [代码开发](#3-代码开发)
   - [Kernel/Host代码开发](#31-kernelhost-代码开发)
   - [测试代码开发](#32-测试代码开发)
      - [添加 GTest](#321-添加-gtest)
      - [添加测试用例](#322-添加测试用例)
- [算子测试](#4-算子测试)
   - [测试覆盖率](#41-测试覆盖率)
   - [算子验收标准](#42-算子验收标准)
      - [精度验收标准](#421-精度验收标准)
      - [性能验收标准](#422-性能验收标准)
- [代码提交流程](#5-代码提交)

## 1. 概述

MLU-OPS™ 是面向 MLU 平台的人工智能网络加速库，算子的功能实现可以利用寒武纪特有的 BANG C 语言实现。

在介绍算子前，需明确一个理念，算子的设计文档与代码实现同样重要。优秀的设计文档可以帮助其他开发者快速了解算子功能与接口使用方式，从而提升算子质量，故算子设计文档的撰写需格式规范、层次清晰、功能完整。当修改算子的实现代码时，文档需同步进行修改，保持二者的一致性。

一个算子需要入库的文件包含：算子设计文档、算子实现代码、算子测试代码、测试报告。

- 算子设计文档：包含算子的需求分析、API 接口设计、实现方案设计、算子性能优化记录

- 算子实现代码：包含 C++ 实现源码、C 风格对外接口、BANG C 的 kernel 实现源码

- 算子测试代码：算子开发者需要编写算子的测试代码，该代码需能够测试到算子的多种使用场景

- 算子测试报告：包含算子测试结果，如算子测试规模和数据类型、内存泄漏情况、代码覆盖率情况、算子性能及稳定性等

（下面以添加一个 abs 算子为例详细说明在整个算子开发和维护过程中需要添加和修改的文件及其所在目录。算子开发者在开发过程中务必按照说明中的目录和文件进行添加和修改，以确保 MLU-OPS™ 库文件和目录结构的简洁。）

## 2 文档开发

### 2.1 算子设计文档
在 docs/design_docs/ 目录下新建以算子名命名的目录，目录名首字母小写，并在算子目录下新建以算子名命名的 md 文件。如：

```bash
$ cd docs/design_docs/
$ mkdir abs
$ cd abs
$ vim abs.md
```

在 abs 目录下添加的 abs.md 文件，为算子的设计文档，设计文档模板可参考[MLU-OPS™ 算子设计文档模板](./MLU-OPS-Operator-Design-Doc-Template.md)。

如果一个算子存在正向和反向，那么正反向算子当做两个不同的算子来处理，目录结构应为

```bash
|-- docs
   |-- design_docs
      |-- border_align_forward
         |-- border_align_forward.md
      |-- border_align_backward
         |-- border_align_backward.md
```

文档中如有涉及公式的地方， 使用 md 的公式格式，不能使用图片的形式插入。

### 2.2 其它
1. 在`docs/MLU_OPS-OpList.md`中添加算子名和算子类型，具体参考文档描述。
2. 在`docs/user_guide/9_operators/index.rst`中添加对应算子功能简述。

## 3 代码开发

### 3.1 Kernel/Host 代码开发

在 kernels 目录下，添加以算子名命名的目录，然后在该目录下添加算子功能的实现文件、接口声明文件以及 bangc 编写的 kernel 文件，文件名首字母小写。如：

```bash
$ cd kernels
$ mkdir abs  // 添加以算子名命名的目录
$ cd abs
$ touch abs.cpp // abs.cpp  ->  mluop 接口的实现文件
$ touch abs.h   // abs.h    ->  kernel 接口声明文件
$ touch abs.mlu // abs.mlu  ->  以 bangc 编程的 kernel 函数的实现文件
$ cd ../../
$ mlu_op.h // mluop 接口声明文件
```

文件命名及组织规则为：

1. cpp 及 h 文件的文件名为算子名，如 abs.cpp / abs.h 。

2. mlu 文件根据算子的实现以 "算子名 + 实现方式" 的规则进行命名。如算子的实现方式为以 Union1 为最小单位，需命名为 abs_union1.mlu，如以 Union2 为最小单位，需命名为 abs_union2.mlu 。

此外，算子开发者还需要在 mlu_op.h 中添加该算子的对外接口的详细注释；对于可用模板实现的算子，可调用 binary_op 和 unary_op 文件中函数进行实现；对于通用函数可以在 /kernels/utils/common.h 中查找调用。

### 3.2 测试代码开发

算子贡献者在完成算子开发任务后，需要添加 GTest 测试。具体添加方式及注意事项如下：

#### 3.2.1 添加 GTest

GTest 测试例的添加原则为能够测试到该算子的各种应用场景，包括：

- 算子输入输出支持的各种数据类型

- 算子输入输出支持的各种规模

- 算子输入输出支持的各种 layout

- 算子在框架端网络中用到的规模及数据类型

- 必要的边界测试

添加 GTest 的流程大体分为:

1. 在 [mlu_op_test.proto](https://github.com/Cambricon/mlu-ops-proto/blob/master/mlu_op_test.proto) 文件中增加算子信息，算子没有额外参数可以不添加
2. 增加测试代码
3. 手写测例用于测试

详细可参考[GTest-User-Guide-zh.md](../GTest-User-Guide-zh.md)。

`注意`: cpuCompute()函数中需要统计算子的理论计算量 theory_ops，具体添加方法可以参考库内的已有算子。

#### 3.2.2 添加测试用例

代码开发完成后，添加测试文件，格式可以参考现有算子，根据 proto 中定义的数据结构，注意格式写好即可。

  - prototxt 文件可以自己手动添加，也可以通过工具批量生成
  - pb 文件为序列化后的测试例文件，可以通过工具批量生成
  - MLU-OPS™ GTest 支持解析 prototxt 以及 pb 两种文件类型 

## 4 算子测试

详见：[MLU-OPS™ 测试报告模板](./MLU-OPS-Test-Report-Template.md)

### 4.1 测试覆盖率

MLU-OPS™ coverage test 是面向 bangc 语言的代码覆盖率测试工具。
关于 MLU-OPS™ coverage test 的使用方法见[GTest-User-Guide-zh](../GTest-User-Guide-zh.md)。

### 4.2 算子验收标准

#### 4.2.1 精度验收标准

kernels 下的 bangc 算子实现需要与 GTest 中的 cpuCompute()实现作为 baseline 进行精度对比验证，具体精度标准见 [MLU-OPS™ 精度验收标准](../MLU-OPS-Accuracy-Acceptance-Standard.md)。

#### 4.2.2 性能验收标准

见 [MLU-OPS™ 性能验收标准](../MLU-OPS-Performance-Acceptance-Standard.md)。

## 5 代码提交

本地编译测试通过后可以执行以下操作提交修改的代码到远程分支。更多 `PR` 操作见[Pull-Request.md](https://github.com/Cambricon/mlu-ops/blob/master/docs/Pull-Request.md)。

```bash
1. source env.sh                   \\ 使能 pre-commit ，在 commit 阶段触发代码格式检查。
2. git add FileName                \\ 将所有修改的文件添加到 git 暂存区。
3. git commit -m "**"              \\ 将添加到暂存区的修改提交。
4. git pull origin master -r       \\ rebase master, 确保自己的分支领先于最新 master 分支。
5. git push origin your_branch     \\ 将本地分支推到远程, 若子仓mlu_op_test.proto有修改，先在子仓按照2～5步提交PR。
```
