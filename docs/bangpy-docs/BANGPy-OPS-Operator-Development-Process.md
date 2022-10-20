# BANGPy-OPS 算子开发流程

[概述](./BANGPy-OPS-Operator-Development-Process.md#概述)

[需要添加的文件](./BANGPy-OPS-Operator-Development-Process.md#需要添加的文件)

[算子设计文档](./BANGPy-OPS-Operator-Development-Process.md#算子设计文档)

[算子测试报告](./BANGPy-OPS-Operator-Development-Process.md#算子测试报告)

[代码提交流程](./BANGPy-OPS-Operator-Development-Process.md#代码提交流程)

## 概述

BANGPy-Ops 是面向 MLU 平台的神经网络加速库，算子的功能实现可以利用寒武纪特有的 BANG Python 语言实现。

BANGPy-Ops 库中算子的开发过程，文档与算子的代码实现同样重要。文档是 BANGPy-Ops 库中参与者沟通的桥梁，是了解算子功能和接口使用的入口，故算子设计文档的撰写需格式规范、层次清晰、功能完整。当算子的实现代码修改时，文档需同步进行修改，保持文档与代码的一致性。

一个算子需要入库的文件包含：算子设计文档、算子实现代码、算子测试代码、测试报告。

- 算子设计文档：包含算子的需求分析、API 接口设计、算子的实现设计

- 算子实现代码：包含 Python 实现源码、Pytest 测试代码

- 算子测试代码：算子开发者需要编写算子的测试代码，该代码需能够测试到算子的多种使用场景

- 算子测试报告：包含算子测试结果，如算子测试规模和数据类型、算子性能及稳定性等

## 需要添加的文件

本章节会详细说明在整个算子开发和维护过程中需要添加和修改的文件及其所在目录。算子开发者在开发过程中务必按照说明中的目录和文件进行添加和修改，以确保 BANGPy-Ops 库文件和目录结构的简洁。

下面以添加一个加法算子为例说明整个算子开发过程中添加的文件。

### 1. 文档开发阶段

在 docs/bangpy-docs/design_docs/ 目录下新建以算子名命名的目录，目录名首字母小写，算子名的格式只能取 `xxx` 和 `xxx_xxx` 的其中一种，并在算子目录下新建以算子名命名的 md 文件。如：

```bash
$ cd docs/bangpy-docs/design\_docs/
$ mkdir add
$ cd add
$ vim add.md
```

在 add 目录下添加的 add.md 文件，为算子的设计文档，设计文档模板可参考[BANGPy-OPS 算子设计文档模板](./BANGPy-OPS-Operator-Design-Doc-Template.md)。

如果一个算子存在正向和反向，那么正反向算子当做两个不同的算子来处理。如卷积算子存在卷积前向与卷积反向，目录结构应为

```bash
|-- docs
   |-- bangpy-docs
      |-- design_docs
         |-- convolution_forward
            |-- convolution_forward.md
         |-- convolution_backward
            |-- convolution_backward.md
```
文档中如有涉及公式的地方， 使用 md 的公式格式，不能使用图片的形式插入。

### 2. 代码开发阶段

在 bangpy-ops/ops/ 目录下，添加以算子名命名的目录，然后在该目录下添加算子功能的实现文件、测试文件，文件名首字母小写，如：

```bash
$ cd bangpy-ops/ops/
$ mkdir add         // 添加以算子名命名的目录
$ cd add
$ touch add.py      // add.py  ->  BANGPy op的实现文件
$ touch test_add.py // test_add.py  ->  BANGPy op的测试文件
```

文件命名及组织规则为：
1. 算子实现的文件名为算子名， 如 add.py, pool.py等
2. 测试文件以`test_`开头，后接算子名， 如 test_add.py, test_pool.py

算子实现中需要调用注册装饰函数来对不同数据类型和不同device target进行注册：

```python
@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_add(dtype=None, target=None):
    # tasktype fixed in UNION1
    task_num = 4
    f = Add(SHAPE, dtype, target, task_num).compute_body()
    return f
```

### 3. 测试阶段

算子开发者在完成算子开发任务后, 需要添加测试文件，具体添加方式及注意事项如下：
1. 测试采用pytest的标准形式，利用`@pytest.mark.parametrize`装饰器函数来测试覆盖到算子的各自应用场景，包括：

- 算子输入输出支持的各种数据类型

- 算子输入输出支持的各种规模

- 算子输入输出支持的各种 layout

- 算子在框架端网络中用到的规模及数据类型

- 必要的边界测试

2. 测试函数接口必须以`test_`开口，以用于pytest自动识别测例。第一个参数必须是target，用于测试时指定device target。

示例：

```python
@pytest.mark.parametrize(
    "shape", [SHAPE],
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)
def test_add(target, shape, dtype):
```

#### 3.2 算子验收标准

##### 1. 精度验收标准

ops 下的 BANGPy 算子实现需要自己使用numpy接口来实现CPU baseline 进行精度对比验证，具体精度标准建
[MLU-OPS 精度验收标准](../MLU-OPS-Accuracy-Acceptance-Standard.md)。


##### 2. 性能验收标准

见 [MLU-OPS 性能验收标准](../MLU-OPS-Performance-Acceptance-Standard.md)。


## 算子设计文档

算子负责人接到算子开发任务后，必须先进行算子的需求分析，完成算子设计文档。

算子设计文档模板链接：[BANGPy-OPS 算子设计文档模板](./BANGPy-OPS-Operator-Design-Doc-Template.md)

算子设计文档需包含以下 5 个部分：

- 算子需求分析

- 算子实现设计

- 算子性能优化记录

- 方案实施

### 1. 算子需求分析

在算子需求分析阶段，算子开发者要了解算子实现的数学原理，以确保算子开发的准确、高效。在该阶段，算子开发者需要调研框架的算子的功能和需求，例如 TensorFlow 以及 PyTorch 框架。若该算子在多个框架中都有对应实现，需要进行综合调研分析，以确保开发的算子可以同时满足多个框架的需求。

注意，有些算子在两个框架的需求和使用场景不同，比如 Pooling 算子的 add pad 操作，当 padding_width == 3 时，在 TensorFlow 框架下 pad_left == 1, pad_right == 2，而在 PyTorch 框架下， pad_left == 2, pad_right == 1，因此该算子的接口和内部实现需同时满足两个框架的需求。 因此，一定要确保和两个框架都对接完成后，再进行下一步，否则可能会做许多无用功。

需要和框架对接的内容包括但不限于以下几点：

- 算子的输入和输出：该算子在框架层的调用需要有几个输入和几个输出、需要支持哪些数据类型、输入输出的物理布局以及该算子是否需要支持多维等

- 算子的规模限制：算子对于输入输出的规模限制、维度、layout、数据类型信息等

- 算子的性能需求：两个框架对该算子的性能要求

- 算子的接口需求：接口设计需满足框架的使用需求

- 算子是否需要支持原位操作/stride 机制/广播

- 算子对于 0 元素是直接返回还是需要做特殊处理

- 算子是否有其他特殊需求（量化，融合等）

### 2. API 接口设计

在进行算子的接口设计时，需要对框架算子功能做完备的需求分析后进行设计，包括：

- PyTorch
- TensorFlow
- Caffe
- ...

在设计文档中需写明接口设计时所参考的框架接口，并对所设计的接口参数列表进行详细说明，同时需要说明用户在使用接口时需要注意的事项等。


## 算子测试报告

算子开发者除了编写算子的代码实现部分外，还需要编写算子测试代码，测试代码需能够测试到算子使用的多种场景。测试报告至少需要包含：

- 功能测试
- 性能测试
- 稳定性测试

具体见：[MLU-OPS 测试报告模板](../MLU-OPS-Test-Report-Template.md)。

## 代码提交流程

本地编译测试通过后可以执行以下操作提交修改的代码到远程分支。

```bash
1. source env.sh                   \\ 使能 pre-commit ，在 commit 阶段触发代码格式检查。
2. git add FileName                \\ 将所有修改的文件添加到 git 暂存区。
3. git commit                      \\ 将添加到暂存区的修改提交。
4. git pull origin master -r       \\ rebase master, 确保自己的分支领先于最新 master 分支。
5. git push origin your_branch     \\ 将本地分支推到远程。
```
