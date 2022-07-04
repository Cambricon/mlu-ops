# MLU-OPS 性能验收标准

### 1. 计算效率与 IO 效率的概念

所有算子，粗略可以分成两种类型：纯 IO 类算子，计算类算子。

**纯 IO 类算子**，不涉及运算，需要计算理论 IO 量，与实际运行时间对应的 IO 量进行对比，分子与分母均为 IO 量，得到 IO 效率。

```
io_efficiency = theory_io_size / (latency * IO_BANDWIDTH)
```

`io_efficiency` 是评价算子 IO 效率的指标，越接近 1 代表算子 IO 效率越高。

其中，`theory_io_size` 表示从算法上需要访问的数据量（与实现无关），按照算子 load 一次所有输入，store 一次所有输出的 IO 字节数来计算，不考虑 workspace。以两个数 concat 为例，需要 load AB 两个输入， store C 一个输出，则其 IO 量为 sizeof(A)+sizeof(B)+sizeof(C)。

`lantency` 表示算子运行的实际硬件时间

`IO_BANDWIDTH` 为硬件的理论带宽，以 270 为例， IO_BANDWIDTH = 102.4 GB/s，对于只启动 1 个 job 或者其余没有占满带宽的，也以 102.4 GB/s 计算

**计算类算子**，包括了 IO 操作和运算过程，需要计算理论计算量，与实际运行时间对应的峰值算力进行对比，分子与分母均为计算 OP 量，得到计算效率。

```
compute_efficiency = theory_compute_ops / (latency * peak_compute_force)
```

`compute_efficiency` 是评价算子计算效率的指标，越接近 1 代表算子计算效率越高。

其中 `theory_compute_ops` 表示该算子从算法上需要执行多少次操作（与实现无关）。以向量 add 算子为例（A+B = C），从算法上，需要执行 countof(A)这么多次加法，因此其理论计算量就是 countof(A)。

`theory_compute_ops` 可以类比算法复杂度，只不过我们给出的不是量级，是确数。

`lantency` 表示算子运行的实际时间单位为秒（s）。

`peak_compute_force` 为硬件平台的峰值算力，其单位是 op/s（每秒执行多少次操作）。不同平台不同数据类型算力不同。

### 2. 计算效率与 IO 效率的标准

对于 IO 类算子，在适当规模 case 下，io_efficiency 应达到 50%以上。

对于计算类算子，在适当规模 case 下，compute_efficiency 应达到 50%以上。

该数据需要在测试报告中给出，若无法达标，需要给出分析说明。

### 3. GTest 实现

计算 IO 效率和计算效率时的分母 IO_BANDWIDTH 和 PEAK_COMPUTE_FORCE，与硬件平台和计算数据类型有关。

封装成公共函数，计算时得到硬件平台的版本号，根据计算数据类型通过查找表直接查找，确保支持之后的板卡和平台。

算子开发者需要在 GTest 的`XXXExecutor::getTheoryOps()` 函数中给出算子理论计算量的公式，GTest 会自动根据算子规模计算理论 IO 量，并根据测试规模计算理论 IO 量和理论计算量。
