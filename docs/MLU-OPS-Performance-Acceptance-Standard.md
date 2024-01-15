# MLU-OPS 性能验收标准

### 1. 性能标准相关概念

**计算效率与 IO 效率**

为了评估算子实际运行时的状态，引入计算效率与 IO 效率作为参考标准。

**计算效率( $compute \\_ efficiency$ )**，表征算子在实际运行中对设备上算力的利用效率。其计算方式如下：

$$
compute \\_ efficiency = \frac{theory \\_ compute \\_ ops}{latency \times PEAK \\_ COMPUTE \\_ FORCE}
$$

其中 $theory \\_ compute \\_ ops$ 表示该算子从算法上需要执行多少次操作（与实现无关）。

以向量 add 算子为例（ $A+B=C$ ）。假设两个长度为 $N$ 的 Tensor 进行 add 操作。 从算法上，需要执行 $N$ 次加法，因此其理论计算量为 Tensor 的长度 $N$ 。

$lantency$ 表示算子运行的实际时间，单位为秒（s）。

$PEAK \\_ COMPUTE \\_ FORCE$ 为硬件设备的峰值算力，其单位是 op/s（Operations per second ，每秒执行多少次操作）。

峰值算力受以下因素影响：

- 设备类型
- 数据类型

一般而言，算子的计算效率受以下因素影响：

- 算子开发时设计算法中引入额外的计算量。
- 算子开发时设计的计算 pattern。
- 算子运行时的设备类型。

**IO 效率( $io \\_ efficiency$ )**，表征算子在实际运行中对设备 IO 带宽的利用效率。其计算方式如下：

$$
io \\_ efficiency = \frac{theory \\_ io \\_ size}{latency \times IO \\_ BANDWIDTH}
$$

其中， $theory \\_ io \\_ size$ 表示从算法上需要访问的数据量（与实现无关）。

以向量 add 算子为例（ $A+B=C$ ）。假设两个长度为 $N$ 的 Tensor 进行 add 操作。 从算法上，需要将 Tensor $A、B$ 加载到设备上，同时在获得结果 $C$ 后，需将结果保存到设备外存上。因此该场景下的 $theory \\_ io \\_ size$ 为 $A、B、C$ Tensor 的数据总量。

$lantency$ 表示算子运行的实际时间，单位为秒（s）。

$IO \\_ BANDWIDTH$ 为硬件的理论带宽，以 270 为例， $IO \\_ BANDWIDTH$ = 102.4 GB/s，对于只启动 1 个 job 或者其余没有占满带宽的，也以 102.4 GB/s 计算。

理论带宽受以下因素影响：

- 设备类型

一般而言，算子的 IO 效率受以下因素影响：

- 算子开发时算法中引入的额外 IO 量，如数据在 workspace 暂存。
- 算子开发时算法中 IO pattern 。
- 算子运行时的设备类型。

**效率瓶颈**，反映算子算法中 IO 操作和计算操作对测试规模运行时间的影响。

一般来说，根据 io_efficiency 和 compute_efficiency 的比较结果来确定当前规模属于 `IO 瓶颈` 还是 `计算瓶颈` 。

$$
\begin{cases}
io \\_ efficiency > compute \\_ efficiency, \space IO瓶颈 \\
io \\_ efficiency \leq compute \\_ efficiency, \space 计算瓶颈
\end{cases}
$$

`IO 瓶颈` 下，优化上更倾向优化 IO 相关操作；`计算瓶颈` 下，优化上更倾向优化计算相关操作。

合理地解读 `IO 瓶颈` 与 `计算瓶颈`，能够为算子优化提供方向。

注意： 针对不同型号的计算设备，同一个测试规模 `IO 瓶颈` 与 `计算瓶颈` 会发生变化，这是由计算设备的带宽与算力共同决定的。因此，不同计算设备的优化方向往往有所区别。

**纯 IO 类算子**，数据在设备上不涉及运算。该类算子开发者应当关注其 IO 效率。

**计算类算子**，数据加载到设备上后进行运算。该类算子开发者应当关注起 IO 效率与计算效率，确认测试规模的 `效率瓶颈` 类型，并专注优化瓶颈侧性能。

### 2. 效率评判标准

**绝对标准**

在 MLU-OPS 仓库中，计算效率与 IO 效率遵循以下规则：

*纯 IO 类算子*：在常见网络规模 case 下， $io \\_ efficiency$ 应达到 60% 以上。对于一般规模， $io \\_ efficiency$ 应达到 30%以上。若未达到对应标准，应在 Pull Request 中说明原因。

*计算类算子*: 对于计算类算子，需分析测试规模所归属的瓶颈类型。

在常见网络规模 case 下，对应瓶颈侧的 $efficiency$ 应达到 60% 以上。对于一般规模，对应瓶颈侧的 $efficiency$ 应达到 30% 以上。若未达到对应标准，应在 Pull Request 中说明原因。

**相对标准**

效率评判标准构建的目的在于，帮助开发者确定算子性能与理想状态下性能的差距。对于网络模型，用户更关心算子在具体网络中实际运行的时间，因此 `运行时间` 可以作为评判算子性能相对优劣的另一标准。一般而言，用算子在近似规格的其他计算设备下的运行时间作为 $baseline$ ，与算子在 MLU 上的运行时间作比较，从而得出算子的相对性能。

### 3. GTest 实现

GTest测试框架依赖 `getTheoryIoSize` 以及 `getTheoryOps` 计算 IO 效率与计算效率。

**getTheoryIoSize**

GTest测试框架中默认实现 `Executor::getTheoryIoSize` 接口。该接口预设 IO 的数据量为输入 Tensor 的总数据量加上输出 Tensor 的总数据量，如有特殊需求，请在继承的 `XXXExecutor` 类中重写 `XXXExecutor::getTheoryIoSize` 。相关函数原型见 [test/mlu_op_gtest/pb_gtest/include/executor.h](https://github.com/Cambricon/mlu-ops/blob/master/test/mlu_op_gtest/pb_gtest/include/executor.h)。算子理论 IO 量概念、 IO 效率概念及 IO 效率公式请参考本文档第一节。

**getTheoryOps**

GTest测试框架中未实现 `Executor::getTheoryOps` 接口。该接口需由开发者在继承的 `XXXExecutor` 类中实现`XXXExecutor::getTheoryOps` ，返回算子理论计算量数值。相关函数原型见 [test/mlu_op_gtest/pb_gtest/include/executor.h](https://github.com/Cambricon/mlu-ops/blob/master/test/mlu_op_gtest/pb_gtest/include/executor.h)。算子理论计算量概念、计算效率概念及计算效率公式请参考本文档第一节。

### 4. mlu-only 模式

仓库提供 `mlu-only` 模式以加速性能测试。该模式下，测试框架调用 mluOp 接口，跳过算子的 cpu 计算以及结果的 diff 比对。

**注意** ：该模式下测试数据随机数据。对于**依赖真实值**的算子，不应当在该模式下测试性能数据，请在 [test/mlu_op_gtest/pb_gtest/gtest_config/test_list](https://github.com/Cambricon/mlu-ops/blob/master/test/mlu_op_gtest/pb_gtest/gtest_config/test_list) 中添加算子名可屏蔽该模式的影响。添加后即使传如 `--mlu_only` 现象，GTest将默认忽略该选项。
