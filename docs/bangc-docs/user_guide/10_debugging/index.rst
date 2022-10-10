.. _调试方法:

调试方法
========

本章介绍Cambricon BANGC OPS算子的调试方法。


.. _算子调试信息的保存:

算子调试信息的保存
--------------------------------

Cambricon BANGC OPS提供了一个用于辅助算子调试的工具，开启功能后，该工具会将Cambricon BANGC OPS API接受到的部分输入参数信息保存起来，方便用户更快捷地重现问题。启用本功能将降低算子性能，因此仅能用于调试。

保存的调试信息主要包括：

- 输入输出张量描述符相关信息：

  -  规模（ ``dims`` ）；
  -  数据排布（ ``mluOpLayout_t`` ）；
  -  数据类型（ ``mluOpDataType_t`` ）；
  -  device上的数据（配合其他变量使用）；

- 算子自身的名字 （``op_name``） ，算子类别 （``op_type``）；
- 算子运行所需参数。

算子接受到的参数将会以Cambricon BANGC OPS内部定义的 ``protobuf`` 格式保存到当前路径下的 ``gen_case`` 文件夹内。

开启方法
>>>>>>>>>>>>>
该工具可以通过设置环境变量 ``export MLUOP_GEN_CASE=k`` 或者调用函数 ``mluOpSetGenCaseMode(k)`` （其中k为整数）来切换工具的功能等级，具体为：

- 当k为1时：开启算子调试信息保存功能。在调用任意Cambricon BANGC OPS算子接口后，会在当前目录生成名为 ``gen_case`` 的文件夹，里面包含算子的调试信息文件。调试信息文件中会记录算子本次被调用时的输入输出形状以及算子参数等信息。

- 当k为2时：开启算子调试信息保存功能。在算子被调用时生成调试信息文件，调试信息文件会记录算子本次被调用时的输入输出形状、算子参数等信息以及算子输入输出数据。此功能需要配合环境变量MLUOP_GEN_CASE_DUMP_DATA一起使用，MLUOP_GEN_CASE_DUMP_DATA等于1时，会在调试文件中以文本形式保存数据，等于2时，会以二进制形式保存浮点数据，MLUOP_GEN_CASE_DUMP_DATA_OUTPUT记录输出数据，用法类似。

- 当k为3时：开启算子调试信息保存功能。在算子被调用时，将算子的调试信息打印在屏幕上，不会保存成文件。调试信息包括算子本次被调用时的输入输出形状以及算子参数等信息。

- 当k为其他值时：关闭算子调试信息保存功能。

有关函数 ``mluOpSetGenCaseMode(int mode)`` 详情，可参考《Cambricon BANGC OPS Developer Guide》。

Cambricon BANGC OPS会自动创建路径并保存文件，并打印log信息提示调试信息已成功保存至上述路径。以Abs算子为例，设置 ``export MLUOP_GEN_CASE=1`` 后，log信息打印如下：

::

  [2022-9-28 15:55:42] [MLUOP] [Info]:[gen_case] Generate gen_case/abs/abs_20220928_07_55_42_801549_tid44019_device1.prototxt

调试信息文件会输出至 ``gen_case`` 文件夹下的 ``abs`` 算子文件夹，文件名格式为”算子名_日期_时_分_秒_随机数_tid线程ID.prototxt”。

调试信息文件
>>>>>>>>>>>>>>>

文件内容示例（片段）如下：

::

  op_name: "abs"  // 算子名
  op_type: ABS    // 算子类别
  input {         // 输入tensor的相关信息
    id: "x"       // 输入tensor的名字
    shape: {      // 输入tensor的规模
      dims: 10
      dims: 10
      dims: 10
      dims: 10
    }
    layout: LAYOUT_ARRAY  // 输入tensor的数据排布
    dtype: DTYPE_FLOAT    // 输入tensor的数据类型
  }
  output {        // 输出tensor的相关信息，格式与输入一致，略。
    /*...*/
  }

.. _`MLU Unfinished问题定位`:

mlu unfinished问题定位
--------------------------------

当出现mlu unfinished时, 可以参考本节描述的方法进行定位, 并在github上提issue来获取帮助。

具体过程如下（以CNToolkit-2.3.1为例。随CNToolkit的更新迭代，不同版本的CNToolkit在细节上可能存在差异，但基本流程一致）：

当出现 ``mlu unfinished`` 时，将会有下述信息打印在屏幕上:

::

  2021-04-20 17:30:05.370267: [cnrtError] [11621] [Card : 0] cnrtQueueSync: MLU queue sync failed.

1. 当出现mlu unfinished时，Cambricon BANGC OPS会自动保存名为 ``core_***.cndump`` 的文件。保存路径为当前调用Cambricon BANGC OPS API的可执行文件的同级路径。该文件是二进制文件，可以用CNToolkit工具链中的CNGDB工具进行解析。CNGDB的具体使用方式，请参考《寒武纪CNGDB用户手册》。从解析结果中，能获取到以下信息：

   - ``Device name`` 出现异常时硬件的型号信息。

   - ``MLU Kernel name`` 出现异常的 ``kernel name`` 信息。 ``kernel name`` 是在MLU上运行的函数。用户可以在github上提issue，并在issue中描述异常 ``kernel name`` 的信息。

   - 出现异常时硬件的状态信息， ``exception`` 代表异常的类型。例如 ``barrier sync timeout`` ，说明是硬件同步出现了问题。

#. 用户可单独运行上述过程得到的Cambricon BANGC OPS算子，以确认算子是否存在问题。

#. 如果单算子可以复现问题，设置环境变量 ``MLUOP_GEN_CASE`` ，或者在调用此算子前先调用 ``mluOpSetGenCaseMode`` 函数，然后重新运行该算子（此过程可以保存包含算子调试信息的 ``*.prototxt`` 文件）。最终将出现异常时的 ``core_***.cndump`` 文件、包含 ``*.prototxt`` 文件的 ``gen_case`` 文件夹，以及出现问题时操作系统的dmesg信息提交到github，我们会尽快修复问题。关于如何保存算子的调试信息，详情参看 算子调试信息的保存_。

#. 如果在同参数下验证上述Cambricon BANGC OPS算子没有问题，那导致问题的原因可能是较为底层或其他影响范围更大的特性没有正常工作。如果条件允许（复现问题成本不高），请尽可能缩小复现问题的条件范围，明确问题算子和出现问题的条件，可以帮助Cambricon BANGC OPS更快地定位和解决问题。
