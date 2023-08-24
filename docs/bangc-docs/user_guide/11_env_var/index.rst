.. _环境变量:

Cambricon BANG C OPS库环境变量
================================


本章介绍 Cambricon BANG C OPS 库环境变量。

.. _MLUOP_BUILD_GTEST:
 
MLUOP_BUILD_GTEST
######################

**功能描述**

编译MLU-OPS的GTEST。

**使用方法**

- export MLUOP_BUILD_GTEST=ON。

在build脚本中默认设为ON。

.. _MLUOP_GTEST_DUMP_DATA:
 
MLUOP_GTEST_DUMP_DATA
######################

**功能描述**

将MLU-OPS的GTEST的输入输出数据打印至文件中。

**使用方法**

- export MLUOP_GTEST_DUMP_DATA=ON： 保存 GTEST 测试过程中用到的输入输出数据。

不使用此环境变量时需要unset环境变量。

.. _MLUOP_GEN_CASE:
 
MLUOP_GEN_CASE 
######################

**功能描述**

运行前设置，设置gen_case工具功能等级。

**使用方法**

- export MLUOP_GEN_CASE=0：关闭 gen_case 模块功能。
- export MLUOP_GEN_CASE=1：生成 prototxt，输入输出只保留 shape 等信息。
- export MLUOP_GEN_CASE=2：生成 prototxt，并保留输入真实真。
- export MLUOP_GEN_CASE=3：不生成 prototxt，只在屏幕上打印输入输出的 shape 等信息。

更详细请参见 `MLU-OPS GEN_CASE 使用指南<https://github.com/Cambricon/mlu-ops/blob/master/docs/Gencase-User-Guide-zh.md>`_ 。

.. _MLUOP_MIN_VLOG_LEVEL:
 
MLUOP_MIN_VLOG_LEVEL
######################

**功能描述**

设置外部LOG()宏的最小打印级别，屏蔽不需要级别的日志信息。

**使用方法**

- export MLUOP_MIN_VLOG_LEVEL=0：显示FATAL、ERROR、WARNING、INFO级别的日志。

- export MLUOP_MIN_VLOG_LEVEL=1：显示FATAL、ERROR、WARNING级别的日志。

- export MLUOP_MIN_VLOG_LEVEL=2：显示ERROR和WARNING级别的日志。

- export MLUOP_MIN_VLOG_LEVEL=3：显示FATAL级别LOG。

默认值为0。

.. _MLUOP_LOG_ONLY_SHOW:

MLUOP_LOG_ONLY_SHOW
####################

**功能描述**

设置将日志信息打印到屏幕上和生成日志文件。

**使用方法**

- export MLUOP_LOG_ONLY_SHOW=ON：只打印日志信息到屏幕上而不会生成日志文件。

- export MLUOP_LOG_ONLY_SHOW=OFF：打印日志信息到屏幕上并且生成日志文件。日志文件名为 ``mluop_auto_log`` ，默认保存在程序运行目录下。不支持更改文件存放路径。

默认值为ON。

.. _MLUOP_LOG_COLOR_PRINT:

MLUOP_LOG_COLOR_PRINT
######################

**功能描述**

设置打印日志时是否开启颜色字体特效。该环境变量仅用于打印日志信息到屏幕。生产的日志文件不会有颜色字体特性。
**使用方法**

- export MLUOP_LOG_COLOR_PRINT=ON：打印日志时开启颜色字体特效。

- export MLUOP_LOG_COLOR_PRINT=OFF：打印日志时不开启颜色字体特效。

默认值为ON。


.. _MLUOP_BUILD_ASAN_CHECK:
 
MLUOP_BUILD_ASAN_CHECK
#######################

**功能描述**

在编译的时候设置是否打开ASAN内存检查。

**使用方法**

- export MLUOP_BUILD_ASAN_CHECK=ON： 表示编译ASAN内存检查。

默认不开启。该工具仅在Ubuntu上与Debian上有效。无论环境变量如何设置，Centos上都不会编译该工具。如果没有检测到内存问题，运行算子case时将不会输出任何内容; 若检测到内存问题，运行算子case时将输出错误内容。

.. _MLUOP_SET_JOB_LIMIT_CAPABILITY:

MLUOP_SET_JOB_LIMIT_CAPABILITY
################################

**功能描述**

设置最大JOB限制数量，默认不设置。

**使用方法**

- export MLUOP_SET_JOB_LIMIT_CAPABILITY=1：CN_KERNEL_CLASS_UNION。
- export MLUOP_SET_JOB_LIMIT_CAPABILITY=2：CN_KERNEL_CLASS_UNION2。
- export MLUOP_SET_JOB_LIMIT_CAPABILITY=3：CN_KERNEL_CLASS_UNION4。
- export MLUOP_SET_JOB_LIMIT_CAPABILITY=4：CN_KERNEL_CLASS_UNION8。
- export MLUOP_SET_JOB_LIMIT_CAPABILITY=5：CN_KERNEL_CLASS_UNION16。
- export MLUOP_SET_JOB_LIMIT_CAPABILITY=6：CN_KERNEL_CLASS_BLOCK不使用。
- export MLUOP_SET_JOB_LIMIT_CAPABILITY=7：CN_KERNEL_CLASS_NONE不使用。

JOB_LIMIT和CLUSTER_LIMIT需要同时设置来保证合法性。

.. _MLUOP_GTEST_CLUSTER_LIMIT_CAPABILITY:

MLUOP_GTEST_CLUSTER_LIMIT_CAPABILITY
######################################

**功能描述**

设置最大JOB限制数量，默认不设置。

**使用方法**

- export MLUOP_GTEST_CLUSTER_LIMIT_CAPABILITY=1：1cluster。
- export MLUOP_GTEST_CLUSTER_LIMIT_CAPABILITY=3：2cluster。
- export MLUOP_GTEST_CLUSTER_LIMIT_CAPABILITY=7：3cluster。
- export MLUOP_GTEST_CLUSTER_LIMIT_CAPABILITY=15：4cluster。
- export MLUOP_GTEST_CLUSTER_LIMIT_CAPABILITY=...：从右往左，每多一个连续的1表示1个cluster。

JOB_LIMIT 和CLUSTER_LIMIT 需要同时设置来保证合法性。原理是：1的二进制是0000,0001: 1号cluster可用; 3的二进制是0000,0011: 1号和2好cluster可用; 如果有特殊需求，如只想用2号cluster:设置为2: 0000,0010。

.. _MLUOP_GTEST_SET_GDRAM:

MLUOP_GTEST_SET_GDRAM
#######################

**功能描述**

作用是在GDRAM前后刷NAN/INF。

**使用方法**

- export MLUOP_GTEST_SET_GDRAM=NAN：在GDRAM前后刷NAN。
- export MLUOP_GTEST_SET_GDRAM=INF：在GDRAM前后刷INF。

若不设置则根据日期，偶数天刷NAN，奇数天刷INF。

.. _MLUOP_GTEST_UNALIGNED_ADDRESS_RANDOM:

MLUOP_GTEST_UNALIGNED_ADDRESS_RANDOM
#####################################

**功能描述**

设置在gdram上申请的空间地址是非64 bytes对齐的，偏移量为1~63的随机值。

**使用方法**

- export MLUOP_GTEST_UNALIGNED_ADDRESS_RANDOM=ON。
- export MLUOP_GTEST_UNALIGNED_ADDRESS_RANDOM=OFF。

.. _MLUOP_GTEST_UNALIGNED_ADDRESS_SET:

MLUOP_GTEST_UNALIGNED_ADDRESS_SET
#####################################

**功能描述**

设置在gdram上申请的空间地址是64 bytes对齐的。

**使用方法**

- export MLUOP_GTEST_UNALIGNED_ADDRESS_SET=NUM。

