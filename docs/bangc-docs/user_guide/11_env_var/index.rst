.. _环境变量:

Cambricon BANGC OPS库环境变量
=============================


本章介绍 Cambricon BANGC OPS 库环境变量。

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
###################

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

.. _MLUOP_GEN_CASE:

MLUOP_GEN_CASE
######################

**功能描述**

开启算子调试信息保存功能。开启本功能后，调用算子时，将自动保存算子调试信息。

开启本功能将降低算子性能，只支持整数值，输入不为整数值时取默认值0。

**使用方法**

- export MLUOP_GEN_CASE=1：开启算子调试信息保存功能。在调用任意Cambricon BANGC OPS算子接口后，会在当前目录生成名为 ``gen_case`` 的文件夹，里面包含被调用到的算子的调试信息文件。调试信息文件中会记录算子本次被调用时的输入输出形状以及算子参数等信息。

- export MLUOP_GEN_CASE=2：开启算子调试信息保存功能。在等级1的基础上，生成的算子调试信息文件还会记录算子输入输出数据。export MLUOP_GEN_CASE_DUMP_DATA=1后会以文本形式保存数据，MLUOP_GEN_CASE_DUMP_DATA=2会以二进制形式保存浮点数据。MLUOP_GEN_CASE_DUMP_DATA_OUTPUT用于记录算子输出，用法相同。

- export MLUOP_GEN_CASE=3：开启算子调试信息保存功能。在调用算子时，将算子的调试信息打印在屏幕上，不会保存成文件。

- export MLUOP_GEN_CASE=0 或 unset MLUOP_GEN_CASE：关闭算子调试信息保存功能。

默认值为0。

