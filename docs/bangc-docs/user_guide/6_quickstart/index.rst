快速入门
=================

本章重点介绍了如何部署和使用Cambricon BANGC OPS，以及如何运行寒武纪提供的BANGC OPS样例。

部署Cambricon BANGC OPS
------------------------

Cambricon BANGC OPS库的使用依赖于CNToolkit工具包。用户在使用Cambricon BANGC OPS库之前需要先安装寒武纪CNToolkit工具包。安装详细步骤，请参考《寒武纪CNToolkit安装升级使用手册》。

安装完成CNToolkit工具包后，执行以下命令部署Cambricon BANGC OPS。默认头文件mlu_op.h会安装在 ``/usr/local/neuware/include`` 目录下，链接库文件libmluops.so会安装在 ``/usr/local/neuware/lib64`` 目录下，示例程序会安装在 ``/usr/local/neuware/samples/bangc-ops`` 目录下。


Ubuntu系统
>>>>>>>>>>

- 执行安装命令：

  ::

    sudo apt install ./mluops<x.y.z>-1.ubuntu<a.b>_<arch>.deb

  <x.y.z>为Cambricon BANGC OPS版本号，<a.b>为操作系统版本号，<arch>为CPU架构（AMD64）。例如Cambricon BANGC OPS v0.3.0版本在x86_64的Ubuntu18.04系统下的包名为 ``mluops_0.3.0-1.ubuntu18.04_amd64.deb`` 。


CentOS系统
>>>>>>>>>>

- 执行安装命令：

::

    sudo yum install mluops-<x.y.z>-1.el7.<arch>.rpm

<x.y.z>为Cambricon BANGC OPS版本号，<arch>为CPU架构（x86_64），例如Cambricon BANGC OPS v0.3.0版本在 CentOS7 系统下的包名为 ``mluops-0.3.0-1.el7.x86_64.rpm``。


Debian系统
>>>>>>>>>>

- 执行安装命令：

  ::

    sudo apt install mluops-<x.y.z>-1.debian10.<arch>.deb

  <x.y.z>为Cambricon BANGC OPS版本号，<arch>为CPU架构（AMD64），例如Cambricon BANGC OPS v0.3.0版本在 debian10 系统下的包名为 ``mluops_0.3.0-1.debian10_amd64.deb``。


Kylin系统
>>>>>>>>>>

- 执行安装命令：

  ::

    sudo yum install mluops-<x.y.z>-1.ky10.<arch>.rpm

  <x.y.z>为Cambricon BANGC OPS版本号，<arch>为CPU架构（AArch64），例如Cambricon BANGC OPS v0.3.0版本在 KylinV10 系统下的包名为 ``mluops-0.3.0-1.ky10.aarch64.rpm``。


.. _卸载或升降级BANGC_OPS版本:

卸载或升降级Cambricon BANGC OPS版本
------------------------------------

卸载Cambricon BANGC OPS
>>>>>>>>>>>>>>>>>>>>>>>>>

卸载Cambricon BANGC OPS，可使用 ``.deb`` 包和 ``.rpm`` 包的操作方式，详细说明，请参见《寒武纪CNToolkit安装升级使用手册》。

升级Cambricon BANGC OPS版本
>>>>>>>>>>>>>>>>>>>>>>>>>>>>

由于Cambricon BANGC OPS依赖CNToolkit提供的CNDrv、CNRT、CNBin包，在处理CNToolkit的版本更替时，如果遇到依赖冲突报错，可以先卸载Cambricon BANGC OPS，更新完CNToolkit后，再更新Cambricon BANGC OPS（即可以尝试先全部卸载，再重新安装的方式完成CNToolkit和Cambricon BANGC OPS的版本升降级）。

- ``.deb`` 包升级示例

  使用 ``apt`` 命令：

  ::

    apt install ./cntoolkit_<x.y.z>-1.<distro><id>_<arch>.deb  # 先更新CNToolkit包
    apt update  # 更新CNToolkit本地源
    apt install cndrv cnrt cnbin ./mluops_<x.y.z>-1.<distro><id>_<arch>.deb # 更新源里的cndrv、cnrt、cnbin和本地的mluops deb包

- ``.rpm`` 包升级示例

  使用 ``yum`` 命令：

  ::

    yum update cntoolkit-<x.y.z>-1.el<id>.<arch>.rpm  # 先更新CNToolkit包
    yum clean metadata && yum makecache  # 重建yum包管理数据库信息
    yum update cndrv cnrt cnbin mluops-<x.y.z>-1.el<id>.<arch>.rpm  # 更新cndrv、cnrt、cnbin、mluops


降级Cambricon BANGC OPS版本
>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Cambricon BANGC OPS的降级推荐直接卸载当前版本，再进行安装。

Cambricon BANGC OPS编程和运行
-----------------------------

了解如何使用Cambricon BANGC OPS构造一个算子或者搭建一个简单的网络，请参考 :ref:`programming_guide` 。

Cambricon BANGC OPS开发样例
----------------------------

Cambricon BANGC OPS开发样例为用户提供了abs、polyNms算子运算的样例代码，帮助用户快速体验如何使用Cambricon BANGC OPS来开发、编译以及运行一个算子。用户可以直接通过脚本运行样例代码，无需修改任何配置。

开发样例目录结构与使用方法以实际发布的samples/README.md中描述为准，以下描述样例执行步骤：

1. 设置环境变量。

    a. 确认CNToolkit和Cambricon BANGC OPS完成安装，安装目录的include子目录下包含 ``mlu_op.h`` 头文件，lib64目录下包含 ``libmluops.so`` 、 ``libcnrt.so`` 、 ``libcndrv.so`` 、 ``libcnbin.so`` 。
    b. 设置 ``NEUWARE_HOME`` 环境变量指向安装目录，如 ``export NEUWARE_HOME=/usr/local/neuware`` 。
    c. 在 ``samples/bangc-ops/abs_sample`` 和 ``samples/bangc-ops/poly_nms_sample`` 下执行 ``source env.sh`` ，自动设置 ``PATH`` 、 ``LD_LIBRARY_PATH`` 。

2. 编译并运行开发样例。

  - 编译全部样例

    a. 在 ``samples/bangc-ops/`` 目录下运行下面命令：

      ::

        source env.sh
        ./build.sh

      在 ``samples/bangc-ops/build/bin`` 目录下生成可执行文件 ``abs_sample`` 和 ``poly_nms_sample`` 。

    b. 在 ``samples/bangc-ops/build/bin`` 目录下运行样例：

      ::

        ./abs_sample [dims_vaule] [shape0] [shape1] [shape2] ...  # 运行 abs_sample 样例
        ./poly_nms_sample  # 运行 poly_nms_sample 样例

  - 编译 abs_sample 样例

    a. 在 ``samples/bangc-ops/abs_sample`` 目录下运行下面命令：

      ::

        source env.sh
        ./build.sh

      在 ``samples/bangc-ops/abs_sample/build/bin`` 目录下生成可执行文件 ``abs_sample`` 。

    b. 在 ``samples/bangc-ops/abs_sample/build/bin`` 目录下运行样例：

      ::

        ./abs_sample [dims_vaule] [shape0] [shape1] [shape2] ...  # 运行 abs_sample 样例

      e.g.

      ::

        ./abs_sample 4 10 10 10 10

  - 编译 poly_nms_sample 样例

    a. 在 ``samples/bangc-ops/poly_nms_sample`` 目录下运行下面命令：

      ::

        source env.sh
        ./build.sh

      在 ``samples/bangc-ops/poly_nms_sample/build/bin`` 目录下生成可执行文件 ``poly_nms_sample`` 。

    b. 在 ``samples/bangc-ops/poly_nms_sample/build/bin`` 目录下运行样例：

      ::

        ./poly_nms_sample
