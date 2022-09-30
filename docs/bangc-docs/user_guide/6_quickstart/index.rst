快速入门
=================

本章重点介绍了如何部署和使用Cambricon BANGC OPS，以及如何运行寒武纪提供的BANGC OPS样例。

部署Cambricon BANGC OPS
------------------------

Cambricon BANGC OPS的使用依赖于CNToolkit工具包。用户在使用Cambricon BANGC OPS库之前需要先安装寒武纪CNToolkit工具包。安装详细步骤，请参考《寒武纪CNToolkit安装升级使用手册》。

安装完成CNToolkit工具包后，执行下面命令部署Cambricon BANGC OPS。默认头文件（mlu_op.h）会安装在 ``/usr/local/neuware/include`` 目录下，链接库文件（libmluops.so）会安装在 ``/usr/local/neuware/lib64`` 目录下，示例程序会安装在 ``/usr/local/neuware/samples/bangc-ops`` 目录下。


Ubuntu系统
>>>>>>>>>>

- 执行安装命令：

  ::

    sudo apt install ./mluops<x.y.z>-1.ubuntu<a.b>_<arch>.deb

  <x.y.z>为Cambricon BANGC OPS版本号，<a.b>为操作系统版本号，<arch>为CPU架构（amd64或arm64）。例如Cambricon BANGC OPS v0.2.0版本在X86_64的Ubuntu18.04系统下的包名为 ``mluops_0.2.0-1.ubuntu18.04_amd64.deb``。


CentOS系统
>>>>>>>>>>

- 执行安装命令：

::

   sudo yum install mluops-<x.y.z>-1.el7.<arch>.rpm

   <x.y.z>为Cambricon BANGC OPS版本号，<arch>为CPU架构，例如Cambricon BANGC OPS v0.2.0版本在CentOS系统下的包名为 ``mluops-0.2.0-1.el7.x86_64.rpm   ` 。


Debian系统
>>>>>>>>>>

- 执行安装命令：

::

   sudo yum install mluops-<x.y.z>-1.ky10.<arch>.rpm
   
   <x.y.z>为Cambricon BANGC OPS版本号，<arch>为CPU架构，例如Cambricon BANGC OPS v0.2.0版本在Kylin系统下的包名为 ``mluops_0.2.0-1.debian10_amd64.deb`` 。
   
.. _寒武纪边缘端嵌入式Linux 系统:

寒武纪边缘端嵌入式Linux系统
>>>>>>>>>>>>>>>>>>>>>>>>>>>

寒武纪边缘端嵌入式Linux系统基于ARM64，不具备包管理工具，因此需要通过解压的方式进行安装Cambricon BANGC OPS，或者先在中心主机端系统下进行部署，然后拷贝到边缘端。

同时，边缘端系统不具备直接编译、生成应用的能力，如果要在端侧运行Cambricon BANGC OPS的示例程序或者二次开发，需要在ARM64主机端直接编译或者x86主机端交叉编译，再将生成的应用拷贝到边缘端。

对于ARM64架构的边缘端设备，Cambricon BANGC OPS发行基于Ubuntu 16.04环境生成的DEB包、基于CentOS 7环境生成的RPM包以及自解压安装包，并提供两种部署方式：


- 边缘端系统下运行安装包，自动解压到目标目录下：

  ::

    bash mluops_<x.y.z>-<build>.ubuntu16.04_arm64.run --target <target_directory>

  <x.y.z>为Cambricon BANGC OPS版本号，<build>为打包版本，一般为1，<target_directory>为目标目录，会在该目录下安装 ``lib64`` 、 ``include`` 等目录，如果不指定 ``--target`` 参数，默认安装到 ``/usr/local/neuware`` 下。

  注意， ``mluops_<x.y.z>-<build>.ubuntu16.04_arm64.run`` 中的库文件只提供动态链接库。

- 从主机端系统拷贝文件到边缘端：

  可以先在主机环境下进行部署，再通过 ``scp`` 或其它拷贝工具拷贝到边缘端。

  + 对于DEB包，可以在主机端直接进行解压操作：

     ::

       dpkg -X mluops_<x.y.z>-<build>.ubuntu16.04_arm64.deb <target_directory>

    解压到 ``<target_directory>/usr/local/neuware`` 下。

  + 通过 ``ar`` 和 ``tar`` 命令解压DEB包：

     ::

       ar x mluops_<x.y.z>-<build>.ubuntu16.04_arm64.deb && tar -xf data.tar.gz

    解压到当前目录的 ``usr/local/neuware`` 下。

.. _卸载或升降级Cambricon BANGC OPS版本:

卸载或升降级Cambricon BANGC OPS版本
------------------------------------

卸载Cambricon BANGC OPS
>>>>>>>>>>>>>>>>>>>>>>>>>

卸载Cambricon BANGC OPS，可使用 ``.deb`` 包和 ``.rpm`` 包的操作方式，详细说明，请参看《寒武纪CNToolkit安装升级使用手册》。注意，如果安装了 ``<package_name>-static`` ，卸载时需要同时卸载 ``<package_name>`` 和 ``<package_name>-static`` 。

升级Cambricon BANGC OPS版本
>>>>>>>>>>>>>>>>>>>>>>>>>>>>

由于Cambricon BANGC OPS依赖CNToolkit提供的CNDrv、CNRT、CNBin包，在处理CNToolkit的版本更替时，如果遇到依赖冲突报错，可以先卸载Cambricon BANGC OPS，更新完CNToolkit后，再更新Cambricon BANGC OPS（即可以尝试先全部卸载，再重新安装的方式完成CNToolkit和Cambricon BANGC OPS的版本升降级）。

- ``.deb`` 包升级示例

  使用 ``apt`` 命令：

  ::

    apt install ./cntoolkit_<x.y.z>-1.<distro><id>_<arch>.deb  # 先更新CNToolkit包
    apt update  # 更新CNToolkit本地源
    apt install cndrv cnrt cnbin ./mluops_<x.y.z>-1.<distro><id>_<arch>.deb  # 更新源里的cndrv、cnrt、cnbin和本地的mluops deb包

- ``.rpm`` 包升级示例

  使用 ``yum`` 命令：

  ::

    yum update cntoolkit-<x.y.z>-1.el7update.<arch>.rpm  # 先更新CNToolkit包
    yum clean metadata && yum makecache  # 重建yum包管理数据库信息
    yum update cndrv cnrt cnbin mluops-<x.y.z>-1.el7.<arch>.rpm  # 更新cndrv、cnrt、cnbin、mluops


降级Cambricon BANGC OPS版本
>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Cambricon BANGC OPS的降级推荐直接卸载当前版本，再进行安装。

Cambricon BANGC OPS编程和运行
-----------------------------

了解如何使用Cambricon BANGC OPS构造一个算子或者搭建一个简单的网络，请参考 :ref:`programming_guide` 。

Cambricon BANGC OPS开发样例
----------------------------

寒武纪Cambricon BANGC OPS开发样例为用户提供了abs、polyNms算子运算的样例代码，帮助用户快速体验如何使用Cambricon BANGC OPS来开发、编译以及运行一个算子。用户可以直接通过脚本运行样例代码，无需修改任何配置。

开发样例目录结构与使用方法以实际发布的samples/README.md中描述为准，下面描述样例执行步骤：

1. 设置环境变量。

   a. 确认CNToolkit和Cambricon BANGC OPS完成安装，安装目录的include子目录下包含 ``mlu_op.h`` 头文件，lib64目录下包含 ``libmluops.so`` 、 ``libcnrt.so`` 、 ``libcndrv.so`` 、 ``libcnbin.so`` 。
   b. 设置 ``NEUWARE_HOME`` 环境变量指向安装目录，如 ``export NEUWARE_HOME=/usr/local/neuware`` 。
   c. 在 ``samples/bangc-ops/abs_sample`` 和``samples/bangc-ops/poly_nms_sample`` 下执行 ``source env.sh`` ，自动设置 ``PATH`` 、 ``LD_LIBRARY_PATH`` 。

2. 编译并运行开发样例。

  - 编译全部样例

    a. 在``samples/bangc-ops/`` 目录下运行下面命令：

       ::

         source env.sh
		 
         ./build.sh

       在 ``samples/bangc-ops/build/bin`` 目录下生成可执行文件 ``abs_sample`` 和 ``poly_nms_sample`` 。
    b. 在 ``samples/bangc-ops/build/bin`` 目录下运行样例：

      ::

        ./abs_sample  # 运行 abs_sample 样例

        ./poly_nms_sample  # 运行 poly_nms_sample 样例

  - 编译 abs_sample 样例

    a. 在 ``samples/bangc-ops/abs_sample`` 目录下运行下面命令：

      ::

        source env.sh

        ./build.sh

      在 ``samples/bangc-ops/abs_sample/build/bin`` 目录下生成可执行文件 ``abs_sample`` 。

    b. 在 ``samples/bangc-ops/abs_sample/build/bin`` 目录下运行样例：

      ::

        ./abs_sample

   - 编译 poly_nms_sample 样例

    a. 在 ``samples/bangc-ops/poly_nms_sample`` 目录下运行下面命令：

      ::

        source env.sh

        ./build.sh

      在 ``samples/bangc-ops/poly_nms_sample/build/bin`` 目录下生成可执行文件 ``poly_nms_sample`` 。

    b. 在 ``samples/bangc-ops/poly_nms_sample/build/bin`` 目录下运行样例：

      ::

        ./poly_nms_sample
