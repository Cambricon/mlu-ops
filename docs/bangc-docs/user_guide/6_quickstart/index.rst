快速入门
=================

本章重点介绍了如何部署和使用寒武纪BANGC OPS，以及如何运行寒武纪提供的BANGC OPS样例。

部署寒武纪BANGC OPS
--------------------

Cambricon CNNL的使用依赖于CNToolkit工具包。用户在使用Cambricon CNNL库之前需要先安装寒武纪CNToolkit工具包。安装详细步骤，请参考《寒武纪CNToolkit安装升级使用手册》。

Cambricon CNNL提供了满足运行时及动态链接开发需要的软件包，并且发布了只包含静态库 ``libcnnl.a`` 的软件包（包名称含 ``-static`` 后缀），用户可以在安装原有软件包基础上，安装静态库软件包用于静态链接，具体发布信息可参看 :ref:`软件包发布信息`。软件包操作方式都相同，下面以 ``cnnl`` 为例， 对发布包操作进行说明。

安装完成CNToolkit工具包后，执行下面命令部署Cambricon CNNL。默认头文件（cnnl.h）会安装在 ``/usr/local/neuware/include`` 目录下，链接库文件（libcnnl.so、libcnnl.a）会安装在 ``/usr/local/neuware/lib64`` 目录下，示例程序会安装在 ``/usr/local/neuware/samples/cnnl`` 目录下。


Ubuntu系统
>>>>>>>>>>

- 执行安装命令：

  ::

    sudo apt install ./cnnl_<x.y.z>-1.ubuntu<a.b>_<arch>.deb

  <x.y.z>为Cambricon CNNL版本号，<a.b>为操作系统版本号，<arch>为CPU架构（amd64或arm64）。例如Cambricon CNNL v1.0.0版本在X86_64的Ubuntu18.04系统下的包名为 ``cnnl_1.0.0-1.ubuntu18.04_amd64.deb`` ，在ARM64的Ubuntu18.04系统下的包名为 ``cnnl_1.0.0-1.ubuntu18.04_arm64.deb`` 。


CentOS系统
>>>>>>>>>>

- 执行安装命令：

  ::

    sudo yum install cnnl-<x.y.z>-1.el7.<arch>.rpm

  <x.y.z>为Cambricon CNNL版本号，<arch>为CPU架构，例如Cambricon CNNL v1.0.0版本在CentOS系统下的包名为 ``cnnl-1.0.0-1.el7.x86_64.rpm`` 。


Kylin系统
>>>>>>>>>>

- 执行安装命令：

  ::

    sudo yum install cnnl-<x.y.z>-1.ky10.<arch>.rpm

  <x.y.z>为Cambricon CNNL版本号，<arch>为CPU架构（目前只支持arm64），例如Cambricon CNNL v1.0.0版本在Kylin系统下的包名为 ``cnnl-1.0.0-1.ky10.aarch64.rpm`` 。


.. _寒武纪边缘端嵌入式Linux 系统:

寒武纪边缘端嵌入式Linux系统
>>>>>>>>>>>>>>>>>>>>>>>>>>>

寒武纪边缘端嵌入式Linux系统基于ARM64，不具备包管理工具，因此需要通过解压的方式进行安装Cambricon CNNL，或者先在中心主机端系统下进行部署，然后拷贝到边缘端。

同时，边缘端系统不具备直接编译、生成应用的能力，如果要在端侧运行Cambricon CNNL的示例程序或者二次开发，需要在ARM64主机端直接编译或者x86主机端交叉编译，再将生成的应用拷贝到边缘端。

对于ARM64架构的边缘端设备，Cambricon CNNL发行基于Ubuntu 16.04环境生成的DEB包、基于CentOS 7环境生成的RPM包以及自解压安装包，并提供两种部署方式：

.. TODO 考虑静态库文件.a的处理

- 边缘端系统下运行安装包，自动解压到目标目录下：

  ::

    bash cnnl_<x.y.z>-<build>.ubuntu16.04_arm64.run --target <target_directory>

  <x.y.z>为Cambricon CNNL版本号，<build>为打包版本，一般为1，<target_directory>为目标目录，会在该目录下安装 ``lib64`` 、 ``include`` 等目录，如果不指定 ``--target`` 参数，默认安装到 ``/usr/local/neuware`` 下。

  注意， ``cnnl_<x.y.z>-<build>.ubuntu16.04_arm64.run`` 中的库文件只提供动态链接库，如果是基于Cambricon CNNL静态链接库进行开发，则不需要在边缘端部署CNNL。

- 从主机端系统拷贝文件到边缘端：

  可以先在主机环境下进行部署，再通过 ``scp`` 或其它拷贝工具拷贝到边缘端。

  + 对于DEB包，可以在主机端直接进行解压操作：

     ::

       dpkg -X cnnl_<x.y.z>-<build>.ubuntu16.04_arm64.deb <target_directory>

    解压到 ``<target_directory>/usr/local/neuware`` 下。

  + 通过 ``ar`` 和 ``tar`` 命令解压DEB包：

     ::

       ar x cnnl_<x.y.z>-<build>.ubuntu16.04_arm64.deb && tar -xf data.tar.gz

    解压到当前目录的 ``usr/local/neuware`` 下。

.. _卸载或升降级CNNL版本:

卸载或升降级Cambricon CNNL版本
----------------------------------

卸载Cambricon CNNL
>>>>>>>>>>>>>>>>>>>

卸载Cambricon CNNL，可使用 ``.deb`` 包和 ``.rpm`` 包的操作方式，详细说明，请参看《寒武纪CNToolkit安装升级使用手册》。注意，如果安装了 ``<package_name>-static`` ，卸载时需要同时卸载 ``<package_name>`` 和 ``<package_name>-static`` 。

升级Cambricon CNNL版本
>>>>>>>>>>>>>>>>>>>>>>>>>>>

由于Cambricon CNNL依赖CNToolkit提供的CNDrv、CNRT、CNBin包，在处理CNToolkit的版本更替时，如果遇到依赖冲突报错，可以先卸载Cambricon CNNL，更新完CNToolkit后，再更新Cambricon CNNL（即可以尝试先全部卸载，再重新安装的方式完成CNToolkit和Cambricon CNNL的版本升降级）。

- ``.deb`` 包升级示例

  使用 ``apt`` 命令：

  ::

    apt install ./cntoolkit_<x.y.z>-1.<distro><id>_<arch>.deb  # 先更新CNToolkit包
    apt update  # 更新CNToolkit本地源
    apt install cndrv cnrt cnbin ./cnnl_<x.y.z>-1.<distro><id>_<arch>.deb  # 更新源里的cndrv、cnrt、cnbin和本地的cnnl deb包

- ``.rpm`` 包升级示例

  使用 ``yum`` 命令：

  ::

    yum update cntoolkit-<x.y.z>-1.el7update.<arch>.rpm  # 先更新CNToolkit包
    yum clean metadata && yum makecache  # 重建yum包管理数据库信息
    yum update cndrv cnrt cnbin cnnl-<x.y.z>-1.el7.<arch>.rpm  # 更新cndrv、cnrt、cnbin、cnnl


降级Cambricon CNNL版本
>>>>>>>>>>>>>>>>>>>>>>>

Cambricon CNNL的降级推荐直接卸载当前版本，再进行安装。

Cambricon CNNL编程和运行
--------------------------

了解如何使用Cambricon CNNL构造一个算子或者搭建一个简单的网络，请参考 :ref:`programming_guide` 。

Cambricon CNNL开发样例
----------------------------

寒武纪Cambricon CNNL开发样例为用户提供了卷积算子运算的样例代码，帮助用户快速体验如何使用Cambricon CNNL来开发、编译以及运行一个算子。用户可以直接通过脚本运行样例代码，无需修改任何配置。

开发样例目录结构与使用方法以实际发布的samples/README.md中描述为准，下面以conv_sample为例描述样例执行步骤：

1. 设置环境变量。

   a. 确认CNToolkit和Cambricon CNNL完成安装，安装目录的include子目录下包含 ``cnnl.h`` 头文件，lib64目录下包含 ``libcnnl.so`` 、 ``libcnrt.so`` 、 ``libcndrv.so`` 、 ``libcnbin.so`` 。
   b. 设置 ``NEUWARE_HOME`` 环境变量指向安装目录，如 ``export NEUWARE_HOME=/usr/local/neuware`` 。
   c. 在 ``samples/cnnl`` 下执行 ``source env.sh`` ，自动设置 ``PATH`` 、 ``LD_LIBRARY_PATH`` 。

2. 编译并运行开发样例。

   - 如果编译所有示例：

     a. 在 ``samples/cnnl`` 目录下运行下面脚本：

        ::

          ./build.sh

        生成的可执行文件会存放到 ``samples/cnnl/build/bin`` 目录下。

     b. 在 ``samples/cnnl/build/bin`` 目录下运行样例。其中 ``xxx_sample`` 需替换为要运行的算子样例名。

        ::

          ./xxx_sample -[param1][param1_value]  -[param2][param2_value] -[param3][param3_value] ... -[paramN][paramN_value]

        支持的参数和参数值可参考 ``samples/cnnl/conv_sample/run_conv_sample.sh`` 文件。

        示例如下：

        ::

           ./conv_sample -ni1 -hi14 -wi14 -ci256 -co256 -kh3 -kw3 -sh1 -sw1 -dh1 -dw1 -pt1 -pb1 -pl1 -pr1 -id3 -wd3 -od1 -hb0 -gc1
           ./conv_sample -ni1 -hi16 -wi16 -ci64 -co512 -kh3 -kw3 -sh1 -sw1 -dh1 -dw1 -pt0 -pb0 -pl0 -pr0 -id3 -wd3 -od1 -hb0 -gc1
           ./conv_sample -ni1 -hi16 -wi16 -ci64 -co512 -kh3 -kw3 -sh1 -sw1 -dh1 -dw1 -pt1 -pb1 -pl1 -pr1 -id4 -wd4 -od1 -hb1 -gc1
           ./conv_sample -ni1 -hi16 -wi16 -ci64 -co512 -kh3 -kw3 -sh1 -sw1 -dh1 -dw1 -pt3 -pb3 -pl1 -pr1 -id3 -wd3 -od2 -hb0 -gc1
           ./conv_sample -ni1 -hi16 -wi16 -ci64 -co512 -kh3 -kw3 -sh1 -sw1 -dh1 -dw1 -pt1 -pb1 -pl0 -pr0 -id4 -wd4 -od2 -hb1 -gc1


   - 如果只编译某一个样例，如conv_sample。

     a. 在 ``samples/cnnl/conv_sample`` 目录下运行下面命令：

        ::

           make clean
           make

        在 ``samples/cnnl/conv_sample`` 目录下生成可执行文件 ``conv_sample`` 。

     b. 在 ``samples/cnnl/conv_sample`` 目录下运行样例：

        ::

          ./conv_sample -[param1][param1_value]  -[param2][param2_value] -[param3][param3_value] ... -[paramN][paramN_value]

        支持的参数和参数值请查看 ``run_conv_sample.sh`` 文件。

        示例如下：

        ::

           ./conv_sample -ni1 -hi14 -wi14 -ci256 -co256 -kh3 -kw3 -sh1 -sw1 -dh1 -dw1 -pt1 -pb1 -pl1 -pr1 -id3 -wd3 -od1 -hb0 -gc1
           ./conv_sample -ni1 -hi16 -wi16 -ci64 -co512 -kh3 -kw3 -sh1 -sw1 -dh1 -dw1 -pt0 -pb0 -pl0 -pr0 -id3 -wd3 -od1 -hb0 -gc1
           ./conv_sample -ni1 -hi16 -wi16 -ci64 -co512 -kh3 -kw3 -sh1 -sw1 -dh1 -dw1 -pt1 -pb1 -pl1 -pr1 -id4 -wd4 -od1 -hb1 -gc1
           ./conv_sample -ni1 -hi16 -wi16 -ci64 -co512 -kh3 -kw3 -sh1 -sw1 -dh1 -dw1 -pt3 -pb3 -pl1 -pr1 -id3 -wd3 -od2 -hb0 -gc1
           ./conv_sample -ni1 -hi16 -wi16 -ci64 -co512 -kh3 -kw3 -sh1 -sw1 -dh1 -dw1 -pt1 -pb1 -pl0 -pr0 -id4 -wd4 -od2 -hb1 -gc1

        或可直接运行 ``run_conv_sample.sh`` 文件。

交叉编译开发样例（针对边缘端）
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

交叉编译相比于直接编译，需要准备交叉编译工具链和目标平台的CNDrv、CNRT、CNBin库（包含于CNToolkit）。进一步地，如果需要编译mlu文件，还需要额外准备主机端平台的 CNCC、CNAS工具。

1. 主机端设置环境变量。

   a. 安装x86的CNToolkit工具包所提供的CNCC和CNAS。
   b. 安装或解压ARM64的Cambricon CNNL、CNRT、CNDrv。
   c. 设置 ``NEUWARE_HOME`` 环境变量，指向安装的目录。
   d. 在 ``samples/cnnl`` 目录下执行 ``source env.sh`` ，自动设置 ``PATH``、 ``LD_LIBRARY_PATH`` 。

2. 主机端 ``samples/cnnl`` 目录下执行 ``build.sh`` 进行编译:

   ::

      ./build.sh --aarch64

   或者

   ::

      ./build.sh --target=aarch64-linux-gnu

   可以执行 ``./build.sh --help`` 查询其它选项和环境变量，例如可以设置 ``CC`` 和 ``CXX`` 指定交叉编译用的编译器命令。

3. 拷贝到边缘端运行:

   需要在边缘端部署ARM64的CNDrv、CNRT、CNBin（CNToolkit提供），以及部署Cambricon CNNL。例如将库文件（.so）都部署到了 ``/neuware/lib64`` 下，然后设置环境变量:

   ::

      export LD_LIBRARY_PATH=/neuware/lib64:$LD_LIBRARY_PATH


   同时将生成的可执行程序拷贝到边缘端，即可运行示例程序（如conv_sample):

   ::

      ./conv_sample -ni1 -hi16 -wi16 -ci64 -co512 -kh3 -kw3 -sh1 -sw1 -dh1 -dw1 -pt0 -pb0 -pl0 -pr0 -id3 -wd3 -od1 -hb0 -gc1
