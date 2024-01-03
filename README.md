<div align="center">
  <img src="./docs/MLU-OPS-LOGO.png"/>

<div align="center">
  <b>
    <a href="https://www.cambricon.com/docs/sdk_1.15.0/cambricon_bang_c_ops_0.9.0/user_guide/index.html">
      <font size="4"> 📖 MLU-OPS 用户手册</font>
    </a>
  </b>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <b>
    <a href="https://developer.cambricon.com/">
      <font size="4"> 🌏 寒武纪开发者社区</font>
    </a>
  </b>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <b>
    <a href="https://sdk.cambricon.com/download?sdk_version=V1.15.0&component_name=Basis">
      <font size="4"> 🛠️ 依赖组件获取</font>
    </a>
  </b>
</div>
  
<div>&nbsp;</div>

[![ci](https://github.com/Cambricon/mlu-ops/actions/workflows/ci.yaml/badge.svg)](https://github.com/Cambricon/mlu-ops/actions/workflows/ci.yaml)
[![license](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Cambricon/mlu-ops/blob/master/LICENSE)
![python](https://img.shields.io/badge/python-3.8,_3.9,_3.10-yellow)
![system](https://img.shields.io/badge/system-x86_Ubuntu18.04,_Ubuntu20.04,_Centos7.6,_Centos8.5,_Kylin10-cyan)

</div>

## 简介
MLU-OPS 提供基于寒武纪人工智能单元（MLU），使用 C 接口或者 Python 接口开发高性能算子的示例代码。
MLU-OPS 旨在通过提供示例代码，供开发者参考使用，可用于开发自定义算子，实现对应模型的计算。

MLU-OPS 提供了以下功能：
- [算子精度标准](https://github.com/Cambricon/mlu-ops/blob/master/docs/MLU-OPS-Accuracy-Acceptance-Standard.md)
- [算子性能标准](https://github.com/Cambricon/mlu-ops/blob/master/docs/MLU-OPS-Performance-Acceptance-Standard.md)
- [Op List (高质量实现 BANG C 算子)](https://github.com/Cambricon/mlu-ops/blob/master/docs/bangc-docs/BANGC-OPS-OpList.md)
- [CNNL基础算子使用](https://github.com/Cambricon/mlu-ops/blob/master/docs/MLU-OPS-How-To-Use-CNNL-API.md)
- [测试模块 GTest](https://github.com/cambricon/mlu-ops/blob/master/docs/GTest-User-Guide-zh.md) 支持 [内存泄露测试](https://github.com/cambricon/mlu-ops/blob/master/docs/GTest-User-Guide-zh.md#6-%E5%86%85%E5%AD%98%E6%B3%84%E6%BC%8F%E6%A3%80%E6%B5%8B)、[代码覆盖率测试](https://github.com/cambricon/mlu-ops/blob/master/docs/GTest-User-Guide-zh.md#7-%E4%BB%A3%E7%A0%81%E8%A6%86%E7%9B%96%E7%8E%87)
- [Gen-case (运行时测例生成工具)](https://github.com/Cambricon/mlu-ops/blob/master/docs/Gencase-User-Guide-zh.md)
- [Perf-Analyse (算子性能分析工具)](https://github.com/Cambricon/mlu-ops/tree/master/tools/perf_analyse#readme)

## 依赖条件

- 操作系统：
  - 支持 x86_64 下 Ubuntu18.04、Ubuntu20.04、Centos7.6、Centos8.5、Kylin10
- 寒武纪 MLU SDK：
  - 编译和运行时依赖 CNToolkit v3.7.0 或更高版本，CNNL v1.21.1 或者更高版本
- 寒武纪 MLU 驱动：
  - 运行时依赖驱动 v5.10.15 或更高版本
- 外部链接库：
  - libxml2-dev、libprotobuf-dev<=3.8.0、protobuf-compiler<=3.8.0、llvm-6.0-dev、libeigen3-dev>=3.4
- Python环境：
  - 依赖Python-3.8.0版本


## 依赖环境准备

- 获取 MLU-OPS 代码

  ```sh
  git clone https://github.com/Cambricon/mlu-ops.git
  cd mlu-ops
  git submodule update --init --recursive
  ```

- 准备 CNToolkit、CNNL 环境

  ```sh
  wget https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu18.04/cntoolkit_x.x.x-1.ubuntuxx.xx_amd64.deb
  wget https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu18.04/cnnl_x.x.x-1.ubuntuxx.xx_amd64.deb
  sudo apt-get install ./cntoolkit-x.x.x_ubuntuxx.xx_amd64.deb
  sudo apt-get update
  sudo apt-get install cncc cnas cnbin cndrv cndev cnrt cnrtc cngdb cnperf
  sudo apt-get install ./cnnl_x.x.x-x.ubuntuxx.xx_amd64.deb
  ```

- 准备 Python-3.8.0 环境

  ```sh
  wget https://www.python.org/ftp/python/3.8.0/Python-3.8.0.tgz
  tar -xvf Python-3.8.0.tgz
  cd Python-3.8.0
  make -j24 && make install
  ```

- 准备 BANGPy 环境

  获取 BANGPy 最新版发布包：(https://cair.cambricon.com/)
  ```sh
  pip3.8 install bangpy-x.x.x-py3-none-any.whl
  ```

- 准备链接库环境

  ```sh
  sudo apt-get update
  sudo apt-get install protobuf-compiler libxml2-dev libprotobuf-dev llvm-6.0-dev
  ```
## 开发、编译及测试

当前 C 接口（`BANGC`）、 Python 接口（`BANGPy`）算子开发、编译及测试相互独立
- `BANGC` 算子见 [BANGC-OPS 算子开发流程](docs/bangc-docs/BANGC-OPS-Operator-Development-Process.md)、[README.md](bangc-ops/README.md)
- `BANGPy` 算子见 [BANGPy-OPS 算子开发流程](docs/bangpy-docs/BANGPy-OPS-Operator-Development-Process.md)、[README.md](bangpy-ops/README.md)

更多内容见 docs 目录下文档。

## 获取关于 BANG 语言基础和开发相关工具介绍的文档
可查看最新版 [开发者文档](https://developer.cambricon.com/index/document/index/classid/3.html)
- [BANG C/C++ 编程指南](https://www.cambricon.com/docs/sdk_1.13.0/cntoolkit_3.5.2/programming_guide_1.5.0/index.html)
- [BANG C Developer Guide](https://www.cambricon.com/docs/sdk_1.13.0/cntoolkit_3.5.2/cambricon_bang_c_4.5.1/index.html)
- [CNNL Developer Guide](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_cnnl_1.21.1/developer_guide/index.html)
- [MLU 架构调优指南](https://www.cambricon.com/docs/sdk_1.13.0/cntoolkit_3.5.2/cntoolkit_tuning_0.4.1/index.html)
- [CNRT Developer Guide](https://www.cambricon.com/docs/sdk_1.13.0/cntoolkit_3.5.2/cnrt_6.5.2/index.html)
- [CNRTC Developer Guide](https://www.cambricon.com/docs/sdk_1.13.0/cntoolkit_3.5.2/cambricon_cnrtc_0.6.0/index.html)
- [CNDrv Developer Guide](https://www.cambricon.com/docs/sdk_1.13.0/cntoolkit_3.5.2/cndrv_2.5.2/index.html)
- [CNGDB Developer Guide](https://www.cambricon.com/docs/sdk_1.13.0/cntoolkit_3.5.2/cngdb_3.5.0/index.html)
- [Libdevice Developer Guide](https://www.cambricon.com/docs/sdk_1.13.0/cntoolkit_3.5.2/libdevice_4.5.1/index.html)


## 目录文件结构

| 目录/文件                 | 描述                                    |
| ------------------------ | -------------------------------------- |
| [bangc-ops](bangc-ops)   | C 接口算子开发目录                        |
| [bangpy-ops](bangpy-ops) | Python 接口算子开发目录                   |
| [docker](docker)         | 存放 docker 打包脚本，提供 CI 构建环境。    |
| [docs](docs)             | 算子开发、测试、精度验收的说明文档。         |
