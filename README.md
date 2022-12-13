# Cambricon MLU-OPS

[![ci](https://github.com/Cambricon/mlu-ops/actions/workflows/ci.yaml/badge.svg)](https://github.com/Cambricon/mlu-ops/actions/workflows/ci.yaml)

MLU-OPS 提供基于寒武纪人工智能单元（MLU），使用 C 接口或者 Python 接口开发高性能算子的示例代码。
MLU-OPS 旨在通过提供示例代码，供开发者参考使用，可用于开发自定义算子，实现对应模型的计算。

## 依赖条件

- 操作系统：
  - 支持 x86_64 下 Ubuntu18.04、Ubuntu20.04、Centos7.6、Centos8.5;
  - 支持 AArch64 下 KylinV10。
- 寒武纪 MLU SDK：
  - 编译和运行时依赖 CNToolkit v3.1.2 或更高版本。
- 寒武纪 MLU 驱动：
  - 运行时依赖驱动 v4.20.11 或更高版本。
- 外部链接库：
  - libxml2-dev、libprotobuf-dev、protobuf-compiler、llvm-6.0-dev
- Python环境：
  - 依赖Python-3.8.0版本。

## MLU-OPS 依赖环境准备

- 获取 MLU-OPS 代码

  ```sh
  git clone https://github.com/Cambricon/mlu-ops.git
  ```

- 准备 CNToolkit 环境

  ```sh
  sudo apt-get install ./cntoolkit-x.x.x_ubuntuxx.xx_amd64.deb
  sudo apt-get update
  sudo apt-get install cncc cnas cnbin cndrv cnrt cnrtc
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

## 编译和运行测试用例

当前 C 接口（`BANGC`）和 Python 接口（`BANGPy`）开发编译和测试分开，后续会将两种接口开发编译、测试统一到一起。

- C 接口请参考文档 [README.md](bangc-ops/README.md)。
- Python 接口请参考文档 [README.md](bangpy-ops/README.md)。

## 新算子开发流程

详情可以参考文档 [BANGC-OPS 算子开发流程.md](docs/bangc-docs/BANGC-OPS-Operator-Development-Process.md)、
[BANGPy-OPS 算子开发流程.md](docs/bangpy-docs/BANGPy-OPS-Operator-Development-Process.md) 以及 docs 目录下的其它补充说明，
同时也需要参考 C 接口说明文档[README.md](bangc-ops/README.md) 和 Python 接口说明文档[README.md](bangpy-ops/README.md)。

## 获取开发手册
查看最新版 BANGPy 开发手册(https://developer.cambricon.com/index/document/index/classid/3.html)，获取安装说明、教程、示例。


## 目录文件结构

| 目录/文件                 | 描述                                    |
| ------------------------ | -------------------------------------- |
| [bangc-ops](bangc-ops)   | C 接口算子开发目录                        |
| [bangpy-ops](bangpy-ops) | Python 接口算子开发目录                   |
| [docker](docker)         | 存放docker打包脚本，提供CI构建环境。        |
| [docs](docs)             | 算子开发、测试、精度验收的说明文档。         |
