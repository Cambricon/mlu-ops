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


## 常用环境变量

|   |        环境变量        |                         功能说明                        |                 备注                    |
|---|------------------------|---------------------------------------------------------|-----------------------------------------|
| 1 | MLUOP_BUILD_GTEST      | ON: build.sh 会编译 GTEST;<br>其它情况下不会编译        | 在 build 脚本中默认设为 ON              |
| 2 | MLUOP_GTEST_DUMP_DATA  | ON: 保存 GTEST 测试过程中用到的输入输出数据             | 不使用此环境变量时需要unset环境变量     |
| 3 | MLUOP_BUILD_ASAN_CHECK | ON: 表示编译ASAN内存检查;<br>OFF: 表示编译ASAN内存不检查| 默认不开启                              |
| 4 | MLUOP_GEN_CASE         |0: 关闭 gen_case 模块功能;<br>1: 生成 prototxt，输入输出只保留 shape 等信息（GEN_CASE_DATA_REAL 将无效）;<br>2: 生成 prototxt,并保留输入真实真;<br>3: 不生成 prototxt,只在屏幕上打印输入输出的 shape 等信息;<br> 详情见: [Gencase-User-Guide-zh.md](docs/Gencase-User-Guide-zh.md)|   |
| 5 | MLUOP_MIN_LOG_LEVEL    | 0: enable INFO/WARNING/ERROR/FATAL;<br>1: enable WARNING/ERROR/FATAL;<br>2: enable ERROR/FATAL;<br>3: enable FATAL |默认为0  |
| 6 | MLUOP_MIN_VLOG_LEVEL   |0: enable VLOG(0);<br>1: enable VLOG(0)-VLOG(1);<br>2: enable VLOG(0)-VLOG(2);<br>3: enable VLOG(0)-VLOG(3);<br>4: enable VLOG(0)-VLOG(4);<br>5: enable VLOG(0)-VLOG(5);<br>6: enable VLOG(0)-VLOG(6);<br>7: enable VLOG(0)-VLOG(7); | 默认为0| 