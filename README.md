# Cambricon mlu-ops

[![ci](https://github.com/Cambricon/mlu-ops/actions/workflows/ci.yaml/badge.svg)](https://github.com/Cambricon/mlu-ops/actions/workflows/ci.yaml)

mlu-ops 提供基于寒武纪机器学习单元（Machine Learning Unit，MLU）开发高性能算子、C 接口封装的示例代码。

## 依赖条件

- 操作系统：
  - 目前只支持 Ubuntu 16.04 x86_64
- 寒武纪 MLU SDK：
  - 编译和运行时依赖 CNToolkit v2.3.2 或更高版本
- 寒武纪 MLU 驱动：
  - 运行时依赖驱动 v4.15.3 或更高版本

## 编译 mlu-ops

- 获取 mlu-ops 代码

  ```sh
  git clone https://github.com/Cambricon/mlu-ops.git
  ```

- 准备 CNToolkit 环境

  ```sh
  sudo apt-get install ./cntoolkit-x.x.x_ubuntuxx.xx_amd64.deb
  sudo apt-get update
  sudo apt-get install cncc cnas cnbin cndrv cnrt
  export NEUWARE_HOME=/usr/local/neuware/
  ```

- 编译 mlu-ops 库及测试程序

  ```sh
  cd mlu-ops
  ./build.sh
  ```

  编译成功后在 `build/lib` 目录下生成算子库文件 `libmluops.so`，在 `build/test` 目录下生成测试用的可执行文件 `mluop_gtest` 。

## 运行测试用例

各算子的测试用例在 `test/mlu_op_gtest/src/zoo/*/test_case` 目录下，可以用如下命令执行 abs 算子对应的测试：

```bash
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib64:${PWD}/build/lib
cd build/test/
./mluop_gtest --gtest_filter=*abs*
```

## 新算子开发流程

详情可以参考文档 [MLU-OPS 算子开发流程.md](docs/MLU-OPS算子开发流程.md)以及 docs 目录下的其它补充说明。

1. 在`mlu-ops/kernels/`路径下，创建算子文件夹，添加算子实现，可以参考现有的 abs 算子进行添加。
2. 在`test/mlu_op_gtest/src/zoo`创建算子文件夹，添加测试代码。
3. 在算子测试目录 `test/mlu_op_gtest/src/zoo/xxx` 下进一步创建子目录`test_case`，用于存放测试用例。

## 目录文件结构

| 目录/文件            | 描述                                                           |
| -------------------- | -------------------------------------------------------------- |
| [mlu_op.h](mlu_op.h) | 公共数据类型描述，以及 kernels 目录中的算子对外提供的 C 接口。 |
| [core](core)         | 包含公共数据类型的操作、运行时管理、日志等公共实现。           |
| [kernels](kernels)   | 算子代码实现，包含一元、二元算子模板供其他算子调用。           |
| [test](test)         | 存放测试算子用的代码。                                         |
| [docs](docs)         | 算子开发、测试、精度验收的说明文档。                           |
