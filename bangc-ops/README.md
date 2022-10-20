# Cambricon BANGC operators

提供基于寒武纪机器学习单元（Machine Learning Unit，MLU）开发高性能算子、C 接口封装的示例代码。

## 编译 BANGC operators
- 环境依赖准备
环境准备参看[README.md](../README.md)。

- 在mlu-ops目录下，可以使用以下命令完成环境变量的设置。
  ```sh
  cd mlu-ops
  source env.sh
  ```

- 编译 bangc-ops
  ```sh
  cd mlu-ops/bangc-ops
  ./build.sh
  ```

  编译成功后在 `bangc-ops/build/lib` 目录下生成算子库文件 `libmluops.so`，在 `bangc-ops/build/test` 目录下生成测试用的可执行文件 `mluop_gtest` 。


## 运行测试用例

各算子的测试用例实现在 `bangc-ops/test/mlu_op_gtest/src/zoo/*/test_case` 目录下。可以用如下命令执行 abs 算子对应的测试：

```bash
cd bangc-ops/build/test/
./mluop_gtest --gtest_filter=*abs*
```

## 新算子开发流程

详情可以参考文档 [BANGC-OPS 算子开发流程.md](../docs/bangc-docs/BANGC-OPS-Operator-Development-Process.md)以及 docs 目录下的其它补充说明。

1. 在`mlu-ops/bangc-ops/kernels/`路径下，创建算子文件夹，添加算子实现，可以参考现有的 abs 算子进行添加。
2. 在`mlu-ops/bangc-ops/test/mlu_op_gtest/src/zoo`创建算子文件夹，添加测试代码。
3. 在算子测试目录 `mlu-ops/bangc-ops/test/mlu_op_gtest/src/zoo/xxx` 下进一步创建子目录`test_case`，用于存放测试用例。

## 目录文件结构

| 目录/文件            | 描述                                                           |
| -------------------- | -------------------------------------------------------------- |
| [mlu_op.h](mlu_op.h) | 公共数据类型描述，以及 kernels 目录中的算子对外提供的 C 接口。 |
| [core](core)         | 包含公共数据类型的操作、运行时管理、日志等公共实现。           |
| [kernels](kernels)   | 算子代码实现，包含一元、二元算子模板供其他算子调用。           |
| [test](test)         | 存放测试算子用的代码。                                         |
