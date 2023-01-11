# Cambricon BANGPy operators

提供基于寒武纪人工智能单元（MLU）开发高性能算子、Python 接口封装的示例代码。

## 编译 BANGPy operators
- 环境依赖准备
环境准备参看[README.md](../README.md)。

- 在mlu-ops目录下，可以使用以下命令完成环境变量的设置。
  ```sh
  cd mlu-ops
  source env.sh
  ```

- 编译 bangpy-ops
  - 编译ops目录下的全部算子
    ```sh
    cd mlu-ops/bangpy-ops
    ./utils/build_operators.sh
    ```
  - 使用 `--filter` 编译指定的一个或多个算子
    ```sh
    cd mlu-ops/bangpy-ops
    ./utils/build_operators.sh --filter=axx,bxx,cxx
    ```
  - 使用 `--opsfile` 从存放算子列表的文件中读取指定算子并进行编译
    ```sh
    cd mlu-ops/bangpy-ops
    ./utils/build_operators.sh --opsfile=./ops_xxx.txt
    ```

  `注意` : 如果将指定算子列表存放在文件中，其格式是每行一个算子。
  
  编译成功后在 `bangpy-ops/outs` 目录下生成与算子同名的输出文件夹，其中包含 `libmluops.so`等文件。

## 创建测试用例
测试用例可通过 mlu-ops-generator 测例生成框架进行创建，步骤如下：

- 在 mlu-ops-generator 框架下进行本地环境配置
  - 根据对第三方计算库的依赖，安装对应的环境，例如pytorch, tensorflow。
- 添加算子在 GPU/CPU 上的计算逻辑
  - 将使用第三方库实现的算子逻辑文件夹添加至 mlu-ops-generator/nonmlu_ops/ 下。
- 撰写算子 Manual 格式的 Json 文件
  - 撰写完成的Json 文件添加至 mlu-ops-generator/manual_config/ ，json 文件的具体格式要求需参照 mlu-ops-generator 对应部分的使用说明。
- 生成测例文件
  - 算子 Manual 格式的 Json 文件撰写完成后，在 mlu-ops-generator 框架下运行以下脚本生成测例文件，
  ```
  python3 run_manual.py <opname>
  ```
  测例文件的保存格式可以是 pb 和 prototxt。测例创建成功后，将保存的 prototxt 文件移至 'bangpy-ops/ops/' 的不同算子目录下 testcase 文件夹。

`注意` :mlu-ops-generator模块的详细介绍见(https://github.com/Cambricon/mlu-ops-generator)

## 运行测试用例

首先确定当前的测试平台，如 `mlu370`，之后可以用如下几种命令对算子进行测试：

- 测试ops目录下的全部算子
  ```sh
  cd mlu-ops/bangpy-ops
  ./utils/test_operators.sh --target=mlu3xx
  ```
- 使用 `--filter` 测试指定的一个或多个算子
  ```sh
  cd mlu-ops/bangpy-ops
  ./utils/test_operators.sh --filter=axx,bxx,cxx --target=mlu3xx
  ```
- 使用 `--opsfile` 从存放算子列表的文件中读取指定算子并进行测试
  ```sh
  cd mlu-ops/bangpy-ops
  ./utils/test_operators.sh --opsfile=./ops_xxx.txt --target=mlu3xx
  ```

`注意` :
- 如果将指定算子列表存放在文件中，其格式是每行一个算子。
- 该脚本默认是每个算子先进行编译，再进行测试，如果想跳过编译阶段而直接进行测试，那么请在保证已完成算子编译的情况下对以上三种不同的 `./utils/test_operators.sh` 调用加上 `--only_test` 选项，例如
  ```sh
  cd mlu-ops/bangpy-ops
  ./utils/test_operators.sh --target=mlu3xx --only_test
  ```
测试结果会在所有算子测试完毕后显示。

## 生成算子库文件

在确认新增的算子编译及测试无误后，使用以下几种命令可以生成包含所有指定算子的算子库文件：

- 生成包含ops目录下全部算子的算子库文件
  ```sh
  cd mlu-ops/bangpy-ops
  ./release.sh -r
  ```
- 生成并测试该包含ops目录下全部算子的算子库文件
  ```sh
  cd mlu-ops/bangpy-ops
  ./release.sh -r -t --target=mlu3xx
  ```
- `--filter` 及 `--opsfile` 参数对该脚本同样适用
  ```sh
  cd mlu-ops/bangpy-ops
  ./release.sh -r --filter=axx,bxx,cxx
  ```

`注意` :
- 注意加入 `-r` 选项，否则各算子编译后不会再链接生成最终的算子库文件
- 脚本正常执行完成后，会生成算子库文件 `bangpy-ops/outs/libmluops.so` 及包含算子接口的头文件 `bangpy-ops/outs/mlu_ops.h`，而 `bangpy-ops/outs` 文件夹下有且仅有这两个文件
- 可以使用 `./release.sh -h` 查看其参数介绍，其他脚本类似

算子库文件生成后会放在 `bangpy-ops/outs` 文件夹下。

## 新算子开发流程

详情可以参考文档 [BANGPy-OPS 算子开发流程.md](../docs/bangpy-docs/BANGPy-OPS-Operator-Development-Process.md)。

1. 在 `mlu-ops/bangpy-ops/ops/` 路径下，创建算子文件夹，添加算子实现文件，可以参考现有的 add 算子中的[add.py](./ops/add/add.py)进行添加。
2. 在算子文件夹下创建以 `test_` 为前缀的算子测试文件，添加测试代码，可以参考[test_add.py](./ops/add/test_add.py)进行添加。

## 目录文件结构

| 目录/文件            | 描述                                                           |
| -------------------- | -------------------------------------------------------------- |
| [include](include)   | 包含数据类型以及张量描述符等多种数据结构的描述。                          |
| [ops](ops)           | 算子代码实现，包含全部使用BANGPy编写的算子。                       |
| [utils](utils)       | 存放编译及测试算子的工具文件。                                    |
