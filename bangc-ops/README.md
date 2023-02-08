# Cambricon BANGC operators

提供基于寒武纪人工智能单元（MLU）开发高性能算子、C 接口封装的示例代码。

## 编译 BANGC operators
- 环境依赖准备
环境准备参看[README.md](../README.md)。

- 在mlu-ops目录下，可以使用以下命令完成环境变量的设置。
  ```sh
  cd mlu-ops
  source env.sh
  ```

- 编译所有算子
  ```sh
  cd mlu-ops/bangc-ops
  ./build.sh
  ```

  编译成功后在 `bangc-ops/build/lib` 目录下生成算子库文件 `libmluops.so`，在 `bangc-ops/build/test` 目录下生成测试用的可执行文件 `mluop_gtest` 。

- 编译指定算子

  支持编译指定的一个或多个算子

  ```sh
  cd mlu-ops/bangc-ops
  ./build.sh --filter="abs;expand" # '--filter'参数后接要编译的算子，构建系统会根据'kernel_depends.toml'文件描述的依赖自动编译依赖的算子
  ```

  算子名指的是`bangc-ops/kernels`目录下面的文件夹名。

  注意，该功能对算子开发者有一定要求：

  - `kernels/`、`test/mlu_op_gtest/pb_gtest/src/zoo`、`test/mlu_op_gtest/api_gtest/src/gtest/`三个目录下的算子文件夹命名要完全一致
  - 相关算子依赖需要更新[kernel_depends.toml](./kernel_depends.toml)文件，请严格按照字母顺序添加

  当算子存在正反向，且在kernel下的同一个文件夹下实现时

  - 文件结构
  
    `kernels/op_name`、`test/mlu_op_gtest/pb_gtest/src/zoo/op_name_forward(op_name_backward)`、`test/mlu_op_gtest/api_gtest/src/gtest/op_name_forward(op_name_backward)`

  - 添加依赖
  
    在[kernel_depends.toml](./kernel_depends.toml)文件中的[bangc-ops.gtest]下添加依赖说明

    ```sh
    op_name_backward = ["op_name"]
    op_name_forward = ["op_name"]
    ```

  - 编译方式

    ```sh
    cd mlu-ops/bangc-ops
    ./build.sh --filter="op_name_forward(或op_name_backward)" 
    ```

- 多MLU平台架构编译

  - 当不指定架构时，默认编译支持`MLU370`板卡的 `libmluops.so`，运行时动态选择`MLU370`

  - 编译指定MLU板卡

      ```sh
      ./build.sh            # 编译多架构的版本，libmluops.so 体积较大，cncc使用多arch的cnfatbin封装
      ./build.sh  --mlu370  # 编译 MLU370 板卡专用版本，cncc使用选项--bang-mlu-arch=mtp_372
      ./build.sh  --mlu370 --filter="abs;expand"  # mlu370 下编译 abs 算子和 expand 算子
      ```

- kernel_depends.toml

  TOML格式的配置文件（一种类似于INI文件的格式，但是具有JSON同等的表达能力，支持注释，对人类可读性更友好），记录`kernels/`目录下的算子编译依赖关系，需要算子开发者进行维护{op1}的依赖{dep\_op1},{dep\_op2}（维护直接的第一级依赖即可），确保`--filter={op1}`时，能正确编译。格式如下：

  ```sh
  <op_name> = ["dep_op1", "dep_op2", ...]
  ```

- gen_symbol_visibility_map.py

  - `gen_symbol_visibility_map.py`脚本用于解析`mlu_op.h`头文件，获取函数名，生成`symbol_visibility.map`配置文件。
    ```sh
    MLUOP_ABI {
	    global: op1_func;op2_func;
	    local: *;
    };
    ```
    global：表示符号是全局的（外部的）
    local：表示符号是本地的，即对外不可见
  - 执行build.sh编译时，将自动执行`gen_symbol_visibility_map.py`生成`symbol_visibility.map`配置文件。
  - 在编译阶段依据`symbol_visibility.map`文件中global字段定义的符号表，将动态库`libmluops.so`中除global中定义的符号外其他符号定义为local。

- 命令行参数

  可通过`./build.sh -h`或`./build.sh --help`，查看命令行参数

  | 变量名                      | 默认值                             | 说明                                                   | 关联cmake选项               | 关联命令行参数                       |
  | --------------------------- | ---------------------------------- | ------------------------------------------------------ | --------------------------- | ------------------------------------ |
  | `BUILD_MODE`                | release                            | release/debug，编译模式                                | `CMAKE_BUILD_TYPE`          | -d<br />--debug                      |
  | `NEUWARE_HOME`              | 用户声明，或`source ../env.sh`设置 | neuware路径，包含cnrt,cndrv                            | `NEUWARE_HOME`              |                                      |
  | `MLUOP_BUILD_COVERAGE_TEST` | OFF                                | 代码覆盖率测试                                         | `MLUOP_BUILD_COVERAGE_TEST` | -c<br />--coverage                   |
  | `MLUOP_BUILD_ASAN_CHECK`    | OFF                                | 开启ASAN内存检查工具                                   | `MLUOP_BUILD_ASAN_CHECK`    | --asan                               |
  | `MLUOP_MLU_ARCH_LIST`       | `mtp_372`          | 目标mlu架构列表，分号分割的字符串，如"mtp_372" | `MLUOP_MLU_ARCH_LIST`       | --mlu370 |
  | `MLUOP_BUILD_SPECIFIC_OP`   | 空                                 | 编译指定的算子                                         | `MLUOP_BUILD_SPECIFIC_OP`   | --filter                             |

  

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
