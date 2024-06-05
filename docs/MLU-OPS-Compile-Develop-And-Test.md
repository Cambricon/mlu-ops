# MLU-OPS™ 算子编译、开发与测试

## 编译 Operators
- 环境依赖准备

  环境准备参看“依赖环境准备”章节。

- 在mlu-ops目录下，可以使用以下命令完成环境变量的设置。
  ```sh
  cd mlu-ops
  mlu-ops$ source env.sh
  ```

- 编译所有算子
  ```sh
  cd mlu-ops
  mlu-ops$ ./build.sh
  ```

  编译成功后在 `mlu-ops/build/lib` 目录下生成算子库文件 `libmluops.so`，在 `mlu-ops/build/test` 目录下生成测试用的可执行文件 `mluop_gtest` 。

- 编译指定算子

  支持编译指定的一个或多个算子

  ```sh
  cd mlu-ops
  mlu-ops$ ./build.sh --filter="abs;div;sqrt" # '--filter'参数后接要编译的算子，构建系统会根据'kernel_depends.toml'文件描述的依赖自动编译依赖的算子
  ```

  算子名指的是`mlu-ops/kernels`目录下面的文件夹名。

  注意，该功能对算子开发者有一定要求：

  - `mlu-ops/kernels/`、`mlu-ops/test/mlu_op_gtest/pb_gtest/src/zoo`、`mlu-ops/test/mlu_op_gtest/api_gtest/src/gtest/`三个目录下的算子文件夹命名要完全一致
  - 相关算子依赖需要更新[kernel_depends.toml](../kernel_depends.toml)文件，请严格按照字母顺序添加

  当算子存在正反向，且在kernel下的同一个文件夹下实现时

  - 文件结构
  
    `mlu-ops/kernels/op_name`、`mlu-ops/test/mlu_op_gtest/pb_gtest/src/zoo/op_name_forward(op_name_backward)`、`mlu-ops/test/mlu_op_gtest/api_gtest/src/gtest/op_name_forward(op_name_backward)`

  - 添加依赖
  
    在[kernel_depends.toml](../kernel_depends.toml)文件中的[gtest]下添加依赖说明

    ```sh
    op_name_backward = ["op_name"]
    op_name_forward = ["op_name"]
    ```

  - 编译方式

    ```sh
    cd mlu-ops
    mlu-ops$ ./build.sh --filter="op_name_forward(或op_name_backward)" 
    ```

- 多MLU平台架构编译

  - 当不指定架构时，默认编译支持`MLU370`板卡的 `libmluops.so`，运行时动态选择`MLU370`

  - 编译指定MLU板卡

      ```sh
      mlu-ops$ ./build.sh            # 编译多架构的版本，libmluops.so 体积较大，cncc使用多arch的cnfatbin封装
      mlu-ops$ ./build.sh  --mlu370  # 编译 MLU370 板卡专用版本，cncc使用选项--bang-mlu-arch=mtp_372
      mlu-ops$ ./build.sh  --mlu370 --filter="abs;div"  # mlu370 下编译 abs 算子和 div 算子
      ```

- kernel_depends.toml

  TOML格式的配置文件（一种类似于INI文件的格式，但是具有JSON同等的表达能力，支持注释，对人类可读性更友好），记录`mlu-ops/kernels/`目录下的算子编译依赖关系，需要算子开发者进行维护{op1}的依赖{dep\_op1},{dep\_op2}（维护直接的第一级依赖即可），确保`--filter={op1}`时，能正确编译。格式如下：

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
  | `NEUWARE_HOME`              | 用户声明，或`source env.sh`设置 | neuware路径，包含cnrt,cndrv                            | `NEUWARE_HOME`              |                                      |
  | `MLUOP_BUILD_COVERAGE_TEST` | OFF                                | 代码覆盖率测试                                         | `MLUOP_BUILD_COVERAGE_TEST` | -c<br />--coverage                   |
  | `MLUOP_BUILD_ASAN_CHECK`    | OFF                                | 开启ASAN内存检查工具                                   | `MLUOP_BUILD_ASAN_CHECK`    | --asan                               |
  | `MLUOP_MLU_ARCH_LIST`       | `mtp_372`          | 目标mlu架构列表，分号分割的字符串，如"mtp_372" | `MLUOP_MLU_ARCH_LIST`       | --mlu370 |
  | `MLUOP_BUILD_SPECIFIC_OP`   | 空                                 | 编译指定的算子                                         | `MLUOP_BUILD_SPECIFIC_OP`   | --filter                             |
  | `BUILD_JOBS`   | 16                                 | 编译指定的线程数                                         | `BUILD_JOBS`   | -j<br />--jobs                             |

  

## 运行测试用例

各算子的测试用例实现在 `mlu-ops/test/mlu_op_gtest/src/zoo/<op_name>/test_case` 目录下。可以用如下命令执行 abs 算子对应的测试：

```bash
mlu-ops$ cd build/test/
test$ ./mluop_gtest --gtest_filter=*abs*
```

## 新算子开发流程

详情可以参考文档 [MLU-OPS™ 算子开发流程.md](./MLU-OPS-Operator-Development-Process.md)以及 docs 目录下的其它补充说明。

1. 在`mlu-ops/kernels/`路径下，创建算子文件夹，添加算子实现，可以参考现有的 abs 算子进行添加。
2. 在`mlu-ops/test/mlu_op_gtest/src/zoo`创建算子文件夹，添加测试代码。
3. 在算子测试目录 `mlu-ops/test/mlu_op_gtest/src/zoo/<op_name>/` 下进一步创建子目录`test_case`，用于存放测试用例。

## 常用环境变量

简单环境变量可直接执行以下命令：

```bash
# 使能dump data
mlu-ops$ source env_dumpdata_set.sh on
# 关闭dump data
mlu-ops$ source env_dumpdata_set.sh off
```
```bash
# 使能gencase
mlu-ops$ source env_gencase_set.sh on
# 关闭gencase
mlu-ops$ source env_gencase_set.sh off
```

|   |        环境变量        |                         功能说明                        |  使用方法 |               备注                    |
|---|------------------------|---------------------------------------------------------|----|-----------------------------------------|
| 1 | MLUOP_BUILD_GTEST  | 编译MLU-OPS™ 的GTEST| ON时使能，其他情况不使能           | 在build脚本中默认设为ON     |
| 2 | MLUOP_GTEST_DUMP_DATA  | 将MLU-OPS™ 的GTEST的输入输出数据打印至文件中| ON: 保存 GTEST 测试过程中用到的输入输出数据             | 不使用此环境变量时需要unset环境变量     |
| 3 | MLUOP_GEN_CASE         |运行前设置，设置gen_case工具功能等级 |0: 关闭 gen_case 模块功能;<br>1: 生成 prototxt，输入输出只保留 shape 等信息（GEN_CASE_DATA_REAL 将无效）;<br>2: 生成 prototxt,并保留输入真实值;<br>3: 不生成 prototxt,只在屏幕上打印输入输出的 shape 等信息;<br> 详情见: [Gencase-User-Guide-zh.md](./Gencase-User-Guide-zh.md)|   |
| 4 | MLUOP_MIN_LOG_LEVEL    | 设置外部LOG()宏的最小打印级别，用来让外部用户屏蔽不需要的LOG|0: enable INFO/WARNING/ERROR/FATAL;<br>1: enable WARNING/ERROR/FATAL;<br>2: enable ERROR/FATAL;<br>3: enable FATAL |默认为0  |
| 5 | MLUOP_MIN_VLOG_LEVEL   |设置内部VLOG()宏的最小打印级别，用来控制软件内部不同层级调试需要的LOG |0: enable VLOG(0);<br>1: enable VLOG(0)-VLOG(1);<br>2: enable VLOG(0)-VLOG(2);<br>3: enable VLOG(0)-VLOG(3);<br>4: enable VLOG(0)-VLOG(4);<br>5: enable VLOG(0)-VLOG(5);<br>6: enable VLOG(0)-VLOG(6);<br>7: enable VLOG(0)-VLOG(7); | 默认为0| 
| 6 | MLUOP_LOG_ONLY_SHOW  | 是否之展示LOG 而不生成mluop_auto_log 文件  |=ON时，表示不会生产mluop_auto_log文件;<br>=OFF时，表示会生成mluop_auto_log文件 | 默认为ON|
| 7 | MLUOP_LOG_COLOR_PRINT | 决定打印LOG是否开启颜色字体特效  |=ON时，表示打印带颜色的字体加粗等特效;<br>=OFF时，表示关闭打印字体特效 | 默认为ON,但重定向到文件时，不会带颜色字体特效|
| 8 | MLUOP_BUILD_ASAN_CHECK | 在编译的时候设置是否打开ASAN内存检查  |=ON时，表示编译ASAN内存检查;<br>！=ON时，表示不编译ASAN内存检查 | 1.默认不开启 <br>2.该工具仅在Ubuntu上与Debian上有效。无论环境变量如何设置，Centos上都不会编译该工具。<br>3.如果没有检测到内存问题，运行算子case时将不会输出任何内容; 若检测到内存问题，运行算子case时将输出错误内容。|
|9|MLUOP_SET_JOB_LIMIT_CAPABILITY|设置最大JOB限制数量，默认不设置。|=1 CN_KERNEL_CLASS_UNION<br>=2 CN_KERNEL_CLASS_UNION2<br>=3 CN_KERNEL_CLASS_UNION4<br>=4 CN_KERNEL_CLASS_UNION8<br>=5 CN_KERNEL_CLASS_UNION16<br>=6 CN_KERNEL_CLASS_BLOCK不使用<br>=7 CN_KERNEL_CLASS_NONE不使用<br>|JOB_LIMIT和CLUSTER_LIMIT需要同时设置来保证合法性|
|10|MLUOP_GTEST_CLUSTER_LIMIT_CAPABILITY|设置最大cluster限制数量，默认不设置|=1 1cluster<br>=3 2cluster<br>=7 3cluster<br>=15 4cluster<br>...<br>从右往左，每多一个连续的1表示1个cluster |JOB_LIMIT 和CLUSTER_LIMIT 需要同时设置来保证合法性<br>原理是：<br>1的二进制是0000,0001: 1号cluster可用<br>3的二进制是0000,0011: 1号和2好cluster可用<br>...<br>如果有特殊需求，如只想用2号cluster:设置为2: 0000,0010|
|11|MLUOP_GTEST_SET_GDRAM|作用是在GDRAM前后刷NAN/INF| NAN/INF  在GDRAM前后刷NAN/INF|若不设置则根据日期，偶数天刷NAN，奇数天刷INF|
|12|MLUOP_GTEST_UNALIGNED_ADDRESS_RANDOM|设置在GDRAM上申请的空间地址是非64 bytes对齐的，偏移量为1~63的随机值| ON/OFF  ||
|13|MLUOP_GTEST_UNALIGNED_ADDRESS_SET|设置在GDRAM上申请的空间地址是64 bytes对齐的| = NUM ||
