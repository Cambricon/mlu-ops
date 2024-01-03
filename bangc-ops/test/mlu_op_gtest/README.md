## 使用方法

### 1. 编译和运行

#### 1.1. 拉代码

```
git clone https://github.com/Cambricon/mlu-ops.git
```

#### 1.2. 编译代码

可以使用 `./build.sh` 脚本，编译 mlu-ops(和 mluop_gtest)

分步过程为:

1. 准备依赖项

   a. 准备 CNToolkit 环境

   ```sh
   sudo apt-get install ./cntoolkit-x.x.x_ubuntuxx.xx_amd64.deb
   sudo apt-get update
   sudo apt-get install cncc cnas cnbin cndrv cnrt
   export NEUWARE_HOME=/usr/local/neuware/
   ```

   b. 其他外部依赖:

   - `protobuf: libprotoc 2.6.1`

   - `libxml2: libxml2-2.7.4`

   可在系统包管理器安装，或自行编译源码并安装，并 `export PATH` 和 `LD_LIBRARY`。

2. 编译

   - 设置环境变量

   ```
   source env.sh
   ```

   - 编译

   ```
   ./build.sh
   ```

3. 运行

   ```
   cd build/test/    // 默认测例使用相对路径，因此要到该路径下执行
   ./mluop_gtest     // 更多参数可以执行./mluop_gtest -h
   ```

#### 1.3. 运行 `mluop_gtest`

##### 运行参数

| 运行参数              | 说明                                                                                   |
| --------------------- | -------------------------------------------------------------------------------------- |
| --gtest_filter=${exp} | 后接正则表达式                                                                         |
| --gtest_list_tests    | 列出所有测例                                                                           |
| --gtest_output=xml    | 后接 xml/json，生成结果报告                                                            |
| --case_path=${path}   | 后接测例路径，且路径中必须包含算子名                                                   |
| --cases_dir=${path}   | 后接测例的根路径，根路径下存放各个算子的测例文件夹                                     |
| --cases_list=${path}  | 后接存放测例路径的文件                                                                 |
| --rand_n=n            | 随机选取 n 的测例，仅用于调试                                                          |
| --perf_repeat=n       | 用于测试性能，重复计算 n 次，取硬件时间的平均值                                        |
| --thread=n            | 多线程运行，n 为线程数. 建议 4/8 线程，超过 10 线程收益不明显，但会造成服务器资源紧张  |

更详细介绍，请执行 `./mluop_gtest -h` 参看说明.

##### 环境变量

| 环境变量                      | 取值    | 说明                                                                        |
| ----------------------------- | ------- | --------------------------------------------------------------------------- |
| MLUOP_GTEST_DUMP_DATA         | ON/else | 保存测试例的输入和输出数据                                                  |
| MLUOP_GTEST_ALL_CRITERION     | ON/else | 无视 pb 中公式，计算 diff1-3                                                |
| CNRT_DEFAULT_DEVICE           | 数字    | 指定计算所用设备，请参看 cnrt 说明文档                                      |
| GTEST_TOTAL_SHARDS            | 数字    | 将 gtest 切分成多进程运行，总切分份数                                       |
| GTEST_SHARD_INDEX             | 数字    | 将 gtest 切分成多进程运行，指定其中第 x 份                                  |
| MLUOP_GTEST_OVERWRITTEN_CHECK | ON/OFF  | 打开/关闭写越界检查                                                         |
| MLUOP_GTEST_SET_GDRAM         | NAN/INF | 在 GDRAM 前后刷 NAN/INF，若不设置，则根据日期偶数日期刷 NAN，奇数日期刷 INF |

##### 多进程运行

可以使用 `GTEST_TOTAL_SHARDS` 和 `GTEST_SHARD_INDEX` 将测例进行拆分，分配给多个进程(每个进程可设置一个卡，即可以多个卡)分别执行。

```
export GTEST_TOTAL_SHARDS=4    // 将测试拆分为4份.

export GTEST_SHARD_INDEX=0     // 执行测例中的第0份测例
export CNRT_DEFAULT_DEVICE=0   // 指定选择卡0，该进程将在device:0上执行
./mluop_gtest &                // 后台执行

export GTEST_SHARD_INDEX=1     // 执行测例中的第1份测例
export CNRT_DEFAULT_DEVICE=1   // 指定选择卡1
./mluop_gtest &                // 后台执行

....                           // 其他进程也一样，略
```

##### 多线程运行

多线程执行会使用线程池机制，并行执行当前算子的所有测例

```
./mluop_gtest --thread=4       // 4线程执行
```

每个线程使用一个 `queue` ，多线程会将多个 `queue` 下到同一个 `device` 上运行。

线程数不宜过大，开发服务器建议 4/8，过多线程会占用过多资源; 空闲服务器可以尝试 16/32，再大没有收益(因服务器而异)。

### 2. 现有工具脚本

| 工具             | 说明                                                                                                                                                             |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| pb2prototxt      | 将*pb 文件转换为*prototxt(可读)文件。 第一个输入参数为 pb 文件名或路径; 第二个参数为输出路径，输出文件名与输入文件同名，但后缀不同，用于查看 pb 文件中内容       |
| prototxt2pb      | 将*prototxt 文件转换为*pb 文件。 第一个输入参数为 prototxt 文件名或路径; 第二个参数为输出路径，输出文件名与输入文件同名，但后缀不同，用于将手写 prototxt 转为 pb |
| generate_case.py | 可以批量生产 prototxt 文件                                                                                                                                       |
