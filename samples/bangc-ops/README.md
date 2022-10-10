快速入门
===============

mluops 样例
----------------

```
|-- samples
  |-- build.sh
  |-- CMakeLists.txt
  |-- env.sh
  |-- abs_sample
  |   |-- abs_sample.cc
  |   |-- build.sh
  |   |-- run.sh
  |   |-- CMakeLists.txt 
  |-- poly_nms_sample
  |   |-- poly_nms_sample.cc
  |   |-- build.sh
  |   |-- run.sh
  |   |-- CMakeLists.txt
  |-- fault_sample
  |   |-- fault_demo.mlu
  |   |-- fault_kernel.h
  |   |-- fault_kernel.mlu
  |   |-- build.sh
  |   |-- run.sh
  |   |-- CMakeLists.txt
```

样例所有文件介绍

**sample文件夹**

- env.sh: 设置环境变量，指定库路径；
- CMakeLists.txt：cmake 描述文件， 用于编译全部样例；
- build.sh：自动化编译脚本，其内部对cmake命令进行了封装；
- abs_sample: 调用 mluOpAbs 的示例文件；
- poly_nms_sample: 调用 mluOpPolyNms 的示例文件。
- fault_sample: 故障处理示例文件

**sample/abs_sample**
- abs_sample.cc: 调用 mluOpAbs 的示例文件；
- build.sh: 自动化编译脚本，其内部对cmake命令进行了封装；
- CMakeLists.txt: cmake 描述文件， 用于编译样例；
- run.sh: 自动化运行脚本。

**sample/poly_nms_sample**
- poly_nms_sample.cc: 调用 mluOpPolyNms 的示例文件；
- build.sh: 自动化编译脚本，其内部对cmake命令进行了封装；
- CMakeLists.txt: cmake 描述文件， 用于编译样例；
- run.sh: 自动化运行脚本。

**sample/fault_sample**
- fault_demo.mlu: fault_sample的host代码文件；
- fault_kernel.h: kernel的头文件；
- fault_kernel.mlu: fault_sample的device端代码文件；
- build.sh: 自动化编译脚本，其内部对cmake命令进行了封装；
- CMakeLists.txt: cmake 描述文件， 用于编译样例；
- run.sh: 自动化运行脚本。

运行样例：
- 全部样例的编译与运行：
```
   source env.sh  # 设置环境变量，指定库路径
   ./build.sh
   ./build/bin/abs_sample [dims_vaule] [shape0] [shape1] [shape2] ... # 运行abs_sample示例
   ./build/bin/poly_nms_sample  # 运行poly_nms_sample示例
```

- abs_sample 编译与运行
```
   ## 编译
   source env.sh  # 设置环境变量，指定库路径
   cd abs_sample
   ./build.sh     # 运行自动化编译脚本，编译abs_sample，生成的可执行文件会存放在 build/test 目录下

   ## 运行
   ./run.sh  # 运行 abs_sample
   ## 或
   cd build/bin 
   ./abs_sample [dims_vaule] [shape0] [shape1] [shape2] ... # 运行abs_sample
   # 运行示例
   ./abs_sample 4 10 10 10 10
   ./abs_sample 3 10 5 6
   ./abs_sample 2 8 8
```

- poly_nms_sample 编译与运行
```
    ## 编译
   source env.sh  # 设置环境变量，指定库路径
   cd poly_nms_sample
   ./build.sh     # 运行自动化编译脚本，编译abs_sample，生成的可执行文件会存放在 build/test 目录下

   ## 运行
   ./run.sh  # 运行 poly_nms_sample
   或
   cd build/bin  
   ./poly_nms_sample   # 运行示例
```

**sample/fault_sample**:

故障处理示例文件目录，其中代码可以引发典型的mlu unfinished错误，用于验证用户手册中调试方法的可行性。

具体可以参考BANGC OPS用户手册中“调试方法”一章的“MLU Unfinished问题定位”一节。
- 文件介绍
1. fault_demo.cc：运行代码样例，执行后将会引发mlu unfinished错误。
2. fault_kernel.h：可以引发mlu unfinished的kernel代码。
3. fault_kernel.mlu：可以引发mlu unfinished的kernel代码。

- fault_sample 编译与运行
```
    ## 编译
   source env.sh  # 设置环境变量，指定库路径
   cd fault_sample
   ./build.sh     # 运行自动化编译脚本，编译abs_sample，生成的可执行文件会存放在 build/test 目录下

   ## 运行
   ./run.sh  # 运行 fault_sample
   或
   cd build/bin  
   ./fault_sample   # 运行示例
```
- 运行流程：

1. 参照用户手册中“部署BANGC OPS”一节中内容配置CNtookit，获得neuware文件夹。
2. export NEUWARE_HOME=/path/to/your/neuware
3. 执行source env.sh
4. 执行./build.sh，默认编译所有样例代码。
5. 可执行文件路径为：sample/build/bin，执行./fault_sample 即可引发mlu unfinished错误，之后可以依照用户手册中“调试方法”一章中“MLU Unfinished问题定位”一节中提供的方法进行定位，以验证用户手册中所提供方法的可行性。
