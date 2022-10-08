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

**sample/abs_sample**
- abs_sample.cc: 调用 mluOpAbs 的示例文件；
- build.sh: 自动化编译脚本，其内部对cmake命令进行了封装；
- CMakeLists.txt: cmake 描述文件， 用于编译样例；
- run.sh: 自动化编译运行脚本。

**sample/poly_nms_sample**
- poly_nms_sample.cc: 调用 mluOpPolyNms 的示例文件；
- build.sh: 自动化编译脚本，其内部对cmake命令进行了封装；
- CMakeLists.txt: cmake 描述文件， 用于编译样例；
- run.sh: 自动化编译运行脚本。

**sample/fault_sample**
- fault_demo.mlu: fault_sample的host代码文件；
- fault_kernel.h: kernel的文件；
- fault_kernel.mlu: fault_sample的device端代码文件；
- build.sh: 自动化编译脚本，其内部对cmake命令进行了封装；
- CMakeLists.txt: cmake 描述文件， 用于编译样例；
- run.sh: 自动化编译运行脚本。

运行样例：
- 全部样例的编译与运行：
```
   source env.sh  # 设置环境变量，指定库路径
   ./build.sh
   ./build/bin/abs_sample [dims_vaule] [shape0] [shape1] [shape2] ... # 运行abs_sample示例
   ./build/bin/poly_nms_sample  # 运行poly_nms_sample示例
```

- abs_sample编译与运行
```
   ## 方式1
   source env.sh  # 设置环境变量，指定库路径
   cd abs_sample
   ./build.sh     # 运行自动化编译脚本，编译abs_sample，生成的可执行文件会存放在 build/test 目录下
   cd build/bin 
   ./abs_sample [dims_vaule] [shape0] [shape1] [shape2] ... # 运行abs_sample
   # 运行示例
   ./abs_Sample 4 10 10 10 10
   ./abs_Sample 3 10 5 6
   ./abs_Sample 2 8 8

   ## 方式2
   source env.sh
   cd abs_sample
   ./run.sh  # 编译和运行abs_sample
```

- poly_nms_sample编译与运行
```
   ## 方式1
   source env.sh  # 设置环境变量，指定库路径
   cd poly_nms_sample
   ./build.sh     # 运行自动化编译脚本，编译abs_sample，生成的可执行文件会存放在 build/test 目录下
   cd build/bin  
   ./poly_nms_sample   # 运行示例

    ## 方式2
   source env.sh
   cd poly_nms_sample
   ./run.sh  # 编译和运行poly_nms_sample
```

**sample/fault_sample**:

故障处理示例文件目录，其中代码可以引发典型的mlu unfinished错误，用于验证用户手册中调试方法的可行性。

具体可以参考CNNL用户手册中“调试方法”一章的“MLU Unfinished问题定位”一节。

- fault_demo.cc：运行代码样例，执行后将会引发mlu unfinished错误。
- fault_kernel.h：可以引发mlu unfinished的kernel代码。
- fault_kernel.mlu：可以引发mlu unfinished的kernel代码。

运行流程：

1. 参照用户手册中“部署CNNL”一节中内容配置CNtookit，获得neuware文件夹。
2. export NEUWARE_HOME=/path/to/your/neuware
3. 执行source env.sh
4. 执行./build.sh，默认编译所有样例代码。
5. 可执行文件路径为：sample/build/bin，执行./fault_sample 即可引发mlu unfinished错误，之后可以依照用户手册中“调试方法”一章中“MLU Unfinished问题定位”一节中提供的方法进行定位，以验证用户手册中所提供方法的可行性。
