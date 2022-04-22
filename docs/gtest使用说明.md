# Gtest测试框架使用说明

[TOC]

# 								

##　 1.mlu-ops gtest 介绍

​		mlu-ops gtest(以下简称 gtest)是 mlu-ops 集成的自动测试框架，算子开发者在开发完算子任务之后，可通过一定的规则修改算子对应的配置文件，

即添加 gtest 来进行大规模的算子性能，稳定性测试。

## 2.添加 gtest 测例

​		确认已根据 BANGC-OPS 算子开发流程文档，添加算子的 gtest。
添加完 gtest 之后我们需要通过修改配置文件，给算子自动生成测例，来进行算子测试，目前 mlu-ops 支持 prototxt 和 pb 两种不同格式的测例，并且支持 prototxt 和 pb 文件之间的相互转换。

### 2.1 prototxt 测例

这里以 div 算子为例：

测例路径：mlu-ops/bangc-ops/test/mlu_op_gtest/src/zoo/div/test_case

```
├── div
│   ├── div.cpp
│   ├── div.h
│   └── test_case
│       └── case_0.prototxt
```

case_0.prototxt 格式说明:

```
op_name: "div" //算子名
input {
  id: "input1"  //第一个输入
  shape: {
    dims: 128
    dims: 7
    dims: 7
    dims: 512
  }
  layout: LAYOUT_ARRAY
  dtype: DTYPE_FLOAT
  random_data: {   //随机数
    seed: 23
    upper_bound: 1
    lower_bound: 1
    distribution: UNIFORM
  }
}
input {
  id: "input2"  //第二个输入
  shape: {      //指定数据维度信息
    dims: 128
    dims: 7
    dims: 7
    dims: 512
  }
  layout: LAYOUT_ARRAY  //输入数据形状
  dtype: DTYPE_FLOAT   //dtype 输入数据类型
  random_data: { //随机数
    seed: 25
    upper_bound: 10
    lower_bound: 0.1
    distribution: UNIFORM
  }
}
output {
  id: "output"  //输出维度和输入保持一致
  shape: {
    dims: 128
    dims: 7
    dims: 7
    dims: 512
  }
  layout: LAYOUT_ARRAY  //输出数据形状
  dtype: DTYPE_FLOAT  //dtype 输出数据类型
}
test_param: {
  error_func: DIFF1
  error_func: DIFF2
  error_threshold: 0.003
  error_threshold: 0.003  //测试误差范围
  baseline_device: CPU
}
```

执行 prototxt 测试:

```bash
cd bangc-ops/build/test
./mluop_gtest --gtest_filter="*div*" //--gtest_filter指定具体的执行算子
```

这里就可以修改 prototxt 的维度信息，类型，取值范围，对算子进行各种输入条件下的测试。



### 2.2 pb 测例

​		pb 测例一般是将 tensorflow，pytorch 的算子计算结果通过 pb 文件的形式保存下来,同时记录算子的输入信息，做为 mlu-ops 相同算子的输入,计算得到 mlu 侧的结果和两者的误差等。

 执行 pb 测例:

```bash
cd bangc-ops/build/test
./mluop_gtest --gtest_filter="*div*" --cases_dir="./pb/"  //--gtest_filter指定具体的执行算子 --cases_dir指定pb测例的路径
```



### 2.3 prototxt 和 pb 互相转化

pb2prototxt

```bash
cd bangc-ops/build/test
./pb2prototxt case_0.pb case_0.prototxt  //第一个参数是pb路径,第二个参数是prototxt路径,支持文件夹批量转换
```

prototxt2pb

```
cd bangc-ops/build/test
./prototxt2pb case_0.prototxt case_0.pb  //第一个参数是prototxt路径,第二个参数是pb路径,支持文件夹批量转换
```

