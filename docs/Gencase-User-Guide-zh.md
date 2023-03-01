## MLU-OPS GEN_CASE 使用指南

## 概述

`gen_case` 是每次调用任一算子接口 api 时，根据 api 中形参数，生成一份该算子的 `prototxt` 测例，实现该功能的两点收益：

1）在框架跑一遍网络后，便可生成该网络调用到的各算子的所有规模测例，方便收集真实网络的规模数据，作为性能测试的数据集。

2）当框架调用网络时因为算子错误报错，或算子单元测试失败时，我们可以开启自动生成测例功能，获取会导致算子执行失败的测例规模与参数、更方便定位问题。

相比 `generator`，`gen_case` 是在运行算子时保存测试用例，`generator` 是运行算子前创建测试用例供算子使用。在 MLUOPS 工程中，`GEN_CASE` 主要以`gen_case.h` 和 `gen_case.cpp` 两个文件添加到工程中，文件中涉及到的数据结构或是类型，主要存在于 `mlu_op_core.h`、`core` 文件夹和 `mlu_op_test.proto` 中。

### 1. 算子中调用 GEN_CASE

#### 1.1 生成算子测例 `prototxt` 文件

1) 在测试机上打开环境变量
   ```
   export MLUOP_GEN_CASE = 1
   ```
2) 执行
   ```
   cd build/test/
   ./mluop_gtest --gtest_filter = *abs*
   ```
3) 结果
   会在当前目录下生成 gen_case/abs 文件夹，在文件夹里的 *.prototxt 文件保存了算子测试过程中测例规模

#### 1.2 GEN_CASE 环境变量说明

|          环境变量             |                  功能                                                 |    默认状态         |
|-------------------------------|-----------------------------------------------------------------------|---------------------|
|MLUOP_GEN_CASE                 |export MLUOP_GEN_CASE = 0: 关闭gen_case模块功能;<br>export MLUOP_GEN_CASE = 1: 生成 prototxt, 输入输出只保留 shape 等信息(GEN_CASE_DATA_REAL将无效);<br>export MLUOP_GEN_CASE = 2: 生成 proto, 并保留输入真实值;<br>export MLUOP_GEN_CASE = 3: 不生成 prototxt, 只在屏幕上打印输入输出的shape等信息。                                                          |     默认 0          |
|MLUOP_GEN_CASE_OP_NAME         |export MLUOP_GEN_CASE_OP_NAME = "算子A; 算子B……": 指定只使能算子 A/B……的 gen_case 功能;<br>export MLUOP_GEN_CASE_OP_NAME = "-算子A; -算子B……": 指定只不使能算子 A/B……的 gen_case 功能。                    | 默认全部算子使能     |
|MLUOP_GEN_CASE_DUMP_DATA       |在MLUOP_GEN_CASE = 2时生效;<br>export MLUOP_GEN_CASE_DUMP_DATA = 0: prototxt 中不保存输入的真值(此时的GEN_CASE_DATA_REAL有效);<br>export MLUOP_GEN_CASE_DUMP_DATA = 1: prototxt 中保存输入的文本形式真值;<br>export MLUOP_GEN_CASE_DUMP_DATA = 2: prototxt 中保存输入的二进制真值。                                                                         |     默认 0          |
|MLUOP_GEN_CASE_DUMP_DATA_OUTPUT|export MLUOP_GEN_CASE_DUMP_DATA_OUTPUT = 0: prototxt 中不保存 mlu 的输出值;<br>export MLUOP_GEN_CASE_DUMP_DATA_OUTPUT = 1: prototxt 中保存文本形式的 mlu 输出值;<br>export MLUOP_GEN_CASE_DUMP_DATA_OUTPUT = 2: prototxt 中保存二进制形式的 mlu 输出值。                                                                                       |     默认 0           |
|MLUOP_GEN_CASE_DUMP_DATA_FILE  |在 MLUOP_GEN_CASE = 2时生效;<br>export MLUOP_GEN_CASE_DUMP_DATA_FILE = 0: 保存方式以 MLUOP_GEN_CASE_DUMP_DATA 为准 export MLUOP_GEN_CASE_DUMP_DATA_FILE = 1: 真实值以一个二进制文件单独存储, prototxt 文件中保存 path。 |      默认 0          |

### 2. 算子中添加 GEN_CASE 功能

以上内容主要是面向使用者，如何使用 gen_case 来获取算子测试用例，这里将面向算子开发着如何去写代码调用 gen_case 的模块生成算子测例。

#### 2.1 GEN_CASE功能模块说明

gen_case宏函数说明，gen_case.h 主要定义了多个宏函数，在使用的时候主要利用这些宏来生成测试用例。

|         宏函数名             |                             函数功能说明                                           |接口数据|   数据类型    |
|------------------------------|------------------------------------------------------------------------------------|--------|---------------|
|`GEN_CASE_START`(op_name)     |根据输入的 op_name，在当前目录下创建 gen_case/op_name 文件夹，并在文件夹里创建名为 "op_name_时间.prototxt文件"，同时将 op_name 写入文件开头|op_name|std::string|
|`GEN_CASE_DATA`(is_input, id,<br>data, data_desc, upper_bound,<br>lower_bound)|根据给入的参数对 prototxt 文件写入 input{} 结构或者 output{} 结构。|is_input<br>id<br>data<br>data_desc<br>upper_bound<br>lower_bound|bool<br>std::string<br>void*<br>mluOPTensorDescriptor_t<br>double<br>double|
|`GEN_CASE_DATA_v2`(is_input, id,<br>data, data_desc, upper_bound,<br>lower_bound, distribution)|根据给入的参数对 prototxt 文件写入 input{}结构或者 output{}结构。|is_input<br>id<br>data<br>data_desc<br>upper_bound<br>lower_bound<br>distribution|bool<br>std::string<br>void*<br>mluOPTensorDescriptor_t<br>double<br>double<br>std::string|
|`GEN_CASE_DATA_UNFOLD`(is_input,<br>id,data,dim,dims,dtype,layout,<br>upper_bound, lower_bound)|根据给入的参数对 prototxt 文件写入 input{}结构或者 output{}结构,将 `data_desc` 展开成 dim, dims, dtype 和 layout|is_input<br>id<br>data<br>dim<br>dims<br>dtype<br>layout<br>upper_bound<br>lower_bound|bool<br>std::string<br>void*<br>const int<br>std::vector<int>/int*<br>mluOPDataType_t<br>mluOPTensorLayout_t<br>double<br>double|
|`GEN_CASE_DATA_UNFOLD_v2`(is_input,<br>id, data, dim,dims, dtype, layout,<br>upper_bound, lower_bound, distribution)|相比 GEN_CASE_DATA_UNFOLD 增加了 distribution 参数，用来表示生成数据服从的分布|is_input<br>id<br>data<br>dim<br>dims<br>dtype<br>layout<br>upper_bound<br>lower_bound<br>distribution|bool<br>std::string<br>void*<br>const int<br>std::vector<int>/int*<br>mluOPDataType_t<br>mluOPTensorLayout_t<br>double<br>double<br>std::string|
|`GEN_CASE_OP_PARAM_SINGLE`(flag,<br>param_node_name, param_name,<br>value)|根据传入的参数生成 op_name_param:{} 数据结构，其中该函数要求参数必须是单个数据。|flag<br>param_node_name<br>param_name<br>value|int<br>	std::string<br>	std::string<br>基础数据类型|
|`GEN_CASE_TEST_PARAM`(is_diff1,<br>is_diff2, is_diff3, diff1_threshould,<br>diff2_threshold, diff3_threshold…)|根据传入的参数决定测例的 diff 模式和精度误差阈值|is_diff1<br>is_diff2<br>is_diff3<br>diff1_threshold<br>diff2_threshold<br>diff3_threshold<br>diff1_threshold_imag<br>diff2_threshold_imag<br>diff3_threshold_imag|bool<br>bool<br>bool<br>const float<br>const float<br>const float<br>const float<br>const float<br>const float|
|`GEN_CASE_DATA_REAL`(is_input,<br>id,data, data_desc)|根据给入的参数对 prototxt 文件写入 input{} 结构，并保存 input tensor 里的真实数据|is_input<br>id<br>data<br>data_desc|bool<br>std::string<br>void*<br>mluOpTensorDescriptor_t|
|`GEN_CASE_HANDLE`(handle)       |设置 handle|handle|mluOPHandle_t|
|`GEN_CASE_HANDLE_PARAM`()       |打印设置的 handle 中的信息|||
|`GEN_CASE_OP_PARAM_SINGLE_NAME`(flag,<br>param_node_name, param_name,<br>value)|如果 op_name 和参数名字不一致时，可以用这个函数设置|flag<br>param_node_name<br>param_name<br>value|int<br>	std::string<br>	std::string<br>基础数据类型|
|`GEN_CASE_OP_PARAM_SINGLE_SUB`(flag,<br>param_node_name, param_name,<br>value, new_child)|如果参数又嵌套子参数，可以用这个函数这设置|flag<br>param_node_name<br>param_name<br>value<br>new_child|int<br>std::string<br>std::string<br>基础数据类型<br>bool|
|`GEN_CASE_OP_PARAM_ARRAY`(flag,<br>param_node_name, param_name,<br>value, num)|根据传入的参数生成 op_name_param:{}结构。其中该函数要求参数必须是数组类型|flag<br>param_node_name<br>param_name<br>value[]<br>num|int<br>std::string<br>std::string<br>基础数据类型<br>const int|
|`GEN_CASE_END`()                |恢复gen_case相关状态|||

#### 2.2 GEN_CASE功能模块开发

以下内容以 mlu-ops 仓库中 roi_crop_forward 算子为例在 host 端进行 gen_case 功能代码开发

1）添加头文件

在 bangc-docs/kernel/算子名文件夹/op_name.cpp 文件下添加头文件
```
#include "core/context.h"
#include "core/gen_case.h"
```

2）功能代码编写

```
mluOpStatus_t MLUOP_WIN_API mluOpRoiCropForward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const mluOpTensorDescriptor_t grid_desc,
    const void *grid, const mluOpTensorDescriptor_t output_desc, void *output) {
    
    ……

    if (MLUOP_GEN_CASE_ON_NEW)
    {
        GEN_CASE_START("roi_crop_forward");
        GEN_CASE_HANDLE(handle);
        GEN_CASE_DATA(true, "input", input, input_desc, -10, 10);
        GEN_CASE_DATA(true, "grid", grid, grid_desc, -1, 1);
        GEN_CASE_DATA(false, "output", output, output_desc, 0, 0);
        GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
    }
 
    cnrtDim3_t k_dim;
    cnrtFunctionType_t k_type;
    
    policyFunc(handle, bin_num, &k_dim, &k_type);
    VLOG(5) << "[mluOpRoiCropForward] launch kernel policyFunc[" << k_dim.x
            << ", " << k_dim.y << ", " << k_dim.z << "].";
    
    KERNEL_CHECK((mluOpBlockKernelRoiCropForwardFloat(
        k_dim, k_type, handle->queue, input, grid, batch, height, width, channels,
        grid_n, output_h, output_w, output)));
    VLOG(5) << "Kernel mluOpBlockKernelRoiCropForwardFloat.";
    
    GEN_CASE_END();
    
    return MLUOP_STATUS_SUCCESS;
}

```