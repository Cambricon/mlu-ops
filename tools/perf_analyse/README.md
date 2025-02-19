# README

## 0 文档说明

对于Perf_Analyse每次API更新，维护：

- [README](http://gitlab.software.cambricon.com/neuware/mlu-ops/-/blob/master/tools/Perf_Analyse/README.md)
- [用户/开发文档](http://wiki.cambricon.com/pages/viewpage.action?pageId=137672096)
- [更新日志](http://wiki.cambricon.com/pages/viewpage.action?pageId=137673594)

关于性能数据库的设计，见[性能数据库设计文档](http://wiki.cambricon.com/pages/viewpage.action?pageId=133040673)。

有啥问题欢迎[提issue](http://gitlab.software.cambricon.com/neuware/mlu-ops/-/issues)。

## 1 Getting Started

**使用-h选项均可查看脚本参数的用法**

### gtest_log_to_xlsx.py

用于解析gtest生成的文件，输入支持xml和log格式。输出支持基本模式，tpi模式，simple_tpi模式；生成excel文件以及tar文件，可选更新到数据库；支持两个文件对比，生成对比的excel和图片。

##### 输入模式

根据输入的文件数量决定是否进行文件对比。相关参数有log_path, compare_path。

##### 输出模式

  1. **基本模式**：此模式下会生成excel文件，包含所有case的详细信息，逐算子/网络的统计信息等。可生成比对图片。
  2. **tpi模式**：此模式下会生成excel和tar文件，包含逐网络/框架的统计信息和各网络下的case信息。tpi名词解释见黄区wiki 63238625。
  3. **simple_tpi模式**：此模式基于tpi模式，只对重要网络进行处理。生成excel文件，包含逐网络的统计信息和所有网络中device时长占比前20的算子信息。

### prototxt_to_excel.py

将文件夹中的case解析成excel表格，方便分析gencase抓出来的case内容。同时支持prototxt和pb格式的case。

### analyse_compile_time.py

分析CMake的log，解析编译生成.mlu/.o文件的时间。（目前仅支持Ninja）

### so_analyser.py

解析libmluops.so的大小。支持两个文件对比。

### h5_creator.py

基于网络的json配置文件生成h5文件，用于批量添加多个网络case。


## 2 Prerequisites

运行python脚本需要依赖第三方包，可通过virtualenv进行安装。

virtualenv需要使用内部pip镜像源，在文件~/.pip/pip.conf（若不存在则手动创建）中添加以下内容：

```
[global]
index-url = http://mirrors.cambricon.com/pypi/web/simple
find-links = /opt/shared/tensorflow/tf-python-pkgs
trusted-host = mirrors.cambricon.com
```

第三方包安装步骤如下：
1. `cd ~` (推荐安装到自己的home目录下)
2. `virtualenv venv --python=python3` 或者 `python3 -m venv venv`
3. `source ~/venv/bin/activate`
4. `pip install -r {your_mluops_path}/tools/Perf_Analyse/requirements.txt`

## 3 Usage Example

### analysis_suite.py

本文件为后续的各子模块提供统一的接口。

```bash
python3 analysis_suite.py -h # 查看帮助文档，可使用positional arguments中的选项选择子模块
python3 analysis_suite.py compile_time -h # 同 `python3 analyse_compile_time.pye -h` （暂未支持）
python3 analysis_suite.py gteste -h # 同 `python3 gtest_analyser.pye -h`
python3 analysis_suite.py h5e -h # 同 `python3 h5_creator.pye -h`
python3 analysis_suite.py pt2excele -h # 同 `python3 prototxt_to_excel.pye -h` （暂未支持）
python3 analysis_suite.py soe -h # 同 `python3 so_analyser.pye -h`
```

### analyse_compile_time.py

构建时添加选项`-g Ninja -v`

```bash
./independent_build.sh -g Ninja -v .... 2>&1 | tee compile.log
```

分析Ninja log（注意此时在mluops的目录下）

```
python3 ./tools/Perf_Analyse/analyse_compile_time.py  --log-path /DEV_SOFT_TRAIN/(user)/compile.log  --working-directory $PWD/build
```

### gtest_analyser.py

以下只列出常用选项。本模块大部分选项均可独立使用，若有依赖会指出。具体可-h查看帮助文档。

#### 基本模式

##### 分析gtest生成的xml文件

默认使用数据库

```bash
python gtest_log_to_xlsx.py  --log_path=output.xml
```

##### 分析gtest生成的xml文件（不使用数据库）

在xml中case少的时候会比使用数据库快

```bash
python gtest_log_to_xlsx --log_path=conv.xml --use_db 0
```

##### 分析gtest生成的xml文件并跳过解析pt/pb文件（不使用数据库）

会缺失input/output/params字段，在数据库缺失的case太多且解析pt/pb文件太太慢时可以用。

```bash
python3 gtest_analyser.py --log_path output.xml  --need_case_info False
```

##### 分析gtest_repeat生成的一组xml文件

假设`xmls_dir`路径下是`--gtest_repeat`模式生成的文件：`590_0.xml`、`590_1.xml`、`590_2.xml`……

```bash
python gtest_log_to_xlsx.py  --log_path=xmls_dir
```

##### 分析gtest生成的xml文件，过滤失败case并将它们导出

```bash
python gtest_log_to_xlsx --log_path=conv.xml --filter_failed_cases True --export_failed_cases True
```

##### 根据指定算子进行筛选

开发中...

##### 比较gtest生成的两个文件（不生成信息对比图）

考虑到生成图片比较耗时，建议加`--generate_pic False`选项避免生成对比信息图。

```bash
python3 gtest_analyser.py --log_path new.xml --compare_path baseline.xml --generate_pic 0
```

1. 有时会想要上面baseline.xml的信息，目前未加参数支持，可以再调一次该脚本分析baseline.xml

```bash
python3 gtest_analyser.py --log_path baseline.xml
```

2. 关于比较的指标，目前涉及的指标通常为device time、host time和device space。均为越小越好的，故计算方式为

```
promotion = baseline - new
promotion_ratio = promotion / baseline
```

##### 比较gtest生成的两个文件（生成对比信息图）

直接比较即可，默认生成图片。


```bash
python gtest_log_to_xlsx.py  --log_path=conv_new.xml --compare_path=conv_baseline.xml --generate_pic 1
```

##### 比较gtest生成的两个文件（不使用数据库，不生成信息对比图）


```bash
python3 gtest_analyser.py --log_path new.xml --compare_path baseline.xml --generate_pic 0  --use_db 0
```

#### TPI模式

##### TPI模式下，分析gtest生成的xml文件

```bash
python3 gtest_analyser.py  --log_path output.xml --tpi
```

##### TPI模式下，对比gtest生成的两个文件

```bash
python gtest_log_to_xlsx.py --log_path new.xml --compare_path baseline.xml --tpi
```

#### simple TPI模式（基于TPI模式）

##### simple TPI模式下，分析gtest生成的xml文件

```bash
python3 gtest_analyser.py  --log_path output.xml --tpi --simple_tpi
```

##### simple TPI模式下，分析gtest生成的xml文件，并按框架筛选

```bash
python3 gtest_analyser.py  --log_path output.xml --tpi --simple_tpi --framework "pt1.13"
```

> --frameworks取值只能是"pytorch", "tf", "pt1.13"中的一种或若干种，默认为"pytorch"。若有多种则用逗号","隔开，如"pytorch,tf"。

##### simple TPI模式下，对比gtest生成的两个文件

```bash
python3 gtest_analyser.py  --log_path new.xml --compare_path baseline.xml --tpi --simple_tpi --frameworks pt1.13
```

---------
### h5_creator.py

基于网络的json配置文件生成h5文件。

文件结构：

```
<cases_dir>
├─mluops_benchmark_config.json
└─<op_name>/
```

**基于case_dir生成h5文件**

```bash
python3 h5_creator.py --case_dir <case_dir>
```

----

### prototxt_to_excel.py

#### 分析case文件夹

不指定excel的文件夹路径，此时excel默认保存在当前文件夹下，名为op_tensor.xlsx

```bash
python prototxt_to_excel.py --case_path=/SOFT_TRAIN/test/op_tensor_five_dim/op_tensor
```

指定excel的保存路径，结果文件在test内

```bash
python prototxt_to_excel.py --case_path=/SOFT_TRAIN/test/op_tensor_five_dim/op_tensor  --xlsx_path=/projs/mluops/(名字)/mluops/tools/Perf_Analyse/test/tensor.xlsx
```

------

### so_analyser.py

#### 解析libmluops*.so*的大小

```bash
python3 so_analyser.py --so_path libmluops_0.so
```

#### 对比两个libmluops*.so*文件的大小

```bash
python3 so_analyser.py --so_path libmluops_0.so --so_path_compare libmluops_1.so
```

------
