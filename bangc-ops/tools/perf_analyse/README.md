README
=================================

<!-- GETTING STARTED -->
## Getting Started

**使用-h选项均可查看脚本参数的用法**

* gtest_log_to_xlsx.py
    用于解析gtest生成的文件，支持xml和log格式，输出成excel文件，可选更新到数据库。支持两个文件对比，生成对比的excel和图片。

* prototxt_to_excel.py
    将case文件夹解析成excel表格，方便分析gencase抓出来的case内容支持prototxt及pb文件。

## Prerequisites

运行python脚本需要依赖第三方包，可通过virtualenv进行安装，需要使用内部pip镜像源，在文件~/.pip/pip.conf中添加以下内容，不存在则创建：<br>
[global]<br>
index-url = http://mirrors.cambricon.com/pypi/web/simple<br>
find-links = /opt/shared/tensorflow/tf-python-pkgs<br>
trusted-host = mirrors.cambricon.com<br>

第三方包安装步骤如下：
1. cd ~ (推荐安装到自己的home目录下)
2. virtualenv venv --python=python3 或者 python3 -m venv venv
3. source ~/venv/bin/activate
4. pip install -r {your_mluop_path}/tools/Perf_Analyse/requirements.txt

更多说明可见http://wiki.cambricon.com/pages/viewpage.action?pageId=76996704
<!-- USAGE EXAMPLES -->
## Usage Example

### gtest_log_to_xlsx.py

###### 分析gtest生成的xml文件
```Bash
python gtest_log_to_xlsx.py  --log_path=conv.xml
```

###### 分析gtest_repeat生成的一组xml文件

假设`xmls_dir`路径下是`--gtest_repeat`模式生成的文件：`290_0.xml`、`290_1.xml`、`290_2.xml`……
```Bash
python gtest_log_to_xlsx.py  --log_path=xmls_dir
```
###### 比较gtest生成的两个文件
```Bash
python gtest_log_to_xlsx.py  --log_path=conv_new.xml --compare_path=conv_baseline.xml
```

对于越小越好的指标，提升比例的计算公式为
```math
promotion\_ratio = (baseline - new) / baseline
```
###### 比较gtest_repeat生成的两组文件

假设`xmls_dir_new`和`xmls_dir_baseline`路径下是`--gtest_repeat`模式生成的文件：`290_0.xml`、`290_1.xml`、`290_2.xml`……
```Bash
python gtest_log_to_xlsx.py  --log_path=xmls_dir_new --compare_path=xmls_dir_baseline
```

---------
### prototxt_to_excel.py

###### 分析 /SOFT_TRAIN/test/op_tensor_five_dim/op_tensor 下多个case

```Bash
 python  prototxt_to_excel.py --case_path=/SOFT_TRAIN/test/op_tensor_five_dim/op_tensor #不指定excel的文件夹路径，此时excel默认保存在当前文件夹下,名为op_tensor.xlsx
 python  prototxt_to_excel.py --case_path=/SOFT_TRAIN/test/op_tensor_five_dim/op_tensor  --xlsx_path=/projs/mluOp/(名字)/mluOp/tools/Perf_Analyse/test/tensor.xlsx  #指定excel的保存路径,结果文件在test内
```
