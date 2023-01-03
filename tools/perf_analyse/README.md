README
=================================

<!-- GETTING STARTED -->
## Getting Started

**使用-h选项均可查看脚本参数的用法**

* gtest_log_to_xlsx.py
    用于解析gtest生成的文件，支持xml，json和log格式，输出成excel文件，可选更新到数据库。支持两个log文件对比，生成对比的excel和图片。

## Prerequisites

运行 mlu-ops/tools/perf_analyse/ 下 python 脚本需要依赖第三方包，可通过 virtualenv 进行安装

第三方包安装步骤如下：
1. cd ~ (推荐安装到自己的home目录下)
2. virtualenv venv --python=python3
3. source ~/venv/bin/activate
4. pip install -r mlu-ops/tools/perf_analyse/requirements.txt

说明： 通过第 2 步后会在当前目录下生成 venv  目录，第三方包就安装在这里面，下次使用只需进行第 3 步激活环境，无需重新生成，如果 requirements.txt 文件有更新，请再次执行第 3、4步。

<!-- USAGE EXAMPLES -->
## Usage Example

### gtest_log_to_xlsx.py

###### 分析gtest生成的xml文件
```Bash
python gtest_log_to_xlsx.py  --log_path={path}/20221108/290.xml
```

###### 分析gtest_repeat生成的一组xml文件

假设`20221108`路径下是`--gtest_repeat`模式生成的文件：`290_0.xml`、`290_1.xml`、`290_2.xml`……
```Bash
python gtest_log_to_xlsx.py  --log_path={path}/20221108
```
###### 比较gtest生成的两个文件

```Bash
python gtest_log_to_xlsx.py  --log_path={path}/20221108/290.xml --compare_path={path}/20221210/290.xml
```
###### 比较gtest_repeat生成的两组文件

假设`20221108`和`20221210`路径下是`--gtest_repeat`模式生成的文件：`290_0.xml`、`290_1.xml`、`290_2.xml`……
```Bash
python gtest_log_to_xlsx.py  --log_path={path}/20221108 --compare_path={path}/20221210
```
