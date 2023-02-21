# h2rst

C++ H files to RST files auto-build tools

#### Install
> pip install -r requirements.txt

#### Usage
> python main.py <file1\> <file2\>

例子：python main.py ~/Documents/mluops.h ~/Documents/docs_h/doc/pipeline.hpp

指定输出文件目录：
> python main.py -o <directory\> <file1\> <file2\>

例子：python main.py -o ~/Documents ~/Documents/mluops.h ~/Documents/docs_h/doc/pipeline.hpp

##### 可修改 config.py 配置文件配置参数

datatype是否按字母排序：datatype_sort

api是否按字母排序：api_sort

是否添加引用标签：link_label

是否解析类的私有成员：get_private

是否解析类的保护成员：get_protected

是否解析宏定义：get_define

是否解析类中的成员变量：get_class_variable

index.rst中内容排序： api_index_sort

api文件的一级标题是否支持引用标签： api_title_link_label

是否对api分组： api_group

忽略掉的分组： hidden_groups

index.rst的标题: api_group_index_title

api.rst的标题: api_title

##### 命令行方式配置参数

api接口按字母排序：
> python main.py --api-s <file1\> <file2\>

例子：python main.py --api-s ~/Documents/mluops.h ~/Documents/docs_h/doc/pipeline.hpp

数据类型按字母排序：
> python main.py --dt-s <file1\> <file2\>

例子：python main.py --dt-s ~/Documents/mluops.h ~/Documents/docs_h/doc/pipeline.hpp

提取宏信息写入define.rst：
> python main.py -d <file1\> <file2\>

例子：python main.py -d ~/Documents/mluops.h ~/Documents/docs_h/doc/pipeline.hpp
