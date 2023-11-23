#! /usr/bin/python3

# author: Wangguichun
# date: 2020/5/14
# eg: ./generate_case.py ../test/mlu_op_gtest/src/zoo/cycle_op/test_case/template


import sys
import os

def main():
    if (len(sys.argv) != 2):
        print("使用方法：./replace_case.py 模板文件")
        return

    template = ""
    dirname = os.path.dirname(sys.argv[1])
    title = []
    cases = []

    # 删除旧文件
    os.system('rm ' + dirname + '/case_*')
    # 解析模板文件
    with open(sys.argv[1], "r", encoding='utf-8') as template_file:
        template = template_file.read()

    # 解析ini文件
    try:
      with open(os.path.join(dirname, 'cases.ini'), "r", encoding='utf-8') as replace_file:
          all = replace_file.read()
          all = all.split("\n")
          title = all[0].split()
          for item in all[1:]:
              item = item.strip()
              if item == '': continue
              if item[0] == '#': continue
              if item[0] == ';': continue
              cases.append(item)
    except FileNotFoundError:
        print(os.path.join(dirname, 'cases.ini') + " 文件不存在")
        return



    print("Title: " + str(title))

    id = 0
    for case in cases:
        prototxt = template
        values = case.strip()
        for i in range(len(title)):
            value = ""
            # 处理多维度shape
            if "shape" in title[i]:
                if values[0] != "{":
                    print("shape 格式错误，请用{}包围")
                    return
                end = values.find("}") + 1
                shape = values[:end]
                dim_list = shape[1:-1].split()
                value = "{\n"
                for dim in dim_list:
                    value = value + "    dims: " + dim + "\n"
                value = value + "  }"
                values = values[end+1:]
            else:
                # 其他参数
                split = values.split()
                value = split[0]
                values = " ".join(split[1:])
            prototxt = prototxt.replace(title[i], value)
            # print(title[i] + ":" + value)
            # print("case_" + str(id) + ".prototxt:\n" + prototxt)
        with open(os.path.join(dirname, "case_" + "{0:04d}".format(id) + ".prototxt"), "w") as file:
            file.write(prototxt)
        print(str(case) + " \t\t==>  case_" + "{0:04d}".format(id) + ".prototxt 生成成功")
        id += 1

    print("Over")


if __name__ == "__main__":
    main()
    
