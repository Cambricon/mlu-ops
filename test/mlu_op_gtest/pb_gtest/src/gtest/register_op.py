#!/usr/bin/python

# configparser python2/python3
try:
    import ConfigParser as configparser
except:
    import configparser

from string import Template

import os
import sys
import re

def grep_info(ini_file_path):
    print("registering %s" % os.path.basename(ini_file_path))
    ini = configparser.ConfigParser()
    ini.read(ini_file_path)

    op_name = ini.get('meta', 'name')
    op_branches = [i.strip() for i in
        ini.get('meta', 'branches').split(",")]

    return (op_name, op_branches)

def upper_camel(op_name):
    res = ""
    slices = op_name.split('_')
    for item in slices:
        res += item.capitalize()
    return res

def get_black_list_op():
    return os.environ.get("MLUOP_BLACK_LIST_OP", "").strip().split(";")
if __name__ == "__main__":
    # grep all ini files in zoo/
    mlu_op_gtest_src_dir_path = os.path.dirname(os.path.abspath(__file__))
    zoo_path = os.path.join(mlu_op_gtest_src_dir_path, "../zoo")
    inis = []
    if len(sys.argv) == 1:
        inis = os.listdir(zoo_path)
        # inis = [os.path.join(mlu_op_gtest_src_dir_path, "zoo", f)
        #                      for f in inis if os.path.splitext(f)[1] == ".ini"]
        black_list = get_black_list_op()
        inis = sorted(filter(lambda x: x not in black_list, inis))
    else:
        # build specific operator only
        for i in range(1, len(sys.argv)):
          op_path = os.path.join(zoo_path, sys.argv[i])
          if (os.path.exists(op_path)) and not sys.argv[i] in inis:
            inis.append(sys.argv[i])
    print(inis)

    # generate mlu_op_gtest_case_list.cpp
    filename1 = os.path.join(mlu_op_gtest_src_dir_path, "template/case_list.tpl")
    filename2 = os.path.join(mlu_op_gtest_src_dir_path, "case_list.cpp")
    file1 = open(filename1, "r")
    file2 = open(filename2, "w")
    gen = False
    for eachline in file1:
      if "AUTO GENERATE START" in eachline:
        gen = True
        file2.write(eachline)
        for op in inis:
          addstr="INSTANTIATE_TEST_CASE_P(%s, TestSuite, Combine(Values(\"%s\"), Range(size_t(0), Collector(\"%s\").num())));\n" %(op, op, op)
          file2.write(addstr)
      elif "AUTO GENERATE END" in eachline:
        gen = False
        file2.write(eachline)
      elif gen == False:
        file2.write(eachline)

    file1.close()
    file2.close()
    print("Success generate case_list.cpp.")

    # generate op_register.py
    filename1 = os.path.join(mlu_op_gtest_src_dir_path, "template/op_register.tpl")
    filename2 = os.path.join(mlu_op_gtest_src_dir_path, "op_register.h")
    file1 = open(filename1, "r")
    file2 = open(filename2, "w")
    gen_header = False
    gen = False
    for eachline in file1:
      if "AUTO GENERATE HEADER START" in eachline:
        gen_header = True
        file2.write(eachline)
        for op in inis:
          addstr = "#include \"../zoo/%s/%s.h\"\n" % (op, op)
          file2.write(addstr)
      elif "AUTO GENERATE HEADER END" in eachline:
        gen_header = False
        file2.write(eachline)
      elif "AUTO GENERATE START" in eachline:
        gen = True
        file2.write(eachline)
        for op in inis:
          addstr = "  } else if (op_name == \"%s\") {\n" % (op)
          file2.write(addstr)
          addstr = "    return std::make_shared<mluoptest::%sExecutor>();\n" % (upper_camel(op))
          file2.write(addstr)
      elif "AUTO GENERATE END" in eachline:
        gen = False
        file2.write(eachline)
      elif gen == False and gen_header == False:
        file2.write(eachline)

    file1.close()
    file2.close()
    print("Success generate op_register.h.")
    exit()
