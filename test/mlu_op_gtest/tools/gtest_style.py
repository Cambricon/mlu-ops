#!/usr/bin/python

import os
import sys
import re

# 1.don't use malloc or new in test code(check field), use shared_ptr or cpu_runtime instead.
# 2.don't use CNRT_CHECK/ CHECK, use MLUOP_CHECK or GTEST_CHECK instead.

# check field
check_func = ["paramCheck()", "compute()", "cpuCompute()", "workspaceMalloc()", "workspaceFree()", "diffPreprocess()"]
smart_ptr = ["shared_ptr", "unique_ptr", "auto_ptr", "weak_ptr", "reset"]

# get begin/end of function need to check.
def find_func(lines):
    res = []
    for func in check_func:
        func_begin = 0
        func_end   = 0
        for line_id in range(0, len(lines)):
            if func in lines[line_id]:
                func_begin = line_id
                break
        for line_id in range(func_begin, len(lines)):
            if lines[line_id][0] == "}":
                func_end = line_id
                break
        if func_begin != 0 and func_end != 0:
            res.append([func_begin, func_end])
    return res

def in_func(line_id, func_mark):
    for mark in func_mark:
        if mark[0] <= line_id and line_id <= mark[1]:
            return True
    return False

def get_file(root_path, all_files=[]):
    files = os.listdir(root_path)
    for it in files:
        current = os.path.join(root_path, it)
        if not os.path.isdir(current):
            all_files.append(current)
        else:
            get_file(current, all_files)
    return all_files

def gather_file(path):
    files = []
    get_file(path, files)
    return [i for i in files if ".cpp" in i]

# only allowed shared_ptr and cpu_runtime
def is_smart_ptr(line):
    return any(i in line for i in smart_ptr)

def check_context(line):
    context = ""
    if line.find("//") != -1:
        context = line[0:line.find("//")]  # get words before //
    seg = line.split("\"")   # get word not in ""
    for i in range(0, len(seg)):
        if i%2 == 0:
            context += seg[i]
    return context

def check_malloc(line):
    line = check_context(line)
    if ("new " in line or "malloc(" in line) and (not "allocate(" in line and not is_smart_ptr(line)):
        return True
    elif "cnrtMalloc(" in line and not "allocate(" in line:
        return True
    else:
        return False

def check_macro(line):
    line = check_context(line)
    if " CNRT_CHECK(" in line:
        return True
    elif " CHECK(" in line:
        return True
    else:
        return False

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))
    zoo_path = os.path.join(file_path, "../pb_gtest/src/zoo")

    all_cpps = []
    if len(sys.argv) == 1:
        all_cpps = gather_file(zoo_path)
        all_cpps = sorted(all_cpps)
    else:
        # build specific operator only
        for i in range(1, len(sys.argv)):
            op_path = os.path.join(zoo_path, sys.argv[i])
            if (os.path.exists(op_path)):
                all_cpps = gather_file(op_path)
                all_cpps = sorted(all_cpps)

    ok = True
    macro_error = []
    malloc_error = []
    for cpp in all_cpps:
        print("Checking %s \n" %(cpp))
        fin = open(cpp, "r")
        lines = fin.readlines()
        func_mark = find_func(lines)
        if len(func_mark) == 0:
            continue

        # print("Found function:\n")
        # print(func_mark)

        # check each line
        for line_id in range(0, len(lines)):
            line = lines[line_id]
            if check_malloc(line):
                if in_func(line_id, func_mark):
                    malloc_error.append([cpp, line_id])
                    ok = False
                else:
                    # print(warning maybe)
                    pass

            if check_macro(line):
                macro_error.append([cpp, line_id])
                ok = False
        fin.close()

    if ok:
        exit(0)
    else:
        if len(malloc_error) != 0:
            print("-- ERROR: malloc/new, please use smart ptr instead.")
            for e in malloc_error:
                print("%s +%d" %(e[0], e[1] + 1))

        if len(macro_error) != 0:
            print("-- ERROR: CHECK/CNRT_CHECK, please use GTEST_CHECK instead.")
            for e in macro_error:
                print("%s +%d" %(e[0], e[1] + 1))
        exit(1)
