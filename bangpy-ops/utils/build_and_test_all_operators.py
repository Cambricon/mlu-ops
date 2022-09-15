# Copyright (C) [2021] by Cambricon, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# pylint: disable=missing-docstring, too-many-locals, missing-function-docstring
from test import test_op
import os
import sys


build_entrys = []
test_entrys = []
test_files = []


def is_build_func(name, obj):
    return callable(obj) and name.startswith("build")


def collect_build_funcs(op):
    dicts = [getattr(op, "__dict__", {})]
    build_exist = 0

    for dic in dicts:
        for name, obj in list(dic.items()):
            if is_build_func(name, obj):
                if dicts[0][name].__qualname__.find("register_mlu_op") != 0:
                    raise TypeError(
                        "Please use 'register_mlu_op' to decorate your build function in '%s.py'."
                        % (op.__name__)
                    )
                build_entrys.append(obj)
                build_exist = 1
    return build_exist


def build_all_op():
    print("======================")
    print("Build all operators...")
    for obj in build_entrys:
        obj(None, None)

def test_all_op(target, opname):
    print("======================")
    print("Test all operators...")
    test_op(target, opname)


def main():
    ops_path = "/".join(os.path.abspath(__file__).split("/")[:-2])
    ops_path += "/ops/"
    build_enable = True
    test_enable = True
    target = None
    oper_idx = 1
    if len(sys.argv) == 1:
        raise ValueError("Please input operators list.")
    if sys.argv[1] == "-b" or sys.argv[1] == "--build":
        test_enable = False
        oper_idx += 1
    elif sys.argv[1] == "-t" or sys.argv[1] == "--test":
        build_enable = False
        oper_idx += 1
        for arg in sys.argv[2:]:
            if arg.find("--target=") != -1:
                target = arg[arg.find("--target=") + len("--target=") :]
    if len(sys.argv) == 2 and oper_idx != 1:
        raise ValueError("Please input operators list.")

    operator_lists = sys.argv[oper_idx].split(",")
    operator_lists = [i for i in operator_lists if i != ""]
    cur_work_path = ops_path
    operator_statuts = {}

    for op in operator_lists:
        if operator_lists.count(op) > 1:
            raise ValueError(
                'Duplicate operator "%s", please check the input operators.' % (op)
            )

    for op_name in operator_lists:
        files = os.listdir(ops_path + op_name)
        status = 0
        if files:
            os.chdir(cur_work_path + op_name)
            for f in files:
                if f.endswith(".py"):
                    sys.path.append(os.getcwd())
                    a = __import__(f[:-3])
                    status |= collect_build_funcs(a)
                    sys.path.pop()
            os.chdir(cur_work_path)
        operator_statuts[op_name] = status

    if build_enable:
        build_all_op()
        for k, v in operator_statuts.items():
            if not v & 1:
                print(
                    "Build Warning: Operator %s was skipped, please check whether\
                     there is a function start with 'build' prefix in the operator."
                    % (k)
                )
    if test_enable:
        for op_name in operator_lists:
            test_all_op(target, op_name)
            for k, v in operator_statuts.items():
                if not v & 1:
                    print(
                        "Test Warning: Operator %s was skipped, please check whether\
                            there is a function start with 'test' prefix in the operator."
                        % (k)
                    )


if __name__ == "__main__":
    main()
