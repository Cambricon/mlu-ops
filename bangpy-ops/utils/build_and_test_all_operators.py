# Copyright (C) [2022] by Cambricon, Inc.
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
import os
import sys
from test import test_op
import pytest


build_entrys = []
test_entrys = []
test_files = []
pb_test_op = [
    "add",
    "logaddexp2",
    "kldivloss",
    "cross",
    "hard_sigmoid",
    "cosine_embedding_loss",
    "lerp",
    "frac",
    "hardshrink",
]


def collect_build_test_funcs(op, cur_file_name):
    dicts = [getattr(op, "__dict__", {})]
    build_exist = test_exist = 0

    for dic in dicts:
        for name, obj in list(dic.items()):
            if callable(obj) and name.startswith("build"):
                if dicts[0][name].__qualname__.find("register_mlu_op") != 0:
                    raise TypeError(
                        "Please use 'register_mlu_op' to decorate your build function in '%s.py'."
                        % (op.__name__)
                    )
                build_entrys.append(obj)
                build_exist = 1
            if callable(obj) and name.startswith("test"):
                test_entrys.append(obj)
                if test_files.count(cur_file_name) == 0:
                    test_files.append(cur_file_name)
                    test_exist = 2
    return build_exist | test_exist


def build_all_op():
    print("======================")
    print("Build all operators...")
    for obj in build_entrys:
        obj(None, None)


def test_all_op(target, opname, cases_dir):
    print("======================")
    print("Test all operators...")
    flag = False
    if opname in ["add"]:
        test_op(target, opname, cases_dir)
    else:
        flag = True
        if target is not None:
            pytest.main(["-s", "-x", "--target=" + target, *test_files])
        else:
            pytest.main(["-s", "-x", *test_files])
    return flag


def main():
    ops_path = "/".join(os.path.abspath(__file__).split("/")[:-2])
    ops_path += "/ops/"
    build_enable = True
    test_enable = True
    target = None
    cases_dir = None
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
            if arg.find("--cases_dir=") != -1:
                cases_dir = arg[arg.find("--cases_dir=") + len("--cases_dir=") :]
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
                    status |= collect_build_test_funcs(a, os.getcwd() + "/" + f)
                    sys.path.pop()
            os.chdir(cur_work_path)
        operator_statuts[op_name] = status

    if build_enable:
        build_all_op()
        for k, v in operator_statuts.items():
            if not v & 1:
                print(
                    "Build Warning: Operator %s was skipped, please check whether" % (k)
                    + " there is a function start with 'build' prefix in the operator."
                )
    if test_enable:
        print("======================")
        print("Test all operators with pb case...")
        for op_name in operator_lists:
            if op_name in pb_test_op:
                test_op(target, op_name, cases_dir)
        flag = False
        if len(test_files) != 0:
            flag = test_all_op(target, "", cases_dir)
        for k, v in operator_statuts.items():
            if k not in pb_test_op:
                if not v & 2 and flag is True:
                    print(
                        "Test Warning: Operator %s was skipped, please check whether"
                        % (k)
                        + " there is a function start with 'test' prefix in the operator."
                    )


if __name__ == "__main__":
    main()
