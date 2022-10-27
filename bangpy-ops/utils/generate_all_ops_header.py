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
# pylint: disable=missing-module-docstring, missing-function-docstring
import os
import sys


def main():
    if len(sys.argv) == 1:
        raise ValueError("Please input at least one operator header file.")
    header_lists = sys.argv[1].split(",")
    header_lists = [i for i in header_lists if i != ""]
    build_path = os.environ.get("BANGPY_BUILD_PATH", "")
    if build_path == "":
        raise ValueError("Could not find BANGPY_BUILD_PATH environment variable.")
    build_path += "/" if build_path[-1] != "/" else ""

    with open(build_path + "mlu_ops.h", "w") as mlu_ops:
        for i, h in enumerate(header_lists):
            with open(h, "r") as one_opertor_header:
                lines = one_opertor_header.readlines()
                if i != 0:
                    lines[0] = "\n"
                mlu_ops.writelines(lines)


if __name__ == "__main__":
    main()
