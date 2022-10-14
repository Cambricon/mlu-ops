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
# pylint: disable=too-many-locals, assignment-from-no-return
"""Invoking Kernel with Parameters from Parser"""
import os
from pathlib import Path
import sys
import bangpy
from .parser import Parser
from .optest import OpTestFactory


def execuate_kernel(proto_file, target: str, op_name):
    """execuate_kernel is used for invoking kernel and comparing
    difference between mlu result and prototxt result.
    """
    data = Parser(proto_file, op_name)
    cambricon_dir = str(Path(__file__).parent.parent.resolve())
    sys.path.append(cambricon_dir)
    for register_op_name in op_name:
        register_op_dir = Path(cambricon_dir + "/ops/" + register_op_name)
        register_op_pys = register_op_dir.rglob("*.py")
        for file in register_op_pys:
            filedir = str(file.resolve())
            filename = file.name
            if not filename.startswith("_"):
                reldir = os.path.relpath(filedir, cambricon_dir)
                modulename = os.path.splitext(reldir)
                importname = str(modulename).replace("/", ".")
                __import__(importname)
    # get input and output tensors
    inputs, output = data.get_inp_oup()
    op_name = data.get_opname()
    if data.get_output_dtype() == "DTYPE_FLOAT":
        dtype = bangpy.float32
    elif data.get_output_dtype() == "DTYPE_HALF":
        dtype = bangpy.float16
    elif data.get_output_dtype() == "DTYPE_INT":
        dtype = bangpy.int32
    else:
        raise TypeError("Unsupported data type %s" % (data.get_output_dtype()))
    # launch kernel in factory model
    op_test = OpTestFactory.factory(op_name)(target, dtype, inputs, output)
    op_test.compute()
