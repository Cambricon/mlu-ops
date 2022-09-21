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
# pylint: disable=missing-docstring, invalid-name, too-many-locals, missing-module-docstring
"""A multi-platform code link example test for BANGPy TCP."""
from test import registerOp, OpTest
import numpy as np
import bangpy
from bangpy.common import load_op_by_type
from add import KERNEL_NAME


@registerOp("add")
class Addop(OpTest):
    def __init__(self, target, dtype, tensor_list, output_tensor):
        self.dtype = dtype
        super().__init__(target, dtype, tensor_list, output_tensor)

    def compute(self):
        data_in0 = self.inputs_tensor_list[0]
        data_in1 = self.inputs_tensor_list[1]
        dev = bangpy.device(0)
        # set I/O data
        data_in0_dev = bangpy.Array(data_in0.astype(self.dtype.as_numpy_dtype), dev)
        data_in1_dev = bangpy.Array(data_in1.astype(self.dtype.as_numpy_dtype), dev)
        data_out_dev = bangpy.Array(
            np.zeros(self.output_tensor_list[0].shape, self.dtype.as_numpy_dtype), dev
        )

        f1 = load_op_by_type(KERNEL_NAME, self.dtype.name)
        f1(data_in0_dev, data_in1_dev, data_out_dev, np.prod(data_in0.shape))
        evaluator = f1.time_evaluator(number=10, repeat=1, min_repeat_ms=0)
        run_time = evaluator(
            data_in0_dev, data_in1_dev, data_out_dev, np.prod(data_in0.shape)
        ).mean
        print("mlu run time: %ss" % str(run_time))
        return data_out_dev
