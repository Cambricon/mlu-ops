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
# The above copyright notice and this permission notice shall self.tcp included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Test Lerp operator with multi-platform code link."""
# pylint: skip-file

from test import registerOp, OpTest
import numpy as np
import bangpy
from bangpy.common import load_op_by_type
from lerp import KERNEL_NAME

def cal_diff(result, data_out):
    diff1 = np.sum(np.abs(np.subtract(result, data_out))) / np.sum(np.abs(data_out))
    diff2 = np.sqrt(
        np.sum(np.power(np.subtract(data_out, result), 2)) / np.sum(np.power(data_out, 2))
    )
    assert round(diff1 * 100, 5) < 3e-3 * 100
    assert round(diff2 * 100, 5) < 3e-3 * 100
    print("DIFF1:", str(round(diff1 * 100, 5)) + "%")
    print("DIFF2:", str(round(diff2 * 100, 5)) + "%")

@registerOp("lerp")
class LerpOp(OpTest):
    def __init__(self, target, dtype, tensor_list, output_tensor, params):
        self.dtype = dtype
        super().__init__(target, dtype, tensor_list, output_tensor, params)

    def compute(self):
        data_in_start = self.inputs_tensor_list[0]
        data_in_end = self.inputs_tensor_list[1]
        data_weight = self.inputs_tensor_list[2]
        data_out = self.output_tensor_list[0]
        # set I/O data
        dev = bangpy.device(0)
        data_in_start_dev = bangpy.Array(data_in_start.astype(self.dtype.as_numpy_dtype), dev)
        data_in_end_dev = bangpy.Array(data_in_end.astype(self.dtype.as_numpy_dtype), dev)
        data_weight_dev = bangpy.Array(data_weight.astype(self.dtype.as_numpy_dtype), dev)
        data_out_dev = bangpy.Array(np.zeros(data_out.shape, self.dtype.as_numpy_dtype), dev)

        f_lerp = load_op_by_type(KERNEL_NAME, self.dtype.name)

        f_lerp(
            data_in_start_dev,
            data_in_end_dev,
            data_weight_dev,
            data_in_start.shape[0],
            data_in_start.shape[1],
            data_in_start.shape[2],
            data_in_start.shape[3],
            data_out_dev,
        )

        theory_io_size = data_in_start.shape[0] * data_in_start.shape[1] * data_in_start.shape[2] * data_in_start.shape[3] * self.dtype.bytes * 4
        IO_BANDWIDTH = 2 ** 40 if self.target == "mlu290" else 307.2 * 2 ** 30
        evaluator = f_lerp.time_evaluator(number=2, repeat=1, min_repeat_ms=0)
        latency = evaluator(
            data_in_start_dev,
            data_in_end_dev,
            data_weight_dev,
            data_in_start.shape[0],
            data_in_start.shape[1],
            data_in_start.shape[2],
            data_in_start.shape[3],
            data_out_dev,
        ).mean
        print("Hardware time : %f us" % (latency * 1000 ** 2))
        
        # io_efficiency
        io_efficiency = theory_io_size / (latency * IO_BANDWIDTH)
        print("theory_io_size : %f GB" % (theory_io_size / (2 ** 30)))
        print("io_efficiency:", str(round(io_efficiency * 100, 2)) + "%")

        
        cal_diff(data_out_dev.numpy(),data_out.astype(self.dtype.as_numpy_dtype))
