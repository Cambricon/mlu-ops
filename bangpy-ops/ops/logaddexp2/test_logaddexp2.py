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
"""Logaddexp2 testfile for pytest."""
from test import registerOp, OpTest
import numpy as np
import bangpy
from bangpy.common import load_op_by_type
from logaddexp2 import KERNEL_NAME

def cal_diff(result, data_out):
    diff1 = np.sum(np.abs(np.subtract(result, data_out))) / np.sum(result)
    diff2 = np.sqrt(
        np.sum(np.power(np.subtract(data_out, result), 2,))
        / np.sum(np.power(result, 2))
    )
    assert round(diff1 * 100, 5) < 3e-3 * 100
    assert round(diff2 * 100, 5) < 3e-3 * 100
    return diff1, diff2

@registerOp("logaddexp2")
class LogAddExp2Op(OpTest):
    """Use proto_test to test logaddexp2."""
    def __init__(self, target, dtype, tensor_list, output_tensor, param):
        self.dtype = dtype
        super().__init__(target, dtype, tensor_list, output_tensor, param)

    def compute(self):
        self.shape = self.inputs_tensor_list[0].shape
        if self.shape == (0):
            return
        print("shape :", self.shape)
        data_in0 = self.inputs_tensor_list[0].astype("int32")
        data_in1 = self.inputs_tensor_list[1].astype("int32")
        data_out = self.output_tensor_list[0]
        # device
        dev = bangpy.device(0)
        # set I/O data
        data_in0_dev = bangpy.Array(data_in0.flatten().astype(self.dtype.as_numpy_dtype), dev)
        data_in1_dev = bangpy.Array(data_in1.flatten().astype(self.dtype.as_numpy_dtype), dev)
        data_out_dev = bangpy.Array(
            np.zeros(data_out.flatten().shape, self.dtype.as_numpy_dtype), dev
        )
        # calculate and check
        data_total = len(data_in0.flatten())
        f = load_op_by_type(KERNEL_NAME, self.dtype.name)
        evaluator = f.time_evaluator(number=1, repeat=100, min_repeat_ms=0)

        # calculate
        time = (
            evaluator(
                data_in0_dev, data_in1_dev, data_out_dev, self.shape[0]
            ).mean
            * 1e3
        )  # ms

        data_out_dev2host = data_out_dev.numpy().reshape(self.shape)
        diff1, diff2 = cal_diff(data_out.astype(self.dtype.as_numpy_dtype), data_out_dev2host)

        io_speed = (
            (data_total * 3 * self.dtype.bytes)
            / time
            * 1e3
            / (2 ** 30)
        )  # GB/s

        # Output results.
        out_str = "data_type: {} data_amount: {:2.4f}GB shape: "+\
            "{:8d} time cost: {:3.2f}us io_efficiency: {:4.3f}%, diff1: "+\
            "{:1.5f}%, diff2: {:1.5f}%"
        print(out_str.format(
                self.dtype.name,
                (data_total * 3 * self.dtype.bytes) / (2 ** 30),
                data_total,
                time * 1e3,
                io_speed / (1024 if self.target=="mlu290" else 307.2) * 100,
                round(diff1 * 100, 5),
                round(diff2 * 100, 5),
            )
        )
