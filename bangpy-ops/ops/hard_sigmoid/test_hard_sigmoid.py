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
# pylint: disable=missing-docstring, invalid-name, too-many-locals
# pylint: disable=too-many-function-args
"""Test hardSigmoid operator with multi-platform code link."""
from test import registerOp, OpTest
import numpy as np
import bangpy
from bangpy.common import load_op_by_type
from hard_sigmoid import KERNEL_NAME


def cal_diff(result, data_out):
    diff1 = np.sum(np.abs(np.subtract(result, data_out))) / np.sum(result)
    diff2 = np.sqrt(
        np.sum(np.power(np.subtract(data_out, result), 2)) / np.sum(np.power(result, 2))
    )
    assert round(diff1 * 100, 5) < 3e-3 * 100
    assert round(diff2 * 100, 5) < 3e-3 * 100
    print("DIFF1:", str(round(diff1 * 100, 5)) + "%")
    print("DIFF2:", str(round(diff2 * 100, 5)) + "%")


@registerOp("hard_sigmoid")
class HardSigmoidOp(OpTest):
    def __init__(self, target, dtype, tensor_list, output_tensor, params):
        self.dtype = dtype
        super().__init__(target, dtype, tensor_list, output_tensor, params)

    def compute(self):
        self.shape = self.inputs_tensor_list[0].shape
        print("shape :", self.shape)
        data_in = self.inputs_tensor_list[0]
        data_out = self.output_tensor_list[0]
        inplace = self.test_param_.get("hardSigmoidParam").get("inplace")
        # device
        dev = bangpy.device(0)
        # set I/O data
        data_in_dev = bangpy.Array(
            data_in.flatten().astype(self.dtype.as_numpy_dtype), dev
        )
        data_out_dev = bangpy.Array(
            np.zeros(data_out.flatten().shape, self.dtype.as_numpy_dtype), dev
        )
        # calculate and check
        data_total = len(data_in.flatten())
        f = load_op_by_type(KERNEL_NAME, self.dtype.name)
        f(data_in_dev, data_out_dev, data_total, inplace)
        data_in_dev2host = data_in_dev.numpy().reshape(self.shape)
        data_out_dev2host = data_out_dev.numpy().reshape(self.shape)
        if inplace == 1:
            cal_diff(data_in_dev2host, data_out.astype(self.dtype.as_numpy_dtype))
        else:
            cal_diff(data_out_dev2host, data_out.astype(self.dtype.as_numpy_dtype))

        # Hardware time
        evaluator = f.time_evaluator(number=1, repeat=100, min_repeat_ms=0)
        latency = evaluator(data_in_dev, data_out_dev, data_total, inplace).median * 1e3
        print("Hardware time : %f us" % (latency * 1000))

        # io_efficiency
        theory_io_size = data_total * self.dtype.bytes * 2
        IO_BANDWIDTH = 2 ** 40 if self.target == "mlu290" else 307.2 * 2 ** 30
        # MLU290: 1024GB/s
        # MLU370-s4: 307.2GB/s
        io_efficiency = 1000 * theory_io_size / (latency * IO_BANDWIDTH)
        print("theory_io_size : %f GB" % (theory_io_size / (2 ** 30)))
        print("io_efficiency:", str(round(io_efficiency * 100, 2)) + "%")
