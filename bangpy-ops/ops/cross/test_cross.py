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
# pylint: disable=missing-docstring, invalid-name, too-many-locals, too-many-function-args

"""test file for ops: cross"""
from test import registerOp, OpTest
import numpy as np
import bangpy
from bangpy.common import load_op_by_type
from cross import KERNEL_NAME, TARGET_LIST

EPSILON=1e-9
# np.set_printoptions(threshold=np.inf)

def cal_diff(result, data_out):
    """
    Compute diff1 & 2 between cpu result and mlu result.
    """
    result_ = result
    data_out_ = data_out
    result_[np.isnan(result) & np.isnan(data_out)] = 0
    result_[np.isinf(result) & np.isinf(data_out)] = 0
    data_out_[np.isnan(result) & np.isnan(data_out)] = 0
    data_out_[np.isinf(result) & np.isinf(data_out)] = 0
    diff1 = np.sum(np.abs(np.subtract(result_, data_out_))) / (np.sum(result_) +EPSILON)
    diff2 = np.sqrt(
        np.sum(np.power(np.subtract(data_out_, result_), 2,))
        / np.sum(np.power(result_, 2) + EPSILON)
    )
    return diff1, diff2

@registerOp("cross")
class CrossOp(OpTest):
    """Use proto_test to test cosine_embedding_loss."""
    def __init__(self, target, dtype, tensor_list, output_tensor, param):
        self.dtype = dtype
        super().__init__(target, dtype, tensor_list, output_tensor, param)
        self.param = param

    def evaluate(self, f, dtype, target):
        """
        Evaluate IO efficiency and accuracy of cosineEmbeddingLoss of given parameters
        """
        if target not in TARGET_LIST:
            return

        if target == "mlu370-s4":
            IO_BANDWIDTH = 307.2 * 2 ** 30  # MLU370-s4: 307.2GB/s
        else:
            IO_BANDWIDTH = 2**40  # MLU290: 1024GB/s

        data_in0 = self.inputs_tensor_list[0]
        data_in1 = self.inputs_tensor_list[1]
        dim = self.param.get("crossParam").get("dim")

        data_out = self.output_tensor_list[0]

        dev = bangpy.device(0)
        shape = data_out.shape
        # set I/O datas

        data_in0_dev = bangpy.Array(data_in0.flatten().astype(dtype.as_numpy_dtype), dev)
        data_in1_dev = bangpy.Array(data_in1.flatten().astype(dtype.as_numpy_dtype), dev)
        data_out_dev = bangpy.Array(np.zeros(data_out.flatten().shape, dtype.as_numpy_dtype), dev)
        dimshape = bangpy.Array(np.array(shape).astype(bangpy.int32.as_numpy_dtype), dev)

        f(
            data_in0_dev,
            data_in1_dev,
            dimshape,
            len(shape),
            data_in0.size,
            dim,
            data_out_dev,
        )

        # Hardware time
        evaluator = f.time_evaluator(number=1, repeat=1, min_repeat_ms=0)
        latency = (
            evaluator(
                data_in0_dev,
                data_in1_dev,
                dimshape,
                len(shape),
                data_in0.size,
                dim,
                data_out_dev,
            ).median
            * 1e3
        )
        print("Shape:", shape)
        print("dim:", dim)
        print("dtype:", dtype)
        print("Hardware time : %f ms" % latency)

        # io_efficiency
        length = 1
        for i in shape:
            length = length * i
        theory_io_size = length * dtype.bytes * 3
        io_efficiency = 1000 * theory_io_size / (latency * IO_BANDWIDTH)
        print("theory_io_size : %f GB" % (theory_io_size / (2**30)))
        print("io_efficiency:", str(round(io_efficiency * 100, 2)) + "%\n")

        diff1, diff2 = cal_diff(data_out.flatten(), data_out_dev.numpy().flatten())
        assert round(diff1 * 100, 5) < 3e-3 * 100
        assert round(diff2 * 100, 5) < 3e-3 * 100
        print("DIFF1:", str(round(diff1 * 100, 5)) + "%")
        print("DIFF2:", str(round(diff2 * 100, 5)) + "%")
        # diff3 test
        # calculate diff3 will cost lots of time, to test efficiency quickly,
        # strongly advise you to skip this part
        # data_out = data_out.flatten()
        # data_out_dev = data_out_dev.numpy().flatten()
        # diff = np.abs(data_out - data_out_dev)
        # data_out = np.abs(data_out)
        # maxdiff3 = 0
        # if dtype == bangpy.float16:
        #     th = 1e-4
        # elif dtype == bangpy.float32:
        #     th = 1e-6
        # for i, data in enumerate(data_out):
        #     if data > th:
        #         diff3 = diff[i] / data
        #     else:
        #         diff3 = diff[i]
        #     if diff3 > maxdiff3:
        #         maxdiff3 = diff3
        # assert maxdiff3 == 0

    def compute(self):
        if self.target not in TARGET_LIST:
            return
        f = load_op_by_type(KERNEL_NAME, self.dtype.name)

        self.evaluate(f, self.dtype, self.target)
