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
"""Test cosineEmbeddingLoss operator with multi-platform code link"""
from test import registerOp, OpTest
import numpy as np
from cosine_embedding_loss import KERNEL_NAME
import bangpy
from bangpy.common import load_op_by_type
EPSILON=1e-9
np.set_printoptions(threshold=np.inf)

def cal_diff(result, data_out):
    """
    Compute diff1 & 2 between cpu result and mlu result.
    """
    result_ = result.astype(np.float32)
    data_out_ = data_out.astype(np.float32)
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

@registerOp("cosine_embedding_loss")
class CosineEmbeddingLossOp(OpTest):
    """Use proto_test to test cosine_embedding_loss."""
    def __init__(self, target, dtype, tensor_list, output_tensor, param):
        self.dtype = dtype
        super().__init__(target, dtype, tensor_list, output_tensor, param)
        self.param = param

    def evaluate(self, f, dtype, target):
        """
        Evaluate IO efficiency and accuracy of cosineEmbeddingLoss of given parameters
        """
        data_input_x1 = self.inputs_tensor_list[0]
        data_input_x2 = self.inputs_tensor_list[1]

        data_input_y = self.inputs_tensor_list[2]
        data_height = self.inputs_tensor_list[0].shape[0]
        data_width = self.inputs_tensor_list[0].shape[1]
        # float16 is 64 aligned
        if data_width == 32 and dtype == bangpy.float16:
            return

        margin = self.param.get("cosineEmbeddingLossParam").get("margin")
        data_out = np.zeros((data_height,))

        dev = bangpy.device(0)

        data_input_x1_dev = bangpy.Array(data_input_x1.astype(dtype.as_numpy_dtype), dev)
        data_input_x2_dev = bangpy.Array(data_input_x2.astype(dtype.as_numpy_dtype), dev)
        data_input_y_dev = bangpy.Array(data_input_y.astype(dtype.as_numpy_dtype), dev)
        data_out_dev = bangpy.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)

        data_out = self.output_tensor_list[0]
        f(
            data_input_x1_dev,
            data_input_x2_dev,
            data_input_y_dev,
            margin,
            data_out_dev,
            data_height,
            data_width
        )

        dev_out = data_out_dev.numpy()

        diff1, diff2 = cal_diff(dev_out, data_out)
        evaluator = f.time_evaluator(dev, 1, 10)
        time = (
            evaluator(
                data_input_x1_dev,
                data_input_x2_dev,
                data_input_y_dev,
                margin,
                data_out_dev,
                data_height,
                data_width
            ).mean
            * 1e3
        )  # ms
        io_speed = (
            (data_width * data_height * 2 * dtype.bytes + data_height * dtype.bytes)
            / time
            * 1e3
            / (2 ** 30)
        )  # GB/s

        # Output results.
        out_str = "data_type: {} data_amount: {:2.4f}GB data_shape: "+\
            "{} time cost: {:3.2f}us io_efficiency: {:4.3f}%, diff1: "+\
            "{:1.5f}%, diff2: {:1.5f}%"
        print(out_str.format(
                dtype.name,
                data_height * data_width * dtype.bytes / (2 ** 30),
                (data_height, data_width),
                time * 1000,
                io_speed / (1024 if target == "mlu290" else 307.2) * 100,
                round(diff1 * 100, 5),
                round(diff2 * 100, 5),
            )
        )

        assert round(diff1 * 100, 5) < 3e-3 * 100
        assert round(diff2 * 100, 5) < 3e-3 * 100

    def compute(self):
        f = load_op_by_type(KERNEL_NAME, self.dtype.name)

        self.evaluate(f, self.dtype, self.target)
