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
# pylint: disable=too-many-locals, line-too-long, too-many-function-args, unused-argument
"""KlDivloss test demo."""
from test import registerOp, OpTest
import numpy as np
import bangpy as bp
from bangpy.common import load_op_by_type
from kldivloss import KERNEL_NAME


def cal_diff(result, data_out, reduction):
    """compute diff"""
    if reduction == 0:
        result[np.isnan(result)] = 0
        result[np.isinf(result)] = 1
        data_out[np.isnan(data_out)] = 0
        data_out[np.isinf(data_out)] = 1
    else:
        if np.isinf(result):
            print("DIFF1:", str(round(0 * 100, 5)) + "%")
            print("DIFF2:", str(round(0 * 100, 5)) + "%")
            return
        if (np.isnan(result) and np.isnan(data_out)) or (np.isinf(result) and np.isinf(data_out)):
            print("DIFF1:", str(round(0 * 100, 5)) + "%")
            print("DIFF2:", str(round(0 * 100, 5)) + "%")
            return


    bp.assert_allclose(np.isnan(result), np.isnan(data_out))
    bp.assert_allclose(np.isinf(result), np.isinf(data_out))

    diff1 = np.sum(
        np.abs(np.subtract(result, data_out, dtype=np.float64)), dtype=np.float64
    ) / np.sum(result, dtype=np.float64)
    diff2 = np.sqrt(
        np.sum(
            np.power(
                np.subtract(data_out, result, dtype=np.float64), 2, dtype=np.float64
            ),
            dtype=np.float64,
        )
        / np.sum(np.power(result, 2), dtype=np.float64),
        dtype=np.float64,
    )

    print("DIFF1:", str(round(diff1 * 100, 5)) + "%")
    print("DIFF2:", str(round(diff2 * 100, 5)) + "%")
    assert round(diff1 * 100, 5) < 3e-3 * 100
    assert round(diff2 * 100, 5) < 3e-3 * 100


@registerOp("kldivloss")
class KlDivLossOp(OpTest):
    """Use proto_test to test cosine_embedding_loss."""
    def __init__(self, target, dtype, tensor_list, output_tensor, param):
        self.dtype = dtype
        super().__init__(target, dtype, tensor_list, output_tensor, param)
        self.param = param

    def evaluate(self, f, dtype, target):
        """Test kldivloss operator by giving multiple sets of parameters."""
        reduction = self.param.get("kldivlossParam").get("reduction")
        log_target =  self.param.get("kldivlossParam").get("logTarget")
        shape =   self.inputs_tensor_list[0].shape
        # Convert the size of the input data to the new style (batchnum, length)
        print(
            "shape is :", shape, "reduction is :", reduction, "log_target is :", log_target
        )
        data_in0 = self.inputs_tensor_list[0]
        data_in1 = self.inputs_tensor_list[1]
        data_out = self.output_tensor_list[0]

        # Reduction operation
        # if reduction == 0:
        #     print("data_out : ", data_out)
        if reduction == 1:
            print("data_out_sum : ", data_out[0])
        if reduction == 2:
            print("data_out_mean : ", data_out[0])
        if reduction == 3:
            print("data_out_batchmean : ", data_out[0])

        dev = bp.device(0)
        # Set I/O data
        data_input_dev = bp.Array(data_in0.reshape(shape[0], data_in0.size // shape[0]).astype(dtype.as_numpy_dtype), dev)
        data_target_dev = bp.Array(data_in1.reshape(shape[0], data_in0.size // shape[0]).astype(dtype.as_numpy_dtype), dev)
        data_out_dev = bp.Array(
            np.zeros(data_in0.flatten().shape, dtype.as_numpy_dtype), dev
        )

        f = load_op_by_type(KERNEL_NAME, dtype.name)
        f(
            data_input_dev,
            data_target_dev,
            data_out_dev,
            shape[0],
            data_in0.size // shape[0],
            reduction,
            log_target,
        )

        evaluator = f.time_evaluator(number=100, repeat=2, min_repeat_ms=0)
        t = (
            evaluator(
                data_input_dev,
                data_target_dev,
                data_out_dev,
                shape[0],
                data_in0.size // shape[0],
                reduction,
                log_target,
            ).mean
            * 1e3
        )

        data_out_dev = data_out_dev.numpy().reshape(shape[0], data_in0.size // shape[0])
        if reduction == 0:
            # print("data_out_dev : ", data_out_dev)
            data_out = data_out.reshape(shape[0], data_in0.size // shape[0])
            cal_diff(data_out, data_out_dev, reduction)
        elif reduction == 1:
            print("data_out_sum_dev : ", data_out_dev[0][0])
            cal_diff(data_out[0], data_out_dev[0][0], reduction)
        elif reduction == 2:
            print("data_out_mean_dev : ", data_out_dev[0][0])
            cal_diff(data_out[0], data_out_dev[0][0], reduction)
        elif reduction == 3:
            print("data_out_batchmean_dev : ", data_out_dev[0][0])
            cal_diff(data_out[0], data_out_dev[0][0], reduction)

        print("tutorial : %f ms" % t)

        # io_efficiency
        theory_io_size = (
            3 * data_in0.size * dtype.bytes if reduction == 0 else 2 * data_in0.size * dtype.bytes
        )
        # io_bandwidth = 2 ** 40  # MLU290: 1024GB/s
        io_bandwidth = 307.2 * 2 ** 30  # MLU370-s4: 307.2GB/s
        io_efficiency = 1000 * theory_io_size / (t * io_bandwidth)
        print("theory_io_size : %f GB" % (theory_io_size / (2 ** 30)))
        print("io_efficiency:", str(round(io_efficiency * 100, 2)) + "%")
        print("------------------------------------------------------------")

    def compute(self):
        f = load_op_by_type(KERNEL_NAME, self.dtype.name)

        self.evaluate(f, self.dtype, self.target)
