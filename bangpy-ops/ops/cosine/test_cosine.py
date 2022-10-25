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
"""Test cosine operator with multi-platform code link."""
# pylint: skip-file
import os
import time
import numpy as np
import pytest

import bangpy
from bangpy.common import utils, load_op_by_type
from cosine import DTYPES, KERNEL_NAME, TARGET_LIST

@pytest.mark.parametrize(
    "shape", [(1, 128, 32, 512),],
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)
def test_cosine(target, shape, dtype):
    target = "mlu370-s4"
    if target not in TARGET_LIST:
        return
    dim_var = 3
    data_in0 = np.random.uniform(low=-10, high=10, size=shape)
    data_in1 = np.random.uniform(low=-10, high=10, size=shape)

    shape0 = (shape[1], shape[2], shape[3])
    shape1 = (shape[0], shape[2], shape[3])
    shape2 = (shape[0], shape[1], shape[3])
    shape3 = (shape[0], shape[1], shape[2])
    def torch_test(data_in0, data_in1):
        import torch
        tensor0 = torch.from_numpy(data_in0)
        tensor1 = torch.from_numpy(data_in1)

        output_tensor = torch.nn.functional.cosine_similarity(tensor0, tensor1, dim=dim_var)
        data_out = output_tensor.numpy()
        return data_out
    data_out = torch_test(data_in0, data_in1)
    dev = bangpy.device(0)

    data_in0_dev = bangpy.Array(data_in0.astype(dtype.as_numpy_dtype), dev)
    data_in1_dev = bangpy.Array(data_in1.astype(dtype.as_numpy_dtype), dev)
    data_out0_dev = bangpy.Array(np.zeros(shape0, dtype.as_numpy_dtype), dev)
    data_out1_dev = bangpy.Array(np.zeros(shape1, dtype.as_numpy_dtype), dev)
    data_out2_dev = bangpy.Array(np.zeros(shape2, dtype.as_numpy_dtype), dev)
    data_out3_dev = bangpy.Array(np.zeros(shape3, dtype.as_numpy_dtype), dev)

    f1 = load_op_by_type(KERNEL_NAME, dtype.name)
    f1(
        data_in0_dev,
        data_in1_dev,
        data_out0_dev,
        data_out1_dev,
        data_out2_dev,
        data_out3_dev,
        dim_var,
        shape[0],
        shape[1],
        shape[2],
        shape[3]
    )
    
    theory_io_size = shape[0] * shape[1] * shape[2] * shape[3] * 4 * 2
    IO_BANDWIDTH = 2 ** 40 if target == "mlu290" else 307.2 * 2 ** 30
    evaluator = f1.time_evaluator(number=2, repeat=1, min_repeat_ms=0)
    run_time = evaluator(
        data_in0_dev,
        data_in1_dev,
        data_out0_dev,
        data_out1_dev,
        data_out2_dev,
        data_out3_dev,
        dim_var,
        shape[0],
        shape[1],
        shape[2],
        shape[3]
    ).mean
    print("Hardware time : %f us" % (latency * 1000 ** 2))

    # io_efficiency
    io_efficiency = theory_io_size / (latency * IO_BANDWIDTH)
    print("theory_io_size : %f GB" % (theory_io_size / (2 ** 30)))
    print("io_efficiency:", str(round(io_efficiency * 100, 2)) + "%")

    def cal_diff(result, data_out):
        diff1 = np.sum(np.abs(np.subtract(result, data_out))) / np.sum(np.abs(data_out))
        diff2 = np.sqrt(
        np.sum(np.power(np.subtract(result, data_out), 2)) / np.sum(np.power(data_out, 2))
        )
        print("DIFF1:", str(round(diff1 * 100, 5)) + "%")
        print("DIFF2:", str(round(diff2 * 100, 5)) + "%")
        assert round(diff1 * 100, 5) < 3e-3 * 100
        assert round(diff2 * 100, 5) < 3e-3 * 100

    if dim_var == 0:
        cal_diff(data_out0_dev.numpy(),data_out.astype(dtype.as_numpy_dtype))
    elif dim_var == 1:
        cal_diff(data_out1_dev.numpy(),data_out.astype(dtype.as_numpy_dtype))
    elif dim_var == 2:
        cal_diff(data_out2_dev.numpy(),data_out.astype(dtype.as_numpy_dtype))
    else:
        cal_diff(data_out3_dev.numpy(),data_out.astype(dtype.as_numpy_dtype))