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
"""A multi-platform code link example test for BANGPy TCP."""
import numpy as np
import pytest
import bangpy
from bangpy.common import load_op_by_type
from hard_sigmoid import DTYPES, KERNEL_NAME, TARGET_LIST


def cal_diff(result, data_out):
    diff1 = np.sum(np.abs(np.subtract(result, data_out))) / np.sum(result)
    diff2 = np.sqrt(
        np.sum(np.power(np.subtract(data_out, result), 2)) / np.sum(np.power(result, 2))
    )
    assert round(diff1 * 100, 5) < 3e-3 * 100
    assert round(diff2 * 100, 5) < 3e-3 * 100
    print("DIFF1:", str(round(diff1 * 100, 5)) + "%")
    print("DIFF2:", str(round(diff2 * 100, 5)) + "%")


@pytest.mark.parametrize(
    "shape",
    [
        (2 ** 23 + 1,),
        (1, 2 ** 24 + 1),
        (1, 1, 2 ** 25 + 1),
        (1, 1, 1, 2 ** 26 + 1),
        (1, 1, 1, 1, 2 ** 27 + 1),
        (1, 1, 1, 1, 1, 2 ** 28 + 1),
        (1, 1, 1, 1, 1, 1, 2 ** 29 + 1),
        (1, 1, 1, 1, 1, 1, 1, 2 ** 30 + 1),
    ],
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)
# @pytest.mark.repeat(1000)
def test_hard_sigmoid(target, shape, dtype):
    if target not in TARGET_LIST:
        return
    data_in = np.random.uniform(low=-5, high=5, size=shape).astype(dtype.as_numpy_dtype)
    # Hardsigmoid function
    data_out = data_in * 1 / 6 + 1 / 2
    data_out = np.minimum(data_out, 1)
    data_out = np.maximum(data_out, 0)
    # data_out = torch.nn.functional.hardsigmoid(data_in)
    # device
    dev = bangpy.device(0)
    # set I/O data
    data_in_dev = bangpy.Array(data_in.flatten().astype(dtype.as_numpy_dtype), dev)
    data_out_dev = bangpy.Array(
        np.zeros(data_out.flatten().shape, dtype.as_numpy_dtype), dev
    )

    f = load_op_by_type(KERNEL_NAME, dtype.name)
    f(data_in_dev, data_out_dev)
    data_out_dev2host = data_out_dev.numpy().reshape(shape)
    cal_diff(data_out_dev2host, data_out.astype(dtype.as_numpy_dtype))
    # bangpy.assert_allclose(data_out_dev2host, data_out.astype(dtype.as_numpy_dtype))

    evaluator = f.time_evaluator(number=10, repeat=1, min_repeat_ms=0)
    latency = evaluator(data_in_dev, data_out_dev).mean * 1e3
    print("Hardware time : %f ms" % latency)

    # io_efficiency
    data_total = len(data_out.flatten())
    theory_io_size = data_total * dtype.bytes * 2
    IO_BANDWIDTH = 2 ** 40  # 1024GB/s
    io_efficiency = 1000 * theory_io_size / (latency * IO_BANDWIDTH)
    print("theory_io_size : %f GB" % (theory_io_size / (2 ** 30)))
    print("io_efficiency:", str(round(io_efficiency * 100, 2)) + "%")
