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
"""Test expm1 operator with multi-platform code link."""
# pylint: skip-file
import os
import time

import numpy as np
import pytest

import bangpy
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType
from expm1 import DTYPES, KERNEL_NAME, TARGET_LIST

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
    "shape", [(559, 194,4,571),],
    # "shape", [np.random.randint(i, 600+i, size=(4,),dtype='l') for i in range(15)],
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)
def test_expm1(target, shape, dtype):
    if target not in TARGET_LIST:
        return

    dim0 = shape[0]
    dim1 = shape[1]
    dim2 = shape[2]
    dim3 = shape[3]
    data_in = np.random.uniform(low=1, high=5, size=shape)
    data_out = np.expm1(data_in.astype(dtype.as_numpy_dtype))

    dev = bangpy.device(0)

    data_in_dev = bangpy.Array(data_in.astype(dtype.as_numpy_dtype), dev)
    data_out_dev = bangpy.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)
    f1 = load_op_by_type(KERNEL_NAME, dtype.name)
    f1(data_in_dev, data_out_dev, dim0, dim1, dim2, dim3)

    theory_io_size = shape[0] * shape[1] * shape[2] * shape[3] * dtype.bytes * 2
    # IO_BANDWIDTH = 1024 * 1024 * 1024 * 1024
    IO_BANDWIDTH = 2 ** 40 if target == "mlu290" else 307.2 * 2 ** 30
    evaluator = f1.time_evaluator(number=2, repeat=1, min_repeat_ms=0)
    latency = evaluator(
        data_in_dev, data_out_dev, dim0, dim1, dim2, dim3
    ).median
    print("Hardware time : %f us" % (latency * 1000 ** 2))
    
    # io_efficiency
    io_efficiency = theory_io_size / (latency * IO_BANDWIDTH)
    print("theory_io_size : %f GB" % (theory_io_size / (2 ** 30)))
    print("io_efficiency:", str(round(io_efficiency * 100, 2)) + "%")
    cal_diff(data_out_dev.numpy(), data_out.astype(dtype.as_numpy_dtype))
