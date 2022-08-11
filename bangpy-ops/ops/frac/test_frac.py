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
"""Frac operator implementation using BANGPy TCP API."""
import numpy as np
import pytest

import bangpy
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType
from frac import DTYPES, KERNEL_NAME, TARGET_LIST

import os
import time


@pytest.mark.parametrize(
    "shape", [(4, 4, 1024, 1024),],
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)
def test_frac(target, shape, dtype):
    if target not in TARGET_LIST:
        return

    dim0 = shape[0]
    dim1 = shape[1]
    dim2 = shape[2]
    dim3 = shape[3]
    data_in = np.random.uniform(low=-10, high=10, size=shape)
    data_absolute = np.absolute(data_in)
    data_floor = np.floor(data_absolute)
    data_sign = np.sign(data_in)
    data_tem = np.multiply(data_floor, data_sign)
    data_out = np.subtract(data_in, data_tem)

    dev = bangpy.device(0)
    # set I/O data
    data_in_dev = bangpy.Array(data_in.astype(dtype.as_numpy_dtype), dev)
    data_out_dev = bangpy.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)
    f1 = load_op_by_type(KERNEL_NAME, dtype.name)
    mlu_start_time = time.time()
    f1(data_in_dev, data_out_dev, dim0, dim1, dim2, dim3)
    mlu_end_time = time.time()

    bangpy.assert_allclose(
        data_out_dev.numpy(), data_out.astype(dtype.as_numpy_dtype), rtol=1, atol=1
    )

    theory_io_size = shape[0] * shape[1] * shape[2] * shape[3] * 4 * 2
    IO_BANDWIDTH = 1024 * 1024 * 1024 * 1024
    latency = mlu_end_time - mlu_start_time
    io_efficiency = theory_io_size / (latency * IO_BANDWIDTH)
    print(io_efficiency)
