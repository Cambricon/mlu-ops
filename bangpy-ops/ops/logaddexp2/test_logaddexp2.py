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

import numpy as np
import pytest
import bangpy
from bangpy.common import load_op_by_type
from logaddexp2 import DTYPES, KERNEL_NAME, TARGET_LIST

def cal_diff(result, data_out):
    diff1 = np.sum(np.abs(np.subtract(result, data_out))) / np.sum(result)
    diff2 = np.sqrt(
        np.sum(np.power(np.subtract(data_out, result), 2,))
        / np.sum(np.power(result, 2))
    )
    assert round(diff1 * 100, 5) < 3e-3 * 100
    assert round(diff2 * 100, 5) < 3e-3 * 100
    print("DIFF1:", str(round(diff1 * 100, 5)) + "%")
    print("DIFF2:", str(round(diff2 * 100, 5)) + "%")

@pytest.mark.parametrize(
    "shape", [(0), (100), (2**10),(2**18-1), (2**20), (2**26)],
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)
def test_logaddexp2(target, shape, dtype):
    """Use pytest to test logaddexp2."""
    if target not in TARGET_LIST:
        return
    dev = bangpy.device(0)

    # set data
    data_in0 = np.random.randint(low=-200, high=-190, size=shape).astype(dtype.as_numpy_dtype)
    data_in1 = np.random.randint(low=-200, high=-190, size=shape).astype(dtype.as_numpy_dtype)
    data_out = np.logaddexp2(data_in0, data_in1)
    data_in0_dev = bangpy.Array(data_in0, dev)
    data_in1_dev = bangpy.Array(data_in1, dev)
    data_out_dev = bangpy.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)

    # calculate
    func = load_op_by_type(KERNEL_NAME, dtype.name)
    func(data_in0_dev, data_in1_dev, data_out_dev)

    if shape != (0):
        cal_diff(data_out, data_out_dev.numpy())
