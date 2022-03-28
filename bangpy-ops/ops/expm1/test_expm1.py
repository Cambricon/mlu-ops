# Copyright (C) [2021] by Cambricon, Inc.
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
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType
from expm1 import DTYPES, KERNEL_NAME, TARGET_LIST
import time


@pytest.mark.parametrize(
    "shape",
    [
        (8, 2, 7, 9),
    ],
)
@pytest.mark.parametrize(
    "dtype", [bangpy.float32],
)
def test_expm1(target, shape, dtype):
    if target not in TARGET_LIST:
        return
    data_in = np.random.uniform(low=-1, high=1, size=shape)
    cpu_start = time.time()
    data_out = np.expm1(data_in.astype(dtype.as_numpy_dtype))
    cpu_end = time.time()
    print("cpu exe time: " + str(cpu_end - cpu_start) + "s")
    dev = bangpy.device(0)
    # set I/O data
    data_in_dev = bangpy.Array(data_in.astype(dtype.as_numpy_dtype), dev)
    data_out_dev = bangpy.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)
    f1 = load_op_by_type(KERNEL_NAME, dtype.name)
    mlu_start = time.time()
    f1(
        data_in_dev,
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        data_out_dev
    )
    mlu_end = time.time()
    print("mlu exe time: " + str(mlu_end - mlu_start) + "s")
    bangpy.assert_allclose(
        data_out_dev.numpy(), data_out.astype(dtype.as_numpy_dtype), rtol=3e-3, atol=3e-3
    )
