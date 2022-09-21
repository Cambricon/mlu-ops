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
from bangpy.common import load_op_by_type
from add import DTYPES, KERNEL_NAME, TARGET_LIST


@pytest.mark.parametrize(
    "shape", [(4096,), (8192,),],
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)
def test_add(target, shape, dtype):
    if target not in TARGET_LIST:
        return
    data_in0 = np.random.uniform(low=-10, high=10, size=shape)
    data_in1 = np.random.uniform(low=-10, high=10, size=shape)

    data_out = data_in0.astype(dtype.as_numpy_dtype) + data_in1.astype(
        dtype.as_numpy_dtype
    )
    dev = bangpy.device(0)
    # set I/O data
    data_in0_dev = bangpy.Array(data_in0.astype(dtype.as_numpy_dtype), dev)
    data_in1_dev = bangpy.Array(data_in1.astype(dtype.as_numpy_dtype), dev)
    data_out_dev = bangpy.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)
    f1 = load_op_by_type(KERNEL_NAME, dtype.name)
    f1(data_in0_dev, data_in1_dev, data_out_dev, data_in0.shape[0])
    bangpy.assert_allclose(data_out_dev.numpy(), data_out.astype(dtype.as_numpy_dtype))
