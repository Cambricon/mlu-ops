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
# The above copyright notice and this permission notice shall self.tcp included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Test Lerp operator with multi-platform code link."""
# pylint: skip-file

import numpy as np
import pytest
from lerp import TARGET_LIST, DTYPES
import bangpy as bp
from bangpy.common import load_op_by_type


@pytest.mark.parametrize(
    "dtype", DTYPES,
)
@pytest.mark.parametrize(
    "shape", [(4, 16, 1024, 1024), (4, 16, 1, 64), (3, 5, 197, 255),],
)
def test_lerp(target, shape, dtype):
    if target not in TARGET_LIST:
        return
    data_in_start = np.random.uniform(low=-1, high=1, size=shape)
    data_in_end = np.random.uniform(low=-1, high=1, size=shape)
    data_weight = np.random.uniform(low=-1, high=1, size=shape)

    data_temp = np.subtract(data_in_end, data_in_start)
    data_temp = np.multiply(data_temp, data_weight)
    data_out = np.add(data_in_start, data_temp)

    dev = bp.device(0)
    data_in_start_dev = bp.Array(data_in_start.astype(dtype.as_numpy_dtype), dev)
    data_in_end_dev = bp.Array(data_in_end.astype(dtype.as_numpy_dtype), dev)
    data_weight_dev = bp.Array(data_weight.astype(dtype.as_numpy_dtype), dev)
    data_out_dev = bp.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)

    f_lerp = load_op_by_type("lerp", dtype.name)

    f_lerp(
        data_in_start_dev,
        data_in_end_dev,
        data_weight_dev,
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        data_out_dev,
    )

    evaluator = f_lerp.time_evaluator(number=10, repeat=1, min_repeat_ms=0)
    run_time = evaluator(
        data_in_start_dev,
        data_in_end_dev,
        data_weight_dev,
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        data_out_dev,
    ).mean
    print("mlu run time: " + str(run_time) + "s")

    bp.assert_allclose(
        data_out_dev.numpy(),
        data_out.astype(dtype.as_numpy_dtype),
        rtol=3e-3,
        atol=3e-3,
    )
