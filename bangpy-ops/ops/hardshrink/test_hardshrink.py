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
"""Test HardShrink operator with multi-platform code link."""
import numpy as np
import pytest
from hardshrink import HardShrink, TARGET_LIST, DTYPES
import bangpy as bp
from bangpy import tcp
from bangpy.common import load_op_by_type
import os
import time

# float16 has poor accuracy
@pytest.mark.parametrize(
    "dtype",
    DTYPES,
)
@pytest.mark.parametrize(
    "shape",
    [   
        (10, 4, 4096, 4096),
        (4, 16, 1024, 1024),
        (4,16,1,64),
        (3, 5, 197, 175)
    ],
)
@pytest.mark.parametrize(
    "lambdaPara",
    [0.5,]
)

def test_hardshrink(target,shape,dtype,lambdaPara):
    if target not in TARGET_LIST:
        return
    data_in = np.random.uniform(low = -1, high = 1, size = shape)
    data_out = data_in.astype(dtype.as_numpy_dtype)
    dev = bp.device(0)

    data_in_dev = bp.Array(data_in.astype(dtype.as_numpy_dtype), dev)
    data_out_dev = bp.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)

    f_hardshrink = load_op_by_type("hardshrink", dtype.name)
    mlu_start_time = time.time()
    f_hardshrink(
        data_in_dev,
        lambdaPara,
        # 支持原位操作，可替换为data_in_dev
        data_out_dev
    )

    # compute the cpu data
    eps = 1e-8
    cpu_out = np.where((data_in.astype(dtype.as_numpy_dtype) + lambdaPara > -eps) & (data_in - lambdaPara < eps), 0, data_in)

    evaluator = f_hardshrink.time_evaluator(number=10, repeat=1, min_repeat_ms=0)
    run_time = evaluator(data_in_dev, lambdaPara, data_out_dev).mean
    print("mlu run time: " + str(run_time) + "s")

    bp.assert_allclose(
        data_out_dev.numpy(),
        cpu_out.astype(dtype.as_numpy_dtype),
        rtol=1e-6,
        atol=1e-6
    )