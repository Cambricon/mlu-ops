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
"""Cosine_similarity operator implementation using BANGPy TCP API."""
import numpy as np
import pytest
import torch
import bangpy
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType
from cosine_similarity import DTYPES, KERNEL_NAME, TARGET_LIST


@pytest.mark.parametrize(
    "shape", [(1, 1, 32, 512),],
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)
def test_cosine_similarity(target, shape, dtype):
    if target not in TARGET_LIST:
        return
    dim_var = 3
    data_in0 = np.random.uniform(low=-10, high=10, size=shape)
    data_in1 = np.random.uniform(low=-10, high=10, size=shape)

    shape0 = (shape[1], shape[2], shape[3])
    shape1 = (shape[0], shape[2], shape[3])
    shape2 = (shape[0], shape[1], shape[3])
    shape3 = (shape[0], shape[1], shape[2])

    tensor0 = torch.tensor(data_in0)
    tensor1 = torch.tensor(data_in1)

    output_tensor = torch.nn.functional.cosine_similarity(tensor0, tensor1, dim=dim_var)
    data_out = output_tensor.numpy()

    dev = bangpy.device(0)
    # set I/O data
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
        dim_var,
        data_out0_dev,
        data_out1_dev,
        data_out2_dev,
        data_out3_dev,
    )

    if dim_var == 0:
        bangpy.assert_allclose(
            data_out0_dev.numpy(),
            data_out.astype(dtype.as_numpy_dtype),
            rtol=3e-3,
            atol=3e-3,
        )
    elif dim_var == 1:
        bangpy.assert_allclose(
            data_out1_dev.numpy(),
            data_out.astype(dtype.as_numpy_dtype),
            rtol=3e-3,
            atol=3e-3,
        )
    elif dim_var == 2:
        bangpy.assert_allclose(
            data_out2_dev.numpy(),
            data_out.astype(dtype.as_numpy_dtype),
            rtol=3e-3,
            atol=3e-3,
        )
    else:
        bangpy.assert_allclose(
            data_out3_dev.numpy(),
            data_out.astype(dtype.as_numpy_dtype),
            rtol=3e-3,
            atol=3e-3,
        )
