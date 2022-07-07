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
import torch
import pytest

import bangpy as bp
from bangpy.common import load_op_by_type
from renorm import DTYPES, KERNEL_NAME, TARGET_LIST

def create_random_shape(length, size):
    return np.random.randint(low=1, high=size, size=length)

ranshp = create_random_shape(5, 3)
ranshp2 = create_random_shape(2, 10)

@pytest.mark.parametrize(
    "shape",
    [
        (4, 40, 3, 2, 1, 1, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1),
        (1, 3 * 11, 3, 1, 1),
        (6, 2, 4, 1, 5),
        (6, 2, 4, 7, 5),
        ranshp,
        ranshp2
    ],
)

@pytest.mark.parametrize(
    "dtype", DTYPES,
)

@pytest.mark.parametrize(
    "p", [2.2, 1.0, -12.0, 3.11],
)

@pytest.mark.parametrize(
    "dim", [0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5, -100],
)

@pytest.mark.parametrize(
    "maxnorm", [5.0, 1.0, -1.0],
)



def test_renorm(target, shape, p, dim, dtype, maxnorm):
    if target not in TARGET_LIST:
        return

    try:
        total_input_len = 1
        for s in shape:
            total_input_len *= s

        input_tensor = np.random.uniform(low=-5, high=5, size=shape).astype(dtype.as_numpy_dtype)

        dev = bp.device(0)

        flat_input = input_tensor.flatten()
        mlu_input = bp.Array(flat_input, dev)
        paras = np.array([p, maxnorm]).astype(dtype.as_numpy_dtype)
        mlu_paras = bp.Array(paras, dev)
        mlu_output = bp.Array(flat_input, dev)

        if maxnorm <= 0:
            raise Exception("expected maxnorm to be >= 0")

        if p < 0:
            raise Exception("non-positive-norm not supported")

        if dim < 0:
            dim += len(shape)

        if dim < 0 or dim >= len(shape):
            raise Exception("dim err")

        h = 1
        for i in range(dim):
            h *= shape[i]
        w = total_input_len // h
        sub_t_count = shape[dim]
        sub_wid = w // sub_t_count

        func = load_op_by_type(KERNEL_NAME, dtype.name)
        func(mlu_input, mlu_paras,
             h, w, sub_wid
             , mlu_output)

        result = mlu_output.numpy()
        mlu_ret = result.reshape(shape)


        x = torch.Tensor(input_tensor)
        cpu_ret = torch.renorm(x, p, dim, maxnorm)

        bp.assert_allclose( cpu_ret.numpy(), mlu_ret, rtol = 0.01, atol = 0.01)

    except Exception as err:
        print(str(err))
        if str(err) == "dim err":
            return

        if str(err) == "non-positive-norm not supported":
            return

        if str(err) == "expected maxnorm to be >= 0":
            return

        raise Exception(str(err)) from err
