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
from cmath import pi
from traceback import print_tb
import numpy as np
import torch
import pytest
import math

import bangpy
import bangpy as bp
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType
from renorm import DTYPES, KERNEL_NAME, TARGET_LIST

import time

@pytest.mark.parametrize(
    "shape", 
    [        
        (1, 3, 5, 3)
    ],
)

@pytest.mark.parametrize(
    "dtype", DTYPES,
)

@pytest.mark.parametrize(
    "p", [1.3],
)

@pytest.mark.parametrize(
    "dim", [1],
)

@pytest.mark.parametrize(
    "maxnorm", [15.0],
)



def test_renorm(target, shape, p, dim, dtype, maxnorm): 
    if target not in TARGET_LIST:
        return

    total_input_len = 1
    for s in shape:
        total_input_len *= s

    input_tensor = np.random.uniform(low=1, high=3, size=shape).astype(dtype.as_numpy_dtype)
    
    # 准备mlu计算
    dev = bp.device(0)

    mlu_start = time.time()

    flat_input = input_tensor.flatten()
    mlu_input = bp.Array(flat_input, dev)
    paras = np.array([p, maxnorm]).astype(dtype.as_numpy_dtype) # 这里需要考虑
    mlu_paras = bp.Array(paras, dev)
    mlu_output = bp.Array(flat_input, dev)

    if dim < 0:
        dim += len(shape)

    if dim < 0 or dim >= len(shape):
        print("dim err!")
        return None

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

    print('mlu cost ', time.time() - mlu_start)
    print(mlu_ret)

    
    print("============torch calc==================")

    x = torch.Tensor(input_tensor)
    cpu_start = time.time()
    cpu_ret = torch.renorm(x, p, dim, maxnorm)
    print('cpu cost ', time.time() - cpu_start)
    print(cpu_ret)

    bangpy.assert_allclose( cpu_ret.numpy(), mlu_ret,rtol = 0.01, atol = 0.01)
    
