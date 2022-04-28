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
from logsumexp import DTYPES, KERNEL_NAME, TARGET_LIST

import time

@pytest.mark.parametrize(
    "shape", 
    [        
        (31, 31)
    ],
)

@pytest.mark.parametrize(
    "dtype", DTYPES,
)


@pytest.mark.parametrize(
    "dim", [1],
)

@pytest.mark.parametrize(
    "keepdim", [True],
)



def test_logsumexp(target, shape, dim, dtype, keepdim): 
    if target not in TARGET_LIST:
        return

    total_input_len = 1
    for s in shape:
        total_input_len *= s

    input_tensor = np.random.uniform(low=-5, high=5, size=shape).astype(dtype.as_numpy_dtype)
    
    def get_total_size(shp):
            size = 1
            for s in shp:
                size *= s
            return size        
            
    _dev = bp.device(0)
    shp_len = len(shape)

    # mlu 输入参数
    _pd_len = shape[dim]
    _pd_height = 1
    _pd_width = 1

    if dim < 0:
        dim += len(shape)

    if dim < 0 or dim >= len(shape):
        print("dim err!")
        return None

    for i in range(0, dim + 1):
        _pd_height *= shape[i]

    if dim == shp_len - 1:
        pass
    else:
        for i in range(dim + 1, shp_len):
            _pd_width *= shape[i] 

    # mlu 输入
    _mlu_input1 = bp.Array(input_tensor.flatten(), _dev)

    # mlu 输出
    _output_len = get_total_size(shape) // shape[dim]
    output_buffer = np.zeros(_output_len, dtype=dtype.as_numpy_dtype)
    _mlu_output = bp.Array(output_buffer, _dev)

    output_count = 256
    output_buffer2 = np.zeros(output_count, dtype=dtype.as_numpy_dtype)
    _mlu_border_output = bp.Array(output_buffer2, _dev)

    output_buffer3 = -np.ones(output_count, dtype=np.int32)
    _mlu_border_idx_output = bp.Array(output_buffer3, _dev)


    # 调用mlu
    func = load_op_by_type(KERNEL_NAME, dtype.name)
    func(_mlu_input1,
         get_total_size(shape), 
         _pd_len, _pd_height, _pd_width, _output_len
         , _mlu_border_output, _mlu_border_idx_output, _mlu_output)

    result = _mlu_output.numpy()
    result_border = _mlu_border_output.numpy()
    result_border_idx = _mlu_border_idx_output.numpy()


    outputshape = []
    if keepdim:
        for item in shape:
            outputshape.append(item)
        outputshape[dim_index] = 1
    else:
        for i in range(0, len(shape) - 1):
            outputshape.append(shape[i])

    mlu_ret = result.reshape(outputshape)


   
    print("============torch calc==================")

    x = torch.Tensor(input_tensor)
    cpu_start = time.time()
    cpu_ret = torch.logsumexp(a, dim, keepdim)
    print('cpu cost ', time.time() - cpu_start)
    #print(cpu_ret)

    bangpy.assert_allclose( cpu_ret.numpy(), mlu_ret,rtol = 0.01, atol = 0.01)
    
