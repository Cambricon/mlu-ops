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
import pytest

import bangpy
import bangpy as bp
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType
from pairwise_distance import DTYPES, KERNEL_NAME, TARGET_LIST

@pytest.mark.parametrize(
    "shape", 
    [        
        ((2, 3, 4), (3, 4))
    ],
)

@pytest.mark.parametrize(
    "dtype", DTYPES,
)

@pytest.mark.parametrize(
    "p", [2.],
)

@pytest.mark.parametrize(
    "eps", [0,],
)

@pytest.mark.parametrize(
    "keepdim", [False],
)



def test_pairwise_distance(target, shape, p, eps, keepdim, dtype): 
    if target not in TARGET_LIST:
        print('no')
        return

    # input data
    shape_x = np.array(shape[0]).astype('int32')
    shape_y = np.array(shape[1]).astype('int32')

    data_x = np.random.uniform(low=-10, high=10, size=shape_x).astype(dtype.as_numpy_dtype)
    data_y = np.random.uniform(low=-10, high=10, size=shape_y).astype(dtype.as_numpy_dtype)

    dev = bp.device(0)
    flat_x = data_x.flatten()
    flat_x_dev = bp.Array(flat_x, dev)

    flat_y = data_y.flatten()
    flat_y_dev = bp.Array(flat_y, dev)

    shp_x_dev = bp.Array(shape_x, dev)
    shp_y_dev = bp.Array(shape_y, dev)

    output_len = len(flat_x) // shape_x[1]
    output_buffer = np.zeros(output_len, dtype=dtype.as_numpy_dtype)
    output_dev = bp.Array(output_buffer, dev)
    
    func = load_op_by_type(KERNEL_NAME, dtype.name)
    if len(shape_x) > len(shape_y):
        func(flat_x_dev, flat_y_dev, shp_x_dev, shp_y_dev, 
            len(flat_x), len(flat_y), len(shape_x), len(shape_y), 
            output_len, output_dev)
    else:
        func(flat_y_dev, flat_x_dev, shp_y_dev, shp_x_dev, 
            len(flat_y), len(flat_x), len(shape_y), len(shape_x), 
            output_len, output_dev)

    '''bangpy.assert_allclose(
        output_dev.numpy(), data_out.astype(dtype.as_numpy_dtype), rtol=0.1, atol=0.1
    )'''
   