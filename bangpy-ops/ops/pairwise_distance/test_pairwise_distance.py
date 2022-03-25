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
        return

    def get_total_size(shp):
        size = 1
        for s in shp:
            size *= s
        return size

    def create_origin_input(shpx, shpy):
        shape_x = np.array(shpx).astype('int32')
        shape_y = np.array(shpy).astype('int32')
        data_x = np.random.uniform(low=-10, high=10, size=shape_x).astype(dtype.as_numpy_dtype)
        data_y = np.random.uniform(low=-10, high=10, size=shape_y).astype(dtype.as_numpy_dtype)
        return data_x, data_y

    def create_mlu_input(x, y, dev):        
        reshp_x = x.flatten()
        x_dev = bp.Array(reshp_x, dev)

        reshp_y = y.flatten()
        y_dev = bp.Array(reshp_y, dev)

        shp_x = np.array(x.shape).astype('int32')
        shp_y = np.array(y.shape).astype('int32')
        shp_x_dev = bp.Array(shp_x, dev)
        shp_y_dev = bp.Array(shp_y, dev)

        return x_dev, y_dev, shp_x_dev, shp_y_dev

    def create_output(output_len, dev):
        output_buffer = np.zeros(output_len, dtype=dtype.as_numpy_dtype)
        return bp.Array(output_buffer, dev)

    shpx = shape[0]
    shpy = shape[1]

    x, y = create_origin_input(shpx, shpy)

    dev = bp.device(0)

    x_dev, y_dev, shp_x_dev, shp_y_dev = create_mlu_input(x, y, dev)

    output_len = get_total_size(shpx) // shpx[1]
    print('output size')

    output_dev = create_output(output_len, dev)
    
    func = load_op_by_type(KERNEL_NAME, dtype.name)
    if len(shpx) > len(shpy):
        func(x_dev, y_dev, shp_x_dev, shp_y_dev, 
            get_total_size(shpx), get_total_size(shpy), len(shpx), len(shpy), 
            output_len, output_dev)
    else:
        func(flat_y_dev, flat_x_dev, shp_y_dev, shp_x_dev, 
            len(flat_y), len(flat_x), len(shpy), len(shpx), 
            output_len, output_dev)

    '''bangpy.assert_allclose(
        output_dev.numpy(), data_out.astype(dtype.as_numpy_dtype), rtol=0.1, atol=0.1
    )'''
   