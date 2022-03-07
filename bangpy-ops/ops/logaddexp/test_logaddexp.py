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
from logaddexp import DTYPES, KERNEL_NAME, TARGET_LIST

def numcheck(input_pow_arr):
    natural_exponential_res=np.exp(input_pow_arr)
    natural_power_exponent_one_cent=np.reciprocal(natural_exponential_res)
    numerator_res=np.subtract(natural_exponential_res,natural_power_exponent_one_cent)
    return numerator_res*0.5
@pytest.mark.parametrize(
    "shape", 
    [
        (100),
        (3, 4),
        (100, 120, 140),
    ],
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)

#
#一共测了两种数据类型  在算子源码dtype数组中写的那两种  每种测了三次  每次的数组长度 看37行开始的那个数组
#

def test_logaddexp(target, shape, dtype):
    if target not in TARGET_LIST:
        return

    dev = bp.device(0)

    data_x = np.random.uniform(low=-5, high=5, size=shape).astype(dtype.as_numpy_dtype)
    data_x_dev = bp.Array(data_x, dev)

    data_y = np.random.uniform(low=-5, high=5, size=shape).astype(dtype.as_numpy_dtype)
    data_y_dev = bp.Array(data_y, dev)

    #dim = np.array(data_x).shape
    shape_dev = bp.Array(shape, dev)
    total_count = 1

    if isinstance(shape, int):
        total_count = shape
    else:
        for c in shape:
            total_count *= c

    #print(total_count)


    task_type = TaskType(TARGET(target).cluster_num)
    log_add_exp_func = load_op_by_type("LogAddExp", dtype.name)

    task_num = TARGET(target).cluster_num * TARGET(target).core_num
    task_type = TaskType(TARGET(target).cluster_num)

    output_buffer = np.zeros(total_count, dtype=np.float32)
    output_dev = bp.Array(output_buffer, dev)

    #with tcp.runtime.Run(task_type):
    #    log_add_exp_func(data_x_dev, data_y_dev, shape_dev, len(shape_dev))


    
   
    