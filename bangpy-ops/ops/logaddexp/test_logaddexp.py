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
import time

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
        (1024, 1024, 128),
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

    print(dtype)
    # origin input
    
    data_x = np.random.uniform(low=-1000, high=1000, size=shape).astype(dtype.as_numpy_dtype)
    data_y = np.random.uniform(low=-1000, high=1000, size=shape).astype(dtype.as_numpy_dtype)
    # data_x =np.array([326.15338,]).astype(dtype.as_numpy_dtype)
    # data_y = np.array([320.5677,]).astype(dtype.as_numpy_dtype)
    mlu_start =time.time()
    print("mlu start.")

    dev = bp.device(0)
    data_x_flat = data_x.flatten()
    data_x_dev = bp.Array(data_x_flat, dev)
    
    data_y_flat = data_y.flatten()
    data_y_dev = bp.Array(data_y_flat, dev)

    task_type = TaskType(TARGET(target).cluster_num)
    log_add_exp_func = load_op_by_type("LogAddExp", dtype.name)

    task_type = TaskType(TARGET(target).cluster_num)

    output_buffer = np.zeros(len(data_x_flat), dtype=dtype.as_numpy_dtype)
    output_dev = bp.Array(output_buffer, dev)

    with tcp.runtime.Run(task_type):
        log_add_exp_func(data_x_dev, data_y_dev, output_dev)
    
    np_ret = output_dev.numpy()
    #ret1 = np_ret.reshape(shape)
    mlu_end = time.time()
    print('mlu cost ', mlu_end - mlu_start)
    
    #print(ret1)
    print("cpu start.")
    # cpu_start_time = time.time()
    ret2 = np.logaddexp(data_x_flat, data_y_flat)
    # cpu_end_time = time.time()
    #print(ret2)
    # print("cpu_time", cpu_end_time - cpu_start_time)
    print("data_x:",data_x_flat)
    print("data_y:",data_y_flat)
    print("mlu_out:",np_ret)
    print("cpu_out:",ret2)
    #bp.testing.assert_allclose(ret1.numpy(), ret2.astype(dtype.as_numpy_dtype), rtol = 0.2, atol = 0.2)
    bangpy.assert_allclose( np_ret, ret2.astype(dtype.as_numpy_dtype),rtol = 0.1, atol = 0.1)
    
   
       