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
import pytest
import time

import bangpy
import bangpy as bp
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType
from celu import DTYPES, KERNEL_NAME, TARGET_LIST

@pytest.mark.parametrize(
    "shape", 
    [
        (3,4,),
    ],
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)

#
#一共测了两种数据类型  在算子源码dtype数组中写的那两种  每种测了三次  每次的数组长度 看37行开始的那个数组
#

def test_celu(target, shape, dtype):
    if target not in TARGET_LIST:
        return  
    data_x = np.random.uniform(low=-7.75, high=10, size=shape).astype(dtype.as_numpy_dtype)
    #data_x = np.array([83.35]).astype(dtype.as_numpy_dtype)
   
  
    
    dev = bp.device(0)
    data_x_flat = data_x.flatten()
    data_x_dev = bp.Array(data_x_flat, dev)
    
    task_type = TaskType(TARGET(target).cluster_num)
    celu_func = load_op_by_type("Celu", dtype.name)
    task_type = TaskType(TARGET(target).cluster_num)
    output_buffer = np.zeros(len(data_x_flat), dtype=dtype.as_numpy_dtype)
    output_dev = bp.Array(output_buffer, dev)
    alpha = -2.0 #定义alpha
    buffer_alpha = bp.Array(np.array([alpha]).astype(dtype.as_numpy_dtype), dev) 
    with tcp.runtime.Run(task_type):
        celu_func(data_x_dev,buffer_alpha,False, output_dev)
    
    np_ret = output_dev.numpy()
    print("data_x:",data_x_flat)
    print("res:",np_ret)
   
    #bangpy.assert_allclose( np_ret, ret2.astype(dtype.as_numpy_dtype),rtol = 0.1, atol = 0.1)
    
   
            