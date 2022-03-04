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
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType
from logaddexp import DTYPES, KERNEL_NAME, TARGET_LIST

def numcheck(input_pow_arr_x,input_pow_arr_y):
    natural_exponential_x=np.exp(input_pow_arr_x)
    natural_exponential_y=np.exp(input_pow_arr_y)
    antilogarithm=np.add(natural_exponential_x,natural_exponential_y) 
    return np.log(antilogarithm)
@pytest.mark.parametrize(
    "shape", 
    [
        
        (1,),
        (2,),
        (3,),
        (15,),
        (17,),
        (99,),
        (128,),
        (1946,),
        (8192,),
    ],
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)

#
#一共测了两种数据类型  在算子源码dtype数组中写的那两种  每种测了三次  每次的数组长度 看37行开始的那个数组
#



def test_logaddexp(target, shape, dtype): #shape 用来设置数组到底长个啥样子 （2048，）（为1时 显示空的）    长度为2048的一维数组 （4096，3） 每行4096个元素共3行 （x,y,z）参考多维数组定义
    print("check____start")
    if target not in TARGET_LIST:#如果给定的设备不在支持的设备型号列表中 退出
        return
    
    #np.set_printoptions(precision=4)
    data_in0 = np.random.uniform(low=-5, high=5, size=shape).astype(dtype.as_numpy_dtype)# 从-5开始（包含）到5（不包含）的shape形状的数组（这里是一维的）
    data_in1 = np.random.uniform(low=-5, high=5, size=shape).astype(dtype.as_numpy_dtype)# 从-5开始（包含）到5（不包含）的shape形状的数组（这里是一维的）
    #data_in1 = np.random.uniform(low=-5, high=5, size=shape)
    #data_out = data_in0.astype(dtype.as_numpy_dtype) 
    data_out = numcheck(data_in0,data_in1)
   
   
    # data_out = data_in0.astype(dtype.as_numpy_dtype) + data_in1.astype(
    #     dtype.as_numpy_dtype
    # )#astype numpy库数组的转换数组数据类型的函数 参数是类型名的字符串
    #as_numpy_dtype  应该是将传进来的python类型转为为numpy自己的内置类型  （numpy自己定义了自己的一套数据类型）
    # print("primitive______")
    # print(data_in0)
    dev = bangpy.device(0)#找设备
    # set I/O data
    data_in0_dev = bangpy.Array(data_in0, dev)
    
    data_in1_dev = bangpy.Array(data_in1, dev)
    data_out_dev = bangpy.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)
   
    f1 = load_op_by_type(KERNEL_NAME, dtype.name)
    #f1(data_in0_dev,  data_out_dev)
    f1(data_in0_dev,data_in1_dev,  data_out_dev)
    # print("cpu_in:_______________")
    # print(data_in0)
    # print("dev_in:_______________")
    # print(data_in0_dev)
    print("cpu_out:______________")
    print(data_out)
    print("dev_out:_______________")
    print(data_out_dev)
    
    bangpy.assert_allclose(
        data_out_dev.numpy(), data_out.astype(dtype.as_numpy_dtype),rtol=0.1,atol=0.1
    )
    # evaluator = f1.time_evaluator(number=100, repeat=1, min_repeat_ms=0)
    # print('time consuming : %f ms' % (evaluator(data_in0_dev, data_in1_dev, data_out_dev).mean, 1e7))

    print("----------logaddexp-test-end---------------------")
