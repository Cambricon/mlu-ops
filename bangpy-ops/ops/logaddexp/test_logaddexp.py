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
import sys
sys.path.append("..")
from create_shape import *
test_shape_list = CreatShapeList(nram_single_buffer_size_by_byte = (512 - 40) * 1024 // 8,append_test_count = 30,max_dim_length = 5 ,each_dim_max_length = 64 )
@pytest.mark.parametrize(
    "shape", 
    test_shape_list,
    #[(5,),]
    
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)

def test_logaddexp(target, shape, dtype):
    if target not in TARGET_LIST:
        return
   
    print("dtype->",dtype,"___shape->",shape)
    # origin input
    
    sub_shape = shape[random.randint(0, len(shape) - 1):len(shape)]
    data_x = np.random.uniform(low=-1000, high=1000, size=shape)
    data_y = np.random.uniform(low=-1000, high=1000, size=sub_shape)
    # data_x = np.random.uniform(low=-1000, high=1000, size=(1,))
    # data_y = np.random.uniform(low=-1000, high=1000, size=(9,2))

    def logaddexp(input_x,input_y):
        max = input_x
        min = input_y

        is_sub = True #是否是子集
        scale_up = 1 #缩放倍数
        is_single_element = False #是否存在单元素
        print("x_shape",input_x.shape)
        print("y_shape",input_y.shape)
        
        if len(max.shape) - len(min.shape) < 0 :#不同维度找出高维的    
            max = input_y
            min = input_x   
        if len(max.shape) ==1 and len(min.shape) == 1:#当同为一维时 根据数组长度确定哪个是大的
            if max.shape[0] - min.shape[0] < 0 :
                max = input_y
                min = input_x        
        for i in range(len(min.shape)): #倒序比较 shape各元素 
            if max.shape[-1 - i] != min.shape[-1 - i] :#循环未结束 出现不同 说明短的不是长的子集  不能进行计算
                is_sub = False
                break
        if len(min.shape) == 1 and min.shape[0] == 1 :#特殊情况  当短的只有一个元素时是可以计算的
            is_sub = True
            is_single_element =True
        if is_sub :
            if not is_single_element :#如果是多个元素 计算差值部分的乘积作为缩放短的缩放倍数
                for j in range(len(max.shape) - len(min.shape)) :
                    scale_up *= max.shape[ -1*len(min.shape) -j -1]
            else:#如果只有一个元素 则缩放倍数为长的shape各元素的乘积
                for k in range(len(max.shape)):
                    scale_up *= max.shape[k]
            dev = bp.device(0)         
            x = bp.Array(max.astype(dtype.as_numpy_dtype).flatten(), dev)         
            y = bp.Array(np.tile(min.astype(dtype.as_numpy_dtype).flatten(),scale_up), dev)
            output_dev = bp.Array(np.zeros(max.size, dtype=dtype.as_numpy_dtype), dev)
            task_type = TaskType(TARGET(target).cluster_num)
            log_add_exp_func = load_op_by_type("LogAddExp", dtype.name)
            with tcp.runtime.Run(task_type):
                log_add_exp_func(x, y, output_dev)
            return output_dev.numpy().reshape(max.shape)
        else :
            print("need the same shape or sub shape")
            
    mlu_ret = logaddexp(data_x,data_y)
    ret2 = np.logaddexp(data_x, data_y)
    if dtype.name == "float16":
        bangpy.assert_allclose( mlu_ret, ret2.astype(dtype.as_numpy_dtype),rtol = 0.01, atol = 0.01)
    else:
        bangpy.assert_allclose( mlu_ret, ret2.astype(dtype.as_numpy_dtype),rtol = 0.001, atol = 0.01)
    


