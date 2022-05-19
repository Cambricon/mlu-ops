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
import sys
import torch
import numpy as np
import pytest
import bangpy
import bangpy as bp
from bangpy.common import load_op_by_type
from celu import DTYPES,TARGET_LIST
sys.path.append("..")
from create_shape import *



test_shape_list = CreatShapeList(
    nram_single_buffer_size_by_byte = (512 - 40) * 1024 // 8,
    append_test_count = 50,
    max_dim_length = 5 ,
    each_dim_max_length = 64
)
@pytest.mark.parametrize(
    "shape",
    test_shape_list,
    #[(30207,),]
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)
def test_celu(target, shape, dtype):
    if target not in TARGET_LIST:
        return
    def celu_out (alpha = 1,inplace = False):
        dev = bp.device(0)
        celu_func = load_op_by_type("Celu",dtype.name)
        def celu_inner(input_param):
            primative = input_param.shape#记录shape
            data_x_flat = input_param.flatten()#压平
            buffer_alpha_param = bp.Array(np.array([alpha]).astype( dtype=dtype.as_numpy_dtype),dev)
            data_x_dev_param = bp.Array(data_x_flat,dev)
            output_dev_param = bp.Array(np.zeros(len(data_x_flat), dtype=dtype.as_numpy_dtype),dev)
            celu_func(data_x_dev_param,buffer_alpha_param,inplace,output_dev_param)#计算
            res = output_dev_param.numpy().reshape(primative)#还原shape
            if inplace :
                input_param=res#如果在原位改动
            return res#返回结果 在原位改动直接拿input 反正俩地址现在一样
        return celu_inner#返回函数
    data_x = np.random.uniform(low=-1000, high=1000, size=shape)
    print("current_shape->",shape,"___",dtype.name)
    gala = celu_out(2,True)
    res = gala(data_x.astype(dtype.as_numpy_dtype))
    torch_value = torch.tensor(data_x)
    m = torch.nn.CELU(2)
    t_res = m(torch_value)
    if dtype.name == "float16":
        bangpy.assert_allclose( t_res.numpy(), res,rtol = 0.01, atol = 0.01)
    else:
        bangpy.assert_allclose( t_res.numpy(), res,rtol = 0.00001, atol = 0.01)
