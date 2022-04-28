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
import random
import bangpy
import bangpy as bp
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType
from pairwise_distance import DTYPES, KERNEL_NAME, TARGET_LIST

import time

#max_append_test_count = 50 #附加测试的最大次数
current_append_test_count = 1 # random.randint(0,max_append_test_count)#本次运行附加测试次数    值为0-最大次数之间随机数
max_dim_length =5 #最大维度数
each_dim_max_length =80 #每个维度最大长度


#################################################
#    此值应与kernel中的值保持同步更新状态
#################################################
current_mlu_single_buffer_bytes = (512 - 40) * 1024 // 8 #当前kernel中为每个nram_buffer 预留的字节数 


const_float32_128_align_element_count = 32 #float32 下 128字节对应元素个数
const_float16_128_align_element_count = 64 #float16 下 128字节对应元素个数
const_current_mlu_single_buffer_float32_max_element_size = int(current_mlu_single_buffer_bytes / 4) # float32下 单个nram_buffer的最大元素数
const_current_mlu_single_buffer_float16_max_element_size = int(current_mlu_single_buffer_bytes / 2) # float16下 单个nram_buffer的最大元素数


#此处填充必测shape
test_shape_list=[
    # 1,
    # 2,
    # const_float32_128_align_element_count - 1, #不足128对齐
    # const_float32_128_align_element_count    , #满足128对齐
    # const_float32_128_align_element_count + 1, #128对齐后多1个

    # const_float16_128_align_element_count - 1,
    # const_float16_128_align_element_count    ,
    # const_float16_128_align_element_count + 1,

    # const_current_mlu_single_buffer_float32_max_element_size - 1, #比空间大小少一个元素 
    # const_current_mlu_single_buffer_float32_max_element_size    , #刚好用完空间
    # const_current_mlu_single_buffer_float32_max_element_size + 1, #比空间大小多一个元素 

    # const_current_mlu_single_buffer_float16_max_element_size - 1,
    # const_current_mlu_single_buffer_float16_max_element_size    ,
    # const_current_mlu_single_buffer_float16_max_element_size + 1,
]

def random_int_list( max_dim_length, each_dim_max_length):
    random_list = []
    for i in range(max_dim_length):
        random_list.append(random.randint(1, each_dim_max_length))
    return tuple(random_list)

for i in range(current_append_test_count):
    
    test_shape_list.append(random_int_list(random.randint(2, max_dim_length),random.randint(1, each_dim_max_length)))


@pytest.mark.parametrize(
    "shape", 
    test_shape_list,
)

@pytest.mark.parametrize(
    "dtype", DTYPES,
)

@pytest.mark.parametrize(
    "p", [3.],
)

@pytest.mark.parametrize(
    "eps", [0.000001,],
)

@pytest.mark.parametrize(
    "keepdim", [False],
)




def test_pairwise_distance(target, shape, p, eps, keepdim, dtype): 
    if target not in TARGET_LIST:
        return 

    def mlu_pairwise_distance(p, eps, keepdim):
        def get_total_size(shp):
            size = 1
            for s in shp:
                size *= s
            return size

        def f(a, b):
            #拿到shape
            if len(a.shape) > len(b.shape):
                _shape1 = a.shape
                _shape2 = b.shape
            else:
                _shape1 = b.shape
                _shape2 = a.shape
            
            _dev = bp.device(0)

            shp_len = len(_shape1)
            dim_index = shp_len - 1

            # mlu 输入参数
            _pd_len = _shape1[shp_len - 1]
            _pd_height = 1
            _pd_width = 1

            for i in range(0, dim_index + 1):
                _pd_height *= _shape1[i]

            if dim_index == shp_len - 1:
                pass
            else:
                for i in range(dim_index + 1, shp_len):
                    _pd_width *= _shape1[i] 

            # mlu 输入
            _mlu_input1 = bp.Array(a.flatten(), _dev)
            _mlu_input2 = bp.Array(b.flatten(), _dev)
            paras = np.array([p, eps]).astype(dtype.as_numpy_dtype) # 这里需要考虑
            _mlu_paras = bp.Array(paras, _dev)

            # mlu 输出
            _output_len = get_total_size(_shape1) // _shape1[dim_index]
            output_buffer = np.zeros(_output_len, dtype=dtype.as_numpy_dtype)
            _mlu_output = bp.Array(output_buffer, _dev)

            output_count = 256
            output_buffer2 = np.zeros(output_count, dtype=dtype.as_numpy_dtype)
            _mlu_border_output = bp.Array(output_buffer2, _dev)

            output_buffer3 = -np.ones(output_count, dtype=np.int32)
            _mlu_border_idx_output = bp.Array(output_buffer3, _dev)

            # 调用mlu
            func = load_op_by_type(KERNEL_NAME, dtype.name)
            func(_mlu_input1, _mlu_input2,
                 _mlu_paras, 
                 get_total_size(_shape1), get_total_size(_shape2),
                 _pd_len, _pd_height, _pd_width, _output_len
                 , _mlu_border_output, _mlu_border_idx_output, _mlu_output)

            result = _mlu_output.numpy()
            result_border = _mlu_border_output.numpy()
            result_border_idx = _mlu_border_idx_output.numpy()

            #收尾
            s = set()
            for i in result_border_idx:
                s.add(i)

            for item in s:
                if item >= 0:
                    result[item] = math.pow(result[item], 1 / p)

            outputshape = []
            if keepdim:
                for item in _shape1:
                    outputshape.append(item)
                outputshape[dim_index] = 1
            else:
                for i in range(0, len(_shape1) - 1):
                    outputshape.append(_shape1[i])

            ret = result.reshape(outputshape)
            return ret

        return f

    print("current_shape->",shape)
    # shape1 = np.array(shape).astype('int32')
    # shape2 = np.array(shape).astype('int32')
    _ori_input1 = np.random.uniform(low=-1.5, high=1.5, size=shape)
    _ori_input2 = np.random.uniform(low=-1.5, high=1.5, size=shape)

    
    pdist = mlu_pairwise_distance(p=p, eps=eps, keepdim=keepdim)
    mlu_ret = pdist(_ori_input1.astype(dtype.as_numpy_dtype), _ori_input2.astype(dtype.as_numpy_dtype))
    


    cpu_start = time.time()
    pdist = torch.nn.PairwiseDistance(p=p, eps=eps, keepdim=keepdim)
    tensor1 = torch.Tensor(_ori_input1)
    tensor2 = torch.Tensor(_ori_input2)
    cpu_ret = pdist(tensor1, tensor2)
    print("mlu->",mlu_ret)
    print("cpu->",cpu_ret)
    bangpy.assert_allclose( cpu_ret.numpy(), mlu_ret,rtol = 0.01, atol = 0.01)
    



   
#   改动说明 
#   1. 将shape修改成随机生成 +  必测shape 的结合
#   2. 暂未测shape不等的情况
#   3. 将原本input的定值 改成随机值
#   4. 将input的数据类型转化放在kernel入参时再做
#   5. 使用断言来判断结果
#