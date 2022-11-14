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
import math
import torch
import pytest
import bangpy as bp
from bangpy.common import load_op_by_type
from pairwise_distance import DTYPES, KERNEL_NAME, TARGET_LIST, PairwiseDistance
import logging

def create_random_shape(length, size):
    return np.random.randint(low=1, high=size, size=length)

ranshp = create_random_shape(2, 512)
ranshp1 = create_random_shape(5, 3)
ranshp2 = create_random_shape(10, 2)

@pytest.mark.parametrize(
    "shape",
    [
    [(1, 2, 10241 * 100 ), (1, 2, 10241 * 100)],
    [(2, 1, 1, 1, 3, 2, 2, 3, 1, 2, 5, 4 ), (5, 4,)],
    [(30000, 1), (30000, 1)],
    [(1, 3), (1, 3)],
    [(4,5,2), (234)],
    [(1, 482 * 1024), (1, 482 * 1024)],
    [(32, 482 * 1024), (32, 482 * 1024)],
    [ranshp, ranshp],
    [ranshp1, ranshp1],
    [ranshp2, ranshp2],
        [[112, 2], [2]],
        [[112, 12], [1]]
    ],
)

@pytest.mark.parametrize(
    "dtype", DTYPES,
)

@pytest.mark.parametrize(
    "p", [1, 2.2, 3.5, -1.2],
)

@pytest.mark.parametrize(
    "eps", [0.000001, 0.0001],
)

@pytest.mark.parametrize(
    "keepdim", [False, True],
)


def test_pairwise_distance(target, shape, dtype, p, eps, keepdim):    
    if target not in TARGET_LIST:
        return

    def mlu_pairwise_distance(p=2.0, eps=1e-06, keepdim=False):
        def get_total_size(shp):
            size = 1
            for s in shp:
                size *= s
            return size

        def check_shape(s1, s2):
            if len(s2) == 1:
                if s2[0] == 1:
                    return True

            offset = len(s1) - len(s2)
            i = 0
            for _ in s2:
                if s1[offset + i] != s2[i]:
                    return False
                i += 1
            return True

        def f(a, b):
            if len(a) == 0 or len(b) == 0:
                raise Exception("shape err")

            if len(a.shape) == 1 and len(b.shape) == 1:
                raise Exception("shape err")

            #拿到shape
            if len(a.shape) > len(b.shape):
                _shape1 = a.shape
                _shape2 = b.shape
            else:
                _shape1 = b.shape
                _shape2 = a.shape

            if not check_shape(_shape1, _shape2):
                raise Exception("shape err")


            _dev = bp.device(0)

            dim_index = len(_shape1) - 1

            # mlu input parameters
            _pd_len = _shape1[len(_shape1) - 1]
            _pd_height = 1
            _pd_width = 1

            for i in range(0, dim_index + 1):
                _pd_height *= _shape1[i]

            if dim_index == len(_shape1) - 1:
                pass
            else:
                for i in range(dim_index + 1, len(_shape1)):
                    _pd_width *= _shape1[i]

            # mlu input
            _mlu_input1 = bp.Array(a.flatten(), _dev)
            _mlu_input2 = bp.Array(b.flatten(), _dev)
            paras = np.array([p, eps]).astype(dtype.as_numpy_dtype)
            _mlu_paras = bp.Array(paras, _dev)

            # mlu output
            _output_len = get_total_size(_shape1) // _shape1[dim_index]
            output_buffer = np.zeros(_output_len, dtype=dtype.as_numpy_dtype)
            _mlu_output = bp.Array(output_buffer, _dev)

            output_count = 256
            output_buffer2 = np.zeros(output_count, dtype=dtype.as_numpy_dtype)
            _mlu_border_output = bp.Array(output_buffer2, _dev)

            output_buffer3 = -np.ones(output_count, dtype=np.int32)
            _mlu_border_idx_output = bp.Array(output_buffer3, _dev)


            # call mlu interface

            eager_mode = True
            if not eager_mode:
                func = load_op_by_type(KERNEL_NAME, dtype.name, dtype.bytes)
                func(_mlu_input1, _mlu_input2,
                     _mlu_paras,
                     get_total_size(_shape1), get_total_size(_shape2),
                     _pd_len, _pd_height, _pd_width, _output_len
                     , _mlu_border_output
                     , _mlu_border_idx_output
                     , _mlu_output
                     )
                result = _mlu_output.numpy()
                result_border_idx = _mlu_border_idx_output.numpy()
            else:
                func = PairwiseDistance(dtype.name, dtype.bytes)
                func(a.flatten(), b.flatten(),
                     paras,
                     get_total_size(_shape1), get_total_size(_shape2),
                     _pd_len, _pd_height, _pd_width, _output_len
                     , output_buffer2
                     , output_buffer3
                     , output_buffer
                     )

                result = output_buffer
                result_border_idx = output_buffer3

            s = set()
            for i in result_border_idx:
                s.add(i)

            for item in s:
                if item >= 0:
                    result[item] = math.pow(result[item], 1 / p)


            def create_output_shape(shp, dim_idx):
                outputshape = []
                if keepdim:
                    for item in shp:
                        outputshape.append(item)
                    outputshape[dim_idx] = 1
                else:
                    for i in range(0, len(shp) - 1):
                        outputshape.append(shp[i])
                return outputshape

            return result.reshape(create_output_shape(_shape1, dim_index))

        return f


    m_ori_input1 = np.random.uniform(low=-5, high=5, size=shape[0])
    m_ori_input2 = np.random.uniform(low=-5, high=5, size=shape[1])

    try:
        mlu_ret = mlu_pairwise_distance(p=p, eps=eps, keepdim=keepdim)\
            (m_ori_input1.astype(dtype.as_numpy_dtype), \
            m_ori_input2.astype(dtype.as_numpy_dtype))
    except Exception as err:
        print(str(err))
        if str(err) == "shape err":
            return

        raise Exception(str(err)) from err


    cpu_ret = torch.nn.PairwiseDistance(p=p, eps=eps, keepdim=keepdim)\
        (torch.Tensor(m_ori_input1), torch.Tensor(m_ori_input2)).numpy()

    bp.assert_allclose(cpu_ret, mlu_ret, rtol = 0.01, atol = 0.01)
