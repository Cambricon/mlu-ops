# Copyright (C) [2022] by Cambricon, Inc.
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
"""Celu operator implementation using BANGPy TCP API."""
import random
import torch
import numpy as np
import pytest
import bangpy as bp
from bangpy.common import load_op_by_type
from celu import DTYPES, TARGET_LIST


def random_int_list(max_dim_length, each_dim_max_length):
    random_list = []
    for _ in range(max_dim_length):
        random_list.append(random.randint(2, each_dim_max_length))
    return tuple(random_list)


def create_shape_list(
        nram_single_buffer_size_by_byte, append_test_count=50,
        max_dim_length=5, each_dim_max_length=64
):
    const_float32_128_align_element_count = 32
    const_float16_128_align_element_count = 64
    const_current_mlu_single_buffer_float32_max_element_size = int(
        nram_single_buffer_size_by_byte / 4)
    const_current_mlu_single_buffer_float16_max_element_size = int(
        nram_single_buffer_size_by_byte / 2)

    test_shape_list = [
        (0,),
        (1,),
        (2,),

        (const_float32_128_align_element_count - 1,),
        (const_float32_128_align_element_count,),
        (const_float32_128_align_element_count + 1,),
        (const_float16_128_align_element_count - 1,),
        (const_float16_128_align_element_count,),
        (const_float16_128_align_element_count + 1,),

        (const_current_mlu_single_buffer_float32_max_element_size - 1,),
        (const_current_mlu_single_buffer_float32_max_element_size,),
        (const_current_mlu_single_buffer_float32_max_element_size + 1,),
        (const_current_mlu_single_buffer_float16_max_element_size,),
        (const_current_mlu_single_buffer_float16_max_element_size + 1,),

        (246783,),
        (246784,),
        (246785,),
        (123391,),
        (123392,),
        (123393,),
        (2, 2, 3, 3, 4, 3, 2, 4, 2, 3, 4, 4, 2, 3, 5,),
    ]
    for _ in range(append_test_count):
        test_shape_list.append(random_int_list(random.randint(
            2, max_dim_length), random.randint(2, each_dim_max_length)))
    return test_shape_list


shape_list = create_shape_list(
    nram_single_buffer_size_by_byte=(768 - 40) * 1024 // 4,
    append_test_count=0,
    max_dim_length=5,
    each_dim_max_length=64
)


@pytest.mark.parametrize(
    "shape",
    shape_list,
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)
@pytest.mark.parametrize(
    "alpha", [-1.0, -3.1, 2.0, 1.5, 10.1],
)
def test_celu(target, shape, dtype, alpha):
    if target not in TARGET_LIST:
        return

    def celu_out(alpha_param=1.0):
        dev = bp.device(0)
        celu_func = load_op_by_type("Celu", dtype.name)

        def celu_inner(input_param):
            primative = input_param.shape
            data_x_flat = input_param.flatten()
            data_x_dev_param = bp.Array(data_x_flat, dev)
            output_dev_param = bp.Array(np.zeros(len(data_x_flat), dtype=dtype.as_numpy_dtype), dev)
            celu_func(
                data_x_dev_param,
                output_dev_param,
                data_x_flat.size,
                alpha_param,
                )
            celu_res = output_dev_param.numpy().reshape(primative)
            return celu_res
        return celu_inner
    data_x = np.random.uniform(low=-7.5, high=10, size=shape)
    f1 = celu_out(alpha)
    res = f1(data_x.astype(dtype.as_numpy_dtype))
    torch_value = torch.tensor(data_x)
    m = torch.nn.CELU(alpha)
    t_res = m(torch_value)
    bp.assert_allclose(t_res.numpy(), res, rtol=0.1, atol=0.1)
