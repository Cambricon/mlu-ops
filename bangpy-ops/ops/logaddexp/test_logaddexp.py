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
"""Test LogAddExp operator with multi-platform code link."""

import random
import numpy as np
import pytest
import bangpy as bp
from bangpy.common import load_op_by_type
from logaddexp import DTYPES, TARGET_LIST


def run(input_x, input_y, dtype):
    max_size_buffer = input_x
    min_size_buffer = input_y
    is_sub = True
    scale_up = 1
    is_single_element = False
    if max_size_buffer.ndim < min_size_buffer.ndim:
        max_size_buffer = input_y
        min_size_buffer = input_x

    for i in range(1, min_size_buffer.ndim + 1):
        if max_size_buffer.shape[-1 * i] != min_size_buffer.shape[-1 * i]:
            is_sub = False
            break

    if min_size_buffer.ndim == 1 and min_size_buffer.shape[0] == 1:
        is_sub = True
        is_single_element = True

    if is_sub:
        if not is_single_element:
            for j in range(max_size_buffer.ndim - min_size_buffer.ndim):
                scale_up *= max_size_buffer.shape[-1 * min_size_buffer.ndim - j - 1]
        else:
            for k in max_size_buffer.shape:
                scale_up *= k
        dev = bp.device(0)
        x = bp.Array(max_size_buffer.astype(dtype.as_numpy_dtype).flatten(), dev)
        y = bp.Array(
            np.tile(min_size_buffer.astype(dtype.as_numpy_dtype).flatten(), scale_up),
            dev
        )
        output_dev = bp.Array(np.zeros(max_size_buffer.size, dtype = dtype.as_numpy_dtype), dev)
        log_add_exp_func = load_op_by_type("LogAddExp", dtype.name)
        log_add_exp_func(x, y, output_dev,max_size_buffer.size)
        return output_dev.numpy().reshape(max_size_buffer.shape)

    raise Exception("shape err")


def random_int_list(max_dim_length, each_dim_max_length):
    random_list = []
    for _ in range(max_dim_length):
        random_list.append(random.randint(2, each_dim_max_length))
    return tuple(random_list)


def create_shape_list(
        nram_single_buffer_size_by_byte,
        append_test_count = 50,
        max_dim_length = 5,
        each_dim_max_length = 64
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
        (const_current_mlu_single_buffer_float16_max_element_size - 1,),
        (const_current_mlu_single_buffer_float16_max_element_size,),
        (const_current_mlu_single_buffer_float16_max_element_size + 1,),
        (23295,),
        (23296,),
        (23297,),
        (186368,),
        (186369,),
        (232960,),
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
    nram_single_buffer_size_by_byte = 93184,
    append_test_count = 50,
    max_dim_length = 5,
    each_dim_max_length = 64
)



@pytest.mark.parametrize(
    "shape",
    shape_list,
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)
def test_logaddexp(target, shape, dtype):
    if target not in TARGET_LIST:
        return

    # origin input
    sub_shape = shape[random.randint(0, len(shape) - 1):len(shape)]
    data_x = np.random.uniform(low = -800, high = 800, size = shape)
    data_y = np.random.uniform(low = -800, high = 800, size = sub_shape)
    try:
        mlu_ret = run(data_x, data_y, dtype)
    except Exception as err:
        if str(err) == "shape err":
            return
        raise Exception(str(err)) from err

    cpu_ret = np.logaddexp(data_x, data_y)
    if dtype.name == "float16":
        bp.assert_allclose(mlu_ret, cpu_ret.astype(dtype.as_numpy_dtype), rtol = 0.1, atol = 0.1)
    else:
        bp.assert_allclose(mlu_ret, cpu_ret.astype(dtype.as_numpy_dtype), rtol = 0.001, atol = 0.01)
