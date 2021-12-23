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
# pylint: disable=too-many-locals
"""NonZero test demo."""
import numpy as np
import pytest
from nonzero import NonZero, TARGET_LIST
from nonzero_count import NonZeroCount
import bangpy as bp
from bangpy import tcp
from bangpy.tcp.runtime import TaskType
from bangpy.platform.bang_config import TARGET
from bangpy.common import load_op_by_type
import os


@pytest.mark.parametrize(
    "trans",
    [0, 1],
)
@pytest.mark.parametrize(
    "dtype",
    [bp.float16],
)
@pytest.mark.parametrize(
    "shape",
    [
        (4, 16, 120, 64),
        (4, 32, 120, 128),
        (2, 2, 16, 120),
        (4, 8, 16, 1280),
    ],
)
def test_nonzero(target, trans, dtype, shape):
    """Test case."""
    if target not in TARGET_LIST:
        return
    task_num = TARGET(target).cluster_num * TARGET(target).core_num
    dim_num = 4
    in_np = np.random.randint(0, 2, size=shape).astype(dtype.name)

    count_np = np.zeros((task_num,), dtype="uint32")
    dev = bp.device(0)
    in_data = bp.Array(in_np, dev)
    count_data = bp.Array(count_np, dev)

    task_type = TaskType(TARGET(target).cluster_num)
    f_nonzero_count = load_op_by_type("NonZeroCount", dtype.name)
    f_nonzero = load_op_by_type("NonZero", dtype.name)

    with tcp.runtime.Run(task_type):
        f_nonzero_count(in_data, shape[0], shape[1], shape[2], shape[3], count_data)

    num_nonzero = int(np.sum(count_data.numpy()))
    out_np = np.ones((dim_num * num_nonzero,), dtype="int64")
    out_data = bp.Array(out_np.astype("int64"), dev)

    with tcp.runtime.Run(task_type):
        f_nonzero(
            in_data,
            count_data,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            dim_num,
            num_nonzero,
            trans,
            out_data,
        )

    cpu_out = np.nonzero(in_np.reshape((shape[-dim_num:])))
    cpu_out = np.array(list(cpu_out)).astype("int64")
    mlu_out = out_data.numpy().astype("int64").reshape((dim_num, num_nonzero))

    if trans:
        cpu_out = cpu_out.transpose((1, 0))
        mlu_out = mlu_out.reshape((num_nonzero, dim_num))

    bp.assert_allclose(
        mlu_out,
        cpu_out,
        rtol=0.0,
        atol=0.0,
    )
