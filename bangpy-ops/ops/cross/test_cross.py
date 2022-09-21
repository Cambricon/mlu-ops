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

"""test file for ops: cross"""
import numpy as np
import pytest
import bangpy
from bangpy.common import load_op_by_type
from cross import DTYPES, KERNEL_NAME, TARGET_LIST


@pytest.mark.parametrize(
    "shape,dim",
    [
        # It's not suggested to test all cases in one time
        # you may consider to divide them into two groups
        ((1, 1, 1, 1, 2, 3, 4, 5), 5),
        ((2, 1, 2, 1, 2, 2, 2, 3), 7),
        ((2, 1, 2, 1, 2, 2, 3, 3), 6),
        ((3, 2, 2, 1, 1, 1, 1, 1), 0),
        ((2, 2, 2, 3, 3, 4, 4, 4), 4),
        ((1, 2, 2, 2, 3, 128, 1, 1), 4),
        ((1, 2, 2, 2, 3, 128, 1, 1), -4),
        ((1024, 2, 2, 3, 3, 4, 4, 4), 4),
        ((1, 1, 2, 4, 3, 2, 3, 1024), 4),
        ((1, 1024, 2, 4, 3, 2, 3, 1024), 4),
        ((2, 1024, 4, 4, 3, 2, 3, 1024), 4),
        ((1, 1024, 2, 4, 3, 2, 3, 1024), 6),
        # below are illegal input
        # ((1, 2, 2, 2, 3, 128, 1, 1), 1),  # shape[dim]!=3, should fail
        # ((1, 2, 2, 2, 3, 128, 1, 1), -9),  # dim not in [-8,7], should fail
        # ((2,1024,2,4,3,2,8192,2),4),  # step > pipeline buffer, should fail
    ],
)
@pytest.mark.parametrize(
    "dtype",
    DTYPES,
)
def test_cross(target, shape, dim, dtype):
    if target not in TARGET_LIST:
        return
    data_in0 = np.random.uniform(low=-1, high=1, size=shape)
    data_in1 = np.random.uniform(low=-1, high=1, size=shape)

    if not -8 <= dim <= 7:
        raise KeyError("dim shall be in [-8,7], but not")

    if dim < 0:
        dim = dim + 8

    if shape[dim] != 3:
        raise KeyError("shape[dim] is not 3!")

    # To check if step is larger than pipeline buffer
    step = 1
    for i in range(dim + 1, 8):
        step = step * shape[i]

    nram_size = 512 * 1024
    buffer_size = (nram_size - 30 * 1024) // (128 * 18) * (128 * 18) / 18
    # how to compute pipeline buffer's size, refer to cross.py (line 121)
    if dtype == bangpy.float32:
        buffer_size = buffer_size / 4
    elif dtype == bangpy.float16:
        buffer_size = buffer_size / 2

    if step > buffer_size:
        raise KeyError("step is too large!")

    # computation below is functionally similar to torch.cross(data_in0,data_in1,dim),
    # except data type is numpy array instead of tensor.
    axes = list(np.arange(dim))
    axes += list(np.arange(dim + 1, len(shape)))
    axes.append(dim)
    axes2 = list(np.arange(dim))
    axes2.append(len(shape) - 1)
    axes2 += list(np.arange(dim, len(shape) - 1))

    data0 = data_in0.transpose(axes).astype(dtype.as_numpy_dtype)
    data1 = data_in1.transpose(axes).astype(dtype.as_numpy_dtype)
    dataout = np.cross(data0, data1)
    data_out = dataout.transpose(axes2).astype(dtype.as_numpy_dtype)

    dev = bangpy.device(0)

    # set I/O datas

    data_in0_dev = bangpy.Array(data_in0.astype(dtype.as_numpy_dtype), dev)
    data_in1_dev = bangpy.Array(data_in1.astype(dtype.as_numpy_dtype), dev)
    data_out_dev = bangpy.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)
    dimshape = bangpy.Array(np.array(shape).astype(bangpy.int32.as_numpy_dtype), dev)

    f1 = load_op_by_type(KERNEL_NAME, dtype.name)
    f1(
        data_in0_dev,
        data_in1_dev,
        dimshape,
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        shape[5],
        shape[6],
        shape[7],
        dim,
        data_out_dev,
    )

    # Hardware time
    evaluator = f1.time_evaluator(number=1, repeat=1, min_repeat_ms=0)
    latency = (
        evaluator(
            data_in0_dev,
            data_in1_dev,
            dimshape,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            shape[4],
            shape[5],
            shape[6],
            shape[7],
            dim,
            data_out_dev,
        ).median
        * 1e3
    )
    print("Shape:", shape)
    print("dim:", dim)
    print("dtype:", dtype)
    print("Hardware time : %f ms" % latency)

    # io_efficiency
    length = 1
    for i in shape:
        length = length * i
    theory_io_size = length * dtype.bytes * 3
    IO_BANDWIDTH = 2**40  # MLU290: 1024GB/s
    # IO_BANDWIDTH = 307.2 * 2 ** 30  # MLU370-s4: 307.2GB/s
    io_efficiency = 1000 * theory_io_size / (latency * IO_BANDWIDTH)
    print("theory_io_size : %f GB" % (theory_io_size / (2**30)))
    print("io_efficiency:", str(round(io_efficiency * 100, 2)) + "%\n")

    # diff3 test
    # calculate diff3 will cost lots of time, to test efficiency quickly,
    # strongly advise you to skip this part
    data_out = data_out.flatten()
    data_out_dev = data_out_dev.numpy().flatten()
    diff = np.abs(data_out - data_out_dev)
    data_out = np.abs(data_out)
    maxdiff3 = 0
    if dtype == bangpy.float16:
        th = 1e-4
    elif dtype == bangpy.float32:
        th = 1e-6
    for i, data in enumerate(data_out):
        if data > th:
            diff3 = diff[i] / data
        else:
            diff3 = diff[i]
        if diff3 > maxdiff3:
            maxdiff3 = diff3
    assert maxdiff3 == 0
