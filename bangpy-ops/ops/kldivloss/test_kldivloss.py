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
# pylint: disable=too-many-locals
"""KlDivloss test demo."""
import numpy as np
import pytest
import bangpy as bp
from bangpy.common import load_op_by_type
from kldivloss import DTYPES, KERNEL_NAME, TARGET_LIST


def cal_diff(result, data_out, reduction):
    """compute diff"""
    if reduction == 0:
        result[np.isnan(result)] = 0
        result[np.isinf(result)] = 0
        data_out[np.isnan(data_out)] = 0
        data_out[np.isinf(data_out)] = 0

    bp.assert_allclose(np.isnan(result), np.isnan(data_out))
    bp.assert_allclose(np.isinf(result), np.isinf(data_out))

    diff1 = np.sum(
        np.abs(np.subtract(result, data_out, dtype=np.float64)), dtype=np.float64
    ) / np.sum(result, dtype=np.float64)
    diff2 = np.sqrt(
        np.sum(
            np.power(
                np.subtract(data_out, result, dtype=np.float64), 2, dtype=np.float64
            ),
            dtype=np.float64,
        )
        / np.sum(np.power(result, 2), dtype=np.float64),
        dtype=np.float64,
    )

    print("DIFF1:", str(round(diff1 * 100, 5)) + "%")
    print("DIFF2:", str(round(diff2 * 100, 5)) + "%")
    assert round(diff1 * 100, 5) < 3e-3 * 100
    assert round(diff2 * 100, 5) < 3e-3 * 100


@pytest.mark.parametrize(
    "shape",
    [
        (2 ** 10,),
        (2 ** 12,),
        (2 ** 20,),
        (2 ** 26,),
        (2, 2 ** 20,),
        (2, 4, 8, 1, 2 ** 12,),
    ],
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)
@pytest.mark.parametrize(
    "reduction", [0, 1, 2, 3],
)
@pytest.mark.parametrize(
    "log_target", [0, 1],
)
def test_kldivloss(target, shape, dtype, reduction, log_target):
    """Test kldivloss operator by giving multiple sets of parameters."""
    if target not in TARGET_LIST:
        return

    # Convert the size of the input data to the new style (batchnum, length)
    print(
        "shape is :",
        shape,
        "reduction is :",
        reduction,
        "log_target is :",
        log_target,
        "dtype is :",
        dtype,
    )
    inputdim = len(shape)

    batchnum = shape[0]
    length = 1
    if inputdim == 1:
        batchnum = 1
        length = shape[0]
    else:
        for i in range(1, inputdim):
            length = length * shape[i]
    shape = (batchnum, length)
    totaldata = batchnum * length

    #  By default, input has been logged
    data_input = np.random.uniform(low=0, high=1, size=shape).astype(
        dtype.as_numpy_dtype
    )
    data_input = np.log(data_input)

    data_target = np.random.uniform(low=0, high=1, size=shape).astype(
        dtype.as_numpy_dtype
    )

    # Output argument
    if log_target == 0:
        data_out = np.multiply(
            data_target, np.subtract(np.log(data_target), (data_input))
        )
    elif log_target == 1:
        data_out = np.multiply(
            np.exp(data_target), np.subtract((data_target), (data_input))
        )
    # Reduction operation
    if reduction == 0:
        print("data_out : ", data_out)
    if reduction == 1:
        data_out_sum = np.sum(data_out)
        print("data_out_sum : ", data_out_sum)
    if reduction == 2:
        data_out_mean = np.mean(data_out)
        print("data_out_mean : ", data_out_mean)
    if reduction == 3:
        data_out_batchmean = np.sum(data_out) / batchnum
        print("data_out_batchmean : ", data_out_batchmean)

    dev = bp.device(0)
    # Set I/O data
    data_input_dev = bp.Array(data_input.astype(dtype.as_numpy_dtype), dev)
    data_target_dev = bp.Array(data_target.astype(dtype.as_numpy_dtype), dev)
    data_out_dev = bp.Array(
        np.zeros(data_out.flatten().shape, dtype.as_numpy_dtype), dev
    )
    reduction_result_dev = bp.Array(np.zeros((1,), np.float32), dev)

    f = load_op_by_type(KERNEL_NAME, dtype.name)
    f(
        data_input_dev,
        data_target_dev,
        data_out_dev,
        batchnum,
        length,
        reduction,
        log_target,
        reduction_result_dev,
    )

    evaluator = f.time_evaluator(number=100, repeat=2, min_repeat_ms=0)
    t = (
        evaluator(
            data_input_dev,
            data_target_dev,
            data_out_dev,
            batchnum,
            length,
            reduction,
            log_target,
            reduction_result_dev,
        ).mean
        * 1e3
    )

    data_out_dev = data_out_dev.numpy().reshape(shape)
    reduction_result_dev = reduction_result_dev.numpy()
    if reduction == 0:
        print("data_out_dev : ", data_out_dev)
        cal_diff(data_out, data_out_dev, reduction)
    elif reduction == 1:
        print("data_out_sum_dev : ", reduction_result_dev[0])
        cal_diff(data_out_sum, reduction_result_dev[0], reduction)
    elif reduction == 2:
        print("data_out_mean_dev : ", reduction_result_dev[0])
        cal_diff(data_out_mean, reduction_result_dev[0], reduction)
    elif reduction == 3:
        print("data_out_batchmean_dev : ", reduction_result_dev[0])
        cal_diff(data_out_batchmean, reduction_result_dev[0], reduction)

    print("tutorial : %f ms" % t)

    # io_efficiency
    theory_io_size = (
        3 * totaldata * dtype.bytes if reduction == 0 else 2 * totaldata * dtype.bytes
    )
    # io_bandwidth = 2 ** 40  # MLU290: 1024GB/s
    io_bandwidth = 307.2 * 2 ** 30  # MLU370-s4: 307.2GB/s
    io_efficiency = 1000 * theory_io_size / (t * io_bandwidth)
    print("theory_io_size : %f GB" % (theory_io_size / (2 ** 30)))
    print("io_efficiency:", str(round(io_efficiency * 100, 2)) + "%")
    print("------------------------------------------------------------")
