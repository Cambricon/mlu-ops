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
# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring
"""Test adjustHue operator with multi-platform code link."""
import numpy as np
import matplotlib
import pytest
from adjust_hue import DTYPES, KERNEL_NAME, TARGET_LIST
import bangpy as bp
from bangpy.common import load_op_by_type

np.set_printoptions(threshold=np.inf)


def adjust_hue_cpu(image, delta):
    image = matplotlib.colors.rgb_to_hsv(image)
    image[:, :, 0] += delta
    # matplotlib takes input from [0, 1],round new_h value to [0, 1]
    image[image > 1] -= 1
    image[image < 0] += 1
    image = matplotlib.colors.hsv_to_rgb(image)
    return image


def cal_diff(result, data_out):
    temp_l = 0
    temp_r = 0
    for i in range(result.shape[0]):
        temp = np.subtract(result[i], data_out[i])
        temp = np.abs(temp)
        temp_l += np.sum(temp)
        temp_r += np.sum(result)
        del temp
    diff1 = temp_l / temp_r
    assert round(diff1 * 100, 5) < 3e-3 * 100
    print("DIFF1:", str(round(diff1 * 100, 5)) + "%")
    temp_l = 0
    temp_r = 0
    for i in range(result.shape[0]):
        temp = np.subtract(data_out[i], result[i])
        temp = np.power(temp, 2)
        temp_l += np.sum(temp)
        temp = np.power(result[i], 2)
        temp_r += np.sum(temp)
        del temp
    diff2 = temp_l / temp_r
    diff2 = np.sqrt(diff2)
    assert round(diff2 * 100, 5) < 3e-3 * 100
    print("DIFF2:", str(round(diff2 * 100, 5)) + "%")


@pytest.mark.parametrize(
    "shape",
    [
        (4, 64, 64, 3),
        (1, 339, 576, 3),
        (4, 480, 720, 3),
        # (6, 2160, 4096, 3),
        (1, 512, 512, 3),
        (6, 720, 1080, 3),
        (4, 1080, 1920, 3),
        # (6, 1080, 2048, 3),
        # (1, 4320, 7680, 3),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    DTYPES,
)
@pytest.mark.parametrize(
    "delta",
    [-0.45932, 0.7373],
)
def test_adjust_hue(target, shape, delta, dtype):
    if target not in TARGET_LIST:
        return
    n = shape[0]
    h = shape[1]
    w = shape[2]
    c = shape[3]
    print("shape:", [n, h, w, c], " delta:", delta, " dtype:", dtype.name)
    # set device
    dev = bp.device(0)
    # generate input data
    data_in = np.zeros((n, h, w, c), dtype="float32")
    for i in range(n):
        data_in[i] = np.random.uniform(low=0, high=1, size=(1, h, w, c))

    data_out = np.zeros((n, h, w, c), dtype="float32")
    data_in_handle = bp.Array(data_in.astype(dtype.name), dev)
    data_out_handle = bp.Array(data_out.astype(dtype.name), dev)
    f = load_op_by_type(KERNEL_NAME, dtype.name)
    f(data_in_handle, data_out_handle, n, h, w, c, delta)
    # convert all output to float for diff comparison
    mlu_result = np.zeros((n, h, w, c), dtype="float32")
    for i in range(n):
        mlu_result[i] = data_out_handle.numpy()[i, :, :, :].astype("float32")
    cpu_result = np.zeros((n, h, w, c), dtype="float32")
    for i in range(n):
        cpu_result[i] = adjust_hue_cpu(data_in_handle.numpy()[i, :, :, :], delta)
    # calculate difference between cpu and mlu results
    cal_diff(cpu_result, mlu_result)
    del cpu_result
    del data_out
    del data_in
