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
# The above copyright notice and this permission notice shall self.tcp included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Test adjustHue operator with multi-platform code link."""
import os
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
    diff1 = np.sum(np.abs(np.subtract(result, data_out))) / np.sum(result)
    diff2 = np.sqrt(
        np.sum(
            np.power(
                np.subtract(data_out, result),
                2,
            )
        )
        / np.sum(np.power(result, 2))
    )
    assert round(diff1 * 100, 5) < 10
    assert round(diff2 * 100, 5) < 10
    print("DIFF1:", str(round(diff1 * 100, 5)) + "%")
    print("DIFF2:", str(round(diff2 * 100, 5)) + "%")

@pytest.mark.parametrize(
    "shape",
    [
        (4, 64, 64, 3),
        (1, 339, 576, 3),
        (4, 480, 720, 3),
        (6, 2160, 4096, 3),
        (1, 512, 512, 3),
        (6, 720, 1080, 3),
        (4, 1080, 1920, 3),
        (6, 1080, 2048, 3),
        (1, 4320, 7680, 3),
    ],
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)
@pytest.mark.parametrize(
    "delta", [-0.45932, 0.7373],
)
def test_adjust_hue(target, shape, delta, dtype):
    if target not in TARGET_LIST:
        return
    n = shape[0]
    h = shape[1]
    w = shape[2]
    c = shape[3]
    stride = w *c * dtype.bytes
    print("shape:", [n, h, w, c], " delta:", delta, " dtype:", dtype.name)
    # set device
    dev = bp.device(0)
    # generate input data
    data_in = np.zeros((n, h, stride // dtype.bytes), dtype="float32")
    for i in range(n):
        data_in[i] = np.random.uniform(
            low=0, high=1, size=(1, h, stride // dtype.bytes)
        )

    data_out = np.zeros((n, h, stride // dtype.bytes), dtype="float32")
    data_in_handle = bp.Array(data_in.astype(dtype.name), dev)
    data_out_handle = bp.Array(data_out.astype(dtype.name), dev)
    f = load_op_by_type(
        KERNEL_NAME, dtype.name
    )
    f(
        data_in_handle,
        delta,
        n,
        h,
        w,
        c,
        stride,
        stride,
        data_out_handle,
    )
    # convert all output to float for diff comparison
    mlu_result = np.zeros((n, h, w, c), dtype="float32")
    for i in range(n):
        mlu_result[i] = (
            data_out_handle
            .numpy()[i, :, : w * c]
            .reshape(h, w, c)
            .astype("float32")
        )
    cpu_result = np.zeros((n, h, w, c), dtype="float32")
    for i in range(n):
        cpu_result[i] = adjust_hue_cpu(
            data_in_handle.numpy()[i, :, : w * c].reshape([h, w, c]), delta
        )
    # calculate difference between cpu and mlu results
    cal_diff(cpu_result, mlu_result)
