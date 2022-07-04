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
"""
测试运行算子所需函数，随机数据生成，计时以及准确度
function needed when testing operation.
including generating random data, timing and compute accuracy
"""
import json
import random
import numpy as np

import bangpy
from cosine_embedding_loss import CosineEmbeddingLoss, DTYPES
from bangpy.tcp.runtime import TaskType


def compute_simple_test(x_1, x_2, y, margin):
    """
    Cpu type of operation.
    """
    x_1 = x_1.astype(np.float32)
    x_2 = x_2.astype(np.float32)
    y = y.astype(np.float32)
    upper = np.sum(np.multiply(x_1, x_2), axis=1)
    lower1 = np.sum(np.multiply(x_1, x_1), axis=1)
    lower2 = np.sum(np.multiply(x_2, x_2), axis=1)
    result = (upper / lower1 * upper / lower2) ** 0.5 * (upper / abs(upper)).reshape(
        (-1,)
    )
    return ((y + 1) * (1 - result) + (1 - y) * np.maximum(0, result - margin)) / 2


# Diffs.
def cal_diff(result, data_out):
    """
    Compute diff1 & 2 between cpu result and mlu result.
    """
    result = result.astype(np.float32)
    data_out = data_out.astype(np.float32)
    diff1 = np.sum(np.abs(np.subtract(result, data_out))) / np.sum(result)
    diff2 = np.sqrt(
        np.sum(np.power(np.subtract(data_out, result), 2,))
        / np.sum(np.power(result, 2))
    )
    return diff1, diff2


###################################################
# Testing Parameters
###################################################
# Data amounts 1,2,4,8GB.
data_amounts = [2 ** 20 * 10, 2 ** 30, 2 ** 30 * 2, 2 ** 30 * 4, 2 ** 30 * 8]

# Data width.
data_widths = [
    2 ** 5,
    2 ** 5 + 1,
    2 ** 5 - 1,
    2 ** 6,
    2 ** 7,
    2 ** 8,
    2 ** 9,
    2 ** 10,
    2 ** 11 + 1,
    2 ** 11 - 1,
    2 ** 12,
    2 ** 13,
    2 ** 14,
    2 ** 15,
    2 ** 16,
    2 ** 17,
    2 ** 18,
    2 ** 19,
]

def evaluate(f, dtype, data_amount, data_width):
    """
    Evaluate function.
    """
    data_height = data_amount // dtype.bytes // data_width

    data_input_x1 = np.random.rand(data_width * data_height).reshape(
        (data_height, data_width)
    )
    data_input_x2 = np.random.rand(data_width * data_height).reshape(
        (data_height, data_width)
    )
    data_input_y = np.random.randint(-1, 1, (data_height,))
    data_input_y = data_input_y * 2 + 1
    margin = random.random()
    data_out = np.zeros((data_height,))

    dev = bangpy.device(0)

    data_input_x1_dev = bangpy.Array(data_input_x1.astype(dtype.as_numpy_dtype), dev)
    data_input_x2_dev = bangpy.Array(data_input_x2.astype(dtype.as_numpy_dtype), dev)
    data_input_y_dev = bangpy.Array(data_input_y.astype(dtype.as_numpy_dtype), dev)
    data_out_dev = bangpy.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)

    data_out = compute_simple_test(
        data_input_x1.astype(dtype.as_numpy_dtype),
        data_input_x2.astype(dtype.as_numpy_dtype),
        data_input_y.astype(dtype.as_numpy_dtype),
        margin,
    ).astype(dtype.as_numpy_dtype)
    f(data_input_x1_dev, data_input_x2_dev, data_input_y_dev, margin, data_out_dev)

    dev_out = data_out_dev.numpy()

    diff1, diff2 = cal_diff(dev_out, data_out.astype(dtype.as_numpy_dtype))

    evaluator = f.time_evaluator(dev, 1, 10)
    time = (
        evaluator(
            data_input_x1_dev, data_input_x2_dev, data_input_y_dev, margin, data_out_dev
        ).mean
        * 1e3
    )  # ms
    io_speed = (
        (data_width * data_height * 2 * dtype.bytes + data_height * dtype.bytes)
        / time
        * 1e3
        / (2 ** 30)
    )  # GB/s

    # Output results.
    print(
        "data_type: {} data_amount: {:2.4f}GB data_width: \
        {:7d} time cost: {:3.2f}ms IO speed: {:4.3f}GB/s diff1: \
        {:1.5f}%, diff2: {:1.5f}%".format(
            dtype.name,
            data_amount / (2 ** 30),
            data_width,
            time,
            io_speed,
            round(diff1 * 100, 5),
            round(diff2 * 100, 5),
        )
    )
    return [
        dtype.name,
        data_amount / (2 ** 30),
        data_width,
        time,
        io_speed,
        round(diff1 * 100, 5),
        round(diff2 * 100, 5),
    ]


def func():
    """
    Test function.
    """
    results = [
        [
            "data_type",
            "data_amount",
            "data_width",
            "time_cost",
            "IO_speed",
            "diff1",
            "diff2",
        ]
    ]
    for dtype in DTYPES:
        f = CosineEmbeddingLoss(dtype, 1, "mlu290", TaskType.UNION16).compute_body()
        for data_amount in data_amounts:
            for data_width in data_widths:
                results.append(evaluate(f, dtype, data_amount, data_width))
    # Store testing result into json file.
    filename = "perfomance_log.json"
    with open(filename, "w") as file_obj:
        json.dump(results, file_obj)


# Main function.
if __name__ == "__main__":
    func()
