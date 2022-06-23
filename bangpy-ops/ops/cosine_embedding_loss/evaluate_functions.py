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
"""
测试运行算子所需函数，随机数据生成，计时以及准确度
"""
import json
import random
import numpy as np

import bangpy
from cosine_embedding_loss import CosineEmbeddingLoss, DTYPES
from bangpy.tcp.runtime import TaskType


# numpy格式的数据计算。算子的原始逻辑
def compute_simple_test(x_1, x_2, y, margin):
    upper = np.sum(np.multiply(x_1, x_2), axis = 1)
    lower1 = np.sum(np.multiply(x_1, x_1), axis = 1)
    lower2 = np.sum(np.multiply(x_2, x_2), axis = 1)
    result = (upper / ((lower1 * lower2) ** 0.5)).reshape((-1, ))
    return ((y + 1) * (1 - result) + (1 - y) * np.maximum(0, result - margin)) / 2

# mlu的数据误差
def cal_diff(result, data_out):
    diff1 = np.sum(np.abs(np.subtract(result, data_out))) / np.sum(result)
    diff2 = np.sqrt(
        np.sum(np.power(np.subtract(data_out, result), 2,))
        / np.sum(np.power(result, 2))
    )
    return diff1, diff2

###################################################
# 测试参数
###################################################
# 数据量 1,2,4,8GB
data_amounts = [
    2 ** 20 * 10,
    # 2 ** 30 * 2,
    # 2 ** 30 * 4,
    # 2 ** 30 * 8
]

# 数据宽度，即一行数据的尺寸
data_widths = [
    2 ** 5,
    # 2 ** 6,
    # 2 ** 7,
    # 2 ** 8,
    # 2 ** 9,
    # 2 ** 10,
    2 ** 11,
    # 2 ** 12,
    # 2 ** 13,
    # 2 ** 14,
    # 2 ** 15,
    # 2 ** 16,
    # 2 ** 17,
    2 ** 18,
    2 ** 19,
]
# 数据类型，float16, float32
dtypes = DTYPES[1:2]

def evaluate(f, dtype, data_amount, data_width):
    """
    对每种参数进行计算和时间评估
    """
    data_height = data_amount // dtype.bytes // data_width

    data_input_x1 = np.random.rand(data_width * data_height).reshape((data_height, data_width))
    data_input_x2 = np.random.rand(data_width * data_height).reshape((data_height, data_width))
    data_input_y = np.random.randint(-1, 1, (data_height, ))
    data_input_y = data_input_y * 2 + 1
    margin = random.random()
    data_out = np.zeros((data_height, ))

    dev = bangpy.device(0)

    data_input_x1_dev = bangpy.Array(data_input_x1.astype(dtype.as_numpy_dtype), dev)
    data_input_x2_dev = bangpy.Array(data_input_x2.astype(dtype.as_numpy_dtype), dev)
    data_input_y_dev = bangpy.Array(data_input_y.astype(dtype.as_numpy_dtype), dev)
    data_out_dev = bangpy.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)

    data_out = compute_simple_test(data_input_x1, data_input_x2, data_input_y, margin)
    f(data_input_x1_dev, data_input_x2_dev, data_input_y_dev, margin, data_out_dev)
    # bangpy.assert_allclose(data_out_dev.numpy(), data_out.astype(dtype.as_numpy_dtype))

    dev_out = data_out_dev.numpy()

    diff1, diff2 = cal_diff(dev_out, data_out)

    # 输出结果的差，调试用
    # for i in range(data_height):
    #     print(dev_out[i] - data_out[i], data_input_y[i])

    evaluator = f.time_evaluator(dev, 1, 10)
    time = (
        evaluator(
            data_input_x1_dev,
            data_input_x2_dev,
            data_input_y_dev,
            margin,
            data_out_dev).mean * 1e3
        ) # ms
    io_speed = (
        (data_width * data_height * 2 * dtype.bytes + data_height * dtype.bytes) \
            / time * 1e3 / (2 ** 30)) # GB/s

    # 输出结果
    print("data_type: {} data_amount: {:2.4f}GB data_width: \
        {:7d} time cost: {:3.2f}ms IO speed: {:4.3f}GB/s diff1: \
        {:1.5f}%, diff2: {:1.5f}%".format (
        dtype.name,
        data_amount / (2 ** 30),
        data_width,
        time,
        io_speed,
        round(diff1 * 100, 5),
        round(diff2 * 100, 5),
    ))
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
    测试函数，对每种参数组合进行遍历
    对每种数据类型分别编译生成算子核
    """
    results = [
        ["data_type", "data_amount", "data_width", "time_cost", "IO_speed", "diff1", "diff2"]
    ]
    for dtype in dtypes:
        f = CosineEmbeddingLoss(dtype, 1, "mlu290", TaskType.UNION16).compute_body()
        for data_amount in data_amounts:
            for data_width in data_widths:
                results.append(evaluate(f, dtype, data_amount, data_width))
    # 性能及精确度结果存到json
    filename = "perfomance_log.json"
    with open(filename, "w") as file_obj:
        json.dump(results, file_obj)

# # 主函数
if __name__ == "__main__":
    func()
