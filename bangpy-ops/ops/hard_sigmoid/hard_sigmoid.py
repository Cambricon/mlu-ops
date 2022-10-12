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
"""HardSigmoid operator implementation using BANGPy TCP Script API."""

import bangpy
from bangpy import tcp
from bangpy.script import ty, build_module
from bangpy.platform.bang_config import TARGET

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu290", "mlu370-s4"]
KERNEL_NAME = "hard_sigmoid"


class HardSigmoid(object):
    """Operator description
    Applies the Hardsigmoid function element-wise.
    hard_sigmoid function:
        h(x) = 0 if x <= -3,
        h(x) = 1 if x >= +3,
        h(x) = x * 1/6 + 1/2 otherwise.
        input: tensor.
        output: tensor(same shape as the input).

    Parameters of split strategy
    ----------------------------
        data_total: total number of data.
        self.task_num: number of task of MLU.
        data_each_task: number of data of per task.
        data_rem: the remainder data after distributing to per task.
        self.nram.size: the size of NRAM (Bytes).
        self.nram_size_each_buffer: the space of NRAM was divided into three parts.
        data_each_time: number of data of IPU(task) calculation per time.
        loop_num: number of times each task needs to be copied into NRAM for computation.
        data_rem_n: less than one calculation.
    """

    def __init__(
        self, task_num: ty.int32, nram_size: ty.int32, dtype: ty.string
    ) -> None:
        self.task_num = task_num
        self.nram_size = nram_size
        self.dtype = dtype
        # Notes:
        # (1)The space of NRAM was divided into three parts：buffer_io_n * 2 and buffer_temp_n
        # (2)Buffer size must be 128-byte aligned
        # (3)NRAM needs to reserve a little space(it is 4KB here)
        self.nram_size_each_buffer = ((self.nram_size - 4 * 1024) // 3) // 128 * 128

    def hardSigmoid_body(
        self, local_x: ty.Buffer("nram"), local_temp: ty.Buffer("nram"),
    ) -> None:
        """The body of hard_sigmoid function."""
        tcp.multiply(local_x, local_x, 1.0 / 6)  # x * 1/6
        tcp.add(local_x, local_x, 1.0 / 2)  # x * 1/6 + 1/2
        tcp.assign(local_temp, tcp.cast(1, self.dtype))  # tcp.assign(local_temp, 1)
        tcp.minimum(local_x, local_x, local_temp)  # min(x * 1/6 + 1/2, 1)
        tcp.assign(local_temp, tcp.cast(0, self.dtype))  # tcp.assign(local_temp, 0)
        tcp.maximum(local_x, local_x, local_temp)  # max(x * 1/6 + 1/2, 0)

    def main(
        self,
        buffer_in: ty.handle,
        buffer_out: ty.handle,
        length: ty.int32,
        inplace: ty.int32,
    ) -> None:
        # declare I/O buffer
        buffer_in = tcp.match_buffer(buffer_in, [length], dtype=self.dtype)
        buffer_out = tcp.match_buffer(buffer_out, [length], dtype=self.dtype)

        tgt = tcp.target()

        # calculate split strategy
        data_total = length
        data_each_task = data_total // self.task_num
        data_rem = data_total % self.task_num
        data_each_time = (
            self.nram_size_each_buffer // 2
            if self.dtype == "float16"
            else self.nram_size_each_buffer // 4  # self.dtype == "float32"
        )
        loop_num = data_each_task // data_each_time
        data_rem_n = data_each_task % data_each_time
        if data_rem_n > 0:
            loop_num = loop_num + 1  # copy the remaining data into NRAM for calculation

        # calculate
        for cluster_id in tcp.thread_binding(0, tgt.cluster_num, thread="blockIdx.x"):
            for core_id in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):
                task_id = cluster_id * tgt.core_num + core_id
                for i in range(loop_num, pipeline=True):
                    buffer_io_n = tcp.alloc_buffer(
                        [data_each_time], dtype=self.dtype, scope="nram"
                    )
                    buffer_temp_n = tcp.alloc_buffer(
                        [data_each_time], dtype=self.dtype, scope="nram"
                    )
                    with tcp.block("data_copy"):
                        start = task_id * data_each_task + i * data_each_time
                        stop = start + data_each_time
                        if i == loop_num - 1 and data_rem_n > 0:
                            tcp.memcpy(
                                buffer_io_n[0:data_rem_n],
                                buffer_in[start : start + data_rem_n],
                            )
                        else:
                            tcp.memcpy(
                                buffer_io_n, buffer_in[start:stop],
                            )
                    with tcp.block("compute"):
                        self.hardSigmoid_body(buffer_io_n, buffer_temp_n)

                    with tcp.block("data_copy"):
                        start = task_id * data_each_task + i * data_each_time
                        stop = start + data_each_time
                        if i == loop_num - 1 and data_rem_n > 0:
                            if inplace == 1:
                                tcp.memcpy(
                                    buffer_in[start : start + data_rem_n],
                                    buffer_io_n[0:data_rem_n],
                                )
                            else:
                                tcp.memcpy(
                                    buffer_out[start : start + data_rem_n],
                                    buffer_io_n[0:data_rem_n],
                                )
                        else:
                            if inplace == 1:
                                tcp.memcpy(buffer_in[start:stop], buffer_io_n)
                            else:
                                tcp.memcpy(buffer_out[start:stop], buffer_io_n)
                if data_rem > 0:
                    if (
                        task_id == self.task_num - 1
                    ):  # 0 < data_rem < self.task_num, give it to the last task
                        stop = data_total
                        start = stop - data_rem
                        buffer_io_n = tcp.alloc_buffer(
                            [128], dtype=self.dtype, scope="nram"
                        )
                        buffer_temp_n = tcp.alloc_buffer(
                            [128], dtype=self.dtype, scope="nram"
                        )
                        tcp.memcpy(buffer_io_n[0:data_rem], buffer_in[start:stop])
                        self.hardSigmoid_body(
                            buffer_io_n[0:128],
                            buffer_temp_n[0:128],
                            # buffer size must be 128-byte aligned
                            # float16: 128 * 2B
                            # float32: 128 * 4B
                            # Note: We assume that self.task_num <= 128
                        )
                        if inplace == 1:
                            tcp.memcpy(buffer_in[start:stop], buffer_io_n[0:data_rem])
                        else:
                            tcp.memcpy(buffer_out[start:stop], buffer_io_n[0:data_rem])


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_hard_sigmoid(dtype=None, target=None):
    task_num = TARGET(target).cluster_num * TARGET(target).core_num
    nram_size = TARGET(target).nram_size
    f = build_module.build(
        HardSigmoid(task_num, nram_size, dtype.name),
        target_tag=target,
        name=KERNEL_NAME,
    )
    return f
