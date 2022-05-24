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
# pylint: disable=useless-object-inheritance, too-many-instance-attributes
# pylint: disable=attribute-defined-outside-init, too-many-statements
# pylint: disable=too-many-arguments, too-many-locals
"""Lerp operator implementation using BANGPy TCP API."""
import numpy as np

import bangpy
from bangpy import tcp
from bangpy.tcp.util import round_up, round_down
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu370-s4", "mlu270", "mlu290"]
KERNEL_NAME = "lerp"


class Lerp(object):
    """Operator description:
    Does a linear interpolation of two tensors start and end based on a scalar
    or tensor weight and returns the resulting out tensor.
    """

    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.bp = tcp.TCP(target)
        self.dim_n = self.bp.SizeVar("dim_n")
        self.dim_h = self.bp.SizeVar("dim_h")
        self.dim_w = self.bp.SizeVar("dim_w")
        self.dim_c = self.bp.SizeVar("dim_c")
        self.length = self.bp.SizeVar("length")
        self.nram_size = TARGET(target).nram_size
        self.dtype_sz = dtype.bytes
        self.single_nram_size = round_down(
            (self.nram_size - 30 * 1024) // 4 // self.dtype_sz, ALIGN_LENGTH
        )
        self.bp.launch_task(self.task_num, 1, 1)

        # global
        self.buffer_start = self.bp.Buffer(
            shape=(self.dim_n, self.dim_h, self.dim_w, self.dim_c),
            dtype=self.dtype,
            name="buffer_start",
            scope="global"
        )
        self.buffer_end = self.bp.Buffer(
            shape=(self.dim_n, self.dim_h, self.dim_w, self.dim_c),
            dtype=self.dtype,
            name="buffer_end",
            scope="global"
        )
        self.buffer_weight = self.bp.Buffer(
            shape=(self.dim_n, self.dim_h, self.dim_w, self.dim_c),
            dtype=self.dtype,
            name="buffer_weight",
            scope="global"
        )
        self.buffer_out = self.bp.Buffer(
            shape=(self.dim_n, self.dim_h, self.dim_w, self.dim_c),
            dtype=self.dtype,
            name="buffer_out",
            scope="global"
        )

    def compute_body(self):
        data_num = self.bp.Scalar(dtype=bangpy.int32, name="data_num")
        data_num.assign(self.dim_n * self.dim_h * self.dim_w * self.dim_c)
        average_core = self.bp.Scalar(dtype=bangpy.int32, name="average_core")
        average_core.assign(data_num / self.task_num)
        remain_core = self.bp.Scalar(dtype=bangpy.int32, name="remain")
        remain_core.assign(data_num % self.task_num)

        # flatten
        flatten_buffer_strat = self.buffer_start.reshape((data_num,))
        flatten_buffer_end = self.buffer_end.reshape((data_num,))
        flatten_buffer_weight = self.buffer_weight.reshape((data_num,))
        flatten_buffer_out = self.buffer_out.reshape((data_num,))

        task_id = self.bp.taskId
        core_start = task_id * average_core
        core_end = core_start + average_core
        repeat = average_core // self.single_nram_size
        remain = average_core % self.single_nram_size

        with self.bp.for_range(0, repeat) as i:
            start = core_start + i * self.single_nram_size
            end = start + self.single_nram_size
            # nram
            self.buffer_start_n = self.bp.Buffer(
                shape=(self.single_nram_size,),
                name="INPUT_START_N",
                dtype=self.dtype,
                scope="nram",
            )
            self.buffer_end_n = self.bp.Buffer(
                shape=(self.single_nram_size,),
                name="INPUT_END_N",
                dtype=self.dtype,
                scope="nram",
            )
            self.buffer_weight_n = self.bp.Buffer(
                shape=(self.single_nram_size,),
                name="INPUT_WEIGHT_N",
                dtype=self.dtype,
                scope="nram",
            )
            self.buffer_out_n = self.bp.Buffer(
                shape=(self.single_nram_size,),
                name="OUTPUT_N",
                dtype=self.dtype,
                scope="nram",
            )

            # compute
            self.bp.memcpy(self.buffer_start_n, flatten_buffer_strat[start:end])
            self.bp.memcpy(self.buffer_end_n, flatten_buffer_end[start:end])
            self.bp.memcpy(self.buffer_weight_n, flatten_buffer_weight[start:end])

            self.bp.subtract(self.buffer_end_n, self.buffer_end_n, self.buffer_start_n)
            self.bp.muladd(self.buffer_out_n, self.buffer_weight_n, self.buffer_end_n, self.buffer_start_n)
            self.bp.multiply(self.buffer_end_n, self.buffer_end_n, self.buffer_weight_n)
            self.bp.add(self.buffer_out_n, self.buffer_start_n, self.buffer_end_n)

            self.bp.memcpy(flatten_buffer_out[start:end], self.buffer_out_n)

        with self.bp.if_scope(remain != 0):
            start = core_start + repeat * self.single_nram_size
            end = start + remain

            self.bp.memcpy(self.buffer_start_n[:remain], flatten_buffer_strat[start:end])
            self.bp.memcpy(self.buffer_end_n[:remain], flatten_buffer_end[start:end])
            self.bp.memcpy(self.buffer_weight_n[:remain], flatten_buffer_weight[start:end])

            self.bp.subtract(self.buffer_end_n, self.buffer_end_n, self.buffer_start_n)
            # self.bp.muladd(self.buffer_out_n, self.buffer_weight_n, self.buffer_end_n, self.buffer_start_n)
            self.bp.multiply(self.buffer_end_n, self.buffer_end_n, self.buffer_weight_n)
            self.bp.add(self.buffer_out_n, self.buffer_start_n, self.buffer_end_n)

            self.bp.memcpy(flatten_buffer_out[start:end], self.buffer_out_n[:remain])

        with self.bp.if_scope(remain_core != 0):
            with self.bp.if_scope(task_id == self.task_num - 1):
                start = task_id * average_core
                end = start + remain_core

                self.bp.memcpy(self.buffer_start_n[:remain_core], flatten_buffer_strat[start:end])
                self.bp.memcpy(self.buffer_end_n[:remain_core], flatten_buffer_end[start:end])
                self.bp.memcpy(self.buffer_weight_n[:remain_core], flatten_buffer_weight[start:end])

                self.bp.subtract(self.buffer_end_n, self.buffer_end_n, self.buffer_start_n)
                # self.bp.muladd(self.buffer_out_n, self.buffer_weight_n, self.buffer_end_n, self.buffer_start_n)
                self.bp.multiply(self.buffer_end_n, self.buffer_end_n, self.buffer_weight_n)
                self.bp.add(self.buffer_out_n, self.buffer_start_n, self.buffer_end_n)

                self.bp.memcpy(flatten_buffer_out[start:end], self.buffer_out_n[:remain_core])

        self.buffer_out = flatten_buffer_out.reshape((self.dim_n, self.dim_h, self.dim_w, self.dim_c))

        return self.bp.BuildBANG(
            inputs=[
                self.buffer_start,
                self.buffer_end,
                self.buffer_weight
            ],
            outputs=[
                self.buffer_out
            ],
            kernel_name=KERNEL_NAME,
        )

@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_lerp(dtype=None, target=None):
    task_num = TARGET(target).cluster_num * TARGET(target).core_num
    f = Lerp(dtype, target, task_num).compute_body()
    return f
