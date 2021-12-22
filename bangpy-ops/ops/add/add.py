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
# pylint: disable=missing-docstring, invalid-name, too-many-locals
"""A multi-platform code link example test for BANGPy TCP."""
import numpy as np

import bangpy
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float16]
TRAGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]
SHAPE = (64000,)
KERNEL_NAME = "add"


class Add(object):
    """Operator description:
    Add the data in the two buffers.
    """

    def __init__(self, shape, dtype, target, task_num):
        self.shape = shape
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.length = np.prod(shape)
        self.nram_size = TARGET(target).nram_size
        self.dtype_sz = dtype.bytes
        self.bp = tcp.TCP(target)
        self.bp.launch_task(self.task_num, 1, 1)

    def compute_body(self):
        # calculate split strategy
        # ensure the data size can be divisible by task_num and 128 bytes aligned
        assert (self.dtype_sz * self.length) % self.task_num % ALIGN_LENGTH == 0
        # gets the data length to be calculated for each task
        data_calculated_each_task = self.length // self.task_num
        loop_num = np.ceil(
            3 * data_calculated_each_task * self.dtype_sz / self.nram_size
        )
        # ensure the data size is 128 bytes aligned for each calculation
        while (
            data_calculated_each_task % loop_num != 0
            or data_calculated_each_task // loop_num % ALIGN_LENGTH != 0
        ):
            loop_num += 1
        data_calculated_each_time = int(data_calculated_each_task // loop_num)
        # declare I/O buffer
        buffer_in0 = self.bp.Buffer(
            shape=self.shape, name="INPUT0", dtype=self.dtype, scope="global"
        )
        buffer_in1 = self.bp.Buffer(
            shape=self.shape, name="INPUT1", dtype=self.dtype, scope="global"
        )
        buffer_out = self.bp.Buffer(
            shape=self.shape, name="OUTPUT", dtype=self.dtype, scope="global"
        )
        task_id = self.bp.taskId
        # declare on-chip buffer
        buffer_in0_n = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="INPUT0_N",
            dtype=self.dtype,
            scope="nram",
        )
        buffer_in1_n = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="INPUT1_N",
            dtype=self.dtype,
            scope="nram",
        )
        buffer_out_n = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="OUTPUT_N",
            dtype=self.dtype,
            scope="nram",
        )
        # split and compute
        with self.bp.for_range(0, loop_num) as i:
            start = task_id * data_calculated_each_task + i * data_calculated_each_time
            stop = start + data_calculated_each_time
            self.bp.memcpy(buffer_in0_n, buffer_in0[start:stop])
            self.bp.memcpy(buffer_in1_n, buffer_in1[start:stop])
            self.bp.add(buffer_out_n, buffer_in0_n, buffer_in1_n)
            self.bp.memcpy(buffer_out[start:stop], buffer_out_n)
        # build a executable module
        f = self.bp.BuildBANG(
            inputs=[buffer_in0, buffer_in1],
            outputs=[buffer_out],
            kernel_name=KERNEL_NAME,
        )
        return f


@tcp.register_mlu_op(DTYPES, TRAGET_LIST, KERNEL_NAME)
def build_add(dtype=None, target=None):
    # tasktype fixed in UNION1
    task_num = 4
    f = Add(SHAPE, dtype, target, task_num).compute_body()
    return f
