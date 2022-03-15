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

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]
KERNEL_NAME = "add"


class Add(object):
    """Operator description:
    Add the data in the two buffers.
    """

    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.bp = tcp.TCP(target)
        self.length = self.bp.SizeVar("length")
        self.nram_size = TARGET(target).nram_size
        self.dtype_sz = dtype.bytes
        self.single_buffer_size = 1024*128 #增加128倍
        self.bp.launch_task(self.task_num, 1, 1)

    def compute_body(self):
        # calculate split strategy
        # gets the data length to be calculated for each task
        data_calculated_each_task = self.length // self.task_num
        # gets the number of cycles required for each task
        loop_num = data_calculated_each_task * self.dtype_sz // self.single_buffer_size
        # gets the data length for each calculation
        data_calculated_each_time = self.single_buffer_size // self.dtype_sz

        gala = self.single_buffer_size // self.dtype_sz
        # declare I/O buffer
        buffer_in0 = self.bp.Buffer(
            shape=(self.length,), name="INPUT0", dtype=self.dtype, scope="global"
        )
        buffer_in1 = self.bp.Buffer(
            shape=(self.length,), name="INPUT1", dtype=self.dtype, scope="global"
        )
        buffer_out = self.bp.Buffer(
            shape=(self.length,), name="OUTPUT", dtype=self.dtype, scope="global"
        )
        task_id = self.bp.taskId
        # declare on-chip buffer
        buffer_in0_n = self.bp.Buffer(
            shape=(gala,),
            name="INPUT0_N",
            dtype=self.dtype,
            scope="nram",
        )
        nram_buffer_in1 = self.bp.Buffer(
            shape=(gala,),
            name="INPUT1_N",
            dtype=self.dtype,
            scope="nram",
        )
        buffer_out_n = self.bp.Buffer(
            shape=(gala,),
            name="OUTPUT_N",
            dtype=self.dtype,
            scope="nram",
        )
     
        # split and compute
        with self.bp.for_range(begin = 0, end = loop_num,stage = 1) as i:
            start = task_id * data_calculated_each_task + i * data_calculated_each_time
            stop = start + data_calculated_each_time
            with self.bp.block("data_copy"):
                self.bp.memcpy(buffer_in0_n, buffer_in0[start:stop])
                self.bp.memcpy(nram_buffer_in1, buffer_in1[start:stop])
            with self.bp.block("compute"):
                self.bp.add(buffer_out_n, buffer_in0_n, nram_buffer_in1)     
            with self.bp.block("data_copy"):
                self.bp.memcpy(buffer_out[start:stop], buffer_out_n)
        # build a executable module
        f = self.bp.BuildBANG(
            inputs=[buffer_in0, buffer_in1],
            outputs=[buffer_out],
            kernel_name=KERNEL_NAME,
        )
        return f


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_add(dtype=None, target=None):
    # tasktype fixed in UNION1
    task_num = 64 #由4 改为64
    f = Add(dtype, target, task_num).compute_body()
    return f
