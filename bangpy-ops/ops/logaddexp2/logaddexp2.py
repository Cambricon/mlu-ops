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
"""Logaddexp2 for bangpy tcp."""
import math
import bangpy
from bangpy import tcp
from bangpy.tcp.runtime import TaskType
from bangpy.platform.bang_config import TARGET

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]

KERNEL_NAME = "logaddexp2"


class Logaddexp2():
    """Operator description:
    logaddexp2 the data in the two buffers.
    """

    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.task_num = task_num
        self.tcp = tcp.TCP(target)
        self.length = self.tcp.SizeVar("length")
        # 3*2+2=8 buffers: 3 pipeline double buffer(input1, input2, output), and 2 extra buffer.
        # buffer size need to be 128 aligned
        self.single_buffer_size = (TARGET(target).nram_size - 128*2**10) // 8
        self.single_buffer_size = self.single_buffer_size // 128 * 128
        self.tcp.launch_task(self.task_num, 1, 1)
        self.tcp.launch_cluster(TaskType.BLOCK)

    def compute_body(self):
        """Return logaddexp2 function."""
        # calculate split strategy
        # gets the data length for each calculation
        data_each_time = self.single_buffer_size // self.dtype.bytes
        # gets the data length to be calculated for each task
        # the last task need to handle the extra remainder
        data_each_task  = self.tcp.Scalar(bangpy.int32, name="data_each_task")
        with self.tcp.if_scope(self.tcp.taskId == self.task_num - 1):
            data_each_task.assign(self.length // self.task_num + self.length % self.task_num)
        with self.tcp.else_scope():
            data_each_task.assign(self.length // self.task_num)
        # gets the number of cycles required for each task, round down
        loop_num  = self.tcp.Scalar(bangpy.int32, name="loop_num")
        loop_num.assign(data_each_task // data_each_time)

        # declare I/O buffer
        buffer_in0 = self.tcp.Buffer( shape=(self.length,), name="INPUT0", dtype=self.dtype,
            scope="global")
        buffer_in1 = self.tcp.Buffer( shape=(self.length,), name="INPUT1", dtype=self.dtype,
            scope="global")
        buffer_out = self.tcp.Buffer( shape=(self.length,), name="OUTPUT", dtype=self.dtype,
            scope="global")
        buffer_extra0 = self.tcp.Buffer( shape=(data_each_time,), name="E0_N", dtype=self.dtype,
            scope="nram")
        buffer_extra1 = self.tcp.Buffer( shape=(data_each_time,), name="E1_N", dtype=self.dtype,
            scope="nram")

        # compute
        start = self.tcp.Scalar(bangpy.int32, name="start")
        start.assign(self.tcp.taskId * (self.length // self.task_num))
        stop = self.tcp.Scalar(bangpy.int32, name="stop")

        def compute(out, in0, in1, ex0, ex1):
            # swap in0 and in1 to make sure in0 >= in1 (anthor implemention, use active_relu?)
            # in0 = max(in0, in1)
            # in1 = min(in0, in1)
            self.tcp.minimum(ex0, in0, in1)
            self.tcp.maximum(in0, in0, in1)
            in1, ex0 = ex0, in1 # equal to self.tcp.memcpy(in1, ex0), but reduce copy time

            # ex0 = in0 - in1
            # out = in1 + log2(1+2**(in0-in1))
            self.tcp.subtract(ex0, in0, in1)
            self.tcp.exp2(out, ex0)
            self.tcp.add(out, out, 1)
            self.tcp.log(out, out, high_precision=False)
            self.tcp.multiply(out, out, 1/math.log(2))
            self.tcp.add(out, out, in1)
            # if in0-in1 > 15, out = in0
            # in1: mask, if greater than 15, set 1
            # ex0: ~in1
            self.tcp.greater(in1, ex0, 15)
            self.tcp.less_equal(ex1, ex0, 15)
            self.tcp.multiply(in0, in0, in1)
            self.tcp.multiply(out, out, ex1)
            self.tcp.add(out, out, in0)

        with self.tcp.for_range(0, loop_num, stage=1) as i:
            # declare on-chip buffer
            buffer_in0_n = self.tcp.Buffer( shape=(data_each_time,), name="INPUT0_N",
                dtype=self.dtype, scope="nram",)
            buffer_in1_n = self.tcp.Buffer( shape=(data_each_time,), name="INPUT1_N",
                dtype=self.dtype, scope="nram",)
            buffer_out_n = self.tcp.Buffer( shape=(data_each_time,), name="OUTPUT_N",
                dtype=self.dtype, scope="nram",)

            with self.tcp.block("data_copy"):
                self.tcp.memcpy(buffer_in0_n[:data_each_time],
                    buffer_in0[start + data_each_time * i: start + data_each_time * (i+1)])
                self.tcp.memcpy(buffer_in1_n[:data_each_time],
                    buffer_in1[start + data_each_time * i: start + data_each_time * (i+1)])
            with self.tcp.block("compute"):
                compute(buffer_out_n, buffer_in0_n, buffer_in1_n, buffer_extra0, buffer_extra1)
            with self.tcp.block("data_copy"):
                self.tcp.memcpy(buffer_out[start + data_each_time * i:
                    start + data_each_time * (i+1)],
                    buffer_out_n[:data_each_time])

        # compute remainder
        with self.tcp.if_scope(data_each_task % data_each_time != 0):
            start.assign(start + data_each_time * loop_num)
            stop.assign(start + data_each_task % data_each_time)
            self.tcp.memcpy(buffer_in0_n[:stop-start], buffer_in0[start:stop])
            self.tcp.memcpy(buffer_in1_n[:stop-start], buffer_in1[start:stop])
            compute(buffer_out_n, buffer_in0_n, buffer_in1_n, buffer_extra0, buffer_extra1)
            self.tcp.memcpy(buffer_out[start:stop], buffer_out_n[:stop-start])
        # build a executable module
        func = self.tcp.BuildBANG(
            inputs=[buffer_in0, buffer_in1],
            outputs=[buffer_out],
            kernel_name=KERNEL_NAME,
        )
        return func


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_logaddexp2(dtype=None, target=None):
    # tasktype is BLOCK
    task_num = TARGET(target).cluster_num * TARGET(target).core_num
    func = Logaddexp2(dtype, target, task_num).compute_body()
    return func
