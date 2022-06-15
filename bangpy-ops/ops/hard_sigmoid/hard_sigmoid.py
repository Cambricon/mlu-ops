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
import bangpy
from bangpy import tcp
from bangpy.platform.bang_config import TARGET
from bangpy.tcp.runtime import TaskType


DTYPES = [bangpy.float16,bangpy.float32]
TARGET_LIST = ["mlu290"]
KERNEL_NAME = "hard_sigmoid"


class Hard_sigmoid(object):
    """Operator description
    tensor-->activation function-->another tensor after activation.
    """

    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.target = target
        self.cluster_num=task_num // 4
        self.task_num = task_num
        self.tcp = tcp.TCP(target)
        self.length = self.tcp.SizeVar("length")
        self.nram_size = TARGET(target).nram_size
        self.dtype_sz = dtype.bytes
        # 3*buffer_n:   buffer_io_n*2(double buffering) + buffer_temp_n
        # align: the amounts of elements (dtype:float32)
        # 需要按64个float32对齐，也就是256B对齐，然后可能还是很接近NRAM的最大空间，会有错误，所以又减小了1*256
        self.nram_size_buffer=(((self.nram_size // 3) // 256 - 1) * 256)
        self.tcp.launch_cluster(TaskType.BLOCK)
        self.tcp.launch_task(task_num,1,1)

    def compute_body(self):
        # declare I/O buffer
        buffer_in = self.tcp.Buffer(
            shape=(self.length, ),
            name="INPUT", dtype=self.dtype, scope="global"
        )
        buffer_out = self.tcp.Buffer(
            shape=(self.length, ),
            name="OUTPUT", dtype=self.dtype, scope="global"
        )
        task_id=self.tcp.taskId
        # calculate split strategy
        data_total = self.tcp.Scalar(bangpy.int32,"data_total")
        data_total.assign(self.length)
        data_each_task = self.tcp.Scalar(bangpy.int32,"data_each_task")
        data_each_task.assign(data_total // self.task_num)
        data_rem = self.tcp.Scalar(bangpy.int32,"data_rem")
        data_rem.assign(data_total % self.task_num)
        data_each_time = self.nram_size_buffer // self.dtype_sz
        # data_each_time: can't use Scalar:(error)caanot handle this data type
        loop = self.tcp.Scalar(bangpy.int32,"loop")
        loop.assign(data_each_task // data_each_time)
        data_rem_n = self.tcp.Scalar(bangpy.int32,"data_rem_n")
        data_rem_n.assign(data_each_task  % data_each_time)
        # parameters:
        # data_total: total number of data
        # self.task_num: number of task(s)
        # data_each_task: number of data to be calculated per task
        # data_rem: the remainder after averaging over all IPUs
        # self.nram.size: the calculation size of NRAM (Bytes)
        # self.dtype_sz: number of bytes occupied by different data types
        # data_each_time: number of data per time of IPU calculation
        # loop: number of times each task needs to be copied into NRAM for computation
        # data_rem_n: less than one calculation

        # calculate:
        with self.tcp.for_range(0, loop, stage=1) as i:
            start = task_id * data_each_task + i * data_each_time
            stop = start + data_each_time
            buffer_io_n = self.tcp.Buffer(
                shape=(data_each_time,),
                name="IO_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_temp_n = self.tcp.Buffer(
                shape=(data_each_time,),
                name="TEMP_N",
                dtype=self.dtype,
                scope="nram",
            )
            with self.tcp.block("data_copy"):
                self.tcp.memcpy(buffer_io_n,buffer_in[start:stop])
            with self.tcp.block("compute"):
                self.tcp.assign(buffer_temp_n,1/6)
                self.tcp.multiply(buffer_io_n,buffer_io_n,buffer_temp_n)
                self.tcp.assign(buffer_temp_n,1/2)
                self.tcp.add(buffer_io_n,buffer_io_n,buffer_temp_n)
                self.tcp.assign(buffer_temp_n,1)
                self.tcp.minimum(buffer_io_n,buffer_io_n,buffer_temp_n)
                self.tcp.zeros(buffer_temp_n)
                self.tcp.maximum(buffer_io_n,buffer_io_n,buffer_temp_n)
            with self.tcp.block("data_copy"):
                self.tcp.memcpy(buffer_out[start:stop], buffer_io_n)
        # if data_rem_n > 0
        with self.tcp.if_scope(data_rem_n > 0):
            start = task_id * data_each_task + loop * data_each_time
            stop = start + data_rem_n
            buffer_io_n = self.tcp.Buffer(
                shape=(data_each_time,),
                name="IO_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_temp_n = self.tcp.Buffer(
                shape=(data_each_time,),
                name="TEMP_N",
                dtype=self.dtype,
                scope="nram",
            )
            self.tcp.memcpy(buffer_io_n[0:data_rem_n],buffer_in[start:stop])
            self.tcp.assign(buffer_temp_n,1/6)
            self.tcp.multiply(buffer_io_n,buffer_io_n,buffer_temp_n)
            self.tcp.assign(buffer_temp_n,1/2)
            self.tcp.add(buffer_io_n,buffer_io_n,buffer_temp_n)
            self.tcp.assign(buffer_temp_n,1)
            self.tcp.minimum(buffer_io_n,buffer_io_n,buffer_temp_n)
            self.tcp.zeros(buffer_temp_n)
            self.tcp.maximum(buffer_io_n,buffer_io_n,buffer_temp_n)
            self.tcp.memcpy(buffer_out[start:stop], buffer_io_n[0:data_rem_n])
        # if data_rem > 0:
        # 1 <= data_rem <= task_num-1, let master thread to adress it
        with self.tcp.if_scope(data_rem > 0):
            with self.tcp.if_scope(task_id==0):
                stop = data_total
                start = data_total - data_rem
                buffer_io_n = self.tcp.Buffer(
                    shape=(data_each_time,),
                    name="IO_N",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_temp_n = self.tcp.Buffer(
                    shape=(data_each_time,),
                    name="TEMP_N",
                    dtype=self.dtype,
                    scope="nram",
                )
                self.tcp.memcpy(buffer_io_n[0:data_rem],buffer_in[start:stop])
                self.tcp.assign(buffer_temp_n,1/6)
                self.tcp.multiply(buffer_io_n,buffer_io_n,buffer_temp_n)
                self.tcp.assign(buffer_temp_n,1/2)
                self.tcp.add(buffer_io_n,buffer_io_n,buffer_temp_n)
                self.tcp.assign(buffer_temp_n,1)
                self.tcp.minimum(buffer_io_n,buffer_io_n,buffer_temp_n)
                self.tcp.zeros(buffer_temp_n)
                self.tcp.maximum(buffer_io_n,buffer_io_n,buffer_temp_n)
                self.tcp.memcpy(buffer_out[start:stop], buffer_io_n[0:data_rem])

        # build a executable module
        f = self.tcp.BuildBANG(
            inputs=[buffer_in],
            outputs=[buffer_out],
            kernel_name=KERNEL_NAME,
        )
        return f

@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_hard_sigmoid(dtype=None, target=None):
    # tasktype fixed in BLOCK
    task_num = 64
    f = Hard_sigmoid(dtype, target, task_num).compute_body()
    return f
