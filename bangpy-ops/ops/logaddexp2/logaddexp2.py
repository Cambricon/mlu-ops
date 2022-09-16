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
import bangpy
from bangpy.script import tcp, build_module, ty

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]

KERNEL_NAME = "logaddexp2"

class Logaddexp2:
    """Operator description:
    2 inputs and 1 ouput,
    output = log2(2^input0 + 2^input1), calculate element-wise.
    """

    def __init__(self, dtype: ty.string, arch: ty.string) -> None:
        self.dtype = dtype
        if self.dtype == "float32":
            self.dtype_size = 4
        else: # dtype == "float16"
            self.dtype_size = 2
        self.log2 = 0.6931471805599453
        self.arch = arch

    def compute_body(
        self,
        out: ty.Buffer("nram"),
        in0: ty.Buffer("nram"),
        in1: ty.Buffer("nram"),
        ex0: ty.Buffer("nram"),
        ex1: ty.Buffer("nram"),
    ) -> None:
        """compute logic of logaddexp2 is here"""
        # move in0 and in1 to ex0 and ex1, and make sure ex0 >= ex1
        tcp.maximum(ex0, in0, in1)
        tcp.minimum(ex1, in0, in1)

        # in0 = ex0 - ex1
        # out = ex1 + log2(1+2**(ex0-ex1))
        tcp.subtract(in0, ex0, ex1)
        tcp.exp2(out, in0)
        tcp.add(out, out, 1)
        tcp.log(out, out)
        tcp.multiply(out, out, 1 / self.log2)
        tcp.add(out, out, ex1)
        # if ex0-ex1 > 15, out = ex0
        # ex1: 15
        # in1: mask, if greater than 15, set 1
        tcp.less_equal(in1, in0, 15.0)
        tcp.multiply(out, out, in1)
        tcp.greater(in1, in0, 15.0)
        tcp.multiply(ex0, ex0, in1)
        tcp.add(out, out, ex0)

    def main(
        self, input0: ty.handle, input1: ty.handle, output: ty.handle, length: ty.int32
    ) -> None:
        """operator main body"""
        # declare I/O buffer(gdram)
        buffer_in0 = tcp.match_buffer(input0, [length], dtype=self.dtype)
        buffer_in1 = tcp.match_buffer(input1, [length], dtype=self.dtype)
        buffer_out = tcp.match_buffer(output, [length], dtype=self.dtype)

        tgt = tcp.target()
        task_num = tgt.cluster_num * tgt.core_num
        for cluster_id in tcp.thread_binding(0, tgt.cluster_num, thread="blockIdx.x"):
            for core_id in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):
                # calculate split strategy
                task_id = cluster_id * tgt.core_num + core_id
                # 3*2+2=8 buffers: 3 pipeline double buffer(input1, input2, output)
                # and 2 extra buffer.
                # buffer size need to be 128 aligned
                single_buffer_size = (tgt.nram_size - 128 * 1024) // 8
                single_buffer_size = single_buffer_size // 128 * 128
                # gets the data length for each calculation
                data_each_time = single_buffer_size // self.dtype_size
                # gets the data length to be calculated for each task
                # the last task need to handle the extra remainder
                data_each_task = length // task_num
                if task_id == task_num - 1:
                    data_each_task = length // task_num + length % task_num
                loop_num = data_each_task // data_each_time

                buffer_extra0_n = tcp.alloc_buffer(
                    [data_each_time], dtype=self.dtype, scope="nram"
                )
                buffer_extra1_n = tcp.alloc_buffer(
                    [data_each_time], dtype=self.dtype, scope="nram"
                )
                # compute
                start = task_id * (length // task_num)
                stop = start
                for i in range(loop_num, pipeline=True):
                    buffer_in0_n = tcp.alloc_buffer(
                        [data_each_time], dtype=self.dtype, scope="nram"
                    )
                    buffer_in1_n = tcp.alloc_buffer(
                        [data_each_time], dtype=self.dtype, scope="nram"
                    )
                    buffer_out_n = tcp.alloc_buffer(
                        [data_each_time], dtype=self.dtype, scope="nram"
                    )
                    with tcp.block("data_copy"):
                        tcp.memcpy(
                            buffer_in0_n[:data_each_time],
                            buffer_in0[
                                start
                                + data_each_time * i : start
                                + data_each_time * (i + 1)
                            ],
                        )
                        tcp.memcpy(
                            buffer_in1_n[:data_each_time],
                            buffer_in1[
                                start
                                + data_each_time * i : start
                                + data_each_time * (i + 1)
                            ],
                        )
                    with tcp.block("compute"):
                        self.compute_body(
                            buffer_out_n,
                            buffer_in0_n,
                            buffer_in1_n,
                            buffer_extra0_n,
                            buffer_extra1_n,
                        )
                    with tcp.block("data_copy"):
                        tcp.memcpy(
                            buffer_out[
                                start
                                + data_each_time * i : start
                                + data_each_time * (i + 1)
                            ],
                            buffer_out_n[:data_each_time],
                        )

                # compute remainder
                buffer_in0_n = tcp.alloc_buffer(
                    [data_each_time], dtype=self.dtype, scope="nram"
                )
                buffer_in1_n = tcp.alloc_buffer(
                    [data_each_time], dtype=self.dtype, scope="nram"
                )
                buffer_out_n = tcp.alloc_buffer(
                    [data_each_time], dtype=self.dtype, scope="nram"
                )
                if data_each_task % data_each_time != 0:
                    start = task_id * (length // task_num) + data_each_time * loop_num
                    stop = start + data_each_task % data_each_time
                    tcp.memcpy(buffer_in0_n[: stop - start], buffer_in0[start:stop])
                    tcp.memcpy(buffer_in1_n[: stop - start], buffer_in1[start:stop])
                    self.compute_body(
                        buffer_out_n,
                        buffer_in0_n,
                        buffer_in1_n,
                        buffer_extra0_n,
                        buffer_extra1_n,
                    )
                    tcp.memcpy(buffer_out[start:stop], buffer_out_n[: stop - start])


@bangpy.tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_logaddexp2(dtype=None, target=None):
    """build a executable module"""
    func = build_module.build(
        Logaddexp2(dtype.name, target), target_tag=target, name=KERNEL_NAME
    )
    return func
