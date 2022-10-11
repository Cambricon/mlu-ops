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
"""Add operator implementation using BANGPy TCP API."""
import bangpy
from bangpy import tcp
from bangpy.script import ty, build_module


DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]
KERNEL_NAME = "add"


class Add(object):
    """Operator description:
    Add the data in the two buffers.
    """

    def __init__(self, buffer_size: ty.int32, dtype: ty.string) -> None:
        self.dtype = dtype
        self.single_buffer_size = buffer_size

    def add_body(
        self,
        local_a: ty.Buffer("nram"),  # type: ignore
        local_b: ty.Buffer("nram"),  # type: ignore
        local_c: ty.Buffer("nram"),  # type: ignore
    ) -> None:
        # The body of add function
        tcp.add(local_a, local_b, local_c)

    def main(self, a: ty.handle, b: ty.handle, c: ty.handle, length: ty.int32) -> None:
        A = tcp.match_buffer(a, [length], dtype=self.dtype)
        B = tcp.match_buffer(b, [length], dtype=self.dtype)
        C = tcp.match_buffer(c, [length], dtype=self.dtype)
        tgt = tcp.target()
        # calculate split strategy
        # gets the data length to be calculated for each task
        data_calculated_each_task = length // (tgt.cluster_num * tgt.core_num)
        # gets the number of cycles required for each task
        loop_num = data_calculated_each_task // self.single_buffer_size

        buffer_in0 = tcp.alloc_buffer(
            [self.single_buffer_size], dtype=self.dtype, scope="nram"
        )
        buffer_in1 = tcp.alloc_buffer(
            [self.single_buffer_size], dtype=self.dtype, scope="nram"
        )
        buffer_out = tcp.alloc_buffer(
            [self.single_buffer_size], dtype=self.dtype, scope="nram"
        )
        for cluster_id in tcp.thread_binding(0, tgt.cluster_num, thread="blockIdx.x"):
            for core_id in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):
                for i in range(loop_num):
                    task_id = cluster_id * tgt.core_num + core_id
                    start = (
                        task_id * data_calculated_each_task
                        + i * self.single_buffer_size
                    )
                    stop = start + self.single_buffer_size
                    tcp.memcpy(buffer_in0, A[start:stop])
                    tcp.memcpy(buffer_in1, B[start:stop])
                    self.add_body(buffer_out, buffer_in0, buffer_in1)
                    tcp.memcpy(C[start:stop], buffer_out)


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_add(dtype=None, target=None):
    f = build_module.build(Add(64, dtype.name), target_tag=target, name=KERNEL_NAME)
    return f
