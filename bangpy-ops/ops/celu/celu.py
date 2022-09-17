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
"""A multi-platform code link example test for BANGPy TCP."""
import bangpy
from bangpy.script import tcp, ty, build_module

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]
KERNEL_NAME = "Celu"


class Celu:
    def __init__(
                self,
                cluster_num: ty.int32,
                dtype_bits: ty.int32,
                dtype: ty.string
                ) -> None:
        self.dtype = dtype
        self.dtype_sz = dtype_bits
        self.cluster_num = cluster_num

    def main(
            self,
            input0: ty.handle,
            output: ty.handle,
            length: ty.int32,
            alpha: ty.float32,
            ) -> None:
        gram_input0 = tcp.match_buffer(input0, [length], dtype=self.dtype)
        gram_output = tcp.match_buffer(output, [length], dtype=self.dtype)
        target = tcp.target()
        one_core_count = length // (self.cluster_num * target.core_num)
        remain = length % (self.cluster_num * target.core_num)
        nram_avable_size = (((target.nram_size - 40 * 1024) // 4) // 128) * 128
        process_count = nram_avable_size // self.dtype_sz
        nram_buffer_in0 = tcp.alloc_buffer(
            [process_count], dtype=self.dtype, scope="nram"
        )
        nram_middle_value = tcp.alloc_buffer(
            [process_count], dtype=self.dtype, scope="nram"
        )
        nram_max = tcp.alloc_buffer(
            [process_count], dtype=self.dtype, scope="nram"
        )
        nram_min = tcp.alloc_buffer(
            [process_count], dtype=self.dtype, scope="nram"
        )
        current_core_start = 0
        current_core_end = 0
        total_count_in_core = 0
        once_loop_start = 0
        calc_size = 0

        for cluster_id in tcp.thread_binding(0, self.cluster_num, thread="blockIdx.x"):
            for core_id in tcp.thread_binding(0, target.core_num, thread="threadIdx.x"):
                current_task_id = cluster_id * target.core_num + core_id
                if current_task_id < remain:
                    current_core_start = (one_core_count + 1) * current_task_id
                    current_core_end = (one_core_count + 1) * (current_task_id + 1) - 1
                else:
                    current_core_start = one_core_count * current_task_id + remain
                    current_core_end = current_core_start + one_core_count - 1
                total_count_in_core = current_core_end - current_core_start + 1
                calc_loop_count = (total_count_in_core + process_count - 1) // process_count
                for i in range(0, calc_loop_count):
                    once_loop_start = current_core_start + process_count * i
                    if i < calc_loop_count - 1:
                        calc_size = process_count
                    else:
                        calc_size = total_count_in_core % process_count
                        if calc_size == 0:
                            calc_size = process_count
                    tcp.memcpy(
                        nram_buffer_in0[0:calc_size],
                        gram_input0[once_loop_start:once_loop_start + calc_size]
                    )
                    if alpha != 0.0:
                        tcp.divide(nram_middle_value, nram_buffer_in0, alpha)
                        tcp.exp(nram_middle_value, nram_middle_value)
                        tcp.subtract(nram_middle_value, nram_middle_value, 1.0)
                        tcp.multiply(nram_middle_value, nram_middle_value, alpha)
                        tcp.minimum(nram_min, nram_middle_value, 0.0)
                    else:
                        tcp.assign(nram_min, 0.0)
                    tcp.maximum(nram_max, nram_buffer_in0, 0.0)
                    tcp.add(nram_buffer_in0, nram_max, nram_min)
                    tcp.memcpy(
                        gram_output[once_loop_start:once_loop_start + calc_size],
                        nram_buffer_in0[:calc_size])


@bangpy.tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_celu(dtype=None, target=None):
    f = build_module.build(
        Celu(
            1,
            4 if dtype.name == "float32" else 2,
            dtype.name
            ),
        target_tag=target,
        name=KERNEL_NAME
    )
    return f
