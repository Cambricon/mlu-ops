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
"""LogAddExp operator implementation using BANGPy TCP API."""
import bangpy
from bangpy.script import tcp, ty, build_module

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]
KERNEL_NAME = "LogAddExp"


class LogAddExp:
    def __init__(self, cluster_num: ty.int32, dtype_bits: ty.int32, dtype: ty.string) -> None:
        self.dtype = dtype
        self.dtype_sz = dtype_bits
        self.cluster_num = cluster_num

    def replace_the_marked_value(
        self,
        changed_buffer: ty.Buffer("nram"),
        value_buffer: ty.Buffer("nram"),
        marked_buffer: ty.Buffer("nram")
        ):
        tcp.multiply(changed_buffer, changed_buffer, marked_buffer)
        tcp.logic_not(marked_buffer, marked_buffer)
        tcp.multiply(marked_buffer, value_buffer, marked_buffer)
        tcp.add(changed_buffer, changed_buffer, marked_buffer)

    def mark_value_compare_with_threshold_value(
            self, input_buffer: ty.Buffer("nram"),
            bool_mark: ty.Buffer("nram"),
            is_min: ty.int32,
            threshold_value: ty.float32
            ) -> None:
        if is_min == 1:
            tcp.greater_equal(bool_mark, input_buffer, threshold_value)
        else:
            tcp.less_equal(bool_mark, input_buffer, threshold_value)

    def mark_the_out_of_range_vlaue(
        self,
        input_buffer: ty.Buffer("nram"),
        x: ty.Buffer("nram"),
        y: ty.Buffer("nram")
        ) -> None:
        max_threshold = 10
        min_threshold = -7.5
        self.mark_value_compare_with_threshold_value(input_buffer, x, 1, min_threshold)
        self.mark_value_compare_with_threshold_value(input_buffer, y, 0, max_threshold)

    def logaddexp_calc(self,
        res: ty.Buffer("nram"),
        x: ty.Buffer("nram"),
        y: ty.Buffer("nram"),
        mark_x: ty.Buffer("nram"),
        mark_y: ty.Buffer("nram")
        ) -> None:
        tcp.subtract(res, y, x)
        self.mark_the_out_of_range_vlaue(res, mark_x, mark_y)
        tcp.multiply(res, res, mark_y)
        tcp.multiply(res, res, mark_x)
        tcp.exp(res, res)
        tcp.add(res, res, 1)
        tcp.log(res, res)
        tcp.add(res, res, x)

    def main(self,
        input1: ty.handle,
        input2: ty.handle,
        output: ty.handle,
        length: ty.int32
        ) -> None:
        gram_input1 = tcp.match_buffer(input1, [length], dtype=self.dtype)
        gram_input2 = tcp.match_buffer(input2, [length], dtype=self.dtype)
        gram_output = tcp.match_buffer(output, [length], dtype=self.dtype)
        target = tcp.target()
        one_core_count = length // (self.cluster_num * target.core_num)
        remain = length % (self.cluster_num * target.core_num)
        nram_avable_size = (((target.nram_size - 40 * 1024) // 8) // 128) * 128
        process_count = nram_avable_size // self.dtype_sz
        nram_buffer_in0 = tcp.alloc_buffer(
            [process_count], dtype=self.dtype, scope="nram"
        )
        nram_buffer_in1 = tcp.alloc_buffer(
            [process_count], dtype=self.dtype, scope="nram"
        )
        nram_x_bool = tcp.alloc_buffer(
            [process_count], dtype=self.dtype, scope="nram"
        )
        nram_y_bool = tcp.alloc_buffer(
            [process_count], dtype=self.dtype, scope="nram"
        )
        nram_middle_value = tcp.alloc_buffer(
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
                if (current_task_id) < remain:
                    current_core_start = (one_core_count + 1) * (current_task_id)
                    current_core_end = (one_core_count + 1) * (current_task_id + 1) - 1
                else:
                    current_core_start = one_core_count * (
                                cluster_id * target.core_num + core_id) + remain
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
                    with tcp.block("data_copy"):
                        tcp.memcpy(
                            nram_buffer_in0[0:calc_size],
                            gram_input1[once_loop_start:once_loop_start + calc_size]
                            )
                        tcp.memcpy(nram_buffer_in1[0:calc_size],
                            gram_input2[once_loop_start:once_loop_start + calc_size]
                            )
                    with tcp.block("compute"):
                        self.logaddexp_calc(
                            nram_middle_value,
                            nram_buffer_in0,
                            nram_buffer_in1,
                            nram_x_bool,
                            nram_y_bool
                            )
                        self.replace_the_marked_value(
                            nram_middle_value,
                            nram_buffer_in1, nram_y_bool
                        )
                        self.replace_the_marked_value(
                            nram_middle_value,
                            nram_buffer_in0,
                            nram_x_bool
                        )
                    with tcp.block("data_copy"):
                        tcp.memcpy(
                            gram_output[once_loop_start:once_loop_start + calc_size],
                            nram_middle_value[:calc_size]
                        )


@bangpy.tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_logaddexp(dtype=None, target=None):
    f = build_module.build(
        LogAddExp(
            1,
            4 if dtype.name == "float32" else 2,
            dtype.name
            ),
        target_tag=target,
        name=KERNEL_NAME
    )
    return f
