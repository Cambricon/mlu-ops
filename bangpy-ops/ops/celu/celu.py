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
from bangpy import tcp
from bangpy.platform.bang_config import TARGET
from bangpy.tcp.util import round_down
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu290"]
KERNEL_NAME = "Celu"


class Celu:
    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.bp = tcp.TCP(target)
        self.inplace = self.bp.Var("inplace")
        self.length = self.bp.SizeVar("length")
        self.dtype_sz = dtype.bytes
        self.bp.launch_task(self.task_num, 1, 1)

    def compute_body(self):
        one_core_count = self.bp.Scalar(bangpy.int32, "one_core_count",
                                        self.length // self.task_num)
        remain = self.bp.Scalar(bangpy.int32, "remain", self.length % self.task_num)
        current_core_start = self.bp.Scalar(bangpy.int32, "current_core_start")
        current_core_end = self.bp.Scalar(bangpy.int32, "current_core_end")
        calc_loop_count = self.bp.Scalar(bangpy.int32, "calc_loop_count")
        once_loop_start = self.bp.Scalar(bangpy.int32, "once_loop_start")
        calc_size = self.bp.Scalar(bangpy.int32, "calc_size")
        nram_avable_size = round_down((TARGET(self.target).nram_size - 40 * 1024) // 4, 128)
        process_count = nram_avable_size // self.dtype_sz
        with self.bp.if_scope(self.bp.taskId < remain):
            current_core_start.assign((one_core_count + 1) * self.bp.taskId)
            current_core_end.assign((one_core_count + 1) * (self.bp.taskId + 1) - 1)
        with self.bp.else_scope():
            current_core_start.assign(one_core_count * self.bp.taskId + remain)
            current_core_end.assign(current_core_start + one_core_count - 1)
        total_count_in_core = self.bp.Scalar(
            bangpy.int32,
            "total_count_in_core",
            current_core_end - current_core_start + 1)
        buffer_in0 = self.bp.Buffer(
            shape = (self.length,), name = "INPUT0", dtype = self.dtype, scope = "global"
        )
        buffer_alpha = self.bp.Buffer(
            shape = (1,), name = "ALPHA_PARAM", dtype = self.dtype, scope = "global"
        )
        buffer_out = self.bp.Buffer(
            shape = (self.length,), name = "OUTPUT", dtype = self.dtype, scope = "global"
        )
        alpha = self.bp.Scalar(dtype = self.dtype, name = "alpha")
        alpha.assign(buffer_alpha[0])
        nram_buffer_in0 = self.bp.Buffer(
            shape = (process_count,),
            name = "INPUT0_N",
            dtype = self.dtype,
            scope = "nram",
        )
        nram_middle_value = self.bp.Buffer(
            shape = (process_count,),
            name = "N_MAX",
            dtype = self.dtype,
            scope = "nram",
        )
        nram_max = self.bp.Buffer(
            shape = (process_count,),
            name = "N_MAX",
            dtype = self.dtype,
            scope = "nram",
        )
        nram_min = self.bp.Buffer(
            shape = (process_count,),
            name = "N_MIN",
            dtype = self.dtype,
            scope = "nram",
        )
        const_zero = self.bp.Scalar(dtype = self.dtype, name = "const_zero", value = 0)
        const_one = self.bp.Scalar(dtype = self.dtype, name = "const_one", value = 1)
        calc_loop_count.assign((total_count_in_core + process_count - 1) // process_count)
        with self.bp.for_range(0, calc_loop_count) as i:
            once_loop_start.assign(current_core_start + process_count * i)
            with self.bp.if_scope(i < calc_loop_count - 1):
                calc_size.assign(process_count)
            with self.bp.else_scope():
                calc_size.assign(total_count_in_core % process_count)
                with self.bp.if_scope(calc_size == 0):
                    calc_size.assign(process_count)
            with self.bp.block("data_copy"):
                self.bp.memcpy(
                    nram_buffer_in0[0:calc_size],
                    buffer_in0[once_loop_start:once_loop_start + calc_size]
                )
            with self.bp.if_scope(alpha != 0):
                self.bp.divide(nram_middle_value, nram_buffer_in0, alpha)
                self.bp.exp(nram_middle_value, nram_middle_value)
                self.bp.subtract(nram_middle_value, nram_middle_value, const_one)
                self.bp.multiply(nram_middle_value, nram_middle_value, alpha)
                self.bp.minimum(nram_min, nram_middle_value, const_zero)
            with self.bp.else_scope():
                self.bp.zeros(nram_min)
            self.bp.maximum(nram_max, nram_buffer_in0, const_zero)
            self.bp.add(nram_buffer_in0, nram_max, nram_min)
            self.bp.memcpy(
                buffer_out[once_loop_start:once_loop_start + calc_size],
                nram_buffer_in0[:calc_size])
        f = self.bp.BuildBANG(
            inputs = [buffer_in0, buffer_alpha, self.inplace],
            outputs = [buffer_out],
            kernel_name = KERNEL_NAME, )
        return f


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_celu(dtype=None, target=None):
    task_type = TaskType.UNION16
    task_num = task_type.value * 4
    f = Celu(dtype, target, task_num).compute_body()
    return f
