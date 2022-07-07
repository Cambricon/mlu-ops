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
from bangpy.tcp.util import round_down
from bangpy import tcp
from bangpy.platform.bang_config import TARGET

DTYPES = [bangpy.float32]
TARGET_LIST = ["mlu290"]
KERNEL_NAME = "PairwiseDistance"


class DataMan:
    def init(self, bp):
        self.bp = bp
        self.m_current_core_start = self.bp.Scalar(bangpy.int32, "current_core_start")
        self.m_current_core_end = self.bp.Scalar(bangpy.int32, "current_core_end")
        self.m_total_count_in_core = self.bp.Scalar(bangpy.int32, "total_count_in_core")

    def calc_core_process_count(self, data_total_len, task_num):
        one_core_count = self.bp.Scalar(bangpy.int32, "one_core_count")
        one_core_count.assign(data_total_len // task_num)

        remain = self.bp.Scalar(bangpy.int32, "remain")
        remain.assign(data_total_len % task_num)

        # If remains exists, averagely assigning remains to cores.
        # Small taskId cores will has high priority to be assigned.
        with self.bp.if_scope(self.bp.taskId < remain):
            self.m_current_core_start.assign((one_core_count + 1) * self.bp.taskId )
            self.m_current_core_end.assign((one_core_count + 1) * \
                (self.bp.taskId + 1) - 1)
        with self.bp.else_scope():
            self.m_current_core_start.assign((one_core_count + 1) * \
                remain + one_core_count * (self.bp.taskId - remain))
            self.m_current_core_end.assign((one_core_count + 1) * remain + \
                one_core_count * (self.bp.taskId - remain) + one_core_count - 1)

        self.m_total_count_in_core.assign(self.m_current_core_end - self.m_current_core_start + 1)


class PairwiseDataPara:
    len_tensor1 = None
    len_tensor2 = None

    pd_len = None
    pd_height = None
    pd_width = None

    output_len = None


class PairwiseDistance:
    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.dtype_size  = dtype.bytes
        self.bp = tcp.TCP(target)

        self.PdPara = PairwiseDataPara()
        self.dman = DataMan()

    def sub_tensor(self, t1, t2, len_t2):
        nram_available_size = round_down((TARGET(self.target).nram_size - 30 * 1024) // 2, 128)
        nram_process_count = nram_available_size // self.dtype_size

        nram_buffer_in = self.bp.Buffer(
            shape=(2, nram_process_count),
            name="nram_buffer_in",
            dtype=self.dtype,
            scope="nram")

        nram_buffer_in0 = nram_buffer_in[0][:]
        nram_buffer_in1 = nram_buffer_in[1][:]

        total_count_in_core = self.dman.m_total_count_in_core
        once_loop_start = self.bp.Scalar(bangpy.int32, "once_loop_start")

        calc_size = self.bp.Scalar(bangpy.int32, "calc_size")
        calc_loop_count = self.bp.Scalar(bangpy.int32, "calc_loop_count", \
            (total_count_in_core + nram_process_count - 1) // nram_process_count)

        with self.bp.for_range(0, calc_loop_count) as i:
            with self.bp.if_scope(i < calc_loop_count - 1):
                calc_size.assign(nram_process_count)
            with self.bp.else_scope():
                calc_size.assign(total_count_in_core % nram_process_count)
                with self.bp.if_scope(calc_size == 0):
                    calc_size.assign(nram_process_count)

            once_loop_start.assign(self.dman.m_current_core_start + \
                nram_process_count * i)

            with self.bp.block("data_copy"):
                # tensor1 copy
                self.bp.memcpy(nram_buffer_in0[0:calc_size], \
                    t1[once_loop_start:once_loop_start + calc_size])

                # tensor2 copy
                head_offset = self.bp.Scalar(bangpy.int32, "head_offset", once_loop_start % len_t2)

                with self.bp.if_scope(head_offset == 0):
                    head_len = self.bp.Scalar(bangpy.int32, "head_len", 0)
                with self.bp.else_scope():
                    head_len = self.bp.Scalar(bangpy.int32, "head_len", len_t2 - head_offset)

                with self.bp.if_scope(head_len >= calc_size):
                    self.bp.memcpy(nram_buffer_in1[0:calc_size], \
                        t2[head_offset:head_offset + calc_size])
                with self.bp.else_scope():
                    with self.bp.if_scope(head_len > 0):
                        self.bp.memcpy(nram_buffer_in1[0:head_len], t2[head_offset:len_t2])

                    total_offset = self.bp.Scalar(bangpy.int32, "total_offset")
                    total_offset.assign(head_len)

                    body_cp_count = self.bp.Scalar(bangpy.int32, "body_cp_count", \
                        (calc_size - head_len) // len_t2)

                    with self.bp.for_range(0, body_cp_count):
                        self.bp.memcpy(nram_buffer_in1[total_offset:total_offset \
                            + len_t2], t2[0:len_t2])
                        total_offset.assign(total_offset + len_t2)

                    offset_end = self.bp.Scalar(bangpy.int32, "offset_end", \
                        (once_loop_start + calc_size) % len_t2)

                    with self.bp.if_scope(offset_end > 0):
                        self.bp.memcpy(nram_buffer_in1[total_offset:total_offset + \
                            offset_end], t2[0:offset_end])

            with self.bp.block("compute"):
                self.bp.subtract(nram_buffer_in0, nram_buffer_in0, nram_buffer_in1)
                self.bp.abs(nram_buffer_in0, nram_buffer_in0)

                # subtract eps
                eps = self.bp.Scalar(name='input_eps', dtype=self.dtype, \
                    value=-self.nram_pd_paras[1])
                self.bp.add(nram_buffer_in0, nram_buffer_in0, eps)

                self.bp.log(nram_buffer_in0, nram_buffer_in0)
                p = self.bp.Scalar(name='p', dtype=self.dtype, value=self.nram_pd_paras[0])
                self.bp.multiply(nram_buffer_in0, nram_buffer_in0, p)
                self.bp.exp(nram_buffer_in0, nram_buffer_in0)

            with self.bp.block("data_copy"):
                self.bp.memcpy(t1[once_loop_start:once_loop_start + calc_size], \
                    nram_buffer_in0[:calc_size])


    def copy_from_2d_tensor(self, dst, offset_dst, src, offset_src, dim_len, width, cp_len):
        big_row = offset_src // (width * dim_len)
        m = offset_src % dim_len + big_row * dim_len

        big_n = offset_src % dim_len
        n = big_n % width

        with self.bp.if_scope(offset_dst != offset_dst + cp_len // 2):
            self.bp.memcpy(dst[offset_dst:offset_dst + cp_len // 2, 0:1], \
                src[m:m + cp_len  // 2, n:n + 1])

        with self.bp.if_scope(offset_dst + cp_len // 2 != offset_dst + cp_len):
            self.bp.memcpy(dst[offset_dst + cp_len // 2:offset_dst + cp_len, 0:1], \
                src[m + cp_len // 2:m + cp_len, n:n + 1])

    def calc_norm(self, buffer, start, end):
        result = self.bp.Scalar(self.dtype, "result", 0.0)
        size = self.bp.Scalar(bangpy.int32, "size", end - start)
        with self.bp.for_range(0, size) as i:
            result.assign(result + buffer[start + i])
        return result

    def scalar_pow(self, value, p):
        return self.bp.scalar_pow(value, p)

    def compute_body(self):
        self.dman.init(self.bp)
        self.bp.launch_task(self.task_num, 1, 1)

        self.PdPara.len_tensor1 = self.bp.SizeVar("len_tensor1")
        self.PdPara.len_tensor2 = self.bp.SizeVar("len_tensor2")

        self.PdPara.pd_len = self.bp.SizeVar("pd_len")
        self.PdPara.pd_height = self.bp.SizeVar("pd_height")
        self.PdPara.pd_width = self.bp.SizeVar("pd_width")

        self.PdPara.output_len = self.bp.SizeVar("output_len")

        gram_tensor1 = self.bp.Buffer(
            shape=(self.PdPara.len_tensor1, ), name="gram_tensor1", dtype=self.dtype, scope="global"
        )

        gram_tensor2 = self.bp.Buffer(
            shape=(self.PdPara.len_tensor2, ), name="gram_tensor2", dtype=self.dtype, scope="global"
        )

        gram_paras = self.bp.Buffer(
            shape=(2, ), name="gram_paras", dtype=self.dtype, scope="global"
        )

        gram_buffer_out = self.bp.Buffer(
            shape=(self.PdPara.output_len, ), \
            name="gram_buffer_out", dtype=self.dtype, scope="global"
        )

        border_array_size = 128
        gram_border_buf_out = self.bp.Buffer(
            shape=(border_array_size * 2, ), name="gram_border_buf_out", \
            dtype=self.dtype, scope="global"
        )
        gram_border_idx_out = self.bp.Buffer(
            shape=(border_array_size * 2, ), name="gram_border_idx_out", \
            dtype=bangpy.int32, scope="global"
        )

        self.nram_pd_paras = self.bp.Buffer(
            shape=(2, ),
            name="nram_pd_paras",
            dtype=self.dtype,
            scope="nram")
        self.bp.memcpy(self.nram_pd_paras[0:2], gram_paras[0:2])

        self.nram_pow_buffer = self.bp.Buffer(
            shape=(128, ),
            name="nram_pow_buffer",
            dtype=self.dtype,
            scope="nram")

        self.dman.calc_core_process_count(self.PdPara.len_tensor1, self.task_num)

        self.sub_tensor(gram_tensor1, gram_tensor2, self.PdPara.len_tensor2)
        self.bp.sync_all()

        with self.bp.if_scope(self.bp.taskId == 0):
            gram_reshape_tensor = gram_tensor1.\
            reshape([self.PdPara.pd_height, self.PdPara.pd_width])
        self.bp.sync_all()

        nram_available_size = round_down(TARGET(self.target).nram_size - 30 * 1024, 128)
        self.nram_process_count = nram_available_size // self.dtype_size
        self.nram_calc_buffer = self.bp.Buffer(
            shape=(self.nram_process_count, 1),
            name="nram_calc_buffer",
            dtype=self.dtype,
            scope="nram")

        with self.bp.if_scope(self.PdPara.pd_len > self.nram_process_count):
            self.calc_pairwise_distance1(gram_reshape_tensor, gram_border_buf_out, \
                gram_border_idx_out, gram_buffer_out)
        with self.bp.else_scope():
            with self.bp.if_scope(self.PdPara.len_tensor1 // \
            self.task_num + 1 < self.PdPara.pd_len):
                self.calc_pairwise_distance1(gram_reshape_tensor, \
                    gram_border_buf_out, gram_border_idx_out, gram_buffer_out)
            with self.bp.else_scope():
                self.calc_pairwise_distance2(gram_reshape_tensor, gram_border_buf_out, \
                    gram_border_idx_out, gram_buffer_out)

        self.bp.sync_all()

        with self.bp.if_scope(self.bp.taskId == 0):
            with self.bp.for_range(0, border_array_size) as i:
                index1 = gram_border_idx_out[2 * i]
                index2 = gram_border_idx_out[2 * i + 1]
                norm_value1 = gram_border_buf_out[2 * i]
                norm_value2 = gram_border_buf_out[2 * i + 1]

                with self.bp.if_scope(index1 >= 0):
                    gram_buffer_out[index1] = gram_buffer_out[index1] + norm_value1

                with self.bp.if_scope(index2 >= 0):
                    gram_buffer_out[index2] = gram_buffer_out[index2] + norm_value2

        f = self.bp.BuildBANG(
            inputs=[gram_tensor1, gram_tensor2, gram_paras,
                    self.PdPara.len_tensor1, self.PdPara.len_tensor2,
                    self.PdPara.pd_len, self.PdPara.pd_height, self.PdPara.pd_width,
                    self.PdPara.output_len
                    ],
            outputs=[gram_border_buf_out, gram_border_idx_out, gram_buffer_out],
            kernel_name=KERNEL_NAME
            )
        return f

    def get_norm_index(self, data_pos, dim_len):
        index = (data_pos + dim_len - 1) // dim_len
        return index - 1

    def calc_pairwise_distance1(self, gram_tensor, border_outputs, \
        idx_outputs, outputs):
        current_core_start = self.dman.m_current_core_start
        total_count_in_core = self.dman.m_total_count_in_core
        calc_loop_count = self.bp.Scalar(bangpy.int32, "calc_loop_count", \
            (total_count_in_core + self.nram_process_count - 1) // \
            self.nram_process_count)
        norm_value = self.bp.Scalar(self.dtype, "norm_value", 0.0)

        once_loop_start = self.bp.Scalar(bangpy.int32, "once_loop_start")

        oper_type = self.bp.Scalar(bangpy.int32, "oper_type", 0)

        pw = self.bp.Scalar(self.dtype, "pw", 1 / self.nram_pd_paras[0])

        dim_len = self.PdPara.pd_len
        norm_offset = self.bp.Scalar(bangpy.int32, "norm_offset", \
            current_core_start % dim_len)
        with self.bp.if_scope(norm_offset == 0):
            oper_type.assign(2)
        with self.bp.else_scope():
            oper_type.assign(0)

        flat_nram = self.nram_calc_buffer.reshape([self.nram_process_count, ])

        norm_value = self.bp.Scalar(self.dtype, "norm_value", 0.0)

        calc_size = self.bp.Scalar(bangpy.int32, "calc_size", \
            self.nram_process_count)

        once_norm_ok = self.bp.Scalar(bangpy.int32, "once_norm_ok", 0)
        cp_data_len = self.bp.Scalar(bangpy.int32, "cp_data_len", 0)
        with self.bp.for_range(0, calc_loop_count) as i:
            once_loop_start.assign(current_core_start + self.nram_process_count * i)
            with self.bp.if_scope(i == calc_loop_count - 1):
                calc_size.assign(total_count_in_core % self.nram_process_count)
                with self.bp.if_scope(calc_size == 0):
                    calc_size.assign(self.nram_process_count)

            norm_offset.assign(once_loop_start % dim_len)
            expect_cp_len = self.bp.Scalar(bangpy.int32, "expect_cp_len", \
                dim_len - norm_offset)

            with self.bp.if_scope(expect_cp_len > calc_size):
                expect_cp_len.assign(calc_size)
                self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, once_loop_start, \
                    dim_len, self.PdPara.pd_width, expect_cp_len)
                cp_data_len.assign(cp_data_len + expect_cp_len)
                seg_norm_value = self.calc_norm(flat_nram, 0, expect_cp_len)
                norm_value.assign(norm_value + seg_norm_value)
                with self.bp.if_scope(i == calc_loop_count - 1):  # last loop
                    index = self.get_norm_index(once_loop_start + expect_cp_len, dim_len)
                    with self.bp.if_scope(once_norm_ok == 0):
                        border_outputs[self.bp.taskId * 2] = norm_value
                        idx_outputs[self.bp.taskId * 2] = index
                    with self.bp.else_scope():
                        border_outputs[self.bp.taskId * 2 + 1] = norm_value
                        idx_outputs[self.bp.taskId * 2 + 1] = index

            with self.bp.else_scope():
                self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, once_loop_start, \
                    dim_len, self.PdPara.pd_width, expect_cp_len)
                cp_data_len.assign(cp_data_len + expect_cp_len)
                seg_norm_value = self.calc_norm(flat_nram, 0, expect_cp_len)
                norm_value.assign(norm_value + seg_norm_value)

                once_norm_ok.assign(1)
                index = self.get_norm_index(once_loop_start + expect_cp_len, dim_len)
                with self.bp.if_scope(cp_data_len < dim_len):
                    border_outputs[self.bp.taskId * 2] \
                        = norm_value
                    idx_outputs[self.bp.taskId * 2] = index
                with self.bp.else_scope():
                    outputs[index] = self.scalar_pow(norm_value, pw)  # norm complete

                norm_value.assign(0.0)

                cp_data_len.assign(calc_size - expect_cp_len)
                with self.bp.if_scope(cp_data_len > 0):
                    self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, \
                        once_loop_start + expect_cp_len, dim_len, \
                        self.PdPara.pd_width, cp_data_len)
                    calc_result = self.calc_norm(flat_nram, 0, cp_data_len)
                    norm_value.assign(calc_result)
                    with self.bp.if_scope(i == calc_loop_count - 1):
                        border_outputs[self.bp.taskId * 2 + 1] = norm_value
                        idx_outputs[self.bp.taskId * 2 + 1] = index + 1

    def calc_pairwise_distance2(self, gram_tensor, border_outputs, idx_outputs, outputs):
        current_core_start = self.dman.m_current_core_start
        total_count_in_core = self.dman.m_total_count_in_core
        dim_len = self.PdPara.pd_len
        norm_value = self.bp.Scalar(self.dtype, "norm_value", 0.0)
        pw = self.bp.Scalar(self.dtype, "pw", 1 / self.nram_pd_paras[0])

        flat_nram = self.nram_calc_buffer.reshape([self.nram_process_count, ])

        norm_offset = self.bp.Scalar(bangpy.int32, "norm_offset", current_core_start % dim_len)
        expect_cp_len = self.bp.Scalar(bangpy.int32, "expect_cp_len", 0)
        with self.bp.if_scope(norm_offset != 0):
            expect_cp_len.assign(dim_len - norm_offset)
            self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, current_core_start, \
                dim_len, self.PdPara.pd_width, expect_cp_len)
            calc_result = self.calc_norm(flat_nram, 0, expect_cp_len)
            norm_value.assign(calc_result)
            index = self.get_norm_index(current_core_start + expect_cp_len, dim_len)
            border_outputs[self.bp.taskId * 2] = norm_value
            idx_outputs[self.bp.taskId * 2] = index

        norm_start_pos = self.bp.Scalar(bangpy.int32, "norm_start_pos", \
            current_core_start + expect_cp_len)

        nram_norm_count = self.bp.Scalar(bangpy.int32, "nram_norm_count", \
            self.nram_process_count // dim_len)

        total_norm_in_core = self.bp.Scalar(bangpy.int32, "total_norm_in_core", \
            (total_count_in_core - expect_cp_len) // dim_len)

        calc_loop_count = self.bp.Scalar(bangpy.int32, "calc_loop_count", \
            (total_norm_in_core + \
            nram_norm_count - 1) // nram_norm_count)

        once_loop_norm_count = self.bp.Scalar(bangpy.int32, "nram_norm_count", nram_norm_count)
        with self.bp.for_range(0, calc_loop_count) as i:
            once_loop_start = self.bp.Scalar(bangpy.int32, "once_loop_start", norm_start_pos + \
                nram_norm_count * dim_len * i)
            with self.bp.if_scope(i == calc_loop_count - 1):
                once_loop_norm_count.assign(total_norm_in_core % nram_norm_count)
                with self.bp.if_scope(once_loop_norm_count == 0):
                    once_loop_norm_count.assign(nram_norm_count)

            start_index = self.bp.Scalar(bangpy.int32, "norm_offset", \
            once_loop_start // dim_len)
            with self.bp.for_range(0, once_loop_norm_count) as j:
                self.copy_from_2d_tensor(self.nram_calc_buffer, \
                    0, gram_tensor, once_loop_start + j * \
                    dim_len, dim_len, self.PdPara.pd_width, dim_len)
                calc_result = self.calc_norm(flat_nram, 0, dim_len)
                norm_value.assign(calc_result)
                outputs[start_index + j] = self.scalar_pow(norm_value, pw)

        norm_loop_end_pos = self.bp.Scalar(bangpy.int32, "norm_loop_end_pos", \
            norm_start_pos + total_norm_in_core * dim_len)
        cur_loop_end_pos = self.bp.Scalar(bangpy.int32, "cur_loop_end_pos", \
            current_core_start + total_count_in_core)
        with self.bp.if_scope(norm_loop_end_pos < cur_loop_end_pos):
            self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, norm_loop_end_pos, \
                dim_len, \
                self.PdPara.pd_width, cur_loop_end_pos - norm_loop_end_pos)
            calc_result = self.calc_norm(flat_nram, 0, cur_loop_end_pos - norm_loop_end_pos)
            norm_value.assign(calc_result)
            index = self.get_norm_index(norm_loop_end_pos + 1, dim_len)
            border_outputs[self.bp.taskId * 2 + 1] = norm_value
            idx_outputs[self.bp.taskId * 2 + 1] = index

@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_pairwisedistance(dtype=None, target=None):
    task_num = 32
    f = PairwiseDistance(dtype, target, task_num).compute_body()
    return f
