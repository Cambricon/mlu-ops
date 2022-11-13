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
from bangpy.script import ty, build_module
import bangpy.eager as eg

DTYPES = [#bangpy.float16,
           bangpy.float32]
TARGET_LIST = ["mlu370-s4"]
KERNEL_NAME = "pairwise_distance"


@eg.module
class PairwiseDistance(object):
    def calc_core_process_count(self, data_total_len: ty.int32, task_num: ty.int32, task_id: ty.int32):
        one_core_count = data_total_len // task_num
        remain = data_total_len % task_num
        m_current_core_start = 0
        m_current_core_end = 0
        m_total_count_in_core = 0
        if task_id < remain:
            m_current_core_start = (one_core_count + 1) * task_id
            m_current_core_end = (one_core_count + 1) * (task_id + 1) - 1
            m_total_count_in_core = m_current_core_end - m_current_core_start + 1
        else:
            m_current_core_start = (one_core_count + 1) * \
                remain + one_core_count * (task_id - remain)
            m_current_core_end = (one_core_count + 1) * remain + \
                one_core_count * (task_id - remain) + one_core_count - 1
            m_total_count_in_core = m_current_core_end - m_current_core_start + 1

        self.m_total_count_in_core = m_total_count_in_core
        self.m_current_core_start = m_current_core_start
        self.m_current_core_end = m_current_core_end

    def calc_avaible_nram_count(self):
        return tcp.round_down((self.bp.nram_size - 30 * 1024) // 2, 128)

    def sub_tensor(self, t1: ty.handle, t2: ty.handle, len_t2: ty.int32):
#        nram_available_size = tcp.round_down((self.bp.nram_size - 30 * 1024) // 2, 128)
        nram_available_size = self.calc_avaible_nram_count()

        nram_process_count = nram_available_size // self.dtype_size

        nram_buffer_in = tcp.alloc_buffer(
            shape=(2, nram_process_count),
            dtype=self.dtype,
            scope="nram")

        nram_buffer_in0 = nram_buffer_in[0]
        nram_buffer_in1 = nram_buffer_in[1]

        total_count_in_core = self.m_total_count_in_core
        calc_loop_count = (total_count_in_core + nram_process_count - 1) // nram_process_count

        for i in range(calc_loop_count):
            calc_size = 0
            if i < calc_loop_count - 1:
                calc_size = nram_process_count
            else:
                calc_size = total_count_in_core % nram_process_count
                if calc_size == 0:
                    calc_size = nram_process_count

            once_loop_start = self.m_current_core_start + nram_process_count * i

            # tensor1 copy
            tcp.memcpy(nram_buffer_in0[0:calc_size], t1[once_loop_start:once_loop_start + calc_size])

            # tensor2 copy
            head_offset = once_loop_start % len_t2

            head_len = 0
            if head_offset == 0:
                head_len = 0
            else:
                head_len = len_t2 - head_offset

            if head_len >= calc_size:
                tcp.memcpy(nram_buffer_in1[0:calc_size], t2[head_offset:head_offset + calc_size])
            else:
                if head_len > 0:
                    tcp.memcpy(nram_buffer_in1[0:len_t2 - head_offset], t2[head_offset:len_t2])

                total_offset = head_len
                body_cp_count = (calc_size - head_len) // len_t2

                for i in range(body_cp_count):
                    tcp.memcpy(nram_buffer_in1[total_offset:total_offset + len_t2], t2[0:len_t2])
                    total_offset = total_offset + len_t2

                offset_end = (once_loop_start + calc_size) % len_t2

                if offset_end > 0:
                    tcp.memcpy(nram_buffer_in1[total_offset:total_offset + offset_end], t2[0:offset_end])


            tcp.subtract(nram_buffer_in0, nram_buffer_in0, nram_buffer_in1)
            tcp.abs(nram_buffer_in0, nram_buffer_in0)

            # subtract eps
            eps = -self.gram_paras[1]
            tcp.add(nram_buffer_in0, nram_buffer_in0, eps)

            tcp.log(nram_buffer_in0, nram_buffer_in0)
            p = self.gram_paras[0]
            tcp.multiply(nram_buffer_in0, nram_buffer_in0, p)
            tcp.exp(nram_buffer_in0, nram_buffer_in0)

            tcp.memcpy(t1[once_loop_start:once_loop_start + calc_size], nram_buffer_in0[:calc_size])

    def copy_from_2d_tensor(self, dst: ty.handle, offset_dst: ty.int32, src: ty.handle, \
        offset_src: ty.int32, dim_len: ty.int32, width: ty.int32, cp_len: ty.int32):
        big_row = offset_src // (width * dim_len)
        m = offset_src % dim_len + big_row * dim_len

        big_n = offset_src % dim_len
        n = big_n % width

        if offset_dst != offset_dst + cp_len // 2:
            tcp.memcpy(dst[offset_dst:offset_dst + cp_len // 2, 0:1], \
                src[m:m + cp_len  // 2, n:n + 1])

        if offset_dst + cp_len // 2 != offset_dst + cp_len:
            tcp.memcpy(dst[offset_dst + cp_len // 2:offset_dst + cp_len, 0:1], \
                src[m + cp_len // 2:m + cp_len, n:n + 1])

    def calc_norm(self, buffer: ty.handle, start: ty.int32, end: ty.int32):
        result = 0.0
        size = end - start
        for i in range(size):
            result += buffer[start + i]
        return result

    """Operator description:
    Add the data in the two buffers.
    """
    def __init__(self, dtype: ty.string, dtype_size: ty.int32) -> None:
        self.dtype = dtype
        self.dtype_size = dtype_size

    def get_norm_index(self, data_pos: ty.int32, dim_len: ty.int32):
        index = (data_pos + dim_len - 1) // dim_len
        return index - 1

    def calc_pairwise_distance1(self, gram_tensor: ty.handle, border_outputs: ty.handle, idx_outputs: ty.handle, outputs: ty.handle):
        current_core_start = self.m_current_core_start
        total_count_in_core = self.m_total_count_in_core
        calc_loop_count = (total_count_in_core + self.nram_process_count - 1) // self.nram_process_count
        norm_value = 0.0

        once_loop_start = 0
        oper_type = 0
        pw = 1 / self.gram_paras[0]
        dim_len = self.pd_len
        norm_offset = current_core_start % dim_len

        if norm_offset == 0:
            oper_type = 2
        else:
            oper_type = 0

        flat_nram = self.nram_calc_buffer[:self.nram_process_count].reshape([self.nram_process_count, ])
        norm_value = 0.0
        calc_size = self.nram_process_count

        once_norm_ok = 0
        cp_data_len = 0
        for i in range(calc_loop_count):
            once_loop_start = current_core_start + self.nram_process_count * i
            if i == calc_loop_count - 1:
                calc_size = total_count_in_core % self.nram_process_count
                if calc_size == 0:
                    calc_size = self.nram_process_count

            norm_offset = once_loop_start % dim_len
            expect_cp_len = dim_len - norm_offset

            if expect_cp_len > calc_size:
                expect_cp_len = calc_size
                self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, once_loop_start, \
                    dim_len, self.pd_width, expect_cp_len)
                cp_data_len = cp_data_len + expect_cp_len
                seg_norm_value = self.calc_norm(flat_nram, 0, expect_cp_len)
                norm_value = norm_value + seg_norm_value
                if i == calc_loop_count - 1:  # last loop
                    index = self.get_norm_index(once_loop_start + expect_cp_len, dim_len)
                    if once_norm_ok == 0:
                        border_outputs[self.taskId * 2] = norm_value
                        idx_outputs[self.taskId * 2] = index
                    else:
                        border_outputs[self.taskId * 2 + 1] = norm_value
                        idx_outputs[self.taskId * 2 + 1] = index
            else:
                self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, once_loop_start, \
                    dim_len, self.pd_width, expect_cp_len)
                cp_data_len = cp_data_len + expect_cp_len
                seg_norm_value = self.calc_norm(flat_nram, 0, expect_cp_len)

                norm_value = norm_value + seg_norm_value

                once_norm_ok = 1
                index = self.get_norm_index(once_loop_start + expect_cp_len, dim_len)

                if cp_data_len < dim_len:
                    border_outputs[self.taskId * 2] \
                        = norm_value
                    idx_outputs[self.taskId * 2] = index
                else:
                    outputs[index] = tcp.scalar_pow(norm_value, pw)  # norm complete


                norm_value = 0.0
                cp_data_len = calc_size - expect_cp_len
                if cp_data_len > 0:
                    self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, \
                        once_loop_start + expect_cp_len, dim_len, \
                        self.pd_width, cp_data_len)
                    calc_result = self.calc_norm(flat_nram, 0, cp_data_len)

                    norm_value = calc_result
                    if i == calc_loop_count - 1:
                        border_outputs[self.taskId * 2 + 1] = norm_value
                        idx_outputs[self.taskId * 2 + 1] = index + 1

         


    def calc_pairwise_distance2(self, gram_tensor: ty.handle, border_outputs: ty.handle, idx_outputs: ty.handle, outputs: ty.handle):
        current_core_start = self.m_current_core_start
        total_count_in_core = self.m_total_count_in_core
        dim_len = self.pd_len
        norm_value = 0.0
        pw = 1 / self.gram_paras[0]

        flat_nram = self.nram_calc_buffer[:self.nram_process_count].reshape([self.nram_process_count, ])

        norm_offset = current_core_start % dim_len
        expect_cp_len = 0

        if norm_offset != 0:
            expect_cp_len = dim_len - norm_offset
            self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, current_core_start, \
                dim_len, self.pd_width, expect_cp_len)
            calc_result = self.calc_norm(flat_nram, 0, expect_cp_len)
            norm_value = calc_result
            index = self.get_norm_index(current_core_start + expect_cp_len, dim_len)
            border_outputs[self.taskId * 2] = norm_value
            idx_outputs[self.taskId * 2] = index

        norm_start_pos = current_core_start + expect_cp_len

        nram_norm_count = self.nram_process_count // dim_len

        total_norm_in_core = (total_count_in_core - expect_cp_len) // dim_len

        calc_loop_count = (total_norm_in_core + nram_norm_count - 1) // nram_norm_count

        once_loop_norm_count = nram_norm_count

        for i in range(calc_loop_count):
            once_loop_start = norm_start_pos + nram_norm_count * dim_len * i
            if i == calc_loop_count - 1:
                once_loop_norm_count = total_norm_in_core % nram_norm_count
                if once_loop_norm_count == 0:
                    once_loop_norm_count = nram_norm_count

            start_index = once_loop_start // dim_len
            for j in range(once_loop_norm_count):
                self.copy_from_2d_tensor(self.nram_calc_buffer, \
                    0, gram_tensor, once_loop_start + j * \
                    dim_len, dim_len, self.pd_width, dim_len)
                calc_result = self.calc_norm(flat_nram, 0, dim_len)
                norm_value = calc_result
                outputs[start_index + j] = tcp.scalar_pow(norm_value, pw)

        norm_loop_end_pos = norm_start_pos + total_norm_in_core * dim_len
        cur_loop_end_pos = current_core_start + total_count_in_core

        if norm_loop_end_pos < cur_loop_end_pos:
            self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, norm_loop_end_pos, \
                dim_len, \
                self.pd_width, cur_loop_end_pos - norm_loop_end_pos)
            calc_result = self.calc_norm(flat_nram, 0, cur_loop_end_pos - norm_loop_end_pos)
            norm_value = calc_result
            index = self.get_norm_index(norm_loop_end_pos + 1, dim_len)
            border_outputs[self.taskId * 2 + 1] = norm_value
            idx_outputs[self.taskId * 2 + 1] = index

    def main(self, Gram_tensor1: ty.handle, Gram_tensor2: ty.handle, Gram_paras: ty.handle,
                    len_tensor1: ty.int32, len_tensor2: ty.int32,
                    pd_len: ty.int32, pd_height: ty.int32, pd_width: ty.int32,
                    output_len: ty.int32,
                    Gram_border_buf_out: ty.handle,
                    Gram_border_idx_out: ty.handle,
                    Gram_buffer_out: ty.handle
                    ) -> None:
        tgt = tcp.target()
        self.bp = tgt

        gram_tensor1 = tcp.match_buffer(Gram_tensor1, [len_tensor1], dtype=self.dtype)
        gram_tensor2 = tcp.match_buffer(Gram_tensor2, [len_tensor2], dtype=self.dtype)
        self.gram_paras = tcp.match_buffer(Gram_paras, [2], dtype=self.dtype)

        border_array_size = 128
        gram_border_buf_out = tcp.match_buffer(Gram_border_buf_out, [border_array_size * 2], dtype=self.dtype)
        gram_border_idx_out = tcp.match_buffer(Gram_border_idx_out, [border_array_size * 2], dtype='int32')
        gram_buffer_out = tcp.match_buffer(Gram_buffer_out, [output_len], dtype=self.dtype)

        for cluster_id in tcp.thread_binding(0, tgt.cluster_num, thread="blockIdx.x"):
            for core_id in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):
                task_num = tgt.cluster_num * tgt.core_num
                task_id = tgt.core_num * cluster_id + core_id
                self.taskId = task_id

                self.calc_core_process_count(len_tensor1, task_num, task_id)
                self.sub_tensor(gram_tensor1, gram_tensor2, len_tensor2)
                tcp.sync_all()

                self.pd_width = pd_width
                #    tcp.print(len_tensor1, pd_height, pd_width)
                gram_reshape_tensor = gram_tensor1[:pd_height * pd_width].reshape([pd_height, pd_width])
                    #gram_reshape_tensor = gram_tensor1[:len_tensor1].reshape([pd_height, pd_width])
                tcp.sync_all()

                nram_available_size = self.calc_avaible_nram_count()
                self.nram_process_count = nram_available_size // self.dtype_size

                self.nram_calc_buffer = tcp.alloc_buffer(
                    shape=(self.nram_process_count, 1),
                    dtype=self.dtype,
                    scope="nram")

                self.pd_len = pd_len

                if self.pd_len > self.nram_process_count:
                    self.calc_pairwise_distance1(gram_reshape_tensor, gram_border_buf_out, \
                        gram_border_idx_out, gram_buffer_out)
                else:
                    if len_tensor1 // task_num + 1 < self.pd_len:
                        self.calc_pairwise_distance1(gram_reshape_tensor, \
                            gram_border_buf_out, gram_border_idx_out, gram_buffer_out)
                    else:
                        self.calc_pairwise_distance2(gram_reshape_tensor, gram_border_buf_out, \
                            gram_border_idx_out, gram_buffer_out)

                tcp.sync_all()

                #tcp.print(gram_border_buf_out)
                #tcp.print(gram_border_idx_out)

                if task_id == 0:
                    for i in range(border_array_size):
                        index1 = gram_border_idx_out[2 * i]
                        index2 = gram_border_idx_out[2 * i + 1]
                        norm_value1 = gram_border_buf_out[2 * i]
                        norm_value2 = gram_border_buf_out[2 * i + 1]

                        if index1 >= 0:
                            gram_buffer_out[index1] = gram_buffer_out[index1] + norm_value1

                        if index2 >= 0:
                            gram_buffer_out[index2] = gram_buffer_out[index2] + norm_value2


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_add(dtype=None, target=None):
    f = build_module.build(
        PairwiseDistance(dtype.name, dtype.bytes), target_tag=target, name=KERNEL_NAME
    )
    return f

