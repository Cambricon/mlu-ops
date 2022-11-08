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
        # tcp.print("task id : ", task_id)
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

    def sub_tensor(self, t1: ty.handle, t2: ty.handle, len_t2: ty.int32):
        nram_available_size = tcp.round_down((self.bp.nram_size - 30 * 1024) // 2, 128)
        nram_process_count = nram_available_size // self.dtype_size

        nram_buffer_in = tcp.alloc_buffer(
            shape=(2, nram_process_count),
            dtype=self.dtype,
            scope="nram")

        nram_buffer_in0 = nram_buffer_in[0][:]
        nram_buffer_in1 = nram_buffer_in[1][:]

        total_count_in_core = self.m_total_count_in_core
        calc_loop_count = (total_count_in_core + nram_process_count - 1) // nram_process_count

        for i in range(calc_loop_count):
            if i < calc_loop_count - 1:
                calc_size = nram_process_count
            else:
                calc_size = total_count_in_core % nram_process_count
                if calc_size == 0:
                    calc_size = nram_process_count

            once_loop_start = self.m_current_core_start + nram_process_count * i
            tcp.print("loop start ", self.m_current_core_start)

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
                    tcp.memcpy(nram_buffer_in1[0:head_len], t2[head_offset:len_t2])

                total_offset = head_len
                body_cp_count = (calc_size - head_len) // len_t2

                for _ in body_cp_count:
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

    
    """Operator description:
    Add the data in the two buffers.
    """
    def __init__(self, dtype: ty.string, dtype_size: ty.int32) -> None:
        self.dtype = dtype
        self.dtype_size = dtype_size

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

        gram_border_buf_out = tcp.match_buffer(Gram_border_buf_out, [256], dtype=self.dtype)
        gram_border_idx_out = tcp.match_buffer(Gram_border_idx_out, [256], dtype='int32')
        gram_buffer_out = tcp.match_buffer(Gram_buffer_out, [output_len], dtype=self.dtype)        

        a = 0
        for cluster_id in tcp.thread_binding(0, tgt.cluster_num, thread="blockIdx.x"):
            for core_id in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):
                task_num = tgt.cluster_num * tgt.core_num
                task_id = tgt.core_num * cluster_id + core_id
                #tcp.print(task_num, task_id)
                self.calc_core_process_count(len_tensor1, task_num, task_id)
                self.sub_tensor(gram_tensor1, gram_tensor2, len_tensor2)
                tcp.print(self.m_total_count_in_core)
                


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_add(dtype=None, target=None):
    f = build_module.build(
        PairwiseDistance(dtype.name, dtype.bytes), target_tag=target, name=KERNEL_NAME
    )
    return f
 