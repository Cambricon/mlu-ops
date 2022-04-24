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
import numpy as np

import bangpy
import bangpy as bp
from bangpy.tcp.util import round_up, round_down
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType
import sys
sys.path.append("..")
from data_man import *

DTYPES = [bangpy.float32] #支持的类型
TARGET_LIST = ["mlu290"]#支持的设备
KERNEL_NAME = "PairwiseDistance"#算子名


class PairwiseDistance(object):
    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.dtype_sz = dtype.bytes
        self.bp = tcp.TCP(target)
        self._data_man = data_man()

    def sub_tensor(self, t1, t2, len_t1, len_t2):
        nram_avable_size = round_down( (TARGET(self.target).nram_size - 30 * 1024) // 2, 128)
        nram_process_count = nram_avable_size // self.dtype_sz

        nram_buffer_in = self.bp.Buffer(
            shape=(2, nram_process_count),
            name="nram_buffer_in",
            dtype=self.dtype,
            scope="nram")

        nram_buffer_in0 = nram_buffer_in[0][:] 
        nram_buffer_in1 = nram_buffer_in[1][:]  


        current_core_start = self._data_man._current_core_start
        total_count_in_core = self._data_man._total_count_in_core
        once_loop_start = self.bp.Scalar(bangpy.int32, "once_loop_start")

        calc_size = self.bp.Scalar(bangpy.int32, "calc_size")
        calc_loop_count = self.bp.Scalar(bangpy.int32, "calc_loop_count", (total_count_in_core + nram_process_count - 1) // nram_process_count)

        with self.bp.for_range(0, calc_loop_count) as i:          
            with self.bp.if_scope(i < calc_loop_count - 1):
                calc_size.assign(nram_process_count)
            with self.bp.else_scope():
                calc_size.assign(total_count_in_core % nram_process_count)

            once_loop_start.assign(current_core_start + nram_process_count * i) #当前核心数据开始的位置 + 第i次循环所应偏移的长度

            with self.bp.block("data_copy"):
                # tensor1 copy
                self.bp.memcpy(nram_buffer_in0[0:calc_size], t1[once_loop_start:once_loop_start + calc_size]) 

                # tensor2 copy
                head_offset = self.bp.Scalar(bangpy.int32, "head_len", once_loop_start % len_t2)

                with self.bp.if_scope(head_offset == 0):
                    head_len = self.bp.Scalar(bangpy.int32, "head_len", 0)
                with self.bp.else_scope():
                    head_len = self.bp.Scalar(bangpy.int32, "head_len", len_t2 - head_offset)

                with self.bp.if_scope(head_len >= calc_size):
                    self.bp.memcpy(nram_buffer_in1[0:calc_size], t2[head_offset:head_offset + calc_size])
                with self.bp.else_scope():
                    with self.bp.if_scope(head_len > 0):
                        self.bp.memcpy(nram_buffer_in1[0:head_len], t2[head_offset:len_t2])             

                    total_offset = self.bp.Scalar(bangpy.int32, "total_offset")
                    total_offset.assign(head_len)

                    body_cp_count = self.bp.Scalar(bangpy.int32, "body_cp_count", (calc_size - head_len) // len_t2)

                    with self.bp.for_range(0, body_cp_count) as j: 
                        self.bp.memcpy(nram_buffer_in1[total_offset:total_offset + len_t2], t2[0:len_t2])    
                        total_offset.assign(total_offset + len_t2)                            

                    offset_end = self.bp.Scalar(bangpy.int32, "offset_end", (once_loop_start + calc_size) % len_t2)
                    
                    with self.bp.if_scope(offset_end > 0):
                        self.bp.memcpy(nram_buffer_in1[total_offset:total_offset + offset_end], t2[0:offset_end])      

            with self.bp.block("compute"):
                self.bp.subtract(nram_buffer_in0, nram_buffer_in0, nram_buffer_in1)
                self.bp.abs(nram_buffer_in0, nram_buffer_in0)

            with self.bp.block("data_copy"):
                self.bp.memcpy(t1[once_loop_start:once_loop_start + calc_size], nram_buffer_in0[:calc_size])   


    def copy_from_2d_tensor(self, dst, offset_dst, src, offset_src, dim_len, height, width, cp_len):
        dim_col_count = height // dim_len

        big_row = offset_src // (width * dim_len)
        m = offset_src % dim_len + big_row * dim_len

        big_n = (offset_src + dim_len - 1) % dim_len
        n = big_n % width

        with self.bp.if_scope(offset_dst != offset_dst + cp_len // 2):
            self.bp.memcpy(dst[offset_dst:offset_dst + cp_len // 2, 0:1], src[m:m + cp_len  // 2, n:n + 1])

        with self.bp.if_scope(offset_dst + cp_len // 2 != offset_dst + cp_len):
            self.bp.memcpy(dst[offset_dst + cp_len // 2:offset_dst + cp_len, 0:1], src[m + cp_len // 2:m + cp_len, n:n + 1])                            

    def calc_norm(self, buffer, start, end):
        result = self.bp.Scalar(self.dtype, "result", 0.0)
        size = self.bp.Scalar(bangpy.int32, "size", end - start)
        with self.bp.for_range(0, size) as i:
            result.assign(result + buffer[start + i])
        return result
    
    def compute_body(self):
        self._data_man.init(self.bp)
        self.bp.launch_task(self.task_num, 1, 1)

        self.len_tensor1 = self.bp.SizeVar("len_tensor1")
        self.len_tensor2 = self.bp.SizeVar("len_tensor2")
        
        self.pd_len = self.bp.SizeVar("pd_len")
        self.pd_height = self.bp.SizeVar("pd_height")
        self.pd_width = self.bp.SizeVar("pd_width")

        self.output_len = self.bp.SizeVar("output_len")

        gram_tensor1 = self.bp.Buffer(
            shape=(self.len_tensor1, ), name="gram_tensor1", dtype=self.dtype, scope="global"
        )

        gram_tensor2 = self.bp.Buffer(
            shape=(self.len_tensor2, ), name="gram_tensor2", dtype=self.dtype, scope="global"
        )

        gram_buffer_out = self.bp.Buffer(
            shape=(self.output_len, ), name="gram_buffer_out", dtype=self.dtype, scope="global"
        )

        border_array_size = 5
        gram_border_buf_out = self.bp.Buffer(
            shape=(border_array_size * 2, ), name="gram_border_buf_out", dtype=self.dtype, scope="global"
        )
        gram_border_idx_out = self.bp.Buffer(
            shape=(border_array_size * 2, ), name="gram_border_idx_out", dtype=bangpy.int32, scope="global"
        )

        self._data_man.calc_core_process_count(self.len_tensor1, self.task_num)

        self.sub_tensor(gram_tensor1, gram_tensor2, self.len_tensor1, self.len_tensor2)
        self.bp.sync_all()

        with self.bp.if_scope(self.bp.taskId == 0):
            gram_reshape_tensor = gram_tensor1.reshape([self.pd_height, self.pd_width])
        self.bp.sync_all()

        nram_avable_size = round_down(TARGET(self.target).nram_size - 500 * 1024, 128)
        self.nram_process_count = nram_avable_size // self.dtype_sz
        self.nram_calc_buffer = self.bp.Buffer(
            shape=(self.nram_process_count, 1),
            name="nram_calc_buffer",
            dtype=self.dtype,
            scope="nram")

        with self.bp.if_scope(self.pd_len > self.nram_process_count):
            self.calc_pairwise_distance1(gram_reshape_tensor, gram_border_buf_out, gram_border_idx_out, gram_buffer_out)
        with self.bp.else_scope():
            self.calc_pairwise_distance2(gram_reshape_tensor, gram_border_buf_out, gram_border_idx_out, gram_buffer_out)

        self.bp.sync_all()

        # 处理边界数据
        with self.bp.if_scope(self.bp.taskId == 0):
            with self.bp.for_range(0, border_array_size) as i:
                index1 = gram_border_idx_out[2 * i]
                index2 = gram_border_idx_out[2 * i + 1]
                norm_value1 = gram_border_buf_out[2 * i]
                norm_value2 = gram_border_buf_out[2 * i + 1]

                gram_buffer_out[index1] = gram_buffer_out[index1] + norm_value1
                gram_buffer_out[index2] = gram_buffer_out[index2] + norm_value2


        f = self.bp.BuildBANG(
            inputs=[gram_tensor1, gram_tensor2,
                    self.len_tensor1, self.len_tensor2,
                    self.pd_len, self.pd_height, self.pd_width,
                    self.output_len],
            outputs=[gram_border_buf_out, gram_border_idx_out, gram_buffer_out],
            kernel_name=KERNEL_NAME
            )
        return f

    def get_norm_index(self, data_pos, dim_len):
        index = (data_pos + dim_len - 1) // dim_len
        return index - 1

    def calc_pairwise_distance1(self, gram_tensor, border_outputs, idx_outputs, outputs):# nram 一次还存不下一个元素
        current_core_start = self._data_man._current_core_start
        total_count_in_core = self._data_man._total_count_in_core
        calc_loop_count = self.bp.Scalar(bangpy.int32, "calc_loop_count", (total_count_in_core + self.nram_process_count - 1) // self.nram_process_count)
        norm_value = self.bp.Scalar(self.dtype, "norm_value", 0.0)

        once_loop_start = self.bp.Scalar(bangpy.int32, "once_loop_start")        

        oper_type = self.bp.Scalar(bangpy.int32, "oper_type", 0)

        dim_len = self.pd_len
        norm_offset = self.bp.Scalar(bangpy.int32, "norm_offset", current_core_start % dim_len)
        with self.bp.if_scope(norm_offset == 0):
            oper_type.assign(2)
        with self.bp.else_scope():
            oper_type.assign(0)

        flat_nram = self.nram_calc_buffer.reshape([self.nram_process_count, ])
        '''
        0 : 要压缩的维度，从中间开始，且比nram还要长
        1 : 要压缩的维度，从中间开始，比nram小
        2 : 要压缩的维度，从头开始
        '''
        #以上记录一下，是否是从半截开始处理的，如果是，要缓存

        complete_norm_count = self.bp.Scalar(bangpy.int32, "complete_norm_count", 0)

        norm_value = self.bp.Scalar(self.dtype, "norm_value", 0.0)
        # 确认本次循环要从gram拷贝回nram的数量      
        calc_size = self.bp.Scalar(bangpy.int32, "calc_size", self.nram_process_count)

        once_norm_ok = self.bp.Scalar(bangpy.int32, "once_norm_ok", 0)     
        cp_data_len = self.bp.Scalar(bangpy.int32, "cp_data_len", 0)
        with self.bp.for_range(0, calc_loop_count) as i:
            once_loop_start.assign(current_core_start + self.nram_process_count * i)
            with self.bp.if_scope(i == calc_loop_count - 1):
                calc_size.assign(total_count_in_core % self.nram_process_count)

            norm_offset.assign(once_loop_start % dim_len)
            expect_cp_len = self.bp.Scalar(bangpy.int32, "expect_cp_len", dim_len - norm_offset)

            with self.bp.if_scope(expect_cp_len > calc_size):
                expect_cp_len.assign(calc_size)
                # 一口气拷贝不完，那就尽可能多的拷贝.
                self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, once_loop_start, dim_len, self.pd_height, self.pd_width, expect_cp_len)
                cp_data_len.assign(cp_data_len + expect_cp_len)               
                seg_norm_value = self.calc_norm(flat_nram, 0, expect_cp_len)
                norm_value.assign(norm_value + seg_norm_value)
                with self.bp.if_scope(i == calc_loop_count - 1): # 最后一个循环了
                    # 缓存一下
                    index = self.get_norm_index(once_loop_start + expect_cp_len, dim_len)
                    with self.bp.if_scope(once_norm_ok == 0):
                        border_outputs[self.bp.taskId * 2] = norm_value # 走到这里了，说明这个core一直在处理一个norm的中间部分
                        idx_outputs[self.bp.taskId * 2] = index
                    with self.bp.else_scope():
                        border_outputs[self.bp.taskId * 2 + 1] = norm_value 
                        idx_outputs[self.bp.taskId * 2 + 1] = index
                
            with self.bp.else_scope():
                #这个norm可以拷贝完了
                self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, once_loop_start, dim_len, self.pd_height, self.pd_width, expect_cp_len)
                cp_data_len.assign(cp_data_len + expect_cp_len)
                seg_norm_value = self.calc_norm(flat_nram, 0, expect_cp_len)
                norm_value.assign(norm_value + seg_norm_value)

                # 标记一下
                once_norm_ok.assign(1)
                # 看看这个norm是不是半截
                index = self.get_norm_index(once_loop_start + expect_cp_len, dim_len)
                with self.bp.if_scope(cp_data_len < dim_len):   
                    border_outputs[self.bp.taskId * 2] = norm_value # 走到这里了，说明这个core一直在处理一个norm的中间部分
                    idx_outputs[self.bp.taskId * 2] = index
                with self.bp.else_scope():
                    outputs[index] = norm_value # 一个完整的norm算出来了 

                norm_value.assign(0.0)
                
                # 接下来，拷贝下一个norm
                cp_data_len.assign(calc_size - expect_cp_len)
                with self.bp.if_scope(cp_data_len > 0):
                    self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, once_loop_start + expect_cp_len, dim_len, self.pd_height, self.pd_width, cp_data_len)
                    calc_result = self.calc_norm(flat_nram, 0, cp_data_len)
                    norm_value.assign(calc_result)
                    with self.bp.if_scope(i == calc_loop_count - 1): # 最后一个循环了
                        # 肯定没有拷贝完
                        border_outputs[self.bp.taskId * 2 + 1] = norm_value 
                        idx_outputs[self.bp.taskId * 2 + 1] = index + 1     
                        
                
                        



    def calc_pairwise_distance2(self, gram_tensor, border_outputs, idx_outputs, outputs):
        pass

@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_pairwisedistance(dtype=None, target=None):
    task_num = 4
    f = PairwiseDistance(dtype, target, task_num).compute_body()
    return f



