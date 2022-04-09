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

        self.bp.memcpy(dst[offset_dst:offset_dst + cp_len, 0:1], src[m:m + cp_len, n:n + 1])

    def calc_pairwise_distance(self, tensor, dim_len, height, width, head_tail_buf, outputs):
        nram_avable_size = round_down( (TARGET(self.target).nram_size - 30 * 1024), 128)
        nram_process_count = nram_avable_size // self.dtype_sz

        nram_norm_buffer = self.bp.Buffer(
            shape=(nram_process_count, 1),
            name="nram_norm_buffer",
            dtype=self.dtype,
            scope="nram",
        )

        current_core_start = self._data_man._current_core_start
        total_count_in_core = self._data_man._total_count_in_core
        once_loop_start = self.bp.Scalar(bangpy.int32, "once_loop_start")

        
        calc_loop_count = self.bp.Scalar(bangpy.int32, "calc_loop_count", (total_count_in_core + nram_process_count - 1) // nram_process_count)

        norm_value = 0.0
        norm_total_count = 0

        '''
        0 : 要压缩的维度，从中间开始，且比nram还要长
        1 : 要压缩的维度，从中间开始，比nram小
        2 : 要压缩的维度，从头开始
        '''
        oper_type = self.bp.Scalar(bangpy.int32, "oper_type", 0)

        # 确认本次循环要从gram拷贝回nram的数量      
        calc_size = self.bp.Scalar(bangpy.int32, "calc_size", nram_process_count)

        with self.bp.for_range(0, calc_loop_count) as i:
            with self.bp.if_scope(i == calc_loop_count - 1):
                calc_size.assign(total_count_in_core % nram_process_count)
                
            self.bp.print("cur core calc data size ", total_count_in_core, calc_size)


            # 确认本次要处理的数据开头，在gram中的偏移量
            #当前核心数据开始的位置 + 第i次循环所应偏移的长度
            cur_loop_start = self.bp.Scalar(bangpy.int32, "cur_loop_start", current_core_start + nram_process_count * i)

            # 先判断一下，是那种类型
            norm_offset = self.bp.Scalar(bangpy.int32, "norm_offset", cur_loop_start % dim_len)
            with self.bp.if_scope(norm_offset == 0):
                # 如果正好是开头，那么就走正常流程
                oper_type.assign(2)
            with self.bp.else_scope():
                with self.bp.if_scope(dim_len - norm_offset >= calc_size):
                    oper_type.assign(0)
                with self.bp.else_scope():
                    oper_type.assign(1)

            # 开始拷贝数据了
            nram_pos_offset = self.bp.Scalar(bangpy.int32, "nram_pos_offset", 0)
            tail_size = self.bp.Scalar(bangpy.int32, "tail_size", 0)
            body_cp_count = self.bp.Scalar(bangpy.int32, "body_cp_count", 0)
            with self.bp.block("data_copy"):
                with self.bp.if_scope(oper_type == 0):
                    self.copy_from_2d_tensor(nram_norm_buffer, 0, tensor, cur_loop_start, dim_len, height, width, calc_size)
                    norm_offset.assign(norm_offset + calc_size)
                with self.bp.else_scope():                    
                    with self.bp.if_scope(oper_type == 1):
                        # 拷贝头，身，尾巴
                        head_len = self.bp.Scalar(bangpy.int32, "head_len", dim_len - norm_offset)
                        nram_pos_offset.assign(dim_len - norm_offset)
                        self.bp.print(" bada ", head_len)
                    
                    # 拷贝身，尾巴, 如果state是2，norm_offset 就是 0
                    # 先计算一下，能拷贝多少
                    nram_remain_len = self.bp.Scalar(bangpy.int32, "nram_remain_len", calc_size - nram_pos_offset)
                    body_cp_count.assign(nram_remain_len // dim_len) # 这个可能是0
                    with self.bp.for_range(0, body_cp_count) as j:
                        self.copy_from_2d_tensor(nram_norm_buffer, nram_pos_offset + j * dim_len, 
                                                 tensor, cur_loop_start + nram_pos_offset + j * dim_len, 
                                                 dim_len, height, width, dim_len)
                        body_norm_value = self.calc_norm(nram_norm_buffer, nram_pos_offset + j * dim_len, nram_pos_offset + (j + 1) * dim_len)
                        outputs[nram_pos_offset // dim_len + j] = body_norm_value # 计算完毕，直接拷贝回输出
                        norm_total_count += 1

                    # 处理尾巴
                    tail_size.assign(nram_remain_len % dim_len)
                    with self.bp.if_scope(tail_size > 0):
                        self.copy_from_2d_tensor(nram_norm_buffer, nram_pos_offset + body_cp_count * dim_len, 
                                                 tensor, cur_loop_start +  nram_pos_offset + body_cp_count * dim_len, 
                                                 dim_len, height, width, tail_size)    

            seg_norm_value = None
            with self.bp.block("compute"):
                with self.bp.if_scope(oper_type == 0):
                    seg_norm_value = self.calc_norm(nram_norm_buffer, 0, calc_size)
                    norm_value = seg_norm_value + norm_value
                    with self.bp.if_scope(norm_offset == dim_len):
                        #norm_offset 不用累加，cur_loop_start 已经累加过了，一个元素已经处理完毕了
                        norm_total_count += 1
                with self.bp.else_scope():   
                    # 身子直接计算，拷贝完成了。只考虑tail的问题。
                    tail_start = nram_pos_offset + body_cp_count * dim_len
                    norm_value = self.calc_norm(nram_norm_buffer, tail_start, tail_start + tail_size)

            with self.bp.block("data_copy"):
                with self.bp.if_scope(oper_type == 0):        
                    with self.bp.if_scope(norm_offset == dim_len):
                        #norm_offset 不用累加，cur_loop_start 已经累加过了，一个元素已经处理完毕了
                        if norm_total_count == 1:
                            # 这个 core 处理的第一个张量
                            head_tail_buf[2 * self.bp.taskId] = norm_value
                            norm_value = 0.0
                with self.bp.else_scope():   
                    # 身子也不管了，只考虑tail
                    with self.bp.if_scope(tail_size > 0): # 尾巴长度不是0
                        with self.bp.if_scope(i == calc_loop_count - 1): # 最后一个循环
                            head_tail_buf[2 * self.bp.taskId + 1] = norm_value # 彻底完了.

            self.bp.sync_all()
            with self.bp.if_scope(self.bp.taskId != 0):
                # 把接缝处给它补上。0号core不需要补
                with self.bp.if_scope(current_core_start % dim_len != 0):
                    head = head_tail_buf[2 * (self.bp.taskId - 1) + 1]
                    tail = head_tail_buf[2 * self.bp.taskId]
                    outputs[(current_core_start + dim_len - 1) // dim_len - 1] = head + tail
                            




    def calc_norm(self, buffer, start, end):
        result = self.bp.Scalar(self.dtype, "size", 0.0)
        size = self.bp.Scalar(bangpy.int32, "size", end - start - 1)
        with self.bp.for_range(0, size) as i:          
            result.assign(buffer[i] + result)
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

        gram_border_buf_out = self.bp.Buffer(
            shape=(256, ), name="gram_border_buf_out", dtype=self.dtype, scope="global"
        )

        self._data_man.calc_core_process_count(self.len_tensor1, self.task_num)

        self.sub_tensor(gram_tensor1, gram_tensor2, self.len_tensor1, self.len_tensor2)
        self.bp.sync_all()

        with self.bp.if_scope(self.bp.taskId == 0):
            gram_reshape_tensor = gram_tensor1.reshape([self.pd_height, self.pd_width])
            self.bp.print("************************")
            self.bp.print(gram_tensor1)
            self.bp.print(gram_reshape_tensor)
            self.bp.print("************************")
        self.bp.sync_all()

        self.calc_pairwise_distance(gram_reshape_tensor, self.pd_len, self.pd_height, self.pd_width, gram_border_buf_out, gram_buffer_out)

        self.bp.sync_all()


        f = self.bp.BuildBANG(
            inputs=[gram_tensor1, gram_tensor2,
                    self.len_tensor1, self.len_tensor2,
                    self.pd_len, self.pd_height, self.pd_width,
                    self.output_len],
            outputs=[gram_buffer_out, gram_border_buf_out],
            kernel_name=KERNEL_NAME,
        )
        return f

@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_pairwisedistance(dtype=None, target=None):
    task_num = 1
    f = PairwiseDistance(dtype, target, task_num).compute_body()
    return f



