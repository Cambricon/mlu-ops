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
        self._data_man.calc_core_process_count(len_t1, self.task_num)

        nram_avable_size = round_down( (TARGET(self.target).nram_size - 30 * 1024) // 2, 128)
        nram_process_count = nram_avable_size // self.dtype_sz

        nram_buffer_in0 = self.bp.Buffer(
            shape=(nram_process_count, ),
            name="nram_buffer_in0",
            dtype=self.dtype,
            scope="nram",
        ) 

        nram_buffer_in1 = self.bp.Buffer(
            shape=(nram_process_count, ),
            name="nram_buffer_in1",
            dtype=self.dtype,
            scope="nram",
        )

        current_core_start = self._data_man._current_core_start
        once_loop_start = self.bp.Scalar(bangpy.int32,"once_loop_start")
        total_count_in_core = self._data_man._total_count_in_core
        calc_size = self.bp.Scalar(bangpy.int32,"calc_size")
        const_one = self.bp.Scalar(dtype = self.dtype, name = "const_one", value = 1)
        calc_loop_count = self.bp.Scalar(bangpy.int32,"calc_loop_count")
        calc_loop_count.assign((total_count_in_core + nram_process_count - 1) // nram_process_count)

        self.bp.print('bada hutong calc_loop_count ', calc_loop_count)

        with self.bp.for_range(0, calc_loop_count) as i:            
            once_loop_start.assign(current_core_start + nram_process_count * i) #当前核心数据开始的位置 + 第i次循环所应偏移的长度
            with self.bp.if_scope(i < calc_loop_count - 1):
                calc_size.assign(nram_process_count)
            with self.bp.else_scope():
                calc_size.assign(total_count_in_core % nram_process_count)

            with self.bp.block("data_copy"):
                self.bp.memcpy(nram_buffer_in0[0:calc_size], t1[once_loop_start:once_loop_start + calc_size]) 

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

                    body_cp_count = self.bp.Scalar(bangpy.int32, "body_cp_count")
                    body_cp_count.assign((calc_size - head_len) // len_t2)

                    with self.bp.for_range(0, body_cp_count) as j: 
                        self.bp.memcpy(nram_buffer_in1[total_offset:total_offset + len_t2], t2[0:len_t2])    
                        total_offset.assign(total_offset + len_t2)                            

                    offset_end = self.bp.Scalar(bangpy.int32, "offset_end")
                    offset_end.assign((once_loop_start + calc_size) % len_t2)
                    
                    with self.bp.if_scope(offset_end > 0):
                        self.bp.memcpy(nram_buffer_in1[total_offset:total_offset + offset_end], t2[0:offset_end])      

            with self.bp.block("compute"):
                self.bp.subtract(nram_buffer_in0, nram_buffer_in0, nram_buffer_in1)# y-x

            with self.bp.block("data_copy"):
                self.bp.memcpy(t1[once_loop_start:once_loop_start + calc_size], nram_buffer_in0[:calc_size])   

    
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

        self.sub_tensor(gram_tensor1, gram_tensor2, self.len_tensor1, self.len_tensor2)
        self.bp.sync_all()


       
        '''
        nram_buff = self.bp.Buffer(
            shape=(2, 1), dtype=self.dtype, name="count", scope="nram"
        )

        self.bp.print(gram_x)
        self.bp.memcpy(nram_buff[0:2, 0:1], gram_x[0:2, 0:1])

        self.bp.print(nram_buff)
        self.bp.print("-------", )


        self.bp.print(gram_y)
        self.bp.print(gram_shp_x)
        self.bp.print(gram_shp_y)
        self.bp.print('shp x len ', self.shp_x_len)
        self.bp.print(self.shp_y_len)
        #self.bp.print(buffer_out)
        '''

        f = self.bp.BuildBANG(
            inputs=[gram_tensor1, gram_tensor2, 
                    self.len_tensor1, self.len_tensor2,
                    self.pd_len, self.pd_height, self.pd_width,
                    self.output_len],
            outputs=[gram_buffer_out],
            kernel_name=KERNEL_NAME,
        )
        return f

@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_pairwisedistance(dtype=None, target=None):
    task_num = 8
    f = PairwiseDistance(dtype, target, task_num).compute_body()
    return f



