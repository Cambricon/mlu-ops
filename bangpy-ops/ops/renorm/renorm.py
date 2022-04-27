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
KERNEL_NAME = "Renorm"#算子名


class Renorm(object):
    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.dtype_sz = dtype.bytes
        self.bp = tcp.TCP(target)
        self._data_man = data_man()

    def calc_pow(self, gram_tensor, p):
        nram_avable_size = round_down( (TARGET(self.target).nram_size - 30 * 1024) // 2, 128)
        nram_process_count = nram_avable_size // self.dtype_sz

        nram_buffer_in = self.bp.Buffer(
            shape=(2 * nram_process_count, ),
            name="nram_buffer_in",
            dtype=self.dtype,
            scope="nram")

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
                self.bp.memcpy(nram_buffer_in[0:calc_size], gram_tensor[once_loop_start:once_loop_start + calc_size]) 

            with self.bp.block("compute"):
                # 求绝对值
                self.bp.abs(nram_buffer_in, nram_buffer_in)

                # 求指数
                self.bp.log(nram_buffer_in, nram_buffer_in)
                pw = self.bp.Scalar(name='p', dtype=self.dtype, value=p)
                self.bp.multiply(nram_buffer_in, nram_buffer_in, pw)
                self.bp.exp(nram_buffer_in, nram_buffer_in)

            with self.bp.block("data_copy"):
                self.bp.memcpy(gram_tensor[once_loop_start:once_loop_start + calc_size], nram_buffer_in[:calc_size])   



    def copy_from_2d_tensor(self, dst, md, nd, src, m, n, cp_len):
        with self.bp.if_scope(cp_len // 2 != 0):
            self.bp.memcpy(dst[md:md + cp_len // 2, nd:nd + 1], src[m:m + cp_len // 2, n:n + 1])

        with self.bp.if_scope(cp_len // 2 != cp_len):
            self.bp.memcpy(dst[md + cp_len // 2:md + cp_len, nd:nd + 1], src[m + cp_len // 2:m + cp_len, n:n + 1])                            
    

    def calc_norm(self, buffer, start, end):
        result = self.bp.Scalar(self.dtype, "result", 0.0)
        size = self.bp.Scalar(bangpy.int32, "size", end - start)
        with self.bp.for_range(0, size) as i:
            result.assign(result + buffer[start + i])
        return result

    def scalar_pow(self, value, p):
        self.nram_pow_buffer[0] = value
        self.bp.log(self.nram_pow_buffer, self.nram_pow_buffer)
        pw = self.bp.Scalar(name='pw', dtype=self.dtype, value=p)
        self.bp.multiply(self.nram_pow_buffer, self.nram_pow_buffer, pw)
        self.bp.exp(self.nram_pow_buffer, self.nram_pow_buffer)
        return self.nram_pow_buffer[0]
    
    def process_sub_tensor(self, gram_tensor, p, maxnorm, h, w, sub_wid, output_tensor):
        # 先拆分数据
        st_start = self._data_man._current_core_start
        st_count = self._data_man._total_count_in_core

        nram_avable_size = round_down(TARGET(self.target).nram_size - 30 * 1024, 128)
        self.nram_process_count = nram_avable_size // self.dtype_sz
        self.nram_calc_buffer = self.bp.Buffer(
            shape=(self.nram_process_count, 1),
            name="nram_calc_buffer",
            dtype=self.dtype,
            scope="nram")

        once_loop_start = self.bp.Scalar(bangpy.int32, "once_loop_start")   


        cp_row_count = self.bp.Scalar(bangpy.int32, "cp_row_count", (h + self.nram_process_count - 1 ) // self.nram_process_count)
        cp_len = self.bp.Scalar(bangpy.int32, "cp_len", self.nram_process_count)

        with self.bp.for_range(0, st_count) as i: # 这个nram上要处理的子向量个数
            st_norm_value = self.bp.Scalar(self.dtype, "st_norm_value", 0.0)
            start_col = self.bp.Scalar(bangpy.int32, "start_col", sub_wid * (st_start + i))
            end_col = self.bp.Scalar(bangpy.int32, "end_col", sub_wid * (st_start + i + 1))
            with self.bp.for_range(0, end_col) as j: #要拷贝的列的起止
                with self.bp.if_scope(j >= start_col): #不这么写，编译不过，
                    #一次拷贝一列
                    with self.bp.for_range(0, cp_row_count) as k:
                        with self.bp.if_scope(k == cp_row_count - 1):
                            cp_len.assign(h % self.nram_process_count)

                        self.copy_from_2d_tensor(self.nram_calc_buffer, 0, 0, 
                                                 gram_tensor, self.nram_process_count * k, j, cp_len)


                        # 求绝对值
                        self.bp.abs(self.nram_calc_buffer, self.nram_calc_buffer)

                        # 求指数
                        self.bp.log(self.nram_calc_buffer, self.nram_calc_buffer)
                        pw = self.bp.Scalar(name='p', dtype=self.dtype, value=p)
                        self.bp.multiply(self.nram_calc_buffer, self.nram_calc_buffer, pw)
                        self.bp.exp(self.nram_calc_buffer, self.nram_calc_buffer)

                        calc_ret = self.calc_norm(self.nram_calc_buffer, 0, cp_len)
                        st_norm_value.assign(st_norm_value + calc_ret)

            #计算一下norm，
            st_norm_value.assign(self.scalar_pow(st_norm_value, 1 / p))
            with self.bp.if_scope(st_norm_value > maxnorm):
                tmp = self.scalar_pow(maxnorm / st_norm_value, p)
                with self.bp.for_range(0, end_col) as j: #要拷贝的列的起止
                    with self.bp.if_scope(j >= start_col): #不这么写，编译不过，
                        #一次拷贝一列
                        with self.bp.for_range(0, cp_row_count) as k:
                            with self.bp.if_scope(k == cp_row_count - 1):
                                cp_len.assign(h % self.nram_process_count)

                            self.copy_from_2d_tensor(self.nram_calc_buffer, 0, 0, 
                                                     gram_tensor, self.nram_process_count * k, j, cp_len)

                            scalar_tmp = self.bp.Scalar(name='p', dtype=self.dtype, value=tmp)
                            self.bp.multiply(self.nram_calc_buffer, self.nram_calc_buffer, scalar_tmp)

                            #拷贝到output里面去
                            self.copy_from_2d_tensor(output_tensor, self.nram_process_count * k, j, 
                                                     self.nram_calc_buffer, 0, 0, cp_len)


    def compute_body(self):
        self._data_man.init(self.bp)
        self.bp.launch_task(self.task_num, 1, 1)

        self.h = self.bp.SizeVar("h")
        self.w = self.bp.SizeVar("w")
        
        self.sub_wid = self.bp.SizeVar("sub_wid")

        gram_tensor = self.bp.Buffer(
            shape=(self.h * self.w, ), name="gram_tensor", dtype=self.dtype, scope="global"
        )

        gram_rshp_tensor = gram_tensor.reshape([self.h, self.w])

        gram_buffer_out = self.bp.Buffer(
            shape=(self.h * self.w, ), name="gram_buffer_out", dtype=self.dtype, scope="global"
        )


        gram_reshp_buffer_out = gram_buffer_out.reshape([self.h, self.w])

        gram_paras = self.bp.Buffer(
            shape=(2, ), name="gram_paras", dtype=self.dtype, scope="global"
        )
        self.nram_paras = self.bp.Buffer(
            shape=(2, ),
            name="nram_paras",
            dtype=self.dtype,
            scope="nram")
        self.bp.memcpy(self.nram_paras[0:2], gram_paras[0:2]) 

        self.nram_pow_buffer = self.bp.Buffer(
            shape=(128, ),
            name="nram_pow_buffer",
            dtype=self.dtype,
            scope="nram")        

        #self._data_man.calc_core_process_count(self.h * self.w, self.task_num)
        #self.calc_pow(gram_tensor, self.nram_paras[0])

        self._data_man.calc_core_process_count(self.w // self.sub_wid, self.task_num)
        self.process_sub_tensor(gram_rshp_tensor, self.nram_paras[0], self.nram_paras[1], 
            self.h, self.w, self.sub_wid
            , gram_reshp_buffer_out)

        

        f = self.bp.BuildBANG(
            inputs=[gram_tensor, gram_paras,
                    self.h, self.w,
                    self.sub_wid                 
                    ],
            outputs=[gram_buffer_out],
            kernel_name=KERNEL_NAME
            )
        return f

    
            

@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_renorm(dtype=None, target=None):
    task_num = 4
    f = Renorm(dtype, target, task_num).compute_body()
    return f



