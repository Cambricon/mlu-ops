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

                # 先加上eps
                eps = self.bp.Scalar(name='input_eps', dtype=self.dtype, value=self.nram_pd_paras[1])
                self.bp.add(nram_buffer_in0, nram_buffer_in0, eps)

                # 求指数
                self.bp.log(nram_buffer_in0, nram_buffer_in0)
                p = self.bp.Scalar(name='p', dtype=self.dtype, value=self.nram_pd_paras[0])
                self.bp.multiply(nram_buffer_in0, nram_buffer_in0, p)
                self.bp.exp(nram_buffer_in0, nram_buffer_in0)

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
    # #end_index 是c风格的数组索引  不是py的  
    # def calc_norm(self, buffer, start_index, end_index):
    #     natural_base = self.bp.Scalar(bangpy.float32,"natural_base",2.7182818284590452353602874713526624977572470936999)#5957496696762772407663035354759457138217852516642742746639193200305992181741359662904357290033429526059563)# 07381 32328 62794 34907 63233 82988 07531 95251 01901 15738 34187 93070 21540 89149 93488 41675 09244 76146 06680 82264 80016 84774 11853 74234 54424 37107 53907 77449 92069 55170 27618 38606 26133 13845 83000 75204 49338 26560 29760 67371 13200 70932 87091 27443 74704 72306 96977 20931 01416 92836 81902 55151 08657 46377 21112 52389 78442 50569 53696 77078 54499 69967 94686 44549 05987 93163 68892 30098 79312 77361 78215 42499 92295 76351 48220 82698 95193 66803 31825 28869 39849 64651 05820 93923 98294 88793 32036 25094 43117 30123 81970 68416 14039 70198 37679 32068 32823 76464 80429 53118 02328 78250 98194 55815 30175 67173 61332 06981 12509 96181 88159 30416 90351 59888 85193 45807 27386 67385 89422 87922 84998 92086 80582 57492 79610 48419 84443 63463 24496 84875 60233 62482 70419 78623 20900 21609 90235 30436 99418 49146 31409 34317 38143 64054 62531 52096 18369 08887 07016 76839 64243 78140 59271 45635 49061 30310 72085 10383 75051 01157 47704 17189 86106 87396 96552 12671 54688 95703 50354 )
    #     const_one = self.bp.Scalar(bangpy.float32,"const_one",1)
    #     max_threshold_valu = self.bp.Scalar(bangpy.float32,"max_threshold_valu")
    #     min_threshold_valu = self.bp.Scalar(bangpy.float32,"min_threshold_valu")
    #     #这些数我是网上查的该类型大于0时的最大最小值 然后取了个ln得到的 
    #     max_threshold_valu.assign(88.722008965395851698332450562653)
    #     min_threshold_valu.assign(-87.332719095296162600686375692197)
    #     data_length = self.bp.Scalar(bangpy.int32,"data_length",end_index - start_index +1 )#传进来得数据长度
    #     sub_value = self.bp.Scalar(bangpy.float32,"sub_value")#y-x的差值
    #     sum_value = self.bp.Scalar(bangpy.float32,"sum_value",buffer[start_index].astype(bangpy.float32))#
    #     with self.bp.for_range(0,data_length -1) as i:#这里 -1 是为了循环内省掉一个if
    #         sub_value.assign(sum_value - buffer [i + 1].astype(bangpy.float32))
    #         with self.bp.if_scope(tcp.all(sub_value <= max_threshold_valu,sub_value >= min_threshold_valu)):
    #             sum_value.assign(self.bp.scalar_pow(natural_base,sub_value)+const_one)
    #             sum_value.assign(self.bp.scalar_log(sum_value)/self.bp.scalar_log(natural_base))
    #             sum_value.assign(sum_value + buffer [i + 1])
    #         with self.bp.else_scope():
    #             with self.bp.if_scope(sub_value < min_threshold_valu):
    #                 sum_value.assign(buffer[i + 1])
    #     return sum_value
    def scalar_pow(self, value, p):
        self.nram_pow_buffer[0] = value
        self.bp.log(self.nram_pow_buffer, self.nram_pow_buffer)
        #pw = self.bp.Scalar(name='p', dtype=self.dtype, value=p)
        self.bp.multiply(self.nram_pow_buffer, self.nram_pow_buffer, p)
        self.bp.exp(self.nram_pow_buffer, self.nram_pow_buffer)
        return self.nram_pow_buffer[0]
    
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

        gram_paras = self.bp.Buffer(
            shape=(2, ), name="gram_paras", dtype=self.dtype, scope="global"
        )

        gram_buffer_out = self.bp.Buffer(
            shape=(self.output_len, ), name="gram_buffer_out", dtype=self.dtype, scope="global"
        )

        border_array_size = 128
        gram_border_buf_out = self.bp.Buffer(
            shape=(border_array_size * 2, ), name="gram_border_buf_out", dtype=self.dtype, scope="global"
        )
        gram_border_idx_out = self.bp.Buffer(
            shape=(border_array_size * 2, ), name="gram_border_idx_out", dtype=bangpy.int32, scope="global"
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
        with self.bp.else_scope(): #nram 虽然够了，但是要计算的数据量很小，以至于分摊到每个core上面的数据，还不够一个norm
            with self.bp.if_scope(self.len_tensor1 // self.task_num < self.pd_len):
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

                with self.bp.if_scope(index1 >= 0):
                    gram_buffer_out[index1] = gram_buffer_out[index1] + norm_value1

                with self.bp.if_scope(index2 >= 0):
                    gram_buffer_out[index2] = gram_buffer_out[index2] + norm_value2


        f = self.bp.BuildBANG(
            inputs=[gram_tensor1, gram_tensor2, gram_paras,
                    self.len_tensor1, self.len_tensor2,
                    self.pd_len, self.pd_height, self.pd_width,
                    self.output_len                    
                    ],
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

        pw = self.bp.Scalar(self.dtype, "pw", 1 / self.nram_pd_paras[0])

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
                    outputs[index] = self.scalar_pow(norm_value, pw) # 一个完整的norm算出来了 

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
        current_core_start = self._data_man._current_core_start
        total_count_in_core = self._data_man._total_count_in_core        
        dim_len = self.pd_len
        norm_value = self.bp.Scalar(self.dtype, "norm_value", 0.0)
        pw = self.bp.Scalar(self.dtype, "pw", 1 / self.nram_pd_paras[0])
        
        flat_nram = self.nram_calc_buffer.reshape([self.nram_process_count, ])

        # 1 先看看有没有上个norm残留的尾巴
        norm_offset = self.bp.Scalar(bangpy.int32, "norm_offset", current_core_start % dim_len)
        expect_cp_len = self.bp.Scalar(bangpy.int32, "expect_cp_len", 0)
        with self.bp.if_scope(norm_offset != 0):
            #有残留，拷贝过来
            expect_cp_len.assign(dim_len - norm_offset)
            self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, current_core_start, dim_len, self.pd_height, self.pd_width, expect_cp_len)
            calc_result = self.calc_norm(flat_nram, 0, expect_cp_len)
            norm_value.assign(calc_result)
            index = self.get_norm_index(current_core_start + expect_cp_len, dim_len)
            #保存一下
            border_outputs[self.bp.taskId * 2] = norm_value 
            idx_outputs[self.bp.taskId * 2] = index  

        #开始循环拷贝norm了，先计算开始位置
        norm_start_pos = self.bp.Scalar(bangpy.int32, "norm_start_pos", current_core_start + expect_cp_len)

        #计算一下一个nram里最多能存多少个
        nram_norm_count = self.bp.Scalar(bangpy.int32, "nram_norm_count", self.nram_process_count // dim_len)

        #计算一下，这个core能处理的norm总数是多少
        total_norm_in_core = self.bp.Scalar(bangpy.int32, "total_norm_in_core", (total_count_in_core - expect_cp_len) // dim_len)

        #计算一下，要多少个循环
        calc_loop_count = self.bp.Scalar(bangpy.int32, "calc_loop_count", (total_norm_in_core + nram_norm_count - 1) // nram_norm_count)

        with self.bp.for_range(0, calc_loop_count) as i:
            once_loop_start = self.bp.Scalar(bangpy.int32, "once_loop_start", norm_start_pos + nram_norm_count * dim_len * i)   
            with self.bp.if_scope(i == calc_loop_count - 1):
                nram_norm_count.assign(total_norm_in_core % nram_norm_count)

            #这里后续要优化，目前先弄个for循环吧
            start_index = self.bp.Scalar(bangpy.int32, "norm_offset", once_loop_start // dim_len) #肯定可以整除
            with self.bp.for_range(0, nram_norm_count) as j:
                #先拷贝过来
                self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, once_loop_start + j * dim_len, dim_len, self.pd_height, self.pd_width, dim_len)
                calc_result = self.calc_norm(flat_nram, 0, dim_len)
                norm_value.assign(calc_result)
                #outputs[start_index + j] = norm_value                
                outputs[start_index + j] = self.scalar_pow(norm_value, pw) # 一个完整的norm算出来了 

        #再看一下结尾，是不是要缓存下一个norm的前半截
        norm_loop_end_pos = self.bp.Scalar(bangpy.int32, "norm_loop_end_pos", norm_start_pos + total_norm_in_core * dim_len)
        with self.bp.if_scope(norm_loop_end_pos < total_count_in_core):
            #拷贝一下数据
            self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, norm_loop_end_pos, dim_len, self.pd_height, self.pd_width, total_count_in_core - norm_loop_end_pos)
            calc_result = self.calc_norm(flat_nram, 0, total_count_in_core - norm_loop_end_pos)
            norm_value.assign(calc_result)
            index = self.get_norm_index(norm_loop_end_pos + 1, dim_len) #加个1，表示跳到下一个了
            #保存一下
            border_outputs[self.bp.taskId * 2 + 1] = norm_value 
            idx_outputs[self.bp.taskId * 2 + 1] = index  

@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_pairwisedistance(dtype=None, target=None):
    task_num = 32
    f = PairwiseDistance(dtype, target, task_num).compute_body()
    return f



