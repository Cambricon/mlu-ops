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
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType
from bangpy.tcp.util import round_up, round_down
DTYPES = [bangpy.float32]
TARGET_LIST = ["mlu290"]
KERNEL_NAME = "exp"


class Exp(object):
    """Operator description:
    Add the data in the two buffers.
    """

    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.bp = tcp.TCP(target)
        self.length = self.bp.SizeVar("length")
        self.nram_size = TARGET(target).nram_size
        self.dtype_sz = dtype.bytes
        self.col_count = self.bp.Var("col_count")
        self.row_count = self.bp.Var("row_count")
        self.bp.launch_task(self.task_num, 1, 1)

    # buffer 源数据
    # start_index 起始索引
    # end_index 结束索引
    # 计算一维数组 start_index 至 end_index 范围内的和   并将结果写在start_index除
    def gala_sum_pool(self,buffer,start_index,end_index):
        data_length = self.bp.Scalar(bangpy.int32,"data_length",end_index - start_index +1 )#传进来得数据长度
        count_for_128_align =self.bp.Scalar(bangpy.int32,"count_for_128_align",128 // self.dtype_sz)#128字节是几个占几个索引
        remain = self.bp.Scalar(bangpy.int32,"remain",data_length % count_for_128_align)#128对齐后 不足对齐得数据个数
        current_end_index = self.bp.Scalar(bangpy.int32,"current_end_index",end_index - remain +1)#刨去不足后 剩余可以对齐长度得末尾索引   +1是因为python数组切片语法[a:b]会对b自动-1  这里图省事就直接加上
        #将末尾不能对齐的部分循环加到第一个元素上
        with self.bp.if_scope(remain != 0):
            with self.bp.if_scope(current_end_index != 0):        
                with self.bp.for_range(0,remain) as i:
                    buffer[start_index] = buffer[start_index] + buffer[current_end_index + i]
            with self.bp.else_scope(): 
                with self.bp.for_range(0,remain -1) as j:
                    buffer[start_index] = buffer[start_index] + buffer[current_end_index + j +1]
        data_length.assign(data_length - remain)#刨除不足部分 重新定义数据长度
        #当数据长度不足一次对齐时 不进行下面
        #当满足一次对齐时 对其直接进行sum 
        #1.每行128字节 
        #2.算出多少行 
        #3.reshape （行，128字节数据个数）
        #3.对其sumpool 因为之后每行第一个元素是需要的 所以最终结果直接在buffer[start_index]上         
        with self.bp.if_scope(data_length>=count_for_128_align):
              self.bp.print(buffer[0:64])
              self.bp.sum(buffer[start_index:current_end_index],buffer[start_index:current_end_index]) 
              #self.bp.print("sumpool->",buffer[start_index])
              row = self.bp.Scalar(bangpy.int32,"row",data_length/count_for_128_align)
              reshape_buffer = buffer[start_index:current_end_index].reshape([row,count_for_128_align])
              self.bp.sumpool(reshape_buffer,reshape_buffer,(row,),(1,))
              # self.bp.print("sumpool->",buffer[start_index])
   

    def two_dimension_row_sum(self,buffer,row_count,col_count):#按行 计算二维数组每行的和 结果放在每行首位
        with self.bp.for_range(0,row_count) as i:
            self.gala_sum_pool(buffer[i][:],0,col_count-1)
            self.bp.print("buffer[",i,"][0]->",buffer[i][0])


    #buffer 源数据
    #temp_buffer 与buffer所占内存空间大小相等的nram存储
    #row_count 行数
    #col_count 列数
    #该函数将计算传入的行数 0 - row_count   列数0 - col_count的这个矩形范围内每列的和 并将结果写在源数据的首行
    def two_dimension_col_sum(self,buffer,temp_buffer,row_count,col_count):
        count_for_128_align =self.bp.Scalar(bangpy.int32,"count_for_128_align",128 // self.dtype_sz) # 当前数据类型下 128个字节 对应了多少个元素
        col_remain = self.bp.Scalar(bangpy.int32,"col_remain",col_count % count_for_128_align)
        current_col_count = self.bp.Scalar(bangpy.int32,"current_col_count",col_count - col_remain)
        with self.bp.if_scope(col_remain != 0):
            with self.bp.for_range(0,col_remain) as i:
                current_col_index = self.bp.Scalar(bangpy.int32,"current_col_index",col_count - i -1)
                with self.bp.for_range(0,row_count - 1) as j:
                    buffer[0][current_col_index] = buffer[0][current_col_index] + buffer[j + 1][current_col_index]
        with self.bp.if_scope(col_count >= count_for_128_align):
            reshape_buffer = temp_buffer.reshape([row_count,current_col_count])
            #self.bp.print("data_before_calc->",buffer[0])
            self.bp.memcpy(reshape_buffer[:,:],buffer[:,0:current_col_count])
            self.bp.sumpool(reshape_buffer,reshape_buffer,(row_count,),(1,))
            #self.bp.print("temp_after_calc->",reshape_buffer[0])
            self.bp.memcpy(buffer[0][0:current_col_count],reshape_buffer[0][0:current_col_count])
            #self.bp.print("data_res->",buffer[0])


  


   
    def compute_body(self):
        one_core_count = self.bp.Scalar(bangpy.int32,"one_core_count")

        remain =  self.bp.Scalar(bangpy.int32,"remain")       
        current_core_start = self.bp.Scalar(bangpy.int32,"current_core_start") #当前核心数据开始索引
        current_core_end = self.bp.Scalar(bangpy.int32,"current_core_end") #当前核心数据结束索引
        total_count_in_core = self.bp.Scalar(bangpy.int32,"total_count_in_core")
        calc_loop_count = self.bp.Scalar(bangpy.int32,"calc_loop_count")
        once_loop_start = self.bp.Scalar(bangpy.int32,"once_loop_start")
        calc_size = self.bp.Scalar(bangpy.int32,"calc_size")
        nram_avable_size = round_down( (TARGET(self.target).nram_size - 30* 1024) // 2 ,128)#self.bp.Scalar(bangpy.int32,"nram_avable_size")
        one_core_count.assign(self.length // self.task_num)#每个核均摊计算量（按索引分）
        remain.assign(self.length % self.task_num)#分任务时的余数
        
             
        process_count = nram_avable_size // self.dtype_sz #核心一次最多计算的长度
      
        with self.bp.if_scope(self.bp.taskId < remain): #如果存在余数 将其均摊给各核   taskId从0起
            current_core_start.assign((one_core_count + 1) * self.bp.taskId )
            current_core_end.assign((one_core_count + 1) * (self.bp.taskId + 1) - 1) #此处应该不需要减1 待验证  python切片会自动将上标减1
        with self.bp.else_scope():
            current_core_start.assign((one_core_count + 1) * remain + one_core_count * (self.bp.taskId - remain))
            current_core_end.assign((one_core_count + 1) * remain + one_core_count * (self.bp.taskId - remain) + one_core_count - 1)  
        total_count_in_core.assign(current_core_end - current_core_start + 1)
        # buffer_in0 = self.bp.Buffer(
        #     shape=(self.length,), name="INPUT0", dtype=self.dtype, scope="global"
        # )
        buffer_in0 = self.bp.Buffer(
            shape=(self.length,), name="INPUT0", dtype=self.dtype, scope="global"
        )
        buffer_out = self.bp.Buffer(
            shape=(self.length,), name="OUTPUT", dtype=self.dtype, scope="global"
        )
        nram_buffer_in0 = self.bp.Buffer(
            shape=(process_count,),
            name="GALA_IN",
            dtype=self.dtype,
            scope="nram",
        )





        test_buffer = self.bp.Buffer(
            shape=(process_count,),
            name="test_buffer",
            dtype=self.dtype,
            scope="nram",
        )
       
       
       


        

        calc_loop_count.assign((total_count_in_core + process_count - 1) // process_count)

       
       

        with self.bp.for_range(0, calc_loop_count) as i:            
            once_loop_start.assign(current_core_start + process_count * i) #当前核心数据开始的位置 + 第i次循环所应偏移的长度
            with self.bp.if_scope(i < calc_loop_count - 1):
                calc_size.assign(process_count)
            with self.bp.else_scope():
                calc_size.assign(total_count_in_core % process_count)
            with self.bp.block("data_copy"):
                self.bp.memcpy(nram_buffer_in0[0:calc_size], buffer_in0[once_loop_start:once_loop_start + calc_size])   
            self.bp.print("calc_size-->",calc_size)
            #self.gala_sum_pool(nram_buffer_in0,0,calc_size -1)
            row_count = self.bp.Scalar(dtype = bangpy.int32,name = "row_count",value = self.row_count)
            col_count = self.bp.Scalar(dtype = bangpy.int32,name = "col_count",value = self.col_count)
            reshape_buffer = nram_buffer_in0[0:calc_size].reshape([row_count,col_count])# (33,33)
           

            #二维数组按列求和 
            #此处需注意 第二个buffer参数上标索引越界的问题 此处只是展示 并未进行处理  实际使用时应以每次传入函数数据的实际尺寸计算
            self.two_dimension_col_sum(reshape_buffer,test_buffer[0:row_count*col_count],row_count,col_count)

           

            self.bp.memcpy(buffer_out[once_loop_start:once_loop_start + calc_size], nram_buffer_in0[:calc_size])

        # build a executable module
        f = self.bp.BuildBANG(
            inputs=[buffer_in0,self.row_count,self.col_count,],
            outputs=[buffer_out],
            kernel_name=KERNEL_NAME,
        )
        return f


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_exp(dtype=None, target=None):
    # tasktype fixed in UNION1
    task_num = 1 #由4 改为64
    f = Exp(dtype, target, task_num).compute_body()
    return f

    


