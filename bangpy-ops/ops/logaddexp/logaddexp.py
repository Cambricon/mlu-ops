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

DTYPES = [bangpy.float16, bangpy.float32] #支持的类型
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]#支持的设备
KERNEL_NAME = "logaddexp"#算子名


class LogAddExp(object):
    """Operator description:
    Add the data in the two buffers.
    """

    def __init__(self, dtype, target, task_num):#self 即this指针  dtype 传入的数据类型 target目标芯片的型号
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
       
        self.bp = tcp.TCP(target)
        self.length = self.bp.SizeVar("length")#得到数据的长度  此处应该是数组的长度
        self.nram_size = TARGET(target).nram_size#每个核的Nram大小 每个core自己的存储空间
        self.dtype_sz = dtype.bytes#类型占用空间的大小(字节)
        self.single_buffer_size =1024#每个buffer 占用的空间大小
        self.bp.launch_task(self.task_num, 1, 1)#将任务维度值设置为在此内核中启动。  三个参数其实就是 taskdimx,y,z   
    def compute_body(self):
        # calculate split strategy
        # gets the data length to be calculated for each task
        #因为要进行128字节对齐 这里向下取整 不足128字节的将会被舍弃掉不进行运算
        data_calculated_each_task = self.length // self.task_num #数据长度除以投入的核心数 向下取整 得出每个核心计算的长度
        insufficient_one_core_length = self.length % self.task_num #取余 确认是否有没参加运算的数据
        # gets the number of cycles required for each task
        loop_num = data_calculated_each_task * self.dtype_sz // self.single_buffer_size #每次计算的长度乘以数据类型大小得到数据所占字节数（每个核分到的） 除以核内单个缓冲区的大小 来计算出每个核需要循环计算的次数（向下取整）
        tail_size = (data_calculated_each_task* self.dtype_sz  % self.single_buffer_size) / self.dtype_sz#每个核 分到的数据长度 取余运算  确认是否有没有原酸的数据
        # gets the data length for each calculation
        data_calculated_each_time = self.single_buffer_size // self.dtype_sz   #核内单个缓冲区大小除以数据类型大小 得到一个核一次可以处理多长的数据
       
        # declare I/O buffer
        
      
       
        buffer_in0 = self.bp.Buffer(
            shape=(self.length,), name="INPUT0", dtype=self.dtype, scope="global"
        )
        buffer_in1 = self.bp.Buffer(
            shape=(self.length,), name="INPUT1", dtype=self.dtype, scope="global"
        )
        buffer_out = self.bp.Buffer(
            shape=(self.length,), name="OUTPUT", dtype=self.dtype, scope="global"
        )
        #self.bp.print("self.length",self.length)
        task_id = self.bp.taskId
        # declare on-chip buffer
        buffer_in0_n = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="INPUT0_N",
            dtype=self.dtype,
            scope="nram",
        ) 
        buffer_in1_n = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="INPUT1_N",
            dtype=self.dtype,
            scope="nram",
        )       
        natural_power_x=self.bp.Buffer(#e的x次方
            shape=(data_calculated_each_time,),
            name="NPX",
            dtype=self.dtype,
            scope="nram",
        )
        natural_power_y=self.bp.Buffer(#e的x次方
            shape=(data_calculated_each_time,),
            
            name="NPY",
            dtype=self.dtype,
            scope="nram",
        )
        antilogarithm=self.bp.Buffer(#真数 
            shape=(data_calculated_each_time,),
            name="ANTI",
            dtype=self.dtype,
            scope="nram",
        )
       
      
        out_res = self.bp.Buffer(#最终结果
            shape=(data_calculated_each_time,),
            name="OR",
            dtype=self.dtype,
            scope="nram",
        )
       
        
        with self.bp.for_range(0, loop_num) as i:   #正常循环       
            start = task_id * data_calculated_each_task + i * data_calculated_each_time
            stop = start + data_calculated_each_time
            self.bp.memcpy(buffer_in0_n, buffer_in0[start:stop]) 
            self.bp.memcpy(buffer_in1_n, buffer_in1[start:stop])         
            self.bp.exp(natural_power_x,buffer_in0_n,"hp")
            self.bp.exp(natural_power_y,buffer_in1_n,"hp")
            self.bp.add(antilogarithm, natural_power_x, natural_power_y)
            self.bp.log(out_res,antilogarithm)
            self.bp.memcpy(buffer_out[start:stop], out_res)
       
        with self.bp.if_scope(tail_size > 0):# 当每个核有不够一次循环计算长度的  尾部补0 至一次计算的长度 单独算一次
            self.bp.print("length",self.length)
            self.bp.print("tail_zize",tail_size)
            self.bp.zeros(buffer_in0_n)
            self.bp.zeros(buffer_in1_n)
            self.bp.zeros(natural_power_x)
            self.bp.zeros(natural_power_y)
            self.bp.zeros(antilogarithm)
            self.bp.zeros(out_res)
            start = (task_id)*data_calculated_each_task-loop_num*data_calculated_each_time
            stop =start+ tail_size
            self.bp.memcpy(buffer_in0_n[:tail_size], buffer_in0[start:stop]) 
            self.bp.memcpy(buffer_in1_n[:tail_size], buffer_in1[start:stop])  
            self.bp.exp(natural_power_x,buffer_in0_n,"hp")
            self.bp.exp(natural_power_y,buffer_in1_n,"hp") 
            self.bp.add(antilogarithm, natural_power_x, natural_power_y)
            self.bp.log(out_res,antilogarithm)
            self.bp.memcpy(buffer_out[start:stop], out_res[:tail_size])
        with self.bp.if_scope(insufficient_one_core_length>0):#当整体拆分任务至每个核时 如果有余  在这里计算  
            self.bp.zeros(buffer_in0_n)
            self.bp.zeros(buffer_in1_n)
            self.bp.zeros(natural_power_x)
            self.bp.zeros(natural_power_y)
            self.bp.zeros(antilogarithm)
            self.bp.zeros(out_res)
            start = self.length - insufficient_one_core_length
            stop = self.length
            self.bp.memcpy(buffer_in0_n[:insufficient_one_core_length], buffer_in0[start:stop]) 
            self.bp.memcpy(buffer_in1_n[:insufficient_one_core_length], buffer_in1[start:stop]) 
            self.bp.exp(natural_power_x,buffer_in0_n,"hp")
            self.bp.exp(natural_power_y,buffer_in1_n,"hp")
            self.bp.add(antilogarithm, natural_power_x, natural_power_y)
            self.bp.log(out_res,antilogarithm)
            self.bp.memcpy(buffer_out[start:stop], out_res[:insufficient_one_core_length])
        f = self.bp.BuildBANG(
            inputs=[buffer_in0,buffer_in1],
            outputs=[buffer_out],
            kernel_name=KERNEL_NAME,
        )
        return f
@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_logaddexp(dtype=None, target=None):
    # tasktype fixed in UNION1    调度说明在这里  默认设置为union1 只启用了一个cluster
    task_type=TaskType.UNION1  #设置为UNION4  即当空闲4个cluster时 这玩意开始干活   union1指只要有一个cluster空闲时就可以干活了
    task_num =task_type.value*4 #这里可能是这么理解  一个cluster 4个核   根据union的类型乘4确定投入的core
    f = LogAddExp(dtype, target, task_num).compute_body()
    return f
