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
KERNEL_NAME = "sinh"#算子名


class Sinh(object):
    """Operator description:
    Add the data in the two buffers.
    """

    def __init__(self, dtype, target, task_num):#self 即this指针  dtype 传入的数据类型 target目标芯片的型号
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
       
        self.bp = tcp.TCP(target)#开发API的TCP制作功能  应该是连接设备？？
        self.length = self.bp.SizeVar("length")#创建一个新的TCP变量表示一个缓冲区形状大小，它是非负的
        self.nram_size = TARGET(target).nram_size#每个核的Nram大小（nram一般用来存储标量或张量数据）
        self.dtype_sz = dtype.bytes#类型占用空间的大小
        self.single_buffer_size =1024
        self.bp.launch_task(self.task_num, 1, 1)#将任务维度值设置为在此内核中启动。  三个参数其实就是 taskdimx,y,z   
    def compute_body(self):
        # calculate split strategy
        # gets the data length to be calculated for each task
        data_calculated_each_task = self.length // self.task_num
        # gets the number of cycles required for each task
        loop_num = data_calculated_each_task * self.dtype_sz // self.single_buffer_size
        # gets the data length for each calculation
        data_calculated_each_time = self.single_buffer_size // self.dtype_sz   
        # self.bp.print("data_calculated_each_time")
        # self.bp.print(data_calculated_each_time) 
        # declare I/O buffer
        buffer_in0 = self.bp.Buffer(
            shape=(self.length,), name="INPUT0", dtype=self.dtype, scope="global"
        )
        # buffer_in1 = self.bp.Buffer(
        #     shape=(self.length,), name="INPUT1", dtype=self.dtype, scope="global"
        # )
        buffer_out = self.bp.Buffer(
            shape=(self.length,), name="OUTPUT", dtype=self.dtype, scope="global"
        )
        
        task_id = self.bp.taskId
        # declare on-chip buffer
        buffer_in0_n = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="INPUT0_N",
            dtype=self.dtype,
            scope="nram",
        )     
        numerator_res=self.bp.Buffer(#分子部分计算结果
            shape=(data_calculated_each_time,),
            name="NS",
            dtype=self.dtype,
            scope="nram",
        )
        #e的x次方的倒数
        natural_power_exponent_one_cent=self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="NPEOC",
            dtype=self.dtype,
            scope="nram",
        )
        natural_exponential_res = self.bp.Buffer(#e的x次方
            shape=(data_calculated_each_time,),
            name="NER",
            dtype=self.dtype,
            scope="nram",
        )
      
        sinh_res = self.bp.Buffer(#最终结果
            shape=(data_calculated_each_time,),
            name="SR",
            dtype=self.dtype,
            scope="nram",
        )
       
        denominator=self.bp.Scalar(name='denominator', dtype=self.dtype, value=0.5)
        with self.bp.for_range(0, loop_num) as i:          
            start = task_id * data_calculated_each_task + i * data_calculated_each_time
            stop = start + data_calculated_each_time
            self.bp.memcpy(buffer_in0_n, buffer_in0[start:stop])         
            self.bp.exp(natural_exponential_res,buffer_in0_n,"hp")#计算e的x次方                         
            self.bp.reciprocal(natural_power_exponent_one_cent,natural_exponential_res,"hp")                              
            self.bp.subtract(numerator_res,natural_exponential_res,natural_power_exponent_one_cent)#分子部分相减  
            self.bp.multiply(sinh_res,numerator_res,denominator)#除2 即乘以0.5         
            self.bp.memcpy(buffer_out[start:stop], sinh_res)#拷贝到输出
        f = self.bp.BuildBANG(
            inputs=[buffer_in0],
            outputs=[buffer_out],
            kernel_name=KERNEL_NAME,
        )
        return f
@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_sinh(dtype=None, target=None):
    # tasktype fixed in UNION1    调度说明在这里  默认设置为union1 只启用了一个cluster
    task_num = 4 #这里可能是这么理解  一个cluster 4个核  因为默认是union1  只启用了一个cluster 那么这里顶多设到4  
    f = Sinh(dtype, target, task_num).compute_body()
    return f
