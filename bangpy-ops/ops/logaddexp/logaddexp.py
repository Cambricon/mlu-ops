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
from bangpy.tcp.util import round_up, round_down
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float16, bangpy.float32] #支持的类型
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]#支持的设备
KERNEL_NAME = "LogAddExp" #算子名


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
        self.bp.launch_task(self.task_num, 1, 1)#将任务维度值设置为在此内核中启动。  三个参数其实就是 taskdimx,y,z   
    
    #将input中得数据和阈值比较，不在范围得用阈值替换
    #input 原始输入 将要替换得输入
    #nram_temp_bool_res 真值buffer，（应该在该函数内部创建，但考虑到可能出啥幺蛾子，所以直接在外部调用前就创建好）
    #is_min bool值 控制逻辑分支是使用>=还是<=,
    #threshold_value 要进行判断得阈值
    def replace_out_of_range_value_to_threshold_value(self,input,nram_temp_bool_res,is_min,threshold_value):
        #  with self.bp.if_scope(is_min):#通过将输入和阈值进行比较 将真值存入nram_temp_bool_res中 符合条件为1 不符为0
         if  is_min :   
              self.bp.greater_equal(nram_temp_bool_res,input,threshold_value,'elemwise') #
        #  with self.bp.else_scope():
         else :
              self.bp.less_equal(nram_temp_bool_res,input,threshold_value,'elemwise')
         self.bp.multiply(input,input,nram_temp_bool_res) #输入和真值相乘  可使输入中不符和条件得项变为0
         self.bp.logical_not(nram_temp_bool_res,nram_temp_bool_res) #对真值取反
         self.bp.multiply(nram_temp_bool_res,nram_temp_bool_res,threshold_value) #真值与阈值相乘  
         self.bp.add(input,input,nram_temp_bool_res) #为原始数据附加阈值

    def is_value_out_of_min_or_max_range_and_change_it(self,input,buffer_length):
        nram_temp_bool_res = self.bp.Buffer(
            shape=(buffer_length,),
            name="NTBR",
            dtype=self.dtype,
            scope="nram",
        )
        max_threshold_valu = self.bp.Scalar(self.dtype,"max_threshold_valu")
        min_threshold_valu = self.bp.Scalar(self.dtype,"min_threshold_valu")
        #with self.bp.if_scope(self.dtype == bangpy.float16): #当为16位是 max min 采用以下值
            # max.assign(11.089866488461016076210728979771)  #这些数我是网上查的该类型大于0时的最大最小值 然后取了个ln得到的
            # min.assign(-9.703981170988072538409566077448)
        if self.dtype == bangpy.float16 :
            max_threshold_valu.assign(11)
            min_threshold_valu.assign(-9)
        #with self.bp.else_scope():                         #32位时使用以下值
            # max.assign(88.722008965395851698332450562653)
            # min.assign(-87.332719095296162600686375692197)
        else:
            max_threshold_valu.assign(88)
            min_threshold_valu.assign(-87)
        self.replace_out_of_range_value_to_threshold_value(input,nram_temp_bool_res,1,min_threshold_valu)
        self.replace_out_of_range_value_to_threshold_value(input,nram_temp_bool_res,0,max_threshold_valu)
    def compute_body(self):
        # calculate split strategy
        # gets the data length to be calculated for each task
        one_core_count = self.bp.Scalar(bangpy.int32,"one_core_count")
        remain =  self.bp.Scalar(bangpy.int32,"remain")       
        current_core_start = self.bp.Scalar(bangpy.int32,"current_core_start") #当前核心数据开始索引
        current_core_end = self.bp.Scalar(bangpy.int32,"current_core_end") #当前核心数据结束索引
        total_count_in_core = self.bp.Scalar(bangpy.int32,"total_count_in_core")
        calc_loop_count = self.bp.Scalar(bangpy.int32,"calc_loop_count")
        once_loop_start = self.bp.Scalar(bangpy.int32,"once_loop_start")
        calc_size = self.bp.Scalar(bangpy.int32,"calc_size")
        nram_avable_size = round_down( (TARGET(self.target).nram_size - 200* 1024) // 4  ,128)#self.bp.Scalar(bangpy.int32,"nram_avable_size")
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
        buffer_in0 = self.bp.Buffer(
            shape=(self.length,), name="INPUT0", dtype=self.dtype, scope="global"
        )
        buffer_in1 = self.bp.Buffer(
            shape=(self.length,), name="INPUT1", dtype=self.dtype, scope="global"
        )
        buffer_out = self.bp.Buffer(
            shape=(self.length,), name="OUTPUT", dtype=self.dtype, scope="global"
        )
       
        nram_buffer_in0 = self.bp.Buffer(
            shape=(process_count,),
            name="INPUT0_N",
            dtype=self.dtype,
            scope="nram",
        ) 
        nram_buffer_in1 = self.bp.Buffer(
            shape=(process_count,),
            name="INPUT1_N",
            dtype=self.dtype,
            scope="nram",
        )
        const_one = self.bp.Scalar(dtype = self.dtype,name = "const_one",value = 1)
        calc_loop_count.assign((total_count_in_core + process_count - 1) // process_count)
        with self.bp.for_range(0, calc_loop_count) as i:            
            once_loop_start.assign(current_core_start + process_count * i) #当前核心数据开始的位置 + 第i次循环所应偏移的长度
            with self.bp.if_scope(i < calc_loop_count - 1):
                calc_size.assign(process_count)
            with self.bp.else_scope():
                calc_size.assign(total_count_in_core % process_count)

            with self.bp.block("data_copy"):
                self.bp.memcpy(nram_buffer_in0[0:calc_size], buffer_in0[once_loop_start:once_loop_start + calc_size]) 
                self.bp.memcpy(nram_buffer_in1[0:calc_size], buffer_in1[once_loop_start:once_loop_start + calc_size]) 
            self.bp.subtract(nram_buffer_in1, nram_buffer_in1, nram_buffer_in0)
            self.is_value_out_of_min_or_max_range_and_change_it(nram_buffer_in1,process_count)
            self.bp.exp(nram_buffer_in1, nram_buffer_in1)
            self.bp.add(nram_buffer_in1,nram_buffer_in1,const_one)
            self.bp.log(nram_buffer_in1, nram_buffer_in1)
            self.bp.add(nram_buffer_in0,nram_buffer_in0,nram_buffer_in1)
            self.bp.memcpy(buffer_out[once_loop_start:once_loop_start + calc_size], nram_buffer_in0[:calc_size])
            
      
        f = self.bp.BuildBANG(
            inputs=[buffer_in0, buffer_in1,],
            outputs=[buffer_out],
            kernel_name=KERNEL_NAME,
        )
        return f
@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_logaddexp(dtype=None, target=None):
    # tasktype fixed in UNION1    调度说明在这里  默认设置为union1 只启用了一个cluster
    task_type=TaskType.UNION16 #设置为UNION4  即当空闲4个cluster时 这玩意开始干活   union1指只要有一个cluster空闲时就可以干活了
    task_num =task_type.value*4 #这里可能是这么理解  一个cluster 4个核   根据union的类型乘4确定投入的core
    f = LogAddExp(dtype, target, task_num).compute_body()
    return f
