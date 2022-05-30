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
from bangpy.platform.bang_config import TARGET
from bangpy.tcp.util import round_down
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float32] #支持的类型
TARGET_LIST = ["mlu290"]#支持的设备
KERNEL_NAME = "Celu" #算子名


class Celu:
    """Operator description:
    Add the data in the two buffers.
    """

    def __init__(self, dtype, target, task_num):#self 即this指针  dtype 传入的数据类型 target目标芯片的型号
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.bp = tcp.TCP(target)
        #alpha = self.bp.SizeVar("alpha", self.dtype)
        self.inplace = self.bp.Var("inplace")
        self.length = self.bp.SizeVar("length")#得到数据的长度  此处应该是数组的长度
        self.dtype_sz = dtype.bytes#类型占用空间的大小(字节)
        self.bp.launch_task(self.task_num, 1, 1)#将任务维度值设置为在此内核中启动。  三个参数其实就是 taskdimx,y,z
    def compute_body(self):
        one_core_count = self.bp.Scalar(bangpy.int32,"one_core_count",self.length // self.task_num)
        remain =  self.bp.Scalar(bangpy.int32,"remain",self.length % self.task_num)
        current_core_start = self.bp.Scalar(bangpy.int32,"current_core_start") #当前核心数据开始索引
        current_core_end = self.bp.Scalar(bangpy.int32,"current_core_end") #当前核心数据结束索引
        calc_loop_count = self.bp.Scalar(bangpy.int32,"calc_loop_count")
        once_loop_start = self.bp.Scalar(bangpy.int32,"once_loop_start")
        calc_size = self.bp.Scalar(bangpy.int32,"calc_size")
        nram_avable_size = round_down( (TARGET(self.target).nram_size - 40* 1024) // 4  ,128)
        process_count = nram_avable_size // self.dtype_sz #核心一次最多计算的长度
        with self.bp.if_scope(self.bp.taskId < remain): #如果存在余数 将其均摊给各核   taskId从0起
            current_core_start.assign((one_core_count + 1) * self.bp.taskId )
            current_core_end.assign((one_core_count + 1) * (self.bp.taskId + 1) - 1)
        with self.bp.else_scope():
            current_core_start.assign(
                (one_core_count + 1) * remain + one_core_count * (self.bp.taskId - remain)
                )
            current_core_end.assign(current_core_start  + one_core_count - 1)
        total_count_in_core = self.bp.Scalar(
            bangpy.int32,
            "total_count_in_core",
            current_core_end - current_core_start + 1)
        buffer_in0 = self.bp.Buffer(
            shape=(self.length,), name="INPUT0", dtype=self.dtype, scope="global"
        )
        buffer_alpha = self.bp.Buffer(
            shape=(1,), name="ALPHA_PARAM", dtype=self.dtype, scope="global"
        )
        buffer_out = self.bp.Buffer(
            shape=(self.length,), name="OUTPUT", dtype=self.dtype, scope="global"
        )
        alpha = self.bp.Scalar(dtype = self.dtype,name = "alpha")
        alpha.assign(buffer_alpha[0])
        nram_buffer_in0 = self.bp.Buffer(
            shape=(process_count,),
            name="INPUT0_N",
            dtype=self.dtype,
            scope="nram",
        )
        nram_middle_value = self.bp.Buffer(
            shape=(process_count,),
            name="N_MAX",
            dtype=self.dtype,
            scope="nram",
        )
        nram_max = self.bp.Buffer(
            shape=(process_count,),
            name="N_MAX",
            dtype=self.dtype,
            scope="nram",
        )
        nram_min = self.bp.Buffer(
            shape=(process_count,),
            name="N_MIN",
            dtype=self.dtype,
            scope="nram",
        )
        const_zero = self.bp.Scalar(dtype = self.dtype,name = "const_zero",value = 0)
        const_one = self.bp.Scalar(dtype = self.dtype,name = "const_one",value = 1)
        calc_loop_count.assign((total_count_in_core + process_count - 1) // process_count)
        with self.bp.for_range(0, calc_loop_count) as i:
            #当前核心数据开始的位置 + 第i次循环所应偏移的长度
            once_loop_start.assign(current_core_start + process_count * i)
            with self.bp.if_scope(i < calc_loop_count - 1):
                calc_size.assign(process_count)
            with self.bp.else_scope():
                calc_size.assign(total_count_in_core % process_count)
                with self.bp.if_scope(calc_size == 0):
                    calc_size.assign(process_count)
            with self.bp.block("data_copy"):
                self.bp.memcpy(
                    nram_buffer_in0[0:calc_size],
                    buffer_in0[once_loop_start:once_loop_start + calc_size]
                    )
            #这里开始计算min
            with self.bp.if_scope(alpha != 0):
                self.bp.divide(nram_middle_value,nram_buffer_in0,alpha)#获得x/a
                self.bp.exp(nram_middle_value,nram_middle_value)#计算exp(x/a)
                self.bp.subtract(nram_middle_value, nram_middle_value, const_one)#-1
                self.bp.multiply(nram_middle_value, nram_middle_value, alpha)#*a
                self.bp.minimum(nram_min,nram_middle_value,const_zero)#min(0,...)
            with self.bp.else_scope():#当alpha为0时  min全为0
                self.bp.zeros(nram_min)
            #这里开始计算max
            self.bp.maximum(nram_max,nram_buffer_in0,const_zero)
            self.bp.add(nram_buffer_in0,nram_max,nram_min)
            self.bp.memcpy(
                buffer_out[once_loop_start:once_loop_start + calc_size],
                nram_buffer_in0[:calc_size])
        f = self.bp.BuildBANG(
            inputs=[buffer_in0,buffer_alpha,self.inplace],
            outputs=[buffer_out],
            kernel_name=KERNEL_NAME,)
        return f
@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_celu(dtype=None, target=None):
    task_type=TaskType.UNION16
    task_num =task_type.value*4 #这里可能是这么理解  一个cluster 4个核   根据union的类型乘4确定投入的core
    f = Celu(dtype, target, task_num).compute_body()
    return f
