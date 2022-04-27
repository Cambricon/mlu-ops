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
#from tkinter import Scale
from multiprocessing import synchronize
import bangpy
from bangpy import tcp
from bangpy.platform.bang_config import TARGET
from bangpy.tcp.runtime import TaskType


DTYPES = [bangpy.float32]
TARGET_LIST = ["mlu290"]
KERNEL_NAME = "hard_sigmoid"


class Hard_sigmoid(object):
    """Operator description
       tensor-->activate function-->another tensor with same shape.
    """

    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.target = target
        self.cluster_num=task_num // 4
        self.task_num = task_num
        self.tcp = tcp.TCP(target)
       
        # how to dynamically pass in tensors of different shapes? 
        # now it is not supported
        self.dim_0 = self.tcp.SizeVar("dim_0")
        self.dim_1 = self.tcp.SizeVar("dim_1")
        self.dim_2 = self.tcp.SizeVar("dim_2")
        self.dim_3 = self.tcp.SizeVar("dim_3")
        self.dim_4 = self.tcp.SizeVar("dim_4")
        self.dim_5 = self.tcp.SizeVar("dim_5")
        self.dim_6 = self.tcp.SizeVar("dim_6")
        self.dim_7 = self.tcp.SizeVar("dim_7")
        
        self.nram_size = TARGET(target).nram_size
        self.dtype_sz = dtype.bytes
        
        # 3 buffer_n:   1.buffer_io_n*2      2.buffer_temp_n
        # buffer_io_n*2: double buffering
        # max:  512KB=524288/3=174762.666...(B)
        # note: 174080 to align (128bytes)
        self.sram_size_buffer=2088960
        self.nram_size_buffer=174080
        self.tcp.launch_cluster(TaskType.BLOCK)
        self.tcp.launch_task(self.task_num,1,1)

    def compute_body(self):
        
        # declare I/O buffer
        buffer_in = self.tcp.Buffer(
            shape=(self.dim_0,self.dim_1,self.dim_2,self.dim_3,self.dim_4,self.dim_5,self.dim_6,self.dim_7),
            name="INPUT", dtype=self.dtype, scope="global"
        )
        buffer_out = self.tcp.Buffer(
            shape=(self.dim_0,self.dim_1,self.dim_2,self.dim_3,self.dim_4,self.dim_5,self.dim_6,self.dim_7),
            name="OUTPUT", dtype=self.dtype, scope="global"
        )

        # Data Processing
        buffer_in=buffer_in.flatten() # Reducing tensor to one dimension to manipulate
        buffer_out=buffer_out.flatten() # keep the shape same with buffer_in
        
        cluster_id=self.tcp.clusterId
        core_id = self.tcp.coreId
        task_id=self.tcp.taskId
        
        # calculate split strategy
        data_all=self.tcp.Scalar(bangpy.int32,"data_all")
        data_all.assign(self.dim_0 * self.dim_1 * self.dim_2 * self.dim_3 * self.dim_4 * self.dim_5 * self.dim_6 * self.dim_7)

        data_each_task = data_all // self.task_num
        data_rem = data_all % self.task_num
        data_each_s = self.sram_size_buffer // self.dtype_sz
        data_each_time = self.nram_size_buffer // self.dtype_sz
        loop = data_each_task  // data_each_time
        data_rem_n = data_each_task  % data_each_time
        
        
        '''
        存储体系:GDRAM-->NRAM-->GDRAM
        这里NRAM“有效计算空间”较大,引入SRAM起不了缓存的作用,反而重复拷贝开销较大
        当NRAM“有效计算空间”远小于SRAM的空间时,SRAM缓存作用较好,并且此时SRAM双缓冲作用显著
        注:有效计算空间指每次计算需要从GDRAM拷入NRAM计算的数据所占空间,NRAM=需要拷入计算的数据所占空间+其它辅助计算数据所占空间
        data_all: self.dim_0*self.dim_1*...(数据总个数)
        self.task_num:任务个数
        data_each_task: 每个任务需要计算的数据个数
        data_rem: 平均分给所有IPU后的余数
        self.nram.size: NRAM大小(B) # 见__init__
        self.dtype_sz:每个数据类型所占字节数
        data_each_time: 每次NRAM计算的数据个数
        loop:每个task需要拷入NRAM进行计算的次数
        data_rem_n: 不足一次计算

        
        若引入SRAM作为缓存,则:
        存储体系:GDRAM-->SRAM-->NRAM-->SRAM-->GDRAM
        data_all: self.dim_0*self.dim_1*...(数据总个数)
        self.task_cluster:参与计算的簇的个数
        self.task_num:任务个数
        data_each_sram: 每个SRAM(簇)需要计算的数据个数
        self.sram.size: SRAM大小(B)
        self.dtype_sz:每个数据类型所占字节数
        data_each_sram_copy: 每次SRAM的copy数据个数
        loop1: SRAM完整copy次数
        data_rem_s: 不足一次copy
        data_each_task: 每个任务需要计算的数据个数
        self.nram.size: NRAM大小(B)
        self.dtype_sz:每个数据类型所占字节数
        data_each_time: 每次NRAM计算的数据个数
        loop2: NRAM完整compute次数
        data_rem_n: 不足一次计算
        
        其它:
        1.引入SRAM时注意尾部数据的处理,包括data_rem,data_rem_s,data_rem_n,其中data_rem_s特别注意
        2.引入SRAM时可特殊处理SRAM与NRAM容量,NRAM整除SRAM时,便没有了尾部数据
        3.WRAM作缓存类似SRAM,但效果要比SRAM好,因为它容量似乎较大(如果是1MB的话)且属于私有(不用sync)
        
        '''

        # declare it to use api of mul,add,maximum and minimum 
        buffer_temp_n = self.tcp.Buffer(
            shape=(data_each_time,),
            name="TEMP_N",
            dtype=self.dtype,
            scope="nram",
        )

        # declare it as cache
        buffer_out_s = self.tcp.Buffer(
            shape=(data_each_s,),
            name="OUTPUT_S",
            dtype=self.dtype,
            scope="sram",
        )

        st = self.tcp.Scalar(bangpy.int32,"st")
        with self.tcp.for_range(0, loop, stage=1) as i:
            start = task_id * data_each_task + i * data_each_time
            stop = start + data_each_time
            j =  i % 3
            begin = core_id * (data_each_s//4) + j * data_each_time
            end = begin + data_each_time
            buffer_io_n = self.tcp.Buffer(
                shape=(data_each_time,),
                name="IO_N",
                dtype=self.dtype,
                scope="nram",
            )
            with self.tcp.block("data_copy"):
                self.tcp.memcpy(buffer_io_n,buffer_in[start:stop])
            with self.tcp.block("compute"):
                self.tcp.assign(buffer_temp_n,1/6)
                self.tcp.multiply(buffer_io_n,buffer_io_n,buffer_temp_n)
                self.tcp.assign(buffer_temp_n,1/2)
                self.tcp.add(buffer_io_n,buffer_io_n,buffer_temp_n)
                self.tcp.assign(buffer_temp_n,1)
                self.tcp.minimum(buffer_io_n,buffer_io_n,buffer_temp_n)
                self.tcp.zeros(buffer_temp_n)
                self.tcp.maximum(buffer_io_n,buffer_io_n,buffer_temp_n)
            with self.tcp.block("data_copy"):
                self.tcp.memcpy(buffer_out_s[begin:end], buffer_io_n)
                with self.tcp.if_scope(j==2):
                    st.assign(start-2*data_each_time)
                    # with self.tcp.if_scope(tcp.all(task_id==0,i==2)):
                    #     self.tcp.print(buffer_out_s[begin:begin+data_each_time])
                    self.tcp.memcpy(buffer_out[st:stop],buffer_out_s[begin-2*data_each_time:end])
                    self.tcp.sync_cluster()
                    # with self.tcp.if_scope(tcp.all(task_id==0,i==2)):
                    #     self.tcp.print(buffer_out[st:st+data_each_time])
                with self.tcp.if_scope(tcp.all(i==(loop-1),j<2)):
                    st.assign(start-j*data_each_time)
                    self.tcp.memcpy(buffer_out[st:stop],buffer_out_s[begin-j*data_each_time:end])
       
        # if data_rem_n > 0
        with self.tcp.if_scope(data_rem_n > 0):
            start = task_id * data_each_task + loop * data_each_time
            stop = start + data_rem_n
            buffer_io_n = self.tcp.Buffer(
                shape=(data_each_time,),
                name="INPUT_N",
                dtype=self.dtype,
                scope="nram",
            )
            self.tcp.memcpy(buffer_io_n[0:data_rem_n],buffer_in[start:stop])
            self.tcp.assign(buffer_temp_n,1/6)
            self.tcp.multiply(buffer_io_n,buffer_io_n,buffer_temp_n)
            self.tcp.assign(buffer_temp_n,1/2)
            self.tcp.add(buffer_io_n,buffer_io_n,buffer_temp_n)
            self.tcp.assign(buffer_temp_n,1)
            self.tcp.minimum(buffer_io_n,buffer_io_n,buffer_temp_n)
            self.tcp.zeros(buffer_temp_n)
            self.tcp.maximum(buffer_io_n,buffer_io_n,buffer_temp_n)
            self.tcp.memcpy(buffer_out[start:stop], buffer_io_n[0:data_rem_n])
        
        # if data_rem > 0:
        # 1<=data_rem<=task_num-1,let master thread to adress it
        with self.tcp.if_scope(data_rem > 0):
            with self.tcp.if_scope(task_id==0):
                stop = data_all
                start = data_all - data_rem
                buffer_io_n = self.tcp.Buffer(
                    shape=(data_each_time,),
                    name="INPUT_N",
                    dtype=self.dtype,
                    scope="nram",
                )
                self.tcp.memcpy(buffer_io_n[0:data_rem],buffer_in[start:stop])
                self.tcp.assign(buffer_temp_n,1/6)
                self.tcp.multiply(buffer_io_n,buffer_io_n,buffer_temp_n)
                self.tcp.assign(buffer_temp_n,1/2)
                self.tcp.add(buffer_io_n,buffer_io_n,buffer_temp_n)
                self.tcp.assign(buffer_temp_n,1)
                self.tcp.minimum(buffer_io_n,buffer_io_n,buffer_temp_n)
                self.tcp.zeros(buffer_temp_n)
                self.tcp.maximum(buffer_io_n,buffer_io_n,buffer_temp_n)
                self.tcp.memcpy(buffer_out[start:stop], buffer_io_n[0:data_rem])
        
        # Data Processing
        # Recovering the shape of the tensor
        buffer_out.reshape((self.dim_0,self.dim_1,self.dim_2,self.dim_3,self.dim_4,self.dim_5,self.dim_6,self.dim_7))
        
        # build a executable module
        f = self.tcp.BuildBANG(
            inputs=[buffer_in],
            outputs=[buffer_out],
            kernel_name=KERNEL_NAME,
        )
        return f

@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_hard_sigmoid(dtype=None, target=None):
    # tasktype fixed in UNION16
    task_num=64
    f = Hard_sigmoid(dtype, target, task_num).compute_body()
    return f
