# Copyright (C) [2022] by Cambricon, Inc.
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
import bangpy
from bangpy import tcp
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu290"]
KERNEL_NAME = "cross"


class Cross(object):
    """Operator description:
    torch.cross
    torch.cross(input, other, dim, output)
    计算张量input和other在维度dim上的三维叉乘，结果赋给output
    详见设计文档
    """

    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.tcp = tcp.TCP(target)
        self.dim = self.tcp.Var("dim", dtype=bangpy.int32)
        self.dim0 = self.tcp.SizeVar("dim0")
        self.dim1 = self.tcp.SizeVar("dim1")
        self.dim2 = self.tcp.SizeVar("dim2")
        self.dim3 = self.tcp.SizeVar("dim3")
        self.dim4 = self.tcp.SizeVar("dim4")
        self.dim5 = self.tcp.SizeVar("dim5")
        self.dim6 = self.tcp.SizeVar("dim6")
        self.dim7 = self.tcp.SizeVar("dim7")
        self.dimlength = self.tcp.SizeVar("dimlength")
        self.dtype_sz = dtype.bytes
        # self.nram_size = 512*1024byte,要开18个buffer,乘法和减法要128byte对齐
        # 流水线需要a0,a1,a2,b0,b1,b2,c0,c1,c2实现(a0,a1,a2)x(b0,b1,b2)=(c0,c1,c2)
        # 一轮需要9个buffer，两级流水需要18个buffer
        # 128*18=2304,512*1024//2304*2304=523008(512*1024=524288),
        # 523008/18=29056,29056可以被128整除，当然也可以被4(byte)和2(byte)整除
        self.single_buffer_size = 29056
        self.tcp.launch_task(self.task_num, 1, 1)
        self.tcp.launch_cluster(TaskType.BLOCK)

    def compute_body(self):
        shape = self.tcp.Buffer(shape = (self.dimlength, ),
                name="shape", dtype=bangpy.int32, scope="global")
        #如果直接传元组shape，不支持循环中的shape[i]，i必须是python int/float而循环的i是bangpy类型

        mydim=self.tcp.Scalar(bangpy.int32, "mydim")

        #防呆检测，需要控制流支持return
        with self.tcp.if_scope(tcp.all(self.dim < -8, self.dim > 7)):
            self.tcp.print("dim shall be in [-8,7], but not")
            #return

        with self.tcp.elif_scope(tcp.all(self.dim < 0, self.dim >= -8)):
            mydim.assign(self.dim + 8)
            with self.tcp.if_scope(shape[mydim]!=3):
                self.tcp.print("shape[dim] is not 3!")
                #return
            mydim.assign(mydim + 1)
        with self.tcp.else_scope():
            mydim.assign(self.dim)
            with self.tcp.if_scope(shape[mydim]!=3):
                self.tcp.print("shape[dim] is not 3!")
                #return
            mydim.assign(mydim + 1)

        maxdim=self.tcp.Scalar(bangpy.int32, "maxdim")
        maxdim.assign(8)
        #如果用self.dimlength也是可以的，但还是必须写死有几个参数，因为下面的buffer的shape必须用SizeVar定义

       #例如shape=(1,1,1,1,2,3,4,5),dim=5指向维度为3的位置，设dim之前的维度乘积是group,dim之后的维度乘积是step,均不包括dim
        length=self.tcp.Scalar(bangpy.int32, "length")
        length.assign(1)
        step=self.tcp.Scalar(bangpy.int32, "step")
        step.assign(1)
        with self.tcp.for_range(0, maxdim) as i:
            length.assign(length * shape[i])
        #总共1x1x1x1x2x3x4x5=120个元素

        with self.tcp.for_range(mydim, maxdim) as i:
            step.assign(step * shape[i])
        #隔step=4x5=20个是下一个要取元素

        group = self.tcp.Scalar(bangpy.int32,"group")
        rest = self.tcp.Scalar(bangpy.int32,"rest")

        #有1x1x1x1x1x2组
        group.assign(length / 3 / step)

        # calculate split strategy
        # declare I/O buffer
        buffer_in0 = self.tcp.Buffer( shape = (self.dim0, self.dim1, self.dim2,
            self.dim3,self.dim4, self.dim5, self.dim6, self.dim7),
            name="INPUT0", dtype=self.dtype, scope="global")
        buffer_in1 = self.tcp.Buffer( shape = (self.dim0, self.dim1, self.dim2,
            self.dim3, self.dim4, self.dim5, self.dim6, self.dim7),
            name="INPUT1", dtype=self.dtype, scope="global")
        buffer_out = self.tcp.Buffer( shape = (self.dim0, self.dim1, self.dim2,
            self.dim3,self.dim4, self.dim5, self.dim6, self.dim7),
            name="OUTPUT", dtype=self.dtype, scope="global")
        task_id = self.tcp.taskId

        start = self.tcp.Scalar(bangpy.int32, name="start")
        stop = self.tcp.Scalar(bangpy.int32, name="stop")
        start_ping = self.tcp.Scalar(bangpy.int32, name="start_ping")
        start_pong = self.tcp.Scalar(bangpy.int32, name="start_pong")
        group_each_task  = self.tcp.Scalar(bangpy.int32, name="group_each_task")
        threshold = self.tcp.Scalar(bangpy.int32, name="threshold")
        loop_num = self.tcp.Scalar(bangpy.int32, name="loop_num")
        leap1 =self.tcp.Scalar(bangpy.int32, name="leap1")
        leap2 =self.tcp.Scalar(bangpy.int32, name="leap2")
        leap1_ping =self.tcp.Scalar(bangpy.int32, name="leap1_ping")
        leap1_pong =self.tcp.Scalar(bangpy.int32, name="leap2_ping")
        leap2_ping =self.tcp.Scalar(bangpy.int32, name="leap1_pong")
        leap2_pong =self.tcp.Scalar(bangpy.int32, name="leap2_pong")
        last_loop = self.tcp.Scalar(bangpy.int32, name="last_loop")
        flag_ping = self.tcp.Scalar(bangpy.int32, name="flag_ping")
        flag_pong = self.tcp.Scalar(bangpy.int32, name="flag_pong")
        data_calculated_each_time = self.tcp.Scalar(bangpy.int32, name="data_calculated_each_time")
        data_calculated_each_time_last = self.tcp.Scalar(bangpy.int32,
            name="data_calculated_each_time_last")

        data_each_buffer = self.single_buffer_size // self.dtype_sz
        last_loop.assign(0)

        with self.tcp.if_scope(step  <= data_each_buffer):
            step_each_time = data_each_buffer//step
            #step<= data_each_buffer，以step为单位每次拷贝尽可能多的step做叉乘

            data_calculated_each_time = step_each_time * step

            # split and compute

            buffer_in0 = buffer_in0.reshape((group*3,step))
            buffer_in1 = buffer_in1.reshape((group*3,step))
            buffer_out = buffer_out.reshape((group*3,step))

            #计算每个task分到的group数量，余数从task_id=0开始往后各分一个直到分完
            #例如7个group分给3个task，那么需要计算的group_each_task分别是(3,2,2)
            group_each_task.assign(group // self.task_num)
            rest.assign(group % self.task_num)
            with self.tcp.if_scope(task_id < rest):
                group_each_task.assign(group_each_task + 1)
                start.assign(group_each_task * task_id * 3)
                # stop:当前task中buffer索引允许的最大值，不能扩张到下个task的计算范围里,也就是说等于(task_id+1)的start
                stop.assign(group_each_task * (task_id + 1) * 3)
            with self.tcp.else_scope():
                # start.assign(((group_each_task+1)*task_id-(task_id-rest))*3)
                start.assign( 3 * (group_each_task * task_id + rest))
                stop.assign(3 * (group_each_task * (task_id+1) + rest))
            # start.assign(start-3*step_each_time)
            stop.assign(stop-2)

            #余数额外计算一轮，也就是用到stop的地方
            loop_num.assign(group_each_task//step_each_time)
            with self.tcp.if_scope(group_each_task%step_each_time!=0):
                loop_num.assign(loop_num+1)
                last_loop.assign(1)
                #表示有余数，要单独处理最后一轮
                data_calculated_each_time_last.assign(
                    ((stop - start - (loop_num-1)*3*step_each_time)//3) * step)
                with self.tcp.if_scope((stop - start - (loop_num-1)*3*step_each_time)%3 != 0):
                    data_calculated_each_time_last.assign(data_calculated_each_time_last+step)

            with self.tcp.for_range(0, loop_num, stage = 1) as i:
                # declare on-chip buffer
                buffer_a0 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_a0",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_a1 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_a1",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_a2 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_a2",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_b0 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_b0",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_b1 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_b1",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_b2 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_b2",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_c0 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_c0",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_c1 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_c1",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_c2 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_c2",
                    dtype=self.dtype,
                    scope="nram",
                )
                #(a0,a1,a2)x(b0,b1,b2)=(c0,c1,c2)
                with self.tcp.block("data_copy"):
                    with self.tcp.if_scope(tcp.all(i == loop_num-1, last_loop==1)):

                        self.tcp.memcpy(buffer_a0[0:data_calculated_each_time_last]
                            .reshape((data_calculated_each_time_last/step,step)),
                            buffer_in0[(start+i*3*step_each_time).get(): stop.get() : 3])

                        self.tcp.memcpy(buffer_a1[0:data_calculated_each_time_last]
                            .reshape((data_calculated_each_time_last/step,step)),
                            buffer_in0[(start+i*3*step_each_time).get()+1 : stop.get()+1 : 3])

                        self.tcp.memcpy(buffer_a2[0:data_calculated_each_time_last]
                            .reshape((data_calculated_each_time_last/step,step)),
                            buffer_in0[(start+i*3*step_each_time).get()+2 : stop.get()+2 : 3])

                        self.tcp.memcpy(buffer_b0[0:data_calculated_each_time_last]
                            .reshape((data_calculated_each_time_last/step,step)),
                            buffer_in1[(start+i*3*step_each_time).get() : stop.get() : 3])

                        self.tcp.memcpy(buffer_b1[0:data_calculated_each_time_last]
                            .reshape((data_calculated_each_time_last/step,step)),
                            buffer_in1[(start+i*3*step_each_time).get()+1 : stop.get()+1 : 3])

                        self.tcp.memcpy(buffer_b2[0:data_calculated_each_time_last]
                            .reshape((data_calculated_each_time_last/step,step)),
                            buffer_in1[(start+i*3*step_each_time).get()+2 : stop.get()+2 : 3])

                    with self.tcp.else_scope():

                        self.tcp.memcpy(buffer_a0[0:data_calculated_each_time]
                            .reshape((data_calculated_each_time/step,step)),
                            buffer_in0[(start+i*3*step_each_time).get():
                            (start+(i+1)*3*step_each_time).get() : 3])

                        self.tcp.memcpy(buffer_a1[0:data_calculated_each_time]
                            .reshape((data_calculated_each_time/step,step)),
                            buffer_in0[(start+i*3*step_each_time).get()+1:
                            (start+(i+1)*3*step_each_time).get()+1 : 3])

                        self.tcp.memcpy(buffer_a2[0:data_calculated_each_time]
                            .reshape((data_calculated_each_time/step,step)),
                            buffer_in0[(start+i*3*step_each_time).get()+2 :
                            (start+(i+1)*3*step_each_time).get()+2 : 3])

                        self.tcp.memcpy(buffer_b0[0:data_calculated_each_time]
                            .reshape((data_calculated_each_time/step,step)),
                            buffer_in1[(start+i*3*step_each_time).get() :
                            (start+(i+1)*3*step_each_time).get() : 3])

                        self.tcp.memcpy(buffer_b1[0:data_calculated_each_time]
                            .reshape((data_calculated_each_time/step,step)),
                            buffer_in1[(start+i*3*step_each_time).get()+1 :
                            (start+(i+1)*3*step_each_time).get()+1 : 3])

                        self.tcp.memcpy(buffer_b2[0:data_calculated_each_time]
                            .reshape((data_calculated_each_time/step,step)),
                            buffer_in1[(start+i*3*step_each_time).get()+2 :
                            (start+(i+1)*3*step_each_time).get()+2 : 3])

                with self.tcp.block("compute"):
                    self.tcp.multiply(buffer_c0,buffer_a1,buffer_b2)
                    self.tcp.multiply(buffer_c1,buffer_a2,buffer_b1)
                    self.tcp.subtract(buffer_c0,buffer_c0,buffer_c1)

                    self.tcp.multiply(buffer_c1,buffer_a2,buffer_b0)
                    self.tcp.multiply(buffer_c2,buffer_a0,buffer_b2)
                    self.tcp.subtract(buffer_c1,buffer_c1,buffer_c2)

                    self.tcp.multiply(buffer_c2,buffer_a0,buffer_b1)
                    self.tcp.multiply(buffer_a0,buffer_a1,buffer_b0)
                    self.tcp.subtract(buffer_c2,buffer_c2,buffer_a0)

                with self.tcp.block("data_copy"):
                    with self.tcp.if_scope(tcp.all(i == loop_num-1, last_loop==1)):

                        self.tcp.memcpy(buffer_out
                            [(start+i*3*step_each_time).get():stop.get():3],
                            buffer_c0[0:data_calculated_each_time_last]
                            .reshape((data_calculated_each_time_last/step,step)))

                        self.tcp.memcpy(buffer_out
                            [(start+i*3*step_each_time).get()+1:stop.get()+1:3],
                            buffer_c1[0:data_calculated_each_time_last]
                            .reshape((data_calculated_each_time_last/step,step)))

                        self.tcp.memcpy(buffer_out
                            [(start+i*3*step_each_time).get()+2:stop.get()+2:3],
                            buffer_c2[0:data_calculated_each_time_last]
                            .reshape((data_calculated_each_time_last/step,step)))

                    with self.tcp.else_scope():

                        self.tcp.memcpy(buffer_out
                            [(start+i*3*step_each_time).get():
                            (start+(i+1)*3*step_each_time).get():3],
                            buffer_c0[0:data_calculated_each_time]
                            .reshape((data_calculated_each_time/step,step)))

                        self.tcp.memcpy(buffer_out
                            [(start+i*3*step_each_time).get()+1:
                            (start+(i+1)*3*step_each_time).get()+1:3],
                            buffer_c1[0:data_calculated_each_time]
                            .reshape((data_calculated_each_time/step,step)))

                        self.tcp.memcpy(buffer_out
                            [(start+i*3*step_each_time).get()+2:
                            (start+(i+1)*3*step_each_time).get()+2:3],
                            buffer_c2[0:data_calculated_each_time]
                            .reshape((data_calculated_each_time/step,step)))

        with self.tcp.else_scope():
            #step大于data_each_buffer，每次尽可能地将buffer填满做计算，
            #但是如果若干次拷贝后一个step中剩余的数据小于Buffer,那么就跳到下一个step的位置开始拷贝将buffer填满
            #同时没有像上面那样reshape以step为单位做strided copy的话，start不是线性的，因为每次拷贝的长度是buffer长度而不是step
            buffer_in0 = buffer_in0.flatten()
            buffer_in1 = buffer_in1.flatten()
            buffer_out = buffer_out.flatten()

            flag_ping.assign(0)
            flag_pong.assign(0)

            group_each_task.assign(group // self.task_num)
            rest.assign(group % self.task_num)
            with self.tcp.if_scope(task_id < rest):
                group_each_task.assign(group_each_task + 1)
                start.assign(group_each_task * task_id * 3 * step)
                stop.assign(group_each_task * (task_id + 1) * 3 * step)
            with self.tcp.else_scope():
                # start.assign(((group_each_task+1)*task_id-(task_id-rest))*3*step)，化简
                start.assign( 3 * (group_each_task * task_id + rest) * step)
                #stop.assign(((group_each_task+1)*(task_id+1)-(task_id+1-rest))*3 * step)，化简
                stop.assign(3 * (group_each_task * (task_id+1) + rest) * step)
            threshold.assign(start + step)
            #当前step的最后一个位置，到达该位置发生截断，从下一个start开始拷贝

            stop.assign(stop-2*step)

            loop_num.assign(group_each_task * step // data_each_buffer)
            with self.tcp.if_scope(group_each_task * step // data_each_buffer!=0):
                loop_num.assign(loop_num+1)
                last_loop.assign(1)

            with self.tcp.for_range(0, loop_num, stage = 1) as i:
                # declare on-chip buffer
                buffer_a0 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_a0",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_a1 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_a1",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_a2 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_a2",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_b0 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_b0",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_b1 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_b1",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_b2 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_b2",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_c0 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_c0",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_c1 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_c1",
                    dtype=self.dtype,
                    scope="nram",
                )
                buffer_c2 = self.tcp.Buffer(
                    shape=(data_each_buffer,),
                    name="INPUT0_c2",
                    dtype=self.dtype,
                    scope="nram",
                )
                #每轮循环ping拷贝数据+ping更新start拷进新的数据/与此同时pong计算数据,因为start不是线性的所以要分开存储
                with self.tcp.block("data_copy"):
                    # 最后一次余数的特殊处理
                    with self.tcp.if_scope(tcp.all(i == loop_num-1 , last_loop == 1)):
                        self.tcp.memcpy(buffer_a0[0:stop-start],
                            buffer_in0[start:stop])
                        self.tcp.memcpy(buffer_a1[0:stop-start],
                            buffer_in0[start+step:stop+step])
                        self.tcp.memcpy(buffer_a2[0:stop-start],
                            buffer_in0[start+2*step:stop+2*step])
                        self.tcp.memcpy(buffer_b0[0:stop-start],
                            buffer_in1[start:stop])
                        self.tcp.memcpy(buffer_b1[0:stop-start],
                            buffer_in1[start+step:stop+step])
                        self.tcp.memcpy(buffer_b2[0:stop-start],
                            buffer_in1[start+2*step:stop+2*step])

                    with self.tcp.else_scope():
                        #当前step剩余部分足够填满buffer
                        with self.tcp.if_scope(start + data_each_buffer < threshold):
                            with self.tcp.if_scope(i%2 == 0):
                                flag_ping.assign(0)
                                start_ping.assign(start)
                            with self.tcp.if_scope(i%2 == 1):
                                flag_pong.assign(0)
                                start_pong.assign(start)
                            start.assign(start + data_each_buffer)
                        #当前step剩余部分足够填满buffer但是start要发生跳跃
                        with self.tcp.elif_scope(start + data_each_buffer == threshold):
                            with self.tcp.if_scope(i%2 == 0):
                                flag_ping.assign(0)
                                start_ping.assign(start)
                            with self.tcp.if_scope(i%2 == 1):
                                flag_pong.assign(0)
                                start_pong.assign(start)
                            #下一个start的位置
                            start.assign((start // (3*step) +1) * 3 * step)
                            #下一个start的截断位置
                            threshold.assign((start // (3*step) +1) * 3 * step + step)
                        #当前step剩余部分不足以填满buffer，发生start的跳跃
                        with self.tcp.else_scope():
                            #第一次拷贝的终止位置
                            leap1.assign(threshold)
                            #下一个start的位置，第二次拷贝开始的位置
                            leap2.assign((start // (3*step) +1) * 3 * step)
                            #下一个start的截断位置
                            threshold.assign( (start // (3*step) +1) * 3 * step + step)
                            with self.tcp.if_scope(i%2 == 0):
                                start_ping.assign(start)
                                flag_ping.assign(1)
                            with self.tcp.if_scope(i%2 == 1):
                                start_pong.assign(start)
                                flag_pong.assign(1)

                        with self.tcp.if_scope(tcp.all(i%2 == 0 , flag_ping==0)):
                            #一次性拷贝完
                            self.tcp.memcpy(buffer_a0,
                                buffer_in0[start_ping : start_ping+data_each_buffer])

                            self.tcp.memcpy(buffer_a1,
                                buffer_in0[start_ping+step : start_ping+data_each_buffer+step])

                            self.tcp.memcpy(buffer_a2,
                                buffer_in0[start_ping+2*step : start_ping+data_each_buffer+2*step])

                            self.tcp.memcpy(buffer_b0,
                                buffer_in1[start_ping : start_ping+data_each_buffer])

                            self.tcp.memcpy(buffer_b1,
                                buffer_in1[start_ping+step : start_ping+data_each_buffer+step])

                            self.tcp.memcpy(buffer_b2,
                                buffer_in1[start_ping+2*step : start_ping+data_each_buffer+2*step])

                        with self.tcp.elif_scope(tcp.all(i%2 == 1 , flag_pong==0)):
                            self.tcp.memcpy(buffer_a0,
                                buffer_in0[start_pong : start_pong+data_each_buffer])

                            self.tcp.memcpy(buffer_a1,
                                buffer_in0[start_pong+step : start_pong+data_each_buffer+step])

                            self.tcp.memcpy(buffer_a2,
                                buffer_in0[start_pong+2*step : start_pong+data_each_buffer+2*step])

                            self.tcp.memcpy(buffer_b0,
                                buffer_in1[start_pong : start_pong+data_each_buffer])

                            self.tcp.memcpy(buffer_b1,
                                buffer_in1[start_pong+step : start_pong+data_each_buffer+step])

                            self.tcp.memcpy(buffer_b2,
                                buffer_in1[start_pong+2*step : start_pong+data_each_buffer+2*step])

                        with self.tcp.elif_scope(tcp.all(i%2 == 0 , flag_ping==1)):
                            #拷贝两次，第一次是前一个step的末尾部分，第二次是下一个step的开始部分
                            leap1_ping.assign(leap1)
                            leap2_ping.assign(leap2)
                            start.assign(leap2_ping + data_each_buffer - leap1_ping + start_ping)

                            self.tcp.memcpy(buffer_a0[0: leap1_ping - start_ping],
                                buffer_in0[start_ping  : leap1_ping])

                            self.tcp.memcpy(buffer_a1[0: leap1_ping - start_ping]
                                ,buffer_in0[start_ping+step : leap1_ping+step])

                            self.tcp.memcpy(buffer_a2[0: leap1_ping - start_ping]
                                ,buffer_in0[start_ping+2*step : leap1_ping+2*step])

                            self.tcp.memcpy(buffer_b0[0: leap1_ping - start_ping]
                                ,buffer_in1[start_ping : leap1_ping])

                            self.tcp.memcpy(buffer_b1[0: leap1_ping - start_ping]
                                ,buffer_in1[start_ping+step : leap1_ping+step])

                            self.tcp.memcpy(buffer_b2[0: leap1_ping - start_ping]
                                ,buffer_in1[start_ping+2*step : leap1_ping+2*step])

                            self.tcp.memcpy(buffer_a0[leap1_ping - start_ping: data_each_buffer],
                                buffer_in0[leap2_ping :
                                leap2_ping + data_each_buffer - leap1_ping + start_ping])

                            self.tcp.memcpy(buffer_a1[leap1_ping - start_ping: data_each_buffer],
                                buffer_in0[leap2_ping+step :
                                leap2_ping + data_each_buffer - leap1_ping + start_ping+step])

                            self.tcp.memcpy(buffer_a2[leap1_ping - start_ping: data_each_buffer],
                                buffer_in0[leap2_ping+2*step :
                                leap2_ping + data_each_buffer - leap1_ping + start_ping+2*step])

                            self.tcp.memcpy(buffer_b0[leap1_ping - start_ping: data_each_buffer],
                                buffer_in1[leap2_ping :
                                leap2_ping + data_each_buffer - leap1_ping + start_ping])

                            self.tcp.memcpy(buffer_b1[leap1_ping - start_ping: data_each_buffer],
                                buffer_in1[leap2_ping+step :
                                leap2_ping + data_each_buffer - leap1_ping + start_ping+step])

                            self.tcp.memcpy(buffer_b2[leap1_ping - start_ping: data_each_buffer],
                                buffer_in1[leap2_ping+2*step :
                                leap2_ping + data_each_buffer - leap1_ping + start_ping+2*step])

                        with self.tcp.elif_scope(tcp.all(i%2 == 1 , flag_pong==1)):
                            leap1_pong.assign(leap1)
                            leap2_pong.assign(leap2)
                            start.assign(leap2_pong+data_each_buffer - leap1_pong + start_pong)

                            self.tcp.memcpy(buffer_a0[0:leap1_pong - start_pong],
                                buffer_in0[start_pong : leap1_pong])

                            self.tcp.memcpy(buffer_a1[0:leap1_pong - start_pong],
                                buffer_in0[start_pong+step : leap1_pong+step])

                            self.tcp.memcpy(buffer_a2[0:leap1_pong - start_pong],
                                buffer_in0[start_pong+2*step : leap1_pong+2*step])

                            self.tcp.memcpy(buffer_b0[0:leap1_pong - start_pong],
                                buffer_in1[start_pong : leap1_pong])

                            self.tcp.memcpy(buffer_b1[0:leap1_pong - start_pong],
                                buffer_in1[start_pong+step : leap1_pong+step])

                            self.tcp.memcpy(buffer_b2[0:leap1_pong - start_pong],
                                buffer_in1[start_pong+2*step : leap1_pong+2*step])

                            self.tcp.memcpy(buffer_a0[leap1_pong - start_pong : data_each_buffer],
                                buffer_in0[leap2_pong :
                                leap2_pong+data_each_buffer - leap1_pong + start_pong])

                            self.tcp.memcpy(buffer_a1[leap1_pong - start_pong : data_each_buffer],
                                buffer_in0[leap2_pong+step :
                                leap2_pong+data_each_buffer - leap1_pong + start_pong+step])

                            self.tcp.memcpy(buffer_a2[leap1_pong - start_pong : data_each_buffer],
                                buffer_in0[leap2_pong+2*step :
                                leap2_pong+data_each_buffer - leap1_pong + start_pong+2*step])

                            self.tcp.memcpy(buffer_b0[leap1_pong - start_pong : data_each_buffer],
                                buffer_in1[leap2_pong :
                                leap2_pong+data_each_buffer - leap1_pong + start_pong])

                            self.tcp.memcpy(buffer_b1[leap1_pong - start_pong : data_each_buffer],
                                buffer_in1[leap2_pong+step :
                                leap2_pong+data_each_buffer - leap1_pong + start_pong+step])

                            self.tcp.memcpy(buffer_b2[leap1_pong - start_pong : data_each_buffer],
                                buffer_in1[leap2_pong+2*step :
                                leap2_pong+data_each_buffer - leap1_pong + start_pong+2*step])

                with self.tcp.block("compute"):
                    self.tcp.multiply(buffer_c0,buffer_a1,buffer_b2)
                    self.tcp.multiply(buffer_c1,buffer_a2,buffer_b1)
                    self.tcp.subtract(buffer_c0,buffer_c0,buffer_c1)

                    self.tcp.multiply(buffer_c1,buffer_a2,buffer_b0)
                    self.tcp.multiply(buffer_c2,buffer_a0,buffer_b2)
                    self.tcp.subtract(buffer_c1,buffer_c1,buffer_c2)

                    self.tcp.multiply(buffer_c2,buffer_a0,buffer_b1)
                    self.tcp.multiply(buffer_a0,buffer_a1,buffer_b0)
                    self.tcp.subtract(buffer_c2,buffer_c2,buffer_a0)


                with self.tcp.block("data_copy"):
                    with self.tcp.if_scope(tcp.all(i == loop_num-1 , last_loop == 1)):
                        self.tcp.memcpy(buffer_out[start:stop],buffer_c0[0:stop-start])
                        self.tcp.memcpy(buffer_out[start+step:stop+step],buffer_c1[0:stop-start])
                        self.tcp.memcpy(buffer_out[start+2*step:stop+2*step],
                            buffer_c2[0:stop-start])

                    with self.tcp.else_scope():
                        with self.tcp.if_scope(tcp.all(i%2 == 0 , flag_ping==0)):
                            self.tcp.memcpy(buffer_out[start_ping:
                                start_ping+data_each_buffer], buffer_c0)
                            self.tcp.memcpy(buffer_out[start_ping+step:
                                start_ping+data_each_buffer+step], buffer_c1)
                            self.tcp.memcpy(buffer_out[start_ping+2*step
                                :start_ping+data_each_buffer+2*step], buffer_c2)

                        with self.tcp.elif_scope(tcp.all(i%2 == 1 , flag_pong==0)):
                            self.tcp.memcpy(buffer_out[start_pong :
                                start_pong+data_each_buffer], buffer_c0)
                            self.tcp.memcpy(buffer_out[start_pong+step :
                                start_pong+data_each_buffer+step], buffer_c1)
                            self.tcp.memcpy(buffer_out[start_pong+2*step :
                                start_pong+data_each_buffer+2*step], buffer_c2)

                        with self.tcp.elif_scope(tcp.all(i%2 == 0 , flag_ping==1)):
                            self.tcp.memcpy(buffer_out[start_ping : leap1_ping],
                                buffer_c0[0:leap1_ping - start_ping])

                            self.tcp.memcpy(buffer_out[start_ping+step : leap1_ping+step],
                                buffer_c1[0:leap1_ping - start_ping])

                            self.tcp.memcpy(buffer_out[start_ping+2*step : leap1_ping+2*step],
                                buffer_c2[0:leap1_ping - start_ping])

                            self.tcp.memcpy(buffer_out[leap2_ping :
                                leap2_ping+data_each_buffer - leap1_ping + start_ping],
                                buffer_c0[leap1_ping - start_ping: data_each_buffer])

                            self.tcp.memcpy(buffer_out[leap2_ping+step :
                                leap2_ping+data_each_buffer - leap1_ping + start_ping+step],
                                buffer_c1[leap1_ping - start_ping: data_each_buffer])

                            self.tcp.memcpy(buffer_out[leap2_ping+2*step :
                                leap2_ping+data_each_buffer - leap1_ping + start_ping+2*step],
                                buffer_c2[leap1_ping - start_ping: data_each_buffer])

                        with self.tcp.elif_scope(tcp.all(i%2 == 1 , flag_pong==1)):
                            self.tcp.memcpy(buffer_out[start_pong : leap1_pong],
                                buffer_c0[0 : leap1_pong - start_pong])
                            self.tcp.memcpy(buffer_out[start_pong+step : leap1_pong+step],
                                buffer_c1[0 : leap1_pong - start_pong])
                            self.tcp.memcpy(buffer_out[start_pong+2*step : leap1_pong+2*step],
                                buffer_c2[0 : leap1_pong - start_pong])
                            self.tcp.memcpy(buffer_out[leap2_pong :
                                leap2_pong+data_each_buffer - leap1_pong + start_pong],
                                buffer_c0[leap1_pong - start_pong : data_each_buffer])
                            self.tcp.memcpy(buffer_out[leap2_pong+step :
                                leap2_pong+data_each_buffer - leap1_pong + start_pong+step],
                                buffer_c1[leap1_pong - start_pong : data_each_buffer])
                            self.tcp.memcpy(buffer_out[leap2_pong+2*step :
                                leap2_pong+data_each_buffer - leap1_pong + start_pong+2*step],
                                buffer_c2[leap1_pong - start_pong : data_each_buffer])


        buffer_out.reshape((self.dim0, self.dim1, self.dim2,
            self.dim3, self.dim4, self.dim5, self.dim6, self.dim7))

        # build a executable module
        f = self.tcp.BuildBANG(
            inputs=[
                buffer_in0,
                buffer_in1,
                shape,
                self.dim
            ],
            outputs=[buffer_out],
            kernel_name=KERNEL_NAME,
        )
        return f


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_cross(dtype=None, target=None):
    task_num = 64
    f = Cross(dtype, target, task_num).compute_body()
    return f
