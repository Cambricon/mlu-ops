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
"""A multi-platform code link example test for BANGPy TCP."""
import bangpy
from bangpy import tcp
from bangpy.tcp.runtime import TaskType
from bangpy.platform.bang_config import TARGET

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu290"]
KERNEL_NAME = "cross"


class Cross(object):
    """Operator description:
    torch.cross
    torch.cross(input, other, dim, output)
    compute the 3D cross product of two tensors: input and other on dimension dim,
    and output is the result tensor.
    To learn more, please refer to design docs.
    """
    def __init__(self, dtype, target, task_num):

        if not ((dtype in DTYPES) and (target in TARGET_LIST)):
            raise KeyError("please pass correct parameters.")

        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.tcp = tcp.TCP(target)
        self.dim = self.tcp.Var("_dim", dtype=bangpy.int32)

        self.dim0 = self.tcp.SizeVar("_dim0")
        self.dim1 = self.tcp.SizeVar("_dim1")
        self.dim2 = self.tcp.SizeVar("_dim2")
        self.dim3 = self.tcp.SizeVar("_dim3")
        self.dim4 = self.tcp.SizeVar("_dim4")
        self.dim5 = self.tcp.SizeVar("_dim5")
        self.dim6 = self.tcp.SizeVar("_dim6")
        self.dim7 = self.tcp.SizeVar("_dim7")
        self.dimlength = self.tcp.SizeVar("_dimlength")
        self.dtype_sz = dtype.bytes
        # pipeline needs buffer (a0,a1,a2,b0,b1,b2,c0,c1,c2)
        # to realize cross:(a0,a1,a2)x(b0,b1,b2)=(c0,c1,c2)
        # so two-level-pipeline needs 9x2=18 buffers totally
        # self.nram_size = 512*1024byte,18 buffer needed,
        # and multiply and substraction should be 128byte aligned,
        # 128*18=2304,512*1024//2304*2304=523008(512*1024=524288),
        # 523008/18=29056,29056 can be exactly divided by 128,
        # and of course can be exactly divided by 4(byte) and 2(byte)
        self.single_buffer_size = 29056
        self.tcp.launch_task(self.task_num, 1, 1)
        self.tcp.launch_cluster(TaskType.BLOCK)

    def compute_body(self):
        shape = self.tcp.Buffer(
            shape=(self.dimlength,), name="shape", dtype=bangpy.int32, scope="global"
        )
        # if 'shape' is set as parameter，then 'shape[i]' would be not supported,
        # because i must be python int/float
        # but i in for_range must be int type in bangpy
        # conflicted, so 'shape' can't be parameter in cross function

        mydim = self.tcp.Scalar(bangpy.int32, "mydim")

        mydim.assign(self.dim)

        with self.tcp.if_scope(tcp.any(self.dim < -8, self.dim > 7)):
            self.tcp.print("dim shall be in [-8,7], but not")
        # should be stopped here, but bangpy has no return or raise error

        with self.tcp.if_scope(tcp.all(self.dim < 0, self.dim >= -8)):
            mydim.assign(mydim + 8)
            with self.tcp.if_scope(shape[mydim] != 3):
                self.tcp.print("shape[dim] is not 3!")
                # should be stopped here, but bangpy has no return or raise error
        with self.tcp.else_scope():
            with self.tcp.if_scope(shape[self.dim] != 3):
                self.tcp.print("shape[dim] is not 3!")
                # should be stopped here, but bangpy has no return or raise error

        mydim.assign(mydim + 1)

        maxdim = self.tcp.Scalar(bangpy.int32, "maxdim")
        maxdim.assign(8)
        # can also use self.dimlength，
        # but pipeline buffer's shape must be defined with SizeVar,
        # so the dimension of shape must be static

        # e.g.,shape=(1,1,1,1,2,3,4,5), dim=5 means shape[5] = 3，
        # suppose the product of dimensions before dim called group,
        # the product of dimensions before dim called step,,
        # both of them do not include dim,
        # in this example, group=1x1x1x1x2=2, step=4x5=20
        length = self.tcp.Scalar(bangpy.int32, "length")
        length.assign(1)
        step = self.tcp.Scalar(bangpy.int32, "step")
        step.assign(1)
        with self.tcp.for_range(0, maxdim) as i:
            length.assign(length * shape[i])
        # 1x1x1x1x2x3x4x5=120 elements totally

        with self.tcp.for_range(mydim, maxdim) as i:
            step.assign(step * shape[i])
        # step=4x5=20, if current element's index is 'i',
        # then next element is in 'i+step'

        group = self.tcp.Scalar(bangpy.int32, "group")
        rest = self.tcp.Scalar(bangpy.int32, "rest")

        # group = 1x1x1x1x1x2
        group.assign(length / 3 / step)

        # calculate split strategy
        # declare I/O buffer
        buffer_in0 = self.tcp.Buffer(
            shape=(
                self.dim0,
                self.dim1,
                self.dim2,
                self.dim3,
                self.dim4,
                self.dim5,
                self.dim6,
                self.dim7,
            ),
            name="INPUT0",
            dtype=self.dtype,
            scope="global",
        )
        buffer_in1 = self.tcp.Buffer(
            shape=(
                self.dim0,
                self.dim1,
                self.dim2,
                self.dim3,
                self.dim4,
                self.dim5,
                self.dim6,
                self.dim7,
            ),
            name="INPUT1",
            dtype=self.dtype,
            scope="global",
        )
        buffer_out = self.tcp.Buffer(
            shape=(
                self.dim0,
                self.dim1,
                self.dim2,
                self.dim3,
                self.dim4,
                self.dim5,
                self.dim6,
                self.dim7,
            ),
            name="OUTPUT",
            dtype=self.dtype,
            scope="global",
        )
        task_id = self.tcp.taskId

        start = self.tcp.Scalar(bangpy.int32, name="start")
        stop = self.tcp.Scalar(bangpy.int32, name="stop")
        start_ping = self.tcp.Scalar(bangpy.int32, name="start_ping")
        start_pong = self.tcp.Scalar(bangpy.int32, name="start_pong")
        group_each_task = self.tcp.Scalar(bangpy.int32, name="group_each_task")
        threshold = self.tcp.Scalar(bangpy.int32, name="threshold")
        loop_num = self.tcp.Scalar(bangpy.int32, name="loop_num")
        leap1 = self.tcp.Scalar(bangpy.int32, name="leap1")
        leap2 = self.tcp.Scalar(bangpy.int32, name="leap2")
        leap1_ping = self.tcp.Scalar(bangpy.int32, name="leap1_ping")
        leap1_pong = self.tcp.Scalar(bangpy.int32, name="leap2_ping")
        leap2_ping = self.tcp.Scalar(bangpy.int32, name="leap1_pong")
        leap2_pong = self.tcp.Scalar(bangpy.int32, name="leap2_pong")
        last_loop = self.tcp.Scalar(bangpy.int32, name="last_loop")
        flag_ping = self.tcp.Scalar(bangpy.int32, name="flag_ping")
        flag_pong = self.tcp.Scalar(bangpy.int32, name="flag_pong")
        data_calculated_each_time = self.tcp.Scalar(
            bangpy.int32, name="data_calculated_each_time"
        )
        data_calculated_each_time_last = self.tcp.Scalar(
            bangpy.int32, name="data_calculated_each_time_last"
        )

        data_each_buffer = self.single_buffer_size // self.dtype_sz
        last_loop.assign(0)

        with self.tcp.if_scope(step <= data_each_buffer):
            step_each_time = data_each_buffer // step
            # step<= data_each_buffer，branch 1
            # every time data_each_buffer//step steps can be computed.

            data_calculated_each_time = step_each_time * step

            # split and compute

            buffer_in0 = buffer_in0.reshape((group * 3, step))
            buffer_in1 = buffer_in1.reshape((group * 3, step))
            buffer_out = buffer_out.reshape((group * 3, step))

            # compute every task's group number,
            # remainder distributed from task_id=0
            # e.g. group=7, task=3, then group_each_task[3] is (3,2,2)
            group_each_task.assign(group // self.task_num)
            rest.assign(group % self.task_num)
            with self.tcp.if_scope(task_id < rest):
                group_each_task.assign(group_each_task + 1)
                start.assign(group_each_task * task_id * 3)
                # stop:the max value which the index of buffer can reach in current task
                # index can not in next task's compute range
                # in other words, stop is next task's start
                stop.assign(group_each_task * (task_id + 1) * 3)
            with self.tcp.else_scope():
                # start.assign(((group_each_task+1)*task_id-(task_id-rest))*3),simplify
                start.assign(3 * (group_each_task * task_id + rest))
                stop.assign(3 * (group_each_task * (task_id + 1) + rest))
            stop.assign(stop - 2)

            # if there exists remainder (can't be exactly divided),
            # that means an extra compute time is needed;
            # in this case we need 'stop' computed before
            loop_num.assign(group_each_task // step_each_time)
            with self.tcp.if_scope(group_each_task % step_each_time != 0):
                loop_num.assign(loop_num + 1)
                last_loop.assign(1)
                # means there exists remainder
                data_calculated_each_time_last.assign(
                    ((stop - start - (loop_num - 1) * 3 * step_each_time) // 3) * step
                )
                with self.tcp.if_scope(
                    (stop - start - (loop_num - 1) * 3 * step_each_time) % 3 != 0
                ):
                    data_calculated_each_time_last.assign(
                        data_calculated_each_time_last + step
                    )

            with self.tcp.for_range(0, loop_num, stage=1) as i:
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
                # (a0,a1,a2)x(b0,b1,b2)=(c0,c1,c2)
                with self.tcp.block("data_copy"):
                    with self.tcp.if_scope(tcp.all(i == loop_num - 1, last_loop == 1)):

                        self.tcp.memcpy(
                            buffer_a0[0:data_calculated_each_time_last].reshape(
                                (data_calculated_each_time_last / step, step)
                            ),
                            buffer_in0[
                                (start + i * 3 * step_each_time).get() : stop.get() : 3
                            ],
                        )

                        self.tcp.memcpy(
                            buffer_a1[0:data_calculated_each_time_last].reshape(
                                (data_calculated_each_time_last / step, step)
                            ),
                            buffer_in0[
                                (start + i * 3 * step_each_time).get()
                                + 1 : stop.get()
                                + 1 : 3
                            ],
                        )

                        self.tcp.memcpy(
                            buffer_a2[0:data_calculated_each_time_last].reshape(
                                (data_calculated_each_time_last / step, step)
                            ),
                            buffer_in0[
                                (start + i * 3 * step_each_time).get()
                                + 2 : stop.get()
                                + 2 : 3
                            ],
                        )

                        self.tcp.memcpy(
                            buffer_b0[0:data_calculated_each_time_last].reshape(
                                (data_calculated_each_time_last / step, step)
                            ),
                            buffer_in1[
                                (start + i * 3 * step_each_time).get() : stop.get() : 3
                            ],
                        )

                        self.tcp.memcpy(
                            buffer_b1[0:data_calculated_each_time_last].reshape(
                                (data_calculated_each_time_last / step, step)
                            ),
                            buffer_in1[
                                (start + i * 3 * step_each_time).get()
                                + 1 : stop.get()
                                + 1 : 3
                            ],
                        )

                        self.tcp.memcpy(
                            buffer_b2[0:data_calculated_each_time_last].reshape(
                                (data_calculated_each_time_last / step, step)
                            ),
                            buffer_in1[
                                (start + i * 3 * step_each_time).get()
                                + 2 : stop.get()
                                + 2 : 3
                            ],
                        )

                    with self.tcp.else_scope():

                        self.tcp.memcpy(
                            buffer_a0[0:data_calculated_each_time].reshape(
                                (data_calculated_each_time / step, step)
                            ),
                            buffer_in0[
                                (start + i * 3 * step_each_time)
                                .get() : (start + (i + 1) * 3 * step_each_time)
                                .get() : 3
                            ],
                        )

                        self.tcp.memcpy(
                            buffer_a1[0:data_calculated_each_time].reshape(
                                (data_calculated_each_time / step, step)
                            ),
                            buffer_in0[
                                (start + i * 3 * step_each_time).get()
                                + 1 : (start + (i + 1) * 3 * step_each_time).get()
                                + 1 : 3
                            ],
                        )

                        self.tcp.memcpy(
                            buffer_a2[0:data_calculated_each_time].reshape(
                                (data_calculated_each_time / step, step)
                            ),
                            buffer_in0[
                                (start + i * 3 * step_each_time).get()
                                + 2 : (start + (i + 1) * 3 * step_each_time).get()
                                + 2 : 3
                            ],
                        )

                        self.tcp.memcpy(
                            buffer_b0[0:data_calculated_each_time].reshape(
                                (data_calculated_each_time / step, step)
                            ),
                            buffer_in1[
                                (start + i * 3 * step_each_time)
                                .get() : (start + (i + 1) * 3 * step_each_time)
                                .get() : 3
                            ],
                        )

                        self.tcp.memcpy(
                            buffer_b1[0:data_calculated_each_time].reshape(
                                (data_calculated_each_time / step, step)
                            ),
                            buffer_in1[
                                (start + i * 3 * step_each_time).get()
                                + 1 : (start + (i + 1) * 3 * step_each_time).get()
                                + 1 : 3
                            ],
                        )

                        self.tcp.memcpy(
                            buffer_b2[0:data_calculated_each_time].reshape(
                                (data_calculated_each_time / step, step)
                            ),
                            buffer_in1[
                                (start + i * 3 * step_each_time).get()
                                + 2 : (start + (i + 1) * 3 * step_each_time).get()
                                + 2 : 3
                            ],
                        )

                with self.tcp.block("compute"):
                    self.tcp.multiply(buffer_c0, buffer_a1, buffer_b2)
                    self.tcp.multiply(buffer_c1, buffer_a2, buffer_b1)
                    self.tcp.subtract(buffer_c0, buffer_c0, buffer_c1)

                    self.tcp.multiply(buffer_c1, buffer_a2, buffer_b0)
                    self.tcp.multiply(buffer_c2, buffer_a0, buffer_b2)
                    self.tcp.subtract(buffer_c1, buffer_c1, buffer_c2)

                    self.tcp.multiply(buffer_c2, buffer_a0, buffer_b1)
                    self.tcp.multiply(buffer_a0, buffer_a1, buffer_b0)
                    self.tcp.subtract(buffer_c2, buffer_c2, buffer_a0)

                with self.tcp.block("data_copy"):
                    with self.tcp.if_scope(tcp.all(i == loop_num - 1, last_loop == 1)):

                        self.tcp.memcpy(
                            buffer_out[
                                (start + i * 3 * step_each_time).get() : stop.get() : 3
                            ],
                            buffer_c0[0:data_calculated_each_time_last].reshape(
                                (data_calculated_each_time_last / step, step)
                            ),
                        )

                        self.tcp.memcpy(
                            buffer_out[
                                (start + i * 3 * step_each_time).get()
                                + 1 : stop.get()
                                + 1 : 3
                            ],
                            buffer_c1[0:data_calculated_each_time_last].reshape(
                                (data_calculated_each_time_last / step, step)
                            ),
                        )

                        self.tcp.memcpy(
                            buffer_out[
                                (start + i * 3 * step_each_time).get()
                                + 2 : stop.get()
                                + 2 : 3
                            ],
                            buffer_c2[0:data_calculated_each_time_last].reshape(
                                (data_calculated_each_time_last / step, step)
                            ),
                        )

                    with self.tcp.else_scope():

                        self.tcp.memcpy(
                            buffer_out[
                                (start + i * 3 * step_each_time)
                                .get() : (start + (i + 1) * 3 * step_each_time)
                                .get() : 3
                            ],
                            buffer_c0[0:data_calculated_each_time].reshape(
                                (data_calculated_each_time / step, step)
                            ),
                        )

                        self.tcp.memcpy(
                            buffer_out[
                                (start + i * 3 * step_each_time).get()
                                + 1 : (start + (i + 1) * 3 * step_each_time).get()
                                + 1 : 3
                            ],
                            buffer_c1[0:data_calculated_each_time].reshape(
                                (data_calculated_each_time / step, step)
                            ),
                        )

                        self.tcp.memcpy(
                            buffer_out[
                                (start + i * 3 * step_each_time).get()
                                + 2 : (start + (i + 1) * 3 * step_each_time).get()
                                + 2 : 3
                            ],
                            buffer_c2[0:data_calculated_each_time].reshape(
                                (data_calculated_each_time / step, step)
                            ),
                        )

        with self.tcp.else_scope():
            # step > data_each_buffer, branch 2
            # every loop, a step cannot be calculated entirely,
            # so we just fill the buffer as long as we can,
            # but it can happen that a step has already been loaded several times,
            # then the rest part cannot fill the buffer, when this happen,
            # index will jump to the next step's 'start' to continue load until buffer is full.
            # In this strategy, every loop's offset is not linear to loop time (i),
            # e.g. data_each_buffer = 80, step = 100, then memcpy ranges are:
            # [start, start+80), [start+80, start+100) & [start + 300, start + 360)
            # (jump 3 steps), so the offset is (0, 80, 360), obviously not linear.
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
                # start.assign(((group_each_task+1)*task_id-(task_id-rest))*3*step)，simplify
                start.assign(3 * (group_each_task * task_id + rest) * step)
                # stop.assign(((group_each_task+1)*(task_id+1)-(task_id+1-rest))*3 * step)，simplify
                stop.assign(3 * (group_each_task * (task_id + 1) + rest) * step)
            threshold.assign(start + step)
            # the last position of current step, when memcpy reaches this index,
            # index should jump to next start to continue memcpy

            stop.assign(stop - 2 * step)

            loop_num.assign(group_each_task * step // data_each_buffer)
            with self.tcp.if_scope(group_each_task * step // data_each_buffer != 0):
                loop_num.assign(loop_num + 1)
                last_loop.assign(1)

            with self.tcp.for_range(0, loop_num, stage=1) as i:
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
                # every loop: ping load/store and in the same time, pong compute
                # because when index reached threshold of current step,
                # its value will jump to next start, it's offset is not linear to the loop time (i),
                # so we need start_ping and start_pong to store their 'start' differently
                with self.tcp.block("data_copy"):
                    # if there exists remainder, that means all data left should be load once
                    with self.tcp.if_scope(tcp.all(i == loop_num - 1, last_loop == 1)):
                        self.tcp.memcpy(
                            buffer_a0[0 : stop - start], buffer_in0[start:stop]
                        )
                        self.tcp.memcpy(
                            buffer_a1[0 : stop - start],
                            buffer_in0[start + step : stop + step],
                        )
                        self.tcp.memcpy(
                            buffer_a2[0 : stop - start],
                            buffer_in0[start + 2 * step : stop + 2 * step],
                        )
                        self.tcp.memcpy(
                            buffer_b0[0 : stop - start], buffer_in1[start:stop]
                        )
                        self.tcp.memcpy(
                            buffer_b1[0 : stop - start],
                            buffer_in1[start + step : stop + step],
                        )
                        self.tcp.memcpy(
                            buffer_b2[0 : stop - start],
                            buffer_in1[start + 2 * step : stop + 2 * step],
                        )

                    with self.tcp.else_scope():
                        # situation 1:
                        # current step's rest part (or just not partly loaded)
                        # is long enough that index will not reach threshold,
                        # that means index would not jump and only need one copy operation
                        with self.tcp.if_scope(start + data_each_buffer < threshold):
                            with self.tcp.if_scope(i % 2 == 0):
                                flag_ping.assign(0)
                                start_ping.assign(start)
                            with self.tcp.if_scope(i % 2 == 1):
                                flag_pong.assign(0)
                                start_pong.assign(start)
                            start.assign(start + data_each_buffer)
                        # situation 2:
                        # current step's rest part is just exactly fits the buffer,
                        # that means only need one copy operation,
                        # but index needs to jump for next loop
                        with self.tcp.elif_scope(start + data_each_buffer == threshold):
                            with self.tcp.if_scope(i % 2 == 0):
                                flag_ping.assign(0)
                                start_ping.assign(start)
                            with self.tcp.if_scope(i % 2 == 1):
                                flag_pong.assign(0)
                                start_pong.assign(start)
                            # the start of next step
                            start.assign((start // (3 * step) + 1) * 3 * step)
                            # the threshold of next step
                            threshold.assign(
                                (start // (3 * step) + 1) * 3 * step + step
                            )
                        # situation 3:
                        # current step's rest part is not long enough for buffer,
                        # and index will reach threshold in this loop,
                        # that means jump will happen and need two copy operation:
                        # [start, threshold) and [next start, next start + buffer's rest length)
                        with self.tcp.else_scope():
                            # first copy's stop place, in other words, threshold
                            leap1.assign(threshold)
                            # second copy's start place, in other words, next start
                            leap2.assign((start // (3 * step) + 1) * 3 * step)
                            # the threshold of next step
                            threshold.assign(
                                (start // (3 * step) + 1) * 3 * step + step
                            )
                            with self.tcp.if_scope(i % 2 == 0):
                                start_ping.assign(start)
                                flag_ping.assign(1)
                            with self.tcp.if_scope(i % 2 == 1):
                                start_pong.assign(start)
                                flag_pong.assign(1)

                        with self.tcp.if_scope(tcp.all(i % 2 == 0, flag_ping == 0)):
                            # situation 1&2: copy once, ping
                            self.tcp.memcpy(
                                buffer_a0,
                                buffer_in0[start_ping : start_ping + data_each_buffer],
                            )

                            self.tcp.memcpy(
                                buffer_a1,
                                buffer_in0[
                                    start_ping
                                    + step : start_ping
                                    + data_each_buffer
                                    + step
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_a2,
                                buffer_in0[
                                    start_ping
                                    + 2 * step : start_ping
                                    + data_each_buffer
                                    + 2 * step
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_b0,
                                buffer_in1[start_ping : start_ping + data_each_buffer],
                            )

                            self.tcp.memcpy(
                                buffer_b1,
                                buffer_in1[
                                    start_ping
                                    + step : start_ping
                                    + data_each_buffer
                                    + step
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_b2,
                                buffer_in1[
                                    start_ping
                                    + 2 * step : start_ping
                                    + data_each_buffer
                                    + 2 * step
                                ],
                            )

                        with self.tcp.elif_scope(tcp.all(i % 2 == 1, flag_pong == 0)):
                            # situation 1&2, pong
                            self.tcp.memcpy(
                                buffer_a0,
                                buffer_in0[start_pong : start_pong + data_each_buffer],
                            )

                            self.tcp.memcpy(
                                buffer_a1,
                                buffer_in0[
                                    start_pong
                                    + step : start_pong
                                    + data_each_buffer
                                    + step
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_a2,
                                buffer_in0[
                                    start_pong
                                    + 2 * step : start_pong
                                    + data_each_buffer
                                    + 2 * step
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_b0,
                                buffer_in1[start_pong : start_pong + data_each_buffer],
                            )

                            self.tcp.memcpy(
                                buffer_b1,
                                buffer_in1[
                                    start_pong
                                    + step : start_pong
                                    + data_each_buffer
                                    + step
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_b2,
                                buffer_in1[
                                    start_pong
                                    + 2 * step : start_pong
                                    + data_each_buffer
                                    + 2 * step
                                ],
                            )

                        with self.tcp.elif_scope(tcp.all(i % 2 == 0, flag_ping == 1)):
                            # situation 3: copy twice, ping
                            leap1_ping.assign(leap1)
                            leap2_ping.assign(leap2)
                            start.assign(
                                leap2_ping + data_each_buffer - leap1_ping + start_ping
                            )
                            # first time
                            self.tcp.memcpy(
                                buffer_a0[0 : leap1_ping - start_ping],
                                buffer_in0[start_ping:leap1_ping],
                            )

                            self.tcp.memcpy(
                                buffer_a1[0 : leap1_ping - start_ping],
                                buffer_in0[start_ping + step : leap1_ping + step],
                            )

                            self.tcp.memcpy(
                                buffer_a2[0 : leap1_ping - start_ping],
                                buffer_in0[
                                    start_ping + 2 * step : leap1_ping + 2 * step
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_b0[0 : leap1_ping - start_ping],
                                buffer_in1[start_ping:leap1_ping],
                            )

                            self.tcp.memcpy(
                                buffer_b1[0 : leap1_ping - start_ping],
                                buffer_in1[start_ping + step : leap1_ping + step],
                            )

                            self.tcp.memcpy(
                                buffer_b2[0 : leap1_ping - start_ping],
                                buffer_in1[
                                    start_ping + 2 * step : leap1_ping + 2 * step
                                ],
                            )
                            # second time
                            self.tcp.memcpy(
                                buffer_a0[leap1_ping - start_ping : data_each_buffer],
                                buffer_in0[
                                    leap2_ping : leap2_ping
                                    + data_each_buffer
                                    - leap1_ping
                                    + start_ping
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_a1[leap1_ping - start_ping : data_each_buffer],
                                buffer_in0[
                                    leap2_ping
                                    + step : leap2_ping
                                    + data_each_buffer
                                    - leap1_ping
                                    + start_ping
                                    + step
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_a2[leap1_ping - start_ping : data_each_buffer],
                                buffer_in0[
                                    leap2_ping
                                    + 2 * step : leap2_ping
                                    + data_each_buffer
                                    - leap1_ping
                                    + start_ping
                                    + 2 * step
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_b0[leap1_ping - start_ping : data_each_buffer],
                                buffer_in1[
                                    leap2_ping : leap2_ping
                                    + data_each_buffer
                                    - leap1_ping
                                    + start_ping
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_b1[leap1_ping - start_ping : data_each_buffer],
                                buffer_in1[
                                    leap2_ping
                                    + step : leap2_ping
                                    + data_each_buffer
                                    - leap1_ping
                                    + start_ping
                                    + step
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_b2[leap1_ping - start_ping : data_each_buffer],
                                buffer_in1[
                                    leap2_ping
                                    + 2 * step : leap2_ping
                                    + data_each_buffer
                                    - leap1_ping
                                    + start_ping
                                    + 2 * step
                                ],
                            )

                        with self.tcp.elif_scope(tcp.all(i % 2 == 1, flag_pong == 1)):
                            # situation 3, pong
                            leap1_pong.assign(leap1)
                            leap2_pong.assign(leap2)
                            start.assign(
                                leap2_pong + data_each_buffer - leap1_pong + start_pong
                            )

                            self.tcp.memcpy(
                                buffer_a0[0 : leap1_pong - start_pong],
                                buffer_in0[start_pong:leap1_pong],
                            )

                            self.tcp.memcpy(
                                buffer_a1[0 : leap1_pong - start_pong],
                                buffer_in0[start_pong + step : leap1_pong + step],
                            )

                            self.tcp.memcpy(
                                buffer_a2[0 : leap1_pong - start_pong],
                                buffer_in0[
                                    start_pong + 2 * step : leap1_pong + 2 * step
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_b0[0 : leap1_pong - start_pong],
                                buffer_in1[start_pong:leap1_pong],
                            )

                            self.tcp.memcpy(
                                buffer_b1[0 : leap1_pong - start_pong],
                                buffer_in1[start_pong + step : leap1_pong + step],
                            )

                            self.tcp.memcpy(
                                buffer_b2[0 : leap1_pong - start_pong],
                                buffer_in1[
                                    start_pong + 2 * step : leap1_pong + 2 * step
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_a0[leap1_pong - start_pong : data_each_buffer],
                                buffer_in0[
                                    leap2_pong : leap2_pong
                                    + data_each_buffer
                                    - leap1_pong
                                    + start_pong
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_a1[leap1_pong - start_pong : data_each_buffer],
                                buffer_in0[
                                    leap2_pong
                                    + step : leap2_pong
                                    + data_each_buffer
                                    - leap1_pong
                                    + start_pong
                                    + step
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_a2[leap1_pong - start_pong : data_each_buffer],
                                buffer_in0[
                                    leap2_pong
                                    + 2 * step : leap2_pong
                                    + data_each_buffer
                                    - leap1_pong
                                    + start_pong
                                    + 2 * step
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_b0[leap1_pong - start_pong : data_each_buffer],
                                buffer_in1[
                                    leap2_pong : leap2_pong
                                    + data_each_buffer
                                    - leap1_pong
                                    + start_pong
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_b1[leap1_pong - start_pong : data_each_buffer],
                                buffer_in1[
                                    leap2_pong
                                    + step : leap2_pong
                                    + data_each_buffer
                                    - leap1_pong
                                    + start_pong
                                    + step
                                ],
                            )

                            self.tcp.memcpy(
                                buffer_b2[leap1_pong - start_pong : data_each_buffer],
                                buffer_in1[
                                    leap2_pong
                                    + 2 * step : leap2_pong
                                    + data_each_buffer
                                    - leap1_pong
                                    + start_pong
                                    + 2 * step
                                ],
                            )

                with self.tcp.block("compute"):
                    self.tcp.multiply(buffer_c0, buffer_a1, buffer_b2)
                    self.tcp.multiply(buffer_c1, buffer_a2, buffer_b1)
                    self.tcp.subtract(buffer_c0, buffer_c0, buffer_c1)

                    self.tcp.multiply(buffer_c1, buffer_a2, buffer_b0)
                    self.tcp.multiply(buffer_c2, buffer_a0, buffer_b2)
                    self.tcp.subtract(buffer_c1, buffer_c1, buffer_c2)

                    self.tcp.multiply(buffer_c2, buffer_a0, buffer_b1)
                    self.tcp.multiply(buffer_a0, buffer_a1, buffer_b0)
                    self.tcp.subtract(buffer_c2, buffer_c2, buffer_a0)

                with self.tcp.block("data_copy"):
                    # store is just reverse operation of load, load twice also means store twice.
                    with self.tcp.if_scope(tcp.all(i == loop_num - 1, last_loop == 1)):
                        # remainder
                        self.tcp.memcpy(
                            buffer_out[start:stop], buffer_c0[0 : stop - start]
                        )
                        self.tcp.memcpy(
                            buffer_out[start + step : stop + step],
                            buffer_c1[0 : stop - start],
                        )
                        self.tcp.memcpy(
                            buffer_out[start + 2 * step : stop + 2 * step],
                            buffer_c2[0 : stop - start],
                        )

                    with self.tcp.else_scope():
                        # siuation 1&2, ping
                        with self.tcp.if_scope(tcp.all(i % 2 == 0, flag_ping == 0)):
                            self.tcp.memcpy(
                                buffer_out[start_ping : start_ping + data_each_buffer],
                                buffer_c0,
                            )
                            self.tcp.memcpy(
                                buffer_out[
                                    start_ping
                                    + step : start_ping
                                    + data_each_buffer
                                    + step
                                ],
                                buffer_c1,
                            )
                            self.tcp.memcpy(
                                buffer_out[
                                    start_ping
                                    + 2 * step : start_ping
                                    + data_each_buffer
                                    + 2 * step
                                ],
                                buffer_c2,
                            )

                        with self.tcp.elif_scope(tcp.all(i % 2 == 1, flag_pong == 0)):
                            # siuation 1&2, pong
                            self.tcp.memcpy(
                                buffer_out[start_pong : start_pong + data_each_buffer],
                                buffer_c0,
                            )
                            self.tcp.memcpy(
                                buffer_out[
                                    start_pong
                                    + step : start_pong
                                    + data_each_buffer
                                    + step
                                ],
                                buffer_c1,
                            )
                            self.tcp.memcpy(
                                buffer_out[
                                    start_pong
                                    + 2 * step : start_pong
                                    + data_each_buffer
                                    + 2 * step
                                ],
                                buffer_c2,
                            )

                        with self.tcp.elif_scope(tcp.all(i % 2 == 0, flag_ping == 1)):
                            # siuation 3, ping
                            # first store
                            self.tcp.memcpy(
                                buffer_out[start_ping:leap1_ping],
                                buffer_c0[0 : leap1_ping - start_ping],
                            )

                            self.tcp.memcpy(
                                buffer_out[start_ping + step : leap1_ping + step],
                                buffer_c1[0 : leap1_ping - start_ping],
                            )

                            self.tcp.memcpy(
                                buffer_out[
                                    start_ping + 2 * step : leap1_ping + 2 * step
                                ],
                                buffer_c2[0 : leap1_ping - start_ping],
                            )
                            # second store
                            self.tcp.memcpy(
                                buffer_out[
                                    leap2_ping : leap2_ping
                                    + data_each_buffer
                                    - leap1_ping
                                    + start_ping
                                ],
                                buffer_c0[leap1_ping - start_ping : data_each_buffer],
                            )

                            self.tcp.memcpy(
                                buffer_out[
                                    leap2_ping
                                    + step : leap2_ping
                                    + data_each_buffer
                                    - leap1_ping
                                    + start_ping
                                    + step
                                ],
                                buffer_c1[leap1_ping - start_ping : data_each_buffer],
                            )

                            self.tcp.memcpy(
                                buffer_out[
                                    leap2_ping
                                    + 2 * step : leap2_ping
                                    + data_each_buffer
                                    - leap1_ping
                                    + start_ping
                                    + 2 * step
                                ],
                                buffer_c2[leap1_ping - start_ping : data_each_buffer],
                            )

                        with self.tcp.elif_scope(tcp.all(i % 2 == 1, flag_pong == 1)):
                            self.tcp.memcpy(
                                buffer_out[start_pong:leap1_pong],
                                buffer_c0[0 : leap1_pong - start_pong],
                            )
                            self.tcp.memcpy(
                                buffer_out[start_pong + step : leap1_pong + step],
                                buffer_c1[0 : leap1_pong - start_pong],
                            )
                            self.tcp.memcpy(
                                buffer_out[
                                    start_pong + 2 * step : leap1_pong + 2 * step
                                ],
                                buffer_c2[0 : leap1_pong - start_pong],
                            )
                            self.tcp.memcpy(
                                buffer_out[
                                    leap2_pong : leap2_pong
                                    + data_each_buffer
                                    - leap1_pong
                                    + start_pong
                                ],
                                buffer_c0[leap1_pong - start_pong : data_each_buffer],
                            )
                            self.tcp.memcpy(
                                buffer_out[
                                    leap2_pong
                                    + step : leap2_pong
                                    + data_each_buffer
                                    - leap1_pong
                                    + start_pong
                                    + step
                                ],
                                buffer_c1[leap1_pong - start_pong : data_each_buffer],
                            )
                            self.tcp.memcpy(
                                buffer_out[
                                    leap2_pong
                                    + 2 * step : leap2_pong
                                    + data_each_buffer
                                    - leap1_pong
                                    + start_pong
                                    + 2 * step
                                ],
                                buffer_c2[leap1_pong - start_pong : data_each_buffer],
                            )

        buffer_out.reshape(
            (
                self.dim0,
                self.dim1,
                self.dim2,
                self.dim3,
                self.dim4,
                self.dim5,
                self.dim6,
                self.dim7,
            )
        )

        # build a executable module
        f = self.tcp.BuildBANG(
            inputs=[buffer_in0, buffer_in1, shape, self.dim],
            outputs=[buffer_out],
            kernel_name=KERNEL_NAME,
        )
        return f


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_cross(dtype=None, target=None):
    task_num = TARGET(target).cluster_num * TARGET(target).core_num
    f = Cross(dtype, target, task_num).compute_body()
    return f
