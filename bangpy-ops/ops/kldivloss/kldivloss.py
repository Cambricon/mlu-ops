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
from asyncio import Task
import bangpy
from bangpy import tcp
from bangpy.tcp.runtime import TaskType
from bangpy.platform.bang_config import TARGET
import numpy as np


DTYPES = [bangpy.float32]
TARGET_LIST = ["mlu290"]
KERNEL_NAME = "kldivloss"


class KlDivLoss(object):
    """Operator description:
        The Kullback-Leibler divergence loss

        Parameters
        ----------
        reduction : Int
            The kind of reduction:
                0 represents "none"
                1 represents "sum"
                2 represents "mean"
                3 represents "batchmean" 
        log_target : if target has been logged(0:no / 1:yes)
    """

    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.target = target
        self.tcp = tcp.TCP(target)
        self.task_num = task_num

        self.batchSize = self.tcp.SizeVar("batchSize")
        self.batchLength = self.tcp.SizeVar("batchLength")
        self.totalData = self.batchLength * self.batchSize
        self.inputShape = (self.batchSize, self.batchLength)
        self.outShape = (self.totalData,)

        self.nram_size = TARGET(target).nram_size
        self.dtype_sz = dtype.bytes  # 每个元素所占字节数
        self.compute_row = 128 // self.dtype_sz  # 128字节表示的元素个数
        self.single_buffer_size = (
            64 * 128 // self.dtype_sz * self.compute_row
        )  # 单个buffer大小

        self.task_type = TaskType.UNION16
        self.tcp.launch_cluster(self.task_type.value)
        self.task_num = self.task_type.value * 4
        self.tcp.launch_task(self.task_num, 1, 1)

    def compute_body(self):

        # calculate split strategy
        cluster_num = self.task_type.value
        cluster_id = self.tcp.Scalar(bangpy.int32, name="cluster_id")
        cluster_id.assign(self.tcp.clusterId)
        task_id = self.tcp.taskId

        # 每次处理的数据量
        data_calculated_each_time = self.single_buffer_size // self.dtype_sz
        data_calculated_each_task = self.totalData // self.task_num
        with self.tcp.if_scope(task_id == self.task_num - 1):
            data_calculated_each_task = (
                self.totalData // self.task_num + (self.totalData) % self.task_num
            )

        # 每个task需要循环多少次
        loop_num = self.tcp.Scalar(bangpy.int32, name="loop_num")
        loop_num.assign(data_calculated_each_task // data_calculated_each_time)

        # global的数据buffer
        input = self.tcp.Buffer(
            shape=self.inputShape, name="input", dtype=self.dtype, scope="global"
        )

        input = input.flatten()

        target = self.tcp.Buffer(
            shape=self.inputShape, name="target", dtype=self.dtype, scope="global"
        )
        target = target.flatten()

        out = self.tcp.Buffer(
            shape=self.outShape, name="out", dtype=self.dtype, scope="global"
        )

        reduction = self.tcp.SizeVar(name="reduction")
        log_target = self.tcp.SizeVar(name="log_target")

        # sram数据 保存所有 cluster 共享的临时变量
        cluster_sram = self.tcp.Buffer(
            shape=(cluster_num,), name="cluster_sram", dtype=self.dtype, scope="sram"
        )
        tmp_sram = self.tcp.Buffer(
            shape=(4,), name="tmp_sram", dtype=self.dtype, scope="sram"
        )
        sum_each_cluster = self.tcp.Buffer(
            shape=(1,), name="sum_each_cluster", dtype=self.dtype, scope="sram"
        )

        # 分块处理相关变量
        start = self.tcp.Scalar(bangpy.int32, name="start")
        start.assign(task_id * (self.totalData // self.task_num))
        stop = self.tcp.Scalar(bangpy.int32, name="stop")

        # 计算sum 相关变量 后面进行修改
        computed_size = 128 // self.dtype_sz

        sumpool_size = data_calculated_each_time // computed_size
        sumpool_kernel_size = (
            data_calculated_each_time // computed_size // self.compute_row
        )

        sumvar = self.tcp.Scalar(name="sumvar", dtype=self.dtype)
        sumvar.assign(0)

        with self.tcp.for_range(begin=0, end=loop_num, stage=1) as i:
            # declare on-chip buffer
            buffer_input = self.tcp.Buffer(
                shape=(data_calculated_each_time,),
                name="INPUT_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_target = self.tcp.Buffer(
                shape=(data_calculated_each_time,),
                name="TARGET_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out = self.tcp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUTPUT_N",
                dtype=self.dtype,
                scope="nram",
            )

            # reduction = sum相关 buffer 变量
            sum_input_pool = buffer_out.reshape((computed_size, sumpool_size))
            temp_buffer = self.tcp.Buffer(
                shape=(computed_size * self.compute_row,),
                name="temp_buffer",
                dtype=self.dtype,
                scope="nram",
            )
            temp_buffer_pool = temp_buffer.reshape((computed_size, self.compute_row))

            sum_buf = self.tcp.Buffer(
                shape=(1,), name="sum_buf", dtype=self.dtype, scope="nram"
            )

            def compute_sum(in1, out_buf, out):
                self.tcp.sumpool(
                    temp_buffer_pool,
                    sum_input_pool,
                    (sumpool_kernel_size,),
                    (sumpool_kernel_size,),
                )
                with self.tcp.for_range(begin=0, end=self.compute_row) as i:
                    self.tcp.sum(
                        out_buf[0],
                        temp_buffer[i * computed_size : (i + 1) * computed_size],
                    )
                    out.assign(out + out_buf[0])

            def numCompute(ou, inp, tar, flag):
                with self.tcp.if_scope(flag == 0):
                    self.tcp.log(ou, tar, high_precision=False)
                    self.tcp.subtract(ou, ou, inp)
                    self.tcp.multiply(ou, tar, ou)
                with self.tcp.if_scope(flag == 1):
                    self.tcp.exp(ou, tar)
                    self.tcp.subtract(tar, tar, inp)
                    self.tcp.multiply(ou, ou, tar)

            def compute(ou, inp, tar, flag):
                numCompute(ou, inp, tar, flag)
                with self.tcp.if_scope(reduction != 0):
                    compute_sum(ou, sum_buf, sumvar)

            def computeTail(ou, inp, tar, flag):
                numCompute(ou, inp, tar, flag)
                with self.tcp.if_scope(self.batchLength >= 2 ** 20):
                    compute_sum(ou, sum_buf, sumvar)
                with self.tcp.else_scope():
                    with self.tcp.if_scope(reduction != 0):
                        with self.tcp.if_scope((stop - start) * self.dtype_sz < 128):
                            with self.tcp.for_range(begin=0, end=stop - start) as k:
                                sumvar.assign(sumvar + buffer_out[k])
                        with self.tcp.else_scope():
                            self.tcp.sum(buffer_out, buffer_out)
                            with self.tcp.for_range(
                                begin=0, end=(stop - start) * self.dtype_sz // 128
                            ) as j:
                                sumvar.assign(
                                    sumvar + buffer_out[j * (128 // self.dtype_sz)]
                                )

            with self.tcp.block("data_copy"):
                self.tcp.memcpy(
                    buffer_input[:data_calculated_each_time],
                    input[
                        start
                        + data_calculated_each_time * i : start
                        + data_calculated_each_time * (i + 1)
                    ],
                )
                self.tcp.memcpy(
                    buffer_target[:data_calculated_each_time],
                    target[
                        start
                        + data_calculated_each_time * i : start
                        + data_calculated_each_time * (i + 1)
                    ],
                )

            with self.tcp.block("compute"):
                compute(buffer_out, buffer_input, buffer_target, log_target)

            with self.tcp.block("data_copy"):
                with self.tcp.if_scope(reduction == 0):
                    self.tcp.memcpy(
                        out[
                            start
                            + data_calculated_each_time * i : start
                            + data_calculated_each_time * (i + 1)
                        ],
                        buffer_out[:data_calculated_each_time],
                    )

        with self.tcp.if_scope(
            data_calculated_each_task % data_calculated_each_time != 0
        ):
            start.assign(start + data_calculated_each_time * loop_num)
            stop.assign(start + data_calculated_each_task % data_calculated_each_time)
            # data copy
            self.tcp.memcpy(buffer_input[: stop - start], input[start:stop])
            self.tcp.memcpy(buffer_target[: stop - start], target[start:stop])
            # compute
            computeTail(buffer_out, buffer_input, buffer_target, log_target)
            # data copy
            with self.tcp.if_scope(reduction == 0):
                self.tcp.memcpy(out[start:stop], buffer_out[: stop - start])

        with self.tcp.if_scope(reduction != 0):
            # 每个task计算的sum总值保存在sumvar变量中
            sum_buf[0] = sumvar
            self.tcp.memcpy(tmp_sram[task_id % 4], sum_buf)
            self.tcp.sync_cluster()

            # 计算每个cluster的sum值
            sum_out = self.tcp.Scalar(name="sumout", dtype=self.dtype)
            sum_out.assign(0)
            with self.tcp.for_range(begin=0, end=4) as i:
                with self.tcp.block("compute"):
                    sum_out.assign(sum_out + tmp_sram[i])
            sum_each_cluster[0] = sum_out

            with self.tcp.if_scope(task_id == 0):
                cluster_sram[0] = sum_out

            # 传输不同cluster的sum值
            zeroid = self.tcp.Scalar(name="zeroid", dtype=self.dtype, value=0)
            with self.tcp.if_scope(cluster_id != 0):
                self.tcp.memcpy(cluster_sram[cluster_id], sum_each_cluster, zeroid)
            self.tcp.sync_all()

            total = self.tcp.Scalar(name="total", dtype=self.dtype)
            total.assign(0)
            with self.tcp.if_scope(cluster_id == 0):
                with self.tcp.for_range(begin=0, end=cluster_num) as i:
                    total.assign(total + cluster_sram[i])
                # reduction = mean
                with self.tcp.if_scope(reduction == 2):
                    total.assign(total / self.totalData)
                # reduction = batchmen
                with self.tcp.if_scope(reduction == 3):
                    total.assign(total / self.batchSize)
                # redunction = sum 直接输出total
                out[0] = total

        # build a executable module
        f = self.tcp.BuildBANG(
            inputs=[input, target, reduction, log_target],
            outputs=[out],
            kernel_name=KERNEL_NAME,
        )
        return f


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_kldivloss(dtype=None, target=None):
    # tasktype fixed in UNION1
    task_num = 64
    f = KlDivLoss(dtype, target, task_num).compute_body()
    return f
