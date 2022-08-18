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
"""KlDivLoss for bangpy tcp. The Kullback-Leibler divergence loss is
used to measure the distance between two distributions
"""

import bangpy
from bangpy import tcp
from bangpy.tcp.runtime import TaskType
from bangpy.platform.bang_config import TARGET

DTYPES = [bangpy.float32]
TARGET_LIST = ["mlu290"]
KERNEL_NAME = "kldivloss"
CORES_PER_CLUSTER = 4


class KlDivLoss(object):
    """Operator description:
        The Kullback-Leibler divergence loss is used to measure the distance
        between two distributions (discrete and continuous)

        log_target : if target has been logged(0:no / 1:yes)
            if log_target == 0 : out = target * (log(target) - input)
            if log_target == 1 : out = exp(target) * (target - input)

        reduction : perform reduction operation according to the reduction argument
            The kind of reduction:
                0 represents "none": out
                1 represents "sum" : sum(out)
                2 represents "mean" : sum(out) // dataSize
                3 represents "batchmean" : sum(out) // batchSize
    """

    def __init__(self, dtype, target, task_type):
        """Construct a new KlDivLoss class.

        Parameters
        ----------
        dtype : bangpy.DType
            The data type of input.

        target : str
            Target MLU device name.

        task_type : str
            The task type of runtime.

        Attributes
        ----------
        dtype : bangpy.DType
            The data type of input.

        target : str
            Target MLU device name.

        tcp : tcp.TCP
            TCP container.

        task_num : int
            The task number of runtime.

        batchSize : tcp.SizeVar
            The batch size of input.

        batchLength : tcp.SizeVar
            The length of each batch.

        totalData : int
            The data size of input.

        inputShape :
            The shape of input.

        outShape :
            The shape of output.

        nram_size : int
            The size of nram.

        dtype_sz : int
            The byte of each element.

        compute_row : int
            How many bytes equals 128 bits.

        single_buffer_size : int
            The size of single buffer.
        """
        # Check parameters.
        if not ((dtype in DTYPES) and (target in TARGET_LIST)):
            raise KeyError("please pass correct parameters.")

        self.dtype = dtype
        self.target = target
        self.tcp = tcp.TCP(target)
        self.task_num = task_type.value * CORES_PER_CLUSTER

        self.batchSize = self.tcp.SizeVar("batchSize")
        self.batchLength = self.tcp.SizeVar("batchLength")
        self.totalData = self.batchLength * self.batchSize
        self.inputShape = (self.batchSize, self.batchLength)
        self.outShape = (self.totalData,)

        self.nram_size = TARGET(self.target).nram_size
        self.dtype_sz = dtype.bytes
        self.compute_row = 128 // self.dtype_sz
        self.single_buffer_size = self.nram_size // 4 // self.dtype.bytes

        self.tcp.launch_task(self.task_num, 1, 1)
        self.tcp.launch_cluster(task_type.value)

    def compute_body(self):
        """Function description:
            Split data for each buffer, and calculate formula mentioned above
            If reduction is not equal 0, do the corresponding reduction operation
        """
        # calculate split strategy
        cluster_num = self.task_num // 4
        cluster_id = self.tcp.clusterId
        task_id = self.tcp.taskId

        # gets the data length to be calculated for each calculate
        data_calculated_each_time = self.single_buffer_size // self.dtype_sz
        # gets the number of cycles required for each task
        data_calculated_each_task = self.tcp.Scalar(
            bangpy.int32, name="data_calculated_each_task"
        )
        data_calculated_each_task.assign(self.totalData // self.task_num)
        with self.tcp.if_scope(task_id == self.task_num - 1):
            data_calculated_each_task.assign(
                self.totalData // self.task_num + (self.totalData) % self.task_num
            )

        # loop time of eacb task
        loop_num = data_calculated_each_task // data_calculated_each_time

        # declare on-chip buffer
        inputG = self.tcp.Buffer(
            shape=self.inputShape, name="inputG", dtype=self.dtype, scope="global"
        )

        inputG = inputG.flatten()

        targetG = self.tcp.Buffer(
            shape=self.inputShape, name="targetG", dtype=self.dtype, scope="global"
        )
        targetG = targetG.flatten()

        outG = self.tcp.Buffer(
            shape=self.outShape, name="outG", dtype=self.dtype, scope="global"
        )

        reduction = self.tcp.SizeVar(name="reduction")
        log_target = self.tcp.SizeVar(name="log_target")

        # declare sram buffer which saves temporary variables shared between clusters
        cluster_nram = self.tcp.Buffer(
            shape=(cluster_num,), name="cluster_nram", dtype=self.dtype, scope="nram"
        )
        tmp_sram = self.tcp.Buffer(
            shape=(4,), name="tmp_sram", dtype=self.dtype, scope="sram"
        )
        sum_each_cluster = self.tcp.Buffer(
            shape=(1,), name="sum_each_cluster", dtype=self.dtype, scope="sram"
        )

        # variables related to split
        start = self.tcp.Scalar(bangpy.int32, name="start")
        start.assign(task_id * (self.totalData // self.task_num))
        stop = self.tcp.Scalar(bangpy.int32, name="stop")

        # variables related to sum
        computed_size = 128 // self.dtype_sz

        sumpool_size = data_calculated_each_time // computed_size
        sumpool_kernel_size = (
            data_calculated_each_time // computed_size // self.compute_row
        )

        sumvar = self.tcp.Scalar(name="sumvar", dtype=self.dtype)
        sumvar.assign(0)

        with self.tcp.for_range(begin=0, end=loop_num, stage=1) as i:
            # declare nram buffer
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

            # variables related to reduction
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

            def compute_sum(out_buf, outG):
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
                    outG.assign(outG + out_buf[0])

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
                    compute_sum(sum_buf, sumvar)

            with self.tcp.block("data_copy"):
                self.tcp.memcpy(
                    buffer_input[:data_calculated_each_time],
                    inputG[
                        start
                        + data_calculated_each_time * i : start
                        + data_calculated_each_time * (i + 1)
                    ],
                )
                self.tcp.memcpy(
                    buffer_target[:data_calculated_each_time],
                    targetG[
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
                        outG[
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
            self.tcp.memcpy(buffer_input[: stop - start], inputG[start:stop])
            self.tcp.memcpy(buffer_target[: stop - start], targetG[start:stop])
            # compute
            numCompute(buffer_out, buffer_input, buffer_target, log_target)
            with self.tcp.if_scope(self.batchLength >= 2 ** 20):
                compute_sum(sum_buf, sumvar)
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

            # data copy
            with self.tcp.if_scope(reduction == 0):
                self.tcp.memcpy(outG[start:stop], buffer_out[: stop - start])

        with self.tcp.if_scope(reduction != 0):
            sum_buf[0] = sumvar
            tmp_sram[task_id % 4] = sum_buf

            # calculate sum of each cluster
            sum_out = self.tcp.Scalar(name="sumout", dtype=self.dtype)
            sum_out.assign(0)
            with self.tcp.for_range(begin=0, end=4) as i:
                sum_out.assign(sum_out + tmp_sram[i])
            sum_each_cluster[0] = sum_out

            outG[cluster_id] = sum_each_cluster[0]
            self.tcp.sync_all()

            with self.tcp.for_range(begin=0, end=cluster_num) as i:
                cluster_nram[i] = outG[i]

            total = self.tcp.Scalar(name="total", dtype=self.dtype)
            total.assign(0)
            with self.tcp.if_scope(cluster_id == 0):
                with self.tcp.for_range(begin=0, end=cluster_num) as i:
                    total.assign(total + cluster_nram[i])
                # reduction = mean
                with self.tcp.if_scope(reduction == 2):
                    total.assign(total / self.totalData)
                # reduction = batchmen
                with self.tcp.if_scope(reduction == 3):
                    total.assign(total / self.batchSize)
                outG[0] = total

        # build a executable module
        f = self.tcp.BuildBANG(
            inputs=[inputG, targetG, reduction, log_target],
            outputs=[outG],
            kernel_name=KERNEL_NAME,
        )
        return f


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_kldivloss(dtype=None, target=None):
    task_type = TaskType.UNION1
    f = KlDivLoss(dtype, target, task_type).compute_body()
    return f
