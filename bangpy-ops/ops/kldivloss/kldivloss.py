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
from bangpy.script import ty, build_module
from bangpy.platform.bang_config import TARGET

DTYPES = [bangpy.float16, bangpy.float32]
# TARGET_LIST = ["mlu290"]
TARGET_LIST = ["mlu370-s4"]
KERNEL_NAME = "kldivloss"


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

    def __init__(
        self, task_num: ty.int32, nram_size: ty.int32, dtype: ty.string
    ) -> None:
        """Construct a new KlDivLoss class.
        Parameters
        ----------
        task_num : bangpy.common.DType
            The task number of runtime.
        nram_size : bangpy.common.DType
            The size of nram.
        dtype : bangpy.common.DType
            The data type of input.
        Attributes
        ----------
        task_num : int
            The task number of runtime.
        nram_size : bangpy.common.DType
            The size of nram.
        dtype : bangpy.DType
            The data type of input.
        dtype_sz : int
            The byte of each element.
        single_buffer_size : int
            The size of single buffer.
        """
        self.task_num = task_num
        self.nram_size = nram_size
        self.dtype = dtype
        if self.dtype == "float16":
            self.dtype_sz = 2
        else:
            self.dtype_sz = 4  # "float32"
        self.single_buffer_size = (((self.nram_size - 4 * 1024) // 8) // (128 * 64)) * (
            (128 * 64)  # todo: why 128 * 64 is ok, but 128 causes errors
        )
        # NRAM is divided into 6 main parts
        # 4 * 1024：reserved for other variables
        # # Check parameters.
        # if not ((self.dtype in DTYPES) and (self.target in TARGET_LIST)):
        #     raise KeyError("please pass correct parameters.")
        # todo: check parameters.

    # for reduction
    def compute_sum(
        self,
        sumvar: ty.Buffer("nram"),
        sum_buf: ty.Buffer("nram"),
        temp_buffer_pool: ty.Buffer("nram"),
        temp_buffer: ty.Buffer("nram"),
        sum_input_pool: ty.Buffer("nram"),
        sumpool_kernel_size: ty.int32,
        computed_size: ty.int32,
    ) -> None:

        tcp.sumpool(
            temp_buffer_pool,
            sum_input_pool,
            (sumpool_kernel_size,),
            (sumpool_kernel_size,),
        )
        temp_buffer = temp_buffer_pool.reshape((computed_size,))
        tcp.sum(sum_buf, temp_buffer)
        sumvar = sumvar + tcp.cast(sum_buf[0], "float32")

    # for calculation
    def numCompute(
        self,
        ou: ty.Buffer("nram"),
        inp: ty.Buffer("nram"),
        tar: ty.Buffer("nram"),
        flag: ty.int32,
        temp1: ty.Buffer("nram"),
    ) -> None:
        if flag == 0:
            if self.dtype == "float16":
                tcp.type_convert(temp1, tar, 0)
                tcp.log(temp1, temp1)
                tcp.type_convert(ou, temp1, 0, "rd")
            else:
                tcp.log(ou, tar)  # high_precision=False
            tcp.subtract(ou, ou, inp)
            tcp.multiply(ou, tar, ou)
        else:  # flag == 1:
            tcp.exp(ou, tar, "exp_less_0")  # todo: delete "exp_less_0"?
            tcp.subtract(tar, tar, inp)
            tcp.multiply(ou, ou, tar)

    def main(
        self,
        inputG: ty.handle,
        targetG: ty.handle,
        outG: ty.handle,
        batchSize: ty.int32,
        batchLength: ty.int32,
        reduction: ty.int32,
        log_target: ty.int32,
        reduction_resultG: ty.handle,
    ) -> None:

        # declare I/O buffer
        totalData = batchSize * batchLength
        inputG = tcp.match_buffer(inputG, [batchSize, batchLength], dtype=self.dtype)
        targetG = tcp.match_buffer(targetG, [batchSize, batchLength], dtype=self.dtype)
        outG = tcp.match_buffer(outG, [totalData], dtype=self.dtype)
        reduction_resultG = tcp.match_buffer(reduction_resultG, [1], dtype="float32")
        data_tempG = tcp.alloc_buffer(
            [64], dtype="float32", scope="gdram"
        )  # save the sum of each core, we assume that task_num <= 64

        inputG = inputG.flatten()
        targetG = targetG.flatten()

        tgt = tcp.target()

        # calculate split strategy
        data_calculated_each_time = self.single_buffer_size // self.dtype_sz
        data_calculated_each_task = totalData // self.task_num

        # loop time of each task
        loop_num = data_calculated_each_task // data_calculated_each_time

        # variables related to sum
        computed_size = 128 // self.dtype_sz
        sumpool_size = data_calculated_each_time // computed_size
        sumpool_kernel_size = sumpool_size

        for cluster_id in tcp.thread_binding(0, tgt.cluster_num, thread="blockIdx.x"):
            for core_id in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):
                # parallel region
                task_id = cluster_id * tgt.core_num + core_id
                if task_id == (self.task_num - 1):
                    data_calculated_each_task = totalData // self.task_num + (
                        totalData % self.task_num
                    )
                # variables related to split
                start = task_id * (totalData // self.task_num)
                sumvar = 0.0
                sum_buf = tcp.alloc_buffer(
                    [64], dtype=self.dtype, scope="nram"
                )  # Save temporary result and tcp.assign needs to be 64-element aligned
                tcp.assign(sum_buf, tcp.cast(0, self.dtype))
                temp_buffer_pool = tcp.alloc_buffer(
                    [1, computed_size], dtype=self.dtype, scope="nram",
                )
                temp_buffer = tcp.alloc_buffer(
                    [1, computed_size], dtype=self.dtype, scope="nram",
                )

                for i in range(loop_num, pipeline=True):
                    buffer_input = tcp.alloc_buffer(
                        [data_calculated_each_time], dtype=self.dtype, scope="nram"
                    )
                    buffer_target = tcp.alloc_buffer(
                        [data_calculated_each_time], dtype=self.dtype, scope="nram"
                    )
                    buffer_out = tcp.alloc_buffer(
                        [data_calculated_each_time], dtype=self.dtype, scope="nram"
                    )
                    temp1 = tcp.alloc_buffer(
                        [data_calculated_each_time], dtype="float32", scope="nram"
                    )

                    with tcp.block("data_copy"):
                        tcp.memcpy(
                            buffer_input[:data_calculated_each_time],
                            inputG[
                                start
                                + data_calculated_each_time * i : start
                                + data_calculated_each_time * (i + 1)
                            ],
                        )
                        tcp.memcpy(
                            buffer_target[:data_calculated_each_time],
                            targetG[
                                start
                                + data_calculated_each_time * i : start
                                + data_calculated_each_time * (i + 1)
                            ],
                        )

                    with tcp.block("compute"):
                        self.numCompute(
                            buffer_out, buffer_input, buffer_target, log_target, temp1
                        )
                        if reduction != 0:
                            sum_input_pool = buffer_out.reshape(
                                (sumpool_size, computed_size)
                            )
                            self.compute_sum(
                                sumvar,
                                sum_buf,
                                temp_buffer_pool,
                                temp_buffer,
                                sum_input_pool,
                                sumpool_kernel_size,
                                computed_size,
                            )

                    with tcp.block("data_copy"):
                        if reduction == 0:
                            tcp.memcpy(
                                outG[
                                    start
                                    + data_calculated_each_time * i : start
                                    + data_calculated_each_time * (i + 1)
                                ],
                                buffer_out[:data_calculated_each_time],
                            )

                # data_rem_n
                data_rem_n = data_calculated_each_task % data_calculated_each_time
                if data_rem_n != 0:
                    start = start + data_calculated_each_time * loop_num
                    stop = start + data_rem_n
                    buffer_input = tcp.alloc_buffer(
                        [data_calculated_each_time], dtype=self.dtype, scope="nram"
                    )
                    buffer_target = tcp.alloc_buffer(
                        [data_calculated_each_time], dtype=self.dtype, scope="nram"
                    )
                    buffer_out = tcp.alloc_buffer(
                        [data_calculated_each_time], dtype=self.dtype, scope="nram"
                    )
                    temp1 = tcp.alloc_buffer(
                        [data_calculated_each_time], dtype="float32", scope="nram"
                    )

                    # data copy
                    tcp.memcpy(buffer_input[: stop - start], inputG[start:stop])
                    tcp.memcpy(buffer_target[: stop - start], targetG[start:stop])

                    # compute
                    self.numCompute(
                        buffer_out, buffer_input, buffer_target, log_target, temp1
                    )

                    # reduction
                    if reduction != 0:
                        if (  # calculate it by sumpool when the size is large
                            data_rem_n > 128 * 1  # 128 * x | 128
                        ):  # todo: what is the best value of x?
                            data_rem_n_temp = (data_rem_n // (128 * 1)) * (128 * 1)
                            sumpool_kernel_size = data_rem_n_temp // computed_size
                            tcp.sumpool(
                                temp_buffer_pool,
                                buffer_out[
                                    0 : sumpool_kernel_size * computed_size
                                ].reshape((sumpool_kernel_size, computed_size)),
                                (sumpool_kernel_size,),
                                (sumpool_kernel_size,),
                            )
                            temp_buffer = temp_buffer_pool.reshape((computed_size,))
                            tcp.sum(sum_buf, temp_buffer)
                            sumvar = sumvar + tcp.cast(sum_buf[0], "float32")
                            if (data_rem_n % computed_size) > 0:
                                for k in range(begin=data_rem_n_temp, end=data_rem_n):
                                    sumvar = sumvar + tcp.cast(buffer_out[k], "float32")
                        else:  # calculate it one by one when the size is small
                            for k in range(begin=0, end=data_rem_n):
                                sumvar = sumvar + tcp.cast(buffer_out[k], "float32")

                    # data copy
                    if reduction == 0:
                        tcp.memcpy(outG[start:stop], buffer_out[: stop - start])

                if reduction != 0:
                    # todo: tcp.sync_all() is not supported
                    data_tempG[task_id % self.task_num] = sumvar
                    if task_id == 0:
                        total = tcp.alloc_buffer([64], dtype="float32", scope="nram")
                        tcp.memcpy(total, data_tempG)
                        for i in range(begin=1, end=self.task_num):  # 1 to 63
                            total[0] = total[0] + total[i]
                        # reduction = mean
                        if reduction == 2:
                            total[0] = total[0] / totalData
                        # reduction = batchmean
                        if reduction == 3:
                            total[0] = total[0] / batchSize
                        tcp.memcpy(reduction_resultG[0], total[0])


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_kldivloss(dtype=None, target=None):
    task_num = TARGET(target).cluster_num * TARGET(target).core_num
    nram_size = TARGET(target).nram_size
    f = build_module.build(
        KlDivLoss(task_num, nram_size, dtype.name), target_tag=target, name=KERNEL_NAME,
    )
    return f
