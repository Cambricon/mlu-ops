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
"""Cosine_similarity operator implementation using BANGPy TCP API."""
import numpy as np

import bangpy
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType
import os
import shutil

DTYPES = [bangpy.float32, bangpy.float16]
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]
KERNEL_NAME = "cosine_similarity"


class Cosine_similarity(object):
    """Operator description:
    compute cosine similarity of two given tensors
    """

    def __init__(self, dtype, target, task_num, stage):
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.stage = stage
        self.bp = tcp.TCP(target)
        self.dim = self.bp.Var(name="dim", dtype=bangpy.int32)
        self.dim_0 = self.bp.SizeVar("dim_0")
        self.dim_1 = self.bp.SizeVar("dim_1")
        self.dim_2 = self.bp.SizeVar("dim_2")
        self.dim_3 = self.bp.SizeVar("dim_3")

        self.length = self.dim_0 * self.dim_1 * self.dim_2 * self.dim_3
        self.nram_size = TARGET(target).nram_size
        self.dtype_sz = dtype.bytes
        self.single_buffer_size = (self.nram_size) // 8
        self.bp.launch_task(self.task_num, 1, 1)

    def compute_body(self):
        # calculate basic data
        data_calculated_each_task = self.length
        loop_num = data_calculated_each_task * self.dtype_sz // self.single_buffer_size
        data_calculated_each_time = self.single_buffer_size // self.dtype_sz
        remain = (data_calculated_each_task * self.dtype_sz) % self.single_buffer_size

        buffer_in0 = self.bp.Buffer(
            shape=(self.dim_0, self.dim_1, self.dim_2, self.dim_3),
            name="INPUT0",
            dtype=self.dtype,
            scope="global",
        )
        buffer_in1 = self.bp.Buffer(
            shape=(self.dim_0, self.dim_1, self.dim_2, self.dim_3),
            name="INPUT1",
            dtype=self.dtype,
            scope="global",
        )

        task_id = self.bp.taskId
        dim = self.dim
        buffer_reshape0 = buffer_in0.reshape((self.length,))
        buffer_reshape1 = buffer_in1.reshape((self.length,))

        with self.bp.if_scope(dim == 0):
            dim_h = 1
            dim_m = self.dim_0
            dim_l = self.dim_1 * self.dim_2 * self.dim_3
            total_after = dim_h * dim_l

            buffer_out0 = self.bp.Buffer(
                shape=(self.dim_1, self.dim_2, self.dim_3),
                name="OUTPUT0",
                dtype=self.dtype,
                scope="global",
            )
            buffer_out0 = buffer_out0.reshape((1, 1, total_after))
            buffer_in0_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="IN0_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_in1_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="IN1_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_mul = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="MUL_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUTN",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out0_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUT0N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out1_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUT1N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out_final = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUTFINAL",
                dtype=self.dtype,
                scope="nram",
            )
            self.bp.assign(buffer_in0_n, 0)
            self.bp.assign(buffer_in1_n, 0)
            self.bp.assign(buffer_mul, 0)
            self.bp.assign(buffer_out0_n, 0)
            self.bp.assign(buffer_out1_n, 0)
            self.bp.assign(buffer_out_final, 0)
            self.bp.memcpy(buffer_in0_n[0 : self.length], buffer_reshape0)
            self.bp.memcpy(buffer_in1_n[0 : self.length], buffer_reshape1)

            self.bp.multiply(buffer_mul, buffer_in0_n, buffer_in1_n)
            self.bp.square(buffer_in0_n, buffer_in0_n)
            self.bp.square(buffer_in1_n, buffer_in1_n)

            mul_reshape = buffer_mul[0 : self.length].reshape((dim_h, dim_m, dim_l))
            in_reshape0 = buffer_in0_n[0 : self.length].reshape((dim_h, dim_m, dim_l))
            in_reshape1 = buffer_in1_n[0 : self.length].reshape((dim_h, dim_m, dim_l))
            buffer_out_n = buffer_out_n[0:total_after].reshape((dim_h, 1, dim_l))
            buffer_out0_n = buffer_out0_n[0:total_after].reshape((dim_h, 1, dim_l))
            buffer_out1_n = buffer_out1_n[0:total_after].reshape((dim_h, 1, dim_l))
            buffer_out_final = buffer_out_final[0:total_after].reshape(
                (dim_h, 1, dim_l)
            )
            self.bp.sumpool(buffer_out_n, mul_reshape, (dim_h, dim_m), (1, 1))
            self.bp.sumpool(buffer_out0_n, in_reshape0, (dim_h, dim_m), (1, 1))
            self.bp.sumpool(buffer_out1_n, in_reshape1, (dim_h, dim_m), (1, 1))
            self.bp.sqrt(buffer_out0_n, buffer_out0_n)
            self.bp.sqrt(buffer_out1_n, buffer_out1_n)

            self.bp.multiply(buffer_out0_n, buffer_out0_n, buffer_out1_n)
            self.bp.divide(buffer_out_final, buffer_out_n, buffer_out0_n)
            self.bp.memcpy(buffer_out0, buffer_out_final)

            buffer_out0 = buffer_out0.reshape((self.dim_1, self.dim_2, self.dim_3))

        with self.bp.if_scope(dim == 1):

            dim_h = self.dim_0
            dim_m = self.dim_1
            dim_l = self.dim_2 * self.dim_3
            lim_h = data_calculated_each_time // (dim_m * dim_l)
            total_after = lim_h * dim_l

            buffer_out1 = self.bp.Buffer(
                shape=(self.dim_0, self.dim_2, self.dim_3),
                name="OUTPUT1",
                dtype=self.dtype,
                scope="global",
            )
            buffer_out1 = buffer_out1.reshape((dim_h, 1, dim_l))
            buffer_in0_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="IN0_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_in1_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="IN1_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_mul = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="MUL_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUTN",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out0_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUT0N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out1_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUT1N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out_final = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUTFINAL",
                dtype=self.dtype,
                scope="nram",
            )
            self.bp.assign(buffer_in0_n, 0)
            self.bp.assign(buffer_in1_n, 0)
            self.bp.assign(buffer_mul, 0)
            self.bp.assign(buffer_out0_n, 0)
            self.bp.assign(buffer_out1_n, 0)
            self.bp.assign(buffer_out_n, 0)
            self.bp.assign(buffer_out_final, 0)
            with self.bp.for_range(0, loop_num) as i:
                start = i * data_calculated_each_time
                end = start + data_calculated_each_time
                self.bp.memcpy(buffer_in0_n, buffer_reshape0[start:end])
                self.bp.memcpy(buffer_in1_n, buffer_reshape1[start:end])

                self.bp.multiply(buffer_mul, buffer_in0_n, buffer_in1_n)
                self.bp.square(buffer_in0_n, buffer_in0_n)
                self.bp.square(buffer_in1_n, buffer_in1_n)
                mul_reshape = buffer_mul.reshape((lim_h, dim_m, dim_l))
                in_reshape0 = buffer_in0_n.reshape((lim_h, dim_m, dim_l))
                in_reshape1 = buffer_in1_n.reshape((lim_h, dim_m, dim_l))

                buffer_out_n = buffer_out_n[0:total_after].reshape((lim_h, 1, dim_l))
                buffer_out0_n = buffer_out0_n[0:total_after].reshape((lim_h, 1, dim_l))
                buffer_out1_n = buffer_out1_n[0:total_after].reshape((lim_h, 1, dim_l))
                buffer_out_final = buffer_out_final[0:total_after].reshape(
                    (lim_h, 1, dim_l)
                )
                self.bp.sumpool(buffer_out_n, mul_reshape, (1, dim_m), (1, 1))
                self.bp.sumpool(buffer_out0_n, in_reshape0, (1, dim_m), (1, 1))
                self.bp.sumpool(buffer_out1_n, in_reshape1, (1, dim_m), (1, 1))
                self.bp.sqrt(buffer_out0_n, buffer_out0_n)
                self.bp.sqrt(buffer_out1_n, buffer_out1_n)
                self.bp.multiply(buffer_out0_n, buffer_out0_n, buffer_out1_n)
                self.bp.divide(buffer_out_final, buffer_out_n, buffer_out0_n)
                self.bp.memcpy(
                    buffer_out1[i * lim_h : (i + 1) * lim_h], buffer_out_final
                )

            buffer_out1 = buffer_out1.reshape((self.dim_0, self.dim_2, self.dim_3))

        with self.bp.if_scope(dim == 2):

            dim_h = self.dim_0 * self.dim_1
            dim_m = self.dim_2
            dim_l = self.dim_3
            lim_h = data_calculated_each_time // (dim_m * dim_l)
            total_after = lim_h * dim_l

            buffer_out2 = self.bp.Buffer(
                shape=(self.dim_0, self.dim_1, self.dim_3),
                name="OUTPUT2",
                dtype=self.dtype,
                scope="global",
            )
            buffer_out2 = buffer_out2.reshape((dim_h, 1, dim_l))
            buffer_in0_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="IN0_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_in1_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="IN1_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_mul = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="MUL_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUTN",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out0_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUT0N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out1_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUT1N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out_final = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUTFINAL",
                dtype=self.dtype,
                scope="nram",
            )
            self.bp.assign(buffer_in0_n, 0)
            self.bp.assign(buffer_in1_n, 0)
            self.bp.assign(buffer_mul, 0)
            self.bp.assign(buffer_out0_n, 0)
            self.bp.assign(buffer_out1_n, 0)
            self.bp.assign(buffer_out_final, 0)
            with self.bp.for_range(0, loop_num) as i:
                start = i * data_calculated_each_time
                end = start + data_calculated_each_time
                self.bp.memcpy(buffer_in0_n, buffer_reshape0[start:end])
                self.bp.memcpy(buffer_in1_n, buffer_reshape1[start:end])

                self.bp.multiply(buffer_mul, buffer_in0_n, buffer_in1_n)
                self.bp.square(buffer_in0_n, buffer_in0_n)
                self.bp.square(buffer_in1_n, buffer_in1_n)
                mul_reshape = buffer_mul.reshape((lim_h, dim_m, dim_l))
                in_reshape0 = buffer_in0_n.reshape((lim_h, dim_m, dim_l))
                in_reshape1 = buffer_in1_n.reshape((lim_h, dim_m, dim_l))

                buffer_out_n = buffer_out_n[0:total_after].reshape((lim_h, 1, dim_l))
                buffer_out0_n = buffer_out0_n[0:total_after].reshape((lim_h, 1, dim_l))
                buffer_out1_n = buffer_out1_n[0:total_after].reshape((lim_h, 1, dim_l))
                buffer_out_final = buffer_out_final[0:total_after].reshape(
                    (lim_h, 1, dim_l)
                )
                self.bp.sumpool(buffer_out_n, mul_reshape, (1, dim_m), (1, 1))
                self.bp.sumpool(buffer_out0_n, in_reshape0, (1, dim_m), (1, 1))
                self.bp.sumpool(buffer_out1_n, in_reshape1, (1, dim_m), (1, 1))
                self.bp.sqrt(buffer_out0_n, buffer_out0_n)
                self.bp.sqrt(buffer_out1_n, buffer_out1_n)
                self.bp.multiply(buffer_out0_n, buffer_out0_n, buffer_out1_n)
                self.bp.divide(buffer_out_final, buffer_out_n, buffer_out0_n)
                self.bp.memcpy(
                    buffer_out2[i * lim_h : (i + 1) * lim_h], buffer_out_final
                )

            buffer_out2 = buffer_out2.reshape((self.dim_0, self.dim_1, self.dim_3))

        with self.bp.if_scope(dim == 3):
            dim_h = self.dim_0 * self.dim_1 * self.dim_2
            dim_m = self.dim_3
            dim_l = 1
            lim_h = data_calculated_each_time // (dim_m * dim_l)
            total = dim_h * dim_l

            buffer_out3 = self.bp.Buffer(
                shape=(self.dim_0, self.dim_1, self.dim_2),
                name="OUTPUT3",
                dtype=self.dtype,
                scope="global",
            )
            buffer_out3 = buffer_out3.reshape((total,))
            buffer_in0_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="IN0_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_in1_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="IN1_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_mul = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="MUL_N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUTN",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out0_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUT0N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out1_n = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUT1N",
                dtype=self.dtype,
                scope="nram",
            )
            buffer_out_final = self.bp.Buffer(
                shape=(data_calculated_each_time,),
                name="OUTFINAL",
                dtype=self.dtype,
                scope="nram",
            )
            self.bp.assign(buffer_in0_n, 0)
            self.bp.assign(buffer_in1_n, 0)
            self.bp.assign(buffer_mul, 0)
            self.bp.assign(buffer_out0_n, 0)
            self.bp.assign(buffer_out1_n, 0)
            self.bp.assign(buffer_out_final, 0)
            with self.bp.for_range(0, loop_num) as i:
                start = i * data_calculated_each_time
                end = start + data_calculated_each_time
                self.bp.memcpy(buffer_in0_n, buffer_reshape0[start:end])
                self.bp.memcpy(buffer_in1_n, buffer_reshape1[start:end])
                self.bp.multiply(buffer_mul, buffer_in0_n, buffer_in1_n)
                self.bp.square(buffer_in0_n, buffer_in0_n)
                self.bp.square(buffer_in1_n, buffer_in1_n)
                mul_reshape = buffer_mul.reshape((lim_h, dim_m))
                in_reshape0 = buffer_in0_n.reshape((lim_h, dim_m))
                in_reshape1 = buffer_in1_n.reshape((lim_h, dim_m))

                buffer_out_n = buffer_out_n[0:lim_h]
                buffer_out0_n = buffer_out0_n[0:lim_h]
                buffer_out1_n = buffer_out1_n[0:lim_h]
                buffer_out_final = buffer_out_final[0:lim_h]

                with self.bp.for_range(0, lim_h) as j:
                    total_sum = self.bp.Scalar(
                        name="total_sum", dtype=self.dtype, value=0
                    )
                    total_sum0 = self.bp.Scalar(
                        name="total_sum0", dtype=self.dtype, value=0
                    )
                    total_sum1 = self.bp.Scalar(
                        name="total_sum1", dtype=self.dtype, value=0
                    )
                    with self.bp.for_range(0, dim_m) as k:
                        total_sum += mul_reshape[j][k]
                        total_sum0 += in_reshape0[j][k]
                        total_sum1 += in_reshape1[j][k]
                    buffer_out_n[j] = total_sum
                    buffer_out0_n[j] = total_sum0
                    buffer_out1_n[j] = total_sum1

                buffer_out_n = buffer_out_n.reshape((lim_h,))
                buffer_out0_n = buffer_out0_n.reshape((lim_h,))
                buffer_out1_n = buffer_out1_n.reshape((lim_h,))
                buffer_out_final = buffer_out_final.reshape((lim_h,))
                self.bp.sqrt(buffer_out0_n, buffer_out0_n)
                self.bp.sqrt(buffer_out1_n, buffer_out1_n)

                self.bp.multiply(buffer_out0_n, buffer_out0_n, buffer_out1_n)
                self.bp.divide(buffer_out_final, buffer_out_n, buffer_out0_n)
                self.bp.memcpy(
                    buffer_out3[i * lim_h : (i + 1) * lim_h], buffer_out_final
                )

            buffer_out3 = buffer_out3.reshape((self.dim_0, self.dim_1, self.dim_2))

        f = self.bp.BuildBANG(
            inputs=[buffer_in0, buffer_in1, dim, buffer_out0, buffer_out1, buffer_out2],
            outputs=[buffer_out3],
            kernel_name=KERNEL_NAME,
            dump_ir=True,
        )
        return f


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_cosine_similarity(dtype=None, target=None):
    # tasktype fixed in UNION4
    task_type = TaskType.UNION4
    task_num = 1
    stage = 1
    f = Cosine_similarity(dtype, target, task_num, stage).compute_body()
    return f
