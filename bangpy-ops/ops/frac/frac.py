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
"""Frac operator implementation using BANGPy TCP API."""
# pylint: skip-file
import bangpy
from bangpy import tcp
from bangpy.script import ty, build_module
from bangpy.platform.bang_config import TARGET

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu370-s4", "mlu290"]
KERNEL_NAME = "frac"


class Frac(object):
    """Operator description:
    Calculate the fraction part of the given data
    """

    def __init__(
        self, cluster_num: ty.int32, dtype: ty.string, dtype_sz: ty.int32
    ) -> None:
        self.dtype = dtype
        self.cluster_num = cluster_num
        self.dtype_sz = dtype_sz

    def compute_body_half(
        self,
        buffer_in_n: ty.Buffer("nram"),
        buffer_abs: ty.Buffer("nram"),
        buffer_floor: ty.Buffer("nram"),
        buffer_floor_after: ty.Buffer("nram"),
        buffer_sgn: ty.Buffer("nram"),
        buffer_tem: ty.Buffer("nram"),
    ) -> None:
        """The body of frac function"""

        tcp.type_convert(buffer_tem,buffer_in_n, 0, "tz") #h2f
        tcp.abs(buffer_abs, buffer_tem)#f
        tcp.type_convert(buffer_floor, buffer_abs, 0, "tz")#  f2i
        tcp.type_convert(buffer_floor_after, buffer_floor, 0, "tz")#i2f
        tcp.sign(buffer_sgn, buffer_tem) #f
        tcp.multiply(buffer_abs, buffer_floor_after, buffer_sgn)# f
        
        tcp.subtract(buffer_tem, buffer_tem, buffer_abs) #f
        tcp.type_convert(buffer_in_n,buffer_tem, 0, "rd")#f2h

    def compute_body_float(
        self,
        buffer_in_n: ty.Buffer("nram"),
        buffer_abs: ty.Buffer("nram"),
        buffer_floor: ty.Buffer("nram"),
        buffer_floor_after: ty.Buffer("nram"),
        buffer_sgn: ty.Buffer("nram"),
        buffer_tem: ty.Buffer("nram"),
    ) -> None:
        """The body of frac function"""

        tcp.memcpy(buffer_tem, buffer_in_n)
        tcp.abs(buffer_abs, buffer_tem)#f
        tcp.type_convert(buffer_floor, buffer_abs, 0, "tz")#  f2i
        tcp.type_convert(buffer_floor_after, buffer_floor, 0, "tz")#i2f
        tcp.sign(buffer_sgn, buffer_tem) #f
        tcp.multiply(buffer_abs, buffer_floor_after, buffer_sgn)# f
        
        tcp.subtract(buffer_in_n, buffer_tem, buffer_abs) #f

    def main(
        self,
        buffer_in: ty.handle,
        buffer_out: ty.handle,
        dim0: ty.int32,
        dim1: ty.int32,
        dim2: ty.int32,
        dim3: ty.int32,
    ) -> None:
        """The main part of frac function"""

        buffer_in = tcp.match_buffer(
            buffer_in, [dim0, dim1, dim2, dim3], dtype=self.dtype
        )
        buffer_out = tcp.match_buffer(
            buffer_out, [dim0, dim1, dim2, dim3], dtype=self.dtype
        )
        tgt = tcp.target()

        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.length = self.dim0 * self.dim1 * self.dim2 * self.dim3

        self.task_num = self.cluster_num * tgt.core_num
        self.single_buffer_size = (tgt.nram_size) // 8
        task_id = 0
        data_calculated_each_task = self.length // self.task_num
        data_remain = self.length % self.task_num
        loop_num = data_calculated_each_task * self.dtype_sz * (10//5) // self.single_buffer_size 
        data_calculated_each_time = self.single_buffer_size // (self.dtype_sz * 10//5)
        each_task_remain = data_calculated_each_task % data_calculated_each_time

        buffer_in = buffer_in.reshape((self.length,))
        buffer_out = buffer_out.reshape((self.length,))


        for cluster_id in tcp.thread_binding(0, self.cluster_num, thread="blockIdx.x"):
            for core_id in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):
                task_id = cluster_id * tgt.core_num + core_id
                for i in range(0, loop_num, pipeline=True):
                    buffer_in_n = tcp.alloc_buffer(
                        [data_calculated_each_time,], dtype=self.dtype, scope="nram"
                    )
                    buffer_out_n = tcp.alloc_buffer(
                        [data_calculated_each_time,], dtype=self.dtype, scope="nram"
                    )
                    buffer_abs = tcp.alloc_buffer(
                        shape=[data_calculated_each_time,], dtype="float32", scope="nram",
                    )
                    buffer_floor = tcp.alloc_buffer(
                        shape=[data_calculated_each_time,], dtype="int16", scope="nram",
                    )
                    buffer_floor_after = tcp.alloc_buffer(
                        shape=[data_calculated_each_time,], dtype="float32", scope="nram",
                    )
                    buffer_sgn = tcp.alloc_buffer(
                        shape=[data_calculated_each_time,], dtype="float32", scope="nram",
                    )
                    buffer_tem = tcp.alloc_buffer(
                        shape=[data_calculated_each_time,], dtype="float32", scope="nram",
                    )
                    with tcp.block("data_copy"):
                        start = (
                            task_id * data_calculated_each_task
                            + i * data_calculated_each_time
                        )
                        stop = start + data_calculated_each_time
                        tcp.memcpy(buffer_in_n, buffer_in[start:stop])
                    with tcp.block("compute"):
                        if self.dtype == "float32":
                            self.compute_body_float(
                                buffer_in_n,
                                buffer_abs,
                                buffer_floor,
                                buffer_floor_after,
                                buffer_sgn,
                                buffer_tem,
                            )
                        elif self.dtype == "float16":
                            self.compute_body_half(
                                buffer_in_n,
                                buffer_abs,
                                buffer_floor,
                                buffer_floor_after,
                                buffer_sgn,
                                buffer_tem,
                            )
                            
                    with tcp.block("data_copy"):
                        start = (
                            task_id * data_calculated_each_task
                            + i * data_calculated_each_time
                        )
                        stop = start + data_calculated_each_time
                        tcp.memcpy(buffer_out[start:stop], buffer_in_n)

                buffer_in_n = tcp.alloc_buffer(
                    [data_calculated_each_time,], dtype=self.dtype, scope="nram"
                )
                buffer_out_n = tcp.alloc_buffer(
                    [data_calculated_each_time,], dtype=self.dtype, scope="nram"
                )
                buffer_abs = tcp.alloc_buffer(
                    shape=[data_calculated_each_time,], dtype="float32", scope="nram",
                )
                buffer_floor = tcp.alloc_buffer(
                    shape=[data_calculated_each_time,], dtype="int16", scope="nram",
                )
                buffer_floor_after = tcp.alloc_buffer(
                    shape=[data_calculated_each_time,], dtype="float32", scope="nram",
                )
                buffer_sgn = tcp.alloc_buffer(
                    shape=[data_calculated_each_time,], dtype="float32", scope="nram",
                )
                buffer_tem = tcp.alloc_buffer(
                    shape=[data_calculated_each_time,], dtype="float32", scope="nram",
                )
                tcp.assign(buffer_in_n, 0)
                tcp.assign(buffer_out_n, 0)
                if each_task_remain != 0:
                    with tcp.block("data_copy"):
                        start = (
                            task_id * data_calculated_each_task
                            + loop_num * data_calculated_each_time
                        )
                        stop = start + each_task_remain
                        tcp.memcpy(
                            buffer_in_n[0:each_task_remain], buffer_in[start:stop]
                        )
                    with tcp.block("compute"):
                        if self.dtype == "float32":
                            self.compute_body_float(
                                buffer_in_n,
                                buffer_abs,
                                buffer_floor,
                                buffer_floor_after,
                                buffer_sgn,
                                buffer_tem,
                            )
                        elif self.dtype == "float16":
                            self.compute_body_half(
                                buffer_in_n,
                                buffer_abs,
                                buffer_floor,
                                buffer_floor_after,
                                buffer_sgn,
                                buffer_tem,
                            )
                    with tcp.block("data_copy"):
                        start = (
                            task_id * data_calculated_each_task
                            + loop_num * data_calculated_each_time
                        )
                        stop = start + each_task_remain
                        tcp.memcpy(
                            buffer_out[start:stop], buffer_in_n[0:each_task_remain]
                        )

                if data_remain != 0:
                    start = task_id * data_calculated_each_task + data_calculated_each_task
                    stop = start + data_remain
                    tcp.assign(buffer_in_n, 0)
                    tcp.assign(buffer_out_n, 0)
                    tcp.memcpy(buffer_in_n[0:data_remain], buffer_in[start:stop])
                    with tcp.block("compute"):
                        if self.dtype == "float32":
                            self.compute_body_float(
                                buffer_in_n,
                                buffer_abs,
                                buffer_floor,
                                buffer_floor_after,
                                buffer_sgn,
                                buffer_tem,
                            )
                        elif self.dtype == "float16":
                            self.compute_body_half(
                                buffer_in_n,
                                buffer_abs,
                                buffer_floor,
                                buffer_floor_after,
                                buffer_sgn,
                                buffer_tem,
                            )
                    tcp.memcpy(buffer_out[start:stop], buffer_in_n[0:data_remain])


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_frac(dtype=None, target=None):
    f = build_module.build(
        Frac(TARGET(target).cluster_num, dtype.name, dtype.bytes),
        target_tag=target,
        name=KERNEL_NAME,
    )
    return f
