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
"""Cosine operator implementation using BANGPy TCP API."""
# pylint: skip-file
import bangpy
from bangpy import tcp
from bangpy.script import ty, build_module
import bangpy.eager as eg


DTYPES = [bangpy.float32]
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]
KERNEL_NAME = "cosine"


class Cosine(object):
    """Operator description:
    Calculate cosine similarity of two given buffers
    """

    def __init__(
        self, cluster_num: ty.int32, dtype: ty.string, dtype_sz: ty.int32
    ) -> None:
        self.dtype = dtype
        self.cluster_num = cluster_num
        self.dtype_sz = dtype_sz

    def compute_dim0(
        self,
        buffer_in0: ty.Buffer("global"),
        buffer_in1: ty.Buffer("global"),
        buffer_out0: ty.Buffer("global"),
        size: ty.int32,
    ):
        """The body of cosine function"""

        dim_h = 1
        dim_m = self.dim0
        dim_l = self.dim1 * self.dim2 * self.dim3 / 32 * 32
        total_after = dim_h * dim_l
        buffer_out0 = buffer_out0.flatten()[:total_after].reshape((1, 1, total_after))
        length = dim_h * dim_m * dim_l

        buffer_in0_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_in1_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_mul = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out0_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out1_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out_final = tcp.alloc_buffer(
            shape=[size,], dtype=self.dtype, scope="nram",
        )
        tcp.assign(buffer_in0_n, 0)
        tcp.assign(buffer_in1_n, 0)
        tcp.assign(buffer_mul, 0)
        tcp.assign(buffer_out0_n, 0)
        tcp.assign(buffer_out1_n, 0)
        tcp.assign(buffer_out_final, 0)
        tcp.memcpy(buffer_in0_n[0 : self.length], buffer_in0)
        tcp.memcpy(buffer_in1_n[0 : self.length], buffer_in1)

        tcp.multiply(buffer_mul, buffer_in0_n, buffer_in1_n)
        tcp.square(buffer_in0_n, buffer_in0_n)
        tcp.square(buffer_in1_n, buffer_in1_n)

        mul_reshape = buffer_mul[0 : self.length][:length].reshape((dim_h, dim_m, dim_l))
        in_reshape0 = buffer_in0_n[0 : self.length][:length].reshape((dim_h, dim_m, dim_l))
        in_reshape1 = buffer_in1_n[0 : self.length][:length].reshape((dim_h, dim_m, dim_l))
        buffer_out_n = buffer_out_n[0:total_after].reshape((dim_h, 1, dim_l))
        buffer_out0_n = buffer_out0_n[0:total_after].reshape((dim_h, 1, dim_l))
        buffer_out1_n = buffer_out1_n[0:total_after].reshape((dim_h, 1, dim_l))
        buffer_out_final = buffer_out_final[0:total_after].reshape((dim_h, 1, dim_l))
        tcp.sumpool(buffer_out_n, mul_reshape, (dim_h, dim_m), (1, 1))
        tcp.sumpool(buffer_out0_n, in_reshape0, (dim_h, dim_m), (1, 1))
        tcp.sumpool(buffer_out1_n, in_reshape1, (dim_h, dim_m), (1, 1))

        tcp.multiply(buffer_out0_n, buffer_out0_n, buffer_out1_n)
        tcp.sqrt(buffer_out0_n, buffer_out0_n)
        tcp.divide(buffer_out_final, buffer_out_n, buffer_out0_n)
        tcp.memcpy(buffer_out0, buffer_out_final)

        buffer_out0 = buffer_out0.flatten()[0:self.dim1 * self.dim2 * self.dim3].reshape((self.dim1, self.dim2, self.dim3))

    def compute_dim1(
        self,
        buffer_in0: ty.Buffer("global"),
        buffer_in1: ty.Buffer("global"),
        buffer_out1: ty.Buffer("global"),
        size: ty.int32,
        loop_num: ty.int32,
    ):
        """The body of cosine function"""

        dim_h = self.dim0
        dim_m = self.dim1
        dim_l = self.dim2 * self.dim3 / 32 * 32
        lim_h = size // (dim_m * dim_l)
        total_after = lim_h * dim_l
        buffer_out1 = buffer_out1.flatten()[0:dim_h * dim_l].reshape((dim_h, 1, dim_l))
        length = lim_h * dim_m * dim_l

        buffer_in0_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_in1_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_mul = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out0_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out1_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out_final = tcp.alloc_buffer(
            shape=[size,], dtype=self.dtype, scope="nram",
        )
        tcp.assign(buffer_in0_n, 0)
        tcp.assign(buffer_in1_n, 0)
        tcp.assign(buffer_mul, 0)
        tcp.assign(buffer_out0_n, 0)
        tcp.assign(buffer_out1_n, 0)
        tcp.assign(buffer_out_n, 0)
        tcp.assign(buffer_out_final, 0)
        for i in range(0, loop_num):
            start = i * data_calculated_each_time
            end = start + data_calculated_each_time
            tcp.memcpy(buffer_in0_n, buffer_in0[start:end])
            tcp.memcpy(buffer_in1_n, buffer_in1[start:end])

            tcp.multiply(buffer_mul, buffer_in0_n, buffer_in1_n)
            tcp.square(buffer_in0_n, buffer_in0_n)
            tcp.square(buffer_in1_n, buffer_in1_n)
            mul_reshape = buffer_mul[0:length].reshape((lim_h, dim_m, dim_l))
            in_reshape0 = buffer_in0_n[0:length].reshape((lim_h, dim_m, dim_l))
            in_reshape1 = buffer_in1_n[0:length].reshape((lim_h, dim_m, dim_l))

            buffer_out_n = buffer_out_n[0:total_after].reshape((lim_h, 1, dim_l))
            buffer_out0_n = buffer_out0_n[0:total_after].reshape((lim_h, 1, dim_l))
            buffer_out1_n = buffer_out1_n[0:total_after].reshape((lim_h, 1, dim_l))
            buffer_out_final = buffer_out_final[0:total_after].reshape(
                (lim_h, 1, dim_l)
            )
            tcp.sumpool(buffer_out_n, mul_reshape, (1, dim_m), (1, 1))
            tcp.sumpool(buffer_out0_n, in_reshape0, (1, dim_m), (1, 1))
            tcp.sumpool(buffer_out1_n, in_reshape1, (1, dim_m), (1, 1))
            tcp.multiply(buffer_out0_n, buffer_out0_n, buffer_out1_n)
            tcp.sqrt(buffer_out0_n, buffer_out0_n)
            tcp.divide(buffer_out_final, buffer_out_n, buffer_out0_n)
            
            tcp.memcpy(buffer_out1[i * lim_h : (i + 1) * lim_h], buffer_out_final)

        buffer_out1 = buffer_out1.flatten()[0:self.dim0 * self.dim2 * self.dim3].reshape((self.dim0, self.dim2, self.dim3))

    def compute_dim2(
        self,
        buffer_in0: ty.Buffer("global"),
        buffer_in1: ty.Buffer("global"),
        buffer_out2: ty.Buffer("global"),
        size: ty.int32,
        loop_num: ty.int32,
    ):
        """The body of cosine function"""

        dim_h = self.dim0 * self.dim1
        dim_m = self.dim2
        dim_l = self.dim3 / 32 * 32
        lim_h = size // (dim_m * dim_l)
        total_after = lim_h * dim_l
        buffer_out2 = buffer_out2.flatten()[0:dim_h * dim_l].reshape((dim_h, 1, dim_l))
        length = lim_h * dim_m * dim_l

        buffer_in0_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_in1_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_mul = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out0_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out1_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out_final = tcp.alloc_buffer(
            shape=[size,], dtype=self.dtype, scope="nram",
        )
        tcp.assign(buffer_in0_n, 0)
        tcp.assign(buffer_in1_n, 0)
        tcp.assign(buffer_mul, 0)
        tcp.assign(buffer_out0_n, 0)
        tcp.assign(buffer_out1_n, 0)
        tcp.assign(buffer_out_n, 0)
        tcp.assign(buffer_out_final, 0)
        for i in range(0, loop_num):
            start = i * data_calculated_each_time
            end = start + data_calculated_each_time
            tcp.memcpy(buffer_in0_n, buffer_in0[start:end])
            tcp.memcpy(buffer_in1_n, buffer_in1[start:end])

            tcp.multiply(buffer_mul, buffer_in0_n, buffer_in1_n)
            tcp.square(buffer_in0_n, buffer_in0_n)
            tcp.square(buffer_in1_n, buffer_in1_n)
            mul_reshape = buffer_mul[0:length].reshape((lim_h, dim_m, dim_l))
            in_reshape0 = buffer_in0_n[0:length].reshape((lim_h, dim_m, dim_l))
            in_reshape1 = buffer_in1_n[0:length].reshape((lim_h, dim_m, dim_l))

            buffer_out_n = buffer_out_n[0:total_after].reshape((lim_h, 1, dim_l))
            buffer_out0_n = buffer_out0_n[0:total_after].reshape((lim_h, 1, dim_l))
            buffer_out1_n = buffer_out1_n[0:total_after].reshape((lim_h, 1, dim_l))
            buffer_out_final = buffer_out_final[0:total_after].reshape(
                (lim_h, 1, dim_l)
            )
            tcp.sumpool(buffer_out_n, mul_reshape, (1, dim_m), (1, 1))
            tcp.sumpool(buffer_out0_n, in_reshape0, (1, dim_m), (1, 1))
            tcp.sumpool(buffer_out1_n, in_reshape1, (1, dim_m), (1, 1))
            tcp.multiply(buffer_out0_n, buffer_out0_n, buffer_out1_n)
            tcp.sqrt(buffer_out0_n, buffer_out0_n)
            tcp.divide(buffer_out_final, buffer_out_n, buffer_out0_n)
            tcp.memcpy(buffer_out2[i * lim_h : (i + 1) * lim_h], buffer_out_final)

        buffer_out2 = buffer_out2.flatten()[0:self.dim0 * self.dim1 * self.dim3].reshape((self.dim0, self.dim1, self.dim3))

    def compute_dim3(
        self,
        buffer_in0: ty.Buffer("global"),
        buffer_in1: ty.Buffer("global"),
        buffer_out2: ty.Buffer("global"),
        size: ty.int32,
        loop_num: ty.int32,
    ):
        """The body of cosine function"""

        dim_h = self.dim0 * self.dim1 * self.dim2 / 32 * 32
        dim_m = self.dim3 / 32 * 32
        dim_l = 1
        lim_h = size // (dim_m * dim_l)
        total = dim_h * dim_l
        buffer_out3 = buffer_out3.flatten()[0:total].reshape((total,))
        length = lim_h * dim_m * dim_l

        buffer_in0_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_in1_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_mul = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out0_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out1_n = tcp.alloc_buffer(shape=[size,], dtype=self.dtype, scope="nram",)
        buffer_out_final = tcp.alloc_buffer(
            shape=[size,], dtype=self.dtype, scope="nram",
        )

        tcp.assign(buffer_in0_n, 0)
        tcp.assign(buffer_in1_n, 0)
        tcp.assign(buffer_mul, 0)
        tcp.assign(buffer_out0_n, 0)
        tcp.assign(buffer_out1_n, 0)
        tcp.assign(buffer_out_final, 0)
        for i in range(0, loop_num):
            start = i * data_calculated_each_time
            end = start + data_calculated_each_time
            tcp.memcpy(buffer_in0_n, buffer_in0[start:end])
            tcp.memcpy(buffer_in1_n, buffer_in1[start:end])
            tcp.multiply(buffer_mul, buffer_in0_n, buffer_in1_n)
            tcp.square(buffer_in0_n, buffer_in0_n)
            tcp.square(buffer_in1_n, buffer_in1_n)
            mul_reshape = buffer_mul[0:length].reshape((lim_h, dim_m))
            in_reshape0 = buffer_in0_n[0:length].reshape((lim_h, dim_m))
            in_reshape1 = buffer_in1_n[0:length].reshape((lim_h, dim_m))

            buffer_out_n = buffer_out_n[0:lim_h]
            buffer_out0_n = buffer_out0_n[0:lim_h]
            buffer_out1_n = buffer_out1_n[0:lim_h]
            buffer_out_final = buffer_out_final[0:lim_h]

            for j in range(0, lim_h):
                total_sum = 0.0
                total_sum0 = 0.0
                total_sum1 = 0.0
                for k in range(0, dim_m):
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
            
            tcp.multiply(buffer_out0_n, buffer_out0_n, buffer_out1_n)
            tcp.sqrt(buffer_out0_n, buffer_out0_n)
            tcp.divide(buffer_out_final, buffer_out_n, buffer_out0_n)

            tcp.memcpy(buffer_out3[i * lim_h : (i + 1) * lim_h], buffer_out_final)

        buffer_out3 = buffer_out3.flatten()[0:self.dim0 * self.dim1 * self.dim2].reshape((self.dim0, self.dim1, self.dim2))

    def main(
        self,
        buffer_in0: ty.handle,
        buffer_in1: ty.handle,
        buffer_out0: ty.handle,
        buffer_out1: ty.handle,
        buffer_out2: ty.handle,
        buffer_out3: ty.handle,
        dim: ty.int32,
        dim0: ty.int32,
        dim1: ty.int32,
        dim2: ty.int32,
        dim3: ty.int32,
    ) -> None:
        """The main part of cosine function"""

        self.dim = dim
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.length = self.dim0 * self.dim1 * self.dim2 * self.dim3

        buffer_in0 = tcp.match_buffer(
            buffer_in0, [dim0, dim1, dim2, dim3], dtype=self.dtype
        )
        buffer_in1 = tcp.match_buffer(
            buffer_in1, [dim0, dim1, dim2, dim3], dtype=self.dtype
        )
        buffer_out0 = tcp.match_buffer(
            buffer_out0, [dim1, dim2, dim3], dtype=self.dtype
        )
        buffer_out1 = tcp.match_buffer(
            buffer_out1, [dim0, dim2, dim3], dtype=self.dtype
        )
        buffer_out2 = tcp.match_buffer(
            buffer_out2, [dim0, dim1, dim3], dtype=self.dtype
        )
        buffer_out3 = tcp.match_buffer(
            buffer_out3, [dim0, dim1, dim2], dtype=self.dtype
        )
        tgt = tcp.target()

        self.single_buffer_size = (tgt.nram_size) // 8
        self.task_num = 1
        task_id = 0
        data_calculated_each_task = self.length // self.task_num
        loop_num = data_calculated_each_task * self.dtype_sz // self.single_buffer_size
        data_calculated_each_time = self.single_buffer_size // self.dtype_sz

        buffer_in0 = buffer_in0.reshape((self.length,))
        buffer_in1 = buffer_in1.reshape((self.length,))

        for core_id in tcp.thread_binding(0, 1, thread="threadIdx.x"):
            if self.dim == 0:
                self.compute_dim0(
                    buffer_in0, buffer_in1, buffer_out0, data_calculated_each_time
                )
            elif self.dim == 1:
                self.compute_dim1(
                    buffer_in0,
                    buffer_in1,
                    buffer_out1,
                    data_calculated_each_time,
                    loop_num,
                )
            elif self.dim == 2:
                self.compute_dim2(
                    buffer_in0,
                    buffer_in1,
                    buffer_out2,
                    data_calculated_each_time,
                    loop_num,
                )
            elif self.dim == 3:
                self.compute_dim3(
                    buffer_in0,
                    buffer_in1,
                    buffer_out3,
                    data_calculated_each_time,
                    loop_num,
                )
            else:
                self.compute_dim0(
                    buffer_in0, buffer_in1, buffer_out0, data_calculated_each_time
                )


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_cosine(dtype=None, target=None):
    f = build_module.build(
        Cosine(1, dtype.name, dtype.bytes), target_tag=target, name=KERNEL_NAME
    )
    return f