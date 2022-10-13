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
from bangpy.common.dtypes import DType
from bangpy.script import tcp, build_module, ty

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu220-m2", "mlu270", "mlu290", "mlu370-s4"]
KERNEL_NAME = "cross"


class Cross(object):
    """Operator description:
    torch.cross
    torch.cross(input, other, dim, output)
    compute the 3D cross product of two tensors: input and other on dimension dim,
    and output is the result tensor.
    To learn more, please refer to design docs.
    """

    def __init__(self, dtype: ty.string) -> None:
        self.dtype = dtype
        self.dtype_size = DType(dtype).bytes

    def cross_body(
        self,
        a0: ty.Buffer("nram"),
        a1: ty.Buffer("nram"),
        a2: ty.Buffer("nram"),
        b0: ty.Buffer("nram"),
        b1: ty.Buffer("nram"),
        b2: ty.Buffer("nram"),
        c0: ty.Buffer("nram"),
        c1: ty.Buffer("nram"),
        c2: ty.Buffer("nram"),
    ) -> None:
        # (c0,c1,c2)=(a0,a1,a2)x(b0,b1,b2)
        # = (a1*b2-a2*b1, a2*b0-a0*b2, a0*b1-a1*b0)
        tcp.multiply(c0, a1, b2)
        tcp.multiply(c1, a2, b1)
        tcp.subtract(c0, c0, c1)

        tcp.multiply(c1, a2, b0)
        tcp.multiply(c2, a0, b2)
        tcp.subtract(c1, c1, c2)

        tcp.multiply(c2, a0, b1)
        tcp.multiply(a0, a1, b0)
        tcp.subtract(c2, c2, a0)

    def main(
        self,
        input0: ty.handle,
        input1: ty.handle,
        dimshape: ty.handle,
        dim0: ty.int32,
        dim1: ty.int32,
        dim2: ty.int32,
        dim3: ty.int32,
        dim4: ty.int32,
        dim5: ty.int32,
        dim6: ty.int32,
        dim7: ty.int32,
        dim: ty.int32,
        output: ty.handle,
    ) -> None:

        # shape=(dim0, dim1, …)
        # if 'shape' is not set as parameter，then 'shape[i]' below would be not supported

        shape = tcp.match_buffer(dimshape, [8], dtype="int32")
        buffer_in0 = tcp.match_buffer(
            input0, [dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7], dtype=self.dtype
        )
        buffer_in1 = tcp.match_buffer(
            input1, [dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7], dtype=self.dtype
        )
        buffer_out = tcp.match_buffer(
            output, [dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7], dtype=self.dtype
        )

        tgt = tcp.target()
        task_num = tgt.cluster_num * tgt.core_num

        mydim = dim + 1
        # Bangpy2 can automatically deal with negative index

        maxdim = 8
        # pipeline buffer's shape must be statically defined,
        # so the dimension of shape must be static

        # calculate split strategy
        for cluster_id in tcp.thread_binding(0, tgt.cluster_num, thread="blockIdx.x"):
            for core_id in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):

                # e.g.,shape=(1,1,1,1,2,3,4,5), dim=5 means shape[5] = 3，
                # suppose the product of dimensions before dim called group,
                # the product of dimensions before dim called step,
                # both of them do not include dim,
                # in this example, group=1x1x1x1x2=2, step=4x5=20
                length = 1
                step = 1
                for i in range(maxdim):
                    length = length * shape[i]
                # 1x1x1x1x2x3x4x5=120 elements totally

                for i in range(mydim, maxdim):
                    step = step * shape[i]
                # step=4x5=20, if current element's index is 'i',
                # then next element is in 'i+step'

                # group = 1x1x1x1x1x2
                group = length / 3 / step

                task_id = cluster_id * tgt.core_num + core_id
                # pipeline needs buffer (a0,a1,a2,b0,b1,b2,c0,c1,c2)
                # to realize cross:(a0,a1,a2)x(b0,b1,b2)=(c0,c1,c2)
                # so two-level-pipeline needs 9x2=18 buffers totally
                # self.nram_size = 512*1024byte,18 buffer needed,
                # spare 30KB, then 482KB can be used,
                # and multiply and substraction should be 128byte aligned,
                # 128*18=2304,482*1024//2304*2304=493056(482*1024=493568),
                # 493056/18=27392,27392 can be exactly divided by 128,
                # and of course can be exactly divided by 4(byte) and 2(byte),
                # means that 13696*float16 or 6848*float32;
                # when target is mlu370, then nram_size = 768KB,
                # max buffer size is 41984 byte,
                # 20992*float16 or 10496*float32.
                single_buffer_size = (
                    (tgt.nram_size - 30 * 1024) // (128 * 18) * (128 * 18) / 18
                )
                data_each_buffer = single_buffer_size // self.dtype_size
                last_loop = 0

                # ATTENTION: step must <= data_each_buffer to run this operator now;
                step_each_time = data_each_buffer // step
                # every time data_each_buffer//step steps can be computed.

                # split and compute
                buffer_in0 = buffer_in0.reshape(
                    (dim0 * dim1 * dim2 * dim3 * dim4 * dim5 * dim6 * dim7 / step, step)
                )
                buffer_in1 = buffer_in1.reshape(
                    (dim0 * dim1 * dim2 * dim3 * dim4 * dim5 * dim6 * dim7 / step, step)
                )
                buffer_out = buffer_out.reshape(
                    (dim0 * dim1 * dim2 * dim3 * dim4 * dim5 * dim6 * dim7 / step, step)
                )

                # compute every task's group number,
                # remainder distributed from task_id=0
                # e.g. group=7, task=3, then group_each_task[3] is (3,2,2)
                group_each_task = group // task_num
                rest = group % task_num
                start = 0
                stop = 0
                if task_id < rest:
                    group_each_task = group_each_task + 1
                    start = group_each_task * task_id * 3
                    # stop: the max value which the index of buffer can reach in current task
                    # index can not in next task's compute range
                    # in other words, stop is next task's start
                    stop = group_each_task * (task_id + 1) * 3
                else:
                    # start.assign(((group_each_task+1)*task_id-(task_id-rest))*3),simplify
                    start = 3 * (group_each_task * task_id + rest)
                    stop = 3 * (group_each_task * (task_id + 1) + rest)

                stop = stop - 2

                # if there exists remainder (can't be exactly divided),
                # that means an extra compute time is needed;
                # in this case we need 'stop' computed before
                loop_num = group_each_task // step_each_time
                if group_each_task % step_each_time != 0:
                    loop_num = loop_num + 1
                    last_loop = 1
                    # means there exists remainder

                for i in range(loop_num, pipeline=True):
                    buffer_a0 = tcp.alloc_buffer(
                        [data_each_buffer], dtype=self.dtype, scope="nram"
                    )
                    buffer_a1 = tcp.alloc_buffer(
                        [data_each_buffer], dtype=self.dtype, scope="nram"
                    )
                    buffer_a2 = tcp.alloc_buffer(
                        [data_each_buffer], dtype=self.dtype, scope="nram"
                    )
                    buffer_b0 = tcp.alloc_buffer(
                        [data_each_buffer], dtype=self.dtype, scope="nram"
                    )
                    buffer_b1 = tcp.alloc_buffer(
                        [data_each_buffer], dtype=self.dtype, scope="nram"
                    )
                    buffer_b2 = tcp.alloc_buffer(
                        [data_each_buffer], dtype=self.dtype, scope="nram"
                    )
                    buffer_c0 = tcp.alloc_buffer(
                        [data_each_buffer], dtype=self.dtype, scope="nram"
                    )
                    buffer_c1 = tcp.alloc_buffer(
                        [data_each_buffer], dtype=self.dtype, scope="nram"
                    )
                    buffer_c2 = tcp.alloc_buffer(
                        [data_each_buffer], dtype=self.dtype, scope="nram"
                    )
                    # (a0,a1,a2)x(b0,b1,b2)=(c0,c1,c2)
                    with tcp.block("data_copy"):
                        if i == loop_num - 1 and last_loop == 1:
                            # to calculate remainder
                            tcp.memcpy(
                                buffer_a0[
                                    0 : (stop - (start + i * 3 * step_each_time) + 2)
                                    // 3
                                    * step
                                ].reshape(
                                    (
                                        (stop - (start + i * 3 * step_each_time) + 2)
                                        // 3,
                                        step,
                                    )
                                    # (stop-start+step-1)//step,
                                    # that's shape[0] of slice(start:stop:step)
                                ),
                                buffer_in0[(start + i * 3 * step_each_time) : stop : 3],
                            )

                            tcp.memcpy(
                                buffer_a1[
                                    0 : (stop - (start + i * 3 * step_each_time) + 2)
                                    // 3
                                    * step
                                ].reshape(
                                    (
                                        (stop - (start + i * 3 * step_each_time) + 2)
                                        // 3,
                                        step,
                                    )
                                ),
                                buffer_in0[
                                    (start + i * 3 * step_each_time) + 1 : stop + 1 : 3
                                ],
                            )

                            tcp.memcpy(
                                buffer_a2[
                                    0 : (stop - (start + i * 3 * step_each_time) + 2)
                                    // 3
                                    * step
                                ].reshape(
                                    (
                                        (stop - (start + i * 3 * step_each_time) + 2)
                                        // 3,
                                        step,
                                    )
                                ),
                                buffer_in0[
                                    (start + i * 3 * step_each_time) + 2 : stop + 2 : 3
                                ],
                            )

                            tcp.memcpy(
                                buffer_b0[
                                    0 : (stop - (start + i * 3 * step_each_time) + 2)
                                    // 3
                                    * step
                                ].reshape(
                                    (
                                        (stop - (start + i * 3 * step_each_time) + 2)
                                        // 3,
                                        step,
                                    )
                                ),
                                buffer_in1[(start + i * 3 * step_each_time) : stop : 3],
                            )

                            tcp.memcpy(
                                buffer_b1[
                                    0 : (stop - (start + i * 3 * step_each_time) + 2)
                                    // 3
                                    * step
                                ].reshape(
                                    (
                                        (stop - (start + i * 3 * step_each_time) + 2)
                                        // 3,
                                        step,
                                    )
                                ),
                                buffer_in1[
                                    (start + i * 3 * step_each_time) + 1 : stop + 1 : 3
                                ],
                            )

                            tcp.memcpy(
                                buffer_b2[
                                    0 : (stop - (start + i * 3 * step_each_time) + 2)
                                    // 3
                                    * step
                                ].reshape(
                                    (
                                        (stop - (start + i * 3 * step_each_time) + 2)
                                        // 3,
                                        step,
                                    )
                                ),
                                buffer_in1[
                                    (start + i * 3 * step_each_time) + 2 : stop + 2 : 3
                                ],
                            )

                        else:
                            # every time data_each_buffer//step steps are computed.
                            tcp.memcpy(
                                buffer_a0[
                                    0 : (3 * step_each_time + 2) // 3 * step
                                ].reshape(((3 * step_each_time + 2) // 3, step)),
                                buffer_in0[
                                    (start + i * 3 * step_each_time) : (
                                        start + (i + 1) * 3 * step_each_time
                                    ) : 3
                                ],
                            )

                            tcp.memcpy(
                                buffer_a1[
                                    0 : (3 * step_each_time + 2) // 3 * step
                                ].reshape(((3 * step_each_time + 2) // 3, step)),
                                buffer_in0[
                                    (start + i * 3 * step_each_time)
                                    + 1 : (start + (i + 1) * 3 * step_each_time)
                                    + 1 : 3
                                ],
                            )

                            tcp.memcpy(
                                buffer_a2[
                                    0 : (3 * step_each_time + 2) // 3 * step
                                ].reshape(((3 * step_each_time + 2) // 3, step)),
                                buffer_in0[
                                    (start + i * 3 * step_each_time)
                                    + 2 : (start + (i + 1) * 3 * step_each_time)
                                    + 2 : 3
                                ],
                            )

                            tcp.memcpy(
                                buffer_b0[
                                    0 : (3 * step_each_time + 2) // 3 * step
                                ].reshape(((3 * step_each_time + 2) // 3, step)),
                                buffer_in1[
                                    (start + i * 3 * step_each_time) : (
                                        start + (i + 1) * 3 * step_each_time
                                    ) : 3
                                ],
                            )

                            tcp.memcpy(
                                buffer_b1[
                                    0 : (3 * step_each_time + 2) // 3 * step
                                ].reshape(((3 * step_each_time + 2) // 3, step)),
                                buffer_in1[
                                    (start + i * 3 * step_each_time)
                                    + 1 : (start + (i + 1) * 3 * step_each_time)
                                    + 1 : 3
                                ],
                            )

                            tcp.memcpy(
                                buffer_b2[
                                    0 : (3 * step_each_time + 2) // 3 * step
                                ].reshape(((3 * step_each_time + 2) // 3, step)),
                                buffer_in1[
                                    (start + i * 3 * step_each_time)
                                    + 2 : (start + (i + 1) * 3 * step_each_time)
                                    + 2 : 3
                                ],
                            )

                    with tcp.block("compute"):
                        self.cross_body(
                            buffer_a0,
                            buffer_a1,
                            buffer_a2,
                            buffer_b0,
                            buffer_b1,
                            buffer_b2,
                            buffer_c0,
                            buffer_c1,
                            buffer_c2,
                        )

                    with tcp.block("data_copy"):
                        # store is the reverse of load
                        if i == loop_num - 1 and last_loop == 1:

                            tcp.memcpy(
                                buffer_out[(start + i * 3 * step_each_time) : stop : 3],
                                buffer_c0[
                                    0 : (stop - (start + i * 3 * step_each_time) + 2)
                                    // 3
                                    * step
                                ].reshape(
                                    (
                                        (stop - (start + i * 3 * step_each_time) + 2)
                                        // 3,
                                        step,
                                    )
                                ),
                            )

                            tcp.memcpy(
                                buffer_out[
                                    (start + i * 3 * step_each_time) + 1 : stop + 1 : 3
                                ],
                                buffer_c1[
                                    0 : (stop - (start + i * 3 * step_each_time) + 2)
                                    // 3
                                    * step
                                ].reshape(
                                    (
                                        (stop - (start + i * 3 * step_each_time) + 2)
                                        // 3,
                                        step,
                                    )
                                ),
                            )

                            tcp.memcpy(
                                buffer_out[
                                    (start + i * 3 * step_each_time) + 2 : stop + 2 : 3
                                ],
                                buffer_c2[
                                    0 : (stop - (start + i * 3 * step_each_time) + 2)
                                    // 3
                                    * step
                                ].reshape(
                                    (
                                        (stop - (start + i * 3 * step_each_time) + 2)
                                        // 3,
                                        step,
                                    )
                                ),
                            )

                        else:

                            tcp.memcpy(
                                buffer_out[
                                    (start + i * 3 * step_each_time) : (
                                        start + (i + 1) * 3 * step_each_time
                                    ) : 3
                                ],
                                buffer_c0[
                                    0 : (3 * step_each_time + 2) // 3 * step
                                ].reshape(((3 * step_each_time + 2) // 3, step)),
                            )

                            tcp.memcpy(
                                buffer_out[
                                    (start + i * 3 * step_each_time)
                                    + 1 : (start + (i + 1) * 3 * step_each_time)
                                    + 1 : 3
                                ],
                                buffer_c1[
                                    0 : (3 * step_each_time + 2) // 3 * step
                                ].reshape(((3 * step_each_time + 2) // 3, step)),
                            )

                            tcp.memcpy(
                                buffer_out[
                                    (start + i * 3 * step_each_time)
                                    + 2 : (start + (i + 1) * 3 * step_each_time)
                                    + 2 : 3
                                ],
                                buffer_c2[
                                    0 : (3 * step_each_time + 2) // 3 * step
                                ].reshape(((3 * step_each_time + 2) // 3, step)),
                            )

        buffer_out.reshape((dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7))

@bangpy.tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_cross(dtype=None, target=None):
    # build a executable module
    func = build_module.build(Cross(dtype.name), target_tag=target, name=KERNEL_NAME)
    return func
