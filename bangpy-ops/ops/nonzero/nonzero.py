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
# The above copyright notice and this permission notice shall self.tcp included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# pylint: disable=useless-object-inheritance, too-many-instance-attributes
# pylint: disable=attribute-defined-outside-init, too-many-statements
# pylint: disable=too-many-arguments, too-many-locals
"""NonZero operator implementation using BANGPy TCP API."""
import bangpy as bp
from bangpy import tcp
from bangpy.tcp.util import round_up, round_down
from bangpy.platform.bang_config import TARGET

DTYPES = [bp.float16, bp.float32]
TARGET_LIST = ["mlu370-s4", "mlu290", "mlu270"]

# Align to 64.
ALIGN_SIZE = 64


class NonZero(object):
    """Operator description:
    ONNX NonZero operator, behaves similar to numpy.nonzero.
    """

    def __init__(self, target, dtype=bp.float32, task_num=16, name="NonZero"):
        """Construct a new NonZero class.

        Parameters
        ----------
        target : str
            Target MLU device name.

        dtype : bangpy.DType
            The data type of input.

        task_num : int
            The task number of runtime.

        name : str
            Kernel entry function name.

        Attributes
        ----------
        tcp : tcp.TCP
            TCP container.

        dtype : bangpy.DType
            The data type of input.

        target : str
            Target MLU device name.

        task_num : int
            The task number of runtime.

        name : str
            Kernel entry function name.

        dim_0 : tcp.SizeVar
            The first dimension size of input data.

        dim_1 : tcp.SizeVar
            The second dimension size of input data.

        dim_2 : tcp.SizeVar
            The third dimension size of input data.

        dim_3 : tcp.SizeVar
            The fourth dimension size of input data.

        num_nonzero : tcp.SizeVar
            The number of nonzero.

        trans : tcp.SizeVar
            Output data will be transposed if trans is equal to 1.
            Otherwise not.

        dim_num : tcp.SizeVar
            The size of dimension of input data.

        nram_size : int
            The size of nram buffer needed for one calculation.

        int_type : bangpy.DType
            The compute int data type.

        float_type : bangpy.DType
            The compute float data type.

        max_dim_num : int
            The maximum size of dimension of input data.
        """
        self.tcp = tcp.TCP(target)
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.name = name
        self.dim_0 = self.tcp.SizeVar("dim_0")
        self.dim_1 = self.tcp.SizeVar("dim_1")
        self.dim_2 = self.tcp.SizeVar("dim_2")
        self.dim_3 = self.tcp.SizeVar("dim_3")
        self.task_num_var = self.tcp.SizeVar("task_num")
        self.num_nonzero = self.tcp.SizeVar("num_nonzero")
        self.trans = self.tcp.SizeVar("need_trans")
        self.dim_num = self.tcp.SizeVar("dim_num")
        # 30 * 1024B reserve for stack, need 13 buffers after storage rewrite pass.
        self.nram_size = round_down(
            (TARGET(target).nram_size - 30 * 1024) // 22 // bp.int32.bytes, 128
        )
        self.int_type = bp.int32
        self.float_type = bp.float32
        self.max_dim_num = 4

    def indices_set(self, index_nram, global_index, size, seg_size, dim_size):
        """Set output indices of each dim."""
        index_0 = self.tcp.Scalar(dtype=bp.int32, name="index_0")
        index_1 = self.tcp.Scalar(dtype=bp.int32, name="index_1")
        remain_front = self.tcp.Scalar(dtype=bp.int32, name="remain_front")
        remain_back = self.tcp.Scalar(dtype=bp.int32, name="remain_back")
        offset = self.tcp.Scalar(dtype=bp.int32, name="offset")
        size_align = self.tcp.Scalar(dtype=bp.int32, name="size_align")
        value = self.tcp.Scalar(dtype=bp.int32, name="value")
        size_align.assign(round_up(size, ALIGN_SIZE))
        index_0.assign(global_index / seg_size)
        index_1.assign((global_index + size) / seg_size)
        with self.tcp.if_scope(index_0 == index_1):
            value.assign(index_0 % dim_size)
            self.tcp.assign(index_nram[:size_align], value.astype(self.int_type))
        with self.tcp.else_scope():
            remain_front.assign(seg_size - global_index % seg_size)
            remain_back.assign((global_index + size) % seg_size)
            value.assign(index_0 % dim_size)
            self.tcp.assign(
                index_nram[: round_up(remain_front, ALIGN_SIZE)],
                value.astype(self.int_type),
            )
            offset.assign(remain_front)
            with self.tcp.for_range(0, (index_1 - index_0 - 1)) as i:
                value.assign((index_0 + i + 1) % dim_size)
                # mlu3xx
                if self.target[:6] == "mlu370":
                    self.tcp.assign(
                        index_nram[offset : offset + round_up(seg_size, ALIGN_SIZE)],
                        value.astype(self.int_type),
                    )
                # mlu2xx
                else:
                    with self.tcp.for_range(0, seg_size) as i:
                        index_nram[offset + i] = value.astype(self.int_type)
                offset.assign(offset + seg_size)
            with self.tcp.if_scope(remain_back != 0):
                value.assign(index_1 % dim_size)
                if self.target[:6] == "mlu370":
                    self.tcp.assign(
                        index_nram[offset : offset + round_up(remain_back, ALIGN_SIZE)],
                        value.astype(self.int_type),
                    )
                else:
                    with self.tcp.for_range(0, remain_back) as i:
                        index_nram[offset + i] = value.astype(self.int_type)

    def gather_data(self, int64_nram, out_nram, align_size, dim_index):
        """Gather each dim output data. Convert int32 to int64 and transpose."""
        int_nram = int64_nram[: self.dim_num * align_size].reinterpret_cast(bp.int32)
        with self.tcp.if_scope(self.trans == 1):
            int_nram0 = int_nram.reshape((align_size, self.dim_num * 2))[
                :, dim_index * 2
            ]
            self.tcp.memcpy(
                int_nram0[:, 0], out_nram[:align_size].reshape((align_size, 1))
            )
        with self.tcp.else_scope():
            int_nram1 = int_nram.reshape((self.dim_num, align_size * 2))[dim_index]
            int_nram1 = int_nram1.reshape((align_size, 2))
            self.tcp.memcpy(
                int_nram1[:, 0], out_nram[:align_size].reshape((align_size, 1))
            )

    def core_compute(self, pre_core, core_index, out_offset, in_buffer, out_buffer):
        """NonZero compute on each core."""
        repeat = self.tcp.Scalar(dtype=bp.int32, name="repeat")
        repeat.assign(pre_core / self.nram_size)
        remain = self.tcp.Scalar(dtype=bp.int32, name="remain")
        remain.assign(pre_core % self.nram_size)
        zero = self.tcp.Scalar(dtype=self.dtype, name="zero", value=0)
        index_0_nram = self.tcp.Buffer(
            shape=(2 * self.nram_size + ALIGN_SIZE,),
            dtype=self.int_type,
            name="index_0_nram",
            scope="nram",
        )
        cast_nram = self.tcp.Buffer(
            shape=(self.nram_size,),
            dtype=self.float_type,
            name="cast_nram",
            scope="nram",
        )
        out_nram = self.tcp.Buffer(
            shape=(self.nram_size,), dtype=self.int_type, name="out_nram", scope="nram"
        )

        with self.tcp.if_scope(self.dim_num > 1):
            index_1_nram = self.tcp.Buffer(
                shape=(self.nram_size + ALIGN_SIZE,),
                dtype=self.int_type,
                name="index_1_nram",
                scope="nram",
            )
        with self.tcp.for_range(0, repeat, stage=1) as i:
            global_index = core_index + i * self.nram_size

            # The count number buffer for count_nonzero, size at least 128 bytes.
            count_num = self.tcp.Buffer(
                shape=(32,), dtype=bp.int32, name="count", scope="nram"
            )
            data_nram = self.tcp.Buffer(
                shape=(self.nram_size,),
                dtype=self.dtype,
                name="data_nram",
                scope="nram",
            )

            out_nram_int64 = self.tcp.Buffer(
                shape=(self.nram_size * self.max_dim_num,),
                dtype=bp.int64,
                name="out_nram",
                scope="nram",
            )

            with self.tcp.block("data_copy"):
                self.tcp.memcpy(
                    data_nram,
                    in_buffer.flatten()[global_index : global_index + self.nram_size],
                )
            with self.tcp.block("compute"):
                self.tcp.assign(
                    out_nram_int64.reinterpret_cast(bp.int32)[
                        : self.dim_num * self.nram_size * 2
                    ],
                    zero.astype(bp.int32),
                )

                # Convert input data type to float32 data type.
                if self.dtype != self.float_type:
                    self.tcp.type_convert(cast_nram, data_nram)
                else:
                    cast_nram = data_nram

                # cast_nram only computes if there are non-zero elements in it.
                self.tcp.count_nonzero(count_num.reinterpret_cast(bp.uint32), cast_nram)
                with self.tcp.if_scope(count_num[0] > 0):

                    # One dims compute.
                    with self.tcp.for_range(0, self.nram_size) as k:
                        index_0_nram[k] = (global_index + k) % self.dim_3
                    self.tcp.take(
                        out_nram.reinterpret_cast(self.float_type),
                        index_0_nram[: self.nram_size].reinterpret_cast(
                            self.float_type
                        ),
                        cast_nram,
                    )

                    self.gather_data(
                        out_nram_int64, out_nram, self.nram_size, self.dim_num - 1
                    )

                    # Two dims compute.
                    with self.tcp.if_scope(self.dim_num > 1):
                        self.indices_set(
                            index_1_nram,
                            global_index,
                            self.nram_size,
                            self.dim_3,
                            self.dim_2,
                        )

                        self.tcp.take(
                            out_nram.reinterpret_cast(self.float_type),
                            index_1_nram[: self.nram_size].reinterpret_cast(
                                self.float_type
                            ),
                            cast_nram,
                        )
                        self.gather_data(
                            out_nram_int64, out_nram, self.nram_size, self.dim_num - 2
                        )

                    # Tree dims compute.
                    with self.tcp.if_scope(self.dim_num > 2):
                        self.indices_set(
                            index_1_nram,
                            global_index,
                            self.nram_size,
                            self.dim_3 * self.dim_2,
                            self.dim_1,
                        )
                        self.tcp.take(
                            out_nram.reinterpret_cast(self.float_type),
                            index_1_nram[: self.nram_size].reinterpret_cast(
                                self.float_type
                            ),
                            cast_nram,
                        )
                        self.gather_data(
                            out_nram_int64, out_nram, self.nram_size, self.dim_num - 3
                        )

                    # Four dims compute.
                    with self.tcp.if_scope(self.dim_num > 3):
                        self.indices_set(
                            index_1_nram,
                            global_index,
                            self.nram_size,
                            self.dim_3 * self.dim_2 * self.dim_1,
                            self.dim_0,
                        )
                        self.tcp.take(
                            out_nram.reinterpret_cast(self.float_type),
                            index_1_nram[: self.nram_size].reinterpret_cast(
                                self.float_type
                            ),
                            cast_nram,
                        )
                        self.gather_data(
                            out_nram_int64, out_nram, self.nram_size, self.dim_num - 4
                        )

            with self.tcp.block("data_copy"):
                with self.tcp.if_scope(count_num[0] > 0):
                    with self.tcp.if_scope(self.trans == 1):
                        out_nram_int64 = out_nram_int64[
                            : self.dim_num * self.nram_size
                        ].reshape((self.nram_size, self.dim_num))
                        out_buffer = out_buffer.reshape(
                            (self.num_nonzero, self.dim_num)
                        )
                        self.tcp.memcpy(
                            out_buffer[
                                out_offset : out_offset + count_num[0],
                            ],
                            out_nram_int64[
                                : count_num[0],
                            ],
                        )
                    with self.tcp.else_scope():
                        out_nram_int64 = out_nram_int64[
                            : self.dim_num * self.nram_size
                        ].reshape((self.dim_num, self.nram_size))
                        out_buffer = out_buffer.reshape(
                            (self.dim_num, self.num_nonzero)
                        )
                        self.tcp.memcpy(
                            out_buffer[:, out_offset : out_offset + count_num[0]],
                            out_nram_int64[:, : count_num[0]],
                        )

            with self.tcp.block("compute"):
                with self.tcp.if_scope(count_num[0] > 0):
                    out_offset.assign(out_offset + count_num[0])
        with self.tcp.if_scope(remain > 0):
            global_index = core_index + repeat * self.nram_size

            remain_align = self.tcp.Scalar(dtype=bp.int32, name="remain_align")
            remain_align.assign(
                round_up(
                    remain,
                    ALIGN_SIZE
                    if self.tcp.mlu_device[:6] == "mlu370"
                    else ALIGN_SIZE * 2,
                )
            )
            data_nram = self.tcp.Buffer(
                shape=(self.nram_size,),
                dtype=self.dtype,
                name="data_nram",
                scope="nram",
            )

            out_nram_int64 = self.tcp.Buffer(
                shape=(self.nram_size * self.max_dim_num,),
                dtype=bp.int64,
                name="out_nram",
                scope="nram",
            )

            self.tcp.assign(data_nram[:remain_align], zero)
            self.tcp.assign(
                out_nram_int64.reinterpret_cast(bp.int32)[
                    : self.dim_num * remain_align * 2
                ],
                zero.astype(bp.int32),
            )
            self.tcp.memcpy(
                data_nram[:remain],
                in_buffer.flatten()[global_index : global_index + remain],
            )
            if self.dtype != self.float_type:
                self.tcp.type_convert(
                    cast_nram[:remain_align], data_nram[:remain_align]
                )
            else:
                cast_nram = data_nram

            # cast_nram only computes if there are non-zero elements in it.
            self.tcp.count_nonzero(
                count_num.reinterpret_cast(bp.uint32), cast_nram[:remain_align]
            )
            with self.tcp.if_scope(count_num[0] > 0):

                # One dims compute.
                with self.tcp.for_range(0, remain) as k:
                    index_0_nram[k] = (global_index + k) % self.dim_3
                self.tcp.take(
                    out_nram.reinterpret_cast(self.float_type),
                    index_0_nram[:remain_align].reinterpret_cast(self.float_type),
                    cast_nram[:remain_align],
                )
                self.gather_data(
                    out_nram_int64, out_nram, remain_align, self.dim_num - 1
                )

                # Two dims compute.
                with self.tcp.if_scope(self.dim_num > 1):
                    self.indices_set(
                        index_1_nram, global_index, remain, self.dim_3, self.dim_2
                    )
                    self.tcp.take(
                        out_nram.reinterpret_cast(self.float_type),
                        index_1_nram[:remain_align].reinterpret_cast(self.float_type),
                        cast_nram,
                    )
                    self.gather_data(
                        out_nram_int64, out_nram, remain_align, self.dim_num - 2
                    )

                # Tree dims compute.
                with self.tcp.if_scope(self.dim_num > 2):
                    self.indices_set(
                        index_1_nram,
                        global_index,
                        remain,
                        self.dim_3 * self.dim_2,
                        self.dim_1,
                    )
                    self.tcp.take(
                        out_nram.reinterpret_cast(self.float_type),
                        index_1_nram[:remain_align].reinterpret_cast(self.float_type),
                        cast_nram,
                    )
                    self.gather_data(
                        out_nram_int64, out_nram, remain_align, self.dim_num - 3
                    )

                # Four dims compute.
                with self.tcp.if_scope(self.dim_num > 3):
                    self.indices_set(
                        index_1_nram,
                        global_index,
                        remain,
                        self.dim_3 * self.dim_2 * self.dim_1,
                        self.dim_0,
                    )
                    self.tcp.take(
                        out_nram.reinterpret_cast(self.float_type),
                        index_1_nram[:remain_align].reinterpret_cast(self.float_type),
                        cast_nram,
                    )
                    self.gather_data(
                        out_nram_int64, out_nram, remain_align, self.dim_num - 4
                    )

                with self.tcp.if_scope(self.trans == 1):
                    out_nram_int64 = out_nram_int64[
                        : self.dim_num * remain_align
                    ].reshape((remain_align, self.dim_num))
                    out_buffer = out_buffer.reshape((self.num_nonzero, self.dim_num))
                    self.tcp.memcpy(
                        out_buffer[
                            out_offset : out_offset + count_num[0],
                        ],
                        out_nram_int64[
                            : count_num[0],
                        ],
                    )
                with self.tcp.else_scope():
                    out_nram_int64 = out_nram_int64[
                        : self.dim_num * remain_align
                    ].reshape((self.dim_num, remain_align))
                    out_buffer = out_buffer.reshape((self.dim_num, self.num_nonzero))
                    self.tcp.memcpy(
                        out_buffer[:, out_offset : out_offset + count_num[0]],
                        out_nram_int64[:, : count_num[0]],
                    )

    def nonzero_compute(self):
        """The entry of the NonZero operator."""
        self.tcp.launch_task(self.task_num, 1, 1)
        self.in_buffer = self.tcp.Buffer(
            shape=(self.dim_0, self.dim_1, self.dim_2, self.dim_3),
            dtype=self.dtype,
            name="in_buffer",
            scope="global",
        )

        self.core_count = self.tcp.Buffer(
            shape=(self.task_num_var,), dtype=bp.uint32, name="core_count", scope="global"
        )
        self.out_buffer = self.tcp.Buffer(
            shape=(self.num_nonzero * self.dim_num,),
            dtype=bp.int64,
            name="out_buffer",
            scope="global",
        )
        # Multi-core compute split.
        elem_size = self.tcp.Scalar(dtype=bp.int32, name="elem_size")
        elem_size.assign(self.dim_0 * self.dim_1 * self.dim_2 * self.dim_3)
        pre_core = self.tcp.Scalar(dtype=bp.int32, name="pre_core")
        pre_core.assign(elem_size / self.tcp.taskDim)
        core_remain = self.tcp.Scalar(dtype=bp.int32, name="core_remain")
        core_remain.assign(elem_size % self.tcp.taskDim)
        core_index = self.tcp.Scalar(dtype=bp.int32, name="core_index")
        out_offset = self.tcp.Scalar(dtype=bp.int32, name="out_offset", value=0)

        with self.tcp.if_scope(self.core_count[self.tcp.taskId] != 0):
            core_index.assign(pre_core * self.tcp.taskId)
            with self.tcp.if_scope(self.tcp.taskId < core_remain):
                pre_core.assign(pre_core + 1)
                core_index.assign(core_index + self.tcp.taskId)
            with self.tcp.else_scope():
                core_index.assign(core_index + core_remain)

            with self.tcp.for_range(0, self.tcp.taskId) as i:
                out_offset.assign(out_offset + self.core_count[i])
            self.core_compute(
                pre_core,
                core_index,
                out_offset,
                self.in_buffer,
                self.out_buffer,
            )

        return self.tcp.BuildBANG(
            inputs=[
                self.in_buffer,
                self.core_count,
                self.dim_0,
                self.dim_1,
                self.dim_2,
                self.dim_3,
                self.dim_num,
                self.num_nonzero,
                self.trans,
            ],
            outputs=[self.out_buffer],
            kernel_name=self.name,
        )


@tcp.register_mlu_op(DTYPES, TARGET_LIST, "NonZero")
def build_nonzero(dtype, target):
    task_num = TARGET(target).cluster_num * TARGET(target).core_num

    nonzero = NonZero(target, task_num=task_num, dtype=dtype)
    f_nonzero = nonzero.nonzero_compute()
    return f_nonzero
