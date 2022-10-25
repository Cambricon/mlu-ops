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
# pylint: disable=useless-object-inheritance, too-many-instance-attributes
# pylint: disable=attribute-defined-outside-init, too-many-statements
# pylint: disable=too-many-arguments, too-many-locals, missing-function-docstring
"""NonZero operator implementation using BANGPy TCP API."""
import bangpy as bp
from bangpy import tcp
from bangpy.script import build_module, ty

DTYPES = [bp.float16, bp.float32]
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]


class NonZero(object):
    """Operator description:
    ONNX NonZero operator, behaves similar to numpy.nonzero.
    """

    def __init__(
        self,
        target: ty.string,
        dtype: ty.string,
        dtype_bits: ty.int32,
        align_size: ty.int32,
    ):
        self.target = target
        self.dtype = dtype
        self.dtype_bits = dtype_bits
        self.align_size = align_size
        self.max_dim_num = 4
        self.mode = "rd" if self.dtype == "float32" else None

    def indices_set(
        self,
        index_nram: ty.Buffer("nram"),  # type: ignore
        global_index: ty.int32,
        size: ty.int32,
        seg_size: ty.int32,
        dim_size: ty.int32,
    ):
        """Set output indices of each dim."""
        size_align = tcp.round_up(size, 64)
        index_0 = global_index / seg_size
        index_1 = (global_index + size) / seg_size
        if index_0 == index_1:
            value = index_0 % dim_size
            tcp.assign(index_nram[:size_align], value)
        else:
            remain_front = seg_size - global_index % seg_size
            remain_back = (global_index + size) % seg_size
            value = index_0 % dim_size
            tcp.assign(
                index_nram[: tcp.round_up(remain_front, 64)],
                value,
            )
            offset = remain_front
            for i in range(index_1 - index_0 - 1):  # type: ignore
                value = (index_0 + i + 1) % dim_size
                # mlu3xx
                if self.target == "mlu370":
                    tcp.assign(
                        index_nram[offset : offset + tcp.round_up(seg_size, 64)],
                        value,
                    )
                # mlu2xx
                else:
                    for j in range(seg_size):  # type: ignore
                        index_nram[offset + j] = value
                offset = offset + seg_size
            if remain_back != 0:
                value = index_1 % dim_size
                if self.target == "mlu370":
                    tcp.assign(
                        index_nram[offset : offset + tcp.round_up(remain_back, 64)],
                        value,
                    )
                else:
                    for j in range(remain_back):  # type: ignore
                        index_nram[offset + j] = value

    def gather_data(
        self,
        int64_nram: ty.Buffer("nram"),  # type: ignore
        out_nram: ty.Buffer("nram"),  # type: ignore
        align_size: ty.int32,
        dim_index: ty.int32,
    ):
        """Gather each dim output data. Convert int32 to int64 and transpose."""
        int_nram = int64_nram[: self.dim_num * align_size].reinterpret_cast("int32")
        if self.trans == 1:
            int_nram0 = int_nram.reshape((align_size, self.dim_num * 2))
            int_nram0 = int_nram0[:align_size, dim_index * 2]
            tcp.memcpy(int_nram0, out_nram[:align_size].reshape((align_size, 1)))
        else:
            int_nram1 = int_nram.reshape((self.dim_num, align_size * 2))[dim_index]
            int_nram1 = int_nram1.reshape((align_size, 2))
            tcp.memcpy(
                int_nram1[:align_size, 0],
                out_nram[:align_size].reshape((align_size, 1)),
            )

    def core_compute(
        self,
        pre_core: ty.int32,
        core_index: ty.int32,
        out_offset: ty.int32,
    ):
        """NonZero compute on each core."""
        repeat = pre_core / self.nram_size
        remain = pre_core % self.nram_size
        zero = tcp.cast(0, self.dtype)
        index_0_nram = tcp.alloc_buffer(
            shape=(2 * self.nram_size + 64,),
            dtype="int32",
            scope="nram",
        )
        cast_nram = tcp.alloc_buffer(
            shape=(self.nram_size,),
            dtype="float32",
            scope="nram",
        )
        out_nram = tcp.alloc_buffer(
            shape=(self.nram_size,), dtype="int32", scope="nram"
        )
        index_1_nram = tcp.alloc_buffer(
            shape=(self.nram_size + 64,),
            dtype="int32",
            scope="nram",
        )
        for i in range(repeat, pipeline=True):  # type: ignore
            global_index = core_index + i * self.nram_size

            # The count number buffer for count_nonzero, size at least 128 bytes.
            count_num = tcp.alloc_buffer(shape=(32,), dtype="int32", scope="nram")
            data_nram = tcp.alloc_buffer(
                shape=(self.nram_size,),
                dtype=self.dtype,
                scope="nram",
            )

            out_nram_int64 = tcp.alloc_buffer(
                shape=(self.nram_size * self.max_dim_num,),
                dtype="int64",
                scope="nram",
            )

            with tcp.block("data_copy"):
                tcp.memcpy(
                    data_nram,
                    self.in_buffer.flatten()[
                        global_index : global_index + self.nram_size
                    ],
                )
            with tcp.block("compute"):
                tcp.assign(
                    out_nram_int64.reinterpret_cast("int32")[
                        : self.dim_num * self.nram_size * 2
                    ],
                    0,
                )

                # Convert input data type to float32 data type.
                if self.dtype != "float32":
                    tcp.type_convert(cast_nram, data_nram, mode=self.mode)
                    tcp.count_nonzero(count_num.reinterpret_cast("uint32"), cast_nram)
                else:
                    # cast_nram = data_nram
                    tcp.count_nonzero(count_num.reinterpret_cast("uint32"), data_nram)

                if count_num[0] > 0:
                    # One dims compute.
                    for k in range(self.nram_size):  # type: ignore
                        index_0_nram[k] = (global_index + k) % self.dim_3
                    if self.dtype != "float32":
                        tcp.take(
                            out_nram.reinterpret_cast("float32"),
                            index_0_nram[: self.nram_size].reinterpret_cast("float32"),
                            cast_nram,
                        )
                    else:
                        tcp.take(
                            out_nram.reinterpret_cast("float32"),
                            index_0_nram[: self.nram_size].reinterpret_cast("float32"),
                            data_nram,
                        )
                    self.gather_data(
                        out_nram_int64, out_nram, self.nram_size, self.dim_num - 1
                    )

                    # Two dims compute.
                    if self.dim_num > 1:
                        self.indices_set(
                            index_1_nram,
                            global_index,
                            self.nram_size,
                            self.dim_3,
                            self.dim_2,
                        )
                        if self.dtype != "float32":
                            tcp.take(
                                out_nram.reinterpret_cast("float32"),
                                index_1_nram[: self.nram_size].reinterpret_cast(
                                    "float32"
                                ),
                                cast_nram,
                            )
                        else:
                            tcp.take(
                                out_nram.reinterpret_cast("float32"),
                                index_1_nram[: self.nram_size].reinterpret_cast(
                                    "float32"
                                ),
                                data_nram,
                            )
                        self.gather_data(
                            out_nram_int64, out_nram, self.nram_size, self.dim_num - 2
                        )

                    # Tree dims compute.
                    if self.dim_num > 2:
                        self.indices_set(
                            index_1_nram,
                            global_index,
                            self.nram_size,
                            self.dim_3 * self.dim_2,
                            self.dim_1,
                        )
                        if self.dtype != "float32":
                            tcp.take(
                                out_nram.reinterpret_cast("float32"),
                                index_1_nram[: self.nram_size].reinterpret_cast(
                                    "float32"
                                ),
                                cast_nram,
                            )
                        else:
                            tcp.take(
                                out_nram.reinterpret_cast("float32"),
                                index_1_nram[: self.nram_size].reinterpret_cast(
                                    "float32"
                                ),
                                data_nram,
                            )
                        self.gather_data(
                            out_nram_int64, out_nram, self.nram_size, self.dim_num - 3
                        )

                    # Four dims compute.
                    if self.dim_num > 3:
                        self.indices_set(
                            index_1_nram,
                            global_index,
                            self.nram_size,
                            self.dim_3 * self.dim_2 * self.dim_1,
                            self.dim_0,
                        )
                        if self.dtype != "float32":
                            tcp.take(
                                out_nram.reinterpret_cast("float32"),
                                index_1_nram[: self.nram_size].reinterpret_cast(
                                    "float32"
                                ),
                                cast_nram,
                            )
                        else:
                            tcp.take(
                                out_nram.reinterpret_cast("float32"),
                                index_1_nram[: self.nram_size].reinterpret_cast(
                                    "float32"
                                ),
                                data_nram,
                            )
                        self.gather_data(
                            out_nram_int64, out_nram, self.nram_size, self.dim_num - 4
                        )

            with tcp.block("data_copy"):
                if count_num[0] > 0:
                    if self.trans == 1:
                        out_nram_int64 = out_nram_int64[
                            : self.dim_num * self.nram_size
                        ].reshape((self.nram_size, self.dim_num))
                        out_buffer = self.out_buffer.reshape(
                            (self.num_nonzero, self.dim_num)
                        )
                        tcp.memcpy(
                            out_buffer[
                                out_offset : out_offset + count_num[0],
                                : self.dim_num,
                            ],
                            out_nram_int64[
                                : count_num[0],
                                : self.dim_num,
                            ],
                        )
                    else:
                        out_nram_int64 = out_nram_int64[
                            : self.dim_num * self.nram_size
                        ].reshape((self.dim_num, self.nram_size))
                        out_buffer = self.out_buffer.reshape(
                            (self.dim_num, self.num_nonzero)
                        )
                        tcp.memcpy(
                            out_buffer[
                                : self.dim_num, out_offset : out_offset + count_num[0]
                            ],
                            out_nram_int64[: self.dim_num, : count_num[0]],
                        )

            with tcp.block("compute"):
                if count_num[0] > 0:
                    out_offset += count_num[0]
        count_num = tcp.alloc_buffer(shape=(32,), dtype="int32", scope="nram")
        if remain > 0:
            global_index = core_index + repeat * self.nram_size

            remain_align = tcp.round_up(remain, self.align_size)

            data_nram = tcp.alloc_buffer(
                shape=(self.nram_size,),
                dtype=self.dtype,
                scope="nram",
            )

            out_nram_int64 = tcp.alloc_buffer(
                shape=(self.nram_size * self.max_dim_num,),
                dtype="int64",
                scope="nram",
            )

            tcp.assign(data_nram[:remain_align], zero)
            tcp.assign(
                out_nram_int64.reinterpret_cast("int32")[
                    : self.dim_num * remain_align * 2
                ],
                0,
            )
            tcp.memcpy(
                data_nram[:remain],
                self.in_buffer.flatten()[global_index : global_index + remain],
            )
            if self.dtype != "float32":
                tcp.type_convert(cast_nram[:remain_align], data_nram[:remain_align])
                # cast_nram only computes if there are non-zero elements in it.
                tcp.count_nonzero(
                    count_num.reinterpret_cast("uint32"), cast_nram[:remain_align]
                )
            else:
                # cast_nram = data_nram
                # cast_nram only computes if there are non-zero elements in it.
                tcp.count_nonzero(
                    count_num.reinterpret_cast("uint32"), data_nram[:remain_align]
                )

            if count_num[0] > 0:

                # One dims compute.
                for k in range(remain):  # type: ignore
                    index_0_nram[k] = (global_index + k) % self.dim_3
                if self.dtype != "float32":
                    tcp.take(
                        out_nram.reinterpret_cast("float32")[:remain_align],
                        index_0_nram[:remain_align].reinterpret_cast("float32"),
                        cast_nram[:remain_align],
                    )
                else:
                    tcp.take(
                        out_nram.reinterpret_cast("float32")[:remain_align],
                        index_0_nram[:remain_align].reinterpret_cast("float32"),
                        data_nram[:remain_align],
                    )
                self.gather_data(
                    out_nram_int64, out_nram, remain_align, self.dim_num - 1
                )

                # Two dims compute.
                if self.dim_num > 1:
                    self.indices_set(
                        index_1_nram, global_index, remain, self.dim_3, self.dim_2
                    )
                    if self.dtype != "float32":
                        tcp.take(
                            out_nram.reinterpret_cast("float32")[:remain_align],
                            index_1_nram[:remain_align].reinterpret_cast("float32"),
                            cast_nram[:remain_align],
                        )
                    else:
                        tcp.take(
                            out_nram.reinterpret_cast("float32")[:remain_align],
                            index_1_nram[:remain_align].reinterpret_cast("float32"),
                            data_nram[:remain_align],
                        )
                    self.gather_data(
                        out_nram_int64, out_nram, remain_align, self.dim_num - 2
                    )

                # Tree dims compute.
                if self.dim_num > 2:
                    self.indices_set(
                        index_1_nram,
                        global_index,
                        remain,
                        self.dim_3 * self.dim_2,
                        self.dim_1,
                    )
                    if self.dtype != "float32":
                        tcp.take(
                            out_nram.reinterpret_cast("float32")[:remain_align],
                            index_1_nram[:remain_align].reinterpret_cast("float32"),
                            cast_nram[:remain_align],
                        )
                    else:
                        tcp.take(
                            out_nram.reinterpret_cast("float32")[:remain_align],
                            index_1_nram[:remain_align].reinterpret_cast("float32"),
                            data_nram[:remain_align],
                        )
                    self.gather_data(
                        out_nram_int64, out_nram, remain_align, self.dim_num - 3
                    )

                # Four dims compute.
                if self.dim_num > 3:
                    self.indices_set(
                        index_1_nram,
                        global_index,
                        remain,
                        self.dim_3 * self.dim_2 * self.dim_1,
                        self.dim_0,
                    )
                    if self.dtype != "float32":
                        tcp.take(
                            out_nram.reinterpret_cast("float32")[:remain_align],
                            index_1_nram[:remain_align].reinterpret_cast("float32"),
                            cast_nram[:remain_align],
                        )
                    else:
                        tcp.take(
                            out_nram.reinterpret_cast("float32")[:remain_align],
                            index_1_nram[:remain_align].reinterpret_cast("float32"),
                            data_nram[:remain_align],
                        )
                    self.gather_data(
                        out_nram_int64, out_nram, remain_align, self.dim_num - 4
                    )

                if self.trans == 1:
                    out_nram_int64 = out_nram_int64[
                        : self.dim_num * remain_align
                    ].reshape((remain_align, self.dim_num))
                    out_buffer = self.out_buffer.reshape(
                        (self.num_nonzero, self.dim_num)
                    )
                    tcp.memcpy(
                        out_buffer[
                            out_offset : out_offset + count_num[0],
                            : self.dim_num,
                        ],
                        out_nram_int64[
                            : count_num[0],
                            : self.dim_num,
                        ],
                    )
                else:
                    out_nram_int64 = out_nram_int64[
                        : self.dim_num * remain_align
                    ].reshape((self.dim_num, remain_align))
                    out_buffer = self.out_buffer.reshape(
                        (self.dim_num, self.num_nonzero)
                    )
                    tcp.memcpy(
                        out_buffer[
                            : self.dim_num, out_offset : out_offset + count_num[0]
                        ],
                        out_nram_int64[: self.dim_num, : count_num[0]],
                    )

    def main(
        self,
        in_buffer: ty.handle,
        core_count: ty.handle,
        dim_0: ty.int32,
        dim_1: ty.int32,
        dim_2: ty.int32,
        dim_3: ty.int32,
        dim_num: ty.int32,
        num_nonzero: ty.int32,
        trans: ty.int32,
        out_buffer: ty.handle,
    ) -> None:
        """The entry of the NonZero operator."""
        tgt = tcp.target()
        self.dim_0 = dim_0
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.dim_3 = dim_3
        self.dim_num = dim_num
        self.num_nonzero = num_nonzero
        self.trans = trans
        self.in_buffer = tcp.match_buffer(
            in_buffer,
            [dim_0, dim_1, dim_2, dim_3],
            self.dtype,
        )
        self.core_count = tcp.match_buffer(
            core_count,
            [
                tgt.core_num,
            ],
            "uint32",
        )
        self.out_buffer = tcp.match_buffer(
            out_buffer,
            [
                self.num_nonzero * self.dim_num,
            ],
            "int64",
        )
        self.nram_size = ((tgt.nram_size - 30 * 1024) // 22 // 4) // 128 * 128
        for i in tcp.thread_binding(0, 1, thread="blockIdx.x"):
            for j in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):
                # Multi-core compute split.
                task_dim = tgt.core_num
                task_id = i * tgt.core_num + j
                elem_size = self.dim_0 * self.dim_1 * self.dim_2 * self.dim_3
                pre_core = elem_size / task_dim
                core_remain = elem_size % task_dim
                out_offset = 0

                if self.core_count[task_id] != tcp.cast(0, "uint32"):
                    core_index = pre_core * task_id
                    if task_id < core_remain:
                        pre_core = pre_core + 1
                        core_index = core_index + task_id
                    else:
                        core_index = core_index + core_remain

                    for task_ in range(task_id):  # type: ignore
                        out_offset = out_offset + self.core_count[task_]
                    self.core_compute(
                        pre_core,
                        core_index,
                        out_offset,
                    )


@tcp.register_mlu_op(DTYPES, TARGET_LIST, "NonZero")
def build_nonzero(dtype, target):
    align_size = 64 if target[:6] == "mlu370" else 128
    f_nonzero = build_module.build(
        NonZero(
            target[:6],
            dtype.name,
            dtype.bytes,
            align_size,
        ),
        target,
        "NonZero",
    )
    return f_nonzero
