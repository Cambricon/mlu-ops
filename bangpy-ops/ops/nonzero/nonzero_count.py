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
# pylint: disable=attribute-defined-outside-init
"""NonZeroCount operator implementation using BANGPy TCP API."""
import bangpy as bp
from bangpy import tcp
from bangpy.script import build_module, ty

DTYPES = [bp.float16, bp.float32]
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]


class NonZeroCount(object):
    """Operator description:
    ONNX NonZero operator, behaves similar to numpy.nonzero.
    NonZeroCount is used to count the number of NonZeros on each core.
    """

    def __init__(
        self,
        dtype: ty.string,
        align_size: ty.int32,
    ):
        self.dtype = dtype
        self.align_size = align_size
        if self.dtype == "float32":
            self.mode = "rd"
        else:
            self.mode = None

    def core_compute(
        self,
        pre_core: ty.int32,
        core_index: ty.int32,
        out_buffer: ty.Buffer("nram"),  # type: ignore
    ):
        """NonZero count compute on each core."""
        repeat = pre_core / self.nram_size
        remain = pre_core % self.nram_size
        # The count number buffer for count_nonzero, size at least 128 bytes.
        count_num = tcp.alloc_buffer(shape=(32,), dtype="uint32", scope="nram")

        cast_nram = tcp.alloc_buffer(
            shape=(self.nram_size,),
            dtype="float16",
            scope="nram",
        )

        total_count = tcp.uint32(0)
        global_index = 0

        for i in range(repeat, pipeline=True):  # type: ignore
            data_nram = tcp.alloc_buffer(
                shape=(self.nram_size,),
                dtype=self.dtype,
                scope="nram",
            )
            with tcp.block("data_copy"):
                global_index = core_index + i * self.nram_size
                tcp.memcpy(
                    data_nram,
                    self.in_buffer.flatten()[
                        global_index : global_index + self.nram_size
                    ],
                )
            with tcp.block("compute"):
                if self.dtype != "float16":
                    tcp.type_convert(cast_nram, data_nram, mode=self.mode)
                    tcp.count_nonzero(count_num, cast_nram)
                else:
                    tcp.count_nonzero(count_num, data_nram)
                total_count += count_num[0]
        if remain > 0:
            global_index = core_index + repeat * self.nram_size
            remain_align = tcp.round_up(remain, self.align_size)
            data_nram = tcp.alloc_buffer(
                shape=(self.nram_size,),
                dtype=self.dtype,
                scope="nram",
            )
            tcp.assign(data_nram[:remain_align], 0.0)
            tcp.memcpy(
                data_nram[:remain],
                self.in_buffer.flatten()[global_index : global_index + remain],
            )
            if self.dtype != "float16":
                tcp.type_convert(
                    cast_nram[:remain_align], data_nram[:remain_align], mode=self.mode
                )
                tcp.count_nonzero(count_num, cast_nram[:remain_align])
            else:
                tcp.count_nonzero(count_num, data_nram[:remain_align])
            total_count += count_num[0]
        out_buffer[0] = total_count

    def main(
        self,
        in_buffer: ty.handle,
        dim_0: ty.int32,
        dim_1: ty.int32,
        dim_2: ty.int32,
        dim_3: ty.int32,
        core_count: ty.handle,
    ) -> None:
        """The entry of the NonZero count operator."""
        tgt = tcp.target()
        self.dim_0 = dim_0
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.dim_3 = dim_3
        self.in_buffer = tcp.match_buffer(
            in_buffer,
            (self.dim_0, self.dim_1, self.dim_2, self.dim_3),
            self.dtype,
        )
        self.core_count = tcp.match_buffer(
            core_count,
            (tgt.core_num,),
            "uint32",
        )
        self.nram_size = ((tgt.nram_size - 30 * 1024) // 4 // 3) // 128 * 128
        for i in tcp.thread_binding(0, 1, thread="blockIdx.x"):
            for j in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):
                task_dim = tgt.core_num
                task_id = i * tgt.core_num + j
                elem_size = self.dim_0 * self.dim_1 * self.dim_2 * self.dim_3
                pre_core = elem_size / task_dim
                core_remain = elem_size % task_dim
                core_index = pre_core * task_id
                if task_id < core_remain:
                    pre_core = pre_core + 1
                    core_index = core_index + task_id
                else:
                    core_index = core_index + core_remain
                self.core_compute(pre_core, core_index, self.core_count[task_id])


@tcp.register_mlu_op(DTYPES, TARGET_LIST, "NonZeroCount")
def build_nonzero_count(dtype, target):
    align_size = 64 if target[:6] == "mlu370" else 128

    f_nonzero_count = build_module.build(
        NonZeroCount(dtype.name, align_size),
        target,
        "NonZeroCount",
    )
    return f_nonzero_count
