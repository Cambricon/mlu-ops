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
# pylint: disable=attribute-defined-outside-init
"""NonZeroCount operator implementation using BANGPy TCP API."""
import bangpy as bp
from bangpy import tcp
from bangpy.tcp.util import round_up, round_down
from bangpy.platform.bang_config import TARGET

DTYPES = [bp.float16, bp.float32]
TARGET_LIST = ["mlu370-s4", "mlu290", "mlu270"]

# Align to 64.
ALIGN_SIZE = 64


class NonZeroCount(object):
    """Operator description:
    ONNX NonZero operator, behaves similar to numpy.nonzero.
    NonZeroCount is used to count the number of NonZeros on each core.
    """

    def __init__(self, target, dtype=bp.float32, task_num=16, name="NonZeroCount"):
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

        nram_size : int
            The size of nram buffer needed for one calculation.

        int_type : bangpy.DType
            The compute int data type.

        float_type : bangpy.DType
            The compute float data type.
        """
        self.tcp = tcp.TCP(target)
        self.dtype = dtype
        self.task_num = task_num
        self.name = name
        self.dim_0 = self.tcp.SizeVar("dim_0")
        self.dim_1 = self.tcp.SizeVar("dim_1")
        self.dim_2 = self.tcp.SizeVar("dim_2")
        self.dim_3 = self.tcp.SizeVar("dim_3")
        # 30 * 1024B reserve for stack, need 2 buffers after storage rewrite pass.
        self.nram_size = round_down(
            (TARGET(target).nram_size - 30 * 1024) // bp.int32.bytes // 3, 128
        )
        self.float_type = bp.float16

    def core_compute(self, pre_core, core_index, in_buffer, out_buffer):
        """NonZero count compute on each core."""
        repeat = self.tcp.Scalar(dtype=bp.int32, name="repeat")
        repeat.assign(pre_core / self.nram_size)
        remain = self.tcp.Scalar(dtype=bp.int32, name="remain")
        remain.assign(pre_core % self.nram_size)
        # The count number buffer for count_nonzero, size at least 128 bytes.
        count_num = self.tcp.Buffer(
            shape=(32,), dtype=bp.uint32, name="count", scope="nram"
        )

        total_count = self.tcp.Buffer(
            shape=(1,), dtype=bp.uint32, name="total_count", scope="nram"
        )

        cast_nram = self.tcp.Buffer(
            shape=(self.nram_size,),
            dtype=self.float_type,
            name="data_nram",
            scope="nram",
        )

        total_count[0] = 0
        global_index = self.tcp.Scalar(dtype=bp.int32, name="global_index")
        zero = self.tcp.Scalar(dtype=self.dtype, name="zero", value=0.0)

        with self.tcp.for_range(0, repeat, stage=1) as i:
            with self.tcp.block("compute"):
                global_index.assign(core_index + i * self.nram_size)
            data_nram = self.tcp.Buffer(
                shape=(self.nram_size,),
                dtype=self.dtype,
                name="data_nram",
                scope="nram",
            )
            with self.tcp.block("data_copy"):
                self.tcp.memcpy(
                    data_nram,
                    in_buffer.flatten()[global_index : global_index + self.nram_size],
                )
            with self.tcp.block("compute"):
                if self.dtype != self.float_type:
                    self.tcp.type_convert(cast_nram, data_nram)
                else:
                    cast_nram = data_nram
                self.tcp.count_nonzero(count_num, cast_nram)
                total_count[0] = total_count[0] + count_num[0]

        with self.tcp.if_scope(remain > 0):
            global_index.assign(core_index + repeat * self.nram_size)
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
            self.tcp.assign(data_nram[:remain_align], zero)
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
            self.tcp.count_nonzero(count_num, cast_nram[:remain_align])
            total_count[0] = total_count[0] + count_num[0]
        out_buffer[0] = total_count[0]

    def nonzero_count_compute(self):
        """The entry of the NonZero count operator."""
        self.tcp.launch_task(self.task_num, 1, 1)
        self.in_buffer = self.tcp.Buffer(
            shape=(self.dim_0, self.dim_1, self.dim_2, self.dim_3),
            dtype=self.dtype,
            name="in_buffer",
            scope="global",
        )
        self.core_count = self.tcp.Buffer(
            shape=(self.task_num,), dtype=bp.uint32, name="core_count", scope="global"
        )

        elem_size = self.tcp.Scalar(dtype=bp.int32, name="elem_size")
        elem_size.assign(self.dim_0 * self.dim_1 * self.dim_2 * self.dim_3)

        pre_core = self.tcp.Scalar(dtype=bp.int32, name="pre_core")
        pre_core.assign(elem_size / self.tcp.taskDim)
        core_remain = self.tcp.Scalar(dtype=bp.int32, name="core_remain")
        core_remain.assign(elem_size % self.tcp.taskDim)
        core_index = self.tcp.Scalar(dtype=bp.int32, name="core_index")
        core_index.assign(pre_core * self.tcp.taskId)
        with self.tcp.if_scope(self.tcp.taskId < core_remain):
            pre_core.assign(pre_core + 1)
            core_index.assign(core_index + self.tcp.taskId)
        with self.tcp.else_scope():
            core_index.assign(core_index + core_remain)
        self.core_compute(
            pre_core, core_index, self.in_buffer, self.core_count[self.tcp.taskId]
        )

        return self.tcp.BuildBANG(
            inputs=[self.in_buffer, self.dim_0, self.dim_1, self.dim_2, self.dim_3],
            outputs=[self.core_count],
            kernel_name=self.name,
        )


@tcp.register_mlu_op(DTYPES, TARGET_LIST, "NonZeroCount")
def build_nonzero_count(dtype, target):
    task_num = TARGET(target).cluster_num * TARGET(target).core_num

    nonzero_count = NonZeroCount(target, task_num=task_num, dtype=dtype)
    f_nonzero_count = nonzero_count.nonzero_count_compute()
    return f_nonzero_count
