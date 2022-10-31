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
"""Lerp operator implementation using BANGPy TCP API."""
import bangpy
from bangpy import tcp
from bangpy.platform.bang_config import TARGET
from bangpy.script import build_module, ty

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu290", "mlu370-s4"]
KERNEL_NAME = "lerp"


class Lerp(object):
    """Operator description:
    A linear interpolation of two tensors, behaves similar to torch.lerp.
    """

    def __init__(self, cluster_num: ty.int32, dtype: ty.string) -> None:
        self.dtype = dtype
        self.cluster_num = cluster_num

    def compute(
        self,
        tensor_input_start_nram: ty.Buffer("nram"),
        tensor_input_end_nram: ty.Buffer("nram"),
        tensor_weight_nram: ty.Buffer("nram"),
    ) -> None:
        """Include the main compute"""
        tcp.subtract(
            tensor_input_end_nram, tensor_input_end_nram, tensor_input_start_nram
        )
        tcp.multiply(tensor_input_end_nram, tensor_input_end_nram, tensor_weight_nram)
        tcp.add(tensor_weight_nram, tensor_input_start_nram, tensor_input_end_nram)

    def lerp_compute(self):
        """The main compute pipeline"""
        self.task_num = self.cluster_num * self.core_num

        task_content = self.element_num // self.task_num
        task_remain = self.element_num % self.task_num

        for cluster_id in tcp.thread_binding(0, self.cluster_num, thread="blockIdx.x"):
            for core_id in tcp.thread_binding(0, self.core_num, thread="threadIdx.x"):
                task_id = cluster_id * self.core_num + core_id
                task_start = task_id * task_content
                task_end = task_start + task_content

                cmpt_times = task_content * 2 // self.nram_use_size
                remain_num = task_content * 2 % self.nram_use_size
                calculated_each_time = self.nram_use_size // 2
                tensor_input_start_flatten = (
                    self.tensor_input_start.flatten()
                )
                tensor_input_end_flatten = (
                    self.tensor_input_end.flatten()
                )
                tensor_output_flatten = (
                    self.tensor_output.flatten()
                )
                tensor_weight_flatten = (
                    self.tensor_weight.flatten()
                )

                for c_t in range(cmpt_times, pipeline=True):
                    tensor_input_start_nram = tcp.alloc_buffer(
                        [calculated_each_time], dtype=self.dtype, scope="nram"
                    )
                    tensor_input_end_nram = tcp.alloc_buffer(
                        [calculated_each_time], dtype=self.dtype, scope="nram"
                    )
                    tensor_weight_nram = tcp.alloc_buffer(
                        [calculated_each_time], dtype=self.dtype, scope="nram"
                    )

                    with tcp.block("data_copy"):
                        cmpt_start = task_start + c_t * calculated_each_time
                        cmpt_end = cmpt_start + calculated_each_time
                        tcp.memcpy(
                            tensor_input_start_nram,
                            tensor_input_start_flatten[cmpt_start:cmpt_end],
                        )
                        tcp.memcpy(
                            tensor_input_end_nram,
                            tensor_input_end_flatten[cmpt_start:cmpt_end],
                        )
                        tcp.memcpy(
                            tensor_weight_nram,
                            tensor_weight_flatten[cmpt_start:cmpt_end],
                        )
                    with tcp.block("compute"):
                        self.compute(
                            tensor_input_start_nram,
                            tensor_input_end_nram,
                            tensor_weight_nram,
                        )
                    with tcp.block("data_copy"):
                        cmpt_start = task_start + c_t * calculated_each_time
                        cmpt_end = cmpt_start + calculated_each_time
                        tcp.memcpy(
                            tensor_output_flatten[cmpt_start:cmpt_end],
                            tensor_weight_nram,
                        )
                cmpt_times = task_content // calculated_each_time
                remain_num = task_content % calculated_each_time

                tensor_input_start_nram = tcp.alloc_buffer(
                    [calculated_each_time], dtype=self.dtype, scope="nram"
                )
                tensor_input_end_nram = tcp.alloc_buffer(
                    [calculated_each_time], dtype=self.dtype, scope="nram"
                )
                tensor_weight_nram = tcp.alloc_buffer(
                    [calculated_each_time], dtype=self.dtype, scope="nram"
                )

                if remain_num != 0:
                    start_pos = task_end - remain_num
                    end_pos = task_end
                    with tcp.block("data_copy"):
                        tcp.memcpy(
                            tensor_input_start_nram[:remain_num],
                            tensor_input_start_flatten[start_pos:end_pos],
                        )
                        tcp.memcpy(
                            tensor_input_end_nram[:remain_num],
                            tensor_input_end_flatten[start_pos:end_pos],
                        )
                        tcp.memcpy(
                            tensor_weight_nram[:remain_num],
                            tensor_weight_flatten[start_pos:end_pos],
                        )
                    with tcp.block("compute"):
                        self.compute(
                            tensor_input_start_nram,
                            tensor_input_end_nram,
                            tensor_weight_nram,
                        )
                    with tcp.block("data_copy"):
                        tcp.memcpy(
                            tensor_output_flatten[start_pos:end_pos],
                            tensor_weight_nram[:remain_num],
                        )
                if task_remain != 0:
                    remain_by_thistask = task_remain // self.task_num

                    tensor_input_small_start_nram = tcp.alloc_buffer(
                        [self.base_align], dtype=self.dtype, scope="nram"
                    )
                    tensor_input_small_end_nram = tcp.alloc_buffer(
                        [self.base_align], dtype=self.dtype, scope="nram"
                    )
                    tensor_weight_small_nram = tcp.alloc_buffer(
                        [self.base_align], dtype=self.dtype, scope="nram"
                    )

                    if task_remain % self.task_num > task_id:
                        remain_by_thistask += 1
                        content_end = task_content * self.task_num
                        remain_task_start = content_end + task_id * remain_by_thistask
                        remain_task_end = remain_task_start + remain_by_thistask
                        with tcp.block("data_copy"):
                            tcp.memcpy(
                                tensor_input_small_start_nram[:remain_by_thistask],
                                tensor_input_start_flatten[
                                    remain_task_start:remain_task_end
                                ],
                            )
                            tcp.memcpy(
                                tensor_input_small_end_nram[:remain_by_thistask],
                                tensor_input_end_flatten[
                                    remain_task_start:remain_task_end
                                ],
                            )
                            tcp.memcpy(
                                tensor_weight_small_nram[:remain_by_thistask],
                                tensor_weight_flatten[
                                    remain_task_start:remain_task_end
                                ],
                            )
                        with tcp.block("compute"):
                            self.compute(
                                tensor_input_small_start_nram,
                                tensor_input_small_end_nram,
                                tensor_weight_small_nram,
                            )
                        with tcp.block("data_copy"):
                            tcp.memcpy(
                                tensor_output_flatten[
                                    remain_task_start:remain_task_end
                                ],
                                tensor_weight_small_nram[:remain_by_thistask],
                            )

    def main(
        self,
        data_in_start_dev: ty.handle,
        data_in_end_dev: ty.handle,
        data_weight_dev: ty.handle,
        dim_0: ty.int32,
        dim_1: ty.int32,
        dim_2: ty.int32,
        dim_3: ty.int32,
        data_out_dev: ty.handle,
    ) -> None:
        """The main entry"""
        tgt = tcp.target()

        self.data_weight_dev = data_weight_dev
        self.dim_0 = dim_0
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.dim_3 = dim_3
        self.element_num = self.dim_0 * self.dim_1 * self.dim_2 * self.dim_3
        self.tensor_shape = (self.dim_0, self.dim_1, self.dim_2, self.dim_3)

        self.nram_size = tgt.nram_size
        self.core_num = tgt.core_num
        self.base_align = 64
        self.nram_use_size = tcp.round_up(tgt.nram_size // 16, self.base_align)

        self.tensor_input_start = tcp.match_buffer(
            data_in_start_dev, self.tensor_shape, dtype=self.dtype
        )
        self.tensor_input_end = tcp.match_buffer(
            data_in_end_dev, self.tensor_shape, dtype=self.dtype
        )
        self.tensor_weight = tcp.match_buffer(
            data_weight_dev, self.tensor_shape, dtype=self.dtype
        )
        self.tensor_output = tcp.match_buffer(
            data_out_dev, self.tensor_shape, dtype=self.dtype
        )

        self.lerp_compute()


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_lerp(dtype=None, target=None):

    f_lerp = build_module.build(
        Lerp(TARGET(target).cluster_num, dtype.name),
        target_tag=target,
        name=KERNEL_NAME,
    )
    return f_lerp
