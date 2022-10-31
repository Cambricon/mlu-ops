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
"""HardShrink operator implementation using BANGPy TCP API."""
import bangpy
from bangpy import tcp
from bangpy.script import build_module, ty
from bangpy.platform.bang_config import TARGET

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu290", "mlu270", "mlu370-s4", "mlu370-m8"]
KERNEL_NAME = "hardshrink"


class HardShrink(object):
    """Operator description:
    An activation function, behaves similar to torch.hardshrink.
    """

    def __init__(self, cluster_num: ty.int32, dtype: ty.string) -> None:
        self.dtype = dtype
        self.cluster_num = cluster_num

    def compute(
        self,
        tensor_input_nram: ty.Buffer("nram"),
        tensor_abs_nram: ty.Buffer("nram"),
        tensor_lambda_nram: ty.Buffer("nram"),
        tensor_greater_nram: ty.Buffer("nram"),
        tensor_output_nram: ty.Buffer("nram"),
    ) -> None:
        """Include the main compute"""
        tcp.abs(tensor_abs_nram, tensor_input_nram)
        tcp.greater_equal(tensor_greater_nram, tensor_abs_nram, tensor_lambda_nram)
        tcp.multiply(tensor_output_nram, tensor_input_nram, tensor_greater_nram)

    def hardshrink_compute(self):
        """The main compute pipeline"""
        self.task_num = self.cluster_num * self.core_num

        task_content = self.element_num // self.task_num
        task_remain = self.element_num % self.task_num

        tensor_input_flatten = self.tensor_input.reshape((self.element_num,))
        tensor_output_flatten = self.tensor_output.reshape((self.element_num,))

        cmpt_times = task_content // self.nram_use_size
        remain_num = task_content % self.nram_use_size


        for cluster_id in tcp.thread_binding(0, self.cluster_num, thread="blockIdx.x"):
            for core_id in tcp.thread_binding(0, self.core_num, thread="threadIdx.x"):
                task_id = cluster_id * self.core_num + core_id
                task_start = task_id * task_content
                task_end = task_start + task_content

                for c_t in range(cmpt_times, pipeline=True):
                    tensor_input_nram = tcp.alloc_buffer(
                        [self.nram_use_size], dtype=self.dtype, scope="nram"
                    )
                    tensor_output_nram = tcp.alloc_buffer(
                        [self.nram_use_size], dtype=self.dtype, scope="nram"
                    )
                    tensor_abs_nram = tcp.alloc_buffer(
                        [self.nram_use_size], dtype=self.dtype, scope="nram"
                    )
                    tensor_lambda_nram = tcp.alloc_buffer(
                        [self.nram_use_size], dtype=self.dtype, scope="nram"
                    )

                    tensor_greater_nram = tcp.alloc_buffer(
                        [self.nram_use_size], dtype=self.dtype, scope="nram"
                    )
                    with tcp.block("data_copy"):
                        cmpt_start = task_start + c_t * self.nram_use_size
                        cmpt_end = cmpt_start + self.nram_use_size
                        tcp.memcpy(
                            tensor_input_nram,
                            tensor_input_flatten[cmpt_start:cmpt_end],
                        )
                    with tcp.block("compute"):
                        tcp.assign(tensor_lambda_nram, tcp.cast(self.lambda_para, self.dtype))
                        self.compute(
                            tensor_input_nram,
                            tensor_abs_nram,
                            tensor_lambda_nram,
                            tensor_greater_nram,
                            tensor_output_nram,
                        )
                    with tcp.block("data_copy"):
                        cmpt_start = task_start + c_t * self.nram_use_size
                        cmpt_end = cmpt_start + self.nram_use_size
                        tcp.memcpy(
                            tensor_output_flatten[cmpt_start:cmpt_end],
                            tensor_output_nram,
                        )

                tensor_input_nram = tcp.alloc_buffer(
                    [self.nram_use_size], dtype=self.dtype, scope="nram"
                )
                tensor_output_nram = tcp.alloc_buffer(
                    [self.nram_use_size], dtype=self.dtype, scope="nram"
                )
                tensor_abs_nram = tcp.alloc_buffer(
                    [self.nram_use_size], dtype=self.dtype, scope="nram"
                )
                tensor_lambda_nram = tcp.alloc_buffer(
                    [self.nram_use_size], dtype=self.dtype, scope="nram"
                )

                tcp.assign(tensor_lambda_nram, tcp.cast(self.lambda_para, self.dtype))
                tensor_greater_nram = tcp.alloc_buffer(
                    [self.nram_use_size], dtype=self.dtype, scope="nram"
                )
                if remain_num != 0:
                    start_pos = task_end - remain_num
                    end_pos = task_end
                    with tcp.block("data_copy"):
                        tcp.memcpy(
                            tensor_input_nram[:remain_num],
                            tensor_input_flatten[start_pos:end_pos],
                        )
                    with tcp.block("compute"):
                        self.compute(
                            tensor_input_nram,
                            tensor_abs_nram,
                            tensor_lambda_nram,
                            tensor_greater_nram,
                            tensor_output_nram,
                        )
                    with tcp.block("data_copy"):
                        tcp.memcpy(
                            tensor_output_flatten[start_pos:end_pos],
                            tensor_output_nram[:remain_num],
                        )

                if task_remain != 0:
                    remain_by_thistask = task_remain // self.task_num
                    tensor_input_small_nram = tcp.alloc_buffer(
                        [self.base_align], dtype=self.dtype, scope="nram"
                    )
                    tensor_abs_small_nram = tcp.alloc_buffer(
                        [self.base_align], dtype=self.dtype, scope="nram"
                    )
                    tensor_lambda_small_nram = tcp.alloc_buffer(
                        [self.base_align], dtype=self.dtype, scope="nram"
                    )
                    tcp.assign(
                        tensor_lambda_small_nram, tcp.cast(self.lambda_para, self.dtype)
                    )
                    tensor_greater_small_nram = tcp.alloc_buffer(
                        [self.base_align], dtype=self.dtype, scope="nram"
                    )
                    tensor_output_small_nram = tcp.alloc_buffer(
                        [self.base_align], dtype=self.dtype, scope="nram"
                    )

                    content_end = task_content * self.task_num
                    if task_remain % self.task_num > task_id:
                        remain_by_thistask += 1
                        remain_task_start = content_end + task_id * remain_by_thistask
                        remain_task_end = remain_task_start + remain_by_thistask
                        with tcp.block("data_copy"):
                            tcp.memcpy(
                                tensor_input_small_nram[:remain_by_thistask],
                                tensor_input_flatten[remain_task_start:remain_task_end],
                            )
                        with tcp.block("compute"):
                            self.compute(
                                tensor_input_small_nram,
                                tensor_abs_small_nram,
                                tensor_lambda_small_nram,
                                tensor_greater_small_nram,
                                tensor_output_small_nram,
                            )
                        with tcp.block("data_copy"):
                            tcp.memcpy(
                                tensor_output_flatten[
                                    remain_task_start:remain_task_end
                                ],
                                tensor_output_small_nram[:remain_by_thistask],
                            )
        self.tensor_output = tensor_output_flatten.reshape(self.tensor_shape)

    def main(
        self,
        data_in_dev: ty.handle,
        lambda_para: ty.float32,
        dim_0: ty.int32,
        dim_1: ty.int32,
        dim_2: ty.int32,
        dim_3: ty.int32,
        data_out_dev: ty.handle,
    ) -> None:
        """The main entry"""
        tgt = tcp.target()

        self.lambda_para = lambda_para
        self.dim_0 = dim_0
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.dim_3 = dim_3
        self.element_num = self.dim_0 * self.dim_1 * self.dim_2 * self.dim_3
        self.tensor_shape = (self.dim_0, self.dim_1, self.dim_2, self.dim_3)

        self.nram_size = tgt.nram_size
        self.core_num = tgt.core_num
        self.base_align = 64
        dtype_bytes = 2 if self.dtype == "float16" else 4
        self.nram_use_size = tcp.round_up(self.nram_size // (8 * dtype_bytes), self.base_align)

        self.tensor_input = tcp.match_buffer(
            data_in_dev, self.tensor_shape, dtype=self.dtype
        )
        self.tensor_output = tcp.match_buffer(
            data_out_dev, self.tensor_shape, dtype=self.dtype
        )

        self.hardshrink_compute()


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_hardshrink(dtype=None, target=None):

    f_hardshrink = build_module.build(
        HardShrink(TARGET(target).cluster_num, dtype.name), target_tag=target, name=KERNEL_NAME
    )
    return f_hardshrink
