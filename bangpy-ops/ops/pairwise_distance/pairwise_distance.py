# Copyright (C) [2021] by Cambricon, Inc.
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
from bangpy import tcp
from bangpy.script import ty, build_module


DTYPES = [#bangpy.float16,
           bangpy.float32]
TARGET_LIST = ["mlu370-s4"]
KERNEL_NAME = "pairwise_distance"


class PairwiseDistance(object):
    """Operator description:
    Add the data in the two buffers.
    """
    def __init__(self, buffer_size: ty.int32, dtype: ty.string) -> None:
        self.dtype = dtype
        self.single_buffer_size = buffer_size

    def add_body(
        self,
        local_a: ty.Buffer("nram"),  # type: ignore
        local_b: ty.Buffer("nram"),  # type: ignore
        local_c: ty.Buffer("nram"),  # type: ignore
    ) -> None:
        # The body of add function
        tcp.add(local_a, local_b, local_c)

    def main(self, Gram_tensor1: ty.handle, Gram_tensor2: ty.handle, Gram_paras: ty.handle,
                    len_tensor1: ty.int32, len_tensor2: ty.int32,
                    pd_len: ty.int32, pd_height: ty.int32, pd_width: ty.int32,
                    output_len: ty.int32,
                    Gram_border_buf_out: ty.handle, 
                    Gram_border_idx_out: ty.handle, 
                    Gram_buffer_out: ty.handle
                    ) -> None:
        gram_tensor1 = tcp.match_buffer(Gram_tensor1, [len_tensor1], dtype=self.dtype)
        gram_tensor2 = tcp.match_buffer(Gram_tensor2, [len_tensor2], dtype=self.dtype)
        gram_paras = tcp.match_buffer(Gram_paras, [2], dtype=self.dtype)

        gram_border_buf_out = tcp.match_buffer(Gram_border_buf_out, [256], dtype=self.dtype)
        gram_border_idx_out = tcp.match_buffer(Gram_border_idx_out, [256], dtype='int32')
        gram_buffer_out = tcp.match_buffer(Gram_buffer_out, [output_len], dtype=self.dtype)        

        tgt = tcp.target()
        self.bp = tgt

        tcp.print(gram_tensor1)
        a = 0
        for cluster_id in tcp.thread_binding(0, tgt.cluster_num, thread="blockIdx.x"):
            for core_id in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):
                #self.bp.print("zouni\n")
                a += 1
                tcp.print("feifei ", a)


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_add(dtype=None, target=None):
    f = build_module.build(
        PairwiseDistance(64, dtype.name), target_tag=target, name=KERNEL_NAME
    )
    return f
