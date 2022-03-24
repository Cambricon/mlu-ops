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
import numpy as np

import bangpy
import bangpy as bp
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float32] #支持的类型
TARGET_LIST = ["mlu290"]#支持的设备
KERNEL_NAME = "PairwiseDistance"#算子名


class PairwiseDistance(object):
    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.bp = tcp.TCP(target)

    def compute_body(self):
        self.bp.launch_task(self.task_num, 1, 1)

        self.x_length = self.bp.SizeVar("x_length")
        self.y_length = self.bp.SizeVar("y_length")
        self.shp_x_len = self.bp.SizeVar("shp_x_len")
        self.shp_y_len = self.bp.SizeVar("shp_y_len")

        self.output_len = self.bp.SizeVar("output_len")

        gram_x = self.bp.Buffer(
            shape=(self.x_length,), name="gram_x", dtype=self.dtype, scope="global"
        )

        gram_y = self.bp.Buffer(
            shape=(self.y_length,), name="gram_y", dtype=self.dtype, scope="global"
        )

        gram_shp_x = self.bp.Buffer(
            shape=(self.shp_x_len,), name="gram_shp_x", dtype=bp.int32, scope="global"
        )

        gram_shp_y = self.bp.Buffer(
            shape=(self.shp_y_len,), name="gram_shp_y", dtype=bp.int32, scope="global"
        )

        buffer_out = self.bp.Buffer(
            shape=(self.output_len,), name="OUTPUT", dtype=self.dtype, scope="global"
        )
        
        self.bp.print(gram_x)
        self.bp.print(gram_y)
        self.bp.print(gram_shp_x)
        self.bp.print(gram_shp_y)
        self.bp.print('shp x len ', self.shp_x_len)
        self.bp.print(self.shp_y_len)
        #self.bp.print(buffer_out)

        f = self.bp.BuildBANG(
            inputs=[gram_x, gram_y, 
                    gram_shp_x, gram_shp_y,
                    self.x_length, self.y_length, self.shp_x_len, self.shp_y_len,
                    self.output_len],
            outputs=[buffer_out],
            kernel_name=KERNEL_NAME,
        )
        return f

@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_pairwisedistance(dtype=None, target=None):
    task_num = 1
    f = PairwiseDistance(dtype, target, task_num).compute_body()
    return f
