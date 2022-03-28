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
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float16]
TARGET_LIST = ["mlu290"]
KERNEL_NAME = "frac"


class Frac(object):
    """Operator description:
    compute the fraction part of the elements in the input tensor
    """

    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.bp = tcp.TCP(target)
        # self.length = self.bp.Scalar(dtype=bangpy.int32, name="length")
        self.dim_0 = self.bp.SizeVar("dim_0")
        self.dim_1 = self.bp.SizeVar("dim_1")
        self.dim_2 = self.bp.SizeVar("dim_2")
        self.dim_3 = self.bp.SizeVar("dim_3")
        self.length = self.dim_0 * self.dim_1 * self.dim_2 * self.dim_3
        self.nram_size = TARGET(target).nram_size
        self.dtype_sz = dtype.bytes
        self.single_buffer_size = 1024
        self.bp.launch_task(self.task_num, 1, 1)

    def compute_body(self):
        #calculate basic data
        
        data_calculated_each_task = self.length
        loop_num = data_calculated_each_task * self.dtype_sz // self.single_buffer_size
        data_calculated_each_time = self.single_buffer_size // self.dtype_sz
        remain = (data_calculated_each_task * self.dtype_sz) % self.single_buffer_size

        buffer_original = self.bp.Buffer(
            shape=(self.dim_0, self.dim_1, self.dim_2, self.dim_3), 
            name="INPUT", dtype=self.dtype, scope="global"
        )
        buffer_in = buffer_original.reshape((self.length,))
        buffer_final = self.bp.Buffer(
            shape=(self.dim_0, self.dim_1, self.dim_2, self.dim_3), 
            name="OUTPUT", dtype=self.dtype, scope="global"
        )
        buffer_out = buffer_final.reshape((self.length,))
        task_id = self.bp.taskId

        buffer_in_n = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="INPUT_N",
            dtype=self.dtype,
            scope="nram",
        )
        buffer_out_n = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="OUTPUT_N",
            dtype=self.dtype,
            scope="nram",
        )
        buffer_abs = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="abs",
            dtype=self.dtype,
            scope="nram",
        )
        buffer_floor = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="floor",
            dtype=bangpy.int16,
            scope="nram",
        )
        buffer_floor_after = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="floorafter",
            dtype=self.dtype,
            scope="nram",
        )
        buffer_sgn = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="sgn",
            dtype=self.dtype,
            scope="nram",
        )
        buffer_tem = self.bp.Buffer(
            shape=(data_calculated_each_time,),
            name="tem",
            dtype=self.dtype,
            scope="nram",
        )
        
        with self.bp.for_range(0, loop_num) as i:
            start = i * data_calculated_each_time
            stop = start + data_calculated_each_time
            self.bp.memcpy(buffer_in_n, buffer_in[start:stop])
            #self.bp.abs(buffer_out_n, buffer_in_n)
            
            self.bp.abs(buffer_abs, buffer_in_n)
            self.bp.type_convert(buffer_floor, buffer_abs, 0, "tz")
            self.bp.type_convert(buffer_floor_after, buffer_floor, 0, "tz")
            self.bp.sign(buffer_sgn, buffer_in_n)
            self.bp.multiply(buffer_tem, buffer_floor_after, buffer_sgn)
            self.bp.subtract(buffer_out_n, buffer_in_n, buffer_tem)
            
            self.bp.memcpy(buffer_out[start:stop], buffer_out_n)
        start = loop_num * data_calculated_each_time
        stop = start + remain
        offset = stop - start
        
        with self.bp.if_scope(start != stop):
            self.bp.memcpy(buffer_in_n[0:offset], buffer_in[start:stop])
            self.bp.abs(buffer_abs[0:offset], buffer_in_n[0:offset])
            self.bp.type_convert(buffer_floor[0:offset], buffer_abs[0:offset], 2, "tz")
            self.bp.type_convert(buffer_floor_after[0:offset], buffer_floor[0:offset], 2, "tz")
            self.bp.sign(buffer_sgn[0:offset], buffer_in_n[0:offset])
            self.bp.multiply(buffer_tem[0:offset], buffer_floor_after[0:offset], buffer_sgn[0:offset])
            self.bp.subtract(buffer_out_n[0:offset], buffer_in_n[0:offset], buffer_tem[0:offset])
            self.bp.abs(buffer_out_n, buffer_in_n)
            self.bp.memcpy(buffer_out[start:stop], buffer_out_n[0:offset])
        
        buffer_original = buffer_in.reshape((self.dim_0, self.dim_1, self.dim_2, self.dim_3))
        buffer_final = buffer_out.reshape((self.dim_0, self.dim_1, self.dim_2, self.dim_3))
        f = self.bp.BuildBANG(
            inputs=[buffer_original,
                    self.dim_0,
                    self.dim_1,
                    self.dim_2,
                    self.dim_3],
            outputs=[buffer_final],
            kernel_name=KERNEL_NAME,
        )
        return f


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_frac(dtype=None, target=None):
    # tasktype fixed in UNION1
    task_num = 1
    f = Frac(dtype, target, task_num).compute_body()
    return f
