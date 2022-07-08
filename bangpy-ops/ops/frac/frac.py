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
"""Frac operator implementation using BANGPy TCP API."""
import numpy as np

import bangpy
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float32]
TARGET_LIST = ["mlu290"]
KERNEL_NAME = "frac"


class Frac(object):
    """Operator description:
    compute the fraction part of the elements in the input tensor
    """

    def __init__(self, dtype, target, task_num, stage):
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.stage = stage
        self.bp = tcp.TCP(target)
        # self.length = self.bp.Scalar(dtype=bangpy.int32, name="length")
        self.dim_0 = self.bp.SizeVar("dim_0")
        self.dim_1 = self.bp.SizeVar("dim_1")
        self.dim_2 = self.bp.SizeVar("dim_2")
        self.dim_3 = self.bp.SizeVar("dim_3")
        self.length = self.dim_0 * self.dim_1 * self.dim_2 * self.dim_3
        self.nram_size = TARGET(target).nram_size
        self.dtype_sz = dtype.bytes
        self.single_buffer_size = (self.nram_size - 52 * 1024) // 8
        self.bp.launch_task(self.task_num, 1, 1)


    def compute_body(self):
        #calculate basic data
        
        data_calculated_each_task = self.length // self.task_num
        data_remain = self.length % self.task_num
        loop_num = data_calculated_each_task * self.dtype_sz // self.single_buffer_size
        data_calculated_each_time = self.single_buffer_size // self.dtype_sz
        each_task_remain = data_calculated_each_task % data_calculated_each_time

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
        
        
        with self.bp.for_range(0, loop_num, stage=self.stage) as i:
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
            start = task_id * data_calculated_each_task + i * data_calculated_each_time
            stop = start + data_calculated_each_time
            with self.bp.block("data_copy"):
                self.bp.memcpy(buffer_in_n, buffer_in[start:stop])
            
            with self.bp.block("compute"):
                self.bp.abs(buffer_abs, buffer_in_n)
                self.bp.type_convert(buffer_floor, buffer_abs, 0, "tz")
                self.bp.type_convert(buffer_floor_after, buffer_floor, 0, "tz")
                self.bp.sign(buffer_sgn, buffer_in_n)
                self.bp.multiply(buffer_tem, buffer_floor_after, buffer_sgn)
                self.bp.subtract(buffer_out_n, buffer_in_n, buffer_tem)
            
            with self.bp.block("data_copy"):
                self.bp.memcpy(buffer_out[start:stop], buffer_out_n)
        
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

        with self.bp.if_scope(each_task_remain != 0):
            start = task_id * data_calculated_each_task + loop_num * data_calculated_each_time
            stop = start + each_task_remain
            offset = stop - start
            self.bp.memcpy(buffer_in_n[0:offset], buffer_in[start:stop])
            self.bp.abs(buffer_abs, buffer_in_n)
            self.bp.type_convert(buffer_floor, buffer_abs, 0, "tz")
            self.bp.type_convert(buffer_floor_after, buffer_floor, 0, "tz")
            self.bp.sign(buffer_sgn, buffer_in_n)
            self.bp.multiply(buffer_tem, buffer_floor_after, buffer_sgn)
            self.bp.subtract(buffer_out_n, buffer_in_n, buffer_tem)
            self.bp.memcpy(buffer_out[start:stop], buffer_out_n[0:offset])
            

        with self.bp.if_scope(data_remain != 0):
            with self.bp.if_scope(task_id == self.task_num - 1):
                start = task_id * data_calculated_each_task + data_calculated_each_task
                stop = start + data_remain
                offset = stop - start
                self.bp.memcpy(buffer_in_n[0:offset], buffer_in[start:stop])
                self.bp.abs(buffer_abs, buffer_in_n)
                self.bp.type_convert(buffer_floor, buffer_abs, 0, "tz")
                self.bp.type_convert(buffer_floor_after, buffer_floor, 0, "tz")
                self.bp.sign(buffer_sgn, buffer_in_n)
                self.bp.multiply(buffer_tem, buffer_floor_after, buffer_sgn)
                self.bp.subtract(buffer_out_n, buffer_in_n, buffer_tem)
                self.bp.memcpy(buffer_out[start:stop], buffer_out_n[0:offset])

        buffer_original = buffer_in.reshape((self.dim_0, self.dim_1, self.dim_2, self.dim_3))
        buffer_final = buffer_out.reshape((self.dim_0, self.dim_1, self.dim_2, self.dim_3))
        f = self.bp.BuildBANG(
            inputs=[buffer_original],
            outputs=[buffer_final],
            kernel_name=KERNEL_NAME,
        )
        return f


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_frac(dtype=None, target=None):
    # tasktype fixed in UNION1
    
    task_type = TaskType.UNION4
    task_num = 4 * task_type.value
    stage = 1
    f = Frac(dtype, target, task_num, stage).compute_body()
    return f
