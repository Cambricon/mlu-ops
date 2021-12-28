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
# pylint: disable=invalid-name, missing-function-docstring, useless-object-inheritance
# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals
# pylint: disable=too-many-statements, attribute-defined-outside-init
"""Non-maximum suppression operator implementation using BANGPy TCP API."""
import pytest
import numpy as np
import bangpy as bp
import os
from bangpy import tcp
from bangpy.tcp.runtime import TaskType
from bangpy.platform.bang_config import TARGET
from bangpy.tcp.util import round_up, round_down
from bangpy.common import compile_for_multi_dtype_platform, utils, load_op_by_type

TARGET_LIST = ["mlu270", "mlu290"]
KERNEL_NAME = "nms"
DTYPES = [bp.float16]
NMS_SIZE = 64


class NMS(object):
    """Operator description.
        Non_maximum_suppression operator for object detection.

    Parameters
    ----------
    task_num : Int
        MLU task number.

    name : Str, optional
        The name of operator.
    """

    def __init__(
        self, dtype=bp.float16, target="mlu270", task_type=TaskType.UNION1, name="nms"
    ):
        self.dtype = dtype
        self.task_type = task_type
        self.name = name
        self.tcp = tcp.TCP(target)
        self.num_boxes = self.tcp.SizeVar("num_boxes")
        self.max_output_size = self.tcp.SizeVar("max_output_size")
        self.iou_threshold = self.tcp.Var("iou_threshold", dtype=bp.float32)
        self.score_threshold = self.tcp.Var("score_threshold", dtype=bp.float32)
        # To avoid copying multiple times, copy data 256 by 256
        self.gdram_save_count = 256
        self.buffer_segment = 11

    def nms_compute_body(self, task_num):
        """The main compute body of the nms operator."""
        # SRAM buffer declaration
        self.tcp.memcpy(self.input_score, self.gdram_score)
        self.sram_buffer = self.tcp.Buffer(
            shape=[(self.tcp.get_ram_size("sram") - 52 * 1024) // self.dtype.bytes],
            dtype=self.dtype,
            name="sram_buffer",
            scope="sram",
        )
        self.sram_pos = self.tcp.Buffer(
            shape=[64], dtype=bp.int32, name="sram_pos", scope="sram"
        )
        self.max_score = self.tcp.Buffer(
            shape=[64], dtype=self.dtype, name="max_score", scope="nram"
        )

        # Calculate the maximum buffer size according to the buffer involved in nms's computation
        nram_size_limit = (
            self.tcp.get_ram_size("nram")
            - 64 * 2 * self.dtype.bytes
            - 52 * 1024
            - self.max_output_size * 5 * self.dtype.bytes
        ) / (self.buffer_segment * self.dtype.bytes)

        self.max_seg_size = self.tcp.Scalar(
            dtype=bp.int32,
            name="max_seg_size",
            value=round_down(nram_size_limit, NMS_SIZE),
        )

        # NRAM buffer declaration
        buffer = self.tcp.Buffer(
            shape=[(self.tcp.get_ram_size("nram") - 52 * 1024) // self.dtype.bytes],
            dtype=self.dtype,
            name="buffer",
            scope="nram",
        )
        self.score = buffer[: self.max_seg_size]
        self.x1 = buffer[self.max_seg_size : 2 * self.max_seg_size]
        self.y1 = buffer[2 * self.max_seg_size : 3 * self.max_seg_size]
        self.x2 = buffer[3 * self.max_seg_size : 4 * self.max_seg_size]
        self.y2 = buffer[4 * self.max_seg_size : 5 * self.max_seg_size]
        self.inter_x1 = buffer[5 * self.max_seg_size : 6 * self.max_seg_size]
        self.inter_y1 = buffer[6 * self.max_seg_size : 7 * self.max_seg_size]
        self.inter_x2 = buffer[7 * self.max_seg_size : 8 * self.max_seg_size]
        self.inter_y2 = buffer[8 * self.max_seg_size : 9 * self.max_seg_size]
        self.max_box = buffer[9 * self.max_seg_size : 10 * self.max_seg_size]
        self.nram_save = buffer[10 * self.max_seg_size : 11 * self.max_seg_size]
        self.max_pos = self.tcp.Buffer(
            shape=[64], dtype=bp.int32, name="max_pos", scope="nram"
        )

        # Scalar declaration
        self.output_box_num = self.tcp.Scalar(
            dtype=bp.int32, name="output_box_num", value=0
        )
        self.nram_save_count = self.tcp.Scalar(
            dtype=bp.int32, name="nram_save_count", value=0
        )
        self.save_time = self.tcp.Scalar(dtype=bp.int32, name="save_time", value=0)
        self.zero_scalar = self.tcp.Scalar(
            dtype=self.dtype, name="zero_scalar", value=0
        )

        core_num = self.tcp.Scalar(
            dtype=bp.int32, name="core_num", value=self.num_boxes / task_num
        )
        len_remain = self.tcp.Scalar(
            dtype=bp.int32, name="len_remain", value=self.num_boxes % task_num
        )
        input_offset = self.tcp.Scalar(
            dtype=bp.int32, name="input_offset", value=self.tcp.taskId * core_num
        )

        with self.tcp.if_scope(len_remain > 0):
            core_num += 1
            with self.tcp.if_scope(self.tcp.taskId < len_remain):
                input_offset += self.tcp.taskId
            with self.tcp.else_scope():
                input_offset += len_remain - 1

        repeat = self.tcp.Scalar(
            dtype=bp.int32, name="repeat", value=core_num / self.max_seg_size
        )
        remain = self.tcp.Scalar(
            dtype=bp.int32, name="remain", value=core_num % self.max_seg_size
        )
        remain_pad = self.tcp.Scalar(
            dtype=bp.int32, name="remain_pad", value=round_up(remain, NMS_SIZE)
        )

        with self.tcp.for_range(0, self.max_output_size) as _:
            self.max_box[0] = self.zero_scalar
            # Look for the max score and its corresponding index.
            max_index = self.score_sort(input_offset, repeat, remain, remain_pad)

            # the max box's x1, y1, x2, y2 on every core
            self.tcp.memcpy(self.max_box[1], self.input_box[0, max_index])
            self.tcp.memcpy(self.max_box[2], self.input_box[1, max_index])
            self.tcp.memcpy(self.max_box[3], self.input_box[2, max_index])
            self.tcp.memcpy(self.max_box[4], self.input_box[3, max_index])

            if task_num == 1:
                max_area = (self.max_box[3] - self.max_box[1]) * (
                    self.max_box[4] - self.max_box[2]
                )
                self.input_score[max_index] = 0
                global_max_index = max_index
                self.max_score[0] = self.max_box[0]
            else:
                # The argmax of each core
                self.max_pos[0] = max_index
                # copy every core's box info to sram, form: score--x1---y1---x2---y2---
                with self.tcp.for_range(0, 5) as i:
                    idx = i * task_num + self.tcp.taskId
                    self.tcp.memcpy(self.sram_buffer[idx], self.max_box[i])
                # copy every core's max_index to sram, use 2 half to memcpy max_index
                self.tcp.memcpy(self.sram_pos[self.tcp.taskId], self.max_pos[0])

                # copy score from sram to nram and find the max
                self.tcp.assign(self.inter_x1[:64], 0)
                self.tcp.memcpy(self.inter_x1[:task_num], self.sram_buffer[:task_num])
                self.tcp.amax(self.max_score, self.inter_x1[:64])

                max_core = self.tcp.uint_reinterpret(self.max_score[1])

                # copy the max box to max_box
                self.max_box[0] = self.max_score[0]
                self.tcp.memcpy(
                    self.max_box[1], self.sram_buffer[1 * task_num + max_core]
                )
                self.tcp.memcpy(
                    self.max_box[2], self.sram_buffer[2 * task_num + max_core]
                )
                self.tcp.memcpy(
                    self.max_box[3], self.sram_buffer[3 * task_num + max_core]
                )
                self.tcp.memcpy(
                    self.max_box[4], self.sram_buffer[4 * task_num + max_core]
                )
                max_area = (self.max_box[3] - self.max_box[1]) * (
                    self.max_box[4] - self.max_box[2]
                )

                self.tcp.memcpy(self.max_pos[:task_num], self.sram_pos[:task_num])
                global_max_index = self.max_pos[max_core]

                self.input_score[global_max_index] = self.zero_scalar

            # by now, we get: max_score|max_index|max_box|max_area
            with self.tcp.if_scope(self.output_box_num != 0):
                with self.tcp.if_scope(
                    tcp.any(
                        self.nram_save_count == self.gdram_save_count,
                        self.max_score[0] <= self.score_threshold,
                    )
                ):
                    with self.tcp.if_scope(self.tcp.taskId == task_num - 1):
                        # score, x1, y1, x2, y2
                        self.tcp.memcpy(
                            self.output[
                                self.save_time
                                * self.nram_save_count : self.save_time
                                * self.nram_save_count
                                + self.gdram_save_count
                            ],
                            self.nram_save[: self.gdram_save_count * 5],
                        )
                        self.save_time += 1
                    self.nram_save_count.assign(0)

            with self.tcp.if_scope(self.max_score[0] >= self.score_threshold):
                # score, x1, y1, x2, y2
                with self.tcp.if_scope(self.tcp.taskId == task_num - 1):
                    idx = self.nram_save_count * 5
                    self.tcp.memcpy(self.nram_save[idx : idx + 5], self.max_box[:5])

                self.nram_save_count += 1
                self.output_box_num += 1

                with self.tcp.if_scope(self.output_box_num == self.max_output_size):
                    with self.tcp.if_scope(self.tcp.taskId == task_num - 1):
                        self.tcp.memcpy(
                            self.output[
                                self.save_time
                                * self.gdram_save_count : self.save_time
                                * self.gdram_save_count
                                + self.nram_save_count * 5
                            ],
                            self.nram_save[: self.nram_save_count * 5],
                        )

                self.score_rewrite(max_area, input_offset, repeat, remain, remain_pad)

        # Pad the output whose number is equal to the self.max_output_size
        with self.tcp.if_scope(self.output_box_num < self.max_output_size):
            with self.tcp.if_scope(self.tcp.taskId == task_num - 1):
                self.tcp.assign(self.x1, 0)
                self.tcp.memcpy(
                    self.output[self.output_box_num : self.max_output_size], self.x1[:]
                )

    def nms_compute(self):
        """The entry of the nms operator."""
        self.gdram_score = self.tcp.Buffer(
            shape=[self.num_boxes], dtype=self.dtype, name="gdram_score", scope="global"
        )
        self.input_score = self.tcp.Buffer(
            shape=[self.num_boxes], dtype=self.dtype, name="input_score", scope="global"
        )
        self.input_box = self.tcp.Buffer(
            shape=[4, self.num_boxes],
            dtype=self.dtype,
            name="input_box",
            scope="global",
        )
        self.output = self.tcp.Buffer(
            shape=[self.max_output_size, 5],
            dtype=self.dtype,
            name="output",
            scope="global",
        )

        # pylint: disable=unexpected-keyword-arg
        # disable cause: the for_range parameters of TCP and IRBuilder differ.
        self.tcp.launch_cluster(self.task_type.value)
        task_num = self.task_type.value * 4 if self.task_type.value != 0 else 1
        self.tcp.launch_task(task_num, 1, 1)
        self.nms_compute_body(task_num)

        return self.tcp.BuildBANG(
            inputs=[
                self.gdram_score,
                self.input_score,
                self.input_box,
                self.num_boxes,
                self.max_output_size,
                self.iou_threshold,
                self.score_threshold,
            ],
            outputs=[self.output],
            kernel_name=self.name,
        )

    def score_sort(self, input_offset, nms_loop, remain, remain_pad):
        """Sort the boxes' score."""
        with self.tcp.if_scope(nms_loop > 0):
            with self.tcp.for_range(0, nms_loop) as i:
                offset = i * self.max_seg_size
                max_index = self.score_sort_each_loop(
                    input_offset, offset, self.max_seg_size, self.max_seg_size
                )

        offset = nms_loop * self.max_seg_size
        with self.tcp.if_scope(remain > 0):
            max_index = self.score_sort_each_loop(
                input_offset, offset, remain, remain_pad
            )
        return max_index

    def score_sort_each_loop(self, input_offset, offset, loop_extent, alignmemt):
        """Sort the boxes' score in each loop."""
        self.tcp.assign(self.score[:alignmemt], 0)
        idx = input_offset + offset
        self.tcp.memcpy(
            self.score[:loop_extent], self.input_score[idx : idx + loop_extent]
        )
        self.tcp.amax(self.inter_x1, self.score[:alignmemt])

        with self.tcp.if_scope(self.inter_x1[0] > self.max_box[0]):
            self.max_box[0] = self.inter_x1[0]
            # offset start from head of input data
            max_index = idx + self.tcp.uint_reinterpret(self.inter_x1[1])
        return max_index

    def score_rewrite(self, max_area, input_offset, nms_loop, remain, remain_pad):
        """Rewrite the score of boxes."""
        with self.tcp.if_scope(nms_loop > 0):
            with self.tcp.for_range(0, nms_loop) as i:
                offset = i * self.max_seg_size
                self.score_rewrite_each_loop(
                    max_area, input_offset, offset, self.max_seg_size, self.max_seg_size
                )

        offset = nms_loop * self.max_seg_size
        with self.tcp.if_scope(remain > 0):
            self.score_rewrite_each_loop(
                max_area, input_offset, offset, remain, remain_pad
            )

    def score_rewrite_each_loop(
        self, max_area, input_offset, offset, loop_extent, alignmemt
    ):
        """Rewrite the score of each loop."""
        self.tcp.assign(self.score[:alignmemt], 0)
        idx = input_offset + offset
        self.tcp.memcpy(
            self.score[:loop_extent], self.input_score[idx : idx + loop_extent]
        )
        self.tcp.memcpy(
            self.x1[:loop_extent], self.input_box[0, idx : idx + loop_extent]
        )
        self.tcp.memcpy(
            self.y1[:loop_extent], self.input_box[1, idx : idx + loop_extent]
        )
        self.tcp.memcpy(
            self.x2[:loop_extent], self.input_box[2, idx : idx + loop_extent]
        )
        self.tcp.memcpy(
            self.y2[:loop_extent], self.input_box[3, idx : idx + loop_extent]
        )

        # 1、 compute IOU
        # get the area_I
        self.tcp.assign(self.inter_y1[:alignmemt], self.max_box[1])
        self.tcp.maximum(
            self.inter_x1[:alignmemt], self.x1[:alignmemt], self.inter_y1[:alignmemt]
        )
        self.tcp.assign(self.inter_y2[:alignmemt], self.max_box[3])
        self.tcp.minimum(
            self.inter_x2[:alignmemt], self.x2[:alignmemt], self.inter_y2[:alignmemt]
        )
        self.tcp.subtract(
            self.inter_x1[:alignmemt],
            self.inter_x2[:alignmemt],
            self.inter_x1[:alignmemt],
        )
        self.tcp.relu(self.inter_x1[:alignmemt], self.inter_x1[:alignmemt])
        self.tcp.assign(self.inter_x2[:alignmemt], self.max_box[2])
        self.tcp.maximum(
            self.inter_y1[:alignmemt], self.y1[:alignmemt], self.inter_x2[:alignmemt]
        )
        self.tcp.assign(self.inter_x2[:alignmemt], self.max_box[4])
        self.tcp.minimum(
            self.inter_y2[:alignmemt], self.y2[:alignmemt], self.inter_x2[:alignmemt]
        )
        self.tcp.subtract(
            self.inter_y1[:alignmemt],
            self.inter_y2[:alignmemt],
            self.inter_y1[:alignmemt],
        )
        self.tcp.relu(self.inter_y1[:alignmemt], self.inter_y1[:alignmemt])

        self.tcp.multiply(
            self.inter_x1[:alignmemt],
            self.inter_x1[:alignmemt],
            self.inter_y1[:alignmemt],
        )
        # get the area of input_box: area = (self.x2 - self.x1) * (y2 - self.y1)
        self.tcp.subtract(
            self.inter_y1[:alignmemt], self.x2[:alignmemt], self.x1[:alignmemt]
        )
        self.tcp.subtract(
            self.inter_y2[:alignmemt], self.y2[:alignmemt], self.y1[:alignmemt]
        )
        self.tcp.multiply(
            self.inter_x2[:alignmemt],
            self.inter_y1[:alignmemt],
            self.inter_y2[:alignmemt],
        )

        # get the area_U: area + max_area - area_I
        self.tcp.assign(self.inter_y1[:alignmemt], max_area)
        self.tcp.add(
            self.inter_x2[:alignmemt],
            self.inter_x2[:alignmemt],
            self.inter_y1[:alignmemt],
        )
        self.tcp.subtract(
            self.inter_x2[:alignmemt],
            self.inter_x2[:alignmemt],
            self.inter_x1[:alignmemt],
        )

        # 2、 select the box
        # if IOU greater than thres, set the score to zero, abort it: area_U * thresh > area_I?
        self.tcp.multiply(
            self.inter_x2[:alignmemt], self.inter_x2[:alignmemt], self.iou_threshold
        )
        self.tcp.greater(
            self.inter_x1[:alignmemt],
            self.inter_x2[:alignmemt],
            self.inter_x1[:alignmemt],
        )
        self.tcp.multiply(
            self.score[:alignmemt], self.score[:alignmemt], self.inter_x1[:alignmemt]
        )

        # update the score
        idx = input_offset + offset
        self.tcp.memcpy(
            self.input_score[idx : idx + loop_extent], self.score[:loop_extent]
        )


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_nms(dtype=None, target=None):
    task_type = TaskType.UNION1
    op_mod = NMS(dtype=dtype, target=target, task_type=task_type).nms_compute()
    return op_mod
