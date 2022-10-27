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
# pylint: disable=invalid-name, missing-function-docstring, useless-object-inheritance
# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals
# pylint: disable=too-many-statements, attribute-defined-outside-init
"""Non-maximum suppression operator implementation using BANGPy TCP API."""
import bangpy as bp
from bangpy import tcp
from bangpy.script import build_module, ty


TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]
KERNEL_NAME = "nms"
DTYPES = [bp.float32, bp.float16]


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

    def __init__(self, dtype: ty.string) -> None:
        self.dtype = dtype
        # To avoid copying multiple times, copy data 256 by 256
        self.gdram_save_count = 256
        self.buffer_segment = 11
        self.NMS_SIZE = 64

    def score_sort_each_loop(
        self,
        offset: ty.int32,
        loop_extent: ty.int32,
        alignment: ty.int32,
        score_nram_buffer: ty.Buffer("nram"),  # type: ignore
        comp_ret_buffer: ty.Buffer("nram"),  # type: ignore
        max_box: ty.Buffer("nram"),  # type: ignore
        int_ret_buffer: ty.Buffer("nram"),  # type: ignore
    ):
        tcp.assign(score_nram_buffer[:alignment], 0)
        tcp.memcpy(
            score_nram_buffer[:loop_extent],
            self.input_score[offset : offset + loop_extent],
        )
        tcp.amax(comp_ret_buffer, score_nram_buffer[:alignment])
        if comp_ret_buffer[0] > max_box[0]:
            max_box[0] = comp_ret_buffer[0]
            # offset start from head of input data
            int_ret_buffer[0] = offset + tcp.uint_reinterpret(comp_ret_buffer[1])

    def score_sort(
        self,
        input_offset: ty.int32,
        repeat_loop: ty.int32,
        remain: ty.int32,
        remain_pad: ty.int32,
        score_nram_buffer: ty.Buffer("nram"),  # type: ignore
        comp_ret_buffer: ty.Buffer("nram"),  # type: ignore
        max_box: ty.Buffer("nram"),  # type: ignore
        max_seg_size: ty.int32,
        int_ret_buffer: ty.Buffer("nram"),  # type: ignore
    ):
        if repeat_loop > 0:
            for i in range(repeat_loop):
                offset = i * max_seg_size + input_offset
                self.score_sort_each_loop(
                    offset,
                    max_seg_size,
                    max_seg_size,
                    score_nram_buffer,
                    comp_ret_buffer,
                    max_box,
                    int_ret_buffer,
                )

        offset = repeat_loop * max_seg_size + input_offset
        if remain > 0:
            self.score_sort_each_loop(
                offset,
                remain,
                remain_pad,
                score_nram_buffer,
                comp_ret_buffer,
                max_box,
                int_ret_buffer,
            )

    def score_rewrite_each_loop(
        self,
        max_area: ty.Buffer("nram"),  # type: ignore
        offset: ty.int32,
        loop_extent: ty.int32,
        alignment: ty.int32,
        iou_threshold: ty.float32,
        score: ty.Buffer("nram"),  # type: ignore
        max_box: ty.Buffer("nram"),  # type: ignore
    ):
        tcp.assign(score[:alignment], 0)
        tcp.memcpy(score[:loop_extent], self.input_score[offset : offset + loop_extent])
        tcp.memcpy(
            self.x1[:loop_extent], self.input_box[0, offset : offset + loop_extent]
        )
        tcp.memcpy(
            self.y1[:loop_extent], self.input_box[1, offset : offset + loop_extent]
        )
        tcp.memcpy(
            self.x2[:loop_extent], self.input_box[2, offset : offset + loop_extent]
        )
        tcp.memcpy(
            self.y2[:loop_extent], self.input_box[3, offset : offset + loop_extent]
        )

        tcp.assign(self.inter_y1[:alignment], max_box[1])
        tcp.maximum(
            self.inter_x1[:alignment], self.x1[:alignment], self.inter_y1[:alignment]
        )
        tcp.assign(self.inter_y2[:alignment], max_box[3])
        tcp.minimum(
            self.inter_x2[:alignment], self.x2[:alignment], self.inter_y2[:alignment]
        )
        tcp.subtract(
            self.inter_x1[:alignment],
            self.inter_x2[:alignment],
            self.inter_x1[:alignment],
        )
        tcp.relu(self.inter_x1[:alignment], self.inter_x1[:alignment])
        tcp.assign(self.inter_x2[:alignment], max_box[2])
        tcp.maximum(
            self.inter_y1[:alignment], self.y1[:alignment], self.inter_x2[:alignment]
        )
        tcp.assign(self.inter_x2[:alignment], max_box[4])
        tcp.minimum(
            self.inter_y2[:alignment], self.y2[:alignment], self.inter_x2[:alignment]
        )
        tcp.subtract(
            self.inter_y1[:alignment],
            self.inter_y2[:alignment],
            self.inter_y1[:alignment],
        )
        tcp.relu(self.inter_y1[:alignment], self.inter_y1[:alignment])

        tcp.multiply(
            self.inter_x1[:alignment],
            self.inter_x1[:alignment],
            self.inter_y1[:alignment],
        )
        # get the area of input_box: area = (x2 - x1) * (y2 - y1)
        tcp.subtract(
            self.inter_y1[:alignment], self.x2[:alignment], self.x1[:alignment]
        )
        tcp.subtract(
            self.inter_y2[:alignment], self.y2[:alignment], self.y1[:alignment]
        )
        tcp.multiply(
            self.inter_x2[:alignment],
            self.inter_y1[:alignment],
            self.inter_y2[:alignment],
        )

        # get the area_U: area + max_area - area_I
        tcp.assign(self.inter_y1[:alignment], max_area)
        tcp.add(
            self.inter_x2[:alignment],
            self.inter_x2[:alignment],
            self.inter_y1[:alignment],
        )
        tcp.subtract(
            self.inter_x2[:alignment],
            self.inter_x2[:alignment],
            self.inter_x1[:alignment],
        )

        # 2. select the box
        # if IOU greater than thres, set the score to zero, abort it: area_U * thresh > area_I?
        tcp.multiply(
            self.inter_x2[:alignment], self.inter_x2[:alignment], iou_threshold
        )
        tcp.greater(
            self.inter_x1[:alignment],
            self.inter_x2[:alignment],
            self.inter_x1[:alignment],
        )
        tcp.multiply(score[:alignment], score[:alignment], self.inter_x1[:alignment])

        # update the score
        tcp.memcpy(self.input_score[offset : offset + loop_extent], score[:loop_extent])

    def score_rewrite(
        self,
        max_area: ty.Buffer("nram"),  # type: ignore
        input_offset: ty.int32,
        nms_loop: ty.int32,
        remain: ty.int32,
        remain_pad: ty.int32,
        iou_threshold: ty.float32,
        score: ty.Buffer("nram"),  # type: ignore
        max_box: ty.Buffer("nram"),  # type: ignore
        max_seg_size: ty.int32,
    ):
        if nms_loop > 0:
            for i in range(nms_loop):
                offset = i * max_seg_size + input_offset
                self.score_rewrite_each_loop(
                    max_area,
                    offset,
                    max_seg_size,
                    max_seg_size,
                    iou_threshold,
                    score,
                    max_box,
                )

        offset = nms_loop * max_seg_size + input_offset
        if remain > 0:
            self.score_rewrite_each_loop(
                max_area,
                offset,
                remain,
                remain_pad,
                iou_threshold,
                score,
                max_box,
            )

    def nms_compute_body(
        self,
        total_box_num: ty.int32,
        max_output_size: ty.int32,
        iou_threshold: ty.float32,
        score_threshold: ty.float32,
        nram_size_limit: ty.int32,
        task_id: ty.int32,
        task_num: ty.int32,
    ):
        # SRAM buffer declaration
        sram_pos = tcp.alloc_buffer(shape=[64], dtype="int32", scope="sram")

        max_seg_size = tcp.round_down(nram_size_limit, self.NMS_SIZE)

        score = self.nram_buffer[:max_seg_size]
        self.x1 = self.nram_buffer[max_seg_size : 2 * max_seg_size]
        self.y1 = self.nram_buffer[2 * max_seg_size : 3 * max_seg_size]
        self.x2 = self.nram_buffer[3 * max_seg_size : 4 * max_seg_size]
        self.y2 = self.nram_buffer[4 * max_seg_size : 5 * max_seg_size]
        self.inter_x1 = self.nram_buffer[5 * max_seg_size : 6 * max_seg_size]
        self.inter_y1 = self.nram_buffer[6 * max_seg_size : 7 * max_seg_size]
        self.inter_x2 = self.nram_buffer[7 * max_seg_size : 8 * max_seg_size]
        self.inter_y2 = self.nram_buffer[8 * max_seg_size : 9 * max_seg_size]
        max_box = self.nram_buffer[9 * max_seg_size : 10 * max_seg_size]
        nram_save = self.nram_buffer[10 * max_seg_size : 11 * max_seg_size]

        max_pos = tcp.alloc_buffer(shape=[64], dtype="int32", scope="nram")
        int_ret_buffer = tcp.alloc_buffer(shape=[1], dtype="int32", scope="nram")
        com_res_buffer = tcp.alloc_buffer(shape=[2], dtype=self.dtype, scope="nram")

        # Scalar declaration
        output_box_num = 0
        nram_save_count = 0

        save_time = 0
        batch_size = total_box_num / task_num
        len_remain = total_box_num % task_num
        input_offset = task_id * batch_size

        if len_remain > 0:
            batch_size += 1
            input_offset = (
                input_offset + task_id
                if task_id < len_remain
                else input_offset + len_remain - 1
            )
        repeat = batch_size / max_seg_size
        remain = batch_size % max_seg_size
        remain_pad = tcp.round_up(remain, self.NMS_SIZE)

        # Worst case that all boxes are irrelevant but hold high score.
        for _ in range(max_output_size):
            max_box[0] = tcp.cast(0, self.dtype)
            # Look for the max score and its corresponding index.
            self.score_sort(
                input_offset,
                repeat,
                remain,
                remain_pad,
                score,
                self.inter_x1,
                max_box,
                max_seg_size,
                int_ret_buffer,
            )
            max_index = int_ret_buffer[0]
            # the max box's x1, y1, x2, y2 on every core
            tcp.memcpy(max_box[1], self.input_box[0, max_index])
            tcp.memcpy(max_box[2], self.input_box[1, max_index])
            tcp.memcpy(max_box[3], self.input_box[2, max_index])
            tcp.memcpy(max_box[4], self.input_box[3, max_index])

            # The argmax of each core
            max_pos[0] = max_index
            # copy every core's box info to sram, form: score--x1---y1---x2---y2---
            for i in range(5):
                idx = i * task_num + task_id
                tcp.memcpy(self.sram_buffer[idx], max_box[i])
            # copy every core's max_index to sram, use 2 half to memcpy max_index
            tcp.memcpy(sram_pos[task_id], max_pos[0])

            # copy score from sram to nram and find the max
            tcp.assign(self.inter_x1[:64], 0)
            tcp.memcpy(self.inter_x1[:task_num], self.sram_buffer[:task_num])
            tcp.amax(com_res_buffer, self.inter_x1[:64])
            task_id_of_max_score = tcp.uint_reinterpret(com_res_buffer[1])

            # copy the max_box info from sram to nram
            max_box[0] = com_res_buffer[0]
            tcp.memcpy(
                max_box[1], self.sram_buffer[1 * task_num + task_id_of_max_score]
            )
            tcp.memcpy(
                max_box[2], self.sram_buffer[2 * task_num + task_id_of_max_score]
            )
            tcp.memcpy(
                max_box[3], self.sram_buffer[3 * task_num + task_id_of_max_score]
            )
            tcp.memcpy(
                max_box[4], self.sram_buffer[4 * task_num + task_id_of_max_score]
            )
            max_area = (max_box[3] - max_box[1]) * (max_box[4] - max_box[2])

            tcp.memcpy(max_pos[:task_num], sram_pos[:task_num])
            global_max_index = max_pos[task_id_of_max_score]

            self.input_score[global_max_index] = tcp.cast(0, self.dtype)

            # by now, we get: max_score|max_index|max_box|max_area
            if output_box_num != 0:
                if nram_save_count == self.gdram_save_count or max_box[0] <= tcp.cast(
                    score_threshold, self.dtype
                ):
                    if task_id == task_num - 1:
                        # score, x1, y1, x2, y2
                        tcp.memcpy(
                            self.output[
                                save_time
                                * nram_save_count : save_time
                                * nram_save_count
                                + self.gdram_save_count
                            ].reshape((self.gdram_save_count * 5,)),
                            nram_save[: self.gdram_save_count * 5],
                        )
                        save_time += 1
                    # TODO((BANGPy-Team)): break not support.
                    # if max_box[0] <= score_threshold:
                    #     break;
                    nram_save_count = 0

            if max_box[0] >= tcp.cast(score_threshold, self.dtype):
                # score, x1, y1, x2, y2
                if task_id == task_num - 1:
                    idx = nram_save_count * 5
                    tcp.memcpy(nram_save[idx : idx + 5], max_box[:5])

                nram_save_count += 1
                output_box_num += 1

                if output_box_num == max_output_size:
                    if task_id == task_num - 1:
                        tcp.memcpy(
                            self.output[
                                save_time
                                * self.gdram_save_count : save_time
                                * self.gdram_save_count
                                + nram_save_count
                            ].reshape((nram_save_count * 5,)),
                            nram_save[: nram_save_count * 5],
                        )

                self.score_rewrite(
                    max_area,
                    input_offset,
                    repeat,
                    remain,
                    remain_pad,
                    iou_threshold,
                    score,
                    max_box,
                    max_seg_size,
                )

            # Read after write, sync cluster here.
            tcp.sync_cluster()

        # Pad the output whose number is equal to the self.max_output_size
        if output_box_num < max_output_size:
            if task_id == task_num - 1:
                tcp.assign(self.x1, 0)
                tcp.memcpy(
                    self.output[output_box_num:max_output_size].reshape(
                        ((max_output_size - output_box_num) * 5),
                    ),
                    self.x1[: 5 * (max_output_size - output_box_num)],
                )

    def main(
        self,
        input_score: ty.handle,
        gdram_score: ty.handle,
        input_box: ty.handle,
        output: ty.handle,
        total_box_num: ty.int32,
        max_output_size: ty.int32,
        iou_threshold: ty.float32,
        score_threshold: ty.float32,
    ) -> None:
        self.gdram_score = tcp.match_buffer(
            gdram_score, [total_box_num], dtype=self.dtype
        )
        self.input_score = tcp.match_buffer(
            input_score, [total_box_num], dtype=self.dtype
        )
        self.input_box = tcp.match_buffer(
            input_box, [4, total_box_num], dtype=self.dtype
        )
        self.output = tcp.match_buffer(output, [max_output_size, 5], dtype=self.dtype)

        # pylint: disable=unexpected-keyword-arg
        # disable cause: the for_range parameters of TCP and IRBuilder differ.
        tgt = tcp.target()
        for _ in tcp.thread_binding(0, 1, thread="blockIdx.x"):
            for i in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):
                self.sram_buffer = tcp.alloc_buffer(
                    shape=[(tgt.sram_size - 52 * 1024) // 4],
                    dtype=self.dtype,
                    scope="sram",
                )
                nram_size_limit = (
                    tgt.nram_size - 64 * 2 * 4 - 52 * 1024 - max_output_size * 5 * 4
                ) / (self.buffer_segment * 4)
                self.nram_buffer = tcp.alloc_buffer(
                    shape=[(tgt.nram_size - 52 * 1024) // 4],
                    dtype=self.dtype,
                    scope="nram",
                )
                tcp.memcpy(self.input_score, self.gdram_score)
                self.nms_compute_body(
                    total_box_num,
                    max_output_size,
                    iou_threshold,
                    score_threshold,
                    nram_size_limit,
                    i,
                    tgt.core_num,
                )


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_nms(dtype=None, target=None):
    op_mod = build_module.build(NMS(dtype.name), target, KERNEL_NAME)
    return op_mod
