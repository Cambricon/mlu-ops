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
# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring
# pylint: disable=attribute-defined-outside-init
"""AdjustHue operator implementation using BANGPy TCP API."""
from active_table import ACTIVE_TABLE1, ACTIVE_TABLE2, ACTIVE_TABLE3
from active_table import ACTIVE_TABLE4, ACTIVE_TABLE5, ACTIVE_TABLE6, ACTIVE_TABLE7
from active_table import CONST_TABLE1, CONST_TABLE2
import bangpy as bp
from bangpy import tcp
from bangpy.tcp.runtime import TaskType
from bangpy.platform.bang_config import TARGET

NRAM_SIZE_LIMIT = lambda x, y, z, align: x / z / y / align * align
ACTIVETAB_ELEM_NUM = 128
CONSTAB_ELEM_NUM = 64
DTYPES = [bp.float16, bp.float32]
TARGET_LIST = ["mlu290", "mlu270"]
KERNEL_NAME = "adjust_hue"


class AdjustHue(object):
    def __init__(self, dtype, stage, target, task_type):
        self.dtype = dtype
        self.stage = stage
        self.target = target
        self.task_type = task_type
        self.tcp = tcp.TCP(target)
        self.tcp.launch_cluster(self.task_type.value)
        self.tcp.launch_task(4 * self.task_type.value, 1, 1)
        self.n = self.tcp.SizeVar("n")
        self.h = self.tcp.SizeVar("h")
        self.w = self.tcp.SizeVar("w")
        self.c = self.tcp.SizeVar("c")
        self.nram_use = (TARGET(target).nram_size - 52 * 1024) // 4
        self.nram_size = self.tcp.Scalar(
            dtype=bp.int32, name="nram_size", value=self.nram_use,
        )
        self.delta = self.tcp.Var("delta", dtype=bp.float32)
        self.line_align = self.tcp.Scalar(
            dtype=bp.int32, name="line_align", value=64 // dtype.bytes
        )
        self.reshape_align = self.tcp.Scalar(
            dtype=bp.int32,
            name="reshape_align",
            value=4096 // dtype.bytes // dtype.bytes,
        )
        nram_limit_size = (
            NRAM_SIZE_LIMIT(
                self.nram_size, dtype.bytes, self.c, 4096 // dtype.bytes // dtype.bytes
            )
            - 4096 // dtype.bytes // dtype.bytes
        )
        self.nram_limit = self.tcp.Scalar(
            dtype=bp.int32, name="nram_limit", value=nram_limit_size
        )
        self.new_delta = self.tcp.Scalar(
            dtype=dtype, name="new_delta", value=self.delta.astype(dtype) * 6.0
        )
        self.core_offset = self.tcp.Scalar(dtype=bp.int32, name="core_offset")
        self.real_task_num = self.tcp.Scalar(dtype=bp.int32, name="real_task_num")
        self.r_h = self.tcp.Scalar(dtype=bp.int32, name="r_h")
        self.r_w = self.tcp.Scalar(dtype=bp.int32, name="r_w")
        self.r_w_tmp = self.tcp.Scalar(dtype=bp.int32, name="r_w_tmp")
        self.r_hw = self.tcp.Scalar(dtype=bp.int32, name="r_hw")
        self.row_each_per = self.tcp.Scalar(
            dtype=bp.int32,
            value=(self.nram_limit / self.w).astype(bp.int32),
            name="row_each_per",
        )
        self.src_gdram = self.tcp.Buffer(
            shape=(self.n, self.h, self.w, self.c),
            dtype=self.dtype,
            scope="global",
            name="src_gdram",
        )
        self.dst_gdram = self.tcp.Buffer(
            shape=(self.n, self.h, self.w, self.c),
            dtype=self.dtype,
            scope="global",
            name="dst_gdram",
        )
        if dtype == bp.float16:
            self.const_tab_1 = self.tcp.Constant(
                value=CONST_TABLE1, shape=(32,), dtype=bp.int16, name="const_1"
            )
        else:

            self.const_tab_1 = self.tcp.Constant(
                value=CONST_TABLE2, shape=(32,), dtype=bp.int16, name="const_1"
            )
        self.const1_nram = self.tcp.Buffer(
            shape=(1, 64 // self.dtype.bytes),
            dtype=self.dtype,
            scope="nram",
            name="const1_nram",
        )

        self.active_tab_1 = self.tcp.Constant(
            value=ACTIVE_TABLE1, shape=(ACTIVETAB_ELEM_NUM,), dtype=dtype, name="active_1"
        )
        self.tab1_nram = self.tcp.Buffer(
            shape=(1, ACTIVETAB_ELEM_NUM), dtype=self.dtype, scope="nram", name="tab1_nram"
        )
        # use for r_c
        self.active_tab_2 = self.tcp.Constant(
            value=ACTIVE_TABLE2, shape=(ACTIVETAB_ELEM_NUM,), dtype=dtype, name="active_2"
        )
        self.tab2_nram = self.tcp.Buffer(
            shape=(1, ACTIVETAB_ELEM_NUM), dtype=self.dtype, scope="nram", name="tab2_nram"
        )
        # use for r_x
        self.active_tab_3 = self.tcp.Constant(
            value=ACTIVE_TABLE3, shape=(ACTIVETAB_ELEM_NUM,), dtype=dtype, name="active_3"
        )
        self.tab3_nram = self.tcp.Buffer(
            shape=(1, ACTIVETAB_ELEM_NUM), dtype=self.dtype, scope="nram", name="tab3_nram"
        )
        # use for g_c
        self.active_tab_4 = self.tcp.Constant(
            value=ACTIVE_TABLE4, shape=(ACTIVETAB_ELEM_NUM,), dtype=dtype, name="active_4"
        )
        self.tab4_nram = self.tcp.Buffer(
            shape=(1, ACTIVETAB_ELEM_NUM), dtype=self.dtype, scope="nram", name="tab4_nram"
        )
        # use for g_x
        self.active_tab_5 = self.tcp.Constant(
            value=ACTIVE_TABLE5, shape=(ACTIVETAB_ELEM_NUM,), dtype=dtype, name="active_5"
        )
        self.tab5_nram = self.tcp.Buffer(
            shape=(1, ACTIVETAB_ELEM_NUM), dtype=self.dtype, scope="nram", name="tab5_nram"
        )
        # use for b_c
        self.active_tab_6 = self.tcp.Constant(
            value=ACTIVE_TABLE6, shape=(ACTIVETAB_ELEM_NUM,), dtype=dtype, name="active_6"
        )
        self.tab6_nram = self.tcp.Buffer(
            shape=(1, ACTIVETAB_ELEM_NUM), dtype=self.dtype, scope="nram", name="tab6_nram"
        )
        # use for b_x
        self.active_tab_7 = self.tcp.Constant(
            value=ACTIVE_TABLE7, shape=(ACTIVETAB_ELEM_NUM,), dtype=dtype, name="active_7"
        )
        self.tab7_nram = self.tcp.Buffer(
            shape=(1, ACTIVETAB_ELEM_NUM), dtype=self.dtype, scope="nram", name="tab7_nram"
        )
        self.hsv_nram = self.tcp.Buffer(
            shape=(self.nram_use // self.dtype.bytes,),
            dtype=self.dtype,
            scope="nram",
            name="hsv_nram",
        )
        self.aux_nram = self.tcp.Buffer(
            shape=(self.nram_use // self.dtype.bytes,),
            dtype=self.dtype,
            scope="nram",
            name="aux_nram",
        )

    def rgb2hsv(self, aux_full, h, s, v, r, g, b, aux, aux1, aux2):
        # get max, min, mid
        # use v to save max
        # use s to save min
        self.tcp.maximum(v, r, g)
        self.tcp.minimum(s, r, g)
        self.tcp.maximum(v, v, b)
        self.tcp.minimum(s, s, b)
        self.tcp.equal(aux, v, s)
        self.tcp.subtract(h, v, s)
        self.tcp.add(h, h, aux)
        self.tcp.reciprocal(h, h, mode="hp")
        self.tcp.subtract(aux, g, b)  # g-b
        self.tcp.subtract(aux1, b, r)  # b-r
        self.tcp.subtract(aux2, r, g)  # r-g
        self.tcp.multiply(aux_full, aux_full, h)
        # r==max
        self.tcp.equal(r, r, v)
        # g==max
        self.tcp.equal(g, g, v)
        # b==max
        self.tcp.logical_or(b, r, g)
        self.tcp.subtract(g, b, r)
        self.tcp.logical_not(b, b)
        self.tcp.add(aux1, aux1, 2)
        self.tcp.add(aux2, aux2, 4)
        self.tcp.multiply(r, r, aux)
        self.tcp.multiply(g, g, aux1)
        self.tcp.multiply(b, b, aux2)
        self.tcp.add(h, r, g)
        self.tcp.add(h, h, b)

    def add_delta(self, h, delta):
        self.tcp.add(h, h, delta)

    def hsv2rgb(self, h, s, v, r, g, b, aux, aux1, aux_int):
        self.tcp.subtract(v, v, s)
        self.tcp.type_convert(aux_int, h, 0, "dn")
        self.tcp.type_convert(aux, aux_int, 0)
        self.tcp.type_convert(aux_int, h, 0, "tz")
        self.tcp.type_convert(aux1, aux_int, 0)
        self.tcp.lut_active(aux1, aux1, self.tab1_nram, self.const1_nram)
        self.tcp.subtract(h, h, aux1)
        self.tcp.abs(h, h)
        self.tcp.multiply(h, h, v)
        # aux:h_category
        # h:ratio
        # s:vmin
        # v:vmax-min
        # r
        self.tcp.lut_active(r, aux, self.tab2_nram, self.const1_nram)
        self.tcp.lut_active(aux1, aux, self.tab3_nram, self.const1_nram)
        self.tcp.multiply(r, r, v)
        self.tcp.multiply(aux1, aux1, h)
        self.tcp.add(r, r, aux1)
        # g
        self.tcp.lut_active(g, aux, self.tab4_nram, self.const1_nram)
        self.tcp.lut_active(aux1, aux, self.tab5_nram, self.const1_nram)
        self.tcp.multiply(g, g, v)
        self.tcp.multiply(aux1, aux1, h)
        self.tcp.add(g, g, aux1)
        # b
        self.tcp.lut_active(b, aux, self.tab6_nram, self.const1_nram)
        self.tcp.lut_active(aux1, aux, self.tab7_nram, self.const1_nram)
        self.tcp.multiply(b, b, v)
        self.tcp.multiply(aux1, aux1, h)
        self.tcp.add(b, b, aux1)

    def prepare_active_tab(self):
        # memcpy for active table.
        self.tcp.memcpy(
            self.tab1_nram.reshape(
                [
                    ACTIVETAB_ELEM_NUM,
                ]
            ),
            self.active_tab_1,
        )
        self.tcp.memcpy(
            self.const1_nram.reshape(
                [
                    64 // self.dtype.bytes,
                ]
            ),
            self.const_tab_1.reinterpret_cast(self.dtype),
        )
        self.tcp.memcpy(
            self.tab2_nram.reshape(
                [
                    ACTIVETAB_ELEM_NUM,
                ]
            ),
            self.active_tab_2,
        )
        self.tcp.memcpy(
            self.tab3_nram.reshape(
                [
                    ACTIVETAB_ELEM_NUM,
                ]
            ),
            self.active_tab_3,
        )
        self.tcp.memcpy(
            self.tab4_nram.reshape(
                [
                    ACTIVETAB_ELEM_NUM,
                ]
            ),
            self.active_tab_4,
        )
        self.tcp.memcpy(
            self.tab5_nram.reshape(
                [
                    ACTIVETAB_ELEM_NUM,
                ]
            ),
            self.active_tab_5,
        )
        self.tcp.memcpy(
            self.tab6_nram.reshape(
                [
                    ACTIVETAB_ELEM_NUM,
                ]
            ),
            self.active_tab_6,
        )
        self.tcp.memcpy(
            self.tab7_nram.reshape(
                [
                    ACTIVETAB_ELEM_NUM,
                ]
            ),
            self.active_tab_7,
        )

    def divide_h_dim(self, dim, dim_id, dim_num, dim_offset):
        dim_num.assign((self.h + dim - 1) / dim)
        dim_offset.assign(dim_num * dim_id)
        with self.tcp.if_scope((self.h / dim) * dim + dim_id >= self.h):
            dim_offset.assign((self.h / dim) * dim_id + self.h % dim)
            with self.tcp.if_scope(dim_num * dim != self.h):
                dim_num.assign(dim_num - 1)

    def divide_w(self, real_w_size, loop_num):
        # divide w when w is too large.
        with self.tcp.if_scope(self.row_each_per == 0):
            real_w_size.assign(
                self.w / ((self.w + self.nram_limit - 1) / self.nram_limit)
            )
            self.row_each_per.assign(1)
        with self.tcp.else_scope():
            real_w_size.assign(self.w)
            with self.tcp.if_scope(tcp.all((loop_num <= 4), (self.row_each_per != 1),)):
                with self.tcp.if_scope(real_w_size <= 3900):
                    self.row_each_per.assign(3900 / self.w)

    def loop_body(self, batch_index, offset_h_start, offset_w_start, db):
        rgb = self.rgb_nram[: self.nram_limit * self.c]
        hsv = self.hsv_nram[: self.nram_limit * self.c]
        aux = self.aux_nram[: self.nram_limit * self.c]
        aux_int = self.aux_nram[
            self.nram_limit * 2 : self.nram_limit * 3
        ].reinterpret_cast(bp.int16)
        with self.tcp.block("data_copy" if db else "null"):
            self.tcp.memcpy(
                rgb.reshape((self.nram_limit / self.r_w / self.c, self.r_w, self.c,))[
                    : self.r_h
                ][:][:],
                self.src_gdram[
                    batch_index,
                    offset_h_start : offset_h_start + self.r_h,
                    offset_w_start : offset_w_start + self.r_w,
                    :,
                ],
            )
        with self.tcp.block("compute" if db else "null"):
            self.tcp.transpose(
                hsv[: self.r_hw * self.c].reshape(
                    (1, self.r_hw / self.line_align * self.c, 1, self.line_align,)
                ),
                rgb[: self.r_hw * self.c].reshape(
                    (1, self.line_align, 1, self.r_hw / self.line_align * self.c,)
                ),
                (0, 3, 1, 2),
            )
            self.tcp.transpose(
                rgb[: self.r_hw * self.c].reshape(
                    (1, self.line_align * self.c, 1, self.r_hw / self.line_align,)
                ),
                hsv[: self.r_hw * self.c].reshape(
                    (1, self.r_hw / self.line_align, 1, self.line_align * self.c,)
                ),
                (0, 3, 1, 2),
            )
            rgb_reshape = rgb[: self.r_hw * self.c].reshape((self.c, self.r_hw))
            aux_reshape = aux[: self.r_hw * self.c].reshape((self.c, self.r_hw))
            h = hsv[: self.r_hw * self.c].reshape((self.c, self.r_hw))[0]
            s = hsv[: self.r_hw * self.c].reshape((self.c, self.r_hw))[1]
            v = hsv[: self.r_hw * self.c].reshape((self.c, self.r_hw))[2]
            r = rgb[: self.r_hw * self.c].reshape((self.c, self.r_hw))[0]
            g = rgb[: self.r_hw * self.c].reshape((self.c, self.r_hw))[1]
            b = rgb[: self.r_hw * self.c].reshape((self.c, self.r_hw))[2]
            aux_nram = aux[: self.r_hw * self.c].reshape((self.c, self.r_hw))[0]
            aux1_nram = aux[: self.r_hw * self.c].reshape((self.c, self.r_hw))[1]
            aux2_nram = aux[: self.r_hw * self.c].reshape((self.c, self.r_hw))[2]
            aux_int_nram = aux_int[: self.r_hw].reshape((self.r_hw,))
            self.rgb2hsv(
                aux_reshape[0:3], h, s, v, r, g, b, aux_nram, aux1_nram, aux2_nram,
            )
            self.add_delta(h, self.new_delta)
            self.hsv2rgb(h, s, v, r, g, b, aux_nram, aux1_nram, aux_int_nram)
            self.tcp.add(rgb_reshape[0:3], rgb_reshape[0:3], s)
            self.tcp.transpose(
                hsv[: self.r_hw * self.c].reshape(
                    (1, self.r_hw / self.line_align, 1, self.line_align * self.c,)
                ),
                rgb[: self.r_hw * self.c].reshape(
                    (1, self.line_align * self.c, self.r_hw / self.line_align, 1,)
                ),
                (0, 2, 3, 1),
            )
            self.tcp.transpose(
                rgb[: self.r_hw * self.c].reshape(
                    (1, self.line_align, 1, self.r_hw / self.line_align * self.c,)
                ),
                hsv[: self.r_hw * self.c].reshape(
                    (1, self.r_hw / self.line_align * self.c, self.line_align, 1,)
                ),
                (0, 2, 3, 1),
            )
        with self.tcp.block("data_copy" if db else "null"):
            self.tcp.memcpy(
                self.dst_gdram[
                    batch_index,
                    offset_h_start : offset_h_start + self.r_h,
                    offset_w_start : offset_w_start + self.r_w,
                    :,
                ],
                rgb.reshape((self.nram_limit / self.r_w / self.c, self.r_w, self.c,))[
                    : self.r_h
                ][:][:],
            )

    def compute_body(self):
        self.prepare_active_tab()

        # modify core_id for memcore core_id error
        self.core_id = self.tcp.Scalar(dtype=bp.int32, name="core_id")
        self.core_id.assign(self.tcp.coreId)
        with self.tcp.if_scope(self.core_id == 0x80):
            self.core_id.assign(0)

        # divide h based on task dim number.
        h_loop_num = self.tcp.Scalar(dtype=bp.int32, name="h_loop_num")
        self.divide_h_dim(
            self.tcp.taskDim, self.tcp.taskId, self.real_task_num, self.core_offset
        )
        h_loop_num.assign(
            (self.real_task_num + self.row_each_per - 1) / self.row_each_per - 1
        )

        # divide w if w is too large
        self.divide_w(self.r_w_tmp, h_loop_num)

        # loop for multiple batch.
        with self.tcp.for_range(begin=0, end=self.n, name="batch_idx") as i:
            # align by reshape align.
            self.r_hw.assign(
                (self.row_each_per * self.r_w_tmp + self.reshape_align - 1)
                / self.reshape_align
                * self.reshape_align
            )
            # loop for multiple w divide.
            offset_w_start = self.tcp.Scalar(dtype=bp.int32, name="offset_w_start")
            with self.tcp.for_range(
                begin=0,
                end=(self.w + self.r_w_tmp - 1) / self.r_w_tmp,
                name="width_idx",
            ) as j:
                self.r_w.assign(self.r_w_tmp)
                with self.tcp.if_scope(
                    j == ((self.w + self.r_w_tmp - 1) / self.r_w_tmp - 1)
                ):
                    self.r_w.assign(self.w - j * self.r_w_tmp)
                offset_w_start.assign(j * self.r_w_tmp)
                self.r_h.assign(self.row_each_per)
                h_loop_num.assign(
                    (self.real_task_num + self.row_each_per - 1) / self.row_each_per - 1
                )
                with self.tcp.for_range(begin=0, end=h_loop_num, stage=self.stage) as k:
                    # h_loop_num = 3
                    # loop for h divide.
                    self.rgb_nram = self.tcp.Buffer(
                        shape=(self.nram_use // self.dtype.bytes,),
                        dtype=self.dtype,
                        scope="nram",
                        name="rgb_nram",
                    )

                    offset_h_start = k * self.row_each_per + self.core_offset
                    self.loop_body(i, offset_h_start, offset_w_start, db=True)

                with self.tcp.if_scope(self.real_task_num > 0):
                    # compute for h divide loop tail.
                    self.r_h.assign(self.real_task_num - h_loop_num * self.row_each_per)
                    offset_h_start = h_loop_num * self.row_each_per + self.core_offset
                    self.r_hw.assign(
                        (self.r_h * self.r_w + self.reshape_align - 1)
                        / self.reshape_align
                        * self.reshape_align
                    )
                    self.loop_body(i, offset_h_start, offset_w_start, db=False)

        return self.tcp.BuildBANG(
            inputs=[self.src_gdram, self.delta,],
            outputs=[self.dst_gdram],
            kernel_name=KERNEL_NAME,
        )


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_adjust_hue(dtype=None, target=None):
    stage = 1
    task_type = TaskType.UNION4
    op_mod = AdjustHue(dtype, stage, target, task_type).compute_body()
    return op_mod
