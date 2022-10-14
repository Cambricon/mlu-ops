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
# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring
# pylint: disable=attribute-defined-outside-init
"""AdjustHue operator implementation using BANGPy TCP API."""
from active_table import ACTIVE_TABLE1, ACTIVE_TABLE2, ACTIVE_TABLE3
from active_table import ACTIVE_TABLE4, ACTIVE_TABLE5, ACTIVE_TABLE6, ACTIVE_TABLE7
from active_table import CONST_TABLE1, CONST_TABLE2
import bangpy as bp
from bangpy import tcp
from bangpy.script import build_module, ty

DTYPES = [bp.float16, bp.float32]
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]
KERNEL_NAME = "adjust_hue"


class AdjustHue(object):
    def __init__(
        self,
        dtype: ty.string,
        dtype_bits: ty.int32,
        stage: ty.boolean,
        active_tab1: ty.Tuple,
        active_tab2: ty.Tuple,
        active_tab3: ty.Tuple,
        active_tab4: ty.Tuple,
        active_tab5: ty.Tuple,
        active_tab6: ty.Tuple,
        active_tab7: ty.Tuple,
        const_tab1: ty.Tuple,
        const_tab2: ty.Tuple,
    ) -> None:
        self.dtype = dtype
        self.dtype_bits = dtype_bits
        self.stage = stage
        self.active_tab1 = active_tab1
        self.active_tab2 = active_tab2
        self.active_tab3 = active_tab3
        self.active_tab4 = active_tab4
        self.active_tab5 = active_tab5
        self.active_tab6 = active_tab6
        self.active_tab7 = active_tab7
        self.const_tab1 = const_tab1
        self.const_tab2 = const_tab2

    def prepare_active_tab(self):
        # memcpy for active table.
        tcp.memcpy(self.tab1_nram, self.active_tab_1)
        tcp.memcpy(self.const1_nram, self.const_tab_1.reinterpret_cast(self.dtype))
        tcp.memcpy(self.tab2_nram, self.active_tab_2)
        tcp.memcpy(self.tab3_nram, self.active_tab_3)
        tcp.memcpy(self.tab4_nram, self.active_tab_4)
        tcp.memcpy(self.tab5_nram, self.active_tab_5)
        tcp.memcpy(self.tab6_nram, self.active_tab_6)
        tcp.memcpy(self.tab7_nram, self.active_tab_7)

    def rgb2hsv(
        self,
        aux_full: ty.Buffer("nram"),  # type: ignore
        h: ty.Buffer("nram"),  # type: ignore
        s: ty.Buffer("nram"),  # type: ignore
        v: ty.Buffer("nram"),  # type: ignore
        r: ty.Buffer("nram"),  # type: ignore
        g: ty.Buffer("nram"),  # type: ignore
        b: ty.Buffer("nram"),  # type: ignore
        aux: ty.Buffer("nram"),  # type: ignore
        aux1: ty.Buffer("nram"),  # type: ignore
        aux2: ty.Buffer("nram"),  # type: ignore
        r_hw: ty.int32,
    ):
        # get max, min, mid
        # use v to save max
        # use s to save min
        tcp.maximum(v, r, g)
        tcp.minimum(s, r, g)
        tcp.maximum(v, v, b)
        tcp.minimum(s, s, b)
        tcp.equal(aux, v, s)
        tcp.subtract(h, v, s)
        tcp.add(h, h, aux)
        if self.dtype == "float16":
            tcp.type_convert(aux[: 2 * r_hw].reinterpret_cast("float32"), h, 0)
            tcp.reciprocal(
                aux[: 2 * r_hw].reinterpret_cast("float32"),
                aux[: 2 * r_hw].reinterpret_cast("float32"),
                mode="hp",
            )
            tcp.type_convert(h, aux[: 2 * r_hw].reinterpret_cast("float32"), 0, "rd")
        else:
            tcp.reciprocal(h, h, mode="hp")
        tcp.subtract(aux, g, b)  # g-b
        tcp.subtract(aux1, b, r)  # b-r
        tcp.subtract(aux2, r, g)  # r-g
        tcp.multiply(aux_full, aux_full, h)
        # r==max
        tcp.equal(r, r, v)
        # g==max
        tcp.equal(g, g, v)
        # b==max
        tcp.logic_or(b, r, g)
        tcp.subtract(g, b, r)
        tcp.logic_not(b, b)
        tcp.add(aux1, aux1, 2)
        tcp.add(aux2, aux2, 4)
        tcp.multiply(r, r, aux)
        tcp.multiply(g, g, aux1)
        tcp.multiply(b, b, aux2)
        tcp.add(h, r, g)
        tcp.add(h, h, b)

    def add_delta(self, h: ty.Buffer("nram"), delta: ty.float16):  # type: ignore
        tcp.add(h, h, delta)

    def hsv2rgb(
        self,
        h: ty.Buffer("nram"),  # type: ignore
        s: ty.Buffer("nram"),  # type: ignore
        v: ty.Buffer("nram"),  # type: ignore
        r: ty.Buffer("nram"),  # type: ignore
        g: ty.Buffer("nram"),  # type: ignore
        b: ty.Buffer("nram"),  # type: ignore
        aux: ty.Buffer("nram"),  # type: ignore
        aux1: ty.Buffer("nram"),  # type: ignore
        aux_int: ty.Buffer("nram"),  # type: ignore
    ):
        tcp.subtract(v, v, s)
        tcp.type_convert(aux_int, h, 0, "dn")
        tcp.type_convert(aux, aux_int, 0)
        tcp.type_convert(aux_int, h, 0, "tz")
        tcp.type_convert(aux1, aux_int, 0)
        tcp.lut_active(aux1, aux1, self.tab1_nram, self.const1_nram)
        tcp.subtract(h, h, aux1)
        tcp.abs(h, h)
        tcp.multiply(h, h, v)
        # aux:h_category
        # h:ratio
        # s:vmin
        # v:vmax-min
        # r
        tcp.lut_active(r, aux, self.tab2_nram, self.const1_nram)
        tcp.lut_active(aux1, aux, self.tab3_nram, self.const1_nram)
        tcp.multiply(r, r, v)
        tcp.multiply(aux1, aux1, h)
        tcp.add(r, r, aux1)
        # g
        tcp.lut_active(g, aux, self.tab4_nram, self.const1_nram)
        tcp.lut_active(aux1, aux, self.tab5_nram, self.const1_nram)
        tcp.multiply(g, g, v)
        tcp.multiply(aux1, aux1, h)
        tcp.add(g, g, aux1)
        # b
        tcp.lut_active(b, aux, self.tab6_nram, self.const1_nram)
        tcp.lut_active(aux1, aux, self.tab7_nram, self.const1_nram)
        tcp.multiply(b, b, v)
        tcp.multiply(aux1, aux1, h)
        tcp.add(b, b, aux1)

    def loop_body(
        self,
        batch_index: ty.int32,
        offset_h_start: ty.int32,
        offset_w_start: ty.int32,
        r_h: ty.int32,
        r_w: ty.int32,
        r_hw: ty.int32,
        line_align: ty.int32,
        nram_limit: ty.int32,
    ):
        rgb = self.rgb_nram[: (nram_limit / r_w) * r_w * self.c]
        hsv = self.hsv_nram[: (nram_limit / r_w) * r_w * self.c]
        aux = self.aux_nram[: (nram_limit / r_w) * r_w * self.c]
        aux_int = self.aux_nram[nram_limit * 2 : nram_limit * 3].reinterpret_cast(
            "int16"
        )
        with tcp.block("data_copy"):
            tcp.memcpy(
                rgb.reshape(
                    (
                        nram_limit / r_w,
                        r_w,
                        self.c,
                    )
                )[:r_h, :r_w, : self.c],
                self.src_gdram[
                    batch_index,
                    offset_h_start : offset_h_start + r_h,
                    offset_w_start : offset_w_start + r_w,
                    : self.c,
                ],
            )
        with tcp.block("compute"):
            tcp.transpose(
                hsv[: (r_hw / line_align) * line_align * self.c].reshape(
                    (
                        1,
                        r_hw / line_align * self.c,
                        1,
                        line_align,
                    )
                ),
                rgb[: (r_hw / line_align) * line_align * self.c].reshape(
                    (
                        1,
                        line_align,
                        1,
                        r_hw / line_align * self.c,
                    )
                ),
                (0, 3, 1, 2),
            )
            tcp.transpose(
                rgb[: (r_hw / line_align) * line_align * self.c].reshape(
                    (
                        1,
                        line_align * self.c,
                        1,
                        r_hw / line_align,
                    )
                ),
                hsv[: (r_hw / line_align) * line_align * self.c].reshape(
                    (
                        1,
                        r_hw / line_align,
                        1,
                        line_align * self.c,
                    )
                ),
                (0, 3, 1, 2),
            )
            rgb_reshape = rgb[: r_hw * self.c].reshape((self.c, r_hw))
            aux_reshape = aux[: r_hw * self.c].reshape((self.c, r_hw))
            h = hsv[: r_hw * self.c].reshape((self.c, r_hw))[0]
            s = hsv[: r_hw * self.c].reshape((self.c, r_hw))[1]
            v = hsv[: r_hw * self.c].reshape((self.c, r_hw))[2]
            r = rgb[: r_hw * self.c].reshape((self.c, r_hw))[0]
            g = rgb[: r_hw * self.c].reshape((self.c, r_hw))[1]
            b = rgb[: r_hw * self.c].reshape((self.c, r_hw))[2]
            aux_nram = aux[: r_hw * self.c].reshape((self.c, r_hw))[0]
            aux1_nram = aux[: r_hw * self.c].reshape((self.c, r_hw))[1]
            aux2_nram = aux[: r_hw * self.c].reshape((self.c, r_hw))[2]
            aux_int_nram = aux_int[:r_hw].reshape((r_hw,))
            self.rgb2hsv(
                aux_reshape[0:3],
                h,
                s,
                v,
                r,
                g,
                b,
                aux_nram,
                aux1_nram,
                aux2_nram,
                r_hw,
            )
            self.add_delta(h, self.delta)
            self.hsv2rgb(h, s, v, r, g, b, aux_nram, aux1_nram, aux_int_nram)
            tcp.add(rgb_reshape[0:3], rgb_reshape[0:3], s)
            tcp.transpose(
                hsv[: (r_hw / line_align) * line_align * self.c].reshape(
                    (
                        1,
                        r_hw / line_align,
                        1,
                        line_align * self.c,
                    )
                ),
                rgb[: (r_hw / line_align) * line_align * self.c].reshape(
                    (
                        1,
                        line_align * self.c,
                        r_hw / line_align,
                        1,
                    )
                ),
                (0, 2, 3, 1),
            )
            tcp.transpose(
                rgb[: (r_hw / line_align) * line_align * self.c].reshape(
                    (
                        1,
                        line_align,
                        1,
                        r_hw / line_align * self.c,
                    )
                ),
                hsv[: (r_hw / line_align) * line_align * self.c].reshape(
                    (
                        1,
                        r_hw / line_align * self.c,
                        line_align,
                        1,
                    )
                ),
                (0, 2, 3, 1),
            )
        with tcp.block("data_copy"):
            tcp.memcpy(
                self.dst_gdram[
                    batch_index,
                    offset_h_start : offset_h_start + r_h,
                    offset_w_start : offset_w_start + r_w,
                    : self.c,
                ],
                rgb.reshape(
                    (
                        nram_limit / r_w,
                        r_w,
                        self.c,
                    )
                )[:r_h, :r_w, : self.c],
            )

    def main(
        self,
        inputs: ty.handle,
        outputs: ty.handle,
        n: ty.int32,
        h: ty.int32,
        w: ty.int32,
        c: ty.int32,
        delta: ty.float32,
    ) -> None:
        tgt = tcp.target()
        self.n = n
        self.h = h
        self.w = w
        self.c = c
        self.delta = tcp.cast(delta * 6.0, self.dtype)
        self.src_gdram = tcp.match_buffer(inputs, [n, h, w, c], self.dtype)
        self.dst_gdram = tcp.match_buffer(outputs, [n, h, w, c], self.dtype)
        for i in tcp.thread_binding(0, tgt.cluster_num, thread="blockIdx.x"):
            for j in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):
                if self.dtype == "float16":
                    self.const_tab_1 = tcp.alloc_const(
                        value=self.const_tab1, shape=(32,), dtype="int16", scope="gdram"
                    )
                else:
                    self.const_tab_1 = tcp.alloc_const(
                        value=self.const_tab2, shape=(32,), dtype="int16", scope="gdram"
                    )
                self.const1_nram = tcp.alloc_buffer(
                    shape=(64 // self.dtype_bits,), dtype=self.dtype, scope="nram"
                )

                self.active_tab_1 = tcp.alloc_const(
                    value=self.active_tab1,
                    shape=(1, 128),
                    dtype=self.dtype,
                    scope="gdram",
                )
                self.tab1_nram = tcp.alloc_buffer(
                    shape=(1, 128), dtype=self.dtype, scope="nram"
                )
                # use for r_c
                self.active_tab_2 = tcp.alloc_const(
                    value=self.active_tab2,
                    shape=(1, 128),
                    dtype=self.dtype,
                    scope="gdram",
                )
                self.tab2_nram = tcp.alloc_buffer(
                    shape=(1, 128), dtype=self.dtype, scope="nram"
                )
                # use for r_x
                self.active_tab_3 = tcp.alloc_const(
                    value=self.active_tab3,
                    shape=(1, 128),
                    dtype=self.dtype,
                    scope="gdram",
                )
                self.tab3_nram = tcp.alloc_buffer(
                    shape=(1, 128), dtype=self.dtype, scope="nram"
                )
                # use for g_c
                self.active_tab_4 = tcp.alloc_const(
                    value=self.active_tab4,
                    shape=(1, 128),
                    dtype=self.dtype,
                    scope="gdram",
                )
                self.tab4_nram = tcp.alloc_buffer(
                    shape=(1, 128), dtype=self.dtype, scope="nram"
                )
                # use for g_x
                self.active_tab_5 = tcp.alloc_const(
                    value=self.active_tab5,
                    shape=(1, 128),
                    dtype=self.dtype,
                    scope="gdram",
                )
                self.tab5_nram = tcp.alloc_buffer(
                    shape=(1, 128), dtype=self.dtype, scope="nram"
                )
                # use for b_c
                self.active_tab_6 = tcp.alloc_const(
                    value=self.active_tab6,
                    shape=(1, 128),
                    dtype=self.dtype,
                    scope="gdram",
                )
                self.tab6_nram = tcp.alloc_buffer(
                    shape=(1, 128), dtype=self.dtype, scope="nram"
                )
                # use for b_x
                self.active_tab_7 = tcp.alloc_const(
                    value=self.active_tab7,
                    shape=(1, 128),
                    dtype=self.dtype,
                    scope="gdram",
                )
                self.tab7_nram = tcp.alloc_buffer(
                    shape=(1, 128), dtype=self.dtype, scope="nram"
                )
                line_align = 64 // self.dtype_bits
                reshape_align = 4096 // self.dtype_bits // self.dtype_bits
                nram_use = (tgt.nram_size - 52 * 1024) // 4 // self.dtype_bits
                nram_limit_size = nram_use / c / reshape_align * reshape_align
                nram_limit_size = nram_limit_size - reshape_align
                row_each_per = tcp.cast(nram_limit_size / w, "int32")
                self.hsv_nram = tcp.alloc_buffer(
                    shape=(nram_use,),
                    dtype=self.dtype,
                    scope="nram",
                )
                self.aux_nram = tcp.alloc_buffer(
                    shape=(nram_use,),
                    dtype=self.dtype,
                    scope="nram",
                )

                self.prepare_active_tab()

                # diveide h dim
                task_dim = tgt.cluster_num * tgt.core_num
                task_id = i * 4 + j
                real_task_num = (self.h + task_dim - 1) / task_dim
                core_offset = real_task_num * task_id
                if (self.h / task_dim) * task_dim + task_id >= self.h:
                    core_offset = (self.h / task_dim) * task_id + self.h % task_dim
                    if real_task_num * task_dim != self.h:
                        real_task_num = real_task_num - 1

                h_loop_num = (real_task_num + row_each_per - 1) / row_each_per - 1

                # divide w dim
                r_w_tmp = self.w / ((self.w + nram_limit_size - 1) / nram_limit_size)
                if row_each_per == 0:
                    r_w_tmp = self.w / (
                        (self.w + nram_limit_size - 1) / nram_limit_size
                    )
                    row_each_per = 1
                else:
                    r_w_tmp = self.w
                    if h_loop_num <= 4 and row_each_per != 1:
                        if r_w_tmp <= 3900:
                            row_each_per = 3900 / self.w

                for batch_idx in range(self.n):
                    # align by reshape align.
                    r_hw = (
                        (row_each_per * r_w_tmp + reshape_align - 1)
                        / reshape_align
                        * reshape_align
                    )
                    # loop for multiple w divide.
                    for width_idx in range((w + r_w_tmp - 1) / r_w_tmp):
                        r_w = r_w_tmp
                        if width_idx == ((w + r_w_tmp - 1) / r_w_tmp - 1):
                            r_w = self.w - width_idx * r_w_tmp
                        offset_w_start = width_idx * r_w_tmp
                        r_h = row_each_per
                        h_loop_num = (
                            real_task_num + row_each_per - 1
                        ) / row_each_per - 1
                        for loop in range(h_loop_num, pipeline=self.stage):  # type: ignore
                            self.rgb_nram = tcp.alloc_buffer(
                                shape=(nram_use,),
                                dtype=self.dtype,
                                scope="nram",
                            )
                            self.loop_body(
                                batch_idx,
                                loop * row_each_per + core_offset,
                                offset_w_start,
                                r_h,
                                r_w,
                                r_hw,
                                line_align,
                                nram_limit_size,
                            )

                        if real_task_num > 0:
                            # compute for h divide loop tail.
                            r_h = real_task_num - h_loop_num * row_each_per
                            offset_h_start = h_loop_num * row_each_per + core_offset
                            r_hw = (
                                (r_h * r_w + reshape_align - 1)
                                / reshape_align
                                * reshape_align
                            )
                            self.loop_body(
                                batch_idx,
                                offset_h_start,
                                offset_w_start,
                                r_h,
                                r_w,
                                r_hw,
                                line_align,
                                nram_limit_size,
                            )


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_adjust_hue(dtype=None, target=None):
    stage = True
    f = build_module.build(
        AdjustHue(
            dtype.name,
            4 if dtype.name == "float32" else 2,
            stage,
            ACTIVE_TABLE1,
            ACTIVE_TABLE2,
            ACTIVE_TABLE3,
            ACTIVE_TABLE4,
            ACTIVE_TABLE5,
            ACTIVE_TABLE6,
            ACTIVE_TABLE7,
            CONST_TABLE1,
            CONST_TABLE2,
        ),
        target,
        KERNEL_NAME,
    )
    return f
