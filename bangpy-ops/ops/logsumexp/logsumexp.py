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
"""A multi-platform code link example test for BANGPy TCP."""
import bangpy
from bangpy.tcp.util import round_down
from bangpy import tcp
from bangpy.platform.bang_config import TARGET

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu290"]
KERNEL_NAME = "Logsumexp"


class NramSet:
    bp = None
    m_size = 0
    m_buff = None

    def init(self, bp, buff):
        self.bp = bp
        self.m_buff = buff

        self.m_size = self.bp.Scalar(bangpy.int32, "m_size", 0)

    def add(self, index):
        self.m_buff[self.m_size] = index
        self.m_size.assign(self.m_size + 1)

    def is_in(self, index):
        ret = self.bp.Scalar(bangpy.int32, "chk_ret", 0)
        with self.bp.for_range(0, self.m_size) as i:
            with self.bp.if_scope(index == self.m_buff[i]):
                ret.assign(1)
        return ret


class DataMan:
    bp = None

    m_current_core_start = None
    m_current_core_end = None
    m_total_count_in_core = None

    def init(self, bp):
        self.bp = bp
        self.m_current_core_start = self.bp.Scalar(bangpy.int32, "current_core_start")
        self.m_current_core_end = self.bp.Scalar(bangpy.int32, "current_core_end")
        self.m_total_count_in_core = self.bp.Scalar(bangpy.int32, "total_count_in_core")

    def calc_core_process_count(self, data_total_len, task_num):
        one_core_count = self.bp.Scalar(bangpy.int32, "one_core_count")
        one_core_count.assign(data_total_len // task_num)

        remain = self.bp.Scalar(bangpy.int32, "remain")
        remain.assign(data_total_len % task_num)

        with self.bp.if_scope(self.bp.taskId < remain):
            self.m_current_core_start.assign((one_core_count + 1) * self.bp.taskId)
            self.m_current_core_end.assign((one_core_count + 1) *
                                           (self.bp.taskId + 1) - 1)
        with self.bp.else_scope():
            self.m_current_core_start.assign((one_core_count + 1) *
                                             remain + one_core_count * (self.bp.taskId - remain))
            self.m_current_core_end.assign((one_core_count + 1) * remain +
                one_core_count * (self.bp.taskId - remain) + one_core_count - 1)

        self.m_total_count_in_core.assign(self.m_current_core_end - self.m_current_core_start + 1)


class LogSumCalcer:
    def __init__(self, bp, dtype):
        self.dtype = dtype
        self.bp = bp
        self.m_value = self.bp.Scalar(self.dtype, "m_value", 0.0)
        self._oper_count = self.bp.Scalar(bangpy.int32, "_oper_count", 0)

    def reset(self):
        self.m_value.assign(0.0)
        self._oper_count.assign(0)

    def calc_value(self, x, y):
        natural_base = self.bp.Scalar(bangpy.float32,
                                      "natural_base", \
                                            2.7182818284590452353602874713526624977572470936999)
        max_threshold_valu = self.bp.Scalar(bangpy.float32,
                                            "max_threshold_valu", 88.722008965395851698332450562653)
        min_threshold_valu = self.bp.Scalar(bangpy.float32,
                                            "min_threshold_valu", \
                                                -87.332719095296162600686375692197)
        const_one = self.bp.Scalar(bangpy.float32, "const_one", 1)
        scalar_res = self.bp.Scalar(bangpy.float32, "scalar_res", (y - x).astype(bangpy.float32))
        with self.bp.if_scope(tcp.all(scalar_res <= max_threshold_valu,
                                      scalar_res >= min_threshold_valu)):
            scalar_res.assign(self.bp.scalar_pow(natural_base, scalar_res))
            scalar_res.assign(scalar_res + const_one)
            scalar_res.assign(self.bp.scalar_log(scalar_res) /
                              self.bp.scalar_log(natural_base))
            scalar_res.assign(scalar_res + x.astype(bangpy.float32))
        with self.bp.else_scope():
            with self.bp.if_scope(scalar_res > max_threshold_valu):
                scalar_res.assign(y.astype(bangpy.float32))
            with self.bp.else_scope():
                scalar_res.assign(x.astype(bangpy.float32))
        return scalar_res.astype(self.dtype)

    def calc_buffer(self, buffer, start_index, end_index):
        natural_base = self.bp.Scalar(bangpy.float32, "natural_base",
                                      2.7182818284590452353602874713526624977572470936999)
        const_one = self.bp.Scalar(bangpy.float32, "const_one", 1)
        max_threshold_valu = self.bp.Scalar(bangpy.float32, "max_threshold_valu")
        min_threshold_valu = self.bp.Scalar(bangpy.float32, "min_threshold_valu")
        max_threshold_valu.assign(88.722008965395851698332450562653)
        min_threshold_valu.assign(-87.332719095296162600686375692197)
        data_length = self.bp.Scalar(bangpy.int32, "data_length", end_index - start_index)
        sub_value = self.bp.Scalar(bangpy.float32, "sub_value")
        sum_value = self.bp.Scalar(bangpy.float32, "sum_value",
                                   buffer[start_index].astype(bangpy.float32))
        with self.bp.for_range(0, data_length - 1) as i:
            sub_value.assign(sum_value - buffer[i + 1].astype(bangpy.float32))
            with self.bp.if_scope(tcp.all(sub_value <= max_threshold_valu,
                                          sub_value >= min_threshold_valu)):
                sum_value.assign(self.bp.scalar_pow(natural_base, sub_value) + const_one)
                sum_value.assign(self.bp.scalar_log(sum_value) / self.bp.scalar_log(natural_base))
                sum_value.assign(sum_value + buffer[i + 1].astype(bangpy.float32))
            with self.bp.else_scope():
                with self.bp.if_scope(sub_value < min_threshold_valu):
                    sum_value.assign(buffer[i + 1].astype(bangpy.float32))
        return sum_value.astype(self.dtype)

    def add_buffer(self, buffer, start_index, end_index):
        with self.bp.if_scope(self._oper_count == 0):
            self.m_value.assign(self.calc_buffer(buffer, start_index, end_index))
        with self.bp.else_scope():
            ret_value = self.calc_buffer(buffer, start_index, end_index)
            tmp_calc_value = self.bp.Scalar(bangpy.float32, "tmp_calc_value",
                self.m_value.astype(bangpy.float32))
            tmp_ret = self.calc_value(tmp_calc_value, ret_value)
            self.m_value.assign(tmp_ret.astype(self.dtype))

        self._oper_count.assign(self._oper_count + 1)
        return self.m_value


def get_norm_index(data_pos, dim_len):
    index = (data_pos + dim_len - 1) // dim_len
    return index - 1


class LogSumExpPara:
    dim_len = None
    h = None
    w = None
    dtype_sz = None
    task_num = None
    target = None

    calc_size = None


class CopyPara:
    def __init__(self, dst, offset_d, src, offset_s):
        self.dst = dst
        self.offset_dst = offset_d
        self.src = src
        self.offset_src = offset_s


class Logsumexp:
    def __init__(self, dtype, target, task_num):
        self.bp = tcp.TCP(target)

        self.para = LogSumExpPara()
        self.para.h = 0
        self.para.w = 0
        self.para.dim_len = 0
        self.para.task_num = task_num
        self.para.dtype_sz = dtype.bytes
        self.para.target = target
        self.para.calc_size = self.bp.Scalar(bangpy.int32, "calc_size")
        self.dtype = dtype
        self.output_len = 0
        self.nram_process_count = None
        self.nram_calc_buffer = None
        self.flat_nram = None
        self.dman = DataMan()

    def compute_body(self):
        self.dman.init(self.bp)
        self.bp.launch_task(self.para.task_num, 1, 1)
        self.para.dim_len = self.bp.SizeVar("dim_len")
        self.para.h = self.bp.SizeVar("h")
        self.para.w = self.bp.SizeVar("w")
        self.output_len = self.bp.SizeVar("output_len")
        gram_tensor = self.bp.Buffer(
            shape=(self.para.h * self.para.w,),
            name="gram_tensor", dtype=self.dtype, scope="global"
        )

        gram_buffer_out = self.bp.Buffer(
            shape=(self.output_len,), name="gram_buffer_out", dtype=self.dtype, scope="global"
        )

        border_array_size = 128
        gram_border_buf_out = self.bp.Buffer(
            shape=(border_array_size * 2,),
            name="gram_border_buf_out", dtype=self.dtype, scope="global"
        )
        gram_border_idx_out = self.bp.Buffer(
            shape=(border_array_size * 2,), name="gram_border_idx_out",
            dtype=bangpy.int32, scope="global"
        )

        with self.bp.if_scope(self.bp.taskId == 0):
            gram_reshape_tensor = gram_tensor.reshape([self.para.h, self.para.w])
        self.bp.sync_all()

        nram_avable_size = round_down(TARGET(self.para.target).nram_size - 30 * 1024, 128)
        self.nram_process_count = nram_avable_size // self.para.dtype_sz
        self.nram_calc_buffer = self.bp.Buffer(
            shape=(self.nram_process_count, 1),
            name="nram_calc_buffer",
            dtype=self.dtype,
            scope="nram")

        self.m_buff = self.bp.Buffer(
            shape=(border_array_size * 2,), name="m_buff",
            dtype=bangpy.int32, scope="nram"
        )

        self.flat_nram = self.nram_calc_buffer.reshape([self.nram_process_count, ])

        self.dman.calc_core_process_count(self.para.h * self.para.w, self.para.task_num)

        with self.bp.if_scope(self.para.dim_len > self.nram_process_count):
            self.calc1(gram_reshape_tensor, gram_border_buf_out,
                       gram_border_idx_out, gram_buffer_out)
        with self.bp.else_scope():
            with self.bp.if_scope((self.para.h * self.para.w)
                                  // self.para.task_num + 1 < self.para.dim_len):
                self.calc1(gram_reshape_tensor, gram_border_buf_out,
                           gram_border_idx_out, gram_buffer_out)
            with self.bp.else_scope():
                self.calc2(gram_reshape_tensor, gram_border_buf_out,
                           gram_border_idx_out, gram_buffer_out)

        self.bp.sync_all()

        lc = LogSumCalcer(self.bp, self.dtype)
        with self.bp.if_scope(self.bp.taskId == 0):
            nset = NramSet()
            nset.init(self.bp, self.m_buff)

            with self.bp.for_range(0, border_array_size) as i:
                index1 = gram_border_idx_out[2 * i]
                index2 = gram_border_idx_out[2 * i + 1]
                norm_value1 = gram_border_buf_out[2 * i]
                norm_value2 = gram_border_buf_out[2 * i + 1]

                with self.bp.if_scope(index1 >= 0):
                    with self.bp.if_scope(nset.is_in(index1) == 0):
                        gram_buffer_out[index1] = norm_value1
                        nset.add(index1)
                    with self.bp.else_scope():
                        gram_buffer_out[index1] = \
                            lc.calc_value(gram_buffer_out[index1], norm_value1)

                with self.bp.if_scope(index2 >= 0):
                    with self.bp.if_scope(nset.is_in(index2) == 0):
                        gram_buffer_out[index2] = norm_value2
                        nset.add(index2)
                    with self.bp.else_scope():
                        gram_buffer_out[index2] = \
                            lc.calc_value(gram_buffer_out[index2], norm_value2)

        f = self.bp.BuildBANG(
            inputs=[gram_tensor,
                    self.para.dim_len, self.para.h, self.para.w,
                    self.output_len
                    ],
            outputs=[gram_border_buf_out, gram_border_idx_out, gram_buffer_out],
            kernel_name=KERNEL_NAME
        )
        return f

    def copy_from_2d_tensor(self, cp_para, dim_len, width, cp_len):
        dst = cp_para.dst
        offset_dst = cp_para.offset_dst
        src = cp_para.src
        offset_src = cp_para.offset_src

        big_row = offset_src // (width * dim_len)

        m = offset_src % dim_len + big_row * dim_len

        big_n = offset_src // dim_len
        n = big_n % width

        with self.bp.if_scope(offset_dst != offset_dst + cp_len // 2):
            self.bp.memcpy(dst[offset_dst:offset_dst + cp_len // 2, 0:1],
                           src[m:m + cp_len // 2, n:n + 1])

        with self.bp.if_scope(offset_dst + cp_len // 2 != offset_dst + cp_len):
            self.bp.memcpy(dst[offset_dst + cp_len // 2:offset_dst + cp_len, 0:1],
                           src[m + cp_len // 2:m + cp_len, n:n + 1])

    def get_calc_loop_count(self, dataman):
        return self.bp.Scalar(bangpy.int32, "calc_loop_count",
                              (dataman.m_total_count_in_core + self.nram_process_count - 1)
                              // self.nram_process_count)

    def calc1(self, gram_tensor, border_outputs, idx_outputs, outputs):
        once_loop_start = self.bp.Scalar(bangpy.int32, "once_loop_start")
        norm_offset = self.bp.Scalar(bangpy.int32, "norm_offset",
                                     self.dman.m_current_core_start % self.para.dim_len)
        norm_value = LogSumCalcer(self.bp, self.dtype)
        self.para.calc_size.assign(self.nram_process_count)
        once_norm_ok = self.bp.Scalar(bangpy.int32, "once_norm_ok", 0)
        cp_data_len = self.bp.Scalar(bangpy.int32, "cp_data_len", 0)
        with self.bp.for_range(0, self.get_calc_loop_count(self.dman)) as i:
            once_loop_start.assign(self.dman.m_current_core_start
                                   + self.nram_process_count * i)
            with self.bp.if_scope(i == self.get_calc_loop_count(self.dman) - 1):
                self.para.calc_size.assign(self.dman.m_total_count_in_core %
                                           self.nram_process_count)
                with self.bp.if_scope(self.para.calc_size == 0):
                    self.para.calc_size.assign(self.nram_process_count)

            norm_offset.assign(once_loop_start % self.para.dim_len)
            expect_cp_len = self.bp.Scalar(bangpy.int32,
                                           "expect_cp_len", self.para.dim_len - norm_offset)

            with self.bp.if_scope(expect_cp_len > self.para.calc_size):
                expect_cp_len.assign(self.para.calc_size)
                cp_para = CopyPara(self.nram_calc_buffer, 0, gram_tensor, once_loop_start)
                self.copy_from_2d_tensor(cp_para, self.para.dim_len, self.para.w, expect_cp_len)
                cp_data_len.assign(cp_data_len + expect_cp_len)
                norm_value.add_buffer(self.flat_nram, 0, expect_cp_len)
                with self.bp.if_scope(i == self.get_calc_loop_count(self.dman) - 1):
                    index = get_norm_index(once_loop_start + expect_cp_len, self.para.dim_len)
                    with self.bp.if_scope(once_norm_ok == 0):
                        border_outputs[self.bp.taskId * 2] = \
                            norm_value.m_value
                        idx_outputs[self.bp.taskId * 2] = index
                    with self.bp.else_scope():
                        border_outputs[self.bp.taskId * 2 + 1] = norm_value.m_value
                        idx_outputs[self.bp.taskId * 2 + 1] = index

            with self.bp.else_scope():
                cp_para = CopyPara(self.nram_calc_buffer, 0, gram_tensor, once_loop_start)
                self.copy_from_2d_tensor(cp_para, self.para.dim_len, self.para.w, expect_cp_len)
                cp_data_len.assign(cp_data_len + expect_cp_len)
                norm_value.add_buffer(self.flat_nram, 0, expect_cp_len)
                once_norm_ok.assign(1)
                index = get_norm_index(once_loop_start + expect_cp_len, self.para.dim_len)
                with self.bp.if_scope(cp_data_len < self.para.dim_len):
                    border_outputs[self.bp.taskId * 2] = \
                        norm_value.m_value
                    idx_outputs[self.bp.taskId * 2] = index
                with self.bp.else_scope():
                    outputs[index] = norm_value.m_value
                norm_value.reset()
                cp_data_len.assign(self.para.calc_size - expect_cp_len)
                with self.bp.if_scope(cp_data_len > 0):
                    cp_para = CopyPara(self.nram_calc_buffer, 0, gram_tensor, \
                        once_loop_start + expect_cp_len)
                    self.copy_from_2d_tensor(cp_para, self.para.dim_len, self.para.w, cp_data_len)
                    norm_value.add_buffer(self.flat_nram, 0, cp_data_len)
                    with self.bp.if_scope(i == self.get_calc_loop_count(self.dman) - 1):
                        border_outputs[self.bp.taskId * 2 + 1] = norm_value.m_value
                        idx_outputs[self.bp.taskId * 2 + 1] = index + 1

    def calc2(self, gram_tensor, border_outputs, idx_outputs, outputs):
        current_core_start = self.dman.m_current_core_start
        total_count_in_core = self.dman.m_total_count_in_core
        dim_len = self.para.dim_len
        norm_value = self.bp.Scalar(self.dtype, "norm_value", 0.0)
        lc = LogSumCalcer(self.bp, self.dtype)
        norm_offset = self.bp.Scalar(bangpy.int32, "norm_offset", current_core_start % dim_len)
        expect_cp_len = self.bp.Scalar(bangpy.int32, "expect_cp_len", 0)
        with self.bp.if_scope(norm_offset != 0):
            expect_cp_len.assign(dim_len - norm_offset)
            cp_para = CopyPara(self.nram_calc_buffer, 0, gram_tensor, current_core_start)
            self.copy_from_2d_tensor(cp_para, dim_len, self.para.w, expect_cp_len)
            calc_result = lc.calc_buffer(self.flat_nram, 0, expect_cp_len)
            norm_value.assign(calc_result)
            index = get_norm_index(current_core_start + expect_cp_len, dim_len)
            border_outputs[self.bp.taskId * 2] = norm_value
            idx_outputs[self.bp.taskId * 2] = index
        norm_start_pos = self.bp.Scalar(bangpy.int32,
                                        "norm_start_pos", current_core_start + expect_cp_len)
        nram_norm_count = self.bp.Scalar(bangpy.int32,
                                         "nram_norm_count", self.nram_process_count // dim_len)

        total_norm_in_core = self.bp.Scalar(bangpy.int32, "total_norm_in_core",
                                            (total_count_in_core - expect_cp_len) // dim_len)
        calc_loop_count = self.bp.Scalar(bangpy.int32, "calc_loop_count",
                                         (total_norm_in_core + nram_norm_count - 1) \
                                            // nram_norm_count)
        once_loop_norm_count = self.bp.Scalar(bangpy.int32, "nram_norm_count", nram_norm_count)
        with self.bp.for_range(0, calc_loop_count) as i:
            once_loop_start = self.bp.Scalar(bangpy.int32, "once_loop_start",
                                             norm_start_pos + nram_norm_count * dim_len * i)
            with self.bp.if_scope(i == calc_loop_count - 1):
                once_loop_norm_count.assign(total_norm_in_core % nram_norm_count)
                with self.bp.if_scope(once_loop_norm_count == 0):
                    once_loop_norm_count.assign(nram_norm_count)
            start_index = self.bp.Scalar(bangpy.int32, "norm_offset",
                                         once_loop_start // dim_len)
            with self.bp.for_range(0, once_loop_norm_count) as j:
                cp_para = CopyPara(self.nram_calc_buffer, 0, gram_tensor, \
                    once_loop_start + j * dim_len)
                self.copy_from_2d_tensor(cp_para, dim_len, self.para.w, dim_len)
                calc_result = lc.calc_buffer(self.flat_nram, 0, dim_len)
                norm_value.assign(calc_result)
                outputs[start_index + j] = norm_value
        norm_loop_end_pos = self.bp.Scalar(bangpy.int32, "norm_loop_end_pos",
                                           norm_start_pos + total_norm_in_core * dim_len)
        cur_loop_end_pos = self.bp.Scalar(bangpy.int32, "cur_loop_end_pos",
                                          current_core_start + total_count_in_core)
        with self.bp.if_scope(norm_loop_end_pos < cur_loop_end_pos):
            cp_para = CopyPara(self.nram_calc_buffer, 0, gram_tensor, norm_loop_end_pos)
            self.copy_from_2d_tensor(cp_para, dim_len, self.para.w,
                                     cur_loop_end_pos - norm_loop_end_pos)
            calc_result = lc.calc_buffer(self.flat_nram, 0, cur_loop_end_pos - norm_loop_end_pos)
            norm_value.assign(calc_result)
            index = get_norm_index(norm_loop_end_pos + 1, dim_len)
            border_outputs[self.bp.taskId * 2 + 1] = norm_value
            idx_outputs[self.bp.taskId * 2 + 1] = index


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_logsumexp(dtype=None, target=None):
    task_num = 4
    f = Logsumexp(dtype, target, task_num).compute_body()
    return f
