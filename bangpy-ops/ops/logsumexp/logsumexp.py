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
from bangpy.tcp.util import round_up, round_down
from bangpy import tcp
from bangpy.common import load_op_by_type
from bangpy.platform.bang_config import TARGET
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float32] #支持的类型
TARGET_LIST = ["mlu290"]#支持的设备
KERNEL_NAME = "Logsumexp"#算子名

class data_man:
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

        with self.bp.if_scope(self.bp.taskId < remain): #如果存在余数 将其均摊给各核   taskId从0起
            self.m_current_core_start.assign((one_core_count + 1) * self.bp.taskId )
            self.m_current_core_end.assign((one_core_count + 1) * \
                (self.bp.taskId + 1) - 1) #此处应该不需要减1 待验证  python切片会自动将上标减1
        with self.bp.else_scope():
            self.m_current_core_start.assign((one_core_count + 1) * \
                remain + one_core_count * (self.bp.taskId - remain))
            self.m_current_core_end.assign((one_core_count + 1) * remain + \
                one_core_count * (self.bp.taskId - remain) + one_core_count - 1)

        self.m_total_count_in_core.assign(self.m_current_core_end - self.m_current_core_start + 1)

class logsum_calcer:
    def __init__(self, bp, dtype):
        self.dtype = dtype
        self.bp = bp
        self.m_value = self.bp.Scalar(bangpy.float32, "m_value", 0.0)
        self._oper_count = self.bp.Scalar(bangpy.int32, "_oper_count", 0)

    def reset(self):
        self.m_value.assign(0.0)
        self._oper_count.assign(0)

    def calc_value(self, x, y):
        natural_base = self.bp.Scalar(bangpy.float32, \
            "natural_base",2.7182818284590452353602874713526624977572470936999)
        const_one = self.bp.Scalar(bangpy.float32, "const_one", 1)
        max_threshold_valu = self.bp.Scalar(bangpy.float32,\
            "max_threshold_valu",88.722008965395851698332450562653)
        min_threshold_valu = self.bp.Scalar(bangpy.float32,\
            "min_threshold_valu",-87.332719095296162600686375692197)

        const_one = self.bp.Scalar(bangpy.float32,"const_one", 1)
        scalar_res = self.bp.Scalar(bangpy.float32,"scalar_res", y - x)#计算结果   初始化为 y-x的差值
        with self.bp.if_scope(tcp.all(scalar_res <= max_threshold_valu, \
                    scalar_res >= min_threshold_valu)):#如果差值在合法范围内
            scalar_res.assign(self.bp.scalar_pow(natural_base, scalar_res))# 作为e的指数
            scalar_res.assign(scalar_res + const_one) # +1
            scalar_res.assign(self.bp.scalar_log(scalar_res) / \
                self.bp.scalar_log(natural_base))# 换底公式 计算自然对数
            scalar_res.assign(scalar_res + x) # +x
        with self.bp.else_scope():#如果y-x 后的结果不和法
            with self.bp.if_scope(scalar_res > max_threshold_valu): #超过上限 返回y
                scalar_res.assign(y)
            with self.bp.else_scope():#小于下限 返回 x
                scalar_res.assign(x)
        return scalar_res

    def calc_buffer(self, buffer, start_index, end_index):
        natural_base = self.bp.Scalar(bangpy.float32, "natural_base", \
            2.7182818284590452353602874713526624977572470936999)
        const_one = self.bp.Scalar(bangpy.float32, "const_one", 1)
        max_threshold_valu = self.bp.Scalar(bangpy.float32, "max_threshold_valu")
        min_threshold_valu = self.bp.Scalar(bangpy.float32, "min_threshold_valu")
        #这些数我是网上查的该类型大于0时的最大最小值 然后取了个ln得到的
        max_threshold_valu.assign(88.722008965395851698332450562653)
        min_threshold_valu.assign(-87.332719095296162600686375692197)
        data_length = self.bp.Scalar(bangpy.int32, "data_length", end_index - start_index)#传进来得数据长度

        sub_value = self.bp.Scalar(bangpy.float32, "sub_value")#y-x的差值
        sum_value = self.bp.Scalar(bangpy.float32, "sum_value",\
            buffer[start_index].astype(bangpy.float32))#
        with self.bp.for_range(0, data_length -1) as i:#这里 -1 是为了循环内省掉一个if
            sub_value.assign(sum_value - buffer [i + 1].astype(bangpy.float32))
            with self.bp.if_scope(tcp.all(sub_value <= max_threshold_valu, \
                    sub_value >= min_threshold_valu)):
                sum_value.assign(self.bp.scalar_pow(natural_base,sub_value)+const_one)
                sum_value.assign(self.bp.scalar_log(sum_value) / self.bp.scalar_log(natural_base))
                sum_value.assign(sum_value + buffer [i + 1])
            with self.bp.else_scope():
                with self.bp.if_scope(sub_value < min_threshold_valu):
                    sum_value.assign(buffer[i + 1])
        return sum_value

    def add_buffer(self, buffer, start_index, end_index):
        with self.bp.if_scope(self._oper_count == 0):
            self.m_value.assign(self.calc_buffer(buffer, start_index, end_index))
        with self.bp.else_scope():
            ret_value = self.calc_buffer(buffer, start_index, end_index)
            tmp_calc_value = self.bp.Scalar(bangpy.float32, "tmp_calc_value", self.m_value)
            tmp_ret = self.calc_value(tmp_calc_value, ret_value)
            self.m_value.assign(tmp_ret)

        self._oper_count.assign(self._oper_count + 1)
        return self.m_value

def get_norm_index(data_pos, dim_len):
    index = (data_pos + dim_len - 1) // dim_len
    return index - 1

class Logsumexp:
    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.dtype_sz = dtype.bytes
        self.bp = tcp.TCP(target)
        self._data_man = data_man()

    def compute_body(self):
        self._data_man.init(self.bp)
        self.bp.launch_task(self.task_num, 1, 1)

        self.dim_len = self.bp.SizeVar("dim_len")
        self.h = self.bp.SizeVar("h")
        self.w = self.bp.SizeVar("w")
        self.output_len = self.bp.SizeVar("output_len")

        gram_tensor = self.bp.Buffer(
            shape=(self.h * self.w, ), name="gram_tensor", dtype=self.dtype, scope="global"
        )

        gram_buffer_out = self.bp.Buffer(
            shape=(self.output_len, ), name="gram_buffer_out", dtype=self.dtype, scope="global"
        )

        border_array_size = 128
        gram_border_buf_out = self.bp.Buffer(
            shape=(border_array_size * 2, ), \
                name="gram_border_buf_out", dtype=self.dtype, scope="global"
        )
        gram_border_idx_out = self.bp.Buffer(
            shape=(border_array_size * 2, ), name="gram_border_idx_out", \
                dtype=bangpy.int32, scope="global"
        )

        with self.bp.if_scope(self.bp.taskId == 0):
            gram_reshape_tensor = gram_tensor.reshape([self.h, self.w])
        self.bp.sync_all()

        nram_avable_size = round_down(TARGET(self.target).nram_size - 30 * 1024, 128)
        self.nram_process_count = nram_avable_size // self.dtype_sz
        #self.nram_process_count = 2
        self.nram_calc_buffer = self.bp.Buffer(
            shape=(self.nram_process_count, 1),
            name="nram_calc_buffer",
            dtype=self.dtype,
            scope="nram")

        self._data_man.calc_core_process_count(self.h * self.w, self.task_num)

        with self.bp.if_scope(self.dim_len > self.nram_process_count):
            self.calc1(gram_reshape_tensor, gram_border_buf_out, \
                gram_border_idx_out, gram_buffer_out)
        with self.bp.else_scope(): #nram 虽然够了，但是要计算的数据量很小，以至于分摊到每个core上面的数据，还不够一个norm
            with self.bp.if_scope((self.h * self.w)  \
                    // self.task_num < self.dim_len):
                self.calc1(gram_reshape_tensor, gram_border_buf_out, \
                    gram_border_idx_out, gram_buffer_out)
            with self.bp.else_scope():
                self.calc2(gram_reshape_tensor, gram_border_buf_out, \
                    gram_border_idx_out, gram_buffer_out)

        # 处理边界数据
        lc = logsum_calcer(self.bp, self.dtype)
        with self.bp.if_scope(self.bp.taskId == 0):
            with self.bp.for_range(0, border_array_size) as i:
                index1 = gram_border_idx_out[2 * i]
                index2 = gram_border_idx_out[2 * i + 1]
                norm_value1 = gram_border_buf_out[2 * i]
                norm_value2 = gram_border_buf_out[2 * i + 1]

                with self.bp.if_scope(index1 >= 0):
                    with self.bp.if_scope(gram_buffer_out[index1] < 0):
                        gram_buffer_out[index1] = norm_value1
                    with self.bp.else_scope():
                        gram_buffer_out[index1] = \
                            lc.calc_value(gram_buffer_out[index1], norm_value1)

                with self.bp.if_scope(index2 >= 0):
                    with self.bp.if_scope(gram_buffer_out[index2] < 0):
                        gram_buffer_out[index2] = norm_value2
                    with self.bp.else_scope():
                        gram_buffer_out[index2] = \
                            lc.calc_value(gram_buffer_out[index2], norm_value2)

        f = self.bp.BuildBANG(
            inputs=[gram_tensor,
                    self.dim_len, self.h, self.w,
                    self.output_len
                    ],
            outputs=[gram_border_buf_out, gram_border_idx_out, gram_buffer_out],
            kernel_name=KERNEL_NAME
            )
        return f

    def copy_from_2d_tensor(self, dst, offset_dst, src, offset_src, dim_len, width, cp_len):
        big_row = offset_src // (width * dim_len)

        m = offset_src % dim_len + big_row * dim_len

        big_n = (offset_src + dim_len - 1) // dim_len
        n = big_n % width

        with self.bp.if_scope(offset_dst != offset_dst + cp_len // 2):
            self.bp.memcpy(dst[offset_dst:offset_dst + cp_len // 2, 0:1], \
                src[m:m + cp_len  // 2, n:n + 1])

        with self.bp.if_scope(offset_dst + cp_len // 2 != offset_dst + cp_len):
            self.bp.memcpy(dst[offset_dst + cp_len // 2:offset_dst + cp_len, 0:1], \
                src[m + cp_len // 2:m + cp_len, n:n + 1])

    def calc_norm(self, buffer, start_index, end_index):
        #with self.bp.if_scope(end_index == start_index + 1):
        #    return buffer[start_index]

        natural_base = self.bp.Scalar(bangpy.float32, "natural_base", \
            2.7182818284590452353602874713526624977572470936999)
        const_one = self.bp.Scalar(bangpy.float32, "const_one", 1)
        max_threshold_valu = self.bp.Scalar(bangpy.float32, "max_threshold_valu")
        min_threshold_valu = self.bp.Scalar(bangpy.float32, "min_threshold_valu")
        #这些数我是网上查的该类型大于0时的最大最小值 然后取了个ln得到的
        max_threshold_valu.assign(88.722008965395851698332450562653)
        min_threshold_valu.assign(-87.332719095296162600686375692197)
        data_length = self.bp.Scalar(bangpy.int32, "data_length", end_index - start_index)#传进来得数据长度

        sub_value = self.bp.Scalar(bangpy.float32, "sub_value")#y-x的差值
        sum_value = self.bp.Scalar(bangpy.float32, \
            "sum_value",buffer[start_index].astype(bangpy.float32))#
        with self.bp.for_range(0, data_length -1) as i:#这里 -1 是为了循环内省掉一个if
            sub_value.assign(sum_value - buffer [i + 1].astype(bangpy.float32))
            with self.bp.if_scope(tcp.all(sub_value <= \
                    max_threshold_valu,sub_value >= min_threshold_valu)):
                sum_value.assign(self.bp.scalar_pow(natural_base,sub_value)+const_one)
                sum_value.assign(self.bp.scalar_log(sum_value) / \
                    self.bp.scalar_log(natural_base))
                sum_value.assign(sum_value + buffer [i + 1])
            with self.bp.else_scope():
                with self.bp.if_scope(sub_value < min_threshold_valu):
                    sum_value.assign(buffer[i + 1])
        return sum_value

    def calc1(self, gram_tensor, border_outputs, idx_outputs, outputs):# nram 一次还存不下一个元素
        current_core_start = self._data_man.m_current_core_start
        total_count_in_core = self._data_man.m_total_count_in_core
        calc_loop_count = self.bp.Scalar(bangpy.int32, "calc_loop_count", \
            (total_count_in_core + self.nram_process_count - 1) // self.nram_process_count)

        once_loop_start = self.bp.Scalar(bangpy.int32, "once_loop_start")

        pw = self.bp.Scalar(self.dtype, "pw", 1)

        dim_len = self.dim_len
        norm_offset = self.bp.Scalar(bangpy.int32, "norm_offset", \
            current_core_start % dim_len)

        flat_nram = self.nram_calc_buffer.reshape([self.nram_process_count, ])

        #0 : 要压缩的维度，从中间开始，且比nram还要长
        #1 : 要压缩的维度，从中间开始，比nram小
        #2 : 要压缩的维度，从头开始
        #以上记录一下，是否是从半截开始处理的，如果是，要缓存

        complete_norm_count = self.bp.Scalar(bangpy.int32, "complete_norm_count", 0)

        norm_value = logsum_calcer(self.bp, self.dtype)

        # 确认本次循环要从gram拷贝回nram的数量
        calc_size = self.bp.Scalar(bangpy.int32, "calc_size", self.nram_process_count)

        once_norm_ok = self.bp.Scalar(bangpy.int32, "once_norm_ok", 0)
        cp_data_len = self.bp.Scalar(bangpy.int32, "cp_data_len", 0)
        with self.bp.for_range(0, calc_loop_count) as i:
            once_loop_start.assign(current_core_start + self.nram_process_count * i)
            with self.bp.if_scope(i == calc_loop_count - 1):
                calc_size.assign(total_count_in_core % self.nram_process_count)
                with self.bp.if_scope(calc_size == 0):
                    calc_size.assign(self.nram_process_count)

            norm_offset.assign(once_loop_start % dim_len)
            expect_cp_len = self.bp.Scalar(bangpy.int32, "expect_cp_len", dim_len - norm_offset)

            with self.bp.if_scope(expect_cp_len > calc_size):
                expect_cp_len.assign(calc_size)
                # 一口气拷贝不完，那就尽可能多的拷贝.
                self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, \
                    once_loop_start, dim_len, self.w, expect_cp_len)
                cp_data_len.assign(cp_data_len + expect_cp_len)
                norm_value.add_buffer(flat_nram, 0, expect_cp_len)

                with self.bp.if_scope(i == calc_loop_count - 1): # 最后一个循环了
                    # 缓存一下
                    index = get_norm_index(once_loop_start + expect_cp_len, dim_len)
                    with self.bp.if_scope(once_norm_ok == 0):
                        border_outputs[self.bp.taskId * 2] = \
                            norm_value.m_value # 走到这里了，说明这个core一直在处理一个norm的中间部分
                        idx_outputs[self.bp.taskId * 2] = index
                    with self.bp.else_scope():
                        border_outputs[self.bp.taskId * 2 + 1] = norm_value.m_value
                        idx_outputs[self.bp.taskId * 2 + 1] = index

            with self.bp.else_scope():
                #这个norm可以拷贝完了
                self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, \
                    once_loop_start, dim_len, self.w, expect_cp_len)
                cp_data_len.assign(cp_data_len + expect_cp_len)

                norm_value.add_buffer(flat_nram, 0, expect_cp_len)

                # 标记一下
                once_norm_ok.assign(1)
                # 看看这个norm是不是半截
                index = get_norm_index(once_loop_start + expect_cp_len, dim_len)
                with self.bp.if_scope(cp_data_len < dim_len):
                    border_outputs[self.bp.taskId * 2] = \
                        norm_value.m_value # 走到这里了，说明这个core一直在处理一个norm的中间部分
                    idx_outputs[self.bp.taskId * 2] = index
                with self.bp.else_scope():
                    outputs[index] = norm_value.m_value # 完整的算出来了

                norm_value.reset()

                # 接下来，拷贝下一个norm
                cp_data_len.assign(calc_size - expect_cp_len)
                with self.bp.if_scope(cp_data_len > 0):
                    self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, \
                        once_loop_start + expect_cp_len, dim_len, self.w, cp_data_len)
                    #calc_result = self.calc_norm(flat_nram, 0, cp_data_len)
                    norm_value.add_buffer(flat_nram, 0, cp_data_len)
                    #norm_value_inited.assign(1)
                    with self.bp.if_scope(i == calc_loop_count - 1): # 最后一个循环了
                        # 肯定没有拷贝完
                        border_outputs[self.bp.taskId * 2 + 1] = norm_value.m_value
                        idx_outputs[self.bp.taskId * 2 + 1] = index + 1

    def calc2(self, gram_tensor, border_outputs, idx_outputs, outputs):
        current_core_start = self._data_man.m_current_core_start
        total_count_in_core = self._data_man.m_total_count_in_core
        dim_len = self.dim_len
        norm_value = self.bp.Scalar(self.dtype, "norm_value", 0.0)

        flat_nram = self.nram_calc_buffer.reshape([self.nram_process_count, ])

        # 1 先看看有没有上个norm残留的尾巴
        norm_offset = self.bp.Scalar(bangpy.int32, "norm_offset", current_core_start % dim_len)
        expect_cp_len = self.bp.Scalar(bangpy.int32, "expect_cp_len", 0)
        with self.bp.if_scope(norm_offset != 0):
            #有残留，拷贝过来
            expect_cp_len.assign(dim_len - norm_offset)
            self.copy_from_2d_tensor(self.nram_calc_buffer, 0, \
                gram_tensor, current_core_start, dim_len, self.w, expect_cp_len)
            calc_result = self.calc_norm(flat_nram, 0, expect_cp_len)
            norm_value.assign(calc_result)
            index = get_norm_index(current_core_start + expect_cp_len, dim_len)
            #保存一下
            border_outputs[self.bp.taskId * 2] = norm_value
            idx_outputs[self.bp.taskId * 2] = index

        #开始循环拷贝norm了，先计算开始位置
        norm_start_pos = self.bp.Scalar(bangpy.int32, \
            "norm_start_pos", current_core_start + expect_cp_len)

        #计算一下一个nram里最多能存多少个
        nram_norm_count = self.bp.Scalar(bangpy.int32, \
            "nram_norm_count", self.nram_process_count // dim_len)

        #计算一下，这个core能处理的norm总数是多少
        total_norm_in_core = self.bp.Scalar(bangpy.int32, "total_norm_in_core", \
            (total_count_in_core - expect_cp_len) // dim_len)

        #计算一下，要多少个循环
        calc_loop_count = self.bp.Scalar(bangpy.int32, "calc_loop_count",\
            (total_norm_in_core + nram_norm_count - 1) // nram_norm_count)

        once_loop_norm_count = self.bp.Scalar(bangpy.int32, "nram_norm_count", nram_norm_count)
        with self.bp.for_range(0, calc_loop_count) as i:
            once_loop_start = self.bp.Scalar(bangpy.int32, "once_loop_start", \
                norm_start_pos + nram_norm_count * dim_len * i)
            with self.bp.if_scope(i == calc_loop_count - 1):
                once_loop_norm_count.assign(total_norm_in_core % nram_norm_count)
                with self.bp.if_scope(once_loop_norm_count == 0):
                    once_loop_norm_count.assign(nram_norm_count)

            #这里后续要优化，目前先弄个for循环吧
            start_index = self.bp.Scalar(bangpy.int32, "norm_offset", \
                once_loop_start // dim_len) #肯定可以整除
            with self.bp.for_range(0, once_loop_norm_count) as j:
                #先拷贝过来
                self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, \
                    once_loop_start + j * dim_len, dim_len, self.w, dim_len)
                calc_result = self.calc_norm(flat_nram, 0, dim_len)
                norm_value.assign(calc_result)
                outputs[start_index + j] = norm_value       # 一个完整的norm算出来了

        #再看一下结尾，是不是要缓存下一个norm的前半截
        norm_loop_end_pos = self.bp.Scalar(bangpy.int32, "norm_loop_end_pos", \
            norm_start_pos + total_norm_in_core * dim_len)
        cur_loop_end_pos = self.bp.Scalar(bangpy.int32, "cur_loop_end_pos", \
            current_core_start + total_count_in_core)
        with self.bp.if_scope(norm_loop_end_pos < cur_loop_end_pos):
            #拷贝一下数据
            self.copy_from_2d_tensor(self.nram_calc_buffer, 0, gram_tensor, norm_loop_end_pos, \
                dim_len, self.w, cur_loop_end_pos - norm_loop_end_pos)
            calc_result = self.calc_norm(flat_nram, 0, cur_loop_end_pos - norm_loop_end_pos)
            norm_value.assign(calc_result)
            index = get_norm_index(norm_loop_end_pos + 1, dim_len) #加个1，表示跳到下一个了
            #保存一下
            border_outputs[self.bp.taskId * 2 + 1] = norm_value
            idx_outputs[self.bp.taskId * 2 + 1] = index

@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_logsumexp(dtype=None, target=None):
    task_num = 4
    f = Logsumexp(dtype, target, task_num).compute_body()
    return f
