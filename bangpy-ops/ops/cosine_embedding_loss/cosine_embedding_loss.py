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
"""cosine embedding loss for bangpy tcp"""

import bangpy
from bangpy import tcp
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu290"]
KERNEL_NAME = "cosine_embedding_loss"


class CosineEmbeddingLoss(object):
    """Operator description:
    Creates a criterion that measures the loss given input tensors x_1, x_2
    and a Tensor label yy with values 1 or -1. This is used for measuring whether
    two inputs are similar or dissimilar, using the cosine distance, and is
    typically used for learning nonlinear embeddings or semi-supervised learning.

    parameters:
    margin: Should be a number from -1 to 1, 0 to 0.5 is suggested.
    reduction: (string)暂无法处理string类型的数据

    data shape:
    input1: (N, D)
    input2: (N, D)
    target: (N),
    output: (N), if reduction, then scalar
    """

    def __init__(self, dtype, stage, target, task_type):

        # 检查参数
        if not (
            (dtype in DTYPES)
            and (stage in [0, 1])
            and (target in TARGET_LIST)
            and (task_type in TaskType.__dict__.values())
        ):
            raise KeyError

        # 设置属性初始值
        self.dtype = dtype
        self.target = target
        self.task_type = task_type
        self.tcp = tcp.TCP(target)
        self.tcp.launch_cluster(self.task_type.value)
        self.tcp.launch_task(self.task_type.value * 4, 1, 1)
        self.task_num = self.task_type.value * 4
        self.pipeline = stage

        # 数据为二维分布，(H, W)
        # length = W
        # data_num = H
        self.length = self.tcp.SizeVar("length_v")
        self.data_num_v = self.tcp.SizeVar("data_num_v")

        # 输入输出全局内存
        self.input_x1 = self.tcp.Buffer(
            shape=(self.data_num_v, self.length),
            name="input_x1",
            dtype=dtype,
            scope="global",
        )
        self.input_x2 = self.tcp.Buffer(
            shape=(self.data_num_v, self.length),
            name="input_x2",
            dtype=dtype,
            scope="global",
        )
        self.input_y = self.tcp.Buffer(
            shape=(self.data_num_v,), name="input_y", dtype=dtype, scope="global"
        )
        self.margin = self.tcp.Var(name="margin", dtype=bangpy.float32)
        self.output = self.tcp.Buffer(
            shape=(self.data_num_v,), name="output", dtype=dtype, scope="global"
        )

        # 参数的标量形式，因为bangpy中的scalar_min 不允许非标量数据作为参数
        self.length_s = self.tcp.Scalar(bangpy.int32, "length_s", value=self.length)
        self.data_num = self.tcp.Scalar(bangpy.int32, "data_num", value=self.data_num_v)

        # 数据的128bytes 对齐尺寸
        self.align_size = 128 // self.dtype.bytes
        self.max_buffer_size = 16 * 512 * 8 // self.dtype.bytes  # TODO: 确定buffer上限
        # 现只支持mlu290的nram尺寸
        # sumpool时kernel的最大尺寸
        self.max_kernel_size = self.tcp.Scalar(
            value=self.align_size, dtype=bangpy.int32, name="max_kernel_size"
        )
        # 由于bangpy暂时无法动态管理nram内存，因此现暂时假设数据宽度大于等于对齐尺寸，这样只需要声明源数据1/32或1/64尺寸的规约数据空间
        self.max_reduced_buffer_size = self.max_buffer_size // self.align_size

    def compute_body(self):
        """
        算子计算主体，返回build对象
        """
        # row指源数据的一行
        # line指nram中的一行数据
        # 将nram划分为128bytes//dtype.bytes行，每行含有固定数量的kernel数
        kernel_size = self.tcp.Scalar(bangpy.int32, name="kernel_size")
        kernels_nram = self.tcp.Scalar(bangpy.int32, name="kernels_nram")
        kernels_per_line = self.tcp.Scalar(bangpy.int32, name="kernels_per_line")
        kernels_per_row = self.tcp.Scalar(bangpy.int32, name="kernels_per_row")
        lines_per_row = self.tcp.Scalar(bangpy.int32, name="lines_per_row")
        rows_per_line = self.tcp.Scalar(bangpy.int32, name="rows_per_line")

        # batch指nram一次能处理的row数量，如果batch == 0则表示需要多一层循环处理这个row
        batch_size = self.tcp.Scalar(bangpy.int32, name="batch_size")
        # nram中存储的数据的row数量
        y_num = self.tcp.Scalar(bangpy.int32, name="y_num")

        kernel_size.assign(self.tcp.scalar_min(self.length_s, self.max_kernel_size))

        kernels_per_line.assign(self.max_buffer_size // self.align_size // kernel_size)
        kernels_nram.assign(kernels_per_line * self.align_size)
        kernels_per_row.assign((self.length + kernel_size - 1) // kernel_size)

        # 源数据一行所需内存行数
        lines_per_row.assign(
            (kernels_per_row + kernels_per_line - 1) // kernels_per_line
        )

        # 内存中一行数据能够存储的源数据行数
        rows_per_line.assign(kernels_per_line // kernels_per_row)

        # 当一行数据装不满一整个buffer时计算这一行数据需要的kernel数量
        kernels_per_line_n = self.tcp.Scalar(
            dtype=bangpy.int32, name="kernels_per_line_n"
        )
        kernels_per_line_n.assign(
            (kernels_per_row + self.align_size - 1) // self.align_size
        )

        # nram一次所能装载的数据行数
        with self.tcp.if_scope(rows_per_line > 0):
            batch_size.assign(rows_per_line * self.align_size)
        with self.tcp.else_scope():
            batch_size.assign(self.align_size // lines_per_row)

        # 一个core所需要的处理的任务数量和起止位置
        task_row_num = self.tcp.Scalar(bangpy.int32, name="cluster_row_num")
        task_row_num.assign(self.data_num // self.task_num)

        task_row_base = self.tcp.Scalar(bangpy.int32, name="task_row_base")
        with self.tcp.if_scope(
            self.data_num - task_row_num * self.task_num > self.tcp.taskId
        ):
            task_row_num.assign(task_row_num + 1)
            task_row_base.assign(task_row_num * self.tcp.taskId)
        with self.tcp.else_scope():
            task_row_base.assign(
                self.data_num - (self.task_num - self.tcp.taskId) * task_row_num
            )

        task_row_end = self.tcp.Scalar(bangpy.int32, name="task_row_end")
        task_row_end.assign(task_row_base + task_row_num)

        # 两层循环的边界条件，如果nram中无法装载下一整行数据，则需要两层循环，外层遍历行，内层在列上遍历处理单行数据
        # 如果nram可以存储一行或者多行数据，则只需要一层内层遍历，外层bound设为1
        # 这样是为了在内层循环设置流水线
        inner_loop_bound = self.tcp.Scalar(bangpy.int32, name="inner_loop_bound")
        outer_loop_bound = self.tcp.Scalar(
            bangpy.int32, name="outer_loop_bound", value=1
        )

        with self.tcp.if_scope(batch_size == 0):
            inner_loop_bound.assign(
                (kernels_per_row + kernels_nram - 1) // kernels_nram
            )
            outer_loop_bound.assign(task_row_num)
            y_num.assign(1)
        with self.tcp.else_scope():
            inner_loop_bound.assign((task_row_num + batch_size - 1) // batch_size)
            outer_loop_bound.assign(1)
            y_num.assign(batch_size)

        with self.tcp.for_range(0, outer_loop_bound) as i:
            # batch size == 0 时表示当前处理哪行数据
            row = self.tcp.Scalar(dtype=bangpy.int32, name="row")
            row.assign(task_row_base + i)

            # 声明标量
            # upper_sum = sum(x1 * x2)
            # lower1_sum = sum(x1 * x1)
            # lower2_sum = sum(x2 * x2)
            upper_sum = self.tcp.Scalar(name="upper_sum", dtype=bangpy.float32, value=0)
            lower1_sum = self.tcp.Scalar(
                name="lower1_sum", dtype=bangpy.float32, value=0
            )
            lower2_sum = self.tcp.Scalar(
                name="lower2_sum", dtype=bangpy.float32, value=0
            )

            with self.tcp.for_range(0, inner_loop_bound, stage=self.pipeline) as j:
                # 声明三个存储x1，x2以及中间结果的nram buffer, 这里声明的是展开的形状
                input_buffer_x1 = self.tcp.Buffer(
                    shape=(self.max_buffer_size,),
                    name="input_buffer_x1",
                    dtype=self.dtype,
                    scope="nram",
                )
                input_buffer_x2 = self.tcp.Buffer(
                    shape=(self.max_buffer_size,),
                    name="input_buffer_x2",
                    dtype=self.dtype,
                    scope="nram",
                )
                inter_buffer = self.tcp.Buffer(
                    shape=(self.max_buffer_size,),
                    name="inter_buffer",
                    dtype=self.dtype,
                    scope="nram",
                )

                # 声明y相关buffer, 以及规约后所需的标量，直接进行相应的截取
                input_buffer_y = self.tcp.Buffer(
                    shape=(self.max_reduced_buffer_size,),
                    name="input_buffer_y",
                    dtype=self.dtype,
                    scope="nram",
                )[
                    : (batch_size + self.align_size - 1)
                    // self.align_size
                    * self.align_size
                ]
                upper = self.tcp.Buffer(
                    shape=(self.max_reduced_buffer_size,),
                    name="upper",
                    dtype=self.dtype,
                    scope="nram",
                )[
                    : (batch_size + self.align_size - 1)
                    // self.align_size
                    * self.align_size
                ]
                lower1 = self.tcp.Buffer(
                    shape=(self.max_reduced_buffer_size,),
                    name="lower1",
                    dtype=self.dtype,
                    scope="nram",
                )[
                    : (batch_size + self.align_size - 1)
                    // self.align_size
                    * self.align_size
                ]
                lower2 = self.tcp.Buffer(
                    shape=(self.max_reduced_buffer_size,),
                    name="lower2",
                    dtype=self.dtype,
                    scope="nram",
                )[
                    : (batch_size + self.align_size - 1)
                    // self.align_size
                    * self.align_size
                ]

                # sumpool时所需的temp存储
                temp_buffer = self.tcp.Buffer(
                    shape=(self.max_buffer_size,),
                    name="temp_buffer",
                    dtype=self.dtype,
                    scope="nram",
                )

                with self.tcp.block("data_copy"):
                    # 当次内层循环所需处理的数据行起止位置
                    base = self.tcp.Scalar(name="base", dtype=bangpy.int32)
                    end = self.tcp.Scalar(name="end", dtype=bangpy.int32)

                    # nram无法装下一整行数据
                    with self.tcp.if_scope(batch_size == 0):
                        base.assign(j * kernels_nram * kernel_size)
                        end.assign(kernels_nram * kernel_size + base)
                        end.assign(self.tcp.scalar_min(end, self.length_s))
                        self.tcp.memcpy(
                            input_buffer_x1[: end - base], self.input_x1[row][base:end]
                        )
                        self.tcp.memcpy(
                            input_buffer_x2[: end - base], self.input_x2[row][base:end]
                        )

                    # nram能装下一行或多行数据，但是无法装下align_size(128bytes//dtype.bytes)
                    # 行数据，即无法直接批量进行sumpool
                    # 这里采用的方法是对数据按行分别计算，用了for循环，可寻找优化方式
                    with self.tcp.elif_scope(rows_per_line == 0):
                        base.assign(batch_size * j + task_row_base)
                        end.assign(base + batch_size)
                        end.assign(self.tcp.scalar_min(end, task_row_end))

                        self.tcp.memcpy(
                            input_buffer_y[: end - base], self.input_y[base:end]
                        )

                        cpy_in_x11 = input_buffer_x1[
                            : batch_size * self.length
                        ].reshape((batch_size, self.length))
                        cpy_in_x21 = input_buffer_x2[
                            : batch_size * self.length
                        ].reshape((batch_size, self.length))
                        self.tcp.memcpy(
                            cpy_in_x11[: end - base], self.input_x1[base:end]
                        )
                        self.tcp.memcpy(
                            cpy_in_x21[: end - base], self.input_x2[base:end]
                        )

                    # nram能装下至少align_size(128bytes//dtype.bytes)行数据
                    # 由于需要对数据进行批量sumpool，这就要求数据从(H, W)转置为(W, H)
                    # 以适应sumpool对二维数据输入要求的(H, C)排布
                    # 由于需要进行转置，bangpy的转置函数要求源和目标地址分处不同内存
                    # 因此拷入时错位拷贝，
                    # 拷贝时x1 -> x2, x2 -> inter_buffer
                    # 转置时x2 -> x1, inter_buffer -> x2
                    # 在计算时将转置后的数据放入正确的位置
                    with self.tcp.else_scope():
                        base.assign(batch_size * j + task_row_base)
                        end.assign(base + batch_size)
                        end.assign(self.tcp.scalar_min(end, task_row_end))

                        self.tcp.memcpy(
                            input_buffer_y[: end - base], self.input_y[base:end]
                        )

                        # 内存错位拷贝
                        cpy_in_x1 = input_buffer_x2[: batch_size * self.length].reshape(
                            (batch_size, self.length)
                        )
                        cpy_in_x2 = inter_buffer[: batch_size * self.length].reshape(
                            (batch_size, self.length)
                        )
                        self.tcp.memcpy(
                            cpy_in_x1[: end - base], self.input_x1[base:end]
                        )
                        self.tcp.memcpy(
                            cpy_in_x2[: end - base], self.input_x2[base:end]
                        )

                with self.tcp.block("compute"):
                    # 计算和数据拷入时一样，也依据数据尺寸不同分三种逻辑
                    # nram装不下一行数据
                    with self.tcp.if_scope(batch_size == 0):
                        inter_n = inter_buffer[: kernel_size * kernels_nram].reshape(
                            (kernels_per_line * kernel_size, self.align_size)
                        )
                        temp_n = temp_buffer[
                            : kernels_per_line * self.align_size
                        ].reshape((kernels_per_line, self.align_size))

                        # 中间数组清零，防止尾部多余数据影响求和结果
                        with self.tcp.if_scope(j <= 1):
                            self.tcp.assign(inter_buffer, 0)

                        # 求和函数，两次sumpool
                        def compute_sum_batch_0(in1, in2, out):
                            self.tcp.multiply(inter_buffer, in1, in2)
                            self.tcp.sumpool(
                                temp_n, inter_n, (kernel_size,), (kernel_size,)
                            )
                            self.tcp.sumpool(
                                inter_n[0:1],
                                temp_n[:kernels_per_line],
                                (kernels_per_line,),
                                (kernels_per_line,),
                            )
                            self.tcp.sum(temp_n[0][0], inter_n[0])
                            out.assign(out + temp_n[0][0])

                        compute_sum_batch_0(input_buffer_x1, input_buffer_x2, upper_sum)
                        compute_sum_batch_0(
                            input_buffer_x1, input_buffer_x1, lower1_sum
                        )
                        compute_sum_batch_0(
                            input_buffer_x2, input_buffer_x2, lower2_sum
                        )

                    with self.tcp.else_scope():
                        # nram能装不到align_size行数据
                        with self.tcp.if_scope(rows_per_line == 0):
                            comps_x1 = input_buffer_x1[
                                : batch_size * self.length
                            ].reshape((batch_size, self.length))
                            comps_x2 = input_buffer_x2[
                                : batch_size * self.length
                            ].reshape((batch_size, self.length))

                            comp_inter = inter_buffer[
                                : kernels_per_line_n * self.align_size * kernel_size
                            ].reshape(
                                (kernel_size * kernels_per_line_n, self.align_size,)
                            )
                            temp_n = temp_buffer[
                                : kernels_per_line_n * self.align_size
                            ].reshape((kernels_per_line_n, self.align_size))

                            with self.tcp.if_scope(j <= 1):
                                self.tcp.assign(inter_buffer, 0)

                            # 计算求和的函数
                            def compute_sum_batch_1(in1, in2, out):
                                self.tcp.multiply(inter_buffer[: self.length], in1, in2)
                                self.tcp.sumpool(
                                    temp_n[:kernels_per_line_n],
                                    comp_inter,
                                    (kernel_size,),
                                    (kernel_size,),
                                )
                                self.tcp.sumpool(
                                    comp_inter[0:1],
                                    temp_n[:kernels_per_line_n],
                                    (kernels_per_line_n,),
                                    (kernels_per_line_n,),
                                )
                                self.tcp.sum(comp_inter[0][0], comp_inter[0])
                                out.assign(comp_inter[0][0].astype(bangpy.float32))

                            # 迭代处理每行数据的求和
                            with self.tcp.for_range(0, batch_size) as k:
                                comp_x1 = comps_x1[k]
                                comp_x2 = comps_x2[k]

                                compute_sum_batch_1(comp_x1, comp_x2, upper_sum)
                                compute_sum_batch_1(comp_x1, comp_x1, lower1_sum)
                                compute_sum_batch_1(comp_x2, comp_x2, lower2_sum)

                                # 求和之后分别计算每行数据的最终结果
                                with self.tcp.if_scope(
                                    tcp.all(lower1_sum != 0, lower2_sum != 0)
                                ):
                                    lower1_sum.assign(lower1_sum * lower2_sum)
                                    lower1_sum.assign(self.tcp.scalar_sqrt(lower1_sum))
                                    upper_sum.assign(upper_sum / lower1_sum)
                                with self.tcp.else_scope():
                                    upper_sum.assign(0)

                                lower1_sum.assign(0)
                                lower2_sum.assign(upper_sum - self.margin)
                                upper[k] = (
                                    (input_buffer_y[k] + 1) * (1 - upper_sum)
                                    + (1 - input_buffer_y[k])
                                    * self.tcp.scalar_max(lower1_sum, lower2_sum)
                                ) / 2

                        # nram能装下不止align_size行数据
                        with self.tcp.else_scope():
                            comps_x1 = input_buffer_x1[
                                : self.length * batch_size
                            ].reshape((self.length, batch_size))
                            comps_x2 = input_buffer_x2[
                                : self.length * batch_size
                            ].reshape((self.length, batch_size))
                            comps_inter = inter_buffer[
                                : self.length * batch_size
                            ].reshape((self.length, batch_size))

                            # 将拷入的数据进行转置
                            self.tcp.transpose(comps_x1, cpy_in_x1)
                            self.tcp.transpose(comps_x2, cpy_in_x2)

                            temp_n = temp_buffer[
                                : kernels_per_row * batch_size
                            ].reshape((kernels_per_row, batch_size))
                            with self.tcp.if_scope(j <= 1):
                                self.tcp.assign(inter_buffer, 0)

                            def compute_sum_batch_2(in1, in2, out):
                                # 两层sumpool直接得到最终求和结果
                                self.tcp.multiply(comps_inter, in1, in2)
                                self.tcp.sumpool(
                                    temp_n[:1],
                                    comps_inter,
                                    (kernel_size,),
                                    (kernel_size,),
                                )
                                self.tcp.sumpool(
                                    out[0:batch_size].reshape((1, batch_size)),
                                    temp_n,
                                    (kernels_per_row,),
                                    (kernels_per_row,),
                                )

                            # 每行数据求和
                            compute_sum_batch_2(comps_x1, comps_x2, upper)
                            compute_sum_batch_2(comps_x1, comps_x1, lower1)
                            compute_sum_batch_2(comps_x2, comps_x2, lower2)

                            # 由于数据可以128bytes对齐，因此使用向量计算提高运行速度
                            # 使用y与1的差与和，避免if语句
                            # 原始逻辑可参照numpy检验函数实现或者前面两种情况中的标量计算
                            self.tcp.multiply(lower1, lower1, lower2)  # lower1 * lower2
                            self.tcp.sqrt(lower1, lower1)  # (lower1 * lower2) ** 0.5
                            self.tcp.maximum(lower2, lower1, 0.004)

                            # 除法这里输出内存必须是新内存，使用与源数据相同的内存会得到
                            # 错误的结果，基本是0和一个很大的数

                            # upper / (lower1 * lower2) ** 0.5
                            # upper <- upper / (lower1 * lower2) ** 0.5
                            self.tcp.divide(lower1, upper, lower2)
                            # upper / (lower1 * lower2) ** 0.5 - margin
                            self.tcp.subtract(lower2, lower1, self.margin)
                            # (1 - upper)
                            self.tcp.subtract(upper, 1, lower1)
                            # input_y + 1
                            self.tcp.add(lower1, input_buffer_y, 1)
                            # (input_y + 1) * (1 - upper)
                            self.tcp.multiply(upper, upper, lower1)
                            # 1 - input_y
                            self.tcp.subtract(input_buffer_y, 1, input_buffer_y)
                            # max(lower1 * lower2, 0)
                            self.tcp.maximum(lower1, lower2, 0)
                            # (1 - input_y) * max(lower1 * lower2, 0)
                            self.tcp.multiply(lower1, lower1, input_buffer_y)
                            self.tcp.add(upper, lower1, upper)
                            self.tcp.multiply(upper, upper, 0.5)

                with self.tcp.block("data_copy"):
                    # 对于后两种情况，单词内循环可以得到一行或者数行的结果，因此进行结果拷出
                    # 在另一份代码中尝试了使用sram进行结果缓存，但是可能是由于sync_cluster语句
                    # 插入，效果不是很理想，遂暂时直接拷出有更好的解决方案了再改
                    with self.tcp.if_scope(batch_size > 0):
                        base.assign(batch_size * j + task_row_base)
                        end.assign(base + batch_size)
                        end.assign(self.tcp.scalar_min(end, task_row_end))
                        self.tcp.memcpy(self.output[base:end], upper[: end - base])
            # 对于第一种情况，在外层循环的每次迭代能够得到一行数据的结果
            # 在这里拷出
            with self.tcp.if_scope(batch_size == 0):
                with self.tcp.if_scope(tcp.all(lower1_sum != 0, lower2_sum != 0)):
                    lower1_sum.assign(lower1_sum * lower2_sum)
                    lower1_sum.assign(self.tcp.scalar_sqrt(lower1_sum))
                    upper_sum.assign(upper_sum / lower1_sum)
                with self.tcp.else_scope():
                    upper_sum.assign(0)
                lower1_sum.assign(0)
                lower2_sum.assign(upper_sum - self.margin)
                self.output[row] = (
                    (self.input_y[row] + 1) * (1 - upper_sum)
                    + (1 - self.input_y[row])
                    * self.tcp.scalar_max(lower1_sum, lower2_sum)
                ) / 2

        return self.tcp.BuildBANG(
            inputs=[self.input_x1, self.input_x2, self.input_y, self.margin],
            outputs=[self.output],
            kernel_name=KERNEL_NAME,
        )


###################################################
# 注册
###################################################
@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_adjust_hue(dtype=None, target=None):
    stage = 1
    task_type = TaskType.UNION16
    op_mod = CosineEmbeddingLoss(dtype, stage, target, task_type).compute_body()
    return op_mod
