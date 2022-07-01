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
"""cosine embedding loss for bangpy tcp"""

import bangpy
from bangpy import tcp
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu290"]
KERNEL_NAME = "cosine_embedding_loss"
CORES_PER_CLUSTER = 4
MLU290_MAX_BUFFER_SIZE = 16 * 512 * 8
ALIGN_BYTES = 128


class CosineEmbeddingLoss(object):
    """Operator description:
    Creates a criterion that measures the loss given input tensors x_1, x_2
    and a Tensor label yy with values 1 or -1. This is used for measuring whether
    two inputs are similar or dissimilar, using the cosine distance, and is
    typically used for learning nonlinear embeddings or semi-supervised learning.

    parameters:
    margin: Should be a number from -1 to 1, 0 to 0.5 is suggested.
    reduction: (string)cannot handle string parameters at this time.

    data shape:
    input1: (N, D)
    input2: (N, D)
    target: (N),
    output: (N), if reduction, then scalar
    """

    def __init__(self, dtype, stage, target, task_type):

        # check parameters
        if not (
            (dtype in DTYPES)
            and (stage in [0, 1])
            and (target in TARGET_LIST)
            and (task_type in TaskType.__dict__.values())
        ):
            raise KeyError("please pass correct parameters.")

        # initialize attributes
        self.dtype = dtype
        self.target = target
        self.task_type = task_type
        self.tcp = tcp.TCP(target)
        self.tcp.launch_cluster(self.task_type.value)
        self.tcp.launch_task(self.task_type.value * CORES_PER_CLUSTER, 1, 1)
        self.task_num = self.task_type.value * CORES_PER_CLUSTER
        self.pipeline = stage

        # 2D data layout，(N, D)
        # length = N
        # data_num = D
        self.length = self.tcp.SizeVar("length_v")
        self.data_num_v = self.tcp.SizeVar("data_num_v")

        # global buffer for input and output.
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

        # Scalar version of SizeVars, cause scalar_min in bangpy does not
        # allow non-Scalar type parameters.
        self.length_s = self.tcp.Scalar(bangpy.int32, "length_s", value=self.length)
        self.data_num = self.tcp.Scalar(bangpy.int32, "data_num", value=self.data_num_v)

        # 128 bytes aligned size of data
        self.align_size = ALIGN_BYTES // self.dtype.bytes
        # upper bound of buffer, need to be modified when using devices other than MLU-290
        self.max_buffer_size = MLU290_MAX_BUFFER_SIZE // self.dtype.bytes

        self.max_kernel_size = self.tcp.Scalar(
            value=self.align_size, dtype=bangpy.int32, name="max_kernel_size"
        )
        # According to align constraints of bang platform
        self.max_reduced_buffer_size = self.max_buffer_size // self.align_size

    def compute_body(self):
        """
        compute body of operator, returns bangpy build module
        """
        # Row is D size of source data (N, D)
        # Line is a line of nram
        # nram buffers are divided into 128bytes//dtype.bytes lines,
        # and each line has fixed number of kernels
        kernel_size = self.tcp.Scalar(bangpy.int32, name="kernel_size")
        kernels_nram = self.tcp.Scalar(bangpy.int32, name="kernels_nram")
        kernels_per_line = self.tcp.Scalar(bangpy.int32, name="kernels_per_line")
        kernels_per_row = self.tcp.Scalar(bangpy.int32, name="kernels_per_row")
        lines_per_row = self.tcp.Scalar(bangpy.int32, name="lines_per_row")
        rows_per_line = self.tcp.Scalar(bangpy.int32, name="rows_per_line")

        # batch means the number of rows can be handled by nram in one memcpy
        # we need one more layer of iteration if batch == 0 to handle one row of data
        batch_size = self.tcp.Scalar(bangpy.int32, name="batch_size")
        # row number can be stored in nram
        y_num = self.tcp.Scalar(bangpy.int32, name="y_num")

        kernel_size.assign(self.max_kernel_size)

        kernels_per_line.assign(self.max_buffer_size // self.align_size // kernel_size)
        kernels_nram.assign(kernels_per_line * self.align_size)
        kernels_per_row.assign((self.length + kernel_size - 1) // kernel_size)

        # nram lines needed by one row of source data
        lines_per_row.assign(
            (kernels_per_row + kernels_per_line - 1) // kernels_per_line
        )

        # number of source data rows can be sored in one line of nram
        rows_per_line.assign(kernels_per_line // kernels_per_row)

        # compute numbers of kernels needed by one row of source data.
        kernels_per_line_n = self.tcp.Scalar(
            dtype=bangpy.int32, name="kernels_per_line_n"
        )
        kernels_per_line_n.assign(
            (kernels_per_row + self.align_size - 1) // self.align_size
        )

        # number of rows of source data can be stored in nram
        with self.tcp.if_scope(rows_per_line > 0):
            batch_size.assign(rows_per_line * self.align_size)
        with self.tcp.else_scope():
            batch_size.assign(self.align_size // lines_per_row)

        # batch num assigned to one IPU core
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

        # loop bounds. Need two layers of loop if nram cannot store one single data row.
        # if nram can store at least one row of data then one layer of loop will be needed.
        # use this structure to use pipeline at inner loop layer
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
            # row id when batch size == 0
            row = self.tcp.Scalar(dtype=bangpy.int32, name="row")
            row.assign(task_row_base + i)

            # Scalars
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
                # nram buffers
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

                # buffers related to input_y.
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

                # temp memory needed by sumpool
                temp_buffer = self.tcp.Buffer(
                    shape=(self.max_buffer_size,),
                    name="temp_buffer",
                    dtype=self.dtype,
                    scope="nram",
                )

                with self.tcp.block("compute"):
                    # initialize buffer
                    self.tcp.assign(input_buffer_x1, 0.0)
                    self.tcp.assign(input_buffer_x2, 0.0)
                    self.tcp.assign(inter_buffer, 0.0)

                with self.tcp.block("data_copy"):
                    base = self.tcp.Scalar(name="base", dtype=bangpy.int32)
                    end = self.tcp.Scalar(name="end", dtype=bangpy.int32)

                    # nram cannot store one single data row
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

                    # nram can store at least one row of data,
                    # but less than align_size(128bytes//dtype.bytes) rows,
                    # which means we cannot use sumpool to compute all the sum at one time.
                    with self.tcp.elif_scope(rows_per_line == 0):
                        base.assign(batch_size * j + task_row_base)
                        end.assign(base + batch_size)
                        end.assign(self.tcp.scalar_min(end, task_row_end))

                        self.tcp.memcpy(
                            input_buffer_y[: end - base], self.input_y[base:end]
                        )

                        cpy_in_x11 = input_buffer_x1[
                            : batch_size * kernel_size * kernels_per_row
                        ].reshape((batch_size, kernel_size * kernels_per_row))
                        cpy_in_x21 = input_buffer_x2[
                            : batch_size * kernel_size * kernels_per_row
                        ].reshape((batch_size, kernel_size * kernels_per_row))
                        self.tcp.memcpy(
                            cpy_in_x11[: end - base, : self.length],
                            self.input_x1[base:end],
                        )
                        self.tcp.memcpy(
                            cpy_in_x21[: end - base, : self.length],
                            self.input_x2[base:end],
                        )

                    # nram can store at least align_size(128bytes//dtype.bytes) rows of data
                    # but in order to sumpool at data, we need to transpose the data from
                    # (N, D) to (D, N)
                    # moreover, transpose function in pangpy requires that addresses
                    # of source and destination buffer are different
                    # so when memcpy:
                    # x1 -> x2, x2 -> inter_buffer
                    # when transpose:
                    # x2 -> x1, inter_buffer -> x2
                    with self.tcp.else_scope():
                        base.assign(batch_size * j + task_row_base)
                        end.assign(base + batch_size)
                        end.assign(self.tcp.scalar_min(end, task_row_end))

                        self.tcp.memcpy(
                            input_buffer_y[: end - base], self.input_y[base:end]
                        )

                        cpy_in_x1 = input_buffer_x2[
                            : batch_size * kernel_size * kernels_per_row
                        ].reshape((batch_size, kernel_size * kernels_per_row))
                        cpy_in_x2 = inter_buffer[
                            : batch_size * kernel_size * kernels_per_row
                        ].reshape((batch_size, kernel_size * kernels_per_row))
                        # use different type of memcpy according to whether source
                        # data is aligned
                        with self.tcp.if_scope(
                            kernel_size * kernels_per_row != self.length
                        ):
                            self.tcp.memcpy(
                                cpy_in_x1[: end - base, : self.length],
                                self.input_x1[base:end],
                            )
                            self.tcp.memcpy(
                                cpy_in_x2[: end - base, : self.length],
                                self.input_x2[base:end],
                            )
                        with self.tcp.else_scope():
                            self.tcp.memcpy(
                                input_buffer_x2[: self.length * batch_size].reshape(
                                    (batch_size, self.length)
                                )[: end - base],
                                self.input_x1[base:end],
                            )
                            self.tcp.memcpy(
                                inter_buffer[: self.length * batch_size].reshape(
                                    (batch_size, self.length)
                                )[: end - base],
                                self.input_x2[base:end],
                            )

                with self.tcp.block("compute"):
                    # similar to data_copy block
                    # nram cannot store one single row
                    with self.tcp.if_scope(batch_size == 0):
                        inter_n = inter_buffer[: kernel_size * kernels_nram].reshape(
                            (kernels_per_line * kernel_size, self.align_size)
                        )
                        temp_n = temp_buffer[
                            : kernels_per_line * self.align_size
                        ].reshape((kernels_per_line, self.align_size))

                        # initialize inter_buffer
                        with self.tcp.if_scope(j <= 1):
                            self.tcp.assign(inter_buffer, 0)

                        # sum function
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
                        # nram store at least one row but less than one aligned data
                        with self.tcp.if_scope(rows_per_line == 0):
                            comps_x1 = input_buffer_x1[
                                : batch_size * kernel_size * kernels_per_row
                            ].reshape((batch_size, kernel_size * kernels_per_row))
                            comps_x2 = input_buffer_x2[
                                : batch_size * kernel_size * kernels_per_row
                            ].reshape((batch_size, kernel_size * kernels_per_row))

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

                            # sum function
                            def compute_sum_batch_1(in1, in2, out):
                                self.tcp.multiply(
                                    inter_buffer[: kernel_size * kernels_per_row],
                                    in1,
                                    in2,
                                )
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

                            # use for loop to compute result of each row
                            with self.tcp.for_range(0, batch_size) as k:
                                comp_x1 = comps_x1[k]
                                comp_x2 = comps_x2[k]

                                compute_sum_batch_1(comp_x1, comp_x2, upper_sum)
                                compute_sum_batch_1(comp_x1, comp_x1, lower1_sum)
                                compute_sum_batch_1(comp_x2, comp_x2, lower2_sum)

                                # compute final result
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

                        # nram can store more than align_size rows of data
                        with self.tcp.else_scope():
                            comps_x1 = input_buffer_x1[
                                : kernel_size * kernels_per_row * batch_size
                            ].reshape((kernel_size * kernels_per_row, batch_size))
                            comps_x2 = input_buffer_x2[
                                : kernel_size * kernels_per_row * batch_size
                            ].reshape((kernel_size * kernels_per_row, batch_size))
                            comps_inter = inter_buffer[
                                : kernel_size * kernels_per_row * batch_size
                            ].reshape((kernel_size * kernels_per_row, batch_size))

                            # transpose
                            self.tcp.transpose(comps_x1, cpy_in_x1)
                            self.tcp.transpose(comps_x2, cpy_in_x2)

                            temp_n = temp_buffer[
                                : kernels_per_row * batch_size
                            ].reshape((kernels_per_row, batch_size))
                            with self.tcp.if_scope(j <= 1):
                                self.tcp.assign(inter_buffer, 0)

                            def compute_sum_batch_2(in1, in2, out):
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

                            compute_sum_batch_2(comps_x1, comps_x2, upper)
                            compute_sum_batch_2(comps_x1, comps_x1, lower1)
                            compute_sum_batch_2(comps_x2, comps_x2, lower2)

                            # transform the original scalar compute into vector compute
                            self.tcp.multiply(lower1, lower1, lower2)  # lower1 * lower2
                            self.tcp.sqrt(lower1, lower1)  # (lower1 * lower2) ** 0.5
                            self.tcp.maximum(lower2, lower1, 0.004)

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
                    # memcpy of output for the last two cases
                    with self.tcp.if_scope(batch_size > 0):
                        base.assign(batch_size * j + task_row_base)
                        end.assign(base + batch_size)
                        end.assign(self.tcp.scalar_min(end, task_row_end))
                        self.tcp.memcpy(self.output[base:end], upper[: end - base])
            # for the first case, transform output data here
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
# register
###################################################
@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_adjust_hue(dtype=None, target=None):
    stage = 1
    task_type = TaskType.UNION16
    op_mod = CosineEmbeddingLoss(dtype, stage, target, task_type).compute_body()
    return op_mod
