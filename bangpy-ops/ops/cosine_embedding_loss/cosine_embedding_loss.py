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
"""cosine embedding loss for bangpy tcp script"""
import bangpy
from bangpy.common.dtypes import DType
from bangpy.script import tcp, build_module, ty

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu290", "mlu270", "mlu370-s4", "mlu370-m8"]
KERNEL_NAME = "cosine_embedding_loss"
CORES_PER_CLUSTER = 4


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

    def __init__(self, dtype: ty.string, pipeline: ty.boolean) -> None:
        # Initialize attributes.
        self.dtype = dtype
        self.dtype_size = DType(dtype).bytes
        self.pipeline = pipeline

    # Sum function.
    def compute_sum_batch_0(
        self,
        in1: ty.Buffer,
        in2: ty.Buffer,
        inter_n: ty.Buffer,
        inter_n_fp: ty.Buffer,
        temp_n: ty.Buffer,
        out_n: ty.float32,
        kernel_size: ty.int32,
        kernels_per_line: ty.int32,
    ) -> ty.float32:
        """compute_sum_batch_0"""
        tcp.multiply(self.inter_buffer, in1, in2)
        if self.dtype == "float16":
            tcp.type_convert(inter_n_fp, inter_n, 0)
        tcp.sumpool(temp_n, inter_n_fp, (kernel_size,), (kernel_size,))
        tcp.sumpool(
            inter_n_fp[0:1],
            temp_n[:kernels_per_line],
            (kernels_per_line,),
            (kernels_per_line,),
        )
        tcp.sum(temp_n[0][0], inter_n_fp[0])
        var = tcp.cast(temp_n[0][0], "float32") + tcp.cast(out_n, "float32")
        return var

    def compute_sum_batch_1(
        self,
        in1: ty.Buffer,
        in2: ty.Buffer,
        comp_inter: ty.Buffer,
        comp_inter_fp: ty.Buffer,
        temp_n: ty.Buffer,
        kernels_per_row: ty.int32,
        kernels_per_line_n: ty.int32,
        kernel_size: ty.int32,
    ) -> ty.float32:
        """compute_sum_batch_1"""
        tcp.multiply(
            self.inter_buffer[: kernel_size * kernels_per_row], in1, in2,
        )
        if self.dtype == "float16":
            tcp.type_convert(comp_inter_fp, comp_inter, 0)
        tcp.sumpool(
            temp_n[:kernels_per_line_n], comp_inter_fp, (kernel_size,), (kernel_size,),
        )
        tcp.sumpool(
            comp_inter_fp[0:1],
            temp_n[:kernels_per_line_n],
            (kernels_per_line_n,),
            (kernels_per_line_n,),
        )
        tcp.sum(comp_inter_fp[0][0], comp_inter_fp[0])
        return tcp.cast(comp_inter_fp[0][0], "float32")

    def compute_sum_batch_2(
        self,
        in1: ty.Buffer,
        in2: ty.Buffer,
        out: ty.Buffer,
        temp_n: ty.Buffer,
        comps_inter: ty.Buffer,
        comps_inter_fp: ty.Buffer,
        kernel_size: ty.int32,
        batch_size: ty.int32,
        kernels_per_row: ty.int32,
    ) -> None:
        """compute_sum_batch_2"""
        tcp.multiply(comps_inter, in1, in2)
        if self.dtype == "float16":
            tcp.type_convert(comps_inter_fp, comps_inter, 0)
        tcp.sumpool(
            temp_n[0:1], comps_inter_fp, (kernel_size,), (kernel_size,),
        )
        tcp.sumpool(
            out[0:batch_size].reshape((1, batch_size)),
            temp_n,
            (kernels_per_row,),
            (kernels_per_row,),
        )

    def compute_final_result(
        self,
        v_0: ty.Buffer,
        v_1: ty.Buffer,
        v_2: ty.Buffer,
        v_3: ty.Buffer,
        tmp: ty.Buffer,
        margin: ty.float32,
    ) -> None:
        """
        Transform the original scalar compute into vector compute.
        """
        tcp.maximum(v_1, v_1, 0.004)
        tcp.maximum(v_2, v_2, 0.004)
        tcp.reciprocal(v_1, v_1)
        tcp.multiply(v_1, v_0, v_1)
        tcp.reciprocal(v_2, v_2)
        tcp.multiply(v_2, v_0, v_2)
        tcp.multiply(v_1, v_1, v_2)
        tcp.sqrt(v_1, v_1)  # (v_0/v_1 * v_0/v_2) ** 0.5
        tcp.greater(tmp, v_0, 0.0)
        tcp.add(tmp, tmp, tmp)
        tcp.subtract(tmp, tmp, 1.0)
        tcp.multiply(v_1, v_1, tmp)  # v1 <- v_1 * v_0/abs(v_0)
        # v_0 / (v_1 * v_2) ** 0.5 - margin
        tcp.subtract(v_2, v_1, margin)
        # (1 - v_0)
        tcp.assign(tmp, 1)
        tcp.subtract(v_0, tmp, v_1)
        # v_3 + 1
        tcp.equal(v_1, v_3, 1.0)
        # tcp.add(v_1, v_3, 1)
        # (v_3 + 1) * (1 - v_0)
        tcp.multiply(v_0, v_0, v_1)
        # 1 - v_3
        # max(v_1 * v_2, 0)

        tcp.maximum(v_1, v_2, 0)
        tcp.equal(v_2, v_3, -1.0)
        # (1 - v_3) * max(v_1 * v_2, 0)
        tcp.multiply(v_1, v_1, v_3)
        tcp.add(v_0, v_1, v_0)
        # tcp.multiply(v_0, v_0, 0.5)

    def main(
        self,
        x1: ty.handle,
        x2: ty.handle,
        y: ty.handle,
        margin: ty.float32,
        out: ty.handle,
        data_num: ty.int32,
        length: ty.int32,
    ) -> None:
        """
        Compute body of operator, returns bangpy build module.
        """
        # 2D data layout，(N, D)
        # data_num = N
        # length = D
        # Global buffer for input and output.
        input_x1 = tcp.match_buffer(x1, [data_num, length], dtype=self.dtype)
        input_x2 = tcp.match_buffer(x2, [data_num, length], dtype=self.dtype)
        input_y = tcp.match_buffer(y, [data_num], dtype=self.dtype)
        output = tcp.match_buffer(out, [data_num], dtype=self.dtype)

        tgt = tcp.target()
        task_num = tgt.cluster_num * tgt.core_num
        # 128 bytes aligned size of data
        ALIGN_BYTES = 128
        align_size = ALIGN_BYTES // self.dtype_size
        # Upper bound of buffer, need to be modified when using devices other than MLU-290.
        max_buffer_size = tgt.nram_size // 8 // self.dtype_size

        # Row is D size of source data (N, D)
        # Line is a line of nram
        # Nram buffers are divided into 128bytes//dtype.bytes lines,
        # and each line has fixed number of kernels
        kernel_size = align_size
        kernels_per_line = max_buffer_size // align_size // kernel_size
        kernels_nram = kernels_per_line * align_size
        kernels_per_row = (length + kernel_size - 1) // kernel_size
        # Nram lines needed by one row of source data.
        lines_per_row = (kernels_per_row + kernels_per_line - 1) // kernels_per_line
        # Number of source data rows can be sored in one line of nram.
        rows_per_line = kernels_per_line // kernels_per_row
        # Compute numbers of kernels needed by one row of source data.
        kernels_per_line_n = (kernels_per_row + align_size - 1) // align_size

        # Batch means the number of rows can be handled by nram in one memcpy.
        # We need one more layer of iteration if batch == 0 to handle one row of data.
        batch_size = align_size // lines_per_row
        # Number of rows of source data can be stored in nram.
        if rows_per_line > 0:
            batch_size = rows_per_line * align_size

        # According to align constraints of bang platform.
        max_reduced_buffer_size = max_buffer_size // align_size

        for cluster_id in tcp.thread_binding(0, tgt.cluster_num, thread="blockIdx.x"):
            for core_id in tcp.thread_binding(0, tgt.core_num, thread="threadIdx.x"):
                task_id = cluster_id * tgt.core_num + core_id
                # Batch num assigned to one MLU Core.
                task_row_num = data_num // task_num
                task_row_base = data_num - (task_num - task_id) * task_row_num
                if data_num - task_row_num * task_num > task_id:
                    task_row_num = task_row_num + 1
                    task_row_base = task_row_num * task_id
                task_row_end = task_row_base + task_row_num

                # Loop bounds. Need two layers of loop if nram cannot store one single data row.
                # If nram can store at least one row of data then one layer of loop will be needed.
                # Use this structure to use pipeline at inner loop layer
                inner_loop_bound = (task_row_num + batch_size - 1) // batch_size
                outer_loop_bound = 1

                if batch_size == 0:
                    inner_loop_bound = (
                        kernels_per_row + kernels_nram - 1
                    ) // kernels_nram
                    outer_loop_bound = task_row_num

                for i in range(outer_loop_bound):
                    # Row id when batch size == 0
                    row = task_row_base + i

                    # Scalars
                    # upper_sum = sum(x1 * x2)
                    # lower1_sum = sum(x1 * x1)
                    # lower2_sum = sum(x2 * x2)
                    upper_sum = 0.0
                    lower1_sum = 0.0
                    lower2_sum = 0.0
                    for j in range(inner_loop_bound, pipeline=self.pipeline):
                        # Nram buffers.
                        input_buffer_x1 = tcp.alloc_buffer(
                            [max_buffer_size], dtype=self.dtype, scope="nram"
                        )
                        input_buffer_x2 = tcp.alloc_buffer(
                            [max_buffer_size], dtype=self.dtype, scope="nram"
                        )
                        self.inter_buffer = tcp.alloc_buffer(
                            [max_buffer_size], dtype=self.dtype, scope="nram"
                        )
                        self.inter_buffer_fp = tcp.alloc_buffer(
                            [max_buffer_size], dtype="float32", scope="nram"
                        )
                        # Temp memory needed by sumpool.
                        temp_buffer = tcp.alloc_buffer(
                            [max_buffer_size], dtype="float32", scope="nram"
                        )
                        input_buffer_y = tcp.alloc_buffer(
                            [max_reduced_buffer_size], dtype=self.dtype, scope="nram",
                        )
                        input_buffer_y = input_buffer_y[
                            : (batch_size + align_size - 1) // align_size * align_size
                        ]
                        upper = tcp.alloc_buffer(
                            [max_reduced_buffer_size], dtype="float32", scope="nram",
                        )
                        upper = upper[
                            : (batch_size + align_size - 1) // align_size * align_size
                        ]
                        upper_fp = tcp.alloc_buffer(
                            [max_reduced_buffer_size], dtype=self.dtype, scope="nram",
                        )
                        upper_fp = upper_fp[
                            : (batch_size + align_size - 1) // align_size * align_size
                        ]
                        lower1 = tcp.alloc_buffer(
                            [max_reduced_buffer_size], dtype="float32", scope="nram",
                        )
                        lower1 = lower1[
                            : (batch_size + align_size - 1) // align_size * align_size
                        ]
                        lower2 = tcp.alloc_buffer(
                            [max_reduced_buffer_size], dtype="float32", scope="nram",
                        )
                        lower2 = lower2[
                            : (batch_size + align_size - 1) // align_size * align_size
                        ]
                        cpy_in_x1 = input_buffer_x2[
                            : batch_size * kernel_size * kernels_per_row
                        ].reshape((batch_size, kernel_size * kernels_per_row))
                        cpy_in_x2 = self.inter_buffer[
                            : batch_size * kernel_size * kernels_per_row
                        ].reshape((batch_size, kernel_size * kernels_per_row))
                        temp_0 = tcp.alloc_buffer(
                            [max_reduced_buffer_size], dtype="float32", scope="nram",
                        )
                        temp_0 = temp_0[
                            : (batch_size + align_size - 1) // align_size * align_size
                        ]

                        with tcp.block("compute"):
                            # Initialize buffer.
                            tcp.assign(input_buffer_x1, 0.0)
                            tcp.assign(input_buffer_x2, 0.0)
                            tcp.assign(self.inter_buffer, 0.0)

                        with tcp.block("data_copy"):
                            # Nram cannot, store one single data row.
                            if batch_size == 0:
                                base = j * kernels_nram * kernel_size
                                end = kernels_nram * kernel_size + base
                                end = tcp.min(end, length)
                                tcp.memcpy(
                                    input_buffer_x1[: end - base],
                                    input_x1[row][base:end],
                                )
                                tcp.memcpy(
                                    input_buffer_x2[: end - base],
                                    input_x2[row][base:end],
                                )

                            # Nram can store at least one row of data,
                            # but less than align_size(128bytes//dtype.bytes) rows,
                            # which means we cannot use sumpool to compute all the sum at one time.
                            elif rows_per_line == 0:
                                base = batch_size * j + task_row_base
                                end = base + batch_size
                                end = tcp.min(end, task_row_end)

                                tcp.memcpy(
                                    input_buffer_y[: end - base], input_y[base:end]
                                )
                                cpy_in_x11 = input_buffer_x1[
                                    : batch_size * kernel_size * kernels_per_row
                                ].reshape((batch_size, kernel_size * kernels_per_row))
                                cpy_in_x21 = input_buffer_x2[
                                    : batch_size * kernel_size * kernels_per_row
                                ].reshape((batch_size, kernel_size * kernels_per_row))
                                tcp.memcpy(
                                    cpy_in_x11[: end - base, :length],
                                    input_x1[base:end],
                                )
                                tcp.memcpy(
                                    cpy_in_x21[: end - base, :length],
                                    input_x2[base:end],
                                )

                            # Nram can store at least align_size(128bytes//dtype.bytes) rows of data
                            # but in order to sumpool at data, we need to transpose the data from
                            # (N, D) to (D, N).
                            # Moreover, transpose function in pangpy requires that addresses
                            # of source and destination buffer are different,
                            # so when memcpy:
                            # x1 -> x2, x2 -> inter_buffer
                            # when transpose:
                            # x2 -> x1, inter_buffer -> x2
                            else:
                                base = batch_size * j + task_row_base
                                end = base + batch_size
                                end = tcp.min(end, task_row_end)

                                tcp.memcpy(
                                    input_buffer_y[: end - base], input_y[base:end]
                                )

                                # Use different type of memcpy according to whether source
                                # data is aligned.
                                if kernel_size * kernels_per_row != length:
                                    tcp.memcpy(
                                        cpy_in_x1[: end - base, :length],
                                        input_x1[base:end],
                                    )
                                    tcp.memcpy(
                                        cpy_in_x2[: end - base, :length],
                                        input_x2[base:end],
                                    )
                                else:
                                    tcp.memcpy(
                                        input_buffer_x2[: length * batch_size].reshape(
                                            (batch_size, length)
                                        )[: end - base],
                                        input_x1[base:end],
                                    )
                                    tcp.memcpy(
                                        self.inter_buffer[
                                            : length * batch_size
                                        ].reshape((batch_size, length))[: end - base],
                                        input_x2[base:end],
                                    )

                        with tcp.block("compute"):
                            # Similar to data_copy block
                            # Nram cannot store one single row.
                            if batch_size == 0:
                                # Initialize inter_buffer.
                                inter_n = self.inter_buffer[
                                    : kernel_size * kernels_nram
                                ].reshape((kernels_per_line * kernel_size, align_size))
                                inter_n_fp = self.inter_buffer_fp[
                                    : kernel_size * kernels_nram
                                ].reshape((kernels_per_line * kernel_size, align_size))
                                temp_n = temp_buffer[
                                    : kernels_per_line * align_size
                                ].reshape((kernels_per_line, align_size))

                                if j <= 1:
                                    tcp.assign(self.inter_buffer, 0)

                                upper_sum = self.compute_sum_batch_0(
                                    input_buffer_x1,
                                    input_buffer_x2,
                                    inter_n,
                                    inter_n_fp,
                                    temp_n,
                                    upper_sum,
                                    kernel_size,
                                    kernels_per_line,
                                )
                                lower1_sum = self.compute_sum_batch_0(
                                    input_buffer_x1,
                                    input_buffer_x1,
                                    inter_n,
                                    inter_n_fp,
                                    temp_n,
                                    lower1_sum,
                                    kernel_size,
                                    kernels_per_line,
                                )
                                lower2_sum = self.compute_sum_batch_0(
                                    input_buffer_x2,
                                    input_buffer_x2,
                                    inter_n,
                                    inter_n_fp,
                                    temp_n,
                                    lower2_sum,
                                    kernel_size,
                                    kernels_per_line,
                                )

                            else:
                                # Nram store at least one row but less than one aligned data.
                                if rows_per_line == 0:
                                    comps_x1 = input_buffer_x1[
                                        : batch_size * kernel_size * kernels_per_row
                                    ].reshape(
                                        (batch_size, kernel_size * kernels_per_row)
                                    )
                                    comps_x2 = input_buffer_x2[
                                        : batch_size * kernel_size * kernels_per_row
                                    ].reshape(
                                        (batch_size, kernel_size * kernels_per_row)
                                    )

                                    comp_inter = self.inter_buffer[
                                        : kernels_per_line_n * align_size * kernel_size
                                    ].reshape(
                                        (kernel_size * kernels_per_line_n, align_size,)
                                    )
                                    comp_inter_fp = self.inter_buffer_fp[
                                        : kernels_per_line_n * align_size * kernel_size
                                    ].reshape(
                                        (kernel_size * kernels_per_line_n, align_size,)
                                    )
                                    temp_n = temp_buffer[
                                        : kernels_per_line_n * align_size
                                    ].reshape((kernels_per_line_n, align_size))

                                    if j <= 1:
                                        tcp.assign(self.inter_buffer, 0)

                                    # Use for loop to compute result of each row.
                                    for k in range(batch_size):
                                        comp_x1 = comps_x1[k]
                                        comp_x2 = comps_x2[k]

                                        upper_sum = self.compute_sum_batch_1(
                                            comp_x1,
                                            comp_x2,
                                            comp_inter,
                                            comp_inter_fp,
                                            temp_n,
                                            kernels_per_row,
                                            kernels_per_line_n,
                                            kernel_size,
                                        )
                                        lower1_sum = self.compute_sum_batch_1(
                                            comp_x1,
                                            comp_x1,
                                            comp_inter,
                                            comp_inter_fp,
                                            temp_n,
                                            kernels_per_row,
                                            kernels_per_line_n,
                                            kernel_size,
                                        )
                                        lower2_sum = self.compute_sum_batch_1(
                                            comp_x2,
                                            comp_x2,
                                            comp_inter,
                                            comp_inter_fp,
                                            temp_n,
                                            kernels_per_row,
                                            kernels_per_line_n,
                                            kernel_size,
                                        )

                                        # Compute final result.
                                        if lower1_sum != 0.0 and lower2_sum != 0.0:
                                            lower1_sum = lower1_sum * lower2_sum
                                            lower1_sum = tcp.scalar_sqrt(lower1_sum)
                                            lower1_sum = 1.0 / lower1_sum
                                            upper_sum = upper_sum * lower1_sum
                                        else:
                                            upper_sum = 0.0

                                        lower1_sum = 0.0
                                        lower2_sum = upper_sum - margin
                                        if input_y[k] == 1.0:
                                            upper[k] = 1 - upper_sum
                                        elif input_y[k] == -1.0:
                                            upper[k] = tcp.max(lower1_sum, lower2_sum)
                                        else:
                                            upper[k] = 0.0
                                # Nram can store more than align_size rows of data.
                                else:
                                    # arch below mlu3xx need 64 byte align
                                    batch_size = batch_size // align_size * align_size
                                    comps_x1 = input_buffer_x1[
                                        : kernel_size * kernels_per_row * batch_size
                                    ].reshape(
                                        (kernel_size * kernels_per_row, batch_size)
                                    )
                                    comps_x2 = input_buffer_x2[
                                        : kernel_size * kernels_per_row * batch_size
                                    ].reshape(
                                        (kernel_size * kernels_per_row, batch_size)
                                    )
                                    comps_inter = self.inter_buffer[
                                        : kernel_size * kernels_per_row * batch_size
                                    ].reshape(
                                        (kernel_size * kernels_per_row, batch_size)
                                    )
                                    comps_inter_fp = self.inter_buffer_fp[
                                        : kernel_size * kernels_per_row * batch_size
                                    ].reshape(
                                        (kernel_size * kernels_per_row, batch_size)
                                    )
                                    # Transpose.
                                    tcp.transpose(comps_x1, cpy_in_x1)
                                    tcp.transpose(comps_x2, cpy_in_x2)

                                    temp_n = temp_buffer[
                                        : kernels_per_row * batch_size
                                    ].reshape((kernels_per_row, batch_size))
                                    if j <= 1:
                                        tcp.assign(self.inter_buffer, 0)

                                    self.compute_sum_batch_2(
                                        comps_x1,
                                        comps_x2,
                                        upper,
                                        temp_n,
                                        comps_inter,
                                        comps_inter_fp,
                                        kernel_size,
                                        batch_size,
                                        kernels_per_row,
                                    )
                                    self.compute_sum_batch_2(
                                        comps_x1,
                                        comps_x1,
                                        lower1,
                                        temp_n,
                                        comps_inter,
                                        comps_inter_fp,
                                        kernel_size,
                                        batch_size,
                                        kernels_per_row,
                                    )
                                    self.compute_sum_batch_2(
                                        comps_x2,
                                        comps_x2,
                                        lower2,
                                        temp_n,
                                        comps_inter,
                                        comps_inter_fp,
                                        kernel_size,
                                        batch_size,
                                        kernels_per_row,
                                    )

                                    if self.dtype == "float16":
                                        tcp.type_convert(
                                            temp_0, input_buffer_y, 0, "rd"
                                        )
                                        self.compute_final_result(
                                            upper,
                                            lower1,
                                            lower2,
                                            temp_0,
                                            temp_buffer[:max_reduced_buffer_size][
                                                : (batch_size + align_size - 1)
                                                // align_size
                                                * align_size
                                            ],
                                            margin,
                                        )
                                    else:
                                        self.compute_final_result(
                                            upper,
                                            lower1,
                                            lower2,
                                            input_buffer_y,
                                            temp_buffer[:max_reduced_buffer_size][
                                                : (batch_size + align_size - 1)
                                                // align_size
                                                * align_size
                                            ],
                                            margin,
                                        )
                        with tcp.block("data_copy"):
                            # Memcpy of output for the last two cases.
                            if batch_size > 0:
                                base = batch_size * j + task_row_base
                                end = base + batch_size
                                end = tcp.min(end, task_row_end)
                                if self.dtype == "float16":
                                    tcp.type_convert(
                                        upper_fp, upper, 0, "rd"
                                    )  # fix later: optional parameter can't ignore
                                    tcp.memcpy(output[base:end], upper_fp[: end - base])
                                else:
                                    tcp.memcpy(output[base:end], upper[: end - base])

                    # For the first case, transform output data here.
                    if batch_size == 0:
                        if lower1_sum != 0.0 and lower2_sum != 0.0:
                            lower1_sum = lower1_sum * lower2_sum
                            lower1_sum = tcp.scalar_sqrt(lower1_sum)
                            lower1_sum = 1.0 / lower1_sum
                            upper_sum = upper_sum * lower1_sum
                        else:
                            upper_sum = 0.0
                        lower1_sum = 0.0
                        lower2_sum = upper_sum - margin
                        if self.dtype == "float16":
                            if input_y[row] == 1.0:
                                output[row] = tcp.cast(1 - upper_sum, "float16")
                            elif input_y[row] == -1.0:
                                output[row] = tcp.cast(tcp.max(lower1_sum, lower2_sum), "float16")
                            else:
                                output[row] = tcp.cast(0.0, "float16")
                        else:
                            if input_y[row] == 1.0:
                                output[row] = 1 - upper_sum
                            elif input_y[row] == -1.0:
                                output[row] = tcp.max(lower1_sum, lower2_sum)
                            else:
                                output[row] = 0.0


###################################################
# Register
###################################################
@bangpy.tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_cosine_embedding_loss(dtype=None, target=None):
    pipeline = True
    op_mod = build_module.build(
        CosineEmbeddingLoss(dtype.name, pipeline),
        target_tag=target,
        name=KERNEL_NAME,
    )
    return op_mod
