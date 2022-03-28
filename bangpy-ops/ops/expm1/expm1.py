import numpy as np

import bangpy
from bangpy import tcp
from bangpy.tcp.util import round_down
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]
KERNEL_NAME = "expm1"
ALIGN_SIZE = 64


class Expm1(object):
    def __init__(self, dtype, target, task_num):
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.bp = tcp.TCP(target)
        self.dim_n = self.bp.SizeVar("dim_n")
        self.dim_h = self.bp.SizeVar("dim_h")
        self.dim_w = self.bp.SizeVar("dim_w")
        self.dim_c = self.bp.SizeVar("dim_c")
        self.nram_size = TARGET(target).nram_size
        self.dtype_sz = dtype.bytes
        self.single_nram_size = round_down(
            (self.nram_size - 30 * 1024) // 32 // self.dtype_sz, 128
        )
        self.bp.launch_task(self.task_num, 1, 1)

    def compute_body(self):
        # global
        buffer_in = self.bp.Buffer(
            shape=(self.dim_n, self.dim_h, self.dim_w, self.dim_c),
            dtype=self.dtype,
            name="buffer_in",
            scope="global"
        )
        buffer_out = self.bp.Buffer(
            shape=(self.dim_n, self.dim_h, self.dim_w, self.dim_c),
            dtype=self.dtype,
            name="buffer_out",
            scope="global"
        )

        data_num = self.bp.Scalar(dtype=bangpy.int32, name="data_num")
        data_num.assign(self.dim_n * self.dim_h * self.dim_w * self.dim_c)
        average_core = self.bp.Scalar(dtype=bangpy.int32, name="average_core")
        average_core.assign(data_num / self.task_num)
        remain_core = self.bp.Scalar(dtype=bangpy.int32, name="remain")
        remain_core.assign(data_num % self.task_num)

        # flatten
        flatten_buffer_in = buffer_in.reshape((data_num,))
        flatten_buffer_out = buffer_out.reshape((data_num,))
        buffer_one = self.bp.Scalar(
            name="ONE",
            dtype=self.dtype,
            value=1,
        )

        # nram
        buffer_in_n = self.bp.Buffer(
            shape=(self.single_nram_size,),
            name="INPUT_N",
            dtype=self.dtype,
            scope="nram",
        )
        buffer_out_n = self.bp.Buffer(
            shape=(self.single_nram_size,),
            name="OUTPUT_N",
            dtype=self.dtype,
            scope="nram",
        )

        task_id = self.bp.taskId
        core_start = task_id * average_core
        with self.bp.if_scope(task_id == self.task_num - 1):
            core_end = core_start + remain_core
            repeat = (average_core + remain_core) // self.single_nram_size
            remain = (average_core + remain_core) % self.single_nram_size
        with self.bp.else_scope():
            core_end = core_start + average_core
            repeat = average_core // self.single_nram_size
            remain = average_core % self.single_nram_size

        with self.bp.for_range(0, repeat) as i:
            start = core_start + i * self.single_nram_size
            end = start + self.single_nram_size
            self.bp.memcpy(buffer_in_n, flatten_buffer_in[start:end])
            self.bp.exp(buffer_in_n, buffer_in_n)
            self.bp.subtract(buffer_out_n, buffer_in_n, buffer_one)
            self.bp.memcpy(flatten_buffer_out[start:end], buffer_out_n)
        with self.bp.if_scope(remain != 0):
            start = core_start + repeat * self.single_nram_size
            end = start + remain
            self.bp.memcpy(buffer_in_n[:remain], flatten_buffer_in[start:end])
            self.bp.exp(buffer_in_n, buffer_in_n)
            self.bp.subtract(buffer_out_n, buffer_in_n, buffer_one)
            self.bp.memcpy(flatten_buffer_out[start:end], buffer_out_n[:remain])

        buffer_out = flatten_buffer_out.reshape((self.dim_n, self.dim_h, self.dim_w, self.dim_c))

        return self.bp.BuildBANG(
            inputs=[
                buffer_in,
                self.dim_n,
                self.dim_h,
                self.dim_w,
                self.dim_c
            ],
            outputs=[
                buffer_out
            ],
            kernel_name=KERNEL_NAME,
        )


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_expm1(dtype=None, target=None):
    # tasktype fixed in UNION1
    task_num = 4
    # print(dtype, target, task_num)
    f = Expm1(dtype, target, task_num).compute_body()
    return f
