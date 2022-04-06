import bangpy
from bangpy import tcp
from bangpy.tcp.util import round_up, round_down
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float16, bangpy.float32]
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]
KERNEL_NAME = "expm1"


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
        self.buffer_one = self.bp.Scalar(
            name="ONE",
            dtype=self.dtype,
            value=1,
        )
        self.buffer_in = self.bp.Buffer(
            shape=(self.dim_n, self.dim_h, self.dim_w, self.dim_c),
            dtype=self.dtype,
            name="buffer_in",
            scope="global"
        )
        self.buffer_out = self.bp.Buffer(
            shape=(self.dim_n, self.dim_h, self.dim_w, self.dim_c),
            dtype=self.dtype,
            name="buffer_out",
            scope="global"
        )
        self.single_nram_size = round_down(
            (self.nram_size - 30 * 1024) // 32 // self.dtype_sz, ALIGN_LENGTH
        )
        self.bp.launch_task(self.task_num, 1, 1)

    def compute(self):
        self.bp.exp(self.buffer_in_n, self.buffer_in_n)
        self.bp.subtract(self.buffer_out_n, self.buffer_in_n, self.buffer_one)

    @property
    def compute_body(self):
        data_num = self.bp.Scalar(dtype=bangpy.int32, name="data_num")
        data_num.assign(self.dim_n * self.dim_h * self.dim_w * self.dim_c)
        average_core = self.bp.Scalar(dtype=bangpy.int32, name="average_core")
        average_core.assign(data_num / self.task_num)
        remain_core = self.bp.Scalar(dtype=bangpy.int32, name="remain")
        remain_core.assign(data_num % self.task_num)

        # flatten
        flatten_buffer_in = self.buffer_in.reshape((data_num,))
        flatten_buffer_out = self.buffer_out.reshape((data_num,))

        task_id = self.bp.taskId
        core_start = task_id * average_core
        core_end = core_start + average_core
        repeat = average_core // self.single_nram_size
        remain = average_core % self.single_nram_size

        with self.bp.for_range(0, repeat, stage=1) as i:
            start = core_start + i * self.single_nram_size
            end = start + self.single_nram_size
            # nram
            self.buffer_in_n = self.bp.Buffer(
                shape=(self.single_nram_size,),
                name="INPUT_N",
                dtype=self.dtype,
                scope="nram",
            )
            self.buffer_out_n = self.bp.Buffer(
                shape=(self.single_nram_size,),
                name="OUTPUT_N",
                dtype=self.dtype,
                scope="nram",
            )
            with self.bp.block(stage_scope="data_copy"):
                self.bp.memcpy(self.buffer_in_n, flatten_buffer_in[start:end])
            with self.bp.block(stage_scope="compute"):
                self.compute()
            with self.bp.block(stage_scope="data_copy"):
                self.bp.memcpy(flatten_buffer_out[start:end], self.buffer_out_n)
        with self.bp.if_scope(remain != 0):
            start = core_start + repeat * self.single_nram_size
            end = start + remain
            self.bp.memcpy(self.buffer_in_n[:remain], flatten_buffer_in[start:end])
            self.compute()
            self.bp.memcpy(flatten_buffer_out[start:end], self.buffer_out_n[:remain])

        with self.bp.if_scope(remain_core != 0):
            with self.bp.if_scope(task_id == self.task_num - 1):
                start = task_id * average_core
                end = start + remain_core
                self.bp.memcpy(self.buffer_in_n[:remain_core], flatten_buffer_in[start:end])
                self.compute()
                self.bp.memcpy(flatten_buffer_out[start:end], self.buffer_out_n[:remain_core])

        self.buffer_out = flatten_buffer_out.reshape((self.dim_n, self.dim_h, self.dim_w, self.dim_c))

        return self.bp.BuildBANG(
            inputs=[
                self.buffer_in,
                self.dim_n,
                self.dim_h,
                self.dim_w,
                self.dim_c
            ],
            outputs=[
                self.buffer_out
            ],
            kernel_name=KERNEL_NAME,
        )


@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_expm1(dtype=None, target=None):
    # tasktype fixed in UNION1
    task_num = 64
    # print(dtype, target, task_num)
    f = Expm1(dtype, target, task_num).compute_body
    return f
