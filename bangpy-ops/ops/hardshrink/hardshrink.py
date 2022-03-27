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
# pylint: disable=useless-object-inheritance, too-many-instance-attributes
# pylint: disable=attribute-defined-outside-init, too-many-statements
# pylint: disable=too-many-arguments, too-many-locals
"""HardShrink operator implementation using BANGPy TCP API."""
import bangpy
from bangpy import tcp
from bangpy.common import utils,load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float32]
TARGET_LIST = ["mlu370-s4", "mlu220-m2", "mlu270", "mlu290"]
KERNEL_NAME = "hardshrink"

class HardShrink(object):
    """Operator description:
    An activation function, behaves similar to torch.hardshrink.
    """
    def __init__(self, target, dtype, task_num, task_type, kernel_name = "hardshrink"):
        """Construct a new HardShrink class."""
        """Parameter
        ------
        target : string
            Target MLU device name.
        dtype : bangpy.DType
            The data type of input.
        task_num : int
            The task number of runtime.
        name : string
            Kernel entry function name.
        scalar_lambda : Scalar
            The lambda para, default is 0.05
        """
        self.bp = tcp.TCP(target)
        self.target = target
        self.dtype = dtype
        self.task_num = task_num
        self.task_type = task_type
        self.kernel_name = kernel_name
        self.lambdaPara = self.bp.Var("lambdaPara", dtype = self.dtype)
        self.scalar_lambda = self.bp.Scalar(name = "scalar_lambda", value = self.lambdaPara, dtype = self.dtype)
        """Attributes
        ------
        tcp : tcp.TCP
            TCP container.
        """
        self.dim_num = self.bp.SizeVar("dim_num")
        self.dim_0 = self.bp.SizeVar("dim_0")
        self.dim_1 = self.bp.SizeVar("dim_1")
        self.dim_2 = self.bp.SizeVar("dim_2")
        self.dim_3 = self.bp.SizeVar("dim_3")

        self.element_num = self.dim_0 * self.dim_1 * self.dim_2 * self.dim_3 # all data num
        self.tensor_shape = (self.dim_0, self.dim_1, self.dim_2, self.dim_3)

        self.nram_size = TARGET(target).nram_size # get the device constraint
        self.nram_use_size = self.nram_size // 32 # per calculate num

        self.base_align = 128 # basic alignment requirement
    def hardshrink_compute(self):
        
        # self.bp.launch_task(TaskType.UNION4.value*4,1,1)
        # can't use "with self.bp.if_scope"
        if self.task_type == "UNION1":
            self.bp.launch_cluster(TaskType.UNION1.value)
            self.task_num = TaskType.UNION1.value * 4
        elif self.task_type == "UNION2":
            self.bp.launch_cluster(TaskType.UNION2.value)
            self.task_num = TaskType.UNION2.value * 4
        elif self.task_type == "UNION4":
            self.bp.launch_cluster(TaskType.UNION4.value)
            self.task_num = TaskType.UNION4.value * 4

        self.bp.launch_task(self.task_num, 1, 1)
        self.task_id = self.bp.taskId
        self.task_content = self.element_num // self.task_num # per task need to calculate num

        """ global """
        self.tensor_input = self.bp.Buffer(shape=self.tensor_shape, name="input_tensor", dtype=self.dtype, scope="global")
        self.tensor_output = self.bp.Buffer(shape=self.tensor_shape, name="output_tensor", dtype=self.dtype, scope="global")
        """ flatten """
        self.tensor_input_flatten = self.tensor_input.reshape((self.element_num,))
        self.tensor_output_flatten = self.tensor_output.reshape((self.element_num,))
        """ nram —— shape only 1 """
        tensor_input_nram = self.bp.Buffer(shape=(self.nram_use_size,), name="input_tensor_nram", dtype=self.dtype, scope="nram")
        tensor_output_nram = self.bp.Buffer(shape=(self.nram_use_size,), name="output_tensor_nram", dtype=self.dtype, scope="nram")
        tensor_abs_nram = self.bp.Buffer(shape=(self.nram_use_size,), name="abs_tensor_nram", dtype=self.dtype, scope="nram")
        tensor_greater_nram = self.bp.Buffer(shape=(self.nram_use_size,), name="greater_tensor_nram", dtype=self.dtype, scope="nram")

        """ core computation —— for """
        task_start = self.task_id * self.task_content
        task_end = task_start + self.task_content
        cmpt_times = self.task_content // self.nram_use_size
        with self.bp.for_range(0,cmpt_times) as c_t:
            cmpt_start = task_start + c_t * self.nram_use_size
            cmpt_end = cmpt_start + self.nram_use_size
            self.bp.memcpy(tensor_input_nram, self.tensor_input_flatten[cmpt_start:cmpt_end])
            self.bp.abs(tensor_abs_nram, tensor_input_nram)
            self.bp.greater(tensor_greater_nram, tensor_abs_nram, self.scalar_lambda)
            self.bp.multiply(tensor_output_nram, tensor_input_nram, tensor_greater_nram) # multiply is the necessary? Other op can replace it?
            self.bp.memcpy(self.tensor_output_flatten[cmpt_start:cmpt_end], tensor_output_nram)
        # there are two aspects for the alignment:
        # 1. per task inner task;
        # 2. no task assgined to do
        # start the first align
        remain_num = self.task_content % self.nram_use_size
        with self.bp.if_scope(remain_num != 0):
            # cal the start and the end pos
            start_pos = task_start + cmpt_times * self.nram_use_size
            end_pos = start_pos + remain_num
            self.bp.memcpy(tensor_input_nram[:remain_num],self.tensor_input_flatten[start_pos:end_pos])
            self.bp.abs(tensor_abs_nram,tensor_input_nram)
            self.bp.greater(tensor_greater_nram,tensor_abs_nram,self.scalar_lambda)
            self.bp.multiply(tensor_output_nram,tensor_input_nram,tensor_greater_nram)
            self.bp.memcpy(self.tensor_output_flatten[start_pos:end_pos],tensor_output_nram[:remain_num])
        # start the second align
        # only need smaller nrams —— the basic alignment
        # There has a problem: task_id from 0 or 1 ?
        task_remain = self.element_num % self.task_num # remain num which has no task to deal
        with self.bp.if_scope(task_remain != 0):
            with self.bp.if_scope(self.task_id == self.task_num-1):
                tensor_input_small_nram = self.bp.Buffer(shape=(self.base_align,),name="input_tensor_small_nram",dtype=self.dtype,scope="nram")
                tensor_output_small_nram = self.bp.Buffer(shape=(self.base_align,),name="output_tensor_small_nram",dtype=self.dtype,scope="nram")
                tensor_abs_small_nram = self.bp.Buffer(shape=(self.base_align,),name="abs_tensor_small_nram",dtype=self.dtype,scope="nram")
                tensor_greater_small_nram = self.bp.Buffer(shape=(self.base_align,),name="greater_tensor_small_nram",dtype=self.dtype,scope="nram")
                self.bp.memcpy(tensor_input_small_nram[:task_remain],self.tensor_input_flatten[task_end:])
                self.bp.abs(tensor_abs_small_nram,tensor_input_small_nram)
                self.bp.greater(tensor_greater_small_nram,tensor_abs_small_nram,self.scalar_lambda)
                self.bp.multiply(tensor_output_small_nram,tensor_input_small_nram,tensor_greater_small_nram)
                self.bp.memcpy(self.tensor_output_flatten[task_end:],tensor_output_small_nram[:task_remain])

        self.tensor_output = self.tensor_output_flatten.reshape(self.tensor_shape)
        
        return self.bp.BuildBANG(
            inputs = [
                self.tensor_input,
                self.dim_num,
                self.lambdaPara
            ],
            outputs = [self.tensor_output],
            kernel_name = self.kernel_name
        )

@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_hardshrink_tensor(dtype, target):
    task_num = 64 # max 64
    task_type = "BLOCK" # choices: BLOCK(need to set thetask_num)、UNION1、UNION2、UNION4
    hardshrink = HardShrink(target=target,dtype=dtype,task_num=task_num,task_type=task_type)
    f_hardshrink = hardshrink.hardshrink_compute()
    return f_hardshrink