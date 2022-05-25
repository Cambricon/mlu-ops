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
import bangpy
from bangpy import tcp
from bangpy.platform.bang_config import TARGET
from bangpy.tcp.util import round_down
from bangpy.tcp.runtime import TaskType

DTYPES = [bangpy.float32] #支持的类型
TARGET_LIST = ["mlu290"]#支持的设备
KERNEL_NAME = "Celu" #算子名


class Celu:
    """Operator description:
    Add the data in the two buffers.
    """

    def __init__(self, dtype, target, task_num):#self 即this指针  dtype 传入的数据类型 target目标芯片的型号
        self.dtype = dtype
        self.target = target
        self.task_num = task_num
        self.bp = tcp.TCP(target)
        #alpha = self.bp.SizeVar("alpha", self.dtype)
        self.inplace = self.bp.Var("inplace")
        self.length = self.bp.SizeVar("length")#得到数据的长度  此处应该是数组的长度
        self.dtype_sz = dtype.bytes#类型占用空间的大小(字节)
        self.bp.launch_task(self.task_num, 1, 1)#将任务维度值设置为在此内核中启动。  三个参数其实就是 taskdimx,y,z
    #根据标记位buffer 从数据buffer中获取指定数值去替换输入中对应位置的数据
    #changed_buffer 输入数据 等待被替换的数据buffer
    #value_buffer 数据buffer 标志位buffer将从这里获取对应位数据来准备替换changed_buffer中的数据
    #marked_bool_buffer 标记位buffer 要准备替换的位置标位0  其余为1
    def replace_the_marked_value(self,changed_buffer,marked_bool_buffer,value_buffer):
        #被标记位是0 相乘后使替换位归零
        self.bp.multiply(changed_buffer,changed_buffer,marked_bool_buffer)
        #对标记位取反 此时所有标记位是1 非标位为0
        self.bp.logical_not(marked_bool_buffer,marked_bool_buffer)
        #标记与填充值相乘 使所有标记位变为填充值 ，非标位仍是0
        self.bp.multiply(marked_bool_buffer,value_buffer,marked_bool_buffer)
        #标记位被归零后得原始数据和新得到得标记位数值相加  替换完毕
        self.bp.add(changed_buffer,changed_buffer,marked_bool_buffer)
    #将input中得数据和阈值比较，在bool_mark对应位置 符合条件为1否则为0
    #input_buffer 原始输入 将要替换得输入
    #bool_mark 真值buffer 存储对应位置判断的结果
    #is_min int值 控制逻辑分支是使用>=还是<=,
    #threshold_value 要进行判断得阈值
    def mark_value_compare_with_threshold_value(self,input_buffer,bool_mark,is_min,threshold_value):
         #通过将输入和阈值进行比较 将真值存入nram_temp_bool_res中 符合条件为1 不符为0
        if is_min == 1:#和最小值比较
            self.bp.greater_equal(bool_mark,input_buffer,threshold_value,'elemwise') #大于等于阈值返回1
        #和最大值比较
        else :
            self.bp.less_equal(bool_mark,input_buffer,threshold_value,'elemwise') #小于等于阈值返回1
    #标记不在范围内的数据的位置
    #input_buffer 原始数据
    #x承载小于阈值的数据对应位置的buffer 对应位置上
    #y承载大于阈值的数据对应位置的buffer
    def mark_the_out_of_range_vlaue(self,input_buffer,x,y):
        max_threshold_valu = self.bp.Scalar(self.dtype,"max_threshold_valu",10)
        min_threshold_valu = self.bp.Scalar(self.dtype,"min_threshold_valu",-7.5)
        #这些数我是网上查的该类型大于0时的最大最小值 然后取了个ln得到的   这里注释掉的原因是 其exp接口最大范围是[-7.5,10] 并不随类型的范围
        #当为16位是 max min 采用以下值
        # if self.dtype == bangpy.float16 :
        #     max_threshold_valu.assign(11.089866488461016076210728979771)
        #     min_threshold_valu.assign(-9.703981170988072538409566077448)
        # #32位时使用以下值
        # else:
        #     max_threshold_valu.assign(88.722008965395851698332450562653)
        #     min_threshold_valu.assign(-87.332719095296162600686375692197)
        #将输入中小于最小值的在x对应位置标为0
        self.mark_value_compare_with_threshold_value(input_buffer,x,1,min_threshold_valu)
        #将输入中大于最大值的在y对应位置标为0
        self.mark_value_compare_with_threshold_value(input_buffer,y,0,max_threshold_valu)
    def compute_body(self):
        one_core_count = self.bp.Scalar(bangpy.int32,"one_core_count")
        remain =  self.bp.Scalar(bangpy.int32,"remain")
        current_core_start = self.bp.Scalar(bangpy.int32,"current_core_start") #当前核心数据开始索引
        current_core_end = self.bp.Scalar(bangpy.int32,"current_core_end") #当前核心数据结束索引
        total_count_in_core = self.bp.Scalar(bangpy.int32,"total_count_in_core")
        calc_loop_count = self.bp.Scalar(bangpy.int32,"calc_loop_count")
        once_loop_start = self.bp.Scalar(bangpy.int32,"once_loop_start")
        calc_size = self.bp.Scalar(bangpy.int32,"calc_size")
        nram_avable_size = round_down( (TARGET(self.target).nram_size - 40* 1024) // 8  ,128)
        one_core_count.assign(self.length // self.task_num)#每个核均摊计算量（按索引分）
        remain.assign(self.length % self.task_num)#分任务时的余数
        process_count = nram_avable_size // self.dtype_sz #核心一次最多计算的长度
        with self.bp.if_scope(self.bp.taskId < remain): #如果存在余数 将其均摊给各核   taskId从0起
            current_core_start.assign((one_core_count + 1) * self.bp.taskId )
            #此处应该不需要减1 待验证  python切片会自动将上标减1
            current_core_end.assign((one_core_count + 1) * (self.bp.taskId + 1) - 1)
        with self.bp.else_scope():
            current_core_start.assign(
                (one_core_count + 1) * remain + one_core_count * (self.bp.taskId - remain)
                )
            current_core_end.assign(current_core_start  + one_core_count - 1)
        total_count_in_core.assign(current_core_end - current_core_start + 1)
        buffer_in0 = self.bp.Buffer(
            shape=(self.length,), name="INPUT0", dtype=self.dtype, scope="global"
        )
        buffer_alpha = self.bp.Buffer(
            shape=(1,), name="ALPHA_PARAM", dtype=self.dtype, scope="global"
        )
        buffer_out = self.bp.Buffer(
            shape=(self.length,), name="OUTPUT", dtype=self.dtype, scope="global"
        )
        alpha = self.bp.Scalar(dtype = self.dtype,name = "alpha")
        alpha.assign(buffer_alpha[0])
        nram_buffer_in0 = self.bp.Buffer(
            shape=(process_count,),
            name="INPUT0_N",
            dtype=self.dtype,
            scope="nram",
        )
        nram_middle_value = self.bp.Buffer(
            shape=(process_count,),
            name="N_MAX",
            dtype=self.dtype,
            scope="nram",
        )
        nram_max = self.bp.Buffer(
            shape=(process_count,),
            name="N_MAX",
            dtype=self.dtype,
            scope="nram",
        )
        nram_min = self.bp.Buffer(
            shape=(process_count,),
            name="N_MIN",
            dtype=self.dtype,
            scope="nram",
        )
        nram_marked_exp_overrun_the_upper_limit = self.bp.Buffer(
            shape=(process_count,),
            name="NMEOTUL",
            dtype=self.dtype,
            scope="nram",
        )
        nram__marked_exp_beyond_the_lower_limit = self.bp.Buffer(
            shape=(process_count,),
            name="NMEBTLL",
            dtype=self.dtype,
            scope="nram",
        )
        nram_marked_zero = self.bp.Buffer(
            shape=(process_count,),
            name="NMZ",
            dtype=self.dtype,
            scope="nram",
        )
        const_zero = self.bp.Scalar(dtype = self.dtype,name = "const_zero",value = 0)
        const_one = self.bp.Scalar(dtype = self.dtype,name = "const_one",value = 1)
        replace_value =  self.bp.Scalar(dtype = self.dtype,name = "replace_value")
        calc_loop_count.assign((total_count_in_core + process_count - 1) // process_count)
        with self.bp.for_range(0, calc_loop_count) as i:
            #当前核心数据开始的位置 + 第i次循环所应偏移的长度
            once_loop_start.assign(current_core_start + process_count * i)
            with self.bp.if_scope(i < calc_loop_count - 1):
                calc_size.assign(process_count)
            with self.bp.else_scope():
                calc_size.assign(total_count_in_core % process_count)
                with self.bp.if_scope(calc_size == 0):
                    calc_size.assign(process_count)
            with self.bp.block("data_copy"):
                self.bp.memcpy(
                    nram_buffer_in0[0:calc_size],
                    buffer_in0[once_loop_start:once_loop_start + calc_size]
                    )
            #这里开始计算min
            with self.bp.if_scope(alpha != 0):
                #大于等于0的标记为0 小于的标记为1
                self.bp.less_equal(nram_marked_zero,nram_buffer_in0,const_zero,'elemwise')
                self.bp.divide(nram_middle_value,nram_buffer_in0,alpha)#获得x/a
                self.mark_the_out_of_range_vlaue(
                    nram_middle_value,
                    nram_marked_exp_overrun_the_upper_limit,
                    nram__marked_exp_beyond_the_lower_limit
                    )#标记出所有超出运算范围的值的位置并分别在两个buffer中用0标注
                #前期准备基本完成 开始常规计算
                self.bp.exp(nram_middle_value,nram_middle_value)#计算exp(x/a)
                self.bp.subtract(nram_middle_value, nram_middle_value, const_one)#-1
                # self.bp.print(nram_middle_value[0])
                self.bp.multiply(nram_middle_value, nram_middle_value, alpha)#*a
                self.bp.minimum(nram_min,nram_middle_value,const_zero)#min(0,...)
                #开始替换
                #将所有x>=0得位置全部替换成0
                self.bp.multiply(nram_middle_value, nram_middle_value,nram_marked_zero)
                #另一种情况  当（x/a）< e 的最小次方值时   将所有标记位替换成 -a和0中小的那个
                with self.bp.if_scope(alpha * -1 > 0):
                    replace_value.assign(0)
                with self.bp.else_scope():
                    replace_value.assign(alpha * -1)
                self.replace_the_marked_value(
                    nram_middle_value,
                    nram__marked_exp_beyond_the_lower_limit,
                    replace_value
                    )
            with self.bp.else_scope():#当alpha为0时  min全为0
                self.bp.zeros(nram_min)
            #这里开始计算max
            self.bp.maximum(nram_max,nram_buffer_in0,const_zero)
            #计算max+min
            self.bp.add(nram_buffer_in0,nram_max,nram_min)
            self.bp.memcpy(
                buffer_out[once_loop_start:once_loop_start + calc_size],
                nram_buffer_in0[:calc_size]
                )
        f = self.bp.BuildBANG(
            inputs=[buffer_in0,buffer_alpha,self.inplace],
            outputs=[buffer_out],
            kernel_name=KERNEL_NAME,
        )
        return f
@tcp.register_mlu_op(DTYPES, TARGET_LIST, KERNEL_NAME)
def build_celu(dtype=None, target=None):
    task_type=TaskType.UNION16
    task_num =task_type.value*4 #这里可能是这么理解  一个cluster 4个核   根据union的类型乘4确定投入的core
    f = Celu(dtype, target, task_num).compute_body()
    return f
