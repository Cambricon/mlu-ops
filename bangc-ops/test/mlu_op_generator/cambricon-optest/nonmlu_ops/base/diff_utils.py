import numpy as np
import warnings
import copy
from .tensor import *


class EvaluatorImpl():
    def __init__(self, **kwargs):
        self.dtype_ulp_ = kwargs.get("dtype_ulp", 0)
        self.is_float_diff_ = kwargs.get("is_float_diff", False)
        self.epsilon_ = 1e-9

    def checkInf(self, base_data, compute_data):
        base_inf = np.isinf(base_data)
        compute_inf = np.isinf(compute_data)
        has_inf = base_inf.any() or compute_inf.any()
        assert has_inf == False, "[Evaluator]: unexpected inf result in dynamic threshold!"

    def checkNan(self, base_data, compute_data):
        base_nan = np.isnan(base_data)
        compute_nan = np.isnan(compute_data)
        has_nan = base_nan.any() or compute_nan.any()
        assert has_nan == False, "[Evaluator]: unexpected nan result in dynamic threshold!"

    def computeDiff1(self, base_data, compute_data):
        self.checkInf(base_data, compute_data)
        self.checkNan(base_data, compute_data)
        self.diff1_ = np.sum(np.abs(base_data-compute_data)) / \
            (np.sum(np.abs(base_data)) + self.epsilon_)
        return self.diff1_

    def computeDiff2(self, base_data, compute_data):
        self.checkInf(base_data, compute_data)
        self.checkNan(base_data, compute_data)
        self.diff2_ = np.sqrt(np.sum(np.square(base_data-compute_data)) /
                              (np.sum(np.square(base_data)) + self.epsilon_))
        return self.diff2_

    def computeDiff3(self, base_data, compute_data):
        self.checkInf(base_data, compute_data)
        self.checkNan(base_data, compute_data)
        numerator = np.abs(base_data - compute_data)
        denominator = np.abs(base_data) + self.epsilon_
        if self.is_float_diff_:
            index = np.abs(base_data) <= self.dtype_ulp_
            denominator[index] = 1
        self.diff3_ = np.max(numerator / denominator)
        return self.diff3_

    def computeDiff3_2(self, base_data, compute_data):
        self.checkInf()
        self.checkNan()
        self.diff3_2 = np.max(np.abs(base_data-compute_data))
        return self.diff3_2

    def computeDiff4(self, diff=0.75):
        self.checkInf()
        self.checkNan()
        self.diff4_ = DiffInfo(diff)
        return self.diff4_


class Evaluator():
    def __init__(self, base_datanode, compute_datanode, half_min_threshold=0, float_min_threshold=0, check_rate=True):
        self.base_datanode_ = copy.deepcopy(base_datanode)
        base_dtype = self.base_datanode_.getDataType()
        assert base_dtype in [DataType.DOUBLE, DataType.COMPLEX128], "[Evaluator]: unexpected base " \
            "data type in dynamic threshold, which should be double, complex128, complex_double!"
        self.compute_datanode_ = copy.deepcopy(compute_datanode)
        self.base_data_ = np.array([])
        self.base_real_ = np.array([])
        self.base_imag_ = np.array([])
        self.compute_data_ = np.array([])
        self.compute_real_ = np.array([])
        self.compute_imag_ = np.array([])

        self.half_min_threshold_ = half_min_threshold
        self.float_min_threshold_ = float_min_threshold
        self.static_threshold_ = 0.0
        self.dtype_ulp_ = 0.0
        self.max_threshold_ = 0.0
        self.diff_dtype_ = DataType.UNSET
        self.is_complex_ = False
        self.is_float_diff_ = False
        # if diff > 0.3, wrong base data or compute data may be passed in.
        self.diff_check_ = DiffInfo(0.3, 0.3)
        self.check_rate_ = check_rate

        self.processDiffParam()
        self.eva_impl_ = EvaluatorImpl(
            dtype_ulp=self.dtype_ulp_, is_float_diff=self.is_float_diff_)

    def processDiffParam(self):
        # set diff dtype
        compute_node_dtype = self.compute_datanode_.getDataType()
        if compute_node_dtype.isComplex():
            self.is_complex_ = True
            if compute_node_dtype == DataType.COMPLEX_HALF:
                self.diff_dtype_ = DataType.FLOAT16
            elif compute_node_dtype == DataType.COMPLEX_FLOAT:
                self.diff_dtype_ = DataType.FLOAT32
            else:
                raise ValueError(
                    "Complex dtype only half and float support dynamic diff")
        else:
            self.diff_dtype_ = compute_node_dtype

        # set diff threshold
        if self.diff_dtype_ == DataType.FLOAT16:
            if self.half_min_threshold_:
                self.static_threshold_ = self.half_min_threshold_
            else:
                self.static_threshold_ = 9.8e-4
            self.dtype_ulp_ = 1e-4
            self.max_threshold_ = 3e-2
            self.is_float_diff_ = True
        elif self.diff_dtype_ == DataType.FLOAT32:
            if self.float_min_threshold_:
                self.static_threshold_ = self.float_min_threshold_
            else:
                self.static_threshold_ = 1.2e-7
            self.dtype_ulp_ = 1e-6
            self.max_threshold_ = 3e-3
            self.is_float_diff_ = True
        else:
            pass

    def computeDynamicDiff(self, diff, rate):
        '''
        diff should be within [min_threshold, max_threshold]
        here, static threshold is minimum_threshold
        if rate is over default rate, minimum_threshold should be extended correspondingly
        '''
        dynamic_diff = diff * rate
        min_threshold = self.static_threshold_
        if rate > 10.0:
            min_threshold *= rate
        dynamic_diff = np.maximum(dynamic_diff, min_threshold)
        if self.is_float_diff_ and rate > 10.0:
            # max_threshold is valid only when rate is over default rate
            dynamic_diff = np.minimum(dynamic_diff, self.max_threshold_)
        return dynamic_diff

    def computeDiffBase(self, diff_func, post_process_func, rate):
        real_diff, imag_diff = -1, -1
        if self.is_complex_:
            # if base not init, init base
            if 0 in [self.base_data_.size, self.base_imag_.size, self.compute_real_.size, self.compute_imag_.size]:
                base_real, base_imag = self.base_datanode_.getComplexData()
                compute_real, compute_imag = self.compute_datanode_.getComplexData()
                compute_node_dtype = compute_real.dtype
                self.base_real_ = base_real.astype(
                    compute_node_dtype).astype(np.float64)
                self.base_imag_ = base_imag.astype(
                    compute_node_dtype).astype(np.float64)
                self.compute_real_ = compute_real.astype(np.float64)
                self.compute_imag_ = compute_imag.astype(np.float64)
            real_diff = diff_func(self.base_real_, self.compute_real_)
            real_diff = post_process_func(real_diff, rate)
            imag_diff = diff_func(self.base_imag_, self.compute_imag_)
            imag_diff = post_process_func(imag_diff, rate)

        else:
            if self.base_data_.size == 0 or self.compute_data_.size == 0:
                compute_node_dtype = self.compute_datanode_.getData().dtype
                self.base_data_ = self.base_datanode_.getData().astype(
                    compute_node_dtype).astype(np.float64)
                self.compute_data_ = self.compute_datanode_.getData().astype(np.float64)
            real_diff = diff_func(self.base_data_, self.compute_data_)
            real_diff = post_process_func(real_diff, rate)
        return DiffInfo(real_diff, imag_diff)

    def computeDiff1(self, rate=10.0):
        if not self.is_float_diff_:
            warnings.warn(
                "\033[1;33m [Evaluator]: diff1 excepts float data type, maybe you should use diff3 ==0!\033[0m")
        diff1 = self.computeDiffBase(
            self.eva_impl_.computeDiff1, self.computeDynamicDiff, rate)
        if self.check_rate_:
            assert diff1 <= self.diff_check_, "[Evaluator]: unexcepted dynamic diff1 threshold {}, which is too large.\
                Please check whether base_data and compute data are correct".format(diff1)
        return diff1

    def computeDiff2(self, rate=10.0):
        if not self.is_float_diff_:
            warnings.warn(
                "\033[1;33m [Evaluator]: diff2 excepts float data type, maybe you shoude use diff3 ==0!\033[0m")
        diff2 = self.computeDiffBase(
            self.eva_impl_.computeDiff2, self.computeDynamicDiff, rate)
        if self.check_rate_:
            assert diff2 <= self.diff_check_, "[Evaluator]: unexpected dynamic diff1 threshold {}, which is too large.\
                Please check whether base_data and compute data are correct".format(diff2)
        return diff2

    def computeDiff3(self, rate=10.0):
        def computeDynamicDiff3(diff, rate):
            if diff == 0:
                warnings.warn(
                    "\033[1;33m [Evaluator]: would better use static_threshold for diff3 if diff rate is 0\033[0m")
            return self.computeDynamicDiff(diff, rate)
        diff3 = self.computeDiffBase(
            self.eva_impl_.computeDiff3, computeDynamicDiff3, rate)
        return diff3

    def computeDiff3_2(self, rate=10.0):
        def computeDynamicDiff3_2(diff, rate):
            if diff == 0:
                warnings.warn(
                    "\033[1;33m [Evaluator]: would better use static_threshold for diff3_2 if diff rate is 0\033[0m")
            return self.computeDynamicDiff(diff, rate)
        diff3_2 = self.computeDiffBase(
            self.eva_impl_.computeDiff3, computeDynamicDiff3_2, rate)
        return diff3_2

    def computeDiff4(self, base=0.75):
        if self.is_complex_:
            return DiffInfo(base, base)
        else:
            return DiffInfo(base)
