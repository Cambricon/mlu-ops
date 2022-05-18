from enum import Enum
import random
import numpy as np
from math import sqrt
from . import quantize_utils


class Layout(Enum):
    UNSET = 1
    NCHW = 2
    NHWC = 3
    HWCN = 4
    NDHWC = 5
    ARRAY = 6
    TNC = 7
    NTC = 8
    NCDHW = 9
    NC = 10

    def getStr(self):
        return str(self).split(".")[-1]


class DataType(Enum):
    FLOAT16 = 1
    FLOAT32 = 2
    INT8 = 3
    INT16 = 4
    INT32 = 6
    DOUBLE = 8
    UINT8 = 9
    UINT16 = 10
    UINT32 = 11
    BOOL = 12
    COMPLEX_HALF = 13
    COMPLEX_FLOAT = 14
    UNSET = 15
    INT64 = 16
    UINT64 = 17
    def getDataSize(self):
        data_type_dict = {
            DataType.FLOAT16: 2,
            DataType.FLOAT32: 4,
            DataType.DOUBLE: 8,
            DataType.UINT8: 1,
            DataType.UINT16: 2,
            DataType.UINT32: 4,
            DataType.UINT64: 8,
            DataType.INT8: 1,
            DataType.INT16: 2,
            DataType.INT32: 4,
            DataType.INT64: 8,
            DataType.COMPLEX_HALF: 4,
            DataType.COMPLEX_FLOAT: 8,
            DataType.BOOL: 1,
        }
        return data_type_dict[self]

    def getDataBits(self):
        return self.getDataSize() * 8

    def getStr(self):
        return str(self).split(".")[-1].lower()

    def getComplexSaveType(self):
        type_dict = {
            DataType.COMPLEX_HALF: "half",
            DataType.COMPLEX_FLOAT: "float",
        }
        return type_dict[self]

    def getNumpyStr(self):
        return self.getStr()

    def isFloatPoint(self):
        float_list = [DataType.FLOAT16, DataType.FLOAT32, DataType.DOUBLE]
        return self in float_list

    def isComplex(self):
        complex_list = [DataType.COMPLEX_HALF,
                        DataType.COMPLEX_FLOAT]
        return self in complex_list

    def exists(self):
        return self != DataType.UNSET


class RandomDistribution(Enum):
    UNSET = 1
    UNIFORM = 2
    GAUSSIAN = 3
    SAMPLE = 4
    BINOMIAL = 5


class DataNode:
    '''
    Descriptor of Data.
    Has dtype_, pos_, data_, data_real_, data_imag_ : np.array.
    data_ use to store non complex.
    data_real_ and data_imag_ use to store complex
    '''

    def __init__(self, dtype: str = "unset"):
        self.dtype_ = DataType.UNSET
        self.position_ = 0
        self.data_ = None
        self.data_real_ = None
        self.data_imag_ = None
        if isinstance(dtype, str):
            self.dtype_ = DataType[dtype.upper()]
        elif isinstance(dtype, list) and len(dtype) == 4:
            self.dtype_ = DataType[dtype[0].upper()]

    def __repr__(self):
        shape_str = ""
        if hasattr(self.data_, "shape"):
            shape_str = ", shape: " + str(self.data_.shape)
        if hasattr(self.data_real_, "shape"):
            shape_str = ", shape: " + str(self.data_real_.shape)
        return (f"{DataNode.__module__}.{DataNode.__qualname__} "
                f"dtype: {self.dtype_}, position: {self.position_}" + shape_str)

    def getDataType(self):
        return self.dtype_

    def setData(self, data):
        if data is not None:
            assert self.data_real_ is None and self.data_imag_ is None, "setData and setComplexData can not be used together!"
        try:
            self.data_ = data.astype(self.dtype_.getStr())
        except:
            self.data_ = data

    def getData(self):
        return self.data_

    def setComplexData(self, real=None, imag=None):
        if real is not None or imag is not None:
            assert self.data_ is None, "setData and setComplexData can not be used together!"
        try:
            self.data_real_ = real.astype(self.dtype_.getComplexSaveType())
            self.data_imag_ = imag.astype(self.dtype_.getComplexSaveType())
        except:
            self.data_real_ = real
            self.data_imag_ = imag

    def getComplexData(self):
        if not self.dtype_.isComplex():
            print("Warning: use getComplexData but dtype is not complex.")
        return self.data_real_, self.data_imag_


class DiffInfo():
    '''
    Return value of compute diff functions of Evaluator class.
    '''

    def __init__(self, real_diff, imag_diff=-1):
        if isinstance(real_diff, (int, float)):
            self.real_diff_ = real_diff
            self.imag_diff_ = imag_diff
        elif isinstance(real_diff, DiffInfo):
            self.real_diff_ = real_diff.real_diff_
            self.imag_diff_ = real_diff.imag_diff_

    def __gt__(self, base):
        base_diffinfo = DiffInfo(base)
        if self.real_diff_ > base_diffinfo.real_diff_ or \
           self.imag_diff_ > base_diffinfo.imag_diff_:
            return True
        else:
            return False

    def __ge__(self, base):
        base_diffinfo = DiffInfo(base)
        if self.real_diff_ >= base_diffinfo.real_diff_ or \
           self.imag_diff_ >= base_diffinfo.imag_diff_:
            return True
        else:
            return False

    def __eq__(self, base):
        base_diffinfo = DiffInfo(base)
        if self.real_diff_ == base_diffinfo.real_diff_ and \
           self.imag_diff_ == base_diffinfo.imag_diff_:
            return True
        else:
            return False

    def __ne__(self, base):
        base_diffinfo = DiffInfo(base)
        if self.real_diff_ != base_diffinfo.real_diff_ and \
           self.imag_diff_ != base_diffinfo.imag_diff_:
            return True
        else:
            return False

    def __lt__(self, base):
        base_diffinfo = DiffInfo(base)
        if self.real_diff_ < base_diffinfo.real_diff_ or \
           self.imag_diff_ < base_diffinfo.imag_diff_:
            return True
        else:
            return False

    def __le__(self, base):
        base_diffinfo = DiffInfo(base)
        if self.real_diff_ <= base_diffinfo.real_diff_ and \
           self.imag_diff_ <= base_diffinfo.imag_diff_:
            return True
        else:
            return False

    def __str__(self):
        if self.imag_diff_ == -1:
            return "DiffInfo: {}".format(self.real_diff_)
        else:
            return "DiffInfo: ({},{})".format(self.real_diff_, self.imag_diff_)


class Tensor:
    '''
    Implement of class TensorList. contains information about data.

    Do not modify this class for specific kernel.
    If you want change tensors behaivor, please see TensorList and inherite it.
    '''

    def __init__(self, *args, **kwargs):
        self.name_ = kwargs.get("name", "")
        self.shape_ = [int(num) for num in kwargs.get("shape", [])]
        self.strides_ = [int(num) for num in kwargs.get("strides", None)] \
            if kwargs.get("strides", None) is not None else None
        self.layout_ = Layout[kwargs.get("layout", "unset").upper()]
        self.random_distribution_ = kwargs.get("random_distribution", None)
        self.datanode_ = DataNode(kwargs["dtype"])
        self.onchip_datanode_ = DataNode(kwargs["onchip_dtype"])
        self.filename_ = kwargs.get("file_name", "")
        self.require_value_ = kwargs.get("require_value", True)
        self.contain_nan_ = kwargs.get("contain_nan", False)
        self.contain_inf_ = kwargs.get("contain_inf", False)

        self.diff1_ = DiffInfo(-1)
        self.diff2_ = DiffInfo(-1)
        self.diff3_ = DiffInfo(-1)
        self.diff3_2_ = DiffInfo(-1)
        self.diff4_ = DiffInfo(-1)

    def __repr__(self):
        shape_str = ""
        if not hasattr(self.datanode_.data_, "shape"):
            shape_str = f" ,shape: {self.shape_}"
        return (f"{Tensor.__module__}.{Tensor.__qualname__}"
                f"data: {self.datanode_}, layout: {self.layout_}" + shape_str)

    def hasNanInf(self):
        return self.contain_nan_ or self.contain_inf_

    def getShape(self):
        return self.shape_

    def setShape(self, shape):
        self.shape_ = shape

    def getDataType(self):
        return self.datanode_.dtype_

    def getOnchipType(self):
        return self.onchip_datanode_.dtype_

    def getData(self):
        return self.datanode_.getData()

    def setData(self, data):
        self.datanode_.setData(data)

    def getComplexData(self):
        return self.datanode_.getComplexData()

    def setComplexData(self, real, imag):
        self.datanode_.setComplexData(real, imag)

    def getOnchipData(self):
        return self.onchip_datanode_.getData()

    def getDataNode(self):
        return self.datanode_

    def getOnchipDataNode(self):
        return self.onchip_datanode_

    def getLayout(self):
        return self.layout_

    def setDiff(self, diff1=DiffInfo(-1),
                diff2=DiffInfo(-1),
                diff3=DiffInfo(-1),
                diff3_2=DiffInfo(-1),
                diff4=DiffInfo(-1)):
        self.diff1_ = DiffInfo(diff1)
        self.diff2_ = DiffInfo(diff2)
        self.diff3_ = DiffInfo(diff3)
        self.diff3_2_ = DiffInfo(diff3_2)
        self.diff4_ = DiffInfo(diff4)

    def setDiff1(self, diff1):
        self.diff1_ = DiffInfo(diff1)

    def setDiff2(self, diff2):
        self.diff2_ = DiffInfo(diff2)

    def setDiff3(self, diff3):
        self.diff3_ = DiffInfo(diff3)

    def setDiff3_2(self, diff3_2):
        self.diff3_2_ = DiffInfo(diff3_2)

    def setDiff4(self, diff4):
        self.diff4_ = DiffInfo(diff4)

    def getDiff1(self):
        return self.diff1_

    def getDiff2(self):
        return self.diff2_

    def getDiff3(self):
        return self.diff3_

    def getDiff3_2(self):
        return self.diff3_2_

    def getDiff4(self):
        return self.diff4_


class RandomData:
    '''
    Default random data generator for TensorList
    '''

    def __init__(self, tensor: Tensor):
        self.tensor_ = tensor

    def random(self):
        '''
        Main process of randomData.
        '''
        data, real, imag, position = self.random_generator()
        self.tensor_.getDataNode().setData(data)
        self.tensor_.getDataNode().setComplexData(real, imag)

    def random_generator(self):
        '''
        Data will be generatored in this function
        '''
        distribution, random_range = next(
            iter(self.tensor_.random_distribution_.items()))
        start = random_range[0]
        end = random_range[1]
        shape = self.tensor_.shape_
        np_value = None
        np_complex_value_real = None
        np_complex_value_imag = None
        if self.tensor_.strides_ is not None:
            total_num = 1
            for i in range(len(shape)):
                total_num += (shape[i]-1) * self.tensor_.strides_[i]
            shape = [total_num]
        data_node = self.tensor_.getDataNode()
        dtype = self.tensor_.getDataType()
        position = 0
        if dtype.isComplex():
            random_dtype = "float" + str(dtype.getDataBits() // 2)
        else:
            random_dtype = dtype.getStr()

        if dtype == DataType.BOOL:
            if distribution == "uniform":
                if [start, end] not in [[0, 1], [0, 2], [1, 2]]:
                    start, end = 0, 2
                np_value = np.array(np.random.randint(
                    start, end, size=shape), dtype="bool")
            elif distribution == "binomial":
                np_value = np.random.binomial(
                    start, end, size=shape).astype(dtype="bool")
        else:
            if distribution == "uniform":
                if dtype.isComplex():
                    complex_shape = shape + [2]
                    np_data = np.random.uniform(
                        start, end, size=complex_shape).astype(random_dtype)
                    real_data, imag_data = np_data[..., 0], np_data[..., 1]
                    np_complex_value_real, np_complex_value_imag = np_data[...,
                                                                           0], np_data[..., 1]
                else:
                    np_value = np.random.uniform(
                        start, end, size=shape).astype(random_dtype)

            elif distribution == "gaussian":
                if dtype.isComplex():
                    complex_shape = shape + [2]
                    np_data = np.random.normal(
                        start, end / sqrt(2), size=complex_shape).astype(random_dtype)
                    real_data, imag_data = np_data[..., 0], np_data[..., 1]
                    np_complex_value_real, np_complex_value_imag = np_data[...,
                                                                           0], np_data[..., 1]
                else:
                    np_value = np.random.normal(
                        start, end, size=shape).astype(random_dtype)

            elif distribution == "sample":
                assert not dtype.isComplex(), "complex type not support sample random"
                np_value = np.array(random.sample(range(start, end), end))

            elif distribution == "binomial":
                assert not dtype.isComplex(), "complex type not support binomial random"
                np_value = np.random.binomial(
                    start, end, size=shape).astype(random_dtype)

        if dtype.isFloatPoint() and self.tensor_.hasNanInf():
            np_value = quantize_utils.fillNanInf(
                np_value, dtype, self.tensor_.contain_nan_, self.tensor_.contain_inf_)

        return np_value, np_complex_value_real, np_complex_value_imag, position


class TensorList:
    '''
    TensorList is used for generating data for input tensors and saving them

    This class can be inherited by kernel and customized behavior for
    adjusting param and generating data. 

    Main process is in function fun.
    See more infomation in function preProcess and generateData.
    If you inherite this class, please use registerTensorList register it.

    If you want to transfer some params from json file to this class, you can 
    write params in keyword tensor_params.
    '''

    def __init__(self, params):
        self.params_ = params
        self.input_tensors_ = []
        self.output_tensors_ = []

    def __len__(self):
        '''
        return nums of tensors in tensorlist, contain input and output tensors
        '''
        return len(self.input_tensors) + len(self.output_tensors_)

    def __repr__(self):
        return (f"{DataNode.__module__}.{DataNode.__qualname__}:\n" +
                "input_tensors: \n" + str(self.input_tensors_) + "\n" +
                "output_tensors: \n" + str(self.output_tensors_))

    def run(self):
        '''
        Main process of TensorList
        Normally do not need rewrite this function
        '''
        self.preProcess()
        self.generateData()
        self.castDataNode()

    def preProcess(self):
        '''
        for adjusting param inherite this class and rewrite function preProcess
        '''
        for input_param in self.params_["inputs"]:
            self.input_tensors_.append(self.createTensor(input_param))
        for output_param in self.params_["outputs"]:
            self.output_tensors_.append(self.createTensor(output_param))

    def generateData(self):
        '''
        for adjusting param inherite this class and rewrite function generateData

        If tensor has attribute filename, data will be loaded from file,
        else will generate random data
        '''
        for input_tensor in self.input_tensors_:
            if input_tensor.filename_:
                shape = input_tensor.shape_
                dtype = input_tensor.getDataType()
                assert not dtype.isComplex(), "complex type do not support generate data from file"
                dtype_str = dtype.getNumpyStr()
                file_data = np.genfromtxt(
                    input_tensor.filename_, dtype=dtype_str).reshape(shape)
                input_tensor.getDataNode().setData(file_data)
            else:
                RandomData(input_tensor).random()

    def createTensor(self, param):
        '''
        Create tensor from param
        '''
        tensor = Tensor(**param)
        return tensor

    def castDataNode(self):
        '''
        Cast input data to onchip data by input_dtype and input_onchip_dtype
        '''
        for input_tensor in self.input_tensors_:
            quantize_utils.caseDataNode(
                input_tensor.onchip_datanode_, input_tensor.getDataNode())

    def getInputTensors(self):
        '''
        return input tensors in tensor list
        '''
        return self.input_tensors_

    def getInputTensor(self, index):
        '''
        return input tensor in tensor list by index
        '''
        return self.input_tensors_[index]

    def getOutputTensors(self):
        '''
        return output tensors in tensor list
        '''
        return self.output_tensors_

    def getOutputTensor(self, index):
        '''
        return output tensor in tensor list by index
        '''
        return self.output_tensors_[index]


class TensorListFactory:
    '''
    Registry for TensorList.
    '''
    registry = {}

    @classmethod
    def register(cls, name: str, register_cls: TensorList):
        '''
        Function for register TensorList
        '''
        if name not in cls.registry:
            print("[TensorListFactory]: register op ", name)
            cls.registry[name] = register_cls
        else:
            raise Exception(
                "[TensorListFactory]: register the same TensorList, please check op name. ")

    @classmethod
    def factory(cls, name: str) -> TensorList:
        '''
        Get TensorList from Factory by name. 
        '''
        if name in cls.registry:
            return cls.registry[name]
        else:
            return TensorList

    @classmethod
    def print(cls):
        print(cls.registry)


def registerTensorList(op_name=""):
    def register(cls: TensorList):
        if op_name:
            TensorListFactory.register(op_name, cls)
        else:
            raise Exception(
                '[TensorListFactory]: please use like @registerTensorList("op_name")')
        return cls
    return register
