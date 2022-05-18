from mlu_op_test_proto import mlu_op_test_pb2
from nonmlu_ops.base.tensor import *
import numpy as np
import time
import os
import struct
from functools import reduce


class str2ProtoParser():
    def __init__(self, proto):
        self.proto = proto

    def str2enum(self, str_name):
        switcher = {
            # DataType
            "float16": self.proto.DTYPE_HALF,
            "float32": self.proto.DTYPE_FLOAT,
            "double": self.proto.DTYPE_DOUBLE,
            "int8": self.proto.DTYPE_INT8,
            "int16": self.proto.DTYPE_INT16,
            "int32": self.proto.DTYPE_INT32,
            "int64": self.proto.DTYPE_INT64,
            "uint8": self.proto.DTYPE_UINT8,
            "uint16": self.proto.DTYPE_UINT16,
            "uint32": self.proto.DTYPE_UINT32,
            "uint64": self.proto.DTYPE_UINT64,
            "complex_half": self.proto.DTYPE_COMPLEX_HALF,
            "complex_float": self.proto.DTYPE_COMPLEX_FLOAT,
            "bool": self.proto.DTYPE_BOOL,

            # TensorLayout
            "NCHW": self.proto.LAYOUT_NCHW,
            "NHWC": self.proto.LAYOUT_NHWC,
            "HWCN": self.proto.LAYOUT_HWCN,
            "NDHWC": self.proto.LAYOUT_NDHWC,
            "ARRAY": self.proto.LAYOUT_ARRAY,
            "TNC": self.proto.LAYOUT_TNC,
            "NTC": self.proto.LAYOUT_NTC,
            "NCDHW": self.proto.LAYOUT_NCDHW,
            "NC": self.proto.LAYOUT_NC,
            "NLC": self.proto.LAYOUT_NLC,

            # Device
            "cpu": self.proto.CPU,
            "gpu": self.proto.GPU,

            # EvaluationCriterion
            "diff1": self.proto.DIFF1,
            "diff2": self.proto.DIFF2,
            "diff3": self.proto.DIFF3,
            "diff3_2": self.proto.DIFF3_2,
            "diff4": self.proto.DIFF4,

            # RandomDistribution
            "uniform": self.proto.UNIFORM,
            "gaussian": self.proto.GAUSSIAN,
            "sample": self.proto.SAMPLE,
            "binomial": self.proto.BINOMIAL,

            # MluPlatform
            "370": self.proto.MLU370,
            "200": self.proto.MLU200
        }
        return switcher.get(str_name, None)


def writeData2File(tensor_data, tensor_data_real, tensor_data_imag, dtype_str, file_name):
    # all data type save as binary mode in file
    if dtype_str in ["int8", "uint8", "bool"]:
        tensor_data.flatten().astype(np.int8).tofile(file_name)
    elif dtype_str in ["int16", "uint16"]:
        tensor_data.flatten().astype(np.int16).tofile(file_name)
    elif dtype_str in ["int32", "uint32"]:
        tensor_data.flatten().astype(np.int32).tofile(file_name)
    elif dtype_str in ["int64", "uint64"]:
        tensor_data.flatten().astype(np.int64).tofile(file_name)
    elif dtype_str == "float16":
        tensor_data.flatten().astype(np.float16).tofile(file_name)
    elif dtype_str == "float32":
        tensor_data.flatten().astype(np.float32).tofile(file_name)
    elif dtype_str == "double":
        tensor_data.flatten().astype(np.float64).tofile(file_name)
    elif dtype_str in ["complex_half"]:
        assert tensor_data_real.shape == tensor_data_imag.shape, "data real and imag shape must be same."
        concat_data = np.stack(
            (tensor_data_real, tensor_data_imag), -1).flatten()
        concat_data.astype(np.float16).to_file(file_name)
    elif dtype_str in ["complex_float"]:
        assert tensor_data_real.shape == tensor_data_imag.shape, "data real and imag shape must be same."
        concat_data = np.stack(
            (tensor_data_real, tensor_data_imag), -1).flatten()
        concat_data.astype(np.float32).to_file(file_name)
    else:
        raise ValueError("unsupported data type: " + dtype_str)


class ProtoWriter:
    def __init__(self, tensor_list, op_params, proto_params):
        '''
        ProtoWriter is used for dump result to prototxt or pb file.

        This class will deal tensor data and public params automatically.
        If your kernel has some proto param define by yourself. You need inherite
        this class ans rewrite function changeParam.

        Main process is in function dump2File.
        '''
        self.tensor_list_ = tensor_list
        self.op_params_ = op_params
        self.proto_params_ = proto_params
        self.folder_name_ = self.mkdirFolder()
        self.case_name_ = self.getCaseName()

    def dump2File(self):
        '''
        Main process of ProtoWriter, contains function dumpPublicParam2File,
        fuction dumpTensor2File and changeParam.
        '''
        self.dumpPublicParam2Node()
        self.dumpTensor2Node()
        self.dumpOpParam2Node()
        self.saveNode2File()

    def saveNode2File(self):
        '''
        Change params to pb params. If kernel own params for themselves,
        please rewrite this function.
        '''
        # save pb file
        file_name_pb = self.folder_name_ + "/" + self.case_name_ + ".pb"
        with open(file_name_pb, "wb") as f:
            serialize = self.proto_node_.SerializeToString()
            f.write(serialize)
            f.close()

        # save prototxt file
        if self.proto_params_.get("save_txt", False):
            file_name_txt = self.folder_name_ + "/" + self.case_name_ + ".prototxt"
            node1 = self.proto_node_
            node1.ParseFromString(serialize)
            with open(file_name_txt, "w") as f1:
                f1.write(str(node1))
                f1.close()
        else:
            pass

    def mkdirFolder(self):
        path_name = self.proto_params_.get("save_path")
        if not os.path.exists(path_name):
            os.mkdir(path_name)
        op_name = self.op_params_.get("op_name")
        if self.proto_params_.get("supported_mlu_platform"):
            platform_name = "_".join(
                self.proto_params_.get("supported_mlu_platform"))
            folder_name = path_name + "/" + platform_name + "/" + op_name
        else:
            folder_name = path_name + "/default_platform/" + op_name

        if os.path.exists(folder_name):
            pass
        else:
            os.makedirs(folder_name)
        return folder_name

    def getCaseName(self):
        case_name = self.op_params_.get("op_name")
        if self.proto_params_.get("large_tensor", False):
            case_name = case_name + "_" + "data_split_"
        else:
            case_name = case_name + "_" + "data_included_"

        input_tensors = self.tensor_list_.getInputTensors()
        output_tensors = self.tensor_list_.getOutputTensors()
        for tensor in input_tensors:
            if tensor.hasNanInf():
                case_name = case_name + "NanInf_"
                break
        dtype_str = ""
        offchip_strs = []
        onchip_strs = []
        for tensor in input_tensors + output_tensors:
            offchip_strs.append(tensor.datanode_.dtype_.getStr())
            if tensor.onchip_datanode_.dtype_ != DataType.UNSET:
                onchip_strs.append(tensor.onchip_datanode_.dtype_.getStr())
        offchip_strs = list(set(offchip_strs))
        onchip_strs = list(set(onchip_strs))
        dtype_str += "_".join(offchip_strs) + "_"
        if len(onchip_strs) > 0:
            dtype_str += "_".join(onchip_strs) + "_"

        time_flag = str(int(round(time.time() * 1000)))
        case_name = case_name + dtype_str + time_flag
        return case_name

    def dumpTensor2Node(self):
        '''
        Write input tensor to proto node
        '''
        for id, input_tensor in enumerate(self.tensor_list_.getInputTensors()):
            input_node = self.proto_node_.input.add()
            input_name = input_tensor.name_
            if input_name == "":
                input_name = "input" + str(id + 1)
            self.writeTensor(input_node, input_name, input_tensor)

        '''
        Write output tensor to proto node
        '''
        for id, output_tensor in enumerate(self.tensor_list_.getOutputTensors()):
            output_node = self.proto_node_.output.add()
            output_name = output_tensor.name_
            if output_name == "":
                output_name = "output" + str(id + 1)
            self.writeTensor(output_node, output_name, output_tensor)
            if self.op_params_.get("if_dynamic_threshold", False):
                for diff in self.proto_params_.get("evaluation_criterion", []):
                    if diff == "diff1":
                        diff = output_tensor.getDiff1()
                    elif diff == "diff2":
                        diff = output_tensor.getDiff2()
                    elif diff == "diff3":
                        diff = output_tensor.getDiff3()
                    elif diff == "diff4":
                        diff = output_tensor.getDiff4()
                    else:
                        raise Exception("not support diff way:" + diff)
                    real_diff = diff.real_diff_
                    imag_diff = diff.imag_diff_
                    output_node.thresholds.evaluation_threshold.append(
                        real_diff)
                    if imag_diff >= 0:
                        output_node.thresholds.evaluation_threshold_imag.append(
                            imag_diff)

    def dumpPublicParam2Node(self):
        self.proto_node_.op_name = self.op_params_.get("op_name", "op")
        if self.proto_params_.get("device", None):
            self.proto_node_.device = self.str_parser_.str2enum(
                self.proto_params_.get("device"))
        if self.proto_params_.get("supported_mlu_platform", None):
            for platform in self.proto_params_.get("supported_mlu_platform"):
                self.proto_node_.supported_mlu_platform.append(
                    self.str_parser_.str2enum(platform))

        # write diff func and threshold
        for diff in self.proto_params_.get("evaluation_criterion", []):
            self.proto_node_.evaluation_criterion.append(
                self.str_parser_.str2enum(diff))

        if not self.op_params_.get("if_dynamic_threshold", False):
            evaluation_criterion_num = len(
                self.proto_params_.get("evaluation_criterion", []))
            evaluation_threshold_num = len(
                self.proto_params_.get("evaluation_threshold", []))
            if evaluation_criterion_num == evaluation_threshold_num:
                for threshold in self.proto_params_.get("evaluation_threshold", []):
                    self.proto_node_.evaluation_threshold.append(threshold)
            elif evaluation_criterion_num * 2 == evaluation_threshold_num:
                evaluation_threshold_list = self.proto_params_.get(
                    "evaluation_threshold", [])
                for i in range(evaluation_criterion_num):
                    self.proto_node_.evaluation_threshold.append(
                        evaluation_threshold_list[2 * i])
                    self.proto_node_.evaluation_threshold_imag.append(
                        evaluation_threshold_list[2 * i + 1])
            else:
                raise Exception(
                    "criterion num not match threshold num, please check input json.")

    def dumpOpParam2Node(self):
        '''
        Change params to pb params. If kernel own params for themselves,
        please rewrite this function
        '''
        pass

    def writeData2Node(self, tensor_node, tensor_data, tensor_data_real, tensor_data_imag, dtype_str):
        # store float data as int to speed up gtest parser pb/prototxt file and save store space
        tensor_data = np.ascontiguousarray(tensor_data)
        if dtype_str in ["int8", "int16", "int32"]:
            tensor_node.value_i.extend(tensor_data.flatten())
        elif dtype_str in ["uint32", "uint16", "uint8"]:
            tensor_node.value_ui.extend(tensor_data.flatten())
        elif dtype_str == "int64":
            tensor_node.value_l.extend(tensor_data.flatten())
        elif dtype_str == "uint64":
            tensor_node.value_ul.extend(tensor_data.flatten())
        elif dtype_str == "bool":
            tensor_node.value_i.extend(tensor_data.flatten().astype(int))
        elif dtype_str == "float16":
            tensor_node.value_i.extend(np.frombuffer(tensor_data, np.int16))
        elif dtype_str == "float32":
            tensor_node.value_i.extend(np.frombuffer(tensor_data, np.int32))
        elif dtype_str == "double":
            tensor_node.value_l.extend(np.frombuffer(tensor_data, np.int64))

        elif dtype_str == "complex_half":
            assert tensor_data_real.shape == tensor_data_imag.shape, "data real and imag shape must be same. "
            concat_data = np.stack(
                (tensor_data_real, tensor_data_imag), -1).flatten()
            tensor_node.value_i.extend(np.frombuffer(concat_data, np.int16))
        elif dtype_str == "complex_float":
            assert tensor_data_real.shape == tensor_data_imag.shape, "data real and imag shape must be same. "
            concat_data = np.stack(
                (tensor_data_real, tensor_data_imag), -1).flatten()
            tensor_node.value_i.extend(np.frombuffer(concat_data, np.int32))
        else:
            raise ValueError("unsupported data type: " + dtype_str)

    def writeTensor(self, tensor_node, tensor_name, tensor):
        tensor_node.id = tensor_name
        tensor_node.layout = self.str_parser_.str2enum(
            str(tensor.layout_.getStr()))

        # write shape
        for dim in tensor.shape_:
            tensor_node.shape.dims.append(int(dim))

        # write tensor stride
        if tensor.strides_ is not None:
            for stride in tensor.strides_:
                tensor_node.shape.dim_stride.append(int(stride))

        # write offchip data type and data value
        data_type_str = tensor.datanode_.dtype_.getStr()
        tensor_node.dtype = self.str_parser_.str2enum(data_type_str)

        tensor_value = tensor.datanode_.data_
        tensor_value_real = tensor.datanode_.data_real_
        tensor_value_imag = tensor.datanode_.data_imag_
        if tensor.datanode_.dtype_.isComplex():
            data_exist = len(tensor_value_real) > 0 and len(
                tensor_value_imag) > 0
        else:
            data_exist = len(tensor_value) > 0
        if tensor.require_value_ and data_exist:
            tensor_data = np.array(tensor_value)
            if self.proto_params_.get("large_tensor", False):
                data_file_name = self.case_name_ + "_" + tensor_name
                tensor_node.path = data_file_name
                writeData2File(tensor_data, tensor_value_real, tensor_value_imag,
                               data_type_str, self.folder_name_ + "/" + data_file_name)
            else:
                self.writeData2Node(
                    tensor_node, tensor_data, tensor_value_real, tensor_value_imag, data_type_str)

        if tensor.onchip_datanode_.dtype_ != DataType.UNSET:
            tensor_node.onchip_dtype = self.str_parser_.str2enum(
                tensor.onchip_datanode_.dtype_.getStr())

        # write distribution
        if tensor.random_distribution_:
            random_mode = list(tensor.random_distribution_.items())[0]
            distribution = random_mode[0]
            value_range = random_mode[1]
            tensor_node.random_data.distribution = self.str_parser_.str2enum(
                distribution)
            if distribution == "gaussian":
                tensor_node.random_data.mu_double = value_range[0]
                tensor_node.random_data.sigma_double = value_range[1]
            else:
                tensor_node.random_data.lower_bound_double = value_range[0]
                tensor_node.random_data.upper_bound_double = value_range[1]


class MluOpProtoWriter(ProtoWriter):
    def __init__(self, tensor_list, op_params, proto_params):
        super(MluOpProtoWriter, self).__init__(
            tensor_list, op_params, proto_params)
        self.proto_node_ = mlu_op_test_pb2.Node()
        self.str_parser_ = str2ProtoParser(mlu_op_test_pb2)


class ProtoWriterFactory:
    '''
    Registry for ProtoWriter
    '''
    registry = {}

    @classmethod
    def register(cls, name: str, register_cls: ProtoWriter):
        '''
        Function for register ProtoWriter
        '''
        if name not in cls.registry:
            print("[ProtoWriterFactory]: register op ", name)
            cls.registry[name] = register_cls
        else:
            raise Exception(
                '[ProtoWriterFactory]: register same ProtoWriter, please check op name.')

    @classmethod
    def factory(cls, name: str) -> ProtoWriter:
        '''
        Get ProtoWriter from Factory by name.
        '''
        if name in cls.registry:
            return cls.registry[name]
        else:
            return ProtoWriter

    @classmethod
    def print(cls):
        print(cls.registry)


def registerProtoWriter(op_name=""):
    def register(cls: ProtoWriter):
        if op_name:
            ProtoWriterFactory.register(op_name, cls)
        else:
            raise Exception(
                '[ProtoWriterFactory]: illegal op name, please use like @registerOp("op_name")')
        return cls
    return register
