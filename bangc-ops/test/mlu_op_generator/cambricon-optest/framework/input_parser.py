import json
import copy
from pathlib import Path
from nonmlu_ops import *
from mlu_op_test_proto import mlu_op_test_pb2
from google.protobuf import text_format


class InputArgsParser():
    '''
    Class for parser input args to tensor params, op params and proto params
    '''

    def __init__(self, args):
        self.save_txt_ = args.save_txt
        self.save_path_ = args.save_path

    def parser(self):
        arg_params = {}
        tensor_params = {}
        op_params = {}
        proto_params = {}
        proto_params["save_txt"] = self.save_txt_
        proto_params["save_path"] = self.save_path_
        arg_params["tensor_params"] = tensor_params
        arg_params["op_params"] = op_params
        arg_params["proto_params"] = proto_params
        return arg_params


class InputFileParser():
    '''
    Class for parser input file to tensor params, op params and proto params.
    Only support manual json now.
    Usage:
        parser = InputFileParser(filename,params)
        cases = parser.genCases() or
        case = parser.genCase()
    '''

    def __init__(self, filename: str, file_type,
                 params={"tensor_params": {}, "op_params": {}, "proto_params": {}}):
        self.file_name_ = filename
        self.base_case_ = params
        self.file_type_ = file_type
        self.global_tensor_param_ = {}
        self.global_op_param_ = {}
        self.global_proto_param_ = {}
        self.global_unknown_param_ = {}
        self.manual_params_ = []
        self.case_num_ = 0
        self.case_count_ = 0

    def __repr__(self):
        return (f"{InputFileParser.__module__}.{InputFileParser.__qualname__}"
                f"filename: {self.file_name_}, case_num: {self.case_num_}")


class InputFileParser_json(InputFileParser):
    def __init__(self, filename, file_type, params):
        super().__init__(filename, file_type, params)
        self.json_ = {}
        self.parser_json()

    def parser_json(self):
        with open(self.file_name_, "r") as file:
            self.json_ = json.load(file)
        self.parseGlobalParam()
        self.parseManualParam()

    def parseGlobalParam(self):
        '''
        Parser global param from json file.
        If you want parser some spicial keys to specified global param dict.
        Please update keyword into appropriate list.
        PS: global_op_param_key, global_op_param_key, global_proto_param_key.
        '''
        self.global_tensor_param_.update(self.json_.get("tensor_params", {}))
        self.global_op_param_.update(self.json_.get("op_params", {}))
        self.global_proto_param_.update(self.json_.get("proto_params", {}))
        global_tensor_param_key = ["data_type", "random_distribution",
                                   "require_value", "if_scale", "if_offset", "contain_nan", "contain_inf"]
        global_op_param_key = [
            "op_name", "if_dynamic_threshold", "supported_mlu_platform"]
        global_proto_param_key = ["device", "evaluation_criterion",
                                  "evaluation_threshold", "supported_mlu_platform", "large_tensor"]
        for json_key in self.json_:
            if json_key in global_tensor_param_key:
                self.global_tensor_param_[json_key] = self.json_[json_key]
            if json_key in global_op_param_key:
                self.global_op_param_[json_key] = self.json_[json_key]
            if json_key in global_proto_param_key:
                self.global_proto_param_[json_key] = self.json_[json_key]

    def parseManualParam(self):
        if "manual_data" in self.json_:
            self.manual_params_ = self.json_["manual_data"]
            self.case_num_ = len(self.manual_params_)
        else:
            raise Exception(f"{self.file_name_} do not have manual data")

    def genCases(self):
        '''
        return a cases list parser from input manual json
        '''
        cases = []
        while self.case_count_ < self.case_num_:
            cases.append(self.genCase())
        return cases

    def genCase(self):
        '''
        Return a case parser from input manual json.
        If call this function repeatedly, it will return all cases
        which can parser from the input manual json. Then it will return
        an empty dict.
        '''
        if self.case_count_ >= self.case_num_:
            return {}
        case = copy.deepcopy(self.base_case_)
        # use local tensor param update global tensor param
        case_tensor_param = {"inputs": [], "outputs": []}
        inputs = self.manual_params_[self.case_count_].get("inputs")
        input_default_dtype = ["unset" for _ in range(len(inputs))]
        for i in range(len(inputs)):
            input_param = copy.deepcopy(self.global_tensor_param_)
            if "data_type" in input_param:
                data_type = input_param.pop("data_type")
            else:
                data_type = {}
            input_param.update(inputs[i])
            input_param.setdefault("dtype", data_type.get(
                "input_dtype", input_default_dtype)[i])
            input_param.setdefault("onchip_dtype", data_type.get(
                "input_onchip_dtype", input_default_dtype)[i])
            if "file_name" in input_param:
                input_param["file_name"] = str(
                    Path(self.file_name_).with_name(input_param["file_name"]))
            case_tensor_param["inputs"].append(input_param)

        outputs = self.manual_params_[self.case_count_].get("outputs", [])
        output_default_dtype = ["unset" for _ in range(len(outputs))]
        for i in range(len(outputs)):
            output_param = copy.deepcopy(self.global_tensor_param_)
            if "data_type" in output_param:
                data_type = output_param.pop("data_type")
            else:
                data_type = {}
            output_param.update(outputs[i])
            output_param.setdefault("dtype", data_type.get(
                "output_dtype", output_default_dtype)[i])
            output_param.setdefault("onchip_dtype", data_type.get(
                "output_onchip_dtype", output_default_dtype)[i])
            if "file_name" in output_param:
                output_param["file_name"] = str(output_param["file_name"])
            case_tensor_param["outputs"].append(output_param)
        case["tensor_params"].update(case_tensor_param)
        # use local op param update global op param
        case_op_param = {}
        case_op_param.update(self.global_op_param_)
        case_op_param.update(self.manual_params_[
                             self.case_count_].get("op_params", {}))
        case["op_params"].update(case_op_param)
        # use local proto param update global proto param
        case_proto_param = {}
        case_proto_param.update(self.global_proto_param_)
        case_proto_param.update(
            self.manual_params_[self.case_count_].get("proto_params", {}))
        case["proto_params"].update(case_proto_param)
        # add case count in multi thread need lock
        self.case_count_ = self.case_count_ + 1
        return case


class InputFileParser_prototxt(InputFileParser):
    def __init__(self, filename, file_type, params):
        super().__init__(filename, file_type, params)
        self.parser_prototxt()

    def parser_prototxt(self):
        with open(self.file_name_, "r") as openfile:
            file_str = openfile.read()
            node_pro = mlu_op_test_pb2.Node()
            self.get_parse_infor = text_format.Parse(file_str, node_pro)

    def parsePrototxtParam(self):
        '''
        Return a case parser from input prototxt file.
        '''
        self.global_tensor_param_.update({"tensor_params": {}})
        self.global_op_param_.update({"op_params": {}})
        self.global_proto_param_.update({"proto_params": {}})
        case = copy.deepcopy(self.base_case_)
        case_tensor_param = {"inputs": [], "outputs": []}
        inputs = self.get_parse_infor.input
        for index, val in enumerate(inputs):
            input_param = {}
            random_distribution = {}
            random_distribution_type = enum2str(
                mlu_op_test_pb2.RandomDistribution.Name(val.random_data.distribution))
            random_distribution_low_up = [
                val.random_data.lower_bound, val.random_data.upper_bound]
            random_distribution.update(
                {random_distribution_type: random_distribution_low_up})
            input_param["random_distribution"] = random_distribution
            input_param["require_value"] = True
            input_param["shape"] = val.shape.dims
            input_param["dtype"] = enum2str(
                mlu_op_test_pb2.DataType.Name(val.dtype))
            input_param["onchip_dtype"] = enum2str(
                mlu_op_test_pb2.DataType.Name(val.onchip_dtype))
            input_param["layout"] = enum2str(
                mlu_op_test_pb2.TensorLayout.Name(val.layout))
            case_tensor_param["inputs"].append(input_param)
        outputs = self.get_parse_infor.output
        for index, val_out in enumerate(outputs):
            output_param = {}
            random_distribution = {}
            random_distribution_type = enum2str(
                mlu_op_test_pb2.RandomDistribution.Name(val_out.random_data.distribution))
            random_distribution_low_up = [
                val_out.random_data.lower_bound, val_out.random_data.upper_bound]
            random_distribution.update(
                {random_distribution_type: random_distribution_low_up})
            output_param["random_distribution"] = random_distribution
            output_param["require_value"] = True
            output_param["shape"] = val_out.shape.dims
            output_param["dtype"] = enum2str(
                mlu_op_test_pb2.DataType.Name(val_out.dtype))
            output_param["onchip_dtype"] = enum2str(
                mlu_op_test_pb2.DataType.Name(val_out.onchip_dtype))
            output_param["layout"] = enum2str(
                mlu_op_test_pb2.TensorLayout.Name(val_out.layout))
            case_tensor_param["outputs"].append(output_param)
        case_op_param = {}
        case_op_param["op_name"] = (self.get_parse_infor.op_name).lower()
        case_proto_param = {}
        case_proto_param["device"] = enum2str(mlu_op_test_pb2.Device.Name(
            self.get_parse_infor.test_param.baseline_device))
        case_proto_param["support_mlu_platform"] = ['200', '370']
        case_proto_param["run_mode"] = ['online']
        case_proto_param["large_tensor"] = False
        evaluation_criterion = []
        for i in self.get_parse_infor.test_param.error_func:
            evaluation_criterion.append(
                enum2str(mlu_op_test_pb2.EvaluationCriterion.Name(i)))
        case_proto_param['evaluation_criterion'] = evaluation_criterion
        case_proto_param['evaluation_threshold'] = self.get_parse_infor.test_param.error_threshold
        case_proto_param['write_data'] = True
        case = ProtoReadFactory().factory(case_op_param["op_name"])(
            self.get_parse_infor, case_tensor_param, case_op_param, case_proto_param, case).run()
        return case


def enum2str(proto_name):
    switcher = {
        # DataType
        "DTYPE_HALF": "float16",
        "DTYPE_FLOAT": "float32",
        "DTYPE_INT8": "int8",
        "DTYPE_INT16": "int16",
        "DTYPE_INT32": "int32",
        "DTYPE_DOUBLE": "double",
        "DTYPE_UINT8": "uint8",
        "DTYPE_UINT16": "uint16",
        "DTYPE_UINT32": "uint32",
        "DTYPE_BOOL": "bool",

        # TensorLayout
        "LAYOUT_NCHW": "NCHW",
        "LAYOUT_NHWC": "NHWC",
        "LAYOUT_HWCN": "HWCN",
        "LAYOUT_NDHWC": "NDHWC",
        "LAYOUT_TNC": "TNC",
        "LAYOUT_NTC": "NTC",
        "LAYOUT_NCDHW": "NCDHW",
        "LAYOUT_ARRAY": "ARRAY",
        "LAYOUT_NC": "NC",
        "LAYOUT_NLC": "NLC",

        # Device
        "CPU": "cpu",
        "GPU": "gpu",

        # EvaluationCritertion
        "DIFF1": "diff1",
        "DIFF2": "diff2",
        "DIFF3": "diff3",
        "DIFF3_2":"diff3_2",
        "DIFF4": "diff4",

        # RandomDistribution
        "UNIFORM": "uniform",
        "GAUSSIAN": "gaussian",
        "SAMPLE": "sample",
        "BINOMIAL": "binomial",

        # MluPlatform
        "MLU370": "370",
        "MLU200": "200"
    }
    return switcher.get(proto_name)
