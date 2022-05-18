import os
import sys
import shutil
from pathlib import Path
from abc import ABC, abstractmethod
from nonmlu_ops import *
from nonmlu_ops.base.proto_writer import MluOpProtoWriter
from .input_parser import *
from utils import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Builder(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self):
        pass


class MluOpBuilder(Builder):
    '''
    MluOpBuilder is used for generating cases for mluop, called by Director.
    '''

    def __init__(self, *args, **kwargs):
        self.args_ = args[0]
        self.arg_params_ = {}
        self.file_dict_ = {}

    def registerOp2Factory(self):
        cambricon_dir = str(Path(__file__).parent.parent.resolve())
        sys.path.append(cambricon_dir)
        for register_op_name in self.args_.op_name:
            register_op_dir = Path(
                cambricon_dir + "/nonmlu_ops/" + register_op_name)
            register_op_pys = register_op_dir.rglob("*.py")
            for file in register_op_pys:
                filedir = str(file.resolve())
                filename = file.name
                if not filename.startswith("_"):
                    reldir = os.path.relpath(filedir, cambricon_dir)
                    modulename, ext = os.path.splitext(reldir)
                    print("modulename", modulename)
                    importname = modulename.replace("/", ".")
                    __import__(importname)

    def run(self):
        self.registerOp2Factory()
        if not os.environ.get('CUDA_VISIBLE_DEVICES') and utils.available_nvidia_smi():
            os.environ['CUDA_VISIBLE_DEVICES'] = str(utils.available_GPU())
        self.clearOldCases()
        self.arg_params_ = InputArgsParser(self.args_).parser()
        if self.args_.file_type == "json":
            if self.args_.op_name:
                self.processJsonFile()
            else:
                raise Exception(
                    "--op_name option cannot be empty when the input file is json")
        else:
            self.processPrototxtFile()

    def processJsonFile(self):
        '''
        Process json input file.
        '''
        if self.args_.json_file:
            assert(len(self.args_.op_name) == 1)
            self.file_dict_[self.args_.json_file] = self.args_.op_name[0]
        else:
            for op_name in self.args_.op_name:
                op_json_path = self.args_.json_path + "/" + op_name
                if self.args_.json_path == "./manual_config":
                    proto_writer = ProtoWriterFactory().factory(op_name)
                    if issubclass(proto_writer, MluOpProtoWriter):
                        op_json_path = self.args_.json_path + "/mlu_ops/" + op_name
                    else:
                        raise Exception(
                            "No corresponding protoWriter is registered, please check you proto_writer!")
                op_file_dict = {}.fromkeys(
                    utils.getJsonFileFromDir(op_json_path), op_name)
                self.file_dict_.update(op_file_dict)

        if not self.file_dict_:
            raise Exception("No json file, please check op name and file dir.")
        for file, op_name_register in self.file_dict_.items():
            input_parser = InputFileParser_json(
                file, self.args_.file_type, self.arg_params_)
            print("begin to generate cases of " + file)
            if self.args_.precheckin:
                random_case = input_parser.genCases()[0]
                MluOpGenerator(op_name_register, random_case).run()
            else:
                for case_param in input_parser.genCases():
                    MluOpGenerator(op_name_register, case_param).run()
        print("Test cases were generated successfully")

    def processPrototxtFile(self):
        '''
        Process prototxt input file
        '''
        op_prototxt_path = self.args_.prototxt_path + "/"
        op_file_dict = {}.fromkeys(
            utils.getPrototxtFileFromDir(op_prototxt_path))
        self.file_dict_.update(op_file_dict)
        if not self.file_dict_:
            raise Exception(
                "No prototxt file, please check op name and file dir.")
        for file, op_name_register in self.file_dict_.items():
            input_parser = InputFileParser_prototxt(
                file, self.args_.file_type, self.args_params_)
            case_param = input_parser.parsePrototxtParam()
            MluOpGenerator(op_name_register, case_param).run()
            print("Test cases  were generated successfully!")

    def clearOldCases(self):
        '''
        Before generate cases, use this function to remove save_path folder.
        '''
        if os.path.exists(self.args_.save_path):
            shutil.rmtree(self.args_.save_path, ignore_errors=True)
            print("Begin to delete old cases")


class MluOpGenerator:
    def __init__(self, opname_register, case_param, **kwargs):
        self.opname_register_ = opname_register
        self.tensor_param_ = case_param["tensor_params"]
        self.op_param_ = case_param["op_params"]
        self.proto_param_ = case_param["proto_params"]

    def run(self):
        op_name = self.op_param_["op_name"]
        tensor_list = TensorListFactory().factory(op_name)(self.tensor_param_)
        tensor_list.run()
        op_test = OpTestFactory().factory(op_name)(tensor_list, self.op_param_)
        op_test.run()
        proto_writer = ProtoWriterFactory().factory(op_name)(
            tensor_list, self.op_param_, self.proto_param_)
        proto_writer.dump2File()
