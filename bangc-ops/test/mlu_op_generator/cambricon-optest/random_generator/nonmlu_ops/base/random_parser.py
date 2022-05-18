import os
import json
import sys
import itertools
import random
import numpy as np
from abc import abstractmethod
from random_utils.utils import *


class RandomParser(object):
    def __init__(self, file):
        self.file = file
        self.inputs = []
        self.op_params = []
        self.outputs = []
        self.proto_params = []
        self.input_key_num = []
        self.input_repeat_num = []
        self.output_key_num = []
        self.iter_params = []
        self.op_param_num = 0
        self.proto_param_num = 0
        self.loadJson2seq()

    @abstractmethod
    def loadJson2seq(self):
        with open(self.file) as f:
            self.config_list = json.load(f)

    @abstractmethod
    def generateRandomInputs(self):
        for single_input in self.config_list["inputs"]:
            input_key_num = len(single_input.keys())
            input_shapes = getRandomParamList(single_input)
            if "repeat" in single_input:
                input_key_num -= 1
                repeat_num = random.randint(
                    single_input["repeat"][0], single_input["repeat"][1])
                same_input = []
                for index in range(repeat_num):
                    same_input.append(input_shapes)
                self.inputs.append(same_input)
                self.input_repeat_num.append(repeat_num)
            else:
                self.inputs.append(input_shapes)
                self.input_repeat_num.append(1)
            self.input_key_num.append(input_key_num)

        remove_keys = ["total_random", "part_random", "shape", "repeat"]
        input_index = 0
        for single_input in self.config_list["inputs"]:
            for key in single_input.keys():
                if key not in remove_keys:
                    if isinstance(single_input[key], list):
                        tmp_list = []
                        for value in single_input[key]:
                            tmp_dict = {key: value}
                            tmp_list.append(tmp_dict)
                        self.iter_params.append(tmp_list)
                    else:
                        tmp_list = [{key: single_input[key]}]
                        self.iter_params.append(tmp_list)
            shape_dict_list = []
            if isinstance(self.inputs[input_index][0], list):  # repeat num > 1
                for value in self.inputs[input_index][0]:
                    tmp_dict = {"shape": value}
                    shape_dict_list.append(tmp_dict)
            else:
                for value in self.inputs[input_index]:
                    tmp_dict = {"shape": value}
                    shape_dict_list.append(tmp_dict)
            self.iter_params.append(shape_dict_list)
            input_index = input_index + 1

    @abstractmethod
    def generateRandomOpParams(self):
        for single_param in self.config_list["op_params"]:
            for key in single_param.keys():
                param_list = getRandomParamList(single_param[key])
                self.op_params.append(param_list)
                self.op_param_num += 1
        param_index = 0
        for single_param in self.config_list["op_params"]:
            for key in single_param.keys():
                param_dict_list = []
                for value in self.op_params[param_index]:
                    param_dict = {key: value}
                    param_dict_list.append(param_dict)
                self.iter_params.append(param_dict_list)
                param_index += 1

    @abstractmethod
    def generatorRandomOutputs(self):
        self.outputs = self.config_list["outputs"]
        for output in self.outputs:
            self.output_key_num.append(len(output.keys()))

        for output in self.outputs:
            for key in output.keys():
                if isinstance(output[key], list):
                    tmp_list = []
                    for value in output[key]:
                        tmp_dict = {key: value}
                        tmp_list.append(tmp_dict)
                    self.iter_params.append(tmp_list)
                else:
                    tmp_list = [{key: output[key]}]
                    self.iter_params.append(tmp_list)

    @abstractmethod
    def generatorRandomProtoParams(self):
        self.proto_params = [self.config_list["proto_params"]]
        self.proto_param_num = len(self.config_list["proto_params"].keys())
        self.iter_params.append(self.proto_params)

    @abstractmethod
    def generateCaseByCase(self):
        assert("inputs" in self.config_list)
        self.generateRandomInputs()

        if "outputs" in self.config_list:
            self.generatorRandomOutputs()

        if "op_params" in self.config_list:
            self.generateRandomOpParams()

        if "proto_params" in self.config_list:
            self.generatorRandomProtoParams()
        print("all iter params as follow:")
        for item in self.iter_params:
            print(item)

        random_cases = list(itertools.product(*(self.iter_params)))
        self.combination_cases = []
        for i in range(len(random_cases)):
            single_case = {"inputs": []}
            inputs = []
            cur_index = 0
            for input_index in range(len(self.config_list["inputs"])):
                input_temp = random_cases[i][cur_index:cur_index +
                                             self.input_key_num[input_index]]
                input_dict = {}
                for item in input_temp:
                    input_dict.update(item)
                for j in range(self.input_repeat_num[input_index]):
                    single_case["inputs"].append(input_dict)
                cur_index = cur_index + self.input_key_num[input_index]

            if "outputs" in self.config_list:
                single_case["outputs"] = []
                for output_index in range(len(self.config_list["outputs"])):
                    output_temp = random_cases[i][cur_index:cur_index +
                                                  self.output_key_num[output_index]]
                    output_dict = {}
                    for item in output_temp:
                        output_dict.update(item)
                    single_case["outputs"].append(output_dict)
                    cur_index = cur_index + self.output_key_num[output_index]

            if "op_params" in self.config_list:
                single_case["op_params"] = {}
                for item in random_cases[i][cur_index:cur_index + self.op_param_num]:
                    for key in item.keys():
                        single_case["op_params"][key] = item[key]
                cur_index = cur_index + self.op_param_num

            if "proto_params" in self.config_list:
                single_case["proto_params"] = {}
                for item in random_cases[i][cur_index:cur_index + self.proto_param_num]:
                    for key in item.keys():
                        single_case["proto_params"][key] = item[key]
                cur_index = cur_index + self.proto_param_num
            assert(cur_index == len(list(random_cases[i])))
            self.combination_cases.append(single_case)

    @abstractmethod
    def generateManmulJson(self, save_file_name):
        self.generateCaseByCase()
        manual_config = {}
        random_keys = ["inputs", "outputs", "op_params", "proto_params"]
        for key in self.config_list.keys():
            if key not in random_keys:
                manual_config[key] = self.config_list[key]
        manual_config["manual_data"] = []
        for case in self.combination_cases:
            manual_config["manual_data"].append(case)
        convert2str_format = json.dumps(manual_config, indent=3)
        with open(save_file_name, "w+") as file:
            file.write(convert2str_format)
            file.close()


class RandomParserFactory:
    '''
    Registry for RandomParser.
    '''
    registry = {}

    @classmethod
    def register(cls, name: str, register_cls: RandomParser):
        '''
        Function for register RandomParser
        '''
        if name not in cls.registry:
            print("[RandomParserFactory]: register op", name)
            cls.registry[name] = register_cls
        else:
            raise Exception(
                '[RandomParserFactory]: register the same TensorList, please check op name.')

    @classmethod
    def factory(cls, name: str) -> RandomParser:
        '''
        Get RandomParser from Factory by name
        '''
        if name in cls.registry:
            return cls.registry[name]
        else:
            return RandomParser

    @classmethod
    def print(cls):
        print(cls.registry)


def registerRandomParser(op_name=""):
    def register(cls: RandomParser):
        if op_name:
            RandomParserFactory.register(op_name, cls)
        else:
            raise Exception(
                '[RandomParserFactory]: illegal op name, please use like @registerOp("op_name")')
        return cls
    return register
