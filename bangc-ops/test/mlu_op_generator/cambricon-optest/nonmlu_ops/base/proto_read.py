from abc import ABC, abstractmethod
from nonmlu_ops.base.test_param import *


class ProtoRead(ABC):
    '''
    ProtoRead is used to get arguments from different operator.

    This class must be inherited by kernel.
    When you inherite this class, please use function registerProtoRead register it.

    If you want to transfer some params from prototxt file to this class, you can
    write params in keyword op_params.
    '''

    def __init__(self, proto_node, tensor_params, op_params, proto_params, case):
        self.proto_node_ = proto_node
        self.tensor_params_ = tensor_params
        self.op_params_ = op_params
        self.proto_params_ = proto_params
        self.case_ = case
        self.protoParamParse()

    def run(self):
        self.case_["tensor_params"].update(self.tensor_params_)
        self.case_["op_params"].update(self.op_params_)
        self.case_["proto_params"].update(self.proto_params_)
        return self.case_

    def protoParamParse(self):
        pass


class ProtoReadFactory:
    '''
    Registry for ProtoRead
    '''
    registry = {}

    @classmethod
    def register(cls, name: str, register_cls: ProtoRead):
        '''
        Function for register ProtoRead
        '''
        if name not in cls.registry:
            print("[ProtoReadFactory]: register op ", name)
            cls.registry[name] = register_cls
        else:
            raise Exception(
                '[ProtoReadFactory]: register same ProtoRead, please check op name.')

    @classmethod
    def factory(cls, name: str) -> ProtoRead:
        '''
        Get ProtoRead from Factory by name
        '''
        if name not in cls.registry:
            raise KeyError(
                "[ProtoReadFactory]: own ProtoRead is not registed. please use @registerRead")
        return cls.registry[name]

    @classmethod
    def print(cls):
        print(cls.registry)


def registerRead(op_name=""):
    def register(cls: ProtoRead):
        if op_name:
            ProtoReadFactory.register(op_name, cls)
        else:
            raise Exception(
                '[ProtoReadFactory]: illegal op name, please use like @registerRead("op_name").')
        return cls
    return register
