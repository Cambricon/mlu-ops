from abc import ABC, abstractmethod
from nonmlu_ops.base.test_param import *


class OpTest(ABC):
    '''
    OpTest is used for kernel computing.

    This class must be inherited by kernel. And rewrite function compute.
    when you inherite this class, please use function registerOp register it.

    If you want to transfer some params from json file to this class, you can
    write params in keyword op_params.
    '''

    def __init__(self, tensor_list, params):
        self.tensor_list_ = tensor_list
        self.params_ = params
        self.test_param_ = None

    def paramCheck(self):
        pass

    @abstractmethod
    def compute(self) -> TestData:
        pass

    def run(self):
        self.paramCheck()
        self.compute()

    @property
    def input_tensors(self):
        return self.tensor_list_.getInputTensors()

    @property
    def output_tensors(self):
        return self.tensor_list_.getOutputTensors()


class OpTestFactory:
    '''
    Registry for OpTest.
    '''
    registry = {}

    @classmethod
    def register(cls, name: str, register_cls: OpTest):
        '''
        Function for register OpTest
        '''
        if name not in cls.registry:
            print("[OpTestFactory]: register_op ", name)
            cls.registry[name] = register_cls
        else:
            raise Exception(
                '[OpTestFactory]: register the same op, please check op name.')

    @classmethod
    def factory(cls, name: str) -> OpTest:
        '''
        Get OpTest from Factory by name
        '''
        if name not in cls.registry:
            raise KeyError(
                "[OpTestFactory]: op are not registed, please use @registerOp.")
        return cls.registry[name]

    @classmethod
    def print(cls):
        print(cls.registry)


def registerOp(op_name=" "):
    def register(cls: OpTest):
        if op_name:
            OpTestFactory.register(op_name, cls)
        else:
            raise Exception(
                '[OpTestFactory]: illegal op name, please use like @registerOp("op_name")')
        return cls
    return register
