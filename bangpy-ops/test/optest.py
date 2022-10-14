# Copyright (C) [2022] by Cambricon, Inc.
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
# pylint: disable=invalid-name, missing-function-docstring
"""OpTest is the Basis Class of Kernels"""
from abc import ABC, abstractmethod


class OpTest(ABC):
    """OpTest is used for kernel computing.

    This class must be inherited by kernel. And rewrite function compute.
    when you inherite this class, please use function registerOp register it.

    If you want to transfer some params from json file to this class, you can
    write params in keyword op_params.
    """

    def __init__(self, target, dtype, input_tensor_list, output_tensor_list):
        self.inputs_list = input_tensor_list
        self.output_tensor = output_tensor_list
        self.dtype = dtype
        self.target = target
        self.test_param_ = None

    def paramCheck(self):
        pass

    @abstractmethod
    def compute(self):
        pass

    def run(self):
        self.paramCheck()
        self.compute()

    @property
    def inputs_tensor_list(self):
        return self.inputs_list

    @property
    def output_tensor_list(self):
        return self.output_tensor


class OpTestFactory:
    """Registry for OpTest."""

    registry = {}

    @classmethod
    def register(cls, name: str, register_cls: OpTest):
        """Function for register OpTest."""
        if name not in cls.registry:
            cls.registry[name] = register_cls
        else:
            raise Exception(
                "[OpTestFactory]: register the same op, please check op name."
            )

    @classmethod
    def factory(cls, name: str) -> OpTest:
        """Get OpTest from Factory by name."""
        if name not in cls.registry:
            raise KeyError(
                "[OpTestFactory]: op are not registered, please use @registerOp."
            )
        return cls.registry[name]


def registerOp(op_name=" "):
    def register(cls: OpTest):
        if op_name:
            OpTestFactory.register(op_name, cls)
        else:
            raise Exception(
                '[OpTestFactory]: illegal op name, please use like @registerOp("op_name")'
            )
        return cls

    return register
