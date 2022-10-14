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
# pylint: disable=missing-function-docstring
"""Using prototxt_parser to parse *.prototxt Files to Python Dict Objects."""
from typing import Dict, List
from google.protobuf import json_format
import numpy as np
from .mlu_op_test_pb2 import Node



class Parser:
    """Class for parser prototxt to tensor params, opname and test criterion."""

    def __init__(self, file, op_name):
        self.file = file
        self.op_name = op_name
        self.encode = {
            "DTYPE_FLOAT": np.int32,
            "DTYPE_HALF": np.int16,
            "INT32": np.int32,
        }
        self.decode = {
            "DTYPE_FLOAT": np.float32,
            "DTYPE_HALF": np.float16,
            "INT32": np.int32,
        }
        message = Node()
        message.ParseFromString(file)
        self.data =  json_format.MessageToDict(message)

    def get_output(self):
        return self.read_prototxt().get("output")[0]

    def read_prototxt(self) -> dict:
        return self.data

    def get_dev(self):
        return self.read_prototxt().get("device")

    def get_criterion(self):
        return self.read_prototxt().get("evaluation_criterion")

    def get_threshold(self):
        return self.read_prototxt().get("evaluation_threshold")

    def get_opname(self):
        return self.read_prototxt().get("opName")

    def get_input_dtype(self):
        input_ = self.read_prototxt().get("input")
        if not isinstance(self.read_prototxt().get("input"), list):
            input_ = [self.read_prototxt().get("input"),]
        return input_[0].get("dtype")

    def get_output_dtype(self):
        return self.read_prototxt().get("output")[0].get("dtype")

    def get_inp_oup(self):
        input_list = []
        input_ = self.read_prototxt().get("input")
        if not isinstance(self.read_prototxt().get("input"), list):
            input_ = [self.read_prototxt().get("input"),]
        for i in input_:
            if i.get("randomData").get("distribution") == "UNIFORM" and hasattr(
                i.get("randomData"), "seed"
            ):
                np.random.seed(i.get("randomData").get("seed"))
                input_list.append(
                    np.random.uniform(size=i.get("shape").get("dims")).astype(
                        self.decode.get(self.get_input_dtype())
                    )
                )
            elif i.get("randomData").get("distribution") == "UNIFORM" and not hasattr(
                i.get("randomData").get("distribution"), "seed"
            ):
                input_list.append(
                    np.frombuffer(
                        np.array(
                            i.get("valueI"), self.encode.get(self.get_input_dtype())
                        ).reshape(i.get("shape").get("dims")),
                        self.decode.get(self.get_input_dtype()),
                    )
                )
            # (TODO:Add more distribution model)
        output_list = []
        if isinstance(oups := self.read_prototxt().get("output"), Dict):
            output = np.frombuffer(
                np.array(
                    oups.get("valueI"),
                    self.encode.get(self.get_output_dtype()),
                ).reshape(oups.get("shape").get("dims")),
                self.decode.get(self.get_output_dtype()),
            )
            output_list.append(output)
        elif isinstance(oups := self.read_prototxt().get("output"), List):
            for j in oups:
                output_list.append(
                    np.frombuffer(
                    np.array(
                        j.get("valueI"),
                        self.encode.get(self.get_output_dtype()),
                    ).reshape(j.get("shape").get("dims")),
                    self.decode.get(self.get_output_dtype()),
                )
                )
                # output_list.append(j)
        return input_list, output_list
