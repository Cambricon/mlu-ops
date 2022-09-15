# Copyright (C) [2021] by Cambricon, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall self.tcp included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# pylint: disable=invalid-name, missing-function-docstring, useless-object-inheritance
# pylint: disable=too-many-instance-attributes, missing-module-docstring, too-many-locals
# pylint: disable=too-many-statements, missing-class-docstring
from prototxt_parser.prototxt import parse
import numpy as np


class Parser(object):
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

    def get_output(self):
        return self.read_prototxt().get("output")

    def read_prototxt(self) -> dict:
        return parse(self.file)

    def get_dev(self):
        return self.read_prototxt().get("device")

    def get_criterion(self):
        return self.read_prototxt().get("evaluation_criterion")

    def get_threshold(self):
        return self.read_prototxt().get("evaluation_threshold")

    def get_opname(self):
        return self.read_prototxt().get("op_name")

    def get_input_dtype(self):
        return self.read_prototxt().get("input")[0].get("dtype")

    def get_output_dtype(self):
        return self.read_prototxt().get("output").get("dtype")

    def get_inp_oup(self):
        input_list = []
        for i in self.read_prototxt().get("input"):
            if i.get("random_data").get("distribution") == "UNIFORM" and hasattr(
                i.get("random_data"), "seed"
            ):
                np.random.seed(i.get("random_data").get("seed"))
                input_list.append(
                    np.random.uniform(size=i.get("shape").get("dims")).astype(
                        self.decode.get(self.get_input_dtype())
                    )
                )
            elif i.get("random_data").get("distribution") == "UNIFORM" and not hasattr(
                i.get("random_data").get("distribution"), "seed"
            ):
                input_list.append(
                    np.frombuffer(
                        np.array(
                            i.get("value_i"), self.encode.get(self.get_input_dtype())
                        ).reshape([int(i.get("shape").get("dims"))]),
                        self.decode.get(self.get_input_dtype()),
                    )
                )
            # (TODO:Add more distribution model)
        output = np.frombuffer(
            np.array(
                self.read_prototxt().get("output").get("value_i"),
                self.encode.get(self.get_output_dtype()),
            ).reshape(
                [int(self.read_prototxt().get("output").get("shape").get("dims"))]
            ),
            self.decode.get(self.get_output_dtype()),
        )
        return input_list, output
