# Copyright (C) [2024] by Cambricon, Inc.
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
# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring
# pylint: disable=attribute-defined-outside-init
#!/usr/bin/env python3
#coding:utf-8

import google.protobuf.json_format as json_format
import mlu_op_test_pb2

# see http://gitlab.software.cambricon.com/neuware/software/test/
# test_mluOp/-/blob/master/scripts/database/protobuf_case_parser.py
class ProtobufCaseParserImplInputOutput:

    def __init__(self, mlu_op_test_pb2):
        self.mlu_op_test_pb2 = mlu_op_test_pb2
        # emun map
        self.dtype_names = {
            k: v.lower().partition('dtype_')[-1]
            for (v, k) in self.mlu_op_test_pb2.DataType.items()
        }
        self.layout_names = {
            k: v.partition('LAYOUT_')[-1]
            for (v, k) in self.mlu_op_test_pb2.TensorLayout.items()
        }

    def parse_param(self, node):
        field_names = [i[0].name for i in node.ListFields()]
        params_dict = {}
        for param_name in field_names:
            if "_param" in param_name and "test_param" != param_name and "handle_param" != param_name:
                params = getattr(node, param_name)
                params_dict.update(json_format.MessageToDict(params))
                return params_dict
        return {}

    def parseDType(self, tensor):
        t1 = self.dtype_names[tensor.dtype]
        if tensor.HasField('onchip_dtype'):
            t2 = self.dtype_names[tensor.onchip_dtype]
            return [t1, t2]
        return t1

    def parseLayout(self, tensor):
        t1 = self.layout_names[tensor.layout]
        return t1

    def __call__(self, node):
        # input
        input_dim = [list(k.shape.dims) for k in node.input]
        input_stride = [list(k.shape.dim_stride) for k in node.input]

        input_dtype = [self.parseDType(k) for k in node.input]
        input_layout = [self.parseLayout(k) for k in node.input]

        # output
        output_dim = [list(k.shape.dims) for k in node.output]
        output_stride = [list(k.shape.dim_stride) for k in node.output]

        output_dtype = [self.parseDType(k) for k in node.output]
        output_layout = [self.parseLayout(k) for k in node.output]

        inputs_key = ["input_dim", "input_layout", "input_dtype"] if all(
            len(stride) == 0 for stride in input_stride) else [
                "input_dim", "input_stride", "input_layout", "input_dtype"
            ]
        inputs_value = list(zip(input_dim, input_layout, input_dtype)) if all(
            len(stride) == 0 for stride in input_stride) else list(
                zip(input_dim, input_stride, input_layout, input_dtype))

        inputs = []
        for input_value in inputs_value:
            inputs.append(dict(zip(inputs_key, input_value)))

        outputs_key = ["output_dim", "output_layout", "output_dtype"] if all(
            len(stride) == 0 for stride in output_stride) else [
                "output_dim", "output_stride", "output_layout", "output_dtype"
            ]
        outputs_value = list(zip(
            output_dim, output_layout, output_dtype)) if all(
                len(stride) == 0 for stride in output_stride) else list(
                    zip(output_dim, output_stride, output_layout,
                        output_dtype))

        outputs = []
        for output_value in outputs_value:
            outputs.append(dict(zip(outputs_key, output_value)))

        op_params = self.parse_param(node)

        params = {
            'input': inputs,
            'output': outputs,
            'params': op_params,
        }
        return params
