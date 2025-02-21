#!/usr/bin/env python3
#coding:utf-8

from typing import Dict
import google.protobuf.json_format as json_format
from analysis_suite.cfg import mlu_op_test_pb2

# see http://gitlab.software.cambricon.com/neuware/software/test/test_mluops/-/blob/master/scripts/database/protobuf_case_parser.py
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

    def parse_param(self, node) -> Dict:
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
