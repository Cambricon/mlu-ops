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

import logging
import json

import mlu_op_test_pb2

# miscellaneous config
class Config:
    pass_port = {
    "user_name": b'Y2FtYnJpY29u',
    "pass_word": b'TXlxbC1jYW0xOTg='
    }

    suffix = ["_new", "_baseline"]

    platform_map = {
        'MLU370-S4': 'MLU370',
        'MLU370-X4': 'MLU370',
        'MLU370-X8': 'MLU370',
        'MLU590': 'MLU590',
        'MLU270-X5K': 'MLU270_X5K',
        'MLU270-F4': 'MLU270_F4',
        'MLU290-M5': 'MLU290'
    }

    case_info_keys = ["input", "output", "params"]
    network_info_keys = [
        'network_id', 'network_name', 'framework', 'mode', 'batchsize',
        'network_additional_information', 'version'
    ]

    # repeat info
    repeat_key = [
        'date', 'cluster_limit', 'job_limit', 'mlu_platform', 'mluop_version', 'commit_id', 'mluop_branch', 'timestamp', 'driver_version', 'cnrt_version'
    ]

    xml_properties_map = {
        'op_name': 'operator',
        'hardware_time_mlu': 'mlu_hardware_time',
        'interface_time_mlu': 'mlu_interface_time',
        'io_efficiency_mlu': 'mlu_io_efficiency',
        'compute_efficiency_mlu': 'mlu_compute_efficiency',
        'case_path': 'file_path',
        'workspace_size_mlu': 'mlu_workspace_size',
        'kernel_names_mlu': 'mlu_kernel_names',
        'theory_ops': 'mlu_theory_ops',
        'theory_ios': 'mlu_theory_ios',
        'compute_force': 'mlu_computeforce',
        'io_bandwidth': 'mlu_iobandwidth',
        'workspace_size_gpu': 'gpu_workspace_size',
        'hardware_time_gpu': 'gpu_hardware_time',
        'io_efficiency_gpu': 'gpu_io_efficiency',
        'compute_efficiency_gpu': 'gpu_compute_efficiency',
    }

    log_keyword_map = {
        'RUN': 'operator',
        'MLU Hardware Time': 'mlu_hardware_time',
        'MLU Interface Time': 'mlu_interface_time',
        'MLU IO Efficiency': 'mlu_io_efficiency',
        'MLU Compute Efficiency': 'mlu_compute_efficiency',
        'MLU Workspace Size': 'mlu_workspace_size',
        'MLU Kernel Name(s)': 'mlu_kernel_names',
        'MLU TheoryOps': 'mlu_theory_ops',
        'MLU TheoryIOs': 'mlu_theory_ios',
        'MLU ComputeForce': 'mlu_computeforce',
        'MLU IoBandWidth': 'mlu_iobandwidth',
        'GPU Hardware Time': 'gpu_hardware_time',
        'GPU IO Efficiency': 'gpu_io_efficiency',
        'GPU Compute Efficiency': 'gpu_compute_efficiency',
        'GPU Workspace Size': 'gpu_workspace_size',
        '^      OK': 'file_path',
        '^  FAILED': 'file_path'
    }

    float_columns = [
        'mlu_hardware_time',
        'mlu_interface_time',
        'mlu_io_efficiency',
        'mlu_compute_efficiency',
        'mlu_workspace_size',
        'gpu_hardware_time',
        'gpu_io_efficiency',
        'gpu_compute_efficiency',
        'gpu_workspace_size',
        'mlu_theory_ios',
        'mlu_theory_ops',
        'mlu_computeforce',
        'mlu_iobandwidth',
    ]

    important_network_keyword = [
        'resnet50v1.5',
        'ssd',
        'maskrcnn',
        'transformer',
        'bert',
        'mobilenetv2',
        'inceptionv3',
        'yolov3',
    ]

    summary_columns = [
        'case_number', 'mlu_io_efficiency_mean', 'mlu_compute_efficiency_mean',
        'mlu_hardware_time_sum', 'good_rate', 'qualified_rate',
        'unqualified_rate'
    ]


class PerfConfig:

    def __init__(self, filename="perf.json"):
        try:
            with open(filename) as f:
                self.attrs = json.load(f)
        except Exception as e:
            logging.warning(e)
            logging.warning("load perf.json failed, use default value")
            self.attrs = {
                "criterion": {
                    "good": [0.6, 1.0],
                    "qualified": [0.3, 0.6],
                    "unqualified": [0.0, 0.3]
                },
                "ignore_case": {
                    "mlu_hardware_time": 30
                }
            }
