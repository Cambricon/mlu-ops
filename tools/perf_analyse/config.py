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


class Config:
    def __init__(self):
        # columns for dataframe in xml
        self.df_columns_xml = [
            'operator', 'mlu_hardware_time', 'mlu_interface_time',
            'mlu_io_efficiency', 'mlu_compute_efficiency',
            'mlu_workspace_size', 'mlu_theory_ops', 'mlu_theory_ios',
            'mlu_computeforce', 'mlu_iobandwidth', 'gpu_workspace_size',
            'gpu_hardware_time', 'gpu_io_efficiency', 'gpu_compute_efficiency',
            'mlu_op_version', 'date', 'mlu_platform', 'file_path', 'job_limit',
            'cluster_limit', 'input_shape', 'input_dtype', 'input_layout',
            'output_shape', 'output_dtype', 'output_layout',
            'params', 'md5'
        ]

        # property name in xml file
        self.xml_columns = [
            'op_name', 'hardware_time_mlu', 'interface_time_mlu',
            'io_efficiency_mlu', 'compute_efficiency_mlu',
            'workspace_size_mlu', 'theory_ops', 'theory_ios', 'compute_force',
            'io_bandwidth', 'workspace_size_gpu', 'hardware_time_gpu',
            'io_efficiency_gpu', 'compute_efficiency_gpu', 'mlu_op_version',
            'date', 'mlu_platform', 'case_path', 'job_limit', 'cluster_limit'
        ]

        self.df_columns_json = self.df_columns_xml
        self.json_columns = self.xml_columns

        # columns for dataframe in log
        self.df_columns_log = [
            'operator', 'mlu_hardware_time', 'mlu_interface_time',
            'mlu_io_efficiency', 'mlu_compute_efficiency',
            'mlu_workspace_size', 'mlu_theory_ops', 'mlu_theory_ios',
            'mlu_computeforce', 'mlu_iobandwidth', 'gpu_hardware_time',
            'gpu_workspace_size', 'gpu_io_efficiency',
            'gpu_compute_efficiency', 'file_path', 'job_limit',
            'cluster_limit', 'input_shape', 'input_dtype', 'input_layout',
            'output_shape', 'output_dtype', 'output_layout',
            'params', 'md5'
        ]

        self.float_columns = [
            'mlu_hardware_time',
            'mlu_interface_time',
            'mlu_io_efficiency',
            'mlu_compute_efficiency',
            'mlu_theory_ios',
            'mlu_theory_ops',
            'mlu_computeforce',
            'mlu_iobandwidth',
            'mlu_workspace_size',
        ]

        self.log_keyword_columns = {
            'RUN': 'operator',
            'MLU Hardware Time': 'mlu_hardware_time',
            'MLU Interface Time': 'mlu_interface_time',
            'MLU IO Efficiency': 'mlu_io_efficiency',
            'MLU Compute Efficiency': 'mlu_compute_efficiency',
            'MLU Workspace Size': 'mlu_workspace_size',
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

        self.important_network_keyword = [
            'resnet50v1.5',
            'ssd',
            'maskrcnn',
            'transformer',
            'bert',
            'mobilenetv2',
            'inceptionv3',
            'yolov3',
        ]
