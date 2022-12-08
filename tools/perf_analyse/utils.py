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

import pandas as pd
import re
import subprocess
import os
import hashlib
import json
import xml.etree.ElementTree as ET
from sqlalchemy import create_engine
import numpy as np
import google.protobuf.text_format
import google.protobuf.json_format as json_format
import mlu_op_test_pb2
import tqdm
import matplotlib
import time

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
from config import Config
from multiprocessing import Pool
from openpyxl.utils import get_column_letter
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union, List, Generator, Iterator

suffix = ["_new", "_baseline"]
case_info_keys_ = [
    "input_shape", "input_dtype", "input_layout", "output_shape",
    "output_dtype", "output_layout", "params", "md5"
]
case_info_keys = ['operator'] + case_info_keys_


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
        input_shape = {
            "input_dim": input_dim
        } if all(len(stride) == 0 for stride in input_stride) else {
            "input_dim": input_dim,
            "input_stride": input_stride
        }

        input_dtype = [self.parseDType(k) for k in node.input]
        input_layout = [self.parseLayout(k) for k in node.input]

        # output
        output_dim = [list(k.shape.dims) for k in node.output]
        output_stride = [list(k.shape.dim_stride) for k in node.output]
        output_shape = {
            "output_dim": output_dim
        } if all(len(stride) == 0 for stride in output_stride) else {
            "output_dim": output_dim,
            "output_stride": output_stride
        }

        output_dtype = [self.parseDType(k) for k in node.output]
        output_layout = [self.parseLayout(k) for k in node.output]

        op_params = self.parse_param(node)

        params = {
            'input_shape': input_shape,
            'input_dtype': input_dtype,
            'input_layout': input_layout,
            'output_shape': output_shape,
            'output_dtype': output_dtype,
            'output_layout': output_layout,
            'params': op_params,
        }
        return params


def parse_input_file(filename, config):
    print("parsing {}...".format(filename))
    try:
        cases = parse_gtest_output(filename, config)

        if len(cases) == 0:
            raise Exception(
                "There are no case records, please check {}".format(filename))

    # can be more specific, such as ParseError JSONDecodeError
    except Exception as e:
        print(e)
        exit()
    else:
        return get_dataframe(cases, config)

def create_arraylike_dict(filename: str, config: Config, replica_num: int = 1):
    if filename.endswith("xml"):
        case_generator = xml_yield(filename, config)
        attrs = config.xml_columns[1:3]
    elif filename.endswith("json"):
        case_generator = json_yield(filename, config)
        attrs = config.xml_columns[1:3]
    else:
        case_generator = log_yield(filename, config)
        attrs = config.df_columns_log[1:3]
    cases = {}
    for case in case_generator:
        case[attrs[0]] = float(case[attrs[0]])/replica_num
        case[attrs[1]] = float(case[attrs[1]])/replica_num
        merge_dict(cases, case)
    return cases

def gen_perf_time_from_xml(filename: str, attr: List[str])->Iterator[Dict[str, float]]:
    tree = ET.parse(filename)
    testsuites = tree.getroot()
    for testsuite in testsuites.iterfind('testsuite'):
        if 'DISABLED' not in testsuite.attrib['name']:
            for testcase in testsuite.iterfind('testcase'):
                data = {}
                for key in attr:
                    match = "./properties/property[@name='{}']".format(key)
                    value = testcase.find(match).attrib['value']
                    data[key] = float(value)
                yield data

def gen_perf_time_from_json(filename: str, attr: List[str])->Iterator[Dict[str, float]]:
    with open(filename) as fp:
        tests = json.load(fp)
    for testsuites in tests['testsuites']:
        if 'DISABLED' not in testsuites['name']:
            for testcase in testsuites['testsuite']:
                data = {}
                for key in attr:
                    data[key] = float(testcase[key])
                yield data

def average_performance(case_attr_dict: Dict[str, List], filename: str, attrs: List[str], replica_num: int) ->Dict[str, List]:
    if filename.endswith("xml"):
        perf_time_generator = gen_perf_time_from_xml(filename, attrs)
    elif filename.endswith("json"):
        perf_time_generator = gen_perf_time_from_json(filename, attrs)
    else:
        raise Exception("average_performance function cannot process {}, only xml or json file is supported!".format(filename))

    index = 0
    for perf_time in perf_time_generator:
        for key in attrs:
            case_attr_dict[key][index] = case_attr_dict[key][index] + perf_time[key]/replica_num
        hardware_time_mlu = case_attr_dict["hardware_time_mlu"][index]
        theory_ios = float(case_attr_dict["theory_ios"][index])
        io_bandwidth = float(case_attr_dict["io_bandwidth"][index])
        case_attr_dict["io_efficiency_mlu"][index] = theory_ios/ (hardware_time_mlu * io_bandwidth*1000)
        theory_ops = float(case_attr_dict["theory_ops"][index])
        compute_force = float(case_attr_dict["compute_force"][index])
        case_attr_dict["compute_efficiency_mlu"][index] = theory_ops*10**6 / (hardware_time_mlu * compute_force)
        index = index + 1
    return case_attr_dict

def parse_gtest_output(file_path: str, config: Config)-> Dict[str, List]:
    output = Path(file_path)
    if output.is_file():
        cases = create_arraylike_dict(output.as_posix(), config)
    else:
        replica_num = len(list(output.glob('*')))
        if replica_num == 0:
            raise Exception("The folder {} is Empty!".format(file_path))
        file_gen = output.glob('*')
        original = next(file_gen)
        cases = create_arraylike_dict(original.as_posix(), config, replica_num)
        perf_time_attr = config.xml_columns[1:3]
        for replica in file_gen:
            cases = average_performance(cases, replica.as_posix(), perf_time_attr, replica_num)
    return cases

def merge_dict(a, b):
    if len(a.keys()) != len(b.keys()) and len(a.keys()) != 0:
        print("{} {}".format(len(a.keys()), len(b.keys())))
        raise Exception("not match {}\n{}".format(a, b))
    for k, v in b.items():
        if k not in a:
            a[k] = []
        a[k].append(v)

def xml_yield(filename, config):
    tree = ET.parse(filename)
    testsuites = tree.getroot()
    # repeat info is in testsuites
    repeat_key = [
        'date', 'cluster_limit', 'job_limit', 'mlu_platform', 'mlu_op_version'
    ]
    for testsuite in testsuites.findall('testsuite'):
        if 'DISABLED' not in testsuite.attrib['name']:
            for testcase in testsuite.findall('testcase'):
                data = {
                    property.attrib['name']: property.attrib['value']
                    for property in testcase.find('properties')
                    if property.attrib['name'] in config.xml_columns
                }
                for k in repeat_key:
                    if k in testsuites.attrib.keys():
                        data[k] = testsuites.attrib[k]
                yield data

def json_yield(filename, config):
    with open(filename) as fp:
        tests = json.load(fp)
    repeat_key = ['date', 'cluster_limit', 'job_limit', 'mlu_platform', 'mlu_op_version']
    for testsuites in tests['testsuites']:
        if 'DISABLED' not in testsuites['name']:
            for one_test in testsuites['testsuite']:
                data = {
                    k: v
                    for k, v in one_test.items() if k in config.xml_columns
                }
                for k in repeat_key:
                    if k in tests.keys():
                        data[k] = tests[k]
                yield data

def log_yield(filename, config):
    result = {}
    with open(filename) as f:
        for line in f:
            for key in config.log_keyword_columns.keys():
                if key in line:
                    value = re.findall(r"\]:?\s*(\S+)\s?", line)
                    if len(value) == 0:
                        raise Exception(
                            "log file may be corrupted, {}".format(line))
                    if key == 'RUN':
                        result[config.log_keyword_columns[key]] = value[
                            0].split('/')[0]
                    else:
                        result[config.log_keyword_columns[key]] = value[0]
                    if 'file_path' in result.keys():
                        yield result
                        result = {}

# change dict int dataframe and preprocess data
def get_dataframe(cases, config):
    df = pd.DataFrame(cases)
    if 'op_name' in df.columns:
        columns_dict = {
            config.xml_columns[i]: config.df_columns_xml[i]
            for i in range(len(config.xml_columns))
        }
        columns = [columns_dict[i] for i in df.columns]
    else:
        columns = df.columns

    df.columns = columns
    # cast data type
    for column in config.float_columns:
        if column in df.columns:
            df[column] = df[column].astype(float)
            # clean invalid data
            df.loc[df[column] < 0, column] = 0

    if 'mlu_theory_ios' in columns:
        df['io_theory_time'] = df['mlu_theory_ios'] / df[
            'mlu_iobandwidth'] / 1000
        df['cp_theory_time'] = df['mlu_theory_ops'] / df[
            'mlu_computeforce'] * 1000 * 1000
        # will be inf or nan (0 / 0)
        df['io_cp'] = df['io_theory_time'] / df['cp_theory_time']
        df['is_io_bound'] = df['io_cp'] > 1

    if 'mlu_platform' in df.columns:
        df['mlu_platform'] = [
            i.split('[')[0] for i in df['mlu_platform'].to_list()
        ]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format="%Y_%m_%d_%H_%M_%S")
        df['date'] = df['date'].dt.strftime("%Y-%m-%d")

    if 'job_limit' not in df.columns or 'cluster_limit' not in df.columns:
        df['job_limit'] = 0
        df['cluster_limit'] = 0
    else:
        df['job_limit'] = [
            int(i) if len(i) > 0 else 0 for i in df['job_limit'].to_list()
        ]
        df['cluster_limit'] = [
            int(i) if len(i) > 0 else 0 for i in df['cluster_limit'].to_list()
        ]

    return df

# helper function for dump excel
def get_max_length(series, col_name):
    """
    get max length of series, used to auto set width
    """
    # avoid nan
    series = series.fillna('-')
    str_list = series.to_list()
    len_list = []
    for ele in str_list + [col_name]:
        # change str to list
        ele_split = list(ele)
        length = 0
        for c in ele_split:
            if ord(c) <= 256:
                length += 1
            else:
                # chinese character will be fat
                length += 2
        len_list.append(length + 2)

    max_length = max(len_list)
    max_length = min(max_length, 50)
    max_length = max(max_length, 12)
    return max_length

# helper function for dump excel
def auto_width(df, writer, sheet_name, cols_list):
    for i in range(0, len(cols_list)):
        col = cols_list[i]
        letter = chr(i + 65)
        max_len = get_max_length(df[col].astype(str), col)
        wb = writer.book
        ws = writer.sheets[sheet_name]
        fmt = wb.add_format({'align': 'left'})
        ws.set_column(i, i, max_len, fmt)

def dfs_to_excel(dfs, sheet_names, xlsx_path):
    with pd.ExcelWriter(xlsx_path) as writer:
        for i in range(0, len(dfs)):
            dfs[i].to_excel(writer, sheet_name=sheet_names[i], index=False)
            auto_width(dfs[i], writer, sheet_names[i], dfs[i].columns)

def dump_to_excel(df, xlsx_path, n):
    df['total_time'] = df.groupby('md5')['mlu_hardware_time'].transform('sum')
    df['count'] = df.groupby('md5')['mlu_hardware_time'].transform('count')

    columns_list = [
        'operator', 'mlu_hardware_time', 'mlu_interface_time',
        'mlu_workspace_size', 'mlu_io_efficiency', 'mlu_compute_efficiency',
        'count'
    ] + case_info_keys_ + ['is_io_bound', 'total_time', 'file_path']
    with pd.ExcelWriter(xlsx_path) as writer:
        if 'io_cp' not in df.columns:
            columns_list.remove('is_io_bound')
        df[-df.duplicated('md5')].to_excel(writer,
            sheet_name='all',
            index=False,
            columns=columns_list)
        auto_width(df, writer, 'all', columns_list)
        summary = get_summary(df)
        summary.to_excel(writer, sheet_name='summary', index=False)
        summary_extra = get_summary_extra(df)
        summary_extra.to_excel(writer,
                               sheet_name='summary',
                               index=False,
                               startrow=summary.shape[0] + 1)
        auto_width(summary, writer, 'summary', summary.columns)
        
        if 'io_cp' in df.columns:
            cand_hw_time = get_candidate_by_hardware_time(df, n)
            cand_hw_time.to_excel(writer,
                                  sheet_name='longest_hw_time',
                                  index=False,
                                  columns=columns_list)
            auto_width(cand_hw_time, writer, 'longest_hw_time', columns_list)
            cand_interface_time = get_candidate_by_interface_time(df, n)
            cand_interface_time.to_excel(writer,
                                         sheet_name='longest_interface_time',
                                         index=False,
                                         columns=columns_list)
            auto_width(cand_interface_time, writer, 'longest_interface_time',
                       columns_list)
            cand_io_eff = get_candidate_by_io_eff(df, n)
            cand_io_eff.to_excel(writer,
                                 sheet_name='smallest_io_eff',
                                 index=False,
                                 columns=columns_list)
            auto_width(cand_io_eff, writer, 'smallest_io_eff', columns_list)
            cand_cp_eff = get_candidate_by_cp_eff(df, n)
            cand_cp_eff.to_excel(writer,
                                 sheet_name='smallest_cp_eff',
                                 index=False,
                                 columns=columns_list)
            auto_width(cand_cp_eff, writer, 'smallest_cp_eff', columns_list)

    print("the ouput excel is " + xlsx_path)

def dump_to_simple_tpi_network_excel(df, xlsx_path,
                                    config: Config,
                                    frameworks):
    important_network_keys = config.important_network_keyword
    all_network_sheets = df.keys()
    important_network_names = get_important_network_names(
            all_network_sheets, important_network_keys, frameworks)
    important_network_names.append('summary')
    #   select important network row
    all_network_rows = df['summary']['网络名称'].to_list()
    important_network = get_important_network_names(
            all_network_rows, important_network_keys, frameworks)
    df['summary'] = df['summary'].loc[
            df['summary'].网络名称.isin(important_network)]
    df['summary']= df['summary'].reset_index(drop = True)
    #   select important network sheets
    get_important_network_sheet(df, important_network_names)
    #   comput device_time_percentage for every sheet
    dfs = []
    for sheet_name in df.keys():
        if sheet_name not in ('summary'):
            all_device_time = df[sheet_name]['总device时间(us)'].sum()
            df[sheet_name]['device_time_percentage'] = df[sheet_name]['总device时间(us)'] / all_device_time
            df[sheet_name] = df[sheet_name].sort_values(
                                            by=[('device_time_percentage')],
                                            ascending=False)
            df[sheet_name]['device_time_percentage'] = df[sheet_name]['device_time_percentage'].apply('{:.2%}'.format)
            df[sheet_name] = move_column_location(df[sheet_name], 1, 'device_time_percentage')
            df[sheet_name] = df[sheet_name].reset_index(drop = True)
            df[sheet_name].rename(
                        columns={'device_time_percentage':'device时间占比'},
                        inplace = True)
            dfs.append(df[sheet_name])

    #   get top20 operators from important network
    #   filter metrics:'总device时间占比'
    top20_ops = pd.DataFrame()
    all_ops_sheet = pd.concat(dfs, ignore_index = True, sort = True)
    all_ops_sheet['IO瓶颈比例'] = all_ops_sheet['IO瓶颈比例'].str.strip("%").astype(float)/100
    group_by_op = all_ops_sheet.groupby('算子名称')

    data_columns = ['平均IO效率', '平均计算效率', '平均workspace(Bytes)',
                    '平均device时间(us)', '总个数','总device时间(us)',
                    'IO瓶颈比例', '总workspace(Bytes)', '总host时间(us)' ,
                    '平均host时间(us)']
    for col in data_columns:
        if col in ['总个数', '总device时间(us)', '总workspace(Bytes)', '总host时间(us)']:
            top20_ops[col] = group_by_op.agg({col: ['sum']})[col]['sum']
        else:
            top20_ops[col] = group_by_op.agg({col: ['mean']})[col]['mean']

    all_ops_time = top20_ops['总device时间(us)'].sum()
    top20_ops['device_time_per'] = top20_ops['总device时间(us)'] / all_ops_time
    top20_ops['device_time_per'] = top20_ops['device_time_per'].apply("{:.2%}".format)
    top20_ops['IO瓶颈比例'] = top20_ops['IO瓶颈比例'].apply("{:.2%}".format)
    top20_ops = top20_ops.sort_values(by=[('总device时间(us)')],
                                        ascending=False)
    top20_ops = top20_ops.reset_index()
    #   move 'device_time_percentage' column after '算子名称'
    top20_ops = move_column_location(top20_ops, 1, 'device_time_per')
    top20_ops.rename(columns = {'device_time_per':'device时间占比'}, inplace = True)
    top20_ops = top20_ops.head(20)

    #   df to excel
    with pd.ExcelWriter(xlsx_path)as writer:
        top20_ops.to_excel(
                writer,
                sheet_name = 'top20_ops_data',
                index = False)
        auto_width(top20_ops, writer, 'top20_ops_data', top20_ops.columns)
        for i in df.keys():
            df[i].to_excel(writer, sheet_name = i, index = False)
            auto_width(df[i], writer, i, df[i].columns)
    print("the ouput simple excel is " + xlsx_path)

def dump_to_simple_comparision_tpi_excel(df, xlsx_path, config:Config,
                                frameworks, version_compare):
    important_network_keys = config.important_network_keyword
    all_network_sheets = df.keys()
    important_network_names = get_important_network_names(
            all_network_sheets, important_network_keys, frameworks)
    important_network_names.append('summary')

    all_network_rows = df['summary']['网络名称'].to_list()
    important_network = get_important_network_names(
            all_network_rows, important_network_keys, frameworks)
    df['summary'] = df['summary'].loc[df['summary'].网络名称.isin(important_network)]
    df['summary']= df['summary'].reset_index(drop = True)
    get_important_network_sheet(df, important_network_names)

    columns = [
        'mlu_hardware_time_new',
        'mlu_hardware_time_baseline', 'mlu_interface_time_new',
        'mlu_interface_time_baseline', 'mlu_workspace_size_new',
        'mlu_workspace_size_baseline', 'count_new', 'count_baseline'
        ]
    columns_bs = ['mlu_hardware_time', 'mlu_interface_time',
            'mlu_workspace_size', 'count']
    columns_bs_ch = ['device时间', 'interface时间', 'workspace大小', '个数']
    update_columns = ['operator', 'device_time_per_new', 'device_time_per_baseline']
    update_columns_ch = ['算子名称', 'device时间占比' + version_compare[0],
                        'device时间占比' + version_compare[1]]

    static_suffix = ['_new', '_baseline' ,'_promotion_ratio']
    columns_promotion_ratio = []
    ver_suffix = version_compare
    promotion = '_提升比例'
    #   generate new final columns names
    for i in  range(0, len(columns_bs)):
        update_columns.append(columns_bs[i]+ static_suffix[0] + '_sum')
        update_columns.append(columns_bs[i]+ static_suffix[1] + '_sum')
        update_columns.append(columns_bs[i]+ '_sum' + static_suffix[2])
        columns_promotion_ratio.append(columns_bs[i]+ '_sum' + static_suffix[2])
        update_columns_ch.append('总' + columns_bs_ch[i] + ver_suffix[0])
        update_columns_ch.append('总' + columns_bs_ch[i] + ver_suffix[1])
        update_columns_ch.append('总' + columns_bs_ch[i] + promotion)
        if columns_bs[i] not in ['count']:
            update_columns.append(columns_bs[i]+ static_suffix[0] + '_mean')
            update_columns.append(columns_bs[i]+ static_suffix[1] + '_mean')
            update_columns.append(columns_bs[i] + '_mean'+ static_suffix[2])
            columns_promotion_ratio.append(columns_bs[i] + '_mean'+ static_suffix[2])
            update_columns_ch.append('平均' + columns_bs_ch[i] + ver_suffix[0])
            update_columns_ch.append('平均' + columns_bs_ch[i] + ver_suffix[1])
            update_columns_ch.append('平均' + columns_bs_ch[i] + promotion)
    for i in ['io_bound_percentage_new', 'io_bound_percentage_baseline']:
        update_columns.append(i)
    for i in ['IO瓶颈比例' + ver_suffix[0], 'IO瓶颈比例' + ver_suffix[1]]:
        update_columns_ch.append(i)

    # generate sum and mean data to groupby op
    data_col_sum = []
    data_col_mean = []
    data_columns_all = []
    for i in columns:
        data_col_sum.append(i + '_sum')
        data_columns_all.append(i + '_sum')
        if i not in ['count_new', 'count_baseline']:
            data_columns_all.append(i + '_mean')
            data_col_mean.append(i + '_mean')

    #   deal with 'summary' sheet
    top20_ops = pd.DataFrame()
    all_sheets = []
    dfs = pd.DataFrame()
    for i in important_network_names:
        if i not in ('summary'):
            dfs = pd.DataFrame()
            for col in columns:
                    group_by_op = df[i].groupby('operator')
                    dfs[col + '_sum']  = group_by_op.agg(
                                {col: ['sum']})[col]['sum']
                    if col not in ['count_new', 'count_baseline']:
                        dfs[col + '_mean']  = group_by_op.agg(
                                {col: ['mean']})[col]['mean']
            for i in range(0, 3):
                dfs[columns_promotion_ratio[2*i]] = (dfs[data_columns_all[4*i+2]] -
                        dfs[data_columns_all[(4*i)]])/dfs[data_columns_all[4*i+2]]
                dfs[columns_promotion_ratio[2*i+1]] = (dfs[data_columns_all[4*i+3]
                    ] - dfs[data_columns_all[(4*i+1)]])/dfs[data_columns_all[4*i+3]]

            # compute io_bound_percentage
            dfs['counts_new'] = group_by_op.agg({'count_new': ['sum']}
                    )['count_new']['sum']
            dfs['io_bound_percentage_new'] = group_by_op.apply(
                    lambda x: x[x['is_io_bound_new']]['count_new'].sum()) / dfs['counts_new']
            dfs['counts_baseline'] = group_by_op.agg({'count_baseline': ['sum']}
                    )['count_baseline']['sum']
            dfs['io_bound_percentage_baseline'] = group_by_op.apply(
                    lambda x: x[x['is_io_bound_baseline']]['count_baseline'].sum()
                    ) / dfs['counts_baseline']

            dfs['count_sum_promotion_ratio'] =(dfs['count_baseline_sum'] -
                    dfs['count_new_sum'])/dfs['count_baseline_sum']
            dfs = dfs.sort_values(by=[('mlu_hardware_time_new_sum')],
                                                ascending=False)
            dfs_ops_time_new = dfs['mlu_hardware_time_new_sum'].sum()
            dfs_ops_time_baseline = dfs['mlu_hardware_time_baseline_sum'].sum()
            dfs['device_time_per_new'] = (dfs['mlu_hardware_time_new_sum']
                    ) / dfs_ops_time_new
            dfs['device_time_per_baseline'] = (dfs['mlu_hardware_time_baseline_sum']
                    ) / dfs_ops_time_baseline
            dfs['device_time_per_new'] = (dfs['device_time_per_new']
                    ).apply("{:.2%}".format)
            dfs['device_time_per_baseline'] = (dfs['device_time_per_baseline']
                    ).apply("{:.2%}".format)
            dfs = dfs.reset_index()
            dfs = dfs[update_columns]
            all_sheets.append(dfs)
            del dfs

    top20_ops = pd.DataFrame()
    all_ops_sheet = pd.concat(all_sheets, ignore_index = True, sort = True)
    #   gen top_20_ops sheet
    for i in columns_promotion_ratio:
        data_columns_all.append(i)

    group_by_op = all_ops_sheet.groupby('operator')
    for column in data_columns_all:
        if column in data_col_sum:
            top20_ops[column] = group_by_op.agg({column: ['sum']})[column]['sum']
        else:
            top20_ops[column] = group_by_op.agg({column: ['mean']})[column]['mean']

    top20_ops = top20_ops.sort_values(by=[('mlu_hardware_time_new_sum')],
                                        ascending=False)

    all_ops_time_new = top20_ops['mlu_hardware_time_new_sum'].sum()
    all_ops_time_baseline = top20_ops['mlu_hardware_time_baseline_sum'].sum()
    top20_ops['io_bound_percentage_new'] = group_by_op.agg(
                    {'io_bound_percentage_new': ['mean']}
                    )['io_bound_percentage_new']['mean']
    top20_ops['io_bound_percentage_baseline'] = group_by_op.agg(
                    {'io_bound_percentage_baseline': ['mean']}
                    )['io_bound_percentage_baseline']['mean']
    top20_ops['device_time_per_new'] = (top20_ops['mlu_hardware_time_new_sum']
            ) / all_ops_time_new
    top20_ops['device_time_per_baseline'] = (top20_ops['mlu_hardware_time_baseline_sum']
            ) / all_ops_time_baseline
    for i in (['io_bound_percentage_new', 'io_bound_percentage_baseline',
                'device_time_per_new', 'device_time_per_baseline'
                ] + columns_promotion_ratio):
        top20_ops[i] = top20_ops[i].apply("{:.2%}".format)

    top20_ops = top20_ops.reset_index()
    top20_ops = top20_ops[update_columns]
    top20_ops.rename(columns=dict(zip(update_columns, update_columns_ch)),
                inplace=True)
    top20_ops = top20_ops.head(20)

    #   modify columns and trans %
    for i in range(0, len(all_sheets)):
        for j in (columns_promotion_ratio + [
            'io_bound_percentage_new', 'io_bound_percentage_baseline']):
            all_sheets[i][j] =  all_sheets[i][j].apply("{:.2%}".format)
    for i in range(0, len(all_sheets)):
        all_sheets[i].rename(columns=dict(zip(all_sheets[i].keys(), update_columns_ch)),
                    inplace=True)

    with pd.ExcelWriter(xlsx_path)as writer:
        top20_ops.to_excel(writer, sheet_name = 'top20_ops_data',
                        index = False)
        df['summary'].to_excel(writer, sheet_name = 'summary', index = False)
        auto_width(top20_ops, writer, 'top20_ops_data', top20_ops.columns)
        auto_width(df['summary'], writer, 'summary', df['summary'].columns)
        for i in range(0, len(all_sheets)):
            all_sheets[i].to_excel(writer, sheet_name=important_network_names[i], index=False)
            auto_width(all_sheets[i], writer, important_network_names[i], all_sheets[i].columns)
    print("the ouput excel is " + xlsx_path)

def move_column_location(df, loc, column_name):
    df_tmp  = df[column_name]
    df = df.drop(column_name, axis=1)
    df.insert(loc, column_name, df_tmp)
    return df

#   select important network sheet
def get_important_network_sheet(df, important_network_names):
    not_important_names = []
    for sheet_name in df.keys():
        if sheet_name not in important_network_names:
            not_important_names.append(sheet_name)
    [df.pop(x) for x in not_important_names]

def get_frameworks_names(input_fw):
    frameworks = []
    frameworks = input_fw.split(",")
    all_frameworks = ['pytorch', 'tf']
    for i in frameworks:
        if i not in all_frameworks:
            print("The framework name entered is incorrect, incorrect name is", i)
            exit()
    return frameworks

def get_version_numer(log_path, compare_path):
    log_filename = os.path.basename(log_path)
    compare_filename = os.path.basename(compare_path)
    version_compare=re.findall(r'\_\d+\.\d+\.\d+', log_filename
                ) + re.findall(r'\_\d+\.\d+\.\d+', compare_filename)
    return version_compare

def get_important_network_names(all_network, important_network_keyword, framework_names):
    network_names = []
    for item in important_network_keyword:
        for fw_name in framework_names:
            for key in all_network:
                if re.search(item, key) and re.search(fw_name, key):
                    network_names.append(key)
    return network_names

def get_candidate_by_hardware_time(df, n):
    # get unique
    return df[-df.duplicated('md5')].nlargest(n, 'mlu_hardware_time')

def get_candidate_by_interface_time(df, n):
    # get unique
    return df[-df.duplicated('md5')].nlargest(n, 'mlu_interface_time')

def get_candidate_by_io_eff(df, n):
    return df[(df['io_cp'] > 1) & -df.duplicated('md5')].nsmallest(
        n, 'mlu_io_efficiency')

def get_candidate_by_cp_eff(df, n):
    return df[(df['io_cp'] < 1) & (df['io_cp'] > 0)
              & -df.duplicated('md5')].nsmallest(n, 'mlu_compute_efficiency')

def get_summary_extra(df):
    data = {
        "硬件总时间": df['mlu_hardware_time'].sum(),
        "interface总时间": df['mlu_interface_time'].sum(),
        "总个数": df.shape[0],
        "平均IO效率": df['mlu_io_efficiency'].mean(),
        "平均计算效率": df['mlu_compute_efficiency'].mean()
    }
    summary_extra = pd.DataFrame(data, index=[0])
    return summary_extra

def get_summary(df):
    group_by_op = df.groupby('operator')
    summary = group_by_op.agg({
        'mlu_hardware_time': ['sum', 'count'],
    })

    summary['mlu_interface_time'] = group_by_op.agg(
        {'mlu_interface_time': ['sum']})['mlu_interface_time']['sum']
    if 'is_io_bound' in df.columns:
        summary['io_bound_count'] = group_by_op.agg({'is_io_bound': ['sum']
                                                     })['is_io_bound']['sum']
    all_time = df['mlu_hardware_time'].sum()
    summary['percentage'] = summary['mlu_hardware_time']['sum'] / all_time
    summary['percentage'] = summary['percentage'].apply("{:.2%}".format)
    summary = summary.sort_values(by=[('mlu_hardware_time', 'sum')],
                                  ascending=False)
    columns = [
        '算子名称', '硬件时间总和(us)', '出现总个数', 'interface时间总和(us)', 'IO瓶颈数目', '硬件时间百分比'
    ]
    if 'is_io_bound' not in df.columns:
        columns.remove('IO瓶颈数目')
    summary = summary.reset_index()
    summary.columns = columns
    return summary

def get_compare_log(df, df_baseline):
    df_baseline['count'] = df_baseline.groupby(
                    'md5')['mlu_hardware_time'].transform('count')
    df['mlu_platform'] = [
        i.split('[')[0] for i in df['mlu_platform'].to_list()
    ]
    df_baseline['mlu_platform'] = [
        i.split('[')[0] for i in df_baseline['mlu_platform'].to_list()
    ]
    mlu_op_case_comparison = pd.merge(
        df[-df.duplicated('md5')][[
            'operator', 'mlu_hardware_time', 'count', 'mlu_io_efficiency',
            'mlu_compute_efficiency', 'file_path', 'is_io_bound', 'date',
            'gpu_io_efficiency', 'gpu_compute_efficiency', 'gpu_hardware_time'
        ] + case_info_keys_],
        df_baseline[-df_baseline.duplicated('md5')][[
            'operator',
            'count',
            'mlu_hardware_time',
            'mlu_io_efficiency',
            'mlu_compute_efficiency', 'md5'
        ]],
        suffixes=suffix,
        on=['operator', 'md5'])
    mlu_op_case_comparison['mlu_hardware_time_promotion'] = mlu_op_case_comparison[
        'mlu_hardware_time' +
        suffix[1]] - mlu_op_case_comparison['mlu_hardware_time' + suffix[0]]
    mlu_op_case_comparison['mlu_hardware_time_promotion_ratio'] = (
        mlu_op_case_comparison['mlu_hardware_time' + suffix[1]] -
        mlu_op_case_comparison['mlu_hardware_time' + suffix[0]]
    ) / mlu_op_case_comparison['mlu_hardware_time' + suffix[1]]
    return mlu_op_case_comparison

def compare_log(df, df_baseline, log_path, compare_path, xlsx_path):
    xlsx_path = os.path.abspath(
        xlsx_path.split('/')[-1].split('.')[0] + '_comparison')
    df_compare_info = get_compare_log(df, df_baseline)
    generate_pic(df_compare_info, xlsx_path)
    comparison_info_to_excel(df_compare_info, xlsx_path)

def comparison_info_to_excel(df_compare_info, xlsx_path):
    xlsx_path = xlsx_path + '.xlsx'
    # change columns name
    # comparison info sheet
    df_compare_info['mlu_hardware_time_promotion_ratio'] = df_compare_info[
        'mlu_hardware_time_promotion_ratio'].apply(lambda x: format(x, '.2%'))
    df_compare_info.rename(columns={
        'mlu_hardware_time' + suffix[0]: 'mlu硬件时间(us)',
        'mlu_hardware_time' + suffix[1]: 'mlu硬件时间基线(us)',
        'mlu_hardware_time_promotion': 'mlu硬件时间提升(us)',
        'mlu_hardware_time_promotion_ratio': 'mlu硬件时间提升百分比',
        'count' + suffix[0]: 'count',
        'count' + suffix[1]: 'count基线'
    },
                           inplace=True)
    df_compare_info_columns_list = [
        'operator', 'mlu硬件时间(us)', 'mlu硬件时间基线(us)', 'mlu硬件时间提升(us)',
        'mlu硬件时间提升百分比', 'count', 'count基线', 'file_path', 'is_io_bound',
        'mlu_io_efficiency' + suffix[0], 'mlu_io_efficiency' + suffix[1],
        'mlu_compute_efficiency' + suffix[0],
        'mlu_compute_efficiency' + suffix[1], 'gpu_io_efficiency',
        'gpu_compute_efficiency', 'gpu_hardware_time', 'date'
    ] + case_info_keys_

    with pd.ExcelWriter(xlsx_path) as writer:
        df_compare_info.to_excel(writer,
                                 sheet_name='comparison info',
                                 index=False,
                                 columns=df_compare_info_columns_list)
        auto_width(df_compare_info, writer, 'comparison info',
                   df_compare_info_columns_list)
    print("the ouput comparison information excel is " + xlsx_path)

def generate_pic(df_compare_info, xlsx_path):
    # the following two sentences support Chinese SimHei
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # matplotlib.rcParams['axes.unicode_minus'] = False
    xlsx_path = xlsx_path + '.png'
    plt.figure(figsize=(8, 12), dpi=1000)
    #show two data
    plt.subplot(311)
    plt.title("mlu_hardware_time")
    plt.plot(df_compare_info['mlu_hardware_time' + suffix[0]].values,
             color='green',
             label='mlu_hardware_time' + suffix[0] + '(us)')
    plt.plot(df_compare_info['mlu_hardware_time' + suffix[1]].values,
             color='red',
             label='mlu_hardware_time' + suffix[1] + '(us)')
    plt.legend()

    #show data1-data2

    plt.subplot(312)
    plt.title('mlu_hardware_time reduction')
    plt.bar(range(0, len(df_compare_info)),
            df_compare_info['mlu_hardware_time_promotion'].values,
            label='mlu_hardware_time_promotion (us)',
            width=1)
    plt.plot([0, len(df_compare_info)], [0, 0], color='red', linewidth=1)
    plt.legend()

    #show data1/data2
    plt.subplot(313)
    plt.title('mlu_hardware_time reduction percentage')
    plt.bar(range(0, len(df_compare_info)),
            (df_compare_info['mlu_hardware_time_promotion_ratio'] *
             100.0).values,
            label='mlu_hardware_time_promotion_ratio(%)',
            width=1)
    plt.plot([0, len(df_compare_info)], [0, 0], color='red', linewidth=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(xlsx_path)

def dump_tpi_excel(dfs, sheet_names, tpi_path):
    dfs_to_excel(dfs, sheet_names, tpi_path)
    print("the tpi excel is " + tpi_path)

def compare_tpi(case_run, case_run_bl, tpi_dfs, tpi_baseline, sheet_names,
                tpi_compare_path, version_compare):
    summary_compare = pd.merge(tpi_dfs[0],
                               tpi_baseline[0],
                               suffixes=version_compare,
                               on=['网络名称'])
    dfs = []
    summary_compare['device时间提升(us)'] = summary_compare[
        '总device时间(us)' + version_compare[1]] - summary_compare['总device时间(us)' +
                                                             version_compare[0]]
    summary_compare['device时间提升比例'] = summary_compare[
        'device时间提升(us)'] / summary_compare['总device时间(us)' + version_compare[1]]
    summary_compare = summary_compare.sort_values(
                                by=[('device时间提升比例')],
                                ascending = False )
    summary_compare['device时间提升比例'] = summary_compare['device时间提升比例'].apply(
        "{:.2%}".format)
    summary_compare['host时间提升(us)'] = summary_compare[
        '总host时间(us)' + version_compare[1]] - summary_compare['总host时间(us)' + version_compare[0]]
    summary_compare['host时间提升比例'] = summary_compare[
        'host时间提升(us)'] / summary_compare['总host时间(us)' + version_compare[1]]
    summary_compare['host时间提升比例'] = summary_compare['host时间提升比例'].apply(
        "{:.2%}".format)

    summary_compare['workspace提升(Bytes)'] = summary_compare[
        '总workspace(Bytes)' +
        version_compare[1]] - summary_compare['总workspace(Bytes)' + version_compare[0]]
    summary_compare['workspace提升比例'] = summary_compare[
        'workspace提升(Bytes)'] / summary_compare['总workspace(Bytes)' +
                                                    version_compare[1]]
    summary_compare['workspace提升比例'] = summary_compare['workspace提升比例'].apply(
        "{:.2%}".format)
    # more hard code
    columns = [
        '网络名称', 'device时间提升(us)', 'device时间提升比例', 'host时间提升(us)', 'host时间提升比例',
        'workspace提升(Bytes)', 'workspace提升比例'
    ]
    for i in [
            '总device时间(us)', '总host时间(us)', '总workspace(Bytes)', '平均IO效率',
            '平均计算效率', '总个数', '平均device时间(us)', '平均host时间(us)',
            '平均workspace(Bytes)', 'IO瓶颈比例'
    ]:
        for j in version_compare:
            columns.append(i + j)
    dfs.append(summary_compare[columns])

    group_by_network = case_run.groupby("whole_name")
    group_by_network_bl = case_run_bl.groupby("whole_name")

    # assume cases are the same
    for network, index in group_by_network.groups.items():
        one_network = case_run[case_run['whole_name'] == network]
        one_network_bl = case_run_bl[case_run_bl['whole_name'] == network]
        merge_columns = ['md5', 'operator', 'file_path']
        tpi_compare = pd.merge(one_network,
                               one_network_bl,
                               suffixes=suffix,
                               on=merge_columns)
        tpi_compare['mlu_hardware_time_promotion'] = tpi_compare[
            'mlu_hardware_time' +
            suffix[1]] - tpi_compare['mlu_hardware_time' + suffix[0]]
        tpi_compare['mlu_hardware_time_promotion_ratio'] = tpi_compare[
            'mlu_hardware_time_promotion'] / tpi_compare['mlu_hardware_time' +
                                                         suffix[1]]
        tpi_compare['mlu_hardware_time_promotion_ratio'] = tpi_compare[
            'mlu_hardware_time_promotion_ratio'].apply("{:.2%}".format)

        tpi_compare['mlu_interface_time_promotion'] = tpi_compare[
            'mlu_interface_time' +
            suffix[1]] - tpi_compare['mlu_interface_time' + suffix[0]]
        tpi_compare['mlu_interface_time_promotion_ratio'] = tpi_compare[
            'mlu_interface_time_promotion'] / tpi_compare['mlu_interface_time'
                                                          + suffix[1]]
        tpi_compare['mlu_interface_time_promotion_ratio'] = tpi_compare[
            'mlu_interface_time_promotion_ratio'].apply("{:.2%}".format)

        tpi_compare['mlu_workspace_size_promotion'] = tpi_compare[
            'mlu_workspace_size' +
            suffix[1]] - tpi_compare['mlu_workspace_size' + suffix[0]]
        tpi_compare['mlu_workspace_size_promotion_ratio'] = tpi_compare[
            'mlu_workspace_size_promotion'] / tpi_compare['mlu_workspace_size'
                                                          + suffix[1]]
        tpi_compare['mlu_workspace_size_promotion_ratio'] = tpi_compare[
            'mlu_workspace_size_promotion_ratio'].apply("{:.2%}".format)
        # arange excel columns
        columns_ = [
            'operator', 'mlu_hardware_time_promotion',
            'mlu_hardware_time_promotion_ratio',
            'mlu_interface_time_promotion',
            'mlu_interface_time_promotion_ratio',
            'mlu_workspace_size_promotion',
            'mlu_workspace_size_promotion_ratio', 'count_new',
            'count_baseline', 'mlu_hardware_time_new',
            'mlu_hardware_time_baseline', 'mlu_interface_time_new',
            'mlu_interface_time_baseline', 'mlu_workspace_size_new',
            'is_io_bound_new', 'is_io_bound_baseline',
            'mlu_workspace_size_baseline', 'file_path'
        ]
        dfs.append(tpi_compare[columns_])

    dfs_to_excel(dfs, sheet_names, tpi_compare_path)
    print("the tpi comparison excel is " + tpi_compare_path)
    return dfs, sheet_names

def append_case_info(df, cpu_count, use_db):
    cases_info = pd.DataFrame(
        parse_file_path(df['file_path'].to_list(), cpu_count))
    for column in cases_info.columns:
        if column not in df.columns:
            df[column] = cases_info[column]

def parse_file_path(paths, cpu_count):
    with Pool(cpu_count) as pool:
        nodes_info = list(
            tqdm.tqdm(pool.imap(resolve_case, paths, chunksize=10),
                      total=len(paths),
                      ncols=80))

    result = {}
    for i in range(len(nodes_info)):
        merge_dict(result, nodes_info[i])

    return result

def resolve_case(path):
    node = mlu_op_test_pb2.Node()
    try:
        if path.endswith(".prototxt"):
            with open(path) as f:
                google.protobuf.text_format.Parse(f.read(), node)
        elif path.endswith(".pb"):
            with open(path, "rb") as f:
                node.ParseFromString(f.read())
    except Exception as e:
        print(e)
        print(
            "parse file {} failed, please check cntest.proto and file!".format(
                path))

    return get_node_info(node)

def get_node_info(node):
    pbParser = ProtobufCaseParserImplInputOutput(mlu_op_test_pb2)
    params = pbParser(node)
    magic_str = json.dumps(params)
    params['md5'] = hashlib.md5(magic_str.encode("utf-8")).hexdigest()
    return params

#  no test case operator and add platform insensitive operator
def update_operator_summary(operator_summary, engine):
    all_operator = get_operator_lists()
    no_case_operator = all_operator - set(
        operator_summary['operator'].to_list())
    df = pd.DataFrame({}, columns=operator_summary.columns)
    df['operator'] = list(no_case_operator)
    df.fillna(0, inplace=True)
    df['up_to_date'] = 1
    df['date'] = operator_summary.loc[operator_summary.shape[0] - 1, 'date']
    df['mlu_platform'] = operator_summary.loc[0, 'mlu_platform']
    df['job_limit'] = operator_summary.loc[0, 'job_limit']
    df['cluster_limit'] = operator_summary.loc[0, 'cluster_limit']
    df['batchsize'] = operator_summary.loc[0, 'batchsize']
    df.to_sql('mlu_op_operator_summary',
              con=engine,
              if_exists='append',
              index=False)

def walk_dir(networks_dir, method="listdir"):
    paths = []
    if method == "listdir":
        # listdir maybe faster if directory structure is fixed
        for network_name in os.listdir(networks_dir):
            network_name_path = os.path.join(networks_dir, network_name)
            if os.path.isdir(network_name_path):
                for network_whole_info in os.listdir(network_name_path):
                    network_whole_path = os.path.join(network_name_path,
                                                      network_whole_info)
                    if os.path.isdir(network_whole_path):
                        for op_name in os.listdir(network_whole_path):
                            op_name_path = os.path.join(
                                network_whole_path, op_name)
                            if os.path.isdir(op_name_path):
                                for fn in os.listdir(op_name_path):
                                    if fn.endswith('.prototxt') or fn.endswith(
                                            '.pb'):
                                        absolute_path = os.path.join(
                                            op_name_path, fn)
                                        paths.append(absolute_path)
    else:
        for root, dirs, files in os.walk(networks_dir):
            for fn in files:
                if fn.endswith(".prototxt") or fn.endswith(".pb"):
                    absolute_path = os.path.join(root, fn)
                    paths.append(absolute_path)

    return paths

def get_operator_lists():
    cwd = os.path.abspath(os.path.realpath(__file__) + "/../")
    paths = os.walk(os.path.abspath(os.path.join(cwd, "../../kernels")))
    operators = set()
    for roots, dirs, files in paths:
        for filename in files:
            if filename.endswith('.mlu') or filename.endswith(
                    '.h') or filename.endswith('.cpp'):
                absolute_path = os.path.join(roots, filename)
                with open(absolute_path) as f:
                    for line in f:
                        if 'GEN_CASE_START("' in line:
                            operator = line.split('GEN_CASE_START("')[1].split(
                                '"')[0]
                            operators.add(operator)
                            break

    return operators

def get_code_size(so_path):
    # get file size of operator.a and libmlu_op.so
    lib_path = os.path.abspath(so_path)
    cmd_args = ["readelf", "-e", lib_path]
    operator = []
    sizes = []

    cmd_ret = subprocess.run(cmd_args, check=True, stdout=subprocess.PIPE)
    so_size = re.findall(r"cn_fatbin(.*?) \[\d+\]", str(cmd_ret.stdout))[0]
    so_size = int(re.findall(r"\w+", so_size)[4], 16)
    operator.append('libmlu_op.so')
    sizes.append(os.path.getsize(lib_path))
    operator.append('cn_fatbin')
    sizes.append(so_size)

    data = {'operator': operator, 'size': sizes}
    df = pd.DataFrame(data)
    return df

def compare_code_size(code_size_bl, code_size_cmp):
    code_size_compare = pd.merge(code_size_bl,
                                 code_size_cmp,
                                 suffixes=suffix,
                                 on=['operator'])
    code_size_compare['size提升(Bytes)'] = code_size_compare[
        'size' + suffix[1]] - code_size_compare['size' + suffix[0]]
    code_size_compare['size提升比例(Bytes)'] = code_size_compare[
        'size提升(Bytes)'] / code_size_compare['size' + suffix[1]]
    code_size_compare['size提升比例(Bytes)'] = code_size_compare[
        'size提升比例(Bytes)'].apply("{:.2%}".format)
    return code_size_compare

def merge_xml(log_path, host_log_path, config):
    try:
        host_cases = parse_gtest_output(host_log_path, config)
        tree = ET.parse(log_path)
        testsuites = tree.getroot()
        for testsuite in testsuites:
            if 'DISABLED' not in testsuite.attrib['name']:
                for testcase in testsuite:
                    # can use xpath
                    found = False
                    for property in testcase.find('properties'):
                        # only update the same case_path
                        if property.attrib['name'] == 'case_path':
                            if property.attrib['value'] in host_cases[
                                    'case_path']:
                                case_index = host_cases['case_path'].index(
                                    property.attrib['value'])
                                host_time = host_cases['interface_time_mlu'][
                                    case_index]
                                found = True
                    if found:
                        for property in testcase.find('properties'):
                            if property.attrib['name'] == 'interface_time_mlu':
                                property.attrib['value'] = host_time

        tree.write("merge.xml")
        print("merge file is " + os.getcwd() + "/merge.xml")

    except Exception as e:
        print(e)
        exit()

# the format of network_name is: framework_name_mode_batchsize(option)_other(option)_version_date
def get_platforms_for_name(network_name, framework_name=None, additional=None):
    try:
        # see wiki 76995583
        info = network_name.split("_")
        if info[2] == "O0" or info[2] == "O1":
            platforms = "MLU290"
        elif info[1] == "cpm" and info[2] != "apex-O0":
            platforms = "MLU290"
        elif "mlu_opbenchmak-290" in network_name:
            platforms = "MLU290"
        elif "mlu_opbenchmak-all-cloud" in network_name:
            platforms = "MLU290 MLU370 MLU590"
        elif "tf32" in network_name:
            platforms = "MLU590"
        # TODO
        else:
            platforms = "MLU370 MLU590"

    except Exception as e:
        print(e)
        platforms = "MLU370-S4 MLU370-X4 MLU370-X8"

    return platforms

def generator_h5(cases_dir, cpu_count):
    import h5py
    try:
        # get file path
        paths = walk_dir(cases_dir, "walk")
        # get md5 for removing duplicates
        with Pool(cpu_count) as pool:
            nodes_info = list(
                tqdm.tqdm(pool.imap(resolve_case, paths, chunksize=10),
                          total=len(paths),
                          ncols=80))
        # get count for every case-network
        unique_filenames = {}
        networks = set()
        for i in range(len(nodes_info)):
            absolute_path = paths[i]
            network, operator = absolute_path.split('/')[-3:-1]
            networks.add(network)
            md5 = nodes_info[i]['md5']
            if md5 not in unique_filenames:
                unique_filenames[md5] = [paths[i], {network: 1}]
            else:
                unique_filenames[md5][1][
                    network] = unique_filenames[md5][1].get(network, 0) + 1
        # check whether pbName is unique
        path_count = {}
        for path in paths:
            pb_name = path.split('/')[-1]
            path_count[pb_name] = path_count.get(pb_name, 0) + 1
        duplicated_path = []
        for path in paths:
            pb_name = path.split('/')[-1]
            if path_count[pb_name] > 1:
                duplicated_path.append(path)
        if len(duplicated_path) > 0:
            raise Exception("PbNames have duplicated values")
        # get network properties
        mlu_op_network_list = pd.DataFrame(sorted(list(networks)),
                                         columns=['whole_name'])
        mlu_op_network_list['name'] = [
            '_'.join(i.split('_')[:-2])
            for i in mlu_op_network_list['whole_name']
        ]
        mlu_op_network_list['up_to_date'] = -mlu_op_network_list.duplicated(
            ['name'], keep='last')
        mlu_op_network_list_dict = mlu_op_network_list.to_dict('list')
        # generator h5
        h5_filename = "mlu_op_benchmark_raw_data.h5"
        with h5py.File(h5_filename, "w") as f:
            for md5, v in unique_filenames.items():
                pb_name = v[0].split('/')[-1]
                pb_group = f.create_group(pb_name)
                pb_group.create_dataset("file_path", data=v[0])
                for network, count in v[1].items():
                    network_group = pb_group.create_group(network)
                    network_info = network.split("_")
                    network_group.create_dataset("framework",
                                                 data=network_info[0])
                    network_group.create_dataset("network_name",
                                                 data=network_info[1])
                    network_group.create_dataset("mode", data=network_info[2])
                    batchsize = float(re.findall(
                        r'bs(\d+)', network)[0]) if len(
                            re.findall(r'bs(\d+)', network)) > 0 else 0
                    network_group.create_dataset("batchsize", data=batchsize)
                    network_group.create_dataset("version",
                                                 data=network_info[-2])
                    additional = "_".join(network.split("_")[3:-2])
                    if "bs" in network:
                        bs_str = "bs"
                        if len(re.findall(r'bs(\d+)', network)) > 0:
                            bs_str = bs_str + re.findall(r'bs(\d+)',
                                                         network)[0]
                        additional = additional.replace(bs_str, "").strip("_")
                    network_group.create_dataset("additional", data=additional)
                    network_index = mlu_op_network_list_dict['whole_name'].index(
                        network)
                    network_group.create_dataset(
                        "up_to_date",
                        data=mlu_op_network_list_dict['up_to_date']
                        [network_index])
                    platforms = set([
                        i.split("-")[0] if i != 'MLU365-D2' else i
                        for i in get_platforms_for_name(
                            network, network_info[0], additional).split(" ")
                    ])
                    platforms = ' '.join(platforms)
                    network_group.create_dataset("mlu_platform",
                                                 data=platforms)
                    network_group.create_dataset("count", data=count)
        print("h5 file is " + os.getcwd() + "/" + h5_filename)

    except Exception as e:
        print(e)
        print("\n".join(duplicated_path))

def pb_name_rename(cases_dir):
    try:
        # get file path
        paths = walk_dir(cases_dir, "walk")
        pb_names = set()
        for path in paths:
            pb_name = path.split("/")[-1]
            if pb_name not in pb_names:
                pb_names.add(pb_name)
            else:
                now_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
                u_time = str(int((time.time() % 1) * 1000000))
                new_pb_name = pb_name.split(
                    ".")[0] + '_' + now_time + '_' + u_time + '.prototxt'
                new_path = '/'.join(path.split("/")[:-1]) + "/" + new_pb_name
                os.rename(path, new_path)

    except Exception as e:
        print(e)