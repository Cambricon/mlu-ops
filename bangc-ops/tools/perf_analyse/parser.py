#!/usr/bin/env python3
#coding:utf-8

import re
import tqdm
import os
import pandas as pd
import logging
import hashlib
import json
import xml.etree.ElementTree as ET
import google.protobuf.text_format
from sqlalchemy import create_engine, types
from multiprocessing import Pool
from pathlib import Path


import mlu_op_test_pb2
from config import Config,PerfConfig
from protobuf_case_parser_impl_inputoutput import ProtobufCaseParserImplInputOutput

class Parser:

    def __init__(self, *args):
        if len(args) > 0:
            self.args_ = args[0]
        if len(args) > 1:
            self.db_ = args[1]
        self.perf_config = PerfConfig()

    def is_testsuite_for_mluop_op(self, test_name):
        if 'DISABLED' in test_name:
            return False
        if "ArrayCastHalfToFloatSelfTest" in test_name:
            return False
        if "/TestSuite" in test_name:
            return True
        return False

    def preprocess(self, df):
        # protoName and mlu_platform are used to merge database
        df['protoName'] = df['file_path'].apply(lambda x: x.split("/")[-1])
        df['mlu_platform'] = [
            i.split('[')[0] for i in df['mlu_platform'].to_list()
        ]
        # is_io_bound is considered in compute mean
        df['is_io_bound'] = df[[
            'mlu_theory_ios', 'mlu_iobandwidth', 'mlu_theory_ops',
            'mlu_computeforce'
        ]].apply(
            lambda x: (x['mlu_theory_ios'] / x['mlu_iobandwidth']) >
            (1000 * 1000 * 1000 * x['mlu_theory_ops'] / x['mlu_computeforce']),
            axis=1)

        def get_status(x, criterion):
            for k in criterion.keys():
                if criterion[k][0] <= x < criterion[k][1]:
                    return k
            return "invalid"

        # use efficiency by the bottleneck side to decide status
        df['status'] = df[[
            'mlu_io_efficiency', 'mlu_compute_efficiency', 'is_io_bound'
        ]].apply(lambda x: get_status(
            x['mlu_io_efficiency'] * x['is_io_bound'] + x[
                'mlu_compute_efficiency'] *
            (1 - x['is_io_bound']), self.perf_config.attrs['criterion']),
                 axis=1)

        if 'date' not in df.columns:
            df['date'] = pd.Timestamp.now()
        else:
            df['date'] = pd.to_datetime(df['date'], format="%Y_%m_%d_%H_%M_%S")
        df['date'] = df['date'].dt.strftime("%Y-%m-%d")

        if 'commit_id' in df.columns:
            commit_id = df.loc[0, 'commit_id']
            df['commit_id'] = commit_id[7:]
        else:
            df['commit_id'] = "UNK"

        if 'mluop_branch' in df.columns:
            df['mluop_branch'] = df['mluop_branch']
        else:
            df['mluop_branch'] = "UNK"

        if 'job_limit' not in df.columns or 'cluster_limit' not in df.columns:
            df['job_limit'] = 0
            df['cluster_limit'] = 0
        else:
            df['job_limit'] = [
                int(i) if len(i) > 0 else 0 for i in df['job_limit'].to_list()
            ]
            df['cluster_limit'] = [
                int(i) if len(i) > 0 else 0
                for i in df['cluster_limit'].to_list()
            ]

    # df.column = xml_properties_map.values() + case_info_keys + repeat_key
    #           + is_io_bound + protoName + md5(?)
    def parse_input(self, path, cpu_count=8, use_db=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise Exception("file {} not exists".format(path))

        print("parsing {}...".format(path))
        if os.path.isfile(path):
            df = self.parse_file(path)
        else:
            df = self.parse_directory(path)

        if len(df) == 0:
            raise Exception("no case has been run, please check cases config in pipeline!")

        self.preprocess(df)

        self.append_case_info(df, cpu_count, use_db)

        return df

    def parse_file(self, file_path):
        if file_path.endswith(".xml"):
            case_gen = self.xml_yield(file_path)
        else:
            case_gen = self.log_yield(file_path)
            logging.warn("{} is not a xml file!".format(file_path))

        cases = {}
        for case in case_gen:
            Parser.merge_dict(cases, case)
        return pd.DataFrame(cases)

    def parse_directory(self, directory_path):
        # parse all xml in the directory and compute mean
        ans = pd.DataFrame()
        dfs = []
        for file_path in Path(directory_path).glob("*"):
            # can check length
            dfs.append(self.parse_file(file_path.as_posix()))

        for column in Config.float_columns:
            s = []
            for df in dfs:
                s.append(df[column])
            ans[column] = pd.concat(s, axis=1).mean(axis=1)
        for column in set(df.columns) - set(Config.float_columns):
            ans[column] = dfs[0][column]

        return ans

    def merge_dict(a, b):
        # combine {'key1' : value1, 'key2' : value2}
        # and {'key1' : value3, 'key2' : value4}
        # to {'key1' : [value1, value3], 'key2' : [value2, value4]}
        if len(a.keys()) != len(b.keys()) and len(a.keys()) != 0:
            print("{} {}".format(len(a.keys()), len(b.keys())))
            raise Exception("not match {}\n{}".format(a, b))
        for k, v in b.items():
            if k not in a:
                a[k] = []
            a[k].append(v)

    def xml_yield(self, filename):
        tree = ET.parse(filename)
        testsuites = tree.getroot()
        for testsuite in testsuites.findall('testsuite'):
            if self.is_testsuite_for_mluop_op(testsuite.attrib['name']):
                for testcase in testsuite.findall('testcase'):
                    data = {
                        Config.xml_properties_map[property.attrib['name']]:
                        property.attrib['value']
                        for property in testcase.find('properties')
                        if property.attrib['name'] in
                        Config.xml_properties_map.keys()
                    }
                    for k in Config.float_columns:
                        data[k] = float(data[k])
                    for k in Config.repeat_key:
                        if k in testsuites.attrib.keys():
                            data[k] = testsuites.attrib[k]
                    yield data

    def log_yield(self, filename):
        result = {}
        with open(filename) as f:
            for line in f:
                for key in Config.log_keyword_map.keys():
                    if key in line:
                        if key == 'MLU Kernel Name(s)':
                            kernels = re.findall(r"\"(\w+)(?:<.*?>)?\":\s\d+", line)
                            value = [", ".join(kernels)]
                        else:
                            value = re.findall(r"\]:?\s*(\S+)\s?", line)
                        if key == 'RUN':
                            result[Config.log_keyword_map[key]] = value[
                                0].split('/')[0]
                        else:
                            result[Config.log_keyword_map[key]] = value[0]
                        if 'file_path' in result.keys():
                            for k in Config.float_columns:
                                result[k] = float(result[k])
                            # TODO(log_yield): use default now
                            result['mlu_platform'] = "MLU370-S4"
                            result['mluop_version'] = "unknown"
                            yield result
                            result = {}

    def append_case_info(self, df, cpu_count, use_db):
        if use_db:
            try:
                columns = Config.case_info_keys + ['protoName']
                tmp = pd.merge(df,
                               self.db_.case_list[columns],
                               on=['protoName'])
                if tmp.shape[0] != df.shape[0]:
                    raise Exception(
                        "some case not in database, parse pt directly")
                else:
                    for k in Config.case_info_keys:
                        df[k] = tmp[k]
            except Exception as e:
                print(e)
                cases_info = pd.DataFrame(
                    Parser.parse_cases(df['file_path'].to_list(), cpu_count))
                # md5 is appended
                for k in cases_info.keys():
                    df[k] = cases_info[k]
        else:
            cases_info = pd.DataFrame(
                Parser.parse_cases(df['file_path'].to_list(), cpu_count))
            # md5 is appended
            for k in cases_info.keys():
                df[k] = cases_info[k]

    def parse_cases(paths, cpu_count):
        with Pool(cpu_count) as pool:
            nodes_info = list(
                tqdm.tqdm(pool.imap(Parser.resolve_case, paths, chunksize=10),
                          total=len(paths),
                          ncols=80))

        result = {}
        for i in range(len(nodes_info)):
            Parser.merge_dict(result, nodes_info[i])

        return result

    def resolve_case(path):
        node = mlu_op_test_pb2.Node()
        if path.endswith(".prototxt"):
            with open(path) as f:
                google.protobuf.text_format.Parse(f.read(), node)
        elif path.endswith(".pb"):
            with open(path, "rb") as f:
                node.ParseFromString(f.read())

        return Parser.get_node_info(node)

    def get_node_info(node):
        pbParser = ProtobufCaseParserImplInputOutput(mlu_op_test_pb2)
        params = pbParser(node)
        # assure stability
        magic_str = json.dumps(params, sort_keys=True)
        params['md5'] = hashlib.md5(magic_str.encode("utf-8")).hexdigest()
        return params

    def run(self):
        logging.info("Parser run start")
        dfs = []
        if self.args_.log_path:
            logging.info("parsing log_path {}".format(self.args_.log_path))
            df = self.parse_input(self.args_.log_path, self.args_.cpu_count,
                                  self.args_.use_db)
            dfs.append(df)
        if self.args_.compare_path:
            logging.info("parsing compare_path {}".format(
                self.args_.compare_path))
            df_baseline = self.parse_input(self.args_.compare_path,
                                           self.args_.cpu_count,
                                           self.args_.use_db)
            dfs.append(df_baseline)

        logging.info("Parser run end")
        return dfs
