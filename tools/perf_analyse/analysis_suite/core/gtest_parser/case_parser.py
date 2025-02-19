
"""
    parse xml/log file output by gtest, and retun test_info.TestInfo
    If there's some case failed, output the list of failed cases.
"""

__all__ = (
    "parse_input",
)

import os
import pandas as pd
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
import json

from analysis_suite.cfg.config import Config, ColDef, PerfConfig
from analysis_suite.core.gtest_parser import gtest_parser_utils, test_info
from analysis_suite.core.gtest_parser.case_parser_details import gtest_xml_parser, gtest_log_parser
from analysis_suite.utils import path_helper

def parse_file(
        file_path: str,
        filter_failed_cases: bool,
        export_failed_cases: bool,
    ) -> test_info.TestInfo:
    if file_path.endswith(".xml"): # parse xml
        rc = gtest_xml_parser.parse_gtest_xml(file_path,
                filter_failed_cases,
                export_failed_cases
            )
    else: # parse log(not support now)
        rc = gtest_log_parser.parse_gtest_log(file_path)

    return rc

# parse all xml in the directory and compute mean
# do not care environment information here
def parse_directory(
        directory_path: str,
        filter_failed_cases: bool,
        export_failed_cases: bool,
    ) -> test_info.TestInfo:
    ans = pd.DataFrame()

    dfs = []
    for file_path in Path(directory_path).glob("*"):
        data = parse_file(file_path.as_posix(),
                filter_failed_cases,
                export_failed_cases,
            )
        dfs.append(data.perf)

    for column in Config.float_columns:
        s = []
        for df in dfs:
            s.append(df[column])
        ans[column] = pd.concat(s, axis=1).mean(axis=1)
    for column in (set(df.columns) - set(Config.float_columns)):
        ans[column] = dfs[0][column]

    return test_info.TestInfo(None, ans)

# append information to performance data
def preprocess(df: pd.DataFrame, perf_config: PerfConfig):
    # protoName and mlu_platform are used to merge database
    df['protoName'] = df['file_path'].apply(lambda x: x.split("/")[-1])
    # is_io_bound is considered in compute mean
    df['is_io_bound'] = \
        df[
            [
                'mlu_theory_ios',
                'mlu_iobandwidth',
                'mlu_theory_ops',
                'mlu_computeforce'
            ]
        ].apply(
            lambda x: (x['mlu_theory_ios'] / x['mlu_iobandwidth']) > \
                (1000 * 1000 * 1000 * x['mlu_theory_ops'] / x['mlu_computeforce']),
            axis=1
        )

    def get_status(x, criterion):
        for k in criterion.keys():
            if criterion[k][0] < x and x <= criterion[k][1]:
                return k
        return "invalid"
    # use efficiency by the bottleneck side to decide status
    df['status'] = \
        df[
            [
                'mlu_io_efficiency',
                'mlu_compute_efficiency',
                'is_io_bound'
            ]
        ].apply(
            lambda x: get_status(
                x['mlu_io_efficiency'] * x['is_io_bound'] + \
                x['mlu_compute_efficiency'] * (1 - x['is_io_bound']),
                perf_config.attrs['criterion']
            ),
            axis=1
        )

# output:
#   TestInfo:
#   env:
#       Config.environment_keys
#   perf: [
#       Config.xml_properties_map.values(),
#       Config.case_info_keys,
#       is_io_bound,
#       protoName,
#       md5
#   ]
def parse_input(
        path: str,
        perf_config: PerfConfig,
        cpu_count: int,
        filter_failed_cases: bool = False,
        export_failed_cases: bool = False,
    ) -> test_info.TestInfo:
    logging.info("parsing {}...".format(path))

    # parse file/directory to test_info.TestInfo
    if os.path.isfile(path):
        data = parse_file(path, filter_failed_cases, export_failed_cases)
    else:
        data = parse_directory(path, filter_failed_cases, export_failed_cases)
    if len(data.perf) == 0:
        raise Exception("no case has been run, please check cases config in pipeline!")

    preprocess(data.perf, perf_config)

    logging.info("finish parsing {}, there are {} cases.".format(path, len(data.perf)))
    return data

