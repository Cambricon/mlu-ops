"""
    gtest Log parser
    ********** NOT SUPPORTED NOW! **********

    Input: Log file.
    Output: test_info.TestInfo. If there's some case failed, export the list.
"""

__all__ = (
    "parse_gtest_log",
)

import logging
import re
from analysis_suite.cfg.config import Config, ColDef, PerfConfig
from analysis_suite.core.gtest_parser import gtest_parser_utils, test_info

def log_yield(filename: str):
    result = {}
    with open(filename) as f:
        for line in f:
            for key in Config.log_keyword_map.keys():
                if key not in line:
                    continue

                if key == 'MLU Kernel Name(s)':
                    kernels = re.findall(r"\"(\w+)(?:<.*?>)?\":\s\d+", line)
                    value = [", ".join(kernels)]
                else:
                    value = re.findall(r"\]:?\s*(\S+)\s?", line)
                if key == 'RUN':
                    result[Config.log_keyword_map[key]] = value[0].split('/')[0]
                else:
                    result[Config.log_keyword_map[key]] = value[0]
                if ColDef.file_path in result.keys():
                    for k in Config.float_columns:
                        result[k] = float(result[k])
                    # TODO(log_yield): use default now
                    result[ColDef.mlu_platform] = "MLU370-S4"
                    result[ColDef.mluops_version] = "unknown"
                    yield result
                    result = {}

def parse_gtest_log(file_path: str) -> test_info.TestInfo:
    # case_gen = log_yield(file_path)
    logging.warn("{} is not a xml file, do not support log".format(file_path))
    return test_info.TestInfo()

