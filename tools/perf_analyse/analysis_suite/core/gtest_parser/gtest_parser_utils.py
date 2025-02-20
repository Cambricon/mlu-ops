"""
    utility for package `gtest_parser`
"""

__all__ = (
    "get_paths",
    "merge_dict",
    "append_env_info",
)

import os
import logging
import pandas as pd
from typing import List, Optional, Dict

from analysis_suite.cfg.config import Config, ColDef
from analysis_suite.core.gtest_parser import test_info

def get_paths(
        log_path: Optional[str],
        compare_path: Optional[str]
    ) -> List[Optional[str]]:
    # get paths from input
    input_paths = []
    if log_path:
        logging.debug("log_path is {}".format(log_path))
        input_paths.append(log_path)
        if compare_path:
            logging.debug("compare_path is {}".format(compare_path))
            input_paths.append(compare_path)

    # check validity
    paths = []
    for input_path in input_paths:
        path = os.path.expanduser(input_path)
        if not os.path.exists(path):
            raise Exception("file {} not exists".format(path))
        paths.append(path)

    return paths

def append_env_info(
        test_info_lst: List[Optional[test_info.TestInfo]]
    ) -> List[Optional[pd.DataFrame]]:
    dfs = []

    for info in test_info_lst:
        df = info.perf
        # add environment information to performance dataframe
        for k, v in info.env.items():
            if k in Config.environment_keys:
                df[k] = v
        # add `date` using `time_stamp`
        if ColDef.time_stamp in info.env.keys():
            df[ColDef.date] = info.env[ColDef.time_stamp].strftime("%Y-%m-%d")
        dfs.append(df)

    return dfs
