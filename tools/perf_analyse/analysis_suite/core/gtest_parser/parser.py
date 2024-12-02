"""
    external interface of package `gtest_parser`
"""

__all__ = (
    "parse_into_dataframes",
)

import sys
import logging
import json
from typing import Optional
from concurrent.futures import ProcessPoolExecutor

from analysis_suite.cfg.config import Config, PerfConfig
from analysis_suite.core.gtest_parser import gtest_parser_utils, case_parser, case_info

def parse_into_dataframes(
        log_path: Optional[str],
        compare_path: Optional[str],
        **kwargs
    ):
    logging.info("{} start".format(sys._getframe().f_code.co_name))

    """
    To adapt to performance database, the logic of this module is as follow:
        1. Check validity of input paths.
        2. Read performance data and environment information,
        and return it with structure `analysis_suite.core.gtest_parser.test_info.TestInfo`.
        These data can be read from:
            a. XML/log file output by gtest;
            b. performance database.
        3. Append case information to performance data.
        4. Add environment information to the performance data, which is stored in
        `pandas.DataFrame`, to adapt to subsequent interfaces.
    """
    # set paths and check its validity
    paths = gtest_parser_utils.get_paths(log_path, compare_path)

    # parse input to get performance data
    # TODO(tanghandi): read from database?
    test_info_lst = []
    perf_config = PerfConfig()
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = \
            [pool.submit(case_parser.parse_input,
                    path,
                    perf_config,
                    kwargs['cpu_count'],
                    kwargs['filter_failed_cases'],
                    kwargs['export_failed_cases'],
                ) for path in paths
            ]
        for future in futures:
            try:
                rc = future.result()
                test_info_lst.append(rc)
            except Exception:
                raise

    # append case info to the performance data
    if kwargs['need_case_info']:
        case_info.append_case_info(test_info_lst,  kwargs['cpu_count'], kwargs['use_db'])

    # merge environment information to performance information
    dfs = gtest_parser_utils.append_env_info(test_info_lst)

    logging.info("{} end".format(sys._getframe().f_code.co_name))
    return dfs

