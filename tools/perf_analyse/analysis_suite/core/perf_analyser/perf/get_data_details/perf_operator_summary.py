"""
    Table: operator_summary
"""

__all__ = (
    "get_operator_summary",
)

import sys
import logging
import pandas as pd

from analysis_suite.cfg.config import ColDef
from analysis_suite.core.perf_analyser.perf.get_data_details import perf_get_data_utils

def get_operator_summary_impl(df, perf_config):
    if df.empty:
        logging.warn("The test cases are all small/abnormal cases.")
        return pd.DataFrame()

    summary = perf_get_data_utils.get_summary(df, ColDef.operator, perf_config)
    if summary.empty:
        logging.warn("The table operator_summary is empty.")
        return pd.DataFrame()

    return summary.sort_values(by=[ColDef.mlu_hardware_time_proportion], ascending=False)

def get_operator_summary(df, perf_config, is_release):
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    # reserve statistic columns
    reserved_column_values = perf_get_data_utils.reserve_column_values(df, is_release)

    summary = get_operator_summary_impl(df, perf_config)

    # append reserved columns
    if not summary.empty:
        for k, v in reserved_column_values.items():
            summary[k] = v

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return summary
