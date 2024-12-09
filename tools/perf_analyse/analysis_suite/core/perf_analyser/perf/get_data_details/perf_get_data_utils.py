"""
    helper functions for get_data_details
"""

__all__ = (
    "ignore_useless_cases",
    "get_summary",
    "select_rows",
    "reserve_column_values",
    "set2df"
)

import sys
import logging
import pandas as pd
from typing import Set

from analysis_suite.cfg.config import Config, ColDef, PerfConfig

# ignore abnormality case
def ignore_useless_cases(
        df: pd.DataFrame,
        perf_config: PerfConfig
    ) -> pd.DataFrame:
    # filter small cases
    df = df[df[ColDef.mlu_hardware_time] > perf_config.attrs['ignore_case'][ColDef.mlu_hardware_time]]
    # filter abnormal cases
    df = df[(df[ColDef.mlu_io_efficiency] >= 0 & df[ColDef.is_io_bound]) | (df[ColDef.mlu_compute_efficiency] >= 0 & (~df[ColDef.is_io_bound]))]
    return df

def get_summary_impl(df, group_key, perf_config):
    if len(df) == 0:
        logging.warn("input table is empty, group_key={}".format(group_key))
        return pd.DataFrame()

    group_all_case = df.groupby(group_key)
    # assure column order
    summary = pd.DataFrame()
    summary[ColDef.all_case_number] = group_all_case.apply(lambda x: x[ColDef.repeat_num].sum())

    # ignore small/abnormal cases
    df = ignore_useless_cases(df, perf_config)
    group_ignore_useless_cases = df.groupby(group_key)
    if len(group_ignore_useless_cases) == 0:
        logging.warn("The cases are all small or abnormal cases,ignored to get_{}_summary!".format(group_key))
        return pd.DataFrame()

    # compute numbers
    summary[ColDef.filtered_case_number]    = group_ignore_useless_cases.apply(lambda x: x[ColDef.repeat_num].sum())
    summary[ColDef.io_bound_case_number]        = group_ignore_useless_cases.apply(lambda x: x[x[ColDef.is_io_bound]][ColDef.repeat_num].sum())
    summary[ColDef.compute_bound_case_number]   = group_ignore_useless_cases.apply(lambda x: x[~x[ColDef.is_io_bound]][ColDef.repeat_num].sum())

    # compute mean efficiency by the bottleneck side
    summary[ColDef.mlu_io_efficiency_mean]      = group_ignore_useless_cases.apply(lambda x: x[x[ColDef.is_io_bound]][ColDef.mlu_io_efficiency].sum()) / summary[ColDef.io_bound_case_number]
    summary[ColDef.mlu_compute_efficiency_mean] = group_ignore_useless_cases.apply(lambda x: x[~x[ColDef.is_io_bound]][ColDef.mlu_compute_efficiency].sum()) / summary[ColDef.compute_bound_case_number]

    # compute sum of device time
    summary[ColDef.mlu_hardware_time_sum]       = group_ignore_useless_cases.apply(lambda x: (x[ColDef.mlu_hardware_time] * x[ColDef.repeat_num]).sum())
    # time proportion
    total_mlu_hardware_time_sum = summary[ColDef.mlu_hardware_time_sum].sum()
    summary[ColDef.mlu_hardware_time_proportion] = summary[ColDef.mlu_hardware_time_sum] / total_mlu_hardware_time_sum

    # calculate percentage of cases in each status
    for k in perf_config.attrs['criterion'].keys():
        summary[k + '_rate'] = group_ignore_useless_cases.apply(lambda x: x[x[ColDef.status] == k][ColDef.repeat_num].sum())
        summary[k + '_rate'] = summary[k + '_rate'] / summary[ColDef.filtered_case_number]

    # efficiency can be nan because of all case is io bound
    summary.fillna(0, inplace=True)
    summary.reset_index(inplace=True)

    return summary

# summary.columns = [
#   group_key,
#   all_case_number, filtered_case_number,
#   io_bound_case_number, compute_bound_case_number,
#   mlu_io_efficiency_mean, mlu_compute_efficiency_mean, mlu_hardware_time_sum,
#   good_rate, qualified_rate, unqualified_rate
# ]
def get_summary(
        df: pd.DataFrame,
        group_key: str,
        perf_config: PerfConfig
    ) -> pd.DataFrame:
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    summary = get_summary_impl(df, group_key, perf_config)

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return summary

# select rows by key
def select_rows(
        df: pd.DataFrame,
        key_column: str,
        key_set: Set[str],
        is_pro: bool
    ) -> pd.DataFrame:
    if None != key_set:
        if self.is_pro:
            df = df[df[key_column].isin(key_set)]
        else:
            df = df[~df[key_column].isin(key_set)]
    return df

def reserve_column_values(
        df: pd.DataFrame,
        is_release: bool
    ):
    column_values = {
        ColDef.mlu_platform: df.loc[0, ColDef.mlu_platform],
        ColDef.mluops_version: df.loc[0, ColDef.mluops_version],
        #ColDef.date: df.loc[0, ColDef.date],
        ColDef.test_time: df.loc[0, ColDef.test_time],
        ColDef.commit_id: df.loc[0, ColDef.commit_id],
        ColDef.mluops_branch: df.loc[0, ColDef.mluops_branch],
        ColDef.is_release: is_release,
    }
    return column_values

# set to dataframe
# the output dataframe only has one column
def set2df(input_set: Set, key_column: str):
    if None != json_op_set:
        output_df = pd.DataFrame(input_set)
        output_df.columns = [key_column]
    else:
        output_df = pd.DataFrame(columns=[key_column])

    return output_df
