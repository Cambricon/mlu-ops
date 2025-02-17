"""
    present performance
"""

__all__ = (
    "process",
)

import sys
import logging
import pandas as pd

from analysis_suite.cfg.config import Config, ColDef
from analysis_suite.core.perf_analyser.perf import perf_utils
from analysis_suite.core.perf_analyser.perf.get_data_details import perf_cases, perf_operator_summary, perf_network_summary, perf_operator_summary_cri

def process(
        df: pd.DataFrame,
        **kwargs # perf_config, is_release, use_db, json_ops, is_pro, need_case_info
    ):
    # TODO: use 1 dict instead of 2 lists
    # TODO: extract filter of operators
    logging.info("{} start".format(sys._getframe().f_code.co_name))

    dfs = []
    sheet_names = []
    df.rename(columns={ColDef.time_stamp: ColDef.test_time}, inplace=True)
    df[ColDef.is_release] = kwargs['is_release']

    # cases 表
    cases = perf_cases.get_cases(df, kwargs['need_case_info'])
    dfs.append(cases)
    sheet_names.append('cases')

    # operator_summary 表
    summary = perf_operator_summary.get_operator_summary(df, kwargs['perf_config'], kwargs['is_release'])
    if None != kwargs['json_ops']:
        if kwargs['is_pro']:
            summary = summary[summary[ColDef.operator].isin(kwargs['json_ops'])]
        else:
            summary = summary[~summary[ColDef.operator].isin(kwargs['json_ops'])]
    if not summary.empty:
        dfs.append(summary)
        sheet_names.append('operator_summary')

    # means use database
    # network_time 表、 case_in_network 表
    if kwargs['use_db'] and len(df) > 0:
        network_time, case_in_network = perf_network_summary.get_network_summary(df, kwargs['perf_config'], kwargs['is_release'])
        if not network_time.empty:
            dfs.append(network_time)
            sheet_names.append('network_summary')
        if not case_in_network.empty:
            dfs.append(case_in_network)
            sheet_names.append('case_in_network')

    # operator_summary(cri) 表
    summary_under_criterion = perf_operator_summary_cri.get_operator_summary_under_criterion(df, kwargs['perf_config'])
    if None != kwargs['json_ops']:
        if kwargs['is_pro']:
            kwargs['json_ops'].add('_all_ops')
            summary_under_criterion = summary_under_criterion[summary_under_criterion[ColDef.operator].isin(kwargs['json_ops'])]
        else:
            summary_under_criterion = summary_under_criterion[~summary_under_criterion[ColDef.operator].isin(kwargs['json_ops'])]
    if not summary_under_criterion.empty:
        dfs.append(summary_under_criterion)
        sheet_names.append('operator_summary(cri)')

    # operators_in_json 表
    if None != kwargs['json_ops']:
        if kwargs['is_pro']:
            kwargs['json_ops'].discard('_all_ops')
        json_ops_df = pd.DataFrame(kwargs['json_ops'])
        json_ops_df.columns = [ColDef.operator]
    else:
        json_ops_df = pd.DataFrame(columns=[ColDef.operator])
    dfs.append(json_ops_df)
    sheet_names.append("operators_in_json")

    # operator_summary(cri)(io) & operator_summary(cri)(comp) 表
    if not summary_under_criterion.empty:
        dfs.append(summary_under_criterion[summary_under_criterion[ColDef.is_io_bound] == True])
        sheet_names.append('operator_summary(cri)(io)')
        dfs.append(summary_under_criterion[summary_under_criterion[ColDef.is_io_bound] == False])
        sheet_names.append('operator_summary(cri)(comp)')

    logging.info("{} end".format(sys._getframe().f_code.co_name))
    return dfs, sheet_names
