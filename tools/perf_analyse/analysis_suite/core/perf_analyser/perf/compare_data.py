"""
    present result of performance comparison 
"""

__all__ = (
    "compare_process",
)

import sys
import logging
import pandas as pd

from analysis_suite.cfg.config import Config, ColDef
from analysis_suite.core.perf_analyser.perf.compare_data_details import (
    perf_cases_compare,
    perf_operator_summary_compare,
    perf_network_summary_compare,
    perf_operator_summary_cri_compare
)

def compare_process(dfs_new, sheet_names_new, dfs_bl, sheet_names_bl, need_case_info):
    logging.info("{} start".format(sys._getframe().f_code.co_name))

    dfs = []
    sheet_names = []

    # 比较 cases 表
    dfs.append(perf_cases_compare.compare_cases(dfs_new[0], dfs_bl[0], need_case_info))
    sheet_names.append("cases_compare")

    # 比较 operator_summary 表
    tmp_idx_new = 1
    tmp_idx_bl = 1
    if 'operator_summary' in sheet_names_new and 'operator_summary' in sheet_names_bl:
        dfs.append(perf_operator_summary_compare.compare_operator_summary(dfs_new[tmp_idx_new], dfs_bl[tmp_idx_bl]))
        sheet_names.append("operator_summary_compare")
        tmp_idx_new += 1
        tmp_idx_bl += 1
    else:
        if 'operator_summary' in sheet_names_new:
            tmp_idx_new += 1
        if 'operator_summary' in sheet_names_bl:
            tmp_idx_bl += 1
        logging.warn("baseline or new's operator summary is empty, ignore to comapre!")

    # 比较 network_summary 表
    if 'network_summary' in sheet_names_new and 'network_summary' in sheet_names_bl:
        dfs.append(perf_network_summary_compare.compare_network_summary(dfs_new[tmp_idx_new], dfs_bl[tmp_idx_bl]))
        sheet_names.append("network_summary_compare")
        tmp_idx_new += 1
        tmp_idx_bl += 1
    else:
        if 'network_summary' in sheet_names_new:
            tmp_idx_new += 1
        if 'network_summary' in sheet_names_bl:
            tmp_idx_bl += 1
        logging.warn("baseline or new's network summary is empty, ignore to comapre!")

    if 'case_in_network' in sheet_names_new and 'case_in_network' in sheet_names_bl:
        tmp_idx_new += 1
        tmp_idx_bl += 1
    else:
        if 'case_in_network' in sheet_names_new:
            tmp_idx_new += 1
        if 'case_in_network' in sheet_names_bl:
            tmp_idx_bl += 1

    # 比较 operator_summary_cri 表
    if 'operator_summary(cri)' in sheet_names_new and 'operator_summary(cri)' in sheet_names_bl:
        dfs.append(perf_operator_summary_cri_compare.compare_operator_summary_cri(dfs_new[tmp_idx_new], dfs_bl[tmp_idx_bl]))
        sheet_names.append("op_summary_compare(cri)")
        tmp_idx_new += 1
        tmp_idx_bl += 1
    else:
        if 'operator_summary(cri)' in sheet_names_new:
            tmp_idx_new += 1
        if 'operator_summary(cri)' in sheet_names_bl:
            tmp_idx_bl += 1
        logging.warn("baseline or new's operator summary under criterion is empty, ignore to comapre!")

    if 'operators_in_json' in sheet_names_new:
        tmp_idx_new += 1
    if 'operators_in_json' in sheet_names_bl:
        tmp_idx_bl += 1

    if 'operator_summary(cri)(io)' in sheet_names_new and 'operator_summary(cri)(io)' in sheet_names_bl:
        dfs.append(perf_operator_summary_cri_compare.compare_operator_summary_cri(dfs_new[tmp_idx_new], dfs_bl[tmp_idx_bl]))
        sheet_names.append("op_summary_compare(cri)(io)")
        tmp_idx_new += 1
        tmp_idx_bl += 1
    else:
        if 'operator_summary(cri)(io)' in sheet_names_new:
            tmp_idx_new += 1
        if 'operator_summary(cri)(io)' in sheet_names_bl:
            tmp_idx_bl += 1
        logging.warn("baseline or new's io bound operator summary under criterion is empty, ignore to comapre!")

    if 'operator_summary(cri)(comp)' in sheet_names_new and 'operator_summary(cri)(comp)' in sheet_names_bl:
        dfs.append(perf_operator_summary_cri_compare.compare_operator_summary_cri(dfs_new[tmp_idx_new], dfs_bl[tmp_idx_bl]))
        sheet_names.append("op_summary_compare(cri)(comp)")
        tmp_idx_new += 1
        tmp_idx_bl += 1
    else:
        if 'operator_summary(cri)(comp)' in sheet_names_new:
            tmp_idx_new += 1
        if 'operator_summary(cri)(comp)' in sheet_names_bl:
            tmp_idx_bl += 1
        logging.warn("baseline or new's comp bound operator summary under critercompn is empty, ignore to comapre!")

    logging.info("{} end".format(sys._getframe().f_code.co_name))
    return dfs, sheet_names
