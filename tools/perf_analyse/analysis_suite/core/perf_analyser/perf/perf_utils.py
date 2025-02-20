"""
    helper functions for perf
"""

__all__ = (
    "dump_perf_result_to_excel",
    "dump_compare_result_to_excel",
    "generate_pic",
)

import sys
import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sqlalchemy import types
from typing import List

from analysis_suite.cfg.config import Config, ColDef
from analysis_suite.utils import excel_helper
# remove duplicate columns
def dfs_deduplication(dfs):
    for i in range(0, len(dfs)):
        if ColDef.md5 in dfs[i].columns:
            dfs[i].drop_duplicates(subset=ColDef.md5, ignore_index=True, inplace=True)

# remove duplicate columns
def dfs_deduplication(dfs):
    for i in range(0, len(dfs)):
        if ColDef.md5 in dfs[i].columns:
            dfs[i].drop_duplicates(subset=ColDef.md5, ignore_index=True, inplace=True)

def dfs_to_excel(dfs, sheet_names, xlsx_path):
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
        for i in range(len(dfs)):
            if sheet_names[i] in ['operator_summary(cri)', 'operator_summary_compare(cri)']:
                excel_helper.to_sheet_helper(
                        dfs[i],
                        writer,
                        sheet_names[i],
                        merge_cell_col_idx_lst=[0, 1]
                    )
            else:
                excel_helper.to_sheet_helper(
                        dfs[i],
                        writer,
                        sheet_names[i]
                    )

def dump_perf_result_to_excel(dfs, sheet_names, xlsx_path, deduplication):
    # remove duplication
    if deduplication:
        dfs_deduplication(dfs)

    # write excel
    dfs_to_excel(dfs, sheet_names, xlsx_path)

    logging.info("the output excel is %s" , xlsx_path)

def dump_compare_result_to_excel(dfs, sheet_names, xlsx_path):
    # write excel
    dfs_to_excel(dfs, sheet_names, xlsx_path)

    logging.info("the output comparison excel is %s", xlsx_path)

def generate_pic(df_compare_info, pic_path):
    # the following two sentences support Chinese SimHei
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # matplotlib.rcParams['axes.unicode_minus'] = False
    logging.info("{} start".format(sys._getframe().f_code.co_name))

    plt.figure(figsize=(8, 12), dpi=1000)

    #show two data
    plt.subplot(311)
    plt.title("mlu_hardware_time")
    plt.plot(df_compare_info[ColDef.mlu_hardware_time +
                                Config.suffix[0]].values,
                color='green',
                label=ColDef.mlu_hardware_time + Config.suffix[0] + '(us)')
    plt.plot(df_compare_info[ColDef.mlu_hardware_time +
                                Config.suffix[1]].values,
                color='red',
                label=ColDef.mlu_hardware_time + Config.suffix[1] + '(us)')
    plt.legend()

    #show data1-data2
    plt.subplot(312)
    plt.title('mlu_hardware_time reduction')
    plt.bar(range(0, len(df_compare_info)),
            df_compare_info[ColDef.mlu_hardware_time_promotion].values,
            label='mlu_hardware_time_promotion (us)',
            width=1)
    plt.plot([0, len(df_compare_info)], [0, 0], color='red', linewidth=1)
    plt.legend()

    #show data1/data2
    plt.subplot(313)
    plt.title('mlu_hardware_time reduction percentage')
    plt.bar(range(0, len(df_compare_info)),
            (df_compare_info[ColDef.mlu_hardware_time_promotion_ratio] *
                100.0).values,
            label='mlu_hardware_time_promotion_ratio(%)',
            width=1)
    plt.plot([0, len(df_compare_info)], [0, 0], color='red', linewidth=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pic_path)

    logging.info("{} end".format(sys._getframe().f_code.co_name))
