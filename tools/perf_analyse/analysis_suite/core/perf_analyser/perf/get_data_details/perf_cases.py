"""
    Table: cases
"""

__all__ = (
    "get_cases"
)

import sys
import logging
import pandas as pd

from analysis_suite.cfg.config import ColDef

def cases_column_helper(df_columns, need_case_info):
    # assure column order
    if need_case_info:
        columns = [
            ColDef.operator,
            ColDef.mlu_hardware_time,
            ColDef.gpu_hardware_time,
            ColDef.mlu_io_efficiency,
            ColDef.mlu_compute_efficiency,
            ColDef.mlu_interface_time,
            ColDef.mlu_workspace_size,
            ColDef.file_path,
            ColDef.repeat_num,
            ColDef.mlu_hardware_time_sum,
            ColDef.input,
            ColDef.output,
            ColDef.params,
            ColDef.mlu_theory_ios,
            ColDef.mlu_theory_ops,
            ColDef.is_io_bound,
            ColDef.commit_id,
            ColDef.mluops_branch,
            ColDef.test_time,
            ColDef.mlu_platform,
            ColDef.mluops_version,
            ColDef.protoName,
            ColDef.status,
            ColDef.is_release
        ]
    else:
        columns = [
            ColDef.operator,
            ColDef.mlu_hardware_time,
            ColDef.gpu_hardware_time,
            ColDef.mlu_io_efficiency,
            ColDef.mlu_compute_efficiency,
            ColDef.mlu_interface_time,
            ColDef.mlu_workspace_size,
            ColDef.file_path,
            ColDef.repeat_num,
            ColDef.mlu_hardware_time_sum,
            #ColDef.input,
            #ColDef.output,
            #ColDef.params,
            ColDef.mlu_theory_ios,
            ColDef.mlu_theory_ops,
            ColDef.is_io_bound,
            ColDef.commit_id,
            ColDef.mluops_branch,
            ColDef.test_time,
            ColDef.mlu_platform,
            ColDef.mluops_version,
            ColDef.protoName,
            ColDef.status,
            ColDef.is_release
        ]

    if ColDef.md5 in df_columns:
        columns.append(ColDef.md5)

    if ColDef.driver_version in df_columns and ColDef.cnrt_version in df_columns:
        columns.append(ColDef.driver_version)
        columns.append(ColDef.cnrt_version)
    if ColDef.mlu_kernel_names in df_columns:
        columns.append(ColDef.mlu_kernel_names)

    return columns

def get_cases(df: pd.DataFrame, need_case_info: bool):
    logging.debug("{} start".format(sys._getframe().f_code.co_name))
    columns = cases_column_helper(df.columns, need_case_info)

    df[ColDef.mlu_hardware_time_sum] = df[ColDef.repeat_num] * df[ColDef.mlu_hardware_time]

    cases = df[columns].sort_values(by=[ColDef.mlu_hardware_time_sum], ascending=False)

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return cases
