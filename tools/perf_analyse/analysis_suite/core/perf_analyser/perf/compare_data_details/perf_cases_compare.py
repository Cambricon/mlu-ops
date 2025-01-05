"""
    Table: cases_compare
"""

__all__ = (
    "compare_cases"
)

import sys
import logging
import pandas as pd

from analysis_suite.cfg.config import Config, ColDef
from analysis_suite.core.perf_analyser import compare_details

def compare_cases(
        df_new: pd.DataFrame,
        df_bl: pd.DataFrame,
        need_case_info
    ) -> pd.DataFrame:
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    # set the columns needed for compare
    if need_case_info:
        info_columns = [
            ColDef.operator,
            ColDef.file_path,
            ColDef.gpu_hardware_time,
            ColDef.input,
            ColDef.output,
            ColDef.params,
            ColDef.is_io_bound
        ]
    else:
        info_columns = [
            ColDef.operator,
            ColDef.file_path,
            ColDef.gpu_hardware_time,
            #ColDef.input,
            #ColDef.output,
            #ColDef.params,
            ColDef.is_io_bound
        ]

    perf_columns = [
        ColDef.mlu_hardware_time,
        ColDef.mlu_io_efficiency,
        ColDef.mlu_compute_efficiency,
        ColDef.mlu_interface_time,
        ColDef.mlu_workspace_size,
        ColDef.mlu_kernel_names,
        ColDef.mlu_hardware_time_sum
    ]
    promotion_columns = [ColDef.mlu_hardware_time, ColDef.mlu_hardware_time_sum]

    cases_compare = \
        compare_details.compare(
            df_new, df_bl,
            ColDef.protoName,
            info_columns,
            perf_columns,
            promotion_columns
        )

    # assure column order
    result_columns = [
        ColDef.operator,
        ColDef.mlu_hardware_time_promotion,
        ColDef.mlu_hardware_time_promotion_ratio
    ]
    result_columns += [
        m + n for m in perf_columns for n in Config.suffix
    ]
    result_columns += info_columns[1:]

    # sort by mlu_hardware_time_promotion
    cases_compare = \
        cases_compare[result_columns].sort_values(
            by=[ColDef.mlu_hardware_time_promotion],
            ascending=False
        )

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return cases_compare
