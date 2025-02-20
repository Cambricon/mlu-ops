"""
    Table: operator_summary
"""

__all__ = (
    "compare_operator_summary"
)

import sys
import logging
import pandas as pd

from analysis_suite.cfg.config import Config, ColDef
from analysis_suite.core.perf_analyser import compare_details

def compare_operator_summary(
        df_new: pd.DataFrame,
        df_bl: pd.DataFrame
    ) -> pd.DataFrame:
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    # set the columns needed for compare
    info_columns = [ColDef.operator]
    perf_columns = Config.summary_columns
    promotion_columns = [ColDef.mlu_hardware_time_sum]

    operator_summary_compare = \
        compare_details.compare(
            df_new, df_bl, ColDef.operator,
            info_columns,
            perf_columns,
            promotion_columns
        )

    # assure column order
    result_columns = [
        ColDef.operator,
        ColDef.mlu_hardware_time_sum_promotion,
        ColDef.mlu_hardware_time_sum_promotion_ratio
    ]
    result_columns += [
        m + n for m in perf_columns for n in Config.suffix
    ]
    # select columns and sort by mlu_hardware_time_sum_promotion
    operator_summary_compare = \
        operator_summary_compare[result_columns].sort_values(
            by=[ColDef.mlu_hardware_time_sum_promotion],
            ascending=False
        )

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return operator_summary_compare
