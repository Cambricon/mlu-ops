"""
    Table: network_summary
"""

__all__ = (
    "compare_network_summary"
)

import sys
import logging
import pandas as pd

from analysis_suite.cfg.config import Config, ColDef
from analysis_suite.core.perf_analyser import compare_details

def compare_network_summary(
        df_new: pd.DataFrame,
        df_bl: pd.DataFrame
    ) -> pd.DataFrame:
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    # set the columns needed for compare
    info_columns = Config.network_info_keys
    perf_columns = Config.summary_columns
    promotion_columns = [ColDef.mlu_hardware_time_sum]

    network_summary_compare = \
        compare_details.compare(
            df_new, df_bl,
            ColDef.network_id,
            info_columns,
            perf_columns,
            promotion_columns
        )
    network_summary_compare[ColDef.network_time] = df_new[ColDef.mlu_hardware_time_sum_database]
    network_summary_compare[ColDef.ops_promotion_to_network] = \
        network_summary_compare[ColDef.mlu_hardware_time_sum_promotion] / \
        network_summary_compare[ColDef.network_time] if network_summary_compare[ColDef.network_time] is not None else None
    
    # assure column order
    result_columns = [
        ColDef.network_id,
        ColDef.network_name,
        ColDef.mlu_hardware_time_sum_promotion,
        ColDef.mlu_hardware_time_sum_promotion_ratio,
        ColDef.ops_promotion_to_network
    ]
    result_columns += [
        m + n for m in perf_columns for n in Config.suffix
    ]
    result_columns += [
        ColDef.network_time,
        ColDef.framework,
        ColDef.precision_mode,
        ColDef.batchsize,
        ColDef.network_additional_information,
        ColDef.project_version
    ]
    # select columns and sort by mlu_hardware_time_sum_promotion
    network_summary_compare = \
        network_summary_compare[result_columns].sort_values(
            by=[ColDef.mlu_hardware_time_sum_promotion],
            ascending=False
        )

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return network_summary_compare