"""
    Table: operator_summary(cri)
    Demand Source: http://jira.cambricon.com/browse/CNNLCORE-19647
    just concat two dataframes
"""

__all__ = (
    "compare_operator_summary_cri"
)

import sys
import logging
import pandas as pd

from analysis_suite.cfg.config import Config, ColDef

def compare_operator_summary_cri(
        df_new: pd.DataFrame,
        df_bl: pd.DataFrame
    ) -> pd.DataFrame:
    logging.debug("{} start".format(sys._getframe().f_code.co_name))
    # add suffix
    df_new[ColDef.operator] = df_new[ColDef.operator] + Config.suffix[0]
    df_bl[ColDef.operator] = df_bl[ColDef.operator] + Config.suffix[1]

    # concat
    df = pd.concat([df_new, df_bl], ignore_index=True)

    # sort by operator name and is_io_bound
    df = df.sort_values(by=[ColDef.operator, ColDef.is_io_bound], ascending=True)
    df = df.reset_index(drop=True)

    # return dataframe
    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return df
