"""
    Table: network_summary & case_in_network
"""

__all__ = (
    "get_network_summary"
)

import os
import sys
import logging
import pandas as pd

from analysis_suite.cfg.config import Config, ColDef, DBConfig
from analysis_suite.core.perf_analyser.perf import perf_utils
from analysis_suite.core.perf_analyser.perf.get_data_details import perf_get_data_utils

from analysis_suite.database import db_op

def append_network_id(df: pd.DataFrame, engine):
    # read case_list in database
    CASE_LIST_QUERY = """
        SELECT {}
        FROM {};
    """.format(
        ", ".join([ColDef.case_id, ColDef.protoName]),
        DBConfig.Table_Name_mp[DBConfig.Table_Name.case_list]
    )
    case_list = pd.read_sql_query(CASE_LIST_QUERY, con=engine)
    # append case_id
    mluops_case_run = \
        pd.merge(
            df,
            case_list,
            on=[ColDef.protoName]
        )

    # read case_in_network in database
    CASE_IN_NETWORK_QUERY = """
        SELECT {}
        FROM {};
    """.format(
        ", ".join([ColDef.case_id, ColDef.network_id, ColDef.count]),
        DBConfig.Table_Name_mp[DBConfig.Table_Name.case_in_network]
    )
    # append network_id
    case_in_network = pd.read_sql_query(CASE_IN_NETWORK_QUERY, con=engine)
    mluops_case_run = \
        pd.merge(
            mluops_case_run,
            case_in_network,
            on=[ColDef.case_id]
        )

    return mluops_case_run

def append_network_info(df: pd.DataFrame, engine):
    # read network_list in database
    NETWORK_LIST_QUERY = """
        SELECT {}
        FROM {};
    """.format(
        ", ".join(Config.network_info_keys),
        DBConfig.Table_Name_mp[DBConfig.Table_Name.network_list]
    )
    network_list = pd.read_sql_query(NETWORK_LIST_QUERY, con=engine)
    # append network information
    df = \
        pd.merge(
            df,
            network_list,
            on=[ColDef.network_id]
        )

    return df

def append_history_network_summary(df: pd.DataFrame, platform: str, engine):
    # read network_summary in database
    NETWORK_SUMMARY_QUERY = """
        SELECT {}
        FROM {};
    """.format(
        ", ".join([ColDef.network_id, ColDef.mlu_platform, ColDef.mlu_hardware_time_sum, ColDef.date]),
        DBConfig.Table_Name_mp[DBConfig.Table_Name.network_summary]
    )
    network_summary = pd.read_sql_query(NETWORK_SUMMARY_QUERY, con=engine)

    # preprocess data in network_summary
    assert not network_summary.empty, "network_summary is empty, please check database: {}".format("mluops_network_summary_test")
    network_summary_max_date = network_summary.groupby([ColDef.network_id,ColDef.mlu_platform]).agg({ColDef.date: 'max'})
    network_summary = pd.merge(network_summary_max_date, network_summary, on=[ColDef.network_id, ColDef.mlu_platform, ColDef.date])
    network_summary.drop_duplicates(subset=[ColDef.network_id, ColDef.mlu_platform], keep='last', inplace=True)
    network_summary.drop(columns=[ColDef.date], inplace=True)

    # add network_summary in database
    # TODO: what about multi version?
    if platform == 'MLU590-M9':
        df = \
            pd.merge(
                df,
                network_summary[network_summary[ColDef.mlu_platform]=='MLU590-M9U'].drop(columns=[ColDef.mlu_platform]),
                how='left',
                on=[ColDef.network_id],
                suffixes=["", "_database"]
            )
    else:
        df = \
            pd.merge(
                df,
                network_summary,
                how='left',
                on=[ColDef.network_id, ColDef.mlu_platform],
                suffixes=["", "_database"]
            )
    return df

def get_network_summary_impl(df, perf_config, is_release):
    # do not handle cases not in benchmark
    if not df.loc[0, ColDef.file_path].startswith('/MLU_OPS/SOFT_TRAIN/benchmark'):
        logging.info("case {} not in /MLU_OPS/SOFT_TRAIN/benchmark".format(df.loc[0, ColDef.file_path]))
        return pd.DataFrame(), pd.DataFrame()

    # connect database: training_solution
    engine = db_op.create_engine(DBConfig.DB_Name.training_solution)
    # append network_id
    df = append_network_id(df, engine)

    reserved_column_values = perf_get_data_utils.reserve_column_values(df, is_release)

    # 准备后续求 sum
    df[ColDef.mlu_hardware_time] = df[ColDef.mlu_hardware_time] * df[ColDef.count]

    # get summary of network
    summary = perf_get_data_utils.get_summary(df, ColDef.network_id, perf_config)
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    # append reserved columns in df
    for k, v in reserved_column_values.items():
        summary[k] = v

    # append network_info, which is in Config.network_info_keys
    summary = append_network_info(summary, engine)

    # connect database: rainbow
    engine = db_op.create_engine(DBConfig.DB_Name.rainbow)
    # read network_summary data in database and append
    summary = append_history_network_summary(summary, df.loc[0, ColDef.mlu_platform], engine)

    return summary, df[[ColDef.protoName, ColDef.network_id, ColDef.count]]

def get_network_summary(df, perf_config, is_release):
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    network_summary, case_in_network = get_network_summary_impl(df, perf_config, is_release)

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return network_summary, case_in_network

