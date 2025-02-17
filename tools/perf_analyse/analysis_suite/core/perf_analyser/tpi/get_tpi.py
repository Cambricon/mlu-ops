"""
    present tpi data
"""

__all__ = (
    "get_tpi_data",
)

import os
import sys
import logging
import pandas as pd

from analysis_suite.cfg.config import Config, ColDef, DBConfig
from analysis_suite.database import db_op
from analysis_suite.core.perf_analyser.tpi import tpi_utils
from analysis_suite.core.perf_analyser.tpi.get_tpi_details import tpi_network_summary, tpi_networks, tpi_framework_summary

def read_tables_from_db(df):
    logging.debug("{} start".format(sys._getframe().f_code.co_name))
    # create engine
    engine = db_op.create_engine(DBConfig.DB_Name.training_solution)
    if None == engine:
        logging.info("cannot connect db")

    NETWORK_LIST_QUERY = """
        SELECT
            network_id, case_source, network_name, framework,
            precision_mode, batchsize, network_additional_information,
            project_version, up_to_date
        FROM {};
    """.format(
        "mluops_network_list_test"
    )
    network_list = pd.read_sql_query(NETWORK_LIST_QUERY, con=engine)
    network_list.fillna(0, inplace=True)

    whole_name_columns = [
        ColDef.network_name,
        ColDef.framework,
        ColDef.precision_mode,
        ColDef.batchsize,
        ColDef.network_additional_information,
        ColDef.project_version,
        ColDef.network_id
    ]
    network_list[ColDef.whole_name] = \
        network_list[whole_name_columns].apply(
            lambda x: '_'.join(
                [str(i) for i in x[whole_name_columns] if i != None]),
            axis=1
        )

    # case_in_network is for connecting network_list and case_list
    CASE_IN_NETWORK_QUERY = """
        SELECT {}
        FROM {};
    """.format(
        ", ".join([ColDef.case_id, ColDef.network_id, ColDef.count]),
        "mluops_case_in_network_test"
    )
    case_in_network = pd.read_sql_query(CASE_IN_NETWORK_QUERY, con=engine)
    case_in_network = \
        pd.merge(
            case_in_network,
            network_list[
                [
                    ColDef.network_id,
                    ColDef.whole_name,
                    ColDef.network_name,
                    ColDef.up_to_date,
                    ColDef.case_source,
                    ColDef.framework
                ]
            ],
            on=[ColDef.network_id]
        )

    # get platform
    df[ColDef.mlu_platform] = [i.split('[')[0] for i in df[ColDef.mlu_platform].to_list()]
    platform = df.loc[0, ColDef.mlu_platform]
    if platform not in Config.platform_map:
        raise Exception("generating tpi failed, platform not supported, check the log/xml file")
    platform = Config.platform_map[platform]

    CASE_LIST_QUERY = """
        SELECT case_id, protoName, {}
        FROM {};
    """.format(
        ", ".join(set(Config.platform_map.values())),
        "mluops_case_information_benchmark_test"
    )
    case_list = pd.read_sql_query(CASE_LIST_QUERY, con=engine)
    #case_list = pd.read_sql_table('mluops_case_information_benchmark_test', con=engine)
    case_in_network = pd.merge(case_in_network, case_list, on=[ColDef.case_id])
    del case_list

    # select rows
    case_in_network = case_in_network[(case_in_network[ColDef.up_to_date] == 1) & (case_in_network[platform] == 1)]

    mluops_case_run = \
        pd.merge(
            df,
            case_in_network[
                [
                    ColDef.protoName,
                    ColDef.count,
                    ColDef.whole_name,
                    ColDef.network_name,
                    ColDef.network_id
                ]
            ],
            on=[ColDef.protoName]
        )

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return mluops_case_run, network_list

def get_tpi_data(df):
    logging.info("{} start".format(sys._getframe().f_code.co_name))

    dfs = []
    sheet_names = []

    # connect to database and read tables
    try:
        mluops_case_run, network_list = read_tables_from_db(df)
    except Exception:
        raise

    if mluops_case_run.empty:
        raise Exception("generating tpi failed, check the log/xml file")

    # 将 tpi_columns 中的值变成原来的 count 倍（每行独立操作）, 为后面求mean作准备
    tpi_columns = \
        [
            ColDef.mlu_hardware_time,
            ColDef.mlu_interface_time,
            ColDef.mlu_workspace_size,
            ColDef.mlu_io_efficiency,
            ColDef.mlu_compute_efficiency
        ]
    for column in tpi_columns:
        mluops_case_run[column] = mluops_case_run[column] * mluops_case_run[ColDef.count]

    # 以 whole_name 对 mluops_case_run 进行划分
    group_by_network = mluops_case_run.groupby(ColDef.whole_name)

    # some hard-code (for Tables: summary & networks)
    # TODO: map columns when writing excel
    origin_columns = [
        ColDef.whole_name,
        ColDef.mlu_io_efficiency,
        ColDef.mlu_compute_efficiency,
        ColDef.mlu_workspace_size,
        ColDef.mlu_workspace_size_mean,
        ColDef.mlu_interface_time,
        ColDef.mlu_interface_time_mean,
        ColDef.mlu_hardware_time,
        ColDef.mlu_hardware_time_mean,
        ColDef.counts,
        ColDef.io_bound_percentage,
        ColDef.case_source,
        ColDef.framework
    ]
    columns = [
        ColDef.network_name_zh,
        ColDef.io_efficiency_mean_zh,
        ColDef.compute_efficiency_mean_zh,
        ColDef.workspace_size_sum_zh,
        ColDef.workspace_size_mean_zh,
        ColDef.host_time_sum_zh,
        ColDef.host_time_mean_zh,
        ColDef.total_device_time_zh,
        ColDef.device_time_mean_zh,
        ColDef.counts_zh,
        ColDef.io_bottleneck_ratio_zh,
        ColDef.case_source_zh,
        ColDef.framework_zh
    ]

    # summary 表
    tpi_network = tpi_network_summary.get_network_summary(group_by_network, network_list, origin_columns, columns)
    dfs.append(tpi_network)
    sheet_names.append("summary")

    '''
        生成各 network 的表 start
        input: group_by_network
    '''
    # 更新字段名的映射
    origin_columns[0] = ColDef.operator
    columns[0] = ColDef.operator_zh

    # TODO: dump network infomation to sqlite database rather than .tar file
    tpi_networks.get_networks_and_append(group_by_network, dfs, sheet_names, origin_columns, columns)

    # get total information
    dfs_dic = dict(zip(sheet_names, dfs))

    dfs.insert(0, tpi_framework_summary.get_framework_summary(dfs_dic, network_list, 'pt1.13'))
    sheet_names.insert(0, "pt1.13_operator_summary")

    dfs.insert(1, tpi_framework_summary.get_framework_summary(dfs_dic, network_list, 'mm'))
    sheet_names.insert(1, "mm_operator_summary")

    logging.info("{} end".format(sys._getframe().f_code.co_name))
    return mluops_case_run, dfs, sheet_names
