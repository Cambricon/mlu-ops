"""
    present simple tpi data
"""

__all__ = (
    "dump_to_simple_tpi_network_excel",
)

import sys
import logging
import pandas as pd

from analysis_suite.cfg.config import Config, ColDef
from analysis_suite.utils import excel_helper
from analysis_suite.core.perf_analyser.tpi import tpi_utils

def dump_to_simple_tpi_network_excel(df_dict, xlsx_path, frameworks):
    logging.info("{} start".format(sys._getframe().f_code.co_name))

    # 从 summary 表根据 network_name_zh 获取 important_network
    # summary 表中 network_name_zh 就是 whole_name
    summary = df_dict['summary']
    all_network_rows = summary[ColDef.network_name_zh].to_list()
    important_network = tpi_utils.get_important_network_names(all_network_rows, Config.important_network_keyword, frameworks)
    # 选出所有 network_name_zh 在 important_network 中的行
    summary = summary.loc[summary[ColDef.network_name_zh].isin(important_network)]
    # 选行之后索引会乱，重置索引
    #   drop=True 表示不将旧的index添加为dataframe的列
    summary.reset_index(drop=True, inplace=True)

    # 获取重要网络名 important_network_names
    # whole_name 全小写后就是表名，这里和前面的操作有区别吗？？ 意味不明
    all_network_sheets = df_dict.keys() # df_dict.keys() = tpi_sheet_names
    important_network_names = tpi_utils.get_important_network_names(all_network_sheets, Config.important_network_keyword, frameworks)
    # 根据重要网络名 important_network_names 从 df_dict 中筛选对应表
    tpi_utils.get_important_network_sheet(df_dict, important_network_names)

    dfs = []
    # compute device_time_percentage for every sheet
    for sheet_name in df_dict.keys(): # df_dict.keys() = important_network_names
        # 若没有 device_time_percentage_zh 根据 total_device_time_zh 构造
        if ColDef.device_time_percentage_zh not in df_dict[sheet_name]:
            all_device_time = df[sheet_name][ColDef.total_device_time_zh].sum()
            df_dict[sheet_name][ColDef.device_time_percentage_zh] = \
                df[sheet_name][ColDef.total_device_time_zh] / all_device_time
        # 按 device_time_percentage_zh 降序
        df_dict[sheet_name].sort_values(
            by=[(ColDef.device_time_percentage_zh)],
            ascending=False,
            inplace=True
        )
        # 把 device_time_percentage_zh 移到第2列
        df_dict[sheet_name] = tpi_utils.move_column_location(df_dict[sheet_name], 1, ColDef.device_time_percentage_zh)
        # 重置索引
        df_dict[sheet_name].reset_index(drop=True, inplace=True)
        dfs.append(df_dict[sheet_name])

    if len(dfs) == 0:
        logging.warn("No data with framework {}, so don't generate simple_tpi".format(frameworks))
        return

    # get top20 operators from important network
    # filter metrics:'总device时间占比'
    top20_ops = pd.DataFrame()
    # 连 dfs 的所有表得到总表
    all_ops_sheet = pd.concat(dfs, ignore_index=True, sort=True)
    all_ops_sheet[ColDef.io_bottleneck_ratio_zh] = \
        all_ops_sheet[ColDef.io_bottleneck_ratio_zh].str.strip("%").astype(float) / 100.0
    # 按 operator_zh 分类
    group_by_op = all_ops_sheet.groupby(ColDef.operator_zh)

    data_columns = [
        ColDef.io_efficiency_mean_zh,
        ColDef.compute_efficiency_mean_zh,
        ColDef.workspace_size_mean_zh,
        ColDef.device_time_mean_zh,
        ColDef.counts_zh,
        ColDef.total_device_time_zh,
        ColDef.io_bottleneck_ratio_zh,
        ColDef.workspace_size_sum_zh,
        ColDef.host_time_sum_zh,
        ColDef.host_time_mean_zh
    ]
    for col in data_columns:
        if col in [
            ColDef.counts_zh,
            ColDef.total_device_time_zh,
            ColDef.workspace_size_sum_zh,
            ColDef.host_time_sum_zh
        ]:
            top20_ops[col] = group_by_op.agg({col: ['sum']})[col]['sum']
        else:
            # col in [
            #   io_efficiency_mean_zh, compute_efficiency_mean_zh, workspace_size_mean_zh,
            #   device_time_mean_zh, io_bottleneck_ratio_zh, host_time_mean_zh
            # ]
            top20_ops[col] = group_by_op.agg({col: ['mean']})[col]['mean']

    # 求 device_time_percentage_zh 并根据其排序
    all_ops_time = top20_ops[ColDef.total_device_time_zh].sum()
    top20_ops[ColDef.device_time_percentage_zh] = \
        top20_ops[ColDef.total_device_time_zh] / all_ops_time
    top20_ops = \
        top20_ops.sort_values(
            by=[(ColDef.total_device_time_zh)],
            ascending=False
        )
    # 重置索引，添加列 operator_zh 到表中
    top20_ops = top20_ops.reset_index()
    # 移动 device_time_percentage 到 operator_zh 后面
    top20_ops = tpi_utils.move_column_location(top20_ops, 1, ColDef.device_time_percentage_zh)
    # 返回前 20 行
    top20_ops = top20_ops.head(20)

    # 写出表到excel
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
        excel_helper.to_sheet_helper(
            df=top20_ops,
            writer=writer,
            sheet_name='top20_ops_data',
            index=False,
            float_to_percentage_cols=[
                ColDef.device_time_percentage_zh,
                ColDef.io_bottleneck_ratio_zh
            ]
        )

        excel_helper.to_sheet_helper(
            df=summary,
            writer=writer,
            sheet_name='summary',
            index=False,
            float_to_percentage_cols=[
                ColDef.io_bottleneck_ratio_zh
            ]
        )

        for i in df_dict.keys(): # df_dict.keys() = important_network_names
            # sheet_name 保证不重复
            excel_helper.to_sheet_helper(
                df=df_dict[i],
                writer=writer,
                sheet_name=i[0:10] + i[-5:],
                index=False,
                float_to_percentage_cols=[
                    ColDef.device_time_percentage_zh,
                    ColDef.io_bottleneck_ratio_zh
                ]
            )
    logging.info("the output simple excel is " + xlsx_path)

    logging.info("{} end".format(sys._getframe().f_code.co_name))
