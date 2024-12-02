"""
    present result of simple tpi comparison
"""

__all__ = (
    "dump_to_simple_comparision_tpi_excel",
)

import sys
import logging
import pandas as pd

from analysis_suite.cfg.config import Config, ColDef
from analysis_suite.utils import excel_helper
from analysis_suite.core.perf_analyser.tpi import tpi_utils

def dump_to_simple_comparision_tpi_excel(df_dict, xlsx_path, frameworks,
                                            version_compare):
    logging.info("{} start".format(sys._getframe().f_code.co_name))

    '''
        生成 summary 表，更新 df_dict start
            input: df_dict, frameworks
            output: df_dict, summary
    '''
    important_network_keys = Config.important_network_keyword

    summary = df_dict['summary']
    all_network_rows = summary[ColDef.network_name_zh].to_list()
    important_network = tpi_utils.get_important_network_names(all_network_rows, important_network_keys, frameworks)
    summary = summary.loc[summary[ColDef.network_name_zh].isin(important_network)]
    summary.reset_index(drop=True, inplace=True)

    all_network_sheets = df_dict.keys() # df_dict.keys() = tpi_comp_sheet_names
    important_network_names = tpi_utils.get_important_network_names(all_network_sheets, important_network_keys, frameworks)
    tpi_utils.get_important_network_sheet(df_dict, important_network_names)
    '''
        end
    '''

    columns = [
        ColDef.mlu_hardware_time_new,
        ColDef.mlu_hardware_time_baseline,
        ColDef.mlu_interface_time_new,
        ColDef.mlu_interface_time_baseline,
        ColDef.mlu_workspace_size_new,
        ColDef.mlu_workspace_size_baseline,
        ColDef.count_new,
        ColDef.count_baseline
    ]
    '''
        生成字段名 start
            input: columns
            output: update_columns, update_columns_ch, columns_promotion_ratio
    '''
    columns_bs = [
        ColDef.mlu_hardware_time,
        ColDef.mlu_interface_time,
        ColDef.mlu_workspace_size,
        ColDef.count
    ]
    columns_bs_zh = [
        ColDef.mlu_hardware_time_zh,
        ColDef.mlu_interface_time_zh,
        ColDef.mlu_workspace_size_zh,
        ColDef.count_zh
    ]
    update_columns = [
        ColDef.operator,
        ColDef.device_time_per_new,
        ColDef.device_time_per_baseline
    ]
    update_columns_ch = [
        ColDef.operator_zh,
        ColDef.device_time_percentage_zh + version_compare[0],
        ColDef.device_time_percentage_zh + version_compare[1]
    ]
    static_suffix = ['_new', '_baseline', '_promotion_ratio']
    ver_suffix = version_compare
    promotion = '_提升比例'
    columns_promotion_ratio = []
    for i in range(0, len(columns_bs)):
        # 对 count 添加字段名xxx_sum
        # 对 [mlu_hardware_time, mlu_interface_time, mlu_workspace_size] 添加字段名xxx_sum和xxx_mean
        update_columns.append(columns_bs[i] + static_suffix[0] + '_sum')
        update_columns.append(columns_bs[i] + static_suffix[1] + '_sum')
        update_columns.append(columns_bs[i] + '_sum' + static_suffix[2])
        update_columns_ch.append('总' + columns_bs_zh[i] + ver_suffix[0])
        update_columns_ch.append('总' + columns_bs_zh[i] + ver_suffix[1])
        update_columns_ch.append('总' + columns_bs_zh[i] + promotion)
        columns_promotion_ratio.append(columns_bs[i] + '_sum' + static_suffix[2])
        if columns_bs[i] not in [ColDef.count]:
            update_columns.append(columns_bs[i] + static_suffix[0] + '_mean')
            update_columns.append(columns_bs[i] + static_suffix[1] + '_mean')
            update_columns.append(columns_bs[i] + '_mean' + static_suffix[2])
            update_columns_ch.append('平均' + columns_bs_zh[i] + ver_suffix[0])
            update_columns_ch.append('平均' + columns_bs_zh[i] + ver_suffix[1])
            update_columns_ch.append('平均' + columns_bs_zh[i] + promotion)
            columns_promotion_ratio.append(columns_bs[i] + '_mean' + static_suffix[2])
    for i in [ColDef.io_bound_percentage_new, ColDef.io_bound_percentage_baseline]:
        update_columns.append(i)
    for i in [ColDef.io_bottleneck_ratio_zh + ver_suffix[0], ColDef.io_bottleneck_ratio_zh + ver_suffix[1]]:
        update_columns_ch.append(i)
    '''
        生成字段名 end
    '''

    # 为对 operator 进行groupby后的结果生成字段名
    data_col_sum = []
    data_columns_all = []
    for i in columns:
        data_col_sum.append(i + '_sum')
        data_columns_all.append(i + '_sum')
        if i not in [ColDef.count_new, ColDef.count_baseline]:
            data_columns_all.append(i + '_mean')

    # columns_promotion_ratio = [
    #   mlu_hardware_time_sum_promotion_ratio, mlu_hardware_time_mean_promotion_ratio,
    #   mlu_interface_time_sum_promotion_ration, mlu_interface_time_mean_promotion_ratio,
    #   mlu_workspace_size_sum_promotion_ration, mlu_workspace_size_mean_promotion_ratio,
    #   count_sum_promotion_ration
    # ]
    # data_columns_all = [
    #   mlu_hardware_time_new_sum, mlu_hardware_time_new_mean,
    #   mlu_hardware_time_baseline_sum, mlu_hardware_time_baseline_mean,
    #   mlu_interface_time_new_sum, mlu_interface_time_new_mean,
    #   mlu_interface_time_baseline_sum, mlu_interface_time_baseline_mean,
    #   mlu_workspace_size_new_sum, mlu_workspace_size_new_mean,
    #   mlu_workspace_size_baseline_sum, mlu_workspace_size_baseline_mean,
    #   count_new,
    #   count_baseline
    # ]
    top20_ops = pd.DataFrame()
    all_sheets = []
    # 每张 important_network_names 中的表
    for i in important_network_names:
        dfs = pd.DataFrame()
        # 都按照 operator 来groupby一次
        group_by_op = df_dict[i].groupby(ColDef.operator)
        # 添加字段xxx_sum和xxx_mean
        for col in columns:
            dfs[col + '_sum'] = group_by_op.agg({col: ['sum']})[col]['sum']
            if col not in [ColDef.count_new, ColDef.count_baseline]:
                dfs[col + '_mean'] = group_by_op.agg({col: ['mean']})[col]['mean']

        # 直接用索引获取list中的字段名，对表中的列进行操作

        for j in range(0, 3):
            dfs[columns_promotion_ratio[2 * j]] = \
                (dfs[data_columns_all[4 * j + 2]] - dfs[data_columns_all[(4 * j)]]) / \
                dfs[data_columns_all[4 * j + 2]]
            dfs[columns_promotion_ratio[2 * j + 1]] = \
                (dfs[data_columns_all[4 * j + 3]] - dfs[data_columns_all[(4 * j + 1)]]) / \
                dfs[data_columns_all[4 * j + 3]]

        # 计算 io_bound_percentage 相关
        # io_bound_percentage_new
        dfs[ColDef.counts_new] = \
            group_by_op.agg({ColDef.count_new: ['sum']})[ColDef.count_new]['sum']
        dfs[ColDef.io_bound_percentage_new] = \
            group_by_op.apply(
                lambda x: x[x[ColDef.is_io_bound_new]][ColDef.count_new].sum()
            ) / dfs[ColDef.counts_new]
        # io_bound_percentage_baseline
        dfs[ColDef.counts_baseline] = \
            group_by_op.agg({ColDef.count_baseline: ['sum']})[ColDef.count_baseline]['sum']
        dfs[ColDef.io_bound_percentage_baseline] = \
            group_by_op.apply(
                lambda x: x[x[ColDef.is_io_bound_baseline]][ColDef.count_baseline].sum()
            ) / dfs[ColDef.counts_baseline]
        # count_sum_promotion_ratio
        dfs[ColDef.count_sum_promotion_ratio] = \
            (dfs[ColDef.count_baseline_sum] - dfs[ColDef.count_new_sum]) / \
            dfs[ColDef.count_baseline_sum]

        # 根据 mlu_hardware_time_new_sum 降序排序
        dfs = dfs.sort_values(by=[(ColDef.mlu_hardware_time_new_sum)], ascending=False)

        # device_time_per_new
        dfs_ops_time_new = dfs[ColDef.mlu_hardware_time_new_sum].sum()
        dfs[ColDef.device_time_per_new] = \
            (dfs[ColDef.mlu_hardware_time_new_sum]) / dfs_ops_time_new
        dfs[ColDef.device_time_per_new] = (dfs[ColDef.device_time_per_new]).apply("{:.2%}".format)
        # device_time_per_baseline
        dfs_ops_time_baseline = dfs[ColDef.mlu_hardware_time_baseline_sum].sum()
        dfs[ColDef.device_time_per_baseline] = \
            (dfs[ColDef.mlu_hardware_time_baseline_sum]) / dfs_ops_time_baseline
        dfs[ColDef.device_time_per_baseline] = (dfs[ColDef.device_time_per_baseline]).apply("{:.2%}".format)

        dfs = dfs.reset_index()
        dfs = dfs[update_columns]
        all_sheets.append(dfs)

    if len(all_sheets) == 0:
        logging.warn(f"No data with framework {frameworks}, so don't generate comparison_simple_tpi")
        return

    '''
        生成 top20_ops 表 start
    '''
    top20_ops = pd.DataFrame()
    # 上面的表全拼接起来
    all_ops_sheet = pd.concat(all_sheets, ignore_index=True, sort=True)
    # gen top_20_ops sheet
    for i in columns_promotion_ratio:
        data_columns_all.append(i)

    group_by_op = all_ops_sheet.groupby(ColDef.operator)
    for column in data_columns_all:
        if column in data_col_sum:
            top20_ops[column] = group_by_op.agg({column: ['sum']})[column]['sum']
        else:
            top20_ops[column] = group_by_op.agg({column: ['mean']})[column]['mean']

    # 根据 mlu_hardware_time_new_sum 降序排序
    top20_ops = top20_ops.sort_values(by=[(ColDef.mlu_hardware_time_new_sum)], ascending=False)

    all_ops_time_new = top20_ops[ColDef.mlu_hardware_time_new_sum].sum()
    top20_ops[ColDef.device_time_per_new] = \
        (top20_ops[ColDef.mlu_hardware_time_new_sum]) / all_ops_time_new
    all_ops_time_baseline = top20_ops[ColDef.mlu_hardware_time_baseline_sum].sum()
    top20_ops[ColDef.device_time_per_baseline] = \
        (top20_ops[ColDef.mlu_hardware_time_baseline_sum]) / all_ops_time_baseline
    top20_ops[ColDef.io_bound_percentage_new] = \
        group_by_op.agg({ColDef.io_bound_percentage_new: ['mean']})[ColDef.io_bound_percentage_new]['mean']
    top20_ops[ColDef.io_bound_percentage_baseline] = \
        group_by_op.agg({ColDef.io_bound_percentage_baseline: ['mean']})[ColDef.io_bound_percentage_baseline]['mean']
    for i in (
        columns_promotion_ratio + \
        [
            ColDef.io_bound_percentage_new, ColDef.io_bound_percentage_baseline,
            ColDef.device_time_per_new, ColDef.device_time_per_baseline
        ]
    ):
        top20_ops[i] = top20_ops[i].apply("{:.2%}".format)

    top20_ops = top20_ops.reset_index()
    top20_ops = top20_ops[update_columns]
    top20_ops.rename(columns=dict(zip(update_columns, update_columns_ch)), inplace=True)
    top20_ops = top20_ops.head(20)
    '''
        生成 top20_ops 表 end
    '''

    # 更新 all_sheets 中各表的字段名
    for sheet_idx in range(0, len(all_sheets)):
        for col in (
            columns_promotion_ratio + \
            [
                ColDef.io_bound_percentage_new, ColDef.io_bound_percentage_baseline
            ]
        ):
            all_sheets[sheet_idx][col] = all_sheets[sheet_idx][col].apply("{:.2%}".format)

        all_sheets[sheet_idx].rename(
            columns=dict(zip(all_sheets[sheet_idx].keys(), update_columns_ch)),
            inplace=True
        )

    # 写出表到excel
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
        excel_helper.to_sheet_helper(
            df=top20_ops,
            writer=writer,
            sheet_name='top20_ops_data',
            index=False
        )

        excel_helper.to_sheet_helper(
            df=summary,
            writer=writer,
            sheet_name='summary',
            index=False,
            float_to_percentage_cols=[
                ColDef.device_time_promotion_ratio_zh,
                ColDef.host_time_promotion_ratio_zh,
                ColDef.workspace_size_promotion_ratio_zh,
                ColDef.io_bottleneck_ratio_zh + version_compare[0],
                ColDef.io_bottleneck_ratio_zh + version_compare[1]
            ]
        )

        for i in range(0, len(all_sheets)):
            excel_helper.to_sheet_helper(
                df=all_sheets[i],
                writer=writer,
                sheet_name=important_network_names[i][0:10] + important_network_names[i][-5:],
                index=False
            )

        logging.info("the simple tpi comparision excel is " + xlsx_path)

    logging.info("{} end".format(sys._getframe().f_code.co_name))
