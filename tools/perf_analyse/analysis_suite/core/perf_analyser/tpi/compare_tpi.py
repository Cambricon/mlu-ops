"""
    present result of tpi comparison
"""

__all__ = (
    "compare_tpi",
)

import os
import sys
import logging
import pandas as pd

from analysis_suite.cfg.config import Config, ColDef
from analysis_suite.core.perf_analyser.tpi import tpi_utils
#from analysis_suite.core.perf_analyser.tpi.compare_tpi_details import tpi_framework_summary_compare, tpi_network_summary_compare, tpi_networks_compare

def compare_summary(tpi_dfs, tpi_baseline, version_compare):
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    if ColDef.network_name_zh in tpi_dfs:
        compare = \
            pd.merge(
                tpi_dfs,
                tpi_baseline,
                suffixes=version_compare,
                on=[
                    ColDef.network_name_zh,
                    ColDef.network_id
                ]
            )
    elif ColDef.operator_zh in tpi_dfs:
        compare = \
            pd.merge(
                tpi_dfs,
                tpi_baseline,
                suffixes=version_compare,
                on=[ColDef.operator_zh]
            )
    else:
        return pd.DataFrame()

    # 计算提升值 & 提升比例
    compare[ColDef.device_time_promotion_zh] = \
        compare[ColDef.total_device_time_zh + version_compare[1]] - \
        compare[ColDef.total_device_time_zh + version_compare[0]]
    compare[ColDef.device_time_promotion_ratio_zh] = \
        compare[ColDef.device_time_promotion_zh] / \
        compare[ColDef.total_device_time_zh + version_compare[1]]
    # 按提升比例降序
    compare = \
        compare.sort_values(
            by=[(ColDef.device_time_promotion_ratio_zh)],
            ascending=False
        )
    compare[ColDef.host_time_promotion_zh] = \
        compare[ColDef.host_time_sum_zh + version_compare[1]] - \
        compare[ColDef.host_time_sum_zh + version_compare[0]]
    compare[ColDef.host_time_promotion_ratio_zh] = \
        compare[ColDef.host_time_promotion_zh] / \
        compare[ColDef.host_time_sum_zh + version_compare[1]]

    compare[ColDef.workspace_size_promotion_zh] = \
        compare[ColDef.workspace_size_sum_zh + version_compare[1]] - \
        compare[ColDef.workspace_size_sum_zh + version_compare[0]]
    compare[ColDef.workspace_size_promotion_ratio_zh] = \
        compare[ColDef.workspace_size_promotion_zh] / \
        compare[ColDef.workspace_size_sum_zh + version_compare[1]]

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return compare

def compare_tpi(case_run, case_run_bl, tpi_dfs, tpi_baseline,
                sheet_names, tpi_compare_path, version_compare,
                tar_compare_name):
    logging.info("{} start".format(sys._getframe().f_code.co_name))
    dfs = []

    #'''
    framework_columns = [
        ColDef.operator_zh,
        ColDef.device_time_promotion_zh,
        ColDef.device_time_promotion_ratio_zh,
        ColDef.host_time_promotion_zh,
        ColDef.host_time_promotion_ratio_zh,
        ColDef.workspace_size_promotion_zh,
        ColDef.workspace_size_promotion_ratio_zh
    ]
    for i in [
        ColDef.operator_devices_time_sum_ratio_in_all_network_zh,
        ColDef.operator_counts_ratio_in_all_networks_zh,
        ColDef.counts_zh,
        ColDef.io_bottleneck_ratio_zh,
        ColDef.io_efficiency_mean_zh,
        ColDef.compute_efficiency_mean_zh,
        ColDef.workspace_size_mean_zh,
        ColDef.device_time_mean_zh,
        ColDef.total_device_time_zh,
        ColDef.workspace_size_sum_zh,
        ColDef.host_time_sum_zh,
        ColDef.host_time_mean_zh
    ]:
        for j in version_compare: # version_compare 的值在 Processor 类的初始化函数中
            framework_columns.append(i + j)
    pt_compare = compare_summary(tpi_dfs[0], tpi_baseline[0], version_compare)
    mm_compare = compare_summary(tpi_dfs[1], tpi_baseline[1], version_compare)
    # pt_compare 和 mm_compare 的结果中只要 framework_columns 中的列
    # if empty, append empty dataframe for station location
    if not pt_compare.empty:
        dfs.append(pt_compare[framework_columns])
    else:
        dfs.append(pt_compare)
    if not mm_compare.empty:
        dfs.append(mm_compare[framework_columns])
    else:
        dfs.append(mm_compare)
    #'''
    '''
    # pt
    dfs.append(
        tpi_framework_summary_compare.get_framework_summary_compare(
            tpi_dfs[0], tpi_baseline[0]
        )
    )
    # mm
    dfs.append(
        tpi_framework_summary_compare.get_framework_summary_compare(
            tpi_dfs[1], tpi_baseline[1]
        )
    )
    '''

    #'''
    summary_columns = [
        ColDef.network_name_zh,
        ColDef.device_time_promotion_zh,
        ColDef.device_time_promotion_ratio_zh,
        ColDef.host_time_promotion_zh,
        ColDef.host_time_promotion_ratio_zh,
        ColDef.workspace_size_promotion_zh,
        ColDef.workspace_size_promotion_ratio_zh,
        ColDef.network_id
    ]
    for i in [
        ColDef.total_device_time_zh,
        ColDef.host_time_sum_zh,
        ColDef.workspace_size_sum_zh,
        ColDef.io_efficiency_mean_zh,
        ColDef.compute_efficiency_mean_zh,
        ColDef.counts_zh,
        ColDef.device_time_mean_zh,
        ColDef.host_time_mean_zh,
        ColDef.workspace_size_mean_zh,
        ColDef.io_bottleneck_ratio_zh,
        ColDef.case_source_zh,
        ColDef.framework_zh
    ]:
        for j in version_compare:
            summary_columns.append(i + j)
    summary_compare = compare_summary(tpi_dfs[2], tpi_baseline[2], version_compare)
    # summary_compare 的结果中只要 summary_columns 中的列
    dfs.append(summary_compare[summary_columns])
    #'''
    #dfs.append(tpi_network_summary_compare.get_network_summary_compare(tpi_dfs[2], tpi_baseline[2]))

    #'''
    # arange excel columns
    tpi_compare_columns = [
        ColDef.operator,
        ColDef.mlu_hardware_time_promotion,
        ColDef.mlu_hardware_time_promotion_ratio,
        ColDef.mlu_interface_time_promotion,
        ColDef.mlu_interface_time_promotion_ratio,
        ColDef.mlu_workspace_size_promotion,
        ColDef.mlu_workspace_size_promotion_ratio,
        ColDef.count_new,
        ColDef.count_baseline,
        ColDef.mlu_hardware_time_new,
        ColDef.mlu_hardware_time_baseline,
        ColDef.mlu_interface_time_new,
        ColDef.mlu_interface_time_baseline,
        ColDef.mlu_workspace_size_new,
        ColDef.mlu_workspace_size_baseline,
        ColDef.is_io_bound_new,
        ColDef.is_io_bound_baseline,
        ColDef.file_path
    ]

    group_by_network = case_run.groupby(ColDef.whole_name)
    group_by_network_bl = case_run_bl.groupby(ColDef.whole_name)

    # dict -> set
    network_set = set(group_by_network.groups)
    network_bl_set = set(group_by_network_bl.groups)
    # 求交集 只取相同的网络
    network_intersection = network_set & network_bl_set
    for network in network_intersection:
        # 用索引来查找行
        one_network = case_run.loc[group_by_network.groups[network]]
        one_network_bl = case_run_bl.loc[group_by_network_bl.groups[network]]

        tpi_compare = pd.merge(
            one_network,
            one_network_bl,
            suffixes=Config.suffix, # suffix = ["_new", "_baseline"]
            on=[
                ColDef.protoName,
                ColDef.operator,
                ColDef.file_path
            ]
        )

        # mlu_hardware_time_promotion
        tpi_compare[ColDef.mlu_hardware_time_promotion] = \
            tpi_compare[ColDef.mlu_hardware_time + Config.suffix[1]] - \
            tpi_compare[ColDef.mlu_hardware_time + Config.suffix[0]]
        # mlu_hardware_time_promotion_ratio
        tpi_compare[ColDef.mlu_hardware_time_promotion_ratio] = \
            tpi_compare[ColDef.mlu_hardware_time_promotion] / \
            tpi_compare[ColDef.mlu_hardware_time + Config.suffix[1]]
        tpi_compare[ColDef.mlu_hardware_time_promotion_ratio] = \
            tpi_compare[ColDef.mlu_hardware_time_promotion_ratio].apply("{:.2%}".format)

        # mlu_interface_time_promotion
        tpi_compare[ColDef.mlu_interface_time_promotion] = \
            tpi_compare[ColDef.mlu_interface_time + Config.suffix[1]] - \
            tpi_compare[ColDef.mlu_interface_time + Config.suffix[0]]
        # mlu_interface_time_promotion_ratio
        tpi_compare[ColDef.mlu_interface_time_promotion_ratio] = \
            tpi_compare[ColDef.mlu_interface_time_promotion] / \
            tpi_compare[ColDef.mlu_interface_time + Config.suffix[1]]
        tpi_compare[ColDef.mlu_interface_time_promotion_ratio] = \
            tpi_compare[ColDef.mlu_interface_time_promotion_ratio].apply("{:.2%}".format)

        # mlu_workspace_size_promotion
        tpi_compare[ColDef.mlu_workspace_size_promotion] = \
            tpi_compare[ColDef.mlu_workspace_size + Config.suffix[1]] - \
            tpi_compare[ColDef.mlu_workspace_size + Config.suffix[0]]
        # mlu_workspace_size_promotion_ratio
        tpi_compare[ColDef.mlu_workspace_size_promotion_ratio] = \
            tpi_compare[ColDef.mlu_workspace_size_promotion] / \
            tpi_compare[ColDef.mlu_workspace_size + Config.suffix[1]]
        tpi_compare[ColDef.mlu_workspace_size_promotion_ratio] = \
            tpi_compare[ColDef.mlu_workspace_size_promotion_ratio].apply("{:.2%}".format)

        dfs.append(tpi_compare[tpi_compare_columns])
    #'''
    #tpi_networks_compare.get_networks_compare_and_append(dfs, case_run, case_run_bl)

    # 同 get_tpi_data 后的结果，前3张表导出为excel，后面的表压到tar中
    tpi_utils.dump_tpi_excel(
        dfs[0:3],
        sheet_names[0:3],
        tpi_compare_path,
        [
            [
                ColDef.device_time_promotion_ratio_zh,
                ColDef.host_time_promotion_ratio_zh, ColDef.workspace_size_promotion_ratio_zh,
                ColDef.operator_devices_time_sum_ratio_in_all_network_zh + version_compare[0],
                ColDef.operator_devices_time_sum_ratio_in_all_network_zh + version_compare[1],
                ColDef.operator_counts_ratio_in_all_networks_zh + version_compare[0],
                ColDef.operator_counts_ratio_in_all_networks_zh + version_compare[1],
                ColDef.io_bottleneck_ratio_zh + version_compare[0],
                ColDef.io_bottleneck_ratio_zh + version_compare[1]
            ],
            [
                ColDef.device_time_promotion_ratio_zh,
                ColDef.host_time_promotion_ratio_zh, ColDef.workspace_size_promotion_ratio_zh,
                ColDef.operator_devices_time_sum_ratio_in_all_network_zh + version_compare[0],
                ColDef.operator_devices_time_sum_ratio_in_all_network_zh + version_compare[1],
                ColDef.operator_counts_ratio_in_all_networks_zh + version_compare[0],
                ColDef.operator_counts_ratio_in_all_networks_zh + version_compare[1],
                ColDef.io_bottleneck_ratio_zh + version_compare[0],
                ColDef.io_bottleneck_ratio_zh + version_compare[1]
            ],
            [
                ColDef.device_time_promotion_ratio_zh,
                ColDef.host_time_promotion_ratio_zh,
                ColDef.workspace_size_promotion_ratio_zh,
                ColDef.io_bottleneck_ratio_zh + version_compare[0],
                ColDef.io_bottleneck_ratio_zh + version_compare[1],
            ]
        ]
    )

    dic_to_txt = dict(zip(sheet_names[3:], dfs[3:]))
    tpi_utils.get_txt_excel_to_tar(dic_to_txt, tar_compare_name)

    logging.info("{} end".format(sys._getframe().f_code.co_name))
    return dfs, sheet_names
