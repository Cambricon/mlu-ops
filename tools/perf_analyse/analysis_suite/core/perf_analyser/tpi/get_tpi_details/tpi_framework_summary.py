"""
    Table: framework
"""

__all__ = (
    "get_framework_summary",
)

import sys
import logging
import typing
import pandas as pd

from analysis_suite.cfg.config import ColDef
from analysis_suite.core.perf_analyser.tpi import tpi_utils

# for each framework, get summary by operator
def get_framework_tpi_data(df, network_need):
    logging.debug("{} start".format(sys._getframe().f_code.co_name))
    # filter out from df
    dfs = []
    # 根据 network_name_zh 列来删除重复项
    summary = df['summary'].drop_duplicates(subset=ColDef.network_name_zh)

    '''
        filtered by network_need
    '''
    # 按 network 遍历表
    index = 1
    networks = list(df.values())
    for network_id in summary[ColDef.network_id]:
        if network_id in network_need[ColDef.network_id].to_list():
            tpi_net = networks[index]

            # 计算各 operator 的 counts_zh(counts) 的百分比
            counts = tpi_net[ColDef.counts_zh].sum()
            tpi_net[ColDef.count_percentage_zh] = tpi_net[ColDef.counts_zh] / counts
            tpi_net[ColDef.count_percentage_zh] = \
                tpi_net[ColDef.count_percentage_zh].apply(lambda x: format(x * 100, ".6f") + '%')

            # 计算各 operator 的 total_device_time_zh(mlu_hardware_time) 的百分比
            devices = tpi_net[ColDef.total_device_time_zh].sum()
            tpi_net[ColDef.device_time_percentage_zh] = tpi_net[ColDef.total_device_time_zh] / devices
            tpi_net[ColDef.device_time_percentage_zh] = \
                tpi_net[ColDef.device_time_percentage_zh].apply(lambda x: format(x * 100, ".6f") + '%')

            # 将 tpi_net 按 total_device_time_zh(mlu_hardware_time) 降序排列（意味不明，感觉可以删掉）
            tpi_net = tpi_net.sort_values(by=[(ColDef.total_device_time_zh)], ascending=False)

            dfs.append(tpi_net)
        index += 1

    # if len(dfs) == 0 return an empty dataframe
    if len(dfs) == 0:
        return pd.DataFrame()

    '''
        计算最后需要的表
        input: dfs
    '''
    # get ops information in network
    total_ops = pd.DataFrame()
    # 把上面的表全部拼接起来
    all_ops_sheet = pd.concat(dfs, ignore_index=True)
    # 对 io_bottleneck_ratio_zh(io_bound_percentage) 列操作
    # 去除字符串末尾的'%'后转为浮点数，并除以100，得到对应的小数形式
    all_ops_sheet[ColDef.io_bottleneck_ratio_zh] = all_ops_sheet[ColDef.io_bottleneck_ratio_zh].str.rstrip('%').astype('float') / 100.0
    all_ops_sheet[ColDef.io_bottleneck_sum_zh] = all_ops_sheet[ColDef.io_bottleneck_ratio_zh] * all_ops_sheet[ColDef.counts_zh]
    
    # 按 operator_zh(operator) 分类
    group_by_op = all_ops_sheet.groupby(ColDef.operator_zh)

    # 总个数(counts_zh)在求IO瓶颈数量时求得
    data_columns = [
        ColDef.io_bottleneck_ratio_zh,
        ColDef.io_efficiency_mean_zh,
        ColDef.compute_efficiency_mean_zh,
        ColDef.workspace_size_mean_zh,
        ColDef.device_time_mean_zh,
        ColDef.total_device_time_zh,
        ColDef.workspace_size_sum_zh,
        ColDef.host_time_sum_zh,
        ColDef.host_time_mean_zh
    ]

    for col in data_columns:
        if col in [
            ColDef.total_device_time_zh,
            ColDef.workspace_size_sum_zh,
            ColDef.host_time_sum_zh
        ]: # 求和
            total_ops[col] = group_by_op.agg({col: ['sum']})[col]['sum']
        elif col in [ColDef.io_bottleneck_ratio_zh]: # 添加了列: [counts_zh, io_bottleneck_ratio_zh]
            # 将列 io_bottleneck_sum_zh 设置为：按 operator_zh(operator) 对 io_bottleneck_sum_zh 求和后的值
            total_ops[ColDef.io_bottleneck_sum_zh] = group_by_op.agg({ColDef.io_bottleneck_sum_zh: ['sum']})
            # 将列 counts_zh 设置为：按 operator_zh(operator) 对 counts_zh(counts) 求和后的值
            total_ops[ColDef.counts_zh]  = group_by_op.agg({ColDef.counts_zh: ['sum']})
            # 将列 io_bottleneck_ratio_zh 设置为：按 operator_zh(operator) 对 io_bottleneck_sum_zh 求和后取平均
            total_ops[col] = total_ops[ColDef.io_bottleneck_sum_zh] / total_ops[ColDef.counts_zh]
            # 去掉列 io_bottleneck_sum_zh
            total_ops.drop(columns=[ColDef.io_bottleneck_sum_zh], inplace=True)
        else: # 取平均，相关列为: [io_efficiency_mean_zh, compute_efficiency_mean_zh, workspace_size_mean_zh, device_time_mean_zh, host_time_mean_zh]
            total_ops[col] = group_by_op.agg({col: ['mean']})[col]['mean']

    # 求各 operator 的 counts_zh 占总数量的比例
    all_op_counts = total_ops[ColDef.counts_zh].sum()
    total_ops[ColDef.operator_counts_ratio_in_all_networks_zh] = total_ops[ColDef.counts_zh] / all_op_counts
    # 求各 operator 的 total_device_time_zh 占总设备时间的比例
    all_op_devices = total_ops[ColDef.total_device_time_zh].sum()
    total_ops[ColDef.operator_devices_time_sum_ratio_in_all_network_zh] = total_ops[ColDef.total_device_time_zh] / all_op_devices
    # 按 total_device_time_zh 降序排序
    total_ops = total_ops.sort_values(by=[(ColDef.total_device_time_zh)], ascending=False)

    # 重置索引
    total_ops = total_ops.reset_index()
    # 把 operator_counts_ratio_in_all_networks_zh 和 operator_devices_time_sum_ratio_in_all_network_zh 这2列移到前面
    total_ops = tpi_utils.move_column_location(total_ops, 1, ColDef.operator_counts_ratio_in_all_networks_zh)
    total_ops = tpi_utils.move_column_location(total_ops, 1, ColDef.operator_devices_time_sum_ratio_in_all_network_zh)

    # 添加 operator_devices_time_ratio_in_network_sum_zh 和 operator_counts_ratio_in_network_sum_zh 列
    total_ops[ColDef.operator_devices_time_ratio_in_network_sum_zh] = 0
    total_ops[ColDef.operator_counts_ratio_in_network_sum_zh] = 0
    for network in dfs:
        network[ColDef.count_percentage_zh] = network[ColDef.count_percentage_zh].apply(lambda x: float(x.strip('%')) / 100)
        network[ColDef.device_time_percentage_zh] = network[ColDef.device_time_percentage_zh].apply(lambda x: float(x.strip('%')) / 100)
        # 对dfs里的每张表都left join
        total_ops = \
            pd.merge(
                total_ops,
                network[
                    [
                        ColDef.operator_zh,
                        ColDef.count_percentage_zh,
                        ColDef.device_time_percentage_zh
                    ]
                ],
                on=ColDef.operator_zh,
                how='left'
            )

        total_ops[ColDef.operator_devices_time_ratio_in_network_sum_zh] += total_ops[ColDef.device_time_percentage_zh].fillna(0)
        total_ops[ColDef.operator_counts_ratio_in_network_sum_zh] += total_ops[ColDef.count_percentage_zh].fillna(0)
        total_ops.drop(columns=[ColDef.count_percentage_zh, ColDef.device_time_percentage_zh], inplace=True)

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return total_ops

def get_framework_tpi_data_new(dfs_dic, network_list_fw):
    logging.debug("{} start".format(sys._getframe().f_code.co_name))
    # TODO
    # filtered by network_need
    # generate framework_summary
    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return framework_summary

def get_framework_summary(dfs_dic: typing.Dict, network_list: pd.DataFrame, framework_name: str) -> pd.DataFrame:
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    # get filter
    # 取 case_souce==2 且 framework==framework_name 的所有行
    network_list_fw = network_list[(network_list[ColDef.case_source] == 2) & (network_list[ColDef.framework] == framework_name)]
    # 取 network_id 和 network_name 这两列，用于筛选 network
    network_list_fw = network_list_fw.loc[:,[ColDef.network_id, ColDef.network_name]]

    framework_summary = get_framework_tpi_data(dfs_dic, network_list_fw)

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return framework_summary