"""
    Table: summary
"""

__all__ = (
    "get_network_summary",
)

import sys
import logging
import pandas as pd

from analysis_suite.cfg.config import Config, ColDef

def get_network_summary(group_by_network, network_list, origin_columns, columns):
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    tpi_network = pd.DataFrame()
    # is_io_bound 为 True/False, 见 parser.py 的 preprocess 函数
    tpi_network[ColDef.mlu_io_efficiency] = \
        group_by_network.apply(
            lambda x: x[x[ColDef.is_io_bound]][ColDef.mlu_io_efficiency].sum() / \
            max(x[x[ColDef.is_io_bound]][ColDef.count].sum(), Config.epi)
        )
    # is_io_bound==False, 对 mlu_compute_efficiency 进行同上操作
    tpi_network[ColDef.mlu_compute_efficiency] = \
        group_by_network.apply(lambda x: x[~x[ColDef.is_io_bound]][ColDef.mlu_compute_efficiency].sum() / \
        max(x[~x[ColDef.is_io_bound]][ColDef.count].sum(), Config.epi)
    )

    # 对 mlu_workspace_size, mlu_hardware_time, mlu_interface_time 求和
    for i in [
        ColDef.mlu_workspace_size,
        ColDef.mlu_hardware_time,
        ColDef.mlu_interface_time
    ]:
        tpi_network[i] = group_by_network.agg({i: ['sum']})[i]['sum']

    # 对 mlu_workspace_size, mlu_hardware_time, mlu_interface_time 求加权平均
    tpi_network[ColDef.counts] = \
        group_by_network.agg({ColDef.count:['sum']})[ColDef.count]['sum']
    for i in [
        ColDef.mlu_workspace_size,
        ColDef.mlu_hardware_time,
        ColDef.mlu_interface_time
    ]:
        tpi_network[i + '_mean'] = tpi_network[i] / tpi_network[ColDef.counts]

    # 有多少比例的case是 is_io_bound==True 的
    tpi_network[ColDef.io_bound_percentage] = \
        group_by_network.apply(
            lambda x: x[x[ColDef.is_io_bound]][ColDef.count].sum()
        ) / tpi_network[ColDef.counts]

    # 重置索引，原索引是groupby的key，whole_name
    tpi_network = tpi_network.reset_index()

    tpi_network = \
        pd.merge(
            tpi_network,
            network_list[
                [
                    ColDef.whole_name,
                    ColDef.network_id,
                    ColDef.case_source,
                    ColDef.framework
                ]
            ],
            on=[ColDef.whole_name]
        )

    # 重命名字段
    # 由上代码可知，所有字段为 
    # [
    #   whole_name,
    #   mlu_io_efficiency, mlu_compute_efficiency,
    #   mlu_workspace_size, mlu_hardware_time, mlu_interface_time,
    #   count,
    #   mlu_workspace_size_mean, mlu_hardware_time_mean, mlu_interface_time_mean,
    #   io_bound_percentage,
    #   network_id, case_source, framework
    # ]
    # 上面 origin_columns -> columns 中缺少了 network_id, 这个项只在调用 get_network_tpi_data 时起索引作用
    tpi_network.rename(columns=dict(zip(origin_columns, columns)), inplace=True)

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return tpi_network