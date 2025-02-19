"""
    Tables: networks
    TODO: use database to save data, see "get_networks_and_dump"
"""

__all__ = (
    "get_networks_and_append",
)

import os
import sys
import logging
import pandas as pd
import base64

from analysis_suite.cfg.config import Config, ColDef
from analysis_suite.core.perf_analyser.tpi import tpi_utils

# iterate each group in group_by_network and append tpi data(by operator) to dfs
def get_networks_and_append(group_by_network, dfs, sheet_names, origin_columns, columns):
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    # this code is very hot
    for network, network_df in group_by_network:
        # append dataframe
        tpi = pd.DataFrame()
        # group by operator
        group_by_op = network_df.groupby(ColDef.operator)

        # is_io_bound==True: get mean of mlu_io_efficiency
        tpi[ColDef.mlu_io_efficiency] = group_by_op.apply(
            lambda x: x[x[ColDef.is_io_bound]][ColDef.mlu_io_efficiency].sum() / \
                max(x[x[ColDef.is_io_bound]][ColDef.count].sum(), Config.epi)
        )
        # is_io_bound==False: get mean of mlu_compute_efficiency
        tpi[ColDef.mlu_compute_efficiency] = group_by_op.apply(
            lambda x: x[~x[ColDef.is_io_bound]][ColDef.mlu_compute_efficiency].sum() / \
                max(x[~x[ColDef.is_io_bound]][ColDef.count].sum(), Config.epi)
        )

        # get sum
        for i in [
            ColDef.mlu_workspace_size,
            ColDef.mlu_hardware_time,
            ColDef.mlu_interface_time
        ]:
            tpi[i] = group_by_op.agg({i: ['sum']})[i]['sum'] # maybe need to add '_sum' in tpi[i]

        # get total number
        tpi[ColDef.counts] = group_by_op.agg({ColDef.count: ['sum']})[ColDef.count]['sum']

        # get mean (percentage)
        for i in [
            ColDef.mlu_workspace_size,
            ColDef.mlu_hardware_time,
            ColDef.mlu_interface_time
        ]:
            tpi[i + '_mean'] = tpi[i] / tpi[ColDef.counts]

        # 有多少比例的case是 is_io_bound==True 的
        tpi[ColDef.io_bound_percentage] = group_by_op.apply(lambda x: x[x[ColDef.is_io_bound]][ColDef.count].sum()) / tpi[ColDef.counts]
        tpi[ColDef.io_bound_percentage] = tpi[ColDef.io_bound_percentage].apply("{:.2%}".format)

        # reset index
        tpi = tpi.reset_index()
        # rename column names
        tpi.rename(columns=dict(zip(origin_columns, columns)), inplace=True)
        # append dataframe
        dfs.append(tpi)
        # append sheet name
        sheet_names.append(network.lower())

    logging.debug("{} end".format(sys._getframe().f_code.co_name))

def get_networks_and_dump():
    logging.debug("{} start".format(sys._getframe().f_code.co_name))
    # TODO: use database to save data
    logging.debug("{} end".format(sys._getframe().f_code.co_name))