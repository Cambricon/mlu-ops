"""
    Table: operator_summary(cri)
    Demand Source: http://jira.cambricon.com/browse/CNNLCORE-19647
    Output Columns:
    [
        operator,
        all case number of current operator,
        case number after filter,
        if is io bound, # True / False
        4 X [ # 4 is decided by criterion
                case number in current status,
                percentage of cases with the current status in all cases after filtering,
                average compute / IO efficiency,
            ]
    ]
"""

__all__ = (
    "get_operator_summary_under_criterion",
)

import sys
import logging
import pandas as pd

from analysis_suite.cfg.config import ColDef, PerfConfig
from analysis_suite.core.perf_analyser.perf.get_data_details import perf_get_data_utils

def get_operator_summary_under_criterion_impl(df: pd.DataFrame, perf_config: PerfConfig):
    if len(df) == 0:
        logging.warn("input table is empty.")
        return pd.DataFrame()

    # prepare result table and columns
    columns = [
        ColDef.operator,
        ColDef.all_case_number,
        ColDef.filtered_case_number,
        ColDef.is_io_bound
    ]
    for status in perf_config.attrs['criterion']:
        columns += [status + '_case_number', status + '_rate', status + '_efficiency_mean']
    summary = pd.DataFrame(columns=columns)
    # _all_ops 
    all_op_statistics = [
        ["_all_ops", 0, 0, False] + len(perf_config.attrs['criterion']) * [0, 0, 0],
        ["_all_ops", 0, 0, True] + len(perf_config.attrs['criterion']) * [0, 0, 0]
    ]

    # status_dict: status -> index
    status_dict = {key: index + 1 for index, key in enumerate(perf_config.attrs['criterion'])}
    # calculate number of cases in each group grouped by operator
    all_case_number_dict = df.groupby(ColDef.operator)[ColDef.file_path].count().to_dict() # TODO: use [ColDef.repeat_count].sum()

    # ignore useless cases
    df = perf_get_data_utils.ignore_useless_cases(df, perf_config)

    # 1. group by operator
    for op, op_group_df in df.groupby(ColDef.operator):
        # 2. group by is_io_bound
        grouped_by_bound = op_group_df.groupby(ColDef.is_io_bound)
        if 2 >= len(grouped_by_bound.groups):
            for bound, bound_group_df in grouped_by_bound:
                # init current row
                row = [op, all_case_number_dict[op], None, bound] + (len(columns)-4) * [None]
                all_op_statistics[bound][1] += all_case_number_dict[op]

                # calculate case number after filtering
                case_num_in_bound = bound_group_df[ColDef.file_path].count() # TODO: use [ColDef.repeat_count].sum()
                row[2] = case_num_in_bound
                all_op_statistics[bound][2] += case_num_in_bound

                # 3. group by status
                grouped_by_status = bound_group_df.groupby(ColDef.status)

                # calculate case number for each status
                # case_num_in_status: status -> num
                case_num_in_status = grouped_by_status[ColDef.file_path].apply(lambda x: x.count()) # TODO: use [ColDef.repeat_count].sum()
                # calculate mean efficiency for each status
                #  efficiency_mean_in_status: status -> efficiency_mean
                if True == bound:
                    efficiency_mean_in_status = grouped_by_status[ColDef.mlu_io_efficiency].apply(lambda x: x.mean())
                else:
                    efficiency_mean_in_status = grouped_by_status[ColDef.mlu_compute_efficiency].apply(lambda x: x.mean())
                # iterate over all status to update current row
                for status in case_num_in_status.index:
                    if status in status_dict:
                        # set case number
                        row[3 * status_dict[status] + 1] = case_num_in_status[status]
                        # set rate
                        row[3 * status_dict[status] + 2] = case_num_in_status[status] / row[1]
                        # set mean efficiency
                        row[3 * status_dict[status] + 3] = efficiency_mean_in_status[status]
                        # update xxx_num in all_op_statistics
                        all_op_statistics[bound][3 * status_dict[status] + 1] += case_num_in_status[status]
                        # accumulate xxx_efficiency_rate
                        all_op_statistics[bound][3 * status_dict[status] + 3] += case_num_in_status[status] * efficiency_mean_in_status[status]
                # append row
                summary.loc[len(summary)] = row
        else:
            logging.error("Error when group by is_io_bound on dataframe, the operator is {}, the length is {}".format(op, len(grouped_by_bound.groups)))

    # zero padding
    summary = summary.fillna(0)

    # update all_op_statistics
    all_op_statistics[0][1] += all_op_statistics[1][1]
    all_op_statistics[1][1] = all_op_statistics[0][1]
    for status in status_dict:
        for bound in [True, False]:
            # calculate xxx_rate
            if 0 != all_op_statistics[bound][2]:
                all_op_statistics[bound][3 * status_dict[status] + 2] = \
                    all_op_statistics[bound][3 * status_dict[status] + 1] / all_op_statistics[bound][2]
            # calculate final xxx_efficiency_rate
            if 0 != all_op_statistics[bound][3 * status_dict[status] + 1]:
                all_op_statistics[bound][3 * status_dict[status] + 3] = \
                    all_op_statistics[bound][3 * status_dict[status] + 3] / all_op_statistics[bound][3 * status_dict[status] + 1]

    # append all_op_statistics
    summary.loc[len(summary)] = all_op_statistics[0]
    summary.loc[len(summary)] = all_op_statistics[1]

    # sort by operator name
    summary = summary.sort_values(by=[ColDef.operator, ColDef.is_io_bound], ascending=True)
    summary = summary.reset_index(drop=True)

    return summary

def get_operator_summary_under_criterion(df: pd.DataFrame, perf_config: PerfConfig):
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    summary = get_operator_summary_under_criterion_impl(df, perf_config)

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return summary
