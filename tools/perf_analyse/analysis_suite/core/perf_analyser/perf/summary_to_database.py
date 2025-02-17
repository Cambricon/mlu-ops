"""
    old database, may be useless oneday...
"""

__all__ = (
    "update_database",
)

import sys
import logging
from sqlalchemy import create_engine, types
import pandas as pd
import base64

from analysis_suite.cfg.config import Config, ColDef, DBConfig

class DBCache:
    def __init__(self, db_table):
        logging.info("DBCache init start")
        self.init(db_table)
        logging.info("DBCache init end")

    def init(self, db_table):
        user_name = str(base64.b64decode(DBConfig.pass_port["user_name"]))[2:-1]
        pass_word = str(base64.b64decode(DBConfig.pass_port["pass_word"]))[2:-1]
        if set(db_table) & set(('case_in_network',
                                'network_list',
                                'case_list')):
            self.engine = create_engine(
                "mysql+pymysql://{0}:{1}@10.101.9.21/training_solution".format(user_name, pass_word)
            )

            if 'case_in_network' in db_table:
                self.case_in_network = pd.read_sql_table(
                    'mluops_case_in_network_test',
                    self.engine,
                    columns=[ColDef.case_id,
                             ColDef.network_id,
                             ColDef.count])

            if 'network_list' in db_table:
                self.network_list = pd.read_sql_table(
                    'mluops_network_list_test',
                    self.engine)
                # dont need date columns
                self.network_list.drop(columns=[ColDef.date, ColDef.gen_date, ColDef.mluops_version], inplace=True)

            if 'case_list' in db_table:
                # only read necessary columns
                case_columns = [ColDef.case_id,
                                ColDef.protoName,
                                ColDef.input,
                                ColDef.output,
                                ColDef.params]
                self.case_list = pd.read_sql_table(
                    'mluops_case_information_benchmark_test',
                    self.engine,
                    columns=case_columns)
        if set(db_table) & set(('network_summary',
                                'owner_resources')):
            self.engine_rainbow = create_engine(
                "mysql+pymysql://{0}:{1}@10.101.9.21/rainbow".format(user_name, pass_word))

            if 'network_summary' in db_table:
                self.network_summary = pd.read_sql_table(
                'mluops_network_summary_test',
                self.engine_rainbow,
                columns=[ColDef.network_id,
                         ColDef.mlu_platform,
                         ColDef.mlu_hardware_time_sum,
                         ColDef.date])
                if not self.network_summary.empty:
                    network_summary_max_date = self.network_summary.groupby([ColDef.network_id,ColDef.mlu_platform]).agg({ColDef.date: 'max'})
                    self.network_summary = pd.merge(network_summary_max_date, self.network_summary, on=[ColDef.network_id, ColDef.mlu_platform, ColDef.date])
                    self.network_summary.drop_duplicates(subset=[ColDef.network_id, ColDef.mlu_platform], keep='last', inplace=True)
                    self.network_summary.drop(columns=[ColDef.date], inplace=True)

            if 'owner_resources' in db_table:
                self.owner_resources = pd.read_sql_table(
                    'mluops_owner_resources_test',
                    self.engine_rainbow,
                    columns=[ColDef.operator,
                             ColDef.owner,
                             ColDef.resources])

def mapping_df_types(df):
    dtype_dict = {}
    for i, j in zip(df.columns, df.dtypes):
        if i in Config.case_info_keys:
            dtype_dict.update({i: types.JSON()})
    return dtype_dict

def append_network_info(df, db_):
    # merge database info
    mluops_case_run = \
        pd.merge(
            df,
            db_.case_list[[ColDef.case_id, ColDef.protoName]],
            on=[ColDef.protoName]
        )
    mluops_case_run = \
        pd.merge(
            mluops_case_run,
            db_.case_in_network,
            on=[ColDef.case_id]
        )
    mluops_case_run = \
        pd.merge(
            mluops_case_run,
            db_.network_list,
            on=[ColDef.network_id]
        )

    # only handle mlu_platform in xml
    paltform = Config.platform_map[df.loc[0, ColDef.mlu_platform]]
    mluops_case_run = mluops_case_run[mluops_case_run[paltform] == 1]
    # drop MLU270_X5K columns
    drop_columns = set(db_.network_list.columns) - set(Config.network_info_keys)
    mluops_case_run.drop(columns=drop_columns, inplace=True)
    return mluops_case_run

def update_database(dfs, sheet_names, is_truncate, perf_config):
    logging.info("{} start".format(sys._getframe().f_code.co_name))

    # when all cases recorded in db, md5 not exists
    if ColDef.md5 in dfs[0].columns:
        dfs[0].drop(columns=[ColDef.md5], inplace=True)

    type_info = mapping_df_types(dfs[0])
    action = "replace" if is_truncate else "append"
    db_ = DBCache(['case_list', 'case_in_network', 'network_list', 'network_summary', 'owner_resources'])
    dfs[0] = pd.merge(dfs[0], db_.owner_resources, how="left").fillna(value="unknown")
    dfs[0].to_sql(
        'mluops_case_run_origin_test',
        con=db_.engine_rainbow,
        if_exists=action,
        dtype=type_info,
        index=False
    )
    logging.info("update mluops_case_run_origin_test successfully")

    # df_ignore_small_case
    df_ignore_small_case = pd.DataFrame()
    df_ignore_small_case = \
        dfs[0][dfs[0][ColDef.mlu_hardware_time] > \
        perf_config.attrs['ignore_case'][ColDef.mlu_hardware_time]]
    df_ignore_small_case = \
        pd.merge(df_ignore_small_case, db_.owner_resources, how="left").fillna(value="unknown")
    df_ignore_small_case.to_sql(
        'mluops_case_run_ignore_small_case_test',
        con=db_.engine_rainbow,
        if_exists=action,
        dtype=type_info,
        index=False
    )
    logging.info("update mluops_case_run_ignore_small_case_test successfully")

    # mluops_case_run
    mluops_case_run = append_network_info(dfs[0], db_)
    mluops_case_run.to_sql(
        'mluops_case_run_test',
        con=db_.engine_rainbow,
        if_exists=action,
        dtype=type_info,
        index=False
    )
    logging.info("update mluops_case_run_test successfully")

    # TODO(operator_summary): what about no cases
    tmp_idx = 1
    if 'operator_summary' in sheet_names:
        operator_summary = \
            pd.merge(
                dfs[tmp_idx],
                db_.owner_resources,
                how="left"
            ).fillna(value="unknown")
        operator_summary.to_sql(
            "mluops_operator_summary_test",
            con=db_.engine_rainbow,
            if_exists=action,
            index=False
        )
        tmp_idx += 1
        logging.info("update mluops_operator_summary_test successfully")
    else:
        logging.warn("The test cases are all small cases(mlu_hardware_time< 30), ignore to update mluops_operator_summary_test")

    if 'network_summary' in sheet_names:
        network_summary = dfs[tmp_idx].copy()
        if "mlu_hardware_time_sum_database" in network_summary.columns:
            network_summary.drop(columns=[ColDef.mlu_hardware_time_sum_database], inplace=True)
        network_summary.to_sql(
            "mluops_network_summary_test",
            con=db_.engine_rainbow,
            if_exists=action,
            index=False
        )
        tmp_idx += 1
        logging.info("update mluops_network_summary_test successfully")
    else:
        logging.warn("warning: empty network_summary get, ignore to update mluops_network_summary_test!")
    logging.info("{} end".format(sys._getframe().f_code.co_name))