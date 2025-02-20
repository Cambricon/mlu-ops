"""
    helper functions for tpi
"""

__all__ = (
    "move_column_location",
    "dump_tpi_excel",
    "get_txt_excel_to_tar",
    "get_important_network_sheet",
    "get_important_network_names",
)

import re
import os
import sys
import logging
import pandas as pd
import tarfile

def move_column_location(df, loc, column_name):
    df_tmp = df[column_name]
    df = df.drop(column_name, axis=1)
    df.insert(loc, column_name, df_tmp)
    return df

# for tpi
def dump_tpi_excel(dfs, sheet_names, tpi_path, float_to_percentage_cols):
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    from analysis_suite.utils import excel_helper

    excel_helper.dfs_to_excel_impl(dfs, sheet_names, tpi_path, float_to_percentage_cols)
    logging.info("TPI excel has been written to {}".format(tpi_path))

    logging.debug("{} end".format(sys._getframe().f_code.co_name))

# for tpi
def get_txt_excel_to_tar(cases, file_name="tarfile.tar"):
    logging.info("{} start".format(sys._getframe().f_code.co_name))

    name_suffix = "_" + file_name.split("/")[-1].replace(".tar", "")
    tar = tarfile.open(file_name, "w")
    for name, case in cases.items():
        case.to_csv(name + name_suffix + '.txt', sep='\t', index=True, header=True)
        case.to_excel(name + name_suffix + '.xlsx')
        tar.add(name + name_suffix + '.txt')
        tar.add(name + name_suffix + '.xlsx')
        os.remove(name + name_suffix + '.txt')
        os.remove(name + name_suffix + '.xlsx')
    tar.close()

    logging.info("TPI tar has been written to {}".format(file_name))
    logging.info("{} end".format(sys._getframe().f_code.co_name))

# for tpi
def dump_tpi_db(df, table_name="origin_table"):
    logging.info("{} start".format(sys._getframe().f_code.co_name))

    from analysis_suite.cfg.config import DBConfig
    from analysis_suite.database import db_op
    # create sqlite engine
    engine = db_op.create_engine(DBConfig.DB_Name.local_db)

    # write to sql
    # index=False: do not save row label
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)

    # close engine
    engine.dispose()

    logging.info("{} end".format(sys._getframe().f_code.co_name))

# for simple_tpi
def get_important_network_sheet(df, important_network_names):
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    not_important_names = []
    for sheet_name in df.keys():
        if sheet_name not in important_network_names:
            not_important_names.append(sheet_name)
    #[df.pop(x) for x in not_important_names] # 从dict中按key进行pop
    for x in not_important_names:
        df.pop(x)

    logging.debug("{} end".format(sys._getframe().f_code.co_name))

# for simple_tpi
def get_important_network_names(all_network, important_network_keyword, framework_names):
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    network_names = []
    for key in all_network:
        for item in important_network_keyword:
            for fw_name in framework_names:
                # all_network 中的 key 要同时满足在 framework_names 中的 important_network_keyword 的网络
                # re.search(pattern, text) 用于正则匹配
                if re.search(item, key) and re.search(fw_name, key):
                    network_names.append(key)

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return network_names
