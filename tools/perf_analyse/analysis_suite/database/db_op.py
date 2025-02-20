"""
    operators for database
"""

__all__ = (
    "create_engine"
)

import enum
import sys
import logging
import pandas as pd
import sqlalchemy
import base64

from analysis_suite.cfg.config import DBConfig

def create_engine(db_name: DBConfig.DB_Name):
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    user_name = str(base64.b64decode(DBConfig.pass_port["user_name"]))[2:-1]
    pass_word = str(base64.b64decode(DBConfig.pass_port["pass_word"]))[2:-1]
    
    engine = None
    if DBConfig.DB_Name.training_solution == db_name:
        engine = sqlalchemy.create_engine(
            "mysql+pymysql://{}:{}@10.101.9.21/{}".format(user_name, pass_word, DBConfig.DB_Name_mp[db_name])
        )
    elif DBConfig.DB_Name.rainbow == db_name:
        engine = sqlalchemy.create_engine(
            "mysql+pymysql://{}:{}@10.101.9.21/{}".format(user_name, pass_word, DBConfig.DB_Name_mp[db_name])
        )
    elif DBConfig.DB_Name.local_db == db_name:
        engine = sqlalchemy.create_engine(
            "sqlite:///{}".format(DBConfig.DB_Name_mp[db_name])
        )
    else:
        logging.error("Unknown name of database")

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return engine
