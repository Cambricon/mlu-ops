#!/usr/bin/env python3
#coding:utf-8

import logging
from sqlalchemy import create_engine, types
import pandas as pd
import base64

from config import Config

class DBCache:

    def __init__(self):
        logging.info("DBCache init start")
        self.init()
        logging.info("DBCache init end")

    def init(self):
        user_name = str(base64.b64decode(Config.pass_port["user_name"]))[2:-1]
        pass_word = str(base64.b64decode(Config.pass_port["pass_word"]))[2:-1]
        self.engine = create_engine(
            "mysql+pymysql://{0}:{1}@10.101.9.21/training_solution".format(user_name, pass_word)
        )
        self.engine_rainbow = create_engine(
            "mysql+pymysql://{0}:{1}@10.101.9.21/rainbow".format(user_name, pass_word))
        self.case_in_network = pd.read_sql_table(
            'mluop_case_in_network_test',
            self.engine,
            columns=['case_id', 'network_id', 'count'])
        self.network_list = pd.read_sql_table('mluop_network_list_test',
                                              self.engine)
        # dont need date columns
        self.network_list.drop(columns=['date'], inplace=True)
        # only read necessary columns
        case_columns = ['case_id', 'protoName', 'input', 'output', 'params']
        self.case_list = pd.read_sql_table(
            'mluop_case_information_benchmark_test',
            self.engine,
            columns=case_columns)

        self.network_summary = pd.read_sql_table(
            'mluop_network_summary_test',
            self.engine_rainbow,
            columns=['network_id', 'mlu_platform', 'mlu_hardware_time_sum', 'date'])
        if not self.network_summary.empty:
            network_summary_max_date = self.network_summary.groupby(['network_id','mlu_platform']).agg({'date': 'max'})
            self.network_summary = pd.merge(network_summary_max_date, self.network_summary, on=['network_id', 'mlu_platform', 'date'])
            self.network_summary.drop_duplicates(subset=['network_id', 'mlu_platform'], keep='last', inplace=True)
        self.network_summary.drop(columns=['date'], inplace=True)
        self.owner_resources = pd.read_sql_table(
            'mluop_owner_resources_test',
            self.engine_rainbow,
            columns=['operator', 'owner', 'resources'])
