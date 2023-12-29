#!/usr/bin/env python3
#coding:utf-8

import pandas as pd
import xml.etree.ElementTree as ET
import google.protobuf.json_format as json_format
import re
import os
import sys
import json
import tqdm
import logging
import hashlib

import mluop_test_pb2
from db_cache import DBCache
from dirty_worker import DirtyWorker
from parser import Parser
from processor import Processor
from protobuf_case_parser_impl_inputoutput import ProtobufCaseParserImplInputOutput
from config import Config,PerfConfig

class Executor:

    def __init__(self, args):
        self.args_ = args
        self.db_ = None
        if self.args_.use_db == 1:
            self.db_ = DBCache()

    def run(self):
        try:
            # handle h5 and so size
            DirtyWorker(self.args_).run()

            # parsing xml or log
            dfs = Parser(self.args_, self.db_).run()

            # processing tpi
            Processor(self.args_, self.db_).run(dfs)

        # can be more specific, such as ParseError JSONDecodeError
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            sys.exit(1)
