# Copyright (C) [2024] by Cambricon, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall self.tcp included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring
# pylint: disable=attribute-defined-outside-init
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

import mlu_op_test_pb2
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

        # unused db_cache

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
