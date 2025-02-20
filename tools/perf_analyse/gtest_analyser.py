#!/usr/bin/env python3
'''
    Entrance to the analyser of gtest result
'''

import sys
import os
import logging
import pandas as pd

# for debugging
df0_path = 'df0.csv'
df1_path = 'df1.csv'

def dfs2csv(dfs):
    dfs[0].to_csv(df0_path, index=False)
    dfs[1].to_csv(df1_path, index=False)

def csv2dfs():
    df0 = None
    df1 = None
    if os.path.exists(df0_path):
        df0 = pd.read_csv(df0_path)
    if os.path.exists(df1_path):
        df1 = pd.read_csv(df1_path)
    if (df0 is None) or (df1 is None):
        return None
    return [df0, df1]

def run(args):
    from analysis_suite.core.gtest_parser import parser
    from analysis_suite.core.perf_analyser.scheduler import Scheduler

    logging.info("run gtest_analyser start")

    if args.log_path:
        # parse output of gtestx
        dfs = parser.parse_into_dataframes(
                args.log_path,
                args.compare_path,
                cpu_count           = args.cpu_count,
                use_db              = args.use_db,
                need_case_info      = args.need_case_info,
                filter_failed_cases = args.filter_failed_cases,
                export_failed_cases = args.export_failed_cases,
            )
        '''
        # for debugging
        logging.info("read csv for debugging...")
        dfs = csv2dfs()
        if dfs is None:
            logging.info("cannot read csv, parse XML directly.")
            dfs = parser.parse_into_dataframes(
                    args.log_path,
                    args.compare_path,
                    cpu_count           = args.cpu_count,
                    use_db              = args.use_db,
                    need_case_info      = args.need_case_info,
                    filter_failed_cases = args.filter_failed_cases,
                    export_failed_cases = args.export_failed_cases,
                )
            dfs2csv(dfs)
        '''

        # analyse result of the last phase
        Scheduler(args).analyse_dataframes(dfs)

    logging.info("run gtest_analyser end")

if __name__ == "__main__":
    import sys
    import traceback
    import argparse
    from analysis_suite.args_parser.gtest_analyser_parser import Gtest_Analyser_Parser

    # initialize arguments parser
    parser = Gtest_Analyser_Parser(argparse.ArgumentParser(prog="gtest_analyser"))

    # parse arguments
    args = parser.parse_args()

    # initialize `logging` module
    from analysis_suite.utils import logger_helper
    logger_helper.logger_init(args)

    logging.debug(args)

    try:
        run(args)
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
