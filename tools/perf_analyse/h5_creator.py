#!/usr/bin/env python3
'''
    Entrance to the .h5 files generator
'''

import logging

def run(args):
    logging.info("run h5_creator start")

    from analysis_suite.core.h5_creator import generator
    generator.gen_h5(args.cases_dir, args.cpu_count)

    logging.info("run h5_creator end")

if __name__ == "__main__":
    import sys
    import traceback
    import argparse
    from analysis_suite.args_parser.h5_creator_parser import H5_Creator_Parser

    # initialize arguments parser
    parser = H5_Creator_Parser(argparse.ArgumentParser())

    # parse arguments
    args = parser.parse_args()

    # initialize `logging` module
    from analysis_suite.utils import logger_helper
    logger_helper.logger_init(args)

    try:
        logging.debug(args)
        run(args)
    except Exception as e:
        print(e)
        traceback.print_exc()
        sys.exit(1)
