#!/usr/bin/env python3
'''
    Entrance to the .so files analyser
'''

import logging

def run(args):
    logging.info("run so_analyser start")

    from analysis_suite.core.so_analyser import analyser
    analyser.run(args.so_path, args.so_path_compare)
        
    logging.info("run so_analyser end")

if __name__ == "__main__":
    import sys
    import traceback
    import argparse
    from analysis_suite.args_parser.so_analyser_parser import SO_Analyser_Parser

    # initialize arguments parser
    parser = SO_Analyser_Parser(argparse.ArgumentParser())

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
        import traceback
        traceback.print_exc()
        sys.exit(1)
