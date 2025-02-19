#!/usr/bin/env python3
"""
    Entrance of removing duplicated cases and export case_list
"""

import logging

def run(args):
    logging.info("run remove_duplicated_cases start")

    from analysis_suite.core.case_deduplicator import deduplicator
    deduplicator.run(args)

    logging.info("run remove_duplicated_cases end")

if __name__ == "__main__":
    import sys
    import traceback
    import argparse
    import textwrap
    from analysis_suite.args_parser.deduplicate_parser import Deduplicate_Parser

    description = """
"""
    class MyFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
        pass

    help_text_example = textwrap.dedent("""
Example:
    # (a) remove duplicated cases from directory, where
    dir_path
    | - op0
    |    | - case0
    |    | - case1
    |    | - ...
    |    | - caseN
    | - ...
    | - opN
         | - case0
         | - case1
         | - ...
         | - caseN
    python3 remove_duplicated_cases.py --src_case_dir /xxx/xxx/dir_path/

    # (b) remove duplicated cases from case list
    python3 remove_duplicated_cases.py --src_case_dir /xxx/xxx/case_list_path

    # (c) select cases in specified operators to deduplicating
    python3 remove_duplicated_cases.py --src_case_dir /xxx/xxx/dir_path/ --ops "abs;matmul"

    see more details in http://wiki.cambricon.com/pages/viewpage.action?pageId=137672096

""")

    # initialize arguments parser
    parser = Deduplicate_Parser(
        argparse.ArgumentParser(
            description=description,
            formatter_class=MyFormatter,
            epilog=help_text_example
        ))

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

