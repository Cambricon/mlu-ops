"""
    arguments for remove duplicate cases
"""

__all__ = (
    "add_deduplicate_group",
)

import argparse

from analysis_suite.args_parser.args_cfg.type_trans_helper import str2bool

def add_deduplicate_group(parser):
    deduplicate_group = parser.add_argument_group('duplicated arguments', 'options for removing duplicate cases')

    # input: path
    deduplicate_group.add_argument('--src_case_dir', # /xxx/.../gen_case
                                    type=str,
                                    help='the dir of gen_case')
    deduplicate_group.add_argument('--src_case_list',
                                    type=str,
                                    help="case list of input cases.")
    # input: mode for option `src_case_dir`
    deduplicate_group.add_argument('--ops',
                                    type=str,
                                    default='all',
                                    help="select operators for option `src_case_dir`, default 'all'. if there're multiple operators, use ';' to concat them.")
    # output: path
    deduplicate_group.add_argument('--dst_case_list',
                                    type=str,
                                    default="case_list_deduplicated.json",
                                    help='case list after deduplication, default `case_list_deduplicated.json`')
