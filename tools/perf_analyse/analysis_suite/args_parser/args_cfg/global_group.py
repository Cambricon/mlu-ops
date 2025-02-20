"""
    arguments for global configure
"""

__all__ = (
    "add_global_group",
)

import argparse

from analysis_suite.args_parser.args_cfg.type_trans_helper import str2bool

# == For global configure ==
def add_global_group(parser):
    global_group = parser.add_argument_group('global arguments', 'optional for global configure')

    global_group.add_argument('--logging_level',
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                            help="logging level, default INFO",
                            default="INFO",
                            type=str)
    global_group.add_argument("--cpu_count",
                            type=int,
                            default=4,
                            help="the count of parsing prototxt cpu cores, default 4")
    global_group.add_argument("--use_db",
                            type=str2bool,
                            default=True,
                            help="whether use database, default True")
