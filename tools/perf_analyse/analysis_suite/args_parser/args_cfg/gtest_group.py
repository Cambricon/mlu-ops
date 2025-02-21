"""
    arguments for gtest_parser and perf_analyser
"""

__all__ = (
    "add_gtest_group",
)

import argparse

from analysis_suite.args_parser.args_cfg.type_trans_helper import str2bool

# == For parsing output of gtest ==
# TODO(tanghandi): 后续把gtest输出存性能数据库后，需要拆分
def add_gtest_group(parser):
    gtest_group = parser.add_argument_group('gtest arguments', 'optional for parsing output of gtest')
    # gtest_parser & perf_analyser: input & output
    gtest_group.add_argument("--log_path",
                            type=str,
                            help="file or folder path of xml or log")
    gtest_group.add_argument("--compare_path",
                            type=str,
                            help="baseline file or folder path of xml or log")
    gtest_group.add_argument("--xlsx_path",
                            type=str,
                            help="path of the output excel file")
    # gtest_parser: for adding case info
    gtest_group.add_argument("--need_case_info",
                            type=str2bool,
                            default=True,
                            help="whether need `input`, `output` and `params` columns, default True. If False, the script will not parse pt/pb files.")
    # gtest_parser: for filtering failed cases
    gtest_group.add_argument("--filter_failed_cases",
                            type=str2bool,
                            default=False,
                            help="whether remove failed cases in log/xml file, default False")
    gtest_group.add_argument("--export_failed_cases",
                            type=str2bool,
                            default=False,
                            help="whether export failed cases in log/xml file, default False")
    # perf_analyser:
    # TODO: use this option for supporting deduplication script
    gtest_group.add_argument("--deduplication",
                            type=str2bool,
                            default=False,
                            help="whether remove duplicated case(input/output/params scale), default False")
    # perf_analyser: for setting the field
    gtest_group.add_argument("--is_release",
                            type=str2bool,
                            default=False,
                            help="whether is release or not(daily), default is daily")
    # perf_analyser: for tpi
    gtest_group.add_argument("--tpi",
                            type=str2bool,
                            help="whether generate tpi excel file")
    # perf_analyser: for simple tpi
    gtest_group.add_argument("--simple_tpi",
                            type=str2bool,
                            help="whether generate simple_tpi excel file")
    gtest_group.add_argument("--frameworks",
                            type=str,
                            default="pytorch",
                            help="filter of frameworks")
    # perf_analyser: for `perf` module to generate pic
    gtest_group.add_argument("--generate_pic",
                            type=str2bool,
                            default=True,
                            help="whether generate images which contain cases' comparison info, default True")
    # perf_analyser: for `perf` module to filter result by operator
    gtest_group.add_argument("--json_file",
                            type=str,
                            help="json file to select operator")
    gtest_group.add_argument("--is_pro",
                            type=str2bool,
                            default=True,
                            help="whether use JSON file operators for positive selection, default True")
    # perf_analyser: old database. no longer used
    gtest_group.add_argument("--case_run",
                            type=str2bool,
                            help="whether update mluops_case_run")
    gtest_group.add_argument("--truncate_case_run",
                            type=str2bool,
                            help="whether truncate mluops_case_run")

