"""
    Parser for gtest_log_to_xlsx
    # 伺候老命令
"""

__all__ = (
    "Gtest_Log_to_Xlsx_Parser",
)

from analysis_suite.args_parser._base_parser import _Base_Parser

class Gtest_Log_to_Xlsx_Parser(_Base_Parser):
    def add_args(self):
        from analysis_suite.args_parser.args_cfg import global_group, gtest_group, h5_group, so_group

        global_group.add_global_group(self.m_parser)
        gtest_group.add_gtest_group(self.m_parser)
        h5_group.add_h5_group(self.m_parser)
        so_group.add_so_group(self.m_parser)

    def add_subparsers(self):
        from analysis_suite.args_parser.gtest_analyser_parser import Gtest_Analyser_Parser
        from analysis_suite.args_parser.h5_creator_parser import H5_Creator_Parser
        from analysis_suite.args_parser.so_analyser_parser import SO_Analyser_Parser

        subparsers = self.m_parser.add_subparsers(dest="subcommand", required=False)
        Gtest_Analyser_Parser(subparsers.add_parser('gtest')).add_args()
        H5_Creator_Parser(subparsers.add_parser('h5')).add_args()
        SO_Analyser_Parser(subparsers.add_parser('so')).add_args()

