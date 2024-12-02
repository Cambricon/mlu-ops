"""
    Parser for gtest_analyser
"""

__all__ = (
    "Gtest_Analyser_Parser",
)

from analysis_suite.args_parser._base_parser import _Base_Parser

class Gtest_Analyser_Parser(_Base_Parser):
    def add_args(self):
        from analysis_suite.args_parser.args_cfg import global_group, gtest_group

        global_group.add_global_group(self.m_parser)
        gtest_group.add_gtest_group(self.m_parser)

    def add_subparsers(self):
        pass

