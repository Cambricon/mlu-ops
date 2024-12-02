"""
    Parser for so_analyser
"""

__all__ = (
    "SO_Analyser_Parser",
)

from analysis_suite.args_parser._base_parser import _Base_Parser

class SO_Analyser_Parser(_Base_Parser):
    def add_args(self):
        from analysis_suite.args_parser.args_cfg import global_group, so_group

        global_group.add_global_group(self.m_parser)
        so_group.add_so_group(self.m_parser)

    def add_subparsers(self):
        pass

