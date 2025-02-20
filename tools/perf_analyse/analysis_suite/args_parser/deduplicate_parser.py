"""
    Parser for remove_duplicate_cases
"""

__all__ = (
    "Deduplicate_Parser",
)

from analysis_suite.args_parser._base_parser import _Base_Parser

class Deduplicate_Parser(_Base_Parser):
    def add_args(self):
        from analysis_suite.args_parser.args_cfg import global_group, deduplicate_group

        global_group.add_global_group(self.m_parser)
        deduplicate_group.add_deduplicate_group(self.m_parser)

    def add_subparsers(self):
        pass
