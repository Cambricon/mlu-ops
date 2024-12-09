"""
    Parser for h5_creator
"""

__all__ = (
    "H5_Creator_Parser",
)

from analysis_suite.args_parser._base_parser import _Base_Parser

class H5_Creator_Parser(_Base_Parser):
    def add_args(self):
        from analysis_suite.args_parser.args_cfg import global_group, h5_group

        global_group.add_global_group(self.m_parser)
        h5_group.add_h5_group(self.m_parser)

    def add_subparsers(self):
        pass

