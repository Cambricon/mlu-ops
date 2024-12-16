"""
    arguments for .h5 file generator
"""

__all__ = (
    "add_h5_group",
)

import argparse

def add_h5_group(parser):
    h5_group = parser.add_argument_group('generate .h5 arguments', 'optional for generating .h5 file')

    h5_group.add_argument("--cases_dir",
                        type=str,
                        help="cases directory for generating h5")
