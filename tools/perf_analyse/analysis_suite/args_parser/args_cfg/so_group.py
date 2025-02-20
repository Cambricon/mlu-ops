"""
    arguments for .so file analyser
"""

__all__ = (
    "add_so_group",
)

import argparse

def add_so_group(parser):
    so_group = parser.add_argument_group('parsing .so arguments', 'optional for getting size of .so file')

    so_group.add_argument("--so_path",
                        type=str,
                        help="path of libcnnl.so to get code size")
    so_group.add_argument("--so_path_compare",
                        type=str,
                        help="path of libcnnl.so to compare")
