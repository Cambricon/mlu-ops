"""
    helper functions for arguement parsers
"""

__all__ = (
    "str2bool",
)

import argparse

def str2bool(v):
    # ref stackoverflow 15008758
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")
