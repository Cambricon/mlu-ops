"""
    Helper functions for input pathes
"""

__all__ = (
    "check_dir",
    "check_file",
)

import os

def to_abspath(path: str) -> str:
    userpath = os.path.expanduser(path)
    abspath = os.path.abspath(userpath)
    return abspath

def check_dir(path: str) -> str:
    abspath = to_abspath(path)

    if not os.path.exists(abspath):
        raise Exception(f"path {path} not exits.")

    if not os.path.isdir(abspath):
        raise Exception(f"path {path} is not a direcoty")

    return abspath

def check_file(path: str) -> str:
    abspath = to_abspath(path)

    if not os.path.exists(abspath):
        raise Exception(f"path {path} not exits.")

    if not os.path.isfile(abspath):
        raise Exception(f"path {path} is not a file")

    return abspath
