"""
    implement of package `so_analyser`
"""

__all__ = (
    "run",
)

import re
import os
import subprocess
import pandas as pd

from analysis_suite.cfg.config import Config
from analysis_suite.utils import excel_helper

def get_code_size(so_path):
    # get file size of operator.a and libcnnl.so
    lib_path = os.path.abspath(so_path)
    cmd_args = ["readelf", "-e", lib_path]
    operator = []
    sizes = []

    cmd_ret = subprocess.run(cmd_args, check=True, stdout=subprocess.PIPE)
    so_size = re.findall(r"cn_fatbin(.*?) \[\d+\]", str(cmd_ret.stdout))[0]
    so_size = int(re.findall(r"\w+", so_size)[4], 16)
    operator.append('libcnnl.so')
    sizes.append(os.path.getsize(lib_path))
    operator.append('cn_fatbin')
    sizes.append(so_size)

    data = {'operator': operator, 'size': sizes}
    df = pd.DataFrame(data)
    return df

def compare_code_size(code_size_bl, code_size_cmp):
    code_size_compare = \
        pd.merge(
            code_size_bl,
            code_size_cmp,
            suffixes=Config.suffix,
            on=['operator']
        )
    code_size_compare['size提升(Bytes)'] = \
        code_size_compare['size' + Config.suffix[1]] - \
        code_size_compare['size' + Config.suffix[0]]
    code_size_compare['size提升比例(Bytes)'] = \
        code_size_compare['size提升(Bytes)'] / \
        code_size_compare['size' + Config.suffix[1]]
    code_size_compare['size提升比例(Bytes)'] = \
        code_size_compare['size提升比例(Bytes)'].apply("{:.2%}".format)
    return code_size_compare

def run(so_path, so_path_compare):
    if so_path:
        code_size = get_code_size(so_path)
        excel_helper.dfs_to_excel_impl(
            [code_size],
            ['code_size'],
            "code_size.xlsx"
        )
        if so_path_compare:
            code_size_cmp = get_code_size(so_path_compare)
            code_size_diff = compare_code_size(code_size, code_size_cmp)
            excel_helper.dfs_to_excel_impl(
                [code_size_diff],
                ['code_size_compare'],
                "code_size_compare.xlsx"
            )
