"""
    information struct obtained by parsing xml/log
"""

__all__ = (
    "TestInfo",
)

import typing
import pandas as pd

class TestInfo:
    def __init__(self,
                env: typing.Dict,
                perf: pd.DataFrame):
        self.env = env
        self.perf = perf

    def __str__(self):
        return "TestInfo(\nenv = {}\n,\nperf = \n{}\n)".format(self.env, self.perf)

