"""
    Helper funcitons for builtin-types
"""

__all__  = (
    "merge_dict",
)

import sys
import logging
from typing import Dict

def merge_dict(a: Dict, b: Dict):
    # combine
    # {'key1' : value1, 'key2' : value2} and {'key1' : value3, 'key2' : value4}
    # to {'key1' : [value1, value3], 'key2' : [value2, value4]}
    if len(a.keys()) != len(b.keys()) and len(a.keys()) != 0:
        logging.error("{} {}".format(len(a.keys()), len(b.keys())))
        raise Exception("size of two dict do not match:\n{}\n{}".format(a, b))

    for k, v in b.items():
        if k not in a:
            a[k] = []
        a[k].append(v)
