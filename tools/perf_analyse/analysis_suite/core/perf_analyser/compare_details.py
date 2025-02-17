"""
    general interface for all compare functions
"""

__all__ = (
    "compare",
)

import sys
import logging
import pandas as pd
import re
from pandas.errors import MergeError

from analysis_suite.cfg.config import Config

def compare(
    df_new, df_bl,
    merge_key,
    info_columns,
    perf_columns,
    promotion_columns,
    sort_key=None
):
    logging.debug("{} start".format(sys._getframe().f_code.co_name))

    # info_columns can not be hashable
    perf_columns_ = perf_columns + [merge_key]
    try:
        df_compare = \
            pd.merge(
                df_new[perf_columns_],
                df_bl[perf_columns_],
                on=merge_key,
                suffixes=Config.suffix,
                validate='one_to_one' # check if merge keys are unique in both left and right datasets
            )
    except MergeError as e:
        logging.warn(e)
        logging.warn("Begin to drop reduplicate rows on key: {}.".format(merge_key))
        df_new.drop_duplicates(subset=merge_key, inplace=True)
        df_bl.drop_duplicates(subset=merge_key, inplace=True)
        df_compare = \
            pd.merge(
                df_new[perf_columns_],
                df_bl[perf_columns_],
                on=merge_key,
                suffixes=Config.suffix,
                validate='one_to_one'
            )

    # for some circumstance, such as small cases ignorance or original cases diffrence,
    # df_new and df_bl operators number will be different; so we preserve index first.
    if merge_key not in info_columns:
        info_columns = info_columns + [merge_key]
    # add info columns
    df_compare = \
        pd.merge(
            df_new[info_columns],
            df_compare,
            on=merge_key
        )

    # compute promotion
    #   promotion = base - new
    #   promotion_ratio = promotion / base
    for column in promotion_columns:
        df_compare[column + Config.promotion_suffix[0]] = \
            df_compare[column + Config.suffix[1]] - \
            df_compare[column + Config.suffix[0]]
        df_compare[column + Config.promotion_suffix[1]] = \
            df_compare[column + Config.promotion_suffix[0]] / \
            df_compare[column + Config.suffix[1]]

    # sort table
    if sort_key != None:
        df_compare = \
            df_compare.sort_values(
                by=sort_key,
                ascending=False
            )

    logging.debug("{} end".format(sys._getframe().f_code.co_name))
    return df_compare

# this code is very hot
# please make sure that:
#   `merge_key` is in `info_columns`
#   `merge_key` is in `perf_columns`
def compare_fast(
    df_new, df_bl,
    merge_key,
    info_columns,
    perf_columns,
    promotion_columns
):
    df_compare = \
        pd.merge(
            df_new[perf_columns],
            df_bl[perf_columns],
            on=merge_key,
            suffixes=Config.suffix
        )

    df_compare = \
        pd.merge(
            df_new[info_columns],
            df_compare,
            on=merge_key
        )

    # compute promotion
    for column in promotion_columns:
        df_compare[column + Config.promotion_suffix[0]] = \
            df_compare[column + Config.suffix[0]] - \
            df_compare[column + Config.suffix[1]]
        df_compare[column + Config.promotion_suffix[1]] = \
            df_compare[column + Config.promotion_suffix[0]] / \
            df_compare[column + Config.suffix[1]]

    return df_compare