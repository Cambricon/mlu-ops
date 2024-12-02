"""
    dump dataframe to xlsx file
"""

__all__ = (
    "to_excel_helper",
    "dfs_to_excel_impl",
)

import sys
import logging
import pandas as pd
from typing import List, Optional

EXCEL_ROW_NUM_LIMIT = 2 ** 20
COL_MIN_WIDTH = 12
COL_MAX_WIDTH = 50

# get max width of a pandas.Series, to adjust the width of current column
def get_max_width(
        series: pd.Series,
        col_name: str
    ) -> int:
    # randomly select at most 11 cells
    if 11 < len(series):
        # avoid nan: use fillna('-')
        str_list = series.sample(n=11, random_state=1).astype(str).fillna('-').to_list()
    else:
        str_list = series.astype(str).fillna('-').to_list()

    # add header
    str_list = str_list + [col_name]
    # set the minimum width of the column
    len_list = [COL_MIN_WIDTH]
    # iterate over the selected elemnets
    for ele in str_list:
        # convert from string to list
        ele_split = list(ele)

        # get width of current element
        width = 0
        for c in ele_split:
            if ord(c) <= 256:
                width += 1
            else:
                # handle chinese characters
                width += 2
        len_list.append(width + 2)
    max_width = max(len_list)

    return min(max_width, COL_MAX_WIDTH)

# helper function for beautifying excel sheet
def auto_format(
        df: pd.DataFrame,
        writer,
        sheet_name: str,
        float_to_percentage_cols: List[str]=None
    ):
    wb = writer.book
    ws = writer.sheets[sheet_name]
    fmt = wb.add_format({'align': 'left'})
    col_list = df.columns
    for idx in range(len(col_list)):
        col_name = col_list[idx]
        letter = chr(idx + 65)
        width = get_max_width(df[col_name], col_name)
        if (float_to_percentage_cols is not None) and (col_name in float_to_percentage_cols):
            fmt = wb.add_format({'align': 'left', 'num_format': '0.00%'})
        ws.set_column(
                first_col=idx,
                last_col=idx,
                width=width,
                cell_format=fmt
            )

def merge_cells(
        df: pd.DataFrame,
        writer,
        sheet_name: str,
        merge_cell_col_idx_lst: Optional[List[int]]=None
    ):
    # if there's no column to merge, return
    if merge_cell_col_idx_lst is None:
        return

    ws = writer.sheets[sheet_name]
    group_key = df.columns[0]
    for group_name, group_df in df.groupby(group_key):
        if len(group_df) <= 1:
            continue
        # please make sure that all rows with the same group_key are adjacent
        start_row = group_df.index[0]
        end_row = start_row + len(group_df)
        # merge cells
        for idx in merge_cell_col_idx_lst:
            ws.merge_range(
                    first_row=start_row + 1,
                    last_row=end_row,
                    first_col=idx,
                    last_col=idx,
                    data=group_df.loc[start_row, df.columns[idx]]
                )

# dump dataframe to sheet
def to_sheet_helper(
        df: pd.DataFrame,
        writer: pd.io.excel._xlsxwriter.XlsxWriter,
        sheet_name: str,
        index: bool=False,
        float_to_percentage_cols: Optional[List[str]]=None,
        merge_cell_col_idx_lst: Optional[List[int]]=None
    ):
    df_len = len(df)
    num_parts = (df_len + EXCEL_ROW_NUM_LIMIT - 1) // EXCEL_ROW_NUM_LIMIT

    for j in range(num_parts):
        start_row = j * EXCEL_ROW_NUM_LIMIT
        end_row = min((j+1) * EXCEL_ROW_NUM_LIMIT, df_len)

        df_part = df[start_row:end_row]
        if 0 == j:
            sheet_name = "{}".format(sheet_name)
        else:
            sheet_name = "{}({})".format(sheet_name, j)

        df_part.to_excel(writer, sheet_name=sheet_name, index=index)

        # merge cells
        merge_cells(df_part, writer, sheet_name, merge_cell_col_idx_lst)

        # set width of cells
        auto_format(df_part, writer, sheet_name, float_to_percentage_cols)

def to_excel_helper(df, xlsx_path):
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
        to_sheet_helper(df=df, writer=writer, sheet_name="sheet1")

def dfs_to_excel_impl(dfs, sheet_names, xlsx_path, float_to_percentage_cols=None):
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
        for i in range(len(dfs)):
            if float_to_percentage_cols is None:
                to_sheet_helper(df=dfs[i], writer=writer, sheet_name=sheet_names[i])
            else:
                to_sheet_helper(df=dfs[i], writer=writer, sheet_name=sheet_names[i], float_to_percentage_cols=float_to_percentage_cols[i])
