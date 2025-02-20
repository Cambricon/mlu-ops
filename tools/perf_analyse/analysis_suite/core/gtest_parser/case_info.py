"""
    get information of cases from .prototxt/.pb files
"""

__all__ = (
    "append_case_info",
)

import sys
import logging
import tqdm
import pandas as pd
import hashlib
import json
from typing import List, Optional
import google.protobuf.text_format as text_format
from multiprocessing import Pool

from analysis_suite.cfg import mlu_op_test_pb2
from analysis_suite.core.gtest_parser.protobuf_case_parser_impl_inputoutput import ProtobufCaseParserImplInputOutput
from analysis_suite.cfg.config import Config, ColDef, DBConfig
from analysis_suite.core.gtest_parser import parser, test_info
from analysis_suite.utils import builtin_types_helper

def read_case_list_from_db(columns: List[str]) -> pd.DataFrame:
    from analysis_suite.database import db_op

    # create engine
    engine = db_op.create_engine(DBConfig.DB_Name.training_solution)

    # read `case_list` from database
    CASE_LIST_QUERY = """
        SELECT {}
        FROM {}
    """.format(
        ", ".join(columns),
        DBConfig.Table_Name_mp[DBConfig.Table_Name.case_list]
    )
    case_info = pd.read_sql_query(CASE_LIST_QUERY, con=engine)
    return case_info

def get_node_info(path: str, node):
    # __init__
    pbParser = ProtobufCaseParserImplInputOutput(mlu_op_test_pb2)
    # __call__
    params = pbParser(node)
    # assure stability
    magic_str = json.dumps(params, sort_keys=True)
    params['file_path'] = path
    params['md5'] = hashlib.md5(magic_str.encode("utf-8")).hexdigest()
    return params

def resolve_case(path: str):
    # TODO(tanghandi): do not parse values of the input and output
    node = mlu_op_test_pb2.Node()
    if path.endswith(".prototxt"):
        # when op is not in mluops, we will return none instad of case_info
        with open(path,'r') as f:
            first_line = f.readline()
            second_line = f.readline()
        if second_line.startswith("op_type:"):
            op_type = second_line.split(":")[1].strip()
            if op_type not in mlu_op_test_pb2.OpType.keys():
                logging.info("{} is not in mluops op_type list.".format(op_type))
                return None
        # parse
        with open(path) as f:
            text_format.Parse(f.read(), node)
    elif path.endswith(".pb"):
        with open(path, "rb") as f:
            node.ParseFromString(f.read())
    else:
        raise Exception("Unsupported file type, please check your input.")

    return get_node_info(path, node)

def parse_cases_info(paths: List[Optional[str]], cpu_count: int):
    with Pool(cpu_count) as pool:
        nodes_info = list(
            tqdm.tqdm(
                pool.imap(resolve_case, paths, chunksize=10),
                total=len(paths),
                ncols=80
            )
        )

    result = {}
    for i in range(len(nodes_info)):
        builtin_types_helper.merge_dict(result, nodes_info[i])

    return result

def append_case_info_impl_with_db(
        info_lst: List[Optional[test_info.TestInfo]],
        cpu_count: int
    ):
    """
    Read case information from database and append it to performance data.
    Maintain a dataframe that stores case information. The source of case information is:
        1.  The database;
        2.  If the information in databse is missing, parse the pt/pb files directly.

    Parameters & Returns:
        See function `append_case_info`.

    For each dataframe(`info`) in `info_lst`, there're 4 tables:
        1.  `case_info_cache`: Maintain all case information read into memory,
          which may be from database or pt/pb files.
        2.  `info.perf`: Performance data in `info`.
        3.  `missing_cases_*`: The rows in table: `info.perf` - `case_info_cache`.
            a. `missing_cases_idx`: reserve index for info.perf so that
              the information of missing cases can be updated based on the indices.
            b. `missing_cases_info`: case information read from pt/pb files.

    The STEPs is as follow:
        1.  Read `case_info_cache` from database.
        2.  `info.perf` LEFT JOIN `case_info_cache` to append existing case information to `info.perf`.
        3.  Find the missing cases and save to `missing_cases_idx`.
        4.  Read information of missing cases from pt/pb files.
        5.  Add information of missing cases to `info.perf` and update `case_info_cache`.
    """
    # STEP 1
    # field names required in case info
    # columns = ['input', 'output', 'params', 'protoName']
    columns = Config.case_info_keys + ['protoName']
    # read case info from database
    # `case_info_cache` will be used to maintain case info later
    case_info_cache = read_case_list_from_db(columns)

    for info in info_lst:
        # STEP 2
        # left join to append case info to the performance data
        info.perf = pd.merge(info.perf, case_info_cache, on=['protoName'], how='left', indicator=True)

        # STEP 3
        # for current table `info.perf`, find missing cases of database
        # columns in `missing_cases_idx`: ['protoName', 'file_path']
        missing_cases_idx = info.perf[info.perf['_merge'] == 'left_only'][['protoName', 'file_path']]
        # reserve index for origin table
        # after this, columns in `missing_cases_idx`: ['index', 'protoName', 'file_path']
        missing_cases_idx = missing_cases_idx.reset_index(drop=False)

        # drop column '_merge'
        info.perf = info.perf.drop(columns=['_merge'])

        '''
        # for debugging
        from analysis_suite.utils import excel_helper
        excel_helper.to_excel_helper(missing_cases_idx, "missing_cases" + str(id(info)) + ".xlsx")
        '''

        # there are some cases missing in the database, parsing .prototxt/.pb files directly.
        if 0 != len(missing_cases_idx):
            logging.warn("Some case not in database: Parse {} .proto/.pb files directly.".format(len(missing_cases_idx)))

            # STEP 4
            # read pt/pb files by 'file_path' in `missing_cases_idx`
            # after this, columns in `missing_cases_info`: ['input', 'output', 'params', 'file_path', 'md5']
            try:
                missing_cases_info = pd.DataFrame(parse_cases_info(missing_cases_idx['file_path'].to_list(), cpu_count))
            except Exception:
                raise
            # adding columns for `missing_cases_info`
            # get 'protoName' from 'file_path'
            missing_cases_info['protoName'] = missing_cases_info['file_path'].apply(lambda x: x.split("/")[-1])
            # inner join to adding index
            # after this, columns in `missing_cases_info`: ['input', 'output', 'params', 'file_path', 'md5', 'protoName', 'index']
            missing_cases_info = pd.merge(missing_cases_info, missing_cases_idx[['index', 'protoName']], on=['protoName'])

            # STEP 5
            # add information to `perf.info`
            for i in range(len(missing_cases_info)):
                row = missing_cases_info.iloc[i]
                idx = row['index']
                for k in Config.case_info_keys:
                    info.perf.at[idx, k] = str(row[k])

            # update `case_info_cache` to read each file once
            case_info_cache = pd.concat([case_info_cache, missing_cases_info[columns]])

# read all cases information by parsing pt/pb files
def append_case_info_impl_no_db(info_lst: List[Optional[test_info.TestInfo]], cpu_count: int):
    for info in info_lst:
        cases_info = pd.DataFrame(parse_cases_info(info.perf['file_path'].to_list(), cpu_count))
        # md5 is appended
        for k in cases_info.keys():
            info.perf[k] = cases_info[k]

def append_case_info(info_lst: List[Optional[test_info.TestInfo]], cpu_count: int, use_db: bool):
    """
    Append case information to the performance data.

    case information: [
        'input', 'output', 'params'
    ]

    Paramters:
        info_lst: The list of test_info.TestInfo. only thge performance data will be used.
        cpu_count: CPU number to use when parsing pt/pb files
        use_db: Whether to read case information from database.
    Returns:
        No return. Modify the data in parameter 'info_lst' directly.
    """
    if use_db:
        append_case_info_impl_with_db(info_lst, cpu_count)
    else:
        append_case_info_impl_no_db(info_lst, cpu_count)
