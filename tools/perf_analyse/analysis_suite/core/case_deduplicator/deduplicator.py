"""
    Deduplicator for case_list/case_path
    Support .prototxt file only
"""

import os
import sys
import logging
import hashlib
from typing import List, Optional, Dict
from collections import defaultdict
import json
from tqdm import tqdm

from analysis_suite.utils import logger_helper, path_helper
from analysis_suite.core.case_deduplicator import deduplicator_mr

def find_pt_files(
        start_abspath: str,
        op_list: List
    ) -> List[Optional[str]]:
    case_list = []

    select_all_op = True # True for all ops; False for selected ops
    if 'all' not in op_list:
        select_all_op = False

    if True == select_all_op:
        for foldername in os.listdir(start_abspath):
            folder_path = os.path.join(start_abspath, foldername)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if not filename.endswith((".prototxt", ".pb")):
                        continue
                    case = os.path.join(folder_path, filename)
                    case_list.append(case)
    else: # False == select_all_op
        entries = os.listdir(start_abspath)
        for entry in entries:
            if entry not in op_list:
                continue
            absentry = os.path.join(start_abspath, entry)
            if not os.path.isdir(absentry):
                continue
            for file in os.listdir(absentry):
                absfile = os.path.join(absentry, file)
                if os.path.isfile(absfile) and absfile.endswith((".prototxt", ".pb")):
                    case_list.append(absfile)

    return case_list

def parse_src_case_dir(
        src_case_dir: str,
        **kwargs,
    ) -> List[Optional[str]]:
    try:
        start_abspath = path_helper.check_dir(src_case_dir)
    except Exception:
        raise

    ops = kwargs['ops']
    op_list = [op.strip() for op in ops.split(';')]

    case_list = find_pt_files(start_abspath, op_list)

    logging.info(f"finish parsing case directory, the number of cases is {len(case_list)}")
    return case_list

def parse_src_case_list(
        src_case_list: str,
    ) -> List[Optional[str]]:
    case_list = []

    try:
        src_case_list_abspath =  path_helper.check_file(src_case_list)
    except Exception:
        raise

    try:
        with open(src_case_list_abspath, 'r', encoding='utf-8') as f:
            for line in f:
                case_list.append(line.strip())
    except Exception:
        raise

    logging.info(f"finish parsing case list, the number of case is {len(case_list)}")
    return case_list

def get_case_list(
        src_case_dir: Optional[str],
        src_case_list: Optional[str],
        **kwargs,
    ) -> List[Optional[str]]:
    case_list = []

    if src_case_dir is not None:
        case_list = parse_src_case_dir(
                src_case_dir    = src_case_dir,
                ops             = kwargs['ops']
            )
    elif src_case_list is not None:
        case_list = parse_src_case_list(src_case_list)
    else:
        raise Exception("Input Error: Please specify src_case_dir or src_case_list.")

    return case_list

@logger_helper.log_debug
def remove_duplicated_cases(
        case_list: List[str],
    ) -> Dict[str, int]:
    # md5 -> case
    md5_to_case = {}
    # md5 -> repeat num
    md5_to_count = defaultdict(int)

    for case_path in case_list:
        # calculate md5
        md5_val = deduplicator_mr.cal_md5(case_path)
        # update case path
        md5_to_case[md5_val] = case_path
        # update repeat num
        md5_to_count[md5_val] += 1

    # for output: case -> repeat num
    case_to_count = {}
    for md5_val, count in md5_to_count.items():
        case_path = md5_to_case[md5_val]
        case_to_count[case_path] = count

    return case_to_count

def export_case_count(
        case_to_count: Dict[str, int],
        dst_case_list: str,
    ):
    # Dict to List[Dict]
    json_list = []
    for case_path, repeat_num in case_to_count.items():
        tmp_dict = {"case_path": case_path, "repeat_num": repeat_num}
        json_list.append(tmp_dict)

    # dump to json file
    try:
        with open(dst_case_list, 'w') as f:
            json.dump(json_list, f, indent=2)
    except TypeError as e:
        raise("error in serialization: {e}")
    except IOError as e:
        raise("error in operating file: {e}")
    except Exception:
        raise
    logging.info(f"write case-count to {dst_case_list}")

@logger_helper.log_info
def run(args):
    # input
    try:
        # handle input cases_path/cases_list
        case_list = get_case_list(
            src_case_dir    = args.src_case_dir,
            src_case_list   = args.src_case_list,
            ops             = args.ops,
        )
    except Exception:
        raise

    # process
    case_count_map = {}
    try:
        # generate dict from case to repeat_num
        if 1 >= args.cpu_count:
            case_count_map = remove_duplicated_cases(case_list)
        else:
            case_count_map = deduplicator_mr.run_deduplicator_mr(case_list, num_mappers=args.cpu_count)
    except Exception:
        raise

    # output
    try:
        export_case_count(case_count_map, args.dst_case_list)
    except Exception:
        raise

