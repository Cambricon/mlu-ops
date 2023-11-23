#!/usr/bin/env python3

"""
在数据库中搜索含真实数据的测例
根据pb实际文件大小、规模信息、vmpeak等约束过滤出满足内存限制的测例
最终输出文件列表

支持的过滤条件：
- MLU硬件架构
- 算子列表
- 预定的运行CPU内存占用大小
- 测例的文件大小
- mode = all, 返回约束条件下（据pb实际文件大小、规模信息、vmpeak等），所有算子的测例
- mode = op, 返回约束条件下（据pb实际文件大小、规模信息、vmpeak等），指定算子的全部用例和其他算子部分测例（可通过limit_num按个数输出，默认100，也可以通过limit_ratio按比例输出）
- mode = random, 返回约束条件下（据pb实际文件大小、规模信息、vmpeak等）, 所有算子部分测例（可通过limit_num按个数输出，默认100，也可以通过limit_ratio按比例输出）
- 测例所在目录(支持release_temp和release_test)
- 测例所在分支(master和其他r分支)
- 测例为功能测例或性能测例
- 测例所在网络
"""

WIKI = "http://wiki.cambricon.com/pages/viewpage.action?pageId=80539039"

import os
import sys
import re
import argparse
import enum
import logging
import functools
from pathlib import Path
from itertools import chain
import textwrap
from typing import Optional, Generator, Tuple

try:
    import pymysql
except ImportError:
    print("Python module import failed: `pymysql` should be installed")
    print('''You need to run: \033[1;33mpip3 install --user --trusted-host mirrors.cambricon.com -i http://mirrors.cambricon.com/pypi/web/simple pymysql\033[0m''')
    print("More details on wiki:", WIKI)
    sys.exit(-1)


logging.basicConfig(format='%(levelname)s:%(name)s [%(process)d-%(thread)d] [%(funcName)s:%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)


# NOTE MUST BE THE SAME WITH repo `neuware/software/test/mluoptest`
# mluoptest/mluoptest_database/mlu_op_to_sql_v2.py has the same enum
@enum.unique
class DirectoryType(enum.IntEnum):
    # 0 for RELEASE_TEMP and RELEASE_TEST
    RELEASE_TEMP=1
    RELEASE_TEST=2
    BENCHMARK=3
    # you can add new enum, but never modify existed value
CasePathMapping = {
    "/SOFT_TRAIN/release_temp": DirectoryType.RELEASE_TEMP,
    "/SOFT_TRAIN/release_test": DirectoryType.RELEASE_TEST,
    "/SOFT_TRAIN/benchmark": DirectoryType.BENCHMARK,
}

Inverse_CasePathMapping = {v.value: k for k,v in CasePathMapping.items()}

CARD_TYPE_ALIAS_V0 = {
    'MLU270-X5K': 'valid',
    'MLU270': 'valid',
    'MLU290-M5': 'MLU290',
    'MLU290': 'MLU290',
    'MLU370-S4': 'MLU370',
    'MLU370': 'MLU370',
    'MLU370-X4':'MLU370',
    'MLU370-X8':'MLU370',
}

CARD_TYPE_ALIAS_V1 = CARD_TYPE_ALIAS_V0.copy()
CARD_TYPE_ALIAS_V1.update({
    'MLU270-X5K': 'MLU270_X5K',
    'MLU270': 'MLU270_X5K',
    'MLU590': 'MLU590',
    'MLU590-H8': 'MLU590',
    'MLU590-M9': 'MLU590',
})

class MyFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass
help_text_example = textwrap.dedent("""
Example:
    --card is required, --mode is `random` by default, and other parameters are optional.

    # mode = all, search cases for all operators (except for inference operators) from master branch under the following constraints.
    # --mode is required and others such as --card, --testset, --memory are optional.
    `python3 search_case_v2.py --mode all --card MLU590 --testset 2 --memory 1638 --specific-filter-out-group inference`
    # seach cases for operators on SD5223
    `python3 search_case_v2.py --mode all --card SD5223 --testset 2 --memory 1638 --specific-filter-out-group 5223`

    # mode = id, search cases according to case_id reemainder. --circle n-i means that all the cases are tested in n times and the current is the i-th time.
    # --mode and --circle are required and others such as --card, --testset, --memory are optional. The following examples test all cases under the constraints (card, directory and memory) in 3 times:
    1-th: `python3 search_case_v2.py --mode id --circle 3-0 --card MLU590 --testset 2 --memory 1638` //case_id%3 == 0
    2-th: `python3 search_case_v2.py --mode id --circle 3-1 --card MLU590 --testset 2 --memory 1638` //case_id%3 == 1
    3-th: `python3 search_case_v2.py --mode id --circle 3-2 --card MLU590 --testset 2 --memory 1638` //case_id%3 == 2

    # mode = op, search 200 cases (or 80% cases) for 'abs' and 'scale' operators from r1.3 branch, and 50 cases (or 20% cases) for other operators randomly under the following constraints.
    # --mode and --ops are required and others such as --card, --limit_num, --limit_ratio, --testset, --memory are optional, where --limit_num defaults to 100.
    `python3 search_case_v2.py --mode op --ops abs,scale --ops_limit_num 200 --limit_num 50 --card MLU270-X5K --testset 1 --exceeding_memory 4096 --branch r1.3`
    `python3 search_case_v2.py --mode op --ops abs,scale --ops_limit_ratio 0.8 --limit_ratio 0.2 --card MLU270-X5K --testset 1 --memory 4096 --branch r1.3`

    # mode = random, search cases for all operators under the following constraints, limit each operator with 120 cases (or 20% cases)  and use r1.5 branch.
    # --mode and --ops are required and others such as --card, --limit_num, --testset, --memory are optional, where --limit_num defaults to 100.
    `python3 search_case_v2.py --mode random --limit_num 120 --card MLU370 --testset 2 --memory 4096 --exceeding_memory 1024 --branch r1.5`
    `python3 search_case_v2.py --mode random --limit_ratio 0.2 --card MLU370 --testset 2 --memory 4096 --branch r1.5`

    # search benchmark cases.
    # search benchmark cases used to test performance. --case-type performance is required.
    # search benchmark cases used to test function. --case-type precision --database training_solution is required.
    `python3 search_case_v2.py --case-type performance --card MLU370 --mode all`
    `python3 search_case_v2.py --case-type precision --database training_solution --card MLU370 --mode all`

    # search benchmark cases according to network, add optional network parameters such as --network-name, --network-framework, --network-mode,--network-batchsize, --network-version, --network-additional-information
    `python3 search_case_v2.py --case-type performance --card MLU370 --mode all --network-name 2dUnet --network-framework mm`
    `python3 search_case_v2.py --case-type precision --database training_solution --card MLU370 --mode all --network-name 2dUnet --network-framework mm`

card suppurt type (MLU220 here only represents MLU220-M.2):
MLU220、MLU220-EDGE、MLU270-X5K、MLU270、MLU290-M5、MLU290、MLU370-S4、MLU370、MLU370-X4、MLU370-X8、MLU365-D2、MLU270-F4、MLU590、SD5223

More details on wiki: {WIKI}
""".format(WIKI=WIKI))
# We will get default value from env
parser=argparse.ArgumentParser(
    prog="search_case_v2",
    description="select cases from search_case database",
    formatter_class=MyFormatter,
    epilog=help_text_example,
)
parser.add_argument("--proto-max-size", type=int, default=0, help="max proto file size in MB, use 0 to disable this filter")
parser.add_argument("--proto-min-size", type=int, default=0, help="min proto file size in MB")
parser.add_argument("-d", "--dump", type=str, default=os.getenv('CASE_OUTPUT_FILENAME', 'mluop_case_list.txt'), help="dump file name")
parser.add_argument("--database", type=str, default=os.getenv("DATABASE", dev"), help="database name to connect")
parser.add_argument("--table", type=str, default=os.getenv("TABLE"), help="search case table to be queried from")
parser.add_argument("--cvid", type=str, default=os.getenv("CVID"), help="search index table to be queried from")
parser.add_argument("--network_table", type=str, default=os.getenv("NETWORK_TABLE", network_list_test"), help="search case table to be queried from")
parser.add_argument("--card", type=str, default=os.getenv("CARD_TYPE", ""), help="mlu type to be tested", required=True)
parser.add_argument("--ops", type=str, nargs='?', const="", default="", help="comma split ops to be tested")
parser.add_argument("--testset", type=int, help="the location of the testset, use `--list_testset` to show all available location", default=0)
parser.add_argument("--list_testset", action="store_true", help="list all supported location type enum")
parser.add_argument('--mode', help="all, op, random, id", type=str, choices = ['all', 'op', 'random', 'id'], default="random")
parser.add_argument("--memory", type=int, default=int(os.getenv("MLUOP_GTEST_LIMIT_MEMORY", -1)), help="search the cases with memory less than limit in MB for mluop_gtest, -1 means no limit")
parser.add_argument("--exceeding_memory", type=int, default=-1, help="search the cases with memory larger than limit in MB for mluop_gtest, -1 means no limit.")
parser.add_argument("--branch", help="source branch to be tested", default="master", type=str)
parser.add_argument("--circle", help="The input is in the format of n-i, whcich means that all the cases are tested in n times and the current is the i-th time", default=None, type=str)
parser.add_argument("--specific-filter-out-group", type=str, help="skip specific group of cases, usually for edge devices which does not support fully training op under mluOp.", default=None)
parser.add_argument("--debug", help="print sql and verbose log for debugging search_case itself", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)
parser.add_argument("--case-type", type=str, help="precision or performance", default=None)
parser.add_argument("--network-name", type=str, help="network name", default=None)
parser.add_argument("--network-id", type=str, help="network id", default=None)
parser.add_argument("--network-framework", type=str, help="framework", default=None)
parser.add_argument("--network-mode", type=str, help="running precision, float or half", default=None)
parser.add_argument("--network-batchsize", type=int, help="batchsize", default=-1)
parser.add_argument("--network-version", type=str, help="version", default=None)
parser.add_argument("--network-additional-information", type=str, help="additional information", default=None)
parser.add_argument("--skip_cpu_cases", action="store_true",help="skip cpu cases, note that ops with only cpu cases will not be skipped")

group_out=parser.add_mutually_exclusive_group()
group_out.add_argument('--limit_ratio', help="max ratio limit for output", type=float, default=-1)
group_out.add_argument('--limit_num', help="max case number limit for output", type=int, default=100)

group_in=parser.add_mutually_exclusive_group()
group_in.add_argument('--ops_limit_ratio', help="max ratio limit for --ops", type=float, default=-1)
group_in.add_argument('--ops_limit_num', help="max case number limit for --ops", type=int, default=-1)


args = parser.parse_args()
args.card = args.card.upper()
logging.getLogger().setLevel(args.loglevel)
logging.debug(args)

_db_passwd='Hello@123'
_db_host='10.101.9.21'
_db_user='mluop_search'
_db_name=args.database
_db_table=args.table if args.table else case_information"
_db_cvid=""
_db_network_table=args.network_table

if args.case_type=="performance":
    _db_name="training_solution"

if args.branch=="master":
    _db_cvid=args.cvid if args.cvid else cvID_information"
elif re.match(r"\w\d+\.\d+", args.branch):
    _db_cvid=args.branch.replace(".", "dot")
else:
    _db_cvid=args.branch

if _db_name=="training_solution":
        _db_passwd='nj#Is9Gk'
        _db_host='10.101.9.21'
        _db_user='mluop_dev'
        _db_table=args.table if args.table else case_information_benchmark_test"
        _db_cvid=args.cvid if args.cvid else case_in_network_test"

conn = pymysql.connect(host=_db_host, user=_db_user, password=_db_passwd, database=_db_name)
cursor = conn.cursor()

class DBHandler:
    conn = None
    cursor = None
    schema_version = 0

    @classmethod
    def _init_db_(cls):
        # TODO move global var into this class
        global conn
        global cursor
        if cls.conn == cls.cursor == None:
            cls.conn = conn if conn else pymysql.connect(host=_db_host, user=_db_user, password=_db_passwd, database=_db_name)
            cls.cursor = cursor if cursor else cls.conn.cursor()

    @classmethod
    def find_table_version(cls, table_name:str):
        """Retrive table schema version

        search_case database will change schema, consider support both old and new table

        - version 0: original search_case
        - version 1: change `valid` to `MLU270_X5K`;
                     also change VARCHAR with value 'True'/'False' to BOOLEAN;
                     for fields like `input`, `output`, etc, change VARCHAR to JSON;
                     add field `directory` for directory filtering;
                     add `base_size`, etc for cpu memory usage filtering.
        """
        cls._init_db_()
        table_name=table_name
        sql = """SHOW COLUMNS FROM `{table_name}`;""".format(table_name=table_name)
        logging.debug(sql)
        cls.cursor.execute(sql)
        columns = cls.cursor.fetchall()
        col_meta = {v[0]: v for v in columns}
        if 'valid' in col_meta:
            return 0
        return 1

def get_available_ops():
    sql = """
    SELECT DISTINCT(op_name) FROM `{_db_table}`;
    """.format(_db_table=_db_table)
    logging.debug(sql)
    cursor.execute(sql)
    columns = cursor.fetchall()
    ops = sorted([v[0] for v in columns])
    logging.debug("awailable test op_name in database: {}".format(ops))
    return ops

def specific_filter_out(group_name: str="inference"):
    logging.info("Special extra search_case filter on {}".format(group_name))
    if group_name == "inference":
        ops = get_available_ops()
        exclude_op = set()
        for op in ops:
            if re.search(r'(adam|grad|loss|filter|training|weight(?!_norm))', op):
                exclude_op.add(op)
            if re.search(r'^(?!.*pool).*backward.*$', op):
                exclude_op.add(op)
            if re.search(r'logits', op):
                exclude_op.add(op)
        exclude_op.add('rnn')
        exclude_op.add('sync_batchnorm_stats')
        exclude_op.add('sync_batchnorm_gather_stats_with_counts')
        exclude_op.add('sync_batchnorm_elemt')
        exclude_op.add('deform_roi_pool_backward')
        return "AND op_name NOT IN ({exclude_ops})".format(
                exclude_ops=','.join(sorted(['"' + o + '"' for o in exclude_op]))
                )
    elif group_name.lower() in ("5223", "1v",):
        include_op = ["abs", "activation_forward", "adaptive_pooling_forward", "ax", "advanced_index", "atan2", "batch2space", "batch_matmul_bcast", "batch_to_space_nd", "cast", "clip", "concat", "crop_and_resize", \
        "convolution_forward", "copy", "cumsum", "deconvolution", "div", "divnonan", "exp", "expm1", "floor_mod", "floor_mod_trunc", "expand", "fill", "fusedOp", "gather_v2", "grep", "grid_sample_forward", \
        "index_put", "index_fill", "instancenorm_forward", "layernorm_forward", "interp", "logic_op", "lrn", "matmul", "maximum", "minimum", "nan_to_num", "neg", "op_tensor", \
        "pad", "pooling_backward", "pooling_forward", "pooling_forward_with_index", "prelu", "quantize", "quantize_param", "reorg", "scale", "scatter", "select", "shufflechannel", "sincos", \
        "softmax_forward", "softplus_forward", "space2batch", "space_to_batch_nd", "split", "std_forward", "strided_slice", "threshold", "tile", "transpose", "unpool_forward", "var_forward", "matmul_inference", "quantize_matmul", \
        "quantize_batch_matmul", "qr", "reduce", "floor", "floor_div", "floor_div_trunc", "groupnorm_forward", "log", "pow", "powr", "pown", "sqrt", "batchnorm_inference", "topk", \
        "cycle_op", "rsqrt", "add_n", "random_uniform", "random_normal", "nms", "addcmul", "addcdiv", "rnn_inference", "det", "std_forward", "var_forward", "std_var_mean", \
        "flip", "dynamic_stitch", "matrix_band_part", "embedding_bag", "softsigh_forward", "cummax", "cummin", "kth_value", \
        "repeat_interleave", "invert_permutation", "erf", "im2col", "cosine_similarity", \
        "sign", "gru", "roi_align_rotated", "is_nan", "tin_shift_backward", "inverse", "hardtanh", \
        "tri", "linspace", "arange", "as_strided", "normalize", "reflection_pad2d", "diag", "scatter_nd", "square", "gather_nd", "embedding_forward", "trigon", "bit_compute_v2", "where",
        "random_multinomial", "random_truncated_normal", "random_uniform_int", "maskzero", "fused_dropout", "masked", "is_finite", "squared_difference", "index_add", "exponential", "poisson", "dcn_forward",
        "roialign_forward", "round", "ceil", "assignadd", "assignsub", "biasadd", "adadelta", "angle", "applyftrlv2", "axpby", "axpy", "bincount", "box_overlap_bev", "bucketize", \
        "complex_abs", "cross", "diag_part", "diagonal", "dynamic_partition", "fake_quantize_per_channel_affine", "fake_quantize_per_tensor_affine", "gather", "is_inf", \
        "logaddexp", "logaddexp2", "matrix_diag", "mul_n", "one_hot", "orgqr", "pdist_forward", "polar", "searchsorted", "softsign_forward", "sorted_segment_reduce", "trace", \
        "transform", "tri_indices", "unpool_backward"]
        return "AND op_name IN ({include_ops})".format(
                include_ops=','.join(sorted(['"' + o + '"' for o in include_op]))
                )
    else:
        logging.error("At present, only support inference and 5223 (1v)!")
        sys.exit(-1)


def gen_mlu_filter(schema_version: int=0, card: str=""):
    if card is None or len(card) == 0:
        return ''

    card_type = ''
    try:
        card_type = globals()["CARD_TYPE_ALIAS_V{i}".format(i=schema_version)][card]
    except KeyError as e:
        print(e, "card type {card}, not supported".format(card=card))

    MLU_FILTER=''
    if card_type:
        MLU_FILTER="AND {card_type}={boolean}".format(
            card_type=card_type,
            boolean="'True'" if schema_version == 0 else True
        )
    logging.debug("MLU_FILTER: {MLU_FILTER}".format(MLU_FILTER=MLU_FILTER))

    return MLU_FILTER

def search_case(schema_version: int=0, ops="", card="MLU370", limit_num=0, limit_ratio: float=0, circle: str=None, mode='by_num', extra_filter="") -> Generator[Tuple, None, None]:
    MLU_FILTER = gen_mlu_filter(schema_version=schema_version, card=card)
    OPS_FILTER_IN=""
    OPS_FILTER_OUT=""
    if len(ops):
        ops = ops.replace(";", ",").split(",")
        op_lists = ",".join(["'{op}'".format(op=op) for op in ops])
        OPS_FILTER_IN="cases_info_tmp.op_name IN ({op_lists})".format(op_lists=op_lists)
        OPS_FILTER_OUT="cases_info_tmp.op_name NOT IN ({op_lists})".format(op_lists=op_lists)

    LIMIT_STR = ''
    JOIN_STR = ''
    if mode == 'random':
        LIMIT_STR = "AND rownum <= {limit_num}".format(limit_num=int(limit_num))
        if limit_ratio >= 0:
            JOIN_STR = """
                LEFT OUTER JOIN
                (
                    SELECT op_name, count(*) as op_name_cnt
                    FROM cases_info
                    GROUP BY op_name
                ) AS c
                ON cases_info.op_name=c.op_name
            """
            LIMIT_STR = "AND rownum <= GREATEST(op_name_cnt*{limit_ratio}, 50)".format(limit_num=int(limit_num), limit_ratio=limit_ratio)
    elif mode == 'op':
        # limit_num > 0
        LIMIT_STR = "AND ((rownum <= {limit_num} AND {OPS_FILTER_OUT}) OR ({OPS_FILTER_IN}))".format(
            limit_num=int(limit_num), OPS_FILTER_IN=OPS_FILTER_IN, OPS_FILTER_OUT=OPS_FILTER_OUT)
        if args.ops_limit_ratio >= 0:
            JOIN_STR = """
                LEFT OUTER JOIN
                (
                    SELECT op_name, count(*) as op_name_cnt
                    FROM cases_info
                    GROUP BY op_name
                ) AS c
                ON cases_info.op_name=c.op_name
            """
            LIMIT_STR = "AND ((rownum <= {limit_num} AND {OPS_FILTER_OUT}) OR ((rownum <= GREATEST(op_name_cnt*{ops_limit_ratio}, 50)) AND {OPS_FILTER_IN}))".format(
                limit_num=int(limit_num), ops_limit_ratio=args.ops_limit_ratio, limit_ratio=limit_ratio, OPS_FILTER_IN=OPS_FILTER_IN, OPS_FILTER_OUT=OPS_FILTER_OUT)
        elif args.ops_limit_num >= 0:
            LIMIT_STR = "AND ((rownum <= {limit_num} AND {OPS_FILTER_OUT}) OR ((rownum <= {ops_limit_num}) AND {OPS_FILTER_IN}))".format(
                limit_num=int(limit_num), ops_limit_num=args.ops_limit_num, OPS_FILTER_IN=OPS_FILTER_IN, OPS_FILTER_OUT=OPS_FILTER_OUT)
        # limit_ratio > 0
        if limit_ratio >= 0:
            JOIN_STR = """
                LEFT OUTER JOIN
                (
                    SELECT op_name, count(*) as op_name_cnt
                    FROM cases_info
                    GROUP BY op_name
                ) AS c
                ON cases_info.op_name=c.op_name
            """
            LIMIT_STR = "AND (((rownum <= GREATEST(op_name_cnt*{limit_ratio}, 50)) AND {OPS_FILTER_OUT}) OR ({OPS_FILTER_IN}))".format(
                limit_ratio=limit_ratio, OPS_FILTER_IN=OPS_FILTER_IN, OPS_FILTER_OUT=OPS_FILTER_OUT)
            if args.ops_limit_ratio >= 0:
                LIMIT_STR = "AND (((rownum <= GREATEST(op_name_cnt*{limit_ratio}, 50)) AND {OPS_FILTER_OUT}) OR ((rownum <= GREATEST(op_name_cnt*{ops_limit_ratio}, 50)) AND {OPS_FILTER_IN}))".format(
                    limit_ratio=limit_ratio, ops_limit_ratio=args.ops_limit_ratio, OPS_FILTER_IN=OPS_FILTER_IN, OPS_FILTER_OUT=OPS_FILTER_OUT)
    elif mode == 'all':
        JOIN_STR = ""
        LIMIT_STR = ''
    elif mode == 'id':
        circle_split = circle.split('-', 1)
        if (len(circle_split) == 2):
            logging.debug("All the cases are tested in {n} times and the current is the {i}-th time".format(n=circle_split[0], i=circle_split[1]))
            JOIN_STR = ""
            LIMIT_STR = "AND (case_id % {n} = {i})".format(n=circle_split[0], i=circle_split[1])
        else:
            logging.error("--circle parameter only supports n-i format, whcich means that all the cases are tested in n times and the current is the i-th time")
            sys.exit(-1)

    sql_query = """
        WITH cases_info AS
        (
            SELECT DISTINCT a.case_id as case_id, op_name, protoName, {file_size} {directory}
            FROM {_db_table} a, {_db_cvid} b
            WHERE  a.case_id=b.case_id
            {MLU_FILTER}
            {extra_filter}
        )
        SELECT case_id, op_name, protoName, {file_size} directory,{op_name_cnt} rownum FROM(
            SELECT case_id, cases_info.op_name, protoName, {file_size} directory,{op_name_cnt} ROW_NUMBER() OVER(partition by cases_info.op_name order by RAND()) rownum
        FROM cases_info
        {JOIN_STR}

            GROUP BY case_id, cases_info.op_name, protoName, {file_size} {op_name_cnt} directory
        ) as cases_info_tmp
        WHERE true {LIMIT_STR}

        ORDER BY RAND()
        ;
    """.format(
        directory="directory" if schema_version else '"/SOFT_TRAIN/release_temp" AS directory',
        file_size = "file_size," if schema_version else "",
        op_name_cnt = "op_name_cnt," if JOIN_STR != "" else "",
        _db_table=_db_table,
        _db_cvid=_db_cvid,
        MLU_FILTER=MLU_FILTER,
        LIMIT_STR=LIMIT_STR,
        JOIN_STR=JOIN_STR,
        extra_filter=extra_filter,
    )
    logging.debug(sql_query)
    cursor.execute(sql_query)
    while True:
        values = cursor.fetchmany(size=2000)
        if len(values) == 0:
            break
        yield from values

def filter_location_type(schema_version: int=0, location_type: int=0) -> str:
    if schema_version < 1:
        logging.error("old search_case database schema does not support location type")
        return ""
    if not location_type:
        return """
        AND directory in (1,2)
        """
    return """
        AND directory = {location_type}
    """.format(location_type=location_type)

def get_case_by_type(case_type: str=""):
    if case_type != "precision" and case_type != "performance" :
        logging.error("At present, case_type only include precision and performance!")
        sys.exit(-1)
    logging.info("Search {} case".format(case_type))
    sql = """
    SELECT DISTINCT(case_id) FROM `{_db_cvid}` WHERE network_id {type}= 1;
    """.format(_db_cvid=_db_cvid,
    type="" if case_type=='precision' else "!",
    )
    logging.debug(sql)
    cursor.execute(sql)
    columns = cursor.fetchall()
    cases_list = sorted([v[0] for v in columns])
    logging.debug("{case_type} cases_list: {cases_list}".format(case_type=case_type,cases_list=cases_list))
    return cases_list

def get_case_by_network(network_name: Optional[str] = None,
        network_id: Optional[str] = None,
        network_framework:Optional[str] = None,
        network_mode: Optional[str] = None,
        network_batchsize: Optional[int] = -1,
        network_version: Optional[str] = None,
        network_additional_information: Optional[str] = None
    ):
    sql = """
    SELECT DISTINCT(case_id) FROM `{_db_network_table}` a, `{_db_cvid}` b WHERE a.network_id=b.network_id
    """.format(_db_network_table=_db_network_table,
        _db_cvid=_db_cvid,
    )
    if network_name:
        network_name = network_name.split(",")
        network_name_lists = ",".join(["'{}'".format(item) for item in network_name])
        sql += """
        AND network_name IN ({})
    """.format(network_name_lists)
    if network_id:
        network_id = network_id.split(",")
        network_id_lists = ",".join(["'{}'".format(item) for item in network_id])
        sql += """
        AND a.network_id IN ({})
    """.format(network_id_lists)
    if network_framework:
        sql += """
        AND framework = '{}'
    """.format(network_framework)
    if network_mode:
        sql += """
        AND mode = '{}'
    """.format(network_mode)
    if network_batchsize > 0:
        sql += """
        AND batchsize = {}
    """.format(network_batchsize)
    if network_version:
        sql += """
        AND version = '{}'
    """.format(network_version)
    if network_additional_information:
        sql += """
        AND network_additional_information = '{}'
    """.format(network_additional_information)
    logging.debug(sql)
    cursor.execute(sql)
    columns = cursor.fetchall()
    cases_list = sorted([v[0] for v in columns])
    logging.debug("network-{network_name},network_id-{network_id},framework-{network_framework} cases_list: {cases_list}".format(network_name=network_name,network_id=network_id,network_framework=network_framework,cases_list=cases_list))
    return cases_list

def filter_case(case_type: Optional[str] = None,
        network_name: Optional[str] = None,
        network_id: Optional[str] = None,
        network_framework:Optional[str] = None,
        network_mode: Optional[str] = None,
        network_batchsize: Optional[int] = -1,
        network_version: Optional[str] = None,
        network_additional_information: Optional[str] = None,
):
    type_cases_list = None
    network_cases_list = None
    cases_list = []
    if case_type:
        type_cases_list = get_case_by_type(case_type)
    if  network_name or network_id or network_framework or network_mode or network_batchsize > 0 or network_version or network_additional_information:
        network_cases_list = get_case_by_network(network_name,network_id,network_framework,network_mode,network_batchsize,network_version,network_additional_information)
    if type_cases_list is not None and network_cases_list is not None :
        cases_list = list(set(type_cases_list)& set(network_cases_list))
    else:
        cases_list = type_cases_list if type_cases_list is not None  else network_cases_list
    sql = ''
    if len(cases_list) == 0:
        return ""
    elif len(cases_list) == 1:
        cases_list.append('default')
    return "AND a.case_id IN {}".format(tuple(cases_list))

def gen_case_list(schema_version: int = 0,
        dump: Optional[str] = None, ops: str = "", exceeding_memory:int= -1,
        memory: int = -1, proto_min_size: int = 0, proto_max_size: int = 0,
        limit_num: int = 100, limit_ratio: float=0, circle: str = None,
        mode: str = 'by_num', card: str = "MLU370", location_type: int = 0,
        specific_filter_out_group: Optional[str] = None,
        case_type: Optional[str] = None,
        network_name: Optional[str] = None,
        network_id: Optional[str] = None,
        network_framework:Optional[str] = None,
        network_mode: Optional[str] = None,
        network_batchsize: Optional[int] = None,
        network_version: Optional[str] = None,
        network_additional_information: Optional[str] = None,
        skip_cpu_cases: Optional[bool] = 0
    ):
    """Generate case lists

    :arg int schema_version: table version, for compatibility
    :arg str dump: file name to be generated to, None for stdout
    :arg str ops: comma separated kernel operators, empty for all cases
    :arg int memory: memory limit in MB for mluop_gtest, -1 for infinity
    :arg int proto_min_size: proto file min size limit in MB
    :arg int proto_max_size: proto file max size limit in MB, set 0 to disable
    :arg int limit_num: case limit number or ratio (related to `mode`)
    :arg enum mode: 'by_num' or 'by_ratio'
    :arg str card: card type
    :arg int location_type: type enum of location (1: release_temp, 2: release_test, 3: benchmark)
    :arg str case_type: case type (precision or performance)
    :arg str network_name: network name (general: precision case)
    :arg str network_id: network id (1: precision case)
    :arg str network_framework: network framework (general: precision case)
    :arg str network_mode: network running precision (float or half)
    :arg int network_batchsize: batchsize
    :arg int network_version: network version
    :arg str network_additional_information: network additional information
    """
    filedump = Path(dump).open(mode='w') if dump else sys.stdout
    extra_filter = ""

    if skip_cpu_cases:
        sql = "SELECT DISTINCT op_name FROM {table} WHERE op_name NOT IN (SELECT DISTINCT op_name FROM {table} WHERE device!='cpu');".format(table=_db_table)
        cursor.execute(sql)
        columns = cursor.fetchall()
        op_name_list = tuple(sorted([v[0] for v in columns]))
        print("These ops' cases are not skipped:",op_name_list)
        if len(op_name_list) > 1:
            extra_filter += "AND (op_name IN {cpu_op} or device != 'cpu')".format(cpu_op=op_name_list)
        else:
            extra_filter += "AND (op_name IN ('{cpu_op}') or device != 'cpu')".format(cpu_op=op_name_list[0])

    if case_type != "performance":
        extra_filter += filter_location_type(schema_version=schema_version,
            location_type=location_type,
        )

    no_cases = False

    if 0 <= proto_min_size <= proto_max_size and proto_max_size > 0:
        extra_filter += """
        AND file_size BETWEEN {proto_min_size} AND {proto_max_size}
        """.format(proto_min_size=proto_min_size, proto_max_size=proto_max_size)

    # TODO: NOTE: memory limit for v2 (which compile with only one mlu) and cloud (support all mlu) is different, we may need to set different base offset
    if memory >= 0:
        vm_peak_bytes = memory * 1024 * 1024
        # if vm_peak is null, use base_size
        # always check vm_peak if has data
        # if memory < 2048 MB, also check base_size
        # For v2, base offset is different from cloud (which support multiple mlu)
        extra_filter += """
        AND (
            ( vm_peak IS NULL AND base_size <= {memory} )
            OR
            ( {memory} > 2048 AND vm_peak <= {vm_peak_bytes} )
            OR
            ( {memory} <= 2048 AND GREATEST(base_size, vm_peak/1024/1024-750) <= {memory} )
        )""".format(vm_peak_bytes=vm_peak_bytes, memory=memory)

    if exceeding_memory >= 0:
        if ((memory >= exceeding_memory) or (memory<0)):
            vm_peak_exceeding_bytes = exceeding_memory * 1024 * 1024
            extra_filter += """
            AND (
                ( vm_peak IS NULL AND base_size >= {exceeding_memory} )
                OR
                ( {exceeding_memory} > 2048 AND vm_peak >= {vm_peak_exceeding_bytes} )
                OR
                ( {exceeding_memory} <= 2048 AND GREATEST(base_size, vm_peak/1024/1024-750) >= {exceeding_memory} )
            )""".format(vm_peak_exceeding_bytes=vm_peak_exceeding_bytes, exceeding_memory=exceeding_memory)
        else:
            logging.error("memory should be larger than exceeding_memory!")
            sys.exit(-1)

    if specific_filter_out_group:
        extra_filter += specific_filter_out(specific_filter_out_group)

    if  _db_name=="training_solution" and (case_type or network_name or network_id or network_framework or network_mode or network_batchsize > 0 or network_version or network_additional_information):
        case_filter = filter_case(case_type,network_name,network_id,network_framework,network_mode,network_batchsize,network_version,network_additional_information)
        if case_filter == "":
            no_cases = True
        else: extra_filter += case_filter

    if no_cases:
            cases = []
    else: cases = search_case(schema_version=schema_version, ops=ops, card=card,
                        limit_num=limit_num, limit_ratio=limit_ratio, circle=circle, mode=mode, extra_filter=extra_filter)

    case_cnt = 0
    for case in cases:
        case_cnt += 1
        #if case_cnt <= 10: print(case) # for debug purpose
        if len(case) == 6:
            case_id, op_name, protoName, file_size, directory, rownum = case
        else:
            case_id, op_name, protoName, file_size, directory, rownum, _ = case
        real_path = Path(Inverse_CasePathMapping[directory]).joinpath(op_name).joinpath(protoName)
        print(str(real_path), file=filedump)
    logging.info("A total of {case_cnt} cases were searched and stored in {dump}".format(case_cnt=case_cnt, dump=dump))

def print_supoorted_localtion_type():
    text = """
    Supported location type enum:
        0: search in all supported directory (just ignore this field)
    """
    for k, v in CasePathMapping.items():
        text += """    {v_value}: search cases under '{k}`
    """.format(v_value=v.value, k=k)
    logging.info(text)

def validate_location_type(location_type: int) -> bool:
    logging.debug("location_type is : {location_type}".format(location_type=location_type))
    if location_type == 0:
        # search in all directory in one query (just ignore the existence of this field)
        return True
    if location_type not in [item.value for item in list(DirectoryType)]:
        logging.error("invalid location type enum {location_type}".format(location_type=location_type))
        return False
    return True

def validate_card_type(card_type: int, schema_version) -> bool:
    logging.debug("card_type: {card_type}".format(card_type=card_type))
    if card_type=="":
        return True

    if schema_version:
        if card_type not in  CARD_TYPE_ALIAS_V1.keys():
            logging.error("invalid card type: {card_type}".format(card_type=card_type))
            return False
        else:
            return True
    else:
        if card_type not in CARD_TYPE_ALIAS_V0.keys():
            logging.error("invalid card type: {card_type}".format(card_type=card_type))
            return False
        else:
            return True

def validate_branch(branch: str) -> bool:
    logging.debug("branch is: {branch}".format(branch=branch))

    branch_list=["master"]
    cursor.execute("SHOW TABLES;")
    columns = cursor.fetchall()

    for column in columns:
        branch_list.append(column[0])

    if branch not in [branchs for branchs in branch_list]:
        logging.error("invalid branch is: {branch}".format(branch=args.branch))
        return False
    else:
        return True

def validate_table(table: str) -> bool:
    logging.debug("table is: {table}".format(table=table))

    table_list=[]
    cursor.execute("SHOW TABLES;")
    columns = cursor.fetchall()

    for column in columns:
        table_list.append(column[0])

    if table not in [tables for tables in table_list]:
        logging.error("invalid table is: {table}".format(table=table))
        return False
    else:
        return True


def validate_ops() -> bool:
    if args.ops=="" and args.mode=="op":
        logging.error("--ops can't be empty")
        return False
    else:
        return True

if args.list_testset:
    print_supoorted_localtion_type()
    sys.exit(0)

if not validate_location_type(args.testset):
    print_supoorted_localtion_type()
    sys.exit(-1)

if not validate_branch(_db_cvid) or not validate_ops():
    sys.exit(-1)

if __name__ == '__main__':
    mode = args.mode
    limit_num = args.limit_num

    schema_version = DBHandler().find_table_version(_db_table)
    logging.debug("table schema_version is {schema_version}".format(schema_version=schema_version))

    if not validate_table(_db_table) and not validate_table(_db_network_table) and  not validate_table(_db_cvid):
        sys.exit(-1)
    if not validate_card_type(args.card, schema_version):
        sys.exit(-1)

    gen_case_list(schema_version=schema_version,
            dump=args.dump, ops=args.ops,
            exceeding_memory=int(args.exceeding_memory),
            memory=int(args.memory),
            proto_min_size=int(args.proto_min_size),
            proto_max_size=int(args.proto_max_size),
            limit_num = limit_num,
            limit_ratio = args.limit_ratio,
            circle = args.circle,
            mode = mode,
            card = args.card,
            location_type=args.testset,
            specific_filter_out_group=args.specific_filter_out_group,
            case_type=args.case_type,
            network_name=args.network_name,
            network_id=args.network_id,
            network_framework=args.network_framework,
            network_mode=args.network_mode,
            network_batchsize=args.network_batchsize,
            network_version=args.network_version,
            network_additional_information=args.network_additional_information,
            skip_cpu_cases=args.skip_cpu_cases
    )
