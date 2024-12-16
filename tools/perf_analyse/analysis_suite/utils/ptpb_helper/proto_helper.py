"""
    generate mluops_test_pb2.py with mlu_op_test_proto
"""

__all__ = (
    "check_mluops_proto",
)

import os
import subprocess
import sys
import argparse
import logging

def check_mluops_proto():
    # generate mluops_test_pb2.py
    # 生成在 analysis_suite.details/cfg/ 目录下
    proto_dir = os.path.abspath(os.path.realpath(__file__) + "/../../../../../../test/mlu_op_gtest/pb_gtest/mlu_op_test_proto")
    cwd = os.path.abspath(os.path.realpath(__file__) + "/../../../cfg/")

    cmd_args = ["protoc", "--python_out", cwd, "--proto_path", proto_dir, proto_dir + "/mlu_op_test.proto"]
    try:
        # check returncode
        cmd_ret = subprocess.run(cmd_args, check=True)
    except Exception:
        logging.error("run {} failed, please check!".format(" ".join(cmd_args)))
        raise
