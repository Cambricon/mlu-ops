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
    proto_dir = os.path.abspath(os.path.realpath(__file__) + "/../../../../../../test/mlu_op_gtest/pb_gtest/mlu_op_test_proto/")
    cwd = os.path.abspath(os.path.realpath(__file__) + "/../../../cfg/")

    cmd1_args = ["pushd", proto_dir, "bash", "build.sh", "popd"]
    cmd2_args = ["cp", proto_dir + "/mlu_op_test.proto", cwd]
    try:
        # change workspace to proto_dir
        original_dir = os.getcwd()
        os.chdir(proto_dir)

        # gen mlu_op_test.proto
        cmd1_args = ["bash", "build.sh"]
        cmd1_ret = subprocess.run(cmd1_args, check=True)

        # cp mlu_op_test.proto to cfg
        os.chdir(original_dir)
        cmd2_args = ["cp", os.path.join(proto_dir, "mlu_op_test.proto"), cwd]
        cmd2_ret = subprocess.run(cmd2_args, check=True)
    except Exception:
        logging.error("run {} and {} failed, please check!".format(" ".join(cmd1_args)," ".join(cmd2_args)))
        raise

