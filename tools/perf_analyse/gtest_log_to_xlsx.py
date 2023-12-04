# Copyright (C) [2022] by Cambricon, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall self.tcp included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring
# pylint: disable=attribute-defined-outside-init

import os
import argparse
import sys
import subprocess
import logging
import config


def check_mluops_proto():
    # generate mluops_test_pb2.py
    proto_dir = os.path.abspath(
        os.path.realpath(__file__) + "/../../../bangc-ops/test/mlu_op_gtest/pb_gtest/mlu_op_test_proto")
    cwd = os.path.abspath(os.path.realpath(__file__) + "/../")
    cmd_args = [
        "protoc", "--python_out", cwd, "--proto_path", proto_dir,
        proto_dir + "/mlu_op_test.proto"
    ]
    try:
        # check returncode
        cmd_ret = subprocess.run(cmd_args, check=True)
    except Exception as e:
        print("run {} failed, please check!".format(" ".join(cmd_args)))
        exit()

if __name__ == "__main__":
    logging.basicConfig(format = '%(asctime)s %(message)s')
    # change str to bool for argparse
    def str2bool(v):
        if v.lower() in ("yes", "true", "1"):
            return True
        elif v.lower() in ("no", "false", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value.")

    # argument parse
    parser = argparse.ArgumentParser()
    # update_case_list can not need log_path
    log_path_required = True
    # for admin
    if "admin" in sys.argv:
        log_path_required = False
        parser.add_argument("--case_list",
                            type=str2bool,
                            help="whether update mluops_case_list, i.e. 1",
                            default=False,
                            required=False)
        parser.add_argument("--batch_size",
                            type=int,
                            help="update network of which batch size, i.e. 1",
                            default=-1,
                            required=False)
        parser.add_argument("--case_run",
                            type=str2bool,
                            help="whether update mluops_case_run, i.e. 1",
                            default=False,
                            required=False)
        parser.add_argument("--truncate_case_run",
                            type=str2bool,
                            help="whether truncate mluops_case_run, i.e. 1",
                            default=False,
                            required=False)
        parser.add_argument("--cases_dir",
                            type=str,
                            default="",
                            help="cases dir for generating h5",
                            required=False)
        parser.add_argument("--rename_pb",
                            type=str2bool,
                            default=False,
                            help="rename pb names",
                            required=False)

    parser.add_argument(
        "--log_path",
        type=str,
        help=
        "path of the input file, can be xml json or log got by tee, see more details in README.md",
        required=log_path_required)
    parser.add_argument(
        "--compare_path",
        type=str,
        help=
        "path of the baseline file. the promotion ratio of hardware time \
        is (compare_path - log_path) / compare_path",
        required=False)
    parser.add_argument(
        "--xlsx_path",
        type=str,
        help=
        "path of the output excel file, i.e. 290.xlsx, default replace \
        xml or json with xlsx in log_path",
        required=False)
    parser.add_argument(
        "--tpi",
        type=str2bool,
        help="whether generate tpi excel file, i.e. 1, default 0",
        default = False,
        required=False)
    parser.add_argument(
        "--simple_tpi",
        type=str2bool,
        help="whether generate simple_tpi excel file, i.e. 1, default 0",
        default = False,
        required=False)
    parser.add_argument(
        "--frameworks",
        type=str,
        help="the frameworks to be filtered, default pytorch",
        default="pytorch",
        required=False)
    parser.add_argument(
        "--topN",
        type=int,
        help="top N for hardware time or efficiency, i.e. 100, default 100",
        required=False,
        default=100)
    parser.add_argument(
        "--cpu_count",
        type=int,
        help=
        "how many cpu core will be used to parse prototxt, i.e. 8, default 8",
        required=False,
        default=8)
    parser.add_argument(
        "--use_db",
        type=str2bool,
        help="whether use database to append case info, i.e. 1, default 0",
        default=False,
        required=False)
    parser.add_argument("--so_path",
                        type=str,
                        help="path of libmluops.so to get code size",
                        default="",
                        required=False)
    parser.add_argument("--so_path_compare",
                        type=str,
                        help="path of libmluops.so to compare",
                        default="",
                        required=False)
    parser.add_argument(
        "--host_log_path",
        type=str,
        default="",
        help=
        "path of the host time file, must be xml, see more details in README.md",
        required=False)
    opt, unknown = parser.parse_known_args()

    # assure mluops_test_pb2.py exists
    check_mluops_proto()
    # config of columns
    c = config.Config()
    
    import utils
    if not log_path_required and opt.log_path == None:
        # generate h5
        if opt.cases_dir != "":
            if opt.rename_pb == True:
                utils.pb_name_rename(opt.cases_dir)
            else:
                utils.generator_h5(opt.cases_dir, opt.cpu_count)
            exit()
    else:
        # merge host.xml into device.xml
        if opt.host_log_path != "":
            utils.merge_xml(os.path.abspath(opt.log_path),
                            os.path.abspath(opt.host_log_path), c)
            exit()
        if opt.so_path != "":
            code_size = utils.get_code_size(opt.so_path)
            utils.dfs_to_excel([code_size], ['code_size'], "code_size.xlsx")
        if opt.so_path_compare != "":
            code_size_cmp = utils.get_code_size(opt.so_path_compare)
            code_size_diff = utils.compare_code_size(code_size, code_size_cmp)
            utils.dfs_to_excel([code_size_diff], ['code_size_compare'],
                               "code_size_compare.xlsx")
        if opt.xlsx_path is None:
            # not handle json
            opt.xlsx_path = opt.log_path.split("/")[-1].replace(".xml",
                                                                "") + ".xlsx"

        # parse gtest ouput file
        df = utils.parse_input_file(os.path.abspath(opt.log_path), c)

        # append case info using database or parsing prototxt
        utils.append_case_info(df, opt.cpu_count, opt.use_db)
        
        utils.dump_to_excel(df, os.path.abspath(opt.xlsx_path), opt.topN)

        if opt.compare_path is not None:
            compare_path = os.path.abspath(opt.compare_path)
            df_baseline = utils.parse_input_file(compare_path, c)
            utils.append_case_info(df_baseline, opt.cpu_count, opt.use_db)
            if opt.simple_tpi:
                version_compare = utils.get_version_numer(opt.log_path, opt.compare_path)
            else:
                version_compare = ['_new', '_baseline']
            utils.compare_log(df, df_baseline, os.path.abspath(opt.log_path),
                              compare_path, os.path.abspath(opt.xlsx_path))
