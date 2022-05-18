import argparse
from framework import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Case Generator",
                                     formatter_class=argparse.RawTextHelpFormatter, allow_abbrev=True)
    parser.add_argument("--json_path", type=str,
                        help="The root dir of all test cases manual json will be loaded."
                        "[default: ./manual_config]",
                        metavar="${JSONFILE_PATH}", default="./manual_config")
    parser.add_argument("--json_file", type=str,
                        help="The filename of manual json will be loaded."
                        "[default: ./manual_config]",
                        metavar="${JSON_FILE}", default=None)
    parser.add_argument("--save_path", type=str,
                        help="The root dir of all test cases (*pb/*prototxt) will be saved."
                        "[default: ./generated_testcases]",
                        metavar="${PROTOTXT_PATH}", default="./generated_testcases")
    parser.add_argument("--save_txt", type=bool,
                        help="except pb, also save prototxt",
                        default=False)
    parser.add_argument("-j", "--jobs", type=int, help="thread_num")
    parser.add_argument("--precheckin", type=bool, default=False,
                        help="is running precheckin, default value is False")
    parser.add_argument("--test_type", metavar="${TYPE}", choices=["MluOp"], default="MluOp",
                        help="Test Type of generator, now only support MluOp")

    parser.add_argument("--prototxt_path", type=str, default="./input_prototxt",
                        help="when input file is prototxt file,"
                        "this option indicates the input path"
                        )
    parser.add_argument("--file_type", type=str, default="json", choices=["json", "prototxt"],
                        help="select the type of input file."
                        "now only support json and prototxt two type"
                        )

    op_str = ("Op names, at least need one op.\n"
              "Input op_name should be the same as registered name in kernel. \n"
              "op_name in json file should be the same as op_name in mluop_proto. \n"
              "Generator run with the registered name in kernel. \n"
              "The generated folder name is consistent with the op_name in input json file.\n"
              "Normally all these op_name should be the same.")
    parser.add_argument("op_name", help=op_str, nargs="*")
    args = parser.parse_args()
    Director(args.test_type, args).run()
