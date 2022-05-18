import os
import sys
from pathlib import Path
import argparse
import shutil
from random_utils import utils
from nonmlu_ops.base.random_parser import RandomParserFactory

class RandomGenerator:
    '''
    MluOpBuilder is used for generating cases for CNNL, called by Director.
    '''
    def __init__(self, *args, **kwargs):
        self.args_ = args[0]
        self.file_dict_ = {}

    def registerOp2Factory(self):
        random_dir = str(Path(__file__).resolve())
        sys.path.append(random_dir)
        for register_op_name in self.args_.op_name:
            register_op_dir = Path(random_dir + "/nonmlu_ops/" + register_op_name)
            register_op_pys = register_op_dir.rglob("*.py")
            for file in register_op_pys:
                filedir = str(file.resolve())
                filename = file.name
                if not filename.startswith("_"):
                    reldir = os.path.relpath(filedir, random_dir)
                    modulename, ext = os.path.splitext(reldir)
                    importname = modulename.replace("/", ".")
                    __import__(importname)

    def run(self):
        self.registerOp2Factory()
        self.clearOldCases()

        for op_name in self.args_.op_name:
            op_json_path = self.args_.json_path + "/" + op_name
            op_file_dict = {}.fromkeys(utils.getJsonFileFromDir(op_json_path), op_name)
            self.file_dict_.update(op_file_dict)
        if not self.file_dict_:
            raise Exception("No json file, please check op name and file dir.")
        for file_name, op_name_register in self.file_dict_.items():
            random_generator = RandomParserFactory().factory(op_name_register)(file_name)
            random_name = file_name.partition(op_name_register)[2]
            random_name = ("_".join(random_name.split("/"))).strip("_")
            final_manual_name = utils.mkdir_folder(self.args_.save_path, op_name_register) + "/" + random_name
            random_generator.generateManmulJson(final_manual_name)
            print("generator manual json:", final_manual_name)

        def clearOldCases(self):
            '''
            Before generate cases, use this function to remove save_path folder.
            '''
            if os.path.exists(self.args_.save_path):
                shutil.rmtree(self.args_.save_path, ignore_errors = True)
                print("Old Cases Cleared")

if __name__ == "__main__":
    op_str = ("Op names, at least need one op.\n"
              "Input op_name should be the same as registered name in kernel.\n"
              "op_name in json file should be the same as op_name in mluopproto.\n"
              "Generator run with the registered name in kernel.\n"
              "The generated folder name is consistent with the op_name in input json file.\n"
              "Normally all these op_name should be the same.")
    parser = argparse.ArgumentParser(description = "Random Case Generator",
                                     formatter_class = argparse.RawTextHelpFormatter,
                                     allow_abbrev = True)
    parser.add_argument("op_name", help = op_str, nargs = "+")
    parser.add_argument("--json_path", type = str,
                        help = "The root dir of all random json will be loaded."
                        "[default: ./random_config]",
                        metavar = "${JSONFILE_PATH}", 
                        default = "./random_config")
    parser.add_argument("--save_path", type = str,
                        help = "The root dir of all test cases(*pb or * prototxt) will be saved."
                        "[default: ./generated_manual]",
                        metavar = "${SAVE_PATH}", 
                        default = "./generated_manual")
    parser.add_argument("-j", "--jobs", type = int, help = "thread number")
    args = parser.parse_args()
    RandomGenerator(args).run()