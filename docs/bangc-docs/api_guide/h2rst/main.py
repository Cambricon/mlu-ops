#coding=utf-8

import generate
import getopt, sys, os
import config

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "do:", ["api-s", "dt-s"])
    except Exception as e:
        print(e)
        sys.exit(-1)
    api_sort = config.api_sort
    datatypes_sort = config.datatype_sort
    get_define = config.get_define
    output_dir = ""
    for k, v in opts:
        if k == "--api-s":
            api_sort = True
        elif k == "--dt-s":
            datatypes_sort = True
        elif k == "-d":
            get_define = True
        elif k == "-o":
            output_dir = v
    if output_dir and not os.path.exists(output_dir):
        print("Output directory dose not exist:", output_dir)
    else:
        output_dir = output_dir if os.path.isdir(output_dir) else os.path.dirname(output_dir)
        files = []
        for file_path in args:
            if not os.path.exists(file_path):
                print("error:", "File dose not exist:", file_path)
                continue
            elif not os.path.isfile(file_path):
                print("error:", file_path, "is not a file.")
                continue
            files.append(file_path)
        if len(files) == 0:
            print("Please input header file(s)")
        else:
            rst_name = generate.hpp_to_rst(files, output_dir, get_define, api_sort, datatypes_sort)
            print("RST files:")
            print(rst_name)
