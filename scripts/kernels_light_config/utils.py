# Copyright (C) [2025] by Cambricon, Inc.
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
#!/usr/bin/env python3

import logging
import json
import os
import sys


# search cpp,h,mlu,mluh
def find_files_in_path(relative_path):
    list_path = []
    for root, dirs, files in os.walk(relative_path):
        for file in files:
            if (
                file.endswith(".h")
                or file.endswith(".cpp")
                or file.endswith(".mlu")
                or file.endswith(".mluh")
            ):
                full_path = os.path.join(root, file)
                list_path.append(full_path)
    return list_path


# search JSON
def find_json_files(absolute_path):
    json_files = []
    for root, dirs, files in os.walk(absolute_path):
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                json_files.append(full_path)
    return json_files


def get_relative_paths(absolute_path):
    relative_paths = []
    # base_folder = os.path.basename(absolute_path)
    base_folder = ""
    for folder, dirs, files in os.walk(absolute_path):
        # get relative_folder path
        relative_folder = os.path.relpath(folder, start=absolute_path)
        if "kernels" in relative_folder:
            relative_paths.append(relative_folder)

        # concat path
        for file in files:
            if (
                file.endswith(".h")
                or file.endswith(".cpp")
                or file.endswith(".mlu")
                or file.endswith(".mluh")
            ):
                if relative_folder == ".":
                    full_path = os.path.join(base_folder, file)
                else:
                    full_path = os.path.join(base_folder, relative_folder, file)
                relative_paths.append(full_path)
    return relative_paths


# parsing JSON file
def extract_headers(
    json_file_path, common_flag=True, header_flag=True, sources_flag=True
):
    header_files = []
    op_name = []
    try:
        with open(json_file_path, "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"The JSON file at {json_file_path} was not found.")
        return [], []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file at {json_file_path}.")
        return [], []

    # JSON.common
    if "common" in config and common_flag:
        header_files.extend(config["common"])

    # JSON.operators {op_name, header, sources}
    if "operators" in config:
        for operator in config["operators"]:
            if "name" in operator:
                if isinstance(operator["name"], list):
                    op_name.extend(operator["name"])
                else:
                    op_name.append(operator["name"])

            if "header" in operator and header_flag:
                if isinstance(operator["header"], list):
                    header_files.extend(operator["header"])
                else:
                    header_files.append(operator["header"])

            if "sources" in operator and sources_flag:
                if isinstance(operator["sources"], list):
                    header_files.extend(operator["sources"])
                else:
                    header_files.append(operator["sources"])

    return header_files, op_name


license_and_pragma_once = """
/*************************************************************************
 * Copyright (C) [2025] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/

#pragma once
"""


def write_string_to_file(long_string, filename):
    try:
        with open(filename, "w") as file:
            file.write(long_string)
            file.write(f"\n")
        print(f"Successfully wrote the license to {filename}")
    except Exception as e:
        print(f"Failed to write the license. Error: {e}")


def write_paths_to_file(path_list, filename):
    try:
        with open(filename, "a") as file:
            for path in path_list:
                file.write(f'#include "{path}"\n')
        print(f"Successfully wrote the {path} to {filename}")
    except Exception as e:
        print(f"Failed to write the {path} . Error: {e}")
