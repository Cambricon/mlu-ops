#!/usr/bin/env python3

import os

from bangc_kernels_path_check import extract_headers,find_json_files

license_and_pragma_once = '''

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
'''

def write_string_to_file(long_string, filename):
    try:
        with open(filename, 'w') as file:
            file.write(long_string)
        print(f"Successfully wrote the license to {filename}")
    except Exception as e:
        print(f"Failed to write the license. Error: {e}")


def write_paths_to_file(path_list, filename):
    try:
        with open(filename, 'a') as file:
            for path in path_list:
                if "/" in path: file.write(f'\n') 
                file.write(f'#include "{path}"\n')
                
                    
        print(f"Successfully wrote the {path} to {filename}")
    except Exception as e:
        print(f"Failed to write the {path} . Error: {e}")


if __name__ == "__main__":
    try:
        # get header_files
        header_files = []
        json_paths = find_json_files(os.path.dirname(__file__))
        for path in json_paths:
            files_path,_ = extract_headers(path, False, True, False)
            header_files.extend(files_path) if isinstance(files_path, list) else header_files.append(files_path)

        header_files_unique = sorted(list(set(header_files)))
        
        # gen bangc_kernels_collection.h
        mlu_ops_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        filename = mlu_ops_path + "/bangc_kernels_collection.h"
        write_string_to_file(license_and_pragma_once, filename)
        write_paths_to_file(header_files_unique, filename)
    except Exception as e:
        print(f"[ERROR] The generation of file bangc_kernels_collection.h failed. : {e}") 
