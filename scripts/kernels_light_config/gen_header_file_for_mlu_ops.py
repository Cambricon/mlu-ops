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

import os
import sys

from utils import *


if __name__ == "__main__":
    try:
        # get header_files
        header_files = []
        json_paths = find_json_files(os.path.dirname(__file__))

        for path in json_paths:
            files_path, _ = extract_headers(path, False, True, False)
            (
                header_files.extend(files_path)
                if isinstance(files_path, list)
                else header_files.append(files_path)
            )
        header_files_unique = sorted(list(set(header_files)))

        # gen bangc_kernels_collection.h
        mlu_ops_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        filename = mlu_ops_path + "/bangc_kernels_collection.h"
        write_string_to_file(license_and_pragma_once, filename)
        write_paths_to_file(header_files_unique, filename)

    except Exception as e:
        print(
            f"[ERROR] The generation of file bangc_kernels_collection.h failed. : {e}"
        )
        sys.exit(1)
