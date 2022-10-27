/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc.
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

#include <mlu_op.h>
#include <string.h>

#include <iostream>
using namespace std;  //NOLINT

/*
Add version check by mluOpGetLibVersion.
*/
int main(int argc, char *argv[]) {
  if (argc != 2) {
    cerr << "Error input" << endl;
    return 1;
  }
  try {
    int version_in_str_size = strlen(argv[1]);
    if (version_in_str_size < 3) {
      char err_info[100];
      sprintf(err_info, "Expect version string length >= 3, but now is: %d",  // NOLINT
              version_in_str_size);
      cerr << err_info << endl;
      return 1;
    }
    int version_in[3] = {0, 0, 0};
    int id = 0;
    for (int i = 0; i < version_in_str_size; i++) {
      if (argv[1][i] != '.') {
        version_in[id] = version_in[id] * 10 + int(argv[1][i] - '0');
      } else {
        id++;
      }
    }
    int major_in = version_in[0];
    int minor_in = version_in[1];
    int patch_in = version_in[2];

    int major;
    int minor;
    int patch;
    mluOpGetLibVersion(&major, &minor, &patch);
    if (major != major_in || minor != minor_in || patch != patch_in) {
      char err_info[100];
      sprintf(err_info, "Expect version:  %d.%d.%d, but get %d.%d.%d", major_in,  // NOLINT
              minor_in, patch_in, major, minor, patch);
      cerr << err_info << endl;
      return 1;
    }
    printf("The mluops version is %d.%d.%d\n", major, minor, patch);
  } catch (const char *msg) {
    cerr << msg << endl;
    return 1;
  }
  return 0;
}
