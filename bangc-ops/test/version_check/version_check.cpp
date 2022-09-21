#include <mlu_op.h>
#include <string.h>

#include <iostream>
using namespace std;

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
      sprintf(err_info, "Expect version string length >= 3, but now is: %d",
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
      sprintf(err_info, "Expect version:  %d.%d.%d, but get %d.%d.%d", major_in,
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
