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

/************************************************************************
 *
 *  @file main.c
 *
 **************************************************************************/
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "mlu_op_test.pb.h"

void usage() {
  std::cout << "Usage:" << std::endl;
  std::cout << "[1]: src_path or src_file. (prototxt or txt)" << std::endl;
  std::cout << "[2]: dst_path or dst_file. (pb)" << std::endl;
}

void listFiles(std::string dir, std::vector<std::string> &files) {
  DIR *dp = opendir(dir.c_str());
  if (dp == NULL) {  // it's not dir or not exist
    return;
  } else {
    // it is dir, grep all files
    struct dirent *dirp;
    while ((dirp = readdir(dp)) != NULL) {
      if (dirp->d_type == DT_DIR) {
        std::string sub_dir = std::string(dirp->d_name);
        if (sub_dir[sub_dir.length() - 1] != '.') {
          // is dir and not "." or ".."
          listFiles(dir + "/" + std::string(dirp->d_name), files);
        } else {
          // is "." or ".."
        }
      } else if (dirp->d_type == DT_REG || dirp->d_type == DT_UNKNOWN) {
        // is file
        files.push_back(dir + "/" + std::string(dirp->d_name));
      }
      // else maybe . or .. or else file type.
    }
    closedir(dp);
  }
}

// return true, is dir and can be visited
// else return false
bool isDir(std::string dir) {
  struct stat s;
  if (stat(dir.c_str(), &s) == 0) {
    if (s.st_mode & S_IFDIR) {
      return true;
    }
  }
  return false;
}

// return true, if not dir and can be visited
// else return false
bool isFile(std::string dir) {
  struct stat s;
  if (stat(dir.c_str(), &s) == 0) {
    if (!(s.st_mode & S_IFDIR)) {
      return true;
    }
  }
  return false;
}

// read in prototxt
bool readIn(const std::string &filename, google::protobuf::Message *proto) {
  size_t dot = filename.rfind(".");
  std::string ext = filename.substr(dot, filename.length());
  if (ext == ".prototxt" || ext == ".txt") {
    std::ifstream fin(filename, std::ios::in);
    if (!fin.is_open()) {
      std::cout << "File not found: " << filename;
      fin.close();
      return false;
    }

    bool status = true;
    google::protobuf::io::IstreamInputStream input(&fin);
    status = google::protobuf::TextFormat::Parse(&input, proto);
    fin.close();
    return status;
  } else {
    std::cout << "Can't parse this file: " << filename << std::endl;
    return false;
  }
}

// write to pb.
bool writeTo(const google::protobuf::Message *proto,
             const std::string &filename) {
  std::ifstream fin(filename, std::ios::in);
  if (fin.is_open()) {
    fin.close();
    std::cout << filename << " already exists, create file failed."
              << std::endl;
    return false;
  }

  std::ofstream fout(filename, std::ios::out);
  if (!fout.is_open()) {
    std::cout << "Create file failed: " << filename << std::endl;
    return false;
  }
  bool status = true;
  google::protobuf::io::OstreamOutputStream output(&fout);
  google::protobuf::io::CodedOutputStream coded_output(&output);
  status = proto->SerializeToCodedStream(&coded_output);
  return status;
}

void makeDir(std::string path) {
  if (0 != access(path.c_str(), 0)) {
    mkdir(path.c_str(), 0777);
  }
}

void dumpProto(std::string src, std::string dst) {
  auto proto_node = new mluoptest::Node;
  std::cout << src << " => " << dst << std::endl;
  readIn(src, proto_node);
  writeTo(proto_node, dst);
  delete proto_node;
}

std::string splitExt(std::string path) {
  size_t dot = path.rfind(".");
  return path.substr(dot, path.length());
}
std::string splitFileName(std::string path) {
  size_t slash = path.rfind("/");
  size_t dot = path.rfind(".");
  return path.substr(slash + 1, dot - slash - 1);
}

std::string splitPrefix(std::string root, std::string path) {
  size_t slash = path.rfind("/");
  std::string res = path.substr(root.length(), slash - root.length());
  return res;
}

std::vector<std::string> grepExtPrototxt(std::vector<std::string> files) {
  std::vector<std::string> res;
  for (int i = 0; i < files.size(); ++i) {
    std::string filename = files[i];
    size_t dot = filename.rfind(".");
    std::string ext = filename.substr(dot, filename.length());
    if (ext == ".prototxt" || ext == ".txt") {
      res.push_back(filename);
    }
  }
  return res;
}

std::string getCurrentDir() {
  char *buffer = NULL;
  if ((buffer = getcwd(NULL, 0)) == NULL) {
    std::cout << "Get current dir failed." << std::endl;
    return std::string("");  // return empty
  } else {
    std::string current_dir = std::string(buffer);
    free(buffer);
    return current_dir + "/";
  }
  return std::string("");
}

int main(int argc, char **argv) {
  if (argc != 3) {
    usage();
    exit(0);
  }

  std::string src_path = argv[1];
  std::string dst_path = argv[2];
  src_path = (src_path[0] == '/') ? src_path : getCurrentDir() + src_path;
  dst_path = (dst_path[0] == '/') ? dst_path : getCurrentDir() + dst_path;

  if (isFile(src_path) && !isDir(dst_path)) {
    dumpProto(src_path, dst_path);
  } else if (isFile(src_path) && isDir(dst_path)) {
    std::string name = splitFileName(src_path);
    dumpProto(src_path, dst_path + "/" + name + ".pb");
  } else if (isDir(src_path) && isDir(dst_path)) {
    std::vector<std::string> all_files;
    std::vector<std::string> proto_files;
    listFiles(src_path, all_files);
    proto_files = grepExtPrototxt(all_files);
    for (int i = 0; i < proto_files.size(); ++i) {
      std::string ext = splitExt(proto_files[i]);
      std::string name = splitFileName(proto_files[i]);
      std::string prefix = splitPrefix(src_path, proto_files[i]);
      makeDir(dst_path + prefix + "/");
      dumpProto(src_path + prefix + "/" + name + ext,
                dst_path + prefix + "/" + name + ".pb");
    }
  } else {
    std::cout << "Can't read in from " << src_path << " and write to "
              << dst_path << std::endl;
  }
}
