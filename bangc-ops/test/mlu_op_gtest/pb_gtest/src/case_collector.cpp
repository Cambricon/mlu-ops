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
#include <string>
#include <set>
#include <algorithm>
#include <vector>
#include "case_collector.h"

extern mluoptest::GlobalVar mluoptest::global_var;
using mluoptest::global_var;

Collector::Collector(const std::string &name) { op_name_ = name; }

size_t Collector::num() {
  if (global_var.thread_num_ > 1) {
    return 1;
  } else {
    return list().size();
  }
}

void Collector::grep_case(std::string dir, std::vector<std::string> &files) {
  DIR *dp = opendir(dir.c_str());
  if (dp == NULL) {  // it's not dir or not exit
    return;
  } else {
    // it is dir, grep all files
    struct dirent *dirp;
    while ((dirp = readdir(dp)) != NULL) {
      if (dirp->d_type == DT_DIR) {
        std::string sub_dir = std::string(dirp->d_name);
        if (sub_dir[sub_dir.length() - 1] != '.') {
          // is dir and not "." or ".."
          grep_case(dir + "/" + std::string(dirp->d_name), files);
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

void Collector::grep_dir(std::string dir, std::vector<std::string> &op_dirs) {
  DIR *dp = opendir(dir.c_str());
  if (dp == NULL) {  // it's not dir or not exit
    return;
  } else {
    // it is dir, grep all files
    struct dirent *dirp;
    while ((dirp = readdir(dp)) != NULL) {
      if (dirp->d_type == DT_DIR) {
        std::string sub_dir = std::string(dirp->d_name);
        if (strcmp(op_name_.c_str(), sub_dir.c_str()) == 0) {
          op_dirs.push_back(dir + "/" + sub_dir);
        } else if (sub_dir[sub_dir.length() - 1] != '.') {
          grep_dir(dir + "/" + sub_dir, op_dirs);
        } else {
          // is "." or ".."
        }
      }
    }
    closedir(dp);
  }
}

std::vector<std::string> Collector::read_case(std::string cases_list) {
  // /**/add/add_***.pb\r
  std::vector<std::string> res;
  std::string op_dir = "/" + op_name_ + "/";
  std::ifstream fin(cases_list, std::ios::in);

  std::string case_path;
  while (getline(fin, case_path)) {
    if (case_path.find(op_dir) != std::string::npos) {
      if (case_path.back() == '\r' || case_path.back() == '\n') {
        res.push_back(
            case_path.substr(0, case_path.length() - 1));  // remove "\r"
      } else {
        res.push_back(case_path.substr(0, case_path.length()));
      }
    }
    case_path.clear();
  }
  fin.close();
  return res;
}

std::string Collector::current_dir() {
  char *buffer = NULL;
  if ((buffer = getcwd(NULL, 0)) == NULL) {
    LOG(ERROR) << "Get current dir failed.";
    return std::string("");  // return empty
  } else {
    std::string current_dir = std::string(buffer);
    free(buffer);
    return current_dir + "/";
  }
  return std::string("");
}

std::vector<std::string> Collector::list_by_case_path(std::string path) {
  std::string abs_case_path = (path[0] != '/') ? (current_dir() + path) : path;
  if (path.find("/" + op_name_ + "/") == std::string::npos) {
    // case path is not belong this op
    assertPath(abs_case_path, caseType::CASE_FILE, __FILE__, __LINE__);
    return {};
  }
  assertPath(abs_case_path, caseType::CASE_FILE, __FILE__, __LINE__);
  RETURN_IF_PATH_INVALID();
  return {abs_case_path};
}

std::vector<std::string> Collector::list_by_case_list(std::string list_file) {
  if (list_file[0] != '/') {
    // turn to abs
    list_file = current_dir() + list_file;
  }
  assertPath(list_file, caseType::CASE_LIST, __FILE__, __LINE__);
  RETURN_IF_PATH_INVALID();
  return read_case(list_file);
}

void Collector::assertPath(std::string &case_path, caseType case_type,
                           std::string file, int line_num) {
  // error message is not displayed like a normal FAILED test would be, so use
  // LOG(ERROR) to highlight it
  switch (case_type) {
    case caseType::CASE_FILE:
    case caseType::CASE_LIST: {
      std::ifstream fin(case_path, std::ios::in);
      if (!fin) {
        LOG(ERROR) << file << ":" << line_num << ":: "
                   << "Can not open case_path/cases_list " << case_path;
        exit(EXIT_FAILURE_MLUOP);
      }
      path_exist = true;
    } break;
    case caseType::CASE_DIR: {
      DIR *dp = opendir(case_path.c_str());
      if (!dp) {
        LOG(ERROR) << file << ":" << line_num << ":: "
                   << "Can not open cases_dir " << case_path;
        exit(EXIT_FAILURE_MLUOP);
      }
      path_exist = true;
      closedir(dp);
    } break;
    default:
      break;
  }
}

std::vector<std::string> Collector::list_by_case_dir(std::string case_dir) {
  assertPath(case_dir, caseType::CASE_DIR, __FILE__, __LINE__);
  RETURN_IF_PATH_INVALID();
  std::vector<std::string> res;

  std::string suffix_txt = ".prototxt";
  std::string suffix_pb = ".pb";

  if (case_dir.back() != '/') {
    case_dir += "/";
  }

  // get all the files in dir
  std::vector<std::string> all_files;
  if (true == mluoptest::getEnv("MLUOP_GTEST_CASE_RECURSIVE_SEARCH", false)) {
    // found env
    std::vector<std::string> op_dirs;
    grep_dir(case_dir, op_dirs);
    for (auto op_dir : op_dirs) {
      grep_case(op_dir, all_files);
    }
  } else {
    // no env
    grep_case(case_dir + op_name_, all_files);
  }

  for (auto name : all_files) {
    if (name.find("invalid") == std::string::npos) {  // no "invalid" in name
      if (name.substr(name.size() - suffix_pb.size(), suffix_pb.size()) ==
          suffix_pb) {
        res.push_back(name);
      } else if (name.substr(name.size() - suffix_txt.size(),
                             suffix_txt.size()) == suffix_txt) {
        res.push_back(name);
      }
    }
  }

  return res;
}

std::vector<std::string> Collector::list() {
  // for --case_path
  // vector<> size is 1
  std::vector<std::string> case_names;
  if (!global_var.case_path_.empty()) {
    case_names = list_by_case_path(global_var.case_path_);
    return case_names;
  }

  // for --case_list
  if (!global_var.cases_list_.empty()) {
    case_names = list_by_case_list(global_var.cases_list_);
  } else if (!global_var.cases_dir_.empty()) {
    // for --case_dir
    case_names = list_by_case_dir(global_var.cases_dir_);
  } else {
    case_names = list_by_case_dir("../../test/mlu_op_gtest/pb_gtest/src/zoo/");
  }

  auto fisher_shuffle = [](std::vector<std::string> res,
                           int n) -> std::vector<std::string> {
    std::vector<std::string> res_n;
    n = std::min(n, int(res.size()));
    std::srand(std::time(nullptr));
    for (int i = 0; i < n; i++) {
      int rand = std::rand() % res.size();
      res_n.push_back(res[rand]);
      res.erase(res.begin() + rand);
    }
    return res_n;
  };

  // rand_n
  if (global_var.rand_n_ != -1) {
    case_names = fisher_shuffle(case_names, global_var.rand_n_);
  }

  // shuffle
  if (global_var.shuffle_) {
    std::random_shuffle(case_names.begin(), case_names.end());
  }

  if (case_names.size() == 0) {
    LOG(INFO) << "No cases found for " << op_name_;
  }

  return case_names;
}
