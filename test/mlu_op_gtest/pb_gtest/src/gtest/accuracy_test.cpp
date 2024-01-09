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
#include "accuracy_test.h"
#include "core/logging.h"
#include "json.h"

#define DEFAULT_THRESHOLD (0.00)

std::string getCaseName(std::string str) {
  std::string::size_type start = str.find_last_of("/");
  std::string result;
  if (start != std::string::npos) {
    start += 1;
    result = str.substr(start);
  } else {
    result = str;
  }
  return result;
}

bool getAccuracyThreshold(std::string op_name, double *threshold) {
  // init for default config
  *threshold = DEFAULT_THRESHOLD;
  bool in_white_list = false;
  std::string file_path;
  if (getenv("MLUOP_GTEST_ACC_THRESHOLD_FILE") != NULL) {
    file_path = getenv("MLUOP_GTEST_ACC_THRESHOLD_FILE");
  } else {
    LOG(INFO) << "getAccuracyThreshold:The env of "
                 "MLUOP_GTEST_ACC_THRESHOLD_FILE is NULL";
    LOG(INFO) << "getAccuracyThreshold:Use default accuracy threshold set";
    return in_white_list;
  }

  Json::Value root;
  Json::CharReaderBuilder build;
  std::string errs;
  std::fstream f;
  f.open(file_path, std::ios::in);
  if (!f.is_open()) {
    LOG(INFO) << "getAccuracyThreshold:Open json file error!";
    LOG(INFO) << "getAccuracyThreshold:Use default accruacy threshold set";
    return in_white_list;
  }

  bool parse_ok = Json::parseFromStream(build, f, &root, &errs);
  if (!parse_ok) {
    LOG(INFO) << "getAccuracyThreshold:Parse json file error!";
    LOG(INFO) << "getAccuracyThreshold:Use default accruacy threshold set";
    f.close();
    return in_white_list;
  }

  // get default accruacy threshold from json file
  in_white_list = root[0]["in_white_list"].asBool();
  double threshold_temp = root[0]["threshold_attrs"]["threshold"].asDouble();

  int i = 0;
  while (i < root.size()) {
    std::string name = root[i]["op_name"].asString();
    if (strcmp(op_name.c_str(), name.c_str()) == 0) {
      in_white_list = root[i]["in_white_list"].asBool();
      threshold_temp = root[i]["threshold_attrs"]["threshold"].asDouble();
      break;
    }
    ++i;
  }

  *threshold = threshold_temp;

  f.close();
  return in_white_list;
}

bool checkAccuracyBaselineStrategy(std::string case_name,
                                   std::vector<double> &base_errors,
                                   std::vector<double> &errors,
                                   double threshold) {
  for (int i = 0; i < base_errors.size(); i++) {
    double error_diff = errors[i] - base_errors[i];
    if (error_diff > base_errors[i] * threshold) {
      LOG(ERROR) << "[Accuracy Baseline:" << case_name
                 << "]:the accuracy result exceed baseline threshold which is "
                 << threshold * 100 << "%.";
      LOG(ERROR) << "[Accuracy Baseline:" << case_name
                 << "]:diff of baseline is " << base_errors[i];
      LOG(ERROR) << "[Accuracy Baseline:" << case_name
                 << "]:diff of this test is " << errors[i];
      return false;
    }
  }
  return true;
}
