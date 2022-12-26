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
#include "perf_test.h"
#include "core/logging.h"
#include "json.h"

// baseline default threshold
#define DEFAULT_SCALE_BOUND (100)
#define DEFAULT_THRESHOLD_ABSOLUTE (5)
#define DEFAULT_THRESHOLD_RELATIVE (0.04f)

// get hardware_time and workspace_size from txt file for better perfermance
bool getTxtData(std::string case_name, double *txt_time,
                double *workspace_size) {
  bool is_get = false;

  if (std::getenv("MLUOP_GTEST_GENERATE_BASELINE_ONLY") != NULL &&
      std::string(std::getenv("MLUOP_GTEST_GENERATE_BASELINE_ONLY"))
              .compare("ON") == 0) {
    return is_get;
  }

  std::string txt_file;
  if (getenv("MLUOP_BASELINE_XML_FILE") != NULL) {
    txt_file = std::string(getenv("MLUOP_BASELINE_XML_FILE")) + ".txt";
  } else {
    LOG(ERROR) << "getTxtData:The env of MLUOP_BASELINE_XML_FILE is NULL.";
    return is_get;
  }

  std::ifstream file(txt_file.c_str(), std::fstream::in);
  if (!file.is_open()) {
    LOG(ERROR) << "getTxtData: failed to open file " << txt_file << "\n";
    return is_get;
  }
  std::string hw_input, ws_input, result;
  std::string::size_type start, end;
  while (getline(file, hw_input)) {
    if (hw_input.find(case_name.c_str()) != std::string::npos) {
      // get hard_time_base
      start = hw_input.find_last_of("value=");
      end = hw_input.find_last_of("/");
      result = hw_input.substr(start + 1, end - start - 1);
      *txt_time = atof(result.c_str());

      // get workspace_size_base
      getline(file, ws_input);
      start = ws_input.find_last_of("value=");
      end = ws_input.find_last_of("/");
      result = ws_input.substr(start + 1, end - start - 1);
      *workspace_size = atof(result.c_str());
      is_get = true;
      break;
    }
  }

  return is_get;
}

// get pb or prototxt file name
std::string getTestCaseName(std::string str) {
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

// get threshold from json config file
bool getThreshold(std::string op_name, double *scale_bound,
                  double *threshold_absolute, double *threshold_relative) {
  // init for default config
  bool in_white_list = false;
  *scale_bound = DEFAULT_SCALE_BOUND;
  *threshold_absolute = DEFAULT_THRESHOLD_ABSOLUTE;
  *threshold_relative = DEFAULT_THRESHOLD_RELATIVE;

  std::string file_path;
  if (getenv("MLUOP_GTEST_THRESHOLD_FILE") != NULL) {
    file_path = getenv("MLUOP_GTEST_THRESHOLD_FILE");
  } else {
    LOG(INFO) << "getThreshold:The env of MLUOP_GTEST_THERSHOLD_FILE is NULL";
    LOG(INFO) << "getThreshold:Use default threshold set";
    return in_white_list;
  }

  Json::Value root;
  Json::CharReaderBuilder build;
  std::string errs;
  std::fstream f;
  f.open(file_path, std::ios::in);
  if (!f.is_open()) {
    LOG(INFO) << "getThreshold:Open json file error!";
    LOG(INFO) << "getThreshold:Use default threshold set";
    return in_white_list;
  }

  bool parse_ok = Json::parseFromStream(build, f, &root, &errs);
  if (!parse_ok) {
    LOG(INFO) << "getThreshold:Parse json file error!";
    LOG(INFO) << "getThreshold:Use default threshold set";
    f.close();
    return in_white_list;
  }

  // get default threshold from json file
  in_white_list = root[0]["in_white_list"].asBool();
  double scale_bound_temp =
      root[0]["threshold_attrs"]["scale_bound"].asDouble();
  double threshold_absolute_temp =
      root[0]["threshold_attrs"]["threshold_absolute"].asDouble();
  double threshold_relative_temp =
      root[0]["threshold_attrs"]["threshold_relative"].asDouble();
  int i = 0;
  while (i < root.size()) {
    std::string name = root[i]["op_name"].asString();
    if (strcmp(op_name.c_str(), name.c_str()) == 0) {
      in_white_list = root[i]["in_white_list"].asBool();
      scale_bound_temp = root[i]["threshold_attrs"]["scale_bound"].asDouble();
      threshold_absolute_temp =
          root[i]["threshold_attrs"]["threshold_absolute"].asDouble();
      threshold_relative_temp =
          root[i]["threshold_attrs"]["threshold_relative"].asDouble();
      break;
    }
    ++i;
  }

  *scale_bound = scale_bound_temp;
  *threshold_absolute = threshold_absolute_temp;
  *threshold_relative = threshold_relative_temp;

  f.close();

  return in_white_list;
}

// update baseline hardware_time_base
bool updateBaselineStrategy(double hw_time_mean, double scale_bound,
                            double threshold_absolute,
                            double threshold_relative, double *hw_time_base) {
  if (*hw_time_base <= 0) {
    LOG(ERROR) << "updateBaselineStrategy:baseline time is error, it must be "
                  "greater than zero";
    return false;
  }
  if (scale_bound <= 0 || threshold_absolute <= 0 || threshold_relative <= 0) {
    LOG(ERROR) << "updateBaselineStrategy:threshold is wrong, it must be "
                  "greater than zero";
    return false;
  }
  double time_base = *hw_time_base;
  double time_diff = hw_time_mean - time_base;
  bool is_baseline_pass = true;
  if (time_base <= scale_bound) {  // small scale
    if (time_diff > threshold_absolute) {
      is_baseline_pass = false;
    } else if (time_diff <= threshold_absolute) {
      is_baseline_pass = true;
      *hw_time_base = (time_base + hw_time_mean) / 2;
    }
  } else {  // big scale
    if (time_diff / time_base > threshold_relative) {
      is_baseline_pass = false;
    } else if (time_diff / time_base <= threshold_relative) {
      is_baseline_pass = true;
      *hw_time_base = (time_base + hw_time_mean) / 2;
    }
  }
  return is_baseline_pass;
}
