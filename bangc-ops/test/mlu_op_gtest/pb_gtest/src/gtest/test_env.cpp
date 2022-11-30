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
#include "test_env.h"

using mluoptest::global_var;

TestEnvInfo TestEnvironment::test_env_;
void TestEnvironment::convertBaselineXml2Txt() {
  if ((std::getenv("MLUOP_GTEST_GENERATE_BASELINE_ONLY") != NULL &&
       std::string(std::getenv("MLUOP_GTEST_GENERATE_BASELINE_ONLY"))
               .compare("ON") == 0) ||
      std::getenv("MLUOP_GTEST_PERF_BASELINE") == NULL ||
      std::string(std::getenv("MLUOP_GTEST_PERF_BASELINE")).compare("ON") !=
          0) {
    return;
  }

  std::string xml_file_name;
  if (std::getenv("MLUOP_BASELINE_XML_FILE") != NULL) {
    xml_file_name = std::getenv("MLUOP_BASELINE_XML_FILE");
  } else {
    LOG(ERROR) << "convertBaselineXml2Txt: MLUOP_BASELINE_XML_FILE is not set.";
    return;
  }

  std::ifstream xml_file(xml_file_name, std::ifstream::in);
  std::ofstream baseline_perf_txt_file;
  baseline_perf_txt_file.open(xml_file_name + ".txt", std::fstream::out);
  std::string line;
  // xml has the following order of properties
  // case_path, mluop_version, mlu_platform, hardware_time_base, <file_name>,
  // workspace_size_mlu
  while (std::getline(xml_file, line, '\n')) {
    if (line.find("name=\"case_path\"") != std::string::npos) {
      int start = line.find_last_of("=") + 2;
      int end = line.find_last_of("/") - 2;
      std::string case_path = line.substr(start, end - start + 1);
      std::string case_path_file_name =
          case_path.substr(case_path.find_last_of("/") + 1);

      // move to hardware_time_base
      for (int i = 0; i < 1; ++i) {
        std::getline(xml_file, line, '\n');
      }

      start = line.find("=") + 2;
      end = line.find("value=") - 3;
      if ((line.substr(start, end - start + 1)).compare("hardware_time_base") !=
          0) {
        LOG(ERROR) << "convertBaselineXml2Txt: expected hardware_time_base, "
                      "but got "
                   << line.substr(start, end - start + 1) << ".";
        baseline_perf_txt_file.close();
        xml_file.close();
        break;
      }

      std::string hardware_time_base_value;
      start = line.find_last_of("=") + 2;
      end = line.find_last_of("/") - 2;
      hardware_time_base_value = line.substr(start, end - start + 1);

      // move to <file_name>
      std::getline(xml_file, line, '\n');
      start = line.find("=") + 2;
      end = line.find("value=") - 3;
      std::string file_name = line.substr(start, end - start + 1);
      if (case_path_file_name.compare(file_name) != 0) {
        LOG(ERROR) << "convertBaselineXml2Txt: case file " << file_name
                   << " does not match the "
                   << "one in case_path " << case_path << ".";
        baseline_perf_txt_file.close();
        xml_file.close();
        break;
      }

      // move to workspace_size_mlu
      std::getline(xml_file, line, '\n');

      start = line.find("=") + 2;
      end = line.find("value=") - 3;
      if ((line.substr(start, end - start + 1)).compare("workspace_size_mlu") !=
          0) {
        LOG(ERROR) << "convertBaselineXml2Txt: expected workspace_size_mlu, "
                      "but got "
                   << line.substr(start, end - start + 1) << ".";
        baseline_perf_txt_file.close();
        xml_file.close();
        break;
      }

      std::string workspace_size_mlu_value;
      start = line.find_last_of("=") + 2;
      end = line.find_last_of("/") - 2;
      workspace_size_mlu_value = line.substr(start, end - start + 1);

      // write this case to txt
      baseline_perf_txt_file << "<" << file_name
                             << " value=" << hardware_time_base_value << "/>\n";
      baseline_perf_txt_file
          << "<workspace_size_mlu value=" << workspace_size_mlu_value << "/>\n";
    }
  }

  baseline_perf_txt_file.close();
  xml_file.close();
}

void TestEnvironment::SetUp() {
  // 1. set up cnrt env
  VLOG(4) << "SetUp CNRT environment.";

  // 2. get device num
  unsigned int dev_num = 0;
  ASSERT_EQ(cnrtGetDeviceCount(&dev_num), CNRT_RET_SUCCESS);
  if (dev_num <= 0) {  // dev_num_ should > 0
    FAIL() << "Can't find device";
  } else {
    VLOG(4) << "Found " << dev_num << " devices.";
  }

  // 3. random device id [0, dev_num)
  // [a, b] => (rand() % (b - a + 1)) + a
  unsigned int seed = time(0);
  global_var.dev_id_ = (rand_r(&seed) % (dev_num - 1 - 0 + 1)) + 0;

  // cnrt set current device using CNRT_DEFAULT_DEVICE
  // in cnrtGetDeviceHandle() CNRT_DEFAULT_DEVICE > id
  VLOG(4) << "Set current device as device: " << global_var.dev_id_;
  if (global_var.thread_num_ == 1) {
    // if single thread is 1, set current device.
    // if multi thread set current in each thread.
    ASSERT_EQ(cnrtSetDevice(global_var.dev_id_), CNRT_RET_SUCCESS);
  }
  // convert baseline xml to txt
  convertBaselineXml2Txt();
  recordEnvXml();
}

void TestEnvironment::TearDown() {
  VLOG(4) << "TearDown CNRT environment.";

  auto summary = global_var.summary_;
  std::cout << "[ SUMMARY  ] "
            << "Total " << summary.case_count << " cases of "
            << summary.suite_count << " op(s).\n";
  if (summary.failed_list.empty()) {
    std::cout << "ALL PASSED.\n";
  } else {
    auto case_list = summary.failed_list;
    std::cout << case_list.size() << " CASES FAILED:\n";
    for (auto it = case_list.begin(); it != case_list.end(); ++it) {
      std::cout << "Failed: " << (*it) << "\n";
    }
  }
}

// save date(Year_month_day_hour_minute_second).
void TestEnvironment::setDate() {
  std::ostringstream date_oss;
  char ymd_time[24];
  time_t timep;
  time(&timep);
  strftime(ymd_time, sizeof(ymd_time), "%Y_%m_%d_%H_%M_%S", localtime(&timep));
  date_oss << ymd_time;
  test_env_.date = date_oss.str();
}

void TestEnvironment::setMluopVersion() {
  std::ostringstream mluop_ver_oss;
  int mluop_ver[3];
  mluOpGetLibVersion(&mluop_ver[0], &mluop_ver[1], &mluop_ver[2]);
  mluop_ver_oss << mluop_ver[0] << "." << mluop_ver[1] << "." << mluop_ver[2];
  test_env_.mluop_version = mluop_ver_oss.str();
}

void TestEnvironment::setMluPlatform() {
  std::ostringstream mlu_platform_oss;
  char dev_name[64];
  cnDeviceGetName(dev_name, 64, 0);
  mlu_platform_oss << dev_name;
  test_env_.mlu_platform = mlu_platform_oss.str();
}

void TestEnvironment::setClusterLimit() {
  std::ostringstream cluster_limit;
  cluster_limit << std::setprecision(10)
                << getenv("MLUOP_SET_ClUSTER_LIMIT_CAPABILITY");
  test_env_.cluster_limit = cluster_limit.str();
}

void TestEnvironment::setJobLimit() {
  std::ostringstream job_limit;
  job_limit << std::setprecision(10)
            << getenv("MLUOP_SET_JOB_LIMIT_CAPABILITY");
  test_env_.job_limit = job_limit.str();
}

void TestEnvironment::setMluopBranch() {
  FILE *fp;
  char buffer[100];
  fp = popen("git rev-parse --abbrev-ref HEAD", "r");
  if (nullptr == fp) {
    perror("popen error.");
    return;
  }
  if (fgets(buffer, sizeof(buffer), fp) != NULL) {
    std::stringstream ss;
    ss << buffer;
    std::string buffer_str = ss.str();
    test_env_.mluop_branch = buffer_str.substr(0, buffer_str.size() - 1);
  }
  pclose(fp);
  if (test_env_.mluop_branch == "") {
    VLOG(4) << "cannot get mluops branch, maybe here is not git reposible.";
  }
}

void TestEnvironment::setCommitId() {
  FILE *fp;
  char buffer[100];
  fp = popen("git log", "r");
  if (nullptr == fp) {
    perror("popen error.");
    return;
  }
  while (fgets(buffer, sizeof(buffer), fp)) {
    std::stringstream ss;
    ss << buffer;
    std::string buffer_str = ss.str();
    std::string commit = "commit";
    std::string cur_str = buffer_str.substr(0, 6);
    if (cur_str == commit) {
      test_env_.commit_id = buffer_str.substr(0, buffer_str.size() - 1);
      break;
    }
  }
  pclose(fp);
  if (test_env_.commit_id == "") {
    VLOG(4) << "cannot get commit id, may here is not a git repository";
  }
}

// only use for local test, if use docker, ip maybe wrong
void TestEnvironment::setIp() {
  FILE *fp;
  char buffer[100];
  fp = popen("ifconfig", "r");
  if (nullptr == fp) {
    perror("popen error.");
    return;
  }
  bool inet_flag = false;
  bool card_flag = false;
  while (fgets(buffer, sizeof(buffer), fp)) {
    std::stringstream ss;
    ss << buffer;
    std::string buffer_str = ss.str();
    std::string docker_card = "docker";
    if (" " != buffer_str.substr(0, 1)) {
      if (docker_card != buffer_str.substr(0, 6)) {
        card_flag = true;
      }
    }
    if (!card_flag) {
      continue;
    } else {
      std::string ip_key = "inet addr";
      int len_ip_key = ip_key.size() + 1;
      int idx_begin = buffer_str.find("inet addr");
      if (std::string::npos != idx_begin) {
        int idx_end = buffer_str.find("Bcast");
        test_env_.ip = buffer_str.substr(idx_begin + len_ip_key,
                                         idx_end - idx_begin - len_ip_key - 2);
        break;
      } else {
        continue;
      }
    }
  }
  pclose(fp);
  // when not use for local test, may cannot has ifconfig.
  if (test_env_.ip == "") {
    VLOG(4) << "cannot get inet address, may \"ifconfig\" is not installed.";
  }
  //   fp = popen("ifconfig | grep \"inet addr\" | awk '{print $2}' | cut -d:
  //   -f2", "r");
}

void TestEnvironment::setTestEnv() {
  setDate();
  setCommitId();
  setMluopBranch();
  setMluopVersion();
  setMluPlatform();
  setClusterLimit();
  setJobLimit();
  setIp();
}

void TestEnvironment::recordEnvXml() {
  setTestEnv();
  testing::Test::RecordProperty("date", test_env_.date);
  testing::Test::RecordProperty("mluop_version", test_env_.mluop_version);
  testing::Test::RecordProperty("mlu_platform", test_env_.mlu_platform);
  testing::Test::RecordProperty("job_limit", test_env_.job_limit);
  testing::Test::RecordProperty("cluster_limit", test_env_.cluster_limit);
  testing::Test::RecordProperty("commit_id", test_env_.commit_id);
  testing::Test::RecordProperty("mluop_branch", test_env_.mluop_branch);
  testing::Test::RecordProperty("ip", test_env_.ip);
  if (global_var.repeat_ != 0) {
    testing::Test::RecordProperty("repeat_count", global_var.repeat_);
  }
}
