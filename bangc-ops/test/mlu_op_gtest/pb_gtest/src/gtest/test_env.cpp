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
#include <sys/types.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>   // open, write
#include <algorithm>  // sort
#include <utility>    // std::pair
#include "cndev.h"    // cndevGetProcessInfo
#include "hardware_monitor.h"

using mluoptest::global_var;

TestEnvInfo TestEnvironment::test_env_;
std::unordered_map<std::string, std::vector<double>> acc_baseline_map;

void TestEnvironment::getAccuracyBaselineXml() {
  if (std::getenv("MLUOP_GTEST_ACC_BASELINE") == NULL ||
      std::string(std::getenv("MLUOP_GTEST_ACC_BASELINE")).compare("ON") != 0) {
    return;
  }
  std::string xml_file_name;
  if (std::getenv("MLUOP_ACC_BASELINE_XML_FILE") != NULL) {
    xml_file_name = std::getenv("MLUOP_ACC_BASELINE_XML_FILE");
  } else {
    LOG(WARNING)
        << "getAccuracyBaseline: MLUOP_ACC_BASELINE_XML_FILE is not set.";
  }
  std::ifstream xml_file(xml_file_name, std::ifstream::in);
  std::string line, error_diff;
  std::string::size_type start, end;
  while (std::getline(xml_file, line, '\n')) {
    if (line.find("name=\"case_path\"") != std::string::npos) {
      start = line.find_last_of("=") + 2;
      end = line.find_last_of("/") - 2;
      std::string case_path = line.substr(start, end - start + 1);
      std::string case_name = case_path.substr(case_path.find_last_of("/") + 1);
      acc_baseline_map[case_name] = std::vector<double>({});

      // move to error_diff
      while (line.find("error_diff") == std::string::npos) {
        std::getline(xml_file, line, '\n');
      }

      while (line.find("properties") == std::string::npos &&
             line.find("error_diff") != std::string::npos) {
        start = line.find_last_of("=") + 2;
        end = line.find_last_of("/") - 2;
        error_diff = line.substr(start, end - start + 1);
        acc_baseline_map[case_name].push_back(atof(error_diff.c_str()));
        std::getline(xml_file, line, '\n');
      }
    }
  }
  xml_file.close();
}

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
      int end = line.find_last_of("/") - 3;
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
        LOG(ERROR)
            << "convertBaselineXml2Txt: expected hardware_time_base, but got "
            << line.substr(start, end - start + 1) << ".";
        baseline_perf_txt_file.close();
        xml_file.close();
        break;
      }
      std::string hardware_time_base_value;
      start = line.find_last_of("=") + 2;
      end = line.find_last_of("/") - 3;
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
        LOG(ERROR)
            << "convertBaselineXml2Txt: expected workspace_size_mlu, but got "
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
  VLOG(4) << "SetUp CNRT environment.";
  // set device
  setDevice();
  mluoptest::monitor->signalMonitorDeviceChosen();
  // convert baseline xml to txt
  convertBaselineXml2Txt();
  getAccuracyBaselineXml();
  setTestEnv();
  recordEnvXml();
  recordEnvLog();
}

void TestEnvironment::TearDown() {
  VLOG(4) << "TearDown CNRT environment.";
  showSummary();

  // set compute mode as default
  restoreComputeMode();
  mluoptest::monitor->signalMonitorOneGRepeatDone();
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

void TestEnvironment::setMluOpVersion() {
  std::ostringstream mluop_ver_oss;
  int mluop_ver[3];
  mluOpGetLibVersion(&mluop_ver[0], &mluop_ver[1], &mluop_ver[2]);
  mluop_ver_oss << mluop_ver[0] << "." << mluop_ver[1] << "." << mluop_ver[2];
  test_env_.mluop_version = mluop_ver_oss.str();
}

void TestEnvironment::setDriverVersion() {
  std::ostringstream driver_ver_oss;
  int driver_vers[3];
  cnGetDriverVersion(&driver_vers[0], &driver_vers[1], &driver_vers[2]);
  driver_ver_oss << driver_vers[0] << "." << driver_vers[1] << "."
                 << driver_vers[2];
  test_env_.driver_version = driver_ver_oss.str();
}

void TestEnvironment::setCnrtVersion() {
  std::ostringstream cnrt_version_oss;
  int cnrt_vers[3];
  cnrtGetLibVersion(&cnrt_vers[0], &cnrt_vers[1], &cnrt_vers[2]);
  cnrt_version_oss << cnrt_vers[0] << "." << cnrt_vers[1] << "."
                   << cnrt_vers[2];
  test_env_.cnrt_version = cnrt_version_oss.str();
}

void TestEnvironment::setMluPlatform() {
  constexpr int MAX_DEVICE_NAME_LEN = 64;
  std::ostringstream mlu_platform_oss;
  char dev_name[MAX_DEVICE_NAME_LEN];
  cnDeviceGetName(dev_name, MAX_DEVICE_NAME_LEN, 0);
  mlu_platform_oss << dev_name;
  test_env_.mlu_platform = mlu_platform_oss.str();
}

void TestEnvironment::setClusterLimit() {
  std::ostringstream cluster_limit;
  cluster_limit << std::setprecision(10)
                << getenv("MLUOP_SET_CLUSTER_LIMIT_CAPABILITY");
  test_env_.cluster_limit = cluster_limit.str();
}

void TestEnvironment::setJobLimit() {
  std::ostringstream job_limit;
  job_limit << std::setprecision(10)
            << getenv("MLUOP_SET_JOB_LIMIT_CAPABILITY");
  test_env_.job_limit = job_limit.str();
}

static void *resolveMluOpSym(const char *name) {
  void *sym = dlsym(RTLD_NEXT, name);  // allow LD_PRELOAD hook
  if (!sym) {
    // for general case (e.g. we are linking static version)
    return dlsym(RTLD_DEFAULT, name);
  }
  return sym;
}

void TestEnvironment::setMluOpBranch() {
  static auto get_branch_fallback = [](TestEnvInfo &test_env_) {
    FILE *fp;
    char buffer[100];
    fp = popen(
        "git branch --contains `git rev-parse HEAD` | grep -v 'HEAD detached' "
        "| head -n 1",
        "r");
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
  };
  void *sym = resolveMluOpSym("mluOpInternalGetBranchInfo");
  if (!sym) {
    LOG(WARNING) << "mluOpInternalGetBranchInfo not found, use fallback method";
    get_branch_fallback(test_env_);
  } else {
    test_env_.mluop_branch = ((const char *(*)(void))sym)();
  }
  if (test_env_.mluop_branch == "") {
    VLOG(4) << "cannot get mluOp branch, maybe here is not git reposible.";
  }
}

void TestEnvironment::setCommitId() {
  static auto get_commit_fallback = [](TestEnvInfo &test_env_) {
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
  };
  void *sym = resolveMluOpSym("mluOpInternalGetCommitId");
  if (!sym) {
    LOG(WARNING) << "mluOpInternalGetCommitId not found, use fallback method";
    return get_commit_fallback(test_env_);
  }
  test_env_.commit_id = ((const char *(*)(void))sym)();
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
      std::string ip_key = "inet";
      int len_ip_key = ip_key.size() + 1;
      int idx_begin = buffer_str.find("inet");
      if (std::string::npos != idx_begin) {
        int idx_end = buffer_str.find("netmask");
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

bool TestEnvironment::getBusId(int device_id, std::string &bus_id_str) {
  constexpr int MAX_DEVICE_NAME_LEN = 64;
  char dev_name[MAX_DEVICE_NAME_LEN];
  cnDeviceGetName(dev_name, MAX_DEVICE_NAME_LEN, device_id);
  char str[100];
  VLOG(4) << "Device " << device_id << "'s platform=" << dev_name;
  if (!strncmp(dev_name, "CE3226", 6)) {
    snprintf(str, sizeof(str), "cabc:0328");
    VLOG(4) << "Device " << device_id << "'s mlu id=" << str;
  } else if (!strncmp(dev_name, "1V", 2)) {
    snprintf(str, sizeof(str), "cabc:0428");
    VLOG(4) << "Device " << device_id << "'s mlu id=" << str;
  } else {
    cnrtRet_t ret = cnrtDeviceGetPCIBusId(str, 100, device_id);
    if (ret != CNRT_RET_SUCCESS) {
      LOG(WARNING) << "Fail to get device " << device_id << "'s bus id.";
      return false;
    } else {
      VLOG(4) << "Device " << device_id << "'s bus id=" << str;
    }
  }
  bus_id_str = str;
  return true;
}

bool TestEnvironment::setComputeMode(std::string bus_id_str, char mode) {
  std::stringstream compute_mode_file_ss;
  compute_mode_file_ss << "/proc/driver/cambricon/mlus/" << bus_id_str
                       << "/exclusive_mode";
  int fd =
      open(compute_mode_file_ss.str().c_str(), O_CLOEXEC | O_SYNC | O_RDWR);
  char compute_mode[1] = {mode};
  ssize_t stat = write(fd, compute_mode, 1);
  if (stat == -1) {
    LOG(WARNING) << "Fail to set bus id " << bus_id_str << "'s Compute-mode to "
                 << mode << ".";
    return false;
  } else {
    VLOG(4) << "Bus id " << bus_id_str << "'s Compute-mode set to " << mode
            << ".";
  }
  return true;
}

bool TestEnvironment::getComputeMode(std::string bus_id_str, char &mode) {
  std::stringstream compute_mode_file_ss;
  compute_mode_file_ss << "/proc/driver/cambricon/mlus/" << bus_id_str
                       << "/exclusive_mode";
  int fd =
      open(compute_mode_file_ss.str().c_str(), O_CLOEXEC | O_SYNC | O_RDWR);
  char compute_mode[1];
  ssize_t stat = read(fd, compute_mode, 1);
  if (stat == -1) {
    LOG(WARNING) << "Fail to get bus id " << bus_id_str << "'s Compute-mode.";
    return false;
  } else {
    mode = compute_mode[0];
    VLOG(4) << "Bus id " << bus_id_str << "'s Compute-mode is " << mode << ".";
    int ret_close = close(fd);
    if (-1 == ret_close) {
      LOG(WARNING) << "close compute_mode_file failed.";
    }
  }
  return true;
}

void TestEnvironment::setDevice() {
  // 1.get device num
  unsigned int dev_num = 0;
  ASSERT_EQ(cnrtGetDeviceCount(&dev_num), CNRT_RET_SUCCESS);
  if (dev_num <= 0) {  // dev_num_ should > 0
    FAIL() << "Can't find device.";
  } else {
    VLOG(4) << "Found " << dev_num << " devices.";
  }

  // 2.random device id [0, dev_num)
  size_t seed_id = mluoptest::RandomUniformNumber(0, dev_num - 1)();

  // 3.pick device id
  std::vector<DeviceInfo> sorted_device_ids;
  // init cndev
  cndevCheckErrors(cndevInit(0));
  for (unsigned int i = 0; i < dev_num; i++) {
    // get card's process info
    DeviceInfo device_info;
    device_info.device_id = (seed_id + i) % dev_num;
    if (!getBusId(device_info.device_id, device_info.bus_id) ||
        !getComputeMode(device_info.bus_id, device_info.compute_mode)) {
      continue;  // discard the device that cannot get bus id or compute mode
    }
    device_info.process_count = 10;
    cndevProcessInfo_t *procInfo = NULL;
    procInfo = (cndevProcessInfo_t *)malloc(device_info.process_count *
                                            sizeof(cndevProcessInfo_t));
    procInfo->version = CNDEV_VERSION_5;
    cndevDevice_t dev_handle;
    cndevCheckErrors(
        cndevGetDeviceHandleByIndex(device_info.device_id, &dev_handle));
    cndevCheckErrors(
        cndevGetProcessInfo(&device_info.process_count, procInfo, dev_handle));
    free(procInfo);
    VLOG(4) << "Device " << device_info.device_id
            << "'s process count: " << device_info.process_count;
    if ((!global_var.exclusive_ && device_info.compute_mode == '1' &&
         device_info.process_count > 0) ||
        (global_var.exclusive_ && device_info.process_count > 0)) {
      continue;
    }
    sorted_device_ids.push_back(device_info);
  }
  sort(sorted_device_ids.begin(), sorted_device_ids.end(),
       [](const DeviceInfo &lhs, const DeviceInfo &rhs) {
         if (lhs.compute_mode == '1' && rhs.compute_mode == '0') {
           return true;
         } else if (lhs.compute_mode == '0' && rhs.compute_mode == '1') {
           return false;
         } else {
           return lhs.process_count < rhs.process_count;
         }
       });

  bool is_picked = false;
  unsigned int device_id = 0;
  unsigned int process_count = 0;
  if (!global_var.exclusive_) {
    for (int i = 0; i < sorted_device_ids.size(); i++) {
      device_id = sorted_device_ids[i].device_id;
      process_count = sorted_device_ids[i].process_count;
      std::string bus_id = sorted_device_ids[i].bus_id;
      if ((process_count == 0 &&
           (global_var.run_on_jenkins_ || setComputeMode(bus_id, '0'))) ||
          process_count > 0) {
        is_picked = true;
        break;
      }
    }
  } else {
    for (int i = 0; i < sorted_device_ids.size(); i++) {
      device_id = sorted_device_ids[i].device_id;
      process_count = sorted_device_ids[i].process_count;
      std::string bus_id = sorted_device_ids[i].bus_id;
      if (setComputeMode(bus_id, '1')) {
        is_picked = true;
        break;
      }
    }
  }
  if (!is_picked) {
    FAIL() << "No suitable card to pick! Please check card state.";
  }

  // 4.set current device
  // cnrt set current device using CNRT_DEFAULT_DEVICE
  // in cnrtGetDevice() CNRT_DEFAULT_DEVICE > id
  global_var.dev_id_ = device_id;
  VLOG(4) << "Set current device as device: " << global_var.dev_id_
          << ", process count: " << process_count;
  if (global_var.thread_num_ == 1) {
    // if single thread is 1, set current device.
    // if multi thread set current in each thread.
    ASSERT_EQ(cnrtSetDevice(global_var.dev_id_), cnrtSuccess);
  }
}

void TestEnvironment::restoreComputeMode() {
  if (global_var.dev_id_ != -1) {
    if (global_var.exclusive_) {
      // use cnrtDeviceReset to destroy cnrt context before change exclusie_mode
      cnrtDeviceReset();
      std::string bus_id;
      getBusId(global_var.dev_id_, bus_id);
      setComputeMode(bus_id, '0');
    }
  } else {
    VLOG(4) << "No card picked.";
  }
  cndevRelease();
}

void TestEnvironment::setTestEnv() {
  setDate();
  setCommitId();
  setMluOpBranch();
  setMluOpVersion();
  setDriverVersion();
  setCnrtVersion();
  setMluPlatform();
  setClusterLimit();
  setJobLimit();
  setIp();
}

void TestEnvironment::recordEnvXml() {
  testing::Test::RecordProperty("date", test_env_.date);
  testing::Test::RecordProperty("mluop_version", test_env_.mluop_version);
  testing::Test::RecordProperty("mlu_platform", test_env_.mlu_platform);
  testing::Test::RecordProperty("job_limit", test_env_.job_limit);
  testing::Test::RecordProperty("cluster_limit", test_env_.cluster_limit);
  testing::Test::RecordProperty("commit_id", test_env_.commit_id);
  testing::Test::RecordProperty("mluop_branch", test_env_.mluop_branch);
  testing::Test::RecordProperty("driver_version", test_env_.driver_version);
  testing::Test::RecordProperty("cnrt_version", test_env_.cnrt_version);
  testing::Test::RecordProperty("ip", test_env_.ip);
  if (global_var.repeat_ != 0) {
    testing::Test::RecordProperty("repeat_count", global_var.repeat_);
  }
}

void TestEnvironment::recordEnvLog() {
  std::cout << "[date                   ]: " << test_env_.date << std::endl;
  std::cout << "[mluop_version           ]: " << test_env_.mluop_version
            << std::endl;
  std::cout << "[mlu_platform           ]: " << test_env_.mlu_platform
            << std::endl;
  std::cout << "[job_limit              ]: " << test_env_.job_limit
            << std::endl;
  std::cout << "[cluster_limit          ]: " << test_env_.cluster_limit
            << std::endl;
  std::cout << "[commit_id              ]: " << test_env_.commit_id
            << std::endl;
  std::cout << "[mluop_branch            ]: " << test_env_.mluop_branch
            << std::endl;
  std::cout << "[driver_version         ]: " << test_env_.driver_version
            << std::endl;
  std::cout << "[cnrt_version           ]: " << test_env_.cnrt_version
            << std::endl;
  std::cout << "[ip                     ]: " << test_env_.ip << std::endl;
  if (global_var.repeat_ != 0) {
    std::cout << "[repeat_count           ]: " << global_var.repeat_
              << std::endl;
  }
}

void TestEnvironment::showSummary() {
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
