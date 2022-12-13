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
#ifndef TEST_MLU_OP_GTEST_SRC_GTEST_TEST_ENV_H_
#define TEST_MLU_OP_GTEST_SRC_GTEST_TEST_ENV_H_

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <tuple>
#include <random>
#include <libxml/xpath.h>  // NOLINT
#include "cn_api.h"
#include "gtest/gtest.h"
#include "mlu_op.h"
#include "core/logging.h"
#include "variable.h"

struct TestEnvInfo {
  std::string commit_id;
  std::string mluop_version;
  std::string date;
  std::string ip;
  std::string mluop_branch;
  std::string mlu_platform;
  std::string job_limit;
  std::string cluster_limit;
};

class TestEnvironment : public testing::Environment {
 public:
  virtual void SetUp();
  virtual void TearDown();
  static TestEnvInfo test_env_;

 private:
  void convertBaselineXml2Txt();
  // set environment of test, such as date, mluop_version.
  static void setTestEnv();
  // record environment info to xml.
  static void recordEnvXml();

  static void setDate();
  static void setMluopVersion();
  static void setMluPlatform();
  static void setClusterLimit();
  static void setJobLimit();
  // CommitId, MluopBranch will be null when not a mluops envrionment.
  static void setCommitId();
  static void setMluopBranch();
  // Ip will be null when "ifconfig" is not installed,
  //  such as testing in docker.
  static void setIp();
};

#endif  // TEST_MLU_OP_GTEST_SRC_GTEST_TEST_ENV_H_
