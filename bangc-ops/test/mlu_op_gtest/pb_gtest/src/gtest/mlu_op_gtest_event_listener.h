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

#pragma once

#include <iostream>
#include <string>

#include "gtest/internal/custom/gtest.h"
#include "gtest/gtest-spi.h"

#include "variable.h"
#include "evaluator.h"
#include "internal_perf.h"
#include "gtest/gtest.h"

namespace mluoptest {

extern GlobalVar global_var;

class MLUOPGtestInternalPerfEventListener
    : public ::testing::EmptyTestEventListener {
  void OnTestIterationStart(const ::testing::UnitTest& unit_test,
                            int iteration) override {
    GTEST_LOG_(INFO) << __func__ << " " << iteration;

    std::string output_file_name =
        fileprefix + "_" + std::to_string(iteration) + file_ext;

    testing::internal::FilePath output_file(output_file_name);
    testing::internal::FilePath output_dir(output_file.RemoveFileName());
    if (output_dir.CreateDirectoriesRecursively()) {
      fout = testing::internal::posix::FOpen(output_file_name.c_str(), "w");
    }

    if (fout == NULL) {
      GTEST_LOG_(FATAL) << "Unable to open file \"" << output_file_name << "\"";
    }
  }

  void OnTestIterationEnd(const ::testing::UnitTest& unit_test,
                          int iteration) override {
    GTEST_LOG_(INFO) << __func__ << " " << iteration;
    if (fout) {
      fclose(fout);
    }
    fout = NULL;
  }

  void OnTestStart(const ::testing::TestInfo& test_info) override {
    GTEST_LOG_(INFO) << __func__ << " " << test_info.test_case_name() << " "
                     << test_info.name();
  }

  void OnTestCaseStart(const ::testing::TestCase& test_case) override {
    GTEST_LOG_(INFO) << __func__ << " " << test_case.name();
  }

  void OnTestCaseEnd(const ::testing::TestCase& test_case) override {
    GTEST_LOG_(INFO) << __func__ << " " << test_case.name();
  }

  void OnTestPartResult(
      const ::testing::TestPartResult& test_part_result) override {
    GTEST_LOG_(INFO) << __func__ << " " << test_part_result.file_name();
  }

  void OnTestEnd(const ::testing::TestInfo& test_info) override {
    GTEST_LOG_(INFO) << __func__ << " " << test_info.test_case_name() << " "
                     << test_info.name();
    // const testing::TestResult *result = test_info.result();

    /*
     * write info into csv, may multiple csv inside single file
     * for header, prefix with 'x', for data, prefix with '_'
     */
#define CSV_HEADER_TOKEN "x"
#define CSV_BODY_TOKEN "_"
    global_var.internal_info_.iterate_invoke([this](const auto& item) {
      auto _csv_header = item.get_csv_header();
      if (csv_header.empty()) {
        need_csv_header = true;
      } else if (csv_header.compare(_csv_header) != 0) {
        GTEST_LOG_(WARNING) << "csv header changed";
        need_csv_header = true;
      }
      if (need_csv_header) {
        csv_header = std::move(_csv_header);
        fprintf(fout, CSV_HEADER_TOKEN "%s\n", csv_header.c_str());
        need_csv_header = false;
      }
      std::string serialized(item.serialize_to_csv());
      fprintf(fout, CSV_BODY_TOKEN "%s\n", serialized.c_str());
    });
    // clear internal state
    global_var.internal_info_.clear_cases();
#undef CSV_HEADER_TOKEN
#undef CSV_BODY_TOKEN

#if 0
    auto _csv_header(global_var.internal_info_.get_csv_header());
    if (csv_header.empty()) {
      csv_header = _csv_header;
    }
    if (csv_header.compare(_csv_header) != 0) {
      GTEST_LOG_(WARNING) << "csv header changed";
      need_csv_header = true;
    }
    if (need_csv_header) {
      csv_header = std::move(_csv_header);  // NOLINT
      fprintf(fout, "x%s\n", csv_header.c_str());
      need_csv_header = false;
    }
    std::string serialized(global_var.internal_info_.serialize_to_csv());
    fprintf(fout, "_%s\n", serialized.c_str());
#endif
  }

  // XXX At present, just use fixed name for convenience
  bool need_csv_header = true;
  FILE* fout = NULL;
  std::string fileprefix = "mluops_gtest_internal_perf";
  std::string file_ext = ".csv";
  std::string csv_header;
};

}  // namespace mluoptest
