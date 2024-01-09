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
#ifndef TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_INTERNAL_PERF_H_
#define TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_INTERNAL_PERF_H_

#include <algorithm>
#include <mutex>  // NOLINT
#include <string>
#include <tuple>
#include <vector>

namespace mluoptest {
using TimePoint_t = std::tuple<std::string, double>;
using TimeSeries_t = std::vector<TimePoint_t>;

/**
 * convert `TimeSeries_t` value into '[ [<key1>,<val1>], [<key1>,<val1>],
 * ...]' format
 */
std::string timeseries_to_array_str(const TimeSeries_t &);

struct GtestInternal {
  size_t parsed_file_size = 0;
  double parsed_cost_seconds = 0.;
  TimeSeries_t time_costs_ms;  // cumulative time on different time point
};

struct GtestInternalMsg {
  std::string serialize_to_csv(std::string sep = "|") const;
  std::string get_csv_header(std::string sep = "|") const;

  GtestInternal gtest_internal_;
  TimeSeries_t timespan_record_;
  std::string case_path_;
};

class TestInternalInfo {
 public:
  /**
   * save gtest internal perf info
   */
  void record_case(const std::string &case_path,
                   const GtestInternal &gtest_internal);

  template <typename Functor>
  void iterate_invoke(Functor f) {
    cases_info_mutex.lock();
    std::for_each(cases_info_.begin(), cases_info_.end(), f);
    cases_info_mutex.unlock();
  }
  void clear_cases() { cases_info_.clear(); }

 private:
  // compute timespan based on cumulative time
  static TimeSeries_t evaluate_timespan(const TimeSeries_t &);
  std::vector<GtestInternalMsg> cases_info_;
  std::mutex cases_info_mutex;
};
}  // namespace mluoptest

#endif  // TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_INTERNAL_PERF_H_
