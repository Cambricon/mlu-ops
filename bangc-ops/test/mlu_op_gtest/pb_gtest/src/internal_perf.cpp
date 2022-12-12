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

#include "internal_perf.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <utility>
#include <thread>   // NOLINT
#include <iostream>

using namespace mluoptest;  // NOLINT

std::string mluoptest::timeseries_to_array_str(const TimeSeries_t &series) {
  std::string s("[ ");
  auto timepoint2str = [](const TimePoint_t &item) {
    return std::string("[") + "\"" + std::get<0>(item) + "\", " +
           std::to_string(std::get<1>(item)) + "]";
  };
  if (series.size()) {
    s += std::accumulate(
        std::next(series.begin()), series.end(), timepoint2str(series[0]),
        [&timepoint2str](std::string s, const auto &item) {
          return std::move(s) + ", " + std::move(timepoint2str(item));
        });
  }
  s += " ]";
  return s;
}

void TestInternalInfo::record_case(const std::string &case_path,
                                   const GtestInternal &gtest_internal) {
  GtestInternalMsg msg;
  msg.gtest_internal_ = gtest_internal;
  msg.timespan_record_ =
      std::move(evaluate_timespan(msg.gtest_internal_.time_costs_ms));
  msg.case_path_ = case_path;

  cases_info_mutex.lock();
  cases_info_.emplace_back(std::move(msg));
  cases_info_mutex.unlock();
}

std::string GtestInternalMsg::serialize_to_csv(std::string sep) const {
  double size_mb = gtest_internal_.parsed_file_size / 1024. / 1024.;
  auto cost = gtest_internal_.parsed_cost_seconds;
  std::string init = sep + case_path_ + sep + std::to_string(size_mb) + sep +
                     std::to_string(cost) + sep;
  return std::accumulate(
      timespan_record_.begin(), timespan_record_.end(), std::move(init),
      [&sep](std::string s, const auto &item) {
        return std::move(s) + std::to_string(std::get<1>(item)) + sep;
      });
}

std::string GtestInternalMsg::get_csv_header(std::string sep) const {
  std::string init =
      sep + "case+path" + sep + "file_size_mb" + sep + "parse_time_s" + sep;
  return std::accumulate(timespan_record_.begin(), timespan_record_.end(),
                         std::move(init),
                         [&sep](std::string s, const auto &item) {
                           return std::move(s) + std::get<0>(item) + sep;
                         });
}

TimeSeries_t TestInternalInfo::evaluate_timespan(
    const TimeSeries_t &gtest_internal) {
  if (gtest_internal.size() == 0) {
    return {};
  }
  if (gtest_internal.size() == 1) {
    const auto &item = gtest_internal[0];
    return {std::make_tuple(std::get<0>(item), std::get<1>(item))};
  }

  TimeSeries_t ret;
  for (auto it = std::next(gtest_internal.cbegin());
       it != gtest_internal.cend(); it++) {
    const auto &end = *it;
    const auto &begin = *std::prev(it);
    ret.emplace_back(
        std::make_tuple(std::get<0>(begin) + "--" + std::get<0>(end),
                        std::get<1>(end) - std::get<1>(begin)));
  }
  return ret;
}
