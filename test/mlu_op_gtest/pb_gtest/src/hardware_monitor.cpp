/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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
#include <chrono>  // NOLINT
#include <sstream>
#include "cndev.h"
#include "hardware_monitor.h"
#include "variable.h"

namespace mluoptest {

hardwareMonitor *hardwareMonitor::getInstance() {
  static hardwareMonitor monitor;
  return &monitor;
}

hardwareMonitor::~hardwareMonitor() {
  if (!monitor_mlu_hardware) {
    return;
  }

  stop();

  for (auto &thread : monitor_threads) {
    thread.join();
  }
  GTEST_WARNING(is_clock_steady, "The monitor clock is not steady.");
}

void hardwareMonitor::setDevice() const {
  GTEST_CHECK(cndevInit(0) == CNDEV_SUCCESS);
  ASSERT_EQ(cnrtSetDevice(global_var.dev_id_), cnrtSuccess);
}

void hardwareMonitor::start() {
  monitor_mlu_hardware = global_var.monitor_mlu_hardware_;
  if (!monitor_mlu_hardware) {
    return;
  }
  std::stringstream cmd;
  cmd << "rm -rf " << results_dir << "; mkdir " << results_dir;
  int _ = system(cmd.str().c_str());
  bool monitor_hwtime = false;
  monitor_threads.emplace_back(std::thread([&, this] {
    monitorAllGRepeat(
        std::bind(&hardwareMonitor::monitorFrequencyOneGRepeat, this),
        monitor_hwtime);
  }));
  monitor_threads.emplace_back(std::thread([&, this] {
    monitorAllGRepeat(std::bind(&hardwareMonitor::monitorPowerOneGRepeat, this),
                      monitor_hwtime);
  }));
  monitor_hwtime = true;
  monitor_threads.emplace_back(std::thread([&, this] {
    monitorAllGRepeat(
        std::bind(&hardwareMonitor::monitorHwtimeOneGRepeat, this),
        monitor_hwtime);
  }));
}

void hardwareMonitor::setHwtimeOfstream(bool monitor_hwtime) {
  if (monitor_hwtime) {
    hwtime_file.open(results_dir + "/hwtime_device_" +
                         std::to_string(global_var.dev_id_) + ".csv",
                     std::ios::app);
    hwtime_file << "relative_time(ns),hardware_time(us)\n";
  }
}

void hardwareMonitor::checkClockSteady(const size_t tp) {
  std::lock_guard<std::mutex> lock(status.monitor_mutex);
  if (tp < start_time_point) {
    is_clock_steady = false;
  }
}

void hardwareMonitor::monitorAllGRepeat(std::function<void()> monitorOneGRepeat,
                                        bool monitor_hwtime) {
  auto t_curr = MONITOR_CLOCK::now().time_since_epoch().count();
  checkClockSteady(t_curr);

  while (!finish_all_tests) {
    bool should_monitor = false;
    while (!finish_all_tests) {
      std::unique_lock<std::mutex> lock(status.monitor_mutex);
      if (status.monitor_cv.wait_for(
              lock, std::chrono::microseconds(10),
              [this] { return status.is_device_chosen; })) {
        should_monitor = true;
        break;
      }
    }
    if (!should_monitor) {
      break;
    }

    setDevice();  // each thread has to set device
    setHwtimeOfstream(
        monitor_hwtime);  // has to be ready before the main thread runs tests
    ++status.num_monitor_threads_started;
    status.monitor_cv.notify_all();
    monitorOneGRepeat();
    ++status.num_monitor_threads_stopped;
    status.monitor_cv.notify_all();
  }
}

void hardwareMonitor::monitorFrequencyOneGRepeat() {
  std::ofstream frequency_file(results_dir + "/frequency_device_" +
                                   std::to_string(global_var.dev_id_) + ".csv",
                               std::ios::app);
  frequency_file << "relative_time(ns),IPU_frequency(MHz)\n";
  cndevDevice_t dev_id;
  GTEST_CHECK(cnrtGetDevice(&dev_id) == cnrtSuccess);
  int i = 1;

  cndevFrequencyInfo_t freq_info_prev, freq_info_curr;
  size_t t_prev, t_curr;
  auto getFrequency = [&, this]() {
    freq_info_curr.version = CNDEV_VERSION_5;
    t_curr = MONITOR_CLOCK::now().time_since_epoch().count() - start_time_point;
    GTEST_CHECK(cndevGetFrequencyInfo(&freq_info_curr, dev_id) ==
                CNDEV_SUCCESS);
  };

  MONITOR_CLOCK::time_point t1 = MONITOR_CLOCK::now();
  getFrequency();
  std::tie(t_prev, freq_info_prev) = std::make_tuple(t_curr, freq_info_curr);
  frequency_file << t_prev << "," << freq_info_prev.boardFreq << "\n";

  while (!status.finish_one_grepeat) {
    ++i;
    getFrequency();
    if (freq_info_prev.boardFreq != freq_info_curr.boardFreq) {
      frequency_file << t_prev << "," << freq_info_prev.boardFreq << "\n";
      frequency_file << t_curr << "," << freq_info_curr.boardFreq << "\n";
      freq_info_prev = freq_info_curr;
    }
    t_prev = t_curr;
  }
  frequency_file << t_curr << "," << freq_info_curr.boardFreq << "\n";
  MONITOR_CLOCK::time_point t2 = MONITOR_CLOCK::now();
  auto time_span =
      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(t2 -
                                                                            t1);
  VLOG(4) << "cndevGetFrequencyInfo took " << time_span.count() / i
          << "us per call.";
}

void hardwareMonitor::monitorPowerOneGRepeat() {
  std::ofstream power_file(results_dir + "/power_device_" +
                               std::to_string(global_var.dev_id_) + ".csv",
                           std::ios::app);
  power_file << "relative_time(ns),instantaneous_power(W),average_power(W)\n";
  cndevDevice_t dev_id;
  GTEST_CHECK(cnrtGetDevice(&dev_id) == cnrtSuccess);
  GTEST_CHECK(cndevInit(0) == CNDEV_SUCCESS);
  int i = 1;

  cndevPowerInfo_t power_info_prev, power_info_curr;
  size_t t_prev, t_curr;
  auto getPower = [&, this]() {
    power_info_curr.version = CNDEV_VERSION_5;
    t_curr = MONITOR_CLOCK::now().time_since_epoch().count() - start_time_point;
    // TODO(None): cntoolkit-3.6, use cndevGetDevicePower
    // GTEST_CHECK(cndevGetDevicePower(&power_info_curr, dev_id) ==
    // CNDEV_SUCCESS);
    GTEST_CHECK(cndevGetPowerInfo(&power_info_curr, dev_id) == CNDEV_SUCCESS);
  };

  MONITOR_CLOCK::time_point t1 = MONITOR_CLOCK::now();
  getPower();
  std::tie(t_prev, power_info_prev) = std::make_tuple(t_curr, power_info_curr);
  power_file << t_prev << ","
             << (uint32_t)(power_info_prev.instantaneousPowerUsage) << ","
             << power_info_prev.usage << "\n";
  while (!status.finish_one_grepeat) {
    ++i;
    getPower();
    if (power_info_prev.instantaneousPowerUsage !=
            power_info_curr.instantaneousPowerUsage ||
        power_info_prev.usage != power_info_curr.usage) {
      power_file << t_prev << ","
                 << (uint32_t)(power_info_prev.instantaneousPowerUsage) << ","
                 << power_info_prev.usage << "\n";
      power_file << t_curr << ","
                 << (uint32_t)(power_info_curr.instantaneousPowerUsage) << ","
                 << power_info_curr.usage << "\n";
      power_info_prev = power_info_curr;
    }
    t_prev = t_curr;
  }
  power_file << t_curr << ","
             << (uint32_t)(power_info_curr.instantaneousPowerUsage) << ","
             << power_info_curr.usage << "\n";
  MONITOR_CLOCK::time_point t2 = MONITOR_CLOCK::now();
  auto time_span =
      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(t2 -
                                                                            t1);
  // TODO(None): cntoolkit-3.6, remove this warning
  LOG(WARNING) << "From cntoolkit-3.6 onward, use cndevGetDevicePower.";
  VLOG(4) << "cndevGetDevicePower took " << time_span.count() / i
          << "us per call.";
}

void hardwareMonitor::monitorHwtimeOneGRepeat() {
  {
    std::unique_lock<std::mutex> lock(status.monitor_mutex);
    status.monitor_cv.wait(lock, [this] { return status.finish_one_grepeat; });
  }
  hwtime_file.close();
}

void hardwareMonitor::recordHwtime(
    const std::list<std::tuple<size_t, float>> &hwtime_list) {
  std::lock_guard<std::mutex> lock(status.monitor_mutex);
  for (auto it = hwtime_list.begin(); it != hwtime_list.end(); ++it) {
    hwtime_file << (std::get<0>(*it) - start_time_point) << ","
                << std::get<1>(*it) << "\n";
  }
}

void hardwareMonitor::signalMonitorDeviceChosen() {
  if (!global_var.monitor_mlu_hardware_) {
    return;
  }
  status.is_device_chosen = true;
  status.monitor_cv.notify_all();

  std::unique_lock<std::mutex> lock(status.monitor_mutex);
  status.monitor_cv.wait(lock, [this] {
    return status.num_monitor_threads_started == getNumMonitorThreads();
  });
}

void hardwareMonitor::signalMonitorOneGRepeatDone() {
  if (!global_var.monitor_mlu_hardware_) {
    return;
  }
  status.num_monitor_threads_started = 0;
  status.is_device_chosen = false;
  status.finish_one_grepeat = true;
  status.monitor_cv.notify_all();  // signal hwtime thread

  std::unique_lock<std::mutex> lock(status.monitor_mutex);
  status.monitor_cv.wait(lock, [this] {
    return status.num_monitor_threads_stopped == getNumMonitorThreads();
  });

  status.num_monitor_threads_stopped = 0;
  status.finish_one_grepeat = false;
}

hardwareMonitor *monitor = hardwareMonitor::getInstance();

}  // namespace mluoptest
