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
#ifndef TEST_MLU_OP_GTEST_INCLUDE_HARDWARE_MONITOR_H_
#define TEST_MLU_OP_GTEST_INCLUDE_HARDWARE_MONITOR_H_

#include <atomic>
#include <condition_variable>  // NOLINT
#include <fstream>
#include <functional>
#include <list>
#include <vector>
#include <thread>  // NOLINT
#include <string>
#include <tuple>

#define HARDWARE_TIME_CSV "hardware_time.csv"
#define MONITOR_CLOCK std::chrono::high_resolution_clock

namespace mluoptest {

struct MonitorStatusParam {
  bool is_device_chosen = false;
  bool finish_one_grepeat = false;
  std::atomic_int num_monitor_threads_started{0};
  std::atomic_int num_monitor_threads_stopped{0};
  std::condition_variable monitor_cv;
  std::mutex monitor_mutex;
};

class hardwareMonitor {
 public:
  hardwareMonitor(const hardwareMonitor &) = delete;
  void operator=(const hardwareMonitor &) = delete;
  ~hardwareMonitor();

  static hardwareMonitor *getInstance();

  void start();                        // by main thread
  void signalMonitorOneGRepeatDone();  // by main thread
  void signalMonitorDeviceChosen();    // by main thread
  void recordHwtime(const std::list<std::tuple<size_t, float>>
                        &hwtime_list);  // by main thread

 private:
  bool finish_all_tests;
  bool monitor_mlu_hardware;  // used in dtr when global_var might be destroyed,
                              // so keep a copy
  bool is_clock_steady;
  size_t start_time_point;
  const std::string results_dir = "monitor_mlu_hardware";
  MonitorStatusParam status;
  std::ofstream hwtime_file;
  std::vector<std::thread> monitor_threads;

  hardwareMonitor()
      : finish_all_tests(false),
        is_clock_steady(true),
        start_time_point(MONITOR_CLOCK::now().time_since_epoch().count()) {}
  // XXX(zhaolianshui): should be equal to monitor_threads.size()
  int getNumMonitorThreads() const { return 3; }
  void setDevice() const;
  void setHwtimeOfstream(bool monitor_hwtime);
  void monitorAllGRepeat(std::function<void()>, bool);
  void monitorFrequencyOneGRepeat();
  void monitorPowerOneGRepeat();
  void monitorHwtimeOneGRepeat();
  void checkClockSteady(const size_t tp);
  void stop() { finish_all_tests = true; }
};

extern hardwareMonitor *monitor;

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_INCLUDE_HARDWARE_MONITOR_H_
