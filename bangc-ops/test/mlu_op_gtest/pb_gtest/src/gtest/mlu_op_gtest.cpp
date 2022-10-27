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
#include <functional>
#include <queue>
#include <set>
#include "mlu_op_gtest.h"
#include "op_register.h"

extern mluoptest::GlobalVar global_var;
std::string TestSuite::op_name_ = "";  // NOLINT
std::vector<std::string> TestSuite::case_path_vec_ = {};
std::shared_ptr<mluoptest::ExecuteConfig> TestSuite::ecfg_ =
    std::make_shared<mluoptest::ExecuteConfig>();
// depends on thread num.
std::shared_ptr<mluoptest::ExecuteContext> TestSuite::ectx_ = nullptr;
// setup for 1 op
void TestSuite::SetUpTestCase() {
  // get op name and case list.
  auto test_case = UnitTest::GetInstance()->current_test_case();
  auto case_name = std::string(test_case->name());
  op_name_ = case_name.substr(0, case_name.find_first_of("/"));
  case_path_vec_ = Collector(op_name_).list();

  // record info.
  global_var.summary_.suite_count += 1;
  global_var.summary_.case_count += case_path_vec_.size();

  // exe config
  ecfg_->perf_repeat = global_var.repeat_;

  // exe context
  // if thread is 1, prepare 1 execute_context(handle queue...).
  // and all case (Thread1()) share these variable.
  // so use this global variable ectx_
  if (global_var.thread_num_ == 1) {
    ectx_ = std::make_shared<mluoptest::ExecuteContext>();
    ectx_->init();
  }
}

// teardown for 1 op
void TestSuite::TearDownTestCase() {
  if (ectx_ != nullptr) {  // only for thread 1 actually.
    ectx_->destroy();
    ectx_.reset();
  }

  op_name_.clear();
  case_path_vec_.clear();
}

void TestSuite::TearDown() {
  // print result on screen or to file.
  for (auto it = res_.begin(); it != res_.end(); ++it) {  // for case_path
    report(*it);
  }
  res_.clear();
}

void TestSuite::Thread1() {
  size_t case_idx = std::get<1>(GetParam());
  auto case_path = case_path_vec_[case_idx];
  try {
    auto exe = getOpExecutor(op_name_);
    exe->result()->op_name = op_name_;
    exe->init(ectx_);
    exe->setup(case_path_vec_[case_idx], ecfg_);
    exe->launch();
    exe->sync();
    auto res = exe->teardown();
    res_.emplace_back(res);
    if (global_var.get_vmpeak_ != "") {
      std::ofstream get_vmpeak_oss;
      get_vmpeak_oss.open(global_var.get_vmpeak_, std::ios::app);
      get_vmpeak_oss << op_name_ << "|" << case_path_vec_[case_idx] << "|"
                     << mluoptest::proc_usage_peak() << std::endl;
      get_vmpeak_oss.close();
    }
  } catch (std::exception &e) {
    ectx_->reset();

    mluoptest::EvaluateResult res;
    res.op_name = case_path;
    res.case_path = case_path;
    res.what.emplace_back(
        "Unknown error: maybe exception raised, other info is lost.");
    res_.emplace_back(res);
    ADD_FAILURE() << "MLUOP GTEST: catched " << e.what()
                  << " in single thread mode. (of " << case_path << ")";
  }
}

// wrap a executor and it status flag
// task buffer is a vector and each element is an ExecutorWrap.
// when setup, set in_used as true, then setup(), then set exe.
// when it's ready, set been_chosen as true, then teardown, then reset().
struct ExecutorWrap {
  ExecutorWrap() = default;
  explicit ExecutorWrap(std::shared_ptr<mluoptest::Executor> e) : exe(e) {
    in_used = true;
  }

  std::shared_ptr<mluoptest::Executor> exe = nullptr;
  // flag for teardown polling and pick up.
  // if 1 thread choose this exe, other thread shouldn't choose it.
  bool been_chosen = false;
  // flag for setup
  // if 1 thread choose this exe, other thread shouldn't choose it.
  bool in_used = false;

  void used() { in_used = true; }
  void set(std::shared_ptr<mluoptest::Executor> e) { exe = e; }
  void reset() {
    been_chosen = false;
    in_used = false;
    exe = nullptr;
  }
  bool is_free() { return !in_used; }
  // ready means ready to teardown:
  // exe is not null and not been chose by other thread and is ready.
  bool ready() {
    if (exe == nullptr) {
      return false;
    } else if (been_chosen == true) {
      return false;
    } else {
      return exe->ready();  // kernel is done, is ready.
    }
  }
};

// wrap a executor context and it status flag
// executor context encapsulates handle queue ... and anything can share.
// ecw_vec is a vector and each element is an ExecuteContextWrap.
// when setup, set in_used as true, then setup(), then set exe.
// when it's ready, set been_chosen as true, then teardown, then reset().
struct ExecuteContextWrap {
  std::shared_ptr<mluoptest::ExecuteContext> ectx = nullptr;
  void init() {
    if (ectx == nullptr) {
      ectx = std::make_shared<mluoptest::ExecuteContext>();
      ectx->init();
      ectx->cmp = std::make_shared<mluoptest::CPUMemoryPool>();
      ectx->mmp = std::make_shared<mluoptest::MLUMemoryPool>();
    }
  }
  void destroy() {
    if (ectx != nullptr) {
      ectx->destroy();
      ectx->cmp.reset();
      ectx->mmp.reset();
      ectx.reset();
    }
  }
  void reset() { ectx->reset(); }
};

struct Context {
  explicit Context(size_t buffer_num) {
    ecw_vec.resize(buffer_num);
    exe_vec.resize(buffer_num);
    for (auto it = ecw_vec.begin(); it != ecw_vec.end(); ++it) {
      (*it) = std::make_shared<ExecuteContextWrap>();
    }
    for (auto it = exe_vec.begin(); it != exe_vec.end(); ++it) {
      (*it) = std::make_shared<ExecutorWrap>();
    }
  }

  void destroy() {
    exe_vec.clear();
    for (auto it = ecw_vec.begin(); it != ecw_vec.end(); ++it) {
      // free each context (handle queue .. in it)
      (*it)->destroy();
    }
    results.clear();
  }

  std::mutex mtx;  // modify anything, remember lock it by this mtx.
  std::condition_variable cond;

  // 1 to 1 corresponding
  // exe_vec saved executor
  // and ecw_vec saved context for executor.
  // their size is same, and exe_vec[0] will use context in ecw_vec[0]
  std::vector<std::shared_ptr<ExecuteContextWrap>> ecw_vec;
  // this is so-called task queue, but kernel's latency is not same,
  // so it is not fifo. so use vector.
  std::vector<std::shared_ptr<ExecutorWrap>> exe_vec;

  std::list<mluoptest::EvaluateResult> results;
  // set current device for all thread.
  std::set<std::thread::id, std::greater<std::thread::id>> been_initialized;
};

void TestSuite::ThreadX() {
  size_t thread_num = global_var.thread_num_;
  size_t max_exe_vec_num = thread_num * 1.5;
  auto thread_pool = std::make_shared<mluoptest::ThreadPool>(thread_num);
  auto context = std::make_shared<Context>(max_exe_vec_num);

  // set device for each thread.
  auto set_device = [](std::shared_ptr<Context> ctx) {
    std::lock_guard<std::mutex> lk(ctx->mtx);
    auto it = ctx->been_initialized.find(std::this_thread::get_id());
    // if current thread has not been set device.
    if (it == ctx->been_initialized.end()) {
      ASSERT_EQ(cnrtSetDevice(global_var.dev_id_), CNRT_RET_SUCCESS);
      ctx->been_initialized.insert(std::this_thread::get_id());
    }
  };

  // find executor that is done
  auto has_done = [](std::shared_ptr<Context> ctx) -> bool {
    auto it =
        find_if(ctx->exe_vec.begin(), ctx->exe_vec.end(),
                [](std::shared_ptr<ExecutorWrap> ew) { return ew->ready(); });
    if (it != ctx->exe_vec.end()) {
      return true;
    } else {
      return false;
    }
  };

  // return unique item.
  // buffer is thread num * 1.5 wont exceed int64_t
  auto any_done = [](std::shared_ptr<Context> ctx) -> int64_t {
    std::lock_guard<std::mutex> lk(ctx->mtx);
    auto it =
        find_if(ctx->exe_vec.begin(), ctx->exe_vec.end(),
                [](std::shared_ptr<ExecutorWrap> ew) { return ew->ready(); });
    if (it != ctx->exe_vec.end()) {
      (*it)->been_chosen = true;
      // mark this exe been chosen, and other thread shouldn't choose it.
      return std::distance(ctx->exe_vec.begin(), it);
    } else {
      return -1;
    }
  };

  auto teardown = [](size_t id, std::shared_ptr<Context> ctx) {
    mluoptest::EvaluateResult res;
    auto exe = ctx->exe_vec[id]->exe;
    try {
      exe->sync();
      res = exe->teardown();
    } catch (std::exception &e) {
      ctx->ecw_vec[id]->reset();  // reset running env

      res = *(exe->result());
      res.what.emplace_back(
          "Unknown error: maybe exception raised, other info is lost.");
      ADD_FAILURE() << "MLUOP GTEST: catched " << e.what()
                    << " in teardown. (of " << res.case_path
                    << ") tid: " << std::this_thread::get_id();
    }
    printf("[ TEARDOWN ]: %s\n",
           res.case_path.c_str());  // printf is thread-safe
    exe.reset();                    // free this exe.
    {
      std::lock_guard<std::mutex> lk(ctx->mtx);
      ctx->exe_vec[id]->reset();  // reset this position as idle.
      ctx->results.emplace_back(res);
    }
    // this thread is free, wake master thread to schedule new task.
    ctx->cond.notify_all();
  };

  auto setup = [](std::string op_name, std::string case_path,
                  std::shared_ptr<Context> ctx, size_t pos) {
    printf("[ SETUP    ]: %s\n", case_path.c_str());  // printf is thread-safe
    // get corresponding executor context which saved handle queue ...
    auto ecw = ctx->ecw_vec[pos];
    ecw->init();  // if initialized, this func will return directly.

    // run
    try {
      auto exe = getOpExecutor(op_name);
      exe->result()->op_name = op_name;
      exe->init(ecw->ectx);
      exe->setup(case_path, ecfg_);
      exe->launch();
      {
        std::lock_guard<std::mutex> lk(ctx->mtx);
        ctx->exe_vec[pos]->set(exe);  // push exe into task queue.
      }
    } catch (std::exception &e) {
      ctx->ecw_vec[pos]->reset();  // reset running env
      ctx->exe_vec[pos]->reset();  // mark pos as free

      mluoptest::EvaluateResult res;
      res.op_name = op_name;
      res.case_path = case_path;
      res.what.emplace_back(
          "Unknown error: maybe exception raised, other info is lost.");
      {
        std::lock_guard<std::mutex> lk(ctx->mtx);
        ctx->results.emplace_back(res);
      }
      ADD_FAILURE() << "MLUOP GTEST: catched " << e.what() << " in setup. (of "
                    << res.case_path << ") tid: " << std::this_thread::get_id();
    }
  };

  // set current device for each thread.
  while (context->been_initialized.size() < thread_num) {
    thread_pool->enqueue(set_device, context);
  }

  for (size_t i = 0;;) {
    auto teardown_pos = any_done(context);
    if (teardown_pos != -1) {
      thread_pool->enqueue(teardown, teardown_pos, context);
    } else {
      // find a idle position.
      auto it =
          find_if(context->exe_vec.begin(), context->exe_vec.end(),
                  [](std::shared_ptr<ExecutorWrap> e) { return e->is_free(); });
      if (it != context->exe_vec.end() && i < case_path_vec_.size()) {
        (*it)->used();  // occupy this position
        auto setup_pos = std::distance(context->exe_vec.begin(), it);
        thread_pool->enqueue(setup, op_name_, case_path_vec_[i], context,
                             setup_pos);
        i++;
      } else {
        // task is full, just wait.
        std::unique_lock<std::mutex> lk(context->mtx);
        context->cond.wait_for(lk, std::chrono::milliseconds(1),
                               [=]() { return has_done(context); });
      }
    }

    // all case been launched and buffer is empty, done.
    if (i == case_path_vec_.size()) {
      // find a exe which is not free
      auto it = find_if(
          context->exe_vec.begin(), context->exe_vec.end(),
          [](std::shared_ptr<ExecutorWrap> e) { return !e->is_free(); });
      if (it == context->exe_vec.end()) {
        // if all exe in task buffer is free, done.
        break;
      }
    }
  }

  // join thread pool
  thread_pool.reset();

  // get results.
  res_ = context->results;

  // free all.
  context->destroy();
  context.reset();

  ASSERT_EQ(case_path_vec_.size(), res_.size());
}

void TestSuite::Run() {
  if (global_var.thread_num_ == 1) {
    Thread1();
  } else {
    ThreadX();
  }
}

// * print result
// * is pass?
// * calc average
void TestSuite::report(mluoptest::EvaluateResult eva) {
  if (global_var.repeat_ > 1) {
    print(eva, true);  // print average log.
  } else {
    print(eva, false);  // normal log.
  }
  if (!eva.is_passed) {
    global_var.summary_.failed_list.emplace_back(eva.case_path);
  }
  recordXml(eva);
  bool passed = eva.is_passed;
  EXPECT_TRUE(passed);
}

std::string showFormula(mluoptest::Evaluator::Formula f) {
  switch (f) {
    case mluoptest::Evaluator::Formula::DIFF1:
      return "DIFF1";
    case mluoptest::Evaluator::Formula::DIFF2:
      return "DIFF2";
    case mluoptest::Evaluator::Formula::DIFF3:
      return "DIFF3";
    case mluoptest::Evaluator::Formula::DIFF3_2:
      return "DIFF3_2";
    case mluoptest::Evaluator::Formula::DIFF4:
      return "DIFF4";
    default:
      GTEST_CHECK(false,
                  "MLUOP GTEST: got an unsupported formula when print it.");
  }
}

void TestSuite::print(mluoptest::EvaluateResult eva, bool average) {
  std::cout << "\n";
  if (true == average) {
    std::cout << "[Average MLU Hardware Time     ]: " << eva.mlu.hardware_time
              << " (us)\n";
    std::cout << "[Average MLU Interface Time    ]: " << eva.mlu.interface_time
              << " (us)\n";
    std::cout << "[Average MLU IO Efficiency     ]: " << eva.mlu.io_efficiency
              << "\n";
    std::cout << "[Average MLU Compute Efficiency]: "
              << eva.mlu.compute_efficiency << "\n";
    std::cout << "[Average MLU Workspace Size    ]: " << eva.mlu.workspace_size
              << " (Bytes)\n";
    std::cout << "[MLU TheoryOps         ]: " << eva.mlu.theory_ops
              << " (Ops)\n";
    std::cout << "[MLU TheoryIOs         ]: " << eva.mlu.theory_io
              << " (Bytes)\n";
    std::cout << "[MLU ComputeForce      ]: " << eva.mlu.compute_force
              << " (op/s)\n";
    std::cout << "[MLU IoBandWidth       ]: " << eva.mlu.io_bandwidth
              << " (GB/s)\n";
  } else {
    std::cout << "[MLU Hardware Time     ]: " << eva.mlu.hardware_time
              << " (us)\n";
    std::cout << "[MLU Interface Time    ]: " << eva.mlu.interface_time
              << " (us)\n";
    std::cout << "[MLU IO Efficiency     ]: " << eva.mlu.io_efficiency << "\n";
    std::cout << "[MLU Compute Efficiency]: " << eva.mlu.compute_efficiency
              << "\n";
    std::cout << "[MLU Workspace Size    ]: " << eva.mlu.workspace_size
              << " (Bytes)\n";
    std::cout << "[MLU TheoryOps         ]: " << eva.mlu.theory_ops
              << " (Ops)\n";
    std::cout << "[MLU TheoryIOs         ]: " << eva.mlu.theory_io
              << " (Bytes)\n";
    std::cout << "[MLU ComputeForce      ]: " << eva.mlu.compute_force
              << " (op/s)\n";
    std::cout << "[MLU IoBandWidth       ]: " << eva.mlu.io_bandwidth
              << " (GB/s)\n";
  }

  auto print_error = [](std::vector<mluoptest::Evaluator::ErrorWrap> errors) {
    std::cout << "[Diffs]:\n";
    std::string name = "";
    for (auto error : errors) {
      if (error.name != name) {
        name = error.name;
        std::cout << "[" << name << "]\n";
      }
      auto func = showFormula(error.criterion.formula);
      std::ostringstream oss;  //  keep precision
      oss.setf(std::ios::scientific);
      oss << error.error;
      std::cout << func << ": " << oss.str() << "\n";
    }
  };

  if (eva.is_passed) {  // if passed, just print errors.
    print_error(eva.errors);
    std::cout << "[^      OK ] " << eva.case_path << "\n";
  } else {  // if failed.
    if (!eva.errors.empty()) {
      print_error(eva.errors);
    }
    for (auto line : eva.what) {
      std::cout << line << "\n";
    }
    std::cout << "[^  FAILED ] " << eva.case_path << "\n";
  }
}

void TestSuite::recordXml(mluoptest::EvaluateResult er) {
  this->RecordProperty("op_name", op_name_);
  if (global_var.repeat_ != 0) {
    this->RecordProperty("repeat_count", global_var.repeat_);
  }

  // save date(Year_month_day_hour_minute_second).
  std::ostringstream date_oss;
  char ymd_time[24];
  time_t timep;
  time(&timep);
  strftime(ymd_time, sizeof(ymd_time), "%Y_%m_%d_%H_%M_%S", localtime(&timep));
  date_oss << ymd_time;
  this->RecordProperty("date", date_oss.str());

  // save case_path
  this->RecordProperty("case_path", er.case_path);

  // save mlu platform.
  std::ostringstream mlu_platform_oss;
  char dev_name[64];
  cnDeviceGetName(dev_name, 64, 0);
  mlu_platform_oss << dev_name;
  this->RecordProperty("mlu_platform", mlu_platform_oss.str());

  // hardware_time is latency in generator.
  std::ostringstream mhwb_oss;
  mhwb_oss << std::setprecision(10) << er.mlu.hardware_time_base;
  this->RecordProperty("hardware_time_base", mhwb_oss.str());

  // save case_name and hardware_time_base
  std::string::size_type start = er.case_path.find_last_of("/");
  std::string case_name;
  if (start != std::string::npos) {
    start += 1;
    case_name = er.case_path.substr(start);
  } else {
    case_name = er.case_path;
  }
  this->RecordProperty(case_name, mhwb_oss.str());

  std::ostringstream mws_oss;
  mws_oss << std::setprecision(10) << er.mlu.workspace_size;
  this->RecordProperty("workspace_size_mlu", mws_oss.str());

  std::ostringstream mhw_oss;
  mhw_oss << std::setprecision(10) << er.mlu.hardware_time;
  this->RecordProperty("hardware_time_mlu", mhw_oss.str());

  std::ostringstream interface_oss;
  interface_oss << std::setprecision(10) << er.mlu.interface_time;
  this->RecordProperty("interface_time_mlu", interface_oss.str());

  std::ostringstream mie_oss;
  mie_oss << std::setprecision(10) << er.mlu.io_efficiency;
  this->RecordProperty("io_efficiency_mlu", mie_oss.str());

  std::ostringstream mce_oss;
  mce_oss << std::setprecision(10) << er.mlu.compute_efficiency;
  this->RecordProperty("compute_efficiency_mlu", mce_oss.str());

  std::ostringstream theory_ops_oss;
  theory_ops_oss << std::setprecision(10) << er.mlu.theory_ops;
  this->RecordProperty("theory_ops", theory_ops_oss.str());

  std::ostringstream theory_ios_oss;
  theory_ios_oss << std::setprecision(10) << er.mlu.theory_io;
  this->RecordProperty("theory_ios", theory_ios_oss.str());

  std::ostringstream compute_force_oss;
  compute_force_oss << std::setprecision(10) << er.mlu.compute_force;
  this->RecordProperty("compute_force", compute_force_oss.str());

  std::ostringstream io_bandwidth_oss;
  io_bandwidth_oss << std::setprecision(10) << er.mlu.io_bandwidth;
  this->RecordProperty("io_bandwidth", io_bandwidth_oss.str());

  auto errors = er.errors;
  for (auto it : errors) {
    auto name = it.name;
    auto error = it.error;
    auto func = showFormula(it.criterion.formula);
    std::transform(func.begin(), func.end(), func.begin(), ::tolower);
    auto key = name + "_error_" + func;  // output1_error1_diff1
    std::ostringstream error_oss;
    error_oss.setf(std::ios::scientific);
    error_oss << error;
    this->RecordProperty(key, error_oss.str());
  }
}
