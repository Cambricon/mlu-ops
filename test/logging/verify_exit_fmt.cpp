// 退出期日志路径 alive-flag 验证：
// 全局对象 g_early 在 main 前构造（比 LogEngine 单例早），其析构在 LogEngine 析构
// 之后运行（atexit LIFO）。析构里打 LOG_* 系列日志 —— 此时 LogEngine 已析构、
// g_log_engine_alive_flag=0，宏层 logEngineAlive() 检查应把日志拦下（不调
// instance()，不悬垂、不 UB），程序正常退出。
// 通过标准：exit==0 且 stdout 无 "exit-period"。未修复时 instance() 返回悬垂引用
// → UB（可能崩溃 exit!=0，或悬垂调用碰巧跑出 "exit-period"）。
#include "core/logging.h"
#include <cstdio>

struct EarlyLogger {
  ~EarlyLogger() {
    // LogEngine 已析构（flag=0）：下面三条都应被宏层 logEngineAlive() 拦下。
    LOG_ERROR("exit-period ERROR");
    LOG_INFO("exit-period INFO");
    LOG_DEBUG1("exit-period DEBUG1 {}", 1);
  }
};
EarlyLogger g_early;  // main 前构造 → 比 LogEngine 晚析构

int main() {
  mluOpSetMaxLogLevel(MLUOP_LOG_LEVEL_DEBUG4);
  LOG_ERROR("main-log-hello");  // 触发引擎构造 + 正常打一条
  std::fflush(stdout);
  return 0;  // 退出：先 ~LogEngine（flag=0），再 ~g_early（打 FMT，被拦）
}
