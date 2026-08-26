// 落盘场景回归（MLUOP_LOG_ONLY_SHOW=0）。setLogFile 语义：总是关旧流、trunc 开新流。
//  1. setLogFile 新文件 → trunc 落盘
//  2. 同名重设 → 也 trunc（旧内容清空）
//  3. 改名切换 → 新文件 trunc
//  4. disable 关流 → 释放文件句柄（/proc/self/fd 验证）；re-enable → trunc 重开
//  5. 默认文件 → 同样 trunc（切走再切回，旧内容清空）
// exit 非 0 表示有失败。
#include "core/logging.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <dirent.h>
#include <unistd.h>

static bool file_contains(const char* p, const char* needle) {
  FILE* f = std::fopen(p, "r");
  if (!f) return false;
  char buf[512];
  std::string content;
  while (std::fgets(buf, sizeof(buf), f)) content += buf;
  std::fclose(f);
  return content.find(needle) != std::string::npos;
}

// 是否有打开的 fd 指向 path（遍历 /proc/self/fd 的符号链接）。
static bool fd_open(const char* path) {
  DIR* d = opendir("/proc/self/fd");
  if (!d) return false;
  bool found = false;
  char link[512], target[512];
  for (dirent* e = readdir(d); e != nullptr; e = readdir(d)) {
    std::snprintf(link, sizeof(link), "/proc/self/fd/%s", e->d_name);
    ssize_t n = readlink(link, target, sizeof(target) - 1);
    if (n > 0) {
      target[n] = '\0';
      if (std::strstr(target, path) != nullptr) { found = true; break; }
    }
  }
  closedir(d);
  return found;
}

static int g_fails = 0;
#define CHECK_ITEM(what, cond)                                         \
  do {                                                            \
    bool ok = (cond);                                             \
    std::fprintf(stderr, "%s: %s\n", ok ? "PASS" : "FAIL", what); \
    if (!ok) ++g_fails;                                           \
  } while (0)

int main() {
  // 落盘模式：LogEngine 首次打日志时才读 env，须在首个 LOG 前 setenv。
  setenv("MLUOP_LOG_ONLY_SHOW", "0", 1);
  mluOpSetMaxLogLevel(MLUOP_LOG_LEVEL_DEBUG4);
  std::setvbuf(stdout, nullptr, _IOFBF, 4096);

  // 1. setLogFile 新文件 → trunc 落盘
  mluOpSetLogFile("/tmp/ghost.txt");
  LOG_ERROR("to-ghost");
  std::fflush(stdout);
  CHECK_ITEM("1 新文件 setLogFile 落盘", file_contains("/tmp/ghost.txt", "to-ghost"));

  // 2. 同名重设 → trunc（旧内容清空）
  mluOpSetLogFile("/tmp/same.txt");
  LOG_ERROR("line1-before-reset");
  mluOpSetLogFile("/tmp/same.txt");  // 同名也 trunc
  LOG_ERROR("line2-after-reset");
  std::fflush(stdout);
  CHECK_ITEM("2 同名重设 trunc（line1 清空）",
        !file_contains("/tmp/same.txt", "line1-before-reset"));
  CHECK_ITEM("2 同名重设 trunc（line2 写入）",
        file_contains("/tmp/same.txt", "line2-after-reset"));

  // 3. 改名切换 → 新文件 trunc
  mluOpSetLogFile("/tmp/other.txt");
  LOG_ERROR("to-other");
  std::fflush(stdout);
  CHECK_ITEM("3 改名切换新文件", file_contains("/tmp/other.txt", "to-other"));

  // 4. disable 关流 → 释放句柄；re-enable → trunc 重开（旧内容清空）
  mluOpSetLogFile("/tmp/toggle.txt");
  LOG_ERROR("toggle-before-disable");
  std::fflush(stdout);
  CHECK_ITEM("4 落盘期间句柄打开", fd_open("/tmp/toggle.txt"));
  mluOpSetLogFile(nullptr);  // disable（关流）
  CHECK_ITEM("4 disable 后句柄释放", !fd_open("/tmp/toggle.txt"));
  LOG_ERROR("toggle-while-disabled");
  mluOpSetLogFile("/tmp/toggle.txt");  // re-enable → trunc 重开
  CHECK_ITEM("4 re-enable 句柄重新打开", fd_open("/tmp/toggle.txt"));
  LOG_ERROR("toggle-after-enable");
  std::fflush(stdout);
  CHECK_ITEM("4 re-enable trunc（before 清空）",
        !file_contains("/tmp/toggle.txt", "toggle-before-disable"));
  CHECK_ITEM("4 re-enable trunc（after 写入）",
        file_contains("/tmp/toggle.txt", "toggle-after-enable"));
  CHECK_ITEM("4 disable 期间不落盘",
        !file_contains("/tmp/toggle.txt", "toggle-while-disabled"));

  // 5. 默认文件 → 同样 trunc（切走再切回，旧内容清空）
  mluOpSetLogFile("mlu_op_auto_log");
  LOG_ERROR("default-line-1");
  mluOpSetLogFile("/tmp/other2.txt");
  LOG_ERROR("temp");
  mluOpSetLogFile("mlu_op_auto_log");  // 切回默认文件 → trunc
  LOG_ERROR("default-line-2");
  std::fflush(stdout);
  CHECK_ITEM("5 默认文件 trunc（旧内容清空）",
        !file_contains("mlu_op_auto_log", "default-line-1"));
  CHECK_ITEM("5 默认文件 trunc（新内容写入）",
        file_contains("mlu_op_auto_log", "default-line-2"));

  // 6. 空串 → BAD_PARAM，落盘状态不变（仍写当前文件）
  CHECK_ITEM("6 空串返回 BAD_PARAM", mluOpSetLogFile("") == MLUOP_STATUS_BAD_PARAM);
  LOG_ERROR("after-empty-string");
  std::fflush(stdout);
  CHECK_ITEM("6 空串后落盘不受影响",
        file_contains("mlu_op_auto_log", "after-empty-string"));

  std::fprintf(stderr, "%s\n", g_fails == 0 ? "ALL PASS" : "SOME FAIL");
  return g_fails;
}
