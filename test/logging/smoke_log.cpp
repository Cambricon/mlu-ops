// 综合冒烟（默认 only_show=true、只屏幕，不落盘）：屏幕打印、callback（body +
// 原始 severity）、only_show=true 时 setLogFile 不创建文件。exit 非 0 表示失败。
// 落盘场景的回归见 verify_logfile（env 控制落盘）。
#include "core/logging.h"
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

static bool exists(const char* p) {
  struct stat st;
  return stat(p, &st) == 0;
}
static bool file_contains(const char* p, const char* needle) {
  FILE* f = std::fopen(p, "r");
  if (!f) return false;
  char buf[512];
  std::string content;
  while (std::fgets(buf, sizeof(buf), f)) content += buf;
  std::fclose(f);
  return content.find(needle) != std::string::npos;
}

static int g_cb_count = 0;
static void my_cb(int sev, const char* fn, const char* msg) {
  ++g_cb_count;
  // message 为 body（无 head，不含 "[MLU-OPS]"）；severity 为原始 Severity
  //（LOG_ERROR → kError=0，非塌缩后的 spdlog level）。
  bool ok = (msg != nullptr) && (strstr(msg, "cb-body") != nullptr) &&
            (strstr(msg, "[MLU-OPS]") == nullptr) && (sev == 0);
  std::fprintf(stderr, "[CB] sev=%d ok=%d msg=%s\n", sev, ok, msg);
  if (!ok) {
    std::fprintf(stderr, ">>> callback body/severity check FAILED\n");
    std::exit(1);
  }
}

static int g_fails = 0;
#define CHECK_ITEM(what, cond)                                         \
  do {                                                            \
    bool ok = (cond);                                             \
    std::fprintf(stderr, "%s: %s\n", ok ? "PASS" : "FAIL", what); \
    if (!ok) ++g_fails;                                           \
  } while (0)

int main() {
  std::setvbuf(stdout, nullptr, _IOFBF, 4096);
  mluOpSetMaxLogLevel(MLUOP_LOG_LEVEL_DEBUG4);
  std::remove("/tmp/ghost.txt");  // 清掉历史运行残留，保证下面的"默认不落盘"检查

  // 屏幕打印（带 head）
  LOG_ERROR("screen-line {} {}", 1, 2.5);
  std::fflush(stdout);

  // callback：body + 原始 severity（kError=0）
  mluOpSetLogCallback(my_cb);
  LOG_ERROR("cb-body arg={}", 42);
  CHECK_ITEM("callback 收到 body + 原始 severity", g_cb_count == 1);
  mluOpSetLogCallback(nullptr);

  // 默认（未 setLogFile）不落盘；setLogFile 后启用落盘并写入。
  CHECK_ITEM("默认（未 setLogFile）不落盘", !exists("/tmp/ghost.txt"));
  mluOpSetLogFile("/tmp/ghost.txt");
  LOG_ERROR("now-to-ghost");
  std::fflush(stdout);
  CHECK_ITEM("setLogFile 启用落盘并写入", file_contains("/tmp/ghost.txt", "now-to-ghost"));

  std::fprintf(stderr, "%s\n", g_fails == 0 ? "ALL PASS" : "SOME FAIL");
  return g_fails;
}
