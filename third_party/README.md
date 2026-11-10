# third_party 依赖清单

本目录内嵌（vendored）日志引擎依赖的两个三方库，构建零网络依赖。

| 库 | 版本 | 上游 | 用途 | 本地修改 |
|---|---|---|---|---|
| fmt | 11.2.0（FMT_VERSION 110200） | github.com/fmtlib/fmt，tag 11.2.0 | 日志格式化，header-only（CMake 目标 fmt::fmt-header-only） | 无 |
| spdlog | 1.15.3 | github.com/gabime/spdlog，tag v1.15.3 | 日志后端（add_subdirectory 静态库） | 无 |

## 接线方式

- 核心库只链接 fmt 的 header-only 目标；spdlog 经 `SPDLOG_FMT_EXTERNAL_HO ON`
  复用外部 fmt（见 `cmake/log_setup.cmake`），`spdlog/include/spdlog/fmt/bundled/`
  不参与编译。
- 外部 fmt 是全库唯一 fmt 实体；`core/logging.h`、`core/log_core/log_engine.*`
  直接 `#include <fmt/...>`。
- 外部 bench（cpu_bench）等价方式：`-DFMT_HEADER_ONLY` + 两个 include 路径。

## 版本注意

- bundled fmt 与外部 fmt 当前同为 11.2.0；bundled 未参与编译，版本不一致也无碍。
- spdlog 对外部 fmt 无显式版本约束（其 CMakeLists 为
  `find_package(fmt CONFIG REQUIRED)`），跨大版本升级 fmt 后必须全量回归。

## 升级流程

1. 下载新版本 release tarball，整体替换对应目录（保留本文件）；
2. 更新上表版本号；
3. 重新构建，跑 `test/logging/` 功能回归（smoke_log / verify_logfile /
   verify_exit_fmt，构建运行方式见其 README.md）；
4. 检查上游 CMake 变更（target 名、`SPDLOG_FMT_EXTERNAL_HO` 等宏语义）。
