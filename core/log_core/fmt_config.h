/*************************************************************************
 * Copyright (C) [2026] by Cambricon, Inc.
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
#ifndef CORE_LOG_CORE_FMT_CONFIG_H_
#define CORE_LOG_CORE_FMT_CONFIG_H_

// fmt 统一选择策略：全仓库（host 与 BANG device 侧）一律使用
// third_party/fmt 这一份外部 fmt（SPDLOG_FMT_EXTERNAL），不再使用
// spdlog bundled 副本。理由：
//   - 单一事实来源：两侧版本永远一致，杜绝同名 fmt::v11 弱符号跨版本
//     混链（ODR）。曾因 <fmt/ostream.h> 漂到系统 fmt 11.0.2 与 bundled
//     11.2.0 在同一 TU 混用，chrono.h 报 use_utf8/copy2 系列错误。
//   - FMT_HEADER_ONLY=1：BANG 目标（.mlu 经 cncc 编译）的 host 侧代码
//     链入 libmluops.so 等不携带 libfmt.a 的产物，必须自包含；与
//     mluopscore 链 fmt::fmt-header-only 是同一模式。
//
// 本头必须在任何 spdlog/fmt 头之前包含（logging.h / log_engine.h 已
// 保证，直接 include fmt/spdlog 头的文件需自行先 include 本头）。
// include 路径由构建保证 third_party/fmt/include 存在：host 目标经
// fmt::fmt / fmt::fmt-header-only 的 INTERFACE 天然具备；BANG 目标因
// FindBANG 不转发 INTERFACE include，需 target_include_directories 显式
// 补——漏配会直接报 "fmt/base.h not found"，不会静默回退 bundled。
#ifndef SPDLOG_FMT_EXTERNAL
#define SPDLOG_FMT_EXTERNAL
#endif
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY 1
#endif

// 期望的 vendored fmt 版本（third_party/fmt，FMT_VERSION 格式 MMmmpp）。
// 升级 third_party/fmt 时同步改这里；log_engine.h 中有 static_assert 编译期
// 校验——防止编译命令漏配 third_party/fmt/include 时，静默漂到版本不符的
// 系统 fmt（如容器内 11.0.2）造成链期弱符号混链。
#define MLUOP_EXPECTED_FMT_VERSION 110200

#endif  // CORE_LOG_CORE_FMT_CONFIG_H_
