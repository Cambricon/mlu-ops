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
#ifndef CORE_LOG_CORE_LOG_PROJECT_H_
#define CORE_LOG_CORE_LOG_PROJECT_H_

#include "core/macros.h"

// 分支预测提示：log_core 内部统一用 LOG_PREDICT_*，此处转发到项目层
// core/macros.h 的宏；迁移项目时改本文件的映射即可。
#define LOG_PREDICT_TRUE(x) MLUOP_PREDICT_TRUE(x)
#define LOG_PREDICT_FALSE(x) MLUOP_PREDICT_FALSE(x)

// 命名空间：log_core 一律用 BEGIN/END_LOG_NAMESPACE 包裹；迁移项目时改这里
// 即整体切换（log_core 其他文件不得再出现项目限定的 namespace 字样）。
#define BEGIN_LOG_NAMESPACE namespace mluop { namespace log {
#define END_LOG_NAMESPACE } }  // namespace mluop::log

BEGIN_LOG_NAMESPACE

// 项目级配置：log_core 唯一的项目适配点。迁移项目时只需改本文件。

// 环境变量：允许打印的最高日志级别（0..7 或 ERROR/WARNING/... 名称）。
inline constexpr const char* kEnvMaxLogLevel = "MLUOP_LOG_MAX_LEVEL";
// 环境变量：是否只输出到屏幕（不落盘）。
inline constexpr const char* kEnvOnlyShow = "MLUOP_LOG_ONLY_SHOW";
// 环境变量：是否彩色打印。
inline constexpr const char* kEnvColorPrint = "MLUOP_LOG_COLOR_PRINT";

// 日志 head 展示名。
inline constexpr const char* kProjectDisplayName = "MLU-OPS";
// 默认落盘文件名（非 only_show 时）。
inline constexpr const char* kLogFileName = "mlu_op_auto_log";

END_LOG_NAMESPACE

#endif  // CORE_LOG_CORE_LOG_PROJECT_H_
