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

#ifndef CORE_LOGGING_H_
#define CORE_LOGGING_H_

#include <type_traits>
#include <utility>
#include <string>
#include <limits>
#include <sstream>

#include "core/log_core/fmt_config.h"
#include "core/log_core/log_engine.h"
#include "core/macros.h"
#include "core/util.h"
#include "mlu_op.h"
#include "type.h"
#include "preprocessor.h"
#include <algorithm>


#define LARGE_TENSOR_NUM ((uint64_t)2147483648)
#define LARGE_TENSOR_SIZE ((uint64_t)2147483648)

// SEV_ 映射表：旧宏层等级 token（ERROR/WARNING/CNPAPI/INFO/DEBUG1..4/FATAL/VLOG）
// 到新枚举 Severity 常量（kError..kDebug4 / kFatal / kVlog）的显式映射。
#define SEV_ERROR kError
#define SEV_WARNING kWarning
#define SEV_CNPAPI kCnpapi
#define SEV_INFO kInfo
#define SEV_DEBUG1 kDebug1
#define SEV_DEBUG2 kDebug2
#define SEV_DEBUG3 kDebug3
#define SEV_DEBUG4 kDebug4
// #define SEV_FATAL kFatal
#define SEV_VLOG kVlog

#define LOG(severity)                                                   \
  mluop::log::LogMessage(__FILE__, __LINE__,                            \
                         mluop::log::Severity::SEV_##severity).stream()   


                         
// shouldLog 先短路：过滤路径不查 alive；alive 仅挡退出期对已析构单例的输出。
#define LOG_GATED_UNLIKELY(sev, fmt_str, ...) \
  (MLUOP_PREDICT_FALSE( \
       mluop::log::LogEngine::shouldLog(mluop::log::Severity::sev)) && \
       MLUOP_PREDICT_TRUE(mluop::log::logEngineAlive()) \
     ? mluop::log::LogEngine::instance().EmitFmt( \
           mluop::log::Severity::sev, __FILE__, __LINE__, fmt_str, \
           ##__VA_ARGS__) \
     : (void)0)

#define LOG_DEBUG1(fmt_str, ...) \
  LOG_GATED_UNLIKELY(kDebug1, fmt_str, ##__VA_ARGS__)
#define LOG_DEBUG2(fmt_str, ...) \
  LOG_GATED_UNLIKELY(kDebug2, fmt_str, ##__VA_ARGS__)
#define LOG_DEBUG3(fmt_str, ...) \
  LOG_GATED_UNLIKELY(kDebug3, fmt_str, ##__VA_ARGS__)
#define LOG_DEBUG4(fmt_str, ...) \
  LOG_GATED_UNLIKELY(kDebug4, fmt_str, ##__VA_ARGS__)
#define LOG_INFO(fmt_str, ...) \
  LOG_GATED_UNLIKELY(kInfo, fmt_str, ##__VA_ARGS__)
#define LOG_WARNING(fmt_str, ...) \
  LOG_GATED_UNLIKELY(kWarning, fmt_str, ##__VA_ARGS__)
#define LOG_APICALL(fmt_str, ...) \
  LOG_GATED_UNLIKELY(kCnpapi, fmt_str, ##__VA_ARGS__)

#define LOG_ERROR(fmt_str, ...) \
  (MLUOP_PREDICT_TRUE(mluop::log::logEngineAlive()) \
     ? mluop::log::LogEngine::instance().EmitFmt(mluop::log::Severity::kError, \
                                                 __FILE__, __LINE__, fmt_str, \
                                                 ##__VA_ARGS__) \
     : (void)0)

#define TOKENPASTE(x, y, z) x##y##z
#define TOKENPASTE2(x, y, z) TOKENPASTE(x, y, z)

#define LOG_FIRST_N(severity, n)                                            \
  static std::atomic<int> TOKENPASTE2(LOG_, __LINE__, _OCCURRENCES)(0);     \
  if (MLUOP_PREDICT_FALSE(TOKENPASTE2(LOG_, __LINE__, _OCCURRENCES)++ < n)) \
  LOG(severity)

// CHECK with a error if condition is not true.
#define CHECK(condition, ...)                                    \
  if (!(condition)) {                                            \
    LOG(ERROR) << " Check failed: " #condition "." #__VA_ARGS__; \
  }

// CHECK_EQ/NE/...
#define CHECK_EQ(val1, val2, ...)                                         \
  if (!(val1 == val2)) {                                                  \
    LOG(ERROR) << " Check failed: " #val1 " == " #val2 ". " #__VA_ARGS__; \
  }
#define CHECK_NE(val1, val2, ...)                                         \
  if (!(val1 != val2)) {                                                  \
    LOG(ERROR) << " Check failed: " #val1 " != " #val2 ". " #__VA_ARGS__; \
  }
#define CHECK_LE(val1, val2, ...)                                         \
  if (!(val1 <= val2)) {                                                  \
    LOG(ERROR) << " Check failed: " #val1 " <= " #val2 ". " #__VA_ARGS__; \
  }
#define CHECK_LT(val1, val2, ...)                                        \
  if (!(val1 < val2)) {                                                  \
    LOG(ERROR) << " Check failed: " #val1 " < " #val2 ". " #__VA_ARGS__; \
  }
#define CHECK_GE(val1, val2, ...)                                         \
  if (!(val1 >= val2)) {                                                  \
    LOG(ERROR) << " Check failed: " #val1 " >= " #val2 ". " #__VA_ARGS__; \
  }
#define CHECK_GT(val1, val2, ...)                                        \
  if (!(val1 > val2)) {                                                  \
    LOG(ERROR) << " Check failed: " #val1 " > " #val2 ". " #__VA_ARGS__; \
  }

// kernel check for crop
#define SYMBOL_CHECK(symbol...)                                                \
  if (MLUOP_PREDICT_FALSE(!&(symbol))) {                                       \
    LOG(ERROR) << "calling undefined symbol which not be linked: " << #symbol; \
  }                                                                            \
  symbol

// Use cnrtGetLastError() to clear error before launch kernel and set
// return value to CN_SUCCESS, then use cnrtPeekAtLastError() to get
// the error occured when launch kernel. Error now is thread local variable

#define KERNEL_CHECK(kernel...)                                    \
  {                                                                \
    cnrtGetLastError();                                            \
    kernel;                                                        \
    cnrtRet_t ret = cnrtPeekAtLastError();                         \
    if (MLUOP_PREDICT_FALSE(cnrtSuccess != ret)) {                 \
      LOG(ERROR) << "Check failed: Found " << cnrtGetErrorStr(ret) \
                 << " after invoke kernel " #kernel;               \
      return MLUOP_STATUS_EXECUTION_FAILED;                        \
    }                                                              \
  }

// CHECK with return value.
#define INTERNAL_CHECK(api, condition, ...)                           \
  if (!(condition)) {                                                 \
    LOG(ERROR) << api << " An internal error occured. " #__VA_ARGS__; \
    return MLUOP_STATUS_INTERNAL_ERROR;                               \
  }

// CHECK if return value equals MLUOP_STATUS_SUCCESS
#define CHECK_RETURN(api, status, ...)                                    \
  {                                                                       \
    mluOpStatus_t __status__ = (status);                                  \
    if ((__status__) != MLUOP_STATUS_SUCCESS) {                           \
      LOG(ERROR) << api << "BAD return status: " << #status << "returns " \
                 << (__status__) << " (FILE: " << __FILE__                \
                 << ", LINE: " << __LINE__ << "). " __VA_ARGS__;          \
      return (__status__);                                                \
    }                                                                     \
  }

#define PARAM_CHECK(api, condition, ...)                                 \
  if MLUOP_PREDICT_FALSE (!(condition)) {                                \
    LOG(ERROR) << api << " Check failed: " #condition ". " #__VA_ARGS__; \
    return MLUOP_STATUS_BAD_PARAM;                                       \
  }

#define NOT_SUPPORT_BFLOAT16_DATATYPE(interface_name, condition) \
  if (condition) {                                               \
    LOG(ERROR) << interface_name                                 \
               << " This api do not support BFLOAT16 data type " \
               << "temporarily.";                                \
    return MLUOP_STATUS_NOT_SUPPORTED;                           \
  }

// This prints out values instead of names of variables inside __VA_ARGS__
#define PARAM_CHECK_V2(api, condition, ...)                             \
  if (!(condition)) {                                                   \
    LOG(ERROR) << api << " Check failed: " #condition ". " __VA_ARGS__; \
    return MLUOP_STATUS_BAD_PARAM;                                      \
  }


#define PARAM_CHECK_PTR_NOT_NULL(api, ptr, ...)                                \
  do {                                                                         \
    if (ptr == nullptr) {                                                      \
      LOG_ERROR("{} Bad param: {} cannot be NULL. {}", api, #ptr,              \
                #__VA_ARGS__);                                                 \
      return MLUOP_STATUS_BAD_PARAM;                                           \
    }                                                                          \
  } while (0)

#define PARAM_CHECK_PTR_MUST_NULL(api, ptr, ...)                               \
  do {                                                                         \
    if (ptr != nullptr) {                                                      \
      LOG_ERROR("{} Bad param: expect argument {} to be NULL, but a "          \
                "non-null pointer({}) was provided. {}",                       \
                api, #ptr, static_cast<const void*>(ptr), #__VA_ARGS__);       \
      return MLUOP_STATUS_BAD_PARAM;                                           \
    }                                                                          \
  } while (0)

#define PARAM_CHECK_DTYPE_SAME(api, dtype1, dtype2, ...)                       \
  do {                                                                         \
    if (dtype1 != dtype2) {                                                    \
      LOG_ERROR("{} Bad param: expect {} and {} same, but get {}({}) vs "      \
                "{}({}). {}",                                                  \
                api, #dtype1, #dtype2, #dtype1, getNameOfDataType(dtype1),     \
                #dtype2, getNameOfDataType(dtype2), #__VA_ARGS__);             \
      return MLUOP_STATUS_BAD_PARAM;                                           \
    }                                                                          \
  } while (0)

template <typename T, typename U, typename... Args>
bool is_in_list(const T &value, std::initializer_list<U> list) {
  return std::find_if(list.begin(), list.end(), [&value](const auto &elem) {
           return elem == value;
         }) != list.end();
}

// 空可变参数列表编译期拦截（供 PARAM_CHECK_* 等带变长参数的宏复用）。
template <typename... Ts>
constexpr void assert_va_args_nonempty(Ts...) {
  static_assert(sizeof...(Ts) > 0, "variadic argument list must not be empty");
}

/**
  ctx_msg: 上下文说明信息，用于说明一些额外的信息，辅助用户了解参数具体情况。
  不需要上下文时用无 _WITH_CTX 后缀的版本。
 */
#define PARAM_CHECK_DTYPE_SUPPORT(api, dtype, ...) \
  PARAM_CHECK_DTYPE_SUPPORT_WITH_CTX(api, dtype, "", __VA_ARGS__)

#define PARAM_CHECK_DTYPE_SUPPORT_WITH_CTX(api, dtype, ctx_msg, ...)           \
  do {                                                                         \
    assert_va_args_nonempty(__VA_ARGS__);                                      \
    if (!is_in_list(dtype, {__VA_ARGS__})) {                                   \
      if constexpr (MLUOP_PP_NUM_ARGS(__VA_ARGS__) > 1) {                      \
        LOG_ERROR("{} Bad param: expect {} be one of {{{}}}, but got {}. {}",  \
                  api, #dtype, #__VA_ARGS__, getNameOfDataType(dtype),         \
                  ctx_msg);                                                    \
      } else {                                                                 \
        LOG_ERROR("{} Bad param: expect {} be {}, but got {}. {}",             \
                  api, #dtype, #__VA_ARGS__, getNameOfDataType(dtype),         \
                  ctx_msg);                                                    \
      }                                                                        \
      return MLUOP_STATUS_BAD_PARAM;                                           \
    }                                                                          \
  } while (0)

#define PARAM_CHECK_DTYPE_NOT_SUPPORT(api, dtype, ...) \
  PARAM_CHECK_DTYPE_NOT_SUPPORT_WITH_CTX(api, dtype, "", __VA_ARGS__)

#define PARAM_CHECK_DTYPE_NOT_SUPPORT_WITH_CTX(api, dtype, ctx_msg, ...)       \
  do {                                                                         \
    assert_va_args_nonempty(__VA_ARGS__);                                      \
    if (is_in_list(dtype, {__VA_ARGS__})) {                                    \
      if constexpr (MLUOP_PP_NUM_ARGS(__VA_ARGS__) > 1) {                      \
        LOG_ERROR("{} Bad param: {} in {{{}}} are not supported, but got {}."  \
                  " {}",                                                       \
                  api, #dtype, #__VA_ARGS__, getNameOfDataType(dtype),         \
                  ctx_msg);                                                    \
      } else {                                                                 \
        LOG_ERROR("{} Bad param: {} cannot be {}. {}",                         \
                  api, #dtype, #__VA_ARGS__, ctx_msg);                         \
      }                                                                        \
      return MLUOP_STATUS_BAD_PARAM;                                           \
    }                                                                          \
  } while (0)

// LAYOUT 检查
#define PARAM_CHECK_LAYOUT_SAME(api, layout1, layout2, ...)                    \
  do {                                                                         \
    if (layout1 != layout2) {                                                  \
      LOG_ERROR("{} Bad param: expect {} and {} same, but get {}({}) vs "      \
                "{}({}). {}",                                                  \
                api, #layout1, #layout2, #layout1,                             \
                mluOpGetNameOfTensorLayout(layout1), #layout2,                 \
                mluOpGetNameOfTensorLayout(layout2), #__VA_ARGS__);            \
      return MLUOP_STATUS_BAD_PARAM;                                           \
    }                                                                          \
  } while (0)

#define PARAM_CHECK_LAYOUT_SUPPORT(api, layout, ...) \
  PARAM_CHECK_LAYOUT_SUPPORT_WITH_CTX(api, layout, "", __VA_ARGS__)

#define PARAM_CHECK_LAYOUT_SUPPORT_WITH_CTX(api, layout, ctx_msg, ...)         \
  do {                                                                         \
    assert_va_args_nonempty(__VA_ARGS__);                                      \
    if (!is_in_list(layout, {__VA_ARGS__})) {                                  \
      if constexpr (MLUOP_PP_NUM_ARGS(__VA_ARGS__) > 1) {                      \
        LOG_ERROR("{} Bad param: expect {} be one of {{{}}}, but got {}. {}",  \
                  api, #layout, #__VA_ARGS__,                                  \
                  mluOpGetNameOfTensorLayout(layout), ctx_msg);                \
      } else {                                                                 \
        LOG_ERROR("{} Bad param: expect {} be {}, but got {}. {}",             \
                  api, #layout, #__VA_ARGS__,                                  \
                  mluOpGetNameOfTensorLayout(layout), ctx_msg);                \
      }                                                                        \
      return MLUOP_STATUS_BAD_PARAM;                                           \
    }                                                                          \
  } while (0)

#define PARAM_CHECK_LAYOUT_NOT_SUPPORT(api, layout, ...) \
  PARAM_CHECK_LAYOUT_NOT_SUPPORT_WITH_CTX(api, layout, "", __VA_ARGS__)

#define PARAM_CHECK_LAYOUT_NOT_SUPPORT_WITH_CTX(api, layout, ctx_msg, ...)     \
  do {                                                                         \
    assert_va_args_nonempty(__VA_ARGS__);                                      \
    if (is_in_list(layout, {__VA_ARGS__})) {                                   \
      if constexpr (MLUOP_PP_NUM_ARGS(__VA_ARGS__) > 1) {                      \
        LOG_ERROR("{} Bad param: {} in {{{}}} are not supported, but got {}."  \
                  " {}",                                                       \
                  api, #layout, #__VA_ARGS__,                                  \
                  mluOpGetNameOfTensorLayout(layout), ctx_msg);                \
      } else {                                                                 \
        LOG_ERROR("{} Bad param: {} cannot be {}. {}",                         \
                  api, #layout, #__VA_ARGS__, ctx_msg);                        \
      }                                                                        \
      return MLUOP_STATUS_BAD_PARAM;                                           \
    }                                                                          \
  } while (0)

#define PARAM_CHECK_DIM_RANGE(api, dim, condition, msg)                        \
  do {                                                                         \
    if (!condition) {                                                          \
      LOG(ERROR) << api << " Bad param: " #dim " expect: " #condition ", but " \
                 << dim << " is provided. " << msg;                            \
      return MLUOP_STATUS_BAD_PARAM;                                           \
    }                                                                          \
  } while (0)

#define PARAM_CHECK_DIM_SAME(api, msg, dim1, dim2, ...)                        \
  do {                                                                         \
    const auto &_mluop_dim_base = (dim1);                                      \
    const auto _mluop_dims[] = {dim2, ##__VA_ARGS__};                          \
    size_t _mluop_dims_size = sizeof(_mluop_dims) / sizeof(_mluop_dims[0]);    \
    auto get_dim_string = [_mluop_dim_base, _mluop_dims, _mluop_dims_size]() { \
      std::ostringstream oss;                                                  \
      oss << "{" << _mluop_dim_base << ", ";                                   \
      for (size_t i = 0; i < _mluop_dims_size; ++i) {                          \
        oss << _mluop_dims[i];                                                 \
        if (i + 1 < _mluop_dims_size) {                                        \
          oss << ", ";                                                         \
        }                                                                      \
      }                                                                        \
      oss << "}";                                                              \
      return oss.str();                                                        \
    };                                                                         \
    for (size_t _i = 0; _i < sizeof(_mluop_dims) / sizeof(_mluop_dims[0]);     \
         ++_i) {                                                               \
      if (!(_mluop_dims[_i] == _mluop_dim_base)) {                             \
        LOG(ERROR) << api << " Bad param: " << msg                             \
                   << " expect {" #dim1 ", " #dim2 ", " #__VA_ARGS__           \
                      "} all same, but get "                                   \
                   << get_dim_string() << ".";                                 \
        return MLUOP_STATUS_BAD_PARAM;                                         \
      }                                                                        \
    }                                                                          \
  } while (0)

// CHECK_EQ/NE/... with return value.
#define PARAM_CHECK_EQ(api, val1, val2, ...)                              \
  if (!(val1 == val2)) {                                                  \
    LOG(ERROR) << api                                                     \
               << " Check failed: " #val1 " == " #val2 ". " #__VA_ARGS__; \
    return MLUOP_STATUS_BAD_PARAM;                                        \
  }
#define PARAM_CHECK_NE(api, val1, val2, ...)                              \
  if (!(val1 != val2)) {                                                  \
    LOG(ERROR) << api                                                     \
               << " Check failed: " #val1 " != " #val2 ". " #__VA_ARGS__; \
    return MLUOP_STATUS_BAD_PARAM;                                        \
  }
#define PARAM_CHECK_LE(api, val1, val2, ...)                              \
  if (!(val1 <= val2)) {                                                  \
    LOG(ERROR) << api                                                     \
               << " Check failed: " #val1 " <= " #val2 ". " #__VA_ARGS__; \
    return MLUOP_STATUS_BAD_PARAM;                                        \
  }
#define PARAM_CHECK_LT(api, val1, val2, ...)                             \
  if (!(val1 < val2)) {                                                  \
    LOG(ERROR) << api                                                    \
               << " Check failed: " #val1 " < " #val2 ". " #__VA_ARGS__; \
    return MLUOP_STATUS_BAD_PARAM;                                       \
  }
#define PARAM_CHECK_GE(api, val1, val2, ...)                              \
  if (!(val1 >= val2)) {                                                  \
    LOG(ERROR) << api                                                     \
               << " Check failed: " #val1 " >= " #val2 ". " #__VA_ARGS__; \
    return MLUOP_STATUS_BAD_PARAM;                                        \
  }
#define PARAM_CHECK_GT(api, val1, val2, ...)                             \
  if (!(val1 > val2)) {                                                  \
    LOG(ERROR) << api                                                    \
               << " Check failed: " #val1 " > " #val2 ". " #__VA_ARGS__; \
    return MLUOP_STATUS_BAD_PARAM;                                       \
  }

#define TENSOR_NUM_CHECK(api, num, max_num, reason, ...)                      \
  if (!(num < max_num)) {                                                     \
    LOG(ERROR) << api << " overflow max supported tensor num " << max_num - 1 \
               << ", "                                                        \
               << "now tensor's total num is " << num << ". " << reason;      \
    return MLUOP_STATUS_NOT_SUPPORTED;                                        \
  }

#define TENSOR_DIM_SIZE_CHECK(api, desc, max_num, reason, ...)        \
  for (int i = 0; i < desc->getDim(); i++) {                          \
    if (!(desc->getDimIndex(i) < max_num)) {                          \
      LOG(ERROR) << api << " overflow max supported tensor dim size " \
                 << max_num - 1 << ", "                               \
                 << "now tensor's dims[" << i << "] is "              \
                 << desc->getDimIndex(i) << ". " << reason;           \
      return MLUOP_STATUS_NOT_SUPPORTED;                              \
    }                                                                 \
  }

extern bool mluop_check_large_tensor_dim_size_;
#define LARGE_TENSOR_CHECK(api, desc)                                         \
  if (desc != NULL) {                                                         \
    if (mluop_check_large_tensor_dim_size_) {                                 \
      TENSOR_DIM_SIZE_CHECK(api, desc, LARGE_TENSOR_NUM, "");                 \
    } else {                                                                  \
      TENSOR_NUM_CHECK(api, mluOpGetTensorElementNum(desc), LARGE_TENSOR_NUM, \
                       "");                                                   \
    }                                                                         \
  }

#define TENSOR_SIZE_CHECK(api, size, max_size, reason, ...)                \
  if (!(size < max_size)) {                                                \
    LOG(ERROR) << api << " overflow max supported tensor size "            \
               << max_size - 1 << "B, "                                    \
               << "now tensor's total size is " << size << "B." << reason; \
    return MLUOP_STATUS_NOT_SUPPORTED;                                     \
  }

#define STRIDE_TENSOR_CHECK(api, desc, reason)                            \
  if (MLUOP_PREDICT_TRUE(desc != NULL)) {                                 \
    if (MLUOP_PREDICT_FALSE(                                              \
            MLUOP_PREDICT_TRUE(0 != mluOpGetTensorElementNum(desc)) &&    \
            isStrideTensor(desc->getDim(), desc->getDims(),               \
                           desc->getStrides()))) {                        \
      LOG(ERROR) << api << " stride tensor is not supported. " << reason; \
      return MLUOP_STATUS_NOT_SUPPORTED;                                  \
    }                                                                     \
  }

void mluOpCheck(mluOpStatus_t result, char const *const func,
                const char *const file, int const line);
#define MLUOP_CHECK(val) mluOpCheck((val), #val, __FILE__, __LINE__)

#define KERNEL_CALL_CHECK(parent_kernel, sub_kernel, status, statement)        \
  do {                                                                         \
    if (status != MLUOP_STATUS_SUCCESS) {                                      \
      std::string error = "[" + std::string(parent_kernel) + "]" + " Error " + \
                          std::string(mluOpGetErrorString(status)) +           \
                          " occured when this kernel "                         \
                          "call the kernel " +                                 \
                          std::string(sub_kernel) + ". " +                     \
                          std::string(statement);                              \
      LOG(ERROR) << error;                                                     \
      return status;                                                           \
    }                                                                          \
  } while (0)

namespace mluop {

const int INFO = 0;     // base_logging::INFO;
const int WARNING = 1;  // base_logging::WARNING;
const int ERROR = 2;    // base_logging::ERROR;
const int FATAL = 3;    // base_logging::FATAL;

namespace internal {

class LogMessage {
 public:
  LogMessage() = delete;
  // Returns whether VLOG level lvl is activated for the file fname.
  static bool VmoduleActivated(const char *fname, int level);

  // Returns the minimum log level for VLOG statements.
  // E.g., if MinVLogLevel() is 2, then VLOG(2) statements will produce output,
  // but VLOG(3) will not. Defaults to 0.
  static int64_t MinVLogLevel();
};

inline namespace {  // NOLINT
// Uses the lower operator & precedence to voidify a LogMessage reference, so
// that the ternary VLOG() implementation is balanced, type wise.
struct Voidifier {
  template <typename T>
  void operator&(const T &) const {}
};
}  // namespace

// Otherwise, set MLUOP_MIN_VLOG_LEVEL environment to update minimum log level
// of VLOG, or MLUOP_CPP_VMODULE to set the minimum log level for individual
// translation units.
#define VLOG_IS_ON(lvl)                                                \
  (([](int level, const char *fname) {                                 \
    static const bool vmodule_activated =                              \
        ::mluop::internal::LogMessage::VmoduleActivated(fname, level); \
    return vmodule_activated;                                          \
  })(lvl, __FILE__))

#define VLOG(level)                                                  \
  MLUOP_PREDICT_TRUE(!VLOG_IS_ON(({                                  \
    static_assert(level > 0, "VLOG level should be greater than 0"); \
    level;                                                           \
  })))                                                               \
  ? (void)0 : ::mluop::internal::Voidifier() & LOG(VLOG)

// This formats a value for a failing CHECK_XX statement.  Ordinarily,
// it uses the definition for operator<<, with a few special cases below.
template <typename T>
inline void MakeCheckOpValueString(std::ostream *os, const T &v) {
  (*os) << v;
}

// Overrides for char types provide readable values for unprintable
// characters.
template <>
void MakeCheckOpValueString(std::ostream *os, const char &v);
template <>
void MakeCheckOpValueString(std::ostream *os, const signed char &v);  // NOLINT
template <>
void MakeCheckOpValueString(std::ostream *os,
                            const unsigned char &v);  // NOLINT

#if LANG_CXX11
// We need an explicit specialization for std::nullptr_t.
template <>
void MakeCheckOpValueString(std::ostream *os, const std::nullptr_t &p);
#endif

// A container for a string pointer which can be evaluated to a bool -
// true iff the pointer is non-NULL.
struct CheckOpString {
  CheckOpString(std::string *str) : str_(str) {}  // NOLINT
  // No destructor: if str_ is non-NULL, we're about to LOG(FATAL),
  // so there's no point in cleaning up str_.
  operator bool() const { return MLUOP_PREDICT_FALSE(str_ != NULL); }
  std::string *str_;
};

// Build the error message string. Specify no inlining for code size.
template <typename T1, typename T2>
std::string *MakeCheckOpString(const T1 &v1, const T2 &v2,
                               const char *exprtext) MLUOP_ATTRIBUTE_NOINLINE;

// A helper class for formatting "expr (V1 vs. V2)" in a CHECK_XX
// statement.  See MakeCheckOpString for sample usage.  Other
// approaches were considered: use of a template method (e.g.,
// base::BuildCheckOpString(exprtext, base::Print<T1>, &v1,
// base::Print<T2>, &v2), however this approach has complications
// related to volatile arguments and function-pointer arguments).
class CheckOpMessageBuilder {
 public:
  // Inserts "exprtext" and " (" to the stream.
  explicit CheckOpMessageBuilder(const char *exprtext);
  // Deletes "stream_".
  ~CheckOpMessageBuilder();
  // For inserting the first variable.
  std::ostream *ForVar1() { return stream_; }
  // For inserting the second variable (adds an intermediate " vs. ").
  std::ostream *ForVar2();
  // Get the result (inserts the closing ")").
  std::string *NewString();

 private:
  std::ostringstream *stream_;
};

template <typename T1, typename T2>
std::string *MakeCheckOpString(const T1 &v1, const T2 &v2,
                               const char *exprtext) {
  CheckOpMessageBuilder comb(exprtext);
  MakeCheckOpValueString(comb.ForVar1(), v1);
  MakeCheckOpValueString(comb.ForVar2(), v2);
  return comb.NewString();
}

// Helper functions for CHECK_OP macro.
// The (int, int) specialization works around the issue that the compiler
// will not instantiate the template version of the function on values of
// unnamed enum type - see comment below.
// The (size_t, int) and (int, size_t) specialization are to handle unsigned
// comparison errors while still being thorough with the comparison.
#define MLUOP_DEFINE_CHECK_OP_IMPL(name, op)                             \
  template <typename T1, typename T2>                                    \
  inline std::string *name##Impl(const T1 &v1, const T2 &v2,             \
                                 const char *exprtext) {                 \
    if (MLUOP_PREDICT_TRUE(v1 op v2))                                    \
      return NULL;                                                       \
    else                                                                 \
      return ::mluop::internal::MakeCheckOpString(v1, v2, exprtext);     \
  }                                                                      \
  inline std::string *name##Impl(int v1, int v2, const char *exprtext) { \
    return name##Impl<int, int>(v1, v2, exprtext);                       \
  }                                                                      \
  inline std::string *name##Impl(const size_t v1, const int v2,          \
                                 const char *exprtext) {                 \
    if (MLUOP_PREDICT_FALSE(v2 < 0)) {                                   \
      return ::mluop::internal::MakeCheckOpString(v1, v2, exprtext);     \
    }                                                                    \
    return name##Impl<size_t, size_t>(v1, v2, exprtext);                 \
  }                                                                      \
  inline std::string *name##Impl(const int v1, const size_t v2,          \
                                 const char *exprtext) {                 \
    if (MLUOP_PREDICT_FALSE(v2 >= std::numeric_limits<int>::max())) {    \
      return ::mluop::internal::MakeCheckOpString(v1, v2, exprtext);     \
    }                                                                    \
    const size_t uval = (size_t)((unsigned)v2);                          \
    return name##Impl<size_t, size_t>(v1, uval, exprtext);               \
  }

// We use the full name Check_EQ, Check_NE, etc. in case the file including
// base/logging.h provides its own #defines for the simpler names EQ, NE, etc.
// This happens if, for example, those are used as token names in a
// yacc grammar.
MLUOP_DEFINE_CHECK_OP_IMPL(Check_EQ, ==)
MLUOP_DEFINE_CHECK_OP_IMPL(Check_NE, !=)
MLUOP_DEFINE_CHECK_OP_IMPL(Check_LE, <=)
MLUOP_DEFINE_CHECK_OP_IMPL(Check_LT, <)
MLUOP_DEFINE_CHECK_OP_IMPL(Check_GE, >=)
MLUOP_DEFINE_CHECK_OP_IMPL(Check_GT, >)
#undef MLUOP_DEFINE_CHECK_OP_IMPL

// In optimized mode, use CheckOpString to hint to compiler that
// the while condition is unlikely.
#define CHECK_OP_LOG(name, op, val1, val2)                       \
  while (::mluop::internal::CheckOpString _result =              \
             ::mluop::internal::name##Impl(                      \
                 ::mluop::internal::GetReferenceableValue(val1), \
                 ::mluop::internal::GetReferenceableValue(val2), \
                 #val1 " " #op " " #val2))                       \
  LOG(ERROR) << "[" << __FUNCTION__ << "] " << *(_result.str_)

#define CHECK_OP(name, op, val1, val2) CHECK_OP_LOG(name, op, val1, val2)

// Function is overloaded for integral types to allow static const
// integrals declared in classes and not defined to be used as arguments to
// CHECK* macros. It's not encouraged though.
template <typename T>
inline const T &GetReferenceableValue(const T &t) {
  return t;
}
inline char GetReferenceableValue(char t) { return t; }
inline unsigned char GetReferenceableValue(unsigned char t) { return t; }
inline signed char GetReferenceableValue(signed char t) { return t; }
inline short GetReferenceableValue(short t) {  // NOLINT
  return t;
}
inline unsigned short GetReferenceableValue(unsigned short t) {  // NOLINT
  return t;
}
inline int GetReferenceableValue(int t) {  // NOLINT
  return t;
}
inline unsigned int GetReferenceableValue(unsigned int t) { return t; }
inline long GetReferenceableValue(long t) {  // NOLINT
  return t;
}
inline unsigned long GetReferenceableValue(unsigned long t) {  // NOLINT
  return t;
}
inline long long GetReferenceableValue(long long t) {  // NOLINT
  return t;
}
inline unsigned long long GetReferenceableValue(unsigned long long t) {  // NOLINT
  return t;
}

}  // namespace internal
}  // namespace mluop

#endif  // CORE_LOGGING_H_
