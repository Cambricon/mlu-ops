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

#include <utility>
#include <string>
#include <limits>
#include <sstream>
#include "core/macros.h"
#include "core/cnlog.h"
#include "mlu_op.h"

#define LARGE_TENSOR_NUM ((uint64_t)2147483648)
#define LARGE_TENSOR_SIZE ((uint64_t)2147483648)

#define LOG(severity) mluop::cnlog::CLOG(MLUOP, severity)

#define TOKENPASTE(x, y, z) x##y##z
#define TOKENPASTE2(x, y, z) TOKENPASTE(x, y, z)

#define LOG_FIRST_N(severity, n)                                            \
  static std::atomic<int> TOKENPASTE2(LOG_, __LINE__, _OCCURRENCES)(0);     \
  if (MLUOP_PREDICT_FALSE(TOKENPASTE2(LOG_, __LINE__, _OCCURRENCES)++ < n)) \
  mluop::cnlog::CLOG(MLUOP, severity)

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
#define SYMBOL_CHECK(symbol...)                                     \
  if (MLUOP_PREDICT_FALSE(!&symbol)) {                              \
    LOG(FATAL) << "calling underfined symbol which not be linked: " \
               << #symbol;                                          \
  }                                                                 \
  symbol

// return if found cnrt error.
#define KERNEL_CHECK(kernel...)                                    \
  {                                                                \
    cnrtGetLastError();                                            \
    kernel;                                                        \
    cnrtRet_t ret = cnrtPeekAtLastError();                         \
    if (MLUOP_PREDICT_FALSE(CNRT_RET_SUCCESS != ret)) {            \
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
#define CHECK_RETURN(api, status, ...)                                     \
  {                                                                        \
    mluOpStatus_t __status__ = (status);                                   \
    if ((__status__) != MLUOP_STATUS_SUCCESS) {                            \
      LOG(ERROR) << api << "BAD return status: " << #status << " returns " \
                 << (__status__) << " (FILE: " << __FILE__                 \
                 << ", LINE: " << __LINE__ << "). " __VA_ARGS__;           \
      return (__status__);                                                 \
    }                                                                      \
  }

#define PARAM_CHECK(api, condition, ...)                                 \
  if (!(condition)) {                                                    \
    LOG(ERROR) << api << " Check failed: " #condition ". " #__VA_ARGS__; \
    return MLUOP_STATUS_BAD_PARAM;                                       \
  }

// this prints out values instead of names of variables inside _VA_ARGS__
#define PARAM_CHECK_V2(api, condition, ...)                             \
  if (!(condition)) {                                                   \
    LOG(ERROR) << api << " Check failed: " #condition ". " __VA_ARGS__; \
    return MLUOP_STATUS_BAD_PARAM;                                      \
  }

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

#define TENSOR_NUM_CHECK(api, num, max_num, reason, ...)                    \
  if (!(num < max_num)) {                                                   \
    LOG(ERROR) << "[" << api << "]: overflow max supported tensor num. "    \
               << max_num - 1 << ", "                                       \
               << "now tensor's total num is " << num << ". " << reason;    \
    return MLUOP_STATUS_NOT_SUPPORTED;                                      \
  }

#define TENSOR_SIZE_CHECK(api, size, max_size, reason, ...)                 \
  if (!(size < max_size)) {                                                 \
    LOG(ERROR) << "[" << api << "]: overflow max supported tensor size. "   \
               << max_size - 1 << "B, "                                     \
               << "now tensor's total size is " << size << "B. " << reason; \
    return MLUOP_STATUS_NOT_SUPPORTED;                                      \
  }

void mluOpCheck(mluOpStatus_t result, char const *const func,
                const char *const file, int const line);

#define MLUOP_CHECK(val) mluOpCheck((val), #val, __FILE__, __LINE__)

#define KERNEL_CALL_CHECK(parent_kernel, sub_kernel, status, statement)       \
  do {                                                                        \
    if (status != MLUOP_STATUS_SUCCESS) {                                     \
      std::string error = "[" + std::string(parent_kernel) + "]" + " Error" + \
                          std::string(mluOpGetErrorString(status)) +          \
                          " occured when this kernel "                        \
                          "call the kernel " +                                \
                          std::string(sub_kernel) + ". " +                    \
                          std::string(statement);                             \
      LOG(ERROR) << error;                                                    \
      return status;                                                          \
    }                                                                         \
  } while (0)

namespace mluop {

const int INFO = 0;     // base_logging::INFO;
const int WARNING = 1;  // base_logging::WARNING;
const int ERROR = 2;    // base_logging::ERROR;
const int FATAL = 3;    // base_logging::FATAL;

namespace internal {

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char *fname, int line, int severity);
  ~LogMessage();

  // Returns the minimum log level for VLOG statements.
  // E.g., if MinVLogLevel() is 2, then VLOG(2) statements will produce output,
  // but VLOG(3) will not. Defaults to 0.
  static int64_t MinVLogLevel();

  // Returns whether VLOG level lvl is activated for the file fname.
  static bool VmoduleActivated(const char *fname, int level);

 protected:
  void GenerateLogMessage();

 private:
  const char *fname_;
  int line_;
  int severity_;
};

// Uses the lower operator & precedence to voidify a LogMessage reference, so
// that the ternary VLOG() implementation is balanced, type wise.
struct Voidifier {
  template <typename T>
  void operator&(const T &) const {}
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char *file, int line) MLUOP_ATTRIBUTE_COLD;
  MLUOP_ATTRIBUTE_NORETURN ~LogMessageFatal();
};

// Otherwise, set MLUOP_MIN_VLOG_LEVEL environment to update minimum log level
// of VLOG, or MLUOP_CPP_VMODULE to set the minimum log level for individual
// translation units.
#define VLOG_IS_ON(lvl)                                                \
  (([](int level, const char *fname) {                                 \
    static const bool vmodule_activated =                              \
        ::mluop::internal::LogMessage::VmoduleActivated(fname, level); \
    return vmodule_activated;                                          \
  })(lvl, __FILE__))

#define VLOG(level)                      \
  MLUOP_PREDICT_TRUE(!VLOG_IS_ON(level)) \
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
inline unsigned long long                      // NOLINT
GetReferenceableValue(unsigned long long t) {  // NOLINT
  return t;
}

}  // namespace internal
}  // namespace mluop

#endif  // CORE_LOGGING_H_
