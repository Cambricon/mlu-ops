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
#ifndef CORE_MACROS_H_
#define CORE_MACROS_H_

#if !defined(PLATFORM_POSIX)
#if defined(_WIN32)
#define PLATFORM_WINDOWS
#else
#define PLATFORM_POSIX
#endif
#endif  // !defined(PLATFORM_POSIX)

// Compiler attributes
#if defined(__GNUC__)

// Compiler supports GCC-style attributes
#define MLUOP_ATTRIBUTE_VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#define MLUOP_ATTRIBUTE_CONSTRUCTOR __attribute__((constructor))
#define MLUOP_ATTRIBUTE_DESTRUCTOR __attribute__((destructor))
#define MLUOP_ATTRIBUTE_NORETURN __attribute__((noreturn))
#define MLUOP_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#define MLUOP_ATTRIBUTE_NOINLINE __attribute__((noinline))
#define MLUOP_ATTRIBUTE_UNUSED __attribute__((unused))
#define MLUOP_ATTRIBUTE_COLD __attribute__((cold))
#define MLUOP_ATTRIBUTE_WEAK __attribute__((weak))
#define MLUOP_ATTRIBUTE_FLATTEN __attribute__((flatten))
#define MLUOP_PACKED __attribute__((packed))
#define MLUOP_MUST_USE_RESULT __attribute__((warn_unused_result))
#define MLUOP_PRINTF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__printf__, string_index, first_to_check)))
#define MLUOP_SCANF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__scanf__, string_index, first_to_check)))
#elif defined(_MSC_VER)
// Non-GCC equivalents
#define MLUOP_ATTRIBUTE_NORETURN __declspec(noreturn)
#define MLUOP_ATTRIBUTE_ALWAYS_INLINE __forceinline
#define MLUOP_ATTRIBUTE_CONSTRUCTOR
#define MLUOP_ATTRIBUTE_DESTRUCTOR
#define MLUOP_ATTRIBUTE_NOINLINE
#define MLUOP_ATTRIBUTE_UNUSED
#define MLUOP_ATTRIBUTE_COLD
#define MLUOP_ATTRIBUTE_WEAK
#define MLUOP_ATTRIBUTE_FLATTEN
#define MLUOP_MUST_USE_RESULT
#define MLUOP_PACKED
#define MLUOP_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define MLUOP_SCANF_ATTRIBUTE(string_index, first_to_check)

#else  // defined(__GNUC__)

// Non-GCC equivalents
#define MLUOP_ATTRIBUTE_NORETURN
#define MLUOP_ATTRIBUTE_ALWAYS_INLINE
#define MLUOP_ATTRIBUTE_CONSTRUCTOR
#define MLUOP_ATTRIBUTE_DESTRUCTOR
#define MLUOP_ATTRIBUTE_NOINLINE
#define MLUOP_ATTRIBUTE_UNUSED
#define MLUOP_ATTRIBUTE_COLD
#define MLUOP_ATTRIBUTE_WEAK
#define MLUOP_MUST_USE_RESULT
#define MLUOP_PACKED
#define MLUOP_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define MLUOP_SCANF_ATTRIBUTE(string_index, first_to_check)

#endif  // defined(__GNUC__)

#ifdef __has_builtin
#define MLUOP_HAS_BUILTIN(x) __has_builtin(x)
#else
#define MLUOP_HAS_BUILTIN(x) 0
#endif

// Compilers can be told that a certain branch is not likely to be taken
// (for instance, a CHECK failure), and use that information in static
// analysis. Giving it this information can help it optimize for the
// common case in the absence of better information (ie.
// -fprofile-arcs).
#if (MLUOP_HAS_BUILTIN(__builtin_expect) || \
     (defined(__GNUC__) && __GNUC__ >= 3))
#define MLUOP_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define MLUOP_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define MLUOP_PREDICT_FALSE(x) (x)
#define MLUOP_PREDICT_TRUE(x) (x)
#endif

#if defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L || \
    (defined(_MSC_VER) && _MSC_VER >= 1900)
// Define this to 1 if the code is compiled in C++11 mode; leave it
// undefined otherwise.  Do NOT define it to 0 -- that causes
// '#ifdef LANG_CXX11' to behave differently from '#if LANG_CXX11'.
#define LANG_CXX11 1
#endif

#define THRESHOLD_MSE (1e-5)
#define THRESHOLD_DIFF1 (0.003)

#endif  // CORE_MACROS_H_
