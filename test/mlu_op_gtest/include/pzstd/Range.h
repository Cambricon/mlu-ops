/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "pzstd/Likely.h"
#include "pzstd/Portability.h"

namespace mluoptest {

namespace detail {
/*
 Use IsCharPointer<T>::type to enable const char or char*.
 Use IsCharPointer<T>::const_type to enable only const char.
*/
template <class T>
struct IsCharPointer {};

template <>
struct IsCharPointer<char*> {
  typedef int type;
};

template <>
struct IsCharPointer<const char*> {
  typedef int const_type;
  typedef int type;
};

}  // namespace detail

template <typename Iter>
class Range {
  Iter b_;
  Iter e_;

 public:
  using size_type = std::size_t;
  using iterator = Iter;
  using const_iterator = Iter;
  using value_type = typename std::remove_reference<
      typename std::iterator_traits<Iter>::reference>::type;
  using reference = typename std::iterator_traits<Iter>::reference;

  constexpr Range() : b_(), e_() {}
  constexpr Range(Iter begin, Iter end) : b_(begin), e_(end) {}

  constexpr Range(Iter begin, size_type size) : b_(begin), e_(begin + size) {}

  template <class T = Iter, typename detail::IsCharPointer<T>::type = 0>
  /* implicit */   Range(Iter str) : b_(str), e_(str + std::strlen(str)) {}  // NOLINT

  template <class T = Iter, typename detail::IsCharPointer<T>::const_type = 0>
  /* implicit */   Range(const std::string& str) : b_(str.data()), e_(b_ + str.size()) {}   // NOLINT

  // Allow implicit conversion from Range<From> to Range<To> if From is
  // implicitly convertible to To.
  template <class OtherIter, typename std::enable_if<
                                 (!std::is_same<Iter, OtherIter>::value &&
                                  std::is_convertible<OtherIter, Iter>::value),
                                 int>::type = 0>
  constexpr /* implicit */ Range(const Range<OtherIter>& other)
      : b_(other.begin()), e_(other.end()) {}

  Range(const Range&) = default;
  Range(Range&&) = default;

  Range& operator=(const Range&) = default;
  Range& operator=(Range&&) = default;

  constexpr size_type size() const { return e_ - b_; }

  bool empty() const { return b_ == e_; }

  Iter data() const { return b_; }
  Iter begin() const { return b_; }

  Iter end() const { return e_; }

  void advance(size_type n) {
    if (UNLIKELY(n > size())) {
      throw std::out_of_range("index out of range");
    }

    b_ += n;
  }

  void subtract(size_type n) {
    if (UNLIKELY(n > size())) {
      throw std::out_of_range("index out of range");
    }

    e_ -= n;
  }

  Range subpiece(size_type first, size_type length = std::string::npos) const {
    if (UNLIKELY(first > size())) {
      throw std::out_of_range("index out of range");
    }

    return Range(b_ + first, std::min(length, size() - first));
  }
};

using ByteRange = Range<const char*>;
using MutableByteRange = Range<char*>;
using StringPiece = Range<const char*>;
}  // namespace mluoptest
