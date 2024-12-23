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

/**
 * Provides `BANG_WRAP_T(ptr_arg)` for .cc and `BANG_UNWRAP_T(ptr_arg)` for .mlu
 * to bridge Eigen:: type and BANGC type
 */

#include <type_traits>

struct bang_half_t;
struct bang_bfloat16_t;

namespace detail {
/*
 * `bang_wrap_data` and `bang_unwrap_data` could be the same thing,
 *  but should be used in different scope
 *
 *  handle 'const DType', 'Dtype *',
 *  could be implemented by SFINAE or just specialization
 */
template <typename DType, template <typename> class Impl,
          typename RawType = DType>
struct bang_trans_impl_ {
  static_assert(std::is_same_v<RawType, DType>);
  typedef DType type;
};

template <typename DType, template <typename> class Impl, typename RawType>
struct bang_trans_impl_<DType*, Impl, RawType> {
  typedef typename Impl<DType>::type* type;
};

template <typename DType, template <typename> class Impl, typename RawType>
struct bang_trans_impl_<const DType, Impl, RawType> {
  typedef const typename Impl<DType>::type type;
};

}  // namespace detail

#define BANG_TRANS_TYPE_FROM_TO(TOKEN, From, To) \
  template <>                                    \
  struct TOKEN<From> {                           \
    typedef To type;                             \
  }

/* For .cc/.cpp trans unknown type to wrapped type */
#if !defined(__BANG__)

namespace Eigen {
struct half;
struct bfloat16;
}  // namespace Eigen

template <typename DType>
struct bang_wrap_data {
  using type = typename detail::bang_trans_impl_<DType, bang_wrap_data>::type;
};

#define BANG_WRAP_TYPE_FROM_TO(From, To) \
  BANG_TRANS_TYPE_FROM_TO(bang_wrap_data, From, To)

BANG_WRAP_TYPE_FROM_TO(Eigen::half, bang_half_t);
BANG_WRAP_TYPE_FROM_TO(Eigen::bfloat16, bang_bfloat16_t);

template <typename T>
using bang_wrap_data_t = typename bang_wrap_data<T>::type;

#define BANG_WRAP_T(a) reinterpret_cast<bang_wrap_data_t<decltype(a)>>(a)

#endif  // !defined(__BANG__)

/* For .mlu trans intermediate type to mlu's underlying type */

#if __BANG__
template <typename DType>
struct bang_unwrap_data {
  using type = typename detail::bang_trans_impl_<DType, bang_unwrap_data>::type;
};

#define BANG_UNWRAP_TYPE_FROM_TO(From, To) \
  BANG_TRANS_TYPE_FROM_TO(bang_unwrap_data, From, To)

BANG_UNWRAP_TYPE_FROM_TO(bang_half_t, half);
BANG_UNWRAP_TYPE_FROM_TO(bang_bfloat16_t, bfloat16_t);

template <typename T>
using bang_unwrap_data_t = typename bang_unwrap_data<T>::type;

#define BANG_UNWRAP_T(a) reinterpret_cast<bang_unwrap_data_t<decltype(a)>>(a)

#endif  // __BANG__
