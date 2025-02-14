/*************************************************************************
 * Copyright (C) [2025] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of chvar_arge, to any person obtaining a
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
#include <tuple>
#include <variant>

#include "mlu_op.h"

namespace Eigen {
struct half;
struct bfloat16;
}  // namespace Eigen

namespace mluoptest {
using Eigen::bfloat16;
using Eigen::half;

template <typename T>
struct TypeToEnum {
  static constexpr mluOpDataType_t value = MLUOP_DTYPE_INVALID;
};

#define typeMapEnum(type, enum)                    \
  template <>                                      \
  struct TypeToEnum<type> {                        \
    static constexpr mluOpDataType_t value = enum; \
  }

typeMapEnum(float, MLUOP_DTYPE_FLOAT);
typeMapEnum(half, MLUOP_DTYPE_HALF);
typeMapEnum(bfloat16, MLUOP_DTYPE_BFLOAT16);
typeMapEnum(double, MLUOP_DTYPE_DOUBLE);
typeMapEnum(int8_t, MLUOP_DTYPE_INT8);
typeMapEnum(int16_t, MLUOP_DTYPE_INT16);
typeMapEnum(int32_t, MLUOP_DTYPE_INT32);
typeMapEnum(int64_t, MLUOP_DTYPE_INT64);
typeMapEnum(uint8_t, MLUOP_DTYPE_UINT8);
typeMapEnum(uint16_t, MLUOP_DTYPE_UINT16);
typeMapEnum(uint32_t, MLUOP_DTYPE_UINT32);
typeMapEnum(uint64_t, MLUOP_DTYPE_UINT64);
typeMapEnum(bool, MLUOP_DTYPE_BOOL);
typeMapEnum(std::complex<half>, MLUOP_DTYPE_COMPLEX_HALF);
typeMapEnum(std::complex<float>, MLUOP_DTYPE_COMPLEX_FLOAT);

/*
 * Variadic templates, using fold expressions.
 *
 * Parameter Explanation:  ...Types represents all supported types.
 */
template <typename... Types>
struct VariantHelper {
  using Type = std::variant<Types...>;

  static Type create(mluOpDataType_t enumDType) {
    Type var;
    bool f = false;
    // The expression (: false) has no practical significance, it's just for
    // correct compilation.
    ((enumDType == TypeToEnum<Types>::value ? var = Types{}, f = true : false),
     ...);

    if (!f) {
      throw std::invalid_argument("unsupported type");
    }
    return var;
  }
};
}  // namespace mluoptest

/*
 * Macro Explanation: Constructs a variant type variable that supports all the
 *  given types and assigns an initial value based on the enum type obtained
 *  during runtime.
 *
 * Parameter Explanation: var is the variable name of the variant type,
 *  runEnumDataType is the enum type obtained during runtime,
 *  ... represents all supported types.
 *
 * Example: VARIANT_INIT(var, run_dtype, float, half, bfloat16).
 */
#define VARIANT_INIT(var, runEnumDataType, ...) \
  auto var = VariantHelper<__VA_ARGS__>::create(runEnumDataType);

/*
 * Macro Explanation:
 *  Traverses the variants type and dispatches a template
 *  function with T_VAR(id) during the traversal. It
 *  should be used in conjunction with the VARIANT_INIT macro.
 *
 * Parameter Explanation:
 *  The T_VAR(id) macro will be used in func_call to specify the types,
 *  ... are the variables constructed by the VARIANT_INIT macro.
 *
 * Key Technology: std::visit.
 *
 * Notes:
 *  1.Do not declare a variable named Types.
 *  2.When using a function with <>, the entire function call needs to be
 *    wrapped in ().
 *  3.Only full traversal can be done, and it will result in an error if
 *    some combinations are not instantiated.
 */
#define DISPATCH_VARIANTS(func_call, ...)                          \
  std::visit(                                                      \
      [&](auto&&... args) {                                        \
        using Types = std::tuple<std::decay_t<decltype(args)>...>; \
        func_call;                                                 \
      },                                                           \
      __VA_ARGS__);

// The first variant type is T_VAR(0), and so on.
#define T_VAR(id) std::tuple_element_t<id, Types>
