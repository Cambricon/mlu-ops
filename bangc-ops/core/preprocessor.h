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
#pragma once

/*!
 * @file preprocessor.h
 * provides useful macros for preprocessor metaprogramming
 * @ref:
 * stackoverflow.com/questions/5957679/is-there-a-way-to-use-c-preprocessor-stringification-on-variadic-macro-arguments
 */

// internal helper method for MLUOP_PP_NUM_ARGS
#define _NUM_ARGS(X29, X28, X27, X26, X25, X24, X23, X22, X21, X20, X19, X18, \
                  X17, X16, X15, X14, X13, X12, X11, X10, X9, X8, X7, X6, X5, \
                  X4, X3, X2, X1, N, ...)                                     \
  N

/*!
 * @brief get number of input arguments
 */
#define MLUOP_PP_NUM_ARGS(...)                                               \
  _NUM_ARGS(__VA_ARGS__, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, \
            16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

#define EXPAND(X) X
#define FIRSTARG(X, ...) (X)
#define RESTARGS(X, ...) (__VA_ARGS__)

/*!
 * @brief apply the same macro on each element of the list
 * @param MACRO function for funciton like macro
 * @param LIST  element sequence, each element x of LIST will be called by
 * MACRO(x)
 *
 * @example `MLUOP_PP_MAP(F, (a, b, c))` will be expanded to `F(a) F(b) F(c)`
 */
#define MLUOP_PP_MAP(MACRO, LIST) MAP_(MLUOP_PP_NUM_ARGS LIST, MACRO, LIST)

// internal helper method for MLUOP_PP_MAP
#define MAP_(N, M, LIST) MAP__(N, M, LIST)
#define MAP__(N, M, LIST) MAP_##N(M, LIST)
#define MAP_1(M, LIST) M LIST
#define MAP_2(M, LIST) EXPAND(M FIRSTARG LIST) MAP_1(M, RESTARGS LIST)
#define MAP_3(M, LIST) EXPAND(M FIRSTARG LIST) MAP_2(M, RESTARGS LIST)
#define MAP_4(M, LIST) EXPAND(M FIRSTARG LIST) MAP_3(M, RESTARGS LIST)
#define MAP_5(M, LIST) EXPAND(M FIRSTARG LIST) MAP_4(M, RESTARGS LIST)
#define MAP_6(M, LIST) EXPAND(M FIRSTARG LIST) MAP_5(M, RESTARGS LIST)
#define MAP_7(M, LIST) EXPAND(M FIRSTARG LIST) MAP_6(M, RESTARGS LIST)
#define MAP_8(M, LIST) EXPAND(M FIRSTARG LIST) MAP_7(M, RESTARGS LIST)
#define MAP_9(M, LIST) EXPAND(M FIRSTARG LIST) MAP_8(M, RESTARGS LIST)
#define MAP_10(M, LIST) EXPAND(M FIRSTARG LIST) MAP_9(M, RESTARGS LIST)
#define MAP_11(M, LIST) EXPAND(M FIRSTARG LIST) MAP_10(M, RESTARGS LIST)
#define MAP_12(M, LIST) EXPAND(M FIRSTARG LIST) MAP_11(M, RESTARGS LIST)
#define MAP_13(M, LIST) EXPAND(M FIRSTARG LIST) MAP_12(M, RESTARGS LIST)
#define MAP_14(M, LIST) EXPAND(M FIRSTARG LIST) MAP_13(M, RESTARGS LIST)
#define MAP_15(M, LIST) EXPAND(M FIRSTARG LIST) MAP_14(M, RESTARGS LIST)
#define MAP_16(M, LIST) EXPAND(M FIRSTARG LIST) MAP_15(M, RESTARGS LIST)
#define MAP_17(M, LIST) EXPAND(M FIRSTARG LIST) MAP_16(M, RESTARGS LIST)
#define MAP_18(M, LIST) EXPAND(M FIRSTARG LIST) MAP_17(M, RESTARGS LIST)
#define MAP_19(M, LIST) EXPAND(M FIRSTARG LIST) MAP_18(M, RESTARGS LIST)
#define MAP_20(M, LIST) EXPAND(M FIRSTARG LIST) MAP_19(M, RESTARGS LIST)
#define MAP_21(M, LIST) EXPAND(M FIRSTARG LIST) MAP_20(M, RESTARGS LIST)
#define MAP_22(M, LIST) EXPAND(M FIRSTARG LIST) MAP_21(M, RESTARGS LIST)
#define MAP_23(M, LIST) EXPAND(M FIRSTARG LIST) MAP_22(M, RESTARGS LIST)
#define MAP_24(M, LIST) EXPAND(M FIRSTARG LIST) MAP_23(M, RESTARGS LIST)
#define MAP_25(M, LIST) EXPAND(M FIRSTARG LIST) MAP_24(M, RESTARGS LIST)
