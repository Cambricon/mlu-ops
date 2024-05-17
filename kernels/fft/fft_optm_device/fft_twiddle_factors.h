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

#define COS_PI_2_3 -0.49999999999999978
#define TW3_1R_F COS_PI_2_3
#define TW3_1I_F -0.86602540378443871

#define COS_PI_2_9 \
  0.76604444311897801  //  cos(2 * pi * 1 / 9) and cos(2 * pi * 8 / 9)
#define COS_PI_4_9 \
  0.17364817766693041  //  cos(2 * pi * 2 / 9) and cos(2 * pi * 7 / 9)
#define COS_PI_6_9 \
  -0.49999999999999978  //  cos(2 * pi * 3 / 9) and cos(2 * pi * 6 / 9)
#define COS_PI_8_9 \
  -0.93969262078590832  //  cos(2 * pi * 4 / 9) and cos(2 * pi * 5 / 9)
#define NEG_SIN_PI_2_9 \
  -0.64278760968653925  //  -sin(2 * pi * 1 / 9) and sin(2 * pi * 8 / 9)
#define NEG_SIN_PI_4_9 \
  -0.98480775301220802  //  -sin(2 * pi * 2 / 9) and sin(2 * pi * 7 / 9)
#define NEG_SIN_PI_6_9 \
  -0.86602540378443871  //  -sin(2 * pi * 3 / 9) and sin(2 * pi * 6 / 9)
#define NEG_SIN_PI_8_9 \
  -0.34202014332566888  //  -sin(2 * pi * 4 / 9) and sin(2 * pi * 5 / 9)
#define SIN_PI_2_9 \
  0.64278760968653925  //  sin(2 * pi * 1 / 9) and -sin(2 * pi * 8 / 9)
#define SIN_PI_4_9 \
  0.98480775301220802  //  sin(2 * pi * 2 / 9) and -sin(2 * pi * 7 / 9)
#define SIN_PI_6_9 \
  0.86602540378443871  //  sin(2 * pi * 3 / 9) and -sin(2 * pi * 6 / 9)
#define SIN_PI_8_9 \
  0.34202014332566888  //  sin(2 * pi * 4 / 9) and -sin(2 * pi * 5 / 9)
#define NEG_SIN_PI_6_9_X2 -1.732050807568877293527446341505872366942805254

/* Twiddles used in Radix-9 FFT */
#define TW9_1R_F COS_PI_2_9
#define TW9_2R_F COS_PI_4_9
#define TW9_3R_F COS_PI_6_9
#define TW9_4R_F COS_PI_8_9
#define TW9_1I_F NEG_SIN_PI_2_9
#define TW9_2I_F NEG_SIN_PI_4_9
#define TW9_3I_F NEG_SIN_PI_6_9
#define TW9_4I_F NEG_SIN_PI_8_9
#define TW9_1I_F_NEG SIN_PI_2_9
#define TW9_2I_F_NEG SIN_PI_4_9
#define TW9_3I_F_NEG SIN_PI_6_9
#define TW9_4I_F_NEG SIN_PI_8_9
#define TW9_3I_X2_F NEG_SIN_PI_6_9_X2
#define TW91_SUB_TW94_R 1.705737063904886419256501927880148143872040591
#define TW92_ADD_TW94_R -1.113340798452838732905825904094046265936583811
#define TW94_SUB_TW91_I 0.300767466360870593278543795225003852144476517
#define TW92_ADD_TW94_I 1.326827896337876792410842639271782594433726619
