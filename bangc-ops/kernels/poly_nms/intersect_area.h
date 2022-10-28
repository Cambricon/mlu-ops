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

#ifndef BANGC_OPS_KERNELS_POLY_NMS_INTERSECT_AREA_H
#define BANGC_OPS_KERNELS_POLY_NMS_INTERSECT_AREA_H

#include <stdint.h>

#include "kernels/poly_nms/enums.h"

namespace {  // NOLINT
constexpr uint32_t BIT_FLOAT_1 = 0x0;
constexpr uint32_t BIT_FLOAT_NEG_1 = 0x80000000;
#define BIT_FLOAT_MUL(x, y) ((x) ^ (y))
#define EPSILON 1e-12

struct Point2D {
  float x;
  float y;
};

struct Line {
  __mlu_func__ void update(const Point2D *__restrict__ A,
                           const Point2D *__restrict__ B) {
    a = B->y - A->y;
    b = A->x - B->x;
    c = a * (A->x) + b * (A->y);
  }
  float a;
  float b;
  float c;
};

__mlu_func__ static uint32_t getDirection(const Point2D *__restrict__ A) {
  float x0 = A[2].x - A[0].x;
  float y0 = A[2].y - A[0].y;
  float x1 = A[3].x - A[1].x;
  float y1 = A[3].y - A[1].y;
  if ((x0 * y1 - y0 * x1) < 0) {
    return BIT_FLOAT_1;
  } else {
    return BIT_FLOAT_NEG_1;
  }
}

union FP32U32 {
  float fp32;
  uint32_t u32;
};

__mlu_func__ static bool isInner(const Line *__restrict__ line,
                                 const Point2D *__restrict__ c,
                                 uint32_t direction) {
  FP32U32 eps;
  eps.fp32 = EPSILON;
  eps.u32 = BIT_FLOAT_MUL(eps.u32, direction);
  FP32U32 result;
  result.fp32 = (line->b * c->y + line->a * c->x - line->c + eps.fp32);
  result.u32 = BIT_FLOAT_MUL(result.u32, direction);
  return result.fp32 > 0;
}

struct QuadClipBox {
  __mlu_func__ void addLines(const Point2D *__restrict__ A) {
    direction = getDirection(A);
    line[0].update(A, A + 1);
    Point2D centerAC = {(A[0].x + A[2].x) / 2, (A[0].y + A[2].y) / 2};
    bool ACFine = false;

    if (isInner(&line[0], &centerAC, direction)) {
      line[5].update(A + 3, A);
      if (isInner(&line[5], &centerAC, direction)) {
        ACFine = true;
      }
      Point2D centerBD = {(A[1].x + A[3].x) / 2, (A[1].y + A[3].y) / 2};
      line[1].update(A + 1, A + 2);
      if (ACFine && isInner(&line[0], &centerBD, direction) &&
          isInner(&line[1], &centerBD, direction)) {
        is_convex = true;
        return noSplit(A);
      }
    }
    if (ACFine) {
      splitByAC(A);
    } else {
      splitByBD(A);
    }
  }

  __mlu_func__ void splitByAC(const Point2D *__restrict__ A) {
    line[2].update(A + 2, A);
    line[3].update(A, A + 2);
    line[4].update(A + 2, A + 3);
  }

  __mlu_func__ void noSplit(const Point2D *__restrict__ A) {
    line[2].update(A + 2, A + 3);
    line[3].update(A + 3, A);
  }

  __mlu_func__ void splitByBD(const Point2D *__restrict__ A) {
    line[1].update(A + 1, A + 3);
    line[2].update(A + 3, A);
    line[3].update(A + 1, A + 2);
    line[4].update(A + 2, A + 3);
    line[5].update(A + 3, A + 1);
  }

  Line line[6];
  uint32_t direction;
  bool is_convex = false;
};

__mlu_func__ static void cross(const Line *__restrict__ line,
                               const Point2D *__restrict__ c,
                               const Point2D *__restrict__ d,
                               float *__restrict__ o_x,
                               float *__restrict__ o_y) {
  float a1 = line->a;
  float b1 = line->b;
  float c1 = line->c;
  float lambda = (a1 * c->x + b1 * c->y - c1);
  float beta = (a1 * d->x + b1 * d->y - c1);
  float ratio = lambda / (lambda - beta);
  float m_ratio = 1 - ratio;
  *o_x = ratio * d->x + m_ratio * c->x;
  *o_y = ratio * d->y + m_ratio * c->y;
}

__mlu_func__ static float polyArea(const Point2D *__restrict__ points, int n) {
  const Point2D *p0 = points;
  float area = 0;
  float x0 = points[1].x - p0->x;
  float y0 = points[1].y - p0->y;
  for (int i = 1; i < (n - 1); ++i) {
    float x1 = points[i + 1].x - p0->x;
    float y1 = points[i + 1].y - p0->y;
    area += (x0 * y1 - y0 * x1);
    x0 = x1;
    y0 = y1;
  }
  area = area / 2;
  FP32U32 result;
  result.fp32 = area;
  result.u32 = (result.u32 & 0x7FFFFFFF);
  return result.fp32;
}

template <int CUTLINE_N>
__mlu_func__ static float clipArea(const float *__restrict__ box_i,
                                   const Line *__restrict__ clip_box_lines,
                                   uint32_t direction) {
  constexpr int MAX_POINT = CUTLINE_N + 5;
  Point2D p_swap0[MAX_POINT];
  Point2D p_swap1[MAX_POINT];
  Point2D *p = p_swap0;
  Point2D *p_next = p_swap1;
#pragma unroll 4
  for (int i = 0; i < 4; ++i) {
    p_next[i].x = *box_i;
    ++box_i;
    p_next[i].y = *box_i;
    ++box_i;
  }
  int n = 4;
#pragma unroll CUTLINE_N
  for (int i = 0; i < CUTLINE_N; ++i) {
    Point2D *tmp = p_next;
    p_next = p;
    p = tmp;
    int new_n = 0;
    const Line *cut_line = &clip_box_lines[i];
    bool prev_inner = isInner(cut_line, p + n - 1, direction);
    for (int j = 0; j < n; ++j) {
      Point2D *current_point = p + j;
      Point2D *prev_point = p + (j - 1 + n) % n;
      bool current_inner = isInner(cut_line, current_point, direction);
      if (current_inner) {
        if (!prev_inner) {
          cross(cut_line, prev_point, current_point, &p_next[new_n].x,
                &p_next[new_n].y);
          ++new_n;
        }
        p_next[new_n].x = current_point->x;
        p_next[new_n].y = current_point->y;
        ++new_n;
      } else if (prev_inner) {
        cross(cut_line, prev_point, current_point, &p_next[new_n].x,
              &p_next[new_n].y);
        ++new_n;
      }
      prev_inner = current_inner;
    }
    n = new_n;
    if (n < 3) {
      return 0;
    }
  }
  return polyArea(p_next, n);
}

__mlu_func__ float intersectArea(const float *__restrict__ box_i,
                                 const QuadClipBox *__restrict__ clip_box) {
  float area = 0;
  if (clip_box->is_convex) {
    area = clipArea<4>(box_i, clip_box->line, clip_box->direction);
  } else {
    area = clipArea<3>(box_i, clip_box->line, clip_box->direction);
    area += clipArea<3>(box_i, &clip_box->line[3], clip_box->direction);
  }

  return area;
}
}  // namespace
#endif  // BANGC_OPS_KERNELS_POLY_NMS_INTERSECT_AREA_H
