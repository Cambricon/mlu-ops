/*************************************************************************
* Copyright (C) [2019-2022] by Cambricon, Inc.
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

#include "kernels/poly_nms/enums.h"

namespace {
struct Point2D {
  float x;
  float y;
};
struct Vector2D {
  float x;
  float y;
};

struct Line {
  __mlu_func__ void Update(const Point2D *__restrict__ A,
                           const Point2D *__restrict__ B) {
    a = B->y - A->y;
    b = A->x - B->x;
    c = a * (A->x) + b * (A->y);
    s.x = A->x;
    s.y = A->y;
  }
  float a;
  float b;
  float c;
  Point2D s;
};

template <PointDirection DIR>
__mlu_func__ static inline bool IsInner(const Line *__restrict__ line,
                                        const Point2D *__restrict__ c);

template <>
__mlu_func__ inline bool
IsInner<PointDirection::CW>(const Line *__restrict__ line,
                            const Point2D *__restrict__ c) {
  return (line->b * (c->y - line->s.y) + line->a * (c->x - line->s.x)) > 0;
}

template <>
__mlu_func__ inline bool
IsInner<PointDirection::CCW>(const Line *__restrict__ line,
                             const Point2D *__restrict__ c) {
  return (line->b * (c->y - line->s.y) + line->a * (c->x - line->s.x)) < 0;
}
template <PointDirection POINT_DIR>
struct QuadClipBox {
  __mlu_func__ void AddLines(const Point2D *__restrict__ A) {
    line[0].Update(A, A + 1);
    Point2D centerAC = {(A[0].x + A[2].x) / 2, (A[0].y + A[2].y) / 2};
    bool ACFine = false;

    if (IsInner<POINT_DIR>(&line[0], &centerAC)) {
      line[5].Update(A + 3, A);
      if (IsInner<POINT_DIR>(&line[5], &centerAC)) {
        ACFine = true;
      }
      Point2D centerBD = {(A[1].x + A[3].x) / 2, (A[1].y + A[3].y) / 2};
      line[1].Update(A + 1, A + 2);
      if (IsInner<POINT_DIR>(&line[0], &centerBD) &&
          IsInner<POINT_DIR>(&line[1], &centerBD)) {
        is_convex = true;
        return NoSplit(A);
      }
    }
    if (ACFine) {
      SplitByAC(A);
    } else {
      SplitByBD(A);
    }
  }
  __mlu_func__ void SplitByAC(const Point2D *__restrict__ A) {
    line[2].Update(A + 2, A);
    line[3].Update(A, A + 2);
    line[4].Update(A + 2, A + 3);
  }

  __mlu_func__ void NoSplit(const Point2D *__restrict__ A) {
    line[2].Update(A + 2, A + 3);
    line[3].Update(A + 3, A);
  }

  __mlu_func__ void SplitByBD(const Point2D *__restrict__ A) {
    line[1].Update(A + 1, A + 3);
    line[2].Update(A + 3, A);
    line[3].Update(A + 1, A + 2);
    line[4].Update(A + 2, A + 3);
    line[5].Update(A + 3, A + 1);
  }
  Line line[6];
  bool is_convex = false;
};

__mlu_func__ static inline void Cross(const Line *__restrict__ line,
                                      const Point2D *__restrict__ c,
                                      const Point2D *__restrict__ d, float *o_x,
                                      float *o_y) {
  float a1 = line->a;
  float b1 = line->b;
  float c1 = line->c;

  float a2 = d->y - c->y;
  float b2 = c->x - d->x;
  float c2 = a2 * (c->x) + b2 * (c->y);

  float determinant = a1 * b2 - a2 * b1;

  float x = (b2 * c1 - b1 * c2) / determinant;
  float y = (a1 * c2 - a2 * c1) / determinant;
  *o_x = x;
  *o_y = y;
}

__mlu_func__ static inline float Area(const Point2D *__restrict__ points,
                                      int n) {
  const Point2D *p0 = points;
  float area = 0;
  Vector2D vec0 = {points[1].x - p0->x, points[1].y - p0->y};
  Vector2D vec1;
  Vector2D *v0 = &vec1;
  Vector2D *v1 = &vec0;
  for (int i = 1; i < (n - 1); ++i) {
    v0->x = points[i + 1].x - p0->x;
    v0->y = points[i + 1].y - p0->y;
    Vector2D *tmp = v0;
    v0 = v1;
    v1 = tmp;
    area += (v0->x * v1->y - v0->y * v1->x);
  }
  return area / 2;
}

template <int CUTLINE_N, PointDirection POINT_DIR>
__mlu_func__ static inline float
ClipArea(Point2D *p, Point2D *p_next, const Line *__restrict__ clip_box_lines) {
  int n = 4;
#pragma unroll CUTLINE_N
  for (int i = 0; i < CUTLINE_N; ++i) {
    Point2D *tmp = p_next;
    p_next = p;
    p = tmp;
    int new_n = 0;
    const Line *cut_line = &clip_box_lines[i];
    for (int j = 0; j < n; ++j) {
      Point2D *current_point = p + j;
      Point2D *prev_point = p + (j - 1 + n) % n;
      if (IsInner<POINT_DIR>(cut_line, current_point)) {
        if (!IsInner<POINT_DIR>(cut_line, prev_point)) {
          Cross(cut_line, prev_point, current_point, &p_next[new_n].x,
                &p_next[new_n].y);
          ++new_n;
        }
        p_next[new_n].x = current_point->x;
        p_next[new_n].y = current_point->y;
        ++new_n;
      } else if (IsInner<POINT_DIR>(cut_line, prev_point)) {
        Cross(cut_line, prev_point, current_point, &p_next[new_n].x,
              &p_next[new_n].y);
        ++new_n;
      }
    }
    n = new_n;
    if (n < 3) {
      return 0;
    }
  }
  return Area(p_next, n);
}

template <PointDirection POINT_DIR>
__mlu_func__ inline float
IntersectArea(const float *__restrict__ box_i,
              const QuadClipBox<POINT_DIR> *__restrict__ clip_box) {
  constexpr int MAX_POINT = 8;
  Point2D p_swap0[MAX_POINT];
  Point2D p_swap1[MAX_POINT];
  Point2D *p = p_swap0;
  Point2D *p_next = p_swap1;
  const float *pp = box_i;
#pragma unroll 4
  for (int i = 0; i < 4; ++i) {
    p_next[i].x = *pp;
    ++pp;
    p_next[i].y = *pp;
    ++pp;
  }
  float area = 0;
  if (clip_box->is_convex) {
    area = ClipArea<4, POINT_DIR>(p, p_next, clip_box->line);
  } else {
    area = ClipArea<3, POINT_DIR>(p, p_next, clip_box->line);
    pp = box_i;
#pragma unroll 4
    for (int i = 0; i < 4; ++i) {
      p_next[i].x = *pp;
      ++pp;
      p_next[i].y = *pp;
      ++pp;
    }
    area += ClipArea<3, POINT_DIR>(p, p_next, &clip_box->line[3]);
  }

  return area > 0 ? area : -area;
}
} // namespace
#endif // BANGC_OPS_KERNELS_POLY_NMS_INTERSECT_AREA_H
