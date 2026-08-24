/*************************************************************************
 * Copyright (C) [2026] by Cambricon, Inc.
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
#ifndef TEST_CNNL_GTEST_SRC_ZOO_BOX_OVERLAP_BEV_BOX_OVERLAP_BEV_H_
#define TEST_CNNL_GTEST_SRC_ZOO_BOX_OVERLAP_BEV_BOX_OVERLAP_BEV_H_
#include "executor.h"

namespace mluoptest {
template <typename T>
struct BBox {
  T x_ctr, y_ctr, w, h, a;
};

template <typename T>
struct PPoint {
  T x, y;
  explicit inline PPoint(const T &px = 0, const T &py = 0) : x(px), y(py) {}
  inline PPoint operator+(const PPoint &p) const {
    return PPoint(x + p.x, y + p.y);
  }
  inline PPoint operator-(const PPoint &p) const {
    return PPoint(x - p.x, y - p.y);
  }
  inline PPoint operator+=(const PPoint &p) const {
    x += p.x;
    y += p.y;
    return *this;
  }
  inline PPoint operator*(const T coeff) const {
    return PPoint(x * coeff, y * coeff);
  }
};
template <typename T>
inline T dot2d(const PPoint<T> &A, const PPoint<T> &B) {
  return A.x * B.x + A.y * B.y;
}
template <typename T>
inline T cross2d(const PPoint<T> &A, const PPoint<T> &B) {
  return A.x * B.y - A.y * B.x;
}

class BoxOverlapBevExecutor : public Executor {
 public:
  BoxOverlapBevExecutor() {}
  ~BoxOverlapBevExecutor() {}

  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;
  int64_t getTheoryIoSize() override;

 private:
  template <typename T>
  void cpuBoxOverlapBev(const T *box1, const T *box2, T *overlaps,
                        const int num_box1, const int num_box2);
  template <typename T>
  T singleBoxOverlapBev(const BBox<T> box1, const BBox<T> box2);
  template <typename T>
  T rotatedBoxesIntersection(const BBox<T> box1, const BBox<T> box2);
  template <typename T>
  void getRotatedVertices(const BBox<T> &box, PPoint<T> (&pts)[4]);
  template <typename T>
  T getIntersectionPoints(const PPoint<T> (&pts1)[4],
                          const PPoint<T> (&pts2)[4],
                          PPoint<T> (&intersections)[24]);
  template <typename T>
  int convexHullGraham(const PPoint<T> (&p)[24], const int &num_in,
                       PPoint<T> (&q)[24]);
  template <typename T>
  T polygonArea(const PPoint<T> (&q)[24], const int &m);
};  // class Executor
}  // namespace mluoptest
#endif  // TEST_CNNL_GTEST_SRC_ZOO_BOX_OVERLAP_BEV_BOX_OVERLAP_BEV_H_
