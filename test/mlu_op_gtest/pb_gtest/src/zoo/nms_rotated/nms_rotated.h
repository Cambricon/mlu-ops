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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_NMS_ROTATED_NMS_ROTATED_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_NMS_ROTATED_NMS_ROTATED_H_
#include "executor.h"
namespace mluoptest {

template <typename T>
struct Box;

template <typename T>
struct Point;

template <typename T>
inline T dot2d(const Point<T> &A, const Point<T> &B);

template <typename T>
inline T cross2d(const Point<T> &A, const Point<T> &B);

class NmsRotatedExecutor : public Executor {
 public:
  NmsRotatedExecutor() {}
  ~NmsRotatedExecutor() {}

  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;
  int64_t getTheoryIoSize() override;
  void workspaceFree();
  void workspaceMalloc();
  int64_t out_num_;

 private:
  template <typename T>
  void cpuNmsRotated(
          const T *boxes, const T *scores, T *output,
          const int num_box, const float iou_threshold, const int box_dim);
  template <typename T>
  T singleBoxIouRotated(const T *box1, const T *box2, const int mode);
  template <typename T>
  T rotatedBoxesIntersection(const Box<T> box1, const Box<T> box2);
  template <typename T>
  void getRotatedVertices(const Box<T> &box, Point<T> (&pts)[4]);
  template <typename T>
  T getIntersectionPoints(const Point<T> (&pts1)[4], const Point<T> (&pts2)[4],
                          Point<T> (&intersections)[24]);
  template <typename T>
  int convexHullGraham(const Point<T> (&p)[24], const int &num_in,
                       Point<T> (&q)[24]);
  template <typename T>
  T polygonArea(const Point<T> (&q)[24], const int &m);
};  // class Executor
}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_NMS_ROTATED_NMS_ROTATED_H_
