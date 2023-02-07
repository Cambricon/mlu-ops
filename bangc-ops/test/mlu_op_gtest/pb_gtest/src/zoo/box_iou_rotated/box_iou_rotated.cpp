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
#include "box_iou_rotated.h"


namespace mluoptest {

void BoxIouRotatedExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 2,
              "box_iou_rotated tensor input number is wrong.");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "box_iou_rotated tensor output number is wrong.");
  GTEST_CHECK(parser_->getProtoNode()->has_box_iou_rotated_param(),
              "box_iou_rotated param not found!");
}

void BoxIouRotatedExecutor::compute() {
  VLOG(4) << "BoxIouRotatedExecutor compute ";

  int mode = parser_->getProtoNode()->box_iou_rotated_param().mode();
  bool aligned = parser_->getProtoNode()->box_iou_rotated_param().aligned();
  auto box1 = tensor_desc_[0].tensor;
  auto box2 = tensor_desc_[1].tensor;
  auto ious = tensor_desc_[2].tensor;
  auto dev_box1 = data_vector_[0].device_ptr;
  auto dev_box2 = data_vector_[1].device_ptr;
  auto dev_ious = data_vector_[2].device_ptr;
  VLOG(4) << "call mluOpBoxIouRotated()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpBoxIouRotated(handle_, mode, aligned, box1, dev_box1, box2,
                                 dev_box2, ious, dev_ious));
  interface_timer_.stop();
  VLOG(4) << "mluOpBoxIouRotated() over";
}

void BoxIouRotatedExecutor::cpuCompute() {
  auto count_box1 = parser_->getInputDataCount(0);
  auto count_box2 = parser_->getInputDataCount(1);
  auto count_out = parser_->getOutputDataCount(0);
  if (count_box1 == 0 || count_box1 == 0 || count_out == 0) {
    return;
  }

  auto box1_desc = tensor_desc_[0].tensor;
  auto box2_desc = tensor_desc_[1].tensor;
  auto num_box1 = box1_desc->dims[0];
  auto num_box2 = box2_desc->dims[0];

  int mode = parser_->getProtoNode()->box_iou_rotated_param().mode();
  bool aligned = parser_->getProtoNode()->box_iou_rotated_param().aligned();

  VLOG(4) << "mode:    " << mode;
  VLOG(4) << "aligned: " << aligned;

  VLOG(4) << "call cpuBoxIouRotated()";
  cpuBoxIouRotated(cpu_fp32_input_[0], cpu_fp32_input_[1], cpu_fp32_output_[0],
                   num_box1, num_box2, mode, aligned);
}

template <typename T>
void BoxIouRotatedExecutor::cpuBoxIouRotated(const T *box1_raw,
                                             const T *box2_raw, T *ious,
                                             const int num_box1,
                                             const int num_box2, const int mode,
                                             const bool aligned) {
  VLOG(4) << "num box1: " << num_box1;
  VLOG(4) << "num box2: " << num_box2;
  if (aligned) {
    int num_ious = tensor_desc_[2].tensor->dims[0];
    VLOG(4) << "num_ious: " << num_ious;
    GTEST_CHECK(num_box1 == num_ious,
                "when aligned, num_box1 should equal to num_ious.");
  } else {
    int num_ious = tensor_desc_[2].tensor->dims[0];
    VLOG(4) << "num_ious[0]: " << num_ious;
    num_ious = tensor_desc_[2].tensor->dims[1];
    VLOG(4) << "num_ious[1]: " << num_ious;
    GTEST_CHECK(((num_box1 == tensor_desc_[2].tensor->dims[0]) ||
                 (num_box2 == tensor_desc_[2].tensor->dims[1])),
                "when not aligned, num_ious should equal to num_box1*num_box2");
  }

  Box<T> box1, box2;
  if (aligned) {
    for (int i = 0; i < num_box1; i++) {
      box1.x_ctr = box1_raw[5 * i];
      box1.y_ctr = box1_raw[5 * i + 1];
      box1.w = box1_raw[5 * i + 2];
      box1.h = box1_raw[5 * i + 3];
      box1.a = box1_raw[5 * i + 4];

      box2.x_ctr = box2_raw[5 * i];
      box2.y_ctr = box2_raw[5 * i + 1];
      box2.w = box2_raw[5 * i + 2];
      box2.h = box2_raw[5 * i + 3];
      box2.a = box2_raw[5 * i + 4];

      ious[i] = singleBoxIouRotated<T>(box1, box2, mode);
      VLOG(6) << "aligned ious: " << ious[i];
    }
  } else {
    for (int i = 0; i < num_box1; i++) {
      for (int j = 0; j < num_box2; j++) {
        box1.x_ctr = box1_raw[5 * i];
        box1.y_ctr = box1_raw[5 * i + 1];
        box1.w = box1_raw[5 * i + 2];
        box1.h = box1_raw[5 * i + 3];
        box1.a = box1_raw[5 * i + 4];

        box2.x_ctr = box2_raw[5 * j];
        box2.y_ctr = box2_raw[5 * j + 1];
        box2.w = box2_raw[5 * j + 2];
        box2.h = box2_raw[5 * j + 3];
        box2.a = box2_raw[5 * j + 4];

        ious[i * num_box2 + j] = singleBoxIouRotated<T>(box1, box2, mode);
        VLOG(6) << "not aligned ious: " << ious[i * num_box2 + j];
      }
    }
  }
}

template <typename T>
T BoxIouRotatedExecutor::singleBoxIouRotated(const Box<T> box1_raw,
                                             const Box<T> box2_raw,
                                             const int mode) {
  // 1. Calculate new points
  Box<T> box1, box2;
  auto center_shift_x = (box1_raw.x_ctr + box2_raw.x_ctr) / 2.0;
  auto center_shift_y = (box1_raw.y_ctr + box2_raw.y_ctr) / 2.0;
  box1.x_ctr = box1_raw.x_ctr - center_shift_x;
  box1.y_ctr = box1_raw.y_ctr - center_shift_y;
  box1.w = box1_raw.w;
  box1.h = box1_raw.h;
  box1.a = box1_raw.a;

  box2.x_ctr = box2_raw.x_ctr - center_shift_x;
  box2.y_ctr = box2_raw.y_ctr - center_shift_y;
  box2.w = box2_raw.w;
  box2.h = box2_raw.h;
  box2.a = box2_raw.a;

  VLOG(7) << "new_box1 x: " << box1.x_ctr << "\tnew_box2 x: " << box2.x_ctr;
  VLOG(7) << "new_box1 y: " << box1.y_ctr << "\tnew_box2 y: " << box2.y_ctr;
  VLOG(7) << "new_box1 w: " << box1.w << "\tnew_box2 w: " << box2.w;
  VLOG(7) << "new_box1 h: " << box1.h << "\tnew_box2 h: " << box2.h;
  VLOG(7) << "new_box1 a: " << box1.a << "\tnew_box2 a: " << box2.a;

  const T area1 = box1.w * box1.h;
  const T area2 = box2.w * box2.h;
  VLOG(7) << "area1: " << area1 << " area2: " << area2;

  if (area1 < 1e-14 || area2 < 1e-14) {
    VLOG(7) << "check area < 1e-14!";
    return 0.f;
  }

  const T intersection = rotatedBoxesIntersection<T>(box1, box2);
  T baseS = 1.0;
  // when mode==0, IOU; mode==1, IOF
  if (mode == 0) {
    baseS = (area1 + area2 - intersection);
  } else {
    baseS = area1;
  }
  const T iou = intersection * (1.0f / baseS);
  return iou;
}

template <typename T>
T BoxIouRotatedExecutor::rotatedBoxesIntersection(const Box<T> box1,
                                                  const Box<T> box2) {
  // There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
  // from rotated_rect_intersection_pts
  Point<T> intersectPts[24], orderedPts[24];
  Point<T> pts1[4];
  Point<T> pts2[4];

  // 2. Calculate rotated vertices
  getRotatedVertices<T>(box1, pts1);
  getRotatedVertices<T>(box2, pts2);

  // 3. Get all intersection points
  int num = getIntersectionPoints<T>(pts1, pts2, intersectPts);
  if (num <= 2) {
    VLOG(7) << "check nums in <= 2!";
    return 0.0;
  }

  // 4. Convex-hull-graham to order the intersection points in clockwise order
  // and find the contour area
  int num_convex = convexHullGraham<T>(intersectPts, num, orderedPts);
  VLOG(7) << "num_convex: " << num_convex;

  // 5. Calculate polygon area
  return polygonArea<T>(orderedPts, num_convex);
}

template <typename T>
void BoxIouRotatedExecutor::getRotatedVertices(const Box<T> &box,
                                               Point<T> (&pts)[4]) {
  // M_PI / 180. == 0.01745329251
  // double theta = box.a * 0.01745329251;
  double theta = box.a;
  T cosTheta2 = (T)cosf(theta) * 0.5f;
  T sinTheta2 = (T)sinf(theta) * 0.5f;

  // y: top->down; x: left->right
  pts[0].x = box.x_ctr - sinTheta2 * box.h - cosTheta2 * box.w;
  pts[0].y = box.y_ctr + cosTheta2 * box.h - sinTheta2 * box.w;
  pts[1].x = box.x_ctr + sinTheta2 * box.h - cosTheta2 * box.w;
  pts[1].y = box.y_ctr - cosTheta2 * box.h - sinTheta2 * box.w;
  pts[2].x = 2 * box.x_ctr - pts[0].x;
  pts[2].y = 2 * box.y_ctr - pts[0].y;
  pts[3].x = 2 * box.x_ctr - pts[1].x;
  pts[3].y = 2 * box.y_ctr - pts[1].y;

  VLOG(7) << "pts[0].x  " << pts[0].x;
  VLOG(7) << "pts[0].y  " << pts[0].y;
  VLOG(7) << "pts[1].x  " << pts[1].x;
  VLOG(7) << "pts[1].y  " << pts[1].y;
  VLOG(7) << "pts[2].x  " << pts[2].x;
  VLOG(7) << "pts[2].y  " << pts[2].y;
  VLOG(7) << "pts[3].x  " << pts[3].x;
  VLOG(7) << "pts[3].y  " << pts[3].y;
}
template <typename T>
T BoxIouRotatedExecutor::getIntersectionPoints(const Point<T> (&pts1)[4],
                                               const Point<T> (&pts2)[4],
                                               Point<T> (&intersections)[24]) {
  // Line vector, from p1 to p2 is: p1+(p2-p1)*t, t=[0,1]
  Point<T> vec1[4], vec2[4];
  for (int i = 0; i < 4; i++) {
    vec1[i] = pts1[(i + 1) % 4] - pts1[i];
    vec2[i] = pts2[(i + 1) % 4] - pts2[i];
  }

  // Line test - test all line combos for intersection, 4x4 posible
  int num = 0;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      T det = cross2d<T>(vec2[j], vec1[i]);

      // deal with parallel lines
      if (fabs(det) <= 1e-14) {
        continue;
      }

      auto vec12 = pts2[j] - pts1[i];

      T t1 = cross2d<T>(vec2[j], vec12) * (1.0f / det);
      T t2 = cross2d<T>(vec1[i], vec12) * (1.0f / det);

      if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
        intersections[num++] = pts1[i] + vec1[i] * t1;
        VLOG(7) << "intersections %" << i * 4 + j << ": "
                << intersections[num - 1].x << ", " << intersections[num - 1].y;
      }
    }
  }

  // Check for vertices of rect1 inside rect2
  {
    const auto &AB = vec2[0];
    const auto &DA = vec2[3];
    auto ABdotAB = dot2d<T>(AB, AB);
    auto ADdotAD = dot2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      // assume ABCD is the rectangle, and P is the point to be judged
      // P is inside ABCD iff. P's projection on AB lines within AB
      // and P's projection on AD lies within AD
      auto AP = pts1[i] - pts2[0];
      auto APdotAB = dot2d<T>(AP, AB);
      auto APdotAD = -dot2d<T>(AP, DA);
      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts1[i];
        VLOG(7) << "intersections %" << i + 16 << ": "
                << intersections[num - 1].x << ", " << intersections[num - 1].y;
      }
    }
  }

  // Reverse the check - check for vertices of rect2 inside rect1
  {
    const auto &AB = vec1[0];
    const auto &DA = vec1[3];
    auto ABdotAB = dot2d<T>(AB, AB);
    auto ADdotAD = dot2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      auto AP = pts2[i] - pts1[0];
      auto APdotAB = dot2d<T>(AP, AB);
      auto APdotAD = -dot2d<T>(AP, DA);
      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts2[i];
        VLOG(7) << "intersections %" << i + 20 << ": "
                << intersections[num - 1].x << ", " << intersections[num - 1].y;
      }
    }
  }
  return num;
}
template <typename T>
int BoxIouRotatedExecutor::convexHullGraham(const Point<T> (&p)[24],
                                            const int &num_in,
                                            Point<T> (&q)[24]) {
  assert(num_in >= 2);
  // Step1:
  // Find point with minimum y
  // if more than 1 points have the same minimum y,
  // pick the one with the minimum x.
  int t = 0;
  for (int i = 0; i < num_in; i++) {
    if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
      t = i;
    }
  }
  auto &start = p[t];  // starting point
  VLOG(7) << "start point: " << start.x << ", " << start.y;

  // Step2:
  // Subtract starting point from every points (for sorting in the next step)
  for (int i = 0; i < num_in; i++) {
    q[i] = p[i] - start;
  }
  // Swap the starting point to position 0
  auto tmp = q[0];
  q[0] = q[t];
  q[t] = tmp;

  // Step3:
  // Sort point 1~num_in according to their relative cross-product values
  // (essentially sorting according to angles)
  // If the angles are the same, sort according to their distance to origin
  T dist[24];
  for (int i = 0; i < num_in; i++) {
    dist[i] = dot2d<T>(q[i], q[i]);
  }

  T temp;
  for (int i = 1; i < num_in - 1; i++) {
    for (int j = i + 1; j < num_in; j++) {
      temp = cross2d<T>(q[i], q[j]);
      if ((temp < -1e-6) || ((fabs(temp) < 1e-6) && (dist[i] > dist[j]))) {
        tmp = q[i];
        q[i] = q[j];
        q[j] = tmp;
        temp = dist[i];
        dist[i] = dist[j];
        dist[j] = temp;
      }
    }
  }

  for (int i = 0; i < num_in; i++) {
    VLOG(7) << "ordered dist " << i << ": " << dist[i];
  }

  // Step4:
  // Make sure there are at least 2 points(that don't overlap with each other)
  // in the stack
  int k;  // index of the non-overlapped second point
  for (k = 1; k < num_in; k++) {
    if (dist[k] > 1e-8) {
      break;
    }
  }
  if (k == num_in) {
    // We reach the end, which means the convex hull is just one point
    q[0] = p[t];
    return 1;
  }
  q[1] = q[k];
  int m = 2;  // 2 points in the stack

  // Step 5:
  // Finally we ca start the scanning process.
  // When a non-convex relationship between the 3 points is found
  // (either concave shape or duplicated points),
  // we pop the previous point from the stack
  // until the 3-point relationship is convex again, or
  // until the stack only contains two points
  for (int i = k + 1; i < num_in; i++) {
    while (m > 1 && cross2d<T>(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0) {
      m--;
    }
    q[m++] = q[i];
  }

  // Step 6: Optional, ignored...
  return m;
}
template <typename T>
T BoxIouRotatedExecutor::polygonArea(const Point<T> (&q)[24], const int &m) {
  if (m <= 2) {
    VLOG(7) << "check polygon nums in <= 2";
    return 0;
  }
  T area = 0;
  for (int i = 1; i < m - 1; i++) {
    area += fabs(cross2d<T>(q[i] - q[0], q[i + 1] - q[0]));
  }
  VLOG(7) << "polygon area: " << area / 2.0;
  return area / 2.0;
}

int64_t BoxIouRotatedExecutor::getTheoryOps() {
  bool aligned = parser_->getProtoNode()->box_iou_rotated_param().aligned();
  int64_t num_box1 = parser_->input(0)->total_count / 5;
  int64_t num_box2 = parser_->input(1)->total_count / 5;
  int64_t theory_ops = aligned ? 60000 * num_box1 : 66000 * num_box1 * num_box2;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

int64_t BoxIouRotatedExecutor::getTheoryIoSize() {
  int64_t theory_ios =
      (parser_->input(0)->total_count + parser_->input(1)->total_count) *
          (1 + 64 / 5) +
      parser_->output(0)->total_count;
  VLOG(4) << "getTheoryIos: " << theory_ios << " ios";
  return theory_ios;
}

}  // namespace mluoptest
