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
#include "nms_rotated.h"
#include <algorithm>
#include <vector>

namespace mluoptest {

template <typename T>
struct Box {
  T x_ctr, y_ctr, w, h, a;
};

template <typename T>
struct Point {
  T x, y;
  explicit inline Point(const T &px = 0, const T &py = 0) : x(px), y(py) {}
  inline Point operator+(const Point &p) const {
    return Point(x + p.x, y + p.y);
  }
  inline Point operator-(const Point &p) const {
    return Point(x - p.x, y - p.y);
  }
  inline Point operator+=(const Point &p) const {
    x += p.x;
    y += p.y;
    return *this;
  }
  inline Point operator*(const T coeff) const {
    return Point(x * coeff, y * coeff);
  }
};
template <typename T>
inline T dot2d(const Point<T> &A, const Point<T> &B) {
  return A.x * B.x + A.y * B.y;
}
template <typename T>
inline T cross2d(const Point<T> &A, const Point<T> &B) {
  return A.x * B.y - A.y * B.x;
}

void NmsRotatedExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 2,
              "nms_rotated tensor input number is wrong.");
  GTEST_CHECK(parser_->outputs().size() == 2,
              "nms_rotated tensor output number is wrong.");
  GTEST_CHECK(parser_->getProtoNode()->has_nms_rotated_param(),
              "nms_rotated param not found!");
}

void NmsRotatedExecutor::workspaceMalloc() {
  size_t workspace_size = 0;
  auto boxes_desc = tensor_desc_[0].tensor;
  MLUOP_CHECK(mluOpGetNmsRotatedWorkspaceSize(handle_, boxes_desc,
                &workspace_size));
  VLOG(4) << "Malloc workspace space.";
  void *temp = mlu_runtime_.allocate(workspace_size);
  workspace_.push_back(temp);
  VLOG(4) << "Malloc addr: " << temp << " , size: " << workspace_size;
  eva_->setMluWorkspaceSize(workspace_size);
}

void NmsRotatedExecutor::workspaceFree() {
  VLOG(4) << "Free device workspace space.";
  if (workspace_[0] != NULL) {
    mlu_runtime_.deallocate(workspace_[0]);
  }
}

void NmsRotatedExecutor::compute() {
  VLOG(4) << "NmsRotatedExecutor compute";

  float iou_threshold =
    parser_->getProtoNode()->nms_rotated_param().iou_threshold();
  auto boxes = tensor_desc_[0].tensor;
  auto dev_boxes = data_vector_[0].device_ptr;
  auto scores = tensor_desc_[1].tensor;
  auto dev_scores = data_vector_[1].device_ptr;
  auto output = tensor_desc_[2].tensor;
  auto dev_output = data_vector_[2].device_ptr;
  auto result_num = data_vector_[3].device_ptr;
  size_t workspace_size = 0;
  size_t output_size = parser_->getMetaTensor("output1").size_in_bytes;
  GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMemset(dev_output, 0, output_size));
  GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMemset(result_num, 0, sizeof(int32_t)));

  // GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMemset(dev_output, 0,
  //   output->dims[0] * sizeof(int64_t)));
  VLOG(4) << "call mluOpNmsRotated()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpGetNmsRotatedWorkspaceSize(
              handle_, boxes, &workspace_size));
  MLUOP_CHECK(mluOpNmsRotated(
      handle_, iou_threshold, boxes, dev_boxes, scores, dev_scores,
      workspace_[0], workspace_size, output, dev_output,
      (int32_t *)result_num));
  interface_timer_.stop();
  VLOG(4) << "mluOpNmsRotated() finished!";
}

void NmsRotatedExecutor::cpuCompute() {
  auto count_boxes = parser_->getInputDataCount(0);
  if (count_boxes == 0) {
    return;
  }

  auto num_box = tensor_desc_[0].tensor->dims[0];
  auto box_dim = tensor_desc_[0].tensor->dims[1];
  float iou_threshold =
    parser_->getProtoNode()->nms_rotated_param().iou_threshold();

  VLOG(4) << "num_box: " << num_box;
  VLOG(4) << "box_dim: " << box_dim;
  VLOG(4) << "iou_threshold: " << iou_threshold;
  VLOG(4) << "call cpuNmsRotated()";
  memset(cpu_fp32_output_[0], 0x0, num_box * sizeof(float));
  cpuNmsRotated(cpu_fp32_input_[0], cpu_fp32_input_[1], cpu_fp32_output_[0],
                num_box, iou_threshold, box_dim);
  VLOG(4) << "cpuNmsRotated() finished!";
  VLOG(4) << "output box num is: " << out_num_;
}

template <typename T>
void NmsRotatedExecutor::cpuNmsRotated(const T *boxes,
                                       const T *scores,
                                       T *output,
                                       const int num_box,
                                       const float iou_threshold,
                                       const int box_dim) {
  std::vector<int32_t> order(num_box);
  for (int i = 0; i < num_box; i++) order[i] = i;
  sort(order.begin(), order.end(), [&scores] (int i1, int i2)
    {return scores[i1] > scores[i2];});

  std::vector<uint8_t> suppressed(num_box, 0);
  int64_t num_to_keep = 0;
  auto nboxes = num_box;

  for (int32_t _i = 0; _i < nboxes; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1) {
      continue;
    }

    output[num_to_keep++] = i;

    for (int32_t _j = _i + 1; _j < nboxes; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) {
        continue;
      }
      auto ovr = singleBoxIouRotated(boxes + i * box_dim,
                                    boxes + j * box_dim, 0);
      if (iou_threshold == 0) {
        if (ovr > iou_threshold) {
          suppressed[j] = 1;
        }
      } else {
        if (ovr >= iou_threshold) {
          suppressed[j] = 1;
        }
      }
    }
  }
  cpu_fp32_output_[1][0] = num_to_keep;
  out_num_ = num_to_keep;
}

template <typename T>
T NmsRotatedExecutor::singleBoxIouRotated(const T *box1_raw,
                                          const T *box2_raw,
                                          const int mode_flag) {
  // 1. Calculate new points
  Box<T> box1, box2;
  auto center_shift_x = (box1_raw[0] + box2_raw[0]) / 2.0;
  auto center_shift_y = (box1_raw[1] + box2_raw[1]) / 2.0;
  box1.x_ctr = box1_raw[0] - center_shift_x;
  box1.y_ctr = box1_raw[1] - center_shift_y;
  box1.w = box1_raw[2];
  box1.h = box1_raw[3];
  box1.a = box1_raw[4];

  box2.x_ctr = box2_raw[0] - center_shift_x;
  box2.y_ctr = box2_raw[1] - center_shift_y;
  box2.w = box2_raw[2];
  box2.h = box2_raw[3];
  box2.a = box2_raw[4];

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
  if (mode_flag == 0) {
    baseS = (area1 + area2 - intersection);
  } else if (mode_flag == 1) {
    baseS = area1;
  }
  const T iou = intersection * (1.0f / baseS);
  return iou;
}

template <typename T>
T NmsRotatedExecutor::rotatedBoxesIntersection(const Box<T> box1,
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
void NmsRotatedExecutor::getRotatedVertices(const Box<T> &box,
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
T NmsRotatedExecutor::getIntersectionPoints(const Point<T> (&pts1)[4],
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
int NmsRotatedExecutor::convexHullGraham(const Point<T> (&p)[24],
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
T NmsRotatedExecutor::polygonArea(const Point<T> (&q)[24], const int &m) {
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

int64_t NmsRotatedExecutor::getTheoryOps() {
  int64_t theory_ops =  60000 * out_num_;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

int64_t NmsRotatedExecutor::getTheoryIoSize() {
  int64_t theory_ios =
      (parser_->input(0)->total_count + parser_->input(1)->total_count +
        parser_->output(0)->total_count) * sizeof(float);
  VLOG(4) << "getTheoryIos: " << theory_ios << " bytes";
  return theory_ios;
}

}  // namespace mluoptest
